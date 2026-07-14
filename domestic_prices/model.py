from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "price_cny_per_tonne",
    "lag_1",
    "lag_5",
    "lag_10",
    "lag_20",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "return_1d",
    "return_5d",
    "ma5_gap",
    "ma10_gap",
    "ma20_gap",
    "volatility_10d",
    "volatility_20d",
    "shfe_basis",
    "inventory_change_5d",
    "premium_discount",
    "import_profit",
]


def generate_forecasts(
    spot_prices: pd.DataFrame,
    market_features: pd.DataFrame,
    forecast_days: int,
    model_version: str,
) -> pd.DataFrame:
    frames = []
    for metal, metal_prices in spot_prices.groupby("metal"):
        metal_features = market_features[market_features["metal"] == metal] if not market_features.empty else pd.DataFrame()
        frames.append(_forecast_one_metal(metal, metal_prices, metal_features, forecast_days, model_version))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _forecast_one_metal(
    metal: str,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    forecast_days: int,
    model_version: str,
) -> pd.DataFrame:
    history = _prepare_history(prices, features)
    if len(history) < 20:
        raise ValueError(f"{metal} needs at least 20 valid daily prices for forecasting")

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    simulated = history.copy()
    last_real_price = float(simulated["price_cny_per_tonne"].iloc[-1])
    residual_std = _safe_std(history["price_cny_per_tonne"].diff().dropna()) / max(last_real_price, 1.0)
    price_model, residual_std = _fit_price_path_model(history, forecast_days, residual_std, last_real_price)
    rows = []
    for step in range(1, forecast_days + 1):
        next_date = pd.Timestamp(simulated["trade_date"].iloc[-1]) + pd.Timedelta(days=1)
        candidate = _next_feature_row(simulated, next_date)
        predicted_price = _predict_price_for_horizon(price_model, candidate, step)
        uncertainty = 1.96 * residual_std * np.sqrt(step) * predicted_price
        lower_bound = max(predicted_price - uncertainty, 1.0)
        upper_bound = predicted_price + uncertainty
        direction = _direction(predicted_price, last_real_price)
        rows.append(
            {
                "metal": metal,
                "forecast_date": next_date,
                "predicted_price_cny_per_tonne": round(predicted_price, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "direction": direction,
                "model_version": model_version,
                "generated_at": generated_at,
            }
        )
        simulated = pd.concat(
            [
                simulated,
                pd.DataFrame(
                    [
                        {
                            "trade_date": next_date,
                            "price_cny_per_tonne": predicted_price,
                            "shfe_futures_price": candidate["shfe_futures_price"].iloc[0],
                            "inventory_tonne": candidate["inventory_tonne"].iloc[0],
                            "premium_discount": candidate["premium_discount"].iloc[0],
                            "import_profit": candidate["import_profit"].iloc[0],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        simulated = _add_derived_features(simulated)
    return pd.DataFrame(rows)


def _fit_price_path_model(
    history: pd.DataFrame,
    forecast_days: int,
    fallback_residual_std: float,
    current_price: float,
) -> tuple[object | None, float]:
    train = history.copy()
    target_data = {
        f"target_price_h{horizon}": train["price_cny_per_tonne"].shift(-horizon)
        for horizon in range(1, forecast_days + 1)
    }
    target_frame = pd.DataFrame(target_data, index=train.index)
    target_cols = list(target_frame.columns)
    train = pd.concat([train, target_frame], axis=1)
    train = train.dropna(subset=FEATURE_COLUMNS + target_cols)
    if len(train) < 80:
        return None, fallback_residual_std

    model = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    model.fit(train[FEATURE_COLUMNS], train[target_cols])
    residual = train[target_cols] - model.predict(train[FEATURE_COLUMNS])
    residual_std = max(float(np.nanmean(np.nanstd(residual, axis=0))) / max(current_price, 1.0), fallback_residual_std * 0.6)
    return model, residual_std


def _predict_price_for_horizon(
    model: object | None,
    candidate: pd.DataFrame,
    horizon: int,
) -> float:
    current_price = float(candidate["price_cny_per_tonne"].iloc[0])
    ma20 = float(candidate["ma20"].iloc[0]) if pd.notna(candidate["ma20"].iloc[0]) else current_price
    ma60 = float(candidate["ma60"].iloc[0]) if pd.notna(candidate["ma60"].iloc[0]) else ma20

    if model is None:
        raw_price = current_price
    else:
        path = model.predict(candidate[FEATURE_COLUMNS])[0]
        raw_price = float(path[min(horizon - 1, len(path) - 1)])

    # Anchor the machine-learning estimate to recent price levels. This keeps long-horizon
    # forecasts from jumping too far away when only price history is available.
    horizon_weight = min(0.55, 0.20 + 0.015 * horizon)
    anchor_price = 0.70 * ma20 + 0.30 * ma60
    predicted_price = (1 - horizon_weight) * raw_price + horizon_weight * anchor_price
    max_move = current_price * min(0.20, 0.035 * np.sqrt(horizon))
    predicted_price = float(np.clip(predicted_price, current_price - max_move, current_price + max_move))
    return max(predicted_price, 1.0)


def build_feature_snapshot(spot_prices: pd.DataFrame, market_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metal, metal_prices in spot_prices.groupby("metal"):
        metal_features = market_features[market_features["metal"] == metal] if not market_features.empty else pd.DataFrame()
        history = _prepare_history(metal_prices, metal_features)
        latest = history.iloc[-1]
        for column in FEATURE_COLUMNS:
            rows.append({"metal": metal, "factor": column, "latest_value": latest.get(column)})
    return pd.DataFrame(rows)


def _prepare_history(prices: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    history = prices[["trade_date", "price_cny_per_tonne"]].copy()
    history["trade_date"] = pd.to_datetime(history["trade_date"])
    history = history.drop_duplicates("trade_date", keep="last").sort_values("trade_date")
    if not features.empty:
        feature_cols = [
            "trade_date",
            "shfe_futures_price",
            "inventory_tonne",
            "premium_discount",
            "import_profit",
        ]
        usable = features[[col for col in feature_cols if col in features]].copy()
        usable["trade_date"] = pd.to_datetime(usable["trade_date"])
        history = history.merge(usable, on="trade_date", how="left")
    for column in ["shfe_futures_price", "inventory_tonne", "premium_discount", "import_profit"]:
        if column not in history:
            history[column] = np.nan
        history[column] = pd.to_numeric(history[column], errors="coerce").ffill()
    history = _add_derived_features(history)
    return history.replace([np.inf, -np.inf], np.nan)


def _add_derived_features(history: pd.DataFrame) -> pd.DataFrame:
    history = history.sort_values("trade_date").copy()
    price = history["price_cny_per_tonne"].astype(float)
    for lag in [1, 5, 10, 20]:
        history[f"lag_{lag}"] = price.shift(lag)
    for window in [5, 10, 20, 60]:
        history[f"ma{window}"] = price.rolling(window, min_periods=max(2, window // 2)).mean()
    history["return_1d"] = price.pct_change(fill_method=None)
    history["return_5d"] = price.pct_change(5, fill_method=None)
    for window in [5, 10, 20]:
        ma = history[f"ma{window}"]
        history[f"ma{window}_gap"] = price / ma - 1
    history["volatility_10d"] = history["return_1d"].rolling(10, min_periods=5).std()
    history["volatility_20d"] = history["return_1d"].rolling(20, min_periods=10).std()
    history["shfe_basis"] = history["shfe_futures_price"] / price - 1
    history["inventory_change_5d"] = history["inventory_tonne"].pct_change(5, fill_method=None)
    for column in FEATURE_COLUMNS:
        if column not in history:
            history[column] = 0.0
        history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)
    return history


def _next_feature_row(history: pd.DataFrame, next_date: pd.Timestamp) -> pd.DataFrame:
    latest = history.iloc[-1].copy()
    row = pd.DataFrame(
        [
            {
                "trade_date": next_date,
                "price_cny_per_tonne": latest["price_cny_per_tonne"],
                "shfe_futures_price": latest.get("shfe_futures_price"),
                "inventory_tonne": latest.get("inventory_tonne"),
                "premium_discount": latest.get("premium_discount"),
                "import_profit": latest.get("import_profit"),
            }
        ]
    )
    expanded = pd.concat([history, row], ignore_index=True)
    expanded = _add_derived_features(expanded)
    return expanded.tail(1)


def _safe_std(series: pd.Series) -> float:
    value = float(series.dropna().std()) if len(series.dropna()) else 0.01
    if not np.isfinite(value) or value <= 0:
        return 0.01
    return value


def _direction(predicted_price: float, last_real_price: float) -> str:
    change = predicted_price / last_real_price - 1
    if change > 0.002:
        return "up"
    if change < -0.002:
        return "down"
    return "flat"
