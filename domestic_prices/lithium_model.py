from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MONTHLY_FACTORS = [
    "新能源汽车产量当期值_环比",
    "汽车产量当期值_环比",
    "制造业PMI_变化",
    "工业增加值同比增长",
    "企业商品价格矿产品环比增长",
    "企业商品价格煤油电环比增长",
    "LC成交量_环比",
    "LC持仓量_环比",
    "LC价格波动率",
    "LC合约切换",
]
MONTHLY_MODEL_VERSION = "lithium-specific-elasticnet-monthly-v3"
DAILY_MODEL_VERSION = "lithium-specific-elasticnet-daily-v3"


@dataclass
class ModelDiagnostics:
    selected_model: str
    model_mae: float
    baseline_mae: float
    improvement_pct: float
    residual_std: float


@dataclass
class LithiumForecastResult:
    monthly_forecast: pd.DataFrame
    daily_forecast: pd.DataFrame
    monthly_coefficients: pd.DataFrame
    monthly_contributions: pd.DataFrame
    monthly_diagnostics: ModelDiagnostics
    daily_diagnostics: ModelDiagnostics


def build_lithium_forecasts(
    daily_prices: pd.DataFrame,
    monthly_factors: pd.DataFrame,
    monthly_periods: int = 12,
    daily_periods: int = 30,
) -> LithiumForecastResult:
    prices = _normalize_daily_prices(daily_prices)
    factors = _normalize_monthly_factors(monthly_factors)
    monthly, monthly_model, monthly_diag, coefficients, contributions = _monthly_forecast(
        prices, factors, monthly_periods
    )
    daily, daily_diag = _daily_forecast(prices, monthly, daily_periods)
    return LithiumForecastResult(
        monthly_forecast=monthly,
        daily_forecast=daily,
        monthly_coefficients=coefficients,
        monthly_contributions=contributions,
        monthly_diagnostics=monthly_diag,
        daily_diagnostics=daily_diag,
    )


def _monthly_forecast(
    prices: pd.DataFrame, factors: pd.DataFrame, periods: int
) -> tuple[pd.DataFrame, Pipeline | None, ModelDiagnostics, pd.DataFrame, pd.DataFrame]:
    target = (
        prices.set_index("date")["settlement_price"]
        .resample("MS")
        .mean()
        .dropna()
        .rename("price")
        .reset_index()
        .rename(columns={"date": "month"})
    )
    target["target_return"] = target["price"].pct_change()
    target["return_lag_1"] = target["target_return"].shift(1)
    target["return_lag_2"] = target["target_return"].shift(2)
    usable_factors = [column for column in MONTHLY_FACTORS if column in factors]
    frame = target.merge(factors[["month", *usable_factors]], on="month", how="left")
    feature_columns = ["return_lag_1", "return_lag_2", *usable_factors]
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    model_frame = frame[["month", "price", "target_return", *feature_columns]].dropna()
    if len(model_frame) < 20:
        return _monthly_baseline(target, periods, "样本不足，使用月度朴素基准")

    x = model_frame[feature_columns]
    y = model_frame["target_return"]
    splits = min(5, max(2, len(model_frame) // 6))
    cv = TimeSeriesSplit(n_splits=splits)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", ElasticNet(max_iter=20000, random_state=42)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {
            "model__alpha": [0.001, 0.005, 0.01, 0.03, 0.08, 0.15],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        cv=cv,
        scoring="neg_mean_absolute_error",
    )
    search.fit(x, y)
    oos_model: list[float] = []
    oos_actual: list[float] = []
    for train_idx, test_idx in cv.split(x):
        fold = search.best_estimator_
        fold.fit(x.iloc[train_idx], y.iloc[train_idx])
        oos_model.extend(fold.predict(x.iloc[test_idx]).tolist())
        oos_actual.extend(y.iloc[test_idx].tolist())
    model_mae = mean_absolute_error(oos_actual, oos_model)
    baseline_mae = mean_absolute_error(oos_actual, np.zeros(len(oos_actual)))
    improvement = 1 - model_mae / baseline_mae if baseline_mae > 0 else 0.0
    residual_std = float(np.std(np.asarray(oos_actual) - np.asarray(oos_model), ddof=0))
    if improvement < 0.05:
        result = _monthly_baseline(target, periods, "Elastic Net未明显优于月度朴素基准")
        result[2].model_mae = float(model_mae)
        result[2].baseline_mae = float(baseline_mae)
        result[2].improvement_pct = float(improvement)
        return result

    model = search.best_estimator_
    model.fit(x, y)
    future_months = pd.date_range(
        target["month"].max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS"
    )
    future_factors = _seasonal_factor_scenario(factors, usable_factors, future_months)
    returns = target["target_return"].dropna().tolist()
    rows: list[dict[str, object]] = []
    contribution_rows: list[dict[str, object]] = []
    last_price = float(target.iloc[-1]["price"])
    scaler = model.named_steps["scale"]
    estimator = model.named_steps["model"]
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    for horizon, month in enumerate(future_months, start=1):
        factor_row = future_factors[future_factors["month"] == month].iloc[0]
        values = {
            "return_lag_1": returns[-1],
            "return_lag_2": returns[-2],
            **{column: factor_row[column] for column in usable_factors},
        }
        future_x = pd.DataFrame([values], columns=feature_columns)
        predicted_return = float(model.predict(future_x)[0])
        predicted_return = float(np.clip(predicted_return, -0.12, 0.12))
        last_price = max(last_price * (1 + predicted_return), 1.0)
        width = 1.282 * residual_std * np.sqrt(horizon)
        rows.append(
            {
                "metal": "lithium_carbonate",
                "forecast_month": month,
                "predicted_price_cny_per_tonne": last_price,
                "lower_bound": max(last_price * (1 - width), 1.0),
                "upper_bound": last_price * (1 + width),
                "direction": _direction(predicted_return),
                "predicted_change_pct": predicted_return,
                "source": "碳酸锂产业逻辑约束Elastic Net月度模型",
                "model_version": MONTHLY_MODEL_VERSION,
                "generated_at": generated_at,
            }
        )
        standardized = scaler.transform(future_x)[0]
        for column, value in zip(feature_columns, standardized * estimator.coef_):
            contribution_rows.append(
                {
                    "metal": "lithium_carbonate",
                    "forecast_period": month.strftime("%Y-%m"),
                    "horizon_type": "monthly",
                    "factor": column,
                    "factor_category": _factor_category(column),
                    "contribution": float(value),
                    "direction": "支撑" if value > 0 else "压制" if value < 0 else "中性",
                    "source_period": factor_row["source_period"].strftime("%Y-%m"),
                    "model_version": MONTHLY_MODEL_VERSION,
                    "generated_at": generated_at,
                }
            )
        returns.append(predicted_return)
    coefficients = pd.DataFrame(
        {
            "变量": feature_columns,
            "系数": estimator.coef_,
            "模型版本": MONTHLY_MODEL_VERSION,
        }
    )
    diagnostics = ModelDiagnostics(
        selected_model="elastic_net",
        model_mae=float(model_mae),
        baseline_mae=float(baseline_mae),
        improvement_pct=float(improvement),
        residual_std=residual_std,
    )
    return (
        pd.DataFrame(rows),
        model,
        diagnostics,
        coefficients,
        pd.DataFrame(contribution_rows),
    )


def _monthly_baseline(
    target: pd.DataFrame, periods: int, reason: str
) -> tuple[pd.DataFrame, None, ModelDiagnostics, pd.DataFrame, pd.DataFrame]:
    future_months = pd.date_range(
        target["month"].max() + pd.offsets.MonthBegin(1), periods=periods, freq="MS"
    )
    last_price = float(target.iloc[-1]["price"])
    residual_std = float(target["target_return"].dropna().tail(24).std(ddof=0))
    if not np.isfinite(residual_std) or residual_std <= 0:
        residual_std = 0.05
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    rows = []
    for horizon, month in enumerate(future_months, start=1):
        width = 1.282 * residual_std * np.sqrt(horizon)
        rows.append(
            {
                "metal": "lithium_carbonate",
                "forecast_month": month,
                "predicted_price_cny_per_tonne": last_price,
                "lower_bound": max(last_price * (1 - width), 1.0),
                "upper_bound": last_price * (1 + width),
                "direction": "平稳",
                "predicted_change_pct": 0.0,
                "source": reason,
                "model_version": MONTHLY_MODEL_VERSION,
                "generated_at": generated_at,
            }
        )
    diagnostics = ModelDiagnostics(
        selected_model="naive",
        model_mae=np.nan,
        baseline_mae=np.nan,
        improvement_pct=0.0,
        residual_std=residual_std,
    )
    coefficients = pd.DataFrame(
        [{"变量": "价格动量基准", "系数": 0.0, "模型版本": MONTHLY_MODEL_VERSION}]
    )
    return pd.DataFrame(rows), None, diagnostics, coefficients, pd.DataFrame()


def _daily_forecast(
    prices: pd.DataFrame, monthly_forecast: pd.DataFrame, periods: int
) -> tuple[pd.DataFrame, ModelDiagnostics]:
    history = prices.copy()
    history["return_1d"] = history["settlement_price"].pct_change()
    history["return_lag_1"] = history["return_1d"].shift(1)
    history["return_lag_2"] = history["return_1d"].shift(2)
    history["return_lag_5"] = history["return_1d"].shift(5)
    history["ma5_gap"] = history["settlement_price"] / history["settlement_price"].rolling(5).mean() - 1
    history["ma20_gap"] = history["settlement_price"] / history["settlement_price"].rolling(20).mean() - 1
    history["volatility_20d"] = history["return_1d"].rolling(20).std(ddof=0)
    history["volume_change"] = history["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    history["open_interest_change_pct"] = (
        history["open_interest"].pct_change().replace([np.inf, -np.inf], np.nan)
    )
    feature_columns = [
        "return_lag_1",
        "return_lag_2",
        "return_lag_5",
        "ma5_gap",
        "ma20_gap",
        "volatility_20d",
        "volume_change",
        "open_interest_change_pct",
    ]
    model_frame = history[["date", "settlement_price", "return_1d", *feature_columns]].dropna()
    x = model_frame[feature_columns].clip(-5, 5)
    y = model_frame["return_1d"].clip(-0.2, 0.2)
    cv = TimeSeriesSplit(n_splits=5)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", ElasticNet(max_iter=20000, random_state=42)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {
            "model__alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        cv=cv,
        scoring="neg_mean_absolute_error",
    )
    search.fit(x, y)
    predictions: list[float] = []
    actual: list[float] = []
    for train_idx, test_idx in cv.split(x):
        fold = search.best_estimator_
        fold.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions.extend(fold.predict(x.iloc[test_idx]).tolist())
        actual.extend(y.iloc[test_idx].tolist())
    model_mae = mean_absolute_error(actual, predictions)
    baseline_mae = mean_absolute_error(actual, np.zeros(len(actual)))
    improvement = 1 - model_mae / baseline_mae if baseline_mae > 0 else 0.0
    residual_std = float(np.std(np.asarray(actual) - np.asarray(predictions), ddof=0))
    use_model = improvement >= 0.05
    model = search.best_estimator_.fit(x, y) if use_model else None
    price_path = history["settlement_price"].astype(float).tolist()
    return_path = history["return_1d"].fillna(0.0).astype(float).tolist()
    future_dates = pd.bdate_range(history["date"].max() + pd.offsets.BDay(1), periods=periods)
    rows: list[dict[str, object]] = []
    last_volume = float(history["volume"].iloc[-1])
    last_open_interest = float(history["open_interest"].iloc[-1])
    for horizon, forecast_date in enumerate(future_dates, start=1):
        price_series = pd.Series(price_path)
        values = {
            "return_lag_1": return_path[-1],
            "return_lag_2": return_path[-2],
            "return_lag_5": return_path[-5],
            "ma5_gap": price_path[-1] / price_series.tail(5).mean() - 1,
            "ma20_gap": price_path[-1] / price_series.tail(20).mean() - 1,
            "volatility_20d": float(pd.Series(return_path).tail(20).std(ddof=0)),
            "volume_change": 0.0 if last_volume else 0.0,
            "open_interest_change_pct": 0.0 if last_open_interest else 0.0,
        }
        predicted_return = float(model.predict(pd.DataFrame([values]))[0]) if model else 0.0
        predicted_return = float(np.clip(predicted_return, -0.05, 0.05))
        predicted_price = max(price_path[-1] * (1 + predicted_return), 1.0)
        price_path.append(predicted_price)
        return_path.append(predicted_return)
        rows.append(
            {
                "forecast_date": forecast_date,
                "predicted_price_cny_per_tonne": predicted_price,
                "predicted_return": predicted_return,
            }
        )
    result = pd.DataFrame(rows)
    result["month"] = result["forecast_date"].dt.to_period("M").dt.to_timestamp()
    monthly_targets = monthly_forecast.set_index("forecast_month")[
        "predicted_price_cny_per_tonne"
    ]
    for month, group in result.groupby("month"):
        if month in monthly_targets.index:
            target = float(monthly_targets.loc[month])
            mask = result["month"] == month
            shift = target - float(group["predicted_price_cny_per_tonne"].mean())
            result.loc[mask, "predicted_price_cny_per_tonne"] += shift
    steps = np.sqrt(np.arange(1, len(result) + 1))
    widths = 1.282 * residual_std * steps
    result["lower_bound"] = (
        result["predicted_price_cny_per_tonne"] * (1 - widths)
    ).clip(lower=1.0)
    result["upper_bound"] = result["predicted_price_cny_per_tonne"] * (1 + widths)
    last_actual = float(history.iloc[-1]["settlement_price"])
    result["direction"] = result["predicted_price_cny_per_tonne"].map(
        lambda value: _direction(value / last_actual - 1)
    )
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    result["metal"] = "lithium_carbonate"
    result["model_version"] = DAILY_MODEL_VERSION
    result["generated_at"] = generated_at
    diagnostics = ModelDiagnostics(
        selected_model="elastic_net" if use_model else "random_walk",
        model_mae=float(model_mae),
        baseline_mae=float(baseline_mae),
        improvement_pct=float(improvement),
        residual_std=residual_std,
    )
    return result, diagnostics


def _normalize_daily_prices(data: pd.DataFrame) -> pd.DataFrame:
    renamed = data.rename(
        columns={
            "trade_date": "date",
            "price_cny_per_tonne": "settlement_price",
        }
    ).copy()
    renamed["date"] = pd.to_datetime(renamed["date"], errors="coerce")
    for column in ["settlement_price", "volume", "open_interest"]:
        if column not in renamed:
            renamed[column] = 0.0
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return renamed.dropna(subset=["date", "settlement_price"]).sort_values("date")


def _normalize_monthly_factors(data: pd.DataFrame) -> pd.DataFrame:
    renamed = data.rename(columns={data.columns[0]: "month"}).copy()
    renamed["month"] = pd.to_datetime(renamed["month"], errors="coerce")
    for column in MONTHLY_FACTORS:
        if column in renamed:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    return renamed.dropna(subset=["month"]).sort_values("month")


def _seasonal_factor_scenario(
    factors: pd.DataFrame, columns: list[str], future_months: pd.DatetimeIndex
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    indexed = factors.set_index("month")
    for month in future_months:
        row: dict[str, object] = {"month": month}
        source_periods: list[pd.Timestamp] = []
        for column in columns:
            series = indexed[column].dropna()
            seasonal_month = month - pd.DateOffset(years=1)
            if seasonal_month in series.index:
                row[column] = float(series.loc[seasonal_month])
                source_periods.append(pd.Timestamp(seasonal_month))
            elif not series.empty:
                row[column] = float(series.iloc[-1])
                source_periods.append(pd.Timestamp(series.index[-1]))
            else:
                row[column] = 0.0
        row["source_period"] = max(source_periods) if source_periods else month
        rows.append(row)
    return pd.DataFrame(rows)


def _factor_category(factor: str) -> str:
    if factor.startswith("return_lag"):
        return "价格动量"
    if factor.startswith("LC"):
        return "碳酸锂期货结构"
    if "汽车" in factor:
        return "需求"
    if "PMI" in factor or "工业增加值" in factor:
        return "宏观"
    return "成本"


def _direction(change: float) -> str:
    if change > 0.005:
        return "上涨"
    if change < -0.005:
        return "下跌"
    return "平稳"
