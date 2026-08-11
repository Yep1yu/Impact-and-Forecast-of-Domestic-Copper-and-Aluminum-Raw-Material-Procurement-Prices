from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "ccmn_changjiang_avg_prices.csv"
MODEL_VERSION = "daily-ensemble-v1"
DRIFT_WINDOWS = [5, 20, 60, 120]
ARIMA_ORDERS = [
    (0, 1, 0),
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (0, 1, 2),
    (2, 1, 1),
    (1, 1, 2),
]
FEATURE_COLUMNS = [
    "price",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_5",
    "lag_10",
    "lag_20",
    "lag_30",
    "lag_60",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_60",
    "ma_120",
    "return_1",
    "return_5",
    "return_20",
    "ma_gap_5",
    "ma_gap_10",
    "ma_gap_20",
    "volatility_10",
    "volatility_20",
    "log_drift_5",
    "log_drift_20",
    "log_drift_60",
    "log_drift_120",
    "volume_change_context",
    "open_interest_change_context",
    "contract_roll_context",
]


@dataclass(frozen=True)
class Args:
    input: Path
    output_dir: Path
    forecast_days: int
    validation_origins: int
    min_train_days: int
    retrain_step: int
    ridge_alpha: float
    no_arima: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Build a 5-series daily 30-day robust ensemble price model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Wide CSV with date + product columns.")
    parser.add_argument("--output-dir", type=Path, default=ROOT, help="Directory for output CSV/JSON files.")
    parser.add_argument("--forecast-days", type=int, default=30, help="Forecast horizon count. Default: 30.")
    parser.add_argument(
        "--validation-origins",
        type=int,
        default=120,
        help="Number of rolling origins used for backtest weighting. Default: 120.",
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=500,
        help="Minimum historical observations before a validation origin. Default: 500.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=20.0, help="Ridge alpha. Default: 20.")
    parser.add_argument(
        "--retrain-step",
        type=int,
        default=20,
        help="Retrain Ridge/ARIMA candidates every N validation origins. Default: 20.",
    )
    parser.add_argument("--no-arima", action="store_true", help="Skip ARIMA candidate.")
    ns = parser.parse_args()
    return Args(
        input=ns.input,
        output_dir=ns.output_dir,
        forecast_days=ns.forecast_days,
        validation_origins=ns.validation_origins,
        min_train_days=ns.min_train_days,
        retrain_step=max(int(ns.retrain_step), 1),
        ridge_alpha=ns.ridge_alpha,
        no_arima=ns.no_arima,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    series_map = load_price_series(args.input)
    all_metric_rows: list[dict[str, object]] = []
    all_weight_rows: list[dict[str, object]] = []
    all_detail_rows: list[dict[str, object]] = []
    all_forecast_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "input_file": str(args.input),
        "forecast_days": args.forecast_days,
        "validation_origins": args.validation_origins,
        "min_train_days": args.min_train_days,
        "retrain_step": args.retrain_step,
        "candidate_models": candidate_model_names(args.no_arima),
        "outputs": {},
        "series_summary": [],
    }

    for series_name, frame in series_map.items():
        print(f"Modeling {series_name} ({len(frame)} rows)...", flush=True)
        result = model_one_series(series_name, frame, args)
        all_metric_rows.extend(result["metrics"])
        all_weight_rows.extend(result["weights"])
        all_detail_rows.extend(result["details"])
        all_forecast_rows.extend(result["forecast"])
        summary["series_summary"].append(result["summary"])

    output_paths = {
        "forecast": args.output_dir / "daily_ensemble_forecast_next_30d.csv",
        "backtest_by_horizon": args.output_dir / "daily_ensemble_backtest_by_horizon.csv",
        "model_weights": args.output_dir / "daily_ensemble_model_weights.csv",
        "actual_vs_pred": args.output_dir / "daily_ensemble_actual_vs_pred.csv",
        "summary": args.output_dir / "daily_ensemble_summary.json",
    }
    pd.DataFrame(all_forecast_rows).to_csv(output_paths["forecast"], index=False, encoding="utf-8-sig")
    pd.DataFrame(all_metric_rows).to_csv(
        output_paths["backtest_by_horizon"], index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_weight_rows).to_csv(output_paths["model_weights"], index=False, encoding="utf-8-sig")
    pd.DataFrame(all_detail_rows).to_csv(output_paths["actual_vs_pred"], index=False, encoding="utf-8-sig")

    summary["outputs"] = {key: str(path) for key, path in output_paths.items()}
    output_paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Done.")
    for key, path in output_paths.items():
        print(f"{key}: {path}")


def load_price_series(path: Path) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    if "date" not in raw.columns:
        raise ValueError("input CSV must contain a date column")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date")
    series_map: dict[str, pd.DataFrame] = {}
    for column in raw.columns:
        if column == "date":
            continue
        frame = raw[["date", column]].rename(columns={column: "price"}).copy()
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame = frame.dropna(subset=["price"])
        frame = frame[frame["price"] > 0].drop_duplicates("date", keep="last").reset_index(drop=True)
        if len(frame) >= 100:
            series_map[column] = frame
    if not series_map:
        raise ValueError("no usable price series found")
    return series_map


def candidate_model_names(no_arima: bool) -> list[str]:
    names = ["Naive_last", *[f"Rolling_log_drift_{window}d" for window in DRIFT_WINDOWS], "Ridge_direct_h"]
    if not no_arima:
        names.append("ARIMA_log")
    names.append("Median_ensemble")
    return names


def model_one_series(series_name: str, frame: pd.DataFrame, args: Args) -> dict[str, object]:
    prices = frame["price"].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(frame["date"]).reset_index(drop=True)
    features = build_features(prices, frame)
    arima_spec = None if args.no_arima else select_arima_spec(prices)
    origin_indices = validation_origin_indices(len(frame), args.forecast_days, args.validation_origins, args.min_train_days)
    if not origin_indices:
        raise ValueError(f"{series_name} has no valid rolling validation origins")

    base_detail = build_backtest_predictions(
        series_name=series_name,
        dates=dates,
        prices=prices,
        features=features,
        origin_indices=origin_indices,
        args=args,
        arima_spec=arima_spec,
    )
    base_metrics = metric_rows_from_details(base_detail)
    weights = weight_rows_from_metrics(base_metrics, series_name, args.forecast_days)
    final_detail = add_final_ensemble_predictions(base_detail, weights)
    metrics = metric_rows_from_details(final_detail)
    forecast = build_future_forecast(series_name, dates, prices, features, weights, final_detail, args, arima_spec)
    summary = summarize_series(series_name, frame, metrics, forecast, arima_spec, origin_indices)
    return {
        "metrics": metrics.to_dict("records"),
        "weights": weights.to_dict("records"),
        "details": final_detail.to_dict("records"),
        "forecast": forecast.to_dict("records"),
        "summary": summary,
    }


def validation_origin_indices(n_rows: int, forecast_days: int, requested_origins: int, min_train_days: int) -> list[int]:
    last_origin = n_rows - forecast_days - 1
    if last_origin < min_train_days:
        return []
    first_origin = max(min_train_days, last_origin - requested_origins + 1)
    return list(range(first_origin, last_origin + 1))


def build_features(prices: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"price": prices.astype(float)})
    log_price = np.log(frame["price"])
    for lag in [1, 2, 3, 5, 10, 20, 30, 60]:
        frame[f"lag_{lag}"] = frame["price"].shift(lag)
    for window in [5, 10, 20, 60, 120]:
        ma = frame["price"].rolling(window, min_periods=max(2, window // 2)).mean()
        frame[f"ma_{window}"] = ma
        if window in [5, 10, 20]:
            frame[f"ma_gap_{window}"] = frame["price"] / ma - 1
    for window in [5, 20, 60, 120]:
        frame[f"log_drift_{window}"] = (log_price - log_price.shift(window)) / window
    frame["return_1"] = frame["price"].pct_change(fill_method=None)
    frame["return_5"] = frame["price"].pct_change(5, fill_method=None)
    frame["return_20"] = frame["price"].pct_change(20, fill_method=None)
    frame["volatility_10"] = frame["return_1"].rolling(10, min_periods=5).std()
    frame["volatility_20"] = frame["return_1"].rolling(20, min_periods=10).std()
    if context is not None:
        volume = pd.to_numeric(context.get("volume", pd.Series(index=prices.index)), errors="coerce")
        open_interest = pd.to_numeric(
            context.get("open_interest", pd.Series(index=prices.index)), errors="coerce"
        )
        frame["volume_change_context"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
        frame["open_interest_change_context"] = open_interest.pct_change().replace(
            [np.inf, -np.inf], np.nan
        )
        contracts = context.get("contract")
        if contracts is not None:
            frame["contract_roll_context"] = pd.Series(contracts).astype(str).ne(
                pd.Series(contracts).astype(str).shift(1)
            ).astype(float)
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame


def build_backtest_predictions(
    *,
    series_name: str,
    dates: pd.Series,
    prices: pd.Series,
    features: pd.DataFrame,
    origin_indices: list[int],
    args: Args,
    arima_spec: tuple[tuple[int, int, int], str] | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    arima_paths = arima_backtest_paths(prices, origin_indices, args.forecast_days, arima_spec, args.retrain_step)
    ridge_preds = ridge_backtest_predictions(features, prices, origin_indices, args)
    for origin_index in origin_indices:
        origin_date = dates.iloc[origin_index]
        current_price = float(prices.iloc[origin_index])
        base_path: dict[str, list[float | None]] = {
            "Naive_last": [current_price for _ in range(args.forecast_days)]
        }
        for window in DRIFT_WINDOWS:
            base_path[f"Rolling_log_drift_{window}d"] = drift_path(prices, origin_index, args.forecast_days, window)
        if arima_spec is not None:
            base_path["ARIMA_log"] = arima_paths.get(origin_index, [None for _ in range(args.forecast_days)])
        for horizon in range(1, args.forecast_days + 1):
            if "Ridge_direct_h" not in base_path:
                base_path["Ridge_direct_h"] = [None for _ in range(args.forecast_days)]
            base_path["Ridge_direct_h"][horizon - 1] = ridge_preds.get((origin_index, horizon))

        median_path = []
        for horizon in range(1, args.forecast_days + 1):
            values = [
                value[horizon - 1]
                for model, value in base_path.items()
                if model != "Naive_last" and value[horizon - 1] is not None and np.isfinite(value[horizon - 1])
            ]
            median_path.append(float(np.median(values)) if values else current_price)
        base_path["Median_ensemble"] = median_path

        for horizon in range(1, args.forecast_days + 1):
            target_index = origin_index + horizon
            actual = float(prices.iloc[target_index])
            for model, path in base_path.items():
                pred = path[horizon - 1]
                if pred is None or not np.isfinite(pred):
                    continue
                pred = float(max(pred, 1.0))
                rows.append(
                    {
                        "series": series_name,
                        "origin_date": origin_date.strftime("%Y-%m-%d"),
                        "target_date": dates.iloc[target_index].strftime("%Y-%m-%d"),
                        "horizon": horizon,
                        "model": model,
                        "actual_price": actual,
                        "predicted_price": pred,
                        "error": pred - actual,
                        "abs_pct_error": abs(pred - actual) / abs(actual) * 100 if actual else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def drift_path(prices: pd.Series, origin_index: int, forecast_days: int, window: int) -> list[float | None]:
    if origin_index < window:
        return [None for _ in range(forecast_days)]
    current = float(prices.iloc[origin_index])
    previous = float(prices.iloc[origin_index - window])
    if current <= 0 or previous <= 0:
        return [None for _ in range(forecast_days)]
    daily_drift = (math.log(current) - math.log(previous)) / window
    return [math.exp(math.log(current) + daily_drift * horizon) for horizon in range(1, forecast_days + 1)]


def ridge_predict_horizon(
    features: pd.DataFrame,
    prices: pd.Series,
    origin_index: int,
    horizon: int,
    args: Args,
) -> float | None:
    train_last = origin_index - horizon
    if train_last < args.min_train_days:
        return None
    x_train = features.loc[:train_last, FEATURE_COLUMNS].copy()
    y_train = prices.shift(-horizon).loc[:train_last].copy()
    usable = x_train.notna().all(axis=1) & y_train.notna()
    x_train = x_train.loc[usable]
    y_train = y_train.loc[usable]
    if len(x_train) < args.min_train_days:
        return None
    model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
    try:
        model.fit(x_train, y_train)
        pred = float(model.predict(features.loc[[origin_index], FEATURE_COLUMNS])[0])
    except Exception:
        return None
    current = float(prices.iloc[origin_index])
    max_move = current * min(0.30, 0.06 * math.sqrt(horizon))
    return float(np.clip(pred, current - max_move, current + max_move))


def ridge_backtest_predictions(
    features: pd.DataFrame,
    prices: pd.Series,
    origin_indices: list[int],
    args: Args,
) -> dict[tuple[int, int], float | None]:
    predictions: dict[tuple[int, int], float | None] = {}
    for horizon in range(1, args.forecast_days + 1):
        for batch in chunked(origin_indices, args.retrain_step):
            train_last = batch[0] - horizon
            if train_last < args.min_train_days:
                for origin_index in batch:
                    predictions[(origin_index, horizon)] = None
                continue
            x_train = features.loc[:train_last, FEATURE_COLUMNS].copy()
            y_train = prices.shift(-horizon).loc[:train_last].copy()
            usable = x_train.notna().all(axis=1) & y_train.notna()
            x_train = x_train.loc[usable]
            y_train = y_train.loc[usable]
            if len(x_train) < args.min_train_days:
                for origin_index in batch:
                    predictions[(origin_index, horizon)] = None
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
            try:
                model.fit(x_train, y_train)
                batch_pred = model.predict(features.loc[batch, FEATURE_COLUMNS])
            except Exception:
                for origin_index in batch:
                    predictions[(origin_index, horizon)] = None
                continue
            for origin_index, pred in zip(batch, batch_pred):
                current = float(prices.iloc[origin_index])
                max_move = current * min(0.30, 0.06 * math.sqrt(horizon))
                predictions[(origin_index, horizon)] = float(np.clip(float(pred), current - max_move, current + max_move))
    return predictions


def select_arima_spec(prices: pd.Series) -> tuple[tuple[int, int, int], str] | None:
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        return None
    log_series = np.log(prices.astype(float))
    best: tuple[float, tuple[int, int, int], str] | None = None
    sample = log_series.iloc[-min(len(log_series), 900) :]
    for order in ARIMA_ORDERS:
        for trend in ["n", "t"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit = ARIMA(
                        sample,
                        order=order,
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                except Exception:
                    continue
            aic = float(fit.aic)
            if best is None or aic < best[0]:
                best = (aic, order, trend)
    if best is None:
        return None
    return best[1], best[2]


def arima_backtest_paths(
    prices: pd.Series,
    origin_indices: Iterable[int],
    forecast_days: int,
    arima_spec: tuple[tuple[int, int, int], str] | None,
    retrain_step: int,
) -> dict[int, list[float | None]]:
    if arima_spec is None:
        return {}
    paths: dict[int, list[float | None]] = {}
    origins = list(origin_indices)
    for batch in chunked(origins, retrain_step):
        anchor = batch[0]
        anchor_path = arima_forecast_path(prices.iloc[: anchor + 1], forecast_days + len(batch) - 1, arima_spec)
        for offset, origin_index in enumerate(batch):
            shifted = anchor_path[offset : offset + forecast_days]
            if len(shifted) < forecast_days:
                shifted = shifted + [None for _ in range(forecast_days - len(shifted))]
            paths[origin_index] = shifted
    return paths


def arima_forecast_path(
    history: pd.Series,
    forecast_days: int,
    arima_spec: tuple[tuple[int, int, int], str] | None,
) -> list[float | None]:
    if arima_spec is None:
        return [None for _ in range(forecast_days)]
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        return [None for _ in range(forecast_days)]
    order, trend = arima_spec
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ARIMA(
                np.log(history.astype(float)),
                order=order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
            forecast = np.exp(fit.forecast(forecast_days))
        except Exception:
            return [None for _ in range(forecast_days)]
    return [float(value) if np.isfinite(value) else None for value in forecast]


def chunked(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def metric_rows_from_details(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (series, horizon, model), group in details.groupby(["series", "horizon", "model"], sort=True):
        actual = group["actual_price"].astype(float)
        pred = group["predicted_price"].astype(float)
        err = pred - actual
        abs_err = err.abs()
        rows.append(
            {
                "series": series,
                "horizon": int(horizon),
                "model": model,
                "MAE": float(abs_err.mean()),
                "RMSE": float(np.sqrt(np.mean(np.square(err)))),
                "MAPE_%": float((abs_err / actual.abs()).mean() * 100),
                "Bias": float(err.mean()),
                "n_obs": int(len(group)),
            }
        )
    metrics = pd.DataFrame(rows)
    naive = metrics[metrics["model"] == "Naive_last"][
        ["series", "horizon", "MAE", "RMSE", "MAPE_%"]
    ].rename(columns={"MAE": "naive_MAE", "RMSE": "naive_RMSE", "MAPE_%": "naive_MAPE_%"})
    metrics = metrics.merge(naive, on=["series", "horizon"], how="left")
    metrics["beats_naive"] = metrics["MAE"] < metrics["naive_MAE"]
    return metrics.sort_values(["series", "horizon", "MAE", "model"]).reset_index(drop=True)


def weight_rows_from_metrics(metrics: pd.DataFrame, series_name: str, forecast_days: int) -> pd.DataFrame:
    rows = []
    for horizon in range(1, forecast_days + 1):
        table = metrics[(metrics["series"] == series_name) & (metrics["horizon"] == horizon)].copy()
        table = table[table["model"] != "Final_ensemble"]
        if table.empty:
            continue
        naive_mae = float(table.loc[table["model"] == "Naive_last", "MAE"].iloc[0])
        eligible = table[(table["MAE"] <= naive_mae) & np.isfinite(table["MAE"])].copy()
        if eligible.empty:
            eligible = table[table["model"] == "Naive_last"].copy()
        if "Naive_last" not in set(eligible["model"]):
            eligible = pd.concat([eligible, table[table["model"] == "Naive_last"]], ignore_index=True)
        eligible["score"] = 1.0 / np.maximum(eligible["MAE"].astype(float), 1e-9) ** 2
        total_score = float(eligible["score"].sum())
        for _, row in table.iterrows():
            match = eligible[eligible["model"] == row["model"]]
            weight = float(match["score"].iloc[0] / total_score) if not match.empty and total_score > 0 else 0.0
            rows.append(
                {
                    "series": series_name,
                    "horizon": horizon,
                    "model": row["model"],
                    "weight": weight,
                    "eligible": bool(weight > 0),
                    "model_MAE": float(row["MAE"]),
                    "naive_MAE": naive_mae,
                    "beats_naive": bool(row["MAE"] < naive_mae),
                }
            )
    return pd.DataFrame(rows).sort_values(["series", "horizon", "model"]).reset_index(drop=True)


def add_final_ensemble_predictions(details: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    final_rows = []
    weight_lookup = {
        (row.series, int(row.horizon), row.model): float(row.weight)
        for row in weights.itertuples(index=False)
        if float(row.weight) > 0
    }
    for key, group in details.groupby(["series", "origin_date", "target_date", "horizon"], sort=False):
        series, origin_date, target_date, horizon = key
        weighted_sum = 0.0
        weight_sum = 0.0
        for row in group.itertuples(index=False):
            weight = weight_lookup.get((series, int(horizon), row.model), 0.0)
            if weight <= 0:
                continue
            weighted_sum += weight * float(row.predicted_price)
            weight_sum += weight
        if weight_sum <= 0:
            fallback = group[group["model"] == "Naive_last"].iloc[0]
            pred = float(fallback["predicted_price"])
        else:
            pred = weighted_sum / weight_sum
        actual = float(group["actual_price"].iloc[0])
        final_rows.append(
            {
                "series": series,
                "origin_date": origin_date,
                "target_date": target_date,
                "horizon": int(horizon),
                "model": "Final_ensemble",
                "actual_price": actual,
                "predicted_price": pred,
                "error": pred - actual,
                "abs_pct_error": abs(pred - actual) / abs(actual) * 100 if actual else np.nan,
            }
        )
    return pd.concat([details, pd.DataFrame(final_rows)], ignore_index=True)


def build_future_forecast(
    series_name: str,
    dates: pd.Series,
    prices: pd.Series,
    features: pd.DataFrame,
    weights: pd.DataFrame,
    final_detail: pd.DataFrame,
    args: Args,
    arima_spec: tuple[tuple[int, int, int], str] | None,
) -> pd.DataFrame:
    last_date = pd.Timestamp(dates.iloc[-1])
    current = float(prices.iloc[-1])
    future_dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=args.forecast_days)
    base_path: dict[str, list[float | None]] = {"Naive_last": [current for _ in range(args.forecast_days)]}
    for window in DRIFT_WINDOWS:
        base_path[f"Rolling_log_drift_{window}d"] = drift_path(prices, len(prices) - 1, args.forecast_days, window)
    if arima_spec is not None:
        base_path["ARIMA_log"] = arima_forecast_path(prices, args.forecast_days, arima_spec)
    ridge_path = []
    for horizon in range(1, args.forecast_days + 1):
        ridge_path.append(ridge_predict_horizon(features, prices, len(prices) - 1, horizon, args))
    base_path["Ridge_direct_h"] = ridge_path
    base_path["Median_ensemble"] = [
        float(
            np.median(
                [
                    path[horizon - 1]
                    for model, path in base_path.items()
                    if model != "Naive_last"
                    and path[horizon - 1] is not None
                    and np.isfinite(path[horizon - 1])
                ]
                or [current]
            )
        )
        for horizon in range(1, args.forecast_days + 1)
    ]

    weight_lookup = {
        (int(row.horizon), row.model): float(row.weight)
        for row in weights[weights["series"] == series_name].itertuples(index=False)
    }
    metric_table = metric_rows_from_details(final_detail)
    final_metrics = metric_table[
        (metric_table["series"] == series_name) & (metric_table["model"] == "Final_ensemble")
    ].set_index("horizon")
    best_model_by_horizon = (
        metric_table[(metric_table["series"] == series_name) & (metric_table["model"] != "Final_ensemble")]
        .sort_values(["horizon", "MAE"])
        .drop_duplicates("horizon")
        .set_index("horizon")["model"]
        .to_dict()
    )

    rows = []
    residual_table = final_detail[
        (final_detail["series"] == series_name) & (final_detail["model"] == "Final_ensemble")
    ]
    for horizon in range(1, args.forecast_days + 1):
        weighted_sum = 0.0
        weight_sum = 0.0
        for model, path in base_path.items():
            pred = path[horizon - 1]
            if pred is None or not np.isfinite(pred):
                continue
            weight = weight_lookup.get((horizon, model), 0.0)
            if weight <= 0:
                continue
            weighted_sum += weight * float(pred)
            weight_sum += weight
        pred = weighted_sum / weight_sum if weight_sum > 0 else current
        residuals = -residual_table[residual_table["horizon"] == horizon]["error"].astype(float)
        if residuals.empty:
            q05 = q10 = -abs(pred * 0.03)
            q90 = q95 = abs(pred * 0.03)
        else:
            q05, q10, q90, q95 = np.percentile(residuals, [5, 10, 90, 95])
        p05, p10, p90, p95 = pred + q05, pred + q10, pred + q90, pred + q95
        rows.append(
            {
                "series": series_name,
                "forecast_date": future_dates[horizon - 1].strftime("%Y-%m-%d"),
                "horizon": horizon,
                "predicted_price": round(float(pred), 3),
                "p10": round(float(max(min(p10, p90), 1.0)), 3),
                "p90": round(float(max(max(p10, p90), 1.0)), 3),
                "p05": round(float(max(min(p05, p95), 1.0)), 3),
                "p95": round(float(max(max(p05, p95), 1.0)), 3),
                "best_base_model": best_model_by_horizon.get(horizon, ""),
                "final_MAE": float(final_metrics.loc[horizon, "MAE"]) if horizon in final_metrics.index else np.nan,
                "naive_MAE": float(final_metrics.loc[horizon, "naive_MAE"]) if horizon in final_metrics.index else np.nan,
                "beats_naive": bool(final_metrics.loc[horizon, "beats_naive"]) if horizon in final_metrics.index else False,
                "model_version": MODEL_VERSION,
            }
        )
    return pd.DataFrame(rows)


def summarize_series(
    series_name: str,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    forecast: pd.DataFrame,
    arima_spec: tuple[tuple[int, int, int], str] | None,
    origin_indices: list[int],
) -> dict[str, object]:
    final = metrics[(metrics["series"] == series_name) & (metrics["model"] == "Final_ensemble")]
    naive = metrics[(metrics["series"] == series_name) & (metrics["model"] == "Naive_last")]
    checkpoints = {}
    for horizon in [1, 7, 14, 30]:
        if horizon in set(final["horizon"]):
            f_row = final[final["horizon"] == horizon].iloc[0]
            n_row = naive[naive["horizon"] == horizon].iloc[0]
            checkpoints[str(horizon)] = {
                "final_MAE": float(f_row["MAE"]),
                "naive_MAE": float(n_row["MAE"]),
                "final_MAPE_%": float(f_row["MAPE_%"]),
                "naive_MAPE_%": float(n_row["MAPE_%"]),
                "beats_naive": bool(f_row["MAE"] < n_row["MAE"]),
            }
    return {
        "series": series_name,
        "sample_start": pd.Timestamp(frame["date"].min()).strftime("%Y-%m-%d"),
        "sample_end": pd.Timestamp(frame["date"].max()).strftime("%Y-%m-%d"),
        "sample_rows": int(len(frame)),
        "validation_origin_count": int(len(origin_indices)),
        "validation_start_origin": pd.Timestamp(frame["date"].iloc[origin_indices[0]]).strftime("%Y-%m-%d"),
        "validation_end_origin": pd.Timestamp(frame["date"].iloc[origin_indices[-1]]).strftime("%Y-%m-%d"),
        "arima_spec": {"order": arima_spec[0], "trend": arima_spec[1]} if arima_spec else None,
        "mean_final_MAE": float(final["MAE"].mean()),
        "mean_naive_MAE": float(naive["MAE"].mean()),
        "beats_naive_horizon_count": int((final["MAE"].values < naive["MAE"].values).sum()),
        "checkpoint_horizons": checkpoints,
        "forecast_start": str(forecast["forecast_date"].min()),
        "forecast_end": str(forecast["forecast_date"].max()),
    }


if __name__ == "__main__":
    main()
