from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MONTHLY_DIR = ROOT / "monthly_price_prediction_outputs"
DEFAULT_DAILY_INPUT = ROOT / "ccmn_changjiang_avg_prices.csv"
DEFAULT_ENSEMBLE_FORECAST = ROOT / "daily_ensemble_forecast_next_30d.csv"
DEFAULT_ENSEMBLE_BACKTEST = ROOT / "daily_ensemble_actual_vs_pred.csv"
MODEL_VERSION = "daily-hybrid-variable-anchor-v1"

PRODUCT_ALIASES = {
    "ADC12": "铝合金ADC12",
    "ZLD104": "铸造铝合金锭(ZLD104)",
}


@dataclass(frozen=True)
class Args:
    daily_prices: Path
    ensemble_forecast: Path
    ensemble_backtest: Path
    monthly_dir: Path
    monthly_workbook: Path | None
    output_dir: Path
    anchor_strength: float


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Anchor daily 30-day ensemble forecasts to the variable-based monthly price model."
    )
    parser.add_argument("--daily-prices", type=Path, default=DEFAULT_DAILY_INPUT)
    parser.add_argument("--ensemble-forecast", type=Path, default=DEFAULT_ENSEMBLE_FORECAST)
    parser.add_argument("--ensemble-backtest", type=Path, default=DEFAULT_ENSEMBLE_BACKTEST)
    parser.add_argument("--monthly-dir", type=Path, default=MONTHLY_DIR)
    parser.add_argument(
        "--monthly-workbook",
        type=Path,
        help="Optional tax-exclusive monthly regression workbook. Uses its test and future forecast sheets.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--anchor-strength",
        type=float,
        default=0.75,
        help="0 keeps the daily ensemble unchanged; 1 fully matches the monthly anchor.",
    )
    ns = parser.parse_args()
    return Args(
        daily_prices=ns.daily_prices,
        ensemble_forecast=ns.ensemble_forecast,
        ensemble_backtest=ns.ensemble_backtest,
        monthly_dir=ns.monthly_dir,
        monthly_workbook=ns.monthly_workbook,
        output_dir=ns.output_dir,
        anchor_strength=float(np.clip(ns.anchor_strength, 0.0, 1.0)),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    daily_actual = load_daily_prices(args.daily_prices)
    monthly_anchors = load_monthly_anchors(args.monthly_dir, args.monthly_workbook)
    ensemble_forecast = pd.read_csv(args.ensemble_forecast, encoding="utf-8-sig")
    ensemble_backtest = pd.read_csv(args.ensemble_backtest, encoding="utf-8-sig")

    future = build_future_hybrid(ensemble_forecast, daily_actual, monthly_anchors, args.anchor_strength)
    backtest = build_backtest_hybrid(ensemble_backtest, daily_actual, monthly_anchors, args.anchor_strength)
    first_pass_metrics = build_metrics(backtest)
    backtest = add_selected_adaptive_model(backtest, first_pass_metrics)
    metrics = build_metrics(backtest)
    future = add_selected_forecast(future, first_pass_metrics)
    summary = build_summary(args, monthly_anchors, future, metrics)

    paths = {
        "forecast": args.output_dir / "daily_hybrid_variable_anchored_forecast_30d.csv",
        "backtest_by_horizon": args.output_dir / "daily_hybrid_backtest_by_horizon.csv",
        "actual_vs_pred": args.output_dir / "daily_hybrid_actual_vs_pred.csv",
        "summary": args.output_dir / "daily_hybrid_summary.json",
    }
    future.to_csv(paths["forecast"], index=False, encoding="utf-8-sig")
    metrics.to_csv(paths["backtest_by_horizon"], index=False, encoding="utf-8-sig")
    backtest.to_csv(paths["actual_vs_pred"], index=False, encoding="utf-8-sig")
    summary["outputs"] = {key: str(value) for key, value in paths.items()}
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    for key, path in paths.items():
        print(f"{key}: {path}")


def load_daily_prices(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    if "date" not in raw.columns:
        raise ValueError("daily price CSV must contain a date column")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    rows = []
    for column in raw.columns:
        if column == "date":
            continue
        part = raw[["date", column]].rename(columns={column: "price"}).copy()
        part["series"] = column
        part["price"] = pd.to_numeric(part["price"], errors="coerce")
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["date", "price"])
    out = out[out["price"] > 0]
    out["month"] = out["date"].dt.to_period("M").astype(str)
    return out[["series", "date", "month", "price"]].sort_values(["series", "date"])


def load_monthly_anchors(monthly_dir: Path, monthly_workbook: Path | None = None) -> pd.DataFrame:
    if monthly_workbook is not None:
        if not monthly_workbook.exists():
            raise FileNotFoundError(monthly_workbook)
        test = pd.read_excel(monthly_workbook, sheet_name="测试集预测对比")
        future = pd.read_excel(monthly_workbook, sheet_name="未来月份预测")
        test_source = "tax_exclusive_monthly_test_prediction"
        future_source = "tax_exclusive_monthly_future_prediction"
    else:
        test_path = monthly_dir / "monthly_price_model_test_actual_vs_pred.csv"
        if not test_path.exists():
            raise FileNotFoundError(test_path)
        test = pd.read_csv(test_path, encoding="utf-8-sig")
        future = pd.read_csv(find_monthly_future_file(monthly_dir), encoding="utf-8-sig")
        test_source = "monthly_test_prediction"
        future_source = "monthly_future_prediction"
    test = test.rename(columns={"品种": "series", "月份": "month", "预测月均价": "monthly_anchor"})
    test["anchor_source"] = test_source
    future = future.rename(columns={"品种": "series", "预测月份": "month", "预测月均价": "monthly_anchor"})
    future["anchor_source"] = future_source

    anchors = pd.concat(
        [
            test[["series", "month", "monthly_anchor", "anchor_source"]],
            future[["series", "month", "monthly_anchor", "anchor_source"]],
        ],
        ignore_index=True,
    )
    anchors["series"] = anchors["series"].map(normalize_product)
    anchors["month"] = pd.to_datetime(anchors["month"]).dt.to_period("M").astype(str)
    anchors["monthly_anchor"] = pd.to_numeric(anchors["monthly_anchor"], errors="coerce")
    anchors = anchors.dropna(subset=["series", "month", "monthly_anchor"])
    source_rank = {test_source: 0, future_source: 1}
    anchors["source_rank"] = anchors["anchor_source"].map(source_rank).fillna(9)
    anchors = anchors.sort_values(["series", "month", "source_rank"]).drop_duplicates(
        ["series", "month"], keep="first"
    )
    return anchors[["series", "month", "monthly_anchor", "anchor_source"]].reset_index(drop=True)


def find_monthly_future_file(monthly_dir: Path) -> Path:
    candidates = [p for p in monthly_dir.glob("*.csv") if "预测结果" in p.name]
    if not candidates:
        raise FileNotFoundError("Cannot find monthly variable forecast CSV containing '预测结果'")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def normalize_product(value: object) -> str:
    text = str(value).strip()
    return PRODUCT_ALIASES.get(text, text)


def build_future_hybrid(
    forecast: pd.DataFrame,
    daily_actual: pd.DataFrame,
    monthly_anchors: pd.DataFrame,
    anchor_strength: float,
) -> pd.DataFrame:
    frame = forecast.copy()
    frame["series"] = frame["series"].map(normalize_product)
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"])
    frame["month"] = frame["forecast_date"].dt.to_period("M").astype(str)
    frame = frame.merge(monthly_anchors, on=["series", "month"], how="left")

    adjusted_parts = []
    for (series, month), group in frame.groupby(["series", "month"], sort=False):
        group = group.sort_values("forecast_date").copy()
        anchor = first_valid(group["monthly_anchor"])
        if anchor is None:
            group = apply_no_anchor(group)
            adjusted_parts.append(group)
            continue
        origin_date = group["forecast_date"].min() - pd.offsets.BDay(1)
        target = remaining_month_target(series, month, origin_date, daily_actual, anchor)
        base_mean = float(group["predicted_price"].mean())
        delta = target - base_mean
        group = apply_delta(group, delta, target, "monthly_variable_anchor", anchor_strength)
        adjusted_parts.append(group)
    out = pd.concat(adjusted_parts, ignore_index=True)
    out["model_version"] = MODEL_VERSION
    out["forecast_date"] = out["forecast_date"].dt.strftime("%Y-%m-%d")
    return out[
        [
            "series",
            "forecast_date",
            "horizon",
            "hybrid_predicted_price",
            "ensemble_predicted_price",
            "monthly_anchor",
            "remaining_month_target",
            "anchor_adjustment",
            "anchor_source",
            "p10",
            "p90",
            "p05",
            "p95",
            "model_version",
        ]
    ]


def build_backtest_hybrid(
    details: pd.DataFrame,
    daily_actual: pd.DataFrame,
    monthly_anchors: pd.DataFrame,
    anchor_strength: float,
) -> pd.DataFrame:
    base = details[details["model"].isin(["Naive_last", "Final_ensemble"])].copy()
    base["series"] = base["series"].map(normalize_product)
    base["origin_date"] = pd.to_datetime(base["origin_date"])
    base["target_date"] = pd.to_datetime(base["target_date"])
    base["month"] = base["target_date"].dt.to_period("M").astype(str)
    base = base.merge(monthly_anchors, on=["series", "month"], how="left")

    hybrid_rows = []
    final = base[base["model"] == "Final_ensemble"].copy()
    for (series, origin_date, month), group in final.groupby(["series", "origin_date", "month"], sort=False):
        group = group.sort_values("target_date").copy()
        anchor = first_valid(group["monthly_anchor"])
        if anchor is None:
            target = np.nan
            delta = 0.0
            source = "missing_monthly_anchor"
        else:
            target = remaining_month_target(series, month, origin_date, daily_actual, anchor)
            delta = target - float(group["predicted_price"].mean())
            source = first_valid(group["anchor_source"]) or "monthly_variable_anchor"
        for row in group.itertuples(index=False):
            scaled_delta = horizon_anchor_strength(int(row.horizon), anchor_strength) * delta
            pred = max(float(row.predicted_price) + scaled_delta, 1.0)
            actual = float(row.actual_price)
            hybrid_rows.append(
                {
                    "series": series,
                    "origin_date": origin_date.strftime("%Y-%m-%d"),
                    "target_date": row.target_date.strftime("%Y-%m-%d"),
                    "horizon": int(row.horizon),
                    "model": "Hybrid_variable_anchor",
                    "actual_price": actual,
                    "predicted_price": pred,
                    "ensemble_predicted_price": float(row.predicted_price),
                    "monthly_anchor": anchor,
                    "remaining_month_target": target,
                    "anchor_adjustment": scaled_delta,
                    "anchor_source": source,
                    "error": pred - actual,
                    "abs_pct_error": abs(pred - actual) / abs(actual) * 100 if actual else np.nan,
                }
            )

    keep = base[base["model"].isin(["Naive_last", "Final_ensemble"])].copy()
    keep["ensemble_predicted_price"] = np.where(
        keep["model"] == "Final_ensemble", keep["predicted_price"], np.nan
    )
    keep["remaining_month_target"] = np.nan
    keep["anchor_adjustment"] = 0.0
    keep["anchor_source"] = keep["anchor_source"].fillna("missing_monthly_anchor")
    cols = [
        "series",
        "origin_date",
        "target_date",
        "horizon",
        "model",
        "actual_price",
        "predicted_price",
        "ensemble_predicted_price",
        "monthly_anchor",
        "remaining_month_target",
        "anchor_adjustment",
        "anchor_source",
        "error",
        "abs_pct_error",
    ]
    keep["origin_date"] = keep["origin_date"].dt.strftime("%Y-%m-%d")
    keep["target_date"] = keep["target_date"].dt.strftime("%Y-%m-%d")
    return pd.concat([keep[cols], pd.DataFrame(hybrid_rows)[cols]], ignore_index=True)


def remaining_month_target(
    series: str,
    month: str,
    origin_date: pd.Timestamp,
    daily_actual: pd.DataFrame,
    monthly_anchor: float,
) -> float:
    period = pd.Period(month, freq="M")
    business_days = pd.bdate_range(period.start_time, period.end_time)
    total_count = len(business_days)
    if total_count == 0:
        return float(monthly_anchor)
    actual = daily_actual[
        (daily_actual["series"] == series)
        & (daily_actual["month"] == month)
        & (daily_actual["date"] <= origin_date)
    ].copy()
    known_count = len(actual)
    remaining_count = max(total_count - known_count, 1)
    target = (float(monthly_anchor) * total_count - float(actual["price"].sum())) / remaining_count
    if not np.isfinite(target) or target <= 0:
        return float(monthly_anchor)
    return float(target)


def apply_delta(
    group: pd.DataFrame,
    delta: float,
    target: float,
    source: str,
    anchor_strength: float,
) -> pd.DataFrame:
    group = group.copy()
    group["ensemble_predicted_price"] = group["predicted_price"]
    strength = group["horizon"].astype(int).map(lambda value: horizon_anchor_strength(value, anchor_strength))
    group["anchor_adjustment"] = strength.astype(float) * delta
    group["hybrid_predicted_price"] = np.maximum(
        group["predicted_price"].astype(float) + group["anchor_adjustment"],
        1.0,
    )
    for col in ["p10", "p90", "p05", "p95"]:
        if col in group:
            group[col] = np.maximum(group[col].astype(float) + group["anchor_adjustment"], 1.0)
    group["remaining_month_target"] = target
    group["anchor_source"] = group["anchor_source"].fillna(source)
    return group


def apply_no_anchor(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["ensemble_predicted_price"] = group["predicted_price"]
    group["hybrid_predicted_price"] = group["predicted_price"]
    group["selected_predicted_price"] = group["predicted_price"]
    group["selected_model"] = "Final_ensemble"
    group["remaining_month_target"] = np.nan
    group["anchor_adjustment"] = 0.0
    group["anchor_source"] = "missing_monthly_anchor"
    return group


def horizon_anchor_strength(horizon: int, max_strength: float) -> float:
    return float(np.clip(max_strength * horizon / 30.0, 0.0, max_strength))


def add_selected_adaptive_model(backtest: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    hybrid_metric = metrics[metrics["model"] == "Hybrid_variable_anchor"].copy()
    use_hybrid = {
        (row.series, int(row.horizon)): bool(row.beats_ensemble)
        for row in hybrid_metric.itertuples(index=False)
    }
    ensemble_rows = backtest[backtest["model"] == "Final_ensemble"].copy()
    hybrid_rows = backtest[backtest["model"] == "Hybrid_variable_anchor"].copy()
    hybrid_lookup = {
        (row.series, row.origin_date, row.target_date, int(row.horizon)): row
        for row in hybrid_rows.itertuples(index=False)
    }
    selected = []
    for row in ensemble_rows.itertuples(index=False):
        key = (row.series, row.origin_date, row.target_date, int(row.horizon))
        take_hybrid = use_hybrid.get((row.series, int(row.horizon)), False) and key in hybrid_lookup
        source_row = hybrid_lookup[key] if take_hybrid else row
        pred = float(source_row.predicted_price)
        actual = float(row.actual_price)
        selected.append(
            {
                "series": row.series,
                "origin_date": row.origin_date,
                "target_date": row.target_date,
                "horizon": int(row.horizon),
                "model": "Selected_adaptive_anchor",
                "actual_price": actual,
                "predicted_price": pred,
                "ensemble_predicted_price": float(row.predicted_price),
                "monthly_anchor": getattr(source_row, "monthly_anchor", np.nan),
                "remaining_month_target": getattr(source_row, "remaining_month_target", np.nan),
                "anchor_adjustment": getattr(source_row, "anchor_adjustment", 0.0) if take_hybrid else 0.0,
                "anchor_source": getattr(source_row, "anchor_source", "selected_ensemble"),
                "error": pred - actual,
                "abs_pct_error": abs(pred - actual) / abs(actual) * 100 if actual else np.nan,
            }
        )
    return pd.concat([backtest, pd.DataFrame(selected)], ignore_index=True)


def add_selected_forecast(future: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    hybrid_metric = metrics[metrics["model"] == "Hybrid_variable_anchor"].copy()
    use_hybrid = {
        (row.series, int(row.horizon)): bool(row.beats_ensemble)
        for row in hybrid_metric.itertuples(index=False)
    }
    out = future.copy()
    selected_price = []
    selected_model = []
    for row in out.itertuples(index=False):
        take_hybrid = use_hybrid.get((row.series, int(row.horizon)), False)
        if take_hybrid:
            selected_price.append(float(row.hybrid_predicted_price))
            selected_model.append("Hybrid_variable_anchor")
        else:
            selected_price.append(float(row.ensemble_predicted_price))
            selected_model.append("Final_ensemble")
    out["selected_predicted_price"] = selected_price
    out["selected_model"] = selected_model
    return out


def first_valid(series: pd.Series) -> object | None:
    values = series.dropna()
    if values.empty:
        return None
    return values.iloc[0]


def build_metrics(backtest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (series, horizon, model), group in backtest.groupby(["series", "horizon", "model"], sort=True):
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
    ensemble = metrics[metrics["model"] == "Final_ensemble"][
        ["series", "horizon", "MAE", "RMSE", "MAPE_%"]
    ].rename(
        columns={"MAE": "ensemble_MAE", "RMSE": "ensemble_RMSE", "MAPE_%": "ensemble_MAPE_%"}
    )
    metrics = metrics.merge(naive, on=["series", "horizon"], how="left")
    metrics = metrics.merge(ensemble, on=["series", "horizon"], how="left")
    metrics["beats_naive"] = metrics["MAE"] < metrics["naive_MAE"]
    metrics["beats_ensemble"] = metrics["MAE"] < metrics["ensemble_MAE"]
    return metrics.sort_values(["series", "horizon", "model"]).reset_index(drop=True)


def build_summary(
    args: Args,
    monthly_anchors: pd.DataFrame,
    future: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict[str, object]:
    series_summary = []
    for series, group in metrics[metrics["model"] == "Hybrid_variable_anchor"].groupby("series"):
        ensemble = metrics[(metrics["series"] == series) & (metrics["model"] == "Final_ensemble")].set_index("horizon")
        selected = metrics[(metrics["series"] == series) & (metrics["model"] == "Selected_adaptive_anchor")]
        checkpoints = {}
        for horizon in [1, 7, 14, 30]:
            row = group[group["horizon"] == horizon]
            if row.empty:
                continue
            item = row.iloc[0]
            selected_row = selected[selected["horizon"] == horizon]
            selected_item = selected_row.iloc[0] if not selected_row.empty else item
            checkpoints[str(horizon)] = {
                "hybrid_MAE": float(item["MAE"]),
                "selected_MAE": float(selected_item["MAE"]),
                "ensemble_MAE": float(item["ensemble_MAE"]),
                "naive_MAE": float(item["naive_MAE"]),
                "hybrid_MAPE_%": float(item["MAPE_%"]),
                "selected_MAPE_%": float(selected_item["MAPE_%"]),
                "beats_ensemble": bool(item["beats_ensemble"]),
                "selected_beats_ensemble": bool(selected_item["beats_ensemble"]),
                "beats_naive": bool(item["beats_naive"]),
                "selected_beats_naive": bool(selected_item["beats_naive"]),
            }
        series_summary.append(
            {
                "series": series,
                "mean_hybrid_MAE": float(group["MAE"].mean()),
                "mean_selected_MAE": float(selected["MAE"].mean()) if not selected.empty else np.nan,
                "mean_ensemble_MAE": float(ensemble["MAE"].mean()) if not ensemble.empty else np.nan,
                "beats_ensemble_horizon_count": int(group["beats_ensemble"].sum()),
                "selected_beats_ensemble_horizon_count": int(selected["beats_ensemble"].sum())
                if not selected.empty
                else 0,
                "beats_naive_horizon_count": int(group["beats_naive"].sum()),
                "selected_beats_naive_horizon_count": int(selected["beats_naive"].sum())
                if not selected.empty
                else 0,
                "checkpoint_horizons": checkpoints,
            }
        )
    return {
        "model_version": MODEL_VERSION,
        "daily_prices": str(args.daily_prices),
        "ensemble_forecast": str(args.ensemble_forecast),
        "ensemble_backtest": str(args.ensemble_backtest),
        "monthly_dir": str(args.monthly_dir),
        "anchor_strength": args.anchor_strength,
        "monthly_anchor_months": sorted(monthly_anchors["month"].unique().tolist()),
        "forecast_start": str(future["forecast_date"].min()),
        "forecast_end": str(future["forecast_date"].max()),
        "series_summary": series_summary,
    }


if __name__ == "__main__":
    main()
