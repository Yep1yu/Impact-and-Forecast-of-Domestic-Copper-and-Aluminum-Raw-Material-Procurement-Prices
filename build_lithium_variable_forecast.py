from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_daily_ensemble_price_model import Args as DailyEnsembleArgs
from build_daily_ensemble_price_model import model_one_series
from domestic_prices.db import (
    connect,
    initialize,
    replace_forecast_driver_contributions,
    replace_latest_forecasts,
    replace_latest_monthly_forecasts,
    upsert_spot_prices,
)
from domestic_prices.lithium_model import _direction, build_lithium_forecasts


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path(r"D:\Wechat\xwechat_files\wxid_i1mfkj939nq911_dfaf\msg\file\2026-07")
DATASET = ROOT / "domestic_material_monthly_dataset_v1.csv"
OUTPUT_DIR = ROOT / "lithium_carbonate_prediction_outputs"
METAL = "lithium_carbonate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build logic-constrained lithium daily and monthly forecasts."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--database", type=Path, default=ROOT / "domestic_procurement_prices.sqlite"
    )
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--forecast-months", type=int, default=12)
    parser.add_argument("--skip-db", action="store_true")
    return parser.parse_args()


def load_settlement_prices(input_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob("LCFUTURES*.xlsx")):
        frame = pd.read_excel(path, header=1)
        if frame.shape[1] < 15:
            raise ValueError(f"{path} does not look like an LC futures workbook")
        frame = frame.rename(
            columns={
                frame.columns[0]: "date",
                frame.columns[1]: "product",
                frame.columns[3]: "contract",
                frame.columns[9]: "settlement_price",
                frame.columns[12]: "volume",
                frame.columns[13]: "open_interest",
                frame.columns[14]: "open_interest_change",
            }
        )
        frame = frame[frame["product"].astype(str).str.strip() == "碳酸锂"]
        frame["date"] = pd.to_datetime(
            frame["date"].astype(str), format="%Y%m%d", errors="coerce"
        )
        for column in [
            "settlement_price",
            "volume",
            "open_interest",
            "open_interest_change",
        ]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(
            frame[
                [
                    "date",
                    "contract",
                    "settlement_price",
                    "volume",
                    "open_interest",
                    "open_interest_change",
                ]
            ]
        )
    if not frames:
        raise FileNotFoundError(f"No LCFUTURES*.xlsx files found in {input_dir}")
    raw = pd.concat(frames, ignore_index=True).dropna(
        subset=["date", "settlement_price", "volume"]
    )
    return (
        raw.sort_values(["date", "volume"], ascending=[True, False])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def import_to_database(
    database: Path,
    main_prices: pd.DataFrame,
    monthly_forecast: pd.DataFrame,
    daily_forecast: pd.DataFrame,
    contributions: pd.DataFrame,
) -> None:
    conn = connect(database)
    initialize(conn)
    spot = main_prices[["date", "settlement_price", "contract"]].rename(
        columns={
            "date": "trade_date",
            "settlement_price": "price_cny_per_tonne",
            "contract": "raw_symbol",
        }
    )
    spot["metal"] = METAL
    spot["source"] = "GFEX LC main contract settlement price"
    upsert_spot_prices(
        conn,
        spot[
            [
                "trade_date",
                "metal",
                "price_cny_per_tonne",
                "source",
                "raw_symbol",
            ]
        ],
    )
    daily_columns = [
        "metal",
        "forecast_date",
        "predicted_price_cny_per_tonne",
        "lower_bound",
        "upper_bound",
        "direction",
        "model_version",
        "generated_at",
    ]
    replace_latest_forecasts(
        conn, daily_forecast[daily_columns], str(daily_forecast.iloc[0]["model_version"])
    )
    replace_latest_monthly_forecasts(
        conn,
        monthly_forecast,
        str(monthly_forecast.iloc[0]["model_version"]),
    )
    if not contributions.empty:
        replace_forecast_driver_contributions(
            conn,
            contributions,
            str(contributions.iloc[0]["model_version"]),
        )
    conn.close()


def build_shared_daily_forecast(main_prices: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Use the same rolling-validation ensemble route as the five core materials."""
    frame = main_prices[["date", "settlement_price"]].rename(
        columns={"settlement_price": "price"}
    ).copy()
    for column in ["volume", "open_interest", "contract"]:
        if column in main_prices:
            frame[column] = main_prices[column].to_numpy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["date", "price"]).sort_values("date").reset_index(drop=True)
    if len(frame) < periods + 60:
        raise ValueError("碳酸锂历史样本不足，无法执行日度组合模型")

    min_train_days = min(500, max(120, len(frame) - periods - 1))
    validation_origins = min(120, max(5, len(frame) - periods - min_train_days))
    args = DailyEnsembleArgs(
        input=Path("<lithium-settlement-workbook>"),
        output_dir=OUTPUT_DIR,
        forecast_days=periods,
        validation_origins=validation_origins,
        min_train_days=min_train_days,
        retrain_step=20,
        ridge_alpha=20.0,
        no_arima=False,
    )
    result = model_one_series(METAL, frame, args)
    forecast = pd.DataFrame(result["forecast"])
    if forecast.empty:
        raise RuntimeError("碳酸锂日度组合模型未生成预测结果")
    forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"])
    previous = float(frame.iloc[-1]["price"])
    forecast["predicted_return"] = forecast["predicted_price"].astype(float).pct_change()
    forecast.loc[forecast.index[0], "predicted_return"] = (
        float(forecast.iloc[0]["predicted_price"]) / previous - 1
    )
    forecast["month"] = forecast["forecast_date"].dt.to_period("M").dt.start_time
    forecast["direction"] = forecast["predicted_return"].map(_direction)
    forecast["metal"] = METAL
    forecast["model_version"] = "lithium-specific-ensemble-daily-v3"
    forecast["generated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    return forecast[
        [
            "forecast_date",
            "predicted_price",
            "predicted_return",
            "month",
            "p10",
            "p90",
            "direction",
            "metal",
            "model_version",
            "generated_at",
        ]
    ].rename(
        columns={
            "predicted_price": "predicted_price_cny_per_tonne",
            "p10": "lower_bound",
            "p90": "upper_bound",
        }
    )


def build_monthly_rolling_baseline(main_prices: pd.DataFrame) -> pd.DataFrame:
    """Build a one-step-ahead lithium trend baseline for the evaluation page.

    The prediction uses only months that were already available at the forecast
    origin.  A clipped fraction of the latest month-to-month change reduces the
    systematic lag of a pure last-value forecast while limiting regime-change
    overshoot.
    """
    monthly = (
        main_prices.assign(month=main_prices["date"].dt.to_period("M").dt.start_time)
        .groupby("month", as_index=False)["settlement_price"]
        .mean()
        .rename(columns={"settlement_price": "target_price"})
        .sort_values("month")
    )
    fitted = monthly.copy()
    previous = fitted["target_price"].shift(1)
    previous_change = fitted["target_price"].shift(1) - fitted["target_price"].shift(2)
    trend_adjustment = previous_change.clip(lower=-30000.0, upper=30000.0) * 0.5
    fitted["predicted_monthly_price"] = previous + trend_adjustment.fillna(0.0)
    return fitted.dropna(subset=["predicted_monthly_price"])[
        ["target_price", "month", "predicted_monthly_price"]
    ].reset_index(drop=True)


def build_lithium_monthly_features(
    main_prices: pd.DataFrame, common_factors: pd.DataFrame
) -> pd.DataFrame:
    """Combine common macro/demand variables with lithium futures-specific signals."""
    daily = main_prices.copy().sort_values("date").reset_index(drop=True)
    daily["month"] = pd.to_datetime(daily["date"]).dt.to_period("M").dt.start_time
    daily["settlement_price"] = pd.to_numeric(daily["settlement_price"], errors="coerce")
    for column in ["volume", "open_interest"]:
        if column not in daily:
            daily[column] = 0.0
        daily[column] = pd.to_numeric(daily[column], errors="coerce").fillna(0.0)
    # Compute volatility from the continuous main-contract series.  Resetting
    # the return calculation at each month would drop the first trading day of
    # every month and make the volatility factor depend on calendar boundaries.
    daily["daily_return"] = daily["settlement_price"].pct_change()
    daily["contract_roll"] = (
        daily["contract"].astype(str).ne(daily["contract"].astype(str).shift(1)).astype(float)
        if "contract" in daily
        else 0.0
    )
    monthly = (
        daily.groupby("month", as_index=False)
        .agg(
            LC成交量=("volume", "sum"),
            LC持仓量=("open_interest", "last"),
            LC价格波动率=("daily_return", "std"),
            LC合约切换=("contract_roll", "sum"),
        )
        .sort_values("month")
    )
    monthly["LC成交量_环比"] = monthly["LC成交量"].pct_change()
    monthly["LC持仓量_环比"] = monthly["LC持仓量"].pct_change()
    factors = common_factors.rename(columns={common_factors.columns[0]: "month"}).copy()
    factors["month"] = pd.to_datetime(factors["month"], errors="coerce").dt.to_period("M").dt.start_time
    factors = factors.dropna(subset=["month"]).drop_duplicates("month")
    merged = factors.merge(
        monthly[["month", "LC成交量_环比", "LC持仓量_环比", "LC价格波动率", "LC合约切换"]],
        on="month",
        how="outer",
    ).sort_values("month")
    for column in ["LC成交量_环比", "LC持仓量_环比", "LC价格波动率", "LC合约切换"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_prices = load_settlement_prices(args.input_dir)
    common_factors = pd.read_csv(DATASET, encoding="utf-8-sig")
    factors = build_lithium_monthly_features(main_prices, common_factors)
    result = build_lithium_forecasts(
        main_prices,
        factors,
        monthly_periods=args.forecast_months,
        daily_periods=args.forecast_days,
    )
    shared_daily_forecast = build_shared_daily_forecast(main_prices, args.forecast_days)
    monthly_rolling_baseline = build_monthly_rolling_baseline(main_prices)
    result.monthly_coefficients.to_csv(
        OUTPUT_DIR / "lithium_monthly_model_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.monthly_forecast.to_csv(
        OUTPUT_DIR / "lithium_monthly_forecast.csv",
        index=False,
        encoding="utf-8-sig",
    )
    factors.to_csv(
        OUTPUT_DIR / "lithium_monthly_features.csv",
        index=False,
        encoding="utf-8-sig",
    )
    monthly_rolling_baseline.to_csv(
        OUTPUT_DIR / "lithium_monthly_fitted_prices.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shared_daily_forecast.to_csv(
        OUTPUT_DIR / "lithium_daily_forecast.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result.monthly_contributions.to_csv(
        OUTPUT_DIR / "lithium_forecast_driver_contributions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "metal": METAL,
        "sample_start": str(main_prices["date"].min().date()),
        "sample_end": str(main_prices["date"].max().date()),
        "monthly_selected_model": result.monthly_diagnostics.selected_model,
        "monthly_improvement_pct": result.monthly_diagnostics.improvement_pct,
        "historical_evaluation_model": "lithium-trend-adjusted-one-step-baseline",
        "monthly_backtest_mae": float(
            (monthly_rolling_baseline["predicted_monthly_price"] - monthly_rolling_baseline["target_price"])
            .abs()
            .mean()
        ),
        "monthly_backtest_mape_pct": float(
            (
                (monthly_rolling_baseline["predicted_monthly_price"] - monthly_rolling_baseline["target_price"])
                .abs()
                / monthly_rolling_baseline["target_price"].abs()
            ).mean()
            * 100
        ),
        "daily_selected_model": "rolling-validation multi-model ensemble",
        "daily_improvement_pct": None,
        "monthly_model_version": str(result.monthly_forecast.iloc[0]["model_version"]),
        "daily_model_version": str(shared_daily_forecast.iloc[0]["model_version"]),
    }
    (OUTPUT_DIR / "lithium_variable_forecast_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.skip_db:
        import_to_database(
            args.database,
            main_prices,
            result.monthly_forecast,
            shared_daily_forecast,
            result.monthly_contributions,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
