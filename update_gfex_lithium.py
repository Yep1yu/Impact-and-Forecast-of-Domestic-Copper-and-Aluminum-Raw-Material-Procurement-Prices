from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_lithium_variable_forecast import (
    DATASET,
    METAL,
    build_lithium_forecasts,
    build_lithium_monthly_features,
    build_monthly_rolling_baseline,
    build_shared_daily_forecast,
    import_to_database,
)
from domestic_prices.gfex import fetch_lithium_days, merge_lithium_history


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / "lithium_carbonate_prediction_outputs" / "lithium_daily_history.csv"
OUTPUT_DIR = ROOT / "lithium_carbonate_prediction_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the latest GFEX lithium contracts and rebuild lithium forecasts."
    )
    parser.add_argument("--database", type=Path, default=ROOT / "domestic_procurement_prices.sqlite")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--as-of", help="Latest date to query, YYYY-MM-DD; defaults to today.")
    parser.add_argument("--lookback-days", type=int, default=10)
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--forecast-months", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="Rebuild even when the latest row is unchanged.")
    parser.add_argument("--skip-db", action="store_true")
    return parser.parse_args()


def load_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["settlement_price", "volume", "open_interest", "open_interest_change"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "settlement_price"]).sort_values("date")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recent = fetch_lithium_days(
        as_of=args.as_of,
        lookback_days=args.lookback_days,
    )
    if recent.empty:
        raise RuntimeError("GFEX returned no lithium futures data in the lookback window")
    latest = recent.tail(1).reset_index(drop=True)
    if args.history.exists() and not args.force:
        cached = load_history(args.history)
        same_date = not cached.empty and cached.iloc[-1]["date"] == latest.iloc[0]["date"]
        same_price = same_date and abs(
            float(cached.iloc[-1]["settlement_price"])
            - float(latest.iloc[0]["settlement_price"])
        ) < 1e-9
        if same_price:
            print(json.dumps({
                "status": "no_update",
                "latest_trade_date": str(latest.iloc[0]["date"].date()),
                "contract_count": int(latest.iloc[0]["contract_count"]),
                "average_settlement_price": float(latest.iloc[0]["settlement_price"]),
            }, ensure_ascii=False, indent=2))
            return
    merged = merge_lithium_history(args.history, recent)
    merged.to_csv(args.history, index=False, encoding="utf-8-sig")
    prices = load_history(args.history)
    common_factors = pd.read_csv(DATASET, encoding="utf-8-sig")
    factors = build_lithium_monthly_features(prices, common_factors)
    result = build_lithium_forecasts(
        prices,
        factors,
        monthly_periods=args.forecast_months,
        daily_periods=args.forecast_days,
    )
    daily_forecast = build_shared_daily_forecast(prices, args.forecast_days, OUTPUT_DIR)
    fitted = build_monthly_rolling_baseline(prices)
    result.monthly_coefficients.to_csv(
        OUTPUT_DIR / "lithium_monthly_model_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    result.monthly_forecast.to_csv(
        OUTPUT_DIR / "lithium_monthly_forecast.csv", index=False, encoding="utf-8-sig"
    )
    factors.to_csv(OUTPUT_DIR / "lithium_monthly_features.csv", index=False, encoding="utf-8-sig")
    fitted.to_csv(
        OUTPUT_DIR / "lithium_monthly_fitted_prices.csv", index=False, encoding="utf-8-sig"
    )
    daily_forecast.to_csv(
        OUTPUT_DIR / "lithium_daily_forecast.csv", index=False, encoding="utf-8-sig"
    )
    result.monthly_contributions.to_csv(
        OUTPUT_DIR / "lithium_forecast_driver_contributions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "metal": METAL,
        "sample_start": str(prices["date"].min().date()),
        "sample_end": str(prices["date"].max().date()),
        "price_aggregation": "GFEX all listed LC contracts simple mean settlement price",
        "latest_contract_count": int(latest.iloc[0]["contract_count"]),
        "monthly_selected_model": result.monthly_diagnostics.selected_model,
        "monthly_improvement_pct": result.monthly_diagnostics.improvement_pct,
        "historical_evaluation_model": "lithium-daily-feature-ridge-monthly-v1",
        "monthly_backtest_mae": float(
            (fitted["predicted_monthly_price"] - fitted["target_price"]).abs().mean()
        ),
        "monthly_backtest_mape_pct": float(
            ((fitted["predicted_monthly_price"] - fitted["target_price"]).abs()
             / fitted["target_price"].abs()).mean() * 100
        ),
        "daily_selected_model": "rolling-validation multi-model ensemble",
        "daily_improvement_pct": None,
        "monthly_model_version": str(result.monthly_forecast.iloc[0]["model_version"]),
        "daily_model_version": str(daily_forecast.iloc[0]["model_version"]),
    }
    (OUTPUT_DIR / "lithium_variable_forecast_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.skip_db:
        import_to_database(
            args.database,
            prices,
            result.monthly_forecast,
            daily_forecast,
            result.monthly_contributions,
        )
    print(json.dumps({
        "status": "success",
        "latest_trade_date": str(latest.iloc[0]["date"].date()),
        "contract_count": int(latest.iloc[0]["contract_count"]),
        "average_settlement_price": float(latest.iloc[0]["settlement_price"]),
        **summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
