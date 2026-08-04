from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from domestic_prices.db import (
    connect,
    initialize,
    replace_forecast_driver_contributions,
    replace_latest_forecasts,
    replace_latest_monthly_forecasts,
    upsert_spot_prices,
)
from domestic_prices.lithium_model import build_lithium_forecasts


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


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_prices = load_settlement_prices(args.input_dir)
    factors = pd.read_csv(DATASET, encoding="utf-8-sig")
    result = build_lithium_forecasts(
        main_prices,
        factors,
        monthly_periods=args.forecast_months,
        daily_periods=args.forecast_days,
    )
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
    result.daily_forecast.to_csv(
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
        "daily_selected_model": result.daily_diagnostics.selected_model,
        "daily_improvement_pct": result.daily_diagnostics.improvement_pct,
        "monthly_model_version": str(result.monthly_forecast.iloc[0]["model_version"]),
        "daily_model_version": str(result.daily_forecast.iloc[0]["model_version"]),
    }
    (OUTPUT_DIR / "lithium_variable_forecast_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.skip_db:
        import_to_database(
            args.database,
            main_prices,
            result.monthly_forecast,
            result.daily_forecast,
            result.monthly_contributions,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
