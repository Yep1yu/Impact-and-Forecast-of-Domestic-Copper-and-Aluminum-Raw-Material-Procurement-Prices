from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests

from domestic_prices.config import load_config
from domestic_prices.analytics import ensure_monthly_horizon
from domestic_prices.db import (
    connect,
    finish_run,
    initialize,
    replace_latest_forecasts,
    replace_latest_monthly_forecasts,
    start_run,
    upsert_spot_prices,
)
from scripts.fetch_ccmn_changjiang_avg_prices import (
    BASE_URL,
    MARKET_VM_ID,
    PRODUCTS,
    date_chunks,
    fetch_product_range,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PRICE_CSV = ROOT / "ccmn_changjiang_avg_prices.csv"
DEFAULT_TAX_EXCLUSIVE_PRICE_CSV = ROOT / "ccmn_changjiang_avg_prices_tax_exclusive.csv"
DEFAULT_MONTHLY_WORKBOOK = Path("D:/BSH实习/铜、铝采购影响分析/国内/引入变量结果分析_不含税价.xlsx")
DEFAULT_MODEL_VERSION = "daily-hybrid-variable-anchor-v1"
VAT_FACTOR = 1.13
SERIES_TO_METAL = {
    "1#铜": "copper_1",
    "A00铝": "aluminum_a00",
    "1#白银": "silver_1",
    "铝合金ADC12": "aluminum_adc12",
    "铸造铝合金锭(ZLD104)": "aluminum_zld104",
}
MONTHLY_SERIES_TO_METAL = {
    **SERIES_TO_METAL,
    "ADC12": "aluminum_adc12",
    "ZLD104": "aluminum_zld104",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch CCMN prices, rebuild hybrid forecasts, and write the website SQLite database."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--database", help="Override SQLite database path.")
    parser.add_argument("--price-csv", type=Path, default=DEFAULT_PRICE_CSV)
    parser.add_argument("--tax-exclusive-price-csv", type=Path, default=DEFAULT_TAX_EXCLUSIVE_PRICE_CSV)
    parser.add_argument("--monthly-workbook", type=Path, default=DEFAULT_MONTHLY_WORKBOOK)
    parser.add_argument("--lookback-days", type=int, default=21, help="Recent CCMN days to refetch.")
    parser.add_argument("--cookie-env", default="CCMN_COOKIE")
    parser.add_argument("--skip-fetch", action="store_true", help="Use the existing price CSV without CCMN fetch.")
    parser.add_argument(
        "--reuse-model-files",
        action="store_true",
        help="Do not rebuild ensemble/hybrid CSV files; import existing outputs into SQLite.",
    )
    parser.add_argument(
        "--refresh-monthly-model",
        action="store_true",
        help="Rebuild the monthly variable model before daily hybrid anchoring.",
    )
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    database_path = Path(args.database) if args.database else config.database_path
    run_id = str(uuid4())
    started_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    source = "ccmn_hybrid_variable_anchor"
    conn = connect(database_path)
    initialize(conn)
    start_run(conn, run_id, started_at, source)

    rows_spot = rows_forecast = rows_monthly_forecast = 0
    messages: list[str] = []
    try:
        if args.skip_fetch:
            messages.append("Skipped CCMN fetch; using existing price CSV.")
        else:
            fetched_rows = update_price_csv_from_ccmn(args.price_csv, args.lookback_days, args.cookie_env)
            messages.append(f"Fetched/merged {fetched_rows} recent CCMN date rows.")

        write_tax_exclusive_price_csv(args.price_csv, args.tax_exclusive_price_csv)
        messages.append(f"Generated tax-exclusive daily price CSV: {args.tax_exclusive_price_csv.name}.")
        spot_prices = load_wide_price_csv_for_db(args.tax_exclusive_price_csv)
        rows_spot = upsert_spot_prices(conn, spot_prices)

        if args.refresh_monthly_model:
            run_command([sys.executable, "build_monthly_price_prediction_models.py"])
        if not args.reuse_model_files:
            run_command([sys.executable, "build_daily_ensemble_price_model.py", "--input", str(args.tax_exclusive_price_csv)])
            run_command(
                [
                    sys.executable,
                    "build_daily_hybrid_variable_anchored_model.py",
                    "--daily-prices",
                    str(args.tax_exclusive_price_csv),
                    "--monthly-workbook",
                    str(args.monthly_workbook),
                ]
            )
        else:
            messages.append("Reused existing daily ensemble/hybrid model CSV files.")

        forecasts = load_hybrid_forecasts_for_db(
            ROOT / "daily_hybrid_variable_anchored_forecast_30d.csv",
            args.model_version,
            spot_prices,
        )
        rows_forecast = replace_latest_forecasts(conn, forecasts, args.model_version)
        monthly_forecasts = load_monthly_forecasts_for_db(
            args.monthly_workbook, args.model_version, spot_prices
        )
        rows_monthly_forecast = replace_latest_monthly_forecasts(conn, monthly_forecasts, args.model_version)
        finish_run(
            conn,
            run_id,
            datetime.now(UTC).replace(microsecond=0).isoformat(),
            "success",
            rows_spot,
            0,
            rows_forecast + rows_monthly_forecast,
            "; ".join(messages) if messages else None,
        )
        print(f"status: success")
        print(f"run_id: {run_id}")
        print(f"database_path: {database_path}")
        print(f"rows_spot: {rows_spot}")
        print(f"rows_forecast: {rows_forecast}")
        print(f"rows_monthly_forecast: {rows_monthly_forecast}")
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        finish_run(
            conn,
            run_id,
            datetime.now(UTC).replace(microsecond=0).isoformat(),
            "failed",
            rows_spot,
            0,
            rows_forecast + rows_monthly_forecast,
            error,
        )
        raise
    finally:
        conn.close()


def update_price_csv_from_ccmn(path: Path, lookback_days: int, cookie_env: str) -> int:
    cookie = os.environ.get(cookie_env)
    if not cookie:
        raise RuntimeError(f"Missing Cookie. Set ${cookie_env} before running automatic CCMN fetch.")

    end = date.today()
    start = end - timedelta(days=max(lookback_days, 1))
    existing = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame(columns=["date"])
    if not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    session = requests.Session()
    session.headers.update(
        {
            "Cookie": cookie,
            "Referer": f"{BASE_URL}/historyprice/",
            "Origin": BASE_URL,
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }
    )

    rows_by_date: dict[str, dict[str, object]] = {}
    for product_name, product_id in PRODUCTS.items():
        for chunk_start, chunk_end in date_chunks(start, end, max_days=365):
            for item in fetch_product_range(session, product_id, chunk_start, chunk_end):
                publish_date = str(item.get("publishDate") or "").strip()
                if not publish_date:
                    continue
                rows_by_date.setdefault(publish_date, {"date": publish_date})[product_name] = item.get("avgPrice")
            time.sleep(0.2)

    recent = pd.DataFrame([rows_by_date[key] for key in sorted(rows_by_date)])
    merged = merge_wide_price_frames(existing, recent)
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return len(recent)


def merge_wide_price_frames(existing: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", *PRODUCTS.keys()]
    for frame in [existing, recent]:
        for column in columns:
            if column not in frame:
                frame[column] = pd.NA
    if existing.empty:
        merged = recent[columns].copy()
    else:
        merged = pd.concat([existing[columns], recent[columns]], ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        merged = merged.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    return merged.sort_values("date").reset_index(drop=True)


def load_wide_price_csv_for_db(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    if "date" not in raw.columns:
        raise ValueError(f"{path} missing date column")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    rows = []
    for series, metal in SERIES_TO_METAL.items():
        if series not in raw.columns:
            raise ValueError(f"{path} missing price column: {series}")
        part = raw[["date", series]].rename(columns={"date": "trade_date", series: "price_cny_per_tonne"})
        part["metal"] = metal
        part["source"] = "CCMN Changjiang tax-exclusive"
        part["raw_symbol"] = series
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out["price_cny_per_tonne"] = pd.to_numeric(out["price_cny_per_tonne"], errors="coerce")
    out = out.dropna(subset=["trade_date", "price_cny_per_tonne"])
    out = out[out["price_cny_per_tonne"] > 0]
    return out[["trade_date", "metal", "price_cny_per_tonne", "source", "raw_symbol"]]


def load_hybrid_forecasts_for_db(path: Path, model_version: str, spot_prices: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path, encoding="utf-8-sig")
    required = {"series", "forecast_date", "selected_predicted_price", "p10", "p90"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    latest_prices = (
        spot_prices.sort_values("trade_date")
        .groupby("metal")["price_cny_per_tonne"]
        .last()
        .to_dict()
    )
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    rows = []
    for row in raw.itertuples(index=False):
        metal = SERIES_TO_METAL.get(str(row.series))
        if not metal:
            continue
        predicted = float(row.selected_predicted_price)
        latest = float(latest_prices.get(metal, predicted))
        rows.append(
            {
                "metal": metal,
                "forecast_date": row.forecast_date,
                "predicted_price_cny_per_tonne": predicted,
                "lower_bound": float(row.p10),
                "upper_bound": float(row.p90),
                "direction": direction(predicted, latest),
                "model_version": model_version,
                "generated_at": generated_at,
            }
        )
    if not rows:
        raise ValueError(f"{path} produced no importable forecast rows")
    return pd.DataFrame(rows)


def load_monthly_forecasts_for_db(
    path: Path, model_version: str, spot_prices: pd.DataFrame
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_excel(path, sheet_name="未来月份预测")
    required = {"品种", "预测月份", "预测月均价"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    out = raw.rename(
        columns={
            "品种": "series",
            "预测月份": "forecast_month",
            "预测月均价": "predicted_price_cny_per_tonne",
        }
    )[["series", "forecast_month", "predicted_price_cny_per_tonne"]].copy()
    out["metal"] = out["series"].map(MONTHLY_SERIES_TO_METAL)
    out["forecast_month"] = pd.to_datetime(out["forecast_month"], errors="coerce")
    out["predicted_price_cny_per_tonne"] = pd.to_numeric(out["predicted_price_cny_per_tonne"], errors="coerce")
    out = out.dropna(subset=["metal", "forecast_month", "predicted_price_cny_per_tonne"])
    out = out[out["predicted_price_cny_per_tonne"] > 0]
    current_month = pd.Timestamp.today().to_period("M").to_timestamp()
    out = out[out["forecast_month"] >= current_month]
    out["source"] = "monthly-variable-model"
    out["model_version"] = model_version
    out["generated_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    extended = []
    for metal, group in out.groupby("metal"):
        metal_spot = spot_prices[spot_prices["metal"] == metal].copy()
        extended.append(ensure_monthly_horizon(group, metal_spot, periods=12))
    out = pd.concat(extended, ignore_index=True) if extended else out
    return out[
        [
            "metal",
            "forecast_month",
            "predicted_price_cny_per_tonne",
            "lower_bound",
            "upper_bound",
            "direction",
            "predicted_change_pct",
            "source",
            "model_version",
            "generated_at",
        ]
    ]


def write_tax_exclusive_price_csv(source_path: Path, destination_path: Path) -> None:
    raw = pd.read_csv(source_path, encoding="utf-8-sig")
    if "date" not in raw.columns:
        raise ValueError(f"{source_path} missing date column")
    for column in raw.columns:
        if column == "date":
            continue
        raw[column] = pd.to_numeric(raw[column], errors="coerce") / VAT_FACTOR
    raw.to_csv(destination_path, index=False, encoding="utf-8-sig")


def direction(predicted: float, latest: float) -> str:
    pct = predicted / latest - 1 if latest else 0.0
    if pct > 0.005:
        return "up"
    if pct < -0.005:
        return "down"
    return "flat"


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
