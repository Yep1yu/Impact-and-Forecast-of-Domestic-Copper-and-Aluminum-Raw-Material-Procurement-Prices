from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://www.ccmn.cn"
MARKET_VM_ID = "40288092327140f601327141c0560001"

PRODUCTS = {
    "1#铜": "40288092327157530132716ac8ab000b",
    "A00铝": "40288092327157530132716d2960000c",
    "1#白银": "402880532fb8e7ce012fb8eae1a90006",
    "铝合金ADC12": "4028809232af3b411132afab21a00020",
    "铸造铝合金锭(ZLD104)": "7a054f7610c340ecb1a2cdfce0e715b0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Changjiang spot daily average prices from ccmn.cn history data."
    )
    parser.add_argument("--start", help="Start date, YYYY-MM-DD. Defaults to five years before --end.")
    parser.add_argument("--end", help="End date, YYYY-MM-DD. Defaults to today.")
    parser.add_argument(
        "--output",
        default="ccmn_changjiang_avg_prices.csv",
        help="Output CSV path. Default: ccmn_changjiang_avg_prices.csv",
    )
    parser.add_argument(
        "--cookie-env",
        default="CCMN_COOKIE",
        help="Environment variable containing the PC-side ccmn.cn Cookie.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.4,
        help="Seconds to wait between requests. Default: 0.4",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    end = parse_date(args.end) if args.end else date.today()
    start = parse_date(args.start) if args.start else end.replace(year=end.year - 5)
    if start > end:
        raise SystemExit("--start cannot be later than --end")

    cookie = os.environ.get(args.cookie_env)
    if not cookie:
        raise SystemExit(f"Missing Cookie. Set ${args.cookie_env} before running this script.")

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

    rows_by_date: dict[str, dict[str, Any]] = {}
    for product_name, product_id in PRODUCTS.items():
        print(f"Fetching {product_name}...", flush=True)
        for chunk_start, chunk_end in date_chunks(start, end, max_days=365):
            price_list = fetch_product_range(session, product_id, chunk_start, chunk_end)
            for item in price_list:
                publish_date = str(item.get("publishDate") or "").strip()
                if not publish_date:
                    continue
                avg_price = item.get("avgPrice")
                rows_by_date.setdefault(publish_date, {"date": publish_date})[product_name] = avg_price
            time.sleep(args.pause)

    output_path = Path(args.output)
    write_wide_csv(output_path, rows_by_date)
    print(f"Wrote {len(rows_by_date)} dates to {output_path}")
    return 0


def fetch_product_range(
    session: requests.Session,
    product_id: str,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    response = session.post(
        f"{BASE_URL}/shop/historyData/getPriceListStartAndEndTime",
        data={
            "marketVmid": MARKET_VM_ID,
            "productSortVmid": product_id,
            "startTime": start.strftime("%Y-%m-%d"),
            "endTime": end.strftime("%Y-%m-%d"),
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        msg = payload.get("msg") or payload
        raise RuntimeError(f"ccmn.cn request failed for {start} to {end}: {msg}")
    return payload.get("body", {}).get("priceList") or []


def write_wide_csv(path: Path, rows_by_date: dict[str, dict[str, Any]]) -> None:
    fieldnames = ["date", *PRODUCTS.keys()]
    rows = [rows_by_date[key] for key in sorted(rows_by_date)]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_chunks(start: date, end: date, max_days: int):
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=max_days - 1), end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


if __name__ == "__main__":
    sys.exit(main())
