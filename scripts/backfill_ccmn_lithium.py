from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from domestic_prices.db import connect, initialize, upsert_spot_prices
from scripts.fetch_ccmn_changjiang_avg_prices import fetch_product_range


LITHIUM_PRODUCT_ID = "5d7f1c291d724b0cbf8d639f777578d7"
LITHIUM_MARKET_VMID = "4028809232fb12120132fb4545a10001"
LITHIUM_PRODUCT_NAME = "电池级碳酸锂99.5%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill CCMN lithium carbonate spot prices.")
    parser.add_argument("--start", default="2026-07-15")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--database", type=Path, default=Path("domestic_procurement_prices.sqlite"))
    parser.add_argument("--cookie-env", default="CCMN_COOKIE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cookie = os.environ.get(args.cookie_env)
    if not cookie:
        raise SystemExit(f"Missing Cookie. Set ${args.cookie_env} before running this one-time backfill.")
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if start > end:
        raise SystemExit("--start cannot be later than --end")

    session = requests.Session()
    session.headers.update(
        {
            "Cookie": cookie,
            "Referer": "https://www.ccmn.cn/",
            "Origin": "https://www.ccmn.cn",
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }
    )
    rows: list[dict[str, object]] = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=364), end)
        print(f"Fetching lithium prices {current} to {chunk_end}...", flush=True)
        for item in fetch_product_range(
            session,
            LITHIUM_PRODUCT_ID,
            current,
            chunk_end,
            market_vmid=LITHIUM_MARKET_VMID,
        ):
            publish_date = str(item.get("publishDate") or "").strip()
            avg_price = item.get("avgPrice")
            if publish_date and avg_price not in (None, ""):
                rows.append(
                    {
                        "trade_date": publish_date,
                        "metal": "lithium_carbonate",
                        "price_cny_per_tonne": avg_price,
                        "source": "CCMN Changjiang historical quote",
                        "raw_symbol": LITHIUM_PRODUCT_NAME,
                    }
                )
        current = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    if not rows:
        raise RuntimeError("CCMN returned no lithium prices for the requested range")
    frame = pd.DataFrame(rows).drop_duplicates("trade_date", keep="last")
    conn = connect(args.database)
    initialize(conn)
    count = upsert_spot_prices(conn, frame)
    conn.close()
    print(f"Upserted {count} lithium price rows into {args.database}")


if __name__ == "__main__":
    main()
