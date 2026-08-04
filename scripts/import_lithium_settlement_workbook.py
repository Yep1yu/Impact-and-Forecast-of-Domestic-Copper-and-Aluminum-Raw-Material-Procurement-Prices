from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from domestic_prices.db import connect, initialize, upsert_spot_prices


def load_main_contract_prices(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=1)
    if raw.shape[1] < 13:
        raise ValueError(f"{path} does not contain the expected futures columns")
    raw = raw.rename(
        columns={
            raw.columns[0]: "trade_date",
            raw.columns[1]: "product",
            raw.columns[3]: "contract",
            raw.columns[9]: "settlement_price",
            raw.columns[12]: "volume",
        }
    )
    raw = raw[raw["product"].astype(str).str.strip() == "碳酸锂"].copy()
    raw["trade_date"] = pd.to_datetime(
        raw["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    raw["settlement_price"] = pd.to_numeric(raw["settlement_price"], errors="coerce")
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    raw = raw.dropna(subset=["trade_date", "settlement_price", "volume"])
    main = (
        raw.sort_values(["trade_date", "volume"], ascending=[True, False])
        .drop_duplicates("trade_date")
        .sort_values("trade_date")
    )
    main["metal"] = "lithium_carbonate"
    main["source"] = "GFEX LC main contract settlement price (provided workbook)"
    return main[
        ["trade_date", "metal", "settlement_price", "source", "contract"]
    ].rename(
        columns={
            "settlement_price": "price_cny_per_tonne",
            "contract": "raw_symbol",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import lithium settlement prices from an Excel workbook.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("domestic_procurement_prices.sqlite"))
    args = parser.parse_args()
    prices = load_main_contract_prices(args.input)
    if prices.empty:
        raise RuntimeError("The workbook contains no usable lithium settlement prices")
    conn = connect(args.database)
    initialize(conn)
    count = upsert_spot_prices(conn, prices)
    conn.close()
    print(
        f"Imported {count} lithium settlement rows from {args.input}; "
        f"range {prices['trade_date'].min():%Y-%m-%d} to {prices['trade_date'].max():%Y-%m-%d}."
    )


if __name__ == "__main__":
    main()
