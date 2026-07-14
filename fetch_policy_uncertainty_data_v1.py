from __future__ import annotations

from pathlib import Path

import pandas as pd


CACHE_DIR = Path("data_cache_v3")
OUTPUT_FILE = CACHE_DIR / "policy_uncertainty_monthly.csv"

FRED_SERIES = {
    "CHNMAINLANDEPU": "中国经济政策不确定性指数",
    "CHNMAINLANDTPU": "中国贸易政策不确定性指数",
    "GEPUCURRENT": "全球经济政策不确定性指数",
}


def fetch_fred_series(series_id: str, label: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    raw = pd.read_csv(url)
    raw["月份"] = pd.to_datetime(raw["observation_date"]).dt.to_period("M").dt.to_timestamp()
    raw[label] = pd.to_numeric(raw[series_id].replace(".", pd.NA), errors="coerce")
    return raw[["月份", label]]


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = None
    for series_id, label in FRED_SERIES.items():
        frame = fetch_fred_series(series_id, label)
        out = frame if out is None else out.merge(frame, on="月份", how="outer")
    out = out.sort_values("月份")
    for col in out.columns:
        if col == "月份":
            continue
        out[f"{col}_变化"] = out[col].diff()
        out[f"{col}_环比"] = out[col].pct_change(fill_method=None) * 100
    out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(OUTPUT_FILE.resolve())
    print(out.shape)
    print(out.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
