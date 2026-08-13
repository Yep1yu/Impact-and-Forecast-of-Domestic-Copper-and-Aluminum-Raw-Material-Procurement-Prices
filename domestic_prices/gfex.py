from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests


GFEX_DAILY_QUOTES_URL = (
    "http://www.gfex.com.cn/u/interfacesWebTiDayQuotes/loadList"
)
GFEX_DAILY_QUOTES_PAGE = "http://www.gfex.com.cn/gfex/rihq/hqsj_tjsj.shtml"
SOURCE = "GFEX LC all-contract simple mean settlement price"
RAW_SYMBOL = "ALL_CONTRACTS_MEAN"


def fetch_daily_quotes(
    trade_date: date | datetime | str,
    *,
    session: requests.Session | None = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> dict[str, Any]:
    normalized = pd.Timestamp(trade_date).strftime("%Y%m%d")
    client = session or requests.Session()
    response = None
    for attempt in range(max(retries, 1)):
        try:
            response = client.post(
                GFEX_DAILY_QUOTES_URL,
                data={"trade_date": normalized, "trade_type": "0"},
                headers={
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                    "Origin": "http://www.gfex.com.cn",
                    "Referer": GFEX_DAILY_QUOTES_PAGE,
                    "User-Agent": "Mozilla/5.0 (compatible; BSHintern/1.0)",
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, ValueError):
            if attempt + 1 >= max(retries, 1):
                raise
            time.sleep(1.5 * (attempt + 1))
    if response is None:
        raise RuntimeError("GFEX daily quotes request produced no response")
    if str(payload.get("code")) != "0":
        raise RuntimeError(f"GFEX daily quotes request failed: {payload.get('msg')}")
    return payload


def lithium_contracts(payload: dict[str, Any], trade_date: date | datetime | str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in payload.get("data") or []:
        if str(item.get("varietyOrder") or "").strip().lower() != "lc":
            continue
        delivery_month = str(item.get("delivMonth") or "").strip()
        if not delivery_month.isdigit():
            continue
        rows.append(
            {
                "date": pd.Timestamp(trade_date).normalize(),
                "contract": f"lc{delivery_month}",
                "settlement_price": item.get("clearPrice"),
                "volume": item.get("volumn"),
                "open_interest": item.get("openInterest"),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=["date", "contract", "settlement_price", "volume", "open_interest"],
    )
    for column in ["settlement_price", "volume", "open_interest"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["settlement_price"])
        .drop_duplicates("contract", keep="last")
        .sort_values("contract")
        .reset_index(drop=True)
    )


def aggregate_lithium_day(contracts: pd.DataFrame) -> pd.DataFrame:
    if contracts.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "settlement_price",
                "volume",
                "open_interest",
                "open_interest_change",
                "contract",
                "contract_count",
                "source",
            ]
        )
    trade_dates = pd.to_datetime(contracts["date"], errors="coerce").dropna().unique()
    if len(trade_dates) != 1:
        raise ValueError("GFEX lithium contracts must contain exactly one trade date")
    valid = contracts.dropna(subset=["settlement_price"]).copy()
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(trade_dates[0]).normalize(),
                "settlement_price": float(valid["settlement_price"].mean()),
                "volume": float(valid["volume"].fillna(0).sum()),
                "open_interest": float(valid["open_interest"].fillna(0).sum()),
                "open_interest_change": 0.0,
                "contract": RAW_SYMBOL,
                "contract_count": int(len(valid)),
                "source": SOURCE,
            }
        ]
    )


def fetch_latest_lithium_day(
    *,
    as_of: date | datetime | str | None = None,
    lookback_days: int = 10,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    end = pd.Timestamp(as_of or date.today()).date()
    client = session or requests.Session()
    for offset in range(max(lookback_days, 1)):
        candidate = end - timedelta(days=offset)
        if candidate.weekday() >= 5:
            continue
        payload = fetch_daily_quotes(candidate, session=client)
        contracts = lithium_contracts(payload, candidate)
        if not contracts.empty:
            return aggregate_lithium_day(contracts)
    return aggregate_lithium_day(pd.DataFrame())


def merge_lithium_history(path: str | Path, recent: pd.DataFrame) -> pd.DataFrame:
    destination = Path(path)
    columns = [
        "date",
        "settlement_price",
        "volume",
        "open_interest",
        "open_interest_change",
        "contract",
        "contract_count",
        "source",
    ]
    existing = (
        pd.read_csv(destination, encoding="utf-8-sig")
        if destination.exists()
        else pd.DataFrame(columns=columns)
    )
    for frame in [existing, recent]:
        for column in columns:
            if column not in frame:
                frame[column] = pd.NA
    merged = pd.concat([existing[columns], recent[columns]], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = (
        merged.dropna(subset=["date", "settlement_price"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged
