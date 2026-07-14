from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from .config import ChangjiangConfig, ShfeConfig, SmmConfig
from .config import ExcelSourceConfig


METAL_ALIASES = {
    "copper_1": "copper_1",
    "aluminum_a00": "aluminum_a00",
    "silver_1": "silver_1",
    "aluminum_adc12": "aluminum_adc12",
    "aluminum_zld104": "aluminum_zld104",
    "cu": "copper",
    "copper": "copper",
    "铜": "copper",
    "1#铜": "copper",
    "1#电解铜": "copper",
    "al": "aluminum",
    "aluminium": "aluminum",
    "aluminum": "aluminum",
    "铝": "aluminum",
    "a00铝": "aluminum",
    "a00铝锭": "aluminum",
}

SPOT_COLUMN_ALIASES = {
    "trade_date": ["trade_date", "date", "日期", "交易日", "报价日期", "publish_date"],
    "metal": ["metal", "品种", "金属", "material", "commodity", "symbol"],
    "price_cny_per_tonne": [
        "price_cny_per_tonne",
        "price",
        "均价",
        "现货均价",
        "avg_price",
        "average",
        "mid",
        "中间价",
    ],
    "raw_symbol": ["raw_symbol", "smm_symbol", "名称", "产品", "product_name"],
}

FEATURE_COLUMN_ALIASES = {
    "trade_date": ["trade_date", "date", "日期", "交易日", "publish_date"],
    "metal": ["metal", "品种", "金属", "material", "commodity", "symbol"],
    "shfe_futures_price": ["shfe_futures_price", "SHFE期货价", "沪期价", "期货价"],
    "inventory_tonne": ["inventory_tonne", "库存", "库存吨", "社会库存"],
    "premium_discount": ["premium_discount", "升贴水", "现货升贴水"],
    "import_profit": ["import_profit", "进口盈亏", "进口利润"],
}


class SmmClient:
    def __init__(self, config: SmmConfig):
        self.config = config

    def fetch_spot_prices(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        frames = []
        for metal in self.config.metals:
            if not metal.spot_endpoint:
                raise ValueError(f"{metal.metal} missing SMM spot_endpoint in config.yaml")
            payload = self._get(metal.spot_endpoint, metal.smm_symbol, start, end)
            frame = normalize_spot_frame(_json_to_frame(payload), default_metal=metal.metal)
            frame["raw_symbol"] = frame["raw_symbol"].fillna(metal.display_name)
            frames.append(frame)
        return _combine(frames)

    def fetch_market_features(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        frames = []
        for metal in self.config.metals:
            if not metal.feature_endpoint:
                continue
            payload = self._get(metal.feature_endpoint, metal.smm_symbol, start, end)
            frames.append(normalize_feature_frame(_json_to_frame(payload), default_metal=metal.metal))
        return _combine(frames)

    def _get(
        self,
        endpoint: str,
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> Any:
        if not self.config.base_url:
            raise ValueError("SMM_API_BASE_URL or smm.base_url must be configured")
        if not self.config.token:
            raise ValueError(f"{self.config.token_env} environment variable is required")
        url = endpoint if endpoint.startswith("http") else urljoin(self.config.base_url.rstrip("/") + "/", endpoint)
        headers = {"Authorization": f"Bearer {self.config.token}", "Accept": "application/json"}
        params = {"symbol": symbol}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()


class ChangjiangClient:
    def __init__(self, config: ChangjiangConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Referer": config.referer_url,
            }
        )

    def fetch_spot_prices(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        end_date = _parse_date(end) or date.today()
        start_date = _parse_date(start) or (end_date - timedelta(days=self.config.lookback_days))
        frames = []
        for metal, product_id in self.config.product_ids.items():
            for chunk_start, chunk_end in _date_chunks(start_date, end_date, 365):
                frames.append(self._fetch_product(metal, product_id, chunk_start, chunk_end))
        return _combine(frames)

    def _fetch_product(self, metal: str, product_id: str, start: date, end: date) -> pd.DataFrame:
        response = self.session.post(
            self.config.history_url,
            data={
                "starttime": start.strftime("%Y-%m-%d"),
                "endtime": end.strftime("%Y-%m-%d"),
                "selectid": product_id,
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        if not response.text.strip():
            raise ValueError(f"Changjiang returned an empty response for {metal}")
        tables = pd.read_html(StringIO(response.text))
        if not tables:
            raise ValueError(f"Changjiang returned no table for {metal}")
        table = tables[0].copy()
        table = table[~table["品名"].astype(str).str.contains("区间均价", na=False)]
        table = table.rename(columns={"日期": "trade_date", "均价": "price_cny_per_tonne", "品名": "raw_symbol"})
        table["metal"] = metal
        table["source"] = "Changjiang/CNAL"
        table["trade_date"] = pd.to_datetime(table["trade_date"], format="%y-%m-%d", errors="coerce")
        table["price_cny_per_tonne"] = pd.to_numeric(table["price_cny_per_tonne"], errors="coerce")
        return table[["trade_date", "metal", "price_cny_per_tonne", "source", "raw_symbol"]].dropna(
            subset=["trade_date", "price_cny_per_tonne"]
        )


class ShfeClient:
    def __init__(self, config: ShfeConfig):
        self.config = config

    def fetch_market_features(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        if not self.config.enabled:
            return pd.DataFrame()
        end_date = _parse_date(end) or date.today()
        requested_start = _parse_date(start) or (end_date - timedelta(days=self.config.lookback_days))
        earliest_start = end_date - timedelta(days=self.config.lookback_days)
        start_date = max(requested_start, earliest_start)
        futures = self._fetch_daily_futures(start_date, end_date)
        warrants = self._fetch_warehouse_receipts(start_date, end_date) if self.config.include_warehouse_receipts else pd.DataFrame()
        if futures.empty:
            return warrants
        if warrants.empty:
            futures["inventory_tonne"] = None
            return futures
        merged = futures.merge(warrants, on=["trade_date", "metal"], how="left")
        return merged

    def _fetch_daily_futures(self, start: date, end: date) -> pd.DataFrame:
        import akshare as ak

        rows = []
        for current in _date_range(start, end):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    day = ak.get_shfe_daily(date=current.strftime("%Y%m%d"))
                except Exception:
                    continue
            if day is None or day.empty:
                continue
            for variety, metal in [("CU", "copper"), ("AL", "aluminum")]:
                subset = day[day["variety"].astype(str).str.upper() == variety].copy()
                if subset.empty:
                    continue
                subset["open_interest"] = pd.to_numeric(subset["open_interest"], errors="coerce")
                subset = subset.sort_values("open_interest", ascending=False)
                row = subset.iloc[0]
                rows.append(
                    {
                        "trade_date": pd.to_datetime(current),
                        "metal": metal,
                        "shfe_futures_price": _first_number(row, ["settle", "close"]),
                        "inventory_tonne": None,
                        "premium_discount": None,
                        "import_profit": None,
                        "source": "SHFE",
                    }
                )
        return pd.DataFrame(rows)

    def _fetch_warehouse_receipts(self, start: date, end: date) -> pd.DataFrame:
        import akshare as ak

        rows = []
        for current in _date_range(start, end):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    payload = ak.futures_shfe_warehouse_receipt(date=current.strftime("%Y%m%d"))
                except Exception:
                    continue
            if not isinstance(payload, dict):
                continue
            for key, metal in [("铜", "copper"), ("铝", "aluminum")]:
                frame = payload.get(key)
                if frame is None or frame.empty or "WRTWGHTS" not in frame:
                    continue
                total_rows = frame[frame.get("ROWSTATUS", 0).astype(str) == "1"] if "ROWSTATUS" in frame else pd.DataFrame()
                usable = total_rows if not total_rows.empty else frame
                rows.append(
                    {
                        "trade_date": pd.to_datetime(current),
                        "metal": metal,
                        "inventory_tonne": pd.to_numeric(usable["WRTWGHTS"], errors="coerce").sum(),
                    }
                )
        return pd.DataFrame(rows).drop_duplicates(["trade_date", "metal"], keep="last")


def fetch_public_changjiang_shfe(
    changjiang: ChangjiangConfig,
    shfe: ShfeConfig,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spot = ChangjiangClient(changjiang).fetch_spot_prices(start, end)
    features = ShfeClient(shfe).fetch_market_features(start, end)
    return spot, features


def fetch_akshare_spot_fallback(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    import akshare as ak

    end_date = _parse_date(end) or date.today()
    start_date = _parse_date(start) or (end_date - timedelta(days=540))
    frame = ak.futures_spot_price_daily(
        start_day=start_date.strftime("%Y%m%d"),
        end_day=end_date.strftime("%Y%m%d"),
        vars_list=["CU", "AL"],
    )
    if frame.empty:
        return pd.DataFrame()
    out = frame.rename(columns={"date": "trade_date", "spot_price": "price_cny_per_tonne", "symbol": "metal"})
    out["metal"] = out["metal"].map({"CU": "copper", "AL": "aluminum"})
    out["source"] = "AkShare/100ppi"
    out["raw_symbol"] = out["metal"]
    return out[["trade_date", "metal", "price_cny_per_tonne", "source", "raw_symbol"]].dropna()


def load_spot_prices_from_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return normalize_spot_frame(frame)


def load_spot_prices_from_excel(config: ExcelSourceConfig) -> pd.DataFrame:
    if not config.path:
        raise ValueError("excel.path must be configured")
    frame = pd.read_excel(config.path, sheet_name=config.sheet_name)
    if config.date_column not in frame:
        raise ValueError(f"Excel data missing date column: {config.date_column}")
    rows = []
    for metal, column in config.columns.items():
        if column not in frame:
            raise ValueError(f"Excel data missing price column: {column}")
        part = frame[[config.date_column, column]].copy()
        part = part.rename(columns={config.date_column: "trade_date", column: "price_cny_per_tonne"})
        part["metal"] = metal
        part["source"] = "Changjiang Excel"
        part["raw_symbol"] = column
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["trade_date", "metal", "price_cny_per_tonne", "source", "raw_symbol"])
    out = pd.concat(rows, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["price_cny_per_tonne"] = pd.to_numeric(out["price_cny_per_tonne"], errors="coerce")
    out = out.dropna(subset=["trade_date", "price_cny_per_tonne"])
    out = out[out["price_cny_per_tonne"] > 0]
    out = out.drop_duplicates(["trade_date", "metal"], keep="last")
    return out[["trade_date", "metal", "price_cny_per_tonne", "source", "raw_symbol"]].sort_values(
        ["metal", "trade_date"]
    )


def load_market_features_from_csv(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return normalize_feature_frame(frame)


def normalize_spot_frame(frame: pd.DataFrame, default_metal: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "metal", "price_cny_per_tonne", "source", "raw_symbol"])
    frame = _expand_wide_spot(frame)
    renamed = _rename_by_alias(frame, SPOT_COLUMN_ALIASES)
    required = ["trade_date", "price_cny_per_tonne"]
    missing = [col for col in required if col not in renamed]
    if missing:
        raise ValueError(f"spot price data missing columns: {missing}")
    if "metal" not in renamed:
        if not default_metal:
            raise ValueError("spot price data must include metal when no default metal is supplied")
        renamed["metal"] = default_metal
    if "raw_symbol" not in renamed:
        renamed["raw_symbol"] = renamed["metal"]
    out = renamed[["trade_date", "metal", "price_cny_per_tonne", "raw_symbol"]].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["metal"] = out["metal"].map(_normalize_metal)
    out["price_cny_per_tonne"] = pd.to_numeric(out["price_cny_per_tonne"], errors="coerce")
    out["source"] = "SMM"
    out = out.dropna(subset=["trade_date", "metal", "price_cny_per_tonne"])
    out = out[out["price_cny_per_tonne"] > 0]
    out = out.drop_duplicates(["trade_date", "metal"], keep="last")
    return out.sort_values(["metal", "trade_date"]).reset_index(drop=True)


def normalize_feature_frame(frame: pd.DataFrame, default_metal: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    renamed = _rename_by_alias(frame, FEATURE_COLUMN_ALIASES)
    if "trade_date" not in renamed:
        raise ValueError("market feature data missing trade_date/date column")
    if "metal" not in renamed:
        if not default_metal:
            raise ValueError("market feature data must include metal when no default metal is supplied")
        renamed["metal"] = default_metal
    out = renamed.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["metal"] = out["metal"].map(_normalize_metal)
    for column in ["shfe_futures_price", "inventory_tonne", "premium_discount", "import_profit"]:
        if column not in out:
            out[column] = None
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["source"] = "SMM"
    out = out.dropna(subset=["trade_date", "metal"])
    return out[
        [
            "trade_date",
            "metal",
            "shfe_futures_price",
            "inventory_tonne",
            "premium_discount",
            "import_profit",
            "source",
        ]
    ].drop_duplicates(["trade_date", "metal"], keep="last")


def _json_to_frame(payload: Any) -> pd.DataFrame:
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        for key in ("data", "rows", "result", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return pd.DataFrame(value)
            if isinstance(value, dict):
                return _json_to_frame(value)
        return pd.DataFrame([payload])
    if isinstance(payload, str):
        try:
            return _json_to_frame(json.loads(payload))
        except json.JSONDecodeError:
            raise ValueError("SMM API returned text that is not JSON") from None
    raise ValueError(f"unsupported SMM API payload type: {type(payload).__name__}")


def _rename_by_alias(frame: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    normalized_names = {_normalize_name(col): col for col in frame.columns}
    rename_map = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            original = normalized_names.get(_normalize_name(candidate))
            if original is not None:
                rename_map[original] = target
                break
    return frame.rename(columns=rename_map)


def _expand_wide_spot(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {_normalize_name(col): col for col in frame.columns}
    date_col = columns.get("date") or columns.get("trade_date") or columns.get("日期")
    copper_col = columns.get("copper_price_cny_per_tonne") or columns.get("铜价") or columns.get("铜现货均价")
    aluminum_col = (
        columns.get("aluminum_price_cny_per_tonne")
        or columns.get("aluminium_price_cny_per_tonne")
        or columns.get("铝价")
        or columns.get("铝现货均价")
    )
    if not date_col or not (copper_col or aluminum_col):
        return frame
    rows = []
    for _, row in frame.iterrows():
        if copper_col:
            rows.append({"trade_date": row[date_col], "metal": "copper", "price_cny_per_tonne": row[copper_col]})
        if aluminum_col:
            rows.append({"trade_date": row[date_col], "metal": "aluminum", "price_cny_per_tonne": row[aluminum_col]})
    return pd.DataFrame(rows)


def _normalize_name(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _normalize_metal(value: Any) -> str | None:
    key = str(value).strip().lower().replace(" ", "")
    return METAL_ALIASES.get(key)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return pd.to_datetime(value).date()


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _date_chunks(start: date, end: date, max_days: int):
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=max_days - 1), end)
        yield chunk_start, chunk_end
        chunk_start = chunk_end + timedelta(days=1)


def _first_number(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(value):
            return float(value)
    return None


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
