from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MetalConfig:
    metal: str
    display_name: str
    smm_symbol: str
    spot_endpoint: str
    feature_endpoint: str | None = None


@dataclass(frozen=True)
class SmmConfig:
    base_url: str = ""
    token_env: str = "SMM_API_TOKEN"
    timeout_seconds: int = 30
    metals: list[MetalConfig] = field(default_factory=list)

    @property
    def token(self) -> str:
        return os.environ.get(self.token_env, "")


@dataclass(frozen=True)
class ChangjiangConfig:
    history_url: str = "https://market.cnal.com/historical/search.html"
    referer_url: str = "https://market.cnal.com/historical/cj.html"
    timeout_seconds: int = 30
    lookback_days: int = 365
    product_ids: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ShfeConfig:
    enabled: bool = True
    lookback_days: int = 14
    include_warehouse_receipts: bool = False


@dataclass(frozen=True)
class ExcelSourceConfig:
    path: str = ""
    sheet_name: str = "ccmn_changjiang_avg_prices"
    date_column: str = "date"
    columns: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AppConfig:
    database_path: Path
    forecast_days: int
    model_version: str
    default_source: str
    excel: ExcelSourceConfig
    changjiang: ChangjiangConfig
    shfe: ShfeConfig
    smm: SmmConfig


def _default_config() -> dict[str, Any]:
    return {
        "database_path": "domestic_procurement_prices.sqlite",
        "forecast_days": 30,
        "model_version": "domestic-price-ridge-v1",
        "default_source": "excel",
        "excel": {
            "path": "D:/BSH实习/铜、铝采购影响分析/原材料日度价格.xlsx",
            "sheet_name": "ccmn_changjiang_avg_prices",
            "date_column": "date",
            "columns": {
                "copper_1": "1#铜",
                "aluminum_a00": "A00铝",
                "silver_1": "1#白银",
                "aluminum_adc12": "铝合金ADC12",
                "aluminum_zld104": "铸造铝合金锭(ZLD104)",
                "lithium_carbonate": "碳酸锂",
            },
        },
        "changjiang": {
            "history_url": "https://market.cnal.com/historical/search.html",
            "referer_url": "https://market.cnal.com/historical/cj.html",
            "timeout_seconds": 30,
            "lookback_days": 365,
            "product_ids": {"copper": "2", "aluminum": "3"},
        },
        "shfe": {
            "enabled": True,
            "lookback_days": 14,
            "include_warehouse_receipts": False,
        },
        "smm": {
            "base_url": "",
            "token_env": "SMM_API_TOKEN",
            "timeout_seconds": 30,
            "metals": [
                {
                    "metal": "copper",
                    "display_name": "SMM 1# Copper Cathode",
                    "smm_symbol": "SMM_1_COPPER_CATHODE",
                    "spot_endpoint": "",
                    "feature_endpoint": "",
                },
                {
                    "metal": "aluminum",
                    "display_name": "SMM A00 Aluminum Ingot",
                    "smm_symbol": "SMM_A00_ALUMINUM_INGOT",
                    "spot_endpoint": "",
                    "feature_endpoint": "",
                },
            ],
        },
    }


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    raw = _default_config()
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        raw = _deep_merge(raw, loaded)

    database_override = os.environ.get("DOMESTIC_PRICE_DB")
    if database_override:
        raw["database_path"] = database_override

    base_url_override = os.environ.get("SMM_API_BASE_URL")
    if base_url_override:
        raw["smm"]["base_url"] = base_url_override

    timeout_override = os.environ.get("SMM_API_TIMEOUT_SECONDS")
    if timeout_override:
        raw["smm"]["timeout_seconds"] = int(timeout_override)

    cj = ChangjiangConfig(
        history_url=raw["changjiang"].get("history_url", "https://market.cnal.com/historical/search.html"),
        referer_url=raw["changjiang"].get("referer_url", "https://market.cnal.com/historical/cj.html"),
        timeout_seconds=int(raw["changjiang"].get("timeout_seconds", 30)),
        lookback_days=int(raw["changjiang"].get("lookback_days", 365)),
        product_ids=dict(raw["changjiang"].get("product_ids", {"copper": "2", "aluminum": "3"})),
    )
    shfe = ShfeConfig(
        enabled=bool(raw["shfe"].get("enabled", True)),
        lookback_days=int(raw["shfe"].get("lookback_days", 14)),
        include_warehouse_receipts=bool(raw["shfe"].get("include_warehouse_receipts", False)),
    )
    excel = ExcelSourceConfig(
        path=str(raw["excel"].get("path", "")),
        sheet_name=str(raw["excel"].get("sheet_name", "ccmn_changjiang_avg_prices")),
        date_column=str(raw["excel"].get("date_column", "date")),
        columns=dict(raw["excel"].get("columns", {})),
    )
    metals = [MetalConfig(**item) for item in raw["smm"].get("metals", [])]
    smm = SmmConfig(
        base_url=raw["smm"].get("base_url", ""),
        token_env=raw["smm"].get("token_env", "SMM_API_TOKEN"),
        timeout_seconds=int(raw["smm"].get("timeout_seconds", 30)),
        metals=metals,
    )
    return AppConfig(
        database_path=Path(raw["database_path"]),
        forecast_days=int(raw.get("forecast_days", 30)),
        model_version=str(raw.get("model_version", "domestic-price-ridge-v1")),
        default_source=str(raw.get("default_source", "changjiang_shfe")),
        excel=excel,
        changjiang=cj,
        shfe=shfe,
        smm=smm,
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
