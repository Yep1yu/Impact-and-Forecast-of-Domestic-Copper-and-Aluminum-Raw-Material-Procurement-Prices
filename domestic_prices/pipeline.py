from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .config import AppConfig
from .db import (
    connect,
    finish_run,
    initialize,
    load_market_features,
    load_spot_prices,
    replace_latest_forecasts,
    start_run,
    upsert_market_features,
    upsert_spot_prices,
)
from .model import generate_forecasts
from .sources import SmmClient, load_market_features_from_csv, load_spot_prices_from_csv
from .sources import fetch_akshare_spot_fallback, fetch_public_changjiang_shfe
from .sources import load_spot_prices_from_excel


def run_update(
    config: AppConfig,
    *,
    source: str | None = None,
    spot_csv: str | Path | None = None,
    features_csv: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, object]:
    conn = connect(config.database_path)
    initialize(conn)
    run_id = str(uuid4())
    started_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    source = source or config.default_source
    start_run(conn, run_id, started_at, source)
    rows_spot = rows_features = rows_forecast = 0
    try:
        spot_prices, market_features = _fetch_inputs(config, source, spot_csv, features_csv, start, end)
        _validate_spot_prices(spot_prices, _required_metals(config, source))
        rows_spot = upsert_spot_prices(conn, spot_prices)
        rows_features = upsert_market_features(conn, market_features)

        full_spot = load_spot_prices(conn)
        full_features = load_market_features(conn)
        forecasts = generate_forecasts(
            full_spot,
            full_features,
            forecast_days=config.forecast_days,
            model_version=config.model_version,
        )
        rows_forecast = replace_latest_forecasts(conn, forecasts, config.model_version)
        finish_run(
            conn,
            run_id,
            datetime.now(UTC).replace(microsecond=0).isoformat(),
            "success",
            rows_spot,
            rows_features,
            rows_forecast,
        )
        return {
            "run_id": run_id,
            "status": "success",
            "rows_spot": rows_spot,
            "rows_features": rows_features,
            "rows_forecast": rows_forecast,
            "database_path": str(config.database_path),
        }
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        finish_run(
            conn,
            run_id,
            datetime.now(UTC).replace(microsecond=0).isoformat(),
            "failed",
            rows_spot,
            rows_features,
            rows_forecast,
            error,
        )
        raise RuntimeError(error) from exc
    finally:
        conn.close()


def _fetch_inputs(
    config: AppConfig,
    source: str,
    spot_csv: str | Path | None,
    features_csv: str | Path | None,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if source == "csv":
        if not spot_csv:
            raise ValueError("--spot-csv is required when --source csv")
        return load_spot_prices_from_csv(spot_csv), load_market_features_from_csv(features_csv)
    if source == "excel":
        return load_spot_prices_from_excel(config.excel), pd.DataFrame()
    if source == "changjiang_shfe":
        return fetch_public_changjiang_shfe(config.changjiang, config.shfe, start, end)
    if source == "akshare_spot":
        return fetch_akshare_spot_fallback(start, end), pd.DataFrame()
    if source == "api":
        client = SmmClient(config.smm)
        return client.fetch_spot_prices(start, end), client.fetch_market_features(start, end)
    raise ValueError("--source must be excel, changjiang_shfe, akshare_spot, api, or csv")


def _validate_spot_prices(prices: pd.DataFrame, required_metals: set[str] | None = None) -> None:
    if prices.empty:
        raise ValueError("no spot price rows were loaded")
    if required_metals:
        metals = set(prices["metal"].dropna().unique())
        missing = sorted(required_metals - metals)
        if missing:
            raise ValueError(f"missing required metals: {missing}")
    if prices.duplicated(["trade_date", "metal"]).any():
        raise ValueError("spot prices contain duplicate trade_date + metal rows")
    if (prices["price_cny_per_tonne"] <= 0).any():
        raise ValueError("spot prices must be positive")


def _required_metals(config: AppConfig, source: str) -> set[str] | None:
    if source == "excel":
        return set(config.excel.columns.keys())
    if source == "api":
        return {metal.metal for metal in config.smm.metals}
    if source in {"changjiang_shfe", "akshare_spot"}:
        return {"copper", "aluminum"}
    return None
