from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS domestic_spot_prices (
    trade_date TEXT NOT NULL,
    metal TEXT NOT NULL,
    price_cny_per_tonne REAL NOT NULL,
    source TEXT NOT NULL,
    raw_symbol TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, metal)
);

CREATE TABLE IF NOT EXISTS market_features (
    trade_date TEXT NOT NULL,
    metal TEXT NOT NULL,
    shfe_futures_price REAL,
    inventory_tonne REAL,
    premium_discount REAL,
    import_profit REAL,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trade_date, metal)
);

CREATE TABLE IF NOT EXISTS daily_forecasts (
    metal TEXT NOT NULL,
    forecast_date TEXT NOT NULL,
    predicted_price_cny_per_tonne REAL NOT NULL,
    lower_bound REAL NOT NULL,
    upper_bound REAL NOT NULL,
    direction TEXT NOT NULL,
    model_version TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    PRIMARY KEY (metal, forecast_date, model_version, generated_at)
);

CREATE TABLE IF NOT EXISTS monthly_forecasts (
    metal TEXT NOT NULL,
    forecast_month TEXT NOT NULL,
    predicted_price_cny_per_tonne REAL NOT NULL,
    source TEXT NOT NULL,
    model_version TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    PRIMARY KEY (metal, forecast_month, model_version, generated_at)
);

CREATE TABLE IF NOT EXISTS update_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    rows_spot INTEGER NOT NULL DEFAULT 0,
    rows_features INTEGER NOT NULL DEFAULT 0,
    rows_forecast INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT
);
"""


def connect(database_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(database_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_spot_prices(conn: sqlite3.Connection, prices: pd.DataFrame) -> int:
    if prices.empty:
        return 0
    rows = prices[
        ["trade_date", "metal", "price_cny_per_tonne", "source", "raw_symbol"]
    ].copy()
    rows["trade_date"] = pd.to_datetime(rows["trade_date"]).dt.strftime("%Y-%m-%d")
    payload = rows.to_records(index=False).tolist()
    conn.executemany(
        """
        INSERT INTO domestic_spot_prices (
            trade_date, metal, price_cny_per_tonne, source, raw_symbol
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(trade_date, metal) DO UPDATE SET
            price_cny_per_tonne = excluded.price_cny_per_tonne,
            source = excluded.source,
            raw_symbol = excluded.raw_symbol,
            updated_at = CURRENT_TIMESTAMP
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def upsert_market_features(conn: sqlite3.Connection, features: pd.DataFrame) -> int:
    if features.empty:
        return 0
    expected = [
        "trade_date",
        "metal",
        "shfe_futures_price",
        "inventory_tonne",
        "premium_discount",
        "import_profit",
        "source",
    ]
    rows = features.copy()
    for column in expected:
        if column not in rows:
            rows[column] = None
    rows["trade_date"] = pd.to_datetime(rows["trade_date"]).dt.strftime("%Y-%m-%d")
    payload = rows[expected].to_records(index=False).tolist()
    conn.executemany(
        """
        INSERT INTO market_features (
            trade_date, metal, shfe_futures_price, inventory_tonne,
            premium_discount, import_profit, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trade_date, metal) DO UPDATE SET
            shfe_futures_price = excluded.shfe_futures_price,
            inventory_tonne = excluded.inventory_tonne,
            premium_discount = excluded.premium_discount,
            import_profit = excluded.import_profit,
            source = excluded.source,
            updated_at = CURRENT_TIMESTAMP
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def replace_latest_forecasts(
    conn: sqlite3.Connection, forecasts: pd.DataFrame, model_version: str
) -> int:
    if forecasts.empty:
        return 0
    metals = sorted(forecasts["metal"].dropna().unique().tolist())
    for metal in metals:
        conn.execute(
            "DELETE FROM daily_forecasts WHERE metal = ? AND model_version = ?",
            (metal, model_version),
        )
    rows = forecasts[
        [
            "metal",
            "forecast_date",
            "predicted_price_cny_per_tonne",
            "lower_bound",
            "upper_bound",
            "direction",
            "model_version",
            "generated_at",
        ]
    ].copy()
    rows["forecast_date"] = pd.to_datetime(rows["forecast_date"]).dt.strftime("%Y-%m-%d")
    payload = rows.to_records(index=False).tolist()
    conn.executemany(
        """
        INSERT INTO daily_forecasts (
            metal, forecast_date, predicted_price_cny_per_tonne, lower_bound,
            upper_bound, direction, model_version, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def replace_latest_monthly_forecasts(
    conn: sqlite3.Connection, forecasts: pd.DataFrame, model_version: str
) -> int:
    if forecasts.empty:
        return 0
    metals = sorted(forecasts["metal"].dropna().unique().tolist())
    for metal in metals:
        conn.execute(
            "DELETE FROM monthly_forecasts WHERE metal = ? AND model_version = ?",
            (metal, model_version),
        )
    rows = forecasts[
        [
            "metal",
            "forecast_month",
            "predicted_price_cny_per_tonne",
            "source",
            "model_version",
            "generated_at",
        ]
    ].copy()
    rows["forecast_month"] = pd.to_datetime(rows["forecast_month"]).dt.strftime("%Y-%m")
    payload = rows.to_records(index=False).tolist()
    conn.executemany(
        """
        INSERT INTO monthly_forecasts (
            metal, forecast_month, predicted_price_cny_per_tonne, source,
            model_version, generated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    conn.commit()
    return len(payload)


def load_spot_prices(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM domestic_spot_prices ORDER BY metal, trade_date",
        conn,
        parse_dates=["trade_date"],
    )


def load_market_features(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM market_features ORDER BY metal, trade_date",
        conn,
        parse_dates=["trade_date"],
    )


def load_latest_forecasts(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    WITH latest AS (
        SELECT metal, MAX(generated_at) AS generated_at
        FROM daily_forecasts
        GROUP BY metal
    )
    SELECT f.*
    FROM daily_forecasts f
    JOIN latest l
      ON f.metal = l.metal
     AND f.generated_at = l.generated_at
    ORDER BY f.metal, f.forecast_date
    """
    return pd.read_sql_query(query, conn, parse_dates=["forecast_date", "generated_at"])


def load_latest_monthly_forecasts(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    WITH latest AS (
        SELECT metal, MAX(generated_at) AS generated_at
        FROM monthly_forecasts
        GROUP BY metal
    )
    SELECT f.*
    FROM monthly_forecasts f
    JOIN latest l
      ON f.metal = l.metal
     AND f.generated_at = l.generated_at
    ORDER BY f.metal, f.forecast_month
    """
    return pd.read_sql_query(query, conn, parse_dates=["forecast_month", "generated_at"])


def load_update_runs(conn: sqlite3.Connection, limit: int = 20) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM update_runs ORDER BY started_at DESC LIMIT ?",
        conn,
        params=(limit,),
        parse_dates=["started_at", "finished_at"],
    )


def start_run(conn: sqlite3.Connection, run_id: str, started_at: str, source: str) -> None:
    conn.execute(
        """
        INSERT INTO update_runs (run_id, started_at, status, source)
        VALUES (?, ?, 'running', ?)
        """,
        (run_id, started_at, source),
    )
    conn.commit()


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    finished_at: str,
    status: str,
    rows_spot: int,
    rows_features: int,
    rows_forecast: int,
    error_summary: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE update_runs
        SET finished_at = ?, status = ?, rows_spot = ?, rows_features = ?,
            rows_forecast = ?, error_summary = ?
        WHERE run_id = ?
        """,
        (
            finished_at,
            status,
            rows_spot,
            rows_features,
            rows_forecast,
            error_summary,
            run_id,
        ),
    )
    conn.commit()
