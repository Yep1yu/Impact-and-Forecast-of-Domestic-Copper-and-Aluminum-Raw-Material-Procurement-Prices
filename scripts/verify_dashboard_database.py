from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


REQUIRED_TABLES = {
    "domestic_spot_prices",
    "daily_forecasts",
    "monthly_forecasts",
    "update_runs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a dashboard SQLite database before publishing it.")
    parser.add_argument("database", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = args.database.resolve()
    if not database.is_file():
        raise SystemExit(f"Database does not exist: {database}")

    uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")

        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        missing = REQUIRED_TABLES - tables
        if missing:
            raise RuntimeError(f"Missing database tables: {sorted(missing)}")

        spot_rows = conn.execute("SELECT COUNT(*) FROM domestic_spot_prices").fetchone()[0]
        forecast_rows = conn.execute("SELECT COUNT(*) FROM daily_forecasts").fetchone()[0]
        monthly_rows = conn.execute("SELECT COUNT(*) FROM monthly_forecasts").fetchone()[0]
        latest_run = conn.execute(
            "SELECT status, started_at FROM update_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if not spot_rows or not forecast_rows or not monthly_rows:
            raise RuntimeError(
                "Database contains empty dashboard datasets: "
                f"spot={spot_rows}, daily={forecast_rows}, monthly={monthly_rows}"
            )
        if not latest_run or latest_run[0] != "success":
            raise RuntimeError(f"Latest update run is not successful: {latest_run}")

        latest_spot = conn.execute(
            "SELECT MAX(trade_date) FROM domestic_spot_prices"
        ).fetchone()[0]
        print(f"integrity: {integrity}")
        print(f"latest_run: {latest_run[0]} at {latest_run[1]}")
        print(f"latest_spot_date: {latest_spot}")
        print(f"rows: spot={spot_rows}, daily={forecast_rows}, monthly={monthly_rows}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
