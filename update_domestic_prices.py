from __future__ import annotations

import argparse

from domestic_prices.config import load_config
from domestic_prices.pipeline import run_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update domestic copper/aluminum daily forecasts.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--source",
        choices=["excel", "changjiang_shfe", "akshare_spot", "api", "csv"],
        help="Input source. Defaults to config.yaml default_source.",
    )
    parser.add_argument("--spot-csv", help="CSV file containing SMM spot prices.")
    parser.add_argument("--features-csv", help="Optional CSV file containing market features.")
    parser.add_argument("--database", help="Override SQLite database path.")
    parser.add_argument("--start", help="Optional API start date, YYYY-MM-DD.")
    parser.add_argument("--end", help="Optional API end date, YYYY-MM-DD.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.database:
        config = type(config)(
            database_path=args.database,
            forecast_days=config.forecast_days,
            model_version=config.model_version,
            default_source=config.default_source,
            excel=config.excel,
            changjiang=config.changjiang,
            shfe=config.shfe,
            smm=config.smm,
        )
    result = run_update(
        config,
        source=args.source,
        spot_csv=args.spot_csv,
        features_csv=args.features_csv,
        start=args.start,
        end=args.end,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
