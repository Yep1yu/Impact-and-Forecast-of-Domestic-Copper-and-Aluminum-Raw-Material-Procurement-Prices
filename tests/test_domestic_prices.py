from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from domestic_prices.config import load_config
from domestic_prices.db import connect, initialize, load_latest_forecasts, load_spot_prices
from domestic_prices.pipeline import run_update


class DomesticPricePipelineTest(unittest.TestCase):
    def test_csv_pipeline_generates_30_day_forecasts_for_both_metals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "spot.csv"
            db_path = tmp_path / "prices.sqlite"
            self._write_sample_spot_csv(csv_path)

            config = load_config("missing-config.yaml")
            config = type(config)(
                database_path=db_path,
                forecast_days=30,
                model_version=config.model_version,
                default_source=config.default_source,
                excel=config.excel,
                changjiang=config.changjiang,
                shfe=config.shfe,
                smm=config.smm,
            )

            result = run_update(config, source="csv", spot_csv=csv_path)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["rows_forecast"], 60)

            conn = connect(db_path)
            initialize(conn)
            try:
                spot = load_spot_prices(conn)
                forecast = load_latest_forecasts(conn)
            finally:
                conn.close()

            self.assertEqual(set(spot["metal"]), {"copper", "aluminum"})
            self.assertFalse(spot.duplicated(["trade_date", "metal"]).any())
            self.assertTrue((spot["price_cny_per_tonne"] > 0).all())
            self.assertEqual(len(forecast[forecast["metal"] == "copper"]), 30)
            self.assertEqual(len(forecast[forecast["metal"] == "aluminum"]), 30)
            self.assertTrue((forecast["lower_bound"] <= forecast["predicted_price_cny_per_tonne"]).all())
            self.assertTrue((forecast["upper_bound"] >= forecast["predicted_price_cny_per_tonne"]).all())

            read_only_conn = connect(db_path, read_only=True)
            try:
                self.assertEqual(len(load_spot_prices(read_only_conn)), len(spot))
                with self.assertRaises(sqlite3.OperationalError):
                    read_only_conn.execute("DELETE FROM domestic_spot_prices")
            finally:
                read_only_conn.close()

    @staticmethod
    def _write_sample_spot_csv(path: Path) -> None:
        dates = pd.date_range("2025-01-01", periods=140, freq="D")
        rows = []
        for idx, date in enumerate(dates):
            copper = 72000 + idx * 18 + 600 * np.sin(idx / 7)
            aluminum = 19500 + idx * 4 + 180 * np.sin(idx / 11)
            rows.append({"date": date.strftime("%Y-%m-%d"), "metal": "copper", "price": round(copper, 2)})
            rows.append({"date": date.strftime("%Y-%m-%d"), "metal": "aluminum", "price": round(aluminum, 2)})
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    unittest.main()
