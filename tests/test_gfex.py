from __future__ import annotations

import unittest

from domestic_prices.gfex import aggregate_lithium_day, lithium_contracts


class GfexLithiumTest(unittest.TestCase):
    def test_filters_subtotal_and_averages_contract_settlements(self) -> None:
        payload = {
            "code": "0",
            "data": [
                {
                    "variety": "碳酸锂",
                    "varietyOrder": "lc",
                    "delivMonth": "2609",
                    "clearPrice": 148840,
                    "volumn": 158164,
                    "openInterest": 233810,
                },
                {
                    "variety": "碳酸锂",
                    "varietyOrder": "lc",
                    "delivMonth": "2610",
                    "clearPrice": 149140,
                    "volumn": 4927,
                    "openInterest": 21175,
                },
                {
                    "variety": "碳酸锂小计",
                    "varietyOrder": "lc",
                    "delivMonth": None,
                    "clearPrice": None,
                    "volumn": 163091,
                    "openInterest": 254985,
                },
            ],
        }
        contracts = lithium_contracts(payload, "2026-08-12")
        daily = aggregate_lithium_day(contracts)
        self.assertEqual(contracts["contract"].tolist(), ["lc2609", "lc2610"])
        self.assertEqual(int(daily.iloc[0]["contract_count"]), 2)
        self.assertEqual(float(daily.iloc[0]["settlement_price"]), 148990.0)
        self.assertEqual(float(daily.iloc[0]["volume"]), 163091.0)


if __name__ == "__main__":
    unittest.main()
