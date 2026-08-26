from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app import (
    NAVIGATION_PAGES,
    PAGE_DESCRIPTIONS,
    apply_compact_price_ranges,
    build_factor_strength_figure,
    build_price_history_figure,
    build_trend_figure,
    plain_factor_name,
    price_unit,
)
from domestic_prices.analytics import (
    FACTOR_WHITELIST,
    build_market_relationship_snapshot,
    ensure_monthly_horizon,
    filter_price_history,
    load_curated_factor_catalog,
    load_verified_events,
    terminal_snapshot,
)
from domestic_prices.db import (
    connect,
    initialize,
    load_latest_forecast_driver_contributions,
    replace_forecast_driver_contributions,
)
from domestic_prices.lithium_model import (
    DAILY_MODEL_VERSION,
    MONTHLY_MODEL_VERSION,
    build_lithium_forecasts,
)


class DashboardAnalysisTest(unittest.TestCase):
    def test_navigation_and_core_page_descriptions(self) -> None:
        self.assertEqual(
            NAVIGATION_PAGES,
            [
                "首页概览",
                "影响分析",
                "预测总览",
                "模型评估",
                "模型说明",
                "报告中心",
                "更新记录",
            ],
        )
        self.assertEqual(
            set(PAGE_DESCRIPTIONS),
            {"首页概览", "影响分析", "预测总览", "模型评估"},
        )
        self.assertIn("历史不含税现货均价", PAGE_DESCRIPTIONS["首页概览"])
        self.assertIn("不把现货或期货价格本身作为影响因子展示", PAGE_DESCRIPTIONS["影响分析"])
        self.assertIn("未来30天日度预测", PAGE_DESCRIPTIONS["预测总览"])
        self.assertIn("MAE、MAPE和RMSE", PAGE_DESCRIPTIONS["模型评估"])

    def test_factor_strength_hover_is_left_aligned_and_neutral(self) -> None:
        factors = pd.DataFrame(
            [
                {
                    "factor": "制造业PMI_变化",
                    "impact_strength": 0.42,
                    "direction": "正向",
                    "category": "宏观",
                    "p_value": 0.03,
                },
                {
                    "factor": "SHFE铝仓单库存_环比",
                    "impact_strength": 0.28,
                    "direction": "负向",
                    "category": "库存",
                    "p_value": 0.08,
                },
            ]
        )
        fig = build_factor_strength_figure(factors)
        self.assertEqual(fig.layout.hoverlabel.align, "left")
        self.assertIn("影响强度排序", fig.data[0].hovertemplate)
        self.assertNotIn("显著影响排序", fig.data[0].hovertemplate)

    def test_silver_uses_price_per_kilogram(self) -> None:
        self.assertEqual(price_unit("silver_1"), "元/千克")
        self.assertEqual(price_unit("copper_1"), "元/吨")

    def test_price_axes_follow_data_instead_of_annotations(self) -> None:
        fig = go.Figure()
        dates = pd.Series(pd.date_range("2026-01-01", periods=10, freq="D"))
        values = pd.Series(np.linspace(90.0, 100.0, 10))
        apply_compact_price_ranges(fig, dates, values)
        self.assertLess(float(fig.layout.yaxis.range[1]), 105.0)
        self.assertGreater(pd.Timestamp(fig.layout.xaxis.range[0]), pd.Timestamp("2025-12-20"))
        self.assertLess(pd.Timestamp(fig.layout.xaxis.range[1]), pd.Timestamp("2026-01-20"))

    def test_daily_forecast_band_uses_lower_upper_label(self) -> None:
        spot = pd.DataFrame(
            {
                "trade_date": pd.date_range("2026-01-01", periods=3, freq="D"),
                "price_cny_per_tonne": [100.0, 101.0, 102.0],
            }
        )
        forecast = pd.DataFrame(
            {
                "forecast_date": pd.date_range("2026-01-04", periods=2, freq="D"),
                "predicted_price_cny_per_tonne": [103.0, 104.0],
                "lower_bound": [98.0, 99.0],
                "upper_bound": [108.0, 109.0],
            }
        )
        fig = build_trend_figure(spot, forecast, "#A32035", pd.DataFrame())
        trace_names = [trace.name for trace in fig.data]
        self.assertIn("预测下限-上限", trace_names)
        self.assertNotIn("80% 可能范围", trace_names)
        self.assertEqual(fig.layout.hovermode, "x unified")
        self.assertEqual(fig.layout.hoverlabel.align, "left")
        self.assertEqual(fig.layout.hoverlabel.font.color, "#2A2A2D")

    def test_price_history_hover_text_has_explicit_contrast(self) -> None:
        prices = pd.DataFrame(
            {
                "trade_date": pd.date_range("2026-01-01", periods=3, freq="D"),
                "price_cny_per_tonne": [100.0, 101.0, 102.0],
            }
        )
        fig = build_price_history_figure(prices, pd.DataFrame(), "#A32035")
        self.assertEqual(fig.layout.hoverlabel.bgcolor, "#FFFFFF")
        self.assertEqual(fig.layout.hoverlabel.font.color, "#2A2A2D")

    def test_ambiguous_import_factor_uses_full_name(self) -> None:
        self.assertEqual(
            plain_factor_name("当月进口额环比增长"),
            "中国海关货物进口总额月度环比增速（美元计价）",
        )

    def test_price_windows_and_monthly_horizon(self) -> None:
        prices = pd.DataFrame(
            {
                "trade_date": pd.date_range("2025-01-01", periods=120, freq="D"),
                "price_cny_per_tonne": np.linspace(100, 130, 120),
            }
        )
        self.assertEqual(len(filter_price_history(prices, "全部历史")), 120)
        self.assertEqual(len(filter_price_history(prices, "近7日")), 8)
        forecast = pd.DataFrame(
            {
                "forecast_month": pd.date_range("2025-05-01", periods=3, freq="MS"),
                "predicted_price_cny_per_tonne": [131.0, 132.0, 133.0],
                "source": "test",
                "model_version": "test",
                "generated_at": "2025-04-30T00:00:00+00:00",
            }
        )
        extended = ensure_monthly_horizon(forecast, prices, periods=12)
        self.assertEqual(len(extended), 12)
        self.assertTrue((extended["lower_bound"] <= extended["predicted_price_cny_per_tonne"]).all())
        self.assertTrue((extended["upper_bound"] >= extended["predicted_price_cny_per_tonne"]).all())
        expected_change = extended["predicted_price_cny_per_tonne"].iloc[1] / extended[
            "predicted_price_cny_per_tonne"
        ].iloc[0] - 1
        self.assertAlmostEqual(float(extended.iloc[1]["predicted_change_pct"]), expected_change)

    def test_events_require_verification_and_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            pd.DataFrame(
                [
                    {
                        "event_id": "valid",
                        "metal": "copper_1",
                        "event_date": "2026-01-01",
                        "title": "有效事件",
                        "summary": "摘要",
                        "source_name": "报告",
                        "source_date": "2026-01",
                        "source_reference": "第1页",
                        "source_url": "https://example.com/source",
                        "verified": True,
                    },
                    {
                        "event_id": "invalid",
                        "metal": "copper_1",
                        "event_date": "2026-02-01",
                        "title": "无来源事件",
                        "summary": "摘要",
                        "source_name": "",
                        "source_date": "",
                        "source_reference": "",
                        "source_url": "",
                        "verified": True,
                    },
                    {
                        "event_id": "missing-source-date",
                        "metal": "copper_1",
                        "event_date": "2026-03-01",
                        "title": "缺少报告日期",
                        "summary": "摘要",
                        "source_name": "报告",
                        "source_date": "",
                        "source_reference": "第2页",
                        "source_url": "",
                        "verified": True,
                    },
                    {
                        "event_id": "missing-web-url",
                        "metal": "copper_1",
                        "event_date": "2026-04-01",
                        "title": "缺少网页链接",
                        "summary": "摘要",
                        "source_name": "网络新闻",
                        "source_date": "2026-04-01",
                        "source_reference": "正文",
                        "source_url": "",
                        "verified": True,
                    },
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")
            events = load_verified_events(path)
            self.assertEqual(events["event_id"].tolist(), ["valid"])

    def test_relationship_snapshot_uses_raw_monthly_series(self) -> None:
        months = pd.date_range("2025-01-01", periods=14, freq="MS")
        monthly = pd.DataFrame(
            {
                "月份": months,
                "汽车产量当期值": np.arange(100, 114, dtype=float),
                "汽车产量当期值_环比": np.linspace(0.01, 0.03, 14),
                "SHFE铝仓单库存": np.arange(200, 214, dtype=float),
                "SHFE铝仓单库存_环比": np.linspace(-0.02, 0.01, 14),
            }
        )
        catalog = pd.DataFrame(
            [
                {
                    "metal": "aluminum_a00",
                    "factor": "汽车产量当期值_环比",
                    "category": "需求",
                },
                {
                    "metal": "aluminum_a00",
                    "factor": "SHFE铝仓单库存_环比",
                    "category": "库存",
                },
            ]
        )
        snapshot = build_market_relationship_snapshot(
            monthly,
            catalog,
            "aluminum_a00",
        )
        self.assertEqual(set(snapshot["category"]), {"需求", "库存"})
        demand = snapshot[snapshot["category"] == "需求"].iloc[0]
        self.assertEqual(demand["raw_column"], "汽车产量当期值")
        self.assertAlmostEqual(float(demand["mom"]), 113 / 112 - 1)
        self.assertAlmostEqual(float(demand["yoy"]), 113 / 101 - 1)

    def test_curated_lithium_factors_exclude_copper_inputs(self) -> None:
        months = pd.date_range("2023-01-01", periods=36, freq="MS")
        data = pd.DataFrame({"月份": months})
        for factor in FACTOR_WHITELIST["lithium_carbonate"]:
            data[factor] = np.arange(len(months), dtype=float)
        catalog = load_curated_factor_catalog(data)
        lithium = catalog[catalog["metal"] == "lithium_carbonate"]
        self.assertEqual(set(lithium["factor"]), set(FACTOR_WHITELIST["lithium_carbonate"]))
        self.assertFalse(lithium["factor"].str.contains("废铜|铜矿").any())

    def test_terminal_snapshot_uses_latest_non_null_period(self) -> None:
        months = pd.date_range("2025-01-01", periods=15, freq="MS")
        data = pd.DataFrame(
            {
                "月份": months,
                "新能源汽车产量当期值": [100 + index for index in range(13)] + [np.nan, np.nan],
                "汽车产量当期值": [200 + index for index in range(13)] + [np.nan, np.nan],
                "制造业PMI": [50 + index * 0.1 for index in range(14)] + [np.nan],
            }
        )
        snapshot, _ = terminal_snapshot(data, "lithium_carbonate")
        nev = snapshot[snapshot["indicator"] == "新能源汽车"].iloc[0]
        self.assertEqual(pd.Timestamp(nev["latest_month"]), months[12])

    def test_database_migration_and_contributions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.sqlite"
            raw = sqlite3.connect(path)
            raw.execute(
                """
                CREATE TABLE monthly_forecasts (
                    metal TEXT, forecast_month TEXT,
                    predicted_price_cny_per_tonne REAL, source TEXT,
                    model_version TEXT, generated_at TEXT
                )
                """
            )
            raw.commit()
            raw.close()
            conn = connect(path)
            initialize(conn)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(monthly_forecasts)")}
            self.assertTrue(
                {"lower_bound", "upper_bound", "direction", "predicted_change_pct"}.issubset(columns)
            )
            contributions = pd.DataFrame(
                [
                    {
                        "metal": "lithium_carbonate",
                        "forecast_period": "2026-08",
                        "horizon_type": "monthly",
                        "factor": "新能源汽车产量当期值_环比",
                        "factor_category": "需求",
                        "contribution": 0.2,
                        "direction": "支撑",
                        "source_period": "2026-05",
                        "model_version": "test",
                        "generated_at": "2026-07-01T00:00:00+00:00",
                    }
                ]
            )
            replace_forecast_driver_contributions(conn, contributions, "test")
            loaded = load_latest_forecast_driver_contributions(conn)
            conn.close()
            self.assertEqual(len(loaded), 1)

    def test_lithium_models_return_required_horizons(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2023-01-02", periods=760)
        returns = rng.normal(0.0002, 0.012, len(dates))
        prices = 90000 * np.exp(np.cumsum(returns))
        daily = pd.DataFrame(
            {
                "date": dates,
                "settlement_price": prices,
                "volume": rng.integers(10000, 50000, len(dates)),
                "open_interest": rng.integers(50000, 90000, len(dates)),
            }
        )
        months = pd.date_range("2023-01-01", periods=43, freq="MS")
        factors = pd.DataFrame({"月份": months})
        for index, factor in enumerate(FACTOR_WHITELIST["lithium_carbonate"]):
            factors[factor] = np.sin(np.arange(len(months)) / (4 + index)) * 0.05
        result = build_lithium_forecasts(daily, factors, monthly_periods=12, daily_periods=30)
        self.assertEqual(len(result.monthly_forecast), 12)
        self.assertEqual(len(result.daily_forecast), 30)
        self.assertEqual(set(result.monthly_forecast["model_version"]), {MONTHLY_MODEL_VERSION})
        self.assertEqual(set(result.daily_forecast["model_version"]), {DAILY_MODEL_VERSION})
        self.assertTrue(
            (
                result.daily_forecast["lower_bound"]
                <= result.daily_forecast["predicted_price_cny_per_tonne"]
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
