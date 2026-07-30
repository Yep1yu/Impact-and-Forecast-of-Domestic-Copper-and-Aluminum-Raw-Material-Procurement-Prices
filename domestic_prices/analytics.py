from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MONTHLY_DATA_PATH = ROOT / "domestic_material_monthly_dataset_v1.csv"
FACTOR_COEFFICIENTS_PATH = ROOT / "domestic_material_factor_coefficients_v1.csv"
LITHIUM_COEFFICIENTS_PATH = (
    ROOT / "lithium_carbonate_prediction_outputs" / "lithium_monthly_model_coefficients.csv"
)
EVENTS_PATH = ROOT / "verified_market_events.csv"

MATERIAL_NAMES = {
    "copper_1": "1#铜",
    "aluminum_a00": "A00铝",
    "silver_1": "1#白银",
    "aluminum_adc12": "ADC12",
    "aluminum_zld104": "ZLD104",
    "lithium_carbonate": "碳酸锂",
}

FACTOR_WHITELIST = {
    "copper_1": {
        "SHFE铜主连收盘价_环比": "价格",
        "SHFE铜仓单库存_环比": "库存",
        "SHFE铜主连成交量_环比": "价格",
        "电线电缆光缆及电工器材制造PPI_环比": "需求",
        "光缆产量当期值_环比": "需求",
        "工业增加值同比增长": "宏观",
        "制造业PMI_变化": "宏观",
        "当月进口额环比增长": "供应",
    },
    "aluminum_a00": {
        "SHFE铝主连收盘价_环比": "价格",
        "SHFE铝仓单库存_环比": "库存",
        "汽车产量当期值_环比": "需求",
        "新能源汽车产量当期值_环比": "需求",
        "房间空气调节器产量当期值_环比": "需求",
        "家用电冰箱产量当期值_环比": "需求",
        "制造业PMI_变化": "宏观",
        "企业商品价格煤油电环比增长": "成本",
    },
    "silver_1": {
        "电线电缆光缆及电工器材制造PPI_环比": "需求",
        "光缆产量当期值_环比": "需求",
        "发电量当期值_环比": "需求",
        "制造业PMI_变化": "宏观",
        "PPI当月同比增长": "宏观",
        "中国经济政策不确定性指数_变化": "宏观",
        "中国贸易政策不确定性指数_变化": "宏观",
    },
    "aluminum_adc12": {
        "A00铝_价格月环比": "价格",
        "ADC12_A00价差_滞后1期变化": "成本",
        "汽车产量当期值_环比": "需求",
        "新能源汽车产量当期值_环比": "需求",
        "汽车销量Top50厂商合计_环比": "需求",
        "制造业PMI_变化": "宏观",
        "废铝进口量_环比": "供应",
    },
    "aluminum_zld104": {
        "A00铝_价格月环比": "价格",
        "ZLD104_A00价差_滞后1期变化": "成本",
        "汽车产量当期值_环比": "需求",
        "发电机组产量当期值_环比": "需求",
        "发电量当期值_环比": "需求",
        "工业增加值同比增长": "宏观",
        "制造业PMI_变化": "宏观",
        "SHFE铝仓单库存_环比": "库存",
    },
    "lithium_carbonate": {
        "新能源汽车产量当期值_环比": "需求",
        "汽车产量当期值_环比": "需求",
        "制造业PMI_变化": "宏观",
        "工业增加值同比增长": "宏观",
        "企业商品价格矿产品环比增长": "成本",
        "企业商品价格煤油电环比增长": "成本",
    },
}

FACTOR_RAW_COLUMNS = {
    "PPI当月同比增长": "PPI当月指数",
    "企业商品价格矿产品环比增长": "企业商品价格矿产品指数",
    "企业商品价格煤油电环比增长": "企业商品价格煤油电指数",
    "ADC12_A00价差_滞后1期变化": "ADC12_A00价差",
    "ZLD104_A00价差_滞后1期变化": "ZLD104_A00价差",
}


@dataclass(frozen=True)
class TerminalIndicator:
    label: str
    column: str
    unit: str
    source: str = "国家统计局"


TERMINAL_INDICATORS = {
    "copper_1": [
        TerminalIndicator("光缆", "光缆产量当期值", "万芯千米"),
        TerminalIndicator("电力设备", "发电机组产量当期值", "万千瓦"),
        TerminalIndicator("汽车", "汽车产量当期值", "万辆"),
        TerminalIndicator("新能源汽车", "新能源汽车产量当期值", "万辆"),
        TerminalIndicator("房地产景气", "房地产开发景气指数", "指数"),
    ],
    "aluminum_a00": [
        TerminalIndicator("汽车", "汽车产量当期值", "万辆"),
        TerminalIndicator("新能源汽车", "新能源汽车产量当期值", "万辆"),
        TerminalIndicator("空调", "房间空气调节器产量当期值", "万台"),
        TerminalIndicator("冰箱", "家用电冰箱产量当期值", "万台"),
        TerminalIndicator("电力设备", "发电机组产量当期值", "万千瓦"),
        TerminalIndicator("房地产景气", "房地产开发景气指数", "指数"),
    ],
    "aluminum_adc12": [
        TerminalIndicator("汽车", "汽车产量当期值", "万辆"),
        TerminalIndicator("新能源汽车", "新能源汽车产量当期值", "万辆"),
    ],
    "aluminum_zld104": [
        TerminalIndicator("汽车", "汽车产量当期值", "万辆"),
        TerminalIndicator("电力设备", "发电机组产量当期值", "万千瓦"),
        TerminalIndicator("工业增加值", "工业增加值同比增长", "%"),
    ],
    "silver_1": [
        TerminalIndicator("光缆", "光缆产量当期值", "万芯千米"),
        TerminalIndicator("发电", "发电量当期值", "亿千瓦时"),
        TerminalIndicator("制造业景气", "制造业PMI", "指数"),
    ],
    "lithium_carbonate": [
        TerminalIndicator("新能源汽车", "新能源汽车产量当期值", "万辆"),
        TerminalIndicator("汽车", "汽车产量当期值", "万辆"),
        TerminalIndicator("制造业景气", "制造业PMI", "指数"),
    ],
}


def load_monthly_dataset(path: Path = MONTHLY_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path, encoding="utf-8-sig")
    data["月份"] = pd.to_datetime(data["月份"], errors="coerce")
    return data.dropna(subset=["月份"]).sort_values("月份").reset_index(drop=True)


def load_verified_events(path: Path = EVENTS_PATH) -> pd.DataFrame:
    columns = [
        "event_id",
        "metal",
        "event_date",
        "title",
        "summary",
        "source_name",
        "source_date",
        "source_reference",
        "source_url",
        "verified",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    events = pd.read_csv(path, encoding="utf-8-sig")
    for column in columns:
        if column not in events:
            events[column] = None
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events["verified"] = (
        events["verified"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    )
    required = [
        "event_id",
        "metal",
        "event_date",
        "title",
        "source_name",
        "source_date",
        "source_reference",
        "source_url",
    ]
    valid = events["verified"]
    for column in required:
        valid &= events[column].notna() & events[column].astype(str).str.strip().ne("")
    valid &= events["source_url"].astype(str).str.strip().str.startswith(("https://", "http://"))
    return events.loc[valid, columns].sort_values("event_date").reset_index(drop=True)


def filter_price_history(prices: pd.DataFrame, window: str) -> pd.DataFrame:
    data = prices.sort_values("trade_date").copy()
    if data.empty or window == "全部历史":
        return data
    days = {"近7日": 7, "近30日": 30, "近60日": 60}[window]
    return data[data["trade_date"] >= data["trade_date"].max() - pd.Timedelta(days=days)]


def events_in_range(
    events: pd.DataFrame, metal: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    return events[
        (events["metal"] == metal)
        & (events["event_date"] >= pd.Timestamp(start))
        & (events["event_date"] <= pd.Timestamp(end))
    ].copy()


def load_curated_factor_catalog(
    monthly_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data = monthly_data if monthly_data is not None else load_monthly_dataset()
    coefficients = _load_coefficients()
    rows: list[dict[str, object]] = []
    for metal, allowed in FACTOR_WHITELIST.items():
        material = MATERIAL_NAMES[metal]
        material_coefficients = coefficients[coefficients["品种"] == material]
        for factor, category in allowed.items():
            if factor not in data.columns:
                continue
            match = material_coefficients[material_coefficients["变量"] == factor]
            row = match.iloc[0] if not match.empty else None
            rows.append(
                {
                    "metal": metal,
                    "material": material,
                    "factor": factor,
                    "category": category,
                    "impact_strength": float(row["影响强度_绝对值"]) if row is not None else np.nan,
                    "direction": str(row["回归方向"]) if row is not None else "待验证",
                    "p_value": float(row["p值"]) if row is not None and pd.notna(row["p值"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def factor_series(monthly_data: pd.DataFrame, factor: str) -> pd.DataFrame:
    if monthly_data.empty or factor not in monthly_data:
        return pd.DataFrame(columns=["month", "value"])
    values = pd.to_numeric(monthly_data[factor], errors="coerce")
    return (
        pd.DataFrame({"month": monthly_data["月份"], "value": values})
        .dropna()
        .sort_values("month")
        .reset_index(drop=True)
    )


def build_market_relationship_snapshot(
    monthly_data: pd.DataFrame,
    catalog: pd.DataFrame,
    metal: str,
) -> pd.DataFrame:
    """Build traceable supply, demand, inventory and cost changes from raw series."""
    columns = [
        "category",
        "factor",
        "raw_column",
        "latest_month",
        "mom",
        "yoy",
        "direction",
    ]
    if monthly_data.empty or catalog.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    relevant = catalog[
        (catalog["metal"] == metal)
        & catalog["category"].isin({"供应", "需求", "库存", "成本"})
    ]
    for item in relevant.itertuples(index=False):
        raw_column = _raw_column_for_factor(str(item.factor), monthly_data.columns)
        if raw_column is None:
            continue
        history = pd.DataFrame(
            {
                "month": monthly_data["月份"],
                "value": pd.to_numeric(monthly_data[raw_column], errors="coerce"),
            }
        ).dropna()
        history = history.sort_values("month").reset_index(drop=True)
        if history.empty:
            continue
        latest = history.iloc[-1]
        previous = history.iloc[-2] if len(history) >= 2 else None
        prior_year = history[
            history["month"] <= pd.Timestamp(latest["month"]) - pd.DateOffset(months=12)
        ]
        year_ago = prior_year.iloc[-1] if not prior_year.empty else None
        mom = (
            float(latest["value"] / previous["value"] - 1)
            if previous is not None and previous["value"] != 0
            else np.nan
        )
        yoy = (
            float(latest["value"] / year_ago["value"] - 1)
            if year_ago is not None and year_ago["value"] != 0
            else np.nan
        )
        rows.append(
            {
                "category": item.category,
                "factor": item.factor,
                "raw_column": raw_column,
                "latest_month": pd.Timestamp(latest["month"]),
                "mom": mom,
                "yoy": yoy,
                "direction": _change_direction(mom),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def terminal_snapshot(
    monthly_data: pd.DataFrame, metal: str
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    histories: dict[str, pd.DataFrame] = {}
    for indicator in TERMINAL_INDICATORS.get(metal, []):
        if indicator.column not in monthly_data:
            continue
        history = pd.DataFrame(
            {
                "month": monthly_data["月份"],
                "value": pd.to_numeric(monthly_data[indicator.column], errors="coerce"),
            }
        ).dropna()
        if history.empty:
            continue
        history = history.sort_values("month").reset_index(drop=True)
        latest = history.iloc[-1]
        previous = history.iloc[-2] if len(history) >= 2 else None
        year_ago = history[history["month"] <= latest["month"] - pd.DateOffset(months=12)]
        year_ago_row = year_ago.iloc[-1] if not year_ago.empty else None
        mom = (
            float(latest["value"] / previous["value"] - 1)
            if previous is not None and previous["value"] != 0
            else np.nan
        )
        yoy = (
            float(latest["value"] / year_ago_row["value"] - 1)
            if year_ago_row is not None and year_ago_row["value"] != 0
            else np.nan
        )
        rows.append(
            {
                "indicator": indicator.label,
                "latest_value": float(latest["value"]),
                "unit": indicator.unit,
                "latest_month": pd.Timestamp(latest["month"]),
                "mom": mom,
                "yoy": yoy,
                "direction": _change_direction(mom),
                "source": indicator.source,
            }
        )
        histories[indicator.label] = history
    return pd.DataFrame(rows), histories


def ensure_monthly_horizon(
    forecast: pd.DataFrame,
    spot_history: pd.DataFrame,
    periods: int = 12,
) -> pd.DataFrame:
    if forecast.empty:
        return forecast.copy()
    data = forecast.sort_values("forecast_month").drop_duplicates("forecast_month").copy()
    data["forecast_month"] = pd.to_datetime(data["forecast_month"])
    data["predicted_price_cny_per_tonne"] = pd.to_numeric(
        data["predicted_price_cny_per_tonne"], errors="coerce"
    )
    actual_monthly = (
        spot_history.assign(month=pd.to_datetime(spot_history["trade_date"]).dt.to_period("M"))
        .groupby("month")["price_cny_per_tonne"]
        .mean()
        .astype(float)
    )
    anchor = float(actual_monthly.iloc[-1]) if not actual_monthly.empty else float(
        data["predicted_price_cny_per_tonne"].iloc[0]
    )
    existing_returns = (
        pd.concat(
            [pd.Series([anchor]), data["predicted_price_cny_per_tonne"].reset_index(drop=True)],
            ignore_index=True,
        )
        .pct_change()
        .dropna()
    )
    base_return = float(existing_returns.tail(3).median()) if not existing_returns.empty else 0.0
    base_return = float(np.clip(base_return, -0.08, 0.08))
    last_month = data["forecast_month"].max()
    last_price = float(data.iloc[-1]["predicted_price_cny_per_tonne"])
    template = data.iloc[-1].to_dict()
    rows = [data]
    for step in range(1, max(periods - len(data), 0) + 1):
        damped_return = base_return * (0.72**step)
        last_price = max(last_price * (1 + damped_return), 1.0)
        row = dict(template)
        row["forecast_month"] = last_month + pd.offsets.MonthBegin(step)
        row["predicted_price_cny_per_tonne"] = last_price
        rows.append(pd.DataFrame([row]))
    result = pd.concat(rows, ignore_index=True).head(periods)
    result["predicted_change_pct"] = (
        pd.concat(
            [pd.Series([anchor]), result["predicted_price_cny_per_tonne"]],
            ignore_index=True,
        )
        .pct_change()
        .iloc[1:]
        .to_numpy()
    )
    historical_returns = actual_monthly.pct_change().dropna()
    residual_scale = float(historical_returns.tail(24).std(ddof=0))
    if not np.isfinite(residual_scale) or residual_scale <= 0:
        residual_scale = 0.04
    widths = np.sqrt(np.arange(1, len(result) + 1)) * residual_scale * 1.282
    fallback_lower = result["predicted_price_cny_per_tonne"] * (1 - widths)
    fallback_upper = result["predicted_price_cny_per_tonne"] * (1 + widths)
    if "lower_bound" not in result:
        result["lower_bound"] = fallback_lower
    else:
        result["lower_bound"] = pd.to_numeric(result["lower_bound"], errors="coerce").fillna(
            fallback_lower
        )
    if "upper_bound" not in result:
        result["upper_bound"] = fallback_upper
    else:
        result["upper_bound"] = pd.to_numeric(result["upper_bound"], errors="coerce").fillna(
            fallback_upper
        )
    result["direction"] = np.select(
        [result["predicted_change_pct"] > 0.005, result["predicted_change_pct"] < -0.005],
        ["上涨", "下跌"],
        default="平稳",
    )
    return result


def build_driver_snapshot(
    monthly_data: pd.DataFrame, catalog: pd.DataFrame, metal: str, limit: int = 5
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor in catalog[catalog["metal"] == metal].itertuples(index=False):
        history = factor_series(monthly_data, factor.factor)
        if history.empty:
            continue
        recent = history.tail(36)
        std = float(recent["value"].std(ddof=0))
        signal = 0.0 if not np.isfinite(std) or std == 0 else float(
            (recent.iloc[-1]["value"] - recent["value"].mean()) / std
        )
        direction_sign = -1.0 if factor.direction == "负向" else 1.0
        strength = 0.0 if pd.isna(factor.impact_strength) else float(factor.impact_strength)
        contribution = signal * direction_sign * strength
        rows.append(
            {
                "factor": factor.factor,
                "category": factor.category,
                "contribution": contribution,
                "direction": "支撑" if contribution > 0.05 else "压制" if contribution < -0.05 else "中性",
                "source_period": pd.Timestamp(history.iloc[-1]["month"]),
                "latest_value": float(history.iloc[-1]["value"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["absolute_contribution"] = result["contribution"].abs()
    return result.sort_values("absolute_contribution", ascending=False).head(limit)


def _load_coefficients() -> pd.DataFrame:
    if not FACTOR_COEFFICIENTS_PATH.exists():
        return pd.DataFrame(
            columns=["品种", "变量", "影响强度_绝对值", "回归方向", "p值"]
        )
    coefficients = pd.read_csv(FACTOR_COEFFICIENTS_PATH, encoding="utf-8-sig")
    if LITHIUM_COEFFICIENTS_PATH.exists():
        lithium = pd.read_csv(LITHIUM_COEFFICIENTS_PATH, encoding="utf-8-sig")
        if {"变量", "系数"}.issubset(lithium.columns):
            lithium = lithium[lithium["变量"] != "截距"].copy()
            lithium["品种"] = "碳酸锂"
            lithium["影响强度_绝对值"] = lithium["系数"].abs()
            lithium["回归方向"] = np.where(lithium["系数"] >= 0, "正向", "负向")
            lithium["p值"] = np.nan
            coefficients = pd.concat(
                [
                    coefficients,
                    lithium[["品种", "变量", "影响强度_绝对值", "回归方向", "p值"]],
                ],
                ignore_index=True,
            )
    return coefficients


def _change_direction(change: float) -> str:
    if not np.isfinite(change) or abs(change) < 0.005:
        return "持平"
    return "上升" if change > 0 else "下降"


def _raw_column_for_factor(factor: str, columns: pd.Index) -> str | None:
    explicit = FACTOR_RAW_COLUMNS.get(factor)
    if explicit in columns:
        return explicit
    candidate = factor
    for suffix in (
        "_滞后1期变化",
        "滞后1期变化",
        "当月同比增长",
        "同比增长",
        "环比增长",
        "_环比",
        "环比",
        "_变化",
        "变化",
    ):
        candidate = candidate.replace(suffix, "")
    if candidate in columns:
        return candidate
    indexed = f"{candidate}指数"
    return indexed if indexed in columns else None
