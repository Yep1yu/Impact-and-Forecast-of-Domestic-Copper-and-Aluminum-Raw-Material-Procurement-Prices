from __future__ import annotations

import random
import sqlite3
import base64
import html
import re
from io import BytesIO
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots

from domestic_prices.analytics import (
    build_market_relationship_snapshot,
    build_driver_snapshot,
    ensure_monthly_horizon,
    events_in_range,
    factor_series,
    filter_price_history,
    load_curated_factor_catalog,
    load_lithium_screening_coefficients,
    load_monthly_dataset,
    load_verified_events,
    terminal_snapshot,
)
from domestic_prices.config import load_config
from domestic_prices.db import (
    connect,
    load_latest_forecast_driver_contributions,
    load_latest_monthly_forecasts,
    load_latest_forecasts,
    load_market_features,
    load_spot_prices,
    load_update_runs,
)
from domestic_prices.news import load_news_cache
DEFAULT_DISPLAY = {
    "copper": "铜",
    "aluminum": "铝",
    "copper_1": "1#铜",
    "aluminum_a00": "A00铝",
    "silver_1": "1#白银",
    "aluminum_adc12": "铝合金ADC12",
    "aluminum_zld104": "铸造铝合金锭(ZLD104)",
    "lithium_carbonate": "碳酸锂",
}
PALETTE = ["#8B1E2D", "#A63D2F", "#4B5563", "#B45359", "#7F1D1D"]
METAL_COLORS = {
    "copper_1": "#A63D2F",
    "aluminum_a00": "#8B1E2D",
    "silver_1": "#4B5563",
    "aluminum_adc12": "#B45359",
    "aluminum_zld104": "#7F1D1D",
    "lithium_carbonate": "#A32035",
}
NAVIGATION_PAGES = [
    "首页概览",
    "影响分析",
    "预测总览",
    "模型评估",
    "模型说明",
    "报告中心",
    "更新记录",
]
PAGE_DESCRIPTIONS = {
    "首页概览": (
        "汇总各原材料最新价格、历史不含税现货均价、价格走势和关键市场事件，"
        "并展示每日更新的铜、铝、白银和碳酸锂等相关资讯。"
    ),
    "影响分析": (
        "围绕供应、需求、库存、成本、宏观和交易状态等方面，收集可能影响原材料价格的指标，"
        "不把现货或期货价格本身作为影响因子展示；价格历史只作为预测模型的价格基准和惯性项。"
    ),
    "预测总览": (
        "展示当前原材料的最新现货价和未来价格判断，包括未来30天日度预测、预测上下限和月度均价趋势，"
        "帮助了解价格可能的变化方向和幅度。"
    ),
    "模型评估": (
        "将模型预测结果与真实价格进行比较。通过MAE、MAPE和RMSE了解平均偏差及较大误差情况；"
        "三项指标越低，代表模型在所选历史区间内的预测效果越好。"
    ),
}


def price_unit(metal: str) -> str:
    return "元/千克" if metal == "silver_1" else "元/吨"


FACTOR_CN = {
    "price_cny_per_tonne": "最新不含税现货均价",
    "lag_1": "1日前价格",
    "lag_5": "5日前价格",
    "lag_10": "10日前价格",
    "lag_20": "20日前价格",
    "ma5": "5日均价",
    "ma10": "10日均价",
    "ma20": "20日均价",
    "ma60": "60日均价",
    "return_1d": "1日涨跌幅",
    "return_5d": "5日涨跌幅",
    "ma5_gap": "偏离5日均线",
    "ma10_gap": "偏离10日均线",
    "ma20_gap": "偏离20日均线",
    "volatility_10d": "10日波动率",
    "volatility_20d": "20日波动率",
    "shfe_basis": "期现价差",
    "inventory_change_5d": "5日库存变化",
    "premium_discount": "升贴水",
    "import_profit": "进口盈亏",
}
HEADER_IMAGE = Path(__file__).resolve().parent / "assets" / "dashboard_header.png"
LANDING_HERO_IMAGE = Path(__file__).resolve().parent / "assets" / "metalpulse_landing_hero.png"
LANDING_DETAIL_IMAGE = Path(__file__).resolve().parent / "assets" / "metalpulse_landing_detail.png"
FACTOR_COEFFICIENTS = Path(__file__).resolve().parent / "domestic_material_factor_coefficients_v1.csv"
MONTHLY_RAW_DATA = Path(__file__).resolve().parent / "domestic_material_monthly_dataset_v1.csv"
NEWS_CACHE = Path(__file__).resolve().parent / "daily_news.json"
FACTOR_MATERIAL = {
    "copper_1": "1#铜",
    "aluminum_a00": "A00铝",
    "silver_1": "1#白银",
    "aluminum_adc12": "ADC12",
    "aluminum_zld104": "ZLD104",
    "lithium_carbonate": "碳酸锂",
}

FACTOR_SOURCE_LINKS = {
    "SHFE": ("上海期货交易所", "https://www.shfe.com.cn/"),
    "PPI": ("国家统计局数据查询", "https://data.stats.gov.cn/"),
    "PMI": ("国家统计局数据查询", "https://data.stats.gov.cn/"),
    "工业增加值": ("国家统计局数据查询", "https://data.stats.gov.cn/"),
    "房地产": ("国家统计局数据查询", "https://data.stats.gov.cn/"),
    "光缆": ("国家统计局数据查询", "https://data.stats.gov.cn/"),
    "电线电缆": ("国家统计局数据查询", "https://data.stats.gov.cn/"),
    "发电量": ("国家统计局数据查询", "https://data.stats.gov.cn/"),
    "当月进口额": (
        "海关总署统计月报",
        "http://www.customs.gov.cn/customs/302249/zfxxgk/2799825/302274/302277/index.html",
    ),
    "废铝进口量": (
        "海关总署统计月报",
        "http://www.customs.gov.cn/customs/302249/zfxxgk/2799825/302274/302277/index.html",
    ),
    "汽车销量Top50": ("盖世汽车销量排行榜", "https://auto.gasgoo.com/qcxl/"),
    "汽车": ("中国汽车工业协会", "http://www.caam.org.cn/"),
    "中国贸易政策不确定性": ("经济政策不确定性指数数据库", "https://www.policyuncertainty.com/china_monthly.html"),
    "中国经济政策不确定性": ("经济政策不确定性指数数据库", "https://www.policyuncertainty.com/china_monthly.html"),
    "A00铝": ("长江有色金属网", "https://www.ccmn.cn/"),
    "企业商品价格": ("国家统计局数据查询", "https://data.stats.gov.cn/"),
}
MONTHLY_EXPORT_MATERIALS = {
    "1#铜": "1#铜",
    "A00铝": "A00铝",
    "1#白银": "1#白银",
    "ADC12": "ADC12",
    "ZLD104": "ZLD104",
}
DISPLAY_EXCLUDED_FACTORS = {
    "SHFE铜主连收盘价_环比",
    "SHFE铝主连收盘价_环比",
    "A00铝_价格月环比",
    "ADC12_A00价差_滞后1期变化",
    "ZLD104_A00价差_滞后1期变化",
}

FACTOR_DISPLAY_NAMES = {
    "SHFE铜仓单库存_环比": "上海期货交易所（SHFE）铜仓单库存月环比",
    "SHFE铜主连成交量_环比": "上海期货交易所（SHFE）铜主力连续合约成交量月环比",
    "SHFE铝仓单库存_环比": "上海期货交易所（SHFE）铝仓单库存月环比",
    "电线电缆光缆及电工器材制造PPI_环比": "电线电缆、光缆及电工器材制造业工业生产者出厂价格指数（PPI）月环比",
    "光缆产量当期值_环比": "全国光缆产量月环比",
    "工业增加值同比增长": "全国规模以上工业增加值同比增速",
    "制造业PMI_变化": "制造业采购经理指数（PMI）月度变化",
    "当月进口额环比增长": "中国海关货物进口总额月度环比增速（美元计价）",
    "汽车产量当期值_环比": "全国汽车产量月环比",
    "新能源汽车产量当期值_环比": "全国新能源汽车产量月环比",
    "房间空气调节器产量当期值_环比": "全国房间空气调节器产量月环比",
    "家用电冰箱产量当期值_环比": "全国家用电冰箱产量月环比",
    "PPI当月同比增长": "全国工业生产者出厂价格指数（PPI）同比增速",
    "中国经济政策不确定性指数_变化": "中国经济政策不确定性指数月度变化",
    "中国贸易政策不确定性指数_变化": "中国贸易政策不确定性指数月度变化",
    "汽车销量Top50厂商合计_环比": "盖世汽车销量榜前50家厂商销量合计月环比",
    "废铝进口量_环比": "中国废铝进口量月环比",
    "发电量当期值_环比": "全国发电量月环比",
    "发电机组产量当期值_环比": "全国发电机组产量月环比",
    "企业商品价格矿产品环比增长": "矿产品企业商品价格指数月度环比增速",
    "企业商品价格煤油电环比增长": "煤油电企业商品价格指数月度环比增速",
}


def plain_factor_name(name: str) -> str:
    if name in FACTOR_DISPLAY_NAMES:
        return FACTOR_DISPLAY_NAMES[name]
    cleaned = str(name)
    for suffix in (
        "_滞后1期变化",
        "滞后1期变化",
        "_当月同比增长",
        "当月同比增长",
        "同比增长",
        "_环比增长",
        "环比增长",
        "价格月环比",
        "_环比",
        "环比",
        "_变化",
        "变化",
        "当期值",
        "当月",
    ):
        cleaned = cleaned.replace(suffix, "")
    cleaned = cleaned.replace("SHFE", "上海期货交易所（SHFE）")
    cleaned = cleaned.replace("主连", "主力连续合约")
    cleaned = cleaned.replace("PPI", "工业生产者出厂价格指数（PPI）")
    cleaned = cleaned.replace("制造业PMI", "制造业采购经理指数（PMI）")
    cleaned = cleaned.replace("非制造业PMI", "非制造业采购经理指数（PMI）")
    return cleaned.strip("_")


def factor_source(name: str) -> tuple[str, str]:
    for keyword, source in FACTOR_SOURCE_LINKS.items():
        if keyword in name:
            return source
    return "国家统计局数据查询", "https://data.stats.gov.cn/"


def excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, data in sheets.items():
            data.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes = "A2"
            for column_cells in worksheet.columns:
                width = min(max(len(str(cell.value or "")) for cell in column_cells) + 2, 36)
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
    return buffer.getvalue()


def build_raw_monthly_export() -> bytes | None:
    if not MONTHLY_RAW_DATA.exists():
        return None
    raw = pd.read_csv(MONTHLY_RAW_DATA, encoding="utf-8-sig")
    sheets: dict[str, pd.DataFrame] = {}
    excluded_terms = ("环比", "同比", "变化", "滞后")
    for material, prefix in MONTHLY_EXPORT_MATERIALS.items():
        columns = [
            column
            for column in raw.columns
            if column == "月份" or (column.startswith(prefix) and not any(term in column for term in excluded_terms))
        ]
        data = raw[columns].copy()
        data = data.rename(columns={"月份": "月份", f"{prefix}_月均价": f"{material}月均价"})
        data = data.rename(columns={column: plain_factor_name(column) for column in data.columns if column != "月份"})
        sheets[material] = data
    return excel_bytes(sheets)


def build_factor_export() -> bytes | None:
    coefficients = load_factor_coefficients()
    if coefficients.empty:
        return None
    sheets: dict[str, pd.DataFrame] = {}
    for material in MONTHLY_EXPORT_MATERIALS:
        data = coefficients[coefficients["品种"] == material].copy()
        if data.empty:
            continue
        data["影响因素"] = data["变量"].map(plain_factor_name)
        data["数据来源"] = data["变量"].map(lambda value: factor_source(str(value))[0])
        data["数据来源链接"] = data["变量"].map(lambda value: factor_source(str(value))[1])
        sheets[material] = data.rename(columns={"影响强度_绝对值": "影响强度", "回归方向": "关联方向"})[
            ["影响因素", "影响强度", "关联方向", "数据来源", "数据来源链接"]
        ].sort_values("影响强度", ascending=False)
    return excel_bytes(sheets) if sheets else None


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


@st.cache_data
def load_factor_coefficients() -> pd.DataFrame:
    if not FACTOR_COEFFICIENTS.exists():
        return pd.DataFrame()
    coefficients = pd.read_csv(FACTOR_COEFFICIENTS, encoding="utf-8-sig")
    lithium_screening = load_lithium_screening_coefficients()
    if not lithium_screening.empty:
        coefficients = pd.concat(
            [coefficients, lithium_screening],
            ignore_index=True,
        )
        return coefficients
    lithium_path = Path(__file__).resolve().parent / "lithium_carbonate_prediction_outputs" / "lithium_monthly_model_coefficients.csv"
    if lithium_path.exists():
        lithium = pd.read_csv(lithium_path, encoding="utf-8-sig")
        if {"变量", "系数"}.issubset(lithium.columns):
            # The lithium model has its own factor set.  Keep both the shared
            # macro/demand variables and the LC-prefixed futures-structure
            # variables in the influence-analysis table.
            lithium = lithium[lithium["变量"] != "截距"].copy()
            lithium["品种"] = "碳酸锂"
            lithium["模型版本"] = lithium.get("模型版本", "碳酸锂产业逻辑约束模型")
            lithium["目标变量"] = "碳酸锂价格月环比"
            lithium["标准化系数"] = lithium["系数"]
            lithium["影响强度_绝对值"] = lithium["系数"].abs()
            lithium["p值"] = np.nan
            lithium["显著性"] = "时间序列验证"
            lithium["回归方向"] = np.where(lithium["系数"] >= 0, "正向", "负向")
            lithium["强弱排名"] = (
                lithium["影响强度_绝对值"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
            columns = [
                "品种",
                "模型版本",
                "目标变量",
                "变量",
                "标准化系数",
                "影响强度_绝对值",
                "p值",
                "显著性",
                "回归方向",
                "强弱排名",
            ]
            coefficients = pd.concat([coefficients, lithium[columns]], ignore_index=True)
    return coefficients


def load_dashboard_analysis_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = load_monthly_dataset()
    catalog = load_curated_factor_catalog(monthly)
    events = load_verified_events()
    return monthly, catalog, events


def inject_style() -> None:
    hero_bg = image_data_uri(HEADER_IMAGE) if HEADER_IMAGE.exists() else ""
    st.markdown(
        f"""
        <style>
        :root {{
            --ink: #1f2937;
            --muted: #7b8798;
            --line: #e5eaf1;
            --surface: #ffffff;
            --canvas: #f6f8fc;
            --accent: #2e78f6;
            --accent-dark: #1c61d1;
            --navy: #ffffff;
        }}
        .stApp {{
            background:
                linear-gradient(180deg, #fbfcfe 0, var(--canvas) 180px, var(--canvas) 100%);
            color: var(--ink);
        }}
        header[data-testid="stHeader"] {{
            background: rgba(255, 255, 255, .94);
        }}
        .block-container {{
            padding-top: 1.75rem;
            padding-bottom: 3rem;
            max-width: 1480px;
        }}
        section[data-testid="stSidebar"] {{
            background: var(--navy);
            border-right: 1px solid var(--line);
        }}
        section[data-testid="stSidebar"] > div {{
            padding-top: 1.5rem;
        }}
        .brand {{
            color: #1f2937;
            font-size: 21px;
            font-weight: 800;
            letter-spacing: -.8px;
            margin: 4px 8px 32px;
        }}
        .brand span {{ color: #2e78f6; }}
        section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] span {{
            color: #64748b !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
            color: #7b8798;
            font-size: 12px;
            font-weight: 700;
        }}
        .sidebar-note {{
            background: #f4f8ff;
            border: 1px solid #e1ebfd;
            border-radius: 10px;
            color: #5f6f82;
            font-size: 12px;
            line-height: 1.7;
            margin-top: 24px;
            padding: 14px;
        }}
        .sidebar-note b {{ color: #334155; }}
        section[data-testid="stSidebar"] hr {{ border-color: var(--line); }}
        section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] {{
            align-items: center;
            border-radius: 10px;
            color: #64748b;
            font-weight: 650;
            margin: 3px 0;
            padding: 8px 10px;
            transition: background-color .15s ease, color .15s ease;
        }}
        section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:hover {{
            background: #f2f6ff;
            color: #2e78f6;
        }}
        section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {{
            background: #eaf2ff;
            color: #2e78f6;
            box-shadow: none;
        }}
        section[data-testid="stSidebar"] .stRadio label p {{ color: #64748b !important; }}
        section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) p {{ color: #2e78f6 !important; }}
        div[data-testid="stRadio"] label[data-baseweb="radio"] p {{ color: #64748b !important; }}
        div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {{ color: #2e78f6 !important; }}
        section[data-testid="stSidebar"] [data-baseweb="select"],
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: #ffffff;
            border-color: #dce5f0 !important;
            color: #334155;
        }}
        section[data-testid="stSidebar"] [data-baseweb="select"] input,
        section[data-testid="stSidebar"] [data-baseweb="select"] svg {{ color: #64748b !important; fill: #64748b !important; }}
        section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
            background: #edf4ff;
            border: 1px solid #dce9ff;
            border-radius: 9px;
            color: #2e78f6;
            font-weight: 700;
        }}
        .hero {{
            min-height: 82px;
            border: 0;
            border-radius: 0;
            padding: 8px 2px 18px;
            color: var(--ink);
            background: transparent;
            box-shadow: none;
            margin-bottom: 8px;
        }}
        .eyebrow {{
            color: #2e78f6;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1.1px;
            margin-bottom: 9px;
            text-transform: uppercase;
        }}
        .hero h1 {{
            font-size: 25px;
            line-height: 1.15;
            margin: 0 0 10px 0;
            font-weight: 760;
        }}
        .hero p {{
            margin: 0;
            max-width: 780px;
            color: var(--muted);
            font-size: 13px;
        }}
        .section-title {{
            margin: 28px 0 12px;
            font-size: 17px;
            font-weight: 720;
            color: var(--ink);
        }}
        div[data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 9px;
            box-shadow: 0 4px 12px rgba(34, 54, 86, .035);
            padding: 13px 15px;
            position: relative;
            overflow: hidden;
        }}
        div[data-testid="stMetric"]::before {{
            background: var(--accent);
            content: "";
            height: 3px;
            left: 0;
            position: absolute;
            top: 0;
            width: 30px;
        }}
        div[data-testid="stMetricLabel"] p {{
            color: #7b8798 !important;
            font-size: 13px;
            font-weight: 650;
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--ink);
            font-size: 1.55rem;
            white-space: normal;
        }}
        .comparison-note {{
            background: #f0f6ff;
            color: #3e5f8c;
            border: 1px solid #dce9ff;
            border-radius: 12px;
            padding: 10px 14px;
            margin: 8px 0 12px;
        }}
        .daily-news-item {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 10px;
            margin: 8px 0;
            padding: 11px 14px;
        }}
        .daily-news-title {{
            color: var(--ink);
            font-size: 14px;
            font-weight: 700;
            line-height: 1.45;
            text-decoration: none;
            display: block;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .daily-news-title:hover {{ color: var(--accent); }}
        .daily-news-meta {{ color: var(--muted); font-size: 11px; margin-top: 5px; }}
        .daily-news-summary {{
            color: #64748b;
            font-size: 12px;
            line-height: 1.55;
            margin-top: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        div[data-baseweb="tab-list"] {{
            gap: 5px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 5px;
        }}
        button[data-baseweb="tab"] {{
            border-radius: 10px;
            color: var(--muted);
            font-weight: 650;
            height: 38px;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background: #edf4ff;
            border-bottom: 2px solid var(--accent);
            color: var(--accent);
        }}
        .forecast-note {{
            background: #f3f7ff;
            border: 1px solid #deebff;
            border-left: 3px solid #2e78f6;
            border-radius: 10px;
            color: #426187;
            font-size: 13px;
            line-height: 1.65;
            margin: 10px 0 4px;
            padding: 10px 14px;
        }}
        div[data-testid="stAlert"] {{
            background: #fff8e7;
            border: 1px solid #f5dda2;
            border-radius: 10px;
            color: #85620d;
        }}
        div[data-testid="stAlert"] p {{ color: #85620d; }}
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
        }}
        .future-card {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 10px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            height: 190px;
            padding: 14px;
        }}
        .market-card {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 10px;
            box-sizing: border-box;
            min-height: 118px;
            min-width: 0;
            padding: 14px;
        }}
        .market-card-title {{
            color: #4f5f73;
            font-size: 13px;
            font-weight: 700;
            line-height: 1.35;
            min-height: 2.7em;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .market-card-value {{
            color: var(--ink);
            font-size: clamp(.95rem, 1.35vw, 1.45rem);
            font-weight: 760;
            letter-spacing: -.035em;
            margin-top: 8px;
            white-space: nowrap;
        }}
        .market-card-change {{
            display: inline-block;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            margin-top: 8px;
            padding: 3px 7px;
        }}
        .market-card-change.up {{ background: #fbe7e9; color: #b73745; }}
        .market-card-change.down {{ background: #e7f5ec; color: #258553; }}
        .market-card-change.steady {{ background: #eeeeef; color: #68686d; }}
        .future-card-title {{ color: #4f5f73; font-size: 13px; font-weight: 700; line-height: 1.35; min-height: 2.7em; overflow-wrap: anywhere; }}
        .future-card-value {{ color: var(--ink); font-size: 18px; font-weight: 760; letter-spacing: -.03em; margin-top: 12px; white-space: nowrap; }}
        .future-card-meta {{ color: var(--muted); font-size: 11px; line-height: 1.5; margin-top: 5px; min-height: 3em; }}
        .future-card-trend {{ font-size: 12px; font-weight: 750; margin-top: auto; padding-top: 8px; }}
        .future-card-trend.up {{ color: #c9372c; }}
        .future-card-trend.down {{ color: #258553; }}
        .future-card-trend.steady {{ color: #87859b; }}
        .st-key-overview_trend_window [data-testid="stWidgetLabel"] {{ display: none; }}
        .st-key-overview_trend_window [role="radiogroup"] {{ justify-content: flex-end; gap: 6px; }}
        .st-key-overview_trend_window label[data-baseweb="radio"] {{
            background: #ffffff;
            border: 1px solid #e1e8f1;
            border-radius: 7px;
            margin: 0;
            padding: 5px 9px;
        }}
        .st-key-overview_trend_window label[data-baseweb="radio"] > div:first-child {{ display: none; }}
        .st-key-overview_trend_window label[data-baseweb="radio"]:has(input:checked) {{
            background: #edf4ff;
            border-color: #cfe0ff;
        }}
        @media (max-width: 768px) {{
            .block-container {{ padding: 1rem 1rem 2.25rem; }}
            .hero {{ min-height: auto; padding: 26px 22px; }}
            .hero h1 {{ font-size: 28px; }}
        }}
        @media (min-width: 769px) and (max-width: 1240px) {{
            div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
                flex: 1 1 calc(50% - .5rem) !important;
                min-width: 0 !important;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # The dashboard keeps Streamlit's native interaction model, while this
    # final layer gives it a lively editorial identity without hiding data in
    # decorative glass effects or heavy gradients.
    st.markdown(
        """
        <style>
        :root {
            --pulse-ink: #172033;
            --pulse-muted: #667085;
            --pulse-line: #dce3ee;
            --pulse-canvas: #f3f6fb;
            --pulse-card: #ffffff;
            --pulse-blue: #246bfd;
            --pulse-cyan: #00a8c7;
            --pulse-coral: #f05a47;
            --pulse-yellow: #f3b63f;
        }

        .stApp {
            color: var(--pulse-ink);
            background:
                radial-gradient(circle at 94% 5%, rgba(36,107,253,.11), transparent 23rem),
                radial-gradient(circle at 4% 46%, rgba(0,168,199,.07), transparent 26rem),
                var(--pulse-canvas);
        }

        .block-container {
            max-width: 1480px;
            padding-top: 1.35rem;
            padding-bottom: 4rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #17243a 0%, #1c2c47 58%, #132035 100%);
            border-right: 0;
            box-shadow: 16px 0 42px rgba(24, 40, 72, .12);
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] small {
            color: #d9e4f4;
        }

        section[data-testid="stSidebar"] .brand {
            color: #ffffff;
            letter-spacing: -.02em;
        }

        section[data-testid="stSidebar"] .brand::after {
            content: "";
            display: inline-block;
            width: .55rem;
            height: .55rem;
            margin-left: .45rem;
            border-radius: 50%;
            background: #4fd1c5;
            box-shadow: 0 0 0 6px rgba(79,209,197,.12);
            animation: pulseDot 2.8s ease-in-out infinite;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 10px;
            padding: .6rem .7rem;
            transition: transform .18s ease, background-color .18s ease;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            transform: translateX(3px);
            background: rgba(255,255,255,.07);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(69,126,255,.2);
            box-shadow: inset 3px 0 0 #65d6e7;
        }

        .hero {
            position: relative;
            overflow: hidden;
            display: grid;
            grid-template-columns: minmax(0, 1.55fr) minmax(250px, .65fr);
            align-items: end;
            min-height: 220px;
            padding: 2.4rem 2.6rem;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(36,107,253,.13);
            border-radius: 24px;
            background:
                linear-gradient(115deg, rgba(255,255,255,.98) 0 54%, rgba(243,248,255,.92) 54% 100%);
            box-shadow: 0 20px 60px rgba(39, 69, 123, .10);
            animation: riseIn .52s cubic-bezier(.2,.8,.2,1) both;
        }

        .hero::before,
        .hero::after {
            content: "";
            position: absolute;
            pointer-events: none;
        }

        .hero::before {
            width: 310px;
            height: 310px;
            right: -90px;
            top: -155px;
            border: 34px solid rgba(36,107,253,.10);
            border-radius: 50%;
        }

        .hero::after {
            width: 210px;
            height: 90px;
            right: 8%;
            bottom: -42px;
            border-radius: 50%;
            background: rgba(0,168,199,.10);
            transform: rotate(-12deg);
            filter: blur(2px);
        }

        .hero h1 {
            max-width: 760px;
            margin: .35rem 0 .65rem;
            color: #172033;
            font-size: clamp(2.15rem, 4vw, 3.9rem);
            line-height: 1.04;
            letter-spacing: -.055em;
        }

        .hero p {
            max-width: 720px;
            margin: 0;
            color: #5f6f86;
            font-size: 1rem;
            line-height: 1.75;
        }

        .hero .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            padding: .38rem .72rem;
            border-radius: 999px;
            color: #1752c3;
            background: #eaf1ff;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .09em;
            text-transform: uppercase;
        }

        .hero .eyebrow::before {
            content: "";
            width: .45rem;
            height: .45rem;
            border-radius: 50%;
            background: var(--pulse-coral);
        }

        .hero-meta {
            position: relative;
            z-index: 1;
            align-self: center;
            padding: 1rem 1.1rem;
            border-left: 3px solid var(--pulse-cyan);
            color: #34445e;
            background: rgba(255,255,255,.7);
            font-size: .88rem;
            line-height: 1.8;
        }

        .section-title {
            margin-top: 2rem;
            padding: 0 0 .8rem;
            border-bottom: 1px solid var(--pulse-line);
            color: #1d2b43;
            font-size: 1.22rem;
            letter-spacing: -.02em;
        }

        .section-title::before {
            content: "";
            display: inline-block;
            width: .55rem;
            height: .55rem;
            margin-right: .6rem;
            border-radius: 3px;
            background: linear-gradient(135deg, var(--pulse-blue), var(--pulse-cyan));
            transform: rotate(8deg);
        }

        div[data-testid="stMetric"] {
            position: relative;
            overflow: hidden;
            min-height: 112px;
            padding: 1rem 1.05rem;
            border: 1px solid #e1e7f0;
            border-radius: 15px;
            background: rgba(255,255,255,.94);
            box-shadow: 0 10px 28px rgba(49, 72, 112, .07);
            transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
            animation: riseIn .45s cubic-bezier(.2,.8,.2,1) both;
        }

        div[data-testid="stMetric"]::before {
            width: 4px;
            height: 42%;
            left: 0;
            top: 29%;
            border-radius: 0 5px 5px 0;
            background: linear-gradient(180deg, var(--pulse-blue), var(--pulse-cyan));
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: #cbd9f2;
            box-shadow: 0 16px 34px rgba(49,72,112,.12);
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #172033;
            font-size: clamp(1.18rem, 1.75vw, 1.65rem);
            font-weight: 760;
            letter-spacing: -.035em;
            line-height: 1.18;
            white-space: normal !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] > div,
        div[data-testid="stMetric"] [data-testid="stMetricValue"] p {
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: normal !important;
            word-break: keep-all;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stAlert"] {
            overflow: hidden;
            border: 1px solid #e1e7f0;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 12px 34px rgba(49,72,112,.065);
        }

        div[data-testid="stPlotlyChart"] {
            overflow: visible;
            border: 1px solid #e1e7f0;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 12px 34px rgba(49,72,112,.065);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-testid="stDateInput"] > div > div,
        div[role="radiogroup"] {
            border-radius: 11px !important;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: .35rem;
            padding: .3rem;
            border: 1px solid #e0e7f1;
            border-radius: 13px;
            background: rgba(255,255,255,.82);
        }

        div[data-testid="stTabs"] button[role="tab"] {
            border-radius: 9px;
            transition: background-color .18s ease, color .18s ease, transform .18s ease;
        }

        div[data-testid="stTabs"] button[role="tab"]:hover {
            transform: translateY(-1px);
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #174daf;
            background: #eaf1ff;
        }

        .stButton > button,
        .stDownloadButton > button,
        a[data-testid="stLinkButton"] {
            min-height: 2.65rem;
            border-radius: 11px;
            font-weight: 700;
            transition: transform .18s ease, box-shadow .18s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        a[data-testid="stLinkButton"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 9px 22px rgba(36,107,253,.14);
        }

        .future-card {
            border: 1px solid #e1e7f0;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 10px 28px rgba(49,72,112,.06);
            transition: transform .2s ease, border-color .2s ease;
        }

        .future-card:hover {
            transform: translateY(-3px);
            border-color: #bfd1f3;
        }

        @keyframes riseIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulseDot {
            0%, 100% { transform: scale(1); box-shadow: 0 0 0 5px rgba(79,209,197,.10); }
            50% { transform: scale(1.12); box-shadow: 0 0 0 9px rgba(79,209,197,.04); }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: .01ms !important;
            }
        }

        @media (max-width: 760px) {
            .block-container { padding: .9rem .8rem 3rem; }
            .hero {
                grid-template-columns: 1fr;
                min-height: auto;
                padding: 1.55rem 1.3rem;
                border-radius: 18px;
            }
            .hero h1 { font-size: 2.15rem; }
            .hero-meta { margin-top: 1.15rem; }
            div[data-testid="stMetric"] { min-height: 96px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        :root {
            --pulse-ink: #202124;
            --pulse-muted: #6f7278;
            --pulse-line: #dedfe2;
            --pulse-canvas: #f5f5f4;
            --pulse-card: #ffffff;
            --pulse-blue: #8b1e2d;
            --pulse-cyan: #c96f7c;
            --pulse-coral: #8b1e2d;
        }
        .stApp {
            background:
                linear-gradient(180deg, rgba(247,229,232,.72) 0, rgba(245,245,244,0) 190px),
                var(--pulse-canvas);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #242426 0%, #303033 100%);
            box-shadow: 12px 0 30px rgba(32,32,35,.10);
        }
        section[data-testid="stSidebar"] .brand span { color: #e4a5ae; }
        section[data-testid="stSidebar"] .brand::after { display: none; }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: #f7edef;
            box-shadow: inset 3px 0 0 #a83345;
        }
        section[data-testid="stSidebar"] .stRadio label p {
            color: #b8bac0 !important;
        }
        section[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
            color: #8b1e2d !important;
        }
        div[data-testid="stCheckbox"] label[data-baseweb="checkbox"] p {
            color: #2a2a2d !important;
            font-weight: 600;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            background: #39393d;
            border-color: #55555b;
            color: #f4f4f2;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            background: #48484d;
            border-color: #77777d;
            color: #ffffff;
        }
        .dashboard-header {
            align-items: center;
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin: .15rem 0 .9rem;
            padding: .25rem 0 .8rem;
            border-bottom: 1px solid var(--pulse-line);
        }
        .dashboard-header h1 {
            color: #202124;
            font-size: clamp(1.55rem, 2.5vw, 2.15rem);
            letter-spacing: -.035em;
            line-height: 1.15;
            margin: 0;
        }
        .dashboard-status {
            color: #66686d;
            font-size: .78rem;
            line-height: 1.55;
            text-align: right;
        }
        .page-description {
            color: #252528;
            font-size: .94rem;
            line-height: 1.75;
            margin: 0 0 1rem;
            max-width: 72rem;
        }
        .section-title {
            border-bottom-color: var(--pulse-line);
            color: #252528;
        }
        .section-title::before {
            background: #8b1e2d;
            border-radius: 1px;
            height: .72rem;
            transform: none;
            width: .22rem;
        }
        div[data-testid="stMetric"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stDataFrame"],
        div[data-testid="stAlert"],
        .market-card,
        .future-card {
            border-color: #dedfe2;
            border-radius: 12px;
            box-shadow: 0 8px 22px rgba(48,43,45,.055);
        }
        div[data-testid="stMetric"]::before {
            background: #8b1e2d;
        }
        .stApp div[data-testid="stMetric"] [data-testid="stMetricLabel"]
        div[data-testid="stMarkdownContainer"] p {
            color: #68686d !important;
            opacity: 1 !important;
        }
        div[data-testid="stMetric"]:hover,
        .market-card:hover,
        .future-card:hover {
            border-color: #d1a7ad;
            box-shadow: 0 12px 26px rgba(86,44,51,.09);
        }
        div[role="radiogroup"] label:has(input:checked) p { color: #8b1e2d !important; }
        .st-key-overview_trend_window label[data-baseweb="radio"]:has(input:checked) {
            background: #f7e5e8;
            border-color: #e4bcc2;
        }
        span[data-baseweb="tag"] {
            background: #2f2f32 !important;
            color: #f7f7f5 !important;
        }
        span[data-baseweb="tag"] svg { fill: #e6b7be !important; }
        .analysis-note {
            color: #74767b;
            font-size: .78rem;
            margin: -.25rem 0 .75rem;
        }
        .balance-empty {
            background: #faf4f5;
            border: 1px solid #ead2d6;
            border-radius: 10px;
            color: #775a5f;
            padding: .8rem .9rem;
        }
        @media (max-width: 760px) {
            .dashboard-header { align-items: flex-start; flex-direction: column; }
            .dashboard-status { text-align: left; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_landing_page() -> None:
    hero_image = image_data_uri(LANDING_HERO_IMAGE) if LANDING_HERO_IMAGE.exists() else ""
    detail_image = image_data_uri(LANDING_DETAIL_IMAGE) if LANDING_DETAIL_IMAGE.exists() else ""
    st.markdown(
        f"""
        <style>
        header[data-testid="stHeader"] {{ background: rgba(255, 255, 255, .92); }}
        .stApp {{ background: #f8fafc; color: #172033; }}
        .block-container {{ max-width: 1280px; padding: 0 2rem 4rem; }}
        section[data-testid="stSidebar"] {{ display: none; }}
        .landing-nav {{ align-items: center; border-bottom: 1px solid #e6ebf2; display: flex; height: 76px; justify-content: space-between; }}
        .landing-logo {{ color: #15213a; font-size: 21px; font-weight: 800; letter-spacing: -.8px; }}
        .landing-logo span {{ color: #2367e8; }}
        .landing-nav-copy {{ color: #64748b; font-size: 14px; }}
        .landing-hero {{ align-items: stretch; display: grid; gap: 3.5rem; grid-template-columns: 1.02fr .98fr; min-height: 570px; padding: 86px 0 70px; }}
        .landing-copy {{ align-self: center; }}
        .landing-kicker {{ color: #2367e8; font-size: 12px; font-weight: 800; letter-spacing: .12em; margin-bottom: 18px; text-transform: uppercase; }}
        .landing-hero h1 {{ color: #172033; font-size: clamp(42px, 5vw, 70px); font-weight: 780; letter-spacing: -.065em; line-height: 1.02; margin: 0; max-width: 680px; }}
        .landing-hero p {{ color: #5f6f82; font-size: 18px; line-height: 1.7; margin: 24px 0 30px; max-width: 530px; }}
        .landing-cta {{ background: #2367e8; border-radius: 10px; color: #fff !important; display: inline-flex; font-size: 15px; font-weight: 750; padding: 14px 20px; text-decoration: none; transition: background .16s ease, transform .16s ease; }}
        .landing-cta:hover {{ background: #1756cd; transform: translateY(-1px); }}
        .landing-visual {{ align-self: center; border-radius: 18px; box-shadow: 0 28px 60px rgba(47, 64, 88, .16); min-height: 420px; object-fit: cover; width: 100%; }}
        .landing-proof {{ border-bottom: 1px solid #e6ebf2; border-top: 1px solid #e6ebf2; display: grid; grid-template-columns: 1.2fr 1fr 1fr; padding: 24px 0; }}
        .proof-item {{ border-left: 1px solid #e6ebf2; padding: 0 28px; }}
        .proof-item:first-child {{ border-left: 0; padding-left: 0; }}
        .proof-value {{ color: #172033; font-size: 22px; font-weight: 760; letter-spacing: -.04em; }}
        .proof-label {{ color: #6b7a8e; font-size: 13px; margin-top: 5px; }}
        .landing-section {{ padding: 112px 0; }}
        .landing-section h2 {{ color: #172033; font-size: clamp(30px, 3.3vw, 48px); letter-spacing: -.05em; line-height: 1.08; margin: 0; max-width: 670px; }}
        .landing-section > p {{ color: #64748b; font-size: 17px; line-height: 1.7; margin: 19px 0 0; max-width: 590px; }}
        .landing-detail {{ align-items: center; display: grid; gap: 6rem; grid-template-columns: .82fr 1.18fr; }}
        .landing-detail img {{ border-radius: 16px; display: block; max-height: 500px; object-fit: cover; width: 100%; }}
        .detail-points {{ display: grid; gap: 0; margin-top: 30px; }}
        .detail-point {{ border-top: 1px solid #e1e7ef; padding: 18px 0; }}
        .detail-point strong {{ color: #172033; display: block; font-size: 16px; }}
        .detail-point span {{ color: #6b7a8e; display: block; font-size: 14px; line-height: 1.6; margin-top: 5px; }}
        .landing-capabilities {{ display: grid; gap: 18px; grid-template-columns: 1.15fr .85fr; margin-top: 42px; }}
        .capability-main {{ background: #eaf1ff; border-radius: 16px; min-height: 280px; padding: 34px; }}
        .capability-main h3, .capability-small h3 {{ color: #172033; font-size: 23px; letter-spacing: -.04em; margin: 0; }}
        .capability-main p, .capability-small p {{ color: #5e6e82; font-size: 15px; line-height: 1.65; margin: 12px 0 0; max-width: 420px; }}
        .capability-stack {{ display: grid; gap: 18px; }}
        .capability-small {{ background: #fff; border: 1px solid #e1e7ef; border-radius: 16px; padding: 26px; }}
        .landing-closing {{ align-items: end; border-top: 1px solid #e6ebf2; display: flex; justify-content: space-between; padding: 80px 0 34px; }}
        .landing-closing h2 {{ color: #172033; font-size: clamp(28px, 3vw, 42px); letter-spacing: -.055em; line-height: 1.08; margin: 0; max-width: 570px; }}
        .landing-closing p {{ color: #64748b; font-size: 15px; line-height: 1.6; margin: 0; max-width: 270px; }}
        .landing-footer {{ color: #7a8797; font-size: 13px; padding-top: 28px; }}
        @media (max-width: 768px) {{
            .block-container {{ padding: 0 1.25rem 3rem; }}
            .landing-nav-copy {{ display: none; }}
            .landing-hero, .landing-detail, .landing-capabilities {{ grid-template-columns: 1fr; }}
            .landing-hero {{ gap: 34px; min-height: auto; padding: 58px 0; }}
            .landing-visual {{ min-height: 300px; }}
            .landing-proof {{ grid-template-columns: 1fr; gap: 20px; }}
            .proof-item, .proof-item:first-child {{ border-left: 0; padding: 0; }}
            .landing-section {{ padding: 76px 0; }}
            .landing-detail {{ gap: 36px; }}
            .landing-closing {{ align-items: flex-start; flex-direction: column; gap: 26px; padding-top: 58px; }}
        }}
        </style>
        <nav class="landing-nav">
            <div class="landing-logo">Metal<span>Pulse</span></div>
            <div class="landing-nav-copy">原材料采购价格智能平台</div>
        </nav>
        <section class="landing-hero">
            <div class="landing-copy">
                <div class="landing-kicker">Material price intelligence</div>
                <h1>让采购判断，建立在清晰的价格信号上。</h1>
                <p>以统一的不含税口径，连接现货价格、预测区间与模型评估，为原材料采购提供可追溯的日度判断依据。</p>
                <a class="landing-cta" href="?view=dashboard">进入预测平台</a>
            </div>
            <img class="landing-visual" src="{hero_image}" alt="铜材与铝锭的原材料采购场景">
        </section>
        <section class="landing-proof">
            <div class="proof-item"><div class="proof-value">统一口径</div><div class="proof-label">现货、预测与月均价均按不含税价格呈现</div></div>
            <div class="proof-item"><div class="proof-value">日度更新</div><div class="proof-label">长江有色数据更新后自动重训</div></div>
            <div class="proof-item"><div class="proof-value">区间判断</div><div class="proof-label">同时呈现趋势、预测与不确定性范围</div></div>
        </section>
        <section class="landing-section landing-detail">
            <img src="{detail_image}" alt="铝材与采购分析资料">
            <div>
                <h2>不只看到价格，更理解价格所处的位置。</h2>
                <p>平台将近期现货走势、未来 30 天预测和模型回测放到同一套视图中，降低跨来源比对的成本。</p>
                <div class="detail-points">
                    <div class="detail-point"><strong>多品种同步跟踪</strong><span>覆盖 1#铜、A00铝、1#白银、铝 ADC12、ZLD104 与碳酸锂。</span></div>
                    <div class="detail-point"><strong>预测区间可见</strong><span>使用 P10 至 P90 区间表达价格判断的边界，而非单一结论。</span></div>
                    <div class="detail-point"><strong>历史表现可复核</strong><span>保留模型评估，以相同口径检查模型的实际表现。</span></div>
                </div>
            </div>
        </section>
        <section class="landing-section">
            <h2>为采购节奏设计的价格工作台。</h2>
            <p>从晨间判断到月度计划，关键数据保持在一个清晰、稳定的分析入口。</p>
            <div class="landing-capabilities">
                <div class="capability-main"><h3>价格预测总览</h3><p>将已公布现货均价、未来预测曲线和预测区间组合呈现，快速识别趋势变化。</p></div>
                <div class="capability-stack">
                    <div class="capability-small"><h3>模型评估</h3><p>用真实价格检查预测表现，让模型结果可被持续验证。</p></div>
                    <div class="capability-small"><h3>模型说明</h3><p>用通俗语言说明数据来源、价格范围和预测逻辑，方便内部协作与复盘。</p></div>
                </div>
            </div>
        </section>
        <section class="landing-closing">
            <h2>把市场波动，转化为更从容的采购决策。</h2>
            <p>统一口径、持续更新、可复核的价格判断，已在预测平台中准备就绪。</p>
        </section>
        <footer class="landing-footer">MetalPulse · 国内原材料采购价格预测</footer>
        """,
        unsafe_allow_html=True,
    )


def _wrap_annotation(value: object, width: int = 15, max_lines: int = 3) -> str:
    text = str(value).strip()
    lines = [text[index : index + width] for index in range(0, len(text), width)]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"
    return "<br>".join(html.escape(line) for line in lines)


def add_market_event_annotations(
    fig: go.Figure,
    prices: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    if prices.empty or events.empty:
        return
    marker_rows: list[dict[str, object]] = []
    date_min = prices["trade_date"].min()
    date_span = max(prices["trade_date"].max() - date_min, pd.Timedelta(days=1))
    vertical_offsets = [-58, 76, 148, 230, -128]
    horizontal_offsets = [105, 185, 90, 165, 120]
    metal_key = str(prices["metal"].iloc[0]) if "metal" in prices.columns and not prices.empty else ""
    is_copper = metal_key == "copper_1"
    if is_copper:
        value_min = float(pd.to_numeric(prices["price_cny_per_tonne"], errors="coerce").min())
        value_max = float(pd.to_numeric(prices["price_cny_per_tonne"], errors="coerce").max())
        value_span = max(value_max - value_min, 1.0)
        copper_lane_ratios = [0.88, 0.66, 0.44, 0.22, 0.10]
    for number, event in enumerate(events.sort_values("event_date").head(5).itertuples(index=False), start=1):
        nearest_index = (prices["trade_date"] - event.event_date).abs().idxmin()
        point = prices.loc[nearest_index]
        marker_rows.append(
            {
                "date": point["trade_date"],
                "price": point["price_cny_per_tonne"],
                "title": event.title,
                "summary": event.summary,
                "source": f"{event.source_name} {event.source_reference}",
            }
        )
        relative_position = (point["trade_date"] - date_min) / date_span
        horizontal_offset = horizontal_offsets[number - 1]
        if relative_position >= 0.68:
            ax = -horizontal_offset
        elif relative_position <= 0.32:
            ax = horizontal_offset
        else:
            ax = -horizontal_offset if number % 2 else horizontal_offset
        ay = vertical_offsets[number - 1]
        annotation_position = {"ax": ax, "ay": ay}
        if is_copper:
            lane_index = min(number - 1, len(copper_lane_ratios) - 1)
            annotation_position = {
                "ax": point["trade_date"] - pd.Timedelta(days=45 + 15 * lane_index),
                "ay": value_min + value_span * copper_lane_ratios[lane_index],
                "axref": "x",
                "ayref": "y",
            }
        fig.add_annotation(
            x=point["trade_date"],
            y=point["price_cny_per_tonne"],
            text=(
                f"<b>{number}. {_wrap_annotation(event.title, 12, 2)}</b>"
                f"<br>{event.event_date:%Y-%m-%d}"
                f"<br>{_wrap_annotation(event.summary, 15, 2)}"
            ),
            showarrow=True,
            arrowhead=0,
            arrowwidth=1.2,
            arrowcolor="#5f5f63",
            **annotation_position,
            align="left",
            bordercolor="#5f5f63",
            borderwidth=1,
            borderpad=6,
            bgcolor="rgba(255,255,255,.96)",
            font={"size": 10, "color": "#29292c"},
        )
    markers = pd.DataFrame(marker_rows)
    fig.add_trace(
        go.Scatter(
            x=markers["date"],
            y=markers["price"],
            mode="markers",
            marker={"size": 8, "color": "#8B1E2D", "line": {"color": "white", "width": 1.5}},
            showlegend=False,
            customdata=np.column_stack(
                [markers["title"], markers["summary"], markers["source"]]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]}"
                "<br>来源：%{customdata[2]}<extra></extra>"
            ),
        )
    )


def apply_compact_price_ranges(
    fig: go.Figure,
    dates: pd.Series,
    values: pd.Series,
) -> None:
    clean_dates = pd.to_datetime(dates, errors="coerce").dropna()
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    if clean_dates.empty or clean_values.empty:
        return
    date_span_days = max((clean_dates.max() - clean_dates.min()).days, 1)
    date_padding = pd.Timedelta(days=max(1, min(21, round(date_span_days * 0.015))))
    value_min = float(clean_values.min())
    value_max = float(clean_values.max())
    value_span = max(value_max - value_min, abs(value_max) * 0.01, 1.0)
    value_padding = value_span * 0.08
    fig.update_xaxes(
        range=[clean_dates.min() - date_padding, clean_dates.max() + date_padding],
        autorange=False,
    )
    fig.update_yaxes(
        range=[max(0.0, value_min - value_padding), value_max + value_padding],
        autorange=False,
    )


def build_trend_figure(
    metal_spot: pd.DataFrame,
    metal_forecast: pd.DataFrame,
    color: str,
    events: pd.DataFrame | None = None,
    unit: str = "元/吨",
) -> go.Figure:
    fig = go.Figure()
    metal_spot = metal_spot.sort_values("trade_date").copy()
    if not metal_spot.empty:
        cutoff = metal_spot["trade_date"].max() - pd.Timedelta(days=31)
        metal_spot = metal_spot[metal_spot["trade_date"] >= cutoff]
    fig.add_trace(
        go.Scatter(
            x=metal_spot["trade_date"],
            y=metal_spot["price_cny_per_tonne"],
            mode="lines",
            name="最近一个月现货均价",
            line={"color": color, "width": 2.4},
            hovertemplate=(
                f"实际不含税均价<br>%{{x|%Y-%m-%d}}<br><b>%{{y:,.0f}} {unit}</b>"
                "<extra></extra>"
            ),
        )
    )
    if events is not None:
        add_market_event_annotations(fig, metal_spot, events)
    if not metal_forecast.empty:
        metal_forecast = metal_forecast.sort_values("forecast_date").copy()
        band = widened_forecast_band(metal_forecast)
        fig.add_trace(
            go.Scatter(
                x=band["forecast_date"],
                y=band["display_upper_bound"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
        go.Scatter(
            x=band["forecast_date"],
            y=band["display_lower_bound"],
            mode="lines",
            fill="tonexty",
            line={"width": 0},
            name="预测下限-上限",
            fillcolor="rgba(46, 120, 246, 0.13)",
            hoverinfo="skip",
        )
        )
        if not metal_spot.empty:
            fig.add_trace(
                go.Scatter(
                    x=[metal_spot["trade_date"].iloc[-1], metal_forecast["forecast_date"].iloc[0]],
                    y=[metal_spot["price_cny_per_tonne"].iloc[-1], metal_forecast["predicted_price_cny_per_tonne"].iloc[0]],
                    mode="lines",
                    line={"color": "#2E78F6", "width": 3.5},
                    name="实际至预测衔接",
                    showlegend=False,
                    hovertemplate="实际末值至首个预测值衔接<extra></extra>",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=metal_forecast["forecast_date"],
                y=metal_forecast["predicted_price_cny_per_tonne"],
                mode="lines+markers",
            name="未来30天预测",
                line={"color": "#2E78F6", "width": 3.5},
            marker={"color": "#2E78F6", "size": 5, "symbol": "circle"},
            customdata=np.column_stack(
                [
                    metal_forecast["lower_bound"].astype(float),
                    metal_forecast["upper_bound"].astype(float),
                ]
            ),
            hovertemplate=(
                f"预测不含税均价<br>%{{x|%Y-%m-%d}}<br><b>%{{y:,.0f}} {unit}</b>"
                f"<br>预测下限-上限：%{{customdata[0]:,.0f}}–%{{customdata[1]:,.0f}} {unit}"
                "<extra></extra>"
            ),
            )
        )
        if not metal_spot.empty:
            split_date = metal_spot["trade_date"].max()
            fig.add_vline(
                x=split_date,
                line_width=1,
                line_dash="dash",
                line_color="rgba(80, 90, 110, 0.55)",
            )
            fig.add_annotation(
                x=split_date,
                y=1,
                yref="paper",
                text="预测起点",
                showarrow=False,
                xanchor="left",
                yanchor="top",
                font={"size": 12, "color": "#36527a"},
                bgcolor="rgba(255,255,255,0.9)",
            )
    fig.update_layout(
        height=560,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b"},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title=f"{unit}（不含税）",
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#CBD5E1",
            "align": "left",
            "font": {
                "color": "#2A2A2D",
                "family": "Arial, sans-serif",
                "size": 12,
            },
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": "#4B4B50"},
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(76, 106, 146, 0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(76, 106, 146, 0.10)")
    axis_dates = [metal_spot["trade_date"]]
    axis_values = [metal_spot["price_cny_per_tonne"]]
    if not metal_forecast.empty:
        axis_dates.append(metal_forecast["forecast_date"])
        axis_values.extend(
            [
                metal_forecast["lower_bound"],
                metal_forecast["upper_bound"],
                metal_forecast["predicted_price_cny_per_tonne"],
            ]
        )
    apply_compact_price_ranges(
        fig,
        pd.concat(axis_dates, ignore_index=True),
        pd.concat(axis_values, ignore_index=True),
    )
    return fig


def widened_forecast_band(forecast: pd.DataFrame) -> pd.DataFrame:
    band = forecast.copy()
    band["display_upper_bound"] = band["upper_bound"].astype(float)
    band["display_lower_bound"] = band["lower_bound"].astype(float)
    return band


def render_monthly_forecast(
    monthly_forecast: pd.DataFrame,
    metal_spot: pd.DataFrame,
    color: str,
    unit: str = "元/吨",
) -> None:
    st.markdown('<div class="section-title">未来12个月均价预测</div>', unsafe_allow_html=True)
    if monthly_forecast.empty:
        st.info("暂无月度均价预测。请先运行数据更新任务。")
        return
    data = ensure_monthly_horizon(monthly_forecast, metal_spot, periods=12)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=data["forecast_month"],
            y=data["upper_bound"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["forecast_month"],
            y=data["lower_bound"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(158, 174, 183, 0.22)",
            name="预测下限-上限",
            hoverinfo="skip",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=data["forecast_month"],
            y=data["predicted_price_cny_per_tonne"],
            marker_color="#7F9FBE",
            name="预测月均价",
            customdata=np.column_stack(
                [
                    data["predicted_change_pct"].map(lambda value: f"{value:+.2%}"),
                    data["lower_bound"].map(lambda value: f"{value:,.0f}"),
                    data["upper_bound"].map(lambda value: f"{value:,.0f}"),
                ]
            ),
            hovertemplate=(
                f"<b>%{{x|%Y-%m}}</b><br>预测月均价：%{{y:,.0f}} {unit}"
                "<br>预测月环比：%{customdata[0]}"
                f"<br>预测下限-上限：%{{customdata[1]}} - %{{customdata[2]}} {unit}"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["forecast_month"],
            y=data["predicted_change_pct"] * 100,
            mode="lines+markers",
            marker_color=color,
            line={"color": color, "width": 2.6},
            marker={"size": 6},
            name="预测月环比",
            hoverinfo="skip",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        height=360,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b"},
        margin={"l": 20, "r": 20, "t": 58, "b": 20},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "x": 0,
            "font": {"color": "#4B4B50"},
        },
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#CBD5E1",
            "align": "left",
            "font": {
                "color": "#2A2A2D",
                "family": "Arial, sans-serif",
                "size": 12,
            },
        },
    )
    fig.update_yaxes(
        title_text=f"{unit}（不含税）",
        showgrid=True,
        gridcolor="rgba(76, 106, 146, 0.10)",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="预测月环比（%）",
        showgrid=False,
        secondary_y=True,
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        data.assign(
            预测月份=data["forecast_month"].dt.strftime("%Y-%m"),
            预测月均价=data["predicted_price_cny_per_tonne"].map(
                lambda value: f"{value:,.0f} {unit}"
            ),
            预测下限=data["lower_bound"].map(lambda value: f"{value:,.0f}"),
            预测上限=data["upper_bound"].map(lambda value: f"{value:,.0f}"),
            预测月环比=data["predicted_change_pct"].map(lambda value: f"{value:+.2%}"),
        )[["预测月份", "预测月均价", "预测下限", "预测上限", "预测月环比", "direction"]].rename(
            columns={"direction": "方向"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption("柱形表示预测月均价，折线表示预测月环比；阴影表示预测下限-上限。")


def build_price_history_figure(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    color: str,
    unit: str = "元/吨",
) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=prices["trade_date"],
            y=prices["price_cny_per_tonne"],
            mode="lines",
            line={"color": color, "width": 2.7},
            name="不含税现货均价",
            hovertemplate=(
                f"%{{x|%Y-%m-%d}}<br><b>%{{y:,.0f}} {unit}</b><extra></extra>"
            ),
        )
    )
    add_market_event_annotations(fig, prices, events)
    fig.update_layout(
        height=510,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b"},
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        yaxis_title=f"{unit}（不含税）",
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#CBD5E1",
            "align": "left",
            "font": {
                "color": "#2A2A2D",
                "family": "Arial, sans-serif",
                "size": 12,
            },
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"color": "#4B4B50"},
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(76, 106, 146, 0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(76, 106, 146, 0.10)")
    apply_compact_price_ranges(
        fig,
        prices["trade_date"],
        prices["price_cny_per_tonne"],
    )
    return fig


def render_verified_event_details(events: pd.DataFrame) -> None:
    if events.empty:
        return
    st.markdown('<div class="section-title">变动分析</div>', unsafe_allow_html=True)
    for number, event in enumerate(events.itertuples(index=False), start=1):
        source_url = str(event.source_url).strip() if pd.notna(event.source_url) else ""
        source_label = f"{event.source_name}（{event.source_date}，{event.source_reference}）"
        source_text = (
            f'<a href="{html.escape(source_url, quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">打开来源：{html.escape(source_label)}</a>'
            if source_url.startswith(("https://", "http://"))
            else html.escape(source_label)
        )
        st.markdown(
            f"<p><strong>{number}. {html.escape(str(event.title))}</strong>　"
            f"{event.event_date:%Y-%m-%d}<br>"
            f"{html.escape(str(event.summary))}<br>{source_text}</p>",
            unsafe_allow_html=True,
        )


def render_daily_news(news_payload: dict[str, object]) -> None:
    st.markdown('<div class="section-title">每日资讯</div>', unsafe_allow_html=True)
    items = news_payload.get("items", []) if isinstance(news_payload, dict) else []
    if not isinstance(items, list) or not items:
        st.info("暂无可展示的原材料相关资讯，等待定时更新。")
        return
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        title = html.escape(str(item.get("title", "")))
        url = str(item.get("url", "")).strip()
        if not title or not url.startswith(("https://", "http://")):
            continue
        source = html.escape(str(item.get("source", "")))
        published = html.escape(str(item.get("published", "")).strip())
        updated_at = html.escape(str(news_payload.get("updated_at", "")).replace("T", " ")[:16])
        meta = " · ".join(value for value in [source, published or updated_at] if value)
        # 旧缓存中可能保留了网页片段，显示前再次清理，避免将 HTML 标签当作摘要文本。
        raw_summary = re.sub(r"<[^>]*>", " ", str(item.get("summary", "")))
        summary = html.escape(" ".join(raw_summary.split()).strip())
        summary_html = f'<div class="daily-news-summary">{summary}</div>' if summary else ""
        card_html = (
            '<div class="daily-news-item">'
            f'<a class="daily-news-title" href="{html.escape(url, quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">{title}</a>'
            f'{summary_html}<div class="daily-news-meta">{meta}</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)
    st.caption("资讯来自长江有色和 SMM，展示标题及摘要；点击标题查看原文。")


def render_impact_analysis(
    metal: str,
    monthly_data: pd.DataFrame,
    catalog: pd.DataFrame,
    color: str,
    key_prefix: str,
) -> None:
    material_catalog = catalog[
        (catalog["metal"] == metal) & ~catalog["factor"].isin(DISPLAY_EXCLUDED_FACTORS)
    ].copy()
    if material_catalog.empty or monthly_data.empty:
        return
    material_catalog["sort_strength"] = material_catalog["impact_strength"].fillna(-1)
    material_catalog = material_catalog.sort_values("sort_strength", ascending=False)
    st.markdown('<div class="section-title">影响分析</div>', unsafe_allow_html=True)
    category_titles = {
        "供应": "供应端因子分析",
        "需求": "需求端因子分析",
        "库存": "库存端影响因子分析",
        "成本": "成本端影响因子分析",
        "宏观": "宏观端影响因子分析",
        "交易": "市场交易状态分析",
    }
    for category, title in category_titles.items():
        category_data = material_catalog[material_catalog["category"] == category]
        if category_data.empty:
            continue
        st.markdown(f"#### {title}")
        chart_columns = st.columns(2)
        for index, metadata in enumerate(category_data.itertuples(index=False)):
            factor = metadata.factor
            history = factor_series(monthly_data, factor)
            if history.empty:
                continue
            plot_data = history.assign(
                year=history["month"].dt.year,
                month_number=history["month"].dt.month,
            )
            current_year = int(plot_data["year"].max())
            years = sorted(plot_data["year"].unique())
            with chart_columns[index % 2]:
                fig = go.Figure()
                factor_name = plain_factor_name(factor)
                for year, year_data in plot_data.groupby("year"):
                    if year == current_year:
                        line_color, line_width, opacity = "#8B1E2D", 2.8, 1
                    elif year == current_year - 1:
                        line_color, line_width, opacity = "#D9A4AC", 1.8, .9
                    else:
                        line_color, line_width, opacity = "#B8B8BC", 1.15, .48
                    fig.add_trace(
                        go.Scatter(
                            x=year_data["month_number"],
                            y=year_data["value"],
                            mode="lines+markers" if year == current_year else "lines",
                            name=f"{int(year)}年",
                            line={"color": line_color, "width": line_width},
                            marker={"size": 4},
                            opacity=opacity,
                            customdata=np.column_stack(
                                [year_data["month"].dt.strftime("%Y-%m")]
                            ),
                            hovertemplate=(
                                f"<b>{html.escape(factor_name)}</b>"
                                "<br>月份：%{customdata[0]}"
                                "<br>数值：%{y:,.2f}<extra></extra>"
                            ),
                        )
                    )
                strength_text = (
                    f"{metadata.impact_strength:.2f}"
                    if pd.notna(metadata.impact_strength)
                    else "待验证"
                )
                fig.update_layout(
                    title={
                        "text": (
                            f"{factor_name}"
                            f"<br><sup>历史统计关联：{metadata.direction}　影响强度：{strength_text}</sup>"
                        ),
                        "font": {"size": 14, "color": "#2A2A2D"},
                        "y": 0.93,
                        "yanchor": "top",
                    },
                    height=320,
                    template="plotly_white",
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="#ffffff",
                    margin={"l": 30, "r": 15, "t": 96, "b": 28},
                    showlegend=True,
                    dragmode=False,
                    hovermode="x unified",
                    hoverlabel={
                        "bgcolor": "#FFFFFF",
                        "bordercolor": "#8B1E2D",
                        "align": "left",
                        "font": {"color": "#2A2A2D", "size": 12},
                    },
                    legend={
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.01,
                        "xanchor": "left",
                        "x": 0,
                        "font": {"size": 9, "color": "#4B4B50"},
                    },
                )
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=list(range(1, 13)),
                    ticktext=[f"{month}月" for month in range(1, 13)],
                    showgrid=False,
                )
                fig.update_yaxes(showgrid=True, gridcolor="rgba(80,80,84,.10)")
                st.plotly_chart(
                    fig,
                    width="stretch",
                    key=f"{key_prefix}_{category}_{factor}_chart",
                    config={"displayModeBar": False, "scrollZoom": False},
                )
                source_name, source_url = factor_source(factor)
                source_text = f"[{source_name}]({source_url})" if source_url else source_name
                st.caption(f"来源：{source_text}")


def build_factor_strength_figure(factors: pd.DataFrame) -> go.Figure:
    factors = factors.copy()
    factors["strength_rank"] = factors["impact_strength"].rank(
        method="first", ascending=False
    ).astype(int)
    factors = factors.sort_values("impact_strength")
    max_strength = float(factors["impact_strength"].max())
    min_strength = float(factors["impact_strength"].min())
    strength_span = max(max_strength - min_strength, 1e-12)
    bar_colors = factors["impact_strength"].map(
        lambda value: (
            "rgba(79, 119, 154, "
            f"{0.46 + 0.44 * (float(value) - min_strength) / strength_span:.2f})"
        )
    )
    fig = go.Figure(
        go.Bar(
            x=factors["impact_strength"],
            y=factors["factor"].map(plain_factor_name),
            orientation="h",
            marker_color=bar_colors,
            text=factors["impact_strength"].map(lambda value: f"{value:.2f}"),
            textposition="outside",
            customdata=np.column_stack(
                [
                    factors["strength_rank"],
                    factors["direction"],
                    factors["category"],
                    factors["p_value"],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>影响强度排序：第 %{customdata[0]} 位"
                "<br>影响强度：%{x:.2f}<br>类别：%{customdata[2]}"
                "<br>方向：%{customdata[1]}<br>p值：%{customdata[3]:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(360, 46 * len(factors) + 95),
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#4B4B50"},
        margin={"l": 360, "r": 70, "t": 20, "b": 40},
        showlegend=False,
        xaxis_title="影响强度（绝对系数）",
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#CBD5E1",
            "align": "left",
            "font": {"color": "#2A2A2D", "size": 12},
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(80,80,84,.10)",
        range=[0, max_strength * 1.16 if max_strength > 0 else 1],
    )
    fig.update_yaxes(tickfont={"color": "#343438", "size": 11}, automargin=True)
    return fig


def render_full_factor_strength_overview(
    metal: str,
    catalog: pd.DataFrame,
    key_prefix: str,
) -> None:
    factors = catalog[
        (catalog["metal"] == metal) & catalog["impact_strength"].notna()
    ].copy()
    if factors.empty:
        return

    st.markdown(
        '<div class="section-title">全部变量影响强度排序</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"共纳入 {len(factors)} 个变量，按影响强度绝对值从高到低排序；"
        "横向条越长、蓝色越深，影响强度越高。"
    )
    if metal == "lithium_carbonate":
        st.caption(
            "碳酸锂这里展示的是历史影响因子筛选强度；当前月度预测验证后采用朴素基准，"
            "这些数值不是当前预测模型的直接权重。"
        )
    fig = build_factor_strength_figure(factors)
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_full_factor_strength")


def render_supply_demand_relationship(
    metal: str,
    monthly_data: pd.DataFrame,
    catalog: pd.DataFrame,
    key_prefix: str,
) -> None:
    del key_prefix
    snapshot = build_market_relationship_snapshot(monthly_data, catalog, metal)
    if snapshot.empty:
        return
    st.markdown('<div class="section-title">供需关系分析</div>', unsafe_allow_html=True)
    group_names = ["供应", "需求", "成本", "库存"]
    for start in range(0, len(group_names), 2):
        columns = st.columns(2)
        for column, category in zip(columns, group_names[start : start + 2]):
            with column:
                st.markdown(f"#### {category}")
                group = snapshot[snapshot["category"] == category].copy()
                if group.empty:
                    st.markdown(
                        '<div class="balance-empty">暂无适用于该材料的真实指标。</div>',
                        unsafe_allow_html=True,
                    )
                    continue
                group["指标"] = group["factor"].map(plain_factor_name)
                group["环比"] = group["mom"].map(
                    lambda value: f"{value:+.2%}" if pd.notna(value) else "暂无"
                )
                group["同比"] = group["yoy"].map(
                    lambda value: f"{value:+.2%}" if pd.notna(value) else "暂无"
                )
                group["方向"] = group["direction"].map(
                    {"上升": "↑ 上升", "下降": "↓ 下降", "持平": "→ 持平"}
                )
                group["月份"] = group["latest_month"].dt.strftime("%Y-%m")
                group["来源"] = group["factor"].map(lambda value: factor_source(value)[0])
                st.dataframe(
                    group[["指标", "环比", "同比", "方向", "月份", "来源"]],
                    width="stretch",
                    hide_index=True,
                )
    st.info("暂无真实供需平衡量数据。现有预测月环比不作为供需平衡吨数展示。")


def render_terminal_demand(
    metal: str,
    monthly_data: pd.DataFrame,
    color: str,
    key_prefix: str,
) -> None:
    snapshot, histories = terminal_snapshot(monthly_data, metal)
    if snapshot.empty:
        return
    st.markdown('<div class="section-title">终端消费变化汇总</div>', unsafe_allow_html=True)
    for start in range(0, len(snapshot), 3):
        columns = st.columns(min(3, len(snapshot) - start))
        for column, (_, row) in zip(columns, snapshot.iloc[start : start + 3].iterrows()):
            delta = f"{row['mom']:+.2%} 环比" if pd.notna(row["mom"]) else "环比暂无"
            column.metric(
                row["indicator"],
                f"{row['latest_value']:,.1f} {row['unit']}",
                delta,
            )
            if pd.isna(row["mom"]):
                column.caption("环比暂无可比数据。")
            elif row["mom"] > 0:
                column.caption(f"环比 {row['mom']:+.2%} 表示本月较上月增加 {abs(row['mom']):.2%}。")
            elif row["mom"] < 0:
                column.caption(f"环比 {row['mom']:+.2%} 表示本月较上月减少 {abs(row['mom']):.2%}。")
            else:
                column.caption("环比 0.00% 表示本月与上月持平。")
    st.markdown("#### 终端消费指标历史走势")
    chart_columns = st.columns(2)
    snapshot_by_indicator = snapshot.set_index("indicator")
    for index, (indicator, history) in enumerate(histories.items()):
        row = snapshot_by_indicator.loc[indicator]
        with chart_columns[index % 2]:
            fig = go.Figure(
                go.Scatter(
                    x=history["month"],
                    y=history["value"],
                    mode="lines+markers",
                    line={"color": color, "width": 2.4},
                    marker={"size": 4},
                    name=indicator,
                    customdata=np.column_stack(
                        [history["month"].dt.strftime("%Y-%m")]
                    ),
                    hovertemplate=(
                        f"<b>{html.escape(indicator)}</b>"
                        "<br>月份：%{customdata[0]}"
                        f"<br>数值：%{{y:,.2f}} {html.escape(str(row['unit']))}<extra></extra>"
                    ),
                )
            )
            fig.update_layout(
                title={"text": indicator, "font": {"size": 14, "color": "#2A2A2D"}},
                height=285,
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="#ffffff",
                margin={"l": 30, "r": 15, "t": 42, "b": 25},
                showlegend=False,
                yaxis_title=row["unit"],
                dragmode=False,
                hovermode="x unified",
                hoverlabel={
                    "bgcolor": "#FFFFFF",
                    "bordercolor": "#8B1E2D",
                    "align": "left",
                    "font": {"color": "#2A2A2D", "size": 12},
                },
            )
            fig.update_yaxes(showgrid=True, gridcolor="rgba(80,80,84,.10)")
            st.plotly_chart(
                fig,
                width="stretch",
                key=f"{key_prefix}_terminal_{indicator}_chart",
                config={"displayModeBar": False, "scrollZoom": False},
            )
            st.caption(f"来源：{row['source']}")


def render_forecast_driver_analysis(
    metal: str,
    monthly_data: pd.DataFrame,
    catalog: pd.DataFrame,
    stored_contributions: pd.DataFrame,
) -> None:
    contributions = stored_contributions[stored_contributions["metal"] == metal].copy()
    if not contributions.empty:
        contributions["absolute_contribution"] = contributions["contribution"].abs()
        first_period = contributions["forecast_period"].min()
        contributions = (
            contributions[contributions["forecast_period"] == first_period]
            .sort_values("absolute_contribution", ascending=False)
            .head(5)
            .rename(
                columns={
                    "factor_category": "category",
                    "source_period": "source_period",
                }
            )
        )
    else:
        contributions = build_driver_snapshot(monthly_data, catalog, metal, limit=5)
    if contributions.empty:
        return
    contributions["absolute_contribution"] = pd.to_numeric(
        contributions["contribution"], errors="coerce"
    ).abs()
    contributions = contributions.sort_values(
        "absolute_contribution", ascending=False
    ).head(5)
    st.markdown('<div class="section-title">主要影响因子</div>', unsafe_allow_html=True)
    display_data = contributions.copy()
    display_data["影响因子"] = display_data["factor"].map(plain_factor_name)
    display_data["类别"] = display_data["category"]
    numeric_contribution = pd.to_numeric(display_data["contribution"], errors="coerce")
    display_data["影响方向"] = np.select(
        [
            numeric_contribution > 0,
            numeric_contribution < 0,
        ],
        ["正向", "负向"],
        default="无方向",
    )
    display_data["贡献强度"] = display_data["contribution"].map(lambda value: f"{value:+.3f}")
    st.dataframe(
        display_data[["类别", "影响因子", "影响方向", "贡献强度"]],
        width="stretch",
        hide_index=True,
    )


def render_top_impact_factors(metal: str) -> None:
    material = FACTOR_MATERIAL.get(metal)
    coefficients = load_factor_coefficients()
    if material is None or coefficients.empty:
        return

    factors = coefficients[coefficients["品种"] == material].copy()
    if factors.empty:
        return
    factors = factors.sort_values("强弱排名").head(5).copy()
    factors["展示名称"] = factors["变量"].map(plain_factor_name)
    factors["数据来源"] = factors["变量"].map(lambda value: factor_source(str(value))[0])
    factors["数据链接"] = factors["变量"].map(lambda value: factor_source(str(value))[1])
    factors = factors.sort_values("影响强度_绝对值")
    colors = np.where(factors["回归方向"].eq("正向"), "#A8C5B9", "#C98A91")
    fig = go.Figure(
        go.Bar(
            x=factors["影响强度_绝对值"],
            y=factors["展示名称"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:.2f}" for value in factors["影响强度_绝对值"]],
            textposition="outside",
            hovertemplate=(
                "%{y}<br>影响强度：%{customdata:.2f}<br>数值越大，说明该因素与价格变化的关联越强。<extra></extra>"
            ),
            customdata=factors["影响强度_绝对值"],
        )
    )
    fig.update_layout(
        height=320,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b", "size": 12},
        margin={"l": 210, "r": 55, "t": 10, "b": 28},
        showlegend=False,
        xaxis_title="影响强度",
        hovermode="closest",
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#CBD5E1",
            "align": "left",
            "font": {"color": "#2A2A2D", "size": 12},
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(76, 106, 146, 0.10)",
        zeroline=True,
        zerolinecolor="#94a3b8",
        zerolinewidth=1.4,
    )
    fig.update_yaxes(showgrid=False)
    st.markdown('<div class="section-title">影响因素 Top 5</div>', unsafe_allow_html=True)
    st.caption("影响强度用于比较各因素与价格变化的关联程度；它不表示因果关系，也不代表价格会按同样幅度变化。")
    if metal == "lithium_carbonate":
        st.caption("碳酸锂当前月度预测采用朴素基准，以下强度来自历史筛选结果，不代表当前模型直接入模权重。")
    st.plotly_chart(fig, width="stretch")
    st.caption("“ADC12 与 A00铝的价格差”及“ZLD104 与 A00铝的价格差”分别等于对应合金价格减去 A00铝价格。它们反映再生铝或铸造合金相对原铝的成本、加工溢价和替代关系，是模型的市场信号，不表示价差会单向导致价格变化。")
    st.caption("点击下方因素可查看对应官方或原始数据发布页面。")
    source_columns = st.columns(len(factors))
    for column, (_, factor) in zip(source_columns, factors.sort_values("强弱排名").iterrows()):
        column.link_button(f"{factor['展示名称']} · 数据来源", factor["数据链接"], width="stretch")


def render_market_cards(spot: pd.DataFrame, metals: list[str], display: dict[str, str]) -> None:
    cards = st.columns(len(metals))
    for column, metal in zip(cards, metals):
        metal_spot = spot[spot["metal"] == metal].sort_values("trade_date")
        if metal_spot.empty:
            continue
        latest_price = metal_spot.iloc[-1]["price_cny_per_tonne"]
        change_5d = metal_spot["price_cny_per_tonne"].pct_change(5).iloc[-1]
        if pd.isna(change_5d) or abs(float(change_5d)) < 0.00005:
            change_text, direction_class = "暂无" if pd.isna(change_5d) else "0.00%", "steady"
        elif change_5d > 0:
            change_text, direction_class = f"↑ {change_5d:.2%}", "up"
        else:
            change_text, direction_class = f"↓ {abs(change_5d):.2%}", "down"
        column.markdown(
            f'''<div class="market-card">
                <div class="market-card-title">{html.escape(display.get(metal, metal))}</div>
                <div class="market-card-value">{latest_price:,.0f} {price_unit(metal)}</div>
                <div class="market-card-change {direction_class}">{change_text}</div>
            </div>''',
            unsafe_allow_html=True,
        )


def render_forecast_summary_cards(
    forecasts: pd.DataFrame,
    metals: list[str],
    display: dict[str, str],
) -> None:
    forecast_summaries: list[tuple[str, str, float, float, float]] = []
    for metal in metals:
        forecast = forecasts[forecasts["metal"] == metal].sort_values("forecast_date")
        if len(forecast) < 2:
            continue
        first_price = float(forecast.iloc[0]["predicted_price_cny_per_tonne"])
        last_price = float(forecast.iloc[-1]["predicted_price_cny_per_tonne"])
        forecast_summaries.append(
            (
                metal,
                display.get(metal, metal),
                first_price,
                last_price,
                last_price / first_price - 1,
            )
        )
    if not forecast_summaries:
        return

    st.markdown('<div class="section-title">第 30 日预测价格</div>', unsafe_allow_html=True)
    future_cards = st.columns(len(forecast_summaries))
    for column, (metal, name, first_price, last_price, change) in zip(
        future_cards, forecast_summaries
    ):
        if abs(change) < 0.00005:
            change_text, direction_class = "0.00%", "steady"
        elif change > 0:
            change_text, direction_class = f"↑ {change:.2%}", "up"
        else:
            change_text, direction_class = f"↓ {abs(change):.2%}", "down"
        column.markdown(
            f'''<div class="market-card">
                <div class="market-card-title">{html.escape(name)}</div>
                <div class="market-card-value">{last_price:,.0f} {price_unit(metal)}</div>
                <div class="market-card-change {direction_class}">{change_text}</div>
            </div>''',
            unsafe_allow_html=True,
        )
    st.caption("数值表示相对首日预测的变化。")


def render_home_overview(
    spot: pd.DataFrame,
    metals: list[str],
    display: dict[str, str],
    colors: dict[str, str],
    verified_events: pd.DataFrame,
    news_payload: dict[str, object] | None = None,
) -> None:
    render_daily_news(news_payload or {})
    st.markdown('<div class="section-title">市场概览</div>', unsafe_allow_html=True)
    render_market_cards(spot, metals, display)
    st.caption("数值表示最新不含税现货均价相较 5 个交易日前的变化。")

    st.markdown('<div class="section-title">综合价格趋势</div>', unsafe_allow_html=True)
    material_column, window_column = st.columns([2, 3])
    default_metal = "aluminum_a00" if "aluminum_a00" in metals else metals[0]
    selected_trend_metal = material_column.selectbox(
        "原材料",
        metals,
        index=metals.index(default_metal),
        format_func=lambda item: display.get(item, item),
        key="overview_trend_metal",
    )
    selected_window = window_column.radio(
        "趋势区间",
        ["近7日", "近30日", "近60日", "全部历史"],
        index=2,
        horizontal=True,
        key="overview_trend_window",
    )
    full_history = spot[spot["metal"] == selected_trend_metal].sort_values("trade_date")
    selected_history = filter_price_history(full_history, selected_window)
    selected_events = events_in_range(
        verified_events,
        selected_trend_metal,
        selected_history["trade_date"].min(),
        selected_history["trade_date"].max(),
    )
    start_price = float(selected_history.iloc[0]["price_cny_per_tonne"])
    latest_price = float(selected_history.iloc[-1]["price_cny_per_tonne"])
    selected_unit = price_unit(selected_trend_metal)
    stats = st.columns(4)
    stats[0].metric("最新价格", f"{latest_price:,.0f} {selected_unit}")
    stats[1].metric("区间变化", f"{latest_price / start_price - 1:+.2%}")
    stats[2].metric(
        "区间最高",
        f"{selected_history['price_cny_per_tonne'].max():,.0f} {selected_unit}",
    )
    stats[3].metric(
        "区间最低",
        f"{selected_history['price_cny_per_tonne'].min():,.0f} {selected_unit}",
    )
    st.plotly_chart(
        build_price_history_figure(
            selected_history,
            selected_events,
            colors[selected_trend_metal],
            selected_unit,
        ),
        width="stretch",
        key="overview_single_price_trend",
    )
    st.caption(
        f"数据截止 {selected_history['trade_date'].max():%Y-%m-%d}，"
        f"价格口径为不含税{selected_unit}。"
    )
    render_range_stats(
        selected_trend_metal,
        full_history,
        colors[selected_trend_metal],
    )
    render_verified_event_details(selected_events)


def render_report_center(
    forecasts: pd.DataFrame,
    monthly_forecasts: pd.DataFrame,
    runs: pd.DataFrame,
    display: dict[str, str],
) -> None:
    st.markdown('<div class="section-title">报告中心</div>', unsafe_allow_html=True)
    catalog_column, status_column = st.columns([3, 1])
    reports = pd.DataFrame(
        [
            {"报告": "日度价格预测", "内容": "六种原材料未来30天预测", "状态": "可用" if not forecasts.empty else "暂无"},
            {"报告": "月度均价预测", "内容": "各原材料月度预测均价", "状态": "可用" if not monthly_forecasts.empty else "暂无"},
            {"报告": "影响因素分析", "内容": "影响强度与 Top 5 主要因素", "状态": "可用" if not load_factor_coefficients().empty else "暂无"},
            {"报告": "数据更新记录", "内容": "最近数据更新与模型重训状态", "状态": "可用" if not runs.empty else "暂无"},
        ]
    )
    with catalog_column:
        st.dataframe(reports, width="stretch", hide_index=True)
        st.markdown('<div class="section-title">数据导出</div>', unsafe_allow_html=True)
        export = forecasts.copy()
        if not export.empty:
            export["品种"] = export["metal"].map(display).fillna(export["metal"])
            export["预测日期"] = pd.to_datetime(export["forecast_date"]).dt.strftime("%Y-%m-%d")
            export = export[["品种", "预测日期", "predicted_price_cny_per_tonne", "lower_bound", "upper_bound"]].rename(
                columns={"predicted_price_cny_per_tonne": "预测价格", "lower_bound": "下限", "upper_bound": "上限"}
            )
            st.download_button(
                "下载日度预测 CSV",
                export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="国内原材料日度预测.csv",
                mime="text/csv",
            )
        monthly_export = monthly_forecasts.copy()
        if not monthly_export.empty:
            monthly_export["品种"] = monthly_export["metal"].map(display).fillna(monthly_export["metal"])
            monthly_export["预测月份"] = pd.to_datetime(monthly_export["forecast_month"]).dt.strftime("%Y-%m")
            monthly_export = monthly_export[["品种", "预测月份", "predicted_price_cny_per_tonne", "source", "model_version"]].rename(
                columns={
                    "predicted_price_cny_per_tonne": "预测月度均价",
                    "source": "数据来源",
                    "model_version": "模型版本",
                }
            )
            st.download_button(
                "下载月度预测 CSV",
                monthly_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="国内原材料月度预测.csv",
                mime="text/csv",
            )
        raw_monthly_export = build_raw_monthly_export()
        if raw_monthly_export is not None:
            st.download_button(
                "下载原始月度数据 Excel（5个工作表）",
                raw_monthly_export,
                file_name="国内原材料原始月度数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="按 1#铜、A00铝、1#白银、ADC12、ZLD104 分为五个工作表。",
            )
        factor_export = build_factor_export()
        if factor_export is not None:
            st.download_button(
                "下载影响因素分析 Excel（5个工作表）",
                factor_export,
                file_name="国内原材料影响因素分析.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="按 1#铜、A00铝、1#白银、ADC12、ZLD104 分为五个工作表。",
            )
    with status_column:
        st.metric("日度预测", "已生成" if not forecasts.empty else "暂无")
        st.metric("月度预测", "已生成" if not monthly_forecasts.empty else "暂无")
        st.metric("最近更新", str(runs.iloc[0]["status"]) if not runs.empty else "暂无")


def render_range_stats(metal: str, metal_spot: pd.DataFrame, color: str) -> None:
    with st.expander("区间统计", expanded=False):
        unit = price_unit(metal)
        min_date = metal_spot["trade_date"].min().date()
        max_date = metal_spot["trade_date"].max().date()
        state_prefix = f"range_{metal}"

        if f"{state_prefix}_start" not in st.session_state:
            st.session_state[f"{state_prefix}_start"] = min_date
            st.session_state[f"{state_prefix}_end"] = max_date

        left, right, button_col = st.columns([1, 1, 0.7])
        if button_col.button("随机区间", key=f"{state_prefix}_random"):
            start, end = random_date_range(min_date, max_date)
            st.session_state[f"{state_prefix}_start"] = start
            st.session_state[f"{state_prefix}_end"] = end

        start = left.date_input(
            "开始日期",
            value=st.session_state[f"{state_prefix}_start"],
            min_value=min_date,
            max_value=max_date,
            key=f"{state_prefix}_start",
        )
        end = right.date_input(
            "结束日期",
            value=st.session_state[f"{state_prefix}_end"],
            min_value=min_date,
            max_value=max_date,
            key=f"{state_prefix}_end",
        )

        if start > end:
            st.warning("开始日期不能晚于结束日期。")
            return

        selected = metal_spot[
            (metal_spot["trade_date"].dt.date >= start) & (metal_spot["trade_date"].dt.date <= end)
        ].copy()
        if selected.empty:
            st.info("这个区间没有价格数据。")
            return

        values = selected["price_cny_per_tonne"]
        first_price = float(values.iloc[0])
        last_price = float(values.iloc[-1])
        price_range = float(values.max() - values.min())
        pct_change = (last_price / first_price - 1) * 100 if first_price else np.nan
        daily_vol = values.pct_change(fill_method=None).std() * 100
        stat_cols = st.columns(4)
        stat_cols[0].metric("区间均价", f"{values.mean():,.2f} {unit}")
        stat_cols[1].metric("最高价", f"{values.max():,.2f} {unit}")
        stat_cols[2].metric("最低价", f"{values.min():,.2f} {unit}")
        stat_cols[3].metric("有效天数", f"{len(selected)}")
        stat_cols2 = st.columns(4)
        stat_cols2[0].metric("区间首日价", f"{first_price:,.2f} {unit}")
        stat_cols2[1].metric("区间末日价", f"{last_price:,.2f} {unit}")
        stat_cols2[2].metric("区间涨跌幅", f"{pct_change:.2f}%")
        stat_cols2[3].metric("价格极差", f"{price_range:,.2f} {unit}")
        stat_cols3 = st.columns(4)
        stat_cols3[0].metric("日度波动率", f"{daily_vol:.2f}%" if pd.notna(daily_vol) else "暂无")
        stat_cols3[1].metric("中位数", f"{values.median():,.2f} {unit}")
        stat_cols3[2].metric("标准差", f"{values.std():,.2f} {unit}")
        stat_cols3[3].metric("变异系数", f"{values.std() / values.mean() * 100:.2f}%")
        st.caption(
            "日度波动率 = 所选区间每日涨跌幅的样本标准差 × 100%；"
            "每日涨跌幅 = 当日价格 ÷ 前一交易日价格 - 1。"
            "变异系数 = 价格标准差 ÷ 区间均价 × 100%。"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=selected["trade_date"],
                y=selected["price_cny_per_tonne"],
                mode="lines",
                name="所选区间",
                line={"color": color, "width": 2},
                hovertemplate=(
                    f"<b>%{{x|%Y-%m-%d}}</b>"
                    f"<br>不含税均价：%{{y:,.2f}} {unit}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[selected["trade_date"].min(), selected["trade_date"].max()],
                y=[values.mean(), values.mean()],
                mode="lines",
                name="区间均价",
                line={"dash": "dash", "color": "#4A4A4A", "width": 1.4},
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            height=260,
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="#ffffff",
            font={"color": "#64748b"},
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            yaxis_title=unit,
            hovermode="closest",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "x": 0,
                "font": {"color": "#4B4B50"},
            },
        )
        st.plotly_chart(fig, width="stretch")


def render_daily_backtest_legacy(
    spot: pd.DataFrame,
    market_features: pd.DataFrame,
    metals: list[str],
    display: dict[str, str],
    colors: dict[str, str],
    model_version: str,
) -> None:
    st.subheader("历史截断预测对比")
    selected_metal = st.selectbox("选择品种", metals, format_func=lambda item: display.get(item, item))
    unit = price_unit(selected_metal)
    metal_spot = spot[spot["metal"] == selected_metal].sort_values("trade_date")
    metal_features = (
        market_features[market_features["metal"] == selected_metal].sort_values("trade_date")
        if not market_features.empty
        else pd.DataFrame()
    )

    min_date = metal_spot["trade_date"].min().date()
    max_date = metal_spot["trade_date"].max().date()
    default_train_end = min(max(date(min_date.year + 1, 12, 31), min_date), max_date)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 0.8])
    train_start = c1.date_input("训练开始", value=min_date, min_value=min_date, max_value=max_date)
    train_end = c2.date_input("训练结束", value=default_train_end, min_value=min_date, max_value=max_date)
    eval_end = c3.date_input("对比结束", value=max_date, min_value=min_date, max_value=max_date)
    max_compare_days = c4.number_input("最多对比天数", min_value=30, max_value=730, value=365, step=30)

    if train_start >= train_end:
        st.warning("训练开始日期必须早于训练结束日期。")
        return
    if eval_end <= train_end:
        st.warning("对比结束日期必须晚于训练结束日期。")
        return

    train_rows = metal_spot[
        (metal_spot["trade_date"].dt.date >= train_start) & (metal_spot["trade_date"].dt.date <= train_end)
    ]
    if len(train_rows) < 60:
        st.warning("训练区间有效样本少于 60 天，预测会不稳定。")
        return

    if not st.button("运行回测", type="primary"):
        st.info("选择训练区间和对比结束日期后，点击“运行回测”生成预测与真实价格对比。")
        return

    requested_horizon_days = (eval_end - train_end).days
    horizon_days = min(requested_horizon_days, int(max_compare_days))
    effective_eval_end = min(eval_end, train_end + timedelta(days=horizon_days))
    if effective_eval_end < eval_end:
        st.info(f"当前回测最多展示训练截止后的 {horizon_days} 天；如需更长区间，可调大“最多对比天数”。")
    with st.spinner("正在用所选历史区间重新训练并回测..."):
        result = run_backtest(
            metal=selected_metal,
            train_start=train_start,
            train_end=train_end,
            eval_end=effective_eval_end,
            metal_spot=metal_spot,
            metal_features=metal_features,
            horizon_days=horizon_days,
            model_version=model_version,
        )

    if result.empty:
        st.info("预测日期与真实日期没有可对比的重叠数据。")
        return

    error = result["predicted_price_cny_per_tonne"] - result["actual_price_cny_per_tonne"]
    mae = error.abs().mean()
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mape = (error.abs() / result["actual_price_cny_per_tonne"]).mean() * 100
    bias = error.mean()

    metric_cols = st.columns(4)
    metric_cols[0].metric("平均绝对误差", f"{mae:,.2f}")
    metric_cols[1].metric("MAPE", f"{mape:.2f}%")
    metric_cols[2].metric("RMSE", f"{rmse:,.2f}")
    metric_cols[3].metric("平均偏差", f"{bias:,.2f}")
    st.markdown(
        f'<div class="comparison-note">指标说明：平均绝对误差表示预测价与真实价平均相差多少{unit}；MAPE 是平均误差率，越低越好；RMSE 会更重视较大的预测错误；平均偏差为正表示整体预测偏高，为负表示整体预测偏低。</div>',
        unsafe_allow_html=True,
    )

    latest_compare = result.iloc[-1]
    st.markdown('<div class="section-title">预测与真实价格对比</div>', unsafe_allow_html=True)
    cmp_cols = st.columns(3)
    cmp_cols[0].metric("最后对比日真实价格", f"{latest_compare['actual_price_cny_per_tonne']:,.2f}")
    cmp_cols[1].metric("最后对比日预测价格", f"{latest_compare['predicted_price_cny_per_tonne']:,.2f}")
    cmp_cols[2].metric("最后对比日误差", f"{latest_compare['predicted_price_cny_per_tonne'] - latest_compare['actual_price_cny_per_tonne']:,.2f}")

    table = result.copy()
    table["误差"] = table["predicted_price_cny_per_tonne"] - table["actual_price_cny_per_tonne"]
    table["误差率"] = table["误差"] / table["actual_price_cny_per_tonne"] * 100
    latest10 = table.tail(10).copy()
    latest10_display = latest10.rename(
        columns={
            "forecast_date": "日期",
            "actual_price_cny_per_tonne": "真实价格",
            "predicted_price_cny_per_tonne": "预测价格",
        }
    )[["日期", "真实价格", "预测价格", "误差", "误差率"]]
    latest10_display["误差率"] = latest10_display["误差率"].map(lambda value: f"{value:.2f}%")
    st.markdown('<div class="section-title">最后10个对比交易日</div>', unsafe_allow_html=True)
    st.dataframe(latest10_display, width="stretch", hide_index=True)

    recent_fig = go.Figure()
    recent_fig.add_trace(
        go.Bar(
            x=latest10["forecast_date"],
            y=latest10["actual_price_cny_per_tonne"],
            name="真实价格（实际）",
            marker_color="rgba(37, 99, 235, 0.58)",
        )
    )
    recent_fig.add_trace(
        go.Bar(
            x=latest10["forecast_date"],
            y=latest10["predicted_price_cny_per_tonne"],
            name="预测价格",
            marker_color="rgba(208, 46, 46, 0.58)",
        )
    )
    recent_fig.update_layout(
        barmode="group",
        height=300,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b"},
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        yaxis_title=unit,
    )
    st.plotly_chart(recent_fig, width="stretch")

    fig = go.Figure()
    full_window = metal_spot[
        (metal_spot["trade_date"].dt.date >= train_start) & (metal_spot["trade_date"].dt.date <= effective_eval_end)
    ]
    fig.add_trace(
        go.Scatter(
            x=full_window["trade_date"],
            y=full_window["price_cny_per_tonne"],
            mode="lines",
            name="真实价格（实际）",
            line={"color": colors[selected_metal], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result["forecast_date"],
            y=result["predicted_price_cny_per_tonne"],
            mode="lines",
            name="历史截断预测（预测）",
            line={"color": "#2E78F6", "width": 1.6},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result["forecast_date"],
            y=result["actual_price_cny_per_tonne"],
            mode="markers",
            name="真实价格点（实际）",
            marker={"color": colors[selected_metal], "size": 4, "opacity": 0.55},
        )
    )
    train_end_x = pd.Timestamp(train_end).strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=train_end_x,
        x1=train_end_x,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line={"color": "#555", "width": 1, "dash": "dash"},
    )
    fig.add_annotation(
        x=train_end_x,
        y=1.04,
        xref="x",
        yref="paper",
        text="训练截止",
        showarrow=False,
        font={"size": 12, "color": "#555"},
    )
    fig.update_layout(
        height=500,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b"},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title=unit,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    error_fig = go.Figure()
    error_fig.add_trace(
        go.Bar(
            x=table["forecast_date"],
            y=table["误差"],
            name="预测误差",
            marker_color=np.where(table["误差"] >= 0, "#d97706", "#2563eb"),
        )
    )
    error_fig.add_hline(y=0, line_color="#475467", line_width=1)
    error_fig.update_layout(
        height=280,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b"},
        margin={"l": 20, "r": 20, "t": 10, "b": 20},
        yaxis_title=f"预测 - 真实（{unit}）",
    )
    st.plotly_chart(error_fig, width="stretch")

    recent90 = table.tail(90).copy()
    detail = recent90.rename(
        columns={
            "forecast_date": "日期",
            "actual_price_cny_per_tonne": "真实价格",
            "predicted_price_cny_per_tonne": "预测价格",
        }
    )[["日期", "真实价格", "预测价格", "误差", "误差率"]].copy()
    detail["误差率"] = detail["误差率"].map(lambda value: f"{value:.2f}%")
    st.markdown('<div class="section-title">最近90个交易日实际值与预测值对比</div>', unsafe_allow_html=True)
    st.caption(f"当前表格展示 {len(detail)} 个有真实价格的交易日；若不足 90 个，说明所选回测区间内可对比交易日不足 90 个。")
    st.dataframe(
        detail,
        width="stretch",
        hide_index=True,
    )


def monthly_history_cache_token() -> str:
    """Change the cache key only when the local SQLite database changes."""
    config = load_config()
    database_path = Path(config.database_path)
    if not database_path.is_absolute():
        database_path = Path(__file__).resolve().parent / database_path
    try:
        stat = database_path.stat()
    except OSError:
        return "missing"
    return f"{stat.st_mtime_ns}:{stat.st_size}"


@st.cache_data(show_spinner=False)
def load_monthly_history_predictions(cache_token: str = "") -> pd.DataFrame:
    del cache_token
    output_rows: list[pd.DataFrame] = []
    output_dir = Path(__file__).resolve().parent / "monthly_price_prediction_outputs" / "drop_limited_vars"
    raw_workbook = None
    for candidate in sorted(output_dir.glob("*.xlsx")):
        try:
            if "v2_modeling_data_full" in pd.ExcelFile(candidate).sheet_names:
                raw_workbook = candidate
                break
        except (OSError, ValueError):
            continue

    sample_metals = {
        1: "silver_1",
        2: "copper_1",
        3: "aluminum_a00",
        4: "aluminum_adc12",
        5: "aluminum_zld104",
    }
    if raw_workbook is not None:
        for sample_index, metal in sample_metals.items():
            try:
                sample = pd.read_excel(raw_workbook, sheet_name=f"sample_{sample_index}")
            except ValueError:
                continue
            if sample.shape[1] < 3:
                continue
            month_col, target_col = sample.columns[:2]
            predictors = list(sample.columns[2:])
            frame = sample.replace([np.inf, -np.inf], np.nan).dropna().copy()
            frame[month_col] = pd.to_datetime(frame[month_col])
            predictors = [column for column in predictors if frame[column].std(ddof=0) > 0]
            if len(frame) < max(15, len(predictors) + 8):
                continue
            fit = sm.OLS(
                frame[target_col],
                sm.add_constant(frame[predictors], has_constant="add"),
            ).fit()
            predicted = fit.predict(sm.add_constant(frame[predictors], has_constant="add"))
            output_rows.append(
                pd.DataFrame(
                    {
                        "metal": metal,
                        "month": frame[month_col],
                        "actual_monthly_price": frame[target_col],
                        "predicted_monthly_price": predicted,
                    }
                )
            )

    # The legacy monthly workbook is refreshed less often than the spot-price
    # database.  Extend each of the five legacy series with the newest complete
    # months from SQLite so the date controls do not stop at the workbook's
    # last month (currently 2026-05).  These appended values use a transparent
    # one-month-lag baseline and are kept separate from the original OLS sample.
    try:
        config = load_config()
        database_path = Path(config.database_path)
        if not database_path.is_absolute():
            database_path = Path(__file__).resolve().parent / database_path
        if database_path.exists():
            conn = connect(database_path, read_only=True)
            current_spot = load_spot_prices(conn)
            conn.close()
            if not current_spot.empty:
                current_spot["trade_date"] = pd.to_datetime(current_spot["trade_date"])
                current_spot["month"] = current_spot["trade_date"].dt.to_period("M").dt.start_time
                monthly_spot = (
                    current_spot.groupby(["metal", "month"], as_index=False)["price_cny_per_tonne"]
                    .mean()
                    .rename(columns={"price_cny_per_tonne": "actual_monthly_price"})
                )
                for metal in sample_metals.values():
                    existing = next(
                        (row for row in output_rows if not row.empty and row.iloc[0]["metal"] == metal),
                        None,
                    )
                    if existing is None or existing.empty:
                        continue
                    last_month = pd.to_datetime(existing["month"]).max()
                    extra = monthly_spot[
                        (monthly_spot["metal"] == metal) & (monthly_spot["month"] > last_month)
                    ].sort_values("month")
                    if extra.empty:
                        continue
                    previous_actual = float(
                        existing.loc[existing["month"].idxmax(), "actual_monthly_price"]
                    )
                    extra_rows = []
                    for row in extra.itertuples(index=False):
                        extra_rows.append(
                            {
                                "metal": metal,
                                "month": row.month,
                                "actual_monthly_price": float(row.actual_monthly_price),
                                "predicted_monthly_price": previous_actual,
                            }
                        )
                        previous_actual = float(row.actual_monthly_price)
                    output_rows.append(pd.DataFrame(extra_rows))
    except (OSError, sqlite3.DatabaseError, ValueError):
        # The static workbook remains available if the optional extension
        # source is unavailable during a deployment.
        pass

    lithium_dir = Path(__file__).resolve().parent / "lithium_carbonate_prediction_outputs"
    lithium_file = next(iter(sorted(lithium_dir.glob("*monthly*fitted*.csv"))), None)
    if lithium_file is not None:
        lithium = pd.read_csv(lithium_file, encoding="utf-8-sig")
        if {"month", "target_price", "predicted_monthly_price"}.issubset(lithium.columns):
            output_rows.append(
                pd.DataFrame(
                    {
                        "metal": "lithium_carbonate",
                        "month": pd.to_datetime(lithium["month"]),
                        "actual_monthly_price": lithium["target_price"],
                        "predicted_monthly_price": lithium["predicted_monthly_price"],
                    }
                )
            )

    if not output_rows:
        return pd.DataFrame(columns=["metal", "month", "actual_monthly_price", "predicted_monthly_price"])
    return pd.concat(output_rows, ignore_index=True).sort_values(["metal", "month"]).reset_index(drop=True)


def render_backtest(
    spot: pd.DataFrame,
    market_features: pd.DataFrame,
    metals: list[str],
    display: dict[str, str],
    model_version: str,
) -> None:
    del spot, market_features, model_version
    st.subheader("历史月均价与模型预测")
    history = load_monthly_history_predictions(monthly_history_cache_token())
    available_metals = [metal for metal in metals if metal in set(history["metal"])]
    if not available_metals:
        st.info("暂未找到可展示的月度模型历史结果。")
        return

    selected_metal = st.selectbox(
        "原材料",
        available_metals,
        format_func=lambda item: display.get(item, item),
        key="monthly_history_metal",
    )
    unit = price_unit(selected_metal)
    series = history[history["metal"] == selected_metal].copy().sort_values("month")
    min_month = series["month"].min().date()
    max_month = series["month"].max().date()
    start_col, end_col = st.columns(2)
    selected_start = start_col.date_input(
        "开始月份",
        value=min_month,
        min_value=min_month,
        max_value=max_month,
        key="monthly_history_start",
    )
    selected_end = end_col.date_input(
        "结束月份",
        value=max_month,
        min_value=min_month,
        max_value=max_month,
        key="monthly_history_end",
    )
    if selected_start > selected_end:
        st.warning("开始月份不能晚于结束月份。")
        return

    selected = series[
        (series["month"] >= pd.Timestamp(selected_start)) & (series["month"] <= pd.Timestamp(selected_end))
    ].copy()
    if selected.empty:
        st.info("所选区间没有完整的模型样本。")
        return

    error = selected["predicted_monthly_price"] - selected["actual_monthly_price"]
    mae = error.abs().mean()
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mape = (error.abs() / selected["actual_monthly_price"]).mean() * 100
    metrics = st.columns(4)
    metrics[0].metric("历史月数", f"{len(selected)}")
    metrics[1].metric("MAE", f"{mae:,.2f}")
    metrics[2].metric("MAPE", f"{mape:.2f}%")
    metrics[3].metric("RMSE", f"{rmse:,.2f}")
    st.markdown(
        f'<div class="comparison-note">指标说明：MAE（平均绝对误差）表示每个月的预测价格平均相差多少{unit}；MAPE（平均绝对百分比误差）表示平均相差真实价格的百分之多少；RMSE（均方根误差）会对较大的偏差给予更高权重。三项指标均为越低越好。</div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=selected["month"],
            y=selected["actual_monthly_price"],
            mode="lines+markers",
            name="历史月度均价",
            line={"color": "#D97745", "width": 2.8},
            marker={"color": "#D97745", "size": 5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=selected["month"],
            y=selected["predicted_monthly_price"],
            mode="lines+markers",
            name="模型预测值",
            line={"color": "#A32035", "width": 2.2, "dash": "dash"},
            marker={"color": "#A32035", "size": 4, "symbol": "diamond"},
        )
    )
    fig.update_layout(
        height=440,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b"},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title=unit,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "font": {"color": "#4B4B50"}},
    )
    st.plotly_chart(fig, width="stretch")

    table = selected.assign(
        月份=selected["month"].dt.strftime("%Y-%m"),
        **{
            f"历史月度均价（{unit}）": selected["actual_monthly_price"],
            f"模型预测值（{unit}）": selected["predicted_monthly_price"],
            f"预测误差（{unit}）": error,
        },
        误差率=error / selected["actual_monthly_price"] * 100,
    )[
        [
            "月份",
            f"历史月度均价（{unit}）",
            f"模型预测值（{unit}）",
            f"预测误差（{unit}）",
            "误差率",
        ]
    ]
    table["误差率"] = table["误差率"].map(lambda value: f"{value:.2f}%")
    st.markdown('<div class="section-title">月度价格明细</div>', unsafe_allow_html=True)
    st.dataframe(table, width="stretch", hide_index=True)


def render_model_formula() -> None:
    st.subheader("模型说明")
    st.markdown(
        """
        平台采用“日度组合预测 + 月度价格校准”的方法，不依赖单一算法。系统先根据历史价格生成未来 30 天的日度预测，
        再结合月度模型的判断进行校准，并通过历史回测选择表现更稳定的结果。所有价格均为不含税价格，预测结果仅用于辅助判断。
        """
    )
    st.markdown(
        "[查看或下载详细版《模型方法与显著因子筛选说明》](https://github.com/Yep1yu/Impact-and-Forecast-of-Domestic-Copper-and-Aluminum-Raw-Material-Procurement-Prices/raw/refs/heads/main/%E5%8E%9F%E6%9D%90%E6%96%99%E4%BB%B7%E6%A0%BC%E9%A2%84%E6%B5%8B%E5%B9%B3%E5%8F%B0_%E6%A8%A1%E5%9E%8B%E6%96%B9%E6%B3%95%E4%B8%8E%E6%98%BE%E8%91%97%E5%9B%A0%E5%AD%90%E7%AD%9B%E9%80%89%E8%AF%B4%E6%98%8E.docx)"
    )

    st.markdown('<div class="section-title">1. 日度组合预测</div>', unsafe_allow_html=True)
    st.markdown(
        """
        系统会同时计算多种结果：以最近价格为基准、根据不同窗口的近期变化速度推演、根据历史价格规律推演，
        以及使用 Ridge 回归等方法。系统先用滚动回测评估每个候选模型，再按各自误差给同一天的预测加权，
        误差越小的候选模型权重越高；中位数组合本身也是候选结果之一。
        """
    )
    st.latex(r"P^{ensemble}_{t+h}=\frac{\sum_j w_{j,h}P^{(j)}_{t+h}}{\sum_j w_{j,h}},\quad w_{j,h}\propto \frac{1}{MAE_{j,h}^{2}}")
    st.caption("P 表示价格，t 表示当前日期，h 表示预测到未来第几天；仅保留回测不劣于 Naive_last 的候选，MAE 越小权重越高。")
    st.table(
        pd.DataFrame(
            [
                {
                    "模型": "最近价格基准（Naive_last）",
                    "核心思路": "将当前最新价格作为未来价格的基准",
                    "作用": "提供最稳妥的基准结果",
                },
                {
                    "模型": "滚动对数趋势（5/20/60/120天）",
                    "核心思路": "按不同时间窗口的近期涨跌速度推演未来",
                    "作用": "捕捉短期、中期和较长期趋势",
                },
                {
                    "模型": "Ridge直接多步回归（Ridge_direct_h）",
                    "核心思路": "使用滞后价格、均线、涨跌幅和波动率等特征，分别预测第1～30天",
                    "作用": "利用多个历史特征进行结构化预测",
                },
                {
                    "模型": "ARIMA对数价格模型（ARIMA_log）",
                    "核心思路": "根据价格序列自身的趋势和时间相关性进行预测",
                    "作用": "补充时间序列规律，按配置启用",
                },
                {
                    "模型": "中位数组合（Median_ensemble）",
                    "核心思路": "对多个候选模型的同日预测取中位数",
                    "作用": "作为稳健候选结果，参与最终组合选择",
                },
            ]
        ),
    )

    st.markdown('<div class="section-title">2. 月度价格校准</div>', unsafe_allow_html=True)
    st.markdown(
        """
        月度模型以不含税现货月均价为预测目标，以“上月月均价”表示价格惯性，
        再使用供应、需求、库存、成本、宏观、事件和交易状态等非价格变量判断未来月均价格。
        直接现货/期货价格变化及产品价差不作为网页影响因子展示。本轮剔除价格类因子的重训结果先独立回测，暂不替换网页现有预测结果。
        """
    )
    st.latex(r"T_m=\alpha+\rho P^{month}_{m-1}+\sum_j\beta_jX_{j,m}+\varepsilon_m")
    st.caption("Tₘ 是未来月份月均价；Pᵐᵒⁿᵗʰₘ₋₁ 是上月月均价（价格惯性项，不作为影响因子展示）；X 是入选的非价格变量。")
    st.table(
        pd.DataFrame(
            [
                {"参数项": "预测目标", "当前设定": "未来月份不含税现货月均价"},
                {"参数项": "价格惯性", "当前设定": "上月月均价，以递推方式进入未来预测"},
                {"参数项": "变量筛选", "当前设定": "月度显著性筛选 p＜0.10"},
                {"参数项": "剔除范围", "当前设定": "直接期货/现货价格变化及产品价差；网页不展示为影响因子"},
                {"参数项": "未来变量假设", "当前设定": "连续变量沿用最近可得值；事件冲击默认为 0"},
                {"参数项": "本轮重训结果", "当前设定": "先用于独立回测，不同步替换网页预测"},
            ]
        )
    )
    st.latex(r"P^{final}_{t+h}=P^{daily}_{t+h}+w_h\,(T_m-\overline{P}^{daily}_m)")
    st.caption("Tₘ 是月度模型判断的月均价格，P̄ᵈᵃⁱˡʸₘ 是日度预测的月均值，wₕ 随预测期限增加，当前最大校准强度为 0.75。")

    st.markdown('<div class="section-title">3. 影响因素与显著性</div>', unsafe_allow_html=True)
    st.markdown(
        """
        影响因素分析以“价格月环比”为目标，候选因素包括库存、PPI、PMI、下游行业、进口、宏观和事件等指标。
        直接价格变化及产品价差不纳入本轮网页影响因子展示。系统将数据按月对齐并标准化后进行多元回归：
        """
    )
    st.latex(r"Z(Y)=\beta_0+\beta_1Z(X_1)+\beta_2Z(X_2)+\cdots+\varepsilon")
    st.markdown(
        """
        标准化系数的正负表示影响方向，绝对值越大表示相对影响越强。显著性按 p 值判断：p＜0.01 为极显著，
        p＜0.05 为显著，p＜0.10 为边际显著。需要注意，显著因子主要用于解释历史变化，
        不等于它会直接作为日度预测公式的固定权重；部分因素会通过月度模型间接参与校准。
        """
    )

    st.markdown('<div class="section-title">4. 回测与预测区间</div>', unsafe_allow_html=True)
    st.markdown(
        """
        系统会用过去的真实价格反复回测不同方法，并按“原材料种类”和“预测天数”比较误差。
        页面中的预测下限和上限反映历史误差带，预测越远通常越宽。
        """
    )
    st.markdown("MAE、MAPE 和 RMSE 均用于衡量预测误差，数值越低表示历史表现越好。")
    st.latex(r"MAE=\frac{1}{n}\sum_{i=1}^{n}|\hat{y}_i-y_i|")
    st.latex(r"MAPE=\frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{\hat{y}_i-y_i}{y_i}\right|")
    st.latex(r"RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2}")
    st.caption("MAE 表示平均差多少；MAPE 表示平均误差占实际价格的比例；RMSE 对较大的单次误差更敏感。")


@st.cache_data(show_spinner=False)
def run_backtest(
    metal: str,
    train_start: date,
    train_end: date,
    eval_end: date,
    metal_spot: pd.DataFrame,
    metal_features: pd.DataFrame,
    horizon_days: int,
    model_version: str,
) -> pd.DataFrame:
    train_spot = metal_spot[
        (metal_spot["trade_date"].dt.date >= train_start) & (metal_spot["trade_date"].dt.date <= train_end)
    ][["trade_date", "metal", "price_cny_per_tonne", "source", "raw_symbol"]]
    train_features = (
        metal_features[metal_features["trade_date"].dt.date <= train_end].copy()
        if not metal_features.empty
        else pd.DataFrame()
    )
    forecast = generate_forecasts(
        train_spot,
        train_features,
        forecast_days=horizon_days,
        model_version=f"{model_version}-backtest",
    )
    actual = metal_spot[
        (metal_spot["trade_date"].dt.date > train_end) & (metal_spot["trade_date"].dt.date <= eval_end)
    ][["trade_date", "price_cny_per_tonne"]].rename(
        columns={"trade_date": "forecast_date", "price_cny_per_tonne": "actual_price_cny_per_tonne"}
    )
    forecast["forecast_date"] = pd.to_datetime(forecast["forecast_date"])
    actual["forecast_date"] = pd.to_datetime(actual["forecast_date"])
    return actual.merge(
        forecast[["forecast_date", "predicted_price_cny_per_tonne"]],
        on="forecast_date",
        how="inner",
    )


def random_date_range(min_date: date, max_date: date) -> tuple[date, date]:
    total_days = (max_date - min_date).days
    if total_days <= 1:
        return min_date, max_date
    length = random.randint(max(1, min(20, total_days)), max(1, min(180, total_days)))
    start_offset = random.randint(0, max(0, total_days - length))
    start = min_date + timedelta(days=start_offset)
    end = start + timedelta(days=length)
    return start, min(end, max_date)


def main() -> None:
    st.set_page_config(page_title="国内原材料采购价格预测", layout="wide")
    if "show_landing" not in st.session_state:
        st.session_state["show_landing"] = True
    if st.query_params.get("view") == "dashboard":
        st.session_state["show_landing"] = False
    if st.session_state["show_landing"]:
        render_landing_page()
        return

    inject_style()

    config = load_config()
    display = {**DEFAULT_DISPLAY, **{metal: name for metal, name in config.excel.columns.items()}}
    conn = connect(config.database_path, read_only=True)

    try:
        spot = load_spot_prices(conn)
        market_features = load_market_features(conn)
        forecasts = load_latest_forecasts(conn)
        monthly_forecasts = load_latest_monthly_forecasts(conn)
        driver_contributions = load_latest_forecast_driver_contributions(conn)
        runs = load_update_runs(conn, limit=5)
    except sqlite3.DatabaseError as exc:
        st.error(f"数据库读取失败：{exc}")
        st.stop()
    finally:
        conn.close()
    monthly_analysis, factor_catalog, verified_events = load_dashboard_analysis_data()
    daily_news = load_news_cache(NEWS_CACHE)

    if spot.empty:
        st.warning("暂无价格数据。请先运行 `python update_hybrid_price_database.py` 更新数据库。")
        st.stop()

    spot["trade_date"] = pd.to_datetime(spot["trade_date"])
    if not market_features.empty:
        market_features["trade_date"] = pd.to_datetime(market_features["trade_date"])

    latest_run = runs.iloc[0] if not runs.empty else None
    latest_date = spot["trade_date"].max().strftime("%Y-%m-%d")
    latest_generated = ""
    latest_model_version = config.model_version
    if not forecasts.empty:
        latest_generated = pd.to_datetime(forecasts["generated_at"]).max().strftime("%Y-%m-%d %H:%M")
        latest_model_version = str(forecasts.sort_values("generated_at").iloc[-1]["model_version"])

    metals = sorted(
        spot["metal"].dropna().unique().tolist(),
        key=lambda item: list(display).index(item) if item in display else 999,
    )
    colors = {metal: METAL_COLORS.get(metal, PALETTE[idx % len(PALETTE)]) for idx, metal in enumerate(metals)}
    with st.sidebar:
        st.markdown('<div class="brand">Metal<span>Pulse</span></div>', unsafe_allow_html=True)
        if st.button("产品首页", width="stretch"):
            st.session_state["show_landing"] = True
            st.query_params.clear()
            st.rerun()
        page = st.radio(
            "导航",
            NAVIGATION_PAGES,
            label_visibility="collapsed",
        )
        st.markdown("---")
        selected_metal = st.selectbox("跟踪品种", metals, format_func=lambda item: display.get(item, item))
        st.markdown(
            '<div class="sidebar-note"><b>数据口径</b><br>所有现货、日度预测及月度预测均为不含税价。<br><br><b>同步状态</b><br>长江有色更新后自动重训。</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="dashboard-header">
            <h1>国内原材料采购价格预测</h1>
            <div class="dashboard-status">
                数据截至 {html.escape(latest_date)}<br>
                预测生成于 {html.escape(latest_generated or "暂无")}　模型 {html.escape(latest_model_version)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page_description = PAGE_DESCRIPTIONS.get(page)
    if page_description:
        st.markdown(
            f'<p class="page-description">{html.escape(page_description)}</p>',
            unsafe_allow_html=True,
        )

    if latest_run is not None and latest_run["status"] in {"failed", "error"}:
        st.warning(f"最近一次更新失败：{latest_run.get('error_summary', '')}")

    if page == "首页概览":
        render_home_overview(
            spot,
            metals,
            display,
            colors,
            verified_events,
            daily_news,
        )

    elif page == "预测总览":
        metal = selected_metal
        metal_spot = spot[spot["metal"] == metal].sort_values("trade_date")
        metal_forecast = forecasts[forecasts["metal"] == metal].sort_values("forecast_date")
        metal_monthly_forecast = monthly_forecasts[monthly_forecasts["metal"] == metal].sort_values("forecast_month")
        if metal_spot.empty:
            st.info(f"暂无{display.get(metal, metal)}价格数据。")
            return

        latest_price = metal_spot.iloc[-1]["price_cny_per_tonne"]
        change_5d = metal_spot["price_cny_per_tonne"].pct_change(5).iloc[-1]
        unit = price_unit(metal)
        render_forecast_summary_cards(forecasts, metals, display)

        recent_history = metal_spot[
            metal_spot["trade_date"]
            >= metal_spot["trade_date"].max() - pd.Timedelta(days=31)
        ]
        recent_events = events_in_range(
            verified_events,
            metal,
            recent_history["trade_date"].min(),
            recent_history["trade_date"].max(),
        )
        fig = build_trend_figure(
            metal_spot,
            metal_forecast,
            colors[metal],
            recent_events,
            unit,
        )
        chart_column, model_column = st.columns([3, 1])
        with chart_column:
            st.markdown('<div class="section-title">价格趋势 · ' + display.get(metal, metal) + '</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch")
            st.markdown(
                '<div class="forecast-note"><b>读图方式：</b>实线为已公布的不含税现货均价；蓝线为未来 30 天每天的预测价格；浅蓝阴影表示每日预测下限与预测上限之间的范围。</div>',
                unsafe_allow_html=True,
            )
        with model_column:
            st.markdown('<div class="section-title">预测摘要</div>', unsafe_allow_html=True)
            model_column.metric("最新现货均价", f"{latest_price:,.0f} {unit}")
            model_column.metric("近5日变化", f"{change_5d * 100:.2f}%" if pd.notna(change_5d) else "暂无")
            if not metal_forecast.empty:
                forecast_end = metal_forecast.iloc[-1]["predicted_price_cny_per_tonne"]
                model_column.metric("30日预测末值", f"{forecast_end:,.0f} {unit}")
                model_column.metric("预测期变化", f"{forecast_end / latest_price - 1:.2%}")
        render_verified_event_details(recent_events)
        render_monthly_forecast(metal_monthly_forecast, metal_spot, colors[metal], unit)

    elif page == "影响分析":
        render_impact_analysis(
            selected_metal,
            monthly_analysis,
            factor_catalog,
            colors.get(selected_metal, "#A32035"),
            "impact",
        )
        render_forecast_driver_analysis(
            selected_metal,
            monthly_analysis,
            factor_catalog,
            driver_contributions,
        )
        render_terminal_demand(
            selected_metal,
            monthly_analysis,
            colors.get(selected_metal, "#A32035"),
            "impact",
        )
        render_full_factor_strength_overview(
            selected_metal,
            factor_catalog,
            "impact",
        )

    elif page == "模型评估":
        render_backtest(
            spot,
            market_features,
            metals,
            display,
            config.model_version,
        )

    elif page == "模型说明":
        render_model_formula()

    elif page == "报告中心":
        render_report_center(forecasts, monthly_forecasts, runs, display)

    else:
        if runs.empty:
            st.info("暂无更新记录。")
        else:
            st.dataframe(runs, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
