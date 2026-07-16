from __future__ import annotations

import random
import sqlite3
import base64
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

from domestic_prices.config import load_config
from domestic_prices.db import (
    connect,
    initialize,
    load_latest_monthly_forecasts,
    load_latest_forecasts,
    load_market_features,
    load_spot_prices,
    load_update_runs,
)
from domestic_prices.model import build_feature_snapshot


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
PALETTE = ["#E77A3D", "#2BB9A8", "#A66BE7", "#F0A83B", "#D8568B"]
METAL_COLORS = {
    "copper_1": "#E77A3D",
    "aluminum_a00": "#2BB9A8",
    "silver_1": "#A66BE7",
    "aluminum_adc12": "#F0A83B",
    "aluminum_zld104": "#D8568B",
    "lithium_carbonate": "#4C78A8",
}
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
FACTOR_MATERIAL = {
    "copper_1": "1#铜",
    "aluminum_a00": "A00铝",
    "silver_1": "1#白银",
    "aluminum_adc12": "ADC12",
    "aluminum_zld104": "ZLD104",
    "lithium_carbonate": "碳酸锂",
}


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


@st.cache_data
def load_factor_coefficients() -> pd.DataFrame:
    if not FACTOR_COEFFICIENTS.exists():
        return pd.DataFrame()
    coefficients = pd.read_csv(FACTOR_COEFFICIENTS, encoding="utf-8-sig")
    lithium_path = Path(__file__).resolve().parent / "lithium_carbonate_prediction_outputs" / "lithium_impact_regression_screening.csv"
    if lithium_path.exists():
        lithium = pd.read_csv(lithium_path, encoding="utf-8-sig")
        lithium = lithium.rename(
            columns={
                "标准化系数": "标准化系数",
                "影响强度_绝对值": "影响强度_绝对值",
                "方向": "回归方向",
            }
        )
        lithium["模型版本"] = "碳酸锂影响变量回归"
        lithium["强弱排名"] = lithium.groupby("品种")["影响强度_绝对值"].rank(method="first", ascending=False).astype(int)
        columns = ["品种", "模型版本", "目标变量", "变量", "标准化系数", "影响强度_绝对值", "p值", "显著性", "回归方向", "强弱排名"]
        coefficients = pd.concat([coefficients, lithium[columns]], ignore_index=True)
    return coefficients


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
        }}
        .comparison-note {{
            background: #f0f6ff;
            color: #3e5f8c;
            border: 1px solid #dce9ff;
            border-radius: 12px;
            padding: 10px 14px;
            margin: 8px 0 12px;
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
            min-height: 126px;
            padding: 14px;
        }}
        .future-card-title {{ color: #4f5f73; font-size: 13px; font-weight: 700; }}
        .future-card-value {{ color: var(--ink); font-size: 20px; font-weight: 760; letter-spacing: -.03em; margin-top: 12px; }}
        .future-card-meta {{ color: var(--muted); font-size: 11px; line-height: 1.5; margin-top: 5px; }}
        .future-card-trend {{ font-size: 12px; font-weight: 750; margin-top: 8px; }}
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
                <p>以统一的不含税口径，连接现货价格、预测区间与历史回测，为原材料采购提供可追溯的日度判断依据。</p>
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
                    <div class="detail-point"><strong>历史表现可复核</strong><span>保留历史回测，以相同口径检查模型的实际表现。</span></div>
                </div>
            </div>
        </section>
        <section class="landing-section">
            <h2>为采购节奏设计的价格工作台。</h2>
            <p>从晨间判断到月度计划，关键数据保持在一个清晰、稳定的分析入口。</p>
            <div class="landing-capabilities">
                <div class="capability-main"><h3>价格预测总览</h3><p>将已公布现货均价、未来预测曲线和预测区间组合呈现，快速识别趋势变化。</p></div>
                <div class="capability-stack">
                    <div class="capability-small"><h3>历史回测</h3><p>用真实价格检查预测表现，让模型结果可被持续验证。</p></div>
                    <div class="capability-small"><h3>模型与口径</h3><p>明确数据来源、价格口径和模型逻辑，方便内部协作与复盘。</p></div>
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


def build_trend_figure(metal_spot: pd.DataFrame, metal_forecast: pd.DataFrame, color: str) -> go.Figure:
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
            hovertemplate="实际不含税均价<br>%{x|%Y-%m-%d}<br><b>%{y:,.0f} 元/吨</b><extra></extra>",
        )
    )
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
            name="80%预测区间（P10–P90）",
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
                "预测不含税均价<br>%{x|%Y-%m-%d}<br><b>%{y:,.0f} 元/吨</b>"
                "<br>80%预测区间：%{customdata[0]:,.0f}–%{customdata[1]:,.0f} 元/吨<extra></extra>"
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
        height=460,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b"},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title="元/吨（不含税）",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(76, 106, 146, 0.10)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(76, 106, 146, 0.10)")
    return fig


def widened_forecast_band(forecast: pd.DataFrame) -> pd.DataFrame:
    band = forecast.copy()
    band["display_upper_bound"] = band["upper_bound"].astype(float)
    band["display_lower_bound"] = band["lower_bound"].astype(float)
    return band


def render_monthly_forecast(monthly_forecast: pd.DataFrame, color: str) -> None:
    st.markdown('<div class="section-title">月度均价预测</div>', unsafe_allow_html=True)
    if monthly_forecast.empty:
        st.info("暂无月度均价预测。请先运行数据更新任务。")
        return
    data = monthly_forecast.sort_values("forecast_month").copy()
    fig = go.Figure(
        go.Scatter(
            x=data["forecast_month"].dt.strftime("%Y-%m"),
            y=data["predicted_price_cny_per_tonne"],
            mode="lines+markers",
            marker_color=color,
            line={"color": color, "width": 3},
            marker={"size": 8},
            name="预测月均价",
            hovertemplate="预测不含税月均价<br>%{x}<br><b>%{y:,.0f} 元/吨</b><extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b"},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title="元/吨（不含税）",
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(76, 106, 146, 0.10)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        data.assign(
            预测月份=data["forecast_month"].dt.strftime("%Y-%m"),
            预测月均价=data["predicted_price_cny_per_tonne"].map(lambda value: f"{value:,.0f} 元/吨"),
        )[["预测月份", "预测月均价"]],
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
    factors = factors.sort_values("强弱排名").head(5).sort_values("影响强度_绝对值")
    colors = np.where(factors["回归方向"].eq("正向"), "#A8C5B9", "#C98A91")
    fig = go.Figure(
        go.Bar(
            x=factors["标准化系数"],
            y=factors["变量"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.3f}" for value in factors["标准化系数"]],
            textposition="outside",
            hovertemplate=(
                "%{y}<br>标准化系数：%{x:+.3f}<br>影响强度：%{customdata:.3f}<extra></extra>"
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
        xaxis_title="标准化回归系数",
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
    st.plotly_chart(fig, width="stretch")


def render_market_cards(spot: pd.DataFrame, metals: list[str], display: dict[str, str]) -> None:
    cards = st.columns(len(metals))
    for column, metal in zip(cards, metals):
        metal_spot = spot[spot["metal"] == metal].sort_values("trade_date")
        latest_price = metal_spot.iloc[-1]["price_cny_per_tonne"]
        change_5d = metal_spot["price_cny_per_tonne"].pct_change(5).iloc[-1]
        column.metric(
            display.get(metal, metal),
            f"{latest_price:,.0f}",
            f"{change_5d * 100:.2f}%" if pd.notna(change_5d) else "暂无",
            delta_color="inverse",
        )


def render_home_overview(
    spot: pd.DataFrame,
    forecasts: pd.DataFrame,
    metals: list[str],
    display: dict[str, str],
    colors: dict[str, str],
) -> None:
    st.markdown('<div class="section-title">市场概览</div>', unsafe_allow_html=True)
    render_market_cards(spot, metals, display)

    trend_title, trend_window = st.columns([3, 2])
    trend_title.markdown('<div class="section-title">综合价格趋势</div>', unsafe_allow_html=True)
    selected_window = trend_window.radio(
        "趋势区间",
        ["近7日", "近30日", "近60日"],
        index=2,
        horizontal=True,
        label_visibility="collapsed",
        key="overview_trend_window",
    )
    lookback_days = {"近7日": 7, "近30日": 30, "近60日": 60}[selected_window]
    fig = go.Figure()
    normalized_values: list[float] = []
    for metal in metals:
        series = spot[spot["metal"] == metal].sort_values("trade_date").tail(lookback_days).copy()
        base_price = float(series.iloc[0]["price_cny_per_tonne"])
        series["price_index"] = series["price_cny_per_tonne"] / base_price * 100
        normalized_values.extend(series["price_index"].tolist())
        fig.add_trace(
            go.Scatter(
                x=series["trade_date"],
                y=series["price_index"],
                mode="lines",
                name=display.get(metal, metal),
                line={"color": colors[metal], "width": 2.3},
                customdata=np.column_stack([series["price_cny_per_tonne"]]),
                hovertemplate="%{x|%Y-%m-%d}<br>价格指数：%{y:.2f}<br>实际价格：%{customdata[0]:,.0f} 元/吨<extra></extra>",
            )
        )
    y_padding = max((max(normalized_values) - min(normalized_values)) * 0.15, 0.5)
    fig.update_layout(
        height=390,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b", "size": 12},
        margin={"l": 45, "r": 20, "t": 22, "b": 25},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(76, 106, 146, 0.10)")
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(76, 106, 146, 0.10)",
        title="价格指数（首日=100）",
        range=[min(normalized_values) - y_padding, max(normalized_values) + y_padding],
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("为便于比较不同价格量级的材料，曲线已按所选区间首日价格标准化为 100；悬停可查看实际元/吨价格。")

    forecast_summaries: list[tuple[str, float, float, float]] = []
    for metal in metals:
        forecast = forecasts[forecasts["metal"] == metal].sort_values("forecast_date")
        if len(forecast) >= 2:
            first_price = float(forecast.iloc[0]["predicted_price_cny_per_tonne"])
            last_price = float(forecast.iloc[-1]["predicted_price_cny_per_tonne"])
            forecast_summaries.append((display.get(metal, metal), first_price, last_price, last_price / first_price - 1))

    st.markdown('<div class="section-title">未来30天走势</div>', unsafe_allow_html=True)
    future_cards = st.columns(len(forecast_summaries))
    for column, (name, first_price, last_price, change) in zip(future_cards, forecast_summaries):
        if abs(change) < 0.005:
            direction, direction_class = "平稳", "steady"
        elif change > 0:
            direction, direction_class = "上行", "up"
        else:
            direction, direction_class = "下行", "down"
        column.markdown(
            f'''<div class="future-card">
                <div class="future-card-title">{name}</div>
                <div class="future-card-value">{last_price:,.0f} 元/吨</div>
                <div class="future-card-meta">预测起点：{first_price:,.0f} 元/吨<br>30日变化：{change:+.2%}</div>
                <div class="future-card-trend {direction_class}">{direction}</div>
            </div>''',
            unsafe_allow_html=True,
        )


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
            {"报告": "影响因素分析", "内容": "标准化回归系数与 Top 5 驱动因素", "状态": "可用" if not load_factor_coefficients().empty else "暂无"},
            {"报告": "数据更新记录", "内容": "最近数据更新与模型重训状态", "状态": "可用" if not runs.empty else "暂无"},
        ]
    )
    with catalog_column:
        st.dataframe(reports, width="stretch", hide_index=True)
        st.markdown('<div class="section-title">预测数据导出</div>', unsafe_allow_html=True)
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
    with status_column:
        st.metric("日度预测", "已生成" if not forecasts.empty else "暂无")
        st.metric("月度预测", "已生成" if not monthly_forecasts.empty else "暂无")
        st.metric("最近更新", str(runs.iloc[0]["status"]) if not runs.empty else "暂无")


def render_range_stats(metal: str, metal_spot: pd.DataFrame, color: str) -> None:
    with st.expander("区间统计", expanded=False):
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
        stat_cols[0].metric("区间均价", f"{values.mean():,.2f}")
        stat_cols[1].metric("最高价", f"{values.max():,.2f}")
        stat_cols[2].metric("最低价", f"{values.min():,.2f}")
        stat_cols[3].metric("有效天数", f"{len(selected)}")
        stat_cols2 = st.columns(4)
        stat_cols2[0].metric("区间首日价", f"{first_price:,.2f}")
        stat_cols2[1].metric("区间末日价", f"{last_price:,.2f}")
        stat_cols2[2].metric("区间涨跌幅", f"{pct_change:.2f}%")
        stat_cols2[3].metric("价格极差", f"{price_range:,.2f}")
        stat_cols3 = st.columns(4)
        stat_cols3[0].metric("日度波动率", f"{daily_vol:.2f}%" if pd.notna(daily_vol) else "暂无")
        stat_cols3[1].metric("中位数", f"{values.median():,.2f}")
        stat_cols3[2].metric("标准差", f"{values.std():,.2f}")
        stat_cols3[3].metric("变异系数", f"{values.std() / values.mean() * 100:.2f}%")
        st.caption("变异系数 = 标准差 ÷ 区间均价，用于衡量价格相对波动程度；数值越大，说明该区间内价格波动越明显。")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=selected["trade_date"],
                y=selected["price_cny_per_tonne"],
                mode="lines",
                name="所选区间",
                line={"color": color, "width": 2},
            )
        )
        fig.add_hline(y=values.mean(), line_dash="dash", line_color="#4A4A4A", annotation_text="均价")
        fig.update_layout(
            height=260,
            template="plotly_white",
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="#ffffff",
            font={"color": "#64748b"},
            margin={"l": 20, "r": 20, "t": 10, "b": 20},
            yaxis_title="元/吨",
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
        '<div class="comparison-note">指标说明：平均绝对误差表示预测价与真实价平均相差多少元/吨；MAPE 是平均误差率，越低越好；RMSE 会更重视较大的预测错误；平均偏差为正表示整体预测偏高，为负表示整体预测偏低。</div>',
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
        yaxis_title="元/吨",
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
        yaxis_title="元/吨",
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
        yaxis_title="预测 - 真实",
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


@st.cache_data(show_spinner=False)
def load_monthly_history_predictions() -> pd.DataFrame:
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
    colors: dict[str, str],
    model_version: str,
) -> None:
    del spot, market_features, model_version
    st.subheader("历史月均价与模型预测")
    history = load_monthly_history_predictions()
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
    st.caption("模型及训练区间固定为该材料最早至最晚的完整样本；选择月份仅筛选展示范围，不会重新训练。")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=selected["month"],
            y=selected["actual_monthly_price"],
            mode="lines+markers",
            name="历史月度均价",
            line={"color": colors.get(selected_metal, "#2E78F6"), "width": 2.6},
            marker={"size": 5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=selected["month"],
            y=selected["predicted_monthly_price"],
            mode="lines+markers",
            name="模型预测值",
            line={"color": "#c9372c", "width": 2, "dash": "dash"},
            marker={"size": 4},
        )
    )
    fig.update_layout(
        height=440,
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#ffffff",
        font={"color": "#64748b"},
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis_title="元/吨",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(fig, width="stretch")

    table = selected.assign(
        月份=selected["month"].dt.strftime("%Y-%m"),
        历史月度均价=selected["actual_monthly_price"],
        模型预测值=selected["predicted_monthly_price"],
        预测误差=error,
        误差率=error / selected["actual_monthly_price"] * 100,
    )[["月份", "历史月度均价", "模型预测值", "预测误差", "误差率"]]
    table["误差率"] = table["误差率"].map(lambda value: f"{value:.2f}%")
    st.markdown('<div class="section-title">月度价格明细</div>', unsafe_allow_html=True)
    st.dataframe(table, width="stretch", hide_index=True)


def render_model_formula() -> None:
    st.subheader("预测模型与公式")
    st.markdown(
        """
        当前看板使用 `daily-hybrid-variable-anchor-v1`。它不是单一模型：先由多个日度模型竞争，
        再在有可用月度预测时进行月均价锚定，最后按每个品种、每个预测步长的历史回测误差择优展示。
        """
    )

    st.markdown('<div class="section-title">1. 日度候选模型</div>', unsafe_allow_html=True)
    st.markdown(
        """
        对每个品种分别回测并比较：最近价（Naive）、不同窗口的对数漂移、Ridge 直接多步预测、
        ARIMA 对数价格预测，以及中位数集成。候选模型的权重来自滚动历史验证，而不是固定指定。
        """
    )

    st.markdown('<div class="section-title">2. 月均价锚定</div>', unsafe_allow_html=True)
    st.latex(r"P^{hybrid}_{t+h}=P^{daily}_{t+h}+w_h\,(T_m-\overline{P}^{daily}_m)")
    st.markdown(
        """
        `T_m` 是月度变量模型给出的该月均价目标，`P_daily_month` 是日度预测在该月的均值。
        锚定权重 `w_h` 随预测步长由 0 平滑升至最多 0.75，使近端预测保留日度模型信息、远端预测与月均价保持一致。
        """
    )

    st.markdown('<div class="section-title">3. 自适应择优</div>', unsafe_allow_html=True)
    st.markdown(
        """
        对每个“品种 × 预测天数”，只有当月度锚定模型在历史回测中 MAE 低于日度集成时，
        看板才采用锚定结果；否则显示日度集成结果。因此不同品种、不同日期可能来自不同分支。
        """
    )

    st.markdown('<div class="section-title">4. 预测区间</div>', unsafe_allow_html=True)
    st.markdown(
        """
        阴影是 P10–P90，即模型给出的 80% 预测区间；并且现已按原始上下界绘制，不再做视觉放宽。
        鼠标停在红色预测点上会显示该日的预测均价及完整区间，单位统一为元/吨。
        """
    )


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
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Metal intelligence · tax-exclusive basis</div>
            <h1>国内原材料采购价格预测</h1>
            <p>基于国内公开日度价格与碳酸锂期货结算价，统一按不含税口径跟踪 1#铜、A00铝、1#白银、铝ADC12、ZLD104 和碳酸锂的历史趋势与未来预测。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    config = load_config()
    display = {**DEFAULT_DISPLAY, **{metal: name for metal, name in config.excel.columns.items()}}
    conn = connect(config.database_path)
    initialize(conn)

    try:
        spot = load_spot_prices(conn)
        market_features = load_market_features(conn)
        forecasts = load_latest_forecasts(conn)
        monthly_forecasts = load_latest_monthly_forecasts(conn)
        runs = load_update_runs(conn, limit=5)
    except sqlite3.DatabaseError as exc:
        st.error(f"数据库读取失败：{exc}")
        st.stop()
    finally:
        conn.close()

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

    st.caption(f"数据截至 {latest_date} · 预测生成于 {latest_generated or '暂无'} · 模型 {latest_model_version}")

    if latest_run is not None and latest_run["status"] in {"failed", "error"}:
        st.warning(f"最近一次更新失败：{latest_run.get('error_summary', '')}")

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
        page = st.radio("导航", ["首页概览", "预测总览", "历史回测", "模型与口径", "报告中心", "更新记录"], label_visibility="collapsed")
        st.markdown("---")
        selected_metal = st.selectbox("跟踪品种", metals, format_func=lambda item: display.get(item, item))
        st.markdown(
            '<div class="sidebar-note"><b>数据口径</b><br>所有现货、日度预测及月度预测均为不含税价。<br><br><b>同步状态</b><br>长江有色更新后自动重训。</div>',
            unsafe_allow_html=True,
        )

    if page == "首页概览":
        render_home_overview(spot, forecasts, metals, display, colors)

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
        render_market_cards(spot, metals, display)

        fig = build_trend_figure(metal_spot, metal_forecast, colors[metal])
        chart_column, model_column = st.columns([3, 1])
        with chart_column:
            st.markdown('<div class="section-title">价格趋势 · ' + display.get(metal, metal) + '</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch")
            st.markdown(
                '<div class="forecast-note"><b>读图方式：</b>实线为已公布的不含税现货均价，蓝线为未来 30 天预测，浅蓝阴影为 P10–P90 的 80% 预测区间。</div>',
                unsafe_allow_html=True,
            )
        with model_column:
            st.markdown('<div class="section-title">模型评估</div>', unsafe_allow_html=True)
            model_column.metric("最新现货均价", f"{latest_price:,.0f} 元/吨")
            model_column.metric("近5日变化", f"{change_5d * 100:.2f}%" if pd.notna(change_5d) else "暂无")
            if not metal_forecast.empty:
                forecast_end = metal_forecast.iloc[-1]["predicted_price_cny_per_tonne"]
                model_column.metric("30日预测末值", f"{forecast_end:,.0f} 元/吨")
                model_column.metric("预测期变化", f"{forecast_end / latest_price - 1:.2%}")
        render_monthly_forecast(metal_monthly_forecast, colors[metal])
        render_top_impact_factors(metal)
        render_range_stats(metal, metal_spot, colors[metal])

        feature_snapshot = build_feature_snapshot(spot, market_features)
        feature_snapshot = feature_snapshot[feature_snapshot["metal"] == metal].copy()
        feature_snapshot["latest_value"] = pd.to_numeric(feature_snapshot["latest_value"], errors="coerce")
        feature_snapshot = feature_snapshot[
            feature_snapshot["latest_value"].notna() & (feature_snapshot["latest_value"].abs() > 1e-12)
        ]
        feature_snapshot["factor"] = feature_snapshot["factor"].map(FACTOR_CN).fillna(feature_snapshot["factor"])
        percentage_factors = {"1日涨跌幅", "5日涨跌幅", "偏离5日均线", "偏离10日均线", "偏离20日均线", "10日波动率", "20日波动率"}
        feature_snapshot["latest_value"] = feature_snapshot.apply(
            lambda row: f"{row['latest_value']:.2%}"
            if row["factor"] in percentage_factors
            else f"{row['latest_value']:,.2f} 元/吨",
            axis=1,
        )
        st.dataframe(
            feature_snapshot.rename(columns={"factor": "因子", "latest_value": "最新值"})[["因子", "最新值"]],
            width="stretch",
            hide_index=True,
        )

    elif page == "历史回测":
        render_backtest(spot, market_features, metals, display, colors, config.model_version)

    elif page == "模型与口径":
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
