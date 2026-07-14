from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PRICE_FILE = Path("ccmn_changjiang_avg_prices.csv")
CACHE_DIR = Path("data_cache_v3")
OUTPUT_FILE = Path("domestic_material_factor_model_v1.xlsx")

TARGET_PRICE_COLUMNS = {
    "1#铜": "1#铜",
    "A00铝": "A00铝",
    "1#白银": "1#白银",
    "ADC12": "铝合金ADC12",
    "ZLD104": "铸造铝合金锭(ZLD104)",
}

MATERIAL_ALIASES = {
    "1#铜": "copper_1",
    "A00铝": "aluminum_a00",
    "1#白银": "silver_1",
    "ADC12": "aluminum_adc12",
    "ZLD104": "aluminum_zld104",
}


def month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def load_monthly_prices() -> pd.DataFrame:
    prices = read_csv(PRICE_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["月份"] = month_start(prices["date"])
    monthly = prices.groupby("月份", as_index=False)[list(TARGET_PRICE_COLUMNS.values())].mean()
    out = pd.DataFrame({"月份": monthly["月份"]})
    for material, source_col in TARGET_PRICE_COLUMNS.items():
        out[f"{material}_月均价"] = monthly[source_col]
        out[f"{material}_价格月环比"] = monthly[source_col].pct_change(fill_method=None) * 100
    out["ADC12_A00价差"] = out["ADC12_月均价"] - out["A00铝_月均价"]
    out["ADC12_A00价差_变化"] = out["ADC12_A00价差"].diff()
    out["ADC12_A00价差_环比"] = out["ADC12_A00价差"].pct_change(fill_method=None) * 100
    out["ADC12_A00价差_滞后1期"] = out["ADC12_A00价差"].shift(1)
    out["ADC12_A00价差_滞后1期变化"] = out["ADC12_A00价差_变化"].shift(1)
    out["ZLD104_A00价差"] = out["ZLD104_月均价"] - out["A00铝_月均价"]
    out["ZLD104_A00价差_变化"] = out["ZLD104_A00价差"].diff()
    out["ZLD104_A00价差_环比"] = out["ZLD104_A00价差"].pct_change(fill_method=None) * 100
    out["ZLD104_A00价差_滞后1期"] = out["ZLD104_A00价差"].shift(1)
    out["ZLD104_A00价差_滞后1期变化"] = out["ZLD104_A00价差_变化"].shift(1)
    return out


def load_domestic_features() -> pd.DataFrame:
    prices = load_monthly_prices()
    out = prices.copy()

    shfe_warrant = read_csv(CACHE_DIR / "shfe_warrant_monthly.csv", parse_dates=["月份"])
    shfe_futures = read_csv(CACHE_DIR / "shfe_futures_monthly.csv", parse_dates=["月份"])
    nbs_monthly = read_csv(CACHE_DIR / "nbs_monthly_indicators.csv", parse_dates=["月份"])
    gasgoo = read_csv(CACHE_DIR / "gasgoo_auto_top50_monthly.csv", parse_dates=["月份"])
    nbs_annual = read_csv(CACHE_DIR / "nbs_annual_indicators.csv")

    out = out.merge(shfe_warrant.drop(columns=["SHFE仓单取数日"], errors="ignore"), on="月份", how="left")
    out = out.merge(shfe_futures, on="月份", how="left")

    nbs_wide = nbs_monthly.pivot_table(index="月份", columns="指标", values="数值", aggfunc="last").reset_index()
    out = out.merge(nbs_wide, on="月份", how="left")
    out = out.merge(gasgoo, on="月份", how="left")
    domestic_macro_path = CACHE_DIR / "domestic_macro_monthly.csv"
    if domestic_macro_path.exists():
        domestic_macro = read_csv(domestic_macro_path, parse_dates=["月份"])
        out = out.merge(domestic_macro, on="月份", how="left")
    policy_uncertainty_path = CACHE_DIR / "policy_uncertainty_monthly.csv"
    if policy_uncertainty_path.exists():
        policy_uncertainty = read_csv(policy_uncertainty_path, parse_dates=["月份"])
        out = out.merge(policy_uncertainty, on="月份", how="left")
    recycling_import_path = CACHE_DIR / "recycling_import_monthly.csv"
    if recycling_import_path.exists():
        recycling_import = read_csv(recycling_import_path, parse_dates=["月份"])
        out = out.merge(recycling_import, on="月份", how="left")

    annual_wide = nbs_annual.pivot_table(index="年份", columns="指标", values="数值", aggfunc="last").reset_index()
    monthly_year = out[["月份"]].copy()
    monthly_year["年份"] = monthly_year["月份"].dt.year
    out = out.merge(monthly_year.merge(annual_wide, on="年份", how="left").drop(columns=["年份"]), on="月份", how="left")

    return out


def add_feature_changes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "月份" or col.endswith("_月均价") or col.endswith("_价格月环比"):
            continue
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        if col.endswith("_变化率"):
            continue
        out[f"{col}_变化"] = out[col].diff()
        out[f"{col}_环比"] = out[col].pct_change(fill_method=None) * 100
    return out


def factor_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(
        material: str,
        category: str,
        name: str,
        direction: str,
        frequency: str,
        lag: str,
        source: str,
        core: str,
        quantifiable: str,
        local_field: str = "",
        availability: str = "待补充",
        modeling_note: str = "",
    ) -> None:
        rows.append(
            {
                "品种": material,
                "品种代码": MATERIAL_ALIASES[material],
                "变量类别": category,
                "变量名": name,
                "预期方向": direction,
                "建议频率": frequency,
                "建议滞后期": lag,
                "建议数据来源": source,
                "是否核心变量": core,
                "是否可量化": quantifiable,
                "本地可用字段": local_field,
                "本地可得性": availability,
                "建模处理建议": modeling_note,
            }
        )

    # 1#铜
    add("1#铜", "供应端", "国内电解铜产量", "供应增加通常压低价格", "月度", "1-3个月", "国家统计局/安泰科/SMM/Wind", "是", "是")
    add("1#铜", "供应端", "冶炼厂开工率", "开工率上升通常压低价格", "周度/月度", "0-2个月", "SMM/Mysteel/百川/Wind", "是", "是")
    add("1#铜", "供应端", "铜精矿进口量", "进口增加通常缓解供应压力", "月度/年度", "1-3个月", "海关总署/国家统计局", "是", "是", "铜矿砂及其精矿进口数量", "部分可用", "本地为年度数据，月度模型中仅作低频背景变量")
    add("1#铜", "库存端", "SHFE铜库存", "库存上升通常压低价格", "周度/月度", "0-1个月", "上海期货交易所", "是", "是", "SHFE铜仓单库存", "可用", "可用仓单库存作为交易所库存代理")
    add("1#铜", "库存端", "社会库存", "库存上升通常压低价格", "周度/月度", "0-1个月", "Mysteel/SMM/Wind", "是", "是")
    add("1#铜", "成本端", "TC/RC加工费", "加工费上升代表矿端偏宽松，通常压低铜价", "周度/月度", "0-2个月", "SMM/Fastmarkets/Wind", "是", "是")
    add("1#铜", "成本端", "铜矿进口成本", "成本上升通常支撑价格", "月度", "0-2个月", "海关总署/国家统计局", "是", "是", "铜矿砂及其精矿进口金额", "部分可用", "本地为年度金额，可与进口数量计算年度均价")
    add("1#铜", "需求端", "电网投资完成额", "需求增加通常推升价格", "月度", "1-3个月", "国家能源局/国家电网", "是", "是")
    add("1#铜", "需求端", "电线电缆产量", "需求增加通常推升价格", "月度", "0-2个月", "国家统计局/Wind", "是", "是", "电线电缆光缆及电工器材制造PPI; 光缆产量当期值", "可用替代", "本地暂无电线电缆产量，用行业PPI和光缆产量代理")
    add("1#铜", "需求端", "房地产施工面积", "需求增加通常推升价格", "月度", "1-3个月", "国家统计局", "是", "是")
    add("1#铜", "回收端", "废铜回收量", "回收供给增加通常压低价格", "月度", "0-2个月", "SMM/Mysteel/再生资源协会/Wind", "否", "是")
    add("1#铜", "回收端", "精废价差", "价差扩大通常提升废铜替代，压制精铜", "日度/周度", "0-1个月", "SMM/Mysteel/Wind", "是", "是")
    add("1#铜", "事件冲击", "限产政策dummy", "限产通常推升价格", "日度/月度", "0-1个月", "政策公告/新闻事件库", "否", "是")
    add("1#铜", "事件冲击", "重大矿山罢工事件", "供应扰动通常推升价格", "日度/月度", "0-1个月", "新闻事件库", "否", "是")

    # A00铝
    add("A00铝", "供应端", "电解铝产量", "供应增加通常压低价格", "月度", "0-2个月", "国家统计局/IAI中国口径/SMM/Wind", "是", "是")
    add("A00铝", "供应端", "运行产能利用率", "利用率上升通常压低价格", "月度", "0-2个月", "SMM/百川/Wind", "是", "是")
    add("A00铝", "库存端", "SHFE铝库存", "库存上升通常压低价格", "周度/月度", "0-1个月", "上海期货交易所", "是", "是", "SHFE铝仓单库存", "可用", "可用仓单库存作为交易所库存代理")
    add("A00铝", "库存端", "社会库存", "库存上升通常压低价格", "周度/月度", "0-1个月", "Mysteel/SMM/Wind", "是", "是")
    add("A00铝", "成本端", "工业电价", "电价上升通常支撑价格", "月度", "0-2个月", "国家发改委/地方电价/企业采购价", "是", "是")
    add("A00铝", "成本端", "煤炭价格", "煤价上升通常支撑价格", "日度/月度", "0-2个月", "秦皇岛煤炭网/发改委/生意社/Wind", "是", "是")
    add("A00铝", "成本端", "氧化铝价格", "氧化铝上涨通常支撑铝价", "日度/月度", "0-1个月", "SMM/百川/生意社/Wind", "是", "是")
    add("A00铝", "回收端", "再生铝产量", "供给增加通常压低价格", "月度", "0-2个月", "SMM/有色协会/Wind", "否", "是")
    add("A00铝", "回收端", "废铝回收量", "回收供给增加通常压低价格", "月度", "0-2个月", "SMM/Mysteel/再生资源协会", "否", "是")
    add("A00铝", "需求端", "汽车产量", "需求增加通常推升价格", "月度", "0-2个月", "国家统计局/中汽协", "是", "是", "汽车产量当期值; 汽车销量Top50厂商合计", "可用", "可用于交通用铝需求代理")
    add("A00铝", "需求端", "家电产量", "需求增加通常推升价格", "月度", "0-2个月", "国家统计局", "是", "是", "房间空气调节器产量当期值; 家用电冰箱产量当期值", "可用", "空调和冰箱作为家电用铝代理")
    add("A00铝", "需求端", "建筑铝型材产量", "需求增加通常推升价格", "月度", "0-2个月", "国家统计局/有色协会/SMM/Wind", "是", "是")
    add("A00铝", "事件冲击", "限电政策dummy", "限电通常推升价格", "日度/月度", "0-1个月", "政策公告/新闻事件库", "是", "是")
    add("A00铝", "事件冲击", "环保限产政策", "限产通常推升价格", "日度/月度", "0-1个月", "政策公告/新闻事件库", "否", "是")

    # 1#白银
    add("1#白银", "供应端", "白银矿产量", "供应增加通常压低价格", "月度/年度", "1-3个月", "国家统计局/有色协会/Wind", "是", "是")
    add("1#白银", "供应端", "冶炼副产银", "供应增加通常压低价格", "月度/年度", "1-3个月", "有色协会/SMM/Wind", "是", "是")
    add("1#白银", "库存端", "上期所库存", "库存上升通常压低价格", "周度/月度", "0-1个月", "上海期货交易所", "是", "是")
    add("1#白银", "库存端", "国内库存", "库存上升通常压低价格", "周度/月度", "0-1个月", "SMM/Mysteel/Wind", "是", "是")
    add("1#白银", "需求端", "光伏装机容量", "需求增加通常推升价格", "月度", "0-3个月", "国家能源局", "是", "是")
    add("1#白银", "需求端", "光伏银浆需求", "需求增加通常推升价格", "月度", "0-3个月", "CPIA/SMM/Wind", "是", "是")
    add("1#白银", "需求端", "电子工业产值", "需求增加通常推升价格", "月度", "0-2个月", "国家统计局/工信部/Wind", "是", "是")
    add("1#白银", "成本端", "冶炼成本指数", "成本上升通常支撑价格", "月度", "0-2个月", "SMM/百川/Wind", "否", "是")
    add("1#白银", "事件冲击", "地缘政治dummy", "风险上升通常支撑避险需求", "日度/月度", "0-1个月", "新闻事件库", "否", "是")
    add("1#白银", "事件冲击", "风险事件指数", "风险上升通常支撑避险需求", "日度/月度", "0-1个月", "国内风险指数/新闻事件库", "否", "是")

    # ADC12
    add("ADC12", "供应端", "再生铝合金产量", "供应增加通常压低价格", "月度", "0-2个月", "SMM/Mysteel/百川/Wind", "是", "是")
    add("ADC12", "供应端", "再生铝开工率", "开工率上升通常压低价格", "周度/月度", "0-2个月", "SMM/Mysteel/百川/Wind", "是", "是")
    add("ADC12", "库存端", "铝合金库存", "库存上升通常压低价格", "周度/月度", "0-1个月", "SMM/Mysteel/Wind", "是", "是")
    add("ADC12", "库存端", "ADC12企业库存", "库存上升通常压低价格", "周度/月度", "0-1个月", "企业内部库存/Mysteel/SMM", "是", "是")
    add("ADC12", "成本端", "A00铝价格", "成本上涨通常推升ADC12", "日度/月度", "0-1个月", "长江有色/SMM/企业采购价", "是", "是", "A00铝_月均价; A00铝_价格月环比", "可用", "核心传导变量")
    add("ADC12", "回收端", "废铝回收量", "回收供给增加通常压低成本", "月度", "0-2个月", "SMM/Mysteel/再生资源协会", "是", "是")
    add("ADC12", "回收端", "再生铝比例", "比例上升通常降低成本", "月度/季度", "0-2个月", "企业配方/SMM/行业协会", "是", "是")
    add("ADC12", "需求端", "新能源汽车产量", "需求增加通常推升价格", "月度", "0-2个月", "国家统计局/中汽协", "是", "是", "新能源汽车产量当期值", "可用", "压铸件需求代理")
    add("ADC12", "需求端", "汽车零部件产量", "需求增加通常推升价格", "月度", "0-2个月", "国家统计局/中汽协/Wind", "是", "是")
    add("ADC12", "事件冲击", "环保督查", "供给受限通常推升价格", "日度/月度", "0-1个月", "政策公告/新闻事件库", "否", "是")
    add("ADC12", "事件冲击", "废料进口政策", "收紧通常推升价格", "日度/月度", "0-2个月", "海关/生态环境部/政策公告", "是", "是")

    # ZLD104
    add("ZLD104", "供应端", "铸造铝合金产量", "供应增加通常压低价格", "月度", "0-2个月", "SMM/Mysteel/百川/Wind", "是", "是")
    add("ZLD104", "供应端", "开工率", "开工率上升通常压低价格", "周度/月度", "0-2个月", "SMM/Mysteel/百川/Wind", "是", "是")
    add("ZLD104", "库存端", "社会库存", "库存上升通常压低价格", "周度/月度", "0-1个月", "SMM/Mysteel/Wind", "是", "是")
    add("ZLD104", "库存端", "企业库存", "库存上升通常压低价格", "周度/月度", "0-1个月", "企业内部库存/SMM/Mysteel", "是", "是")
    add("ZLD104", "成本端", "A00铝价格", "成本上涨通常推升ZLD104", "日度/月度", "0-1个月", "长江有色/SMM/企业采购价", "是", "是", "A00铝_月均价; A00铝_价格月环比", "可用", "核心传导变量")
    add("ZLD104", "成本端", "能源成本", "能源成本上升通常支撑价格", "月度", "0-2个月", "电价/煤价/企业能源成本", "是", "是")
    add("ZLD104", "回收端", "废铝回收量", "回收供给增加通常压低成本", "月度", "0-2个月", "SMM/Mysteel/再生资源协会", "否", "是")
    add("ZLD104", "回收端", "再生料供给", "供给增加通常压低成本", "月度", "0-2个月", "SMM/Mysteel/企业采购", "否", "是")
    add("ZLD104", "需求端", "工程机械产量", "需求增加通常推升价格", "月度", "0-2个月", "国家统计局/工程机械协会/Wind", "是", "是")
    add("ZLD104", "需求端", "工业PMI", "景气上行通常推升价格", "月度", "0-1个月", "国家统计局/中国物流与采购联合会", "是", "是")
    add("ZLD104", "需求端", "通用设备产量", "需求增加通常推升价格", "月度", "0-2个月", "国家统计局/Wind", "是", "是")
    add("ZLD104", "事件冲击", "限产政策", "限产通常推升价格", "日度/月度", "0-1个月", "政策公告/新闻事件库", "否", "是")
    add("ZLD104", "事件冲击", "运输扰动", "扰动通常推升区域价格", "日度/月度", "0-1个月", "物流指数/新闻事件库/企业记录", "否", "是")

    supplemental = {
        "1#铜": [
            ("需求端", "制造业PMI", "景气上行通常支撑价格", "制造业PMI; 制造业PMI_变化"),
            ("需求端", "工业增加值同比", "工业生产改善通常支撑价格", "工业增加值同比增长"),
            ("需求端", "房地产开发景气指数", "地产景气改善通常支撑铜消费", "房地产开发景气指数; 房地产开发景气指数环比"),
            ("成本端", "PPI当月同比", "工业品价格上行通常支撑金属价格", "PPI当月同比增长"),
            ("成本端", "企业商品价格矿产品指数", "矿产品价格上行通常支撑铜价", "企业商品价格矿产品指数; 企业商品价格矿产品环比增长"),
            ("贸易端", "当月进口额", "进口景气改善通常反映国内需求", "当月进口额; 当月进口额环比增长"),
        ],
        "A00铝": [
            ("需求端", "制造业PMI", "景气上行通常支撑价格", "制造业PMI; 制造业PMI_变化"),
            ("需求端", "房地产开发景气指数", "地产景气改善通常支撑建筑用铝", "房地产开发景气指数; 房地产开发景气指数环比"),
            ("成本端", "PPI当月同比", "工业品价格上行通常支撑铝价", "PPI当月同比增长"),
            ("成本端", "企业商品价格煤油电指数", "能源价格上行通常支撑铝成本", "企业商品价格煤油电指数; 企业商品价格煤油电环比增长"),
            ("成本端", "企业商品价格矿产品指数", "矿产品价格上行通常支撑原料成本", "企业商品价格矿产品指数; 企业商品价格矿产品环比增长"),
        ],
        "1#白银": [
            ("需求端", "制造业PMI", "景气上行通常支撑工业银需求", "制造业PMI; 制造业PMI_变化"),
            ("需求端", "工业增加值同比", "工业生产改善通常支撑白银工业需求", "工业增加值同比增长"),
            ("需求端", "当月出口额", "出口改善通常支撑电子与光伏链需求", "当月出口额; 当月出口额环比增长"),
            ("成本端", "PPI当月同比", "工业品价格上行通常支撑商品价格", "PPI当月同比增长"),
            ("成本端", "企业商品价格矿产品指数", "矿产品价格上行通常支撑白银成本侧", "企业商品价格矿产品指数; 企业商品价格矿产品环比增长"),
        ],
        "ADC12": [
            ("需求端", "制造业PMI", "景气上行通常支撑压铸件需求", "制造业PMI; 制造业PMI_变化"),
            ("需求端", "工业增加值同比", "工业生产改善通常支撑铝合金需求", "工业增加值同比增长"),
            ("成本端", "PPI当月同比", "工业品价格上行通常支撑价格", "PPI当月同比增长"),
            ("成本端", "企业商品价格煤油电指数", "能源价格上行通常支撑再生铝成本", "企业商品价格煤油电指数; 企业商品价格煤油电环比增长"),
        ],
        "ZLD104": [
            ("需求端", "制造业PMI", "景气上行通常支撑工业铸造需求", "制造业PMI; 制造业PMI_变化"),
            ("需求端", "非制造业PMI", "施工和服务景气改善通常支撑需求", "非制造业PMI; 非制造业PMI_变化"),
            ("需求端", "工业增加值同比", "工业生产改善通常支撑铸造铝需求", "工业增加值同比增长"),
            ("需求端", "房地产开发景气指数", "地产和施工链改善通常支撑需求", "房地产开发景气指数; 房地产开发景气指数环比"),
            ("成本端", "PPI当月同比", "工业品价格上行通常支撑价格", "PPI当月同比增长"),
            ("成本端", "企业商品价格煤油电指数", "能源价格上行通常支撑铸造成本", "企业商品价格煤油电指数; 企业商品价格煤油电环比增长"),
        ],
    }
    for material, items in supplemental.items():
        for category, name, direction, local_field in items:
            add(
                material,
                category,
                name,
                direction,
                "月度",
                "0-2个月",
                "AkShare公开宏观接口/东方财富/国家统计局口径",
                "否",
                "是",
                local_field,
                "可用",
                "第一批补充国内公开因子，适合做月度解释变量",
            )

    return rows


def build_variable_dictionary() -> pd.DataFrame:
    return pd.DataFrame(factor_rows())


def build_availability_table(variable_dict: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in variable_dict.iterrows():
        fields = [field.strip() for field in str(row["本地可用字段"]).split(";") if field.strip()]
        usable_fields = [field for field in fields if field in data.columns]
        rows.append(
            {
                "品种": row["品种"],
                "变量名": row["变量名"],
                "本地可得性": row["本地可得性"],
                "可用字段数": len(usable_fields),
                "可用字段": "; ".join(usable_fields),
                "缺失字段": "; ".join(field for field in fields if field not in data.columns),
            }
        )
    return pd.DataFrame(rows)


MODEL_CONFIGS = {
    "1#铜": [
        "SHFE铜仓单库存_环比",
        "SHFE铜主连收盘价_环比",
        "SHFE铜主连成交量_环比",
        "SHFE铜主连持仓量_环比",
        "制造业PMI_变化",
        "房地产开发景气指数环比",
        "PPI当月同比增长",
        "工业增加值同比增长",
        "当月进口额环比增长",
        "企业商品价格矿产品环比增长",
        "中国经济政策不确定性指数_变化",
        "中国贸易政策不确定性指数_变化",
        "光缆产量当期值_环比",
        "电线电缆光缆及电工器材制造PPI_环比",
        "铜矿砂及其精矿进口数量_环比",
    ],
    "A00铝": [
        "SHFE铝仓单库存_环比",
        "SHFE铝主连收盘价_环比",
        "SHFE铝主连成交量_环比",
        "SHFE铝主连持仓量_环比",
        "制造业PMI_变化",
        "房地产开发景气指数环比",
        "PPI当月同比增长",
        "企业商品价格煤油电环比增长",
        "企业商品价格矿产品环比增长",
        "中国经济政策不确定性指数_变化",
        "中国贸易政策不确定性指数_变化",
        "ADC12_A00价差_滞后1期变化",
        "汽车产量当期值_环比",
        "房间空气调节器产量当期值_环比",
        "家用电冰箱产量当期值_环比",
        "汽车销量Top50厂商合计_环比",
    ],
    "1#白银": [
        "制造业PMI_变化",
        "PPI当月同比增长",
        "工业增加值同比增长",
        "当月出口额环比增长",
        "企业商品价格矿产品环比增长",
        "中国经济政策不确定性指数_变化",
        "中国贸易政策不确定性指数_变化",
        "光缆产量当期值_环比",
        "电线电缆光缆及电工器材制造PPI_环比",
        "发电量当期值_环比",
    ],
    "ADC12": [
        "A00铝_价格月环比",
        "制造业PMI_变化",
        "PPI当月同比增长",
        "工业增加值同比增长",
        "企业商品价格煤油电环比增长",
        "中国经济政策不确定性指数_变化",
        "ADC12_A00价差_滞后1期变化",
        "新能源汽车产量当期值_环比",
        "汽车产量当期值_环比",
        "汽车销量Top50厂商合计_环比",
        "SHFE铝仓单库存_环比",
    ],
    "ZLD104": [
        "A00铝_价格月环比",
        "制造业PMI_变化",
        "非制造业PMI_变化",
        "工业增加值同比增长",
        "PPI当月同比增长",
        "企业商品价格煤油电环比增长",
        "房地产开发景气指数环比",
        "中国经济政策不确定性指数_变化",
        "ZLD104_A00价差_滞后1期变化",
        "发电机组产量当期值_环比",
        "发电量当期值_环比",
        "SHFE铝仓单库存_环比",
    ],
}


def standardize(frame: pd.DataFrame) -> pd.DataFrame:
    std = frame.std(ddof=0)
    return (frame - frame.mean()) / std.replace(0, np.nan)


def significance(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return "不显著"


def run_regression(data: pd.DataFrame, material: str, predictors: list[str]) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    target = f"{material}_价格月环比"
    usable_predictors = [col for col in predictors if col in data.columns]
    modeling = data[["月份", target] + usable_predictors].replace([np.inf, -np.inf], np.nan).dropna()
    usable_predictors = [
        col for col in usable_predictors
        if pd.to_numeric(modeling[col], errors="coerce").std(ddof=0) > 0
    ]
    modeling = modeling[["月份", target] + usable_predictors]
    if len(modeling) < max(12, len(usable_predictors) + 5) or not usable_predictors:
        fit_row = {
            "品种": material,
            "样本量": len(modeling),
            "R2": np.nan,
            "调整后R2": np.nan,
            "样本起点": "",
            "样本终点": "",
            "入模变量数": len(usable_predictors),
            "备注": "有效样本不足，未回归",
        }
        return pd.DataFrame(), fit_row, modeling

    y = standardize(modeling[[target]])[target]
    x = sm.add_constant(standardize(modeling[usable_predictors]))
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 2})

    rows = []
    for var in usable_predictors:
        coef = float(fit.params[var])
        pvalue = float(fit.pvalues[var])
        rows.append(
            {
                "品种": material,
                "目标变量": target,
                "变量": var,
                "标准化系数": coef,
                "影响强度_绝对值": abs(coef),
                "p值": pvalue,
                "显著性": significance(pvalue),
                "回归方向": "正向" if coef >= 0 else "负向",
            }
        )
    coefs = pd.DataFrame(rows)
    coefs["强弱排名"] = coefs["影响强度_绝对值"].rank(method="first", ascending=False).astype(int)
    coefs = coefs.sort_values("强弱排名")
    fit_row = {
        "品种": material,
        "样本量": int(fit.nobs),
        "R2": float(fit.rsquared),
        "调整后R2": float(fit.rsquared_adj),
        "样本起点": modeling["月份"].min().strftime("%Y-%m"),
        "样本终点": modeling["月份"].max().strftime("%Y-%m"),
        "入模变量数": len(usable_predictors),
        "备注": "国内可得变量基线OLS，解释用，不直接作为最终预测模型",
    }
    return coefs, fit_row, modeling


def main() -> None:
    variable_dict = build_variable_dictionary()
    domestic_data = add_feature_changes(load_domestic_features())
    availability = build_availability_table(variable_dict, domestic_data)

    coef_frames = []
    fit_rows = []
    model_frames = []
    for material, predictors in MODEL_CONFIGS.items():
        coefs, fit_row, modeling = run_regression(domestic_data, material, predictors)
        if not coefs.empty:
            coef_frames.append(coefs)
        fit_rows.append(fit_row)
        model_frames.append(modeling.assign(品种=material))

    coefficients = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    model_dataset = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    fit = pd.DataFrame(fit_rows)

    excluded = pd.DataFrame(
        [
            {"变量/文件": "data_cache_v3/lme_monthly.csv", "处理": "国际库存/注销仓单，第一版国内模型不纳入"},
            {"变量/文件": "FRED宏观金融变量", "处理": "美元、利率、VIX等国际金融变量，第一版国内模型不纳入"},
            {"变量/文件": "CFTC铜持仓", "处理": "美国期货持仓，第一版国内模型不纳入"},
            {"变量/文件": "IAI全球铝产量", "处理": "国际供给变量，第一版国内模型不纳入"},
        ]
    )

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        variable_dict.to_excel(writer, sheet_name="变量字典", index=False)
        availability.to_excel(writer, sheet_name="本地可得性", index=False)
        domestic_data.to_excel(writer, sheet_name="国内月度建模数据", index=False)
        fit.to_excel(writer, sheet_name="基线模型拟合", index=False)
        coefficients.to_excel(writer, sheet_name="基线模型系数", index=False)
        model_dataset.to_excel(writer, sheet_name="入模样本", index=False)
        excluded.to_excel(writer, sheet_name="本轮剔除国际变量", index=False)

    variable_dict.to_csv("domestic_material_variable_dictionary_v1.csv", index=False, encoding="utf-8-sig")
    availability.to_csv("domestic_material_data_availability_v1.csv", index=False, encoding="utf-8-sig")
    domestic_data.to_csv("domestic_material_monthly_dataset_v1.csv", index=False, encoding="utf-8-sig")
    coefficients.to_csv("domestic_material_factor_coefficients_v1.csv", index=False, encoding="utf-8-sig")
    fit.to_csv("domestic_material_factor_model_fit_v1.csv", index=False, encoding="utf-8-sig")

    print(OUTPUT_FILE.resolve())
    print(fit.to_string(index=False))
    if not coefficients.empty:
        print(coefficients.groupby("品种").head(5).to_string(index=False))


if __name__ == "__main__":
    main()
