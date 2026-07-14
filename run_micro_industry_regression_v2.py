from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


INPUT = Path("industry_chain_regression_results_v1.xlsx")
OUTPUT = Path("micro_industry_regression_results_v2.xlsx")


COPPER_TARGET = "copper_price_usd_per_tonne_mom_pct"
ALUMINIUM_TARGET = "aluminium_price_usd_per_tonne_mom_pct"


VAR_CN = {
    "month": "月份",
    COPPER_TARGET: "铜价月环比",
    ALUMINIUM_TARGET: "铝价月环比",
    "brent_usd_per_barrel_month_avg_mom_pct": "Brent原油价格月环比",
    "coal_australia_usd_per_mt_mom_pct": "动力煤价格月环比",
    "natural_gas_europe_usd_per_mmbtu_mom_pct": "天然气价格月环比",
    "freight_expenditures_index_mom_pct": "货运成本指数月环比",
    "china_real_estate_climate_index_mom_diff": "房地产景气指数变化",
    "cftc_copper_noncommercial_net_long_contracts_mom_diff": "CFTC铜非商业净多头变化",
    "cftc_copper_open_interest_contracts_mom_pct": "CFTC铜期货持仓量月环比",
    "iai_primary_aluminium_world_kt_mom_pct": "全球原铝产量月环比",
    "iai_primary_aluminium_china_estimated_kt_mom_pct": "中国原铝产量月环比",
    "iai_alumina_total_world_kt_mom_pct": "全球氧化铝产量月环比",
    "iai_alumina_total_china_estimated_kt_mom_pct": "中国氧化铝产量月环比",
}


VAR_META = {
    "brent_usd_per_barrel_month_avg_mom_pct": ("成本", "原油上涨会推高能源、运输和冶炼成本，通常支撑采购价格。"),
    "coal_australia_usd_per_mt_mom_pct": ("成本", "动力煤上涨会推高电力成本，对高耗电的铝影响更明显。"),
    "natural_gas_europe_usd_per_mmbtu_mom_pct": ("成本", "天然气上涨会推高能源成本，支撑金属价格。"),
    "freight_expenditures_index_mom_pct": ("成本/贸易", "货运成本上涨会提高跨区域采购和交付成本。"),
    "china_real_estate_climate_index_mom_diff": ("需求", "地产景气改善会带动家电、线缆、铝材等需求。"),
    "cftc_copper_noncommercial_net_long_contracts_mom_diff": ("库存/期货", "投机净多头增加代表市场看涨铜价，通常支撑铜价。"),
    "cftc_copper_open_interest_contracts_mom_pct": ("库存/期货", "期货持仓增加代表市场资金参与度提高，可能放大价格波动。"),
    "iai_primary_aluminium_world_kt_mom_pct": ("供应", "全球原铝产量上升代表供应增加，理论上压制铝价。"),
    "iai_primary_aluminium_china_estimated_kt_mom_pct": ("供应", "中国原铝产量上升代表主要供应国供应增加，理论上压制铝价。"),
    "iai_alumina_total_world_kt_mom_pct": ("供应", "氧化铝供应增加会缓解电解铝原料紧张。"),
    "iai_alumina_total_china_estimated_kt_mom_pct": ("供应", "中国氧化铝供应增加会缓解国内铝产业链成本压力。"),
}


COPPER_PREDICTORS = [
    "brent_usd_per_barrel_month_avg_mom_pct",
    "coal_australia_usd_per_mt_mom_pct",
    "natural_gas_europe_usd_per_mmbtu_mom_pct",
    "freight_expenditures_index_mom_pct",
    "china_real_estate_climate_index_mom_diff",
    "cftc_copper_noncommercial_net_long_contracts_mom_diff",
    "cftc_copper_open_interest_contracts_mom_pct",
]


ALUMINIUM_PREDICTORS = [
    "iai_primary_aluminium_world_kt_mom_pct",
    "iai_primary_aluminium_china_estimated_kt_mom_pct",
    "iai_alumina_total_world_kt_mom_pct",
    "iai_alumina_total_china_estimated_kt_mom_pct",
    "brent_usd_per_barrel_month_avg_mom_pct",
    "coal_australia_usd_per_mt_mom_pct",
    "natural_gas_europe_usd_per_mmbtu_mom_pct",
    "freight_expenditures_index_mom_pct",
    "china_real_estate_climate_index_mom_diff",
]


PENDING = [
    ("铜价模型", "供应", "全球铜矿产量", "ICSG月度数据较适合，但完整历史通常需要订阅。"),
    ("铜价模型", "供应", "全球精炼铜产量", "ICSG月度数据较适合，但完整历史通常需要订阅。"),
    ("铜价模型", "供应", "全球铜消费量/用量", "ICSG月度数据较适合，但完整历史通常需要订阅。"),
    ("铜价模型", "供应", "铜TC/RC加工费", "历史数据多来自Fastmarkets、SMM、Wind，通常付费。"),
    ("铜价模型", "需求", "国家电网投资额", "需从国家能源局/国家电网月度累计报告整理并差分。"),
    ("铜价模型", "需求", "电缆产量", "需国家统计局具体指标代码或CEIC/Wind。"),
    ("铜价模型", "需求", "新能源汽车销量", "需中汽协历史月度表或CEIC/Wind。"),
    ("铜价模型", "需求", "光伏新增装机", "需国家能源局月度累计报告整理并差分。"),
    ("铜价模型", "需求", "风电新增装机", "需国家能源局月度累计报告整理并差分。"),
    ("铜价模型", "需求", "空调产量", "需国家统计局具体指标代码或CEIC/Wind。"),
    ("铜价模型", "需求", "冰箱产量", "需国家统计局具体指标代码或CEIC/Wind。"),
    ("铜价模型", "库存/期货", "LME铜库存、LME注销仓单、LME升贴水", "官方历史数据多需下载历史报告或付费。"),
    ("铜价模型", "库存/期货", "SHFE铜库存、SHFE铜仓单", "需批量抓取上期所周报/日报历史。"),
    ("铜价模型", "库存/期货", "COMEX铜库存", "需CME历史库存报表。"),
    ("铜价模型", "贸易", "中国铜矿砂及精矿进口量", "需海关HS编码月度数据。"),
    ("铜价模型", "贸易", "中国未锻轧铜及铜材进口量", "需海关HS编码月度数据。"),
    ("铜价模型", "贸易", "中国精炼铜进口量", "需海关HS编码月度数据。"),
    ("铝价模型", "供应", "全球铝土矿产量", "公开数据多为年度，月度历史较难。"),
    ("铝价模型", "供应", "电解铝开工率/运行产能", "历史多来自SMM、百川、Wind，通常付费。"),
    ("铝价模型", "需求", "新能源汽车销量、汽车产销量", "需中汽协历史月度表或CEIC/Wind。"),
    ("铝价模型", "需求", "光伏新增装机、风电新增装机", "需国家能源局月度累计报告整理并差分。"),
    ("铝价模型", "需求", "房地产新开工面积、房地产投资完成额", "需国家统计局具体指标代码或CEIC/Wind。"),
    ("铝价模型", "需求", "空调产量、冰箱产量", "需国家统计局具体指标代码或CEIC/Wind。"),
    ("铝价模型", "库存/期货", "LME铝库存、LME注销仓单、LME升贴水", "官方历史数据多需下载历史报告或付费。"),
    ("铝价模型", "库存/期货", "SHFE铝库存、SHFE铝仓单", "需批量抓取上期所周报/日报历史。"),
    ("铝价模型", "成本", "工业电价", "中国月度工业电价口径不稳定，通常需Wind或人工整理。"),
    ("铝价模型", "成本", "氧化铝价格、铝土矿价格", "长期历史多来自SMM、百川、Wind，通常付费。"),
    ("铝价模型", "成本", "欧盟碳价/EUA", "公开源可查，但长期日度历史需额外接口或付费。"),
    ("铝价模型", "贸易", "中国铝土矿进口量、氧化铝进口量、铝材进出口量", "需海关HS编码月度数据。"),
]


def zscore(frame):
    return (frame - frame.mean()) / frame.std(ddof=0)


def sig(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return "不显著"


def run_model(df, model_name, target, predictors):
    data = df[["month", target] + predictors].replace([np.inf, -np.inf], np.nan).dropna()
    y = zscore(data[target])
    x = sm.add_constant(zscore(data[predictors]))
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    coef_rows = []
    for var in predictors:
        category, relation = VAR_META[var]
        coef = fit.params[var]
        pvalue = fit.pvalues[var]
        coef_rows.append(
            {
                "模型": model_name,
                "类别": category,
                "指标": VAR_CN[var],
                "标准化系数": coef,
                "p值": pvalue,
                "显著性": sig(pvalue),
                "方向": "正向" if coef >= 0 else "负向",
                "与采购价格的关系": relation,
            }
        )
    coef = pd.DataFrame(coef_rows)
    coef["影响强度排名"] = coef["标准化系数"].abs().rank(ascending=False, method="first").astype(int)
    coef = coef.sort_values("影响强度排名")

    fit_row = {
        "模型": model_name,
        "样本量": int(fit.nobs),
        "R2": fit.rsquared,
        "调整后R2": fit.rsquared_adj,
        "样本起点": data["month"].min().strftime("%Y-%m"),
        "样本终点": data["month"].max().strftime("%Y-%m"),
        "入模变量数": len(predictors),
    }
    return coef, fit_row, data


def chinese_data_table(data, target, predictors):
    out = data[["month", target] + predictors].copy()
    out = out.rename(columns=VAR_CN)
    return out


def variable_description(model_name, target, predictors):
    rows = [
        {
            "模型": model_name,
            "类别": "采购价格",
            "指标": VAR_CN[target],
            "含义": "目标变量，表示采购价格代理变量的月度环比变化。",
            "与采购价格的关系": "被解释变量。",
            "是否入模": "是",
        }
    ]
    for var in predictors:
        category, relation = VAR_META[var]
        rows.append(
            {
                "模型": model_name,
                "类别": category,
                "指标": VAR_CN[var],
                "含义": relation,
                "与采购价格的关系": relation,
                "是否入模": "是",
            }
        )
    return pd.DataFrame(rows)


def main():
    df = pd.read_excel(INPUT, sheet_name="regression_dataset")
    df["month"] = pd.to_datetime(df["month"])

    copper_coef, copper_fit, copper_data = run_model(df, "铜价模型", COPPER_TARGET, COPPER_PREDICTORS)
    aluminium_coef, aluminium_fit, aluminium_data = run_model(
        df, "铝价模型", ALUMINIUM_TARGET, ALUMINIUM_PREDICTORS
    )

    copper_desc = variable_description("铜价模型", COPPER_TARGET, COPPER_PREDICTORS)
    aluminium_desc = variable_description("铝价模型", ALUMINIUM_TARGET, ALUMINIUM_PREDICTORS)
    pending = pd.DataFrame(PENDING, columns=["模型", "类别", "待补指标", "暂未入模原因"])

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        chinese_data_table(copper_data, COPPER_TARGET, COPPER_PREDICTORS).to_excel(
            writer, sheet_name="铜模型数据", index=False
        )
        chinese_data_table(aluminium_data, ALUMINIUM_TARGET, ALUMINIUM_PREDICTORS).to_excel(
            writer, sheet_name="铝模型数据", index=False
        )
        pd.DataFrame([copper_fit, aluminium_fit]).to_excel(writer, sheet_name="模型拟合度", index=False)
        copper_coef.to_excel(writer, sheet_name="铜模型系数", index=False)
        aluminium_coef.to_excel(writer, sheet_name="铝模型系数", index=False)
        copper_desc.to_excel(writer, sheet_name="铜模型变量说明", index=False)
        aluminium_desc.to_excel(writer, sheet_name="铝模型变量说明", index=False)
        pending.to_excel(writer, sheet_name="待补指标", index=False)

    print(OUTPUT.resolve())
    print(pd.DataFrame([copper_fit, aluminium_fit]).to_string(index=False))
    print("\n铜价模型系数")
    print(copper_coef[["类别", "指标", "标准化系数", "p值", "显著性", "方向"]].to_string(index=False))
    print("\n铝价模型系数")
    print(aluminium_coef[["类别", "指标", "标准化系数", "p值", "显著性", "方向"]].to_string(index=False))


if __name__ == "__main__":
    main()
