import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


TARGETS = {
    "铜价模型": "copper_price_usd_per_tonne_mom_pct",
    "铝价模型": "aluminium_price_usd_per_tonne_mom_pct",
}

VARIABLE_META = {
    "brent_usd_per_barrel_month_avg_mom_pct": ("成本", "Brent原油月环比"),
    "wti_usd_per_barrel_month_avg_mom_pct": ("成本", "WTI原油月环比"),
    "coal_australia_usd_per_mt_mom_pct": ("成本", "澳洲动力煤月环比"),
    "natural_gas_europe_usd_per_mmbtu_mom_pct": ("成本", "欧洲天然气月环比"),
    "freight_expenditures_index_mom_pct": ("成本/贸易", "货运成本指数月环比"),
    "china_ppi_yoy_pct_mom_diff": ("成本", "中国PPI同比变化"),
    "iai_primary_aluminium_world_kt_mom_pct": ("供应", "全球原铝产量月环比"),
    "iai_primary_aluminium_china_estimated_kt_mom_pct": ("供应", "中国原铝产量月环比"),
    "iai_alumina_total_world_kt_mom_pct": ("供应", "全球氧化铝产量月环比"),
    "iai_alumina_total_china_estimated_kt_mom_pct": ("供应", "中国氧化铝产量月环比"),
    "china_manufacturing_pmi_mom_diff": ("需求", "中国制造业PMI变化"),
    "china_nonmanufacturing_pmi_mom_diff": ("需求", "中国非制造业PMI变化"),
    "china_industrial_value_added_yoy_pct_mom_diff": ("需求", "中国工业增加值同比变化"),
    "china_real_estate_climate_index_mom_diff": ("需求", "房地产景气指数变化"),
    "china_total_exports_current_usd_mom_pct": ("贸易", "中国出口总额月环比"),
    "china_total_imports_current_usd_mom_pct": ("贸易", "中国进口总额月环比"),
    "cftc_copper_noncommercial_net_long_contracts_mom_diff": ("期货情绪", "CFTC铜非商业净多头变化"),
    "cftc_copper_open_interest_contracts_mom_pct": ("期货情绪", "CFTC铜持仓量月环比"),
    "us_dollar_broad_index_month_avg_mom_pct": ("金融控制", "美元指数月环比"),
    "fed_funds_rate_pct_mom_diff": ("金融控制", "联邦基金利率变化"),
    "us_10y_treasury_yield_pct_month_avg_mom_diff": ("金融控制", "10年期美债收益率变化"),
    "vix_month_avg_mom_diff": ("金融控制", "VIX变化"),
}


def month_start(series):
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def add_missing_changes(raw, changes):
    raw = raw.copy()
    changes = changes.copy()
    raw["month"] = month_start(raw["month"])
    changes["month"] = month_start(changes["month"])
    out = changes.merge(raw[["month"]], on="month", how="right")
    for col in changes.columns:
        if col != "month" and col not in out:
            out[col] = changes[col]

    pct_cols = {
        "us_dollar_broad_index_month_avg": "us_dollar_broad_index_month_avg_mom_pct",
    }
    diff_cols = {
        "fed_funds_rate_pct": "fed_funds_rate_pct_mom_diff",
        "us_10y_treasury_yield_pct_month_avg": "us_10y_treasury_yield_pct_month_avg_mom_diff",
        "vix_month_avg": "vix_month_avg_mom_diff",
    }
    for src, dest in pct_cols.items():
        out[dest] = raw[src].pct_change() * 100
    for src, dest in diff_cols.items():
        out[dest] = raw[src].diff()
    return out


def zscore(df):
    return (df - df.mean()) / df.std(ddof=0)


def significance(pvalue):
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return "不显著"


def run_model(df, model_name, target_col, predictors, variant):
    needed = [target_col] + predictors
    data = df[["month"] + needed].replace([np.inf, -np.inf], np.nan).dropna()
    y = zscore(data[target_col])
    x = zscore(data[predictors])
    x = sm.add_constant(x)
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    rows = []
    for var in predictors:
        dim, label = VARIABLE_META[var]
        coef = fit.params[var]
        pvalue = fit.pvalues[var]
        rows.append(
            {
                "模型": model_name,
                "模型版本": variant,
                "维度": dim,
                "变量": label,
                "字段名": var,
                "标准化系数": coef,
                "影响强度_绝对值": abs(coef),
                "p值": pvalue,
                "显著性": significance(pvalue),
                "方向": "正向" if coef >= 0 else "负向",
            }
        )
    result = pd.DataFrame(rows)
    result["强弱排名"] = (
        result.groupby(["模型", "模型版本"])["影响强度_绝对值"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    fit_row = {
        "模型": model_name,
        "模型版本": variant,
        "样本量": int(fit.nobs),
        "R2": fit.rsquared,
        "调整后R2": fit.rsquared_adj,
        "AIC": fit.aic,
        "BIC": fit.bic,
        "样本起点": data["month"].min().strftime("%Y-%m"),
        "样本终点": data["month"].max().strftime("%Y-%m"),
        "变量数": len(predictors),
    }
    return result, fit_row


def build_models(reg):
    cost = [
        "brent_usd_per_barrel_month_avg_mom_pct",
        "coal_australia_usd_per_mt_mom_pct",
        "natural_gas_europe_usd_per_mmbtu_mom_pct",
        "freight_expenditures_index_mom_pct",
        "china_ppi_yoy_pct_mom_diff",
    ]
    demand_trade = [
        "china_manufacturing_pmi_mom_diff",
        "china_industrial_value_added_yoy_pct_mom_diff",
        "china_real_estate_climate_index_mom_diff",
        "china_total_exports_current_usd_mom_pct",
        "china_total_imports_current_usd_mom_pct",
    ]
    finance = [
        "us_dollar_broad_index_month_avg_mom_pct",
        "fed_funds_rate_pct_mom_diff",
        "us_10y_treasury_yield_pct_month_avg_mom_diff",
        "vix_month_avg_mom_diff",
    ]
    copper_futures = [
        "cftc_copper_noncommercial_net_long_contracts_mom_diff",
        "cftc_copper_open_interest_contracts_mom_pct",
    ]
    aluminium_supply = [
        "iai_primary_aluminium_world_kt_mom_pct",
        "iai_primary_aluminium_china_estimated_kt_mom_pct",
        "iai_alumina_total_world_kt_mom_pct",
        "iai_alumina_total_china_estimated_kt_mom_pct",
    ]

    configs = [
        ("铜价模型", "产业链核心", TARGETS["铜价模型"], cost + demand_trade + copper_futures),
        ("铜价模型", "产业链+金融控制", TARGETS["铜价模型"], cost + demand_trade + copper_futures + finance),
        ("铝价模型", "产业链核心", TARGETS["铝价模型"], aluminium_supply + cost + demand_trade),
        ("铝价模型", "产业链+金融控制", TARGETS["铝价模型"], aluminium_supply + cost + demand_trade + finance),
    ]

    result_frames = []
    fit_rows = []
    for model_name, variant, target, predictors in configs:
        predictors = [p for p in predictors if p in reg.columns]
        result, fit_row = run_model(reg, model_name, target, predictors, variant)
        result_frames.append(result)
        fit_rows.append(fit_row)
    return pd.concat(result_frames, ignore_index=True), pd.DataFrame(fit_rows)


def interpretation_table(results):
    rows = []
    for _, row in results[results["强弱排名"] <= 6].sort_values(
        ["模型", "模型版本", "强弱排名"]
    ).iterrows():
        if row["显著性"] == "不显著":
            comment = "方向可作参考，但统计显著性不足。"
        elif row["方向"] == "正向":
            comment = "该指标上升时，价格月度涨幅通常扩大。"
        else:
            comment = "该指标上升时，价格月度涨幅通常收窄或价格承压。"
        rows.append(
            {
                "模型": row["模型"],
                "模型版本": row["模型版本"],
                "排名": row["强弱排名"],
                "维度": row["维度"],
                "变量": row["变量"],
                "方向": row["方向"],
                "显著性": row["显著性"],
                "解释": comment,
            }
        )
    return pd.DataFrame(rows)


def main():
    input_path = os.environ.get("INPUT_XLSX")
    if not input_path:
        input_path = "copper_aluminum_industry_chain_dataset_v1.xlsx"
    input_path = Path(input_path)

    output_path = Path("industry_chain_regression_results_v1.xlsx").resolve()

    raw = pd.read_excel(input_path, sheet_name="monthly_raw_v1")
    changes = pd.read_excel(input_path, sheet_name="monthly_change_v1")
    reg = add_missing_changes(raw, changes)

    results, fit = build_models(reg)
    interp = interpretation_table(results)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        fit.to_excel(writer, sheet_name="model_fit", index=False)
        results.sort_values(["模型", "模型版本", "强弱排名"]).to_excel(
            writer, sheet_name="standardized_coefficients", index=False
        )
        interp.to_excel(writer, sheet_name="interpretation_top6", index=False)
        reg.to_excel(writer, sheet_name="regression_dataset", index=False)

    results.to_csv("industry_chain_regression_coefficients_v1.csv", index=False, encoding="utf-8-sig")
    fit.to_csv("industry_chain_regression_model_fit_v1.csv", index=False, encoding="utf-8-sig")

    print(output_path)
    print(fit.to_string(index=False))
    print(
        results.sort_values(["模型", "模型版本", "强弱排名"])
        .groupby(["模型", "模型版本"])
        .head(6)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
