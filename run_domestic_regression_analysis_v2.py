from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import build_domestic_material_factor_model_v1 as base


EVENT_CACHE = Path("data_cache_v3/domestic_event_dummies_monthly.csv")
VARIABLE_LIBRARY_XLSX = Path("domestic_material_industry_chain_variable_library_v2.xlsx")
OUTPUT_XLSX = Path("domestic_material_regression_analysis_v2.xlsx")
COEF_CSV = Path("domestic_material_regression_analysis_coefficients_v2.csv")
FIT_CSV = Path("domestic_material_regression_analysis_fit_v2.csv")
SUMMARY_CSV = Path("domestic_material_regression_analysis_summary_v2.csv")

EVENT_COLUMNS = [
    "COVID供应链冲击",
    "限电政策冲击",
    "地缘政治冲击",
    "环保限产冲击",
    "运输扰动冲击",
    "能源价格冲击",
]

RECYCLING_BACKLOG = [
    ("1#铜", "回收端", "废铜进口量", "SMM新闻/海关统计数据在线查询平台", "月度", "可用", "回收端供给代理变量，已补入recycling_import_monthly.csv"),
    ("1#铜", "回收端", "废铜回收量", "中国再生资源回收利用协会/SMM/Mysteel/Wind", "月度", "待补充", "铜再生供给核心变量"),
    ("1#铜", "回收端", "精废价差", "SMM/Mysteel/Wind", "日度/周度/月度", "待补充", "反映废铜替代精铜的经济性"),
    ("A00铝", "回收端", "废铝进口量", "SMM新闻/海关统计数据在线查询平台", "月度", "可用", "回收端供给代理变量，已补入recycling_import_monthly.csv"),
    ("A00铝", "回收端", "废铝回收量", "中国再生资源回收利用协会/SMM/Mysteel/Wind", "月度", "待补充", "再生铝供应增加会削弱原铝需求"),
    ("A00铝", "回收端", "再生铝产量", "有色协会/SMM/Mysteel/百川盈孚/Wind", "月度", "待补充", "原生铝替代变量"),
    ("A00铝", "回收端", "废铝价格", "SMM/Mysteel/生意社/百川盈孚/Wind", "日度/周度/月度", "待补充", "反映废铝回收积极性与成本"),
    ("ADC12", "回收端", "废铝进口量", "SMM新闻/海关统计数据在线查询平台", "月度", "可用", "回收端供给代理变量，已补入recycling_import_monthly.csv"),
    ("ADC12", "回收端", "废铝回收量", "中国再生资源回收利用协会/SMM/Mysteel/Wind", "月度", "待补充", "ADC12成本与供应核心变量"),
    ("ADC12", "供应端", "再生铝合金产量", "SMM/Mysteel/百川盈孚/Wind", "月度", "待补充", "ADC12直接供应变量"),
    ("ADC12", "回收端", "再生铝占比", "企业配方/行业协会/SMM", "月度/季度", "待补充", "反映产业结构"),
    ("ZLD104", "回收端", "废铝进口量", "SMM新闻/海关统计数据在线查询平台", "月度", "可用", "回收端供给代理变量，已补入recycling_import_monthly.csv"),
    ("ZLD104", "回收端", "废铝回收量", "中国再生资源回收利用协会/SMM/Mysteel/Wind", "月度", "待补充", "再生料供给变量"),
    ("ZLD104", "回收端", "再生铝供应量", "SMM/Mysteel/百川盈孚/Wind", "月度", "待补充", "再生料供给变量"),
    ("ZLD104", "回收端", "废料价差", "SMM/Mysteel/企业采购价", "日度/周度/月度", "待补充", "废料经济性变量"),
]

EVENT_VARIABLES = [
    ("1#铜", "政策变量", "中国经济政策不确定性指数", "FRED/PolicyUncertainty.com", "月度", "可用", "覆盖完整建模期，可衡量政策不确定性冲击"),
    ("1#铜", "政策变量", "中国贸易政策不确定性指数", "FRED/PolicyUncertainty.com", "月度", "可用", "覆盖完整建模期，可衡量贸易政策冲击"),
    ("A00铝", "政策变量", "中国经济政策不确定性指数", "FRED/PolicyUncertainty.com", "月度", "可用", "覆盖完整建模期，可衡量政策不确定性冲击"),
    ("A00铝", "政策变量", "中国贸易政策不确定性指数", "FRED/PolicyUncertainty.com", "月度", "可用", "覆盖完整建模期，可衡量贸易政策冲击"),
    ("1#白银", "政策变量", "中国经济政策不确定性指数", "FRED/PolicyUncertainty.com", "月度", "可用", "覆盖完整建模期，可衡量政策不确定性冲击"),
    ("1#白银", "政策变量", "中国贸易政策不确定性指数", "FRED/PolicyUncertainty.com", "月度", "可用", "覆盖完整建模期，可衡量贸易政策冲击"),
    ("ADC12", "政策变量", "中国经济政策不确定性指数", "FRED/PolicyUncertainty.com", "月度", "可用", "覆盖完整建模期，可衡量政策不确定性冲击"),
    ("ZLD104", "政策变量", "中国经济政策不确定性指数", "FRED/PolicyUncertainty.com", "月度", "可用", "覆盖完整建模期，可衡量政策不确定性冲击"),
    ("1#铜", "突发事件", "COVID供应链冲击", "人工构造dummy", "月度", "可用", "疫情封控影响物流、开工与需求"),
    ("1#铜", "突发事件", "地缘政治冲击", "人工构造dummy", "月度", "可用", "海外风险扰动有色金属风险溢价"),
    ("1#铜", "突发事件", "运输扰动冲击", "人工构造dummy", "月度", "可用", "物流扰动影响现货到货"),
    ("A00铝", "突发事件", "限电政策冲击", "人工构造dummy", "月度", "可用", "限电影响电解铝供应"),
    ("A00铝", "突发事件", "能源价格冲击", "人工构造dummy", "月度", "可用", "能源成本上升影响电解铝成本"),
    ("A00铝", "突发事件", "环保限产冲击", "人工构造dummy", "月度", "可用", "环保限产影响供应"),
    ("1#白银", "突发事件", "COVID供应链冲击", "人工构造dummy", "月度", "可用", "影响工业需求和物流"),
    ("1#白银", "突发事件", "地缘政治冲击", "人工构造dummy", "月度", "可用", "影响避险与风险偏好"),
    ("ADC12", "突发事件", "COVID供应链冲击", "人工构造dummy", "月度", "可用", "汽车链和再生铝物流扰动"),
    ("ADC12", "突发事件", "环保限产冲击", "人工构造dummy", "月度", "可用", "再生铝企业开工扰动"),
    ("ADC12", "突发事件", "运输扰动冲击", "人工构造dummy", "月度", "可用", "废料和成品物流扰动"),
    ("ZLD104", "突发事件", "COVID供应链冲击", "人工构造dummy", "月度", "可用", "工业链需求和物流扰动"),
    ("ZLD104", "突发事件", "限电政策冲击", "人工构造dummy", "月度", "可用", "铸造与上游电力约束"),
    ("ZLD104", "突发事件", "运输扰动冲击", "人工构造dummy", "月度", "可用", "区域物流扰动"),
    ("A00铝", "回收端", "ADC12_A00价差", "由现有价格数据构造", "月度", "可用", "再生铝合金相对原铝价差代理"),
    ("ADC12", "回收端", "ADC12_A00价差", "由现有价格数据构造", "月度", "可用", "再生铝合金相对原铝价差代理"),
    ("ZLD104", "回收端", "ZLD104_A00价差", "由现有价格数据构造", "月度", "可用", "铸造铝合金相对原铝价差代理"),
]

EVENT_CONFIGS = {
    "1#铜": ["COVID供应链冲击", "地缘政治冲击", "运输扰动冲击"],
    "A00铝": ["限电政策冲击", "能源价格冲击", "环保限产冲击", "COVID供应链冲击"],
    "1#白银": ["COVID供应链冲击", "地缘政治冲击", "能源价格冲击"],
    "ADC12": ["COVID供应链冲击", "环保限产冲击", "运输扰动冲击", "限电政策冲击"],
    "ZLD104": ["COVID供应链冲击", "限电政策冲击", "运输扰动冲击", "环保限产冲击"],
}

RECYCLING_PROXY_CONFIGS = {
    "1#铜": ["SHFE铜主连收盘价_环比", "电线电缆光缆及电工器材制造PPI_环比", "废铜进口量_环比"],
    "A00铝": ["SHFE铝主连收盘价_环比", "企业商品价格煤油电环比增长", "废铝进口量_环比"],
    "ADC12": ["A00铝_价格月环比", "汽车产量当期值_环比", "废铝进口量_环比"],
    "ZLD104": ["A00铝_价格月环比", "PPI当月同比增长", "废铝进口量_环比"],
}


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
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


def add_event_dummies(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    month = pd.to_datetime(out["月份"])
    out["COVID供应链冲击"] = month.isin(pd.to_datetime(["2022-03-01", "2022-04-01", "2022-05-01", "2022-12-01", "2023-01-01"])).astype(int)
    out["限电政策冲击"] = month.isin(pd.to_datetime(["2021-09-01", "2021-10-01", "2022-08-01"])).astype(int)
    out["地缘政治冲击"] = month.isin(pd.to_datetime(["2022-02-01", "2022-03-01", "2022-04-01"])).astype(int)
    out["环保限产冲击"] = month.isin(pd.to_datetime(["2021-11-01", "2021-12-01", "2022-01-01", "2022-02-01"])).astype(int)
    out["运输扰动冲击"] = month.isin(pd.to_datetime(["2022-04-01", "2022-05-01"])).astype(int)
    out["能源价格冲击"] = month.isin(pd.to_datetime(["2021-09-01", "2021-10-01", "2021-11-01", "2022-02-01", "2022-03-01", "2022-04-01"])).astype(int)
    EVENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out[["月份"] + EVENT_COLUMNS].to_csv(EVENT_CACHE, index=False, encoding="utf-8-sig")
    return out


def build_v2_data() -> pd.DataFrame:
    data = base.add_feature_changes(base.load_domestic_features())
    return add_event_dummies(data)


def v2_model_configs() -> dict[str, list[str]]:
    configs = {material: list(predictors) for material, predictors in base.MODEL_CONFIGS.items()}
    for material, events in EVENT_CONFIGS.items():
        configs.setdefault(material, [])
        configs[material].extend([event for event in events if event not in configs[material]])
    return configs


def field_range(data: pd.DataFrame, field: str) -> str:
    if field not in data.columns:
        return ""
    usable = data.loc[data[field].notna(), "月份"]
    if usable.empty:
        return ""
    return f"{usable.min():%Y-%m}至{usable.max():%Y-%m}"


def build_variable_library(data: pd.DataFrame, configs: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dict = base.build_variable_dictionary().copy()
    rows = []
    for material, category, name, source, freq, status, note in RECYCLING_BACKLOG + EVENT_VARIABLES:
        rows.append(
            {
                "品种": material,
                "变量类别": category,
                "变量名": name,
                "建议数据来源": source,
                "建议频率": freq,
                "本地可得性": status,
                "本地可用字段": name if name in data.columns else "",
                "可获取时间范围": field_range(data, name),
                "建模处理建议": note,
            }
        )
    v2_extra = pd.DataFrame(rows)

    slim_cols = ["品种", "变量类别", "变量名", "建议数据来源", "建议频率", "本地可得性", "本地可用字段", "建模处理建议"]
    available_base_cols = [c for c in slim_cols if c in base_dict.columns]
    base_slim = base_dict[available_base_cols].copy()
    for col in slim_cols:
        if col not in base_slim.columns:
            base_slim[col] = ""
    base_slim["可获取时间范围"] = base_slim["本地可用字段"].fillna("").map(
        lambda fields: "；".join(
            field_range(data, field.strip()) for field in str(fields).split(";") if field.strip() and field_range(data, field.strip())
        )
    )

    library = pd.concat([base_slim[slim_cols + ["可获取时间范围"]], v2_extra], ignore_index=True)
    library = library.drop_duplicates(["品种", "变量类别", "变量名"], keep="last")

    model_rows = []
    combined_configs = {material: list(predictors) for material, predictors in configs.items()}
    for material, predictors in RECYCLING_PROXY_CONFIGS.items():
        combined_configs.setdefault(material, [])
        combined_configs[material].extend([p for p in predictors if p not in combined_configs[material]])
    for material, predictors in combined_configs.items():
        for predictor in predictors:
            model_rows.append(
                {
                    "品种": material,
                    "变量": predictor,
                    "是否进入v2可运行模型": "是" if predictor in data.columns else "否",
                    "数据时间范围": field_range(data, predictor),
                }
            )
    model_fields = pd.DataFrame(model_rows)
    return library, model_fields


def usable_frame(data: pd.DataFrame, target: str, predictors: list[str]) -> tuple[pd.DataFrame, list[str]]:
    usable = [col for col in predictors if col in data.columns]
    frame = data[["月份", target] + usable].replace([np.inf, -np.inf], np.nan).dropna()
    usable = [col for col in usable if col in frame.columns and frame[col].std(ddof=0) > 0]
    return frame[["月份", target] + usable], usable


def run_ols(data: pd.DataFrame, material: str, variant: str, target: str, predictors: list[str]):
    frame, usable = usable_frame(data, target, predictors)
    if len(frame) < max(12, len(usable) + 5) or not usable:
        return pd.DataFrame(), {
            "品种": material,
            "模型版本": variant,
            "样本量": len(frame),
            "R2": np.nan,
            "调整后R2": np.nan,
            "AIC": np.nan,
            "BIC": np.nan,
            "样本起点": "",
            "样本终点": "",
            "入模变量数": len(usable),
            "备注": "有效样本不足，未回归",
        }
    y = zscore(frame[[target]])[target]
    x = sm.add_constant(zscore(frame[usable]))
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    rows = []
    for var in usable:
        coef = float(fit.params[var])
        pvalue = float(fit.pvalues[var])
        rows.append(
            {
                "品种": material,
                "模型版本": variant,
                "目标变量": target,
                "变量": var,
                "标准化系数": coef,
                "影响强度_绝对值": abs(coef),
                "p值": pvalue,
                "显著性": significance(pvalue),
                "方向": "正向" if coef >= 0 else "负向",
            }
        )
    coefs = pd.DataFrame(rows)
    coefs["强弱排名"] = coefs["影响强度_绝对值"].rank(method="first", ascending=False).astype(int)
    return coefs.sort_values("强弱排名"), {
        "品种": material,
        "模型版本": variant,
        "样本量": int(fit.nobs),
        "R2": float(fit.rsquared),
        "调整后R2": float(fit.rsquared_adj),
        "AIC": float(fit.aic),
        "BIC": float(fit.bic),
        "样本起点": frame["月份"].min().strftime("%Y-%m"),
        "样本终点": frame["月份"].max().strftime("%Y-%m"),
        "入模变量数": len(usable),
        "备注": "v2标准化OLS，HAC稳健标准误",
    }


def vif_filter(data: pd.DataFrame, target: str, predictors: list[str], threshold: float = 10.0, max_predictors: int = 8) -> list[str]:
    frame, usable = usable_frame(data, target, predictors)
    selected = list(usable)
    while len(selected) > 1:
        x = zscore(frame[selected]).dropna()
        if len(x) < len(selected) + 3:
            break
        vifs = pd.Series(
            [variance_inflation_factor(x.values, idx) for idx in range(len(selected))],
            index=selected,
        )
        if vifs.max() <= threshold:
            break
        selected.remove(vifs.idxmax())
    if len(selected) <= max_predictors:
        return selected
    corr = []
    for col in selected:
        pair = frame[[target, col]].dropna()
        corr.append((col, abs(pair[target].corr(pair[col])) if len(pair) >= 12 else 0))
    return [col for col, _ in sorted(corr, key=lambda item: item[1], reverse=True)[:max_predictors]]


def lasso_select(data: pd.DataFrame, target: str, predictors: list[str], max_predictors: int = 8) -> list[str]:
    frame, usable = usable_frame(data, target, predictors)
    if len(frame) < 18 or not usable:
        return []
    x = frame[usable].astype(float)
    y = frame[target].astype(float)
    x_scaled = StandardScaler().fit_transform(x)
    y_scaled = StandardScaler().fit_transform(y.to_numpy().reshape(-1, 1)).ravel()
    cv = min(5, max(2, len(frame) // 8))
    model = LassoCV(cv=cv, random_state=42, max_iter=20000).fit(x_scaled, y_scaled)
    coefs = pd.Series(model.coef_, index=usable)
    selected = coefs[coefs.abs() > 1e-6].abs().sort_values(ascending=False).head(max_predictors).index.tolist()
    if selected:
        return selected
    corr = []
    for col in usable:
        corr.append((col, abs(frame[target].corr(frame[col]))))
    return [col for col, _ in sorted(corr, key=lambda item: item[1], reverse=True)[:max_predictors]]


def top_summary(coefficients: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lasso = coefficients[coefficients["模型版本"] == "Lasso筛选模型"]
    for material, group in lasso.groupby("品种"):
        sig = group[group["显著性"] != "不显著"].sort_values("强弱排名")
        top = (sig if not sig.empty else group.sort_values("强弱排名")).head(5)
        fit_row = fit[(fit["品种"] == material) & (fit["模型版本"] == "Lasso筛选模型")]
        rows.append(
            {
                "品种": material,
                "Lasso模型Top因素": "；".join(f"{row['变量']}({row['方向']},{row['显著性']})" for _, row in top.iterrows()),
                "Lasso模型R2": np.nan if fit_row.empty else fit_row["R2"].iloc[0],
                "Lasso模型调整后R2": np.nan if fit_row.empty else fit_row["调整后R2"].iloc[0],
                "说明": "v2已加入事件dummy；废铜/废铝进口量作为回收端供给代理，单独进入回收端代理模型。",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = build_v2_data()
    configs = v2_model_configs()
    library, model_fields = build_variable_library(data, configs)

    coef_frames = []
    fit_rows = []
    selected_rows = []
    for material, predictors in configs.items():
        target = f"{material}_价格月环比"
        variants = {
            "候选全变量模型v2": [p for p in predictors if p in data.columns],
            "VIF筛选模型": vif_filter(data, target, predictors),
            "Lasso筛选模型": lasso_select(data, target, predictors),
        }
        recycle_predictors = [p for p in RECYCLING_PROXY_CONFIGS.get(material, []) if p in data.columns]
        if recycle_predictors:
            variants["回收端代理模型"] = recycle_predictors
        for variant, selected in variants.items():
            selected_rows.append({"品种": material, "模型版本": variant, "变量数": len(selected), "入模变量": "；".join(selected)})
            coefs, fit_row = run_ols(data, material, variant, target, selected)
            if not coefs.empty:
                coef_frames.append(coefs)
            fit_rows.append(fit_row)

    coefficients = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    fit = pd.DataFrame(fit_rows)
    selected = pd.DataFrame(selected_rows)
    summary = top_summary(coefficients, fit)

    with pd.ExcelWriter(VARIABLE_LIBRARY_XLSX, engine="openpyxl") as writer:
        library.to_excel(writer, sheet_name="产业链完整版变量库v2", index=False)
        pd.DataFrame(RECYCLING_BACKLOG, columns=["品种", "变量类别", "变量名", "建议数据来源", "建议频率", "本地可得性", "建模说明"]).to_excel(
            writer, sheet_name="回收端待补清单", index=False
        )
        pd.DataFrame(EVENT_VARIABLES, columns=["品种", "变量类别", "变量名", "数据来源", "频率", "本地可得性", "建模说明"]).to_excel(
            writer, sheet_name="事件变量说明", index=False
        )
        model_fields.to_excel(writer, sheet_name="v2可运行变量清单", index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="核心结论v2", index=False)
        fit.to_excel(writer, sheet_name="模型拟合对比v2", index=False)
        coefficients.to_excel(writer, sheet_name="标准化回归系数v2", index=False)
        selected.to_excel(writer, sheet_name="入模变量清单v2", index=False)
        data.to_excel(writer, sheet_name="v2建模数据", index=False)

    coefficients.to_csv(COEF_CSV, index=False, encoding="utf-8-sig")
    fit.to_csv(FIT_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print(VARIABLE_LIBRARY_XLSX.resolve())
    print(OUTPUT_XLSX.resolve())
    print(fit.to_string(index=False))
    print("\n核心结论v2")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
