from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import build_domestic_material_factor_model_v1 as base


OUTPUT_XLSX = Path("domestic_material_regression_analysis_v1.xlsx")
COEF_CSV = Path("domestic_material_regression_analysis_coefficients_v1.csv")
FIT_CSV = Path("domestic_material_regression_analysis_fit_v1.csv")
SUMMARY_CSV = Path("domestic_material_regression_analysis_summary_v1.csv")


MARKET_SYNC_KEYWORDS = ["主连收盘价"]
TRANSMISSION_VARIABLES = {
    "ADC12": ["A00铝_价格月环比"],
    "ZLD104": ["A00铝_价格月环比"],
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


def relation_note(coef: float, pvalue: float) -> str:
    direction = "推升" if coef >= 0 else "压制"
    if pvalue < 0.05:
        return f"统计显著，变量上升时价格月环比倾向于{direction}。"
    if pvalue < 0.1:
        return f"弱显著，方向可参考：变量上升时价格月环比倾向于{direction}。"
    return f"方向仅作参考，当前样本下显著性不足。"


def usable_model_frame(data: pd.DataFrame, target: str, predictors: list[str]) -> tuple[pd.DataFrame, list[str]]:
    usable = [col for col in predictors if col in data.columns]
    frame = data[["月份", target] + usable].replace([np.inf, -np.inf], np.nan).dropna()
    usable = [
        col
        for col in usable
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").std(ddof=0) > 0
    ]
    frame = frame[["月份", target] + usable]
    return frame, usable


def run_ols(
    data: pd.DataFrame,
    material: str,
    variant: str,
    target: str,
    predictors: list[str],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    frame, usable = usable_model_frame(data, target, predictors)
    min_n = max(12, len(usable) + 5)
    if len(frame) < min_n or not usable:
        fit_row = {
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
        return pd.DataFrame(), fit_row, frame.assign(品种=material, 模型版本=variant)

    y = zscore(frame[[target]])[target]
    x = sm.add_constant(zscore(frame[usable]))
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 2})

    coef_rows = []
    for var in usable:
        coef = float(fit.params[var])
        pvalue = float(fit.pvalues[var])
        coef_rows.append(
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
                "影响解释": relation_note(coef, pvalue),
            }
        )
    coefs = pd.DataFrame(coef_rows)
    coefs["强弱排名"] = coefs["影响强度_绝对值"].rank(method="first", ascending=False).astype(int)
    coefs = coefs.sort_values("强弱排名")

    fit_row = {
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
        "备注": "标准化OLS，HAC稳健标准误；用于影响因素解释",
    }
    return coefs, fit_row, frame.assign(品种=material, 模型版本=variant)


def pairwise_correlation_table(data: pd.DataFrame, material: str, target: str, predictors: list[str]) -> pd.DataFrame:
    rows = []
    for var in predictors:
        if var not in data.columns:
            continue
        frame = data[[target, var]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 12 or frame[var].std(ddof=0) == 0:
            continue
        rows.append(
            {
                "品种": material,
                "目标变量": target,
                "变量": var,
                "可用样本量": len(frame),
                "与目标相关系数": frame[target].corr(frame[var]),
                "相关强度_绝对值": abs(frame[target].corr(frame[var])),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["品种", "相关强度_绝对值"], ascending=[True, False])


def select_predictors(
    data: pd.DataFrame,
    material: str,
    target: str,
    predictors: list[str],
    *,
    max_predictors: int = 5,
    exclude_market_sync: bool = False,
) -> list[str]:
    candidates = [p for p in predictors if p in data.columns]
    if exclude_market_sync:
        forced = TRANSMISSION_VARIABLES.get(material, [])
        candidates = [
            p for p in candidates
            if p in forced or not any(keyword in p for keyword in MARKET_SYNC_KEYWORDS)
        ]
    corr = pairwise_correlation_table(data, material, target, candidates)
    if corr.empty:
        return []

    selected: list[str] = []
    forced = [p for p in TRANSMISSION_VARIABLES.get(material, []) if p in candidates]
    for var in forced:
        if var not in selected:
            selected.append(var)

    for var in corr["变量"].tolist():
        if var in selected:
            continue
        if len(selected) >= max_predictors:
            break
        keep = True
        for chosen in selected:
            pair = data[[var, chosen]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) >= 12 and abs(pair[var].corr(pair[chosen])) >= 0.85:
                keep = False
                break
        if keep:
            selected.append(var)
    return selected[:max_predictors]


def vif_table(data: pd.DataFrame, material: str, variant: str, predictors: list[str]) -> pd.DataFrame:
    usable = [p for p in predictors if p in data.columns]
    frame = data[usable].replace([np.inf, -np.inf], np.nan).dropna()
    usable = [p for p in usable if frame[p].std(ddof=0) > 0]
    if len(usable) < 2 or len(frame) < len(usable) + 3:
        return pd.DataFrame(columns=["品种", "模型版本", "变量", "VIF", "共线性提示"])
    x = zscore(frame[usable]).dropna()
    rows = []
    for idx, var in enumerate(usable):
        vif = float(variance_inflation_factor(x.values, idx))
        rows.append(
            {
                "品种": material,
                "模型版本": variant,
                "变量": var,
                "VIF": vif,
                "共线性提示": "高" if vif >= 10 else ("中" if vif >= 5 else "低"),
            }
        )
    return pd.DataFrame(rows).sort_values(["品种", "模型版本", "VIF"], ascending=[True, True, False])


def top_factor_summary(coefficients: pd.DataFrame, fit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = coefficients[coefficients["模型版本"] == "核心筛选模型"].copy()
    for material, group in selected.groupby("品种"):
        sig = group[group["显著性"] != "不显著"].sort_values("强弱排名")
        top = (sig if not sig.empty else group.sort_values("强弱排名")).head(3)
        fit_row = fit[(fit["品种"] == material) & (fit["模型版本"] == "核心筛选模型")]
        rows.append(
            {
                "品种": material,
                "核心影响因素Top3": "；".join(
                    f"{r['变量']}({r['方向']}, {r['显著性']})" for _, r in top.iterrows()
                ),
                "核心模型R2": np.nan if fit_row.empty else fit_row["R2"].iloc[0],
                "核心模型调整后R2": np.nan if fit_row.empty else fit_row["调整后R2"].iloc[0],
                "解释口径": "按标准化系数绝对值排序，优先列显著变量。",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = base.add_feature_changes(base.load_domestic_features())

    coef_frames = []
    fit_rows = []
    sample_frames = []
    corr_frames = []
    vif_frames = []
    selected_rows = []

    for material, predictors in base.MODEL_CONFIGS.items():
        target = f"{material}_价格月环比"
        all_predictors = [p for p in predictors if p in data.columns]
        target_corr = pairwise_correlation_table(data, material, target, all_predictors)
        corr_frames.append(target_corr)

        variants = {
            "候选全变量模型": all_predictors,
            "核心筛选模型": select_predictors(data, material, target, all_predictors, max_predictors=5),
            "基本面筛选模型": select_predictors(
                data,
                material,
                target,
                all_predictors,
                max_predictors=5,
                exclude_market_sync=True,
            ),
        }

        for variant, variant_predictors in variants.items():
            selected_rows.append(
                {
                    "品种": material,
                    "模型版本": variant,
                    "变量数": len(variant_predictors),
                    "入模变量": "；".join(variant_predictors),
                }
            )
            coefs, fit_row, sample = run_ols(data, material, variant, target, variant_predictors)
            if not coefs.empty:
                coef_frames.append(coefs)
            fit_rows.append(fit_row)
            sample_frames.append(sample)
            vif_frames.append(vif_table(data, material, variant, variant_predictors))

    coefficients = pd.concat(coef_frames, ignore_index=True) if coef_frames else pd.DataFrame()
    fit = pd.DataFrame(fit_rows)
    samples = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    correlations = pd.concat(corr_frames, ignore_index=True) if corr_frames else pd.DataFrame()
    vif = pd.concat(vif_frames, ignore_index=True) if vif_frames else pd.DataFrame()
    selected = pd.DataFrame(selected_rows)
    summary = top_factor_summary(coefficients, fit)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="核心结论", index=False)
        fit.to_excel(writer, sheet_name="模型拟合对比", index=False)
        coefficients.to_excel(writer, sheet_name="标准化回归系数", index=False)
        correlations.to_excel(writer, sheet_name="单变量相关性", index=False)
        vif.to_excel(writer, sheet_name="VIF共线性", index=False)
        selected.to_excel(writer, sheet_name="入模变量清单", index=False)
        samples.to_excel(writer, sheet_name="回归样本", index=False)

    coefficients.to_csv(COEF_CSV, index=False, encoding="utf-8-sig")
    fit.to_csv(FIT_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print(OUTPUT_XLSX.resolve())
    print(fit.to_string(index=False))
    print("\n核心结论")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
