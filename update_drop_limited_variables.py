from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
MODELING_DATA_XLSX = ROOT / "domestic_material_regression_analysis_v2.xlsx"
SOURCE_RESULTS = ROOT / "domestic_material_all_variable_results.csv"
OUT_DIR = ROOT / "monthly_price_prediction_outputs" / "drop_limited_vars"

DROP_VARS = {"SHFE铝仓单库存_环比", "SHFE铜仓单库存_环比", "房地产开发景气指数环比"}
FORECAST_END = pd.Timestamp("2026-12-01")
LAG_COL = "上月月均价"


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


def metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    actual = actual.astype(float)
    pred = pred.astype(float)
    error = pred - actual
    abs_error = error.abs()
    non_zero = actual.abs() > 1e-12
    actual_change = actual.diff()
    pred_change = pred.diff()
    direction_mask = actual_change.notna() & pred_change.notna() & (actual_change.abs() > 1e-12)
    return {
        "MAE": float(abs_error.mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "MAPE_pct": float((abs_error[non_zero] / actual[non_zero].abs()).mean() * 100)
        if non_zero.any()
        else np.nan,
        "Bias": float(error.mean()),
        "Direction_Accuracy_pct": float(
            (np.sign(actual_change[direction_mask]) == np.sign(pred_change[direction_mask])).mean() * 100
        )
        if direction_mask.any()
        else np.nan,
    }


def test_size(n_rows: int, n_predictors: int) -> int:
    requested = max(6, math.ceil(n_rows * 0.2))
    min_train = max(12, n_predictors + 3)
    return max(1, min(requested, n_rows - min_train))


def model_price_target(modeling: pd.DataFrame, source_target: str) -> str:
    idx = modeling.columns.get_loc(source_target)
    return str(modeling.columns[idx - 1])


def latest_value_before(data: pd.DataFrame, month_col: str, field: str, cutoff: pd.Timestamp) -> float:
    usable = data.loc[data[month_col] <= cutoff, [month_col, field]].dropna()
    if usable.empty:
        return np.nan
    return float(usable.sort_values(month_col).iloc[-1][field])


def is_event_dummy(name: str) -> bool:
    return str(name).endswith("冲击")


def run_standardized_regression(
    modeling: pd.DataFrame,
    month_col: str,
    product: str,
    source_target: str,
    predictors: list[str],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    usable_predictors = [p for p in predictors if p in modeling.columns and p not in DROP_VARS]
    frame = (
        modeling[[month_col, source_target] + usable_predictors]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(month_col)
        .reset_index(drop=True)
    )
    usable_predictors = [p for p in usable_predictors if frame[p].std(ddof=0) > 0]
    frame = frame[[month_col, source_target] + usable_predictors]

    y = zscore(frame[[source_target]])[source_target]
    x = sm.add_constant(zscore(frame[usable_predictors]), has_constant="add")
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 2})

    rows = []
    for var in usable_predictors:
        coef = float(fit.params[var])
        pvalue = float(fit.pvalues[var])
        rows.append(
            {
                "品种": product,
                "模型版本": "候选全变量模型v2_删除缺口变量",
                "目标变量": source_target,
                "变量": var,
                "标准化系数": coef,
                "影响强度_绝对值": abs(coef),
                "p值": pvalue,
                "显著性": significance(pvalue),
                "方向": "正向" if coef >= 0 else "负向",
            }
        )
    coefs = pd.DataFrame(rows).sort_values("影响强度_绝对值", ascending=False).reset_index(drop=True)
    coefs["强弱排名"] = np.arange(1, len(coefs) + 1)
    fit_row = {
        "品种": product,
        "模型版本": "候选全变量模型v2_删除缺口变量",
        "样本量": int(fit.nobs),
        "样本起点": frame[month_col].min().strftime("%Y-%m"),
        "样本终点": frame[month_col].max().strftime("%Y-%m"),
        "入模变量数": len(usable_predictors),
        "R2": float(fit.rsquared),
        "调整后R2": float(fit.rsquared_adj),
    }
    return coefs, fit_row, frame


def run_price_prediction(
    modeling: pd.DataFrame,
    month_col: str,
    regression_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    summary_rows = []
    metric_rows = []
    compare_rows = []
    future_rows = []
    coef_rows = []
    formula_rows = []
    sample_frames: dict[str, pd.DataFrame] = {}
    a00_future_mom: dict[pd.Timestamp, float] = {}

    selected = regression_results[regression_results["p值"] < 0.1].copy()
    for sample_idx, (product, group) in enumerate(selected.groupby("品种", sort=False), start=1):
        source_target = str(group["目标变量"].iloc[0])
        price_target = model_price_target(modeling, source_target)
        predictors = [v for v in group.sort_values("强弱排名")["变量"].tolist() if v not in DROP_VARS]
        product_data = modeling[[month_col, price_target] + predictors].copy()
        product_data[LAG_COL] = product_data[price_target].shift(1)
        model_predictors = [LAG_COL] + predictors
        frame = (
            product_data[[month_col, price_target] + model_predictors]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(month_col)
            .reset_index(drop=True)
        )
        model_predictors = [p for p in model_predictors if frame[p].std(ddof=0) > 0]
        frame = frame[[month_col, price_target] + model_predictors]
        if len(frame) < max(15, len(model_predictors) + 8):
            continue
        sample_frames[f"sample_{sample_idx}"] = frame.copy()

        holdout = test_size(len(frame), len(model_predictors))
        train = frame.iloc[:-holdout].copy()
        test = frame.iloc[-holdout:].copy()
        fit = sm.OLS(train[price_target], sm.add_constant(train[model_predictors], has_constant="add")).fit()
        test_pred = fit.predict(sm.add_constant(test[model_predictors], has_constant="add"))
        score = metrics(test[price_target], test_pred)

        summary_rows.append(
            {
                "品种": product,
                "目标变量": price_target,
                "显著变量来源目标": source_target,
                "训练开始": train[month_col].min().strftime("%Y-%m"),
                "训练结束": train[month_col].max().strftime("%Y-%m"),
                "训练样本数": len(train),
                "测试开始": test[month_col].min().strftime("%Y-%m"),
                "测试结束": test[month_col].max().strftime("%Y-%m"),
                "测试样本数": len(test),
                "入模显著变量数": len(predictors),
                "入模显著变量": "；".join(predictors),
                "价格惯性项": LAG_COL,
            }
        )
        metric_row = {"品种": product, "评估对象": "月均价"}
        metric_row.update(score)
        metric_rows.append(metric_row)

        for month, actual, pred in zip(test[month_col], test[price_target], test_pred):
            err = float(pred - actual)
            compare_rows.append(
                {
                    "品种": product,
                    "月份": month.strftime("%Y-%m"),
                    "实际月均价": float(actual),
                    "预测月均价": float(pred),
                    "价格误差": err,
                    "价格绝对百分比误差_%": abs(err) / abs(float(actual)) * 100,
                }
            )

        for name, value in fit.params.items():
            coef_rows.append(
                {
                    "品种": product,
                    "变量": "截距" if name == "const" else name,
                    "非标准化系数": float(value),
                    "p值": float(fit.pvalues[name]),
                }
            )
        terms = []
        for name, value in fit.params.items():
            if name == "const":
                continue
            op = "+" if value >= 0 else "-"
            terms.append(f"{op} {abs(float(value)):.6f} * {name}")
        formula_rows.append(
            {
                "品种": product,
                "因变量": price_target,
                "公式": f"{price_target} = {float(fit.params['const']):.6f} " + " ".join(terms),
                "变量个数_不含截距": len(fit.params) - 1,
            }
        )

        full_fit = sm.OLS(frame[price_target], sm.add_constant(frame[model_predictors], has_constant="add")).fit()
        latest = frame.iloc[-1]
        previous_price = float(latest[price_target])
        forecast_month = pd.Timestamp(latest[month_col]) + pd.offsets.MonthBegin(1)
        while forecast_month <= FORECAST_END:
            future_values = {}
            for predictor in model_predictors:
                if predictor == LAG_COL:
                    future_values[predictor] = previous_price
                elif predictor == "A00铝_价格月环比" and forecast_month in a00_future_mom:
                    future_values[predictor] = a00_future_mom[forecast_month]
                elif is_event_dummy(predictor):
                    future_values[predictor] = 0.0
                else:
                    future_values[predictor] = latest_value_before(modeling, month_col, predictor, pd.Timestamp(latest[month_col]))
            pred_price = float(full_fit.predict(sm.add_constant(pd.DataFrame([future_values]), has_constant="add")).iloc[0])
            future_rows.append(
                {
                    "品种": product,
                    "预测月份": forecast_month.strftime("%Y-%m"),
                    "预测月均价": pred_price,
                    "上月价格输入": previous_price,
                    "解释变量假设": "连续变量沿用最近可得值；事件冲击默认为0；上月月均价递推",
                }
            )
            if product == "A00铝":
                a00_future_mom[forecast_month] = (pred_price / previous_price - 1) * 100
            previous_price = pred_price
            forecast_month += pd.offsets.MonthBegin(1)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(compare_rows),
        pd.DataFrame(future_rows),
        pd.DataFrame(coef_rows),
        pd.DataFrame(formula_rows),
        sample_frames,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    modeling = pd.read_excel(MODELING_DATA_XLSX, sheet_name=4)
    modeling[modeling.columns[0]] = pd.to_datetime(modeling[modeling.columns[0]])
    month_col = modeling.columns[0]
    source = pd.read_csv(SOURCE_RESULTS, encoding="utf-8-sig")
    product_col, target_col, var_col, rank_col = source.columns[[0, 2, 3, 9]]

    all_results = []
    spans = []
    source_filtered = source[~source[var_col].isin(DROP_VARS)].copy()
    for product, group in source_filtered.groupby(product_col, sort=False):
        target = str(group[target_col].iloc[0])
        predictors = group.sort_values(rank_col)[var_col].tolist()
        coefs, fit_row, _ = run_standardized_regression(modeling, month_col, str(product), target, predictors)
        all_results.append(coefs)
        spans.append(fit_row)

    regression_results = pd.concat(all_results, ignore_index=True)
    model_span = pd.DataFrame(spans)

    summary, metrics_df, compare, future, price_coefs, formulas, sample_frames = run_price_prediction(
        modeling, month_col, regression_results
    )

    regression_xlsx = OUT_DIR / "回归全变量结果.xlsx"
    raw_xlsx = OUT_DIR / "回归原始数据.xlsx"
    analysis_xlsx = OUT_DIR / "引入变量结果分析.xlsx"
    forecast_xlsx = OUT_DIR / "引入变量预测结果.xlsx"
    test_compare_xlsx = OUT_DIR / "测试集对比.xlsx"

    with pd.ExcelWriter(regression_xlsx, engine="openpyxl") as writer:
        model_span.to_excel(writer, sheet_name="model_span", index=False)
        regression_results.to_excel(writer, sheet_name="all_variables", index=False)
        for product, frame in regression_results.groupby("品种", sort=False):
            sheet = str(product).replace("#", "").replace("/", "_")[:31]
            frame.to_excel(writer, sheet_name=sheet, index=False)

    sample_ranges = summary[
        ["品种", "目标变量", "显著变量来源目标", "入模显著变量数", "训练开始", "测试结束", "入模显著变量"]
    ].copy()
    sample_ranges = sample_ranges.rename(columns={"训练开始": "完整样本起点", "测试结束": "完整样本终点"})
    with pd.ExcelWriter(raw_xlsx, engine="openpyxl") as writer:
        sample_ranges.to_excel(writer, sheet_name="sample_ranges", index=False)
        regression_results[regression_results["p值"] < 0.1].to_excel(writer, sheet_name="significant_vars", index=False)
        regression_results.to_excel(writer, sheet_name="all_regression_results", index=False)
        summary.to_excel(writer, sheet_name="train_test_summary", index=False)
        modeling.to_excel(writer, sheet_name="v2_modeling_data_full", index=False)
        for sheet, frame in sample_frames.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)

    with pd.ExcelWriter(analysis_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="训练测试集", index=False)
        metrics_df.to_excel(writer, sheet_name="评估指标", index=False)
        compare.to_excel(writer, sheet_name="测试集预测对比", index=False)
        future.to_excel(writer, sheet_name="未来月份预测", index=False)
        price_coefs.to_excel(writer, sheet_name="模型系数", index=False)
        formulas.to_excel(writer, sheet_name="模型公式", index=False)

    with pd.ExcelWriter(forecast_xlsx, engine="openpyxl") as writer:
        future.to_excel(writer, sheet_name="引入变量预测结果", index=False)

    with pd.ExcelWriter(test_compare_xlsx, engine="openpyxl") as writer:
        compare.to_excel(writer, sheet_name="测试集对比", index=False)

    manifest = {
        "dropped_variables": sorted(DROP_VARS),
        "forecast_end": FORECAST_END.strftime("%Y-%m"),
        "outputs": {
            "回归全变量结果.xlsx": str(regression_xlsx),
            "回归原始数据.xlsx": str(raw_xlsx),
            "引入变量结果分析.xlsx": str(analysis_xlsx),
            "引入变量预测结果.xlsx": str(forecast_xlsx),
            "测试集对比.xlsx": str(test_compare_xlsx),
        },
    }
    (OUT_DIR / "更新说明.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(model_span.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
