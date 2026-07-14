from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
MODELING_DATA_XLSX = ROOT / "domestic_material_regression_analysis_v2.xlsx"
SIGNIFICANT_SOURCE = ROOT / "domestic_material_all_variable_results.csv"

OUT_DIR = ROOT / "monthly_price_prediction_outputs"
SUMMARY_CSV = OUT_DIR / "monthly_price_model_train_test_summary.csv"
METRICS_CSV = OUT_DIR / "monthly_price_model_metrics.csv"
TEST_COMPARE_CSV = OUT_DIR / "monthly_price_model_test_actual_vs_pred.csv"
BACKCAST_CSV = OUT_DIR / "monthly_price_model_latest_known_prediction.csv"
FUTURE_FORECAST_CSV = OUT_DIR / "monthly_price_model_future_forecast_to_2026_12.csv"
COEFFICIENTS_CSV = OUT_DIR / "monthly_price_model_coefficients.csv"
FORMULAS_CSV = OUT_DIR / "monthly_price_model_formulas.csv"
REPORT_XLSX = OUT_DIR / "monthly_price_prediction_report_full_significant_vars.xlsx"
SUMMARY_JSON = OUT_DIR / "monthly_price_prediction_summary.json"

FORECAST_END = pd.Timestamp("2026-12-01")
LAG_COL = "lag_monthly_avg_price"


def metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    actual = actual.astype(float)
    pred = pred.astype(float)
    error = pred - actual
    abs_error = error.abs()
    non_zero = actual.abs() > 1e-12
    return {
        "MAE": float(abs_error.mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "MAPE_pct": float((abs_error[non_zero] / actual[non_zero].abs()).mean() * 100)
        if non_zero.any()
        else np.nan,
        "Bias": float(error.mean()),
    }


def direction_accuracy(actual_price: pd.Series, pred_price: pd.Series) -> float:
    actual_change = actual_price.astype(float).diff()
    pred_change = pred_price.astype(float).diff()
    mask = actual_change.notna() & pred_change.notna() & (actual_change.abs() > 1e-12)
    if not mask.any():
        return np.nan
    return float((np.sign(actual_change[mask]) == np.sign(pred_change[mask])).mean() * 100)


def test_size(n_rows: int, n_predictors: int) -> int:
    requested = max(6, math.ceil(n_rows * 0.2))
    min_train = max(12, n_predictors + 3)
    return max(1, min(requested, n_rows - min_train))


def model_target_from_source(monthly: pd.DataFrame, source_target: str) -> str | None:
    if source_target not in monthly.columns:
        return None
    idx = monthly.columns.get_loc(source_target)
    if idx <= 0:
        return None
    return str(monthly.columns[idx - 1])


def is_event_dummy(name: str) -> bool:
    return str(name).endswith("冲击")


def latest_value_before(data: pd.DataFrame, month_col: str, field: str, cutoff: pd.Timestamp) -> float:
    usable = data.loc[data[month_col] <= cutoff, [month_col, field]].dropna()
    if usable.empty:
        return np.nan
    return float(usable.sort_values(month_col).iloc[-1][field])


def display_var(name: str) -> str:
    return "上月月均价" if name == LAG_COL else name


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The final sheet is the v2 modeling dataset. Use an index to avoid relying on
    # Chinese sheet-name literals in the source file.
    monthly = pd.read_excel(MODELING_DATA_XLSX, sheet_name=4)
    sig = pd.read_csv(SIGNIFICANT_SOURCE, encoding="utf-8-sig")

    month_col = monthly.columns[0]
    monthly[month_col] = pd.to_datetime(monthly[month_col])

    product_col = sig.columns[0]
    source_target_col = sig.columns[2]
    variable_col = sig.columns[3]
    p_col = sig.columns[6]
    rank_col = sig.columns[9]

    selected = (
        sig[pd.to_numeric(sig[p_col], errors="coerce") < 0.1]
        .sort_values([product_col, rank_col])
        .copy()
    )

    summary_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    compare_rows: list[dict[str, object]] = []
    backcast_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    formula_rows: list[dict[str, object]] = []
    a00_future_mom: dict[pd.Timestamp, float] = {}

    for product, group in selected.groupby(product_col, sort=False):
        source_target = str(group[source_target_col].iloc[0])
        target = model_target_from_source(monthly, source_target)
        if target is None or target not in monthly.columns:
            continue

        significant_vars = [v for v in group[variable_col].tolist() if v in monthly.columns]
        if not significant_vars:
            continue

        product_data = monthly[[month_col, target] + significant_vars].copy()
        product_data[LAG_COL] = product_data[target].shift(1)
        predictors = [LAG_COL] + significant_vars

        frame = (
            product_data[[month_col, target] + predictors]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(month_col)
            .reset_index(drop=True)
        )
        predictors = [p for p in predictors if frame[p].std(ddof=0) > 0]
        frame = frame[[month_col, target] + predictors]
        if len(frame) < max(15, len(predictors) + 8):
            continue

        holdout = test_size(len(frame), len(predictors))
        train = frame.iloc[:-holdout].copy()
        test = frame.iloc[-holdout:].copy()

        fit = sm.OLS(
            train[target],
            sm.add_constant(train[predictors], has_constant="add"),
        ).fit()

        pred_test = fit.predict(sm.add_constant(test[predictors], has_constant="add"))
        test_out = test[[month_col, target]].copy()
        test_out["predicted_monthly_avg_price"] = pred_test.values
        test_out["price_error"] = test_out["predicted_monthly_avg_price"] - test_out[target]
        test_out["abs_pct_error"] = test_out["price_error"].abs() / test_out[target].abs() * 100

        score = metrics(test_out[target], test_out["predicted_monthly_avg_price"])
        score["Direction_Accuracy_pct"] = direction_accuracy(
            test_out[target], test_out["predicted_monthly_avg_price"]
        )

        summary_rows.append(
            {
                "品种": product,
                "目标变量": target,
                "显著变量来源目标": source_target,
                "训练开始": train[month_col].min().strftime("%Y-%m"),
                "训练结束": train[month_col].max().strftime("%Y-%m"),
                "训练样本数": len(train),
                "测试开始": test[month_col].min().strftime("%Y-%m"),
                "测试结束": test[month_col].max().strftime("%Y-%m"),
                "测试样本数": len(test),
                "入模显著变量数": len(significant_vars),
                "入模显著变量": "；".join(significant_vars),
                "价格惯性项": "上月月均价",
            }
        )

        metric_row = {"品种": product, "评估对象": "月均价"}
        metric_row.update(score)
        metric_rows.append(metric_row)

        for _, row in test_out.iterrows():
            compare_rows.append(
                {
                    "品种": product,
                    "月份": row[month_col].strftime("%Y-%m"),
                    "实际月均价": row[target],
                    "预测月均价": row["predicted_monthly_avg_price"],
                    "价格误差": row["price_error"],
                    "价格绝对百分比误差_%": row["abs_pct_error"],
                }
            )

        for name, value in fit.params.items():
            coefficient_rows.append(
                {
                    "品种": product,
                    "变量": "截距" if name == "const" else display_var(name),
                    "非标准化系数": float(value),
                    "p值": float(fit.pvalues[name]),
                }
            )

        terms = []
        for name, value in fit.params.items():
            if name == "const":
                continue
            op = "+" if value >= 0 else "-"
            terms.append(f"{op} {abs(float(value)):.6f} * {display_var(name)}")
        formula_rows.append(
            {
                "品种": product,
                "因变量": target,
                "公式": f"{target} = {float(fit.params['const']):.6f} " + " ".join(terms),
                "变量个数_不含截距": len(fit.params) - 1,
            }
        )

        full_fit = sm.OLS(
            frame[target],
            sm.add_constant(frame[predictors], has_constant="add"),
        ).fit()

        latest = frame.iloc[-1]
        latest_pred = float(
            full_fit.predict(
                sm.add_constant(latest[predictors].to_frame().T, has_constant="add")
            ).iloc[0]
        )
        backcast_rows.append(
            {
                "品种": product,
                "预测月份": latest[month_col].strftime("%Y-%m"),
                "说明": "使用该月已知因子回代预测；不是未来月份预测",
                "预测月均价": latest_pred,
                "实际月均价": float(latest[target]),
            }
        )

        start_month = pd.Timestamp(latest[month_col]) + pd.offsets.MonthBegin(1)
        previous_price = float(latest[target])
        forecast_month = start_month
        while forecast_month <= FORECAST_END:
            future_values: dict[str, float] = {}
            for predictor in predictors:
                if predictor == LAG_COL:
                    future_values[predictor] = previous_price
                elif predictor == "A00铝_价格月环比" and forecast_month in a00_future_mom:
                    future_values[predictor] = a00_future_mom[forecast_month]
                elif is_event_dummy(predictor):
                    future_values[predictor] = 0.0
                else:
                    future_values[predictor] = latest_value_before(
                        monthly, month_col, predictor, pd.Timestamp(latest[month_col])
                    )

            future_x = sm.add_constant(pd.DataFrame([future_values]), has_constant="add")
            forecast_price = float(full_fit.predict(future_x).iloc[0])
            future_rows.append(
                {
                    "品种": product,
                    "预测月份": forecast_month.strftime("%Y-%m"),
                    "预测月均价": forecast_price,
                    "上月价格输入": previous_price,
                    "解释变量假设": "连续变量沿用最近可得值；事件冲击默认为0；上月月均价递推",
                }
            )

            if product == "A00铝":
                a00_future_mom[forecast_month] = (forecast_price / previous_price - 1) * 100
            previous_price = forecast_price
            forecast_month = forecast_month + pd.offsets.MonthBegin(1)

    summary = pd.DataFrame(summary_rows)
    metrics_df = pd.DataFrame(metric_rows)
    compare = pd.DataFrame(compare_rows)
    backcast = pd.DataFrame(backcast_rows)
    future = pd.DataFrame(future_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    formulas = pd.DataFrame(formula_rows)

    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")
    compare.to_csv(TEST_COMPARE_CSV, index=False, encoding="utf-8-sig")
    backcast.to_csv(BACKCAST_CSV, index=False, encoding="utf-8-sig")
    future.to_csv(FUTURE_FORECAST_CSV, index=False, encoding="utf-8-sig")
    coefficients.to_csv(COEFFICIENTS_CSV, index=False, encoding="utf-8-sig")
    formulas.to_csv(FORMULAS_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(REPORT_XLSX, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="训练测试集", index=False)
        metrics_df.to_excel(writer, sheet_name="评估指标", index=False)
        compare.to_excel(writer, sheet_name="测试集预测对比", index=False)
        future.to_excel(writer, sheet_name="未来月份预测", index=False)
        backcast.to_excel(writer, sheet_name="已知月份回代", index=False)
        coefficients.to_excel(writer, sheet_name="模型系数", index=False)
        formulas.to_excel(writer, sheet_name="模型公式", index=False)

    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "modeling_data": f"{MODELING_DATA_XLSX}#sheet_index_4",
                "significant_source": str(SIGNIFICANT_SOURCE),
                "target_definition": "monthly average price level",
                "predictors": "all p<0.1 variables from the previous monthly mom regression plus lagged monthly average price",
                "future_forecast_end": FORECAST_END.strftime("%Y-%m"),
                "future_forecast_assumptions": "continuous predictors use latest known value, event dummies default to 0, lagged price is recursive",
                "outputs": {
                    "summary": str(SUMMARY_CSV),
                    "metrics": str(METRICS_CSV),
                    "test_compare": str(TEST_COMPARE_CSV),
                    "known_month_backcast": str(BACKCAST_CSV),
                    "future_forecast": str(FUTURE_FORECAST_CSV),
                    "coefficients": str(COEFFICIENTS_CSV),
                    "formulas": str(FORMULAS_CSV),
                    "excel_report": str(REPORT_XLSX),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {REPORT_XLSX}")
    print(f"wrote {FUTURE_FORECAST_CSV}")


if __name__ == "__main__":
    main()
