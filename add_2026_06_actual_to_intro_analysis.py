from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


VAT_FACTOR = 1.13
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "monthly_price_prediction_outputs" / "drop_limited_vars"
TARGET_WORKBOOK = OUTPUT_DIR / "引入变量结果分析.xlsx"
PRICE_CSV = ROOT / "ccmn_changjiang_avg_prices.csv"
OLD_FORECAST_WORKBOOK = Path(r"D:\BSH实习\铜、铝采购影响分析\国内\引入变量结果分析.xlsx")
SOURCE_NAME_BY_MODEL_NAME = {
    "ADC12": "铝合金ADC12",
    "ZLD104": "铸造铝合金锭(ZLD104)",
}


def find_old_forecast_workbook() -> Path:
    if OLD_FORECAST_WORKBOOK.exists():
        return OLD_FORECAST_WORKBOOK
    for root in Path("D:/").iterdir():
        if root.is_dir() and "BSH" in root.name:
            for path in root.rglob("*.xlsx"):
                if path.name == "引入变量结果分析.xlsx":
                    return path
    raise FileNotFoundError("找不到含 2026-06 原预测值的旧版引入变量结果分析.xlsx")


def metrics(group: pd.DataFrame) -> dict[str, float]:
    actual = group["实际月均价"].astype(float)
    pred = group["预测月均价"].astype(float)
    error = pred - actual
    abs_error = error.abs()
    non_zero = actual.abs() > 1e-12

    actual_change = actual.diff()
    pred_change = pred.diff()
    direction_mask = actual_change.notna() & pred_change.notna() & (actual_change.abs() > 1e-12)
    direction_accuracy = (
        float((np.sign(actual_change[direction_mask]) == np.sign(pred_change[direction_mask])).mean() * 100)
        if direction_mask.any()
        else np.nan
    )

    return {
        "MAE": float(abs_error.mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "MAPE_pct": float((abs_error[non_zero] / actual[non_zero].abs()).mean() * 100)
        if non_zero.any()
        else np.nan,
        "Bias": float(error.mean()),
        "Direction_Accuracy_pct": direction_accuracy,
    }


def main() -> None:
    sheets = pd.read_excel(TARGET_WORKBOOK, sheet_name=None)
    sheet_names = list(sheets)

    summary = sheets[sheet_names[0]].copy()
    metric_df = sheets[sheet_names[1]].copy()
    compare = sheets[sheet_names[2]].copy()

    old_future = pd.read_excel(find_old_forecast_workbook(), sheet_name=3)
    old_future["预测月份"] = pd.to_datetime(old_future["预测月份"])
    june_forecast = old_future.loc[old_future["预测月份"].dt.strftime("%Y-%m") == "2026-06"].copy()
    if june_forecast.empty:
        raise ValueError("旧版未来预测表中没有 2026-06 预测值")

    daily = pd.read_csv(PRICE_CSV, encoding="utf-8-sig", parse_dates=["date"])
    june_daily = daily.loc[daily["date"].dt.strftime("%Y-%m") == "2026-06"]
    if june_daily.empty:
        raise ValueError("日度价格文件中没有 2026-06 数据")

    actual_inclusive = june_daily.drop(columns=["date"]).mean(numeric_only=True)
    forecast_inclusive = june_forecast.set_index("品种")["预测月均价"]

    new_rows: list[dict[str, object]] = []
    product_order = list(dict.fromkeys(compare["品种"].tolist()))
    for product in product_order:
        source_product = SOURCE_NAME_BY_MODEL_NAME.get(product, product)
        if source_product not in actual_inclusive.index or source_product not in forecast_inclusive.index:
            continue
        actual = float(actual_inclusive[source_product]) / VAT_FACTOR
        pred = float(forecast_inclusive[source_product]) / VAT_FACTOR
        error = pred - actual
        new_rows.append(
            {
                "品种": product,
                "月份": "2026-06",
                "实际月均价": actual,
                "预测月均价": pred,
                "价格误差": error,
                "价格绝对百分比误差_%": abs(error) / abs(actual) * 100,
            }
        )

    if not new_rows:
        raise ValueError("没有生成任何 2026-06 测试集对比行")

    compare["月份"] = pd.to_datetime(compare["月份"]).dt.strftime("%Y-%m")
    compare = compare.loc[compare["月份"] != "2026-06"].copy()
    compare = pd.concat([compare, pd.DataFrame(new_rows)], ignore_index=True)
    compare["_品种顺序"] = compare["品种"].map({p: i for i, p in enumerate(product_order)})
    compare["_月份排序"] = pd.to_datetime(compare["月份"])
    compare = (
        compare.sort_values(["_品种顺序", "_月份排序"])
        .drop(columns=["_品种顺序", "_月份排序"])
        .reset_index(drop=True)
    )

    metric_rows = []
    for product, group in compare.groupby("品种", sort=False):
        group = group.sort_values("月份")
        row = {"品种": product, "评估对象": "不含税月均价"}
        row.update(metrics(group))
        metric_rows.append(row)
    metric_df = pd.DataFrame(metric_rows, columns=metric_df.columns)

    counts = compare.groupby("品种")["月份"].agg(["min", "max", "count"])
    for idx, row in summary.iterrows():
        product = row["品种"]
        if product not in counts.index:
            continue
        summary.loc[idx, "测试开始"] = counts.loc[product, "min"]
        summary.loc[idx, "测试结束"] = counts.loc[product, "max"]
        summary.loc[idx, "测试样本数"] = int(counts.loc[product, "count"])

    sheets[sheet_names[0]] = summary
    sheets[sheet_names[1]] = metric_df
    sheets[sheet_names[2]] = compare

    with pd.ExcelWriter(TARGET_WORKBOOK, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"updated {TARGET_WORKBOOK}")
    print(pd.DataFrame(new_rows).to_string(index=False))


if __name__ == "__main__":
    main()
