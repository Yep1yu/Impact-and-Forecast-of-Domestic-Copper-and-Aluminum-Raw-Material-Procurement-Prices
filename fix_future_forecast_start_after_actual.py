from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "monthly_price_prediction_outputs" / "drop_limited_vars"
MODELING_DATA_XLSX = ROOT / "domestic_material_regression_analysis_v2.xlsx"
REGRESSION_XLSX = OUT_DIR / "回归全变量结果.xlsx"
ANALYSIS_XLSX = OUT_DIR / "引入变量结果分析.xlsx"
FORECAST_XLSX = OUT_DIR / "引入变量预测结果.xlsx"
FORECAST_CSV = OUT_DIR / "引入变量预测结果.csv"
MANIFEST = OUT_DIR / "更新说明.json"

FORECAST_START = pd.Timestamp("2026-07-01")
FORECAST_END = pd.Timestamp("2026-12-01")
ACTUAL_BASE_MONTH = pd.Timestamp("2026-06-01")
LAG_COL = "上月月均价"


def is_event_dummy(name: str) -> bool:
    return str(name).endswith("冲击")


def latest_value_before(data: pd.DataFrame, month_col: str, field: str, cutoff: pd.Timestamp) -> float:
    rows = data.loc[data[month_col] <= cutoff, [month_col, field]].dropna()
    if rows.empty:
        return np.nan
    return float(rows.sort_values(month_col).iloc[-1][field])


def model_price_target(modeling: pd.DataFrame, source_target: str) -> str:
    idx = modeling.columns.get_loc(source_target)
    return str(modeling.columns[idx - 1])


def build_future_forecast() -> pd.DataFrame:
    modeling = pd.read_excel(MODELING_DATA_XLSX, sheet_name=4)
    month_col = modeling.columns[0]
    modeling[month_col] = pd.to_datetime(modeling[month_col])

    reg = pd.read_excel(REGRESSION_XLSX, sheet_name="all_variables")
    selected = reg[pd.to_numeric(reg["p值"], errors="coerce") < 0.1].copy()

    rows: list[dict[str, object]] = []
    a00_future_mom: dict[pd.Timestamp, float] = {}

    for product, group in selected.groupby("品种", sort=False):
        source_target = str(group["目标变量"].iloc[0])
        price_target = model_price_target(modeling, source_target)
        predictors = group.sort_values("强弱排名")["变量"].tolist()

        data = modeling[[month_col, price_target] + predictors].copy()
        data[LAG_COL] = data[price_target].shift(1)
        model_predictors = [LAG_COL] + predictors
        frame = (
            data[[month_col, price_target] + model_predictors]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(month_col)
            .reset_index(drop=True)
        )
        model_predictors = [p for p in model_predictors if frame[p].std(ddof=0) > 0]
        frame = frame[[month_col, price_target] + model_predictors]

        fit = sm.OLS(
            frame[price_target],
            sm.add_constant(frame[model_predictors], has_constant="add"),
        ).fit()

        actual_base = modeling.loc[modeling[month_col] == ACTUAL_BASE_MONTH, price_target].dropna()
        if actual_base.empty:
            raise ValueError(f"{product} lacks actual monthly average price for {ACTUAL_BASE_MONTH:%Y-%m}")
        previous_price = float(actual_base.iloc[0])

        forecast_month = FORECAST_START
        while forecast_month <= FORECAST_END:
            values: dict[str, float] = {}
            for predictor in model_predictors:
                if predictor == LAG_COL:
                    values[predictor] = previous_price
                elif predictor == "A00铝_价格月环比" and forecast_month in a00_future_mom:
                    values[predictor] = a00_future_mom[forecast_month]
                elif is_event_dummy(predictor):
                    values[predictor] = 0.0
                else:
                    values[predictor] = latest_value_before(modeling, month_col, predictor, ACTUAL_BASE_MONTH)

            pred = float(
                fit.predict(sm.add_constant(pd.DataFrame([values]), has_constant="add")).iloc[0]
            )
            rows.append(
                {
                    "品种": product,
                    "预测月份": forecast_month,
                    "预测月均价": pred,
                    "上月价格输入": previous_price,
                    "解释变量假设": "2026-06价格为实际月均价；连续变量沿用截至2026-06最近可得值；事件冲击默认为0；上月月均价递推",
                }
            )

            if product == "A00铝":
                a00_future_mom[forecast_month] = (pred / previous_price - 1) * 100
            previous_price = pred
            forecast_month += pd.offsets.MonthBegin(1)

    return pd.DataFrame(rows)


def replace_analysis_future_sheet(future: pd.DataFrame) -> None:
    sheets = pd.read_excel(ANALYSIS_XLSX, sheet_name=None)
    sheets["未来月份预测"] = future
    with pd.ExcelWriter(ANALYSIS_XLSX, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)


def main() -> None:
    future = build_future_forecast()
    future.to_csv(FORECAST_CSV, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(FORECAST_XLSX, engine="openpyxl") as writer:
        future.to_excel(writer, sheet_name="引入变量预测结果", index=False)
    replace_analysis_future_sheet(future)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest["future_forecast_fix"] = {
        "reason": "2026-06 has complete actual daily prices, so future forecast starts from 2026-07.",
        "actual_base_month": ACTUAL_BASE_MONTH.strftime("%Y-%m"),
        "forecast_start": FORECAST_START.strftime("%Y-%m"),
        "forecast_end": FORECAST_END.strftime("%Y-%m"),
        "updated_files": [str(FORECAST_XLSX), str(FORECAST_CSV), str(ANALYSIS_XLSX)],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(future.to_string(index=False))
    print(f"wrote {FORECAST_XLSX}")
    print(f"updated {ANALYSIS_XLSX}")


if __name__ == "__main__":
    main()
