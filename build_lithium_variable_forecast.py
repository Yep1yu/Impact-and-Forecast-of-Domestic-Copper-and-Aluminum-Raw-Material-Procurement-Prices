from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from domestic_prices.db import connect, initialize, replace_latest_forecasts, replace_latest_monthly_forecasts, upsert_spot_prices
from domestic_prices.model import generate_forecasts


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path(r"D:\Wechat\xwechat_files\wxid_i1mfkj939nq911_dfaf\msg\file\2026-07")
DATASET = ROOT / "domestic_material_monthly_dataset_v1.csv"
OUTPUT_DIR = ROOT / "lithium_carbonate_prediction_outputs"
METAL = "lithium_carbonate"
DISPLAY_NAME = "碳酸锂"
MODEL_VERSION = "lithium-variable-monthly-daily-v1"
DAILY_MODEL_VERSION = "lithium-daily-hybrid-price-path-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lithium settlement-price variable regression, monthly and daily forecasts.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--database", type=Path, default=ROOT / "domestic_procurement_prices.sqlite")
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--skip-db", action="store_true")
    return parser.parse_args()


def load_settlement_prices(input_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob("LCFUTURES*.xlsx")):
        frame = pd.read_excel(path, header=1)
        if frame.shape[1] < 15:
            raise ValueError(f"{path} does not look like an LC futures workbook")
        frame = frame.rename(
            columns={
                frame.columns[0]: "date",
                frame.columns[3]: "contract",
                frame.columns[9]: "settlement_price",
                frame.columns[12]: "volume",
                frame.columns[13]: "open_interest",
                frame.columns[14]: "open_interest_change",
            }
        )
        frame["date"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d", errors="coerce")
        for column in ["settlement_price", "volume", "open_interest", "open_interest_change"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame[["date", "contract", "settlement_price", "volume", "open_interest", "open_interest_change"]])
    if not frames:
        raise FileNotFoundError(f"No LCFUTURES*.xlsx files found in {input_dir}")
    raw = pd.concat(frames, ignore_index=True).dropna(subset=["date", "settlement_price", "volume"])
    # Main contract is the contract with the largest volume on each trade date.
    main = raw.sort_values(["date", "volume"], ascending=[True, False]).drop_duplicates("date")
    main = main.sort_values("date").reset_index(drop=True)
    main["month"] = main["date"].dt.to_period("M").dt.to_timestamp()
    main["price_source"] = "GFEX LC main contract settlement price"
    return main


def load_monthly_factors() -> pd.DataFrame:
    data = pd.read_csv(DATASET, encoding="utf-8-sig")
    data = data.rename(columns={data.columns[0]: "month"})
    data["month"] = pd.to_datetime(data["month"], errors="coerce")
    return data.sort_values("month").reset_index(drop=True)


def candidate_columns(data: pd.DataFrame) -> list[str]:
    excluded_words = ("月均价", "价格月环比", "价差", "收盘价", "SHFE")
    columns = []
    for column in data.columns:
        if column == "month" or any(word in column for word in excluded_words):
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            columns.append(column)
    return columns


def standardize(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if isinstance(std, pd.Series):
        return (series - series.mean()) / std.replace(0, np.nan)
    return (series - series.mean()) / std if std and np.isfinite(std) else series * np.nan


def regression_screen(monthly_target: pd.DataFrame, factors: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    merged = monthly_target.merge(factors, on="month", how="left")
    for column in candidate_columns(factors):
        frame = merged[["month", "target_price", column]].copy()
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 18 or frame[column].std(ddof=0) == 0:
            continue
        fit = sm.OLS(standardize(frame["target_price"]), sm.add_constant(standardize(frame[[column]]))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 2}
        )
        rows.append(
            {
                "品种": DISPLAY_NAME,
                "目标变量": "碳酸锂月均结算价",
                "变量": column,
                "样本数": int(fit.nobs),
                "标准化系数": float(fit.params[column]),
                "影响强度_绝对值": abs(float(fit.params[column])),
                "p值": float(fit.pvalues[column]),
                "显著性": significance(float(fit.pvalues[column])),
                "方向": "正向" if fit.params[column] >= 0 else "负向",
            }
        )
    screening = pd.DataFrame(rows).sort_values(["p值", "影响强度_绝对值"]).reset_index(drop=True)
    significant = screening.loc[screening["p值"] < 0.1, "变量"].tolist()[:8]
    if not significant:
        significant = screening.head(5)["变量"].tolist()
    return screening, significant


def significance(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return "不显著"


def fit_ols(frame: pd.DataFrame, target: str, predictors: list[str]):
    usable = [column for column in predictors if column in frame and frame[column].std(ddof=0) > 0]
    model = frame[[target] + usable].replace([np.inf, -np.inf], np.nan).dropna()
    if len(model) < max(12, len(usable) + 5) or not usable:
        raise ValueError("Insufficient complete observations for variable model")
    x = sm.add_constant(model[usable], has_constant="add")
    fit = sm.OLS(model[target], x).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    return fit, model, usable


def monthly_model(main: pd.DataFrame, factors: pd.DataFrame, predictors: list[str]):
    target = main.groupby("month", as_index=False)["settlement_price"].mean().rename(columns={"settlement_price": "target_price"})
    merged = target.merge(factors[["month"] + predictors], on="month", how="left")
    for column in predictors:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").ffill().bfill()
    fit, model_frame, usable = fit_ols(merged, "target_price", predictors)
    fitted = model_frame[["target_price"]].copy()
    fitted["month"] = merged.loc[model_frame.index, "month"].values
    fitted["predicted_monthly_price"] = fit.predict(sm.add_constant(model_frame[usable], has_constant="add")).values
    latest_month = target["month"].max()
    future_months = pd.date_range(latest_month + pd.offsets.MonthBegin(1), "2026-12-01", freq="MS")
    future_input = extrapolate_factors(factors, predictors, future_months)
    future_input["predicted_monthly_price"] = fit.predict(sm.add_constant(future_input[usable], has_constant="add")).values
    return target, fit, fitted, future_input, usable


def extrapolate_factors(factors: pd.DataFrame, predictors: list[str], months: pd.DatetimeIndex) -> pd.DataFrame:
    """Forecast exogenous variables from their recent trend instead of freezing them."""
    rows = pd.DataFrame({"month": months})
    for column in predictors:
        series = factors[["month", column]].copy()
        series[column] = pd.to_numeric(series[column], errors="coerce")
        series = series.dropna().sort_values("month").tail(6).reset_index(drop=True)
        if series.empty:
            rows[column] = np.nan
            continue
        if len(series) < 3 or series[column].nunique() <= 1:
            rows[column] = float(series[column].iloc[-1])
            continue
        x = np.arange(len(series), dtype=float)
        slope, intercept = np.polyfit(x, series[column].to_numpy(dtype=float), 1)
        steps = np.arange(len(series), len(series) + len(months), dtype=float)
        values = intercept + slope * steps
        # Prevent extrapolation from producing a sign-inconsistent value for rate/level factors.
        last = float(series[column].iloc[-1])
        if last >= 0:
            values = np.maximum(values, 0.0)
        rows[column] = values
    return rows


def daily_variable_model_legacy(main: pd.DataFrame, factors: pd.DataFrame, predictors: list[str], monthly_forecast: pd.DataFrame, forecast_days: int):
    daily = main[["date", "month", "settlement_price", "contract", "volume", "open_interest", "open_interest_change"]].copy()
    factor_daily = factors[["month"] + predictors].copy()
    for column in predictors:
        factor_daily[column] = pd.to_numeric(factor_daily[column], errors="coerce").ffill().bfill()
    daily = daily.merge(factor_daily, on="month", how="left")
    for column in predictors:
        daily[column] = daily[column].ffill().bfill()
    fit, model_frame, usable = fit_ols(daily.rename(columns={"settlement_price": "target_price"}), "target_price", predictors)
    actual = daily.loc[model_frame.index, ["date", "settlement_price"]].copy()
    actual["predicted_price"] = fit.predict(sm.add_constant(model_frame[usable], has_constant="add")).values
    actual["price_error"] = actual["predicted_price"] - actual["settlement_price"]
    last_date = daily["date"].max()
    future_dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=forecast_days)
    future_months = pd.date_range(
        future_dates.min().to_period("M").to_timestamp(),
        future_dates.max().to_period("M").to_timestamp() + pd.offsets.MonthBegin(1),
        freq="MS",
    )
    future_factor_values = extrapolate_factors(factors, predictors, future_months)
    future = pd.DataFrame([{ "date": date, "month": date.to_period("M").to_timestamp()} for date in future_dates])
    future = future.merge(future_factor_values, on="month", how="left")
    # Expand monthly factor paths to daily values so daily predictions can move
    # within a month instead of becoming a monthly step function.
    future_factor_values = future_factor_values[future_factor_values["month"] > last_date].copy()
    anchor_dates = pd.DatetimeIndex([last_date, *future_factor_values["month"].tolist()])
    for column in predictors:
        last_value = float(daily[column].dropna().iloc[-1])
        anchor_values = np.array([last_value, *future_factor_values[column].to_numpy(dtype=float)])
        future[column] = np.interp(
            future["date"].astype("int64").to_numpy(),
            anchor_dates.astype("int64"),
            anchor_values,
        )
    future["raw_predicted_price"] = fit.predict(sm.add_constant(future[usable], has_constant="add")).values
    future = future.merge(monthly_forecast[["month", "predicted_monthly_price"]], on="month", how="left")
    for month, group in future.groupby("month"):
        mask = future["month"] == month
        target = group["predicted_monthly_price"].iloc[0]
        if pd.notna(target):
            future.loc[mask, "predicted_price"] = future.loc[mask, "raw_predicted_price"] + float(target) - group["raw_predicted_price"].mean()
        else:
            future.loc[mask, "predicted_price"] = future.loc[mask, "raw_predicted_price"]
    future["predicted_price"] = future["predicted_price"].clip(lower=1.0)
    return daily, fit, actual, future, usable


def daily_model(main: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
    """Use the shared price-path model used by the other raw materials."""
    spot = main[["date", "settlement_price", "contract"]].rename(
        columns={"date": "trade_date", "settlement_price": "price_cny_per_tonne", "contract": "raw_symbol"}
    )
    spot["metal"] = METAL
    spot["source"] = "GFEX LC main contract settlement price"
    forecast = generate_forecasts(
        spot[["trade_date", "metal", "price_cny_per_tonne", "source", "raw_symbol"]],
        pd.DataFrame(),
        forecast_days=forecast_days,
        model_version=DAILY_MODEL_VERSION,
    )
    return forecast.rename(columns={"forecast_date": "date", "predicted_price_cny_per_tonne": "predicted_price"})


def coefficient_frame(fit, predictors: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "品种": DISPLAY_NAME,
                "变量": "截距" if name == "const" else name,
                "系数": float(value),
                "p值": float(fit.pvalues[name]),
                "显著性": significance(float(fit.pvalues[name])),
            }
            for name, value in fit.params.items()
        ]
    )


def import_to_website_db(main: pd.DataFrame, future: pd.DataFrame, monthly_forecast: pd.DataFrame, database: Path) -> None:
    conn = connect(database)
    initialize(conn)
    spot = main[["date", "settlement_price"]].rename(columns={"date": "trade_date", "settlement_price": "price_cny_per_tonne"})
    spot["metal"] = METAL
    spot["source"] = "GFEX LC main contract settlement price"
    spot["raw_symbol"] = main["contract"].values
    upsert_spot_prices(conn, spot[["trade_date", "metal", "price_cny_per_tonne", "source", "raw_symbol"]])
    generated = pd.Timestamp.utcnow().isoformat()
    forecast = pd.DataFrame(
        {
            "metal": METAL,
            "forecast_date": future["date"],
            "predicted_price_cny_per_tonne": future["predicted_price"].round(2),
            "lower_bound": future["lower_bound"].round(2),
            "upper_bound": future["upper_bound"].round(2),
            "direction": future["direction"],
            "model_version": DAILY_MODEL_VERSION,
            "generated_at": generated,
        }
    )
    conn.execute("DELETE FROM daily_forecasts WHERE metal = ?", (METAL,))
    conn.commit()
    replace_latest_forecasts(conn, forecast, DAILY_MODEL_VERSION)
    monthly_rows = pd.DataFrame(
        {
            "metal": METAL,
            "forecast_month": monthly_forecast["month"],
            "predicted_price_cny_per_tonne": monthly_forecast["predicted_monthly_price"].round(2),
            "source": "lithium significant-variable monthly model",
            "model_version": MODEL_VERSION,
            "generated_at": generated,
        }
    )
    replace_latest_monthly_forecasts(conn, monthly_rows, MODEL_VERSION)
    conn.close()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main_prices = load_settlement_prices(args.input_dir)
    factors = load_monthly_factors()
    monthly_target = main_prices.groupby("month", as_index=False)["settlement_price"].mean().rename(columns={"settlement_price": "target_price"})
    screening, predictors = regression_screen(monthly_target, factors)
    target, monthly_fit, monthly_fitted, monthly_forecast, usable = monthly_model(main_prices, factors, predictors)
    daily_forecast = daily_model(main_prices, args.forecast_days)

    screening.to_csv(OUTPUT_DIR / "lithium_impact_regression_screening.csv", index=False, encoding="utf-8-sig")
    coefficient_frame(monthly_fit, usable).to_csv(OUTPUT_DIR / "lithium_monthly_model_coefficients.csv", index=False, encoding="utf-8-sig")
    monthly_fitted.to_csv(OUTPUT_DIR / "lithium_monthly_fitted_prices.csv", index=False, encoding="utf-8-sig")
    monthly_forecast.to_csv(OUTPUT_DIR / "lithium_monthly_forecast.csv", index=False, encoding="utf-8-sig")
    daily_forecast.to_csv(OUTPUT_DIR / "lithium_daily_forecast.csv", index=False, encoding="utf-8-sig")
    for obsolete in ["lithium_daily_model_coefficients.csv", "lithium_daily_fitted_prices.csv"]:
        (OUTPUT_DIR / obsolete).unlink(missing_ok=True)
    report_path = OUTPUT_DIR / "lithium_variable_forecast_report.xlsx"
    try:
        writer = pd.ExcelWriter(report_path, engine="openpyxl")
    except PermissionError:
        report_path = OUTPUT_DIR / "lithium_variable_forecast_report_updated.xlsx"
        writer = pd.ExcelWriter(report_path, engine="openpyxl")
    with writer:
        screening.to_excel(writer, sheet_name="影响变量回归", index=False)
        monthly_fitted.to_excel(writer, sheet_name="月度拟合", index=False)
        monthly_forecast.to_excel(writer, sheet_name="月度预测", index=False)
        daily_forecast.to_excel(writer, sheet_name="日度预测", index=False)
    summary = {
        "metal": METAL,
        "price_definition": "每个交易日成交量最大合约的结算价",
        "sample_start": str(main_prices.date.min().date()),
        "sample_end": str(main_prices.date.max().date()),
        "significant_variables": predictors,
        "monthly_model": "月度结算价 ~ 显著外生变量；不含滞后价格",
        "daily_model": "共享日度价格路径模型：滞后价格、移动均线、多步 Ridge 与单步幅度限制；不使用月初平移校准",
        "monthly_model_version": MODEL_VERSION,
        "daily_model_version": DAILY_MODEL_VERSION,
    }
    (OUTPUT_DIR / "lithium_variable_forecast_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.skip_db:
        import_to_website_db(main_prices, daily_forecast, monthly_forecast, args.database)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
