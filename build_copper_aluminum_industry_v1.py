import io
import os
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests


START = "2016-01-01"
END = "2026-05-31"
MONTHS = pd.date_range("2016-01-01", "2026-05-01", freq="MS")


def get_with_retry(url, *, params=None, headers=None, timeout=60, tries=4):
    last_error = None
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise last_error


def month_start(series):
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def fetch_fred(series_id, monthly="native"):
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {
        "id": series_id,
        "observation_start": START,
        "observation_end": END,
    }
    resp = get_with_retry(url, params=params, timeout=60)
    df = pd.read_csv(io.StringIO(resp.text))
    date_col = df.columns[0]
    value_col = df.columns[-1]
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    df["month"] = month_start(df[date_col])
    if monthly == "mean":
        return df.groupby("month", as_index=False)[value_col].mean()
    if monthly == "last":
        return df.sort_values(date_col).groupby("month", as_index=False)[value_col].last()
    return df[["month", value_col]]


def parse_cn_month(series):
    text = series.astype(str)
    return pd.to_datetime(
        text.str.extract(r"(\d{4})")[0] + "-"
        + text.str.extract(r"年(\d{1,2})月")[0].str.zfill(2)
        + "-01",
        errors="coerce",
    )


def fetch_china_macro():
    import akshare as ak

    result = []

    pmi = ak.macro_china_pmi()
    pmi["month"] = parse_cn_month(pmi["月份"])
    result.append(
        pmi.rename(
            columns={
                "制造业-指数": "china_manufacturing_pmi",
                "非制造业-指数": "china_nonmanufacturing_pmi",
            }
        )[["month", "china_manufacturing_pmi", "china_nonmanufacturing_pmi"]]
    )

    gyzjz = ak.macro_china_gyzjz()
    gyzjz["month"] = parse_cn_month(gyzjz["月份"])
    result.append(
        gyzjz.rename(
            columns={
                "同比增长": "china_industrial_value_added_yoy_pct",
                "累计增长": "china_industrial_value_added_ytd_yoy_pct",
            }
        )[
            [
                "month",
                "china_industrial_value_added_yoy_pct",
                "china_industrial_value_added_ytd_yoy_pct",
            ]
        ]
    )

    ppi = ak.macro_china_ppi()
    ppi["month"] = parse_cn_month(ppi["月份"])
    result.append(
        ppi.rename(
            columns={
                "当月": "china_ppi_index",
                "当月同比增长": "china_ppi_yoy_pct",
                "累计": "china_ppi_ytd_index",
            }
        )[["month", "china_ppi_index", "china_ppi_yoy_pct", "china_ppi_ytd_index"]]
    )

    estate = ak.macro_china_real_estate()
    estate["month"] = month_start(estate["日期"])
    result.append(
        estate.rename(columns={"最新值": "china_real_estate_climate_index"})[
            ["month", "china_real_estate_climate_index"]
        ]
    )

    trade = ak.macro_china_hgjck()
    trade["month"] = parse_cn_month(trade["月份"])
    result.append(
        trade.rename(
            columns={
                "当月出口额-金额": "china_total_exports_current_usd",
                "当月进口额-金额": "china_total_imports_current_usd",
                "当月出口额-同比增长": "china_total_exports_yoy_pct",
                "当月进口额-同比增长": "china_total_imports_yoy_pct",
            }
        )[
            [
                "month",
                "china_total_exports_current_usd",
                "china_total_imports_current_usd",
                "china_total_exports_yoy_pct",
                "china_total_imports_yoy_pct",
            ]
        ]
    )

    out = pd.DataFrame({"month": MONTHS})
    for df in result:
        out = out.merge(df, on="month", how="left")
    return out


def fetch_iai_publication(slug):
    token = (
        "VqJoChv3cGZei872eHVKUL4kdbk3CG2qw5RUpq8eV4VmMCbCJxncfzOyCo3nknz"
        "59qoWzPjVsPFffSULSWceeWAuywurxWiRVXdkqADVfKSvItSkOstAcU8yoiL6Hmr6"
    )
    url = "https://alvis.international-aluminium.org/api/publication/"
    resp = get_with_retry(
        url,
        params={"publication": slug},
        headers={"X-AUTH-TOKEN": token, "Accept": "application/json"},
        timeout=60,
    )
    return resp.json()["data"]["publication"]


def get_cell_value(period_item, row_id, col_id):
    row_data = period_item.get("data", {}).get(str(row_id))
    if not row_data:
        return None
    cell = row_data.get(str(col_id))
    if not cell:
        return None
    return cell.get("value")


def fetch_iai_aluminium_supply():
    frames = []

    primary = fetch_iai_publication("primary-aluminium-production")
    primary_cols = [
        c["id"]
        for c in primary["charts"]["publicationCharts"][0]["columnsToUse"]
        if c["id"] != 106
    ]
    rows = []
    for item in primary["charts"]["data"]:
        values = [get_cell_value(item, 85, col_id) for col_id in primary_cols]
        rows.append(
            {
                "month": pd.to_datetime(item["period"]["from"]).to_period("M").to_timestamp(),
                "iai_primary_aluminium_world_kt": sum(v for v in values if v is not None),
                "iai_primary_aluminium_china_estimated_kt": get_cell_value(item, 85, 9),
            }
        )
    frames.append(pd.DataFrame(rows))

    alumina = fetch_iai_publication("alumina-production")
    alumina_cols = [
        c["id"]
        for c in alumina["charts"]["publicationCharts"][0]["columnsToUse"]
        if c["id"] != 71
    ]
    rows = []
    for item in alumina["charts"]["data"]:
        total_values = [get_cell_value(item, 87, col_id) for col_id in alumina_cols]
        rows.append(
            {
                "month": pd.to_datetime(item["period"]["from"]).to_period("M").to_timestamp(),
                "iai_alumina_total_world_kt": sum(v for v in total_values if v is not None),
                "iai_alumina_total_china_estimated_kt": get_cell_value(item, 87, 19),
            }
        )
    frames.append(pd.DataFrame(rows))

    out = pd.DataFrame({"month": MONTHS})
    for df in frames:
        out = out.merge(df, on="month", how="left")
    return out


def fetch_cftc_copper_net_long():
    frames = []
    for year in range(2016, 2027):
        url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
        resp = get_with_retry(url, timeout=60)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = zf.namelist()[0]
            frames.append(pd.read_csv(zf.open(name)))
        time.sleep(0.2)
    df = pd.concat(frames, ignore_index=True)
    market = df["Market and Exchange Names"].astype(str)
    df = df[
        market.str.contains("COPPER", case=False, na=False)
        & market.str.contains("COMMODITY EXCHANGE", case=False, na=False)
        & ~market.str.contains("MICRO", case=False, na=False)
    ]
    df["month"] = month_start(df["As of Date in Form YYYY-MM-DD"])
    df["cftc_copper_noncommercial_net_long_contracts"] = (
        pd.to_numeric(df["Noncommercial Positions-Long (All)"], errors="coerce")
        - pd.to_numeric(df["Noncommercial Positions-Short (All)"], errors="coerce")
    )
    df["cftc_copper_open_interest_contracts"] = pd.to_numeric(
        df["Open Interest (All)"], errors="coerce"
    )
    return (
        df.groupby("month", as_index=False)[
            [
                "cftc_copper_noncommercial_net_long_contracts",
                "cftc_copper_open_interest_contracts",
            ]
        ]
        .mean()
        .sort_values("month")
    )


def add_changes(raw):
    change = raw[["month"]].copy()
    level_pct_cols = [
        "copper_price_usd_per_tonne",
        "aluminium_price_usd_per_tonne",
        "brent_usd_per_barrel_month_avg",
        "wti_usd_per_barrel_month_avg",
        "coal_australia_usd_per_mt",
        "natural_gas_europe_usd_per_mmbtu",
        "freight_expenditures_index",
        "iai_primary_aluminium_world_kt",
        "iai_primary_aluminium_china_estimated_kt",
        "iai_alumina_total_world_kt",
        "iai_alumina_total_china_estimated_kt",
        "china_total_exports_current_usd",
        "china_total_imports_current_usd",
        "cftc_copper_open_interest_contracts",
    ]
    diff_cols = [
        "china_manufacturing_pmi",
        "china_nonmanufacturing_pmi",
        "china_industrial_value_added_yoy_pct",
        "china_ppi_yoy_pct",
        "china_real_estate_climate_index",
        "cftc_copper_noncommercial_net_long_contracts",
    ]
    for col in level_pct_cols:
        if col in raw:
            change[col + "_mom_pct"] = raw[col].pct_change() * 100
    for col in diff_cols:
        if col in raw:
            change[col + "_mom_diff"] = raw[col].diff()
    return change


def build_dataset():
    raw = pd.DataFrame({"month": MONTHS})

    fred_specs = {
        "PCOPPUSDM": ("copper_price_usd_per_tonne", "native"),
        "PALUMUSDM": ("aluminium_price_usd_per_tonne", "native"),
        "DCOILBRENTEU": ("brent_usd_per_barrel_month_avg", "mean"),
        "DCOILWTICO": ("wti_usd_per_barrel_month_avg", "mean"),
        "PCOALAUUSDM": ("coal_australia_usd_per_mt", "native"),
        "PNGASEUUSDM": ("natural_gas_europe_usd_per_mmbtu", "native"),
        "FRGEXPUSM649NCIS": ("freight_expenditures_index", "native"),
        "DTWEXBGS": ("us_dollar_broad_index_month_avg", "mean"),
        "FEDFUNDS": ("fed_funds_rate_pct", "native"),
        "DGS10": ("us_10y_treasury_yield_pct_month_avg", "mean"),
        "VIXCLS": ("vix_month_avg", "mean"),
    }
    for sid, (name, how) in fred_specs.items():
        try:
            df = fetch_fred(sid, how).rename(columns={sid: name})
            raw = raw.merge(df, on="month", how="left")
        except Exception as exc:
            raw[f"fetch_error_fred_{sid}"] = str(exc)

    for fetcher in [fetch_china_macro, fetch_iai_aluminium_supply, fetch_cftc_copper_net_long]:
        try:
            raw = raw.merge(fetcher(), on="month", how="left")
        except Exception as exc:
            raw[f"fetch_error_{fetcher.__name__}"] = str(exc)

    raw = raw[(raw["month"] >= MONTHS.min()) & (raw["month"] <= MONTHS.max())]
    return raw


def coverage_summary(raw):
    rows = []
    for col in raw.columns:
        if col == "month":
            continue
        valid = raw.dropna(subset=[col])
        rows.append(
            {
                "variable": col,
                "non_null_months": len(valid),
                "coverage_pct": round(len(valid) / len(raw) * 100, 1),
                "first_month": valid["month"].min().strftime("%Y-%m") if len(valid) else None,
                "last_month": valid["month"].max().strftime("%Y-%m") if len(valid) else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["coverage_pct", "variable"], ascending=[False, True])


def data_dictionary():
    rows = [
        ("copper_price_usd_per_tonne", "price", "FRED/IMF PCOPPUSDM", "monthly", "Target variable: global copper price"),
        ("aluminium_price_usd_per_tonne", "price", "FRED/IMF PALUMUSDM", "monthly", "Target variable: global aluminium price"),
        ("iai_primary_aluminium_world_kt", "supply", "IAI", "monthly", "World primary aluminium production, thousand tonnes"),
        ("iai_primary_aluminium_china_estimated_kt", "supply", "IAI", "monthly", "China estimated primary aluminium production, thousand tonnes"),
        ("iai_alumina_total_world_kt", "supply", "IAI", "monthly", "World total alumina production, thousand tonnes"),
        ("iai_alumina_total_china_estimated_kt", "supply", "IAI", "monthly", "China estimated total alumina production, thousand tonnes"),
        ("china_industrial_value_added_yoy_pct", "demand", "Eastmoney via AkShare", "monthly", "China industrial value added YoY"),
        ("china_manufacturing_pmi", "demand", "Eastmoney via AkShare", "monthly", "China manufacturing PMI"),
        ("china_real_estate_climate_index", "demand", "Eastmoney via AkShare", "monthly", "China real estate climate index"),
        ("china_ppi_yoy_pct", "cost", "Eastmoney via AkShare", "monthly", "China PPI YoY"),
        ("brent_usd_per_barrel_month_avg", "cost", "FRED DCOILBRENTEU", "daily to monthly average", "Brent oil price"),
        ("wti_usd_per_barrel_month_avg", "cost", "FRED DCOILWTICO", "daily to monthly average", "WTI oil price"),
        ("coal_australia_usd_per_mt", "cost", "FRED/World Bank PCOALAUUSDM", "monthly", "Australia coal price"),
        ("natural_gas_europe_usd_per_mmbtu", "cost", "FRED/World Bank PNGASEUUSDM", "monthly", "European natural gas price"),
        ("freight_expenditures_index", "cost/trade", "FRED FRGEXPUSM649NCIS", "monthly", "Freight expenditures proxy"),
        ("china_total_exports_current_usd", "trade", "Eastmoney via AkShare", "monthly", "China total exports; not metal-specific"),
        ("china_total_imports_current_usd", "trade", "Eastmoney via AkShare", "monthly", "China total imports; not metal-specific"),
        ("cftc_copper_noncommercial_net_long_contracts", "inventory/futures", "CFTC COT", "weekly to monthly average", "COMEX copper non-commercial long minus short"),
        ("cftc_copper_open_interest_contracts", "inventory/futures", "CFTC COT", "weekly to monthly average", "COMEX copper open interest"),
    ]
    return pd.DataFrame(rows, columns=["variable", "dimension", "source", "frequency", "note"])


def pending_indicators():
    rows = [
        ("global_copper_mine_production", "supply", "ICSG monthly data is suitable but full history is subscription-based; use ICSG/Wind/CEIC if available."),
        ("global_refined_copper_production", "supply", "ICSG monthly data is suitable but full history is subscription-based."),
        ("global_refined_copper_usage", "demand", "ICSG monthly data is suitable but full history is subscription-based."),
        ("copper_tc_rc", "cost/supply", "Fastmarkets/SMM/Wind history is usually paid."),
        ("shfe_cu_al_inventory", "inventory", "Public interfaces found but current wrapper only returned recent data; needs direct SHFE historical batch extraction."),
        ("lme_cu_al_inventory_cancelled_warrants", "inventory", "Official historical LME data is paid or needs manual report archive extraction."),
        ("china_metal_specific_customs", "trade", "Need HS-code batch from China Customs/paid database; UN Comtrade now requires API key."),
        ("new_energy_vehicle_sales", "demand", "Current CPCA interface returned only recent two years; needs CAAM/CEIC/Wind historical table."),
        ("appliance_output_air_conditioner_refrigerator", "demand", "Need NBS indicator-code extraction; NBS API was unstable in this run."),
        ("pv_wind_new_installation_grid_investment", "demand", "Can be built from NEA monthly cumulative reports; needs report scraping and differencing."),
        ("industrial_electricity_price", "cost", "Public monthly China industrial electricity price is not stable; likely manual/Wind."),
        ("alumina_price_bauxite_price", "cost", "Long history mainly SMM/Wind/Baichuan paid; SHFE alumina futures starts too late for 2016."),
    ]
    return pd.DataFrame(rows, columns=["indicator", "dimension", "status"])


def main():
    output_dir = Path(os.environ.get("OUTPUT_DIR", ".")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "copper_aluminum_industry_chain_dataset_v1.xlsx"

    raw = build_dataset()
    changes = add_changes(raw)
    coverage = coverage_summary(raw)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        raw.to_excel(writer, sheet_name="monthly_raw_v1", index=False)
        changes.to_excel(writer, sheet_name="monthly_change_v1", index=False)
        coverage.to_excel(writer, sheet_name="coverage_summary", index=False)
        data_dictionary().to_excel(writer, sheet_name="data_dictionary", index=False)
        pending_indicators().to_excel(writer, sheet_name="pending_indicators", index=False)

    print(out_path)
    print(raw.shape)
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
