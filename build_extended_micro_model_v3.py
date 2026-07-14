import time
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


BASE_FILE = Path("industry_chain_regression_results_v1.xlsx")
OUTPUT_FILE = Path("micro_industry_regression_results_v3_extended.xlsx")
CACHE_DIR = Path("data_cache_v3")
START = "2016-01-01"
END = "2026-05-31"
NBS_REFERER = "https://data.stats.gov.cn/dg/website/page.html#/pc/national/search"
NBS_CHART_URL = (
    "https://data.stats.gov.cn/dg/website/publicrelease/web/external/getEsDataByIndicatorIdAndDa"
)


NBS_MONTHLY_INDICATORS = [
    {
        "指标": "光缆产量当期值",
        "cid": "5205fbaf0d11498c935de655720edbab",
        "id": "49b52feff1844db7890da761ff3294fa",
        "rootId": "fc982599aa684be7969d7b90b1bd0e84",
        "单位": "万芯千米",
        "来源": "国家统计局",
    },
    {
        "指标": "电线电缆光缆及电工器材制造PPI",
        "cid": "46a06d3479924949ba801c94d4c50eeb",
        "id": "c47a6d73fe3942b29e2614507c242801",
        "rootId": "fc982599aa684be7969d7b90b1bd0e84",
        "单位": "上年同月=100",
        "来源": "国家统计局",
    },
    {
        "指标": "发电量当期值",
        "cid": "1abb1cfea75847b8bf1a0e395d85966b",
        "id": "baafe3a9a09d4b39a366e5b625574aea",
        "rootId": "fc982599aa684be7969d7b90b1bd0e84",
        "单位": "亿千瓦时",
        "来源": "国家统计局",
    },
    {
        "指标": "发电机组产量当期值",
        "cid": "b4ee019fccbe45c4b3e42870d5fda9de",
        "id": "d3ff9c7560b1432686f5c1c00fcab8d3",
        "rootId": "fc982599aa684be7969d7b90b1bd0e84",
        "单位": "万千瓦",
        "来源": "国家统计局",
    },
    {
        "指标": "房间空气调节器产量当期值",
        "cid": "1236448f609140c2a430b102a1dcf10c",
        "id": "f506a50d525d4d24a3c6c86f509bd933",
        "rootId": "fc982599aa684be7969d7b90b1bd0e84",
        "单位": "万台",
        "来源": "国家统计局",
    },
    {
        "指标": "家用电冰箱产量当期值",
        "cid": "ddfe0d51f69f46f3bafeb537f9f927e7",
        "id": "ac8df8ee237e444ab607244bdc395dac",
        "rootId": "fc982599aa684be7969d7b90b1bd0e84",
        "单位": "万台",
        "来源": "国家统计局",
    },
    {
        "指标": "汽车产量当期值",
        "cid": "508bcbda312b4b1d9a6aba48f6c5c7eb",
        "id": "043652ba8bf34fd59c1092facd0c60d4",
        "rootId": "fc982599aa684be7969d7b90b1bd0e84",
        "单位": "万辆",
        "来源": "国家统计局",
    },
    {
        "指标": "新能源汽车产量当期值",
        "cid": "8dedd34d41004a03ae790313a34cb27f",
        "id": "69dea360563d42f39881a070de56dbee",
        "rootId": "fc982599aa684be7969d7b90b1bd0e84",
        "单位": "万辆",
        "来源": "国家统计局",
    },
]

NBS_ANNUAL_INDICATORS = [
    {
        "指标": "铜矿砂及其精矿进口数量",
        "cid": "df34ddd49fc64229a013d2c31249c67c",
        "id": "6f75f6184ed94caa98825d4207b9365a",
        "rootId": "884c062607104a91967b22742537f44f",
        "单位": "万吨",
        "来源": "国家统计局年度进出口数据",
        "dts": ["2016YY-2025YY"],
    },
    {
        "指标": "铜矿砂及其精矿进口金额",
        "cid": "8632761a5d5d4585914e45d2c84be1fc",
        "id": "23a482e58fc54a91b79a39be392daa12",
        "rootId": "884c062607104a91967b22742537f44f",
        "单位": "百万美元",
        "来源": "国家统计局年度进出口数据",
        "dts": ["2016YY-2025YY"],
    },
]


def month_start(s):
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


def to_monthly_last(df, date_col, value_cols):
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data["月份"] = month_start(data[date_col])
    data = data.sort_values(date_col)
    return data.groupby("月份", as_index=False)[value_cols].last()


def to_monthly_sum_last(df, date_col, sum_cols, last_cols):
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data["月份"] = month_start(data[date_col])
    sum_part = data.groupby("月份", as_index=False)[sum_cols].sum()
    last_part = data.sort_values(date_col).groupby("月份", as_index=False)[last_cols].last()
    return sum_part.merge(last_part, on="月份", how="outer")


def pct_change_cols(df, cols):
    out = df.copy()
    for col in cols:
        out[col + "_变化率"] = out[col].pct_change(fill_method=None) * 100
    return out


def diff_cols(df, cols):
    out = df.copy()
    for col in cols:
        out[col + "_变化"] = out[col].diff()
    return out


def fetch_nbs_series(indicator, dts):
    body = {
        "cid": indicator["cid"],
        "id": indicator["id"],
        "da": "000000000000",
        "dt": "",
        "rootId": indicator["rootId"],
        "dts": dts,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fp:
        json.dump(body, fp, ensure_ascii=False)
        body_path = fp.name
    try:
        cmd = [
            "curl.exe",
            "-s",
            "-L",
            "--http1.1",
            "-k",
            "-A",
            "Mozilla/5.0",
            "-e",
            NBS_REFERER,
            "-H",
            "Content-Type: application/json;charset=UTF-8",
            "--data-binary",
            f"@{body_path}",
            NBS_CHART_URL,
            "--max-time",
            "60",
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        payload = json.loads(result.stdout)
    finally:
        Path(body_path).unlink(missing_ok=True)

    if not payload.get("success"):
        raise RuntimeError(f"NBS接口返回失败: {indicator['指标']} {payload}")

    rows = []
    for item in payload.get("data", []):
        value = pd.to_numeric(item.get("v"), errors="coerce")
        rows.append(
            {
                "月份": pd.NaT if str(item.get("dt", "")).endswith("YY") else pd.to_datetime(item["dt"][:6] + "01"),
                "年份": int(item["dt"][:4]) if str(item.get("dt", "")).endswith("YY") else np.nan,
                "指标": indicator["指标"],
                "数值": value,
                "单位": item.get("unit") or indicator["单位"],
                "原始时间": item.get("dt_name", ""),
                "来源": indicator["来源"],
            }
        )
    return pd.DataFrame(rows)


def fetch_nbs_monthly_indicators():
    cache = CACHE_DIR / "nbs_monthly_indicators.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["月份"])
    print("抓取国家统计局月度电力/家电/汽车指标...")
    pieces = []
    dts = ["201601MM-202605MM"]
    for indicator in NBS_MONTHLY_INDICATORS:
        series = fetch_nbs_series(indicator, dts)
        series = series.drop(columns=["年份"])
        pieces.append(series)
        time.sleep(0.2)

    long_df = pd.concat(pieces, ignore_index=True)
    long_df.to_csv(cache, index=False, encoding="utf-8-sig")
    return long_df


def fetch_nbs_annual_indicators():
    cache = CACHE_DIR / "nbs_annual_indicators.csv"
    if cache.exists():
        return pd.read_csv(cache)
    print("抓取国家统计局铜矿砂年度数据...")
    pieces = []
    for indicator in NBS_ANNUAL_INDICATORS:
        series = fetch_nbs_series(indicator, indicator["dts"])
        series = series.drop(columns=["月份"])
        pieces.append(series)
        time.sleep(0.2)
    out = pd.concat(pieces, ignore_index=True)
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def nbs_monthly_wide():
    long_df = fetch_nbs_monthly_indicators()
    wide = long_df.pivot_table(index="月份", columns="指标", values="数值", aggfunc="last").reset_index()
    pct_cols = [
        "光缆产量当期值",
        "发电量当期值",
        "发电机组产量当期值",
        "房间空气调节器产量当期值",
        "家用电冰箱产量当期值",
        "汽车产量当期值",
        "新能源汽车产量当期值",
    ]
    wide = pct_change_cols(wide, pct_cols)
    wide = diff_cols(wide, ["电线电缆光缆及电工器材制造PPI"])
    return wide


def fetch_lme_monthly():
    import akshare as ak

    cache = CACHE_DIR / "lme_monthly.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["月份"])
    print("抓取 LME 库存/注销仓单...")
    raw = ak.macro_euro_lme_stock()
    raw = raw.rename(
        columns={
            "日期": "日期",
            "铜-库存": "LME铜库存",
            "铜-注销仓单": "LME铜注销仓单",
            "铝-库存": "LME铝库存",
            "铝-注销仓单": "LME铝注销仓单",
        }
    )
    keep = ["LME铜库存", "LME铜注销仓单", "LME铝库存", "LME铝注销仓单"]
    monthly = to_monthly_last(raw, "日期", keep)
    monthly = pct_change_cols(monthly, keep)
    monthly.to_csv(cache, index=False, encoding="utf-8-sig")
    return monthly


def fetch_shfe_futures_monthly():
    import akshare as ak

    cache = CACHE_DIR / "shfe_futures_monthly.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["月份"])
    print("抓取 SHFE 铜铝主连行情...")
    pieces = []
    for symbol, prefix in [("CU0", "SHFE铜主连"), ("AL0", "SHFE铝主连")]:
        raw = ak.futures_zh_daily_sina(symbol=symbol)
        raw = raw.rename(
            columns={
                "date": "日期",
                "volume": prefix + "成交量",
                "hold": prefix + "持仓量",
                "close": prefix + "收盘价",
            }
        )
        raw["日期"] = pd.to_datetime(raw["日期"])
        raw = raw[(raw["日期"] >= START) & (raw["日期"] <= END)]
        monthly = to_monthly_sum_last(
            raw,
            "日期",
            [prefix + "成交量"],
            [prefix + "持仓量", prefix + "收盘价"],
        )
        monthly = pct_change_cols(monthly, [prefix + "成交量", prefix + "持仓量"])
        pieces.append(monthly)

    out = pieces[0].merge(pieces[1], on="月份", how="outer")
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def fetch_shfe_warrant_monthly():
    import akshare as ak

    cache = CACHE_DIR / "shfe_warrant_monthly.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["月份"])
    print("抓取 SHFE 铜铝月末仓单库存...")
    months = pd.date_range(START, END, freq="M")

    def fetch_one(month_end):
        for offset in range(0, 12):
            day = month_end - pd.Timedelta(days=offset)
            if day.weekday() >= 5:
                continue
            try:
                data = ak.futures_shfe_warehouse_receipt(date=day.strftime("%Y%m%d"))
                cu = data.get("铜")
                al = data.get("铝")
                if cu is None or al is None:
                    continue
                cu_total = pd.to_numeric(cu.get("WRTWGHTS"), errors="coerce").sum()
                al_total = pd.to_numeric(al.get("WRTWGHTS"), errors="coerce").sum()
                return {
                    "月份": month_end.to_period("M").to_timestamp(),
                    "SHFE铜仓单库存": cu_total,
                    "SHFE铝仓单库存": al_total,
                    "SHFE仓单取数日": day.strftime("%Y-%m-%d"),
                }
            except Exception:
                continue
        return {
            "月份": month_end.to_period("M").to_timestamp(),
            "SHFE铜仓单库存": np.nan,
            "SHFE铝仓单库存": np.nan,
            "SHFE仓单取数日": "",
        }

    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch_one, m): m for m in months}
        for idx, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if idx % 20 == 0 or idx == len(months):
                print(f"  SHFE仓单进度 {idx}/{len(months)}")

    out = pd.DataFrame(rows).sort_values("月份")
    out = pct_change_cols(out, ["SHFE铜仓单库存", "SHFE铝仓单库存"])
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def fetch_gasgoo_auto_sales_monthly():
    import akshare as ak

    cache = CACHE_DIR / "gasgoo_auto_top50_monthly.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["月份"])
    print("抓取盖世汽车车企榜Top50销量代理...")
    months = pd.date_range(START, END, freq="MS")

    def fetch_one(month):
        yyyymm = month.strftime("%Y%m")
        try:
            df = ak.car_sale_rank_gasgoo(symbol="\u8f66\u4f01\u699c", date=yyyymm)
            month_col = f"{month.year}-{month.month}"
            if month_col in df.columns:
                sales = pd.to_numeric(df[month_col], errors="coerce").sum()
            else:
                sales = pd.to_numeric(df.iloc[:, 1], errors="coerce").sum()
            return {"月份": month, "汽车销量Top50厂商合计": sales}
        except Exception:
            return {"月份": month, "汽车销量Top50厂商合计": np.nan}

    rows = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(fetch_one, m): m for m in months}
        for idx, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if idx % 20 == 0 or idx == len(months):
                print(f"  汽车销量进度 {idx}/{len(months)}")
    out = pd.DataFrame(rows)
    out = pct_change_cols(out, ["汽车销量Top50厂商合计"])
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def load_base_regression_dataset():
    df = pd.read_excel(BASE_FILE, sheet_name="regression_dataset")
    df["月份"] = month_start(df["month"])
    rename = {
        "copper_price_usd_per_tonne_mom_pct": "铜价月环比",
        "aluminium_price_usd_per_tonne_mom_pct": "铝价月环比",
        "brent_usd_per_barrel_month_avg_mom_pct": "Brent原油价格月环比",
        "coal_australia_usd_per_mt_mom_pct": "动力煤价格月环比",
        "natural_gas_europe_usd_per_mmbtu_mom_pct": "天然气价格月环比",
        "freight_expenditures_index_mom_pct": "货运成本指数月环比",
        "iai_primary_aluminium_world_kt_mom_pct": "全球原铝产量月环比",
        "iai_primary_aluminium_china_estimated_kt_mom_pct": "中国原铝产量月环比",
        "iai_alumina_total_world_kt_mom_pct": "全球氧化铝产量月环比",
        "iai_alumina_total_china_estimated_kt_mom_pct": "中国氧化铝产量月环比",
        "china_real_estate_climate_index_mom_diff": "房地产景气指数变化",
        "cftc_copper_noncommercial_net_long_contracts_mom_diff": "CFTC铜非商业净多头变化",
        "cftc_copper_open_interest_contracts_mom_pct": "CFTC铜持仓量月环比",
    }
    keep = ["月份"] + list(rename.keys())
    return df[keep].rename(columns=rename)


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
    data = df[["月份", target] + predictors].replace([np.inf, -np.inf], np.nan).dropna()
    x = sm.add_constant(zscore(data[predictors]))
    y = zscore(data[target])
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    rows = []
    for var in predictors:
        coef = fit.params[var]
        p = fit.pvalues[var]
        rows.append(
            {
                "模型": model_name,
                "类别": CATEGORY.get(var, ""),
                "指标": var,
                "标准化系数": coef,
                "p值": p,
                "显著性": sig(p),
                "方向": "正向" if coef >= 0 else "负向",
            }
        )
    coef_df = pd.DataFrame(rows)
    coef_df["影响强度排名"] = coef_df["标准化系数"].abs().rank(ascending=False, method="first").astype(int)
    coef_df = coef_df.sort_values("影响强度排名")
    fit_row = {
        "模型": model_name,
        "样本量": int(fit.nobs),
        "R2": fit.rsquared,
        "调整后R2": fit.rsquared_adj,
        "样本起点": data["月份"].min().strftime("%Y-%m"),
        "样本终点": data["月份"].max().strftime("%Y-%m"),
        "入模变量数": len(predictors),
    }
    return coef_df, fit_row, data


CATEGORY = {
    "Brent原油价格月环比": "成本",
    "动力煤价格月环比": "成本",
    "天然气价格月环比": "成本",
    "货运成本指数月环比": "成本/贸易",
    "房地产景气指数变化": "需求",
    "汽车销量Top50厂商合计_变化率": "需求",
    "光缆产量当期值_变化率": "需求",
    "电线电缆光缆及电工器材制造PPI_变化": "需求/价格",
    "发电量当期值_变化率": "需求",
    "发电机组产量当期值_变化率": "需求",
    "房间空气调节器产量当期值_变化率": "需求",
    "家用电冰箱产量当期值_变化率": "需求",
    "汽车产量当期值_变化率": "需求",
    "新能源汽车产量当期值_变化率": "需求",
    "CFTC铜非商业净多头变化": "库存/期货",
    "CFTC铜持仓量月环比": "库存/期货",
    "LME铜库存_变化率": "库存/期货",
    "LME铜注销仓单_变化率": "库存/期货",
    "LME铝库存_变化率": "库存/期货",
    "LME铝注销仓单_变化率": "库存/期货",
    "SHFE铜主连成交量_变化率": "库存/期货",
    "SHFE铜主连持仓量_变化率": "库存/期货",
    "SHFE铝主连成交量_变化率": "库存/期货",
    "SHFE铝主连持仓量_变化率": "库存/期货",
    "SHFE铜仓单库存_变化率": "库存/期货",
    "SHFE铝仓单库存_变化率": "库存/期货",
    "全球原铝产量月环比": "供应",
    "中国原铝产量月环比": "供应",
    "全球氧化铝产量月环比": "供应",
    "中国氧化铝产量月环比": "供应",
}


PENDING = [
    ("铜价模型", "供应", "全球铜矿产量、全球精炼铜产量、全球铜消费量/用量", "ICSG完整月度历史通常需要订阅，未发现稳定免费接口。"),
    ("铜价模型", "供应", "铜TC/RC加工费", "Fastmarkets、SMM、Wind为主，历史数据通常付费。"),
    ("铜价模型", "需求", "国家电网投资额、光伏新增装机、风电新增装机", "国家能源局月度报告可整理，但需要批量解析累计口径，未在本轮自动完成。"),
    ("铜价模型", "需求", "电线电缆产量", "国家统计局搜索未返回普通电线电缆月度产量；已纳入光缆产量、电线电缆光缆及电工器材制造PPI、发电量和发电机组产量作为公开代理。"),
    ("铜价模型", "需求", "新能源汽车销量", "已纳入国家统计局新能源汽车产量；销量仍需中汽协/乘联会完整历史或付费库。"),
    ("铜价模型", "贸易", "铜矿砂及精矿月度进口、未锻轧铜及铜材进口、精炼铜进口", "用户给出的海关链接是站内搜索页，海关统计域名本轮访问返回504；已收集国家统计局年度铜矿砂进口数量/金额，因频率不一致未入月度回归。"),
    ("铝价模型", "供应", "全球铝土矿产量、电解铝开工率/运行产能", "铝土矿多为年度；开工率/运行产能历史多需SMM、百川、Wind。"),
    ("铝价模型", "需求", "新能源汽车销量、光伏/风电", "已纳入国家统计局汽车、新能源汽车、空调、冰箱、发电量和发电机组产量；销量和新能源装机需继续从中汽协/乘联会/国家能源局累计口径整理。"),
    ("铝价模型", "成本", "工业电价、氧化铝价格、铝土矿价格、EUA碳价", "工业电价和铝土矿多需付费/人工；氧化铝期货2023年才开始，按用户要求排除。"),
    ("铝价模型", "贸易", "铝土矿进口、氧化铝进口、未锻轧铝及铝材进出口", "海关HS细项需要海关平台批量查询或付费库；本轮海关统计域名访问返回504，未获得稳定免密接口。"),
]


def coverage_table(df):
    rows = []
    for col in df.columns:
        if col == "月份":
            continue
        valid = df.dropna(subset=[col])
        rows.append(
            {
                "指标": col,
                "非空月份数": len(valid),
                "覆盖率": round(len(valid) / len(df) * 100, 1),
                "起始月份": valid["月份"].min().strftime("%Y-%m") if len(valid) else "",
                "结束月份": valid["月份"].max().strftime("%Y-%m") if len(valid) else "",
            }
        )
    return pd.DataFrame(rows)


def main():
    CACHE_DIR.mkdir(exist_ok=True)
    base = load_base_regression_dataset()
    lme = fetch_lme_monthly()
    shfe_fut = fetch_shfe_futures_monthly()
    shfe_warrant = fetch_shfe_warrant_monthly()
    auto_sales = fetch_gasgoo_auto_sales_monthly()
    nbs_monthly = nbs_monthly_wide()
    nbs_annual = fetch_nbs_annual_indicators()

    df = base.merge(lme, on="月份", how="left").merge(shfe_fut, on="月份", how="left").merge(
        shfe_warrant, on="月份", how="left"
    ).merge(auto_sales, on="月份", how="left").merge(nbs_monthly, on="月份", how="left")
    df = df[(df["月份"] >= pd.Timestamp(START)) & (df["月份"] <= pd.Timestamp("2026-05-01"))]

    copper_predictors = [
        "CFTC铜非商业净多头变化",
        "CFTC铜持仓量月环比",
        "LME铜库存_变化率",
        "LME铜注销仓单_变化率",
        "SHFE铜主连成交量_变化率",
        "SHFE铜主连持仓量_变化率",
        "SHFE铜仓单库存_变化率",
        "Brent原油价格月环比",
        "动力煤价格月环比",
        "天然气价格月环比",
        "货运成本指数月环比",
        "房地产景气指数变化",
        "汽车销量Top50厂商合计_变化率",
        "光缆产量当期值_变化率",
        "电线电缆光缆及电工器材制造PPI_变化",
        "发电量当期值_变化率",
        "发电机组产量当期值_变化率",
        "房间空气调节器产量当期值_变化率",
        "家用电冰箱产量当期值_变化率",
        "汽车产量当期值_变化率",
        "新能源汽车产量当期值_变化率",
    ]
    aluminium_predictors = [
        "全球原铝产量月环比",
        "中国原铝产量月环比",
        "全球氧化铝产量月环比",
        "中国氧化铝产量月环比",
        "LME铝库存_变化率",
        "LME铝注销仓单_变化率",
        "SHFE铝主连成交量_变化率",
        "SHFE铝主连持仓量_变化率",
        "SHFE铝仓单库存_变化率",
        "Brent原油价格月环比",
        "动力煤价格月环比",
        "天然气价格月环比",
        "货运成本指数月环比",
        "房地产景气指数变化",
        "汽车销量Top50厂商合计_变化率",
        "发电量当期值_变化率",
        "发电机组产量当期值_变化率",
        "房间空气调节器产量当期值_变化率",
        "家用电冰箱产量当期值_变化率",
        "汽车产量当期值_变化率",
        "新能源汽车产量当期值_变化率",
    ]

    copper_coef, copper_fit, copper_data = run_model(df, "铜价模型", "铜价月环比", copper_predictors)
    aluminium_coef, aluminium_fit, aluminium_data = run_model(df, "铝价模型", "铝价月环比", aluminium_predictors)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        copper_data.to_excel(writer, sheet_name="铜模型数据", index=False)
        aluminium_data.to_excel(writer, sheet_name="铝模型数据", index=False)
        pd.DataFrame([copper_fit, aluminium_fit]).to_excel(writer, sheet_name="模型拟合度", index=False)
        copper_coef.to_excel(writer, sheet_name="铜模型系数", index=False)
        aluminium_coef.to_excel(writer, sheet_name="铝模型系数", index=False)
        coverage_table(df).to_excel(writer, sheet_name="数据覆盖情况", index=False)
        fetch_nbs_monthly_indicators().to_excel(writer, sheet_name="NBS月度原始数据", index=False)
        nbs_annual.to_excel(writer, sheet_name="NBS年度铜矿砂数据", index=False)
        pd.DataFrame(PENDING, columns=["模型", "类别", "未入模指标", "原因"]).to_excel(
            writer, sheet_name="未入模指标说明", index=False
        )
        df.to_excel(writer, sheet_name="合并后全量数据", index=False)

    print(OUTPUT_FILE.resolve())
    print(pd.DataFrame([copper_fit, aluminium_fit]).to_string(index=False))
    print("\n铜模型系数")
    print(copper_coef[["类别", "指标", "标准化系数", "p值", "显著性", "方向"]].to_string(index=False))
    print("\n铝模型系数")
    print(aluminium_coef[["类别", "指标", "标准化系数", "p值", "显著性", "方向"]].to_string(index=False))


if __name__ == "__main__":
    main()
