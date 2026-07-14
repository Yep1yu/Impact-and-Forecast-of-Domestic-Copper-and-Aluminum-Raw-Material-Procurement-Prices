from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


CACHE_DIR = Path("data_cache_v3")
OUTPUT_FILE = CACHE_DIR / "domestic_macro_monthly.csv"


def parse_cn_month(series: pd.Series) -> pd.Series:
    text = series.astype(str)
    year = text.str.extract(r"(\d{4})")[0]
    month = text.str.extract(r"年(\d{1,2})月")[0]
    return pd.to_datetime(year + "-" + month.str.zfill(2) + "-01", errors="coerce")


def month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def clean_numeric(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in exclude:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fetch_domestic_macro_monthly() -> pd.DataFrame:
    import akshare as ak

    frames: list[pd.DataFrame] = []

    pmi = ak.macro_china_pmi()
    pmi["月份"] = parse_cn_month(pmi["月份"])
    frames.append(
        pmi.rename(
            columns={
                "制造业-指数": "制造业PMI",
                "制造业-同比增长": "制造业PMI同比增长",
                "非制造业-指数": "非制造业PMI",
                "非制造业-同比增长": "非制造业PMI同比增长",
            }
        )[["月份", "制造业PMI", "制造业PMI同比增长", "非制造业PMI", "非制造业PMI同比增长"]]
    )

    real_estate = ak.macro_china_real_estate()
    real_estate["月份"] = month_start(real_estate["日期"])
    frames.append(
        real_estate.rename(
            columns={
                "最新值": "房地产开发景气指数",
                "涨跌幅": "房地产开发景气指数环比",
                "近1年涨跌幅": "房地产开发景气指数近1年涨跌幅",
            }
        )[["月份", "房地产开发景气指数", "房地产开发景气指数环比", "房地产开发景气指数近1年涨跌幅"]]
    )

    industrial = ak.macro_china_gyzjz()
    industrial["月份"] = parse_cn_month(industrial["月份"])
    frames.append(
        industrial.rename(
            columns={
                "同比增长": "工业增加值同比增长",
                "累计增长": "工业增加值累计增长",
            }
        )[["月份", "工业增加值同比增长", "工业增加值累计增长"]]
    )

    ppi = ak.macro_china_ppi()
    ppi["月份"] = parse_cn_month(ppi["月份"])
    frames.append(
        ppi.rename(
            columns={
                "当月": "PPI当月指数",
                "当月同比增长": "PPI当月同比增长",
                "累计": "PPI累计指数",
            }
        )[["月份", "PPI当月指数", "PPI当月同比增长", "PPI累计指数"]]
    )

    trade = ak.macro_china_hgjck()
    trade["月份"] = parse_cn_month(trade["月份"])
    frames.append(
        trade.rename(
            columns={
                "当月出口额-金额": "当月出口额",
                "当月出口额-同比增长": "当月出口额同比增长",
                "当月出口额-环比增长": "当月出口额环比增长",
                "当月进口额-金额": "当月进口额",
                "当月进口额-同比增长": "当月进口额同比增长",
                "当月进口额-环比增长": "当月进口额环比增长",
            }
        )[
            [
                "月份",
                "当月出口额",
                "当月出口额同比增长",
                "当月出口额环比增长",
                "当月进口额",
                "当月进口额同比增长",
                "当月进口额环比增长",
            ]
        ]
    )

    commodity = ak.macro_china_qyspjg()
    commodity["月份"] = parse_cn_month(commodity["月份"])
    frames.append(
        commodity.rename(
            columns={
                "总指数-指数值": "企业商品价格总指数",
                "总指数-同比增长": "企业商品价格总指数同比增长",
                "总指数-环比增长": "企业商品价格总指数环比增长",
                "矿产品-指数值": "企业商品价格矿产品指数",
                "矿产品-同比增长": "企业商品价格矿产品同比增长",
                "矿产品-环比增长": "企业商品价格矿产品环比增长",
                "煤油电-指数值": "企业商品价格煤油电指数",
                "煤油电-同比增长": "企业商品价格煤油电同比增长",
                "煤油电-环比增长": "企业商品价格煤油电环比增长",
            }
        )[
            [
                "月份",
                "企业商品价格总指数",
                "企业商品价格总指数同比增长",
                "企业商品价格总指数环比增长",
                "企业商品价格矿产品指数",
                "企业商品价格矿产品同比增长",
                "企业商品价格矿产品环比增长",
                "企业商品价格煤油电指数",
                "企业商品价格煤油电同比增长",
                "企业商品价格煤油电环比增长",
            ]
        ]
    )

    out = pd.concat([frame[["月份"]] for frame in frames]).dropna().drop_duplicates().sort_values("月份")
    for frame in frames:
        frame = clean_numeric(frame, {"月份"})
        out = out.merge(frame, on="月份", how="left")
    return out.sort_values("月份").reset_index(drop=True)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = fetch_domestic_macro_monthly()
    data.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(OUTPUT_FILE.resolve())
    print(data.shape)
    print(data.head(3).to_string(index=False))
    print(data.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
