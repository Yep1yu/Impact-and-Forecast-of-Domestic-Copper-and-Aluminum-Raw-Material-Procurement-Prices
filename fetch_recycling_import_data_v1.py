from __future__ import annotations

import math
import re
import time
from html import unescape
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


CACHE_DIR = Path("data_cache_v3")
OUTPUT_FILE = CACHE_DIR / "recycling_import_monthly.csv"
SEARCH_URL = "https://news.smm.cn/search"
HEADERS = {"User-Agent": "Mozilla/5.0"}

SERIES = {
    "废铜进口量": {
        "keyword": "废铜进口量",
        "volume_patterns": [
            r"中国(?P<year>\d{4})年(?P<month>\d{1,2})月.*?废铜（铜废碎料）进口量.*?为(?P<value>[\d,]+(?:\.\d+)?)吨",
            r"中国(?P<year>\d{4})年(?P<month>\d{1,2})月.*?废铜\(铜废碎料\)进口量.*?为(?P<value>[\d,]+(?:\.\d+)?)吨",
            r"中国(?P<year>\d{4})年(?P<month>\d{1,2})月.*?废铜进口量.*?为(?P<value>[\d,]+(?:\.\d+)?)吨",
            r"(?P<year>\d{4})年(?P<month>\d{1,2})月.*?铜废碎料进口量.*?为(?P<value>[\d,]+(?:\.\d+)?)吨",
        ],
    },
    "废铝进口量": {
        "keyword": "废铝进口量",
        "volume_patterns": [
            r"中国(?P<year>\d{4})年(?P<month>\d{1,2})月.*?废铝进口量.*?为(?P<value>[\d,]+(?:\.\d+)?)吨",
            r"(?P<year>\d{4})年(?P<month>\d{1,2})月.*?铝废料及碎料.*?进口量.*?为(?P<value>[\d,]+(?:\.\d+)?)吨",
        ],
    },
}


def get_html(url: str, *, params: dict[str, str] | None = None) -> str:
    response = requests.get(url, params=params, timeout=30, headers=HEADERS, proxies={"http": None, "https": None})
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def extract_next_data(html: str) -> dict:
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, flags=re.S)
    if not match:
        return {}
    import json

    return json.loads(match.group(1))


def clean_html_text(html: str) -> str:
    text = BeautifulSoup(unescape(html or ""), "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def search_news(keyword: str, max_pages: int = 20) -> list[dict]:
    rows: list[dict] = []
    total = None
    for page in range(1, max_pages + 1):
        data = extract_next_data(get_html(SEARCH_URL, params={"keywords": keyword, "page": str(page)}))
        props = data.get("props", {}).get("pageProps", {}).get("searchListProps", {})
        if total is None:
            total = int(props.get("length") or 0)
        news_list = props.get("newsListList") or []
        if not news_list:
            break
        rows.extend(news_list)
        if total and page >= math.ceil(total / 10):
            break
        time.sleep(0.2)
    seen = set()
    out = []
    for row in rows:
        news_id = row.get("newsId")
        news_url = row.get("newsUrl") or f"https://news.smm.cn/news/{news_id}"
        if news_id in seen or "/news/" not in news_url:
            continue
        seen.add(news_id)
        out.append({"news_id": news_id, "news_url": news_url, "title": clean_html_text(row.get("title", ""))})
    return out


def fetch_news_detail(news_url: str) -> dict:
    data = extract_next_data(get_html(news_url))
    detail = data.get("props", {}).get("pageProps", {}).get("newsDetail", {})
    return {
        "title": clean_html_text(detail.get("title", "")),
        "date": detail.get("date", ""),
        "content": clean_html_text(detail.get("content", "")),
    }


def parse_volume(series_name: str, text: str) -> dict[str, object] | None:
    for pattern in SERIES[series_name]["volume_patterns"]:
        match = re.search(pattern, text)
        if not match:
            continue
        year = int(match.group("year"))
        month = int(match.group("month"))
        value = float(match.group("value").replace(",", ""))
        return {"月份": pd.Timestamp(year=year, month=month, day=1), series_name: value}
    return None


def fetch_series(series_name: str) -> pd.DataFrame:
    items = search_news(SERIES[series_name]["keyword"])
    rows = []
    for item in items:
        try:
            detail = fetch_news_detail(item["news_url"])
        except Exception:
            continue
        parsed = parse_volume(series_name, f"{detail['title']} {detail['content']}")
        if parsed:
            parsed.update(
                {
                    "数据项": series_name,
                    "新闻标题": detail["title"] or item["title"],
                    "新闻日期": detail["date"],
                    "来源URL": item["news_url"],
                    "数据来源": "SMM新闻/海关统计数据在线查询平台",
                }
            )
            rows.append(parsed)
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame(columns=["月份", series_name])
    df = pd.DataFrame(rows).sort_values("月份")
    df = df.drop_duplicates(["月份"], keep="last")
    return df


def add_changes(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.sort_values("月份").copy()
    for col in ["废铜进口量", "废铝进口量"]:
        if col not in out:
            continue
        out[f"{col}_变化"] = out[col].diff()
        out[f"{col}_环比"] = out[col].pct_change(fill_method=None) * 100
    return out


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    long_frames = [fetch_series(name) for name in SERIES]
    source = pd.concat(long_frames, ignore_index=True)
    wide = source.pivot_table(index="月份", values=list(SERIES.keys()), aggfunc="last").reset_index()
    wide = add_changes(wide)
    wide.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    source.to_csv(CACHE_DIR / "recycling_import_monthly_sources.csv", index=False, encoding="utf-8-sig")
    print(OUTPUT_FILE.resolve())
    print(wide.shape)
    print(wide.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
