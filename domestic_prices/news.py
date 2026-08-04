from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "daily_news.json"
CCMN_URL = "https://www.ccmn.cn/news/"
SMM_URL = "https://news.metal.com/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MetalPulse/1.0)"}
MATERIAL_KEYWORDS = (
    "铜",
    "铝",
    "白银",
    "碳酸锂",
    "锂",
    "copper",
    "aluminum",
    "aluminium",
    "silver",
    "lithium",
    "adc12",
    "zld104",
)


def _clean(value: object) -> str:
    text = BeautifulSoup(str(value or ""), "html.parser").get_text(" ", strip=True)
    return " ".join(text.split()).strip()


def _image_url(node: Any, base_url: str) -> str:
    image = node.find("img")
    if image is None:
        return ""
    source = image.get("src") or image.get("data-src") or image.get("data-original") or ""
    if not source:
        return ""
    source_text = str(source).strip()
    lowered = source_text.lower()
    if any(marker in lowered for marker in ("/icon/", "adbar", "logo", "avatar")):
        return ""
    return urljoin(base_url, source_text)


def _class_text(node: Any, fragment: str) -> str:
    for child in node.find_all(True):
        classes = " ".join(child.get("class", []))
        if fragment in classes:
            text = _clean(child.get_text(" ", strip=True))
            if text:
                return text
    return ""


def _parse_ccmn(html: bytes) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", ""))
        title = _clean(anchor.get("title") or anchor.get_text(" ", strip=True))
        if not href.startswith("/news/") or not href.lower().endswith(".html") or len(title) < 8:
            continue
        url = urljoin(CCMN_URL, href)
        if url in seen:
            continue
        seen.add(url)
        summary = ""
        parent = anchor.parent
        if parent is not None and "bulletion_new" in parent.get("class", []):
            detail = parent.find_next_sibling("p", class_="bulletion_detial")
            if detail is not None:
                summary = _clean(detail.get_text(" ", strip=True))
        rows.append(
            {
                "source": "长江有色",
                "title": title,
                "summary": summary[:180],
                "published": "",
                "url": url,
                "image_url": _image_url(anchor, CCMN_URL),
            }
        )
    return rows


def _parse_smm(html: bytes) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    image_by_url: dict[str, str] = {}
    for image_anchor in soup.find_all("a", href=True):
        href = str(image_anchor.get("href", ""))
        if "/en/newscontent/" not in href:
            continue
        image_url = _image_url(image_anchor, SMM_URL)
        if image_url:
            image_by_url.setdefault(urljoin(SMM_URL, href), image_url)
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", ""))
        if "/en/newscontent/" not in href:
            continue
        url = urljoin(SMM_URL, href)
        title = _class_text(anchor, "__title") or _clean(anchor.get_text(" ", strip=True))
        summary = _class_text(anchor, "__summary")
        published = _class_text(anchor, "__time")
        if len(title) < 8:
            continue
        if url in seen:
            continue
        seen.add(url)
        rows.append(
            {
                "source": "SMM",
                "title": title,
                "summary": summary[:180],
                "published": published,
                "url": url,
                "image_url": _image_url(anchor, SMM_URL) or image_by_url.get(url, ""),
            }
        )
    return rows


def _relevance_score(item: dict[str, str]) -> int:
    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    score = sum(3 for keyword in MATERIAL_KEYWORDS if keyword.lower() in text)
    # 同等相关度下优先保留来源页面提供的缩略图，避免资讯卡片全部没有视觉预览。
    return score + (3 if item.get("image_url") else 0)


def select_relevant_news(items: list[dict[str, str]], limit: int = 5) -> list[dict[str, str]]:
    scored = [(index, _relevance_score(item), item) for index, item in enumerate(items)]
    relevant = [row for row in scored if row[1] > 0]
    relevant.sort(key=lambda row: (-row[1], row[0]))
    selected: list[dict[str, str]] = []
    sources: set[str] = set()
    titles: set[str] = set()
    for _, _, item in relevant:
        title_key = item["title"].casefold()
        if item["source"] not in sources and title_key not in titles:
            selected.append(item)
            sources.add(item["source"])
            titles.add(title_key)
    for _, _, item in relevant:
        if len(selected) >= limit:
            break
        title_key = item["title"].casefold()
        if title_key not in titles:
            selected.append(item)
            titles.add(title_key)
    return selected[:limit]


def fetch_daily_news(limit: int = 5) -> tuple[list[dict[str, str]], list[str]]:
    items: list[dict[str, str]] = []
    errors: list[str] = []
    for source, url, parser in (
        ("长江有色", CCMN_URL, _parse_ccmn),
        ("SMM", SMM_URL, _parse_smm),
    ):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            items.extend(parser(response.content))
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    return select_relevant_news(items, limit), errors


def write_news_cache(path: Path, items: list[dict[str, str]], errors: list[str] | None = None) -> None:
    now = datetime.now(timezone(timedelta(hours=8))).replace(microsecond=0).isoformat()
    payload = {
        "updated_at": now,
        "items": items,
        "errors": errors or [],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_news_cache(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if not path.exists():
        return {"updated_at": "", "items": [], "errors": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"updated_at": "", "items": [], "errors": []}
    return payload if isinstance(payload, dict) else {"updated_at": "", "items": [], "errors": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch relevant daily metals news.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    items, errors = fetch_daily_news(max(3, min(args.limit, 5)))
    if not items and args.output.exists():
        print("No new relevant news; kept the existing cache.")
        for error in errors:
            print(f"warning: {error}")
        return
    write_news_cache(args.output, items, errors)
    print(f"Saved {len(items)} news items to {args.output}")
    for error in errors:
        print(f"warning: {error}")


if __name__ == "__main__":
    main()
