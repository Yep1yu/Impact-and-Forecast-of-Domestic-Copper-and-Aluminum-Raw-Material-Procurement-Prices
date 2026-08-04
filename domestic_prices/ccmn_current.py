from __future__ import annotations

import base64
import json
import time
from datetime import date
from typing import Any

import requests
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


CURRENT_URL = "https://www.ccmn.cn/shop/historyData/getCjysAndCjysw"
AES_KEY = b"ccmnCjysw1455881"
CURRENT_MARKET = "长江现货"
PRODUCTS = (
    "1#铜",
    "A00铝",
    "1#白银",
    "铝合金ADC12",
    "铸造铝合金锭(ZLD104)",
)


def _decrypt_json(value: str) -> Any:
    encrypted = base64.b64decode(value)
    decryptor = Cipher(algorithms.AES(AES_KEY), modes.ECB()).decryptor()
    padded = decryptor.update(encrypted) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    raw = unpadder.update(padded) + unpadder.finalize()
    return json.loads(raw.decode("utf-8"))


def fetch_current_prices(
    url: str = CURRENT_URL,
    timeout_seconds: int = 30,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    client = session or requests.Session()
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://www.ccmn.cn/",
        "User-Agent": "Mozilla/5.0 (compatible; MetalPulse/1.0)",
    }
    for attempt in range(3):
        try:
            response = client.get(url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    payload = response.json()
    if not payload.get("success") or not isinstance(payload.get("body"), dict):
        raise RuntimeError(f"CCMN current price request failed: {payload.get('msg') or payload}")

    prices = _decrypt_json(payload["body"]["showPriceList"])
    if not isinstance(prices, list):
        raise ValueError("CCMN current price list is not a list")

    selected: dict[str, dict[str, Any]] = {}
    for table in prices:
        if not isinstance(table, list):
            continue
        for item in table:
            if not isinstance(item, dict) or item.get("marketName") != CURRENT_MARKET:
                continue
            product = str(item.get("productSortName") or "").strip()
            if product not in PRODUCTS:
                continue
            if item.get("avgPrice") in (None, ""):
                continue
            selected[product] = item

    if not selected:
        raise ValueError("CCMN current price response contains no supported products")
    dates = {str(item.get("publishDate") or "").strip() for item in selected.values()}
    trade_date = max((value for value in dates if value), default=date.today().isoformat())
    return {
        "date": trade_date,
        "items": selected,
        "source": "CCMN Changjiang public current quote",
    }


def current_row(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"date": payload["date"]}
    for product in PRODUCTS:
        item = payload["items"].get(product)
        if item is not None:
            row[product] = item.get("avgPrice")
    return row
