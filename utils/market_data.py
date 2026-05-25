"""Market data fetching utilities using Eastmoney and akshare."""
from __future__ import annotations

import random
import time

import akshare as ak
import pandas as pd
import requests
from tqdm import tqdm


EASTMONEY_URL = "https://72.push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
    "Connection": "close",
    "Referer": "https://quote.eastmoney.com/",
}
EASTMONEY_BASE_PARAMS = {
    "pz": "100",
    "po": "1",
    "np": "1",
    "ut": "bd1d9ddb04089700cf9c27f6f7426281",
    "fltt": "2",
    "invt": "2",
    "fid": "f20",
    "fs": "m:105,m:106,m:107",
    "fields": "f12,f13,f14,f2,f20",
}


def fetch_market_cap_page(page_number: int, max_retries: int = 6) -> list[dict]:
    last_exception = None
    for attempt in range(max_retries):
        try:
            params = dict(EASTMONEY_BASE_PARAMS)
            params["pn"] = str(page_number)
            with requests.Session() as session:
                response = session.get(
                    EASTMONEY_URL,
                    params=params,
                    headers=EASTMONEY_HEADERS,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json().get("data", {}).get("diff", [])
                return data or []
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            if attempt == max_retries - 1:
                break
            wait_seconds = min(10.0, 1.5 * (2**attempt)) + random.uniform(0.1, 0.8)
            time.sleep(wait_seconds)
    if last_exception is not None:
        raise last_exception
    return []


def get_market_caps_eastmoney(
    tickers: list[str],
    sleep_time: float = 0.3,
) -> dict[str, float]:
    targets = {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
    if not targets:
        return {}

    market_caps: dict[str, float] = {}
    page_number = 1
    while True:
        page_data = fetch_market_cap_page(page_number)
        if not page_data:
            break

        for item in page_data:
            ticker = str(item.get("f12") or "").strip().upper()
            market_cap = pd.to_numeric(item.get("f20"), errors="coerce")
            if ticker in targets and pd.notna(market_cap) and float(market_cap) > 0:
                market_caps[ticker] = float(market_cap)

        if len(market_caps) >= len(targets):
            break

        page_number += 1
        time.sleep(max(0.0, sleep_time))

    return market_caps


def _normalize_akshare_daily(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = raw_df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "date" not in df.columns:
        df = df.reset_index()
        df.columns = [str(col).strip().lower() for col in df.columns]

    rename_map = {}
    if "vol" in df.columns and "volume" not in df.columns:
        rename_map["vol"] = "volume"
    if "datetime" in df.columns and "date" not in df.columns:
        rename_map["datetime"] = "date"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"AKShare daily history missing columns: {missing}")

    normalized = df.loc[:, required].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = normalized.dropna(subset=["date"])
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["close"])
    normalized = normalized.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    normalized = normalized.set_index("date")
    return normalized


def fetch_daily_history_akshare(
    ticker: str,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            return _normalize_akshare_daily(ak.stock_us_daily(symbol=ticker, adjust=""))
        except Exception as exc:  # noqa: BLE001
            last_exception = exc
            if attempt == max_retries:
                break
            time.sleep(max(0.0, retry_sleep_seconds) * attempt)
    if last_exception is not None:
        raise last_exception
    return pd.DataFrame()


def download_price_history(
    tickers: list[str],
    min_history_days: int = 1200,
    sleep_time: float = 0.2,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    all_close: dict[str, pd.Series] = {}
    target_rows = max(0, int(min_history_days)) + 1

    for ticker in tqdm(tickers, desc="Downloading prices"):
        try:
            data = fetch_daily_history_akshare(
                str(ticker).strip().upper(),
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
            )
            if not data.empty:
                close = data["close"]
                if target_rows > 0:
                    close = close.tail(target_rows)
                all_close[str(ticker).strip().upper()] = close
        except Exception:
            pass
        time.sleep(max(0.0, sleep_time))

    if not all_close:
        return pd.DataFrame()
    return pd.DataFrame(all_close).sort_index()
