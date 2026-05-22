"""Market data fetching utilities using yfinance."""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import yfinance as yf
from tqdm import tqdm


def get_market_caps_yfinance(tickers: list[str], sleep_time: float = 0.5) -> dict[str, float]:
    market_caps: dict[str, float] = {}
    for ticker in tqdm(tickers, desc="Fetching market caps"):
        try:
            info = yf.Ticker(ticker).info
            cap = info.get("marketCap")
            if cap and float(cap) > 0:
                market_caps[ticker] = float(cap)
        except Exception:
            pass
        time.sleep(sleep_time)
    return market_caps


def download_price_history(
    tickers: list[str],
    period: str = "5y",
    sleep_time: float = 1.0,
) -> pd.DataFrame:
    all_close: dict[str, pd.Series] = {}
    for ticker in tqdm(tickers, desc="Downloading prices"):
        try:
            data = yf.download(ticker, period=period, progress=False)
            if not data.empty:
                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                all_close[ticker] = close
        except Exception:
            pass
        time.sleep(sleep_time)
    if not all_close:
        return pd.DataFrame()
    return pd.DataFrame(all_close).sort_index()
