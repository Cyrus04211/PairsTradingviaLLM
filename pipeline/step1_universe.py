"""Step 1: Fetch all US stocks from SEC EDGAR and filter by market cap.

Outputs: outputs/universe.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_io import write_csv
from utils.market_data import get_market_caps_yfinance


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_HEADERS = {"User-Agent": "PairsTradingPipeline research@example.com"}


def fetch_sec_tickers(url: str = SEC_TICKERS_URL) -> pd.DataFrame:
    print("Fetching company tickers from SEC EDGAR...")
    resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for entry in data.values():
        rows.append({
            "cik": str(entry.get("cik_str", "")),
            "ticker": str(entry.get("ticker", "")).strip().upper(),
            "company_name": str(entry.get("title", "")),
        })
    df = pd.DataFrame(rows)
    df = df[df["ticker"].str.len() > 0].drop_duplicates(subset="ticker")
    print(f"Loaded {len(df)} tickers from SEC EDGAR.")
    return df


def filter_by_market_cap(
    df: pd.DataFrame,
    min_market_cap: float,
    sleep_time: float = 0.3,
) -> pd.DataFrame:
    tickers = df["ticker"].tolist()
    print(f"Fetching market caps for {len(tickers)} tickers (this may take a while)...")
    market_caps = get_market_caps_yfinance(tickers, sleep_time=sleep_time)
    df = df.copy()
    df["market_cap_usd"] = df["ticker"].map(market_caps)
    df = df.dropna(subset=["market_cap_usd"])
    df = df[df["market_cap_usd"] >= min_market_cap].copy()
    df = df.sort_values("market_cap_usd", ascending=False).reset_index(drop=True)
    print(f"After market cap filter (>= ${min_market_cap/1e9:.1f}B): {len(df)} stocks remain.")
    return df


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    config = yaml.safe_load(Path(config_path).read_text())
    universe_cfg = config.get("universe", {})
    min_cap = float(universe_cfg.get("min_market_cap_usd", 5_000_000_000))
    sec_url = universe_cfg.get("sec_tickers_url", SEC_TICKERS_URL)

    df = fetch_sec_tickers(sec_url)
    df = filter_by_market_cap(df, min_cap)

    output_path = Path(output_dir) / "universe.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Universe saved to {output_path} ({len(df)} stocks).")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Build US stock universe")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
