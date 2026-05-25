"""Step 1: Build the stock universe from local grouped industry files or SEC EDGAR.

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
from utils.market_data import get_market_caps_eastmoney


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_HEADERS = {"User-Agent": "PairsTradingPipeline research@example.com"}


def _normalize_local_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df


def _normalize_cik(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _column_or_default(df: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def load_local_grouped_industry_universe(
    grouped_dir: str | Path,
    min_market_cap: float,
) -> pd.DataFrame:
    grouped_path = Path(grouped_dir)
    if not grouped_path.exists():
        raise FileNotFoundError(f"Local grouped industry directory not found: {grouped_path}")

    csv_paths = sorted(
        path for path in grouped_path.glob("*.csv")
        if not path.name.startswith("_")
    )
    if not csv_paths:
        raise FileNotFoundError(f"No industry CSV files found in: {grouped_path}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        local_df = _normalize_local_columns(pd.read_csv(csv_path))
        if "primary_ticker" not in local_df.columns:
            raise KeyError(f"Missing primary_ticker column in {csv_path}")

        local_df = local_df.rename(
            columns={
                "primary_ticker": "ticker",
                "Industry Plate": "industry_plate",
                "Original Industry Plate": "original_industry_plate",
                "total_market_cap_usd": "market_cap_usd",
            }
        )

        if "company_tickers_title" in local_df.columns:
            company_name = local_df["company_tickers_title"]
        elif "edgar_name" in local_df.columns:
            company_name = local_df["edgar_name"]
        else:
            company_name = local_df["ticker"]

        local_df["ticker"] = local_df["ticker"].astype(str).str.strip().str.upper()
        local_df["cik"] = _column_or_default(local_df, "cik", "").map(_normalize_cik)
        local_df["company_name"] = company_name.astype(str).str.strip()
        local_df["industry_plate"] = (
            _column_or_default(local_df, "industry_plate", csv_path.stem)
            .fillna(csv_path.stem)
            .astype(str)
            .str.strip()
        )
        local_df["original_industry_plate"] = (
            _column_or_default(local_df, "original_industry_plate", "")
            .fillna(local_df["industry_plate"])
            .astype(str)
            .str.strip()
        )
        local_df["market_cap_usd"] = pd.to_numeric(
            _column_or_default(local_df, "market_cap_usd", pd.NA),
            errors="coerce",
        )

        frames.append(
            local_df[
                [
                    "cik",
                    "ticker",
                    "company_name",
                    "market_cap_usd",
                    "industry_plate",
                    "original_industry_plate",
                ]
            ]
        )

    df = pd.concat(frames, ignore_index=True)
    df = df[df["ticker"].str.len() > 0].copy()
    df = df.dropna(subset=["market_cap_usd"])
    df = df[df["market_cap_usd"] >= min_market_cap].copy()
    df = df.sort_values(["market_cap_usd", "ticker"], ascending=[False, True])

    duplicate_count = int(df["ticker"].duplicated().sum())
    if duplicate_count:
        print(f"[WARN] Found {duplicate_count} duplicate tickers in local industry files; keeping highest market cap row.")
        df = df.drop_duplicates(subset="ticker", keep="first")

    df = df.reset_index(drop=True)
    print(
        f"Loaded {len(df)} stocks across {df['industry_plate'].nunique()} industries "
        f"from local grouped industry files."
    )
    return df


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
    print(f"Fetching market caps for {len(tickers)} tickers from Eastmoney...")
    market_caps = get_market_caps_eastmoney(tickers, sleep_time=sleep_time)
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
    local_grouped_dir = str(universe_cfg.get("local_grouped_industry_dir", "")).strip()

    if local_grouped_dir:
        print(f"Loading universe from local grouped industry directory: {local_grouped_dir}")
        df = load_local_grouped_industry_universe(local_grouped_dir, min_cap)
    else:
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
