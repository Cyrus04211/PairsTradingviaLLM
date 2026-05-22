"""Step 6: Return neutralization and price correlation filtering.

Pipeline:
  1. Download price history for all stocks in the universe
  2. Neutralize returns by removing industry, size (SMB), and volatility factors
  3. Compute multi-window correlation on neutralized residuals
  4. Filter pairs by minimum correlation threshold

Outputs: outputs/return_correlation.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_io import write_csv
from utils.market_data import download_price_history

def build_industry_factor(
    returns: pd.DataFrame,
    ticker: str,
    industry_map: dict[str, str],
    market_caps: dict[str, float],
) -> pd.Series:
    """Market-cap weighted industry return excluding the target ticker."""
    plate = industry_map.get(ticker, "")
    peers = [
        t for t, p in industry_map.items()
        if p == plate and t != ticker and t in returns.columns
    ]
    if not peers:
        return pd.Series(0.0, index=returns.index)
    weights = np.array([max(market_caps.get(t, 1.0), 1.0) for t in peers])
    weights = weights / weights.sum()
    peer_returns = returns[peers].fillna(0.0).values
    return pd.Series((peer_returns * weights).sum(axis=1), index=returns.index)


def build_smb_factor(returns: pd.DataFrame, market_caps: dict[str, float]) -> pd.Series:
    """Fama-French SMB factor: small minus big market cap portfolio returns."""
    tickers = [t for t in returns.columns if market_caps.get(t, 0) > 0]
    if len(tickers) < 20:
        return pd.Series(0.0, index=returns.index)
    tickers_sorted = sorted(tickers, key=lambda t: market_caps.get(t, 0))
    n = len(tickers_sorted)
    small = tickers_sorted[:int(n * 0.3)]
    big = tickers_sorted[int(n * 0.7):]
    small_ret = returns[small].mean(axis=1)
    big_ret = returns[big].mean(axis=1)
    return small_ret - big_ret


def rolling_volatility(returns: pd.Series, window: int = 120) -> pd.Series:
    """Rolling standard deviation as volatility exposure."""
    return returns.rolling(window, min_periods=max(20, window // 2)).std().shift(1)


def neutralize_returns_oos(
    asset_returns: pd.Series,
    factors: dict[str, pd.Series],
    window: int = 120,
) -> pd.Series:
    """Out-of-sample neutralization using rolling OLS."""
    aligned = pd.DataFrame({"asset": asset_returns})
    for name, series in factors.items():
        aligned[name] = series
    aligned = aligned.dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    factor_names = list(factors.keys())
    y = aligned["asset"].values
    X = aligned[factor_names].values
    residuals = np.full(len(aligned), np.nan)

    for t in range(window, len(aligned)):
        train_slice = slice(max(0, t - window), t)
        X_train = np.column_stack([np.ones(t - max(0, t - window)), X[train_slice]])
        y_train = y[train_slice]
        try:
            beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
            x_t = np.concatenate([[1.0], X[t]])
            residuals[t] = y[t] - np.dot(x_t, beta)
        except Exception:
            continue

    return pd.Series(residuals, index=aligned.index)


def compute_pair_correlation(
    resid_a: pd.Series,
    resid_b: pd.Series,
    windows: list[int],
) -> dict[str, float]:
    """Compute correlation metrics across multiple windows."""
    aligned = pd.DataFrame({"a": resid_a, "b": resid_b}).dropna()
    if len(aligned) < 60:
        return {}

    metrics: dict[str, float] = {}
    for w in windows:
        subset = aligned.tail(w)
        if len(subset) < 60:
            continue
        corr = subset["a"].corr(subset["b"])
        rolling_corr = subset["a"].rolling(60, min_periods=30).corr(subset["b"])
        metrics[f"corr_{w}d"] = round(float(corr), 4) if np.isfinite(corr) else 0.0
        metrics[f"corr_{w}d_median"] = round(float(rolling_corr.median()), 4) if not rolling_corr.isna().all() else 0.0
        metrics[f"corr_{w}d_std"] = round(float(rolling_corr.std()), 4) if not rolling_corr.isna().all() else 1.0

    return metrics


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    config = yaml.safe_load(Path(config_path).read_text())
    corr_cfg = config.get("return_correlation", {})
    neut_window = int(corr_cfg.get("neutralization_window", 120))
    corr_windows = corr_cfg.get("correlation_windows", [252, 504, 1000])
    min_corr = float(corr_cfg.get("min_correlation", 0.3))
    price_days = int(corr_cfg.get("price_history_days", 1200))

    # Load universe and text similarity results
    universe_path = Path(output_dir) / "universe_with_industry.csv"
    text_sim_path = Path(output_dir) / "text_similarity.csv"
    if not universe_path.exists():
        raise FileNotFoundError(f"Not found: {universe_path}. Run step2 first.")
    if not text_sim_path.exists():
        raise FileNotFoundError(f"Not found: {text_sim_path}. Run step5 first.")

    universe_df = pd.read_csv(universe_path)
    text_sim_df = pd.read_csv(text_sim_path)
    industry_map = dict(zip(universe_df["ticker"], universe_df["industry_plate"]))
    market_caps = dict(zip(universe_df["ticker"], universe_df["market_cap_usd"]))

    # Get unique tickers from text similarity pairs
    tickers_needed = set(text_sim_df["ticker_a"].tolist() + text_sim_df["ticker_b"].tolist())
    tickers_list = sorted(tickers_needed)
    print(f"Need price data for {len(tickers_list)} tickers...")

    # Check for cached prices
    price_cache = Path(output_dir) / "price_cache.parquet"
    if price_cache.exists():
        print("Loading cached price data...")
        close_df = pd.read_parquet(price_cache)
        missing = [t for t in tickers_list if t not in close_df.columns]
        if missing:
            print(f"Downloading {len(missing)} missing tickers...")
            new_prices = download_price_history(missing, period=f"{price_days // 252 + 1}y")
            if not new_prices.empty:
                close_df = close_df.join(new_prices, how="outer")
                close_df.to_parquet(price_cache)
    else:
        print("Downloading price history (this will take a while)...")
        close_df = download_price_history(tickers_list, period=f"{price_days // 252 + 1}y")
        if close_df.empty:
            raise RuntimeError("Failed to download any price data")
        close_df.to_parquet(price_cache)
        print(f"Cached prices to {price_cache}")

    # Compute returns
    returns = close_df.pct_change().iloc[1:]
    smb_factor = build_smb_factor(returns, market_caps)

    # Neutralize returns for each ticker
    print("Neutralizing returns...")
    residuals: dict[str, pd.Series] = {}
    for ticker in tqdm(tickers_list, desc="Neutralizing"):
        if ticker not in returns.columns:
            continue
        asset_ret = returns[ticker].dropna()
        if len(asset_ret) < neut_window + 30:
            continue
        industry_factor = build_industry_factor(returns, ticker, industry_map, market_caps)
        vol_exposure = rolling_volatility(returns[ticker], window=neut_window)
        factors = {
            "industry": industry_factor,
            "size": smb_factor,
            "vol_exposure": vol_exposure,
        }
        resid = neutralize_returns_oos(asset_ret, factors, window=neut_window)
        if not resid.empty:
            residuals[ticker] = resid

    print(f"Neutralized {len(residuals)} tickers.")

    # Compute correlations for text-similar pairs
    print(f"Computing correlations for {len(text_sim_df)} pairs...")
    results = []
    for _, row in tqdm(text_sim_df.iterrows(), total=len(text_sim_df), desc="Correlations"):
        ticker_a = str(row["ticker_a"])
        ticker_b = str(row["ticker_b"])
        if ticker_a not in residuals or ticker_b not in residuals:
            continue

        metrics = compute_pair_correlation(residuals[ticker_a], residuals[ticker_b], corr_windows)
        if not metrics:
            continue

        # Use the shortest window correlation as primary filter
        primary_corr_key = f"corr_{min(corr_windows)}d"
        primary_corr = metrics.get(primary_corr_key, 0.0)
        if primary_corr < min_corr:
            continue

        result = {
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "industry_plate": row.get("industry_plate", ""),
            "overlap_score": row.get("overlap_score", 0.0),
        }
        result.update(metrics)
        results.append(result)

    results.sort(key=lambda x: -x.get(f"corr_{min(corr_windows)}d", 0))
    output_path = Path(output_dir) / "return_correlation.csv"
    write_csv(output_path, results)
    print(f"Return correlation filtering: {len(text_sim_df)} -> {len(results)} pairs (min_corr={min_corr})")
    print(f"Saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6: Return neutralization and correlation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
