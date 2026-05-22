"""Step 9: Closed-form weight optimization for risk exposure minimization.

For each tradeable pair, compute optimal long/short weights that minimize
the weighted sum of squared risk factor exposures (industry, size, volatility).

Closed-form solution:
  x* = sum(lambda_f * a_f * b_f) / sum(lambda_f * a_f^2)
  where a_f = beta_L^f + beta_S^f, b_f = beta_S^f

Final weights: W_L = x*, W_S = -(1-x*), clipped to [x_min, x_max]

Outputs: outputs/final_portfolio.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_io import write_csv


def estimate_factor_betas(
    returns: pd.DataFrame,
    ticker: str,
    industry_factor: pd.Series,
    size_factor: pd.Series,
    vol_exposure: pd.Series,
    lookback_days: int = 60,
) -> dict[str, float]:
    """Estimate factor betas using recent OLS regression."""
    if ticker not in returns.columns:
        return {"industry": 0.0, "size": 0.0, "volatility": 0.0}

    asset_ret = returns[ticker].tail(lookback_days).dropna()
    if len(asset_ret) < 30:
        return {"industry": 0.0, "size": 0.0, "volatility": 0.0}

    aligned = pd.DataFrame({
        "asset": asset_ret,
        "industry": industry_factor.reindex(asset_ret.index),
        "size": size_factor.reindex(asset_ret.index),
        "vol": vol_exposure.reindex(asset_ret.index),
    }).dropna()

    if len(aligned) < 20:
        return {"industry": 0.0, "size": 0.0, "volatility": 0.0}

    y = aligned["asset"].values
    X = np.column_stack([
        np.ones(len(aligned)),
        aligned["industry"].values,
        aligned["size"].values,
        aligned["vol"].values,
    ])

    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return {
            "industry": float(beta[1]),
            "size": float(beta[2]),
            "volatility": float(beta[3]),
        }
    except Exception:
        return {"industry": 0.0, "size": 0.0, "volatility": 0.0}


def optimize_pair_weight(
    beta_long: dict[str, float],
    beta_short: dict[str, float],
    lambda_industry: float = 1.0,
    lambda_size: float = 1.0,
    lambda_volatility: float = 1.0,
    x_min: float = 0.2,
    x_max: float = 0.8,
) -> float:
    """Closed-form solution for optimal pair weight x*."""
    factors = ["industry", "size", "volatility"]
    lambdas = {"industry": lambda_industry, "size": lambda_size, "volatility": lambda_volatility}

    numerator = 0.0
    denominator = 0.0

    for f in factors:
        bl = beta_long.get(f, 0.0)
        bs = beta_short.get(f, 0.0)
        a_f = bl + bs
        b_f = bs
        lam = lambdas[f]
        numerator += lam * a_f * b_f
        denominator += lam * a_f ** 2

    if abs(denominator) < 1e-12:
        x_star = 0.5
    else:
        x_star = numerator / denominator

    return float(np.clip(x_star, x_min, x_max))


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    config = yaml.safe_load(Path(config_path).read_text())
    weight_cfg = config.get("weight_optimization", {})
    lookback = int(weight_cfg.get("lookback_days", 60))
    lambda_ind = float(weight_cfg.get("lambda_industry", 1.0))
    lambda_vol = float(weight_cfg.get("lambda_volatility", 1.0))
    lambda_size = float(weight_cfg.get("lambda_size", 1.0))
    x_min = float(weight_cfg.get("x_min", 0.2))
    x_max = float(weight_cfg.get("x_max", 0.8))

    # Load tradeable pairs from LLM review
    review_path = Path(output_dir) / "llm_review_results.json"
    if not review_path.exists():
        raise FileNotFoundError(f"Not found: {review_path}. Run step8 first.")

    review_data = json.loads(review_path.read_text(encoding="utf-8"))
    tradeable = review_data.get("tradeable_pairs", [])
    if not tradeable:
        print("No tradeable pairs found. Nothing to optimize.")
        return []

    # Load price data
    price_cache = Path(output_dir) / "price_cache.parquet"
    if not price_cache.exists():
        raise FileNotFoundError(f"Not found: {price_cache}. Run step6 first.")

    close_df = pd.read_parquet(price_cache)
    returns = close_df.pct_change().iloc[1:]

    # Load universe for industry mapping and market caps
    universe_path = Path(output_dir) / "universe_with_industry.csv"
    universe_df = pd.read_csv(universe_path)
    industry_map = dict(zip(universe_df["ticker"], universe_df["industry_plate"]))
    market_caps = dict(zip(universe_df["ticker"], universe_df["market_cap_usd"]))

    # Build global SMB factor
    tickers_with_cap = [t for t in returns.columns if market_caps.get(t, 0) > 0]
    tickers_sorted = sorted(tickers_with_cap, key=lambda t: market_caps.get(t, 0))
    n = len(tickers_sorted)
    if n >= 20:
        small = tickers_sorted[:int(n * 0.3)]
        big = tickers_sorted[int(n * 0.7):]
        size_factor = returns[small].mean(axis=1) - returns[big].mean(axis=1)
    else:
        size_factor = pd.Series(0.0, index=returns.index)

    print(f"Optimizing weights for {len(tradeable)} tradeable pairs...")
    portfolio = []

    for pair in tradeable:
        long_ticker = str(pair.get("long_ticker", "")).upper()
        short_ticker = str(pair.get("short_ticker", "")).upper()
        industry = str(pair.get("industry_plate", ""))

        if long_ticker not in returns.columns or short_ticker not in returns.columns:
            continue

        # Build industry factor for each ticker
        def _industry_factor(ticker):
            plate = industry_map.get(ticker, "")
            peers = [t for t, p in industry_map.items() if p == plate and t != ticker and t in returns.columns]
            if not peers:
                return pd.Series(0.0, index=returns.index)
            weights = np.array([max(market_caps.get(t, 1.0), 1.0) for t in peers])
            weights = weights / weights.sum()
            return pd.Series((returns[peers].fillna(0).values * weights).sum(axis=1), index=returns.index)

        # Volatility exposure
        def _vol_exposure(ticker):
            return returns[ticker].rolling(lookback, min_periods=20).std().shift(1)

        ind_factor_long = _industry_factor(long_ticker)
        ind_factor_short = _industry_factor(short_ticker)

        beta_long = estimate_factor_betas(
            returns, long_ticker, ind_factor_long, size_factor, _vol_exposure(long_ticker), lookback
        )
        beta_short = estimate_factor_betas(
            returns, short_ticker, ind_factor_short, size_factor, _vol_exposure(short_ticker), lookback
        )

        x_star = optimize_pair_weight(
            beta_long, beta_short,
            lambda_industry=lambda_ind,
            lambda_size=lambda_size,
            lambda_volatility=lambda_vol,
            x_min=x_min,
            x_max=x_max,
        )

        portfolio.append({
            "long_ticker": long_ticker,
            "short_ticker": short_ticker,
            "industry_plate": industry,
            "pair_type": pair.get("pair_type", ""),
            "review_confidence": pair.get("review_confidence", 0),
            "supporting_dimensions": pair.get("supporting_dimension_count", 0),
            "weight_long": round(x_star, 4),
            "weight_short": round(-(1 - x_star), 4),
            "x_star": round(x_star, 4),
            "beta_long_industry": round(beta_long["industry"], 4),
            "beta_long_size": round(beta_long["size"], 4),
            "beta_long_vol": round(beta_long["volatility"], 4),
            "beta_short_industry": round(beta_short["industry"], 4),
            "beta_short_size": round(beta_short["size"], 4),
            "beta_short_vol": round(beta_short["volatility"], 4),
        })

    output_path = Path(output_dir) / "final_portfolio.csv"
    write_csv(output_path, portfolio)
    print(f"\nFinal portfolio: {len(portfolio)} pairs")
    print(f"Saved to {output_path}")

    # Summary statistics
    if portfolio:
        weights = [p["x_star"] for p in portfolio]
        print(f"  Weight x* stats: mean={np.mean(weights):.3f}, min={np.min(weights):.3f}, max={np.max(weights):.3f}")
        print(f"  Pairs at boundary (x_min or x_max): {sum(1 for w in weights if w == x_min or w == x_max)}")

    return portfolio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 9: Closed-form weight optimization")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
