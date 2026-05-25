"""Step 4: Pair prefiltering with multi-window neutralized return correlation."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.market_data import download_price_history


def build_industry_factor(
    returns: pd.DataFrame,
    ticker: str,
    industry_map: dict[str, str],
    market_caps: dict[str, float],
) -> pd.Series:
    plate = industry_map.get(ticker, "")
    peers = [t for t, p in industry_map.items() if p == plate and t != ticker and t in returns.columns]
    if not peers:
        return pd.Series(0.0, index=returns.index)
    weights = np.array([max(market_caps.get(t, 1.0), 1.0) for t in peers], dtype=float)
    weights = weights / weights.sum()
    peer_returns = returns[peers].fillna(0.0).to_numpy(dtype=float)
    return pd.Series((peer_returns * weights).sum(axis=1), index=returns.index)


def build_smb_factor(returns: pd.DataFrame, market_caps: dict[str, float]) -> pd.Series:
    tickers = [t for t in returns.columns if market_caps.get(t, 0) > 0]
    if len(tickers) < 20:
        return pd.Series(0.0, index=returns.index)
    tickers_sorted = sorted(tickers, key=lambda t: market_caps.get(t, 0))
    n = len(tickers_sorted)
    small = tickers_sorted[:int(n * 0.3)]
    big = tickers_sorted[int(n * 0.7):]
    return returns[small].mean(axis=1) - returns[big].mean(axis=1)


def rolling_volatility(returns: pd.Series, window: int = 120) -> pd.Series:
    return returns.rolling(window, min_periods=max(20, window // 2)).std().shift(1)


def neutralize_returns_oos(
    asset_returns: pd.Series,
    factors: dict[str, pd.Series],
    window: int = 120,
) -> pd.Series:
    aligned = pd.DataFrame({"asset": asset_returns})
    for name, series in factors.items():
        aligned[name] = series
    aligned = aligned.dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)

    factor_names = list(factors.keys())
    y = aligned["asset"].to_numpy(dtype=float)
    X = aligned[factor_names].to_numpy(dtype=float)
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


def compute_latest_window_correlation(
    resid_a: pd.Series,
    resid_b: pd.Series,
    window_days: int,
) -> dict[str, object]:
    aligned = pd.DataFrame({"a": resid_a, "b": resid_b}).dropna().sort_index()
    if len(aligned) < window_days:
        return {}

    latest = aligned.tail(window_days)
    corr = latest["a"].corr(latest["b"])
    if not np.isfinite(corr):
        return {}

    return {
        "curve_correlation": round(float(corr), 6),
        "window_days": int(window_days),
        "window_start_date": str(latest.index[0].date()),
        "window_end_date": str(latest.index[-1].date()),
    }


def build_output_columns(window_candidates: list[int]) -> list[str]:
    columns = ["ticker_a", "ticker_b", "industry_plate", "pair_end_date"]
    for window_days in window_candidates:
        columns.extend(
            [
                f"corr_{window_days}d",
                f"rank_{window_days}d",
                f"pass_{window_days}d_top_half",
                f"window_start_{window_days}d",
                f"window_end_{window_days}d",
            ]
        )
    columns.extend(["mean_curve_correlation", "min_curve_correlation", "passes_all_windows"])
    return columns


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    config = yaml.safe_load(Path(config_path).read_text())
    corr_cfg = config.get("return_correlation", {})
    neut_window = int(corr_cfg.get("neutralization_window", 120))
    window_candidates = [int(value) for value in corr_cfg.get("window_candidates", [252, 504, 1000])]
    keep_fraction = float(corr_cfg.get("keep_fraction", 0.50))
    price_days = int(corr_cfg.get("price_history_days", 1200))
    end_date_raw = str(corr_cfg.get("end_date", "2026-05-08")).strip()
    end_date = pd.Timestamp(end_date_raw)

    universe_path = Path(output_dir) / "universe_with_industry.csv"
    pairs_path = Path(output_dir) / "all_pairs.csv"
    if not universe_path.exists():
        raise FileNotFoundError(f"Not found: {universe_path}. Run step2 first.")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Not found: {pairs_path}. Run step3 first.")

    universe_df = pd.read_csv(universe_path)
    pairs_df = pd.read_csv(pairs_path)
    industry_map = dict(zip(universe_df["ticker"], universe_df["industry_plate"]))
    market_caps = dict(zip(universe_df["ticker"], universe_df["market_cap_usd"]))

    tickers_needed = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())
    tickers_list = sorted(tickers_needed)
    print(f"Need price data for {len(tickers_list)} tickers...")

    price_cache = Path(output_dir) / "price_cache.parquet"
    if price_cache.exists():
        print("Loading cached price data...")
        close_df = pd.read_parquet(price_cache)
        missing = [t for t in tickers_list if t not in close_df.columns]
        if missing:
            print(f"Downloading {len(missing)} missing tickers...")
            new_prices = download_price_history(missing, min_history_days=price_days)
            if not new_prices.empty:
                close_df = close_df.join(new_prices, how="outer")
                close_df.to_parquet(price_cache)
    else:
        print("Downloading price history (this will take a while)...")
        close_df = download_price_history(tickers_list, min_history_days=price_days)
        if close_df.empty:
            raise RuntimeError("Failed to download any price data")
        close_df.to_parquet(price_cache)
        print(f"Cached prices to {price_cache}")

    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()
    close_df = close_df.loc[close_df.index <= end_date]
    if close_df.empty:
        raise RuntimeError(f"No price history available on or before {end_date_raw}")

    returns = close_df.pct_change().iloc[1:]
    smb_factor = build_smb_factor(returns, market_caps)

    print("Neutralizing returns...")
    residuals: dict[str, pd.Series] = {}
    min_window_days = min(window_candidates)
    for ticker in tqdm(tickers_list, desc="Neutralizing"):
        if ticker not in returns.columns:
            continue
        asset_ret = returns[ticker].dropna()
        if len(asset_ret) < max(neut_window + 30, min_window_days):
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

    print(f"Scoring multi-window correlation for {len(pairs_df)} pairs...")
    scored_rows: list[dict[str, object]] = []
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Correlations"):
        ticker_a = str(row["ticker_a"]).upper()
        ticker_b = str(row["ticker_b"]).upper()
        if ticker_a not in residuals or ticker_b not in residuals:
            continue

        result: dict[str, object] = {
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "industry_plate": row.get("industry_plate", ""),
            "pair_end_date": "",
        }
        corr_values: list[float] = []
        has_all_windows = True

        for window_days in window_candidates:
            metrics = compute_latest_window_correlation(residuals[ticker_a], residuals[ticker_b], window_days)
            corr_col = f"corr_{window_days}d"
            start_col = f"window_start_{window_days}d"
            end_col = f"window_end_{window_days}d"
            if not metrics:
                has_all_windows = False
                result[corr_col] = math.nan
                result[start_col] = ""
                result[end_col] = ""
                continue

            result[corr_col] = float(metrics["curve_correlation"])
            result[start_col] = str(metrics["window_start_date"])
            result[end_col] = str(metrics["window_end_date"])
            result["pair_end_date"] = str(metrics["window_end_date"])
            corr_values.append(float(metrics["curve_correlation"]))

        result["mean_curve_correlation"] = round(float(np.mean(corr_values)), 6) if corr_values else math.nan
        result["min_curve_correlation"] = round(float(np.min(corr_values)), 6) if corr_values else math.nan
        result["passes_all_windows"] = bool(has_all_windows)
        scored_rows.append(result)

    window_stats: list[dict[str, object]] = []
    for window_days in window_candidates:
        corr_col = f"corr_{window_days}d"
        rank_col = f"rank_{window_days}d"
        pass_col = f"pass_{window_days}d_top_half"

        valid_indices = [
            idx for idx, row in enumerate(scored_rows)
            if np.isfinite(float(row.get(corr_col, math.nan)))
        ]
        valid_indices.sort(
            key=lambda idx: (
                -float(scored_rows[idx][corr_col]),
                str(scored_rows[idx]["ticker_a"]),
                str(scored_rows[idx]["ticker_b"]),
            )
        )
        keep_n = max(1, int(len(valid_indices) * keep_fraction)) if valid_indices else 0

        for idx, row in enumerate(scored_rows):
            row[rank_col] = ""
            row[pass_col] = False

        for rank, row_idx in enumerate(valid_indices, start=1):
            scored_rows[row_idx][rank_col] = rank
            scored_rows[row_idx][pass_col] = rank <= keep_n

        window_stats.append(
            {
                "window_days": int(window_days),
                "scored_pairs": int(len(valid_indices)),
                "top_half_cutoff_rank": int(keep_n),
            }
        )

    kept_rows = []
    for row in scored_rows:
        row["passes_all_windows"] = bool(
            row.get("passes_all_windows", False)
            and all(bool(row.get(f"pass_{window_days}d_top_half", False)) for window_days in window_candidates)
        )
        if row["passes_all_windows"]:
            kept_rows.append(row)

    kept_rows.sort(
        key=lambda row: (
            -float(row.get("mean_curve_correlation", math.nan)),
            -float(row.get("min_curve_correlation", math.nan)),
            str(row["ticker_a"]),
            str(row["ticker_b"]),
        )
    )

    output_columns = build_output_columns(window_candidates)
    all_scores_path = Path(output_dir) / "return_correlation_all.csv"
    filtered_path = Path(output_dir) / "return_correlation.csv"
    summary_path = Path(output_dir) / "return_correlation_summary.json"
    pd.DataFrame(scored_rows, columns=output_columns).to_csv(all_scores_path, index=False)
    pd.DataFrame(kept_rows, columns=output_columns).to_csv(filtered_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "end_date": end_date_raw,
                "window_candidates": window_candidates,
                "keep_fraction": keep_fraction,
                "window_stats": window_stats,
                "scored_pair_count": int(len(scored_rows)),
                "kept_pair_count": int(len(kept_rows)),
                "selection_rule": "pair must rank in the top keep_fraction for every candidate window",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Multi-window neutralized return correlation filtering: "
        f"{len(pairs_df)} -> {len(scored_rows)} scored -> {len(kept_rows)} kept"
    )
    print(
        f"  end_date={end_date_raw}, windows={window_candidates}, keep_fraction={keep_fraction}"
    )
    print(f"Saved scores to {all_scores_path}")
    print(f"Saved kept pairs to {filtered_path}")
    print(f"Saved summary to {summary_path}")
    return kept_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Multi-window neutralized return correlation filter")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
