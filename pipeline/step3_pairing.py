"""Step 3: Generate all possible stock pairs within each industry plate.

Outputs: outputs/all_pairs.csv
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_io import write_csv


def generate_intra_industry_pairs(df: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for industry, group in df.groupby("industry_plate"):
        tickers = sorted(group["ticker"].tolist())
        if len(tickers) < 2:
            continue
        for a, b in combinations(tickers, 2):
            pairs.append({
                "ticker_a": a,
                "ticker_b": b,
                "industry_plate": industry,
            })
    result = pd.DataFrame(pairs)
    print(f"Generated {len(result)} pairs across {df['industry_plate'].nunique()} industries.")
    return result


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    universe_path = Path(output_dir) / "universe_with_industry.csv"
    if not universe_path.exists():
        raise FileNotFoundError(f"Not found: {universe_path}. Run step2 first.")

    df = pd.read_csv(universe_path)
    pairs_df = generate_intra_industry_pairs(df)

    output_path = Path(output_dir) / "all_pairs.csv"
    pairs_df.to_csv(output_path, index=False)
    print(f"Saved {len(pairs_df)} pairs to {output_path}.")
    return pairs_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Generate intra-industry pairs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
