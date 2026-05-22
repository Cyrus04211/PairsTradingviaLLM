"""Step 7: Combine text similarity and return correlation into selection_score,
apply top-N filtering and company appearance cap.

Rules:
  1. Compute selection_score = text_weight * overlap_score + corr_weight * correlation
  2. Sort by selection_score descending
  3. Take top candidate_fraction (default 50%)
  4. Enforce company_appearance_cap (default 2): each company appears at most N times

Outputs: outputs/candidates_for_llm.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_io import write_csv


def select_candidates(
    df: pd.DataFrame,
    candidate_fraction: float = 0.50,
    company_appearance_cap: int = 2,
    text_weight: float = 0.5,
    corr_weight: float = 0.5,
) -> pd.DataFrame:
    """Apply scoring, ranking, and concentration control."""
    df = df.copy()

    # Find the primary correlation column
    corr_cols = [c for c in df.columns if c.startswith("corr_") and c.endswith("d") and "median" not in c and "std" not in c]
    if corr_cols:
        primary_corr_col = sorted(corr_cols, key=lambda c: int(c.split("_")[1].replace("d", "")))[0]
    else:
        primary_corr_col = None

    # Compute selection_score
    overlap = pd.to_numeric(df.get("overlap_score", 0), errors="coerce").fillna(0)
    if primary_corr_col:
        corr = pd.to_numeric(df[primary_corr_col], errors="coerce").fillna(0)
    else:
        corr = pd.Series(0.0, index=df.index)

    df["selection_score"] = text_weight * overlap + corr_weight * corr
    df = df.sort_values("selection_score", ascending=False).reset_index(drop=True)

    # Take top fraction
    top_n = max(1, int(len(df) * candidate_fraction))
    df = df.head(top_n).copy()

    # Apply company appearance cap
    company_count: dict[str, int] = {}
    selected_indices: list[int] = []
    for idx, row in df.iterrows():
        ticker_a = str(row["ticker_a"])
        ticker_b = str(row["ticker_b"])
        count_a = company_count.get(ticker_a, 0)
        count_b = company_count.get(ticker_b, 0)
        if count_a >= company_appearance_cap or count_b >= company_appearance_cap:
            continue
        selected_indices.append(idx)
        company_count[ticker_a] = count_a + 1
        company_count[ticker_b] = count_b + 1

    result = df.loc[selected_indices].reset_index(drop=True)
    return result


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    config = yaml.safe_load(Path(config_path).read_text())
    sel_cfg = config.get("candidate_selection", {})
    candidate_fraction = float(sel_cfg.get("candidate_fraction", 0.50))
    appearance_cap = int(sel_cfg.get("company_appearance_cap", 2))
    text_weight = float(sel_cfg.get("text_weight", 0.5))
    corr_weight = float(sel_cfg.get("correlation_weight", 0.5))

    corr_path = Path(output_dir) / "return_correlation.csv"
    if not corr_path.exists():
        raise FileNotFoundError(f"Not found: {corr_path}. Run step6 first.")

    df = pd.read_csv(corr_path)
    print(f"Input pairs: {len(df)}")

    result = select_candidates(
        df,
        candidate_fraction=candidate_fraction,
        company_appearance_cap=appearance_cap,
        text_weight=text_weight,
        corr_weight=corr_weight,
    )

    output_path = Path(output_dir) / "candidates_for_llm.csv"
    result.to_csv(output_path, index=False)
    print(f"Candidate selection: {len(df)} -> {len(result)} pairs")
    print(f"  candidate_fraction={candidate_fraction}, company_appearance_cap={appearance_cap}")
    print(f"Saved to {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Candidate selection with concentration control")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
