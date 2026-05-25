"""Step 7: Enforce company appearance cap after curve and text filtering."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def select_candidates(
    df: pd.DataFrame,
    company_appearance_cap: int = 2,
) -> pd.DataFrame:
    """Apply concentration control on text-filtered pairs."""
    df = df.copy()
    df["overlap_score"] = pd.to_numeric(df.get("overlap_score", 0), errors="coerce").fillna(0.0)
    if "curve_correlation" in df.columns:
        df["curve_correlation"] = pd.to_numeric(df.get("curve_correlation"), errors="coerce").fillna(0.0)
        df = df.sort_values(["overlap_score", "curve_correlation"], ascending=[False, False]).reset_index(drop=True)
    else:
        df = df.sort_values("overlap_score", ascending=False).reset_index(drop=True)
    df["selection_score"] = df["overlap_score"]

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
    appearance_cap = int(sel_cfg.get("company_appearance_cap", 2))
    text_path = Path(output_dir) / "text_similarity.csv"
    if not text_path.exists():
        raise FileNotFoundError(f"Not found: {text_path}. Run step6 first.")

    df = pd.read_csv(text_path)
    print(f"Input pairs: {len(df)}")

    result = select_candidates(
        df,
        company_appearance_cap=appearance_cap,
    )

    output_path = Path(output_dir) / "candidates_for_llm.csv"
    result.to_csv(output_path, index=False)
    print(f"Candidate selection: {len(df)} -> {len(result)} pairs")
    print(f"  company_appearance_cap={appearance_cap}")
    print(f"Saved to {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7: Candidate selection with concentration control")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
