"""Step 5: Compute text similarity between paired companies using embeddings.

For each pair in all_pairs.csv, compute cosine similarity across 4 dimensions,
then apply weighted sum to get overlap_score. Filter by threshold.

Outputs: outputs/text_similarity.csv
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

DIMENSIONS = ["core_business", "main_product_service", "customer_channel", "geography"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    config = yaml.safe_load(Path(config_path).read_text())
    sim_cfg = config.get("text_similarity", {})
    weights = sim_cfg.get("weights", {})
    w_core = float(weights.get("core_business", 0.30))
    w_product = float(weights.get("main_product_service", 0.30))
    w_customer = float(weights.get("customer_channel", 0.25))
    w_geo = float(weights.get("geography", 0.15))
    min_overlap = float(sim_cfg.get("min_overlap_score", 0.7))

    dim_weights = {
        "core_business": w_core,
        "main_product_service": w_product,
        "customer_channel": w_customer,
        "geography": w_geo,
    }

    # Load embeddings
    npz_path = Path(output_dir) / "embeddings_vectors.npz"
    metadata_path = Path(output_dir) / "embeddings_metadata.json"
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Embeddings not found. Run step4 first.")

    data = np.load(npz_path)
    matrices = {dim: data[dim] for dim in DIMENSIONS}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    companies = metadata["companies"]
    ticker_to_idx = {c["ticker"]: c["index"] for c in companies}

    # Load pairs
    pairs_path = Path(output_dir) / "all_pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError("Pairs file not found. Run step3 first.")
    pairs_df = pd.read_csv(pairs_path)

    print(f"Computing text similarity for {len(pairs_df)} pairs...")
    results = []
    for _, row in pairs_df.iterrows():
        ticker_a = str(row["ticker_a"]).upper()
        ticker_b = str(row["ticker_b"]).upper()
        idx_a = ticker_to_idx.get(ticker_a)
        idx_b = ticker_to_idx.get(ticker_b)

        if idx_a is None or idx_b is None:
            continue

        dim_scores = {}
        overlap_score = 0.0
        for dim in DIMENSIONS:
            vec_a = matrices[dim][idx_a]
            vec_b = matrices[dim][idx_b]
            sim = cosine_similarity(vec_a, vec_b)
            dim_scores[dim] = sim
            overlap_score += sim * dim_weights[dim]

        if overlap_score >= min_overlap:
            results.append({
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "industry_plate": row.get("industry_plate", ""),
                "overlap_score": round(overlap_score, 4),
                "core_business_similarity": round(dim_scores["core_business"], 4),
                "main_product_service_similarity": round(dim_scores["main_product_service"], 4),
                "customer_channel_similarity": round(dim_scores["customer_channel"], 4),
                "geography_similarity": round(dim_scores["geography"], 4),
            })

    results.sort(key=lambda x: -x["overlap_score"])
    output_path = Path(output_dir) / "text_similarity.csv"
    write_csv(output_path, results)
    print(f"Text similarity filtering: {len(pairs_df)} -> {len(results)} pairs (threshold={min_overlap})")
    print(f"Saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Text similarity computation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
