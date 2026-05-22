"""Step 8: LLM two-stage subjective analysis for pair trading direction.

Stage 1: Business overlap review - determines if pair has genuine business baseline
Stage 2: Four-dimension directional review (Business Divergence, Product Cycle,
         Financials, Recent News) - determines long/short direction

Outputs: outputs/llm_review_results.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.api_client import APIClient
from utils.data_io import write_json

OVERLAP_REVIEW_SYSTEM_PROMPT = """You are a senior equity analyst evaluating whether two companies share enough business overlap to form a valid pairs trade.

Evaluate whether the two companies share a genuine common business baseline that makes them suitable for relative-value analysis. Consider:
- Do they compete in the same products, services, customers, channels, end markets, supply chains, or industry cycles?
- Is their overlap substantive (not just "both are tech companies")?

Classify valid pairs into:
- Type 1 (Same-Arena Competitive Pair): Direct or indirect competition with substantial overlap in products/services/customers
- Type 2 (Closely Related Structural Divergence Pair): Share clear industry/value-chain/end-market baseline, differences may create divergence signals

Return ONLY a raw JSON object:
{
  "is_high_overlap_competitor": true/false,
  "pair_type": "type_1" or "type_2" or "",
  "review_reason": "1-2 sentence explanation",
  "review_confidence": 0.0 to 1.0
}"""

DIRECTION_REVIEW_SYSTEM_PROMPT = """You are a senior equity analyst determining the relative long/short direction for a stock pair.

Given two companies that have been confirmed as a valid pair, analyze the specific dimension provided and determine which company is relatively stronger.

Search for and consider the most recent public information available.

Return ONLY a raw JSON object:
{
  "is_long_short_candidate": true/false,
  "long_ticker": "TICKER" or "",
  "short_ticker": "TICKER" or "",
  "review_reason": "1-2 sentence explanation",
  "review_confidence": 0.0 to 1.0
}"""

REVIEW_DIMENSIONS = [
    {
        "key": "business_divergence",
        "label": "Business Divergence",
        "prompt": "Analyze business divergence: Under their shared industry/value-chain/end-market baseline, does one company have stronger business drivers, competitive advantages, or structural positioning than the other?",
    },
    {
        "key": "product_cycle",
        "label": "Product Cycle",
        "prompt": "Analyze product cycle: Compare product strength, innovation pace, customer adoption rates, and market share trends. Does one company have a relatively better product cycle outlook?",
    },
    {
        "key": "financials",
        "label": "Financials",
        "prompt": "Analyze financials: Compare margin trends, cost control, balance sheet quality, and free cash flow. Is one company financially stronger on a relative basis?",
    },
    {
        "key": "recent_news",
        "label": "Recent News",
        "prompt": "Analyze recent news and catalysts: Consider new product launches, customer wins/losses, strategic partnerships/M&A, regulatory changes, management changes, and demand trends. Does recent public information favor one company over the other?",
    },
]


def review_overlap(client: APIClient, ticker_a: str, ticker_b: str, industry: str) -> dict[str, Any]:
    user_prompt = (
        f"Company A: {ticker_a}\nCompany B: {ticker_b}\n"
        f"Industry: {industry}\n\n"
        f"Evaluate whether these two companies share enough genuine business overlap "
        f"to form a valid pairs trade."
    )
    try:
        result = client.chat_completion_json(
            messages=[
                {"role": "system", "content": OVERLAP_REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        return result
    except Exception as e:
        return {"is_high_overlap_competitor": False, "error": str(e)}


def review_dimension(
    client: APIClient,
    ticker_a: str,
    ticker_b: str,
    industry: str,
    dimension: dict[str, str],
) -> dict[str, Any]:
    user_prompt = (
        f"Company A (potential long): {ticker_a}\n"
        f"Company B (potential short): {ticker_b}\n"
        f"Industry: {industry}\n\n"
        f"Dimension: {dimension['label']}\n"
        f"{dimension['prompt']}"
    )
    try:
        result = client.chat_completion_json(
            messages=[
                {"role": "system", "content": DIRECTION_REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        return result
    except Exception as e:
        return {"is_long_short_candidate": False, "error": str(e)}


def combine_final_review(
    ticker_a: str,
    ticker_b: str,
    overlap_result: dict[str, Any],
    dimension_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Synthesize overlap + dimension reviews into final verdict."""
    if not overlap_result.get("is_high_overlap_competitor", False):
        return {
            "is_long_short_candidate": False,
            "long_ticker": "",
            "short_ticker": "",
            "pair_type": overlap_result.get("pair_type", ""),
            "review_reason": overlap_result.get("review_reason", "Failed overlap review"),
            "review_confidence": 0.0,
            "supporting_dimensions": [],
            "direction_consensus": "no_overlap",
        }

    # Collect directional signals
    long_votes: dict[str, list[str]] = {}  # ticker -> list of dimension keys
    confidences: list[float] = []

    for dim_result in dimension_results:
        if not dim_result.get("is_long_short_candidate", False):
            continue
        long_t = str(dim_result.get("long_ticker", "")).upper()
        dim_key = dim_result.get("_dimension_key", "")
        conf = float(dim_result.get("review_confidence", 0.0))
        if long_t in (ticker_a.upper(), ticker_b.upper()):
            long_votes.setdefault(long_t, []).append(dim_key)
            confidences.append(conf)

    if not long_votes:
        return {
            "is_long_short_candidate": False,
            "long_ticker": "",
            "short_ticker": "",
            "pair_type": overlap_result.get("pair_type", ""),
            "review_reason": "No dimension provided directional signal",
            "review_confidence": 0.0,
            "supporting_dimensions": [],
            "direction_consensus": "no_signal",
        }

    # Check for direction conflict
    if len(long_votes) > 1:
        return {
            "is_long_short_candidate": False,
            "long_ticker": "",
            "short_ticker": "",
            "pair_type": overlap_result.get("pair_type", ""),
            "review_reason": "Direction conflict between dimensions",
            "review_confidence": 0.0,
            "supporting_dimensions": [],
            "direction_consensus": "conflict",
        }

    # Consensus direction
    long_ticker = list(long_votes.keys())[0]
    short_ticker = ticker_b.upper() if long_ticker == ticker_a.upper() else ticker_a.upper()
    supporting = long_votes[long_ticker]

    overlap_conf = float(overlap_result.get("review_confidence", 0.5))
    avg_conf = (overlap_conf + sum(confidences)) / (1 + len(confidences)) if confidences else overlap_conf

    return {
        "is_long_short_candidate": True,
        "long_ticker": long_ticker,
        "short_ticker": short_ticker,
        "pair_type": overlap_result.get("pair_type", ""),
        "review_reason": overlap_result.get("review_reason", ""),
        "review_confidence": round(avg_conf, 3),
        "supporting_dimension_count": len(supporting),
        "supporting_dimensions": supporting,
        "direction_consensus": "unanimous" if len(supporting) == 4 else "majority",
    }


def process_single_pair(
    client: APIClient,
    row: dict[str, Any],
    dimension_workers: int = 4,
) -> dict[str, Any]:
    """Process one pair through both review stages."""
    ticker_a = str(row.get("ticker_a", "")).upper()
    ticker_b = str(row.get("ticker_b", "")).upper()
    industry = str(row.get("industry_plate", ""))

    # Stage 1: Overlap review
    overlap_result = review_overlap(client, ticker_a, ticker_b, industry)
    if not overlap_result.get("is_high_overlap_competitor", False):
        return {
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "industry_plate": industry,
            "overlap_score": row.get("overlap_score", ""),
            "selection_score": row.get("selection_score", ""),
            "stage1_passed": False,
            **combine_final_review(ticker_a, ticker_b, overlap_result, []),
        }

    # Stage 2: Four-dimension directional review (concurrent)
    dimension_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=dimension_workers) as executor:
        futures = {}
        for dim in REVIEW_DIMENSIONS:
            future = executor.submit(review_dimension, client, ticker_a, ticker_b, industry, dim)
            futures[future] = dim["key"]

        for future in concurrent.futures.as_completed(futures):
            dim_key = futures[future]
            result = future.result()
            result["_dimension_key"] = dim_key
            dimension_results.append(result)

    final = combine_final_review(ticker_a, ticker_b, overlap_result, dimension_results)

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "industry_plate": industry,
        "overlap_score": row.get("overlap_score", ""),
        "selection_score": row.get("selection_score", ""),
        "stage1_passed": True,
        "overlap_review": overlap_result,
        "dimension_reviews": {r["_dimension_key"]: {k: v for k, v in r.items() if k != "_dimension_key"} for r in dimension_results},
        **final,
    }


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    config = yaml.safe_load(Path(config_path).read_text())
    llm_cfg = config.get("llm_review", {})
    api_key_env = llm_cfg.get("api_key_env", "CLOSEAI_API_KEY")
    api_url = llm_cfg.get("api_url", "https://api.openai-proxy.org/v1/chat/completions")
    model = llm_cfg.get("model", "gpt-4o")
    workers = int(llm_cfg.get("workers", 10))
    dimension_workers = int(llm_cfg.get("dimension_workers", 4))
    max_retries = int(llm_cfg.get("max_retries", 3))
    timeout = int(llm_cfg.get("request_timeout", 300))

    base_url = api_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[:-len("/chat/completions")]

    client = APIClient(
        api_key_env=api_key_env,
        api_url=base_url,
        model=model,
        max_retries=max_retries,
        request_timeout=timeout,
    )

    candidates_path = Path(output_dir) / "candidates_for_llm.csv"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Not found: {candidates_path}. Run step7 first.")

    df = pd.read_csv(candidates_path)
    rows = df.to_dict("records")
    print(f"Starting LLM review for {len(rows)} candidate pairs (workers={workers})...")

    # Process pairs concurrently
    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    completed = [0]

    def process_row(row):
        result = process_single_pair(client, row, dimension_workers)
        with lock:
            completed[0] += 1
            status = "PASS" if result.get("is_long_short_candidate") else "SKIP"
            print(f"  [{completed[0]}/{len(rows)}] {row['ticker_a']}::{row['ticker_b']} -> {status}")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_row = {executor.submit(process_row, row): row for row in rows}
        for future in concurrent.futures.as_completed(future_to_row):
            results.append(future.result())

    # Filter to tradeable pairs
    tradeable = [r for r in results if r.get("is_long_short_candidate", False)]
    tradeable.sort(key=lambda x: -float(x.get("review_confidence", 0)))

    print(f"\nLLM Review complete:")
    print(f"  Total candidates: {len(rows)}")
    print(f"  Stage 1 passed (overlap): {sum(1 for r in results if r.get('stage1_passed'))}")
    print(f"  Final tradeable pairs: {len(tradeable)}")

    # Save all results
    output_path = Path(output_dir) / "llm_review_results.json"
    write_json(output_path, {"all_results": results, "tradeable_pairs": tradeable})
    print(f"Saved to {output_path}")

    # Also save tradeable pairs as CSV for easy viewing
    if tradeable:
        csv_rows = []
        for r in tradeable:
            csv_rows.append({
                "long_ticker": r.get("long_ticker", ""),
                "short_ticker": r.get("short_ticker", ""),
                "industry_plate": r.get("industry_plate", ""),
                "pair_type": r.get("pair_type", ""),
                "review_confidence": r.get("review_confidence", 0),
                "supporting_dimension_count": r.get("supporting_dimension_count", 0),
                "direction_consensus": r.get("direction_consensus", ""),
                "overlap_score": r.get("overlap_score", ""),
            })
        csv_path = Path(output_dir) / "tradeable_pairs.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"Tradeable pairs CSV: {csv_path}")

    return tradeable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8: LLM two-stage pair review")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
