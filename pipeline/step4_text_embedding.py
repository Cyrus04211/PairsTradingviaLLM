"""Step 4: Extract SEC 10-K business descriptions, refine into 4 dimensions, and embed.

Pipeline:
  1. For each company, fetch the latest 10-K/20-F Item 1 (Business) from SEC EDGAR
  2. Use LLM to distill into 4 competitive dimensions
  3. Embed each dimension using text-embedding-3-large
  4. Cache embeddings as .npz + metadata .json

Outputs: outputs/embeddings_metadata.json, outputs/embeddings_vectors.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.api_client import APIClient
from utils.data_io import write_json

DIMENSIONS = ["core_business", "main_product_service", "customer_channel", "geography"]

SUMMARY_SYSTEM_PROMPT = """You are a financial analyst. Given a company's business description from their SEC 10-K/20-F filing, extract a structured competitive profile with exactly 4 dimensions. Return ONLY a JSON object with these keys:
- "core_business": One sentence describing the company's core business model
- "main_product_service": One sentence listing the main products or services
- "customer_channel": One sentence describing target customers and distribution channels
- "geography": One sentence describing primary geographic markets

Be specific and factual. Each value should be a single concise English sentence."""


def fetch_business_description(ticker: str, cik: str) -> str:
    """Fetch Item 1 business description from SEC EDGAR using edgartools."""
    try:
        from edgar import Company
        company = Company(cik)
        filings = company.get_filings(form="10-K")
        if not filings or len(filings) == 0:
            filings = company.get_filings(form="20-F")
        if not filings or len(filings) == 0:
            return ""
        latest = filings[0]
        filing_obj = latest.obj()
        if hasattr(filing_obj, "item1") and filing_obj.item1:
            text = str(filing_obj.item1)
        elif hasattr(filing_obj, "__getitem__"):
            text = str(filing_obj["Item 1"])
        else:
            text = ""
        return text[:15000] if text else ""
    except Exception as e:
        print(f"  [WARN] Failed to fetch 10-K for {ticker} (CIK={cik}): {e}")
        return ""


def refine_to_dimensions(client: APIClient, business_text: str, ticker: str) -> dict[str, str]:
    """Use LLM to distill business description into 4 dimensions."""
    if not business_text.strip():
        return {}
    user_prompt = f"Company ticker: {ticker}\n\nBusiness Description:\n{business_text[:10000]}"
    try:
        result = client.chat_completion_json(
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        output = {}
        for dim in DIMENSIONS:
            val = result.get(dim, "")
            if isinstance(val, str) and val.strip():
                output[dim] = val.strip()
        return output
    except Exception as e:
        print(f"  [WARN] LLM refinement failed for {ticker}: {e}")
        return {}


def embed_all_companies(
    client: APIClient,
    companies: list[dict],
    embedding_model: str,
    batch_size: int = 128,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Generate embeddings for all companies across all dimensions."""
    jobs: list[tuple[int, str, str]] = []
    for idx, company in enumerate(companies):
        dims = company.get("dimensions", {})
        for dim in DIMENSIONS:
            text = dims.get(dim, "")
            if text:
                jobs.append((idx, dim, text))

    if not jobs:
        raise ValueError("No texts to embed")

    print(f"Embedding {len(jobs)} texts in batches of {batch_size}...")
    embedding_dim = None
    assigned: dict[tuple[int, str], np.ndarray] = {}

    for batch_start in range(0, len(jobs), batch_size):
        batch = jobs[batch_start:batch_start + batch_size]
        texts = [j[2] for j in batch]
        vectors = client.embed_batch(texts, model=embedding_model)
        for (idx, dim, _), vec in zip(batch, vectors):
            arr = np.asarray(vec, dtype=np.float32)
            if embedding_dim is None:
                embedding_dim = arr.shape[0]
            assigned[(idx, dim)] = arr
        print(f"  Embedded batch {batch_start // batch_size + 1}/{(len(jobs) + batch_size - 1) // batch_size}")

    n = len(companies)
    matrices = {dim: np.zeros((n, embedding_dim), dtype=np.float32) for dim in DIMENSIONS}
    for (idx, dim), vec in assigned.items():
        matrices[dim][idx] = vec

    return matrices, companies


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    config = yaml.safe_load(Path(config_path).read_text())
    embed_cfg = config.get("embedding", {})
    api_key_env = embed_cfg.get("api_key_env", "CLOSEAI_API_KEY")
    api_url = embed_cfg.get("api_url", "https://api.openai-proxy.org/v1")
    embedding_model = embed_cfg.get("model", "text-embedding-3-large")
    summary_model = embed_cfg.get("summary_model", "gpt-4o")
    summary_api_url = embed_cfg.get("summary_api_url", api_url)
    batch_size = int(embed_cfg.get("batch_size", 128))

    # Normalize api_url: strip /embeddings or /chat/completions suffix
    base_url = api_url.rstrip("/")
    if base_url.endswith("/embeddings"):
        base_url = base_url[:-len("/embeddings")]
    if base_url.endswith("/chat/completions"):
        base_url = base_url[:-len("/chat/completions")]

    summary_base = summary_api_url.rstrip("/")
    if summary_base.endswith("/chat/completions"):
        summary_base = summary_base[:-len("/chat/completions")]

    embed_client = APIClient(api_key_env=api_key_env, api_url=base_url, model=embedding_model)
    summary_client = APIClient(api_key_env=api_key_env, api_url=summary_base, model=summary_model)

    universe_path = Path(output_dir) / "universe_with_industry.csv"
    if not universe_path.exists():
        raise FileNotFoundError(f"Not found: {universe_path}. Run step2 first.")

    df = pd.read_csv(universe_path)
    companies: list[dict] = []

    # Check for cached summaries
    cache_path = Path(output_dir) / "business_summaries_cache.json"
    cached: dict[str, dict] = {}
    if cache_path.exists():
        cached_list = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = {item["ticker"]: item for item in cached_list}
        print(f"Loaded {len(cached)} cached business summaries.")

    print(f"Processing {len(df)} companies for business description extraction and refinement...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting & refining"):
        ticker = str(row["ticker"]).strip().upper()
        cik = str(row.get("cik", "")).strip()

        if ticker in cached and cached[ticker].get("dimensions"):
            companies.append(cached[ticker])
            continue

        business_text = fetch_business_description(ticker, cik)
        dimensions = refine_to_dimensions(summary_client, business_text, ticker)

        company = {
            "ticker": ticker,
            "cik": cik,
            "company_name": str(row.get("company_name", "")),
            "industry_plate": str(row.get("industry_plate", "")),
            "market_cap_usd": float(row.get("market_cap_usd", 0)),
            "dimensions": dimensions,
        }
        companies.append(company)
        time.sleep(0.5)

    # Save summaries cache
    write_json(cache_path, companies)
    print(f"Saved business summaries cache to {cache_path}")

    # Filter companies with complete dimensions
    valid_companies = [c for c in companies if len(c.get("dimensions", {})) == len(DIMENSIONS)]
    print(f"Companies with complete 4-dimension summaries: {len(valid_companies)}/{len(companies)}")

    if not valid_companies:
        print("[ERROR] No companies with complete dimension summaries. Cannot proceed.")
        return

    matrices, _ = embed_all_companies(embed_client, valid_companies, embedding_model, batch_size)

    # Save embeddings
    npz_path = Path(output_dir) / "embeddings_vectors.npz"
    np.savez_compressed(npz_path, **{dim: matrices[dim] for dim in DIMENSIONS})
    print(f"Saved embedding vectors to {npz_path}")

    metadata = {
        "model": embedding_model,
        "dimensions": DIMENSIONS,
        "company_count": len(valid_companies),
        "companies": [
            {
                "index": i,
                "ticker": c["ticker"],
                "cik": c["cik"],
                "company_name": c["company_name"],
                "industry_plate": c["industry_plate"],
                "market_cap_usd": c["market_cap_usd"],
                "dimension_texts": c["dimensions"],
            }
            for i, c in enumerate(valid_companies)
        ],
    }
    metadata_path = Path(output_dir) / "embeddings_metadata.json"
    write_json(metadata_path, metadata)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Text embedding pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
