"""Step 5: Reuse cached business summaries/embeddings, then fill missing text features."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.api_client import APIClient
from utils.data_io import write_json

DIMENSIONS = ["core_business", "main_product_service", "customer_channel", "geography"]
DIMENSION_TITLES = {
    "core_business": "Core Business",
    "main_product_service": "Main Product / Service",
    "customer_channel": "Customer / Channel",
    "geography": "Geography",
}
TITLE_TO_DIMENSION = {title: slug for slug, title in DIMENSION_TITLES.items()}

DEFAULT_DOUBLE_TOWER_ROOT = Path("cache/double_tower")
DEFAULT_TEXT_RUNS_ROOT = Path("cache/text_pair_runs")
DEFAULT_STRATEGY_MODEL_TEXT_RUN = Path("cache/strategy_model_text_run")
DEFAULT_SUBMISSIONS_ZIP = Path("cache/submissions_v2.zip")
DEFAULT_EDGAR_DATA_DIR = Path("cache/edgar_data")
DEFAULT_EDGAR_CACHE_DIR = Path("cache/edgar_http_cache")
DEFAULT_SEC_USER_AGENT = "Your Name your_email@example.com"

SUMMARY_SYSTEM_PROMPT = """You are a financial analyst. Given a company's business description from their SEC annual filing, extract a structured competitive profile with exactly 4 dimensions. Return ONLY a JSON object with these keys:
- "core_business": One sentence describing the company's core business model
- "main_product_service": One sentence listing the main products or services
- "customer_channel": One sentence describing target customers and distribution channels
- "geography": One sentence describing primary geographic markets

Be specific and factual. Each value should be a single concise English sentence."""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:  # noqa: BLE001
        pass
    return " ".join(str(value).split()).strip()


def to_float(value: Any) -> float:
    try:
        if value is None or clean_text(value) == "":
            return 0.0
        return float(value)
    except Exception:  # noqa: BLE001
        return 0.0


# ── Robust SEC filing section extraction (ported from stock_matching) ──

MIN_CONTENT_CHARS = 200


def _build_min_content_position(text: str) -> int:
    """Return the earliest character offset where real content is expected,
    to avoid matching table-of-contents entries."""
    return min(max(5000, int(len(text) * 0.03)), 30000)


def extract_text_section(
    text: str,
    start_patterns: list[str],
    end_patterns: list[str],
    *,
    min_chars: int = MIN_CONTENT_CHARS,
) -> str:
    """Regex-based section extraction with multiple start/end pattern candidates.

    Tries each *start_pattern* in order; for each match after the
    minimum-content position, finds the earliest *end_pattern* match and
    returns the clean text between them.  Returns empty string when no
    candidate yields enough text.
    """
    if not text:
        return ""
    min_position = _build_min_content_position(text)
    flags = re.IGNORECASE | re.MULTILINE

    for start_pattern in start_patterns:
        for match in re.finditer(start_pattern, text, flags):
            if match.start() < min_position:
                continue
            start = match.start()
            search_from = min(len(text), match.end() + 20)
            end = len(text)
            for end_pattern in end_patterns:
                end_match = re.search(end_pattern, text[search_from:], flags)
                if end_match is None:
                    continue
                end = min(end, search_from + end_match.start())
            candidate = clean_text(text[start:end])
            if len(candidate) >= min_chars:
                return candidate
    return ""


def _fallback_ten_k_business(filing: Any) -> str:
    text = filing.text()
    return extract_text_section(
        text,
        start_patterns=[
            r"^\s*Items?\s+1\s*(?:and|&)\s*2\b.*$",
            r"^\s*Item\s+1\b.*Business.*$",
            r"^\s*Item\s+1\b(?!.*A\b)",
            r"Business and Properties",
        ],
        end_patterns=[
            r"^\s*Item\s+1A\b",
            r"^\s*Item\s+1B\b",
            r"^\s*Item\s+2\b",
            r"^\s*Item\s+3\b",
        ],
    )


def _fallback_twenty_f_business(filing: Any) -> str:
    text = filing.text()
    return extract_text_section(
        text,
        start_patterns=[
            r"^\s*Item\s+4\b.*$",
            r"^\s*(?:II|I{1,3}|IV|V)\.\s+INFORMATION ON THE COMPANY\b",
            r"^\s*INFORMATION ON THE COMPANY\b",
        ],
        end_patterns=[
            r"^\s*Item\s+4A\b",
            r"^\s*Item\s+5\b",
            r"^\s*UNRESOLVED STAFF COMMENTS\b",
            r"^\s*OPERATING AND FINANCIAL REVIEW AND PROSPECTS\b",
        ],
    )


def _fallback_forty_f_business(report: Any) -> str:
    text = getattr(report, "aif_text", "") or ""
    return extract_text_section(
        text,
        start_patterns=[
            r"^\s*GENERAL DEVELOPMENT OF THE BUSINESS\b",
            r"^\s*DESCRIPTION OF THE BUSINESS\b",
            r"^\s*BUSINESS OVERVIEW\b",
            r"^\s*BUSINESS OPERATIONS\b",
        ],
        end_patterns=[
            r"^\s*RISK FACTORS\b",
            r"^\s*RISKS RELATED TO THE BUSINESS\b",
            r"^\s*DIVIDENDS\b",
            r"^\s*DESCRIPTION OF CAPITAL STRUCTURE\b",
            r"^\s*MARKET FOR SECURITIES\b",
            r"^\s*DIRECTORS AND (?:EXECUTIVE )?OFFICERS\b",
            r"^\s*LEGAL (?:PROCEEDINGS|MATTERS)\b",
            r"^\s*MATERIAL PROPERTIES\b",
            r"^\s*CORPORATE STRUCTURE\b",
        ],
    )


def normalize_dimensions(raw_dimensions: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(raw_dimensions, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in raw_dimensions.items():
        if key in DIMENSIONS:
            dim = key
        else:
            dim = TITLE_TO_DIMENSION.get(clean_text(key))
        if not dim:
            continue
        text = clean_text(value)
        if text:
            normalized[dim] = text
    return normalized


def configure_edgar_identity(user_agent: str) -> None:
    try:
        import edgar

        os.environ["EDGAR_IDENTITY"] = user_agent
        edgar.set_identity(user_agent)
    except Exception:  # noqa: BLE001
        return


def fetch_business_description(ticker: str, cik: str, user_agent: str) -> str:
    """Robustly extract Item-1 / business-section text from the latest annual filing.

    Strategy (mirrors ``stock_matching/extract_latest_annual_items.py``):
      1. Try the structured edgartools ``filing.obj()`` path for the correct form.
      2. If that yields nothing, apply regex-based fallback on the raw filing text.
    """
    if not clean_text(cik):
        return ""
    try:
        configure_edgar_identity(user_agent)
        from edgar import Company

        company = Company(clean_text(cik))

        # Try base forms first, then amended forms
        form_attempts = [
            ("10-K", None),
            ("20-F", None),
            ("40-F", None),
            ("10-K/A", None),
            ("20-F/A", None),
            ("40-F/A", None),
        ]
        latest_filing = None
        used_form = ""
        for form, _ in form_attempts:
            filings = company.get_filings(form=form)
            if filings and len(filings) > 0:
                latest_filing = filings[0]
                used_form = form
                break

        if latest_filing is None:
            return ""

        form_base = used_form.replace("/A", "")

        # ── Step 1: structured extraction via filing.obj() ──
        try:
            filing_obj = latest_filing.obj()
        except Exception:  # noqa: BLE001
            filing_obj = None

        business_text = ""
        if filing_obj is not None:
            if form_base == "10-K":
                business_text = clean_text(
                    getattr(filing_obj, "business", "") or filing_obj["Item 1"]
                )
            elif form_base == "20-F":
                business_text = clean_text(
                    getattr(filing_obj, "business", "") or filing_obj["Item 4"]
                )
            elif form_base == "40-F":
                business_text = clean_text(getattr(filing_obj, "business", ""))
                if not business_text:
                    business_text = _fallback_forty_f_business(filing_obj)

        # ── Step 2: fallback via text regex ──
        if not business_text:
            if form_base == "10-K":
                business_text = _fallback_ten_k_business(latest_filing)
            elif form_base == "20-F":
                business_text = _fallback_twenty_f_business(latest_filing)
            elif form_base == "40-F":
                business_text = _fallback_forty_f_business(filing_obj)

        return business_text

    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Failed to fetch annual business text for {ticker} (CIK={cik}): {exc}")
        return ""


def refine_to_dimensions(client: APIClient, business_text: str, ticker: str) -> dict[str, str]:
    if not business_text.strip():
        return {}
    user_prompt = f"Company ticker: {ticker}\n\nBusiness Description:\n{business_text}"
    try:
        result = client.chat_completion_json(
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] LLM refinement failed for {ticker}: {exc}")
        return {}
    return {dim: clean_text(result.get(dim)) for dim in DIMENSIONS if clean_text(result.get(dim))}


def company_from_local_cache(row: dict[str, Any]) -> dict[str, Any] | None:
    ticker = clean_text(row.get("ticker")).upper()
    if not ticker:
        return None
    return {
        "ticker": ticker,
        "cik": clean_text(row.get("cik")),
        "company_name": clean_text(row.get("company_name")),
        "industry_plate": clean_text(row.get("industry_plate")),
        "market_cap_usd": to_float(row.get("market_cap_usd")),
        "dimensions": normalize_dimensions(row.get("dimensions")),
    }


def company_from_refined_row(row: dict[str, Any]) -> dict[str, Any] | None:
    ticker = clean_text(row.get("primary_ticker")).upper()
    if not ticker:
        return None
    return {
        "ticker": ticker,
        "cik": clean_text(row.get("cik")),
        "company_name": (
            clean_text(row.get("company_tickers_title"))
            or clean_text(row.get("edgar_name"))
            or ticker
        ),
        "industry_plate": clean_text(row.get("Industry Plate")),
        "market_cap_usd": to_float(row.get("total_market_cap_usd")),
        "dimensions": normalize_dimensions(row.get("refined_business_summary")),
    }


def company_from_similarity_meta(row: dict[str, Any]) -> dict[str, Any] | None:
    ticker = clean_text(row.get("primary_ticker")).upper()
    if not ticker:
        return None
    return {
        "ticker": ticker,
        "cik": clean_text(row.get("cik")),
        "company_name": clean_text(row.get("company_name")) or ticker,
        "industry_plate": clean_text(row.get("industry_plate")),
        "market_cap_usd": to_float(row.get("market_cap_usd")),
        "dimensions": normalize_dimensions(row.get("dimension_texts")),
    }


def has_complete_dimensions(company: dict[str, Any] | None) -> bool:
    if not isinstance(company, dict):
        return False
    dimensions = company.get("dimensions", {})
    return isinstance(dimensions, dict) and all(clean_text(dimensions.get(dim)) for dim in DIMENSIONS)


def load_local_summary_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        company = company_from_local_cache(row)
        if company is None:
            continue
        cache[company["ticker"]] = company
    return cache


def is_text_artifact_run_dir(path: Path) -> bool:
    return any((path / name).exists() for name in ("annual_items", "summaries", "similarity"))


def iter_text_artifact_run_dirs(roots: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if is_text_artifact_run_dir(root) else [
            path for path in sorted(root.iterdir()) if path.is_dir() and is_text_artifact_run_dir(path)
        ]
        for candidate in candidates:
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            run_dirs.append(candidate.resolve())
    return run_dirs


def load_external_summary_cache(run_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        refined_path = run_dir / "summaries" / "refined_business_descriptions.json"
        if refined_path.exists():
            payload = json.loads(refined_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for row in payload:
                    if not isinstance(row, dict):
                        continue
                    if clean_text(row.get("refined_business_status")) != "success":
                        continue
                    company = company_from_refined_row(row)
                    if company is None:
                        continue
                    cache.setdefault(company["ticker"], company)

        meta_path = run_dir / "similarity" / "similarity_metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            for row in metadata.get("business_companies", []):
                if not isinstance(row, dict):
                    continue
                company = company_from_similarity_meta(row)
                if company is None:
                    continue
                cache.setdefault(company["ticker"], company)

    return cache


def load_local_embedding_cache(output_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    metadata_path = output_dir / "embeddings_metadata.json"
    npz_path = output_dir / "embeddings_vectors.npz"
    if not metadata_path.exists() or not npz_path.exists():
        return {}

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    companies = metadata.get("companies", [])
    data = np.load(npz_path)
    cache: dict[str, dict[str, np.ndarray]] = {}
    for row in companies:
        if not isinstance(row, dict):
            continue
        ticker = clean_text(row.get("ticker")).upper()
        index = row.get("index")
        if not ticker or not isinstance(index, int):
            continue
        vectors: dict[str, np.ndarray] = {}
        for dim in DIMENSIONS:
            if dim not in data or index >= len(data[dim]):
                vectors = {}
                break
            vectors[dim] = np.asarray(data[dim][index], dtype=np.float32)
        if len(vectors) == len(DIMENSIONS):
            cache[ticker] = vectors
    return cache


def load_external_embedding_cache(run_dirs: list[Path]) -> dict[str, dict[str, np.ndarray]]:
    cache: dict[str, dict[str, np.ndarray]] = {}
    for run_dir in run_dirs:
        metadata_path = run_dir / "similarity" / "similarity_metadata.json"
        npz_path = run_dir / "similarity" / "similarity_embeddings.npz"
        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        tickers_array = data["business_company_tickers"] if "business_company_tickers" in data else []
        matrices = {}
        for dim in DIMENSIONS:
            key = f"{dim}_embeddings"
            matrices[dim] = data[key] if key in data else None
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            business_companies = metadata.get("business_companies", [])
        else:
            business_companies = [{} for _ in range(len(tickers_array))]

        for index, row in enumerate(business_companies):
            if not isinstance(row, dict):
                continue
            ticker = clean_text(row.get("primary_ticker")).upper()
            if not ticker and index < len(tickers_array):
                ticker = clean_text(tickers_array[index]).upper()
            if not ticker or ticker in cache:
                continue

            vectors: dict[str, np.ndarray] = {}
            for dim in DIMENSIONS:
                matrix = matrices[dim]
                if matrix is None or index >= len(matrix):
                    vectors = {}
                    break
                vectors[dim] = np.asarray(matrix[index], dtype=np.float32)
            if len(vectors) == len(DIMENSIONS):
                cache[ticker] = vectors

    return cache


def build_missing_input_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ticker = clean_text(row.get("ticker")).upper()
        if not ticker:
            continue
        company_name = clean_text(row.get("company_name")) or ticker
        rows.append(
            {
                "primary_ticker": ticker,
                "cik": clean_text(row.get("cik")),
                "company_tickers_title": company_name,
                "edgar_name": company_name,
                "Industry Plate": clean_text(row.get("industry_plate")),
                "total_market_cap_usd": to_float(row.get("market_cap_usd")),
            }
        )
    return rows


def load_text_pairing_function(double_tower_root: Path, module_name: str, attr_name: str):
    root_text = str(double_tower_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)


def can_use_external_text_pipeline(double_tower_root: Path | None, submissions_zip: Path) -> bool:
    return bool(double_tower_root and double_tower_root.exists() and submissions_zip.exists())


def extract_missing_annual_rows(
    missing_df: pd.DataFrame,
    output_dir: Path,
    double_tower_root: Path | None,
    submissions_zip: Path,
    edgar_data_dir: Path,
    edgar_cache_dir: Path,
    sec_user_agent: str,
    filing_year: int,
) -> list[dict[str, Any]]:
    if missing_df.empty:
        return []
    if not can_use_external_text_pipeline(double_tower_root, submissions_zip):
        return []

    from utils.data_io import write_csv

    extractor = load_text_pairing_function(
        double_tower_root,
        "text_pairing.extract_annual_items",
        "run_latest_annual_items_extraction",
    )
    input_rows = build_missing_input_rows(missing_df)
    input_csv = output_dir / "annual_items" / "missing_companies.csv"
    write_csv(input_csv, input_rows)
    result = extractor(
        input_csv=input_csv,
        output_dir=output_dir / "annual_items",
        output_prefix="latest_annual_items_missing",
        filing_year=max(0, int(filing_year)),
        submissions_zip=submissions_zip,
        user_agent=sec_user_agent,
        edgar_data_dir=edgar_data_dir,
        edgar_cache_dir=edgar_cache_dir,
        progress_every=10,
        max_retries=2,
    )
    return result["rows"]


def refine_missing_companies(
    annual_rows: list[dict[str, Any]],
    output_dir: Path,
    double_tower_root: Path | None,
    api_key_env: str,
    summary_api_url: str,
    summary_model: str,
    summary_workers: int,
    summary_max_retries: int,
    summary_request_timeout: int,
) -> list[dict[str, Any]]:
    if not annual_rows:
        return []
    if double_tower_root is None:
        return []

    annual_json = output_dir / "annual_items" / "latest_annual_items_missing.json"
    if not annual_json.exists():
        annual_json.parent.mkdir(parents=True, exist_ok=True)
        annual_json.write_text(json.dumps(annual_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    refiner = load_text_pairing_function(
        double_tower_root,
        "text_pairing.refine_descriptions",
        "run_refined_description_pipeline",
    )
    result = refiner(
        input_json=annual_json,
        output_dir=output_dir / "summaries",
        output_json="refined_business_descriptions_missing.json",
        api_key_env=api_key_env,
        api_url=summary_api_url,
        model=summary_model,
        workers=max(1, int(summary_workers)),
        max_retries=max(1, int(summary_max_retries)),
        request_timeout=max(1, int(summary_request_timeout)),
        overwrite=False,
        limit=0,
    )

    companies: list[dict[str, Any]] = []
    for row in result["rows"]:
        if not isinstance(row, dict):
            continue
        if clean_text(row.get("refined_business_status")) != "success":
            continue
        company = company_from_refined_row(row)
        if company is not None:
            companies.append(company)
    return companies


def fallback_refine_missing_companies(
    client: APIClient,
    missing_df: pd.DataFrame,
    sec_user_agent: str,
) -> tuple[list[dict[str, Any]], int]:
    companies: list[dict[str, Any]] = []
    extracted_count = 0
    if missing_df.empty:
        return companies, extracted_count

    print("External text pipeline unavailable; using built-in SEC filing fetch + LLM refinement fallback.")
    for _, row in missing_df.iterrows():
        ticker = clean_text(row.get("ticker")).upper()
        cik = clean_text(row.get("cik"))
        business_text = fetch_business_description(ticker, cik, sec_user_agent)
        if business_text:
            extracted_count += 1
        dimensions = refine_to_dimensions(client, business_text, ticker)
        companies.append(
            {
                "ticker": ticker,
                "cik": cik,
                "company_name": clean_text(row.get("company_name")) or ticker,
                "industry_plate": clean_text(row.get("industry_plate")),
                "market_cap_usd": to_float(row.get("market_cap_usd")),
                "dimensions": dimensions,
            }
        )
        time.sleep(0.2)
    return companies, extracted_count


def embed_companies_with_cache(
    client: APIClient,
    companies: list[dict[str, Any]],
    embedding_model: str,
    batch_size: int,
    embedding_cache: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], int, int]:
    jobs: list[tuple[int, str, str, str]] = []
    assigned: dict[tuple[int, str], np.ndarray] = {}
    reused_tickers: set[str] = set()
    embedding_dim: int | None = None

    for index, company in enumerate(companies):
        ticker = clean_text(company.get("ticker")).upper()
        cached = embedding_cache.get(ticker)
        if cached and all(dim in cached for dim in DIMENSIONS):
            ok = True
            for dim in DIMENSIONS:
                vector = np.asarray(cached[dim], dtype=np.float32)
                if vector.ndim != 1 or vector.size == 0:
                    ok = False
                    break
                if embedding_dim is None:
                    embedding_dim = int(vector.shape[0])
                elif int(vector.shape[0]) != embedding_dim:
                    ok = False
                    break
                assigned[(index, dim)] = vector
            if ok:
                reused_tickers.add(ticker)
                continue
            for dim in DIMENSIONS:
                assigned.pop((index, dim), None)

        for dim in DIMENSIONS:
            text = clean_text(company.get("dimensions", {}).get(dim))
            if text:
                jobs.append((index, ticker, dim, text))

    if jobs:
        total_batches = (len(jobs) + batch_size - 1) // batch_size
        print(f"Embedding {len(jobs)} texts in batches of {batch_size}...")
        for batch_start in range(0, len(jobs), batch_size):
            batch = jobs[batch_start:batch_start + batch_size]
            texts = [job[3] for job in batch]
            vectors = client.embed_batch(texts, model=embedding_model)
            for (index, _ticker, dim, _text), vector in zip(batch, vectors):
                array = np.asarray(vector, dtype=np.float32)
                if embedding_dim is None:
                    embedding_dim = int(array.shape[0])
                elif int(array.shape[0]) != embedding_dim:
                    raise ValueError(
                        f"Embedding dimension changed within the same run: {embedding_dim} -> {array.shape[0]}"
                    )
                assigned[(index, dim)] = array
            print(
                f"  Embedded batch {batch_start // batch_size + 1}/{total_batches}",
                flush=True,
            )

    if embedding_dim is None:
        raise ValueError("No embeddings available")

    matrices = {
        dim: np.zeros((len(companies), embedding_dim), dtype=np.float32)
        for dim in DIMENSIONS
    }
    for (index, dim), vector in assigned.items():
        matrices[dim][index] = vector

    embedded_tickers = {ticker for _, ticker, _, _ in jobs}
    return matrices, len(reused_tickers), len(embedded_tickers)


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    config = yaml.safe_load(Path(config_path).read_text())
    embed_cfg = config.get("embedding", {})
    api_key_env = embed_cfg.get("api_key_env", "CLOSEAI_API_KEY")
    api_url = embed_cfg.get("api_url", "https://api.openai-proxy.org/v1/embeddings")
    embedding_model = embed_cfg.get("model", "text-embedding-3-large")
    summary_model = embed_cfg.get("summary_model", "gpt-5.4")
    summary_api_url = embed_cfg.get("summary_api_url", "https://api.openai-proxy.org/v1/chat/completions")
    batch_size = int(embed_cfg.get("batch_size", 128))
    summary_workers = int(embed_cfg.get("summary_workers", 4))
    summary_max_retries = int(embed_cfg.get("summary_max_retries", 4))
    summary_request_timeout = int(embed_cfg.get("summary_request_timeout", 300))
    reuse_roots_raw = embed_cfg.get(
        "reuse_text_artifact_roots",
        [],
    )
    if isinstance(reuse_roots_raw, str):
        reuse_roots_raw = [reuse_roots_raw]
    reuse_roots = [Path(path).expanduser().resolve() for path in reuse_roots_raw]
    run_dirs = iter_text_artifact_run_dirs(reuse_roots)
    submissions_zip = Path(embed_cfg.get("submissions_zip", DEFAULT_SUBMISSIONS_ZIP)).expanduser().resolve()
    edgar_data_dir = Path(embed_cfg.get("edgar_data_dir", DEFAULT_EDGAR_DATA_DIR)).expanduser().resolve()
    edgar_cache_dir = Path(embed_cfg.get("edgar_cache_dir", DEFAULT_EDGAR_CACHE_DIR)).expanduser().resolve()
    sec_user_agent = clean_text(embed_cfg.get("sec_user_agent")) or DEFAULT_SEC_USER_AGENT
    filing_year = int(embed_cfg.get("filing_year", 0))
    double_tower_root_raw = clean_text(embed_cfg.get("double_tower_root", ""))
    double_tower_root = Path(double_tower_root_raw).expanduser().resolve() if double_tower_root_raw else None

    base_url = api_url.rstrip("/")
    if base_url.endswith("/embeddings"):
        base_url = base_url[:-len("/embeddings")]
    if base_url.endswith("/chat/completions"):
        base_url = base_url[:-len("/chat/completions")]
    embed_client = APIClient(api_key_env=api_key_env, api_url=base_url, model=embedding_model)
    summary_base = summary_api_url.rstrip("/")
    if summary_base.endswith("/chat/completions"):
        summary_base = summary_base[:-len("/chat/completions")]
    summary_client = APIClient(api_key_env=api_key_env, api_url=summary_base, model=summary_model)

    output_path = Path(output_dir)
    universe_path = output_path / "universe_with_industry.csv"
    pairs_path = output_path / "return_correlation.csv"
    if not universe_path.exists():
        raise FileNotFoundError(f"Not found: {universe_path}. Run step2 first.")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Not found: {pairs_path}. Run step4 first.")

    universe_df = pd.read_csv(universe_path)
    pairs_df = pd.read_csv(pairs_path)
    tickers_needed = set(pairs_df["ticker_a"].tolist() + pairs_df["ticker_b"].tolist())
    needed_df = universe_df[universe_df["ticker"].isin(tickers_needed)].copy()
    needed_df["ticker"] = needed_df["ticker"].astype(str).str.upper().str.strip()
    needed_df = needed_df.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    ticker_order = needed_df["ticker"].tolist()

    cache_path = output_path / "business_summaries_cache.json"
    local_summary_cache = load_local_summary_cache(cache_path)
    external_summary_cache = load_external_summary_cache(run_dirs)

    summary_cache = dict(external_summary_cache)
    summary_cache.update(local_summary_cache)

    local_embedding_cache = load_local_embedding_cache(output_path)
    external_embedding_cache = load_external_embedding_cache(run_dirs)
    embedding_cache = dict(external_embedding_cache)
    embedding_cache.update(local_embedding_cache)

    companies_by_ticker: dict[str, dict[str, Any]] = {}
    summary_source_counts = {"local_cache": 0, "external_cache": 0, "newly_refined": 0}
    local_tickers = set(local_summary_cache)
    external_tickers = set(external_summary_cache)

    for ticker in ticker_order:
        company = summary_cache.get(ticker)
        if not has_complete_dimensions(company):
            continue
        company_copy = dict(company)
        if ticker in local_tickers:
            company_copy["summary_source"] = "local_cache"
            summary_source_counts["local_cache"] += 1
        else:
            company_copy["summary_source"] = "external_cache"
            summary_source_counts["external_cache"] += 1
        companies_by_ticker[ticker] = company_copy

    missing_tickers = [ticker for ticker in ticker_order if ticker not in companies_by_ticker]
    missing_df = needed_df[needed_df["ticker"].isin(missing_tickers)].copy()

    print(
        f"Step 5 text prep: {len(ticker_order)} tickers needed, "
        f"{len(ticker_order) - len(missing_tickers)} reused from cache, {len(missing_tickers)} missing."
    )

    extracted_annual_rows: list[dict[str, Any]] = []
    if can_use_external_text_pipeline(double_tower_root, submissions_zip):
        extracted_annual_rows = extract_missing_annual_rows(
            missing_df=missing_df,
            output_dir=output_path,
            double_tower_root=double_tower_root,
            submissions_zip=submissions_zip,
            edgar_data_dir=edgar_data_dir,
            edgar_cache_dir=edgar_cache_dir,
            sec_user_agent=sec_user_agent,
            filing_year=filing_year,
        )
        newly_refined_companies = refine_missing_companies(
            annual_rows=extracted_annual_rows,
            output_dir=output_path,
            double_tower_root=double_tower_root,
            api_key_env=api_key_env,
            summary_api_url=summary_api_url,
            summary_model=summary_model,
            summary_workers=summary_workers,
            summary_max_retries=summary_max_retries,
            summary_request_timeout=summary_request_timeout,
        )
    else:
        newly_refined_companies, extracted_count = fallback_refine_missing_companies(
            summary_client,
            missing_df,
            sec_user_agent,
        )
        extracted_annual_rows = [{}] * extracted_count
    for company in newly_refined_companies:
        ticker = clean_text(company.get("ticker")).upper()
        if not ticker or not has_complete_dimensions(company):
            continue
        company["summary_source"] = "newly_refined"
        companies_by_ticker[ticker] = company
        summary_source_counts["newly_refined"] += 1

    companies = [companies_by_ticker[ticker] for ticker in ticker_order if ticker in companies_by_ticker]
    cache_payload = []
    for company in companies:
        cache_payload.append(
            {
                "ticker": company["ticker"],
                "cik": company.get("cik", ""),
                "company_name": company.get("company_name", ""),
                "industry_plate": company.get("industry_plate", ""),
                "market_cap_usd": company.get("market_cap_usd", 0.0),
                "dimensions": company.get("dimensions", {}),
                "summary_source": company.get("summary_source", ""),
            }
        )
    write_json(cache_path, sorted(cache_payload, key=lambda item: item["ticker"]))
    print(f"Saved business summaries cache to {cache_path}")

    valid_companies = [company for company in companies if has_complete_dimensions(company)]
    print(f"Companies with complete 4-dimension summaries: {len(valid_companies)}/{len(ticker_order)}")
    if not valid_companies:
        raise ValueError("No companies with complete 4-dimension summaries. Cannot proceed.")

    matrices, reused_embedding_count, new_embedding_count = embed_companies_with_cache(
        embed_client,
        valid_companies,
        embedding_model,
        batch_size,
        embedding_cache,
    )

    npz_path = output_path / "embeddings_vectors.npz"
    np.savez_compressed(npz_path, **{dim: matrices[dim] for dim in DIMENSIONS})
    print(f"Saved embedding vectors to {npz_path}")

    metadata = {
        "model": embedding_model,
        "dimensions": DIMENSIONS,
        "company_count": len(valid_companies),
        "companies": [
            {
                "index": index,
                "ticker": company["ticker"],
                "cik": company.get("cik", ""),
                "company_name": company.get("company_name", ""),
                "industry_plate": company.get("industry_plate", ""),
                "market_cap_usd": company.get("market_cap_usd", 0.0),
                "dimension_texts": company.get("dimensions", {}),
                "summary_source": company.get("summary_source", ""),
            }
            for index, company in enumerate(valid_companies)
        ],
    }
    metadata_path = output_path / "embeddings_metadata.json"
    write_json(metadata_path, metadata)
    print(f"Saved metadata to {metadata_path}")

    summary_payload = {
        "requested_ticker_count": len(ticker_order),
        "cached_summary_count": len(ticker_order) - len(missing_tickers),
        "missing_summary_count": len(missing_tickers),
        "summary_source_counts": summary_source_counts,
        "extracted_annual_row_count": len(extracted_annual_rows),
        "valid_company_count": len(valid_companies),
        "reused_embedding_count": reused_embedding_count,
        "new_embedding_count": new_embedding_count,
        "cache_path": str(cache_path),
        "embeddings_vectors_path": str(npz_path),
        "embeddings_metadata_path": str(metadata_path),
    }
    summary_path = output_path / "embedding_summary.json"
    write_json(summary_path, summary_payload)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Text embedding pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
