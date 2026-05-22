"""Main entry point: run the full pairs trading pipeline end-to-end.

Usage:
    python run_pipeline.py                    # Run all steps
    python run_pipeline.py --start 4 --end 9  # Run steps 4 through 9
    python run_pipeline.py --step 8           # Run only step 8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

STEPS = {
    1: ("Universe Construction", "pipeline.step1_universe"),
    2: ("Industry Classification", "pipeline.step2_industry"),
    3: ("Intra-Industry Pairing", "pipeline.step3_pairing"),
    4: ("Text Embedding", "pipeline.step4_text_embedding"),
    5: ("Text Similarity Filtering", "pipeline.step5_text_similarity"),
    6: ("Return Neutralization & Correlation", "pipeline.step6_return_correlation"),
    7: ("Candidate Selection & Concentration Control", "pipeline.step7_candidate_selection"),
    8: ("LLM Subjective Review", "pipeline.step8_llm_review"),
    9: ("Weight Optimization", "pipeline.step9_weight_optimization"),
}


def run_step(step_num: int, config_path: str, output_dir: str):
    name, module_path = STEPS[step_num]
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {name}")
    print(f"{'='*60}\n")

    import importlib
    module = importlib.import_module(module_path)
    start = time.time()
    module.run(config_path, output_dir)
    elapsed = time.time() - start
    print(f"\n  Step {step_num} completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="US Stock Pairs Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. Universe Construction     - Fetch US stocks, filter by market cap (>$5B)
  2. Industry Classification   - Assign Futu industry plates
  3. Intra-Industry Pairing    - Generate all pairs within each industry
  4. Text Embedding            - Extract SEC 10-K, refine to 4 dimensions, embed
  5. Text Similarity           - Compute weighted cosine similarity, filter
  6. Return Correlation        - Neutralize returns, compute multi-window correlation
  7. Candidate Selection       - Score, rank, apply concentration control (max 2 per company)
  8. LLM Review                - Two-stage subjective analysis (overlap + direction)
  9. Weight Optimization       - Closed-form risk-neutral weight calculation

Examples:
  python run_pipeline.py                     # Full pipeline
  python run_pipeline.py --start 4           # From step 4 onwards
  python run_pipeline.py --step 8            # Only step 8
  python run_pipeline.py --config my.yaml    # Custom config
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--start", type=int, default=1, help="First step to run (1-9)")
    parser.add_argument("--end", type=int, default=9, help="Last step to run (1-9)")
    parser.add_argument("--step", type=int, default=None, help="Run only this step")
    args = parser.parse_args()

    config_path = args.config
    output_dir = args.output_dir

    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.step is not None:
        if args.step not in STEPS:
            print(f"Invalid step: {args.step}. Must be 1-9.")
            sys.exit(1)
        run_step(args.step, config_path, output_dir)
    else:
        start = max(1, min(9, args.start))
        end = max(start, min(9, args.end))
        print(f"Running pipeline steps {start} to {end}")
        print(f"Config: {config_path}")
        print(f"Output: {output_dir}")

        total_start = time.time()
        for step_num in range(start, end + 1):
            run_step(step_num, config_path, output_dir)

        total_elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"  Pipeline complete! Total time: {total_elapsed:.1f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
