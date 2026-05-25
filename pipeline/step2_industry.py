"""Step 2: Assign industry classification using preloaded data or Futu OpenAPI.

Requires FutuOpenD gateway running locally.
Outputs: outputs/universe_with_industry.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fetch_futu_industry_mapping(
    host: str = "127.0.0.1",
    port: int = 11111,
    sleep_time: float = 5.0,
) -> dict[str, str]:
    from futu import OpenQuoteContext, Market, Plate, RET_OK

    quote_ctx = OpenQuoteContext(host=host, port=port)
    try:
        ret, plate_df = quote_ctx.get_plate_list(Market.US, Plate.INDUSTRY)
        if ret != RET_OK:
            raise RuntimeError(f"Failed to get plate list: {plate_df}")

        total = len(plate_df)
        print(f"Found {total} US industry plates from Futu.")
        ticker_to_industry: dict[str, str] = {}

        for idx, row in plate_df.iterrows():
            plate_code = row["code"]
            plate_name = row["plate_name"]

            max_retry = 3
            for attempt in range(max_retry):
                ret_s, stocks = quote_ctx.get_plate_stock(plate_code)
                if ret_s == RET_OK:
                    for _, s_row in stocks.iterrows():
                        code = str(s_row["code"])
                        # Futu code format: US.AAPL -> extract ticker
                        if "." in code:
                            ticker = code.split(".", 1)[1].upper()
                        else:
                            ticker = code.upper()
                        if ticker not in ticker_to_industry:
                            ticker_to_industry[ticker] = plate_name
                    break
                else:
                    if attempt < max_retry - 1:
                        time.sleep(3)
                    else:
                        print(f"  [WARN] Failed to get stocks for {plate_name} after {max_retry} retries")

            if (idx + 1) % 20 == 0 or (idx + 1) == total:
                print(f"  Progress: {idx + 1}/{total} plates, {len(ticker_to_industry)} tickers mapped")
            time.sleep(sleep_time)

        print(f"Industry mapping complete: {len(ticker_to_industry)} tickers.")
        return ticker_to_industry
    finally:
        quote_ctx.close()


def run(config_path: str = "config.yaml", output_dir: str = "outputs"):
    config = yaml.safe_load(Path(config_path).read_text())
    industry_cfg = config.get("industry", {})
    host = industry_cfg.get("futu_host", "127.0.0.1")
    port = int(industry_cfg.get("futu_port", 11111))
    sleep_time = float(industry_cfg.get("rate_limit_sleep", 5.0))

    universe_path = Path(output_dir) / "universe.csv"
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}. Run step1 first.")

    df = pd.read_csv(universe_path)
    total = len(df)
    if "industry_plate" in df.columns:
        df["industry_plate"] = df["industry_plate"].fillna("").astype(str).str.strip()
        matched = int((df["industry_plate"] != "").sum())
        if matched:
            df = df[df["industry_plate"] != ""].reset_index(drop=True)
            output_path = Path(output_dir) / "universe_with_industry.csv"
            df.to_csv(output_path, index=False)
            print(f"Using preloaded industry plates from {universe_path}.")
            print(f"Matched {matched}/{total} tickers to industry plates.")
            print(f"Saved to {output_path} ({len(df)} stocks with industry).")
            return df

    ticker_to_industry = fetch_futu_industry_mapping(host, port, sleep_time)

    df["industry_plate"] = df["ticker"].map(ticker_to_industry)
    matched = df["industry_plate"].notna().sum()
    print(f"Matched {matched}/{len(df)} tickers to industry plates.")

    df = df.dropna(subset=["industry_plate"]).reset_index(drop=True)
    output_path = Path(output_dir) / "universe_with_industry.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path} ({len(df)} stocks with industry).")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Industry classification via Futu")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)
