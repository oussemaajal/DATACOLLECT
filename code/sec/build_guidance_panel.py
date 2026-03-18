"""
Step 4: Build clean panel dataset from parsed guidance records.

Merges parsed guidance with company identifiers (CIK -> ticker mapping),
adds derived variables, deduplicates, and produces the final panel.

Input:  data/intermediate/guidance_parsed.parquet (from parse_guidance.py)
Output: data/clean/guidance_panel.parquet

Usage:
    python build_guidance_panel.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_RAW, DATA_INTERMEDIATE, DATA_CLEAN, OUTPUT_DIR,
    SEC_USER_AGENT,
)
from utils import log

RAW_DIR = DATA_RAW / "guidance"
HEADERS = {"User-Agent": SEC_USER_AGENT or "datacollect research@example.com"}


def fetch_company_tickers() -> pd.DataFrame:
    """
    Fetch SEC's CIK-to-ticker mapping from company_tickers.json.

    Returns DataFrame with columns: cik, ticker, company_name_sec.
    """
    cache_path = RAW_DIR / "company_tickers.json"

    if cache_path.exists():
        log.info("Loading cached company_tickers.json")
        data = json.loads(cache_path.read_text())
    else:
        log.info("Fetching company_tickers.json from SEC...")
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data))

    # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    records = []
    for entry in data.values():
        records.append({
            "cik": str(entry["cik_str"]).zfill(10),
            "ticker": entry["ticker"],
            "company_name_sec": entry["title"],
        })

    df = pd.DataFrame(records)
    log.info(f"  {len(df):,} companies in ticker mapping")
    return df


def fetch_company_metadata(cik_padded: str) -> dict | None:
    """
    Fetch company metadata from SEC submissions API.
    Returns SIC code, exchange, state of incorporation, etc.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    try:
        time.sleep(0.1)  # Rate limit
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return {
            "sic": data.get("sic", ""),
            "sic_description": data.get("sicDescription", ""),
            "state_of_incorporation": data.get("stateOfIncorporation", ""),
            "fiscal_year_end": data.get("fiscalYearEnd", ""),
            "exchange": data.get("exchanges", [""])[0] if data.get("exchanges") else "",
            "category": data.get("category", ""),
        }
    except Exception:
        return None


def build_panel(guidance: pd.DataFrame) -> pd.DataFrame:
    """Build clean panel from parsed guidance + company identifiers."""

    # --- Merge with tickers ---
    tickers = fetch_company_tickers()

    # Normalize CIK for merge
    guidance["cik_padded"] = guidance["cik"].astype(str).str.zfill(10)

    panel = guidance.merge(
        tickers,
        left_on="cik_padded",
        right_on="cik",
        how="left",
        suffixes=("", "_ticker"),
    )
    n_matched = panel["ticker"].notna().sum()
    log.info(f"Ticker match rate: {n_matched:,}/{len(panel):,} "
             f"({100 * n_matched / len(panel):.1f}%)")

    # --- Enrich with SIC codes for a sample (optional, slow) ---
    # For the full dataset, SIC enrichment should be a separate batch step.
    # Here we just use the ticker mapping.

    # --- Derived variables ---
    log.info("Adding derived variables...")

    panel["date_filed"] = pd.to_datetime(panel["date_filed"], errors="coerce")
    panel["file_year"] = panel["date_filed"].dt.year
    panel["file_quarter"] = panel["date_filed"].dt.quarter
    panel["file_yearq"] = (
        panel["file_year"].astype(str) + "Q" + panel["file_quarter"].astype(str)
    )

    # Guidance midpoint (already computed, but ensure consistency)
    panel["midpoint"] = panel.apply(
        lambda r: (r["val1"] + r["val2"]) / 2
        if pd.notna(r["val2"]) and r["val2"] != r["val1"]
        else r["val1"],
        axis=1,
    )

    # Is this per-share data? (EPS, FFO per share typically < $100)
    panel["is_per_share"] = (
        panel["metric"].isin(["EPS", "FFO"])
        | ((panel["val1"].abs() < 200) & (panel["val2"].abs() < 200))
    )

    # --- Deduplication ---
    # Same company + date + metric + values = duplicate
    log.info("Deduplicating...")
    before = len(panel)
    panel = panel.drop_duplicates(
        subset=["cik_padded", "date_filed", "metric", "val1", "val2"],
        keep="first",
    )
    log.info(f"  Removed {before - len(panel):,} duplicates")

    # --- Column selection and ordering ---
    cols = [
        # Identifiers
        "cik_padded", "ticker", "company_name", "company_name_sec",
        # Filing info
        "accession_number", "date_filed", "file_year", "file_quarter", "file_yearq",
        "form_type",
        # Guidance details
        "metric", "pdicity", "fiscal_year",
        "val1", "val2", "midpoint", "range_width", "is_range",
        "action",
        "is_per_share",
        # Source
        "sentence", "exhibit_url",
    ]
    cols = [c for c in cols if c in panel.columns]
    panel = panel[cols].sort_values(
        ["ticker", "date_filed", "metric"]
    ).reset_index(drop=True)

    # Rename cik_padded to cik
    panel = panel.rename(columns={"cik_padded": "cik"})

    return panel


def generate_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Generate year x metric summary statistics."""
    summary = panel.groupby(["file_year", "metric"]).agg(
        n_guidance=("cik", "size"),
        n_firms=("cik", "nunique"),
        pct_range=("is_range", "mean"),
        avg_midpoint=("midpoint", "mean"),
    ).reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build clean guidance panel dataset"
    )
    parser.add_argument(
        "--input", type=str, default="guidance_parsed.parquet",
        help="Input parquet filename (in data/intermediate/)"
    )
    args = parser.parse_args()

    # Load parsed guidance
    input_path = DATA_INTERMEDIATE / args.input
    log.info(f"Loading parsed guidance from {input_path}")
    guidance = pd.read_parquet(input_path)
    log.info(f"  {len(guidance):,} guidance records")

    # Build panel
    panel = build_panel(guidance)

    log.info(f"Final panel: {len(panel):,} rows")
    log.info(f"  Unique firms (CIK): {panel['cik'].nunique():,}")
    if "ticker" in panel.columns:
        log.info(f"  Unique tickers: {panel['ticker'].nunique():,}")
    log.info(f"  Date range: {panel['date_filed'].min()} to {panel['date_filed'].max()}")
    log.info(f"  Metrics: {panel['metric'].value_counts().to_dict()}")

    # Save clean panel
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    clean_path = DATA_CLEAN / "guidance_panel.parquet"
    panel.to_parquet(clean_path, index=False)
    log.info(f"Saved clean panel -> {clean_path}")

    # Save summary
    summary = generate_summary(panel)
    summary_path = DATA_CLEAN / "guidance_panel_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved summary -> {summary_path}")

    # Save a small CSV sample for quick inspection
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = OUTPUT_DIR / "guidance_panel_sample.csv"
    panel.head(500).to_csv(sample_path, index=False)
    log.info(f"Saved sample (500 rows) -> {sample_path}")

    log.info("Panel construction complete.")


if __name__ == "__main__":
    main()
