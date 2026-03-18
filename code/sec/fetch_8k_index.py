"""
Step 1: Build a master index of all 8-K filings from SEC EDGAR.

Downloads quarterly master.idx files from the EDGAR full-index archive,
filters for 8-K filings, and produces a single parquet file with all
8-K filing metadata (CIK, company name, date filed, accession number).

Source: https://www.sec.gov/Archives/edgar/full-index/{YEAR}/QTR{Q}/master.idx

Usage:
    python fetch_8k_index.py                        # All years (2004-2025)
    python fetch_8k_index.py --start 2010 --end 2025
"""

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, SEC_USER_AGENT, SEC_RATE_LIMIT, DEFAULT_RETRY_ATTEMPTS
from utils import log

RAW_DIR = DATA_RAW / "guidance"
CACHE_DIR = RAW_DIR / "_index_cache"

EDGAR_INDEX_BASE = "https://www.sec.gov/Archives/edgar/full-index"
HEADERS = {"User-Agent": SEC_USER_AGENT or "datacollect research@example.com"}

# Rate limiting: SEC allows 10 req/sec
_last_request_time = 0.0


def _rate_limit():
    """Enforce SEC rate limit (10 requests/sec)."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    min_interval = 1.0 / SEC_RATE_LIMIT
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.time()


def _fetch_with_retry(url: str, attempts: int = DEFAULT_RETRY_ATTEMPTS) -> str:
    """Fetch URL text with retry and rate limiting."""
    for i in range(attempts):
        try:
            _rate_limit()
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if i == attempts - 1:
                raise
            wait = 2 ** (i + 1)
            log.warning(f"  Attempt {i+1} failed for {url}: {e}, retrying in {wait}s")
            time.sleep(wait)


def download_quarter_index(year: int, quarter: int) -> pd.DataFrame | None:
    """
    Download and parse a single quarter's master.idx.

    The master.idx file is pipe-delimited with columns:
        CIK | Company Name | Form Type | Date Filed | Filename

    Returns DataFrame filtered to 8-K filings only.
    """
    url = f"{EDGAR_INDEX_BASE}/{year}/QTR{quarter}/master.idx"

    # Check cache first
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"master_{year}_Q{quarter}.idx"

    if cache_path.exists():
        log.info(f"  Cache hit: {year} Q{quarter}")
        text = cache_path.read_text()
    else:
        log.info(f"  Downloading: {year} Q{quarter}")
        try:
            text = _fetch_with_retry(url)
        except Exception as e:
            log.warning(f"  Failed to download {year} Q{quarter}: {e}")
            return None
        cache_path.write_text(text)

    # Parse: skip header lines (first ~11 lines until the dashes)
    lines = text.strip().split("\n")
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            data_start = i + 1
            break

    if data_start == 0:
        # Try to find the data start by looking for pipe-delimited lines
        for i, line in enumerate(lines):
            if "|" in line and not line.startswith("CIK"):
                data_start = i
                break

    data_lines = lines[data_start:]
    if not data_lines:
        log.warning(f"  No data found in {year} Q{quarter}")
        return None

    # Parse pipe-delimited data
    records = []
    for line in data_lines:
        parts = line.split("|")
        if len(parts) == 5:
            records.append({
                "cik": parts[0].strip(),
                "company_name": parts[1].strip(),
                "form_type": parts[2].strip(),
                "date_filed": parts[3].strip(),
                "filename": parts[4].strip(),
            })

    if not records:
        return None

    df = pd.DataFrame(records)

    # Filter for 8-K filings (including 8-K/A amendments)
    mask = df["form_type"].str.upper().isin(["8-K", "8-K/A"])
    df = df[mask].copy()

    # Extract accession number from filename
    # filename format: edgar/data/CIK/ACCESSION_NO_DASHES/filing.txt
    df["accession_number"] = df["filename"].str.extract(
        r"edgar/data/\d+/([\d-]+)/", expand=False
    )

    df["year"] = year
    df["quarter"] = quarter

    return df


def build_8k_index(start_year: int, end_year: int) -> pd.DataFrame:
    """Download all quarterly indexes and combine into a single 8-K index."""
    log.info(f"Building 8-K index from {start_year} to {end_year}...")

    all_dfs = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            # Skip future quarters
            df = download_quarter_index(year, quarter)
            if df is not None and len(df) > 0:
                all_dfs.append(df)
                log.info(f"    {year} Q{quarter}: {len(df):,} 8-K filings")

    if not all_dfs:
        log.error("No data retrieved!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Parse date and sort
    combined["date_filed"] = pd.to_datetime(combined["date_filed"], errors="coerce")
    combined = combined.sort_values(["date_filed", "cik"]).reset_index(drop=True)

    # Pad CIK to 10 digits for API compatibility
    combined["cik_padded"] = combined["cik"].str.zfill(10)

    log.info(f"Total 8-K filings: {len(combined):,}")
    log.info(f"  Unique filers: {combined['cik'].nunique():,}")
    log.info(f"  Date range: {combined['date_filed'].min()} to {combined['date_filed'].max()}")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Build index of all 8-K filings from EDGAR"
    )
    parser.add_argument(
        "--start", type=int, default=2004,
        help="Start year (default: 2004, when 8-K reform took effect)"
    )
    parser.add_argument(
        "--end", type=int, default=2025,
        help="End year (default: 2025)"
    )
    args = parser.parse_args()

    index = build_8k_index(args.start, args.end)

    if len(index) == 0:
        log.error("No filings found. Check SEC_USER_AGENT env var.")
        sys.exit(1)

    # Save
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"eight_k_index_{args.start}_{args.end}.parquet"
    index.to_parquet(out_path, index=False)
    log.info(f"Saved 8-K index -> {out_path}")


if __name__ == "__main__":
    main()
