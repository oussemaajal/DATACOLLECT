"""
Step 2: Download press release exhibits from 8-K filings.

For each 8-K in the index, fetches the filing index page to identify
press release exhibits (typically EX-99.1), then downloads the exhibit text.

8-K Items relevant to management guidance:
    Item 2.02 - Results of Operations and Financial Condition
    Item 7.01 - Regulation FD Disclosure
    Item 8.01 - Other Events (sometimes contains guidance)

The filing index page at:
    https://www.sec.gov/Archives/edgar/data/{CIK}/{ACCESSION}/
lists all documents in the filing with their descriptions.

Usage:
    python download_8k_exhibits.py --index eight_k_index_2004_2025.parquet
    python download_8k_exhibits.py --index eight_k_index_2004_2025.parquet --max-filings 1000
    python download_8k_exhibits.py --index eight_k_index_2004_2025.parquet --resume
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, SEC_USER_AGENT, SEC_RATE_LIMIT, DEFAULT_RETRY_ATTEMPTS
from utils import log

RAW_DIR = DATA_RAW / "guidance"
EXHIBITS_DIR = RAW_DIR / "exhibits"
CHECKPOINT_PATH = RAW_DIR / "_download_checkpoint.json"

EDGAR_ARCHIVE_BASE = "https://www.sec.gov/Archives"
HEADERS = {"User-Agent": SEC_USER_AGENT or "datacollect research@example.com"}

_last_request_time = 0.0


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    min_interval = 1.0 / SEC_RATE_LIMIT
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.time()


def _fetch_with_retry(url: str, attempts: int = DEFAULT_RETRY_ATTEMPTS) -> str | None:
    for i in range(attempts):
        try:
            _rate_limit()
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 404:
                log.debug(f"  404 Not Found: {url}")
                return None
            if i == attempts - 1:
                raise
            wait = 2 ** (i + 1)
            log.warning(f"  HTTP {resp.status_code} for {url}, retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            if i == attempts - 1:
                raise
            wait = 2 ** (i + 1)
            log.warning(f"  Error fetching {url}: {e}, retrying in {wait}s")
            time.sleep(wait)


def _strip_html(html: str) -> str:
    """Extract text from HTML, preserving some structure."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_filing_index(index_html: str, base_url: str) -> dict:
    """
    Parse a filing index page to extract 8-K items and exhibit documents.

    Returns dict with:
        items: list of 8-K item numbers mentioned
        exhibits: list of dicts {url, description, filename, sequence}
        primary_doc: URL of the primary 8-K document
    """
    soup = BeautifulSoup(index_html, "html.parser")

    result = {"items": [], "exhibits": [], "primary_doc": None}

    # Find the document table
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            # Typical columns: Seq, Description, Document, Type, Size
            # or: Document, Type, Size, Description
            text_cells = [c.get_text(strip=True) for c in cells]
            link = row.find("a")

            if not link:
                continue

            href = link.get("href", "")
            if not href:
                continue

            # Build full URL
            if href.startswith("/"):
                doc_url = f"https://www.sec.gov{href}"
            elif not href.startswith("http"):
                doc_url = f"{base_url}/{href}"
            else:
                doc_url = href

            desc = " ".join(text_cells).upper()

            # Identify primary 8-K document
            if "8-K" in desc and not result["primary_doc"]:
                if any(ext in href.lower() for ext in [".htm", ".html", ".txt"]):
                    result["primary_doc"] = doc_url

            # Identify press release exhibits (EX-99.x)
            is_exhibit = bool(re.search(r"EX-99|EXHIBIT\s*99|PRESS\s*RELEASE|EARNINGS", desc))
            if is_exhibit and any(ext in href.lower() for ext in [".htm", ".html", ".txt"]):
                result["exhibits"].append({
                    "url": doc_url,
                    "description": " ".join(text_cells),
                    "filename": href.split("/")[-1],
                })

    # Also check for 8-K items in the primary document description or header
    header_text = soup.get_text()
    item_pattern = r"Item\s+([\d.]+)"
    items_found = re.findall(item_pattern, header_text)
    result["items"] = list(set(items_found))

    return result


def load_checkpoint() -> set:
    """Load set of already-processed accession numbers."""
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text())
        return set(data.get("processed", []))
    return set()


def save_checkpoint(processed: set):
    """Save checkpoint of processed accession numbers."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({"processed": sorted(processed)}))


def process_filing(row: pd.Series) -> dict | None:
    """
    Process a single 8-K filing:
    1. Fetch the filing index page
    2. Find press release exhibits
    3. Download exhibit text
    4. Return metadata + text

    Returns dict with filing metadata and exhibit text, or None if no relevant exhibit.
    """
    cik = row["cik"]
    accession = row["accession_number"]

    if not accession or pd.isna(accession):
        return None

    # Filing index URL
    accession_clean = accession.replace("-", "")
    index_url = f"{EDGAR_ARCHIVE_BASE}/edgar/data/{cik}/{accession_clean}"

    # Fetch filing index
    index_html = _fetch_with_retry(f"{index_url}/index.json")
    if index_html is None:
        # Fallback to HTML index
        index_html = _fetch_with_retry(f"{index_url}/")
        if index_html is None:
            return None

    # Try JSON index first (more reliable)
    exhibits_info = []
    items = []
    try:
        index_data = json.loads(index_html)
        # JSON index format: directory -> item -> [{name, type, ...}]
        for item in index_data.get("directory", {}).get("item", []):
            name = item.get("name", "")
            desc = item.get("type", "") + " " + name
            desc_upper = desc.upper()

            # Look for press release exhibits
            if any(x in desc_upper for x in ["EX-99", "EXHIBIT 99", "PRESS"]):
                if any(ext in name.lower() for ext in [".htm", ".html", ".txt"]):
                    exhibits_info.append({
                        "url": f"{index_url}/{name}",
                        "description": desc,
                        "filename": name,
                    })
    except (json.JSONDecodeError, ValueError):
        # Parse HTML index
        parsed = parse_filing_index(index_html, index_url)
        exhibits_info = parsed["exhibits"]
        items = parsed["items"]

    if not exhibits_info:
        return None

    # Download first relevant exhibit (usually the press release)
    exhibit = exhibits_info[0]
    exhibit_html = _fetch_with_retry(exhibit["url"])
    if exhibit_html is None:
        return None

    # Strip HTML to plain text
    if exhibit["filename"].lower().endswith((".htm", ".html")):
        exhibit_text = _strip_html(exhibit_html)
    else:
        exhibit_text = exhibit_html

    # Quick check: does this look like an earnings/guidance document?
    text_upper = exhibit_text[:5000].upper()
    earnings_keywords = [
        "EARNINGS", "RESULTS", "REVENUE", "GUIDANCE", "OUTLOOK",
        "EXPECTS", "FORECAST", "QUARTER", "FISCAL", "NET INCOME",
        "EPS", "PER SHARE", "DILUTED",
    ]
    if not any(kw in text_upper for kw in earnings_keywords):
        return None

    return {
        "cik": cik,
        "company_name": row["company_name"],
        "accession_number": accession,
        "date_filed": str(row["date_filed"]),
        "form_type": row["form_type"],
        "exhibit_url": exhibit["url"],
        "exhibit_filename": exhibit["filename"],
        "exhibit_description": exhibit["description"],
        "items": ",".join(items),
        "text": exhibit_text,
        "text_length": len(exhibit_text),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download press release exhibits from 8-K filings"
    )
    parser.add_argument(
        "--index", type=str, required=True,
        help="Filename of the 8-K index parquet (in data/raw/guidance/)"
    )
    parser.add_argument(
        "--max-filings", type=int, default=0,
        help="Max number of filings to process (0=all)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5000,
        help="Save intermediate results every N filings (default: 5000)"
    )
    args = parser.parse_args()

    # Load 8-K index
    index_path = RAW_DIR / args.index
    log.info(f"Loading 8-K index from {index_path}")
    index = pd.read_parquet(index_path)
    log.info(f"  {len(index):,} filings in index")

    # Resume from checkpoint if requested
    processed = load_checkpoint() if args.resume else set()
    if processed:
        log.info(f"  Resuming: {len(processed):,} already processed")

    # Filter out already-processed filings
    if processed:
        index = index[~index["accession_number"].isin(processed)]
    log.info(f"  {len(index):,} filings to process")

    if args.max_filings > 0:
        index = index.head(args.max_filings)
        log.info(f"  Limited to {len(index):,} filings")

    # Process filings
    EXHIBITS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    n_processed = 0
    n_found = 0
    batch_num = 0

    for idx, row in index.iterrows():
        n_processed += 1

        try:
            result = process_filing(row)
        except Exception as e:
            log.warning(f"  Error processing {row['accession_number']}: {e}")
            result = None

        if result is not None:
            results.append(result)
            n_found += 1

        processed.add(row["accession_number"])

        # Progress logging
        if n_processed % 100 == 0:
            log.info(f"  Processed {n_processed:,}/{len(index):,} "
                     f"({n_found:,} with guidance-relevant exhibits)")

        # Save batch checkpoint
        if n_processed % args.batch_size == 0 and results:
            batch_num += 1
            batch_path = EXHIBITS_DIR / f"exhibits_batch_{batch_num:04d}.parquet"
            pd.DataFrame(results).to_parquet(batch_path, index=False)
            log.info(f"  Saved batch {batch_num} ({len(results):,} records) -> {batch_path}")
            results = []
            save_checkpoint(processed)

    # Save final batch
    if results:
        batch_num += 1
        batch_path = EXHIBITS_DIR / f"exhibits_batch_{batch_num:04d}.parquet"
        pd.DataFrame(results).to_parquet(batch_path, index=False)
        log.info(f"  Saved batch {batch_num} ({len(results):,} records) -> {batch_path}")

    save_checkpoint(processed)

    # Combine all batches into one file
    log.info("Combining all batches...")
    batch_files = sorted(EXHIBITS_DIR.glob("exhibits_batch_*.parquet"))
    if batch_files:
        all_exhibits = pd.concat(
            [pd.read_parquet(f) for f in batch_files],
            ignore_index=True,
        )
        combined_path = RAW_DIR / "eight_k_exhibits.parquet"
        all_exhibits.to_parquet(combined_path, index=False)
        log.info(f"Saved combined exhibits ({len(all_exhibits):,} records) -> {combined_path}")
    else:
        log.warning("No exhibits found!")

    log.info(f"Done. Processed {n_processed:,} filings, "
             f"found {n_found:,} with guidance-relevant exhibits.")


if __name__ == "__main__":
    main()
