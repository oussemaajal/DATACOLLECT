"""
Step 3 (LLM): Extract structured management guidance from 8-K press releases
using an OpenAI-hosted model with structured JSON output.

Drop-in replacement for parse_guidance.py -- reads the same input
(eight_k_exhibits.parquet) and writes the same output (guidance_parsed.parquet)
with additional fields enabled by LLM extraction.

Output fields per guidance statement:
    forecast_metric      EPS, Revenue, EBITDA, Capex, FreeCashFlow, ...
    time_horizon         Q1, Q2, Q3, Q4, H1, H2, FY
    fiscal_year          2025, 2026, ...
    estimate_type        point | range
    value_low            lower bound (raw dollars or percent)
    value_high           upper bound
    value_point          point estimate (when not a range)
    unit                 per_share | dollars | millions | billions | percent | basis_points
    direction            initiates | raises | lowers | narrows | reaffirms | withdraws | issues
    confidence_language  expects | anticipates | projects | guides | ...
    is_adjusted          true if non-GAAP / adjusted / comparable
    context_sentence     source sentence from the press release

Usage:
    python parse_guidance_llm.py
    python parse_guidance_llm.py --input eight_k_exhibits.parquet --resume
    python parse_guidance_llm.py --model gpt-4o --concurrency 5
    python parse_guidance_llm.py --max-cost 50.0 --fallback-regex
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DATA_RAW, DATA_INTERMEDIATE,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_RATE_LIMIT,
    DEFAULT_RETRY_ATTEMPTS,
)
from utils import log

RAW_DIR = DATA_RAW / "guidance"
CHECKPOINT_PATH = RAW_DIR / "_llm_parse_checkpoint.json"
BATCH_DIR = RAW_DIR / "_llm_batches"

# ── Cost per 1M tokens (updated for common models) ──────────────────────
MODEL_COSTS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}

# ── JSON schema for structured output ───────────────────────────────────
GUIDANCE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "forecast_metric": {
            "type": "string",
            "enum": [
                "EPS", "Revenue", "NetIncome", "EBITDA", "EBIT",
                "OperatingIncome", "GrossMargin", "GrossProfit",
                "FreeCashFlow", "OperatingCashFlow", "Capex",
                "TaxRate", "FFO", "SubscriptionRevenue",
                "GrossBookings", "AdjustedEBITDA", "Other",
            ],
        },
        "time_horizon": {
            "type": "string",
            "enum": ["Q1", "Q2", "Q3", "Q4", "H1", "H2", "FY"],
        },
        "fiscal_year": {"type": ["integer", "null"]},
        "estimate_type": {
            "type": "string",
            "enum": ["point", "range"],
        },
        "value_low": {"type": ["number", "null"]},
        "value_high": {"type": ["number", "null"]},
        "value_point": {"type": ["number", "null"]},
        "unit": {
            "type": "string",
            "enum": [
                "per_share", "dollars", "millions", "billions",
                "percent", "basis_points",
            ],
        },
        "direction": {
            "type": "string",
            "enum": [
                "initiates", "raises", "lowers", "narrows",
                "reaffirms", "withdraws", "issues",
            ],
        },
        "confidence_language": {"type": "string"},
        "is_adjusted": {"type": "boolean"},
        "context_sentence": {"type": "string"},
    },
    "required": [
        "forecast_metric", "time_horizon", "fiscal_year",
        "estimate_type", "value_low", "value_high", "value_point",
        "unit", "direction", "confidence_language", "is_adjusted",
        "context_sentence",
    ],
    "additionalProperties": False,
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "guidance_items": {
            "type": "array",
            "items": GUIDANCE_ITEM_SCHEMA,
        },
    },
    "required": ["guidance_items"],
    "additionalProperties": False,
}

# ── System prompt ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a financial data extraction assistant. Your task is to extract every forward-looking management guidance statement from SEC 8-K press release text.

RULES:
1. Each distinct (metric + time_horizon + fiscal_year) combination is a SEPARATE item.
2. Only extract FORWARD-LOOKING guidance (what the company expects for future periods). Do NOT extract reported/actual results for completed periods.
3. For ranges: estimate_type="range", populate value_low and value_high, leave value_point null.
4. For point estimates: estimate_type="point", populate value_point, leave value_low and value_high null.
5. Normalize dollar amounts to raw numbers. "$10.2 billion" → 10200000000. "$1.50" per share → 1.50 with unit="per_share".
6. For percentage metrics (margins, tax rates, growth rates): unit="percent", store the number as-is (46 for 46%).
7. For per-share metrics (EPS, FFO/share): unit="per_share".
8. For dollar amounts in millions: you may use unit="millions" and store the number in millions, OR unit="dollars" and store the fully expanded number. Be consistent within a response -- prefer unit="dollars" with fully expanded numbers when possible.
9. is_adjusted=true when described as "adjusted", "non-GAAP", "comparable", "operating" (when distinguishing from GAAP), or "pro forma".
10. confidence_language: extract the verb/phrase used (e.g., "expects", "anticipates", "projects", "guides for", "sees").
11. direction: "raises" if guidance is increased from prior; "lowers" if decreased; "narrows" if range is tightened; "reaffirms" if maintained/confirmed; "initiates" if first-time guidance; "withdraws" if guidance is suspended/withdrawn; "issues" as default for standard guidance.
12. context_sentence: the key sentence or clause containing the guidance (keep under 300 chars).
13. If the company explicitly states it does not provide guidance, return an empty list.

EXAMPLES:

Input: "The Company expects second quarter fiscal 2025 revenue of $89 billion to $93 billion."
Output: {"forecast_metric": "Revenue", "time_horizon": "Q2", "fiscal_year": 2025, "estimate_type": "range", "value_low": 89000000000, "value_high": 93000000000, "value_point": null, "unit": "dollars", "direction": "issues", "confidence_language": "expects", "is_adjusted": false, "context_sentence": "The Company expects second quarter fiscal 2025 revenue of $89 billion to $93 billion."}

Input: "Palo Alto Networks raises diluted non-GAAP EPS guidance to $3.27 to $3.30 from prior $3.22 to $3.25."
Output: {"forecast_metric": "EPS", "time_horizon": "FY", "fiscal_year": 2025, "estimate_type": "range", "value_low": 3.27, "value_high": 3.30, "value_point": null, "unit": "per_share", "direction": "raises", "confidence_language": "raises guidance to", "is_adjusted": true, "context_sentence": "Palo Alto Networks raises diluted non-GAAP EPS guidance to $3.27 to $3.30 from prior $3.22 to $3.25."}

Input: "The company projects free cash flow of approximately $13.5 billion."
Output: {"forecast_metric": "FreeCashFlow", "time_horizon": "FY", "fiscal_year": 2025, "estimate_type": "point", "value_low": null, "value_high": null, "value_point": 13500000000, "unit": "dollars", "direction": "issues", "confidence_language": "projects", "is_adjusted": false, "context_sentence": "The company projects free cash flow of approximately $13.5 billion."}"""


# ── Metric name mapping (LLM → pipeline schema) ────────────────────────
_METRIC_MAP = {
    "EPS": "EPS",
    "Revenue": "Revenue",
    "NetIncome": "Net_Income",
    "EBITDA": "EBITDA",
    "EBIT": "EBITDA",
    "OperatingIncome": "Operating_Income",
    "GrossMargin": "Gross_Margin",
    "GrossProfit": "Gross_Margin",
    "FreeCashFlow": "Cash_Flow",
    "OperatingCashFlow": "Cash_Flow",
    "Capex": "Capex",
    "TaxRate": "Tax_Rate",
    "FFO": "FFO",
    "SubscriptionRevenue": "Revenue",
    "GrossBookings": "Revenue",
    "AdjustedEBITDA": "EBITDA",
    "Other": "Other",
}


def _map_to_pipeline_record(item: dict, filing_meta: dict) -> dict:
    """Convert an LLM guidance item to the pipeline's column schema."""
    is_range = item["estimate_type"] == "range"

    if is_range:
        val1 = item["value_low"]
        val2 = item["value_high"]
    else:
        val1 = item["value_point"]
        val2 = item["value_point"]

    # Ensure val1 <= val2 for ranges
    if val1 is not None and val2 is not None and val1 > val2:
        val1, val2 = val2, val1

    midpoint = ((val1 or 0) + (val2 or 0)) / 2 if val1 is not None else None

    return {
        **filing_meta,
        # Core fields (compatible with regex parser output)
        "metric": _METRIC_MAP.get(item["forecast_metric"], item["forecast_metric"]),
        "val1": val1,
        "val2": val2,
        "is_range": is_range,
        "midpoint": midpoint,
        "range_width": (val2 - val1) if is_range and val1 is not None and val2 is not None else 0.0,
        "pdicity": item["time_horizon"],
        "fiscal_year": item["fiscal_year"],
        "action": item["direction"],
        "sentence": (item.get("context_sentence") or "")[:500],
        # LLM-enriched fields
        "forecast_metric": item["forecast_metric"],
        "estimate_type": item["estimate_type"],
        "value_low": item["value_low"],
        "value_high": item["value_high"],
        "value_point": item["value_point"],
        "unit": item["unit"],
        "direction": item["direction"],
        "confidence_language": item["confidence_language"],
        "is_adjusted": item["is_adjusted"],
    }


# ── Pre-filter text to reduce token usage ───────────────────────────────
_GUIDANCE_KEYWORDS = {
    "expect", "anticipat", "project", "forecast", "guidanc", "outlook",
    "target", "estimat", "see", "call for", "rais", "lower", "narrow",
    "reaffirm", "reiterat", "confirm", "maintain", "introduc", "initiat",
    "provid", "issu", "updat", "revis", "full-year", "full year",
    "fiscal year", "quarter",
}


def prefilter_text(text: str, max_words: int = 3000) -> str:
    """Keep the first 1500 words + any later paragraphs containing guidance keywords."""
    words = text.split()
    if len(words) <= max_words:
        return text

    # Always keep the first 1500 words (narrative section)
    head = " ".join(words[:1500])

    # Scan remaining paragraphs for guidance language
    remainder = " ".join(words[1500:])
    paragraphs = remainder.split("\n\n")
    relevant = []
    for para in paragraphs:
        para_lower = para.lower()
        if any(kw in para_lower for kw in _GUIDANCE_KEYWORDS):
            relevant.append(para)

    tail = "\n\n".join(relevant)
    combined = head + "\n\n" + tail if tail else head

    # Final cap
    final_words = combined.split()
    if len(final_words) > max_words:
        combined = " ".join(final_words[:max_words])

    return combined


# ── Async LLM extraction ───────────────────────────────────────────────

async def extract_guidance_llm_async(
    text: str,
    filing_meta: dict,
    client,
    model: str,
    semaphore: asyncio.Semaphore,
    token_tracker: dict,
) -> list[dict]:
    """Extract guidance from a single press release via OpenAI API."""
    filtered_text = prefilter_text(text)

    async with semaphore:
        for attempt in range(DEFAULT_RETRY_ATTEMPTS):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Company: {filing_meta.get('company_name', 'Unknown')}\n"
                                f"Filing date: {filing_meta.get('date_filed', 'Unknown')}\n\n"
                                f"Press release text:\n{filtered_text}"
                            ),
                        },
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "guidance_extraction",
                            "strict": True,
                            "schema": RESPONSE_SCHEMA,
                        },
                    },
                    temperature=0.0,
                    timeout=120,
                )

                # Track tokens
                usage = response.usage
                if usage:
                    token_tracker["input"] += usage.prompt_tokens
                    token_tracker["output"] += usage.completion_tokens

                # Parse response
                content = response.choices[0].message.content
                parsed = json.loads(content)
                items = parsed.get("guidance_items", [])

                return [_map_to_pipeline_record(item, filing_meta) for item in items]

            except Exception as e:
                if attempt == DEFAULT_RETRY_ATTEMPTS - 1:
                    log.warning(
                        f"  LLM extraction failed for {filing_meta.get('accession_number')}: {e}"
                    )
                    return []
                wait = 2 ** (attempt + 1)
                log.debug(f"  Retry {attempt+1} in {wait}s: {e}")
                await asyncio.sleep(wait)

    return []


def extract_guidance_from_text_llm_sync(
    text: str,
    filing_meta: dict,
    client,
    model: str,
) -> list[dict]:
    """Synchronous single-filing extraction (for run_sample_test.py)."""
    filtered_text = prefilter_text(text)

    for attempt in range(DEFAULT_RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Company: {filing_meta.get('company_name', 'Unknown')}\n"
                            f"Filing date: {filing_meta.get('date_filed', 'Unknown')}\n\n"
                            f"Press release text:\n{filtered_text}"
                        ),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "guidance_extraction",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    },
                },
                temperature=0.0,
                timeout=120,
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)
            items = parsed.get("guidance_items", [])
            return [_map_to_pipeline_record(item, filing_meta) for item in items]

        except Exception as e:
            if attempt == DEFAULT_RETRY_ATTEMPTS - 1:
                log.warning(
                    f"  LLM extraction failed for {filing_meta.get('accession_number')}: {e}"
                )
                return []
            wait = 2 ** (attempt + 1)
            log.debug(f"  Retry {attempt+1} in {wait}s: {e}")
            time.sleep(wait)

    return []


# ── Checkpoint management ───────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {"processed": [], "input_tokens": 0, "output_tokens": 0}


def save_checkpoint(processed: set, token_tracker: dict):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps({
        "processed": sorted(processed),
        "input_tokens": token_tracker["input"],
        "output_tokens": token_tracker["output"],
    }))


def estimate_cost(token_tracker: dict, model: str) -> float:
    costs = MODEL_COSTS.get(model, MODEL_COSTS["gpt-4o-mini"])
    return (
        token_tracker["input"] / 1_000_000 * costs["input"]
        + token_tracker["output"] / 1_000_000 * costs["output"]
    )


# ── Main async runner ──────────────────────────────────────────────────

async def run_extraction(
    exhibits: pd.DataFrame,
    model: str,
    concurrency: int,
    max_cost: float,
    resume: bool,
    fallback_regex: bool,
    batch_size: int,
):
    from openai import AsyncOpenAI

    api_key = OPENAI_API_KEY
    if not api_key:
        log.error("OPENAI_API_KEY not set. Export it or add to .env")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)

    # Resume
    checkpoint = load_checkpoint() if resume else {"processed": [], "input_tokens": 0, "output_tokens": 0}
    processed = set(checkpoint["processed"])
    token_tracker = {
        "input": checkpoint.get("input_tokens", 0),
        "output": checkpoint.get("output_tokens", 0),
    }

    if processed:
        log.info(f"  Resuming: {len(processed):,} already processed")
        exhibits = exhibits[~exhibits["accession_number"].isin(processed)]

    log.info(f"  {len(exhibits):,} exhibits to process (model={model}, concurrency={concurrency})")

    # Regex fallback
    regex_extract = None
    if fallback_regex:
        from parse_guidance import extract_guidance_from_text
        regex_extract = extract_guidance_from_text

    all_results = []
    batch_num = 0
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Process in chunks for checkpointing
    rows = list(exhibits.iterrows())
    for chunk_start in range(0, len(rows), batch_size):
        chunk = rows[chunk_start:chunk_start + batch_size]

        # Check cost guard
        if max_cost > 0:
            current_cost = estimate_cost(token_tracker, model)
            if current_cost >= max_cost:
                log.warning(f"  Cost limit reached (${current_cost:.2f} >= ${max_cost:.2f}). Stopping.")
                break

        # Build async tasks for this chunk
        tasks = []
        for _, row in chunk:
            filing_meta = {
                "cik": row["cik"],
                "company_name": row["company_name"],
                "accession_number": row["accession_number"],
                "date_filed": row["date_filed"],
                "form_type": row["form_type"],
                "exhibit_url": row["exhibit_url"],
            }
            tasks.append((
                row["accession_number"],
                extract_guidance_llm_async(
                    row["text"], filing_meta, client, model, semaphore, token_tracker,
                ),
                filing_meta,
                row["text"],
            ))

        # Run concurrently
        results_raw = await asyncio.gather(
            *[t[1] for t in tasks], return_exceptions=True
        )

        chunk_results = []
        for (accession, _, filing_meta, text), result in zip(tasks, results_raw):
            if isinstance(result, Exception):
                log.warning(f"  Exception for {accession}: {result}")
                # Fallback to regex
                if regex_extract:
                    result = regex_extract(text, filing_meta)
                else:
                    result = []

            chunk_results.extend(result)
            processed.add(accession)

        all_results.extend(chunk_results)

        # Save batch
        if chunk_results:
            batch_num += 1
            batch_path = BATCH_DIR / f"llm_batch_{batch_num:04d}.parquet"
            pd.DataFrame(chunk_results).to_parquet(batch_path, index=False)

        save_checkpoint(processed, token_tracker)

        cost = estimate_cost(token_tracker, model)
        log.info(
            f"  Processed {min(chunk_start + batch_size, len(rows)):,}/{len(rows):,} "
            f"({len(all_results):,} records, "
            f"~${cost:.2f}, "
            f"{token_tracker['input']:,}+{token_tracker['output']:,} tokens)"
        )

    await client.close()
    return all_results, token_tracker


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse management guidance from 8-K press releases using LLM"
    )
    parser.add_argument(
        "--input", type=str, default="eight_k_exhibits.parquet",
        help="Input parquet file name (in data/raw/guidance/)",
    )
    parser.add_argument("--model", type=str, default=None, help="Override OPENAI_MODEL")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API calls")
    parser.add_argument("--max-cost", type=float, default=0.0, help="Max cost in USD (0=unlimited)")
    parser.add_argument("--fallback-regex", action="store_true", help="Fall back to regex on API failure")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch-size", type=int, default=1000, help="Checkpoint every N filings")
    args = parser.parse_args()

    model = args.model or OPENAI_MODEL

    # Load exhibits
    input_path = RAW_DIR / args.input
    log.info(f"Loading exhibits from {input_path}")
    exhibits = pd.read_parquet(input_path)
    log.info(f"  {len(exhibits):,} press releases to parse")

    # Run async extraction
    results, token_tracker = asyncio.run(run_extraction(
        exhibits=exhibits,
        model=model,
        concurrency=args.concurrency,
        max_cost=args.max_cost,
        resume=args.resume,
        fallback_regex=args.fallback_regex,
        batch_size=args.batch_size,
    ))

    if not results:
        log.warning("No guidance records extracted!")
        return

    df = pd.DataFrame(results)

    # Summary
    cost = estimate_cost(token_tracker, model)
    log.info(f"Extraction complete:")
    log.info(f"  {len(df):,} guidance records")
    log.info(f"  Metrics: {df['metric'].value_counts().to_dict()}")
    log.info(f"  Actions: {df['action'].value_counts().to_dict()}")
    log.info(f"  Periods: {df['pdicity'].value_counts().to_dict()}")
    log.info(f"  Adjusted: {df['is_adjusted'].sum():,}/{len(df):,}")
    log.info(f"  Estimate types: {df['estimate_type'].value_counts().to_dict()}")
    log.info(f"  Total tokens: {token_tracker['input']:,} in + {token_tracker['output']:,} out")
    log.info(f"  Estimated cost: ${cost:.2f} ({model})")

    # Save
    DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    out_path = DATA_INTERMEDIATE / "guidance_parsed.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved parsed guidance -> {out_path}")


if __name__ == "__main__":
    main()
