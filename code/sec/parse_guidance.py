"""
Step 3: Extract structured management guidance from 8-K press release text.

Parses earnings press releases to identify forward-looking guidance statements
and extract structured data: metric (EPS, revenue, etc.), periodicity,
values (point or range), and whether it's new, raised, lowered, or reaffirmed.

Guidance patterns extracted:
    - EPS guidance:    "expects EPS of $1.50 to $1.60"
    - Revenue guidance: "anticipates revenue of $10.2 billion to $10.5 billion"
    - EBITDA guidance:  "projects adjusted EBITDA of $500 million"
    - Net income:       "expects net income of $200 million to $220 million"
    - General:          "raises full-year outlook", "reaffirms guidance"

Usage:
    python parse_guidance.py
    python parse_guidance.py --input eight_k_exhibits.parquet
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_INTERMEDIATE
from utils import log

RAW_DIR = DATA_RAW / "guidance"

# ── Dollar value patterns ─────────────────────────────────────────────────
# Matches: $1.50, $10.2 billion, $500 million, $1,200, ($0.50)
_DOLLAR = r"""
    \(?\$\s*                       # Dollar sign, optional opening paren
    (\d{1,3}(?:,\d{3})*            # Integer part with optional commas
     (?:\.\d{1,4})?)               # Optional decimal
    \s*(billion|million|thousand|   # Optional scale word
        bil|mil|mm|bn|m|k|B|M)?
    \)?                             # Optional closing paren
"""
_DOLLAR_RE = re.compile(_DOLLAR, re.VERBOSE | re.IGNORECASE)

# Range pattern: $X.XX to $Y.YY  or  $X.XX - $Y.YY  or  $X.XX and $Y.YY
_RANGE = re.compile(
    r"\(?\$\s*"
    r"(\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?)"
    r"\s*(billion|million|thousand|bil|mil|mm|bn|m|k|B|M)?"
    r"\)?\s*"
    r"(?:to|through|and|-|–|—)\s*"
    r"\(?\$\s*"
    r"(\d{1,3}(?:,\d{3})*(?:\.\d{1,4})?)"
    r"\s*(billion|million|thousand|bil|mil|mm|bn|m|k|B|M)?"
    r"\)?",
    re.IGNORECASE,
)

# Per-share range: $X.XX to $Y.YY (no scale word needed for per-share)
_PER_SHARE_RANGE = re.compile(
    r"\(?\$\s*"
    r"(\d{1,2}\.\d{2})"
    r"\)?\s*"
    r"(?:to|through|and|-|–|—)\s*"
    r"\(?\$\s*"
    r"(\d{1,2}\.\d{2})"
    r"\)?",
    re.IGNORECASE,
)

# ── Metric patterns ──────────────────────────────────────────────────────
METRIC_PATTERNS = {
    "EPS": re.compile(
        r"(?:earnings?\s+per\s+(?:diluted\s+)?share|"
        r"diluted\s+(?:earnings?\s+per\s+share|EPS)|"
        r"EPS|"
        r"(?:net\s+)?(?:income|earnings?)\s+per\s+(?:common\s+)?share)",
        re.IGNORECASE,
    ),
    "Revenue": re.compile(
        r"(?:(?:total\s+|net\s+)?revenue(?:s)?|"
        r"(?:total\s+|net\s+)?sales|"
        r"top[\s-]line)",
        re.IGNORECASE,
    ),
    "Net_Income": re.compile(
        r"(?:net\s+(?:income|earnings?)(?:\s+attributable)?|"
        r"bottom[\s-]line)",
        re.IGNORECASE,
    ),
    "EBITDA": re.compile(
        r"(?:(?:adjusted\s+)?EBITDA(?:\s+margin)?|"
        r"(?:adjusted\s+)?EBIT(?:\s+margin)?)",
        re.IGNORECASE,
    ),
    "Operating_Income": re.compile(
        r"(?:operating\s+(?:income|profit|earnings?)|"
        r"(?:income|profit)\s+from\s+operations)",
        re.IGNORECASE,
    ),
    "FFO": re.compile(
        r"(?:(?:adjusted\s+)?(?:funds?\s+from\s+operations?|FFO)(?:\s+per\s+share)?)",
        re.IGNORECASE,
    ),
    "Cash_Flow": re.compile(
        r"(?:(?:free\s+)?cash\s+flow(?:\s+from\s+operations)?|"
        r"(?:operating\s+)?cash\s+flow|FCF)",
        re.IGNORECASE,
    ),
    "Capex": re.compile(
        r"(?:capital\s+expenditure(?:s)?|capex|cap[\s-]ex)",
        re.IGNORECASE,
    ),
    "Gross_Margin": re.compile(
        r"(?:gross\s+(?:margin|profit)(?:\s+margin)?)",
        re.IGNORECASE,
    ),
    "Tax_Rate": re.compile(
        r"(?:(?:effective\s+)?tax\s+rate)",
        re.IGNORECASE,
    ),
}

# ── Guidance action patterns ──────────────────────────────────────────────
GUIDANCE_VERBS = re.compile(
    r"(?:(?:the\s+company\s+|management\s+|we\s+)?"
    r"(?:expect(?:s|ing)?|"
    r"anticipate(?:s|d|ing)?|"
    r"project(?:s|ed|ing)?|"
    r"forecast(?:s|ed|ing)?|"
    r"guid(?:e[sd]?|ance|ing)|"
    r"target(?:s|ed|ing)?|"
    r"estimat(?:e[sd]?|ing)|"
    r"see(?:s|ing)?|"
    r"call(?:s|ed|ing)?\s+for|"
    r"rais(?:e[sd]?|ing)|"
    r"lower(?:s|ed|ing)?|"
    r"narrow(?:s|ed|ing)?|"
    r"reaffirm(?:s|ed|ing)?|"
    r"reiterat(?:e[sd]?|ing)|"
    r"confirm(?:s|ed|ing)?|"
    r"maintain(?:s|ed|ing)?|"
    r"introduc(?:e[sd]?|ing)|"
    r"initiat(?:e[sd]?|ing)|"
    r"provid(?:e[sd]?|ing)|"
    r"issu(?:e[sd]?|ing)|"
    r"updat(?:e[sd]?|ing)|"
    r"revis(?:e[sd]?|ing))\s+"
    r"(?:its\s+|the\s+|a\s+|an\s+|full[\s-]year\s+|"
    r"fiscal[\s-]year\s+|quarterly\s+|annual\s+|"
    r"first[\s-]quarter\s+|second[\s-]quarter\s+|"
    r"third[\s-]quarter\s+|fourth[\s-]quarter\s+|"
    r"FY\s*\d*\s+|Q[1-4]\s+)?)",
    re.IGNORECASE,
)

# ── Periodicity patterns ─────────────────────────────────────────────────
PERIOD_PATTERNS = {
    "Q1": re.compile(r"(?:first[\s-]quarter|Q1|1st\s+quarter)", re.IGNORECASE),
    "Q2": re.compile(r"(?:second[\s-]quarter|Q2|2nd\s+quarter)", re.IGNORECASE),
    "Q3": re.compile(r"(?:third[\s-]quarter|Q3|3rd\s+quarter)", re.IGNORECASE),
    "Q4": re.compile(r"(?:fourth[\s-]quarter|Q4|4th\s+quarter)", re.IGNORECASE),
    "FY": re.compile(
        r"(?:full[\s-]year|fiscal[\s-]year|annual|FY\s*\d{2,4}|"
        r"year\s+(?:ending|ended)|for\s+the\s+year|twelve[\s-]month)",
        re.IGNORECASE,
    ),
    "H1": re.compile(r"(?:first[\s-]half|1st\s+half|H1)", re.IGNORECASE),
    "H2": re.compile(r"(?:second[\s-]half|2nd\s+half|H2)", re.IGNORECASE),
}

# Fiscal year extraction
_FISCAL_YEAR_RE = re.compile(
    r"(?:fiscal\s+(?:year\s+)?|FY\s*|calendar\s+(?:year\s+)?)?"
    r"(20\d{2})",
    re.IGNORECASE,
)

# ── Guidance action (direction) ───────────────────────────────────────────
ACTION_PATTERNS = {
    "raises": re.compile(r"\b(?:rais(?:e[sd]?|ing)|increas(?:e[sd]?|ing)|upward)\b", re.IGNORECASE),
    "lowers": re.compile(r"\b(?:lower(?:s|ed|ing)?|reduc(?:e[sd]?|ing)|decreas(?:e[sd]?|ing)|downward|cut)\b", re.IGNORECASE),
    "narrows": re.compile(r"\b(?:narrow(?:s|ed|ing)?|tighten(?:s|ed|ing)?)\b", re.IGNORECASE),
    "reaffirms": re.compile(r"\b(?:reaffirm(?:s|ed|ing)?|reiterat(?:e[sd]?|ing)|confirm(?:s|ed|ing)?|maintain(?:s|ed|ing)?)\b", re.IGNORECASE),
    "initiates": re.compile(r"\b(?:initiat(?:e[sd]?|ing)|introduc(?:e[sd]?|ing)|provid(?:e[sd]?|ing)\s+(?:initial|new|first))\b", re.IGNORECASE),
    "withdraws": re.compile(r"\b(?:withdraw(?:s|n|ing)?|suspend(?:s|ed|ing)?|eliminat(?:e[sd]?|ing)|discontinu(?:e[sd]?|ing))\b", re.IGNORECASE),
}


def _parse_dollar_value(value_str: str, scale_str: str | None) -> float | None:
    """Convert a matched dollar string and scale to a float."""
    try:
        value = float(value_str.replace(",", ""))
    except ValueError:
        return None

    if scale_str:
        scale_str = scale_str.lower().strip()
        if scale_str in ("billion", "bil", "bn", "b"):
            value *= 1_000_000_000
        elif scale_str in ("million", "mil", "mm", "m"):
            value *= 1_000_000
        elif scale_str in ("thousand", "k"):
            value *= 1_000

    return value


def _extract_values_near(text: str, position: int, window: int = 200) -> dict:
    """
    Extract dollar values near a given position in the text.

    Returns dict with val1, val2 (if range), scale, is_range.
    """
    snippet = text[max(0, position - 50):position + window]

    # Try range first
    range_match = _RANGE.search(snippet)
    if range_match:
        val1 = _parse_dollar_value(range_match.group(1), range_match.group(2) or range_match.group(4))
        val2 = _parse_dollar_value(range_match.group(3), range_match.group(4) or range_match.group(2))
        if val1 is not None and val2 is not None:
            return {"val1": min(val1, val2), "val2": max(val1, val2), "is_range": True}

    # Try per-share range
    ps_match = _PER_SHARE_RANGE.search(snippet)
    if ps_match:
        val1 = float(ps_match.group(1))
        val2 = float(ps_match.group(2))
        return {"val1": min(val1, val2), "val2": max(val1, val2), "is_range": True}

    # Try single value
    dollar_match = _DOLLAR_RE.search(snippet)
    if dollar_match:
        val = _parse_dollar_value(dollar_match.group(1), dollar_match.group(2))
        if val is not None:
            return {"val1": val, "val2": val, "is_range": False}

    return {"val1": None, "val2": None, "is_range": None}


def _detect_period(text: str, position: int, window: int = 300) -> dict:
    """Detect the fiscal period being guided near a text position."""
    snippet = text[max(0, position - window):position + window]

    # Detect periodicity
    pdicity = "Unknown"
    for period_name, pattern in PERIOD_PATTERNS.items():
        if pattern.search(snippet):
            pdicity = period_name
            break

    # Detect fiscal year
    fiscal_year = None
    year_match = _FISCAL_YEAR_RE.search(snippet)
    if year_match:
        fiscal_year = int(year_match.group(1))

    return {"pdicity": pdicity, "fiscal_year": fiscal_year}


def _detect_action(text: str, position: int, window: int = 150) -> str:
    """Detect whether guidance is being raised, lowered, reaffirmed, etc."""
    snippet = text[max(0, position - window):position + window]

    for action, pattern in ACTION_PATTERNS.items():
        if pattern.search(snippet):
            return action

    return "issues"  # default: just issuing guidance


def extract_guidance_from_text(text: str, filing_meta: dict) -> list[dict]:
    """
    Extract all guidance statements from a single press release.

    Strategy:
    1. Find sentences containing guidance verbs
    2. For each guidance sentence, identify the metric being guided
    3. Extract the dollar/numeric values
    4. Determine the period and action

    Returns list of dicts, one per guidance statement found.
    """
    if not text or len(text) < 100:
        return []

    records = []
    seen = set()  # Deduplicate: (metric, val1, val2)

    # Split into sentences (roughly)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sent_idx, sentence in enumerate(sentences):
        # Does this sentence contain a guidance verb?
        verb_match = GUIDANCE_VERBS.search(sentence)
        if not verb_match:
            continue

        # Find the position in the original text
        sent_start = text.find(sentence)
        if sent_start == -1:
            sent_start = 0

        # Which metric is being guided?
        for metric_name, metric_pattern in METRIC_PATTERNS.items():
            metric_match = metric_pattern.search(sentence)
            if not metric_match:
                continue

            # Extract values
            values = _extract_values_near(text, sent_start + metric_match.start())
            if values["val1"] is None:
                continue

            # Dedup
            dedup_key = (metric_name, values["val1"], values["val2"])
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Period
            period = _detect_period(text, sent_start)

            # Action
            action = _detect_action(text, sent_start)

            # Build record
            record = {
                **filing_meta,
                "metric": metric_name,
                "val1": values["val1"],
                "val2": values["val2"],
                "is_range": values["is_range"],
                "midpoint": (values["val1"] + values["val2"]) / 2 if values["val2"] else values["val1"],
                "range_width": (values["val2"] - values["val1"]) if values["is_range"] else 0.0,
                "pdicity": period["pdicity"],
                "fiscal_year": period["fiscal_year"],
                "action": action,
                "sentence": sentence.strip()[:500],  # Keep context, truncated
            }
            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Parse management guidance from 8-K press releases"
    )
    parser.add_argument(
        "--input", type=str, default="eight_k_exhibits.parquet",
        help="Input parquet file name (in data/raw/guidance/)"
    )
    args = parser.parse_args()

    input_path = RAW_DIR / args.input
    log.info(f"Loading exhibits from {input_path}")
    exhibits = pd.read_parquet(input_path)
    log.info(f"  {len(exhibits):,} press releases to parse")

    all_guidance = []
    n_with_guidance = 0

    for idx, row in exhibits.iterrows():
        filing_meta = {
            "cik": row["cik"],
            "company_name": row["company_name"],
            "accession_number": row["accession_number"],
            "date_filed": row["date_filed"],
            "form_type": row["form_type"],
            "exhibit_url": row["exhibit_url"],
        }

        records = extract_guidance_from_text(row["text"], filing_meta)
        if records:
            all_guidance.extend(records)
            n_with_guidance += 1

        if (idx + 1) % 10000 == 0:
            log.info(f"  Parsed {idx + 1:,}/{len(exhibits):,} "
                     f"({n_with_guidance:,} with guidance, "
                     f"{len(all_guidance):,} total records)")

    log.info(f"Parsing complete:")
    log.info(f"  {len(exhibits):,} press releases parsed")
    log.info(f"  {n_with_guidance:,} contained extractable guidance")
    log.info(f"  {len(all_guidance):,} total guidance records")

    if not all_guidance:
        log.warning("No guidance records extracted!")
        return

    df = pd.DataFrame(all_guidance)

    # Summary stats
    log.info(f"  Metrics found: {df['metric'].value_counts().to_dict()}")
    log.info(f"  Actions found: {df['action'].value_counts().to_dict()}")
    log.info(f"  Periodicities: {df['pdicity'].value_counts().to_dict()}")

    # Save
    DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    out_path = DATA_INTERMEDIATE / "guidance_parsed.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved parsed guidance -> {out_path}")


if __name__ == "__main__":
    main()
