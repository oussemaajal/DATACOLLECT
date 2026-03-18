# Management Guidance Panel Dataset

## Overview

Panel dataset of management earnings guidance (forecasts) issued by US public firms, extracted directly from **SEC EDGAR 8-K filings**. No WRDS or I/B/E/S subscription required.

The pipeline downloads 8-K press release exhibits, parses guidance statements using regex/NLP, and structures them into a firm-announcement level panel.

## Data Source

**SEC EDGAR** -- all data is freely available from the SEC's public filing system.

| Source | What | URL Pattern |
|--------|------|-------------|
| EDGAR Full Index | Quarterly filing indexes | `sec.gov/Archives/edgar/full-index/{YEAR}/QTR{Q}/master.idx` |
| 8-K Filings | Current reports with earnings press releases | `sec.gov/Archives/edgar/data/{CIK}/{ACCESSION}/` |
| Company Tickers | CIK-to-ticker mapping | `sec.gov/files/company_tickers.json` |

## Pipeline

### Step 1: Build 8-K Filing Index

```bash
cd code/sec
python fetch_8k_index.py --start 2004 --end 2025
```

Downloads quarterly `master.idx` files from EDGAR, filters for 8-K and 8-K/A filings, saves a combined index. Results cached locally for re-runs.

**Output**: `data/raw/guidance/eight_k_index_2004_2025.parquet`

### Step 2: Download Press Release Exhibits

```bash
python download_8k_exhibits.py --index eight_k_index_2004_2025.parquet --resume
```

For each 8-K, fetches the filing index page, identifies EX-99.1 press release exhibits, downloads them, and filters for earnings/guidance-related content. Supports checkpoint/resume for large runs.

**Output**: `data/raw/guidance/eight_k_exhibits.parquet`

### Step 3: Parse Guidance from Text

```bash
python parse_guidance.py --input eight_k_exhibits.parquet
```

Extracts structured guidance from press release text using regex patterns for:
- **Metrics**: EPS, Revenue, Net Income, EBITDA, Operating Income, FFO, Cash Flow, Capex, Gross Margin, Tax Rate
- **Values**: Point estimates and ranges (e.g., "$1.50 to $1.60", "$10.2 billion")
- **Periods**: Quarterly (Q1-Q4), annual (FY), semi-annual (H1/H2)
- **Actions**: Raises, lowers, narrows, reaffirms, initiates, withdraws

**Output**: `data/intermediate/guidance_parsed.parquet`

### Step 4: Build Clean Panel

```bash
python build_guidance_panel.py
```

Merges parsed guidance with SEC ticker mapping, adds derived variables, deduplicates, and produces the final panel.

**Output**:
- `data/clean/guidance_panel.parquet` -- Full panel
- `data/clean/guidance_panel_summary.csv` -- Year x metric summary
- `output/guidance_panel_sample.csv` -- 500-row sample for inspection

## Variable Dictionary

### Identifiers

| Variable | Description |
|----------|-------------|
| `cik` | SEC Central Index Key (10-digit, zero-padded) |
| `ticker` | Stock ticker (from SEC company_tickers.json) |
| `company_name` | Company name (from 8-K filing) |
| `company_name_sec` | Company name (from SEC ticker mapping) |

### Filing Info

| Variable | Description |
|----------|-------------|
| `accession_number` | SEC filing accession number |
| `date_filed` | Filing date |
| `file_year` | Filing calendar year |
| `file_quarter` | Filing calendar quarter |
| `file_yearq` | Filing year-quarter string (e.g., "2023Q1") |
| `form_type` | Filing form type (8-K or 8-K/A) |

### Guidance Details

| Variable | Description |
|----------|-------------|
| `metric` | Guided metric (EPS, Revenue, Net_Income, EBITDA, etc.) |
| `pdicity` | Periodicity (Q1-Q4, FY, H1, H2, Unknown) |
| `fiscal_year` | Fiscal year being guided |
| `val1` | Lower bound or point estimate |
| `val2` | Upper bound (same as val1 for point estimates) |
| `midpoint` | Midpoint of guidance range |
| `range_width` | Range width (val2 - val1; 0 for point estimates) |
| `is_range` | Whether guidance is a range (True/False) |
| `action` | Guidance action (raises, lowers, narrows, reaffirms, initiates, withdraws, issues) |
| `is_per_share` | Whether values are per-share (vs. aggregate) |

### Source

| Variable | Description |
|----------|-------------|
| `sentence` | Source sentence from press release (truncated to 500 chars) |
| `exhibit_url` | URL of the source exhibit on EDGAR |

## Coverage

- **Time period**: 2004--present (post 8-K reform; SEC mandated 8-K for earnings releases)
- **Universe**: All US public firms filing 8-Ks with press release exhibits
- **Guidance types**: EPS, Revenue, EBITDA, Net Income, Operating Income, FFO, Cash Flow, Capex, Gross Margin, Tax Rate
- **Frequency**: Firm-announcement level

## Limitations

1. **Regex-based extraction**: Not all guidance statements follow standard patterns. Some guidance in unusual formats will be missed. Coverage is best for large-cap firms with standardized press releases.
2. **False positives**: Some extracted records may be historical results mislabeled as guidance. The `sentence` column allows manual review.
3. **No analyst consensus**: Unlike I/B/E/S, this dataset does not include concurrent analyst estimates. Merge with analyst forecast data separately if needed.
4. **8-K only**: Guidance in 10-K/10-Q MD&A sections, conference calls, or investor presentations is not captured.

## Practical Tips

```python
import pandas as pd

panel = pd.read_parquet("data/clean/guidance_panel.parquet")

# EPS guidance only
eps = panel[panel["metric"] == "EPS"]

# Annual guidance only
annual = panel[panel["pdicity"] == "FY"]

# Range estimates only
ranges = panel[panel["is_range"] == True]

# Guidance revisions (raised/lowered)
revisions = panel[panel["action"].isin(["raises", "lowers"])]
```

## Estimated Runtime

| Step | Typical Time | Bottleneck |
|------|-------------|------------|
| Step 1 (Index) | ~10 min | ~80 quarterly index downloads |
| Step 2 (Exhibits) | ~24-72 hours | Millions of 8-K filings, 10 req/sec |
| Step 3 (Parse) | ~30-60 min | CPU-bound regex parsing |
| Step 4 (Panel) | ~5 min | One API call for ticker mapping |

Use `--max-filings` and `--resume` flags in Step 2 to manage large runs.

## References

- Anilowski, C., Feng, M., & Skinner, D. (2007). Does earnings guidance affect market returns? The nature and information content of aggregate earnings guidance. *Journal of Accounting and Economics*.
- Houston, J., Lev, B., & Tucker, J. (2010). To guide or not to guide? Causes and consequences of stopping quarterly earnings guidance. *Contemporary Accounting Research*.
- SEC 8-K Reform (2004): Expanded list of reportable events, including Item 2.02 (Results of Operations).
