# Management Guidance Panel Dataset

## Overview

Panel dataset of management earnings guidance (forecasts) issued by US public firms, sourced from the **I/B/E/S Guidance database on WRDS**. Covers all guidance types (EPS, revenue, EBITDA, etc.) with firm identifiers (PERMNO, GVKEY) for easy linking to CRSP and Compustat.

## Source

- **I/B/E/S Detail Guidance** (`ibes.det_guidance`): Individual management guidance records
- **I/B/E/S Identifiers** (`ibes.id_sum`): Ticker-to-CUSIP crosswalk
- **CRSP Stock Names** (`crsp.stocknames`): CUSIP-to-PERMNO mapping
- **Compustat Security** (`comp.security`): CUSIP-to-GVKEY mapping

## Coverage

- **Time period**: 2002--present (I/B/E/S guidance coverage begins ~2002)
- **Universe**: All US public firms issuing management guidance
- **Frequency**: Firm-announcement level (one row per guidance issuance)
- **Guidance types**: EPS, Revenue, EBITDA, FFO, DPS, capex, and others

## How to Build

### Step 1: Pull raw data from WRDS

```bash
cd code/wrds
python guidance.py --start 2002 --end 2025
```

This creates raw parquet files in `data/raw/guidance/`:
- `det_guidance_2002_2025.parquet` -- Raw guidance records
- `ibes_identifiers.parquet` -- I/B/E/S ticker-CUSIP crosswalk
- `crsp_cusip_permno.parquet` -- CRSP identifier mapping
- `compustat_cusip_gvkey.parquet` -- Compustat identifier mapping

### Step 2: Build clean panel

```bash
python build_guidance_panel.py --input-tag 2002_2025
```

This creates:
- `data/intermediate/guidance_panel_merged_2002_2025.parquet` -- Full merged panel
- `data/clean/guidance_panel.parquet` -- Final clean dataset
- `data/clean/guidance_panel_summary.csv` -- Year x measure summary stats

## Variable Dictionary

### Identifiers

| Variable | Description |
|----------|-------------|
| `permno` | CRSP permanent security identifier |
| `permco` | CRSP permanent company identifier |
| `gvkey` | Compustat Global Company Key |
| `ticker` | I/B/E/S ticker |
| `oftic` | Official exchange ticker |
| `cusip` | 8-digit CUSIP |
| `cname` | Company name |

### Guidance Details

| Variable | Description |
|----------|-------------|
| `measure` | Guidance metric code (EPS, SAL, EBI, FFO, etc.) |
| `measure_label` | Human-readable metric name |
| `pdicity` | Periodicity code (QTR, ANN, SAN) |
| `pdicity_label` | Human-readable periodicity |
| `anndats` | Announcement date (when guidance was issued) |
| `anntims` | Announcement time |
| `ann_year` | Announcement calendar year |
| `ann_quarter` | Announcement calendar quarter |
| `ann_yearq` | Announcement year-quarter string (e.g., "2023Q1") |
| `statpers` | Statistical period end date (period being forecast) |
| `stat_year` | Statistical period year |
| `stat_quarter` | Statistical period quarter |

### Guidance Values

| Variable | Description |
|----------|-------------|
| `val1` | First guidance value (point estimate or range lower bound) |
| `val2` | Second guidance value (range upper bound; same as val1 for point) |
| `guidance_mid` | Midpoint of guidance (val1 for point, average of val1/val2 for range) |
| `range_width` | Width of range (val2 - val1; 0 for point estimates) |
| `gession` | Guidance form code (1=point, 2=range, 3=open low, 4=open high, 5=confirming, 6=qualitative) |
| `gession_label` | Human-readable guidance form |
| `guession` | Guidance sub-type detail |
| `is_point` | Indicator: 1 if point estimate |
| `is_range` | Indicator: 1 if range estimate |

### Analyst Context

| Variable | Description |
|----------|-------------|
| `numest` | Number of analyst estimates at time of guidance |
| `meanest` | Consensus mean analyst estimate |
| `medest` | Consensus median analyst estimate |
| `dispersion` | Analyst forecast dispersion |
| `surprise` | Guidance midpoint minus consensus mean |
| `surprise_pct` | Surprise scaled by absolute consensus mean |

### Actuals

| Variable | Description |
|----------|-------------|
| `actual` | Realized actual value for the guided period |
| `anndats_act` | Date when actual was announced |
| `actual_vs_guidance` | Actual minus guidance midpoint (positive = beat guidance) |

### Exchange Info

| Variable | Description |
|----------|-------------|
| `exchcd` | CRSP exchange code (1=NYSE, 2=AMEX, 3=NASDAQ) |

## Notes

- **CUSIP matching**: Identifiers are matched using 8-digit CUSIP with date-range validation to ensure the CUSIP was active at the announcement date.
- **Deduplication**: When multiple CRSP/Compustat matches exist, the primary listing (lowest exchange code) is preferred.
- **Common filters for research**:
  - EPS guidance only: `df[df['measure'] == 'EPS']`
  - Annual guidance: `df[df['pdicity'] == 'ANN']`
  - Point and range only: `df[df['gession'].isin([1, 2])]`
  - Exclude confirming/qualitative: `df[~df['gession'].isin([5, 6])]`

## References

- Chuk, E., Matsumoto, D., & Miller, G. (2013). Assessing methods of identifying management forecasts: CIG vs. researcher collected. *Journal of Accounting and Economics*.
- Houston, J., Lev, B., & Tucker, J. (2010). To guide or not to guide? Causes and consequences of stopping quarterly earnings guidance. *Contemporary Accounting Research*.
