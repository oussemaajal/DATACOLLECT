# CLAUDE.md -- DATACOLLECT

## Purpose

Shared data collection and retrieval toolkit for all RESEARCH projects. Houses reusable scripts for pulling data from WRDS, SEC/EDGAR, FRED, web sources, and APIs. Other projects (SOP, LENDABLES, CROSSBORDER, etc.) reference these scripts rather than duplicating collection logic.

## Collaborators

Solo (Oussema).

## Directory Structure

```
DATACOLLECT/
  code/
    scrapers/      # Web scraping scripts (EDGAR, SEC, websites)
    wrds/          # WRDS data retrieval (CRSP, Compustat, ISS, 13F, etc.)
    sec/           # SEC-specific tools (EDGAR API, XBRL, filing parsers)
    utils/         # Shared utilities (rate limiting, caching, validation)
    scratchpad/    # Temporary exploration scripts
    config.py      # Paths, API keys (env vars), shared constants
    utils.py       # Common helpers (download, cache, retry, logging)
  data/
    raw/           # Raw downloaded data (as retrieved from source)
    intermediate/  # Partially processed data
    clean/         # Final cleaned datasets ready for use by other projects
  output/          # Logs, reports, data quality checks
  docs/            # Documentation, data dictionaries, source notes
```

## Design Principles

1. **Idempotent scripts**: Every collection script should be safe to re-run. Use caching and checkpoint/resume patterns (see BEST_PRACTICES.md).
2. **Rate limiting**: Always respect API rate limits. Build in delays and retries.
3. **Raw data preservation**: Never modify raw downloads in place. Process raw -> intermediate -> clean.
4. **Logging**: Every download should log what was fetched, when, and whether it succeeded.
5. **Cross-project**: Scripts here serve multiple projects. Keep them general. Project-specific transformations belong in the consuming project.

## Data Sources (Expected)

| Source | Module | Notes |
|--------|--------|-------|
| WRDS (CRSP, Compustat, ISS, 13F, etc.) | `code/wrds/` | Via `wrds` Python package or direct PostgreSQL |
| SEC EDGAR | `code/sec/` | EDGAR full-text search, filing downloads, XBRL |
| FRED | `code/scrapers/` | Federal Reserve Economic Data |
| Web scraping | `code/scrapers/` | Generic scrapers for various sources |

## Conventions

- Python for all collection scripts (requests, selenium, wrds, sec-api, etc.)
- R only if a data source has an R-only package
- Use environment variables for credentials (WRDS_USERNAME, SEC_USER_AGENT, etc.)
- Large files: use DuckDB for querying, never load into pandas
- Follow BEST_PRACTICES.md for checkpointing and caching
