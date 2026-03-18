# DATACOLLECT

Shared data collection and retrieval toolkit for RESEARCH projects. Reusable scripts for pulling data from WRDS, SEC/EDGAR, FRED, web sources, and APIs.

## Status

**Setup phase** -- directory structure created, no collection scripts yet.

## Structure

```
code/
  scrapers/      Web scraping (EDGAR, websites, APIs)
  wrds/          WRDS data retrieval (CRSP, Compustat, ISS, 13F)
  sec/           SEC-specific tools (EDGAR API, XBRL, filing parsers)
  utils/         Shared utilities (rate limiting, caching, validation)
  config.py      Paths and shared constants
  utils.py       Common helpers
data/
  raw/           Raw downloads (as retrieved)
  intermediate/  Partially processed
  clean/         Final datasets for other projects
output/          Logs, data quality reports
docs/            Documentation, data dictionaries
```

## Usage

Other projects reference these scripts to avoid duplicating collection logic. Project-specific transformations stay in the consuming project.
