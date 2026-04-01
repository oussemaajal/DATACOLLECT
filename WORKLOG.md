# WORKLOG -- DATACOLLECT

## 2026-03-18 | Session 1: Project Setup

- Created DATACOLLECT directory with Gentzkow-Shapiro structure adapted for a toolkit repo
- Subdirectories: `code/{scrapers,wrds,sec,utils}`, `data/{raw,intermediate,clean}`, `output/`, `docs/`
- Created CLAUDE.md, README.md, config.py, utils.py, .gitignore
- Initialized git repo and pushed to GitHub

### Current state
Empty toolkit -- structure is in place, no collection scripts yet.

## 2026-04-01 | Session: Brazil Payroll Tax Panel

**Goal**: Collect data on Brazilian payroll tax rate variation across sectors,
firm sizes, and time periods. This is the main independent variable for a
project studying the effect of payroll taxes on AI automation intensity
(exploiting the ChatGPT shock and cross-sectional tax differentials).

### What was built

Created `code/scrapers/brazil_payroll_tax/` module with:
- `desoneracao.py` — Desoneração da Folha (CPRB) eligible sectors and rates across 4 legislative phases (2011-2028). 17 sectors post-2018 replace 20% employer INSS with 2-4.5% tax on gross revenue.
- `sat_rat.py` — SAT/RAT workplace accident insurance rates (1/2/3%) mapped to all 87 CNAE divisions.
- `sistema_s.py` — Sistema S/Terceiros contribution rates (~5.8-7.7%) by sector type.
- `simples_nacional.py` — Simples Nacional tax tables (Annexes I-V) with effective rate calculators.
- `fetch_cnae.py` — CNAE 2.0 classification scraper from IBGE API.
- `build_payroll_tax_panel.py` — Combines everything into a CNAE division × year panel (2018-2026).

### Outputs

- `data/clean/brazil_payroll_tax/brazil_payroll_tax_panel.csv` — Main panel (783 rows: 87 divisions × 9 years)
- `data/clean/brazil_payroll_tax/desoneracao_sectors.csv` — Eligible sectors by phase
- `data/clean/brazil_payroll_tax/rat_by_cnae_division.csv` — RAT rates
- `data/clean/brazil_payroll_tax/terceiros_by_cnae_division.csv` — Sistema S rates
- `data/clean/brazil_payroll_tax/simples_nacional_rates.csv` — Simples effective rates
- `docs/brazil_payroll_tax.md` — Full documentation

### Key design notes

- All rates encoded from statutory sources (laws, decrees). No scraping needed for the core panel.
- Study period 2018-2026: desoneração sector composition is stable (17 sectors, Lei 13.670/2018) with phase-out starting 2025.
- Desoneração replaces ONLY the 20% employer INSS; RAT, Terceiros, FGTS are unchanged.
- CPRB is on revenue (not payroll), so effective payroll-equivalent depends on firm's payroll/revenue ratio.
- Simples Nacional threshold at R$ 4.8M creates a sharp ~15-20 pp discontinuity — strongest RDD candidate.

### Not yet done

- Firm-level data (Receita Federal CNPJ registry, RAIS) for RDD around Simples threshold
- 7-digit CNAE subclass-level RAT rates (currently at 2-digit division level)
- FAP multiplier data (firm-specific, requires DATAPREV/Previdência access)
