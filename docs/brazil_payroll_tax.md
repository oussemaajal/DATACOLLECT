# Brazil Payroll Tax Panel Dataset

## Overview

Panel dataset of **effective employer payroll tax rates** for all Brazilian economic sectors (CNAE divisions), covering the **2018-2026 study period**. Built from statutory rates encoded directly from legislation — no external API calls required to generate the panel (though a CNAE scraper is included for enrichment).

Purpose: main independent variable for studying the effect of payroll taxes on AI automation intensity, exploiting the ChatGPT shock (Nov 2022) and cross-sectional tax variation.

## What Varies and Why

Brazilian employer payroll taxes have four components that create cross-sectional variation:

| Component | Rate | Source of Variation |
|-----------|------|---------------------|
| Employer INSS | 20% of payroll (standard) | **Desoneração**: 17 sectors replace this with 2-4.5% on gross revenue |
| SAT/RAT | 1-3% of payroll | Varies by CNAE code (industry risk); firm-level FAP multiplier (0.5-2.0x) |
| Terceiros/Sistema S | 5.8% (most) or 7.7% (rural) | Varies by broad sector type (industry/commerce/transport/rural) |
| FGTS | 8% of payroll | Universal — no variation |

**Simples Nacional** (firms < R$ 4.8M revenue) is a separate regime where employer INSS is embedded in a unified rate, creating a sharp discontinuity at the revenue threshold.

### Key Identifying Variation for Research

1. **Desoneração eligibility (DiD)**: 17/18 CNAE divisions have CPRB (much lower payroll burden) vs. ~70 divisions on standard 20% INSS. Stable composition Sep 2018 – Dec 2024.
2. **Simples Nacional threshold (RDD)**: ~15-20 pp jump in effective payroll tax at R$ 4.8M annual revenue.
3. **SAT/RAT by CNAE (cross-sectional)**: 1%, 2%, or 3% base rate × FAP (0.5-2.0).
4. **Phase-out 2025-2028 (time variation)**: gradual reintroduction of payroll INSS for desoneração sectors.

## Data Sources (Legislative)

All rates are encoded from statutory sources — no scraping or API needed for the core panel:

| Source | What It Provides | Reference |
|--------|-----------------|-----------|
| Lei 12.546/2011 (as amended) | Desoneração eligible CNAE codes and CPRB rates | Arts. 7-8 |
| Lei 13.161/2015 | CPRB rate increases (1%→2.5%, 2%→4.5%) and optionality | |
| Lei 13.670/2018 | Reduction from ~56 to 17 eligible sectors | |
| Lei 14.784/2023 | Extension through 2027 | |
| Lei 14.973/2024 | Gradual phase-out schedule 2025-2028 | |
| Decreto 3.048/1999, Anexo V | CNAE → RAT base rate mapping (1/2/3%) | Updated by Decreto 6.957/2009, 10.410/2020 |
| IN RFB 2.110/2022 | FPAS code → Terceiros/Sistema S rate mapping | |
| LC 123/2006 (amended by LC 155/2016) | Simples Nacional tax tables (Annexes I-V) | |

## Output Files

All outputs are in `data/clean/brazil_payroll_tax/`:

### Main Panel: `brazil_payroll_tax_panel.csv`

**Dimensions**: 783 rows (87 CNAE divisions × 9 years, 2018-2026)

| Column | Description |
|--------|-------------|
| `cnae_division` | 2-digit CNAE division code |
| `division_label` | Sector name |
| `year` | Calendar year |
| `inss_employer_pct` | Employer INSS rate on payroll (0% if desoneração, 20% otherwise) |
| `desoneracao_eligible` | Whether sector can opt for CPRB |
| `cprb_rate_on_revenue_pct` | CPRB rate on gross revenue (null if not eligible) |
| `rat_base_pct` | SAT/RAT base rate (1, 2, or 3%) |
| `rat_min_pct` / `rat_max_pct` | Range after FAP adjustment (base × 0.5 to base × 2.0) |
| `terceiros_pct` | Sistema S + Salário-Educação total |
| `fgts_pct` | FGTS rate (always 8%) |
| `total_payroll_standard_pct` | Total if on standard regime (INSS + RAT + Terceiros + FGTS) |
| `total_payroll_desoneracao_pct` | Payroll-side total if on CPRB (excludes revenue-based CPRB) |
| `desoneracao_inss_savings_pct` | Reduction in payroll-based INSS (20 pp for eligible sectors) |
| `effective_inss_equiv_payroll_*` | CPRB converted to payroll-equivalent at assumed payroll/revenue ratios (30%, 50%, 70%) |

### Component Tables

| File | Description |
|------|-------------|
| `desoneracao_sectors.csv` | All desoneração-eligible CNAE prefixes, CPRB rates, and phase dates |
| `rat_by_cnae_division.csv` | SAT/RAT base rates and FAP-adjusted range by CNAE division |
| `terceiros_by_cnae_division.csv` | Sistema S total rate by CNAE division and sector type |
| `simples_nacional_rates.csv` | Effective rates and embedded CPP across all 5 annexes and revenue brackets |

## Code Structure

All code is in `code/scrapers/brazil_payroll_tax/`:

| File | What It Does |
|------|-------------|
| `desoneracao.py` | Desoneração eligible sectors and CPRB rates across 4 legislative phases. Helper functions: `is_eligible(cnae, date)`, `build_desoneracao_panel()` |
| `sat_rat.py` | RAT base rates by CNAE division (1/2/3%). `get_rat_base(cnae)`, `build_rat_table()` |
| `sistema_s.py` | Terceiros/Sistema S rates by sector type. Maps CNAE divisions to FPAS-based rates. `get_terceiros_rate(cnae)`, `build_terceiros_table()` |
| `simples_nacional.py` | Simples Nacional tax tables (Annexes I-V) with effective rate calculator. `effective_rate(revenue, annex)`, `effective_cpp_rate(revenue, annex)` |
| `fetch_cnae.py` | Scrapes CNAE 2.0 classification hierarchy from IBGE API (`servicodados.ibge.gov.br`). Fetches divisions, classes, or subclasses. |
| `build_payroll_tax_panel.py` | **Main script.** Combines all components into the CNAE division × year panel. Run with `python -m code.scrapers.brazil_payroll_tax.build_payroll_tax_panel` |

## How to Regenerate

```bash
cd /home/user/DATACOLLECT
python -c "
import sys; sys.path.insert(0, '.')
from code.scrapers.brazil_payroll_tax.build_payroll_tax_panel import save_panel, save_component_tables
from code.scrapers.brazil_payroll_tax.simples_nacional import build_simples_table
import logging; logging.basicConfig(level=logging.INFO)

save_component_tables()
save_panel()
build_simples_table().to_csv('data/clean/brazil_payroll_tax/simples_nacional_rates.csv', index=False)
"
```

To fetch fresh CNAE classification data from IBGE (requires internet):
```bash
python -c "
import sys; sys.path.insert(0, '.')
from code.scrapers.brazil_payroll_tax.fetch_cnae import fetch_and_save_cnae
import logging; logging.basicConfig(level=logging.INFO)
fetch_and_save_cnae('classes')     # 5-digit CNAE classes
# fetch_and_save_cnae('subclasses')  # 7-digit subclasses (slower)
"
```

## Limitations and Caveats

1. **Division-level aggregation**: The panel uses 2-digit CNAE divisions. Some desoneração eligibility is defined at the 4-5 digit CNAE group or class level. The panel flags a division as eligible if *any* of its subgroups are eligible. For firm-level analysis, use the `is_eligible(cnae_7digit, date)` function in `desoneracao.py`.

2. **CPRB is on revenue, not payroll**: The desoneração replaces a payroll-based tax (20%) with a revenue-based tax (2-4.5%). The effective payroll-equivalent rate depends on each firm's payroll/revenue ratio. The panel includes illustrative conversions at 30%, 50%, and 70% ratios but firm-level data is needed for precise estimates.

3. **RAT rates are division-level modes**: Individual 7-digit CNAE subclasses within a division may have different RAT rates. The panel uses the predominant rate for each division. For precise rates, parse the full Anexo V of Decreto 3.048/1999.

4. **FAP is firm-specific**: The FAP multiplier (0.5-2.0) on RAT is not in this panel — it requires firm-level data from Previdência Social / DATAPREV, published annually.

5. **Simples Nacional is separate**: Simples firms face a completely different rate structure. The `simples_nacional_rates.csv` file is provided separately. To identify which firms are on Simples, use the Receita Federal CNPJ open data.

6. **Desoneração is optional (post-2015)**: Since Lei 13.161/2015, eligible firms *choose* between CPRB and 20% payroll each year. The panel shows what's *available*, not what firms actually chose. Actual take-up would require eSocial or DCTF filing data (restricted access).

## Relevant Literature

- Gerard & Gonzaga (2021, AER), "Informal Labor and the Efficiency Cost of Social Insurance" — uses desoneração as identification with RAIS microdata
- Dallava (2014), "Impacts of Payroll Tax Reduction in Brazil"
- Ulyssea (2018), "Firms, Informality and Development" — models with Brazilian payroll taxation
