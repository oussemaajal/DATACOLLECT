"""
Build the complete payroll tax rate panel by CNAE code and year.

Combines all components to compute the effective total employer payroll tax
burden for each CNAE division, for each year in the 2018-2026 study period.

Components:
  1. Employer INSS: 20% (standard) or CPRB on revenue (desoneração sectors)
  2. SAT/RAT: 1-3% base, adjustable by FAP (0.5-2.0)
  3. Terceiros / Sistema S: ~5.8% (most sectors) or ~7.7% (rural)
  4. FGTS: 8% (universal, not technically a tax but a mandatory charge)

Output: Panel dataset at CNAE division × year level with columns for each
component and the total effective payroll tax rate.

Notes for your research design:
  - The key variation for DiD is desoneração eligibility (17 sectors post-2018)
  - Desoneração replaces ONLY the 20% INSS; RAT, Terceiros, FGTS are unchanged
  - Since CPRB is on revenue (not payroll), the effective payroll-equivalent
    rate depends on the firm's payroll-to-revenue ratio
  - Simples Nacional firms are a separate regime (excluded from this panel)
"""

import logging
from pathlib import Path

import pandas as pd

from code.config import DATA_CLEAN
from code.scrapers.brazil_payroll_tax.desoneracao import (
    PHASE4_SECTORS,
    PHASE5_TRANSITION,
    build_desoneracao_panel,
)
from code.scrapers.brazil_payroll_tax.sat_rat import RAT_BY_CNAE_DIVISION, build_rat_table
from code.scrapers.brazil_payroll_tax.sistema_s import (
    CNAE_DIVISION_TO_SECTOR_TYPE,
    TERCEIROS_RATES,
    build_terceiros_table,
)

logger = logging.getLogger(__name__)

CLEAN_DIR = DATA_CLEAN / "brazil_payroll_tax"

# Study period
YEARS = list(range(2018, 2027))

# Universal rates (% of payroll)
INSS_EMPLOYER_STANDARD = 20.0  # standard employer INSS contribution
FGTS_RATE = 8.0               # mandatory for all formal employees


def _get_desoneracao_status(cnae_division: str, year: int) -> dict:
    """
    Determine desoneração eligibility and CPRB rate for a CNAE division in a
    given year.

    Returns dict with:
      - eligible: bool
      - cprb_rate_pct: float or None (rate on gross revenue)
      - inss_payroll_pct: float (the payroll-based INSS rate, 0 if fully on CPRB)
      - mandatory: bool
      - notes: str
    """
    # Check if this division has any eligible CNAE codes in Phase 4
    eligible_prefixes = {}
    for prefix, label, rate in PHASE4_SECTORS:
        if cnae_division.startswith(prefix[:2]) and prefix.startswith(cnae_division):
            eligible_prefixes[prefix] = (label, rate)
        elif prefix.startswith(cnae_division):
            eligible_prefixes[prefix] = (label, rate)

    # Also check 2-digit matches
    for prefix, label, rate in PHASE4_SECTORS:
        if prefix == cnae_division:
            eligible_prefixes[prefix] = (label, rate)

    if not eligible_prefixes:
        return {
            "eligible": False,
            "cprb_rate_pct": None,
            "inss_payroll_pct": INSS_EMPLOYER_STANDARD,
            "mandatory": False,
            "notes": "Standard regime (20% employer INSS on payroll)",
        }

    # Get the most representative CPRB rate for this division
    rates = [r for _, (_, r) in eligible_prefixes.items()]
    cprb_rate = max(set(rates), key=rates.count)  # most common rate

    # Handle phase-out (2025-2028)
    if year in PHASE5_TRANSITION:
        cprb_weight, payroll_rate = PHASE5_TRANSITION[year]
        return {
            "eligible": True,
            "cprb_rate_pct": cprb_rate * cprb_weight,
            "inss_payroll_pct": payroll_rate * 100,
            "mandatory": False,
            "notes": f"Transition: {cprb_weight:.0%} CPRB + {payroll_rate:.0%} payroll (Lei 14.973/2024)",
        }

    if year >= 2018:
        return {
            "eligible": True,
            "cprb_rate_pct": cprb_rate,
            "inss_payroll_pct": 0.0,
            "mandatory": False,
            "notes": f"Desoneração Phase 4: optional CPRB at {cprb_rate}% of gross revenue",
        }

    return {
        "eligible": False,
        "cprb_rate_pct": None,
        "inss_payroll_pct": INSS_EMPLOYER_STANDARD,
        "mandatory": False,
        "notes": "Pre-study period",
    }


def build_panel() -> pd.DataFrame:
    """
    Build the main payroll tax panel: CNAE division × year.

    Each row contains the full breakdown of employer payroll tax components.
    For desoneração sectors, the CPRB rate is on gross revenue (not payroll),
    so the 'effective payroll equivalent' depends on the firm's
    payroll-to-revenue ratio (provided as illustrative columns).
    """
    rows = []

    for div, (label, rat_base) in sorted(RAT_BY_CNAE_DIVISION.items()):
        # Terceiros rate
        sector_type = CNAE_DIVISION_TO_SECTOR_TYPE.get(div, "commerce")
        terceiros = TERCEIROS_RATES.get(sector_type, {}).get("total_pct", 5.8)

        for year in YEARS:
            deson = _get_desoneracao_status(div, year)

            row = {
                "cnae_division": div,
                "division_label": label,
                "year": year,
                # Components
                "inss_employer_pct": deson["inss_payroll_pct"],
                "desoneracao_eligible": deson["eligible"],
                "cprb_rate_on_revenue_pct": deson["cprb_rate_pct"],
                "rat_base_pct": rat_base,
                "rat_min_pct": rat_base * 0.5,
                "rat_max_pct": rat_base * 2.0,
                "terceiros_pct": terceiros,
                "fgts_pct": FGTS_RATE,
                # Totals for standard regime (non-desoneração or if firm opts out)
                "total_payroll_standard_pct": (
                    INSS_EMPLOYER_STANDARD + rat_base + terceiros + FGTS_RATE
                ),
                # Total for desoneração firms (payroll components only, excluding CPRB)
                "total_payroll_desoneracao_pct": (
                    deson["inss_payroll_pct"] + rat_base + terceiros + FGTS_RATE
                ) if deson["eligible"] else None,
                # The "savings" from desoneração (on the payroll side)
                "desoneracao_inss_savings_pct": (
                    INSS_EMPLOYER_STANDARD - deson["inss_payroll_pct"]
                ) if deson["eligible"] else 0.0,
                "notes": deson["notes"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Add illustrative effective rates at different payroll/revenue ratios
    # These help researchers understand the magnitude of the desoneração benefit
    for ratio_label, ratio in [("low_30pct", 0.30), ("med_50pct", 0.50), ("high_70pct", 0.70)]:
        col = f"effective_inss_equiv_payroll_{ratio_label}"
        df[col] = df.apply(
            lambda r: (
                r["cprb_rate_on_revenue_pct"] / ratio
                if r["desoneracao_eligible"] and r["cprb_rate_on_revenue_pct"] and ratio > 0
                else r["inss_employer_pct"]
            ),
            axis=1,
        )

    return df


def save_panel(df: pd.DataFrame = None) -> Path:
    """Build (if needed) and save the payroll tax panel."""
    if df is None:
        df = build_panel()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_DIR / "brazil_payroll_tax_panel.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved payroll tax panel ({len(df)} rows) to {out_path}")
    return out_path


def save_component_tables() -> dict:
    """Save individual component tables as separate CSVs."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Desoneração panel
    deson = build_desoneracao_panel()
    p = CLEAN_DIR / "desoneracao_sectors.csv"
    deson.to_csv(p, index=False)
    paths["desoneracao"] = p

    # RAT table
    rat = build_rat_table()
    p = CLEAN_DIR / "rat_by_cnae_division.csv"
    rat.to_csv(p, index=False)
    paths["rat"] = p

    # Terceiros table
    terc = build_terceiros_table()
    p = CLEAN_DIR / "terceiros_by_cnae_division.csv"
    terc.to_csv(p, index=False)
    paths["terceiros"] = p

    logger.info(f"Saved component tables to {CLEAN_DIR}")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Building Brazil payroll tax panel (2018-2026)...")

    # Save individual component tables
    paths = save_component_tables()
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")

    # Build and save the main panel
    panel_path = save_panel()
    logger.info(f"  Main panel: {panel_path}")

    # Print summary
    df = pd.read_csv(panel_path)
    logger.info(f"\nPanel dimensions: {df.shape}")
    logger.info(f"CNAE divisions: {df['cnae_division'].nunique()}")
    logger.info(f"Years: {sorted(df['year'].unique())}")
    logger.info(f"Desoneração eligible: {df[df['desoneracao_eligible']]['cnae_division'].nunique()} divisions")

    # Show the variation
    logger.info("\n--- Standard regime total payroll tax (% of payroll) ---")
    logger.info(f"  Min:  {df['total_payroll_standard_pct'].min():.1f}%")
    logger.info(f"  Max:  {df['total_payroll_standard_pct'].max():.1f}%")
    logger.info(f"  Mean: {df['total_payroll_standard_pct'].mean():.1f}%")

    deson_rows = df[df["desoneracao_eligible"]]
    if len(deson_rows) > 0:
        logger.info("\n--- Desoneração sectors: INSS savings (pp of payroll) ---")
        logger.info(f"  Savings: {deson_rows['desoneracao_inss_savings_pct'].mean():.1f} pp")
        logger.info(f"  CPRB rates: {sorted(deson_rows['cprb_rate_on_revenue_pct'].dropna().unique())}")
