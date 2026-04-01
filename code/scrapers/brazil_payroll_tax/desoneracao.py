"""
Desoneração da Folha de Pagamento — CPRB eligible sectors and rates.

The desoneração replaced the 20% employer INSS contribution with a tax on
gross revenue (CPRB) for eligible sectors. This module encodes:
  - Which CNAE codes were eligible in each time period
  - The applicable CPRB rate for each sector/period
  - Key legislative changes

Sources:
  - Lei 12.546/2011 (as amended)
  - Lei 13.161/2015 (optionality + rate increases)
  - Lei 13.670/2018 (reduction to 17 sectors)
  - Lei 14.784/2023 (extension through 2027)
  - Lei 14.973/2024 (gradual phase-out 2025-2028)
"""

import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Phase definitions: each phase has a date range, list of (cnae_group, cprb_rate),
# and whether participation was mandatory or optional.
# ---------------------------------------------------------------------------

# CNAE groups eligible under each legislative phase.
# Format: (cnae_group_prefix, sector_label, cprb_rate_pct)
# cnae_group_prefix matches the start of a 7-digit CNAE code (e.g. "6201" matches 6201-5/00).

# Phase 1: Lei 12.546/2011 (Dec 2011 – Mar 2012 initial, expanded Apr 2012)
# Mandatory for eligible sectors.
PHASE1_START = "2011-12-01"
PHASE1_END = "2012-07-31"  # before first major expansion

PHASE1_SECTORS = [
    # IT and IT services (TI/TIC)
    ("6201", "IT development", 2.5),
    ("6202", "IT consulting", 2.5),
    ("6203", "IT facilities management", 2.5),
    ("6204", "IT support", 2.5),
    ("6209", "Other IT services", 2.5),
    ("6311", "Data processing/hosting", 2.5),
    ("6319", "Other IT portals", 2.5),
    # Call centers
    ("8220", "Call centers", 2.5),
    # Textiles / apparel / footwear (by NCM codes primarily, but these CNAE groups cover manufacturing)
    ("13", "Textiles manufacturing", 1.5),
    ("14", "Apparel manufacturing", 1.5),
    ("15", "Leather and footwear", 1.5),
    # Semiconductor design
    ("2610", "Semiconductor/electronic components", 2.5),
]

# Phase 2: Major expansion (Aug 2012 – Nov 2015)
# Leis 12.715/2012, 12.794/2013, 12.844/2013, 12.873/2013, 12.995/2014
# Mandatory. ~56 sectors at peak. Rates mostly 1% or 2%.
PHASE2_START = "2012-08-01"
PHASE2_END = "2015-11-30"

PHASE2_SECTORS = [
    # IT/TIC (rates lowered)
    ("6201", "IT development", 2.0),
    ("6202", "IT consulting", 2.0),
    ("6203", "IT facilities management", 2.0),
    ("6204", "IT support", 2.0),
    ("6209", "Other IT services", 2.0),
    ("6311", "Data processing/hosting", 2.0),
    ("6319", "Other IT portals", 2.0),
    # Call centers
    ("8220", "Call centers", 2.0),
    # Textiles / apparel / footwear
    ("13", "Textiles manufacturing", 1.0),
    ("14", "Apparel manufacturing", 1.0),
    ("15", "Leather and footwear", 1.0),
    # Hotels
    ("5510", "Hotels", 2.0),
    # Civil construction
    ("412", "Building construction", 2.0),
    ("421", "Infrastructure - roads/rail", 2.0),
    ("422", "Infrastructure - utilities", 2.0),
    ("429", "Infrastructure - other", 2.0),
    ("431", "Demolition/site preparation", 2.0),
    ("432", "Electrical/plumbing installation", 2.0),
    ("433", "Building finishing", 2.0),
    ("439", "Other specialized construction", 2.0),
    # Road transport
    ("4921", "Urban bus transport", 2.0),
    ("4922", "Intercity bus transport", 2.0),
    ("4930", "Road cargo transport", 1.0),
    # Rail/metro
    ("4911", "Rail freight", 2.0),
    ("4912", "Rail passenger (long distance)", 2.0),
    ("4912", "Metro/urban rail", 2.0),
    # Manufacturing
    ("10", "Food products", 1.0),
    ("25", "Metal products", 1.0),
    ("26", "Electronics/computers", 1.0),
    ("27", "Electrical equipment", 1.0),
    ("28", "Machinery/equipment", 1.0),
    ("29", "Motor vehicles", 1.0),
    ("30", "Other transport equipment", 1.0),
    ("31", "Furniture", 1.0),
    ("22", "Rubber and plastics", 1.0),
    ("24", "Basic metals", 1.0),
    # Retail (selected)
    ("47", "Retail trade", 1.0),
    # Media / communications
    ("58", "Publishing", 1.0),
    ("59", "Film/TV/music", 1.0),
    ("60", "Broadcasting", 1.0),
    # Architecture / engineering
    ("7111", "Architecture services", 2.0),
    ("7112", "Engineering services", 2.0),
    ("7120", "Technical testing", 2.0),
    # Maintenance of vehicles
    ("33", "Repair/installation of machinery", 1.0),
    ("45", "Vehicle trade/repair", 1.0),
]

# Phase 3: Lei 13.161/2015 — Optional regime, increased rates
# Dec 2015 – Aug 2018
PHASE3_START = "2015-12-01"
PHASE3_END = "2018-08-31"

# Same sectors as Phase 2 but with roughly doubled rates and now OPTIONAL
PHASE3_SECTORS = [
    # IT/TIC
    ("6201", "IT development", 4.5),
    ("6202", "IT consulting", 4.5),
    ("6203", "IT facilities management", 4.5),
    ("6204", "IT support", 4.5),
    ("6209", "Other IT services", 4.5),
    ("6311", "Data processing/hosting", 4.5),
    ("6319", "Other IT portals", 4.5),
    # Call centers
    ("8220", "Call centers", 3.0),
    # Textiles / apparel / footwear
    ("13", "Textiles manufacturing", 2.5),
    ("14", "Apparel manufacturing", 2.5),
    ("15", "Leather and footwear", 2.5),
    # Hotels
    ("5510", "Hotels", 4.5),
    # Civil construction
    ("412", "Building construction", 4.5),
    ("421", "Infrastructure - roads/rail", 4.5),
    ("422", "Infrastructure - utilities", 4.5),
    ("429", "Infrastructure - other", 4.5),
    ("431", "Demolition/site preparation", 4.5),
    ("432", "Electrical/plumbing installation", 4.5),
    ("433", "Building finishing", 4.5),
    ("439", "Other specialized construction", 4.5),
    # Road transport
    ("4921", "Urban bus transport", 2.0),
    ("4922", "Intercity bus transport", 2.0),
    ("4930", "Road cargo transport", 2.5),
    # Rail/metro
    ("4911", "Rail freight", 2.0),
    ("4912", "Rail/metro passenger", 2.0),
    # Manufacturing (rates increased from 1% to 2.5%)
    ("10", "Food products", 2.5),
    ("25", "Metal products", 2.5),
    ("26", "Electronics/computers", 2.5),
    ("27", "Electrical equipment", 2.5),
    ("28", "Machinery/equipment", 2.5),
    ("29", "Motor vehicles", 2.5),
    ("30", "Other transport equipment", 2.5),
    ("31", "Furniture", 2.5),
    ("22", "Rubber and plastics", 2.5),
    ("24", "Basic metals", 2.5),
    # Retail
    ("47", "Retail trade", 2.5),
    # Media
    ("58", "Publishing", 2.5),
    ("59", "Film/TV/music", 2.5),
    ("60", "Broadcasting", 2.5),
    # Architecture / engineering
    ("7111", "Architecture services", 4.5),
    ("7112", "Engineering services", 4.5),
    ("7120", "Technical testing", 4.5),
    # Maintenance
    ("33", "Repair/installation of machinery", 2.5),
    ("45", "Vehicle trade/repair", 2.5),
]

# Phase 4: Lei 13.670/2018 — Reduced to 17 sectors (Sep 2018 onward)
# Optional. Extended by Lei 14.288/2021 and Lei 14.784/2023.
PHASE4_START = "2018-09-01"
PHASE4_END = "2024-12-31"  # before gradual phase-out begins

PHASE4_SECTORS = [
    # 1. Apparel (confecção e vestuário)
    ("14", "Apparel manufacturing", 2.5),
    # 2. Footwear (calçados)
    ("1531", "Footwear manufacturing", 2.5),
    ("1532", "Footwear parts", 2.5),
    ("1533", "Footwear manufacturing", 2.5),
    ("1539", "Other footwear", 2.5),
    # 3. Civil construction
    ("412", "Building construction", 4.5),
    ("421", "Infrastructure - roads/rail", 4.5),
    ("422", "Infrastructure - utilities", 4.5),
    ("429", "Infrastructure - other", 4.5),
    ("431", "Demolition/site preparation", 4.5),
    ("432", "Electrical/plumbing installation", 4.5),
    ("433", "Building finishing", 4.5),
    ("439", "Other specialized construction", 4.5),
    # 4. Call centers
    ("8220", "Call centers", 3.0),
    # 5. Communications / media
    ("58", "Publishing", 2.5),
    ("59", "Film/TV/music", 2.5),
    ("60", "Broadcasting", 2.5),
    # 6. IT and IT services (TI/TIC)
    ("6201", "IT development", 4.5),
    ("6202", "IT consulting", 4.5),
    ("6203", "IT facilities management", 4.5),
    ("6204", "IT support", 4.5),
    ("6209", "Other IT services", 4.5),
    ("6311", "Data processing/hosting", 4.5),
    ("6319", "Other IT portals", 4.5),
    # 7. Leather goods (couro)
    ("1510", "Leather tanning/preparation", 2.5),
    ("1521", "Luggage/bags", 2.5),
    ("1529", "Other leather goods", 2.5),
    # 8. Road cargo transport
    ("4930", "Road cargo transport", 2.5),
    # 9. Urban bus transport
    ("4921", "Urban bus transport", 2.0),
    ("4922", "Intercity bus transport", 2.0),
    # 10. Rail/metro transport
    ("4911", "Rail freight", 2.0),
    ("4912", "Rail/metro passenger", 2.0),
    # 11. Machinery and equipment
    ("28", "Machinery/equipment manufacturing", 2.5),
    # 12. Animal protein / meat processing
    ("1011", "Meat slaughtering (cattle)", 2.5),
    ("1012", "Meat slaughtering (poultry)", 2.5),
    ("1013", "Meat products manufacturing", 2.5),
    # 13. Fish processing
    ("1020", "Fish processing", 2.5),
    # 14. Textiles
    ("13", "Textiles manufacturing", 2.5),
    # 15. Automotive / auto parts
    ("29", "Motor vehicles manufacturing", 2.5),
    ("2941", "Auto parts manufacturing", 2.5),
    ("2942", "Auto body manufacturing", 2.5),
    ("2943", "Auto cabin manufacturing", 2.5),
    ("2944", "Auto parts reconditioning", 2.5),
    ("2945", "Auto parts manufacturing", 2.5),
    ("2949", "Other auto parts", 2.5),
    # 16. Circuit design (projeto de circuitos integrados)
    ("2610", "Semiconductor/circuit design", 4.5),
    # 17. Select retail — encoded via NCM codes primarily, CNAE secondary
    # Retail of certain product categories (computers, phones, etc.)
    ("4751", "Retail - computers/electronics", 2.5),
    ("4752", "Retail - telecom equipment", 2.5),
]

# Phase 5: Gradual phase-out (Lei 14.973/2024)
# Same 17 sectors but with blended INSS+CPRB
PHASE5_TRANSITION = {
    # year: (cprb_weight, payroll_inss_rate)
    # Effective rate = cprb_weight * CPRB_rate_on_revenue + payroll_inss_rate * payroll
    2025: (0.80, 0.05),   # 80% CPRB + 5% on payroll
    2026: (0.60, 0.10),   # 60% CPRB + 10% on payroll
    2027: (0.40, 0.15),   # 40% CPRB + 15% on payroll
    2028: (0.00, 0.20),   # Full return to 20% payroll
}


# ---------------------------------------------------------------------------
# Legislative timeline for documentation / panel construction
# ---------------------------------------------------------------------------

LEGISLATIVE_TIMELINE = [
    {
        "date": "2011-12-14",
        "law": "Lei 12.546/2011",
        "event": "Desoneração introduced for 4 sectors (IT, textiles, footwear, call centers)",
        "mandatory": True,
    },
    {
        "date": "2012-09-17",
        "law": "Lei 12.715/2012",
        "event": "Expanded to hotels, construction, furniture",
        "mandatory": True,
    },
    {
        "date": "2013-05-29",
        "law": "Lei 12.794/2013",
        "event": "Added retail, transport, media sectors",
        "mandatory": True,
    },
    {
        "date": "2013-07-19",
        "law": "Lei 12.844/2013",
        "event": "Added road transport, construction materials, more manufacturing",
        "mandatory": True,
    },
    {
        "date": "2014-06-18",
        "law": "Lei 12.995/2014",
        "event": "Peak coverage: ~56 sectors eligible",
        "mandatory": True,
    },
    {
        "date": "2015-12-01",
        "law": "Lei 13.161/2015",
        "event": "CPRB made OPTIONAL; rates roughly doubled (1%→2.5%, 2%→4.5%)",
        "mandatory": False,
    },
    {
        "date": "2018-09-01",
        "law": "Lei 13.670/2018",
        "event": "Reduced from ~56 to 17 eligible sectors",
        "mandatory": False,
    },
    {
        "date": "2021-12-30",
        "law": "Lei 14.288/2021",
        "event": "Extended desoneração through Dec 2023",
        "mandatory": False,
    },
    {
        "date": "2023-12-27",
        "law": "Lei 14.784/2023",
        "event": "Extended through Dec 2027 (congressional override of veto)",
        "mandatory": False,
    },
    {
        "date": "2024-09-16",
        "law": "Lei 14.973/2024",
        "event": "Gradual phase-out: blended CPRB+payroll 2025-2027, full payroll 2028",
        "mandatory": False,
    },
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_phase_for_date(date_str: str) -> Optional[dict]:
    """Return the desoneração phase info for a given date (YYYY-MM-DD)."""
    phases = [
        ("phase1", PHASE1_START, PHASE1_END, PHASE1_SECTORS, True),
        ("phase2", PHASE2_START, PHASE2_END, PHASE2_SECTORS, True),
        ("phase3", PHASE3_START, PHASE3_END, PHASE3_SECTORS, False),
        ("phase4", PHASE4_START, PHASE4_END, PHASE4_SECTORS, False),
    ]
    for name, start, end, sectors, mandatory in phases:
        if start <= date_str <= end:
            return {
                "phase": name,
                "start": start,
                "end": end,
                "sectors": sectors,
                "mandatory": mandatory,
            }
    return None


def is_eligible(cnae_code: str, date_str: str) -> Optional[float]:
    """
    Check if a CNAE code is eligible for desoneração at a given date.
    Returns the CPRB rate if eligible, None otherwise.

    cnae_code: 7-digit CNAE (e.g. "6201500") or shorter prefix (e.g. "6201")
    date_str: "YYYY-MM-DD"
    """
    phase = get_phase_for_date(date_str)
    if phase is None:
        return None

    cnae_clean = cnae_code.replace("-", "").replace("/", "").replace(".", "")

    for prefix, _label, rate in phase["sectors"]:
        if cnae_clean.startswith(prefix):
            return rate
    return None


def build_desoneracao_panel() -> pd.DataFrame:
    """
    Build a panel of desoneração eligibility and CPRB rates by CNAE prefix
    and time period.

    Returns DataFrame with columns:
      cnae_prefix, sector_label, cprb_rate, phase, start_date, end_date, mandatory
    """
    rows = []
    phases = [
        ("phase1", PHASE1_START, PHASE1_END, PHASE1_SECTORS, True),
        ("phase2", PHASE2_START, PHASE2_END, PHASE2_SECTORS, True),
        ("phase3", PHASE3_START, PHASE3_END, PHASE3_SECTORS, False),
        ("phase4", PHASE4_START, PHASE4_END, PHASE4_SECTORS, False),
    ]
    for phase_name, start, end, sectors, mandatory in phases:
        for prefix, label, rate in sectors:
            rows.append({
                "cnae_prefix": prefix,
                "sector_label": label,
                "cprb_rate_pct": rate,
                "phase": phase_name,
                "start_date": start,
                "end_date": end,
                "mandatory": mandatory,
            })

    return pd.DataFrame(rows)
