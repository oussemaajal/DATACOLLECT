"""
Simples Nacional tax tables — embedded payroll tax (CPP) rates.

Under Simples Nacional (LC 123/2006, amended by LC 155/2016), micro and small
enterprises pay a unified tax rate that includes the employer INSS contribution
(CPP) for most annexes. This creates a sharp difference in effective payroll
tax rates vs. the standard regime.

Key thresholds:
  - MEI: up to R$ 81,000/year
  - Microempresa (ME): up to R$ 360,000/year
  - Empresa de Pequeno Porte (EPP): up to R$ 4,800,000/year

Annex IV (construction, security, cleaning services) does NOT include CPP —
these firms pay 20% employer INSS + RAT + Terceiros separately.

Rates are stable across the 2018-2026 study period (tables set by LC 155/2016,
effective Jan 2018).
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Simples Nacional Tax Tables (LC 155/2016, effective 2018-present)
# Each annex has revenue brackets with a nominal rate and a deduction.
# Effective rate = (RBT12 * nominal_rate - deduction) / RBT12
# where RBT12 = gross revenue last 12 months
# ---------------------------------------------------------------------------

# Annex I — Commerce
ANNEX_I = {
    "name": "Annex I - Commerce",
    "includes_cpp": True,
    "cpp_share_pct": 41.5,  # approximate share of CPP in total rate
    "brackets": [
        # (revenue_up_to_BRL, nominal_rate_pct, deduction_BRL)
        (180_000, 4.0, 0),
        (360_000, 7.3, 5_940),
        (720_000, 9.5, 13_860),
        (1_800_000, 10.7, 22_500),
        (3_600_000, 14.3, 87_300),
        (4_800_000, 19.0, 378_000),
    ],
}

# Annex II — Industry
ANNEX_II = {
    "name": "Annex II - Industry",
    "includes_cpp": True,
    "cpp_share_pct": 41.5,
    "brackets": [
        (180_000, 4.5, 0),
        (360_000, 7.8, 5_940),
        (720_000, 10.0, 13_860),
        (1_800_000, 11.2, 22_500),
        (3_600_000, 14.7, 85_500),
        (4_800_000, 30.0, 720_000),
    ],
}

# Annex III — Services (lower labor intensity)
ANNEX_III = {
    "name": "Annex III - Services (low labor intensity)",
    "includes_cpp": True,
    "cpp_share_pct": 43.4,
    "brackets": [
        (180_000, 6.0, 0),
        (360_000, 11.2, 9_360),
        (720_000, 13.5, 17_640),
        (1_800_000, 16.0, 35_640),
        (3_600_000, 21.0, 125_640),
        (4_800_000, 33.0, 648_000),
    ],
}

# Annex IV — Services (construction, security, cleaning)
# DOES NOT include CPP — firms pay 20% INSS + RAT + Terceiros separately!
ANNEX_IV = {
    "name": "Annex IV - Services (construction/security/cleaning)",
    "includes_cpp": False,  # <-- KEY DIFFERENCE
    "cpp_share_pct": 0.0,
    "brackets": [
        (180_000, 4.5, 0),
        (360_000, 9.0, 8_100),
        (720_000, 10.2, 12_420),
        (1_800_000, 14.0, 39_780),
        (3_600_000, 22.0, 183_780),
        (4_800_000, 33.0, 828_000),
    ],
}

# Annex V — Services (higher labor intensity)
ANNEX_V = {
    "name": "Annex V - Services (high labor intensity)",
    "includes_cpp": True,
    "cpp_share_pct": 28.85,
    "brackets": [
        (180_000, 15.5, 0),
        (360_000, 18.0, 4_500),
        (720_000, 19.5, 9_900),
        (1_800_000, 20.5, 17_100),
        (3_600_000, 23.0, 62_100),
        (4_800_000, 30.5, 540_000),
    ],
}

ALL_ANNEXES = [ANNEX_I, ANNEX_II, ANNEX_III, ANNEX_IV, ANNEX_V]

# Revenue thresholds
SIMPLES_MAX_REVENUE = 4_800_000  # R$ 4.8M annual gross revenue
ME_MAX_REVENUE = 360_000         # Microempresa threshold
MEI_MAX_REVENUE = 81_000         # Microempreendedor Individual threshold


def effective_rate(rbt12: float, annex: dict) -> float:
    """
    Calculate the effective Simples Nacional tax rate for a given
    12-month gross revenue (RBT12) under the specified annex.

    Returns the effective rate as a percentage (e.g., 6.0 for 6%).
    """
    for ceiling, nominal, deduction in annex["brackets"]:
        if rbt12 <= ceiling:
            return (rbt12 * nominal / 100 - deduction) / rbt12 * 100
    return None  # above Simples threshold


def effective_cpp_rate(rbt12: float, annex: dict) -> float:
    """
    Calculate the effective employer payroll contribution (CPP) rate
    embedded in the Simples Nacional rate.

    For Annex IV, returns 0 (CPP is paid separately at 20%).
    """
    if not annex["includes_cpp"]:
        return 0.0
    total = effective_rate(rbt12, annex)
    if total is None:
        return None
    return total * annex["cpp_share_pct"] / 100


def build_simples_table() -> pd.DataFrame:
    """
    Build a DataFrame showing effective Simples Nacional rates and embedded
    CPP rates across revenue brackets and annexes.
    """
    rows = []
    # Sample revenue points
    revenue_points = [
        100_000, 180_000, 270_000, 360_000,
        540_000, 720_000, 1_000_000, 1_800_000,
        2_500_000, 3_600_000, 4_200_000, 4_800_000,
    ]
    for annex in ALL_ANNEXES:
        for rev in revenue_points:
            eff = effective_rate(rev, annex)
            cpp = effective_cpp_rate(rev, annex)
            if eff is not None:
                rows.append({
                    "annex": annex["name"],
                    "includes_cpp": annex["includes_cpp"],
                    "rbt12_brl": rev,
                    "effective_rate_pct": round(eff, 2),
                    "embedded_cpp_pct": round(cpp, 2),
                    "separate_inss_20pct": not annex["includes_cpp"],
                })
    return pd.DataFrame(rows)
