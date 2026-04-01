"""
Sistema S / Terceiros contribution rates by sector type.

These mandatory contributions are collected alongside payroll taxes and
directed to sector-specific training and social organizations (SESI, SENAI,
SESC, SENAC, SEST, SENAT, SEBRAE, INCRA, SENAR, SESCOOP, etc.).

Rates are stable across the 2018-2026 study period.
Firms under Simples Nacional are EXEMPT from Sistema S contributions.
Under desoneração (CPRB), firms still pay Terceiros on payroll.

Source: Instrução Normativa RFB 2.110/2022; FPAS code tables.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Terceiros rates by sector type (% of payroll)
# Mapped via FPAS (Fundo de Previdência e Assistência Social) codes.
# ---------------------------------------------------------------------------

TERCEIROS_RATES = {
    "industry": {
        "fpas_codes": [507],
        "label": "Industrial firms",
        "components": {
            "SESI": 1.5,
            "SENAI": 1.0,
            "SEBRAE": 0.6,
            "INCRA": 0.2,
            "Salário-Educação": 2.5,
        },
        "total_pct": 5.8,
    },
    "commerce": {
        "fpas_codes": [515],
        "label": "Commercial firms",
        "components": {
            "SESC": 1.5,
            "SENAC": 1.0,
            "SEBRAE": 0.6,
            "INCRA": 0.2,
            "Salário-Educação": 2.5,
        },
        "total_pct": 5.8,
    },
    "transport": {
        "fpas_codes": [612],
        "label": "Transport firms",
        "components": {
            "SEST": 1.5,
            "SENAT": 1.0,
            "SEBRAE": 0.6,
            "INCRA": 0.2,
            "Salário-Educação": 2.5,
        },
        "total_pct": 5.8,
    },
    "rural": {
        "fpas_codes": [604, 787],
        "label": "Rural / agricultural firms",
        "components": {
            "SENAR": 2.5,
            "INCRA": 2.7,
            "Salário-Educação": 2.5,
        },
        "total_pct": 7.7,
    },
    "cooperatives": {
        "fpas_codes": [736, 779],
        "label": "Cooperatives",
        "components": {
            "SESCOOP": 2.5,
            "SEBRAE": 0.6,
            "INCRA": 0.2,
            "Salário-Educação": 2.5,
        },
        "total_pct": 5.8,
    },
    "financial": {
        "fpas_codes": [566, 574],
        "label": "Financial institutions",
        "components": {
            "SESC": 1.5,  # or equivalent
            "SENAC": 1.0,
            "SEBRAE": 0.6,
            "INCRA": 0.2,
            "Salário-Educação": 2.5,
        },
        "total_pct": 5.8,
    },
}

# ---------------------------------------------------------------------------
# Mapping from CNAE Division (2-digit) to sector type for Terceiros
# ---------------------------------------------------------------------------

CNAE_DIVISION_TO_SECTOR_TYPE = {
    # Agriculture / rural
    "01": "rural", "02": "rural", "03": "rural",
    # Mining → industry
    "05": "industry", "06": "industry", "07": "industry",
    "08": "industry", "09": "industry",
    # Manufacturing → industry
    **{str(d).zfill(2): "industry" for d in range(10, 34)},
    # Utilities → industry
    "35": "industry", "36": "industry", "37": "industry",
    "38": "industry", "39": "industry",
    # Construction → industry
    "41": "industry", "42": "industry", "43": "industry",
    # Trade → commerce
    "45": "commerce", "46": "commerce", "47": "commerce",
    # Transport → transport
    "49": "transport", "50": "transport", "51": "transport",
    "52": "transport", "53": "transport",
    # Accommodation / food → commerce
    "55": "commerce", "56": "commerce",
    # Information / communication → commerce (general)
    "58": "commerce", "59": "commerce", "60": "commerce",
    "61": "commerce", "62": "commerce", "63": "commerce",
    # Finance → financial
    "64": "financial", "65": "financial", "66": "financial",
    # Real estate → commerce
    "68": "commerce",
    # Professional services → commerce
    "69": "commerce", "70": "commerce", "71": "commerce",
    "72": "commerce", "73": "commerce", "74": "commerce",
    "75": "commerce",
    # Admin services → commerce
    "77": "commerce", "78": "commerce", "79": "commerce",
    "80": "commerce", "81": "commerce", "82": "commerce",
    # Education, health, social → commerce
    "84": "commerce", "85": "commerce", "86": "commerce",
    "87": "commerce", "88": "commerce",
    # Arts, entertainment → commerce
    "90": "commerce", "91": "commerce", "92": "commerce",
    "93": "commerce",
    # Other services → commerce
    "94": "commerce", "95": "commerce", "96": "commerce",
    "97": "commerce",
}


def get_terceiros_rate(cnae_code: str) -> dict:
    """Get the Terceiros/Sistema S rate for a CNAE code."""
    division = cnae_code[:2]
    sector_type = CNAE_DIVISION_TO_SECTOR_TYPE.get(division)
    if sector_type is None:
        return None
    info = TERCEIROS_RATES[sector_type]
    return {
        "cnae_division": division,
        "sector_type": sector_type,
        "terceiros_total_pct": info["total_pct"],
        "components": info["components"],
    }


def build_terceiros_table() -> pd.DataFrame:
    """Build a DataFrame of Terceiros rates by CNAE division."""
    rows = []
    for div, sector_type in sorted(CNAE_DIVISION_TO_SECTOR_TYPE.items()):
        info = TERCEIROS_RATES[sector_type]
        rows.append({
            "cnae_division": div,
            "sector_type": sector_type,
            "terceiros_total_pct": info["total_pct"],
        })
    return pd.DataFrame(rows)
