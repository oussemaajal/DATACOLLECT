"""
SAT/RAT (Seguro Acidente de Trabalho / Risco Ambiental do Trabalho) rates.

Base rates are 1%, 2%, or 3% of payroll depending on the CNAE subclass,
adjusted by the firm-specific FAP multiplier (0.5 to 2.0).

Effective RAT = base_rate * FAP

Source: Decreto 3.048/1999, Anexo V (as updated by Decreto 6.957/2009)
CNAE classification: IBGE CNAE 2.0

This module provides the mapping from CNAE Division (2-digit) to the most
common RAT base rate for that division. For precise 7-digit CNAE subclass
rates, use the full Anexo V table (scraped separately).
"""

import pandas as pd


# ---------------------------------------------------------------------------
# RAT base rates by CNAE Division (2-digit code)
# These are the PREDOMINANT rates for each division. Individual 7-digit
# subclasses within a division may differ.
# Source: Decreto 3.048/1999, Anexo V (consolidated)
# ---------------------------------------------------------------------------

RAT_BY_CNAE_DIVISION = {
    # Agriculture, forestry, fishing (01-03)
    "01": ("Agriculture, livestock", 3),
    "02": ("Forestry", 3),
    "03": ("Fishing, aquaculture", 3),
    # Mining (05-09)
    "05": ("Coal mining", 3),
    "06": ("Oil and gas extraction", 3),
    "07": ("Metal ore mining", 3),
    "08": ("Non-metallic mineral mining", 3),
    "09": ("Mining support services", 3),
    # Manufacturing (10-33)
    "10": ("Food products", 3),
    "11": ("Beverages", 3),
    "12": ("Tobacco products", 2),
    "13": ("Textiles", 3),
    "14": ("Apparel", 2),
    "15": ("Leather and footwear", 3),
    "16": ("Wood products", 3),
    "17": ("Paper and pulp", 3),
    "18": ("Printing and recorded media", 2),
    "19": ("Coke and petroleum products", 3),
    "20": ("Chemicals", 3),
    "21": ("Pharmaceuticals", 2),
    "22": ("Rubber and plastics", 3),
    "23": ("Non-metallic minerals", 3),
    "24": ("Basic metals", 3),
    "25": ("Fabricated metal products", 3),
    "26": ("Computers and electronics", 2),
    "27": ("Electrical equipment", 3),
    "28": ("Machinery and equipment", 3),
    "29": ("Motor vehicles", 3),
    "30": ("Other transport equipment", 3),
    "31": ("Furniture", 3),
    "32": ("Other manufacturing", 2),
    "33": ("Repair/installation of machinery", 3),
    # Utilities (35-39)
    "35": ("Electricity, gas, steam", 2),
    "36": ("Water collection/treatment", 2),
    "37": ("Sewerage", 3),
    "38": ("Waste collection/treatment", 3),
    "39": ("Remediation", 3),
    # Construction (41-43)
    "41": ("Building construction", 3),
    "42": ("Civil engineering", 3),
    "43": ("Specialized construction", 3),
    # Trade (45-47)
    "45": ("Vehicle trade and repair", 2),
    "46": ("Wholesale trade", 2),
    "47": ("Retail trade", 1),
    # Transport and storage (49-53)
    "49": ("Land transport", 3),
    "50": ("Water transport", 3),
    "51": ("Air transport", 2),
    "52": ("Warehousing/transport support", 3),
    "53": ("Postal/courier", 2),
    # Accommodation and food (55-56)
    "55": ("Accommodation", 2),
    "56": ("Food and beverage services", 2),
    # Information and communication (58-63)
    "58": ("Publishing", 1),
    "59": ("Film/TV/music production", 1),
    "60": ("Broadcasting", 1),
    "61": ("Telecommunications", 2),
    "62": ("IT services", 1),
    "63": ("Information services", 1),
    # Finance and insurance (64-66)
    "64": ("Financial services", 1),
    "65": ("Insurance", 1),
    "66": ("Auxiliary financial services", 1),
    # Real estate (68)
    "68": ("Real estate", 2),
    # Professional / scientific / technical (69-75)
    "69": ("Legal and accounting", 1),
    "70": ("Management consulting", 1),
    "71": ("Architecture and engineering", 2),
    "72": ("Scientific R&D", 1),
    "73": ("Advertising and market research", 1),
    "74": ("Other professional services", 1),
    "75": ("Veterinary", 2),
    # Administrative services (77-82)
    "77": ("Rental and leasing", 1),
    "78": ("Employment agencies", 2),
    "79": ("Travel agencies", 1),
    "80": ("Security/investigation", 2),
    "81": ("Building services/landscaping", 3),
    "82": ("Office/business support", 1),
    # Public admin (84)
    "84": ("Public administration", 2),
    # Education (85)
    "85": ("Education", 1),
    # Health (86-88)
    "86": ("Human health", 2),
    "87": ("Residential care", 2),
    "88": ("Social work without accommodation", 1),
    # Arts, entertainment, recreation (90-93)
    "90": ("Creative arts", 1),
    "91": ("Libraries, museums", 1),
    "92": ("Gambling", 2),
    "93": ("Sports, recreation", 2),
    # Other services (94-96)
    "94": ("Membership organizations", 1),
    "95": ("Repair of computers/personal goods", 2),
    "96": ("Other personal services", 1),
    # Domestic workers (97)
    "97": ("Domestic workers", 1),
    # International organizations (99)
    "99": ("International organizations", 1),
}


def get_rat_base(cnae_code: str) -> dict:
    """
    Get the RAT base rate for a CNAE code.

    cnae_code: any length (2-digit division, 5-digit group, or 7-digit subclass)
    Returns dict with division, label, base_rate_pct, or None if not found.
    """
    division = cnae_code[:2]
    info = RAT_BY_CNAE_DIVISION.get(division)
    if info is None:
        return None
    return {
        "cnae_division": division,
        "division_label": info[0],
        "rat_base_pct": info[1],
        "rat_min_pct": info[1] * 0.5,  # FAP = 0.5
        "rat_max_pct": info[1] * 2.0,  # FAP = 2.0
    }


def build_rat_table() -> pd.DataFrame:
    """Build a DataFrame of RAT base rates by CNAE division."""
    rows = []
    for div, (label, rate) in RAT_BY_CNAE_DIVISION.items():
        rows.append({
            "cnae_division": div,
            "division_label": label,
            "rat_base_pct": rate,
            "rat_min_pct": rate * 0.5,
            "rat_max_pct": rate * 2.0,
        })
    return pd.DataFrame(rows).sort_values("cnae_division").reset_index(drop=True)
