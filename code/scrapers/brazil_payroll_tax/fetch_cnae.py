"""
Fetch the CNAE 2.0 classification table from IBGE (CONCLA).

Downloads the official CNAE hierarchy (section → division → group → class → subclass)
and saves as CSV for joining with payroll tax rate tables.

Source: https://concla.ibge.gov.br/
API: https://servicodados.ibge.gov.br/api/docs/cnae

The IBGE CNAE API provides structured JSON data for all levels of the
classification. No authentication required. Rate-limit respectfully.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from code.config import DATA_RAW, DEFAULT_RETRY_ATTEMPTS, DEFAULT_RETRY_DELAY

logger = logging.getLogger(__name__)

CNAE_API_BASE = "https://servicodados.ibge.gov.br/api/v2/cnae"

RAW_DIR = DATA_RAW / "brazil_payroll_tax"


def _get_json(url: str, retries: int = DEFAULT_RETRY_ATTEMPTS) -> list:
    """Fetch JSON from IBGE API with retries."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(DEFAULT_RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def fetch_divisions() -> pd.DataFrame:
    """Fetch all CNAE divisions (2-digit codes)."""
    logger.info("Fetching CNAE divisions...")
    data = _get_json(f"{CNAE_API_BASE}/divisoes")
    rows = []
    for item in data:
        rows.append({
            "division_id": item["id"],
            "division_desc": item["descricao"],
        })
    return pd.DataFrame(rows)


def fetch_classes() -> pd.DataFrame:
    """
    Fetch all CNAE classes (5-digit codes) with their group and division info.
    This is the main workhorse — gives us the full hierarchy.
    """
    logger.info("Fetching CNAE classes (this may take a moment)...")
    data = _get_json(f"{CNAE_API_BASE}/classes")
    rows = []
    for item in data:
        class_id = item["id"]
        class_desc = item["descricao"]

        # Extract group info
        grupo = item.get("grupo", {})
        group_id = grupo.get("id", "")
        group_desc = grupo.get("descricao", "")

        # Extract division info
        divisao = grupo.get("divisao", {})
        division_id = divisao.get("id", "")
        division_desc = divisao.get("descricao", "")

        # Extract section info
        secao = divisao.get("secao", {}) if divisao else {}
        section_id = secao.get("id", "")
        section_desc = secao.get("descricao", "")

        rows.append({
            "section_id": section_id,
            "section_desc": section_desc,
            "division_id": str(division_id).zfill(2),
            "division_desc": division_desc,
            "group_id": str(group_id),
            "group_desc": group_desc,
            "class_id": str(class_id),
            "class_desc": class_desc,
        })
    return pd.DataFrame(rows)


def fetch_subclasses() -> pd.DataFrame:
    """
    Fetch all CNAE subclasses (7-digit codes).
    This is the most granular level and what firms actually report.
    """
    logger.info("Fetching CNAE subclasses (this will take a while)...")
    data = _get_json(f"{CNAE_API_BASE}/subclasses")
    rows = []
    for item in data:
        subclass_id = item["id"]
        subclass_desc = item["descricao"]

        # Extract class info
        classe = item.get("classe", {})
        class_id = classe.get("id", "")
        class_desc = classe.get("descricao", "")

        # Extract group
        grupo = classe.get("grupo", {}) if classe else {}
        group_id = grupo.get("id", "")

        # Extract division
        divisao = grupo.get("divisao", {}) if grupo else {}
        division_id = divisao.get("id", "")

        # Extract section
        secao = divisao.get("secao", {}) if divisao else {}
        section_id = secao.get("id", "")

        rows.append({
            "section_id": section_id,
            "division_id": str(division_id).zfill(2) if division_id else "",
            "group_id": str(group_id),
            "class_id": str(class_id),
            "subclass_id": str(subclass_id),
            "subclass_desc": subclass_desc,
        })
    return pd.DataFrame(rows)


def fetch_and_save_cnae(level: str = "classes") -> Path:
    """
    Fetch CNAE data at the specified level and save to CSV.

    level: "divisions", "classes", or "subclasses"
    Returns path to saved CSV.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if level == "divisions":
        df = fetch_divisions()
    elif level == "classes":
        df = fetch_classes()
    elif level == "subclasses":
        df = fetch_subclasses()
    else:
        raise ValueError(f"Unknown level: {level}. Use 'divisions', 'classes', or 'subclasses'.")

    out_path = RAW_DIR / f"cnae_{level}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} CNAE {level} to {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Fetch classes (5-digit) — the best balance of detail and speed
    fetch_and_save_cnae("classes")
    # Optionally fetch subclasses (7-digit) — slower but most granular
    # fetch_and_save_cnae("subclasses")
