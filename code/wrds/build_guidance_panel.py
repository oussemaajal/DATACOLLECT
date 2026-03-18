"""
Build a clean panel dataset from raw I/B/E/S management guidance data.

Steps:
    1. Load raw guidance + identifier crosswalks
    2. Merge guidance with CRSP (PERMNO) and Compustat (GVKEY) via CUSIP
    3. Create derived variables (guidance type labels, surprise, range width)
    4. Output firm-announcement level panel in clean/ directory

Input:  data/raw/guidance/*.parquet  (from guidance.py)
Output: data/clean/guidance_panel.parquet

Usage:
    python build_guidance_panel.py
    python build_guidance_panel.py --input-tag 2002_2025  # specific date range
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_INTERMEDIATE, DATA_CLEAN
from utils import log

RAW_DIR = DATA_RAW / "guidance"

# ── Guidance measure labels ───────────────────────────────────────────────
MEASURE_LABELS = {
    "EPS": "Earnings per share",
    "SAL": "Revenue / Sales",
    "BPS": "Book value per share",
    "CPS": "Cash flow per share",
    "DPS": "Dividends per share",
    "EBI": "EBITDA",
    "EBT": "EBIT",
    "FFO": "Funds from operations",
    "GPS": "Gross profit per share",
    "NAV": "Net asset value",
    "NET": "Net income",
    "OPR": "Operating profit",
    "PRE": "Pre-tax income",
    "ROA": "Return on assets",
    "ROE": "Return on equity",
    "CAP": "Capital expenditure",
}

# ── Guidance form labels ──────────────────────────────────────────────────
GESSION_LABELS = {
    1: "Point estimate",
    2: "Range",
    3: "Open-ended low (at least)",
    4: "Open-ended high (at most)",
    5: "Confirming prior",
    6: "Qualitative",
}

PDICITY_LABELS = {
    "QTR": "Quarterly",
    "ANN": "Annual",
    "SAN": "Semi-annual",
}


def load_raw_guidance(tag: str) -> pd.DataFrame:
    """Load raw guidance parquet."""
    path = RAW_DIR / f"det_guidance_{tag}.parquet"
    log.info(f"Loading raw guidance from {path}")
    df = pd.read_parquet(path)
    log.info(f"  {len(df):,} records loaded")
    return df


def load_crsp_mapping() -> pd.DataFrame:
    """Load CRSP CUSIP->PERMNO mapping."""
    path = RAW_DIR / "crsp_cusip_permno.parquet"
    log.info(f"Loading CRSP mapping from {path}")
    return pd.read_parquet(path)


def load_compustat_mapping() -> pd.DataFrame:
    """Load Compustat CUSIP->GVKEY mapping."""
    path = RAW_DIR / "compustat_cusip_gvkey.parquet"
    log.info(f"Loading Compustat mapping from {path}")
    return pd.read_parquet(path)


def merge_crsp(guidance: pd.DataFrame, crsp: pd.DataFrame) -> pd.DataFrame:
    """
    Merge guidance with CRSP to get PERMNO.

    Uses 8-digit CUSIP from I/B/E/S matched to ncusip from CRSP,
    with date-range validation so the CUSIP was active at announcement time.
    """
    log.info("Merging with CRSP for PERMNO...")

    # Standardize CUSIP to 8 chars
    guidance["cusip8"] = guidance["cusip"].str.strip().str[:8]
    crsp["ncusip8"] = crsp["ncusip"].str.strip().str[:8]

    # Ensure date columns are datetime
    guidance["anndats"] = pd.to_datetime(guidance["anndats"])
    crsp["namedt"] = pd.to_datetime(crsp["namedt"])
    crsp["nameendt"] = pd.to_datetime(crsp["nameendt"])

    # Deduplicate CRSP mapping: keep one row per ncusip-date range
    crsp_dedup = (
        crsp[["permno", "permco", "ncusip8", "exchcd", "shrcd", "namedt", "nameendt"]]
        .drop_duplicates(subset=["ncusip8", "permno", "namedt"])
    )

    # Merge on CUSIP
    merged = guidance.merge(
        crsp_dedup,
        left_on="cusip8",
        right_on="ncusip8",
        how="left",
    )

    # Keep only rows where announcement date falls within CRSP name validity
    mask = (
        merged["namedt"].isna()  # keep unmatched rows too
        | (
            (merged["anndats"] >= merged["namedt"])
            & (merged["anndats"] <= merged["nameendt"])
        )
    )
    merged = merged[mask]

    # If multiple CRSP matches, keep the one with the most common exchcd
    # (prefer NYSE/AMEX/NASDAQ: exchcd in 1,2,3)
    merged = (
        merged.sort_values(["exchcd"], ascending=True)
        .drop_duplicates(
            subset=["ticker", "anndats", "statpers", "measure", "gession", "val1", "val2"],
            keep="first",
        )
    )

    n_matched = merged["permno"].notna().sum()
    log.info(f"  CRSP match rate: {n_matched:,}/{len(merged):,} "
             f"({100 * n_matched / len(merged):.1f}%)")

    # Clean up temp columns
    merged.drop(columns=["cusip8", "ncusip8", "namedt", "nameendt", "shrcd"],
                errors="ignore", inplace=True)
    return merged


def merge_compustat(panel: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """
    Merge with Compustat to get GVKEY.

    Uses 8-digit CUSIP matched with date-range validation.
    """
    log.info("Merging with Compustat for GVKEY...")

    panel["cusip8"] = panel["cusip"].str.strip().str[:8]
    comp["cusip8"] = comp["cusip"].str.strip().str[:8]

    comp["secstartdt"] = pd.to_datetime(comp["secstartdt"])
    comp["secenddt"] = pd.to_datetime(comp["secenddt"])

    comp_dedup = (
        comp[["gvkey", "cusip8", "tic", "secstartdt", "secenddt"]]
        .drop_duplicates(subset=["cusip8", "gvkey", "secstartdt"])
    )

    merged = panel.merge(
        comp_dedup,
        on="cusip8",
        how="left",
        suffixes=("", "_comp"),
    )

    # Date-range filter
    mask = (
        merged["secstartdt"].isna()
        | (
            (merged["anndats"] >= merged["secstartdt"])
            & (merged["anndats"] <= merged["secenddt"])
        )
    )
    merged = merged[mask]

    # Deduplicate
    merged = merged.drop_duplicates(
        subset=["ticker", "anndats", "statpers", "measure", "gession", "val1", "val2"],
        keep="first",
    )

    n_matched = merged["gvkey"].notna().sum()
    log.info(f"  Compustat match rate: {n_matched:,}/{len(merged):,} "
             f"({100 * n_matched / len(merged):.1f}%)")

    merged.drop(columns=["cusip8", "tic_comp", "secstartdt", "secenddt"],
                errors="ignore", inplace=True)
    return merged


def add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable labels and derived analytical variables."""
    log.info("Adding derived variables...")

    # Measure and form labels
    df["measure_label"] = df["measure"].map(MEASURE_LABELS).fillna("Other")
    df["gession_label"] = df["gession"].map(GESSION_LABELS).fillna("Unknown")
    df["pdicity_label"] = df["pdicity"].map(PDICITY_LABELS).fillna("Unknown")

    # Point estimate: midpoint of range, or val1 for point estimates
    df["guidance_mid"] = df.apply(
        lambda r: (r["val1"] + r["val2"]) / 2
        if pd.notna(r["val2"]) and r["val2"] != r["val1"]
        else r["val1"],
        axis=1,
    )

    # Range width (0 for point estimates)
    df["range_width"] = (df["val2"] - df["val1"]).clip(lower=0)
    df.loc[df["val2"].isna(), "range_width"] = 0.0

    # Is this a range estimate?
    df["is_range"] = (df["gession"] == 2).astype(int)

    # Is this a point estimate?
    df["is_point"] = (df["gession"] == 1).astype(int)

    # Guidance surprise: midpoint vs consensus mean at time of guidance
    df["surprise"] = df["guidance_mid"] - df["meanest"]
    df["surprise_pct"] = df["surprise"] / df["meanest"].abs().replace(0, float("nan"))

    # Meet-or-beat actual: did the actual meet/beat the guidance midpoint?
    df["actual_vs_guidance"] = df["actual"] - df["guidance_mid"]

    # Calendar year-quarter for panel structure
    df["anndats"] = pd.to_datetime(df["anndats"])
    df["ann_year"] = df["anndats"].dt.year
    df["ann_quarter"] = df["anndats"].dt.quarter
    df["ann_yearq"] = df["ann_year"].astype(str) + "Q" + df["ann_quarter"].astype(str)

    df["statpers"] = pd.to_datetime(df["statpers"])
    df["stat_year"] = df["statpers"].dt.year
    df["stat_quarter"] = df["statpers"].dt.quarter

    return df


def select_and_order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select final columns and sort for a clean panel."""
    cols = [
        # Identifiers
        "permno", "permco", "gvkey", "ticker", "oftic", "cusip", "cname",
        # Guidance details
        "measure", "measure_label", "pdicity", "pdicity_label",
        "anndats", "anntims", "ann_year", "ann_quarter", "ann_yearq",
        "statpers", "stat_year", "stat_quarter",
        # Values
        "val1", "val2", "guidance_mid", "range_width",
        "gession", "gession_label", "guession",
        "is_point", "is_range",
        # Analyst context
        "numest", "meanest", "medest", "dispersion",
        "surprise", "surprise_pct",
        # Actuals
        "actual", "anndats_act", "actual_vs_guidance",
        # Exchange info
        "exchcd",
    ]
    # Keep only columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(["permno", "gvkey", "anndats", "statpers", "measure"])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build clean guidance panel from raw WRDS data"
    )
    parser.add_argument(
        "--input-tag", type=str, default="2002_2025",
        help="Date range tag matching raw file name (default: 2002_2025)"
    )
    args = parser.parse_args()

    # Load raw data
    guidance = load_raw_guidance(args.input_tag)
    crsp = load_crsp_mapping()
    comp = load_compustat_mapping()

    # Merge identifiers
    panel = merge_crsp(guidance, crsp)
    panel = merge_compustat(panel, comp)

    # Add derived variables
    panel = add_derived_variables(panel)

    # Select and order columns
    panel = select_and_order_columns(panel)

    log.info(f"Final panel: {len(panel):,} rows, {len(panel.columns)} columns")
    log.info(f"  Unique firms (PERMNO): {panel['permno'].nunique():,}")
    log.info(f"  Unique firms (GVKEY):  {panel['gvkey'].nunique():,}")
    log.info(f"  Date range: {panel['anndats'].min()} to {panel['anndats'].max()}")
    log.info(f"  Measures: {panel['measure'].value_counts().to_dict()}")

    # Save intermediate (with all merge artifacts)
    DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    int_path = DATA_INTERMEDIATE / f"guidance_panel_merged_{args.input_tag}.parquet"
    panel.to_parquet(int_path, index=False)
    log.info(f"  Saved intermediate -> {int_path}")

    # Save clean
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    clean_path = DATA_CLEAN / "guidance_panel.parquet"
    panel.to_parquet(clean_path, index=False)
    log.info(f"  Saved clean panel -> {clean_path}")

    # Also save a summary CSV for quick inspection
    summary = panel.groupby(["ann_year", "measure"]).agg(
        n_guidance=("ticker", "size"),
        n_firms=("permno", "nunique"),
        pct_range=("is_range", "mean"),
        avg_surprise=("surprise", "mean"),
    ).reset_index()

    summary_path = DATA_CLEAN / "guidance_panel_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"  Saved summary -> {summary_path}")

    log.info("Panel construction complete.")


if __name__ == "__main__":
    main()
