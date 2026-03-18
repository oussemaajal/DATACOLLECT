"""
Pull management guidance (earnings forecasts) from WRDS I/B/E/S Guidance database.

Creates a panel dataset of all US public-firm management guidance including:
- Earnings per share (EPS) guidance
- Revenue guidance
- Other financial metric guidance (EBITDA, FFO, capex, etc.)

Source tables:
    ibes.det_guidance   -- Individual guidance records
    ibes.id_sum         -- I/B/E/S identifier crosswalk (ticker -> CUSIP)
    comp.security       -- Compustat security mapping (CUSIP -> GVKEY)
    crsp.stocknames     -- CRSP identifier mapping (CUSIP -> PERMNO)

Usage:
    python guidance.py                  # Full pull (all years)
    python guidance.py --start 2000     # From 2000 onwards
    python guidance.py --end 2025       # Up to 2025
    python guidance.py --start 2000 --end 2025  # Date range
"""

import argparse
import sys
from pathlib import Path

import wrds

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, WRDS_USERNAME
from utils import log

# ── Output paths ──────────────────────────────────────────────────────────
RAW_DIR = DATA_RAW / "guidance"


def connect_wrds() -> wrds.Connection:
    """Establish WRDS connection."""
    log.info("Connecting to WRDS...")
    kwargs = {}
    if WRDS_USERNAME:
        kwargs["wrds_username"] = WRDS_USERNAME
    return wrds.Connection(**kwargs)


def pull_guidance(db: wrds.Connection, start_year: int, end_year: int) -> Path:
    """
    Pull raw management guidance from ibes.det_guidance.

    Key columns:
        ticker      - I/B/E/S ticker
        cname       - Company name
        oftic       - Official ticker (exchange ticker)
        measure     - Guidance metric (EPS, SAL, BPS, CPS, DPS, EBI, FFO, etc.)
        pdicity     - Periodicity (QTR=quarterly, ANN=annual, SAN=semi-annual)
        anndats     - Announcement date (when guidance was issued)
        anntims     - Announcement time
        statpers    - Statistical period end date (period being guided)
        val1 / val2 - Guidance values (point estimate or range endpoints)
        gession     - Guidance form (1=point, 2=range, 3=open-ended low,
                       4=open-ended high, 5=confirming, 6=qualitative)
        guession    - Guidance sub-type detail
        cusip       - 8-digit CUSIP
    """
    log.info(f"Pulling guidance data from {start_year} to {end_year}...")

    query = f"""
        SELECT
            g.ticker,
            g.cname,
            g.oftic,
            g.cusip,
            g.measure,
            g.pdicity,
            g.anndats,
            g.anntims,
            g.statpers,
            g.val1,
            g.val2,
            g.gession,
            g.guession,
            g.dispersion,
            g.numest,
            g.medest,
            g.meanest,
            g.actual,
            g.anndats_act
        FROM ibes.det_guidance AS g
        WHERE EXTRACT(YEAR FROM g.anndats) BETWEEN {start_year} AND {end_year}
        ORDER BY g.ticker, g.anndats, g.statpers
    """

    df = db.raw_sql(query)
    log.info(f"  Retrieved {len(df):,} guidance records")

    # Save raw
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"det_guidance_{start_year}_{end_year}.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"  Saved raw guidance -> {out_path}")
    return out_path


def pull_ibes_identifiers(db: wrds.Connection) -> Path:
    """
    Pull I/B/E/S identifier mapping (ticker -> CUSIP + company info).
    Used to crosswalk guidance records to CRSP/Compustat.
    """
    log.info("Pulling I/B/E/S identifier crosswalk...")

    query = """
        SELECT DISTINCT
            ticker,
            cusip,
            oftic,
            cname,
            sdates,
            edates
        FROM ibes.id_sum
        ORDER BY ticker, sdates
    """

    df = db.raw_sql(query)
    log.info(f"  Retrieved {len(df):,} I/B/E/S identifier records")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "ibes_identifiers.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"  Saved I/B/E/S identifiers -> {out_path}")
    return out_path


def pull_crsp_cusip_permno(db: wrds.Connection) -> Path:
    """
    Pull CRSP CUSIP -> PERMNO mapping for merging.
    Uses crsp.stocknames which has historical CUSIP assignments.
    """
    log.info("Pulling CRSP CUSIP-PERMNO mapping...")

    query = """
        SELECT DISTINCT
            permno,
            permco,
            ncusip,
            ticker,
            comnam,
            namedt,
            nameendt,
            exchcd,
            shrcd
        FROM crsp.stocknames
        WHERE shrcd BETWEEN 10 AND 19
        ORDER BY permno, namedt
    """

    df = db.raw_sql(query)
    log.info(f"  Retrieved {len(df):,} CRSP identifier records")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "crsp_cusip_permno.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"  Saved CRSP mapping -> {out_path}")
    return out_path


def pull_compustat_cusip_gvkey(db: wrds.Connection) -> Path:
    """
    Pull Compustat CUSIP -> GVKEY mapping.
    Uses comp.security for the broadest coverage.
    """
    log.info("Pulling Compustat CUSIP-GVKEY mapping...")

    query = """
        SELECT DISTINCT
            gvkey,
            iid,
            cusip,
            tic,
            conm,
            sdate AS secstartdt,
            edate AS secenddt
        FROM comp.security
        ORDER BY gvkey, sdate
    """

    df = db.raw_sql(query)
    log.info(f"  Retrieved {len(df):,} Compustat identifier records")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "compustat_cusip_gvkey.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"  Saved Compustat mapping -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Pull management guidance data from WRDS I/B/E/S"
    )
    parser.add_argument(
        "--start", type=int, default=2002,
        help="Start year (default: 2002, earliest reliable I/B/E/S guidance)"
    )
    parser.add_argument(
        "--end", type=int, default=2025,
        help="End year (default: 2025)"
    )
    args = parser.parse_args()

    db = connect_wrds()
    try:
        pull_guidance(db, args.start, args.end)
        pull_ibes_identifiers(db)
        pull_crsp_cusip_permno(db)
        pull_compustat_cusip_gvkey(db)
        log.info("All guidance raw data pulled successfully.")
    finally:
        db.close()
        log.info("WRDS connection closed.")


if __name__ == "__main__":
    main()
