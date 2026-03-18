"""
DATACOLLECT configuration -- paths, constants, credentials (via env vars).
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERMEDIATE = PROJECT_ROOT / "data" / "intermediate"
DATA_CLEAN = PROJECT_ROOT / "data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ── Credentials (from environment variables) ───────────────────────────
WRDS_USERNAME = os.getenv("WRDS_USERNAME", "")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")  # Required by SEC EDGAR API

# ── Rate limiting defaults ─────────────────────────────────────────────
SEC_RATE_LIMIT = 10  # requests per second (SEC allows 10/sec)
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 5  # seconds
