"""
DATACOLLECT shared utilities -- download, cache, retry, logging helpers.
"""

import hashlib
import json
import logging
import time
from pathlib import Path

from config import DEFAULT_RETRY_ATTEMPTS, DEFAULT_RETRY_DELAY

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("datacollect")


# ── Caching ────────────────────────────────────────────────────────────
def cache_key(url: str, params: dict | None = None) -> str:
    """Generate a deterministic cache key from URL + params."""
    raw = url + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def cached_download(url: str, cache_dir: Path, params: dict | None = None) -> Path:
    """Download a URL, caching the result. Returns path to cached file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(url, params)
    cached_path = cache_dir / key

    if cached_path.exists():
        log.info(f"Cache hit: {url}")
        return cached_path

    import requests
    log.info(f"Downloading: {url}")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    cached_path.write_bytes(resp.content)
    return cached_path


# ── Retry ──────────────────────────────────────────────────────────────
def retry(fn, attempts: int = DEFAULT_RETRY_ATTEMPTS, delay: int = DEFAULT_RETRY_DELAY):
    """Retry a callable with exponential backoff."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise
            wait = delay * (2 ** i)
            log.warning(f"Attempt {i+1} failed ({e}), retrying in {wait}s...")
            time.sleep(wait)
