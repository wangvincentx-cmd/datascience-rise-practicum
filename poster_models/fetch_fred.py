"""Populate cache/fred_*.csv so every model in this folder can run offline.

Exists because `cache/` is gitignored (regenerable) and because urllib against
fredgraph.csv hangs on some networks where curl succeeds -- notably behind a
TLS-inspecting proxy. This retries, falls back to curl, and validates that what
landed on disk is actually a CSV rather than an error page.

    python poster_models/fetch_fred.py
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, SRC  # noqa: E402

sys.path.insert(0, str(SRC))
from truth_data import STOCK_SERIES  # noqa: E402

SERIES = ["INDPRO", "CPIAUCNS", "UNRATE", STOCK_SERIES]
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
CACHE = ROOT / "cache"


def looks_like_csv(path):
    """A truncated download or an HTML error page must not be cached as data."""
    if not path.exists() or path.stat().st_size < 200:
        return False
    head = path.read_text(errors="replace")[:200].lower()
    return "," in head and "<html" not in head


def fetch(sid, tries=8):
    out = CACHE / f"fred_{sid}.csv"
    if looks_like_csv(out):
        print(f"  {sid:<20} already cached ({out.stat().st_size:,} bytes)")
        return True
    for i in range(tries):
        out.unlink(missing_ok=True)
        subprocess.run(
            ["curl", "-sS", "--http1.1", "-m", "30", "-A", "Mozilla/5.0",
             "-o", str(out), URL.format(sid=sid)],
            capture_output=True)
        if looks_like_csv(out):
            print(f"  {sid:<20} ok ({out.stat().st_size:,} bytes, attempt {i + 1})")
            return True
        time.sleep(5)
    out.unlink(missing_ok=True)
    print(f"  {sid:<20} FAILED after {tries} attempts")
    return False


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    ok = [fetch(s) for s in SERIES]
    if not all(ok):
        print("\nSome series are missing. FRED rate-limits bursts -- wait a few "
              "minutes and re-run; already-cached series are skipped.")
        sys.exit(1)
    print("\nAll series cached. poster_models can now run fully offline.")
