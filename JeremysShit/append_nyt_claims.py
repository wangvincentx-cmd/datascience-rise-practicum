"""
Merge NYT post-1963 raw articles into claims_raw.csv so they flow through the
SAME validated pipeline as the LOC 1905-1958 corpus (grade_claims.py's
kappa-checked rubric, then score_claims.py's NBER/FRED scoring) -- one rubric,
one corpus, 1905-2010, instead of a second unvalidated extraction schema.

Reads election_arm/data/raw/nyt_economy_{window}.jsonl for the 9 windows
covering 1963-2010 (LOC full text ends 1963; earlier windows are already in
claims_raw.csv via newspaper_scraper.py and are skipped here on purpose).

Idempotent: safe to rerun after a deeper download_nyt.py pass (e.g. after the
2026-07-16 pagination fix) -- dedupes on page_url against what's already in
claims_raw.csv, only appends genuinely new articles, continues the existing
claim_id sequence.

Usage:
    python append_nyt_claims.py                 # appends into claims_raw.csv
    python append_nyt_claims.py --dry-run        # report counts only
"""

import argparse
import csv
import json
from pathlib import Path

CLAIMS_RAW = Path("claims_raw.csv")
NYT_RAW_DIR = Path("election_arm/data/raw")
FIELDS = ["claim_id", "episode", "kind", "publisher", "state", "date",
          "search_term", "page_url", "quote"]

# window_id -> (episode display name, kind) -- kind/naming matches the
# existing "YYYY Name" / "YYYY Calm (control)" convention in claims_raw.csv.
WINDOWS = {
    "oil_1973":     ("1973 Oil Shock", "crisis"),
    "volcker_1980": ("1980 Recession", "crisis"),
    "crash_1987":   ("1987 Crash", "crisis"),
    "gulf_1990":    ("1990 Recession", "crisis"),
    "dotcom_2001":  ("2001 Dot-com Bust", "crisis"),
    "gfc_2008":     ("2008 Financial Crisis", "crisis"),
    "calm_1965":    ("1965 Calm (control)", "control"),
    "calm_1995":    ("1995 Calm (control)", "control"),
    "calm_2005":    ("2005 Calm (control)", "control"),
}


def load_existing():
    rows = []
    seen_urls = set()
    max_id = 0
    if CLAIMS_RAW.exists():
        with open(CLAIMS_RAW, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                seen_urls.add(row["page_url"])
                max_id = max(max_id, int(row["claim_id"]))
    return rows, seen_urls, max_id


def nyt_rows(seen_urls, next_id):
    new_rows = []
    per_window = {}
    for window_id, (episode, kind) in WINDOWS.items():
        path = NYT_RAW_DIR / f"nyt_economy_{window_id}.jsonl"
        if not path.exists():
            continue
        added = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                url = rec.get("page_id")
                quote = (rec.get("ocr_text") or "").strip()
                if not url or not quote or url in seen_urls:
                    continue
                new_rows.append({
                    "claim_id": next_id,
                    "episode": episode,
                    "kind": kind,
                    "publisher": "the new york times",
                    "state": "",
                    "date": rec.get("date", ""),
                    "search_term": rec.get("matched_phrase", ""),
                    "page_url": url,
                    "quote": quote,
                })
                seen_urls.add(url)
                next_id += 1
                added += 1
        per_window[window_id] = added
    return new_rows, per_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be appended, write nothing")
    args = ap.parse_args()

    existing_rows, seen_urls, max_id = load_existing()
    new_rows, per_window = nyt_rows(seen_urls, max_id + 1)

    print(f"claims_raw.csv currently has {len(existing_rows)} rows (max claim_id {max_id})")
    for wid, n in per_window.items():
        print(f"  {wid}: {n} new articles")
    print(f"total new claims to append: {len(new_rows)}")

    if args.dry_run or not new_rows:
        return

    with open(CLAIMS_RAW, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        for row in new_rows:
            w.writerow(row)
    print(f"appended {len(new_rows)} rows -> {CLAIMS_RAW} "
          f"(now {len(existing_rows) + len(new_rows)} total; "
          f"rerun grade_claims.py to grade only the new ones)")


if __name__ == "__main__":
    main()
