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
import random
from pathlib import Path

CLAIMS_RAW = Path("claims_raw.csv")
NYT_RAW_DIR = Path("election_arm/data/raw")
FIELDS = ["claim_id", "episode", "kind", "publisher", "state", "date",
          "search_term", "page_url", "quote"]

# The 10 LOC-era episodes (1905-1958) run 87-212 raw claims each, avg ~147.
# Capping each NYT window to the same scale keeps every episode -- and both
# eras -- comparably weighted in the pooled corpus, instead of dotcom_2001's
# 861 candidates swamping oil_1973's dozen. There's no clean 1:1 crisis/control
# pairing to match against instead (13 crisis windows share only 6 calm
# baselines across the full corpus), so a flat per-window cap at the LOC
# episodes' own scale is the least arbitrary choice available.
PER_WINDOW_CAP = 150
SAMPLE_SEED = 0

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
    episode_counts = {}
    if CLAIMS_RAW.exists():
        with open(CLAIMS_RAW, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                seen_urls.add(row["page_url"])
                max_id = max(max_id, int(row["claim_id"]))
                episode_counts[row["episode"]] = episode_counts.get(row["episode"], 0) + 1
    return rows, seen_urls, max_id, episode_counts


def nyt_rows(seen_urls, next_id, episode_counts, cap=PER_WINDOW_CAP):
    new_rows = []
    per_window = {}
    rng = random.Random(SAMPLE_SEED)
    for window_id, (episode, kind) in WINDOWS.items():
        path = NYT_RAW_DIR / f"nyt_economy_{window_id}.jsonl"
        if not path.exists():
            continue
        candidates = []
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            url = rec.get("page_id")
            quote = (rec.get("ocr_text") or "").strip()
            if not url or not quote or url in seen_urls:
                continue
            candidates.append(rec)
            seen_urls.add(url)  # dedupe within this file too

        # Cap applies to this episode's TOTAL merged count (already-merged +
        # new), so a window already partially merged doesn't blow past the
        # cap on a later resume.
        room = cap - episode_counts.get(episode, 0) if cap is not None else len(candidates)
        room = max(room, 0)
        if len(candidates) > room:
            candidates = rng.sample(candidates, room)

        for rec in candidates:
            new_rows.append({
                "claim_id": next_id,
                "episode": episode,
                "kind": kind,
                "publisher": "the new york times",
                "state": "",
                "date": rec.get("date", ""),
                "search_term": rec.get("matched_phrase", ""),
                "page_url": rec.get("page_id"),
                "quote": (rec.get("ocr_text") or "").strip(),
            })
            next_id += 1
        per_window[window_id] = len(candidates)
        episode_counts[episode] = episode_counts.get(episode, 0) + len(candidates)
    return new_rows, per_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be appended, write nothing")
    ap.add_argument("--cap", type=int, default=PER_WINDOW_CAP,
                    help=f"max claims per NYT episode, matched to the LOC "
                         f"episodes' own 87-212 scale (default {PER_WINDOW_CAP}); "
                         f"0 or negative disables capping")
    args = ap.parse_args()
    cap = args.cap if args.cap and args.cap > 0 else None

    existing_rows, seen_urls, max_id, episode_counts = load_existing()
    new_rows, per_window = nyt_rows(seen_urls, max_id + 1, episode_counts, cap=cap)

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
