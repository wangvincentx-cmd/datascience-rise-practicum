"""
Merge a newspaper_scraper.py subset run (scraped with --out/--log-out pointed
at separate files, NOT claims_raw.csv/search_log.csv directly -- see that
script's --log-out help text for why) into the main corpus.

Same idea as append_nyt_claims.py, simpler: the subset file is already in
claims_raw.csv's exact schema (both come from newspaper_scraper.py), so no
per-source field mapping is needed -- just dedupe on page_url, continue the
claim_id sequence, and append. Also appends the subset's search_log rows into
the master search_log.csv (pure log data, no IDs to collide, just concat).

Idempotent: safe to rerun -- rows whose page_url is already in claims_raw.csv
are skipped.

Usage:
    python append_loc_claims.py --claims claims_raw_newloc.csv --log search_log_newloc.csv
    python append_loc_claims.py --claims claims_raw_newloc.csv --log search_log_newloc.csv --dry-run
"""

import argparse
import csv
from pathlib import Path

CLAIMS_RAW = Path("claims_raw.csv")
SEARCH_LOG = Path("search_log.csv")
FIELDS = ["claim_id", "episode", "kind", "publisher", "state", "date",
          "search_term", "page_url", "quote"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True, help="subset claims CSV to merge in")
    ap.add_argument("--log", default=None, help="subset search_log CSV to append")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    existing_rows, seen_urls, max_id = [], set(), 0
    episode_counts = {}
    if CLAIMS_RAW.exists():
        with open(CLAIMS_RAW, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)
                seen_urls.add(row["page_url"])
                max_id = max(max_id, int(row["claim_id"]))
                episode_counts[row["episode"]] = episode_counts.get(row["episode"], 0) + 1

    new_rows = []
    next_id = max_id + 1
    with open(args.claims, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["page_url"] in seen_urls:
                continue
            seen_urls.add(row["page_url"])
            row = dict(row)
            row["claim_id"] = next_id
            next_id += 1
            new_rows.append(row)
            episode_counts[row["episode"]] = episode_counts.get(row["episode"], 0) + 1

    print(f"claims_raw.csv currently has {len(existing_rows)} rows (max claim_id {max_id})")
    by_ep = {}
    for row in new_rows:
        by_ep[row["episode"]] = by_ep.get(row["episode"], 0) + 1
    for ep, n in sorted(by_ep.items()):
        print(f"  {ep}: {n} new claims (episode total after merge: {episode_counts[ep]})")
    print(f"total new claims to append: {len(new_rows)}")

    if args.dry_run:
        print("(dry run -- nothing written)")
        return
    if not new_rows:
        print("nothing to append")
        return

    with open(CLAIMS_RAW, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        for row in new_rows:
            w.writerow(row)
    print(f"appended {len(new_rows)} rows -> {CLAIMS_RAW} "
          f"(now {len(existing_rows) + len(new_rows)} total; "
          f"rerun grade_claims.py to grade only the new ones)")

    if args.log:
        log_path = Path(args.log)
        if log_path.exists():
            with open(log_path, newline="", encoding="utf-8") as f:
                log_rows = list(csv.reader(f))[1:]   # skip header
            with open(SEARCH_LOG, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(log_rows)
            print(f"appended {len(log_rows)} rows -> {SEARCH_LOG}")


if __name__ == "__main__":
    main()
