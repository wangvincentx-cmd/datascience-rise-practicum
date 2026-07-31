"""
Progress across the year-sharded ProQuest economy corpus.

`extraction_status.py` prints one row per window, which is right for 9 windows
and unreadable for 110 year shards. This is the rollup: one line per decade plus
a total, and -- the number that actually matters on a corpus this size -- how
many pages are left and how many daily quota runs that implies.

Counting matches extract_gpt.py exactly, or the "left" column lies:
  * a page is DONE if a record for it exists at the CURRENT schema version,
    including a `no_predictions` record (the call happened; it found nothing)
  * records at an older schema version are NOT done -- extract_gpt will redo them
  * a CLAIM is a record that is not `no_predictions`

Runs on the Mac or in the VM; reads only file counts, never article text.

Usage:
  python corpus_progress.py
  python corpus_progress.py --rate 3000      # pages/day you actually observe
  python corpus_progress.py --by-year        # every shard, not decade rollups
  python corpus_progress.py --left           # just the integer: pages left to extract

`--left` is the phase gate run_corpus_economy.sh reads. Verification only starts
once every page has been extracted, and that decision has to be made from what is
on disk (a run interrupted yesterday must not look finished), so both the script
and the human ask the same counter.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

SCHEMA_VERSION = 2          # must track extract_gpt.SCHEMA_VERSION
SHARD_RE = re.compile(r"^(?:pred_)?proquest_economy_(\d{4})\.jsonl$")


def shard_year(path):
    """The year a corpus shard holds, or None if the file is not one.

    Filters out the old per-window files (…_gulf_1990.jsonl) and the .export /
    .dropped side-files, none of which are units of work.
    """
    match = SHARD_RE.match(Path(path).name)
    return match.group(1) if match else None


def count_raw(raw_dir):
    """year -> articles parsed."""
    counts = {}
    for path in sorted(Path(raw_dir).glob("*.jsonl")):
        year = shard_year(path)
        if year:
            with open(path) as f:
                counts[year] = sum(1 for line in f if line.strip())
    return counts


def count_preds(pred_dir):
    """year -> (pages done at current schema, claims, stale pages)."""
    counts = {}
    for path in sorted(Path(pred_dir).glob("*.jsonl")):
        year = shard_year(path)
        if not year:
            continue
        done, claims, stale = set(), 0, set()
        with open(path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                page_id = record.get("page_id")
                if page_id is None:
                    continue
                if record.get("schema_version") != SCHEMA_VERSION:
                    stale.add(page_id)
                    continue
                done.add(page_id)
                claims += not record.get("no_predictions")
        counts[year] = (len(done), claims, len(stale))
    return counts


def rollup(raw, preds, verified, by_year):
    """[(label, parsed, done, claims, kept, stale)] plus a TOTAL row."""
    years = sorted(set(raw) | set(preds) | set(verified))
    groups = defaultdict(list)
    for year in years:
        groups[year if by_year else f"{year[:3]}0s"].append(year)

    rows, totals = [], [0, 0, 0, 0, 0]
    for label in sorted(groups):
        values = [0, 0, 0, 0, 0]
        for year in groups[label]:
            values[0] += raw.get(year, 0)
            values[1] += preds.get(year, (0, 0, 0))[0]
            values[2] += preds.get(year, (0, 0, 0))[1]
            values[3] += verified.get(year, (0, 0, 0))[1]
            values[4] += preds.get(year, (0, 0, 0))[2]
        rows.append((label, *values))
        totals = [t + v for t, v in zip(totals, values)]
    rows.append(("TOTAL", *totals))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".", help="repo root (or unpacked export)")
    ap.add_argument("--rate", type=int, default=3000,
                    help="pages/day the quota actually buys, for the estimate")
    ap.add_argument("--by-year", action="store_true",
                    help="one row per year instead of per decade")
    ap.add_argument("--left", action="store_true",
                    help="print only the number of pages left to extract, for scripts")
    args = ap.parse_args()

    base = Path(args.base)
    raw = count_raw(base / "data/raw")
    preds = count_preds(base / "data/predictions")
    verified = count_preds(base / "data/verified")
    if not raw and not preds:
        if args.left:
            # No shards is not zero work left; it is an unparsed corpus. Saying
            # "0" here would tell the runner to start verifying nothing.
            raise SystemExit("no corpus shards in data/raw")
        raise SystemExit(
            "No corpus shards found. Parse the dataset first:\n"
            "  python tdm_parse.py --arm economy --corpus --dataset-dir <folder>")

    rows = rollup(raw, preds, verified, args.by_year)
    if args.left:
        _, parsed, done, _, _, _ = rows[-1]
        print(max(parsed - done, 0))
        return

    header = f"{'':>8}  {'parsed':>9} {'done':>9} {'claims':>9} {'verified':>9} {'stale':>7}"
    print(header)
    print("-" * len(header))
    for label, parsed, done, claims, kept, stale in rows:
        if label == "TOTAL":
            print("-" * len(header))
        print(f"{label:>8}  {parsed:>9,} {done:>9,} {claims:>9,} {kept:>9,} "
              f"{stale:>7,}")

    _, parsed, done, claims, kept, stale = rows[-1]
    left = parsed - done
    pct = 100 * done / parsed if parsed else 0
    print(f"\n{done:,}/{parsed:,} pages extracted ({pct:.1f}%), {left:,} left")
    if left > 0:
        print(f"  at {args.rate:,} pages/day that is "
              f"~{-(-left // args.rate)} more daily runs")
    if claims:
        print(f"  {claims:,} claims, {claims / max(done, 1):.2f} per page")
    if kept:
        print(f"  {kept:,} survive verification "
              f"({100 * kept / claims:.0f}% of claims kept)")
    elif claims:
        print("  verification has not run yet (data/verified is empty)")

    # Which phase run_corpus_economy.sh will do next. It gates verification on
    # left == 0 (see --left above), so say that here rather than leaving it to be
    # inferred from two numbers. Verification progress is counted in YEARS, not
    # claims: kept < claims permanently once the filter has dropped anything, so
    # comparing those two can never signal "finished".
    if left > 0:
        print("\n  phase: EXTRACT. Verification starts once this reaches 0.")
    else:
        with_claims = {y for y, v in preds.items() if v[1]}
        started = {y for y, v in verified.items() if v[0] or v[1]}
        todo = sorted(with_claims - started)
        if todo:
            print(f"\n  phase: VERIFY. Every page is extracted; "
                  f"{len(started)}/{len(with_claims)} year(s) verified, "
                  f"{len(todo)} to go\n  (next: {todo[0]}). Note a year is "
                  f"counted once it has output, so the one that was mid-run "
                  f"when\n  the quota stopped is already counted -- verify_gpt.py "
                  f"resumes inside it.")
        elif with_claims:
            print("\n  phase: DONE extracting and verifying. Score it:\n"
                  "    python analyze_economy.py --set verified")

    if stale:
        print(f"\n  *** {stale:,} pages sit at an older schema version and will be")
        print(f"  *** re-extracted, costing quota. See extract_gpt.py's check_schema.")


if __name__ == "__main__":
    main()
