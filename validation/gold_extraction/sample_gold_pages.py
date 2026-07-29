"""
Draw the gold-standard page sample for evaluating claim EXTRACTION.

The existing 80-claim gold (handgrade_newspapers/) validates LABELING: given a
sentence the regex already pulled out, does the model grade it the way a human
would? It cannot see the sentences the regex never pulled out, so it measures
precision and is structurally blind to recall. That blind spot is the whole
problem -- a one-page spot check found 7 real predictions where the regex found
5, with zero overlap (CHANGELOG.md, 2026-07-22).

Measuring recall requires the unit of analysis to be the PAGE, not the claim:
annotate every prediction on a page, then ask what each extractor found. A page
with no predictions on it is a gold record too -- that is what makes false
positives countable.

Sampling rules, and why:
  - UNIFORM at random within each stratum. Not "pages that look economic" --
    biasing toward promising pages would inflate every extractor's apparent
    precision at once and make the numbers meaningless.
  - STRATIFIED BY ERA, not by crisis/calm. The cache holds only crisis-window
    pages (the 1905/1925/1955 control windows were never cached), so a
    crisis/calm split is not available. Era is the axis that actually moves
    extraction difficulty anyway: OCR quality, column layout, and house style
    change far more between 1907 and 1957 than between a panic and a calm year.
  - NO LENGTH CAP. Long pages are harder and are where a sentence-window regex
    should struggle most; excluding them would flatter the baseline.

Usage (from JeremysShit/):
    python gold_extraction/sample_gold_pages.py
    python gold_extraction/sample_gold_pages.py --n 24 --seed 20260722

Output: gold_extraction/gold_pages.jsonl -- the sampled pages with full text,
ready to be annotated into gold_claims.jsonl.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# Era buckets. Boundaries are drawn at genuine breaks in the newspaper record
# (WWI, the Depression, WWII), not at even intervals.
ERAS = [
    ("1900s-10s", "1900-01-01", "1919-12-31"),
    ("1920s", "1920-01-01", "1929-12-31"),
    ("1930s", "1930-01-01", "1939-12-31"),
    ("1940s-50s", "1940-01-01", "1963-12-31"),
]


def era_for(date):
    for name, start, end in ERAS:
        if date and start <= date <= end:
            return name
    return None


def sample(pages_path="data/pages.jsonl", n=16, seed=20260722):
    pages = [json.loads(line) for line in open(pages_path, encoding="utf-8")]
    by_era = defaultdict(list)
    for p in pages:
        era = era_for(p.get("date"))
        if era:
            by_era[era].append(p)
    for era in by_era:
        by_era[era].sort(key=lambda p: p["page_id"])  # deterministic pre-shuffle order

    rng = random.Random(seed)
    eras = [e for e, _, _ in ERAS if by_era[e]]
    # Round-robin across eras so the sample is balanced even when n is not a
    # multiple of the era count, and so a large era cannot dominate.
    picked, seen = [], set()
    while len(picked) < n:
        progressed = False
        for era in eras:
            if len(picked) >= n:
                break
            pool = [p for p in by_era[era] if p["page_id"] not in seen]
            if not pool:
                continue
            choice = rng.choice(pool)
            seen.add(choice["page_id"])
            picked.append({**choice, "gold_era": era})
            progressed = True
        if not progressed:
            break
    picked.sort(key=lambda p: (p.get("date") or "", p["page_id"]))
    return picked


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", default="data/pages.jsonl")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--out", default="gold_extraction/gold_pages.jsonl")
    args = ap.parse_args()

    picked = sample(args.pages, args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for p in picked:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"{len(picked)} pages -> {out}  (seed {args.seed})")
    total = 0
    for p in picked:
        total += p["n_chars"]
        print(f"  {p['date']}  {p['gold_era']:<10} {p['n_chars']:>6,} chars  "
              f"{(p['publisher'] or '')[:52]}")
    print(f"  total {total:,} chars to annotate")
