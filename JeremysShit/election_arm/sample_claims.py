"""
Print a random sample of extracted claims so you can eyeball extraction quality
without opening the raw JSONL. A quick manual precision check -- the informal
version of what validate_kappa.py measures rigorously.

Reads data/predictions/pred_{source}_economy_{window}.jsonl (skips the
text-stripped *.export.jsonl copies and no_predictions markers).

Usage (in the workbench):
  python sample_claims.py                          # 10 claims, any proquest window
  python sample_claims.py --window gfc_2008        # only that window
  python sample_claims.py --n 25                   # more samples
  python sample_claims.py --source nyt             # a different source
"""

import argparse
import glob
import json
import random
import textwrap


def load_claims(source, window):
    win = window or "*"
    rows = []
    for path in glob.glob(f"data/predictions/pred_{source}_economy_{win}.jsonl"):
        if path.endswith(".export.jsonl"):
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("no_predictions") or not r.get("claim_text"):
                    continue
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="proquest")
    ap.add_argument("--window", help="one window_id, e.g. gfc_2008; default all")
    ap.add_argument("--n", type=int, default=10, help="how many to sample")
    ap.add_argument("--seed", type=int, default=None, help="fix for a reproducible sample")
    args = ap.parse_args()

    claims = load_claims(args.source, args.window)
    if not claims:
        raise SystemExit("No claims found. Check --source/--window and that "
                         "extraction has produced pred_*.jsonl.")
    if args.seed is not None:
        random.seed(args.seed)
    sample = random.sample(claims, min(args.n, len(claims)))

    print(f"{len(claims)} claims available; showing {len(sample)}:\n")
    for i, c in enumerate(sample, 1):
        head = (f"[{i}] {c.get('window')}  {c.get('date')}  "
                f"{c.get('newspaper_title')}")
        meta = (f"    -> {c.get('predicted_direction')} / "
                f"{c.get('predicted_state_at_horizon')} "
                f"@ {c.get('horizon_months')}mo  "
                f"voice={c.get('voice')} hedged={c.get('hedged')}")
        body = textwrap.fill(c.get("claim_text", ""), width=88,
                             initial_indent="    ", subsequent_indent="    ")
        print(head)
        print(body)
        print(meta + "\n")


if __name__ == "__main__":
    main()
