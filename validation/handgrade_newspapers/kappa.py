"""
Compute inter-coder reliability for the hand-graded validation sample.

Two numbers matter, and they answer different questions:

  human vs human   Is the RUBRIC coherent? If two people who read the same
                   instructions can't agree, no LLM can rescue it. Fix the
                   rubric before touching the model.
  human vs LLM     Can we trust the LLM's labels on the ~1,240 claims nobody
                   checked by hand? This is the number that goes in the paper.

Usage (from JeremysShit/):
    python handgrade_newspapers/kappa.py \
        --graders handgrade_newspapers/handgrade_vincent.csv \
                  handgrade_newspapers/handgrade_jeremy.csv

--graded defaults to this arm's claims_graded.csv. Passing one grader file
skips the human-vs-human comparison.

Fields with a dependency (topic, direction, confidence only mean something for
a real prediction) are scored only on the rows BOTH coders called a prediction.
Scoring them on rows one coder rejected would measure the is_prediction
disagreement twice.
"""

import argparse
import csv
import sys
from pathlib import Path

ECONOMY_ARM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ECONOMY_ARM))
from grade_claims import cohens_kappa  # noqa: E402

# voice is extracted by the LLM and used as a model feature, but is NOT
# hand-graded — so it has no human counterpart to compare against and stays
# out of the kappa report. It is therefore an unvalidated feature: say so if
# it turns up as important in the model.
CONDITIONAL = ["topic", "direction", "confidence"]


def load(path, prefix="human_"):
    with open(path, encoding="utf-8") as f:
        return {r["claim_id"]: {k[len(prefix):] if k.startswith(prefix) else k: v.strip().lower()
                                for k, v in r.items()} for r in csv.DictReader(f)}


def compare(a, b, name_a, name_b):
    ids = sorted(set(a) & set(b), key=int)
    if not ids:
        print(f"  no overlapping claim_ids between {name_a} and {name_b}")
        return
    print(f"\n{name_a} vs {name_b}  (n={len(ids)} shared claims)")

    pairs = [(a[i]["is_prediction"], b[i]["is_prediction"]) for i in ids
             if a[i].get("is_prediction") and b[i].get("is_prediction")]
    if pairs:
        agree = sum(1 for x, y in pairs if x == y) / len(pairs)
        print(f"  {'is_prediction':15s} kappa = {cohens_kappa(pairs):+.2f}   "
              f"raw agreement {agree:.0%}  (n={len(pairs)})")

    both_yes = [i for i in ids
                if a[i].get("is_prediction") == "yes" and b[i].get("is_prediction") == "yes"]
    for field in CONDITIONAL:
        pairs = [(a[i][field], b[i][field]) for i in both_yes
                 if a[i].get(field) and b[i].get(field)]
        if not pairs:
            print(f"  {field:15s} (no rows both coders called predictions)")
            continue
        agree = sum(1 for x, y in pairs if x == y) / len(pairs)
        print(f"  {field:15s} kappa = {cohens_kappa(pairs):+.2f}   "
              f"raw agreement {agree:.0%}  (n={len(pairs)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graders", nargs="+", required=True,
                    help="one or two filled handgrade_*.csv files")
    ap.add_argument("--graded", default=str(ECONOMY_ARM / "claims_graded.csv"),
                    help="the LLM's output, for the human-vs-LLM comparison")
    args = ap.parse_args()

    graders = [(Path(p).stem, load(p)) for p in args.graders]
    for name, g in graders:
        filled = sum(1 for r in g.values() if r.get("is_prediction"))
        print(f"{name}: {filled}/{len(g)} rows filled in")
        if filled < len(g):
            print(f"  WARNING: {len(g) - filled} blank rows will be skipped")

    if len(graders) >= 2:
        print("\n" + "=" * 62)
        print("HUMAN vs HUMAN — is the rubric coherent?")
        print("=" * 62)
        compare(graders[0][1], graders[1][1], graders[0][0], graders[1][0])

    if Path(args.graded).exists():
        llm = load(args.graded, prefix="")
        print("\n" + "=" * 62)
        print("HUMAN vs LLM — can we trust the ungraded remainder?")
        print("=" * 62)
        for name, g in graders:
            compare(g, llm, name, "llm")
    else:
        print(f"\n(skipping human-vs-LLM: {args.graded} not found — run grade_claims.py first)")

    print("\nBenchmarks: kappa >= 0.6 substantial, >= 0.8 near-perfect.")
    print("Below 0.6 on direction, tighten the rubric wording and regrade — do not")
    print("proceed to scoring on labels the coders themselves cannot reproduce.")


if __name__ == "__main__":
    main()
