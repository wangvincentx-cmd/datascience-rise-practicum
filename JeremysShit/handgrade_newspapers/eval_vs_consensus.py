"""
Evaluate the LLM grader against the three-coder CONSENSUS gold standard.

The 80 validation claims were labeled by three coders to consensus (not
independently), so there is no human-human kappa to report — the reported
metric is LLM-vs-consensus agreement, which is a legitimate validation of the
automated labels against a human gold standard.

Two honest adjustments, both applied by an OBJECTIVE rule (not by looking at
which claims the model got right):

  1. Ungradeable rows are dropped from BOTH sides before scoring. A row is
     ungradeable if its quote is raw extraction noise, not a sentence:
       - contains raw NDNP markup ('{"/service/' JSON blobs), OR
       - is <55% alphabetic-word characters (OCR mush).
     Neither a human nor a model can code these; they only add coin-flip noise
     and depress kappa. Removing them is data hygiene, not cherry-picking.

  2. Kappa is reported per field, with n and raw agreement, so a strong field
     (topic) is not hidden behind a weak one (confidence, which the README
     already treats as an unscored limitation).

Usage (from JeremysShit/):
    python handgrade_newspapers/eval_vs_consensus.py \
        --gold handgrade_newspapers/handgrade_vincent.csv \
        --graded claims_graded_val80_deleaked.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grade_claims import cohens_kappa  # noqa: E402

WORD = re.compile(r"[A-Za-z]")


def ungradeable(quote):
    q = quote.strip()
    if not q:
        return True, "empty"
    if '{"/service/' in q or '"full_text"' in q:
        return True, "raw NDNP markup"
    alpha = sum(1 for c in q if WORD.match(c))
    if alpha / max(len(q), 1) < 0.55:
        return True, "OCR mush (<55% letters)"
    return False, ""


def load(path, prefix):
    return {r["claim_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}


def kappa_line(gold, llm, ids, field):
    pairs = [(gold[i]["human_" + field].strip().lower(), llm[i][field].strip().lower())
             for i in ids
             if gold[i].get("human_" + field, "").strip() and llm[i].get(field, "").strip()]
    if not pairs:
        return f"  {field:15s} (no data)"
    n = len(pairs)
    raw = sum(a == b for a, b in pairs) / n
    return f"  {field:15s} kappa = {cohens_kappa(pairs):+.2f}   raw {raw:.0%}   n={n}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--graded", required=True)
    args = ap.parse_args()

    gold = load(args.gold, "human_")
    llm = load(args.graded, "")
    shared = [i for i in gold if i in llm and llm[i].get("is_prediction", "").strip()]

    dropped = [(i, ungradeable(gold[i]["quote"])[1]) for i in shared
               if ungradeable(gold[i]["quote"])[0]]
    kept = [i for i in shared if not ungradeable(gold[i]["quote"])[0]]

    print(f"{len(shared)} claims graded by both | {len(dropped)} dropped as ungradeable "
          f"| {len(kept)} scored")
    for i, why in dropped:
        print(f"    drop [{i}] {why}")

    # is_prediction on all gradeable rows; the other three only where BOTH the
    # gold and the model call it a prediction (a conditional field is undefined
    # otherwise).
    print("\n=== RAW (all graded claims) ===")
    print(kappa_line(gold, llm, shared, "is_prediction"))
    both = [i for i in shared
            if gold[i]["human_is_prediction"].strip().lower() == "yes"
            and llm[i]["is_prediction"].strip().lower() == "yes"]
    for f in ["topic", "direction", "confidence"]:
        print(kappa_line(gold, llm, both, f))

    print("\n=== CLEANED (ungradeable rows removed) ===")
    print(kappa_line(gold, llm, kept, "is_prediction"))
    both_k = [i for i in kept
              if gold[i]["human_is_prediction"].strip().lower() == "yes"
              and llm[i]["is_prediction"].strip().lower() == "yes"]
    for f in ["topic", "direction", "confidence"]:
        print(kappa_line(gold, llm, both_k, f))

    print("\nRemaining is_prediction disagreements (cleaned set):")
    for i in kept:
        g = gold[i]["human_is_prediction"].strip().lower()
        m = llm[i]["is_prediction"].strip().lower()
        if g != m:
            print(f"  [{i}] gold={g:3s} llm={m:3s} | {gold[i]['quote'][:72]}")


if __name__ == "__main__":
    main()
