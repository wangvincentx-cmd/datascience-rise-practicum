"""
Score any claim EXTRACTOR against the gold pages.

The existing kappa harness (../handgrade_newspapers/kappa.py) scores LABELS: it
joins on claim_id, so it can only compare graders on claims some extractor
already found. It cannot see a missed claim, so it cannot measure recall. This
script scores the extraction step itself, which means:

  precision  of what the extractor returned, how much is really a prediction
  recall     of the predictions actually on the page, how many were found
  F1         the harmonic mean

and, on the pairs that matched, per-field agreement (Cohen's kappa) so a
extractor that finds the right sentences but mislabels their direction does not
get credit for it.

Matching is fuzzy, because OCR text cannot be compared literally: spans are
lowercased, stripped to alphanumerics, and compared by CONTAINMENT
(|A and B| / min(|A|,|B|)) at a default threshold of 0.7. Containment rather
than Jaccard because gold spans are whole sentences while a good extractor
often returns the tighter clause carrying the prediction -- see overlap_score.
Matching is one-to-one and greedy from the best pair down, so one long
extracted span cannot claim credit for three gold claims.

Usage (from JeremysShit/):
    python gold_extraction/eval_extraction.py --pred regex_baseline.jsonl --name regex
    python gold_extraction/eval_extraction.py --pred pred_llama70b.jsonl --name llama-3.3-70b

Prediction file: JSONL, one claim per line, with at least `page_id` and `quote`.
Any of the gold label fields that are present are also scored.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grade_claims import cohens_kappa  # noqa: E402

LABEL_FIELDS = ["topic", "direction", "horizon_months", "confidence", "voice"]
TOKEN_RE = re.compile(r"[a-z0-9]+")


def norm_tokens(text):
    """Tokens for fuzzy matching. OCR noise means literal comparison is useless:
    'corporation's mille' vs 'corporations mills' must still match, so case,
    punctuation and hyphen breaks are all discarded."""
    return set(TOKEN_RE.findall((text or "").lower()))


MIN_SPAN_TOKENS = 4


def overlap_score(a, b):
    """Containment, not Jaccard: |A and B| / min(|A|, |B|).

    Jaccard over the UNION was the first thing tried here and it was wrong. Gold
    spans are whole sentences including the attribution clause; a good extractor
    often returns the tighter clause that carries the prediction. Scoring
    'the bottom was close at hand' (6 tokens) against the 24-token gold sentence
    that contains it gives Jaccard 0.25 -- so the same prediction was booked as
    BOTH a miss and a false positive, and every model was penalised twice for
    being more precise than the gold. Containment asks the question that
    actually matters: did the two spans identify the same prediction?

    One prediction still cannot claim several gold claims, because matching
    below is greedy and one-to-one."""
    if len(a) < MIN_SPAN_TOKENS or len(b) < MIN_SPAN_TOKENS:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def match_page(gold_claims, pred_claims, threshold):
    """Greedy one-to-one matching, best pair first.

    Greedy-from-best matters: a lazy extractor that returns one enormous span
    covering a whole column would otherwise match several gold claims at once
    and score inflated recall. One prediction consumes at most one gold claim."""
    gold_tok = [norm_tokens(g["quote"]) for g in gold_claims]
    pred_tok = [norm_tokens(p.get("quote", "")) for p in pred_claims]
    scored = []
    for gi, gt in enumerate(gold_tok):
        for pi, pt in enumerate(pred_tok):
            s = overlap_score(gt, pt)
            if s >= threshold:
                scored.append((s, gi, pi))
    scored.sort(reverse=True)
    used_g, used_p, pairs = set(), set(), []
    for s, gi, pi in scored:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi, s))
    missed = [gi for gi in range(len(gold_claims)) if gi not in used_g]
    spurious = [pi for pi in range(len(pred_claims)) if pi not in used_p]
    return pairs, missed, spurious


def evaluate(gold_path, pred_path, threshold=0.7, name="extractor", verbose=False):
    gold_pages = [json.loads(l) for l in open(gold_path, encoding="utf-8") if l.strip()]
    preds = [json.loads(l) for l in open(pred_path, encoding="utf-8") if l.strip()]
    by_page = {}
    for p in preds:
        by_page.setdefault(p.get("page_id"), []).append(p)

    tp = fp = fn = 0
    field_pairs = {f: [] for f in LABEL_FIELDS}
    per_page = []
    unknown_pages = set(by_page) - {g["page_id"] for g in gold_pages}

    for page in gold_pages:
        gold_claims = page["claims"]
        pred_claims = by_page.get(page["page_id"], [])
        pairs, missed, spurious = match_page(gold_claims, pred_claims, threshold)
        tp += len(pairs)
        fn += len(missed)
        fp += len(spurious)
        for gi, pi, _ in pairs:
            for f in LABEL_FIELDS:
                gv, pv = gold_claims[gi].get(f), pred_claims[pi].get(f)
                if gv is not None and pv is not None:
                    field_pairs[f].append((str(gv), str(pv)))
        per_page.append({
            "page_index": page["page_index"], "date": page["date"],
            "n_gold": len(gold_claims), "n_pred": len(pred_claims),
            "tp": len(pairs), "fn": len(missed), "fp": len(spurious),
            "missed": [gold_claims[gi]["quote"] for gi in missed],
            "spurious": [pred_claims[pi].get("quote", "") for pi in spurious],
        })

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n=== EXTRACTION: {name} ===")
    print(f"  matched (TP)      {tp}")
    print(f"  spurious (FP)     {fp}")
    print(f"  missed (FN)       {fn}")
    print(f"  precision         {precision:.3f}")
    print(f"  recall            {recall:.3f}")
    print(f"  F1                {f1:.3f}")
    if unknown_pages:
        print(f"  ! {len(unknown_pages)} predicted page_id(s) not in the gold set (ignored)")

    zero = [p for p in per_page if p["n_gold"] == 0]
    if zero:
        z_fp = sum(p["fp"] for p in zero)
        print(f"\n  zero-prediction pages ({len(zero)}): {z_fp} false positives")
        print("  (pages with no real forecast on them -- pure precision test)")

    scored_fields = {f: v for f, v in field_pairs.items() if v}
    if scored_fields:
        print("\n  label agreement on matched claims (Cohen's kappa):")
        for f, pairs_ in scored_fields.items():
            agree = sum(1 for a, b in pairs_ if a == b) / len(pairs_)
            print(f"    {f:16s} kappa = {cohens_kappa(pairs_):+.2f}   "
                  f"raw = {agree:.0%}   n = {len(pairs_)}")
    else:
        print("\n  (no label fields in the prediction file -- extraction-only scoring)")

    print("\n  per page:")
    for p in per_page:
        flag = "  <- zero-claim page" if p["n_gold"] == 0 else ""
        print(f"    p{p['page_index']:<3}{p['date']}  gold={p['n_gold']:<3}"
              f"pred={p['n_pred']:<3}tp={p['tp']:<3}fn={p['fn']:<3}fp={p['fp']:<3}{flag}")

    if verbose:
        print("\n  MISSED (false negatives):")
        for p in per_page:
            for q in p["missed"]:
                print(f"    [p{p['page_index']}] {q[:150]}")
        print("\n  SPURIOUS (false positives):")
        for p in per_page:
            for q in p["spurious"]:
                print(f"    [p{p['page_index']}] {q[:150]}")

    return {"name": name, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="gold_extraction/gold_claims.jsonl")
    ap.add_argument("--pred", required=True, help="JSONL with page_id + quote per line")
    ap.add_argument("--name", default=None)
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="containment threshold for a match (default 0.7)")
    ap.add_argument("--verbose", action="store_true",
                    help="list every missed and spurious span")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    result = evaluate(args.gold, args.pred, args.threshold,
                      args.name or Path(args.pred).stem, args.verbose)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nsummary -> {args.json_out}")
