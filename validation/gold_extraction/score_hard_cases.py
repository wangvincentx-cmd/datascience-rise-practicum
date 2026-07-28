"""
Score an extractor on the HARD JUDGEMENT CASES, separately from headline F1.

Aggregate precision/recall hides the decisions that actually distinguish a
model that reasons about the include/exclude boundary from one that pattern-
matches economic vocabulary. Those decisions are enumerated in
hard_cases.jsonl: 44 spans taken from the gold pages, each with the verdict the
protocol requires and why.

Each case is one span plus an expected verdict:
  expect="exclude"  the span is on the page and looks like a forecast, but is an
                    advertisement, fiction, a reprint, a refusal, a conditional,
                    a present-tense report, or a post-mortem on a forecast that
                    had already failed. Returning it is an error.
  expect="include"  the span IS a forecast, but an awkward one -- buried in an
                    unrelated story, phrased in the present tense, negating the
                    very phrase a keyword search looks for, or reporting someone
                    else's live forecast sceptically. Missing it is an error.

The pairs matter more than the totals. h01/h40 are the same construction --
the paper reporting a forecast it doubts -- and differ only in whether the
forecast had already been falsified when the page went to press. An extractor
that gets both right is reading; one that gets both the same way is not.

Usage (from JeremysShit/):
    python gold_extraction/score_hard_cases.py --pred gold_extraction/pred_llama70b.jsonl \\
        --name llama-3.3-70b
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")


def toks(text):
    return TOKEN_RE.findall((text or "").lower())


def contains_probe(pred_quote, probe, threshold=0.7):
    """Did this predicted claim capture the probe span?

    Asymmetric on purpose: the score is the fraction of the PROBE's tokens
    present in the prediction, so an extractor returning a longer span that
    contains the probe still counts as having taken it. A model cannot dodge a
    false positive by padding the quote."""
    p, q = toks(probe), set(toks(pred_quote))
    if not p:
        return False
    return sum(1 for t in p if t in q) / len(p) >= threshold


def validate_probes(cases, pages_path):
    """Every probe must actually appear on its page -- otherwise a transcription
    slip would silently make an exclude-case unfailable."""
    pages = [json.loads(l) for l in open(pages_path, encoding="utf-8") if l.strip()]
    by_index = {i: p for i, p in enumerate(pages)}
    bad = []
    for c in cases:
        page = by_index.get(c["page_index"])
        if page is None:
            bad.append((c["case_id"], "no such page"))
            continue
        page_tokens = " ".join(toks(page["ocr_text"]))
        if " ".join(toks(c["probe"])) not in page_tokens:
            present = sum(1 for t in toks(c["probe"]) if t in set(toks(page["ocr_text"])))
            bad.append((c["case_id"],
                        f"probe not contiguous on page ({present}/{len(toks(c['probe']))} tokens present)"))
    return bad, by_index


def score(cases, preds, by_index, name):
    page_id_of = {i: p["page_id"] for i, p in by_index.items()}
    by_page = defaultdict(list)
    for p in preds:
        by_page[p.get("page_id")].append(p.get("quote", ""))

    rows, by_cat = [], defaultdict(lambda: [0, 0])
    for c in cases:
        pid = page_id_of.get(c["page_index"])
        taken = any(contains_probe(q, c["probe"]) for q in by_page.get(pid, []))
        correct = (taken and c["expect"] == "include") or \
                  (not taken and c["expect"] == "exclude")
        rows.append({**c, "taken": taken, "correct": correct})
        by_cat[c["category"]][0] += int(correct)
        by_cat[c["category"]][1] += 1

    inc = [r for r in rows if r["expect"] == "include"]
    exc = [r for r in rows if r["expect"] == "exclude"]
    n_ok = sum(r["correct"] for r in rows)

    print(f"\n=== HARD CASES: {name} ===")
    print(f"  overall            {n_ok}/{len(rows)} = {n_ok/len(rows):.0%}")
    print(f"  include-side       {sum(r['correct'] for r in inc)}/{len(inc)}"
          f"  (real forecasts it must find)")
    print(f"  exclude-side       {sum(r['correct'] for r in exc)}/{len(exc)}"
          f"  (traps it must refuse)")

    print("\n  by category:")
    for cat, (ok, n) in sorted(by_cat.items(), key=lambda kv: (kv[1][0] / kv[1][1], kv[0])):
        print(f"    {cat:32s} {ok}/{n}")

    print("\n  errors:")
    for r in rows:
        if not r["correct"]:
            verb = "TOOK a trap" if r["expect"] == "exclude" else "MISSED a real forecast"
            print(f"    {r['case_id']} [{r['category']}] {verb}")
            print(f"        \"{r['probe'][:110]}\"")

    pair = {r["case_id"]: r["correct"] for r in rows if r["case_id"] in ("h01", "h40")}
    if len(pair) == 2:
        both = all(pair.values())
        print(f"\n  h01/h40 discrimination pair: "
              f"{'BOTH CORRECT -- reads the frame' if both else 'failed'}"
              f"  (h01 {'ok' if pair['h01'] else 'X'}, h40 {'ok' if pair['h40'] else 'X'})")

    return {"name": name, "overall": n_ok / len(rows), "n": len(rows),
            "include_ok": sum(r["correct"] for r in inc), "include_n": len(inc),
            "exclude_ok": sum(r["correct"] for r in exc), "exclude_n": len(exc)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="gold_extraction/hard_cases.jsonl")
    ap.add_argument("--pages", default="gold_extraction/gold_pages.jsonl")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
    bad, by_index = validate_probes(cases, args.pages)
    if bad:
        print(f"PROBE VALIDATION FAILED ({len(bad)}):")
        for cid, why in bad:
            print(f"  {cid}: {why}")
        raise SystemExit(1)
    print(f"all {len(cases)} probes verified present on their pages")
    if args.validate_only:
        raise SystemExit(0)

    preds = [json.loads(l) for l in open(args.pred, encoding="utf-8") if l.strip()]
    result = score(cases, preds, by_index, args.name or Path(args.pred).stem)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
