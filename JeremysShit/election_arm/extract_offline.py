"""
Offline, no-network extraction of economy-arm predictions from newspaper text.

Drop-in stand-in for extract_predictions.py when the runtime cannot reach the
Claude API -- specifically the ProQuest TDM Studio workbench, whose VM is sealed
off from the internet. Same input/output contract as extract_predictions.py, so
analyze_economy.py reads the result unchanged (it globs pred_*_economy_*.jsonl).

Method: rule-based. Sentence-split ocr_text, keep sentences that pair an economic
DIRECTION word (recession / recovery / ...) with a forward-looking CUE (will,
expected, coming, ahead, ...), map the direction to a recession/expansion call,
flag hedging, and read a horizon in months. This is deliberately transparent and
free, but its recall/precision are lower than the LLM extractor. Validate it with
validate_kappa.py and report the gap; disclose it as the offline-source caveat,
exactly as the README says to disclose the LOC/NYT boundary.

Economy arm only. Elections offline extraction (identifying the predicted winner
and scope) needs its own logic and is not implemented here.

Input:  data/raw/{source}_economy_{window}.jsonl
Output: data/predictions/pred_{source}_economy_{window}.jsonl

Usage (in the TDM Studio workbench):
  python extract_offline.py --source proquest --window gfc_2008
  python extract_offline.py --source proquest --window gfc_2008 --limit 20

No dependencies beyond the standard library; nothing to install, no network.
Resume-safe: already-processed page_ids are skipped on rerun.
"""

import argparse
import json
import re
from pathlib import Path

MAX_OCR_CHARS = 12000
MAX_CLAIMS_PER_ARTICLE = 3   # avoid flooding on long articles

# Direction lexicons. A sentence votes "worsen" or "improve" by which it contains.
WORSEN = [
    "recession", "depression", "downturn", "slump", "contraction", "hard times",
    "crisis", "panic", "collapse", "bear market", "layoff", "unemployment",
    "deflation", "bust", "worsen", "worse", "slowdown", "crash", "shrink",
    "recessionary", "gloom", "weaken",
]
IMPROVE = [
    "recovery", "worst is over", "rebound", "upturn", "prosperity", "boom",
    "expansion", "improve", "better times", "pick up", "bull market",
    "turn the corner", "recover", "revival", "upswing", "rebounding", "growth",
    "strengthen", "brighter",
]
# Forward-looking cues -- at least one must be present for a sentence to count.
CUES = [
    "will", "expected", "expect", "predict", "forecast", "is coming",
    "are coming", "ahead", "likely", "next year", "next month", "months",
    "outlook", "anticipate", "on the horizon", "brace for", "fear", "loom",
    "coming", "in the coming", "by next", "future", "projected", "see ",
]
# Hedging markers -> hedged=True (drives the Brier confidence in the scorer).
HEDGES = [
    "may ", "might", "could", "likely", "probably", "possibly", "some fear",
    "signs of", "appears", "seems", "perhaps", "expected to", "fears", "risk",
    "some economists", "warn",
]


def _hits(text_low, terms):
    return any(t in text_low for t in terms)


def split_sentences(text):
    """Collapse OCR line breaks, then split on sentence punctuation."""
    text = re.sub(r"\s+", " ", text.replace("\n", " "))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def horizon_from(text_low):
    m = re.search(r"(\d{1,2})\s*month", text_low)
    if m and 1 <= int(m.group(1)) <= 24:
        return int(m.group(1))
    if "next year" in text_low or "a year" in text_low or "12 month" in text_low:
        return 12
    if "next quarter" in text_low or "three month" in text_low:
        return 3
    return 6   # same default the LLM prompt uses when unstated


def extract_claims(text):
    claims = []
    for sent in split_sentences(text):
        low = sent.lower()
        if not _hits(low, CUES):
            continue
        worsen, improve = _hits(low, WORSEN), _hits(low, IMPROVE)
        if worsen == improve:
            continue   # neither present, or ambiguous (both) -> skip
        claims.append({
            "claim_text": " ".join(sent.split()[:60]),
            "predicted_direction": "worsen" if worsen else "improve",
            "predicted_state_at_horizon": "recession" if worsen else "expansion",
            "horizon_months": horizon_from(low),
            "voice": "other",          # voice needs an LLM to infer reliably
            "hedged": _hits(low, HEDGES),
            "attributed_to": None,
        })
        if len(claims) >= MAX_CLAIMS_PER_ARTICLE:
            break
    return claims


def enrich(claim, record):
    """Attach the same provenance fields extract_predictions.py attaches."""
    claim.update({
        "page_id": record["page_id"],
        "source": record.get("source"),
        "arm": "economy",
        "window": record.get("window"),
        "window_kind": record.get("window_kind"),
        "cycle": record.get("cycle"),
        "newspaper_title": record.get("newspaper_title"),
        "lccn": record.get("lccn"),
        "date": record.get("date"),
        "publisher_state": record.get("state"),
    })
    return claim


def load_done_ids(out_path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["page_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["loc", "nyt", "proquest"], required=True)
    ap.add_argument("--window", required=True, help="economy window_id, e.g. gfc_2008")
    ap.add_argument("--limit", type=int, default=None, help="max pages to process")
    args = ap.parse_args()

    in_path = Path(f"data/raw/{args.source}_economy_{args.window}.jsonl")
    if not in_path.exists():
        raise SystemExit(f"No raw file at {in_path}. Run tdm_parse.py first.")
    out_dir = Path("data/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{args.source}_economy_{args.window}.jsonl"
    done = load_done_ids(out_path)

    processed = with_pred = total_claims = 0
    with open(in_path) as f, open(out_path, "a") as out:
        for line in f:
            record = json.loads(line)
            if record["page_id"] in done:
                continue
            claims = extract_claims((record.get("ocr_text") or "")[:MAX_OCR_CHARS])
            if not claims:
                out.write(json.dumps({"page_id": record["page_id"],
                                      "no_predictions": True}) + "\n")
            else:
                with_pred += 1
                total_claims += len(claims)
                for c in claims:
                    out.write(json.dumps(enrich(c, record)) + "\n")
            out.flush()
            done.add(record["page_id"])
            processed += 1
            if args.limit and processed >= args.limit:
                break
    print(f"done: {processed} pages, {with_pred} with >=1 prediction, "
          f"{total_claims} claims -> {out_path}")


if __name__ == "__main__":
    main()
