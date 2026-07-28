"""
Run the CURRENT regex extractor over the gold pages, to get the number the
whole rebuild rests on: what recall does it actually have?

This is deliberately GENEROUS to the incumbent, in three ways, so that a loss
cannot be blamed on the harness:

1. **Every search phrase, not just one.** In production, a page was fetched by
   one phrase and `extract_claims` was called with only that phrase, so only
   sentences near THAT phrase could be found. Here it is called once per phrase
   in the corpus's whole 29-term vocabulary and the union is taken. Strictly
   more recall than production could ever have had.

2. **Clean text, not the raw JSON blob.** Production passed the undecoded
   word-coordinates payload straight in, so sentence splitting had to cope with
   a `{"/service/...":{"full_text":"` prefix and literal backslash-n. Here it
   gets the parsed text from build_page_corpus.py, so its sentence splitter
   works as intended.

3. **No page-selection penalty.** All 16 gold pages are scored, including any
   the phrase search would never have retrieved in the first place.

The phrase list is read from the shipped corpus (claims_raw.csv's search_term
column) rather than retyped, so it stays honest if the corpus changes.

Usage (from JeremysShit/):
    python gold_extraction/run_regex_baseline.py
    python gold_extraction/eval_extraction.py --pred gold_extraction/pred_regex.jsonl --name regex
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from newspaper_scraper import EPISODES, extract_claims  # noqa: E402


def corpus_phrases(claims_csv="claims_raw.csv"):
    """Every search phrase the corpus was actually built with, plus the phrase
    lists still declared in newspaper_scraper.EPISODES (in case a term was
    declared but produced no rows)."""
    phrases = set()
    if Path(claims_csv).exists():
        with open(claims_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("search_term"):
                    phrases.add(row["search_term"].strip().lower())
    for ep in EPISODES:
        for term in ep["terms"]:
            phrases.add(term.strip().lower())
    return sorted(phrases)


def run(pages_path, out_path, claims_csv="claims_raw.csv"):
    phrases = corpus_phrases(claims_csv)
    pages = [json.loads(l) for l in open(pages_path, encoding="utf-8") if l.strip()]
    print(f"{len(phrases)} search phrases x {len(pages)} pages")

    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for page in pages:
            seen = set()
            for phrase in phrases:
                if phrase not in page["ocr_text"].lower():
                    continue  # production could not have reached this page via this phrase
                for quote in extract_claims(page["ocr_text"], phrase):
                    if quote in seen:
                        continue
                    seen.add(quote)
                    fh.write(json.dumps({
                        "page_id": page["page_id"],
                        "date": page.get("date"),
                        "quote": quote,
                        "matched_phrase": phrase,
                    }, ensure_ascii=False) + "\n")
                    n += 1
            print(f"  p{page.get('page_index', '?')} {page.get('date')}  "
                  f"{len(seen):>3} candidate sentences")
    print(f"\n{n} candidates -> {out_path}")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", default="gold_extraction/gold_pages.jsonl")
    ap.add_argument("--out", default="gold_extraction/pred_regex.jsonl")
    ap.add_argument("--claims", default="claims_raw.csv")
    args = ap.parse_args()
    run(args.pages, args.out, args.claims)
