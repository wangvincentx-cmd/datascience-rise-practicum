#!/usr/bin/env python3
"""Filter already-extracted CC-News articles down to legitimate English news.

Reads the per-task articles.*.jsonl you already produced, keeps only
English-language articles, drops known junk/spam/parked domains, and writes a
clean JSONL. Also prints volume + the surviving top domains so you can see
what "all open news" actually looks like — no re-download needed.

Usage:
    python filter_articles.py 'ccnews_out/2022-01/articles.*.jsonl' \
        ccnews_out/2022-01/clean.jsonl

Requires:  pip install --user py3langid
"""
import argparse
import collections
import glob
import json
import sys

from py3langid.langid import LanguageIdentifier, MODEL_FILE

# Default py3langid.classify() returns a raw log-probability, not a 0-1 score.
# A normalized identifier gives a real confidence in [0, 1] so --lang-prob works.
_IDENT = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)

# Substrings — any domain containing one of these is dropped. Not exhaustive;
# the language filter does most of the work. These are the obvious offenders:
# domain-parking, stock-photo/payments, and the auto-generated "MarketBeat"
# stock-spam clone network that floods CC-News with fake analyst-rating posts.
JUNK = [
    "dan.com", "shutterstock", "gocardless", "newsbreak", "zazoom",
    "com-unik", "mirtesen", "gulf365", "yahoo.com/lifestyle",
    # stock-spam clone network (shared template, auto-generated):
    "wkrb13", "tickerreport", "themarketsdaily", "dailypolitical",
    "transcriptdaily", "modernreaders", "ledgergazette", "thelincolnianonline",
    "americanbankingnews", "marketbeat", "equities.com", "etfdailynews",
    "watchlistnews", "registertribune", "themarketsdaily", "dakotafinancialnews",
]


def is_junk(domain):
    d = (domain or "").lower()
    return any(j in d for j in JUNK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob", help="quote it, e.g. 'ccnews_out/2022-01/articles.*.jsonl'")
    ap.add_argument("out")
    ap.add_argument("--lang", default="en", help="ISO code to keep (default en)")
    ap.add_argument("--lang-prob", type=float, default=0.90,
                    help="min language-detection confidence (default 0.90)")
    args = ap.parse_args()

    kept = collections.Counter()
    total = dropped_lang = dropped_junk = bad = 0

    with open(args.out, "w", encoding="utf-8") as out:
        for f in glob.glob(args.glob):
            for line in open(f, encoding="utf-8", errors="ignore"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                total += 1
                if is_junk(r.get("domain")):
                    dropped_junk += 1
                    continue
                # detect on title + first 600 chars of body (fast, plenty of signal)
                sample = ((r.get("title") or "") + " " + (r.get("text") or ""))[:600]
                lang, prob = _IDENT.classify(sample)
                if lang != args.lang or float(prob) < args.lang_prob:
                    dropped_lang += 1
                    continue
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept[r.get("domain")] += 1

    k = sum(kept.values())
    print(f"=== FILTER RESULT ===", file=sys.stderr)
    print(f"read:        {total}", file=sys.stderr)
    print(f"dropped junk domains: {dropped_junk}", file=sys.stderr)
    print(f"dropped non-{args.lang}:    {dropped_lang}", file=sys.stderr)
    print(f"KEPT:        {k}  ({len(kept)} domains)", file=sys.stderr)
    print(f"bad lines:   {bad}", file=sys.stderr)
    print(f"\n=== top 40 surviving domains ===", file=sys.stderr)
    for d, n in kept.most_common(40):
        print(f"{n:6d}  {d}", file=sys.stderr)


if __name__ == "__main__":
    main()
