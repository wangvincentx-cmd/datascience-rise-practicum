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
import re
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


# Economic-relevance lexicon. An article is kept if an economic term appears in
# the HEADLINE (strong signal it's an economy story), or if at least --econ-min
# distinct terms appear in the body (catches economy stories with vaguer heads).
# Multi-word phrases are matched as substrings; single words are matched on
# word boundaries so "gdp" doesn't fire inside another token.
ECON_TERMS = [
    "inflation", "recession", "unemployment", "federal reserve", "the fed",
    "interest rate", "rate hike", "rate cut", "gdp", "gross domestic product",
    "stock market", "wall street", "s&p 500", "dow jones", "nasdaq",
    "economy", "economic", "jobless", "layoff", "consumer price", "cpi",
    "labor market", "job market", "bear market", "bull market", "deflation",
    "central bank", "monetary policy", "fiscal", "gas prices", "housing market",
    "mortgage rate", "trade deficit", "tariff", "supply chain", "cost of living",
    "wages", "earnings report", "bond yield", "yield curve", "consumer spending",
    "retail sales", "manufacturing", "jerome powell", "treasury",
]
_SINGLE = {t for t in ECON_TERMS if " " not in t}
_PHRASE = [t for t in ECON_TERMS if " " in t]


def _count_terms(text):
    """Number of distinct economic terms present in text (already lowercased)."""
    words = set(re.findall(r"[a-z&0-9]+", text))
    n = sum(1 for t in _SINGLE if t in words)
    n += sum(1 for p in _PHRASE if p in text)
    return n


def econ_relevant(title, text, min_body):
    t = (title or "").lower()
    if _count_terms(t) >= 1:            # any econ term in the headline = keep
        return True
    return _count_terms((text or "").lower()) >= min_body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob", help="quote it, e.g. 'ccnews_out/2022-01/articles.*.jsonl'")
    ap.add_argument("out")
    ap.add_argument("--lang", default="en",
                    help="ISO code to keep (default en). Use 'any' to keep all "
                         "languages — the 'whole world' option.")
    ap.add_argument("--lang-prob", type=float, default=0.90,
                    help="min language-detection confidence (default 0.90)")
    ap.add_argument("--econ", action="store_true",
                    help="keep only economically-relevant articles (the filter "
                         "that actually shrinks the corpus to on-topic news)")
    ap.add_argument("--econ-min", type=int, default=2,
                    help="min distinct econ terms in the body when the headline "
                         "has none (default 2)")
    args = ap.parse_args()

    check_lang = args.lang.lower() != "any"

    kept = collections.Counter()
    total = dropped_lang = dropped_junk = dropped_econ = bad = 0

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
                if check_lang:
                    # detect on title + first 600 chars of body (fast, enough signal)
                    sample = ((r.get("title") or "") + " " + (r.get("text") or ""))[:600]
                    lang, prob = _IDENT.classify(sample)
                    if lang != args.lang or float(prob) < args.lang_prob:
                        dropped_lang += 1
                        continue
                if args.econ and not econ_relevant(r.get("title"), r.get("text"), args.econ_min):
                    dropped_econ += 1
                    continue
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept[r.get("domain")] += 1

    k = sum(kept.values())
    print(f"=== FILTER RESULT ===", file=sys.stderr)
    print(f"read:        {total}", file=sys.stderr)
    print(f"dropped junk domains: {dropped_junk}", file=sys.stderr)
    if check_lang:
        print(f"dropped non-{args.lang}:    {dropped_lang}", file=sys.stderr)
    if args.econ:
        print(f"dropped non-economic: {dropped_econ}", file=sys.stderr)
    print(f"KEPT:        {k}  ({len(kept)} domains)", file=sys.stderr)
    print(f"bad lines:   {bad}", file=sys.stderr)
    print(f"\n=== top 40 surviving domains ===", file=sys.stderr)
    for d, n in kept.most_common(40):
        print(f"{n:6d}  {d}", file=sys.stderr)


if __name__ == "__main__":
    main()
