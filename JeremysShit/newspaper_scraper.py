"""
Chronicling America prediction-claim scraper — full version.

Searches the Library of Congress loc.gov API for prediction-bearing phrases inside
each crisis episode's date window, downloads the full page OCR text, and extracts
candidate claim sentences (search phrase nearby + future-oriented language) into
claims_raw.csv, ready for grading (grade_claims.py) and scoring (score_claims.py).

Usage:
    python newspaper_scraper.py                          # all episodes, default caps
    python newspaper_scraper.py --episodes 1907 1929     # subset (match on name)
    python newspaper_scraper.py --pages-per-term 100     # scale the corpus up

Everything is cached in cache/ — rerunning never re-downloads, so it is safe to
interrupt and resume. search_log.csv records total hit counts per term (methods
artifact: shows the corpus was sampled, not cherry-picked).

API verified 2026-07-08. Politeness: ~1 req/sec, custom User-Agent, retries.
"""

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.loc.gov/collections/chronicling-america/"
HEADERS = {"User-Agent": "BU-RISE-student-research/0.2 (economic prediction accuracy study)"}
CACHE_DIR = Path("cache")
SLEEP_SECONDS = 1.1

FUTURE_MARKERS = re.compile(
    r"\b(will|shall|expect\w*|predict\w*|forecast\w*|outlook|ahead|coming|"
    r"by (?:spring|summer|fall|autumn|winter|next year)|within \w+ months?|"
    r"next (?:year|month|spring|summer|fall|autumn|winter))\b",
    re.IGNORECASE,
)

# Sentences that are almost always ads/boilerplate, not predictions.
JUNK = re.compile(
    r"(for sale|money to loan|per cent interest|great sale|prices? reduced|"
    r"call on us|subscri(be|ption)|advertis)",
    re.IGNORECASE,
)

# kind: "crisis" = prediction window inside/entering a recession.
#       "control" = calm placebo window with the same search terms — measures the
#       base rate of optimism/accuracy so crisis failure isn't confused with
#       "newspapers are always wrong" (or always right).
EPISODES = [
    {
        "name": "1907 Panic", "kind": "crisis",
        "start": "1907-10-01", "end": "1908-06-30",
        "terms": ["financial panic", "business outlook", "return of prosperity"],
    },
    {
        # NBER 1910-01 .. 1912-01. Added 2026-07-22 (power expansion for the
        # optimism-asymmetry test -- see CHANGELOG): free LOC full-text
        # coverage, no ProQuest needed, previously unscraped.
        "name": "1910 Recession", "kind": "crisis",
        "start": "1910-06-01", "end": "1911-03-31",
        "terms": ["business depression", "business outlook", "return of prosperity"],
    },
    {
        # NBER 1913-01 .. 1914-12 (ends just before WWI -- window kept short
        # of Aug 1914 on purpose, same reasoning the project already applies
        # to skip the 1918-19 war recession: wartime economic claims are a
        # different domain, not a clean peacetime "did they see it coming."
        "name": "1913 Recession", "kind": "crisis",
        "start": "1913-06-01", "end": "1914-03-31",
        "terms": ["business depression", "business outlook", "return of prosperity"],
    },
    {
        "name": "1920 Depression", "kind": "crisis",
        "start": "1920-06-01", "end": "1921-03-31",
        "terms": ["business depression", "business outlook", "return of prosperity"],
    },
    {
        # NBER 1923-05 .. 1924-07. Added 2026-07-22, same reasoning as 1910/1913.
        "name": "1923 Recession", "kind": "crisis",
        "start": "1923-09-01", "end": "1924-05-31",
        "terms": ["business depression", "business outlook", "return of prosperity", "business revival"],
    },
    {
        # NBER 1926-10 .. 1927-11. Added 2026-07-22, same reasoning as 1910/1913.
        "name": "1926 Recession", "kind": "crisis",
        "start": "1927-01-01", "end": "1927-09-30",
        "terms": ["business recession", "business outlook", "business revival"],
    },
    {
        "name": "1929 Crash", "kind": "crisis",
        "start": "1929-11-01", "end": "1930-12-31",
        "terms": ["business outlook", "return of prosperity", "business revival", "business depression"],
    },
    {
        "name": "1937 Recession", "kind": "crisis",
        "start": "1937-09-01", "end": "1938-03-31",
        "terms": ["business recession", "business outlook", "business revival"],
    },
    {
        "name": "1945 Reconversion", "kind": "crisis",
        "start": "1945-08-01", "end": "1946-06-30",
        "terms": ["postwar depression", "business outlook", "unemployment will", "reconversion"],
    },
    {
        "name": "1948 Recession", "kind": "crisis",
        "start": "1948-07-01", "end": "1949-06-30",
        "terms": ["business outlook", "recession", "business slump"],
    },
    {
        "name": "1957 Recession", "kind": "crisis",
        "start": "1957-08-01", "end": "1958-06-30",
        "terms": ["recession", "business outlook", "recovery will"],
    },
    # --- placebo/control windows: mid-expansion, no recession within 12 months ---
    {
        "name": "1905 Calm (control)", "kind": "control",
        "start": "1905-06-01", "end": "1906-05-31",
        "terms": ["financial panic", "business outlook", "return of prosperity"],
    },
    {
        "name": "1925 Calm (control)", "kind": "control",
        "start": "1925-06-01", "end": "1926-05-31",
        "terms": ["business depression", "business outlook", "return of prosperity"],
    },
    {
        "name": "1955 Calm (control)", "kind": "control",
        "start": "1955-06-01", "end": "1956-05-31",
        "terms": ["recession", "business outlook", "business slump"],
    },
]


def _get(url: str) -> bytes:
    """Fetch a URL with disk caching, politeness delay, and retry with backoff."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", url)[-150:] + ".bin")
    if cache_file.exists():
        return cache_file.read_bytes()
    last_err = None
    for attempt in range(4):
        try:
            time.sleep(SLEEP_SECONDS)
            req = urllib.request.Request(url, headers=HEADERS)
            data = urllib.request.urlopen(req, timeout=90).read()
            cache_file.write_bytes(data)
            return data
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"    retry {attempt + 1}: {e} (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"failed after retries: {url} ({last_err})")


def search_pages(phrase, start_date, end_date, max_pages, log_writer, episode_name):
    """Yield page-hit dicts for an exact-phrase, page-dated search."""
    collected, sp, total = 0, 1, None
    while collected < max_pages:
        params = {
            "qs": phrase, "ops": "PHRASE", "searchType": "advanced", "dl": "page",
            "start_date": start_date, "end_date": end_date,
            "fo": "json", "c": min(100, max_pages), "sp": sp,
        }
        try:
            data = json.loads(_get(BASE + "?" + urllib.parse.urlencode(params)))
        except RuntimeError as e:
            if "404" in str(e):
                return  # paged past the last page of results
            raise
        results = data.get("results", [])
        if total is None:
            total = data.get("pagination", {}).get("of", 0) or 0
            print(f"  '{phrase}': {total} total pages, taking up to {max_pages}")
            log_writer.writerow([episode_name, phrase, start_date, end_date, total, min(total, max_pages)])
            max_pages = min(max_pages, total)  # never page past the end (LOC 404s)
        if not results:
            return
        for r in results:
            if collected >= max_pages:
                return
            yield r
            collected += 1
        sp += 1


def fetch_full_text(page_url: str) -> str:
    """Download the full OCR text of one newspaper page (empty string if unavailable)."""
    resource = json.loads(_get(page_url.replace("http://", "https://") + "&fo=json"))
    fulltext_url = (resource.get("resource") or {}).get("fulltext_file")
    return _get(fulltext_url).decode("utf-8", errors="replace") if fulltext_url else ""


def extract_claims(text: str, phrase: str):
    """Sentences near the search phrase that look future-oriented and non-junk."""
    text = re.sub(r"\s+", " ", text.replace("\\n", " ").replace("\n", " "))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?]) ", text) if 40 <= len(s.strip()) <= 600]
    hits = [i for i, s in enumerate(sentences) if phrase.lower() in s.lower()]
    claims = []
    for i in hits:
        for j in range(max(0, i - 2), min(len(sentences), i + 3)):
            s = sentences[j]
            if FUTURE_MARKERS.search(s) and not JUNK.search(s) and s not in claims:
                claims.append(s)
    return claims


def run(episodes, pages_per_term, out_csv, log_csv="search_log.csv"):
    fieldnames = ["claim_id", "episode", "kind", "publisher", "state", "date",
                  "search_term", "page_url", "quote"]
    n_rows, seen_pages, claim_id = 0, set(), 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f, \
         open(log_csv, "w", newline="", encoding="utf-8") as lf:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        log_writer = csv.writer(lf)
        log_writer.writerow(["episode", "term", "start", "end", "total_hits", "pages_taken"])
        for ep in episodes:
            print(f"\nEPISODE: {ep['name']}  ({ep['start']} .. {ep['end']})")
            for phrase in ep["terms"]:
              try:  # one bad term must not kill the whole run
                for hit in search_pages(phrase, ep["start"], ep["end"],
                                        pages_per_term, log_writer, ep["name"]):
                    page_url = hit.get("id", "")
                    if not page_url or page_url in seen_pages:
                        continue
                    seen_pages.add(page_url)
                    try:
                        text = fetch_full_text(page_url)
                    except RuntimeError as e:
                        print(f"    skipping page: {e}")
                        continue
                    for quote in extract_claims(text, phrase):
                        claim_id += 1
                        writer.writerow({
                            "claim_id": claim_id,
                            "episode": ep["name"],
                            "kind": ep.get("kind", "crisis"),
                            "publisher": "; ".join(hit.get("partof_title") or [])[:120],
                            "state": "; ".join(hit.get("location_state") or []),
                            "date": hit.get("date", ""),
                            "search_term": phrase,
                            "page_url": page_url,
                            "quote": quote,
                        })
                        n_rows += 1
              except RuntimeError as e:
                print(f"  term '{phrase}' aborted, moving on: {e}")
              f.flush()
            print(f"  running total: {n_rows} claims from {len(seen_pages)} pages")
    print(f"\nDONE. {n_rows} candidate claims from {len(seen_pages)} pages -> {out_csv}")
    print(f"Search coverage logged to {log_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="substring filter on episode names, e.g. --episodes 1907 1929")
    ap.add_argument("--pages-per-term", type=int, default=30)
    ap.add_argument("--out", default="claims_raw.csv")
    ap.add_argument("--log-out", default="search_log.csv",
                    help="search_log.csv is opened in OVERWRITE mode -- when scraping a "
                         "subset of episodes to merge in later (not a fresh full run), "
                         "point this at a separate file so the real search_log.csv (the "
                         "corpus-transparency record for every episode already scraped) "
                         "isn't wiped down to just the subset's rows.")
    args = ap.parse_args()
    eps = EPISODES
    if args.episodes:
        eps = [e for e in EPISODES if any(k.lower() in e["name"].lower() for k in args.episodes)]
    run(eps, args.pages_per_term, args.out, args.log_out)
