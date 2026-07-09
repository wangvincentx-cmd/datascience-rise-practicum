"""
Chronicling America prediction-claim scraper — starter version.

Searches the Library of Congress loc.gov API for prediction-bearing phrases in a
date window, downloads full page OCR, and extracts candidate "claim" sentences
(search term nearby + future-oriented language) into a CSV for rubric grading.

Usage:  python newspaper_scraper_starter.py
Edit EPISODES / MAX_PAGES_PER_TERM at the bottom to scale up.

API verified 2026-07-08. Be polite: ~1 request/second, cached to disk.
"""

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.loc.gov/collections/chronicling-america/"
HEADERS = {"User-Agent": "BU-RISE-student-research/0.1 (economic prediction accuracy study)"}
CACHE_DIR = Path("cache")
SLEEP_SECONDS = 1.2

FUTURE_MARKERS = re.compile(
    r"\b(will|shall|expect\w*|predict\w*|forecast\w*|outlook|ahead|coming|"
    r"by (?:spring|summer|fall|autumn|winter|next year)|within \w+ months?)\b",
    re.IGNORECASE,
)


def _get(url: str) -> bytes:
    """Fetch a URL with caching, politeness delay, and simple retry."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", url)[-150:] + ".bin")
    if cache_file.exists():
        return cache_file.read_bytes()
    for attempt in range(4):
        try:
            time.sleep(SLEEP_SECONDS)
            req = urllib.request.Request(url, headers=HEADERS)
            data = urllib.request.urlopen(req, timeout=90).read()
            cache_file.write_bytes(data)
            return data
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"    retry {attempt + 1} after error: {e} (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"failed after retries: {url}")


def search_pages(phrase: str, start_date: str, end_date: str, max_pages: int):
    """Yield page-hit dicts for an exact-phrase search in a date window."""
    collected, page_num = 0, 1
    while collected < max_pages:
        params = {
            "qs": phrase, "ops": "PHRASE", "searchType": "advanced", "dl": "page",
            "start_date": start_date, "end_date": end_date,
            "fo": "json", "c": min(100, max_pages), "sp": page_num,
        }
        data = json.loads(_get(BASE + "?" + urllib.parse.urlencode(params)))
        results = data.get("results", [])
        if not results:
            return
        total = data.get("pagination", {}).get("of", "?")
        if page_num == 1:
            print(f"  '{phrase}' {start_date}..{end_date}: {total} total pages, taking up to {max_pages}")
        for r in results:
            if collected >= max_pages:
                return
            yield r
            collected += 1
        page_num += 1


def fetch_full_text(page_url: str) -> str:
    """Download the full OCR text for one newspaper page."""
    resource = json.loads(_get(page_url.replace("http://", "https://") + "&fo=json"))
    fulltext_url = (resource.get("resource") or {}).get("fulltext_file")
    if not fulltext_url:
        return ""
    return _get(fulltext_url).decode("utf-8", errors="replace")


def extract_claims(text: str, phrase: str):
    """Return sentences near the search phrase that contain future-oriented language."""
    text = re.sub(r"\s+", " ", text.replace("\\n", " ").replace("\n", " "))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?]) ", text) if 40 <= len(s.strip()) <= 600]
    hits = [i for i, s in enumerate(sentences) if phrase.lower() in s.lower()]
    claims = []
    for i in hits:
        for j in range(max(0, i - 2), min(len(sentences), i + 3)):
            s = sentences[j]
            if FUTURE_MARKERS.search(s) and s not in claims:
                claims.append(s)
    return claims


def run(episodes, max_pages_per_term, out_csv="claims_raw.csv"):
    rows, seen_pages = [], set()
    for ep in episodes:
        print(f"\nEPISODE: {ep['name']}")
        for phrase in ep["terms"]:
            for hit in search_pages(phrase, ep["start"], ep["end"], max_pages_per_term):
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
                    rows.append({
                        "episode": ep["name"],
                        "publisher": "; ".join(hit.get("partof_title") or [])[:120],
                        "state": "; ".join(hit.get("location_state") or []),
                        "date": hit.get("date", ""),
                        "search_term": phrase,
                        "page_url": page_url,
                        "quote": quote,
                    })
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                ["episode", "publisher", "state", "date", "search_term", "page_url", "quote"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} candidate claims from {len(seen_pages)} pages to {out_csv}")


EPISODES = [
    {
        "name": "Panic of 1907",
        "start": "1907-10-01", "end": "1908-06-30",
        "terms": ["financial panic", "business outlook", "return of prosperity"],
    },
    # Scale up by adding the rest (see project_plan_v2.md section 2):
    # {"name": "Depression of 1920-21", "start": "1920-06-01", "end": "1921-03-31",
    #  "terms": ["business depression", "business outlook", "return of prosperity"]},
    # {"name": "Crash to Great Depression", "start": "1929-11-01", "end": "1930-12-31",
    #  "terms": ["business conditions will", "prosperity", "business outlook"]},
    # {"name": "Postwar reconversion", "start": "1945-08-01", "end": "1946-06-30",
    #  "terms": ["postwar depression", "business outlook", "unemployment will"]},
]

# Smoke-test scale. For the real corpus raise to ~100 pages/term (see plan section 3.3).
MAX_PAGES_PER_TERM = 5

if __name__ == "__main__":
    run(EPISODES, MAX_PAGES_PER_TERM)
