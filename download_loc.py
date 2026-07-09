"""
Download newspaper pages from Chronicling America (Library of Congress)
via the loc.gov JSON API. Two arms share this downloader:

  --arm elections   phrase queries in the weeks before each presidential
                    election, cycles 1896-1960
  --arm economy     phrase queries inside crisis and placebo windows
                    defined in data/windows_economy.csv (pre-1963 windows only;
                    later windows are NYT-only)

The legacy chroniclingamerica.loc.gov API was retired in 2025; this uses the
current loc.gov collections API.

Output: data/raw/loc_{elections|economy}_{cycle_or_window}.jsonl
Corpus transparency: every query's total hits and pages fetched are appended
to data/search_log.csv (the "we sampled, we didn't cherry-pick" artifact).

Usage:
  python download_loc.py --arm elections --cycle 1948 --max-pages 2   # test
  python download_loc.py --arm elections                              # full
  python download_loc.py --arm economy --window crash_1929
  python download_loc.py --arm economy                                # all pre-1963 windows

Resume-safe: already-downloaded page IDs are skipped on rerun.
Set your real email in HEADERS before running; LOC asks for a contact.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import requests

BASE = "https://www.loc.gov/collections/chronicling-america/"
HEADERS = {"User-Agent": "BU-RISE-prediction-research (student project; contact: wangvincentx@gmail.com)"}

ELECTION_WINDOWS = {
    1896: ("1896-09-01", "1896-11-03"), 1900: ("1900-09-01", "1900-11-06"),
    1904: ("1904-09-01", "1904-11-08"), 1908: ("1908-09-01", "1908-11-03"),
    1912: ("1912-09-01", "1912-11-05"), 1916: ("1916-09-01", "1916-11-07"),
    1920: ("1920-09-01", "1920-11-02"), 1924: ("1924-09-01", "1924-11-04"),
    1928: ("1928-09-01", "1928-11-06"), 1932: ("1932-09-01", "1932-11-08"),
    1936: ("1936-09-01", "1936-11-03"), 1940: ("1940-09-01", "1940-11-05"),
    1944: ("1944-09-01", "1944-11-07"), 1948: ("1948-09-01", "1948-11-02"),
    1952: ("1952-09-01", "1952-11-04"), 1956: ("1956-09-01", "1956-11-06"),
    1960: ("1960-09-01", "1960-11-08"),
}

ELECTION_PHRASES = [
    "will be elected", "will carry the state", "certain of election",
    "predicted victory", "straw vote", "betting odds election", "landslide election",
]

ECONOMY_PHRASES = [
    "business depression", "hard times ahead", "prosperity will return",
    "business will improve", "worst is over", "recovery is expected",
    "panic is over", "depression is coming",
]

LOC_CUTOFF = "1964-01-01"   # LOC digitization thins out after ~1963
SEARCH_DELAY = 3.0
ITEM_DELAY = 1.0
MAX_RETRIES = 5


def log_search(arm, window_id, phrase, total_hits, pages_fetched):
    """Append one row to the corpus-transparency log."""
    log_path = Path("data/search_log.csv")
    new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "source", "arm", "window", "phrase",
                        "total_hits", "pages_fetched"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), "loc", arm,
                    window_id, phrase, total_hits, pages_fetched])


def get_json(url, params=None):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        except requests.RequestException as e:
            print(f"  network error: {e}, retrying in {2 ** attempt * 5}s")
            time.sleep(2 ** attempt * 5)
            continue
        if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
            return r.json()
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2 ** attempt * 10))
            print(f"  rate limited, sleeping {wait}s")
            time.sleep(wait)
            continue
        print(f"  HTTP {r.status_code} for {r.url}")
        time.sleep(2 ** attempt * 5)
    return None


def search_pages(arm, window_id, phrase, start_date, end_date, max_pages=None):
    """Yield result dicts, paginating. Logs total hits to search_log.csv."""
    sp = 1
    total_hits = None
    while True:
        params = {
            "qs": phrase, "ops": "PHRASE", "searchType": "advanced",
            "dl": "page", "start_date": start_date, "end_date": end_date,
            "fo": "json", "c": 100, "sp": sp, "at": "results,pagination",
        }
        data = get_json(BASE, params)
        if data is None:
            break
        pagination = data.get("pagination", {}) or {}
        if total_hits is None:
            total_hits = pagination.get("of", 0)
        results = data.get("results", [])
        if not results:
            break
        yield from results
        if not pagination.get("next"):
            break
        sp += 1
        if max_pages and sp > max_pages:
            break
        time.sleep(SEARCH_DELAY)
    log_search(arm, window_id, phrase, total_hits or 0, sp)


def first(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def fetch_ocr_text(resource):
    """The item-detail JSON never embeds page OCR text; it only points to a
    separate word-coordinates-service endpoint (resource.fulltext_file) that
    returns {segment_path: {"full_text": "..."}}. Fetch that separately."""
    url = (first(resource) or {}).get("fulltext_file")
    if not url:
        return ""
    data = get_json(url)
    if not isinstance(data, dict):
        return ""
    for value in data.values():
        if isinstance(value, dict) and value.get("full_text"):
            return value["full_text"]
    return ""


def fetch_page_detail(result):
    item_url = result.get("id", "")
    if not item_url.startswith(("http://www.loc.gov", "https://www.loc.gov")):
        return None
    data = get_json(item_url, params={"fo": "json"})
    if data is None:
        return None
    item = data.get("item", {}) or {}
    return {
        "page_id": item_url,
        "lccn": first(item.get("number_lccn")),
        "newspaper_title": first(item.get("newspaper_title")),
        "date": first(item.get("date")),
        "state": first(item.get("location_state")),
        "city": first(item.get("location_city")),
        "page": (data.get("pagination") or {}).get("current"),
        "ocr_text": fetch_ocr_text(data.get("resource")),
    }


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


def load_economy_windows():
    """Read crisis/placebo windows, keep those inside LOC coverage."""
    windows = {}
    with open("data/windows_economy.csv") as f:
        for row in csv.DictReader(f):
            if row["start_date"] < LOC_CUTOFF:
                windows[row["window_id"]] = (row["start_date"], row["end_date"],
                                             row["kind"])
    return windows


def run_window(arm, window_id, start_date, end_date, phrases, extra_meta,
               max_pages=None):
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"loc_{arm}_{window_id}.jsonl"
    done = load_done_ids(out_path)
    print(f"\n=== LOC {arm} / {window_id} ({start_date} to {end_date}), "
          f"{len(done)} pages already saved ===")

    with open(out_path, "a") as out:
        for phrase in phrases:
            print(f"[{window_id}] phrase: '{phrase}'")
            count = 0
            for result in search_pages(arm, window_id, phrase, start_date,
                                       end_date, max_pages):
                page_id = result.get("id", "")
                if page_id in done:
                    continue
                record = fetch_page_detail(result)
                time.sleep(ITEM_DELAY)
                if record is None or not record["ocr_text"]:
                    continue
                record.update({"source": "loc", "arm": arm,
                               "window": window_id,
                               "matched_phrase": phrase, **extra_meta})
                out.write(json.dumps(record) + "\n")
                out.flush()
                done.add(page_id)
                count += 1
                if count % 25 == 0:
                    print(f"  saved {count} pages")
            print(f"  done, {count} new pages for this phrase")
            time.sleep(SEARCH_DELAY)
    print(f"[{window_id}] total pages on disk: {len(done)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["elections", "economy"], required=True)
    ap.add_argument("--cycle", type=int, help="elections: single year, e.g. 1948")
    ap.add_argument("--window", help="economy: single window_id, e.g. crash_1929")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="cap search result pages per phrase (100 hits each), for testing")
    args = ap.parse_args()

    if args.arm == "elections":
        years = [args.cycle] if args.cycle else sorted(ELECTION_WINDOWS)
        for year in years:
            start, end = ELECTION_WINDOWS[year]
            run_window("elections", str(year), start, end, ELECTION_PHRASES,
                       {"cycle": year}, args.max_pages)
    else:
        windows = load_economy_windows()
        ids = [args.window] if args.window else sorted(windows)
        for wid in ids:
            if wid not in windows:
                raise SystemExit(f"Unknown or post-1963 window '{wid}'. "
                                 f"LOC windows: {sorted(windows)}")
            start, end, kind = windows[wid]
            run_window("economy", wid, start, end, ECONOMY_PHRASES,
                       {"window_kind": kind}, args.max_pages)


if __name__ == "__main__":
    main()
