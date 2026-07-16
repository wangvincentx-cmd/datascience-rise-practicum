"""
Download New York Times articles via the Article Search API. Two arms:

  --arm elections   weeks before each presidential election, 1900-2008
  --arm economy     all crisis and placebo windows in data/windows_economy.csv
                    (including the post-1963 windows LOC cannot cover)

LIMIT TO REPORT: the NYT API returns headline, abstract, lead paragraph,
and snippet only, never full article text. Forecasts usually appear in the
headline or lead, so recall is decent but lower than LOC full OCR.

Rate limits: 5 requests/minute, 500/day. The script sleeps 12s per call and
is resume-safe; spread big runs across days.

Output: data/raw/nyt_{elections|economy}_{cycle_or_window}.jsonl
Every query's total hits and pages fetched go to data/search_log.csv.

Usage:
  python download_nyt.py --arm economy --window gfc_2008 --max-pages 2
  python download_nyt.py --arm elections --cycle 1980
  python download_nyt.py --arm economy                  # all windows

Get a free key at https://developer.nytimes.com/ (enable Article Search API),
then:  export NYT_API_KEY=your_key
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests

BASE = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

ELECTION_WINDOWS = {
    1900: ("19000901", "19001106"), 1904: ("19040901", "19041108"),
    1908: ("19080901", "19081103"), 1912: ("19120901", "19121105"),
    1916: ("19160901", "19161107"), 1920: ("19200901", "19201102"),
    1924: ("19240901", "19241104"), 1928: ("19280901", "19281106"),
    1932: ("19320901", "19321108"), 1936: ("19360901", "19361103"),
    1940: ("19400901", "19401105"), 1944: ("19440901", "19441107"),
    1948: ("19480901", "19481102"), 1952: ("19520901", "19521104"),
    1956: ("19560901", "19561106"), 1960: ("19600901", "19601108"),
    1964: ("19640901", "19641103"), 1968: ("19680901", "19681105"),
    1972: ("19720901", "19721107"), 1976: ("19760901", "19761102"),
    1980: ("19800901", "19801104"), 1984: ("19840901", "19841106"),
    1988: ("19880901", "19881108"), 1992: ("19920901", "19921103"),
    1996: ("19960901", "19961105"), 2000: ("20000901", "20001107"),
    2004: ("20040901", "20041102"), 2008: ("20080901", "20081104"),
}

ELECTION_PHRASES = [
    '"will be elected"', '"will carry"', '"expected to win"',
    '"leads in the polls"', '"favored to win"', '"predicted"',
    '"seen winning"', '"is likely to win"',
]

# Broadened 2026-07-16 to mirror the LOC scraper's terms (newspaper_scraper.py)
# so the pre/post-1963 corpora sample the SAME construct, not different ones.
# The original 9 were all rare crisis-sentiment exact phrases ("recession is
# coming", "hard times ahead") and yielded only ~30 predictions for 1963-2010;
# LOC's top term was the neutral "business outlook" (606 claims, 46% of that
# corpus). This list keeps LOC's proven terms, adds their modern-vocabulary
# equivalents (economic outlook/forecast/recovery/downturn/slowdown), and
# retains a couple of directional forecast phrases. grade_claims.py's rubric
# filters the non-forecasts, so favoring recall here is safe.
ECONOMY_PHRASES = [
    # neutral economic-outlook terms (the high-yield LOC core)
    '"business outlook"', '"economic outlook"', '"business conditions"',
    '"economic forecast"',
    # downturn side (LOC "business recession"/"business slump" + modern vocab)
    '"business recession"', '"recession fears"', '"economic downturn"',
    '"economic slowdown"', '"business slump"',
    # recovery side (LOC "return of prosperity"/"business revival" + modern)
    '"economic recovery"', '"recovery expected"', '"return of prosperity"',
    '"business revival"',
    # explicit forecast framing / directional phrases
    '"economists expect"', '"worst is over"', '"hard times ahead"',
]

REQUEST_DELAY = 12.0
MAX_RETRIES = 5
NYT_PAGE_CAP = 100


def log_search(arm, window_id, phrase, total_hits, pages_fetched):
    log_path = Path("data/search_log.csv")
    new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "source", "arm", "window", "phrase",
                        "total_hits", "pages_fetched"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), "nyt", arm,
                    window_id, phrase, total_hits, pages_fetched])


def get_json(params):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(BASE, params=params, timeout=60)
        except requests.RequestException as e:
            print(f"  network error: {e}, retry in {2 ** attempt * 5}s")
            time.sleep(2 ** attempt * 5)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 60))
            print(f"  rate limited (429), sleeping {wait}s. Repeats mean the "
                  f"500/day cap; rerun tomorrow (resume-safe).")
            time.sleep(wait)
            continue
        if r.status_code in (401, 403):
            raise SystemExit(f"Auth failed ({r.status_code}). Check NYT_API_KEY "
                             f"and that Article Search API is enabled.")
        print(f"  HTTP {r.status_code}: {r.text[:200]}")
        time.sleep(2 ** attempt * 5)
    return None


def combine_text(doc):
    headline = (doc.get("headline") or {}).get("main", "") or ""
    parts = [headline, doc.get("abstract") or "", doc.get("lead_paragraph") or "",
             doc.get("snippet") or ""]
    seen, out = set(), []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return "\n".join(out)


def search_phrase(api_key, arm, window_id, phrase, begin, end, max_pages=None):
    page = 0
    total_hits = None
    cap = min(max_pages or NYT_PAGE_CAP, NYT_PAGE_CAP)
    while page < cap:
        params = {"q": phrase, "begin_date": begin, "end_date": end,
                  "page": page, "sort": "oldest", "api-key": api_key}
        data = get_json(params)
        time.sleep(REQUEST_DELAY)
        if data is None:
            break
        resp = data.get("response", {})
        if total_hits is None:
            total_hits = (resp.get("meta") or {}).get("hits", 0)
        docs = resp.get("docs", [])
        if not docs:
            break
        yield from docs
        # meta.hits is unreliable on these narrow quoted-phrase queries (often
        # reports 0 even when docs come back) -- page on actual page size
        # instead of trusting it, or a real corpus gets truncated at page 1.
        if len(docs) < 10:
            break
        page += 1
    log_search(arm, window_id, phrase, total_hits or 0, page + 1)


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
    windows = {}
    with open("data/windows_economy.csv") as f:
        for row in csv.DictReader(f):
            begin = row["start_date"].replace("-", "")
            end = row["end_date"].replace("-", "")
            windows[row["window_id"]] = (begin, end, row["kind"])
    return windows


def run_window(api_key, arm, window_id, begin, end, phrases, extra_meta,
               max_pages=None):
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"nyt_{arm}_{window_id}.jsonl"
    done = load_done_ids(out_path)
    print(f"\n=== NYT {arm} / {window_id} ({begin}-{end}), "
          f"{len(done)} articles already saved ===")

    with open(out_path, "a") as out:
        for phrase in phrases:
            print(f"[{window_id}] phrase: {phrase}")
            count = 0
            for doc in search_phrase(api_key, arm, window_id, phrase, begin,
                                     end, max_pages):
                url = doc.get("web_url")
                if not url or url in done:
                    continue
                text = combine_text(doc)
                if not text:
                    continue
                record = {
                    "page_id": url, "source": "nyt", "arm": arm,
                    "window": window_id,
                    "newspaper_title": "The New York Times", "lccn": None,
                    "date": (doc.get("pub_date") or "")[:10],
                    "state": None, "city": None,
                    "type_of_material": doc.get("type_of_material"),
                    "section": doc.get("section_name"),
                    "ocr_text": text,
                    "matched_phrase": phrase.strip('"'),
                    **extra_meta,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                done.add(url)
                count += 1
            print(f"  {count} new articles")
    print(f"[{window_id}] total NYT articles on disk: {len(done)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["elections", "economy"], required=True)
    ap.add_argument("--cycle", type=int, help="elections: single year")
    ap.add_argument("--window", help="economy: single window_id")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="cap result pages per phrase (10 hits each), for testing")
    ap.add_argument("--include-pre-1963", action="store_true",
                    help="economy, no --window: also hit windows before 1963 "
                         "(LOC already covers these full-text -- NYT-only "
                         "headline/lead data there is redundant and just "
                         "burns free-tier budget; default is post-1963 only)")
    args = ap.parse_args()

    api_key = os.environ.get("NYT_API_KEY")
    if not api_key:
        raise SystemExit("Set NYT_API_KEY first. Free key at https://developer.nytimes.com/")

    if args.arm == "elections":
        years = [args.cycle] if args.cycle else sorted(ELECTION_WINDOWS)
        for year in years:
            begin, end = ELECTION_WINDOWS[year]
            run_window(api_key, "elections", str(year), begin, end,
                       ELECTION_PHRASES, {"cycle": year}, args.max_pages)
    else:
        windows = load_economy_windows()
        if args.window:
            ids = [args.window]
        elif args.include_pre_1963:
            ids = sorted(windows)
        else:
            ids = sorted(wid for wid, (begin, _, _) in windows.items() if begin >= "19630101")
        for wid in ids:
            if wid not in windows:
                raise SystemExit(f"Unknown window '{wid}'. Options: {sorted(windows)}")
            begin, end, kind = windows[wid]
            run_window(api_key, "economy", wid, begin, end, ECONOMY_PHRASES,
                       {"window_kind": kind}, args.max_pages)


if __name__ == "__main__":
    main()
