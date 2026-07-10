"""
Search the NYT Article Search API for candidate coverage of each bill.

Repoints the reused NYT downloader pattern (see JeremysShit/election_arm/
download_nyt.py) from election/economy phrase search to per-bill queries.
This script only finds CANDIDATE articles -- it does not decide whether an
article is actually about a given bill (that is the LLM's job, in
extract_press.py). Expect a lot of false-positive candidates here; that is
fine, they get filtered downstream.

Query candidates per bill (Section 8): the bill's title, a key-noun-stripped
version of the title (broader recall), and sponsor last name + policy area
(broadest, noisiest -- off by default, use --include-broad-query).

Leakage cutoff: search window is [introduced_date - 7 days, introduced_date
+ 180 days], capped at the bill's latest_action_date if that came sooner.
Coverage found after this window must never be used as a feature.

Rate limits: 5 requests/minute, 500/day. Sleeps 12s per call. Resume-safe via
data/press_search_log.csv, which records every (congress, bill_type, number,
query) already searched, hit or miss, so reruns do not re-spend budget on
bills with zero coverage. Large runs span days.

Input:  data/bills/{congress}.jsonl
Output: data/press_raw/{congress}.jsonl (candidate article hits)
        data/press_search_log.csv (every query issued, for resume + audit)

Usage:
  export NYT_API_KEY=your_key
  python link_coverage.py --congress 118 --limit 25       # tiny test
  python link_coverage.py --congress 118
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
REQUEST_DELAY = 12.0
MAX_RETRIES = 5

STOPWORDS = {
    "a", "an", "and", "the", "of", "to", "for", "in", "on", "or", "act",
    "bill", "amend", "provide", "provides", "require", "requires",
    "establish", "establishes", "relating", "purposes", "other", "with",
    "that", "this", "be", "is", "are", "by", "at", "as", "certain", "such",
    "united", "states", "code", "internal", "revenue",
}


def build_queries(bill, include_broad=False):
    title = (bill.get("title") or "").strip()
    queries = []
    if title:
        queries.append(title)

    words = re.findall(r"[A-Za-z][A-Za-z'-]+", title)
    nouns = [w for w in words if w.lower() not in STOPWORDS and len(w) > 3]
    key_nouns = " ".join(nouns[:8])
    if key_nouns and key_nouns.lower() != title.lower():
        queries.append(key_nouns)

    if include_broad:
        sponsor_last = bill.get("sponsor_last_name")
        policy = bill.get("policy_area")
        if sponsor_last and policy:
            queries.append(f"{sponsor_last} {policy}")

    seen, out = set(), []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def search_window(introduced_date, latest_action_date=None, days_before=7, days_after=180):
    intro = datetime.strptime(introduced_date, "%Y-%m-%d")
    begin = intro - timedelta(days=days_before)
    end = intro + timedelta(days=days_after)
    if latest_action_date:
        try:
            final = datetime.strptime(latest_action_date, "%Y-%m-%d")
            end = min(end, final)
        except ValueError:
            pass
    end = max(end, begin)
    return begin.strftime("%Y%m%d"), end.strftime("%Y%m%d")


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


def search_query(api_key, query, begin, end, max_pages=1):
    page = 0
    total_hits = None
    while page < max_pages:
        params = {"q": query, "begin_date": begin, "end_date": end,
                 "page": page, "sort": "relevance", "api-key": api_key}
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
        page += 1
    return total_hits


def log_search(congress, bill_type, number, query, total_hits):
    log_path = Path("data/press_search_log.csv")
    new = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "congress", "bill_type", "number", "query",
                       "total_hits"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), congress,
                   bill_type, number, query, total_hits])


def load_searched(log_path="data/press_search_log.csv"):
    searched = set()
    p = Path(log_path)
    if p.exists():
        with open(p) as f:
            for row in csv.DictReader(f):
                searched.add((int(row["congress"]), row["bill_type"], row["number"],
                            row["query"]))
    return searched


def load_bills(bills_path):
    bills = []
    with open(bills_path) as f:
        for line in f:
            line = line.strip()
            if line:
                bills.append(json.loads(line))
    return bills


def run_congress(congress, bills_path, api_key, limit, max_pages, include_broad,
                 out_dir):
    bills = load_bills(bills_path)
    if limit:
        bills = bills[:limit]
    searched = load_searched()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{congress}.jsonl"
    print(f"\n=== Congress {congress}: searching NYT coverage for {len(bills)} bills ===")

    hits_found = 0
    with open(out_path, "a") as out:
        for bill in bills:
            bill_type, number = bill["bill_type"], bill["number"]
            if not bill.get("introduced_date"):
                continue
            begin, end = search_window(bill["introduced_date"],
                                       bill.get("latest_action_date"))
            for query in build_queries(bill, include_broad):
                key = (congress, bill_type, number, query)
                if key in searched:
                    continue
                docs = list(search_query(api_key, query, begin, end, max_pages))
                for doc in docs:
                    url = doc.get("web_url")
                    text = combine_text(doc)
                    if not url or not text:
                        continue
                    record = {
                        "congress": congress, "bill_type": bill_type, "number": number,
                        "query": query, "article_url": url,
                        "headline": (doc.get("headline") or {}).get("main"),
                        "pub_date": (doc.get("pub_date") or "")[:10],
                        "section": doc.get("section_name"),
                        "type_of_material": doc.get("type_of_material"),
                        "snippet_text": text,
                    }
                    out.write(json.dumps(record) + "\n")
                    hits_found += 1
                out.flush()
                log_search(congress, bill_type, number, query, len(docs))
                searched.add(key)
    print(f"[{congress}] {hits_found} candidate article hits written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", type=int, required=True)
    ap.add_argument("--bills-dir", default="data/bills")
    ap.add_argument("--out-dir", default="data/press_raw")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of bills searched, for testing")
    ap.add_argument("--max-pages", type=int, default=1,
                    help="NYT result pages per query (10 hits each)")
    ap.add_argument("--include-broad-query", action="store_true",
                    help="also search sponsor-last-name + policy-area (noisy, more API calls)")
    args = ap.parse_args()

    api_key = os.environ.get("NYT_API_KEY")
    if not api_key:
        raise SystemExit("Set NYT_API_KEY first. Free key at https://developer.nytimes.com/")

    bills_path = Path(args.bills_dir) / f"{args.congress}.jsonl"
    if not bills_path.exists():
        raise SystemExit(f"No bills file at {bills_path}. Run download_bills.py first.")

    run_congress(args.congress, bills_path, api_key, args.limit, args.max_pages,
                args.include_broad_query, Path(args.out_dir))


if __name__ == "__main__":
    main()
