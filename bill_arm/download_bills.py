"""
Download bills from the Congress.gov API v3 into one JSONL file per Congress.

Scope: bill types that can become law only -- hr, s, hjres, sjres. Simple and
concurrent resolutions (hres, sres, hconres, sconres) are excluded; they
cannot become law and would pollute the target.

For each bill this fetches:
  - bill detail (sponsor, policy area, introduced date, title, latest action,
    laws array)
  - original cosponsors only (isOriginalCosponsor == True), never the full
    cosponsor list, to respect the introduction-time leakage cutoff
  - the committee(s) the bill was referred to
  - related bills, to derive has_companion_bill
  - optionally, the introduced-version bill text (--fetch-text; expensive,
    off by default)

Enactment target (became_law): True if latestAction text contains "Became
Public Law" / "Became Private Law", OR the bill detail's laws array is
non-empty. See is_enacted().

Output: data/bills/{congress}.jsonl (all four bill types mixed, tagged by
bill_type). Resume-safe: already-downloaded (bill_type, number) pairs are
skipped on rerun.

Rate limits: DEMO_KEY is capped near 40 requests/hour. A registered key
allows 5,000/hour (~1.3/sec); this script paces itself to stay under that
with --sleep (default 0.4s between calls).

Usage:
  export CONGRESS_API_KEY=your_key
  python download_bills.py --congress 118 --limit 25          # tiny test
  python download_bills.py --congress 118                     # full congress
  python download_bills.py --congress 108 --congress 109 ...   # repeatable
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.congress.gov/v3"
BILL_TYPES = ["hr", "s", "hjres", "sjres"]
MAX_RETRIES = 5
PAGE_LIMIT = 250


def get_json(url, api_key, params=None, sleep=0.4):
    params = dict(params or {})
    params["api_key"] = api_key
    params["format"] = "json"
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException as e:
            print(f"  network error: {e}, retry in {2 ** attempt * 5}s")
            time.sleep(2 ** attempt * 5)
            continue
        if r.status_code == 200:
            time.sleep(sleep)
            return r.json()
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 60))
            print(f"  rate limited (429), sleeping {wait}s. Repeats on "
                  f"DEMO_KEY mean the ~40/hour cap; get a registered key.")
            time.sleep(wait)
            continue
        if r.status_code in (401, 403):
            raise SystemExit(f"Auth failed ({r.status_code}) on {url}. "
                             f"Check CONGRESS_API_KEY.")
        if r.status_code == 404:
            return None
        print(f"  HTTP {r.status_code} on {url}: {r.text[:200]}")
        time.sleep(2 ** attempt * 5)
    return None


def iter_bill_list(congress, bill_type, api_key, limit=None, sleep=0.4):
    """Yield bill-list stubs (congress/type/number/title/latestAction) for
    one congress + bill type, paginating until exhausted or `limit` hit."""
    url = f"{BASE}/bill/{congress}/{bill_type}"
    offset = 0
    yielded = 0
    while True:
        page_size = PAGE_LIMIT if limit is None else min(PAGE_LIMIT, limit - yielded)
        if page_size <= 0:
            return
        data = get_json(url, api_key, {"limit": page_size, "offset": offset}, sleep)
        if not data:
            return
        bills = data.get("bills", [])
        if not bills:
            return
        for b in bills:
            yield b
            yielded += 1
            if limit is not None and yielded >= limit:
                return
        offset += len(bills)


def fetch_bill_detail(congress, bill_type, number, api_key, sleep=0.4):
    url = f"{BASE}/bill/{congress}/{bill_type}/{number}"
    data = get_json(url, api_key, sleep=sleep)
    if not data:
        return None
    return data.get("bill")


def fetch_original_cosponsors(congress, bill_type, number, api_key, sleep=0.4):
    """Only cosponsors present on the introduction date (isOriginalCosponsor
    True). Later cosponsors are forbidden -- using them is leakage."""
    url = f"{BASE}/bill/{congress}/{bill_type}/{number}/cosponsors"
    out = []
    offset = 0
    while True:
        data = get_json(url, api_key, {"limit": PAGE_LIMIT, "offset": offset}, sleep)
        if not data:
            break
        cosponsors = data.get("cosponsors", [])
        if not cosponsors:
            break
        for c in cosponsors:
            if c.get("isOriginalCosponsor"):
                out.append({"party": c.get("party"), "state": c.get("state")})
        offset += len(cosponsors)
        if len(cosponsors) < PAGE_LIMIT:
            break
    return out


def fetch_primary_committee(congress, bill_type, number, api_key, sleep=0.4):
    url = f"{BASE}/bill/{congress}/{bill_type}/{number}/committees"
    data = get_json(url, api_key, sleep=sleep)
    if not data:
        return None
    committees = data.get("committees", [])
    if not committees:
        return None
    return committees[0].get("name")


def fetch_has_companion_bill(congress, bill_type, number, api_key, sleep=0.4):
    url = f"{BASE}/bill/{congress}/{bill_type}/{number}/relatedbills"
    data = get_json(url, api_key, sleep=sleep)
    if not data:
        return False
    return len(data.get("relatedBills", [])) > 0


def fetch_introduced_text(congress, bill_type, number, api_key, sleep=0.4):
    """Best-effort plain text of the introduced version. Expensive (extra
    call plus a second fetch of the text document itself); off by default."""
    url = f"{BASE}/bill/{congress}/{bill_type}/{number}/text"
    data = get_json(url, api_key, sleep=sleep)
    if not data:
        return None
    versions = data.get("textVersions", [])
    introduced = next((v for v in versions if "Introduced" in (v.get("type") or "")),
                      versions[0] if versions else None)
    if not introduced:
        return None
    formats = introduced.get("formats", [])
    txt_fmt = next((f for f in formats if f.get("type") == "Formatted Text"), None)
    if not txt_fmt or not txt_fmt.get("url"):
        return None
    try:
        r = requests.get(txt_fmt["url"], timeout=60)
        time.sleep(sleep)
        if r.status_code != 200:
            return None
    except requests.RequestException:
        return None
    import re
    text = re.sub(r"<[^>]+>", " ", r.text)
    return re.sub(r"\s+", " ", text).strip()


def is_enacted(bill_detail):
    """True if the bill became law. See Section 4.1: latestAction text match
    OR a populated laws array. Both are checked; either is sufficient."""
    laws = bill_detail.get("laws") or []
    if laws:
        return True
    latest = (bill_detail.get("latestAction") or {}).get("text") or ""
    return "Became Public Law" in latest or "Became Private Law" in latest


def build_record(congress, bill_type, number, api_key, sleep, fetch_text):
    detail = fetch_bill_detail(congress, bill_type, number, api_key, sleep)
    if not detail:
        return None
    sponsors = detail.get("sponsors") or []
    sponsor = sponsors[0] if sponsors else {}
    original_cosponsors = fetch_original_cosponsors(congress, bill_type, number,
                                                     api_key, sleep)
    primary_committee = fetch_primary_committee(congress, bill_type, number,
                                                api_key, sleep)
    has_companion = fetch_has_companion_bill(congress, bill_type, number,
                                             api_key, sleep)
    record = {
        "congress": congress,
        "bill_type": bill_type,
        "number": str(number),
        "introduced_date": detail.get("introducedDate"),
        "title": detail.get("title"),
        "policy_area": (detail.get("policyArea") or {}).get("name"),
        "sponsor_party": sponsor.get("party"),
        "sponsor_state": sponsor.get("state"),
        "sponsor_bioguide_id": sponsor.get("bioguideId"),
        "sponsor_last_name": sponsor.get("lastName"),
        "sponsor_full_name": sponsor.get("fullName"),
        "latest_action_text": (detail.get("latestAction") or {}).get("text"),
        "latest_action_date": (detail.get("latestAction") or {}).get("actionDate"),
        "laws": detail.get("laws") or [],
        "became_law": is_enacted(detail),
        "original_cosponsors": original_cosponsors,
        "n_original_cosponsors": len(original_cosponsors),
        "primary_committee": primary_committee,
        "has_companion_bill": has_companion,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if fetch_text:
        record["introduced_text"] = fetch_introduced_text(
            congress, bill_type, number, api_key, sleep)
    return record


def load_done(out_path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["bill_type"], rec["number"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def run_congress(congress, api_key, bill_types, limit, sleep, fetch_text, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{congress}.jsonl"
    done = load_done(out_path)
    print(f"\n=== Congress {congress}, {len(done)} bills already saved ===")
    with open(out_path, "a") as out:
        for bill_type in bill_types:
            print(f"[{congress}] bill_type: {bill_type}")
            count = 0
            for stub in iter_bill_list(congress, bill_type, api_key, limit, sleep):
                number = str(stub.get("number"))
                if (bill_type, number) in done:
                    continue
                record = build_record(congress, bill_type, number, api_key,
                                      sleep, fetch_text)
                if record is None:
                    continue
                out.write(json.dumps(record) + "\n")
                out.flush()
                done.add((bill_type, number))
                count += 1
                if count % 25 == 0:
                    print(f"  {count} bills fetched so far")
            print(f"  {count} new bills for {bill_type}")
    print(f"[{congress}] total bills on disk: {len(done)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", type=int, action="append", required=True,
                    help="repeatable, e.g. --congress 117 --congress 118")
    ap.add_argument("--bill-types", default=",".join(BILL_TYPES),
                    help="comma-separated subset of hr,s,hjres,sjres")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap bills per (congress, bill_type), for testing")
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="seconds to sleep after each API call")
    ap.add_argument("--fetch-text", action="store_true",
                    help="also fetch introduced-version bill text (slow, extra calls)")
    ap.add_argument("--out-dir", default="data/bills")
    args = ap.parse_args()

    import os
    api_key = os.environ.get("CONGRESS_API_KEY", "DEMO_KEY")
    if api_key == "DEMO_KEY":
        print("WARNING: using DEMO_KEY, capped near 40 requests/hour. "
              "Set CONGRESS_API_KEY for a registered key (5,000/hour).")

    bill_types = args.bill_types.split(",")
    for bt in bill_types:
        if bt not in BILL_TYPES:
            raise SystemExit(f"Unknown bill type '{bt}'. Options: {BILL_TYPES}")

    out_dir = Path(args.out_dir)
    for congress in args.congress:
        run_congress(congress, api_key, bill_types, args.limit, args.sleep,
                    args.fetch_text, out_dir)


if __name__ == "__main__":
    main()
