"""
Bulk alternative to download_bills.py: instead of 4 Congress.gov API calls
per bill (~4 days for the 108th-118th study at 5,000 req/hour), download
GPO's BILLSTATUS bulk-data zips from govinfo.gov and parse the XML locally.
One zip per (congress, bill type), no API key, no rate limit. BILLSTATUS
bulk data covers the 108th Congress (2003) onward -- exactly the study scope.

Emits the SAME JSONL schema as download_bills.py into the same output files,
so build_features.py and everything downstream is untouched. Resume-safe the
same way: (bill_type, number) pairs already in the output file are skipped,
so bulk runs coexist with earlier API-pulled records.

The BILLSTATUS XML contains everything the API path fetched: sponsors,
cosponsors with the isOriginalCosponsor flag, committees with their
activities, related bills, actions, policy area, title, and the laws array.
introduced_text is NOT in BILLSTATUS; that still needs the API (--fetch-text
on download_bills.py) if text features beyond the title are wanted.

Leakage note: the committees list includes committees involved at any point
in the bill's life. The primary committee feature must be the introduction-
time referral, so the parser prefers the committee whose activities include
a "Referred to" entry over just taking the first list item.

Usage:
  python download_bills_bulk.py --congress 118 --bill-types sjres   # tiny test
  python download_bills_bulk.py --congress 108 --congress 109 ... --congress 118
"""

import argparse
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

from download_bills import BILL_TYPES, is_enacted, load_done

BULK_URL = "https://www.govinfo.gov/bulkdata/BILLSTATUS/{congress}/{bill_type}/BILLSTATUS-{congress}-{bill_type}.zip"
# govinfo rejects the default python-requests User-Agent
HEADERS = {"User-Agent": "Mozilla/5.0 (research pipeline; bill-passage study)"}


def download_zip(congress, bill_type, cache_dir):
    """Download one BILLSTATUS zip, cached on disk so reruns cost nothing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"BILLSTATUS-{congress}-{bill_type}.zip"
    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"  using cached {zip_path.name}")
        return zip_path
    url = BULK_URL.format(congress=congress, bill_type=bill_type)
    print(f"  downloading {url}")
    r = requests.get(url, headers=HEADERS, timeout=300, stream=True)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} for {url}, skipping")
        return None
    tmp_path = zip_path.with_suffix(".part")
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp_path.rename(zip_path)
    print(f"  saved {zip_path.name} ({zip_path.stat().st_size >> 20} MB)")
    return zip_path


def _text(el, path):
    found = el.find(path) if el is not None else None
    return found.text.strip() if found is not None and found.text else None


def _is_true(value):
    return (value or "").strip().lower() == "true"


def parse_original_cosponsors(bill_el):
    """Only cosponsors with isOriginalCosponsor True -- same leakage rule as
    the API path: later-added cosponsors are forbidden features."""
    out = []
    for item in bill_el.findall("./cosponsors/item"):
        if _is_true(_text(item, "isOriginalCosponsor")):
            out.append({"party": _text(item, "party"), "state": _text(item, "state")})
    return out


def parse_primary_committee(bill_el):
    """The introduction-time referral committee: prefer a committee whose
    activities include a 'Referred to' entry; fall back to the first item."""
    items = bill_el.findall("./committees/item")
    if not items:
        return None
    for item in items:
        for activity in item.findall("./activities/item"):
            name = _text(activity, "name") or ""
            if name.startswith("Referred"):
                return _text(item, "name")
    return _text(items[0], "name")


def parse_latest_action(bill_el):
    """latestAction element when present; otherwise the newest dated action."""
    latest = bill_el.find("./latestAction")
    if latest is not None:
        return _text(latest, "text"), _text(latest, "actionDate")
    actions = bill_el.findall("./actions/item")
    dated = [(a, _text(a, "actionDate")) for a in actions]
    dated = [(a, d) for a, d in dated if d]
    if not dated:
        return None, None
    newest = max(dated, key=lambda pair: pair[1])
    return _text(newest[0], "text"), newest[1]


def parse_laws(bill_el):
    return [{"type": _text(item, "type"), "number": _text(item, "number")}
            for item in bill_el.findall("./laws/item")]


def parse_billstatus_xml(xml_bytes):
    """Parse one BILLSTATUS XML document into the download_bills.py record
    schema. Returns None if the document has no <bill> element."""
    root = ET.fromstring(xml_bytes)
    bill_el = root.find("./bill")
    if bill_el is None:
        return None
    sponsor = bill_el.find("./sponsors/item")
    latest_text, latest_date = parse_latest_action(bill_el)
    laws = parse_laws(bill_el)
    original_cosponsors = parse_original_cosponsors(bill_el)
    bill_type = (_text(bill_el, "type") or "").lower()
    related = bill_el.findall("./relatedBills/item")

    detail_for_enactment = {"laws": laws, "latestAction": {"text": latest_text or ""}}
    return {
        "congress": int(_text(bill_el, "congress")),
        "bill_type": bill_type,
        "number": _text(bill_el, "number"),
        "introduced_date": _text(bill_el, "introducedDate"),
        "title": _text(bill_el, "title"),
        "policy_area": _text(bill_el, "policyArea/name"),
        "sponsor_party": _text(sponsor, "party"),
        "sponsor_state": _text(sponsor, "state"),
        "sponsor_bioguide_id": _text(sponsor, "bioguideId"),
        "sponsor_last_name": _text(sponsor, "lastName"),
        "sponsor_full_name": _text(sponsor, "fullName"),
        "latest_action_text": latest_text,
        "latest_action_date": latest_date,
        "laws": laws,
        "became_law": is_enacted(detail_for_enactment),
        "original_cosponsors": original_cosponsors,
        "n_original_cosponsors": len(original_cosponsors),
        "primary_committee": parse_primary_committee(bill_el),
        "has_companion_bill": len(related) > 0,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "govinfo_bulk",
    }


def parse_zip(zip_path, expected_bill_type):
    """Yield parsed records from every XML member of a BILLSTATUS zip."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        for name in sorted(names):
            try:
                record = parse_billstatus_xml(zf.read(name))
            except ET.ParseError as e:
                print(f"  XML parse error in {name}: {e}, skipping")
                continue
            if record is None:
                continue
            if record["bill_type"] != expected_bill_type:
                print(f"  {name}: unexpected bill_type {record['bill_type']}, skipping")
                continue
            yield record


def run_congress(congress, bill_types, cache_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{congress}.jsonl"
    done = load_done(out_path)
    print(f"\n=== Congress {congress} (bulk), {len(done)} bills already saved ===")
    with open(out_path, "a") as out:
        for bill_type in bill_types:
            print(f"[{congress}] bill_type: {bill_type}")
            zip_path = download_zip(congress, bill_type, cache_dir)
            if zip_path is None:
                continue
            count = 0
            for record in parse_zip(zip_path, bill_type):
                key = (record["bill_type"], record["number"])
                if key in done:
                    continue
                out.write(json.dumps(record) + "\n")
                done.add(key)
                count += 1
            out.flush()
            print(f"  {count} new bills for {bill_type}")
    print(f"[{congress}] total bills on disk: {len(done)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--congress", type=int, action="append", required=True,
                    help="repeatable, e.g. --congress 117 --congress 118; "
                        "bulk data covers the 108th Congress onward")
    ap.add_argument("--bill-types", default=",".join(BILL_TYPES),
                    help="comma-separated subset of hr,s,hjres,sjres")
    ap.add_argument("--cache-dir", default="data/bulk",
                    help="where downloaded zips are cached")
    ap.add_argument("--out-dir", default="data/bills")
    args = ap.parse_args()

    bill_types = args.bill_types.split(",")
    for bt in bill_types:
        if bt not in BILL_TYPES:
            raise SystemExit(f"Unknown bill type '{bt}'. Options: {BILL_TYPES}")

    for congress in args.congress:
        if congress < 108:
            print(f"WARNING: BILLSTATUS bulk data starts at the 108th Congress; "
                 f"{congress} will likely 404. Use download_bills.py for older ones.")
        run_congress(congress, bill_types, Path(args.cache_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
