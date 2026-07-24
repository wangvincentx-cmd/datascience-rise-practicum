"""
Continuous monthly LOC scrape, 1900-1963 -- the corpus the press index needs.

SELF-CONTAINED ON PURPOSE. This file imports nothing from the rest of the repo
so it can be copied to another machine and run on its own; see
PORTABLE_SCRAPE.md. It is a ~24 hour job, so it belongs on a machine that stays
on, not on a laptop.

Why a new scraper at all
------------------------
`newspaper_scraper.py` samples 19 disjoint episode windows chosen BECAUSE of
what the economy did, using per-episode phrase lists that differ between
episodes. That is fine for "was this claim right"; it cannot support a time
series. You cannot plot press expectations over time from blocks with
decade-long gaps, and any correlation with macro data is confounded by the fact
that the windows were picked on the outcome.

This samples every month from 1900-01 to 1963-12 on identical terms, so
month-to-month variation reflects the press and not the sampling.

Three design rules, each fixing a specific defect in the episode corpus:

1. **The query set is FIXED and DIRECTION-NEUTRAL.** The v1 terms include
   "return of prosperity", "business revival", "worst is over" and "economic
   recovery" -- all of which select optimistic copy -- while the headline v1
   finding is that forecasts leaned optimistic. Worse, the gold pages show
   "return of prosperity" matching bank New Year ADVERTISEMENTS (see
   gold_extraction/RESULTS.md), so a December window preferentially retrieves
   seasonal boosterism. Every term below is neutral about direction.

2. **The DENOMINATOR is recorded for every month.** LOC's digitised page count
   swings by an order of magnitude across these 64 years. Without total pages
   available and total query hits, a monthly claim count measures digitisation
   density, not the press. Every index series must be a share or a rate, and
   this is where the denominator comes from.

3. **A fixed publisher panel is flagged.** Publisher composition is a live
   confound -- DC alone is ~29% of the v1 corpus. Papers with near-continuous
   coverage are marked so the index can be recomputed on a balanced panel.

Two stages
----------
    search   query LOC month by month     -> monthly_manifest.csv + denominators
    fetch    download OCR for the manifest -> pages_monthly.jsonl

Usage:
    python scrape_monthly.py --stage search          # ~4-6 h
    python scrape_monthly.py --stage fetch           # ~16-24 h
    python scrape_monthly.py --stage both            # sequential, the usual call
    python scrape_monthly.py --stage search --start 1929-01 --end 1929-12   # test

Resumable and safe to interrupt at any point: every HTTP response is cached on
disk, completed months are skipped via the manifest, and already-fetched pages
are skipped via the output file. Just rerun the same command.

No API key required -- loc.gov is public. Politeness: ~1.1 s/request.
"""

import argparse
import csv
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.loc.gov/collections/chronicling-america/"
HEADERS = {"User-Agent": "BU-RISE-student-research/0.2 "
                        "(economic prediction accuracy study)"}
SLEEP_SECONDS = 1.6  # ~37 req/min -- polite enough to avoid LOC's burst 429s
COOLDOWN_AFTER_FAIL = 120  # pause this long after a month is rate-limited out

# Held constant across all 768 months. None of these implies a direction.
# "business outlook" is kept from v1 -- it is neutral (an outlook can be good or
# bad) and was v1's highest-yield term; the loaded ones are dropped.
NEUTRAL_TERMS = [
    "business conditions",
    "business outlook",
    "trade conditions",
    "the business situation",
    "financial outlook",
]

# Papers with long continuous LOC runs. LCCNs, not titles, because titles change
# spelling across decades.
PANEL_LCCNS = {
    "sn83045462": "evening star (washington, d.c.)",
    "sn85038615": "new-york tribune / herald tribune",
    "sn84026749": "the washington times",
    "sn83030214": "new-york daily tribune",
    "sn86063034": "the sun (new york)",
}


def _ssl_context():
    """Verifying SSL context that survives a TLS-intercepting proxy.

    Corporate TLS inspection and some antivirus products present a re-signed
    chain whose root is in the OS trust store but NOT in certifi's bundle, so
    every HTTPS host fails with CERTIFICATE_VERIFY_FAILED. `truststore`
    delegates to the OS store, which fixes it while STILL VERIFYING. If the
    machine has no such proxy this changes nothing."""
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = _ssl_context()
CACHE_DIR = Path("cache")


def _get(url, cache=True):
    """Fetch with disk caching, politeness delay, and retry with backoff."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", url)[-150:] + ".bin")
    if cache and cache_file.exists():
        return cache_file.read_bytes()
    last_err = None
    for attempt in range(5):
        try:
            time.sleep(SLEEP_SECONDS)
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120,
                                        context=SSL_CONTEXT) as resp:
                data = resp.read()
                # Verify the body is complete. LOC drops connections mid-body
                # often enough that a short read would otherwise be cached as if
                # it were the real response, and every later run would reuse the
                # truncated copy from disk without ever retrying.
                declared = resp.headers.get("Content-Length")
                if declared and len(data) < int(declared):
                    raise IOError(f"truncated: {len(data)} of {declared} bytes")
            if cache:
                cache_file.write_bytes(data)
            return data
        except urllib.error.HTTPError as e:
            # 404 is how this API says "you have paged past the last result".
            # It is a definitive answer, not a transient failure, and
            # search_month() relies on it to stop paging. Retrying it burns
            # 5+10+15+20+25 = 75 seconds for nothing, every single time -- which
            # across 768 months x 5 terms is many hours of pure waiting.
            if e.code == 404:
                raise RuntimeError(f"404 (no more results): {url}") from e
            last_err = e
            if e.code == 429:
                # Rate limited. LOC's penalty window is minutes, not seconds, so
                # a 5-25s backoff never clears it -- it just burns the retries and
                # the whole month fails. Honor Retry-After if sent, else back off
                # 30 -> 60 -> 120 -> 240 -> 480 (capped 600). If even that is not
                # enough the month is skipped and retried on the next run.
                wait = int(e.headers.get("Retry-After") or 0) or min(600, 30 * 2 ** attempt)
            else:
                wait = 5 * (attempt + 1)
            print(f"    retry {attempt + 1}/5: HTTP {e.code} (waiting {wait}s)",
                  flush=True)
            time.sleep(wait)
        except Exception as e:
            last_err = e
            wait = 5 * (attempt + 1)
            print(f"    retry {attempt + 1}/5: {type(e).__name__}: "
                  f"{str(e)[:90]} (waiting {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed after retries: {url} ({last_err})")


def _query(params):
    return json.loads(_get(BASE + "?" + urllib.parse.urlencode(params)))


def months(start, end):
    """Yield ('YYYY-MM', first_day, last_day) inclusive."""
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
        last = [31, 29 if leap else 28, 31, 30, 31, 30,
                31, 31, 30, 31, 30, 31][m - 1]
        yield f"{y:04d}-{m:02d}", f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{last:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def month_total_pages(start_date, end_date):
    """Every digitised page LOC holds for this month, regardless of content.

    `searchType=advanced` with an EMPTY `qs` is required: without it the API
    ignores start_date/end_date and returns the whole collection (23,745,587
    for every month), which is silently useless as a denominator.

    `at=pagination` is required for a different reason: without it LOC returns
    the full search payload -- facets, breadcrumbs, the lot -- which is ~1.9 MB
    even at c=1, and the connection routinely dies mid-body with IncompleteRead.
    Asking for just the pagination subtree returns ~1 KB with the identical
    answer. 1,750x less data, and the truncation stops happening."""
    data = _query({"qs": "", "searchType": "advanced", "dl": "page",
                   "start_date": start_date, "end_date": end_date,
                   "fo": "json", "c": 1, "sp": 1, "at": "pagination"})
    return (data.get("pagination") or {}).get("of", 0) or 0


def search_month(phrase, start_date, end_date, max_pages):
    """Page hits for one phrase in one month, plus that phrase's total hits."""
    collected, sp, total = [], 1, None
    while len(collected) < max_pages:
        try:
            data = _query({"qs": phrase, "ops": "PHRASE", "searchType": "advanced",
                           "dl": "page", "start_date": start_date,
                           "end_date": end_date, "fo": "json",
                           "c": min(100, max_pages), "sp": sp,
                           "at": "results,pagination"})
        except RuntimeError as e:
            if "404" in str(e):
                break  # paged past the end
            raise
        if total is None:
            total = (data.get("pagination") or {}).get("of", 0) or 0
        results = data.get("results") or []
        if not results:
            break
        for r in results:
            if len(collected) >= max_pages:
                break
            collected.append(r)
        sp += 1
    return collected, (total or 0)


def _first(v):
    return (v[0] if v else None) if isinstance(v, list) else v


def stage_search(args, out_dir):
    manifest_path = out_dir / "monthly_manifest.csv"
    denom_path = out_dir / "monthly_denominators.csv"

    # "Done" is read from the DENOMINATORS file, not the manifest: every month
    # that completes writes denominator rows (one per term) even if it found
    # zero pages, whereas a legitimately sparse month writes no manifest rows.
    # Keying on the manifest would retry every empty month forever; keying on
    # denominators marks exactly the months that actually finished. Failed
    # months (atomic) write neither file, so they correctly reappear as not-done.
    done = set()
    if denom_path.exists() and not args.overwrite:
        with open(denom_path, encoding="utf-8") as fh:
            done = {row["month"] for row in csv.DictReader(fh)}
        print(f"resuming search: {len(done)} months already done")

    fresh = args.overwrite or not denom_path.exists()
    mode = "w" if args.overwrite else "a"
    all_months = list(months(args.start, args.end))

    with open(manifest_path, mode, newline="", encoding="utf-8") as mf, \
         open(denom_path, mode, newline="", encoding="utf-8") as df:
        mw, dw = csv.writer(mf), csv.writer(df)
        if fresh:
            mw.writerow(["month", "page_id", "date", "publisher", "state",
                         "lccn", "matched_term", "in_panel"])
            dw.writerow(["month", "total_pages_digitised", "term", "term_hits",
                         "pages_taken"])

        n_failed = 0
        for i, (month, d0, d1) in enumerate(all_months, 1):
            if month in done:
                continue
            per_term = max(1, args.pages_per_month // len(NEUTRAL_TERMS))
            # Build the WHOLE month in memory first; only commit to disk if every
            # query succeeded. A month is therefore atomic: fully written (and so
            # skipped next run) or written nowhere (and so retried next run). Any
            # real failure -- a 429 that outlasts the backoff, a dropped
            # connection -- aborts the month cleanly instead of crashing the run
            # or, worse, recording a rate-limited month as "done" with terms
            # missing. This is what makes recoverable failures actually recover.
            try:
                total_pages = month_total_pages(d0, d1)
                seen, month_rows, denom_rows = set(), [], []
                for term in NEUTRAL_TERMS:
                    hits, term_total = search_month(term, d0, d1, per_term)
                    taken = 0
                    for h in hits:
                        pid = h.get("id", "")
                        if not pid or pid in seen:
                            continue
                        seen.add(pid)
                        lccn = _first(h.get("number_lccn"))
                        month_rows.append([month, pid, h.get("date", ""),
                                     "; ".join(h.get("partof_title") or [])[:120],
                                     "; ".join(h.get("location_state") or []),
                                     lccn or "", term,
                                     int(bool(lccn) and lccn in PANEL_LCCNS)])
                        taken += 1
                    denom_rows.append([month, total_pages, term, term_total, taken])
            except RuntimeError as e:
                n_failed += 1
                print(f"[{i}/{len(all_months)}] {month}: SKIPPED "
                      f"({str(e)[:60]}) -- will retry on next run. "
                      f"Cooling down {COOLDOWN_AFTER_FAIL}s.", flush=True)
                time.sleep(COOLDOWN_AFTER_FAIL)
                continue

            for r in month_rows:
                mw.writerow(r)
            for r in denom_rows:
                dw.writerow(r)
            mf.flush()
            df.flush()
            print(f"[{i}/{len(all_months)}] {month}: {len(month_rows):>3} pages "
                  f"of {total_pages:,} digitised", flush=True)

        if n_failed:
            print(f"\n{n_failed} month(s) were skipped on rate-limit/network "
                  f"errors. Just run the SAME command again -- completed months "
                  f"are skipped and only the skipped ones are retried.")

    print(f"\nmanifest      -> {manifest_path}")
    print(f"denominators  -> {denom_path}")


def parse_fulltext(raw):
    """Pull real OCR text out of a word-coordinates-service payload.

    The payload is {"<segment>": {"full_text": "..."}}. Decoding it as one
    string (what the v1 scraper did) leaves a literal JSON prefix on every page
    and turns newlines into backslash-n, which corrupts everything downstream."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        doc = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text.strip()
    if not isinstance(doc, dict):
        return text.strip()
    parts = [seg["full_text"] for _, seg in sorted(doc.items())
             if isinstance(seg, dict) and seg.get("full_text")]
    return "\n".join(parts).strip() if parts else text.strip()


def stage_fetch(args, out_dir):
    manifest_path = out_dir / "monthly_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing -- run --stage search first")
    out_path = out_dir / "pages_monthly.jsonl"

    done = set()
    if out_path.exists() and not args.overwrite:
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["source_page_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resuming fetch: {len(done)} pages already downloaded")

    with open(manifest_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    todo = [r for r in rows if r["page_id"] not in done]
    print(f"{len(todo)} pages to fetch (of {len(rows)} in manifest)")

    mode = "w" if args.overwrite else "a"
    n_ok = n_skip = 0
    with open(out_path, mode, encoding="utf-8") as out:
        for i, r in enumerate(todo, 1):
            url = r["page_id"].replace("http://", "https://") + "&fo=json"
            try:
                res = json.loads(_get(url))
            except (RuntimeError, json.JSONDecodeError) as e:
                print(f"  skip {r['page_id']}: {e}", flush=True)
                n_skip += 1
                continue
            ft = (res.get("resource") or {}).get("fulltext_file")
            if not ft:
                n_skip += 1
                continue
            try:
                text = parse_fulltext(_get(ft))
            except RuntimeError as e:
                print(f"  skip OCR {r['page_id']}: {e}", flush=True)
                n_skip += 1
                continue
            if not text:
                n_skip += 1
                continue
            item = res.get("item") or {}
            out.write(json.dumps({
                "page_id": res.get("segment_id") or r["page_id"],
                "source_page_id": r["page_id"],
                "month": r["month"],
                "date": item.get("date") or r["date"],
                "publisher": _first(item.get("partof_title")) or r["publisher"],
                "state": "; ".join(item.get("location_state") or []) or r["state"],
                "lccn": r["lccn"],
                "matched_term": r["matched_term"],
                "in_panel": int(r["in_panel"]),
                "n_chars": len(text),
                "ocr_text": text,
            }, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += 1
            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] fetched={n_ok} skipped={n_skip}",
                      flush=True)

    print(f"\n{n_ok} pages -> {out_path}  ({n_skip} skipped)")
    print("Bring pages_monthly.jsonl (gzipped) back to the repo; see "
          "PORTABLE_SCRAPE.md.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["search", "fetch", "both"], default="both")
    ap.add_argument("--start", default="1900-01")
    ap.add_argument("--end", default="1963-12")
    ap.add_argument("--pages-per-month", type=int, default=30)
    ap.add_argument("--out-dir", default="data/monthly")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"python {sys.version.split()[0]} | out-dir {out_dir.resolve()}")
    if args.stage in ("search", "both"):
        stage_search(args, out_dir)
    if args.stage in ("fetch", "both"):
        stage_fetch(args, out_dir)
