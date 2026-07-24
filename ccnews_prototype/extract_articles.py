#!/usr/bin/env python3
"""Extract clean article text from one CC-News WARC file.

Reads a local .warc.gz, pulls the main body text (boilerplate stripped) and
metadata out of each HTML response with trafilatura, and appends one JSON
object per article to an output JSONL file.

Usage:
    python extract_articles.py INPUT.warc.gz OUTPUT.jsonl [--domains domains.txt]

--domains: optional file, one substring per line (e.g. "nytimes.com").
           If given, only URLs containing one of those substrings are kept.
           Omit it on the first test run to see the full outlet mix.
"""
import argparse
import json
import sys

from warcio.archiveiterator import ArchiveIterator
import trafilatura


def load_domains(path):
    if not path:
        return None
    with open(path) as f:
        return [line.strip().lower() for line in f if line.strip()]


def url_allowed(url, domains):
    if domains is None:
        return True
    u = url.lower()
    return any(d in u for d in domains)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("warc")
    ap.add_argument("out")
    ap.add_argument("--domains", default=None)
    ap.add_argument("--min-chars", type=int, default=300,
                    help="skip articles with less body text than this")
    ap.add_argument("--max-records", type=int, default=0,
                    help="stop after this many HTML responses (0 = no limit); "
                         "use a small value like 500 for a fast test run")
    args = ap.parse_args()

    domains = load_domains(args.domains)
    kept = seen = 0

    with open(args.warc, "rb") as stream, open(args.out, "a") as out:
        for record in ArchiveIterator(stream):
            if record.rec_type != "response":
                continue
            url = record.rec_headers.get_header("WARC-Target-URI") or ""
            if not url_allowed(url, domains):
                continue
            ctype = (record.http_headers.get_header("Content-Type") or "") if record.http_headers else ""
            if "html" not in ctype.lower():
                continue
            seen += 1
            if args.max_records and seen > args.max_records:
                break
            try:
                html = record.content_stream().read()
                data = trafilatura.bare_extraction(
                    html, url=url, with_metadata=True,
                    include_comments=False, include_tables=False,
                    as_dict=True,   # trafilatura 2.x returns a Document obj otherwise
                )
            except Exception:
                continue
            if not data:
                continue
            text = (data.get("text") or "").strip()
            if len(text) < args.min_chars:
                continue
            rec = {
                "url": url,
                "domain": data.get("hostname"),
                "date": data.get("date"),        # trafilatura's best-guess publish date
                "title": data.get("title"),
                "n_chars": len(text),
                "text": text,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"{args.warc}: {kept} kept / {seen} html responses", file=sys.stderr)


if __name__ == "__main__":
    main()
