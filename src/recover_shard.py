"""
Recover a shard's already-fetched pages from a single-machine pages_monthly.jsonl.

Run this on a machine that did an UN-sharded fetch (produced a plain
pages_monthly.jsonl covering a contiguous chunk of the manifest). It pulls out
just the pages that belong to one shard and writes them in the exact
pages_monthly.shardKofN.jsonl format -- so those pages don't get re-downloaded,
and the file can be combined with the other shards later.

Self-contained (stdlib only). Copy it next to your data/ folder and run.

Needs, in the same --out-dir:
    monthly_manifest.csv     (defines shard membership by row order -- must be
                              the SAME manifest all machines used)
    pages_monthly.jsonl      (the single-machine fetch to recover from)

Usage:
    python recover_shard.py --shard 1/4
    python recover_shard.py --shard 3/4              # same file also holds shard-3 pages
    python recover_shard.py --shard 1/4 --out-dir data/monthly
"""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", required=True, help="K/N, e.g. 1/4")
    ap.add_argument("--out-dir", default="data/monthly")
    ap.add_argument("--pages", default=None,
                    help="single-machine fetch file (default: <out-dir>/pages_monthly.jsonl)")
    args = ap.parse_args()

    k, n = (int(x) for x in args.shard.split("/"))
    if not (1 <= k <= n):
        raise SystemExit(f"--shard {args.shard}: need 1 <= K <= N")

    out_dir = Path(args.out_dir)
    manifest = out_dir / "monthly_manifest.csv"
    pages = Path(args.pages) if args.pages else out_dir / "pages_monthly.jsonl"
    for p in (manifest, pages):
        if not p.exists():
            raise SystemExit(f"missing {p} -- run this in/point --out-dir at the "
                             f"folder holding monthly_manifest.csv and pages_monthly.jsonl")

    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    shard_ids = {rows[i]["page_id"] for i in range(len(rows)) if i % n == (k - 1)}
    print(f"shard {k}/{n} owns {len(shard_ids)} manifest pages")

    out_path = out_dir / f"pages_monthly.shard{k}of{n}.jsonl"
    kept, scanned, chars = 0, 0, []
    with open(out_path, "w", encoding="utf-8") as out:
        for line in open(pages, encoding="utf-8"):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            scanned += 1
            if rec.get("source_page_id") in shard_ids:
                out.write(line + "\n")
                kept += 1
                if "n_chars" in rec:
                    chars.append(rec["n_chars"])

    print(f"scanned {scanned} pages in {pages.name}")
    print(f"recovered {kept} that belong to shard {k} -> {out_path.name}")
    if chars:
        med = statistics.median(chars)
        print(f"  median {med:,.0f} chars/page "
              f"({'healthy' if med > 12000 else 'LOOKS TRUNCATED -- check'})")
    print(f"remaining shard-{k} pages to fetch: {len(shard_ids) - kept}")
    print(f"\nNext: download {out_path.name} back to the repo's data/monthly/, OR")
    print(f"run  python scrape_monthly.py --stage fetch --shard {k}/{n}  here to finish it.")


if __name__ == "__main__":
    main()
