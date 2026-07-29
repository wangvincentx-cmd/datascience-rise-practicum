"""
Combine the per-machine fetch shards into one pages_monthly.jsonl, deduped.

The fetch is split across machines with `scrape_monthly.py --stage fetch
--shard K/N`; each writes pages_monthly.shardKofN.jsonl. This stitches them back
into a single file. Shards are disjoint by construction (row i goes to machine
i % N), so there should be NO overlap -- but this dedups on page_id anyway as a
safety net, and reports if any duplicates were found (which would signal two
machines ran the same shard).

Usage (from the folder holding the shard files):
    python combine_shards.py --in-dir data/monthly --out data/monthly/pages_monthly.jsonl
    python combine_shards.py *.jsonl --out pages_monthly.jsonl   # explicit list
"""

import argparse
import glob
import json
from pathlib import Path


def combine(paths, out_path):
    seen, n_dup, n_bad = set(), 0, 0
    per_file = {}
    with open(out_path, "w", encoding="utf-8") as out:
        for p in paths:
            kept = 0
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    pid = rec["page_id"]
                except (json.JSONDecodeError, KeyError):
                    n_bad += 1
                    continue
                if pid in seen:
                    n_dup += 1
                    continue
                seen.add(pid)
                out.write(line + "\n")
                kept += 1
            per_file[p] = kept
    return per_file, len(seen), n_dup, n_bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="shard files (or use --in-dir)")
    ap.add_argument("--in-dir", default=None,
                    help="folder to glob pages_monthly.shard*.jsonl from")
    ap.add_argument("--out", default="pages_monthly.jsonl")
    args = ap.parse_args()

    paths = list(args.files)
    if args.in_dir:
        paths += sorted(glob.glob(str(Path(args.in_dir) / "pages_monthly.shard*.jsonl")))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no shard files found -- pass them, or --in-dir <folder>")

    print(f"combining {len(paths)} shard file(s):")
    per_file, total, n_dup, n_bad = combine(paths, args.out)
    for p, k in per_file.items():
        print(f"  {k:>6} unique pages from {Path(p).name}")
    print(f"\n{total} unique pages -> {args.out}")
    if n_dup:
        print(f"** {n_dup} duplicate page_ids skipped -- if this is large, two "
              f"machines may have run the SAME --shard K/N. Check the shard "
              f"assignments; the corpus is still correct (dupes were dropped).")
    if n_bad:
        print(f"{n_bad} unparseable lines skipped.")
    # quick health read
    import statistics
    chars = []
    for line in open(args.out, encoding="utf-8"):
        try:
            chars.append(json.loads(line)["n_chars"])
        except Exception:
            pass
    if chars:
        print(f"median {statistics.median(chars):,.0f} chars/page "
              f"(healthy is ~20k-26k; ~7-8k would mean truncation)")
