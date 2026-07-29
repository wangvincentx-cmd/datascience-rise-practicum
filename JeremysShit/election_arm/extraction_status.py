"""
Where does the ProQuest extraction stand, window by window?

One table answering the three questions you ask after every batch run:

  how many CLAIMS does each window have?
  how many pages came back with NO PREDICTIONS?
  how many claims did the verifier DROP?

It only counts what is on disk. Nothing here calls an API, so it is free and
safe to run mid-batch -- including right after a quota stop, to see how far the
run got before it stopped.

Reads three places, all optional:

  data/raw/proquest_economy_<w>.jsonl        pages parsed out of the dataset
  data/predictions/pred_*_economy_<w>.jsonl  extractor output (claims + empties)
  data/verified/pred_*_economy_<w>.jsonl     verifier keeps
    ...jsonl.dropped.jsonl                   verifier rejects

On the Mac the same numbers come out of the exported tarball, which unpacks to
verified/ and unverified/ -- point --base at the folder you unpacked it into.
The .dropped.jsonl is not in the tarball, so there `dropped` is DERIVED as
unverified minus verified and marked `~`; that number also absorbs any claims
the verifier never judged, so it is an upper bound, not a count.

Usage (from election_arm/):
    python extraction_status.py
    python extraction_status.py --window gulf_1990
    python extraction_status.py --base ~/Downloads/proquest_exports   # Mac
"""

import argparse
import glob
import json
import os
from pathlib import Path

SCHEMA_VERSION = 2
WINDOWS_CSV = "data/windows_economy.csv"


class Tally:
    """What one JSONL file contains, at the current schema version."""

    def __init__(self):
        self.claims = 0
        self.empties = 0          # pages the model read and found nothing in
        self.empty_text = 0       # ...of those, pages that had no text to read
        self.v1 = 0               # old-schema records; not part of any count
        self.bad = 0              # unparseable lines
        self.pages = set()

    def add_file(self, path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    self.bad += 1
                    continue
                if r.get("schema_version") != SCHEMA_VERSION:
                    self.v1 += 1
                    continue
                if r.get("page_id"):
                    self.pages.add(r["page_id"])
                if r.get("no_predictions"):
                    self.empties += 1
                    if r.get("empty_source_text"):
                        self.empty_text += 1
                else:
                    self.claims += 1
        return self


def window_of(path):
    """gulf_1990 out of pred_proquest_economy_gulf_1990[.export][.dropped].jsonl
    and out of proquest_economy_gulf_1990.jsonl."""
    name = Path(path).name
    if "_economy_" not in name:
        return None
    tail = name.split("_economy_", 1)[1]
    for suffix in (".jsonl.dropped.jsonl", ".export.jsonl", ".jsonl"):
        if tail.endswith(suffix):
            return tail[: -len(suffix)]
    return None


def pick(paths):
    """Prefer the full file over its stripped .export copy -- the record counts
    are identical, so this only decides which one we open."""
    full = [p for p in paths if ".export.jsonl" not in p]
    return sorted(full or paths)


def collect(base, subdir, keep):
    """window -> [files] under base/subdir whose names pass `keep`."""
    found = {}
    for path in glob.glob(os.path.join(base, subdir, "*.jsonl")):
        if not keep(Path(path).name):
            continue
        w = window_of(path)
        if w:
            found.setdefault(w, []).append(path)
    return {w: pick(paths) for w, paths in found.items()}


def discover(base):
    """Both layouts at once: the VM/repo tree and an unpacked export tarball.

    A window can legitimately appear in only some of these -- parsed but not
    extracted, extracted but not verified -- and the table says which."""
    is_dropped = lambda n: n.endswith(".dropped.jsonl")
    raw = collect(base, "data/raw", lambda n: n.startswith("proquest_"))
    unver = collect(base, "data/predictions", lambda n: not is_dropped(n))
    ver = collect(base, "data/verified", lambda n: not is_dropped(n))
    dropped = collect(base, "data/verified", is_dropped)
    # Exported tarball: verified/ and unverified/ at the top level.
    for w, p in collect(base, "unverified", lambda n: not is_dropped(n)).items():
        unver.setdefault(w, p)
    for w, p in collect(base, "verified", lambda n: not is_dropped(n)).items():
        ver.setdefault(w, p)
    return raw, unver, ver, dropped


def window_order(windows, base="."):
    """Chronological if windows_economy.csv is findable, else alphabetical."""
    known = []
    for csv in (Path(base) / WINDOWS_CSV, Path(WINDOWS_CSV)):
        if csv.exists():
            with open(csv) as f:
                next(f, None)
                known = [line.split(",", 1)[0] for line in f if line.strip()]
            break
    rank = {w: i for i, w in enumerate(known)}
    return sorted(windows, key=lambda w: (rank.get(w, len(rank)), w))


def count_pages(paths):
    """None, not 0, when the parsed pages are simply not here -- an exported
    tarball has no data/raw, and 0 there would read as 'nothing was parsed'."""
    if not paths:
        return None
    n = 0
    for path in paths:
        with open(path) as f:
            n += sum(1 for line in f if line.strip())
    return n


def row_for(w, raw, unver, ver, dropped):
    """One window's numbers, plus the notes that keep them honest."""
    r = {"window": w, "notes": []}
    r["pages"] = count_pages(raw.get(w, []))

    if w in unver:
        t = Tally()
        for p in unver[w]:
            t.add_file(p)
        r["claims"] = t.claims
        r["no_pred"] = t.empties
        r["empty_text"] = t.empty_text
        r["read"] = len(t.pages)
        if t.v1:
            r["notes"].append(f"{t.v1} v1 records ignored (re-extract to use)")
        if t.bad:
            r["notes"].append(f"{t.bad} unreadable line(s)")
    else:
        r["claims"] = r["no_pred"] = r["empty_text"] = r["read"] = None
        if r["pages"]:
            r["notes"].append("parsed but not extracted")

    r["kept"] = r["dropped"] = r["unjudged"] = None
    r["derived"] = False
    if w in ver:
        kept = Tally()
        for p in ver[w]:
            kept.add_file(p)
        r["kept"] = kept.claims
        if w in dropped:
            d = Tally()
            for p in dropped[w]:
                d.add_file(p)
            r["dropped"] = d.claims
            if r["claims"] is not None:
                left = r["claims"] - kept.claims - d.claims
                r["unjudged"] = max(left, 0)
                if left < 0:
                    r["notes"].append(
                        "verified+dropped exceeds extracted -- a file was "
                        "appended twice; check for a double run")
        elif r["claims"] is not None:
            # Export layout: no .dropped.jsonl to count, so infer it.
            r["dropped"] = max(r["claims"] - kept.claims, 0)
            r["derived"] = True
    elif r["claims"]:
        r["notes"].append("not verified yet")
    return r


def fmt(v, derived=False):
    if v is None:
        return "-"
    return f"~{v}" if derived else str(v)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=".",
                    help="repo/VM folder, or the unpacked export tarball")
    ap.add_argument("--window", help="one window_id; default every one found")
    args = ap.parse_args()

    raw, unver, ver, dropped = discover(args.base)
    windows = set(raw) | set(unver) | set(ver)
    if args.window:
        if args.window not in windows:
            raise SystemExit(
                f"No files for window {args.window!r} under {args.base}.\n"
                f"Found: {', '.join(window_order(windows, args.base)) or '(nothing)'}")
        windows = {args.window}
    if not windows:
        raise SystemExit(
            f"No ProQuest files under {args.base}. Run this from election_arm/, "
            f"or point --base at an unpacked export tarball.")

    rows = [row_for(w, raw, unver, ver, dropped)
            for w in window_order(windows, args.base)]

    head = (f"{'window':<16}{'pages':>7}{'read':>7}{'left':>7}  "
            f"{'claims':>8}{'no-pred':>9}{'(empty)':>9}  "
            f"{'kept':>8}{'dropped':>9}{'drop%':>7}{'unjudged':>10}")
    print("=" * len(head))
    print(f"PROQUEST EXTRACTION STATUS   ({args.base})")
    print("=" * len(head))
    print(head)
    print("-" * len(head))

    total = {k: 0 for k in ("pages", "read", "claims", "no_pred", "empty_text",
                            "kept", "dropped", "unjudged")}
    any_derived = False
    for r in rows:
        left = (r["pages"] - r["read"]
                if r["pages"] and r["read"] is not None else None)
        judged = (r["kept"] or 0) + (r["dropped"] or 0)
        pct = (f"{r['dropped'] / judged:.0%}"
               if r["dropped"] is not None and judged else "-")
        any_derived = any_derived or r["derived"]
        print(f"{r['window']:<16}{fmt(r['pages']):>7}{fmt(r['read']):>7}"
              f"{fmt(left):>7}  {fmt(r['claims']):>8}{fmt(r['no_pred']):>9}"
              f"{fmt(r['empty_text']):>9}  {fmt(r['kept']):>8}"
              f"{fmt(r['dropped'], r['derived']):>9}{pct:>7}"
              f"{fmt(r['unjudged']):>10}")
        for k in total:
            v = r.get(k)
            if v:
                total[k] += v

    print("-" * len(head))
    tleft = total["pages"] - total["read"] if total["pages"] else 0
    tjudged = total["kept"] + total["dropped"]
    tpct = f"{total['dropped'] / tjudged:.0%}" if tjudged else "-"
    print(f"{'TOTAL':<16}{total['pages']:>7}{total['read']:>7}{tleft:>7}  "
          f"{total['claims']:>8}{total['no_pred']:>9}{total['empty_text']:>9}  "
          f"{total['kept']:>8}{total['dropped']:>9}{tpct:>7}"
          f"{total['unjudged']:>10}")

    print("\n  pages     parsed out of the ProQuest dataset (data/raw)")
    print("  read      pages the extractor has a record for; left = still to do")
    print("  no-pred   pages read that yielded no forecast; (empty) = the page "
          "had no text at all")
    print("  dropped   claims the verifier rejected -- kept for audit in "
          "<file>.jsonl.dropped.jsonl")
    print("  unjudged  extracted claims the verifier has not ruled on "
          "(verification stopped early)")
    if any_derived:
        print("\n  ~  derived as extracted minus verified, because no "
              ".dropped.jsonl is present\n     (an export tarball omits it). "
              "It includes any unjudged claims, so read it\n     as an upper "
              "bound on drops.")

    notes = [(r["window"], n) for r in rows for n in r["notes"]]
    if notes:
        print("\nNOTES")
        for w, n in notes:
            print(f"  {w:<16} {n}")


if __name__ == "__main__":
    main()
