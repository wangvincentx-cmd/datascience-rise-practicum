"""
Parse a ProQuest TDM Studio dataset (a folder of one-XML-per-article) into the
raw JSONL contract, making ProQuest full text a drop-in third source alongside
loc and nyt. Two arms share this parser, same as the two downloaders:

  --arm elections   one dataset per presidential cycle (pass --cycle)
  --arm economy     one dataset per crisis/placebo window in
                    data/windows_economy.csv (pass --window)

WHERE THIS RUNS: inside the TDM Studio Jupyter workbench. Full text cannot leave
the VM, so this and extract_predictions.py both run there; only the derived
data/predictions/pred_*.jsonl (no ocr_text) is exported. See the project plan.

Why bother: ProQuest's <FullText> is the whole article body, whereas the NYT
Article Search API (download_nyt.py) returns headline+abstract+lead+snippet only.
So ocr_text here is far richer than the nyt source over the same 1963-2010 era.

Two dataset shapes, and they produce different output layouts:

  per-window (--window)  One ProQuest dataset = one window, built in the
                         dashboard for that window's date range with the
                         forecast-catcher query. Mirrors how the downloaders
                         bucket at fetch time.
                         -> data/raw/proquest_economy_{window}.jsonl

  corpus (--corpus)      One ProQuest dataset spans many years (e.g. a single
                         1900-2010 query). No single window applies, so the
                         window is derived per article from its date and is null
                         for the ~90% of years no window covers. Output is
                         sharded by year so the downstream extract/verify stages
                         resume in small units -- which is what makes a corpus
                         of this size survivable against the daily LLM quota.
                         -> data/raw/proquest_economy_{YYYY}.jsonl

Output: data/raw/proquest_{elections|economy}_{cycle_or_window_or_year}.jsonl

VERIFY TAG NAMES FIRST. ProQuest's XML tag names vary by content type. Run with
--inspect on your real dataset before a full parse; it prints the tag matched per
field for a few files. If a field is empty, open one XML in the Jupyter terminal
(not the file browser) or ProQuest's shipped "Getting Started" notebook, find the
real tag, and add it to the *_TAGS / *_XPATHS lists below.

Requires: pip install lxml  (already present in the TDM Studio workbench).

Usage:
  python tdm_parse.py --arm economy --window gfc_2008 \
      --dataset-dir /home/ec2-user/SageMaker/data/MyDataset --inspect
  python tdm_parse.py --arm economy --window gfc_2008 \
      --dataset-dir /home/ec2-user/SageMaker/data/MyDataset
  python tdm_parse.py --arm elections --cycle 1980 \
      --dataset-dir /home/ec2-user/SageMaker/data/Elect1980
  python tdm_parse.py --arm economy --corpus \
      --dataset-dir /home/ec2-user/SageMaker/data/econ19002010 --inspect
  python tdm_parse.py --arm economy --corpus \
      --dataset-dir /home/ec2-user/SageMaker/data/econ19002010

Resume-safe: already-parsed GOIDs are skipped on rerun.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from lxml import etree, html

# Candidate tag names, tried in order. Confirm against your dataset with --inspect.
GOID_TAGS = ["GOID", "Goid", "RecordID"]
DATE_TAGS = ["NumericDate", "NumericPubDate", "AlphaPubDate"]  # YYYY-MM-DD or YYYYMMDD
# Article title: any <Title> NOT inside the publication-metadata block (PubFrosting).
TITLE_XPATHS = ["//Title[not(ancestor::PubFrosting)]", "//RecordTitle", "//Title"]
# Publication (newspaper) name: PubFrosting/Title; PublisherName is the company.
PUBTITLE_XPATHS = ["//PubFrosting/Title", "//PubFrosting//Title", "//SortTitle",
                   "//PublisherName"]
BODY_XPATHS = ["//TextInfo/Text", "//FullText", "//Text", "//HiddenText"]

PARSER = etree.XMLParser(recover=True, huge_tree=True)


def _first(getter, candidates):
    """Return (value, matched_key) for the first candidate that yields text."""
    for key in candidates:
        val = getter(key)
        if val:
            return val, key
    return None, None


def _tag_text(root, tags):
    return _first(lambda t: (root.findtext(f".//{t}") or "").strip() or None, tags)


def _xpath_text(root, xpaths):
    def get(xp):
        hits = root.xpath(xp)
        return ("".join(hits[0].itertext()).strip() or None) if hits else None
    return _first(get, xpaths)


def _norm_date(raw):
    """Normalize a ProQuest date string to YYYY-MM-DD, or None if unparseable."""
    if not raw:
        return None
    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10]
    digits = raw.replace("-", "").strip()
    if len(digits) >= 8 and digits[:8].isdigit():
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return None


def _clean_body(raw):
    """Strip any embedded HTML markup from the article body."""
    if raw and "<" in raw and ">" in raw:
        try:
            return html.fromstring(raw).text_content().strip()
        except (etree.ParserError, etree.XMLSyntaxError, ValueError):
            pass
    return raw


def parse_xml(path):
    """Parse one ProQuest XML into field values (any may be None); None on failure."""
    try:
        root = etree.parse(str(path), PARSER).getroot()
    except (etree.XMLSyntaxError, OSError):
        return None
    if root is None:
        return None
    goid, goid_tag = _tag_text(root, GOID_TAGS)
    title, title_tag = _xpath_text(root, TITLE_XPATHS)
    pub, pub_tag = _xpath_text(root, PUBTITLE_XPATHS)
    date_raw, date_tag = _tag_text(root, DATE_TAGS)
    body, body_tag = _xpath_text(root, BODY_XPATHS)
    # Prepend the headline so the forecast in the title survives, as combine_text
    # in download_nyt.py does.
    ocr_text = "\n".join(p for p in (title, _clean_body(body)) if p)
    return {
        "goid": goid, "newspaper_title": pub, "date": _norm_date(date_raw),
        "ocr_text": ocr_text,
        "_matched": {"goid": goid_tag, "title": title_tag,
                     "newspaper_title": pub_tag, "date": date_tag, "body": body_tag},
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
    """window_id -> {kind, start, end} for every crisis/placebo window."""
    windows = {}
    with open("data/windows_economy.csv") as f:
        for row in csv.DictReader(f):
            windows[row["window_id"]] = {"kind": row["kind"],
                                         "start": row["start_date"],
                                         "end": row["end_date"]}
    return windows


def window_for_date(date, windows):
    """(window_id, kind) for the window containing an ISO date, else (None, None).

    Corpus mode only. The windows are disjoint 6-7 month bands covering roughly
    12 of the 110 years a 1900-2010 query spans, so the overwhelming majority of
    articles legitimately match nothing and get a null window. That is not a
    parse failure -- it is the point of the wider corpus. Dates are ISO strings
    normalized by _norm_date, so string comparison is date comparison.
    """
    if not date:
        return None, None
    for window_id, spec in windows.items():
        if spec["start"] <= date <= spec["end"]:
            return window_id, spec["kind"]
    return None, None


def inspect(xml_files):
    """Print the tag matched per field for a few files, then stop."""
    for path in xml_files[:5]:
        fields = parse_xml(path)
        if fields is None:
            print(f"\n{path.name}: FAILED TO PARSE")
            continue
        print(f"\n{path.name}")
        print(f"  matched tags: {fields['_matched']}")
        print(f"  goid={fields['goid']!r} date={fields['date']!r} "
              f"paper={fields['newspaper_title']!r}")
        print(f"  ocr_text[:140]={fields['ocr_text'][:140]!r}")
    missing = [f for f in ("goid", "date", "newspaper_title", "ocr_text")
               if all((parse_xml(p) or {}).get(f) in (None, "")
                      for p in xml_files[:5])]
    if missing:
        print(f"\n>>> Empty across sample: {missing}. Add the real tag names to "
              f"the *_TAGS / *_XPATHS lists at the top of this file.")
    else:
        print("\nAll core fields populated. Safe to run without --inspect.")


def _paper_only(path):
    """Extract just the publication title (skips body parsing, for a fast tally)."""
    try:
        root = etree.parse(str(path), PARSER).getroot()
    except (etree.XMLSyntaxError, OSError):
        return None
    if root is None:
        return None
    pub, _ = _xpath_text(root, PUBTITLE_XPATHS)
    return pub or "<no publication title>"


def count_papers(xml_files):
    """Tally unique newspaper_title across the whole dataset, most common first."""
    counts = Counter()
    unparsed = 0
    for path in xml_files:
        pub = _paper_only(path)
        if pub is None:
            unparsed += 1
        else:
            counts[pub] += 1
    print(f"{len(xml_files)} files, {len(counts)} unique papers"
          + (f", {unparsed} unparsed" if unparsed else ""))
    for paper, n in counts.most_common():
        print(f"  {n:6d}  {paper}")


def run_dataset(arm, window_id, dataset_dir, extra_meta, limit=None):
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"proquest_{arm}_{window_id}.jsonl"
    done = load_done_ids(out_path)
    xml_files = sorted(dataset_dir.rglob("*.xml"))
    print(f"\n=== ProQuest {arm} / {window_id} ({dataset_dir.name}), "
          f"{len(xml_files)} XMLs, {len(done)} already saved ===")

    written = skipped = 0
    with open(out_path, "a") as out:
        for i, path in enumerate(xml_files):
            if limit and i >= limit:
                break
            fields = parse_xml(path)
            if fields is None or not fields["goid"] or not fields["ocr_text"]:
                skipped += 1
                continue
            if fields["goid"] in done:
                continue
            record = {
                "page_id": fields["goid"], "lccn": None,
                "newspaper_title": fields["newspaper_title"], "date": fields["date"],
                "state": None, "city": None, "ocr_text": fields["ocr_text"],
                "source": "proquest", "arm": arm, "window": window_id,
                "matched_phrase": dataset_dir.name, **extra_meta,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            done.add(fields["goid"])
            written += 1
            if written % 100 == 0:
                print(f"  wrote {written}")
    print(f"[{window_id}] wrote {written} new records "
          f"({skipped} skipped unparseable/empty) -> {out_path}")
    print(f"Next: python extract_predictions.py --source proquest "
          f"--arm {arm} --window {window_id}")


# A corpus shard is named for the year it holds, so it can never collide with a
# window id (which always carries a name_year form). Anything else in data/raw is
# from the old per-window keyword datasets.
CORPUS_SHARD_RE = re.compile(r"^proquest_economy_(\d{4}|nodate)\.jsonl$")


def corpus_shards(out_dir):
    return sorted(p for p in out_dir.glob("proquest_economy_*.jsonl")
                  if CORPUS_SHARD_RE.match(p.name))


def load_corpus_done_ids(out_dir):
    """page_ids already written to any corpus year-shard.

    Deliberately ignores the old per-window files (proquest_economy_gulf_1990
    and friends): those came from the narrow forecast-catcher datasets, and an
    article present in both corpora must still be written to its year shard.
    Treating those GOIDs as done would punch window-shaped holes in the corpus.
    """
    done = set()
    for path in corpus_shards(out_dir):
        done |= load_done_ids(path)
    return done


def warn_stale_window_files(out_dir):
    """Corpus and per-window predictions cannot both sit in data/predictions.

    analyze_economy.py globs pred_*_economy_*.jsonl with no window filter, so an
    article that appears in both a keyword window dataset and the corpus would be
    counted twice -- and the old window files are still schema v1, which the
    scorer cannot read at all. Warn loudly rather than moving a teammate's data.
    """
    stale = [p.name for p in out_dir.glob("proquest_economy_*.jsonl")
             if not CORPUS_SHARD_RE.match(p.name)]
    if not stale:
        return
    print(f"\n*** {len(stale)} per-window raw file(s) from the old keyword "
          f"datasets are still here:")
    for name in sorted(stale)[:10]:
        print(f"      {name}")
    if len(stale) > 10:
        print(f"      ... and {len(stale) - 10} more")
    print("*** Their predictions are globbed by analyze_economy.py alongside the")
    print("*** corpus ones, double-counting any shared article. Before scoring:")
    print("***   mkdir -p data/predictions/keyword_v1 && \\")
    print("***     mv data/predictions/pred_proquest_economy_*_*.jsonl "
          "data/predictions/keyword_v1/\n")


def run_corpus(dataset_dir, limit=None):
    """Parse one date-spanning dataset (e.g. a 1900-2010 query) into year shards.

    A dataset covering 110 years cannot be stamped with a single window, so the
    window is derived per article from its date and left null for the majority
    that fall outside the configured bands. Sharding by year is what makes the
    rest of the run tractable: every downstream stage resumes off its own shard's
    output, so the resume scan stays small, and the batch runner gets a natural
    unit to stop on when the daily LLM quota hits.
    """
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = load_economy_windows()
    warn_stale_window_files(out_dir)
    done = load_corpus_done_ids(out_dir)
    xml_files = sorted(dataset_dir.rglob("*.xml"))
    print(f"=== ProQuest economy corpus ({dataset_dir.name}), "
          f"{len(xml_files)} XMLs, {len(done)} already saved ===")

    handles, per_year, per_window = {}, Counter(), Counter()
    written = skipped = nodate = 0
    try:
        for i, path in enumerate(xml_files):
            if limit and i >= limit:
                break
            fields = parse_xml(path)
            if fields is None or not fields["goid"] or not fields["ocr_text"]:
                skipped += 1
                continue
            if fields["goid"] in done:
                continue
            date = fields["date"]
            # An article with no parseable date cannot be scored -- claim_month
            # is what analyze_economy.py joins on. Park those in their own shard
            # so they are counted and inspectable instead of silently dropped;
            # the batch runner leaves that shard alone so no quota is spent on
            # articles whose claims could never be scored.
            shard = date[:4] if date else "nodate"
            nodate += not date
            window_id, kind = window_for_date(date, windows)
            record = {
                "page_id": fields["goid"], "lccn": None,
                "newspaper_title": fields["newspaper_title"], "date": date,
                "state": None, "city": None, "ocr_text": fields["ocr_text"],
                "source": "proquest", "arm": "economy", "window": window_id,
                "window_kind": kind, "matched_phrase": dataset_dir.name,
            }
            if shard not in handles:
                handles[shard] = open(out_dir / f"proquest_economy_{shard}.jsonl",
                                      "a")
            handles[shard].write(json.dumps(record) + "\n")
            handles[shard].flush()
            done.add(fields["goid"])
            written += 1
            per_year[shard] += 1
            per_window[window_id] += bool(window_id)
            if written % 1000 == 0:
                print(f"  wrote {written}")
    finally:
        for handle in handles.values():
            handle.close()

    years = sorted(y for y in per_year if y != "nodate")
    in_window = sum(per_window.values())
    print(f"\nwrote {written} new records ({skipped} skipped unparseable/empty)")
    if years:
        print(f"  {len(years)} year shards, {years[0]}-{years[-1]}")
    if nodate:
        print(f"  {nodate} records had no parseable date -> "
              f"proquest_economy_nodate.jsonl (not extracted; unscorable)")
    if written:
        print(f"  {in_window} records fall inside a configured window "
              f"({len(per_window)} windows); the rest carry a null window and "
              f"are scored continuously against NBER")
    print(f"\nNext: bash run_corpus_economy.sh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["elections", "economy"], required=True)
    ap.add_argument("--cycle", type=int, help="elections: cycle year, e.g. 1980")
    ap.add_argument("--window", help="economy: window_id, e.g. gfc_2008")
    ap.add_argument("--corpus", action="store_true",
                    help="economy: the dataset spans many years (e.g. one "
                         "1900-2010 query) rather than one window. Shards "
                         "output by year and derives each article's window "
                         "from its date. Use INSTEAD of --window.")
    ap.add_argument("--dataset-dir", required=True,
                    help="ProQuest dataset folder, e.g. "
                         "/home/ec2-user/SageMaker/data/MyDataset")
    ap.add_argument("--limit", type=int, default=None, help="max files, for testing")
    ap.add_argument("--inspect", action="store_true",
                    help="parse a few files, print matched tags, write nothing")
    ap.add_argument("--papers", action="store_true",
                    help="tally unique newspaper_title across the whole dataset, write nothing")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"No such dataset dir: {dataset_dir}")

    if args.inspect or args.papers:
        xml_files = sorted(dataset_dir.rglob("*.xml"))
        if not xml_files:
            raise SystemExit(f"No .xml files under {dataset_dir}")
        (count_papers if args.papers else inspect)(xml_files)
        return

    if args.arm == "elections":
        if args.cycle is None:
            raise SystemExit("--cycle is required for --arm elections")
        run_dataset("elections", str(args.cycle), dataset_dir,
                    {"cycle": args.cycle}, args.limit)
    else:
        if args.corpus:
            if args.window:
                raise SystemExit("--corpus and --window are alternatives: "
                                 "--corpus derives the window from each "
                                 "article's date.")
            run_corpus(dataset_dir, args.limit)
            return
        if not args.window:
            raise SystemExit("--window is required for --arm economy "
                             "(or --corpus for a date-spanning dataset)")
        windows = load_economy_windows()
        if args.window not in windows:
            raise SystemExit(f"Unknown window '{args.window}'. "
                             f"Options: {sorted(windows)}")
        run_dataset("economy", args.window, dataset_dir,
                    {"window_kind": windows[args.window]["kind"]}, args.limit)


if __name__ == "__main__":
    main()
