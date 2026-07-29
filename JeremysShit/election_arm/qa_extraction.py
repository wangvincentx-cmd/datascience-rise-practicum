"""
Sanity-check a v2 extraction before anyone spends time grading it.

Human kappa validation costs two people several hours. This costs a second, and
it catches the failures that would make that time worthless: the model collapsing
to one label, refusing everything, ignoring the schema, or producing horizons
that are all default. Run it right after an extraction finishes.

It reports distributions, not opinions, and compares the ones that have a known
expectation to the reference numbers in main's docs/SCORING.md:

  scope    gold sample was ~68% national, 21% industry, 9% foreign, 3% regional
  rigid    claims whose horizon comes from the CLAIM rather than a default;
           SCORING.md expects ~45-55% on a good extraction (the old regex+grade
           pipeline managed ~24%)

A FLAG is not proof of a bug -- these are priors from a different corpus (LOC
1900-1963 vs ProQuest 1965-2020) and a different model. It means "look at this
before you trust it".

Usage (in the VM, from election_arm/):
    python qa_extraction.py                          # every v2 pred file
    python qa_extraction.py --window gulf_1990
    python qa_extraction.py --window gulf_1990 --show 5   # print sample quotes
"""

import argparse
import glob
import json
from collections import Counter

# Fields the scorers and the model actually read. A missing one is a real defect.
REQUIRED = ["quote", "topic", "direction", "scope", "confidence", "voice",
            "horizon_months", "horizon_hint", "quote_n_words",
            "quote_has_number", "speaker_name", "is_quoted_forecaster", "date"]

VOCAB = {
    "direction": {"improve", "worsen", "no_change", "unclear"},
    "topic": {"general_business", "prices", "employment", "markets", "other"},
    "scope": {"national", "regional", "foreign", "industry"},
    "confidence": {"assertive", "hedged"},
    "voice": {"journalist", "expert", "official", "layperson", "unclear"},
}
SCOPE_REFERENCE = {"national": 0.68, "industry": 0.21, "foreign": 0.09,
                   "regional": 0.03}


def load(window=None):
    claims, empties, v1 = [], 0, 0
    pattern = f"data/predictions/pred_*_economy_{window or '*'}.jsonl"
    files = [p for p in sorted(glob.glob(pattern))
             if not p.endswith(".export.jsonl")]
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("schema_version") != 2:
                    v1 += 1
                    continue
                if r.get("no_predictions"):
                    empties += 1
                else:
                    claims.append(r)
    return files, claims, empties, v1


def pct(n, d):
    return f"{n / d:.1%}" if d else "n/a"


def dist(claims, field, flags):
    """Print a field's distribution and flag anything degenerate."""
    vals = [str(c.get(field, "<missing>")) for c in claims]
    counts = Counter(vals)
    print(f"\n  {field}")
    for v, n in counts.most_common():
        bar = "#" * max(1, int(40 * n / len(vals)))
        print(f"    {v:<20} {n:>6}  {pct(n, len(vals)):>6}  {bar}")
    known = VOCAB.get(field)
    if known:
        off = sorted(set(vals) - known - {"<missing>"})
        if off:
            flags.append(f"{field}: values outside the schema vocabulary: "
                         f"{', '.join(repr(o) for o in off[:5])}")
        if "<missing>" in counts:
            flags.append(f"{field}: {counts['<missing>']} claims are missing it")
    if len(counts) == 1 and len(claims) > 20:
        flags.append(f"{field}: EVERY claim has the same value "
                     f"({vals[0]!r}) -- the model may have collapsed")
    return counts


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", help="one window_id; default all")
    ap.add_argument("--show", type=int, default=0,
                    help="also print N sample quotes (VM only -- article text)")
    args = ap.parse_args()

    files, claims, empties, v1 = load(args.window)
    if not files:
        raise SystemExit("No prediction files found. Check --window, and run "
                         "this from election_arm/.")
    print("=" * 68)
    print(f"v2 EXTRACTION QA  ({len(files)} file(s))")
    print("=" * 68)
    for f in files:
        print(f"  {f}")
    pages = len(claims) and len(set(c.get("page_id") for c in claims)) or 0
    print(f"\n  claims                 {len(claims)}")
    print(f"  pages with >=1 claim   {pages}")
    print(f"  pages with none        {empties}   "
          f"({pct(empties, pages + empties)} of pages read)")
    if v1:
        print(f"  v1 records SKIPPED     {v1}  (not part of this report)")
    if not claims:
        raise SystemExit("\nNo v2 claims at all. Either the extraction has not "
                         "run yet, or every page came back empty.")

    flags = []

    # Missing required fields is the one hard error here.
    for field in REQUIRED:
        missing = sum(1 for c in claims if field not in c)
        if missing:
            flags.append(f"{field}: absent on {missing}/{len(claims)} claims "
                         f"-- downstream scoring/model will break")

    for field in ("direction", "topic", "scope", "confidence", "voice"):
        counts = dist(claims, field, flags)
        if field == "scope":
            nat = counts.get("national", 0) / len(claims)
            if nat > 0.95:
                flags.append(f"scope: {nat:.0%} national. The gold sample was "
                             f"~68%; near-100% suggests the model is not really "
                             f"judging scope, which disables the scope gate.")

    # Horizon: the RIGID share is what the accuracy model rests on.
    print("\n  horizon_hint (RIGID = anything but 'default')")
    hc = Counter(str(c.get("horizon_hint", "<missing>")) for c in claims)
    for v, n in hc.most_common():
        print(f"    {v:<20} {n:>6}  {pct(n, len(claims)):>6}")
    rigid = len(claims) - hc.get("default", 0) - hc.get("<missing>", 0)
    print(f"    -> RIGID subset      {rigid:>6}  {pct(rigid, len(claims)):>6}"
          f"   (SCORING.md expects ~45-55%)")
    if rigid / len(claims) < 0.25:
        flags.append(f"rigid share is {pct(rigid, len(claims))}; SCORING.md "
                     f"expects ~45-55%. A low value means the model rarely "
                     f"states a horizon, which shrinks the accuracy model's "
                     f"honest denominator.")

    # Quote-derived model features.
    lens = [c.get("quote_n_words", 0) for c in claims if isinstance(
        c.get("quote_n_words"), int)]
    if lens:
        lens_sorted = sorted(lens)
        med = lens_sorted[len(lens_sorted) // 2]
        nums = sum(1 for c in claims if c.get("quote_has_number"))
        print(f"\n  quote_n_words          min {min(lens)}  median {med}  "
              f"max {max(lens)}")
        print(f"  quote_has_number       {nums}  ({pct(nums, len(claims))})")
        if max(lens) > 80:
            print("    (note: main clips c_len at 80, so longer quotes are "
                  "capped there, not here)")
        if med <= 3:
            flags.append(f"median quote is {med} words -- suspiciously short; "
                         f"check that quotes are real spans, not fragments")

    if args.show:
        print(f"\n  --- {min(args.show, len(claims))} sample claims "
              f"(ARTICLE TEXT -- do not export) ---")
        for c in claims[:args.show]:
            print(f"\n    \"{c.get('quote', '')}\"")
            print(f"      {c.get('direction')} / {c.get('topic')} / "
                  f"scope={c.get('scope')} / {c.get('confidence')} / "
                  f"voice={c.get('voice')} / horizon={c.get('horizon_months')} "
                  f"({c.get('horizon_hint')})")

    print("\n" + "=" * 68)
    if flags:
        print(f"{len(flags)} THING(S) TO LOOK AT")
        print("=" * 68)
        for f in flags:
            print(f"  - {f}")
        print("\nNone of these is proof of a bug -- the reference numbers come "
              "from\na different corpus and model. Eyeball some claims with "
              "--show before\ncommitting two people to grading:")
        print("  python sample_claims.py --n 20 --window <w>")
    else:
        print("No flags. Distributions look reasonable.")
        print("=" * 68)
        print("\nNext: human validation (two graders, in the VM)")
        print("  python validate_kappa.py sample --arm economy --source proquest")


if __name__ == "__main__":
    main()
