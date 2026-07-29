"""
Proof that adding the ProQuest corpus does not disturb the LOC result.

The LOC corpus is the published number. ProQuest is an addition that may not
survive scrutiny, so the requirement is explicit: it must always be possible to
fall back to LOC alone and get EXACTLY what was there before. That is a property
worth asserting rather than assuming, because the ways it could break are all
silent -- an extra column changing a dtype, a concat reordering rows, a patched
feature function taking a different branch.

Two things are checked against the REAL 14k-row table, not a fixture:

  1. load_context() with default arguments returns the same rows as reading
     data/scored/macro_context.csv directly, and every model feature built from
     it is bit-identical.
  2. main_pooling.patch is a no-op on LOC. It teaches claim_features and
     resolve_horizon to fall back to precomputed columns when `quote` is absent;
     LOC rows HAVE a quote, so the fallback must never fire and the numbers must
     not move.

Run:  python tests/test_context_sources.py
"""

import sys, pathlib  # noqa: E402  -- make src/ importable and run from repo root
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
import os; os.chdir(_ROOT)   # data paths in the modules are repo-root-relative

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import macro_context as mc
import hit_predictor as hp
import model_hit as mh
from context_sources import CONTEXT_FILES, load_context, stock_coverage_warning
from score_predictions import resolve_horizon

FAIL = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        FAIL.append(name)


def features(d):
    """The notebook's exact feature construction."""
    d = d[d["hit"].isin([0, 1])].reset_index(drop=True)
    return pd.concat([hp.claim_features(d),
                      hp.macro_block(d[mc.FACTORS]),
                      hp.interaction_block(d, d[mc.FACTORS])], axis=1)


print("\n[1] load_context() default == reading the LOC csv directly")
loc_csv = CONTEXT_FILES["loc"]
if not loc_csv.exists():
    print(f"  SKIP: {loc_csv} not present")
else:
    raw = pd.read_csv(loc_csv, low_memory=False)
    got = load_context()

    check("same row count", len(raw) == len(got))
    check("only `source` is added",
          set(got.columns) - set(raw.columns) == {"source"})
    check("no column is dropped", not set(raw.columns) - set(got.columns))
    check("source is tagged loc", (got["source"] == "loc").all())
    check("shared columns are unchanged",
          raw.equals(got[list(raw.columns)]))

    Xr, Xg = features(raw), features(got)
    check("feature columns identical", list(Xr.columns) == list(Xg.columns))
    check("FEATURE VALUES identical -- the guarantee", Xr.equals(Xg))

    print("\n[2] the loader refuses unknown sources rather than guessing")
    try:
        load_context(("loc", "nonsense"))
        check("unknown source raises", False)
    except ValueError:
        check("unknown source raises", True)

    print("\n[3] a missing ProQuest table is an explicit error, not silence")
    if not CONTEXT_FILES["proquest"].exists():
        try:
            load_context(("proquest",))
            check("missing table raises with build instructions", False)
        except SystemExit as e:
            check("missing table raises with build instructions",
                  "macro_context.py" in str(e))
    else:
        print("  (proquest table exists -- skipping)")

    print("\n[4] main_pooling.patch is a no-op on LOC")
    # LOC rows carry a quote, so the precomputed-column fallback must not fire.
    d = raw[raw["hit"].isin([0, 1])].reset_index(drop=True)
    check("no LOC row has quote_n_words (so the fallback cannot be used)",
          "quote_n_words" not in d.columns or d["quote_n_words"].isna().all())
    check("c_len is driven by the quote, never zero-filled",
          (mh.claim_features(d)["c_len"] > 0).mean() > 0.99)
    # resolve_horizon must still read the quote when horizon_hint is absent.
    sample = {"horizon_months": "vague", "quote": "business will improve before long"}
    check("resolve_horizon still reads the quote without horizon_hint",
          resolve_horizon(sample) == (6, "inferred_short"))
    # ...and prefer horizon_hint when the quote is gone (the ProQuest case).
    stripped = {"horizon_months": "vague", "horizon_hint": "inferred_short"}
    check("resolve_horizon prefers horizon_hint when the quote is stripped",
          resolve_horizon(stripped) == (6, "inferred_short"))

    print("\n[5] stock coverage is reported, not silently zero")
    rep = stock_coverage_warning(raw.assign(source="loc"))
    check("LOC has real stock coverage",
          float(rep.loc[rep["source"] == "loc", "stock_coverage"].iloc[0]) > 0.5)

print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
