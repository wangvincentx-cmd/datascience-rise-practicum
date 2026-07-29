# new_model — the model that can see both corpora

The published model lives in `src/` and `notebooks/hit_model.ipynb` and is
**unchanged**. Nothing here modifies it. This folder is the parallel version, so
the two can be run side by side and the old result stays reproducible if the new
one does not work out.

## What differs — exactly one input

The stock series. The published model uses FRED's `M1109BUSM293NNBR`, an NBER
historical index that **ends 1968-12**. This one uses Robert Shiller's monthly
S&P composite (1871–), the standard long series for this literature.

Everything else is main's code, imported unmodified: the publication lags, the
factor definitions, the leakage discipline, the scorer, the block bootstrap.
`build_context.py` swaps the series by patching `load_fred` at the module
boundary, so `src/` is untouched and `tests/test_context_sources.py` still
passes.

## Why it was necessary

|  | M1109BUSM293NNBR | Shiller S&P |
|---|---|---|
| LOC 1900–1963 | 76.7% | **100.0%** |
| ProQuest 1965–2009 | 8.9% | **100.0%** |

This is not a peripheral column. In the published permutation importances,
`x_dir_stock_ret6` is the **second strongest feature** (0.0155 AUC drop, behind
only `x_dir_epu` at 0.0378), and the ladder's whole gain is those two
interaction terms — additive AUC 0.581 → 0.647 once `direction × economy` is
added.

So on ProQuest rows the old series would leave the model's number-two predictor
identically zero. Not merely missing: **constant by source**, which a pooled
model can read as "this row is ProQuest" instead of as economics.

## Is it the same signal?

Validated against the series it replaces on their 649-month overlap
(1914-12 → 1968-12):

```
level correlation        0.9963
6m-return correlation    0.9459
12m-return correlation   0.9538
```

Same signal, measured further.

## It changes the LOC result too — on purpose

Swapping the series raises LOC stock coverage from 74.5% to 100%, so the LOC
numbers here will **not** match the published ones. That is the reason this is a
separate folder rather than an edit: `data/scored/macro_context.csv` and every
number in `hit_model.ipynb` stay exactly as published, and the comparison
between old and new is a real comparison rather than a replacement.

## Files

| file | what |
|---|---|
| `stock_series.py` | fetch, parse and cache Shiller's series; `--refresh` re-downloads |
| `build_context.py` | rebuild a scored table's context using it; refuses to write into `data/scored/` |
| `data/shiller_sp500_monthly.csv` | the parsed series, committed so this reproduces offline |
| `data/ie_data.xls` | Shiller's workbook as downloaded, kept for provenance |

## Running it

```bash
# 1. the series (already cached; --refresh to re-download)
python new_model/stock_series.py

# 2. LOC with the new series
python new_model/build_context.py --scored data/scored/monthly_scored.csv \
    --out new_model/data/macro_context_loc.csv

# 3. ProQuest, once its claims are scored
python src/score_predictions.py --claims <proquest.export.jsonl> \
    --out data/scored/proquest_scored.csv
python new_model/build_context.py --scored data/scored/proquest_scored.csv \
    --out new_model/data/macro_context_proquest.csv
```

Step 3 needs the ProQuest export from the TDM Studio VM. See
`JeremysShit/election_arm/PROQUEST_TDM_GUIDE.md`.

## Pooling: still opt-in, and still not the default

Having the stock factor defined for both corpora removes ONE of the three
reasons not to pool. Two remain:

1. **Era.** 1900–1963 and 1965–2009 do not overlap. The CV holds out 3-year
   blocks, so ProQuest adds ~15 new folds rather than strengthening the
   existing 22.
2. **Labeller.** gpt-4.1 for LOC, gpt-4o-mini + a self-verify pass for ProQuest
   (gold F1 0.512, precision 0.700, direction kappa 0.81). `source` is
   confounded with labeller, so a `source` coefficient cannot be attributed.

The two defensible designs:

- **Report separately.** Fit the same ladder on each corpus and compare. If
  `direction × EPU` carries signal in both, that is a replication across a
  40-year gap and a different labelling model — stronger than one pooled number.
- **Train LOC → test ProQuest.** Directly tests whether the skill generalises
  out of era, which is the deployable-scorer question. A null is a real result.

Pooling into one training set — with or without `source` as a feature — is not
recommended: every held-out 3-year block belongs entirely to one corpus, so the
model can identify the source from the fold itself and be scored for it.
