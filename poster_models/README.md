# poster_models

Seven alternatives to the penalized logistic regression in `src/model_hit.py`.

None of these is here because logistic regression is too weak a classifier. The
binding constraint on this project is not model capacity — it is that **14,251
claims sit inside about 21 correlated time blocks**, and the outcome is a near
coin flip (hit rate 0.513). Under that constraint a bigger model buys nothing
and costs interpretability. What is actually hard is *inference*: getting an
estimate that means something and an interval that is honest.

So each model here changes the **estimand** or the **inference**, not the
optimiser.

| id | file | what it changes | needs FRED |
|----|------|-----------------|:---:|
| m1 | `m1_fixed_effects.py` | compares only claims printed in the same month | no |
| m2 | `m2_clustered.py` | error bars that respect ~21 blocks, not 14,251 rows | optional |
| m3 | `m3_dml.py` | the macro delta as a coefficient with a CI | **yes** |
| m4 | `m4_text.py` | reads the quote text, not just its metadata | no |
| m5a | `m5a_direction.py` | predicts *error direction*, not right/wrong | no |
| m5b | `m5b_survival.py` | predicts *when* the forecast came true | partial |
| m5c | `m5c_press_probit.py` | monthly press index → recession onset | partial |

## Running

Everything assumes the repo root is the working directory, like the rest of the
project.

```bash
python poster_models/run_all.py              # all seven
python poster_models/run_all.py --fast       # quick pass
python poster_models/run_all.py --only m1 m3
python poster_models/m1_fixed_effects.py     # or one at a time
```

Console output is teed to `outputs/<id>_console.txt`; every table is also
written there as CSV.

### FRED cache

`cache/` is gitignored, so a fresh clone has no macro data:

```bash
python poster_models/fetch_fred.py
```

Models degrade rather than crash when only some series are cached — a missing
series routes through the pipeline's existing *outside coverage* path, so
affected claims are marked unscorable instead of guessed, and each run prints
what is missing. The exception is **m3**, which fails loudly with no macro data
at all, because "what does wording add over the economy" is not a question you
can answer without the economy.

## What each one does

### m1 — conditional (fixed-effects) logit

Adds one intercept per month and estimates claim-feature effects using only
within-month variation. Two forecasts printed in March 1933 faced the same
economy, so a difference between them cannot be the macro regime. The
confounder is removed by design, not by hoping `m_indpro_g6` is the right
functional form.

Fits a coarseness ladder — month, year, 3-year block — and flags any
coefficient that changes sign across them. Uses exact conditional logit for
small groups and dummy-variable FE for large ones, because the exact
conditional likelihood is recursive and overflows on thousand-claim groups
(and because incidental-parameter bias is negligible once groups are large).

**Cannot** answer across-time questions. That is m2's job.

### m2 — cluster-robust and hierarchical logit

Same point estimates as the standard model; honest intervals. Reports the
naive SE, the block-clustered SE, their ratio (the design effect), the implied
effective sample size per coefficient, and how many findings stop being
significant once clustering is respected. Adds a block bootstrap — the sandwich
estimator is only valid asymptotically in the number of *clusters*, and 22 is
not many — and a hierarchical fit with a random intercept per block, which
partially pools small blocks and yields the intraclass correlation directly.

### m3 — double machine learning

The poster's headline is a *difference of two AUCs*, which is not an estimator:
no standard error, no interval, no population quantity. This replaces it with a
partially-linear model, `hit = θ'claim + g(macro) + e`, where `g` is estimated
by gradient boosting and cross-fitted **by time block**. Residualising both
sides makes `θ` insensitive to first-order error in `g` (Neyman orthogonality),
which throwing macro columns into one logit does not achieve.

Output: partial effects on the probability scale with cluster-robust intervals,
plus a joint χ² test — the correctly-sized replacement for the permutation test
on the AUC delta. Includes an optional placebo check that the test's
false-positive rate really is 5%.

### m4 — text models

`model_hit` knows six things about each quote and never reads the words.

Character 3–5-grams because the OCR is genuinely mangled (`busi ness`,
`prosper f'ra`) and word tokenisation scatters each corruption into its own
hapax. Word n-grams too, not for accuracy but because a list of predictive
*phrases* is readable on a poster.

Three guards: leave-one-block-out CV, a digit-masked variant (a year is a
direct pointer to the outcome), and a within-block permutation baseline —
a sparse model on 22 clusters beats AUC 0.5 for free, so 0.5 is the wrong
benchmark.

LOC only; ProQuest rows ship without verbatim text.

### m5a — error direction

`hit` is one bit and it destroys the most interesting thing in the data. Recast
as **too optimistic / correct / too pessimistic** and fitted as both a
multinomial logit (lets a feature push asymmetrically) and an ordered logit
(one coefficient, stricter). The symmetry table between them says which
summary is honest.

Keeps two scales strictly apart: *series direction* (did it rise) is defined
for every topic; *optimism* (was it good news) is defined only for business,
markets and employment. Price claims are excluded from optimism numbers rather
than assigned a sign by guess — conflating the two would silently encode
"inflation is good".

### m5b — discrete-time survival

Walks each claim forward month by month and records the first month the economy
had moved the way the claim said. Claims whose horizon expires unresolved are
**right-censored**, not scored as failures.

Falls out of this: how *fast* forecasts came true, and the share that were
**transiently right** — vindicated at some point inside the horizon, then
overtaken by events before it closed. Fixed-horizon scoring calls those misses.
That is a defensible rule, but it is a rule, and this is the only place in the
pipeline its cost is visible.

### m5c — aggregate press probit

One row per month instead of per claim. Outcome: did an NBER contraction begin
within the next *h* months. Estimator: probit, the Estrella–Mishkin
specification, so the numbers are comparable to published yield-curve results.

Newey–West HAC standard errors throughout, because overlapping outcome windows
induce autocorrelation by construction. Expanding-window out-of-sample
evaluation with an *h*-month gap between train and test, so the model never
trains on an outcome window that overlaps the month it is forecasting. NBER
dates are only ever the outcome; the macro benchmark is publication-lagged.

## Conventions

Shared by every module, in `_common.py`:

- **Never a random split.** Every CV fold, cluster and bootstrap unit is a
  3-year time block.
- **Always intervals.** `coef_table` prints a 95% interval next to every point
  estimate; there is no path through this code that reports a coefficient
  alone.
- **No hindsight.** Macro features are publication-lagged; NBER status is never
  an input.
- **Unscorable stays unscored.** Missing series produce dropped claims with a
  reason, never imputed outcomes.
- Rare categorical levels are pooled, and a pooled bucket that is *itself* rare
  is folded into the reference level — a 10-observation dummy produced a
  bootstrap standard error of 70 log-odds before this was fixed.
