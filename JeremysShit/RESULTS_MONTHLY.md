# Monthly corpus results, 1900–1963

**Date:** 2026-07-27
**Corpus:** 15,721 LOC newspaper pages, every month 1900-01 → 1963-12 (768/768
months, no gaps), sampled with a fixed direction-neutral query set.
**Extraction:** gpt-oss-120b, low reasoning effort, **$7.47 actual**
(projected $8.15). Total project spend **$24.97** of a $30 cap.
**Claims:** 30,765 extracted → **14,251 scorable US-national** claims.

> **Quality caveat, state it on the poster.** This corpus was extracted with
> gpt-oss-120b (gold-standard F1 ≈ 0.61), not the Gemini extractor used for the
> crisis corpus (F1 ≈ 0.79). Aggregate monthly series tolerate this — per-claim
> error partly averages out over ~40 claims/month — but the two corpora are
> different instruments and must not be compared claim-for-claim.

## 1. The headline: the press expected improvement, always

Net optimism (share improve − share worsen) stays **above zero in essentially
every month for sixty years**, straight through every NBER recession.

| decade | net optimism | disagreement | hedging |
|---|---|---|---|
| 1900s | +0.48 | 0.50 | 0.39 |
| 1910s | +0.53 | 0.46 | 0.33 |
| 1920s | +0.54 | 0.45 | 0.37 |
| 1930s | +0.49 | 0.50 | 0.42 |
| **1940s** | **+0.15** | **0.71** | 0.37 |
| 1950s | +0.36 | 0.59 | 0.42 |
| 1960s | +0.42 | 0.50 | 0.45 |

Figure: `prelim_figures/fig6_index_net_direction.png`

**The 1940s are the exception that proves the instrument works.** Optimism
collapses to +0.15 and disagreement spikes to 0.71 — the wartime and
postwar-depression-scare uncertainty — found by the pipeline unprompted, with no
episode labels and no outcome information. Nothing in the extraction knew what
year mattered.

## 2. Forecast accuracy: near a coin flip, everywhere

Overall directional hit rate on scorable national claims: **0.513** (n = 14,251).

By what was predicted:

| prediction | n | hit rate |
|---|---|---|
| improve | 8,962 | **0.597** |
| prices up | 1,359 | 0.510 |
| worsen | 2,577 | **0.364** |
| prices down | 868 | 0.267 |
| no change / flat | 485 | **0.194** |

Predicting improvement was right ~60% of the time; predicting a downturn ~36%.
That is not forecasting skill — it reflects that the economy expands most of the
time, so a permanent bull is right more often than not. Forecasts of *no change*
were almost always wrong (19%): the economy rarely stands still.

Figure: `prelim_figures/fig8_accuracy_over_time.png`

## 3. The model: how a forecast is written beats what the economy was doing

Out-of-fold ROC-AUC, leave-one-block-out over 21 three-year blocks:

| model | AUC |
|---|---|
| base rate | 0.513 |
| **macro only** (INDPRO/CPI/UNRATE/stocks at print time) | **0.505** |
| claim features only (direction, topic, hedging, voice, horizon…) | **0.561** |
| claim + macro + derived | 0.588 |
| gradient boosting | 0.573 |

**The macro baseline sits at chance (0.505), and this is a genuine null, not a
bug.** Every macro feature correlates with `hit` at |r| < 0.10, checked directly.
The state of the economy when a forecast was printed carries almost no
information about whether that forecast came true.

Because the macro baseline is at chance, the "delta over macro" is not a
meaningful quantity here. What *is* interpretable is the **claim-only model at
0.561** — weak in absolute terms, but it beats both the base rate and the macro
model. The modest, honest conclusion: **how a forecast is phrased predicts its
accuracy slightly better than the economic conditions it was made in.**

## 4. A finding that REVERSED — and why that matters

On the crisis corpus (n = 232) claims that swam against the press consensus hit
**52%** vs 43% for consensus-followers, and I flagged it as suggestive.

On this corpus, with 8× the sample (n = 1,967), it **reverses**:

| | n | hit rate |
|---|---|---|
| follows press consensus | 12,284 | **0.533** |
| swims against consensus | 1,967 | **0.387** |

Contrarians were *worse*, not better. The crisis-corpus result was small-sample
noise. **Do not put the contrarian claim on the poster.** This is the clearest
argument in the project for why the continuous corpus was worth building: it
overturned a finding that a smaller, outcome-selected sample had suggested.

Press disagreement shows no clean monotone relationship with accuracy either
(low 0.537 / mid 0.486 / high 0.514) — the Baker-Bloom-Davis "uncertainty
carries the signal" hypothesis is **not supported** at claim level here.

## 5. What to claim, and what not to

**Solid, verified:**
- The methods result: keyword extraction recovers 27% of forecasts vs 73% for
  whole-page LLM reading, against a gold standard, with a deterministic scorer
  proven by 33 known-answer tests.
- The index itself: 761 months of press expectations, 1900–1963, reproducing
  known history (the 1940s uncertainty spike) unprompted.
- Persistent optimism: the press expected improvement through essentially every
  downturn in sixty years.

**Report with the stated caveat:**
- Hit rates by feature (above) — extracted at F1 ≈ 0.61.
- The claim-only model at AUC 0.561, and the macro null at 0.505.

**Do not claim:**
- That contrarian forecasts were more accurate (it reversed).
- That uncertainty/disagreement predicts accuracy (no clean relationship).
- That the model usefully predicts which forecasts come true — AUC 0.561 is
  barely above chance, and saying so plainly is the honest result.

## Reproduce

```
python score_predictions.py --claims claims_monthly.jsonl --out monthly_scored.csv
python build_press_index.py --claims claims_monthly.jsonl \
    --pages data/monthly/pages_monthly.jsonl --out data/press_index.csv
python model_hit.py --scored monthly_scored.csv --perm 0
python make_index_figures.py
```
