# Economic context & the hit predictor — results

**Date:** 2026-07-28
**Corpus:** the continuous monthly LOC corpus, 14,251 scorable US-national
claims, 1900–1963, hit rate 0.513, 22 three-year periods.
**Scoring is unchanged.** `truth_data.py` still decides what happened, by rule,
with no economic data and no LLM anywhere near the correctness decision. Every
number below comes from a layer that runs *after* scoring.

> Inherits the monthly corpus's standing caveat: extracted with gpt-oss-120b
> (gold-standard F1 ≈ 0.61), not the Gemini extractor used for the crisis corpus
> (F1 ≈ 0.79). Aggregate patterns tolerate this; claim-for-claim comparison
> across the two corpora does not.

## 1. The correction: the macro null was a cancellation, not an absence

[RESULTS_MONTHLY.md](RESULTS_MONTHLY.md) reports a macro-only model at AUC 0.505
and reads it as a genuine null — "the state of the economy when a forecast was
printed carries almost no information about whether it came true." **That
reading was incomplete.** It is true of the *pooled* sample and false within it.

Splitting by what the forecast predicted (`macro_context.py`):

| factor at print time | optimistic forecasts | pessimistic forecasts |
|---|---|---|
| stock market, 6m return | **+0.216** | **−0.283** |
| economic policy uncertainty | **+0.206** | **−0.201** |
| stock market vs 2-year peak | +0.159 | −0.233 |
| stock market, 12m return | +0.155 | −0.213 |
| unemployment rate *(1948+ only)* | **+0.352** | −0.195 |
| industrial output growth (6m, 12m) | ≈ 0 | ≈ 0 |
| inflation (level, acceleration) | ≈ 0 | ≈ 0 |

Correlations with `hit`; intervals and BH q-values in the script output. Bold =
survives Benjamini-Hochberg at q < 0.05 across the 13-factor family.

Optimistic and pessimistic forecasts respond to the same conditions with
**opposite sign**, so pooling them attenuates the effect toward nothing. A
model with only additive macro terms is structurally unable to represent this —
one coefficient has to serve both groups — which is most of why the pooled
macro baseline sat at chance.

**What genuinely does nothing:** industrial output growth and inflation, in
every stratum. That part of the original null holds. It is the *market and
policy-uncertainty* conditions that carry the signal.

Figures: `poster_figures/figI_macro_scissors.png` (the crossing lines),
`poster_figures/figJ_macro_ledger.png` (all 13 factors, both directions,
including the ones that did nothing).

## 2. The hit predictor

`hit_predictor.py` — given a forecast and the economic data already public when
it went to print, the probability it comes true. Out-of-fold, leave-one-3-year-
period-out, pooled across folds:

| model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| base rate | — | 0.513 | 0.250 |
| how the forecast is written | 0.561 | 0.514 | 0.245 |
| the economy at print time | 0.538 | 0.551 | 0.260 |
| both, added together | 0.581 | 0.594 | 0.255 |
| **both, allowed to interact** | **0.647** | **0.652** | **0.241** |
| gradient boosting, same inputs | 0.617 | 0.633 | 0.257 |

The interaction block adds **+0.066 AUC** over the additive model. Gradient
boosting does not beat the linear model, so the logistic model stays primary and
its coefficients are an honest summary rather than a simplification.

**Block-permutation test** (200 shuffles of `hit` *within* each 3-year period,
refit each time): observed AUC 0.647, null mean 0.478, **p = 0.005**. Shuffling
within period preserves the macro-cluster structure, so this tests the
forecast-level signal specifically rather than the model's ability to learn
which decades were good ones.

Figure: `poster_figures/figK_hit_model.png`.

**Permutation importance** (held-out AUC drop when one input is shuffled inside
the test fold, `hit_predictor_importance.csv`) — the gain is almost entirely two
terms:

| input | AUC drop |
|---|---|
| direction × policy uncertainty | **0.038** |
| direction × stock return | **0.016** |
| topic | 0.007 |
| direction | 0.005 |
| *every macro main effect* | ≈ 0 or negative |

That is the same finding as §1, arriving independently: the economy predicts
accuracy only in interaction with the forecast's direction, never on its own.

**Calibration** is monotone but too extreme at both ends (the model says 0.90
where the truth is 0.78, and 0.14 where the truth is 0.27). Use the output as a
**ranking**, not as literal odds.

## 3. Two things that must be said out loud

**(a) Policy uncertainty carries most of the lift, and it is the weakest link.**
Ablating EPU entirely (`python hit_predictor.py --exclude epu`):

| model | with EPU | without EPU |
|---|---|---|
| the economy at print time | 0.538 | 0.502 |
| both, added together | 0.581 | 0.554 |
| both, allowed to interact | **0.647** | **0.587** |

The interaction structure survives without EPU — 0.587 still beats claim-only
(0.561) and the additive model (0.554), and the interaction still adds +0.033.
But EPU alone accounts for roughly 0.060 of the 0.086 total lift over the
claim-only model. Three reasons to hold that number at arm's length:

1. **Shared instrument.** The historical EPU index is built by counting
   policy-uncertainty language *in newspapers* — the same medium this corpus is
   drawn from. A month when the press wrote anxiously is a month when EPU is
   high and when our sampled articles differ too. This is a plausible
   circularity and it is not ruled out here.
2. **It did not replicate** on the crisis corpus (below).
3. Its coverage claim (100%) rests on a single third-party spreadsheet
   (1900–2014) that the rest of the pipeline does not depend on.

The defensible headline is therefore the **0.587 ablated model**, with 0.647
reported alongside it and the EPU caveat attached — not 0.647 on its own.

**(b) The crisis corpus can replicate the pattern but cannot test the model.**
Running the same attribution on `claims_v2_scored.csv` (1,824 claims, Gemini
extraction, F1 ≈ 0.79):

- Market conditions replicate **in sign**: stock 6m return +0.278 optimistic /
  −0.175 pessimistic; stock-vs-peak +0.314 / −0.261; volatility −0.268 / +0.252.
  Same scissors, independent corpus, better labels.
- **EPU does not replicate**: +0.084 / −0.012, essentially nothing.
- Nothing survives BH correction there — 1,824 claims across ~10 outcome-selected
  episodes gives intervals too wide to conclude from.
- The **model ladder on that corpus is non-diagnostic**: every model lands
  *below* chance (0.29–0.37). This is the artefact `model_hit.py` already
  documents — leave-one-block-out removes an entire macro regime per fold, so
  with 13 blocks the models extrapolate and the sign inverts.
  `hit_predictor.py` now prints this warning itself and refuses to report the
  ladder. **Do not read the −0.070 interaction delta from that run as evidence
  against the interaction.**

## 4. What this changes on the poster

- The line "the macro baseline sits at chance, and this is a genuine null" needs
  qualifying. Accurate version: *the economy's effect on accuracy is real but
  direction-dependent, and cancels when optimistic and pessimistic forecasts are
  pooled; industrial output and inflation genuinely carry nothing.*
- It strengthens the existing figA/figB story rather than competing with it.
  figA shows the press forecast the same mix regardless of conditions; this
  shows those same conditions predicted which of those forecasts would be right.
  **The information was on the table and went unused** — which is the mechanism
  behind figB's accuracy collapse, now measured from real-time data rather than
  from hindsight NBER dating.
- Report the model as **0.587 (ablated) / 0.647 (with EPU)**, describe it as a
  ranking rather than calibrated odds, and state that it is driven entirely by
  direction × economy interactions.

## Reproduce

```
python macro_context.py                                  # -> macro_context.csv + attribution
python hit_predictor.py --perm 200                       # -> hit_predictor.joblib + importance
python hit_predictor.py --exclude epu --no-importance    # -> the ablation in §3(a)
python macro_context.py --scored claims_v2_scored.csv --out crisis_context.csv
python hit_predictor.py --context crisis_context.csv     # prints the artefact warning
python make_macro_figures.py                             # -> figI, figJ, figK
python hit_predictor.py --predict-demo                   # score new, unresolved forecasts
```
