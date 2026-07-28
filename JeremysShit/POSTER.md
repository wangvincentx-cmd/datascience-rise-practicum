# Did Anyone See It Coming?
### Machine-reading a century of American newspaper economic forecasts, 1900–1963

*BU RISE — draft poster text. Every number below is reproducible from this repo;
figures are in `poster_figures/`.*

---

## The question

For sixty years American newspapers told readers what the economy would do next.
Were they right? The record has never been scored at scale, because reading a
century of newspapers by hand is infeasible — and the standard shortcut,
keyword search, turns out to lose most of the evidence.

## What we built

**A pipeline that reads whole newspaper pages and grades what it finds against
real economic data.**

```
15,721 LOC newspaper pages          sampled from all 768 months, 1900-01 … 1963-12, no gaps
        ↓  LLM reads each full page, returns structured forecasts
30,765 extracted forecasts          direction, topic, horizon, hedging, speaker, scope
        ↓  deterministic scorer — NBER + Federal Reserve series, no model judgement
14,251 scorable US-national claims  each marked hit / miss / honestly unscorable
   761 months in the press index     months yielding ≥1 scorable US-national claim
```

Note the distinction: **pages were sampled in every one of the 768 months**, but
**761 months** yield at least one scorable US-national forecast. Earlier drafts
wrote "768/768" for both.

**The split that makes this a measurement, not an opinion:** the language model
decides *what was predicted*; real economic data decides *whether it came true*.
The model is never asked whether a forecast was right. Correctness is a lookup
against NBER recession dates and FRED series (industrial production, prices,
unemployment, stock index), computed the same way every time.

---

## PANEL 1 — The method: keyword search loses three forecasts in four

**`poster_figures/` → use `../prelim_figures/fig1_extraction_gap.png`**

| extractor | precision | recall |
|---|---|---|
| keyword regex (the standard approach) | 61% | **27%** |
| whole-page LLM reading | 84% | **73%** |

Measured against a hand-built gold standard: 16 newspaper pages annotated
exhaustively before any model ran, 52 forecasts, plus 44 documented boundary
cases (advertisements, serialized fiction, "Twenty Years Ago" reprints, refusals
to forecast). The scorer itself is proven by **33 known-answer tests** against a
synthetic economy.

**Three specific ways the keyword approach distorts the answer:**
1. Loaded search terms retrieve **advertising**, not forecasts — "return of
   prosperity" appears on one gold page only inside a bank's New Year ad.
2. It extracts a bullish 1907 forecast from a passage headlined *"Prophecies Gone
   Wrong"* — a report that forecasters were **wrong** — which then scores against
   the Panic and manufactures a "nobody saw it coming" signal.
3. 43% of extracted claims are not about the US economy (regional, industry,
   foreign). Without a scope field they get graded against national statistics.

---

## PANEL 2 — The mechanism: the forecast mix never responded to the economy

**`poster_figures/figA_mechanism.png`**

> Share of forecasts predicting a downturn:
> **24.1% in expansions · 24.3% in recessions**

Statistically identical. Month by month across sixty years, the share of downbeat
forecasts does not rise when the economy turns. Roughly **72% of all forecasts
were upbeat**, in booms and busts alike.

---

## PANEL 3 — The consequence: accuracy collapses exactly when it matters

**`poster_figures/figB_consequence.png`**

| | expansions | recessions |
|---|---|---|
| **overall accuracy** | **58.8%** | **39.7%** |

Gap **+19.1 points**, 95% CI **[+12.9, +24.4]** — block-bootstrapped by 3-year
period, so the interval accounts for the fact that forecasts within an era are
not independent. (Recomputed 2026-07-28; `build_poster.py` and
`make_poster_figures.py` now derive this live rather than hardcoding it.)

**Present this as a consequence of Panel 2, not as an independent finding.**
NBER dates are both the scoring ground truth and the split variable: in
recessions 65% of claims are "improve" and 68.3% of all misses are "improve"
claims. The free-standing empirical fact is the *mix* (Panel 2); the accuracy
gap follows from it.

The breakdown shows *why*, and it is not that everyone got worse:

- "business will improve": 76% right in expansions → **37%** in recessions
- "business will worsen": 21% right in expansions → **61%** in recessions

**Pessimists were right when it counted — there were simply never enough of
them.** Because the mix stayed ~3:1 upbeat regardless of conditions, the average
fell anyway.

---

## PANEL 4 — Sixty years, no improvement

**`poster_figures/figC_no_learning.png`**

Annual accuracy 1900–1963: **flat trend**, hovering at a coin flip
(1900s 53.7% → 1960s 49.9%). Two world wars, the Depression, the founding of
modern macroeconomics — no measurable improvement in the press's ability to call
the economy's direction.

**And no publisher was better.** Across seven papers with ≥200 scorable
forecasts, accuracy runs 44.2% to 56.4% — all near chance. There was no smart
newspaper.

---

## PANEL 5 — What predicts whether a forecast came true?

**`poster_figures/figD_what_predicts.png`**

| model (out-of-fold AUC, leave-one-block-out, 21 three-year blocks) | AUC |
|---|---|
| base rate | 0.513 |
| **macroeconomic conditions at print time** | **0.505** ← chance |
| how the forecast was written (direction, hedging, voice, horizon…) | **0.561** |
| both combined | 0.588 |

**Two honest results:**

- **The state of the economy carries almost no information** about whether a
  forecast came true (every macro feature correlates |r| < 0.10 with being
  right). This is a genuine null, verified directly — not a modelling artefact.
- **How a forecast is written does slightly better than chance** (0.561), but
  only slightly. Hedged vs assertive: 50.0% vs 52.0%. Named forecaster vs
  anonymous: 52.0% vs 50.6%. Neither is a useful predictor.

**We do not claim to predict which forecasts come true.** AUC 0.561 is barely
above chance, and saying so plainly is the result: forecast accuracy in this
corpus is close to irreducible.

---

## PANEL 6 — Worse than chance on prices, markets and jobs

**`poster_figures/figF_topics.png`**

Accuracy varies far more by *subject* than by how a forecast was written — a
23-point spread, against 2 points for hedging and 1.4 for a named forecaster.

| topic | n scorable | hit rate | vs 50% |
|---|---|---|---|
| general business | 8,713 | **56.1%** | p = 2e-30 |
| other | 870 | 47.9% | — |
| stock markets | 2,219 | **44.9%** | p = 2e-06 |
| prices / inflation | 2,275 | **41.5%** | p = 5e-16 |
| jobs / unemployment | 174 | **32.8%** | p = 6e-06 |

**Markets, prices and employment forecasts are all significantly worse than a
coin flip**, and the press published 10,400 of them. The only above-chance
category is vague optimism about "business" in general — and only because
"business will improve" is right most years by default.

**Why that edge does not generalise.** Price accuracy is the era's inflation
*regime*, not skill: "prices up" and "prices down" hit rates sum to roughly 1 in
every decade (1910s 0.85/0.01, 1940s 0.83/0.07, 1920s 0.27/0.48). After 1948,
"prices will fall" was right **0 times out of 93**. Consistent with this, topic
scores **AUC 0.495 — chance — out of fold** (leave-one-block-out). It separates
in sample and predicts nothing forward.

*Caveat for the jobs row:* only 174 of 1,663 employment claims (10.5%) are
scorable, because UNRATE starts in 1948. State the n.

---

## PANEL 7 — October 1929: no one saw it coming

**`poster_figures/figG_1929.png`**

Share of US-national forecasts predicting improvement, month by month:

| 1929-06 | 07 | 08 | 09 | **10 (Crash)** | 11 | 12 | 1930-01 |
|---|---|---|---|---|---|---|---|
| 76.5% | 93.3% | 86.4% | 64.3% | **86.1%** | 78.4% | 76.2% | 79.2% |

**In the month of the Great Crash the press was more bullish than it had been in
June.** Forecasts printed August–December 1929 came true **20.8%** of the time
(n = 168) — four in five were wrong.

Found with no episode labels and no outcome information, on the same continuous
corpus as every other panel.

---

## A finding that reversed — and why we report it

On a smaller, crisis-only corpus (n = 232), forecasts that **swam against the
press consensus** appeared *more* accurate: 52% vs 43%.

On the full continuous corpus (n = 1,967 — eight times the sample) it
**reverses**: **38.7% vs 53.3%**. Contrarians were *worse*.

The first result was small-sample noise from an outcome-selected sample. This is
the strongest argument for why the continuous corpus was worth building: it
overturned a finding that the smaller one had suggested.

> ⚠️ **Unresolved — do not print either way yet.** We also tested whether
> **forecaster disagreement** predicts accuracy (the Baker–Bloom–Davis
> "uncertainty" hypothesis). The answer depends on which of two disagreement
> measures in this repo is used, and they disagree:
>
> | measure | low | mid | high |
> |---|---|---|---|
> | `model_hit.py:165` — divides by **all** claims that month | 53.7% | 48.6% | 51.4% |
> | `build_press_index.py:92` — divides by **directional** claims | 56.6% | 49.9% | 47.3% |
>
> The first gives the "no relationship" null currently claimed. The second is
> cleanly monotone, is not a recession proxy (corr with `in_recession` = −0.012),
> and holds within expansions (66.2→58.6→51.7) and within "improve" claims alone
> (64.7→57.4→55.8). Decide which measure is intended before making any claim
> about uncertainty and accuracy.

---

## Limitations, stated plainly

- **Extraction quality.** The monthly corpus was extracted with a cheaper model
  (gold-standard **F1 ≈ 0.53–0.61**) to fit a $30 budget; the methods comparison
  used a stronger one (F1 ≈ 0.79). Aggregate monthly series tolerate this —
  per-claim error partly averages out over ~40 claims/month — but the two are
  different instruments. **The range is unresolved:** `result_oss120b_low.json`
  measures F1 = 0.527 for gpt-oss-120b at *low* reasoning effort, while
  `result_gptoss120b.json` gives 0.612 at default effort, and
  `monthly_extract.log` does not record which setting ran. Confirm and collapse
  this to one number before publication.
- **The gold standard is model-adjudicated, not human.** A two-person recode of
  ~40 claims is the outstanding step before publication.
- **Coverage thins after 1940.** LOC digitization drops from ~600 claims/year
  pre-1940 to ~200–290 after (copyright-driven), so the late index is noisier.
  Every monthly series carries its claim count; figures filter to n ≥ 5.
- **Our index does not correlate with the published historical EPU index**
  (all |r| < 0.08). We measure forecast *direction*, not policy uncertainty —
  they are different constructs, and we report the null rather than hide it.
- **Effective sample is ~21 time blocks, not 14,251 claims.** Forecasts cluster
  hard within eras; all inference uses grouped CV and block bootstraps.

---

## The headline

**Across sixty years and 14,251 scored forecasts, the American press told readers
the economy would improve — in booms and in busts alike, in the same proportion,
with no improvement over six decades and no paper doing better than any other.
When downturns came, three in four forecasts were pointing the wrong way.**

Forecasting the economy's direction was, and remained, close to a coin flip. The
contribution here is that this is now *measured* rather than asserted — and the
pipeline that measures it is validated, reproducible, and cost about $25.

---

## Reproduce everything

```
python score_predictions.py --claims claims_monthly.jsonl --out monthly_scored.csv
python build_press_index.py --claims claims_monthly.jsonl \
    --pages data/monthly/pages_monthly.jsonl --out data/press_index.csv
python model_hit.py --scored monthly_scored.csv --perm 0
python make_poster_figures.py          # all seven poster figures
python make_index_figures.py           # the monthly index series
python build_poster.py                 # -> RISE_Poster_2026.pptx
python test_scoring.py                 # 33 known-answer scorer tests
python test_offline.py                 # 90 offline pipeline tests
```

`build_poster.py` no longer needs `data/monthly/*.jsonl` (untracked, 385 MB) —
it recomputes every headline number from the committed `monthly_scored.csv`,
`data/press_index.csv` and `greenbook_scored.csv`, and asserts that the results
column fits rather than silently overlapping panels.

**Data:** Library of Congress *Chronicling America* (public API, no key), NBER
business-cycle chronology, Federal Reserve FRED series. **Total compute cost:
$24.97.**
