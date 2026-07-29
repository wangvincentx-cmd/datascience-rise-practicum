# How predictions are scored — and how we know the scorer is right

This is the question that decides whether the project measures anything or just
launders an opinion. Written to be the methods paragraph for the poster.

## The one rule: the LLM never decides correctness

Two different questions, two different methods, kept strictly apart:

| Question | Decided by | Validated by |
|---|---|---|
| **What did the paper predict?** (direction, topic, horizon, scope) | the LLM (`extract_llm.py`) | a human/gold standard — extraction F1 0.79, 93% on hard cases |
| **Did that prediction come true?** | a deterministic rule vs. real data (`truth_data.py` + `score_predictions.py`) | unit tests with known answers — `test_scoring.py`, 28/28 |

The language model reads the noisy 1900s prose and says *what was claimed*. Real
economic data — NBER business-cycle dates and Federal Reserve series — decides
*whether it happened*. The model's opinion about whether a forecast was right is
never asked for and never used, here or anywhere downstream. Letting it grade
correctness would be handing a model with hindsight the job of judging
forecasts, which measures nothing.

So the answer to "are we letting the LLM decide, or grounding in real data, or
both?" is **both, for different steps**: the LLM extracts (checked against
humans), the data scores (checked by construction). They do not mix.

## Grounding: exactly how "came true" is computed

For each claim: take its print date, resolve its horizon into a window, look up
what the topic's series actually did over that window, and compare to the
predicted direction.

| topic | scored against | coverage |
|---|---|---|
| general business / markets | INDPRO industrial production | 1919– |
| general business (pre-1919) | NBER recession chronology | full period |
| prices | CPIAUCNS | 1913– |
| employment | UNRATE | 1948– |

Design choices, stated because a reviewer will ask:

- **Final revised values, not vintage.** We score what the economy *actually*
  did, so latest data is correct. (Vintage — what was knowable at print time —
  matters only for the separate *prediction model's features*, not for scoring.)
- **No-change bands.** A move counts as real only past a threshold (INDPRO 2%,
  CPI 1.5%, UNRATE 0.5pt); below it the outcome is "flat", so a forecast of
  "improve" is not rewarded for a 0.1% wiggle. The bands are documented
  parameters, adjustable via `--horizon-scale` and a `bands` argument, so the
  headline can be shown to survive plausible alternatives rather than being
  tuned to one.

## Why we can be *sure* the scorer is correct (and only *confident* about the judge)

An honest asymmetry:

- **The scorer is deterministic, so it can be proven.** `test_scoring.py` builds
  a synthetic economy — INDPRO doubles here, halves there, CPI rises 10%, UNRATE
  jumps 2 points — and asserts the exact hit/miss/unscorable verdict the maths
  must give. 28 cases, all passing, covering direction mapping, the bands,
  coverage boundaries, the NBER fallback, the scope gate, and the no-hindsight
  guards. If the rule were wrong, a test would fail.
- **The judge is an LLM on a partly-subjective task, so it can only be
  measured.** Best evidence: F1 0.79, direction agreement high, 93% on 44 hard
  boundary cases. And its gold standard is **model-built, not human** — a ~40-
  claim human recheck is the outstanding step before publication. We do not
  claim the extractor is certain; we claim it is validated to a stated level
  with a known next step.

## The honest denominator: not every claim is scorable, and we don't pretend

A claim is scored **only** if it has all of: a topic that maps to a series, a
real direction (not "unclear"), a resolvable window, **national scope**, and a
date inside that series' coverage. Everything else is marked unscorable *with a
reason* and left unscored — never guessed.

Measured on the 1,637-prediction v1 corpus with the new scorer:

```
scorable (with an inferred-or-default window)  1,425  (87%)   hit rate 0.474
RIGID  (window comes from the claim, not a default)  401  (24%)   hit rate 0.421
unscorable: 193 employment pre-1948, 12 no direction, 7 prices pre-1913
```

The 0.474 hit rate reproduces the known result (press ≈ economists ≈ ~50%),
which is a sanity check that the rebuilt scorer agrees with the old one on real
data while being independently tested.

### Two tiers, used for different claims

- **Loose scorable (~87%)** — fine for *descriptive* statistics (hit rate by
  decade, by voice, by hedging) where the exact window is not load-bearing.
- **Rigid (~24% on v1)** — the honest set for the *accuracy model*, where the
  window must come from the claim rather than a default, or the target becomes
  circular. **This roughly doubles on the new extraction**: the LLM states a
  horizon on ~54% of gold claims versus ~24% for the old regex+grade pipeline,
  so the rigid stratum on the full new corpus should be ~45–55% of national
  claims — thousands of claims, ample for a fitted model.

### Scope removes more, on purpose

On the gold sample the extracted scope mix is ~68% national, 21% industry, 9%
foreign, 3% regional. Only national claims are graded against national series;
the rest are real forecasts kept as separate strata (a foreign forecast about
Mexican exports must not be scored against US industrial production). The `scope`
field, added before the corpus run, is what makes this filter possible — v1 had
no way to do it and silently graded everything as national.

## What this supports on the poster

- **Solid:** hit rate broken down by claim feature (hedged vs assertive, named
  vs anonymous forecaster, expert vs journalist, specific vs vague), on the
  scorable set, with confidence intervals. Objective ground truth, tested scorer.
- **Attempted, honest either way:** a model predicting whether a forecast came
  true, on the rigid national subset, reported against a macro-only baseline. A
  null ("outcome is governed by the economy, not by how the forecast was
  written") is a legitimate result.
- **Not claimed:** that we can reliably predict which forecasts will be correct —
  effect sizes will not support it, and we say so.

## Files

- `truth_data.py` — the ground-truth layer: FRED series, NBER chronology,
  `realized_direction()`. No LLM anywhere in it.
- `score_predictions.py` — the deterministic scorer: horizon resolution,
  direction normalization, the scope gate, `score_claim()`, honest unscorable
  reasons.
- `test_scoring.py` — the proof: 28 known-answer cases against a synthetic economy.
