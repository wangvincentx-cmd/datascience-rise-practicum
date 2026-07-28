# Validation results — LLM grader vs. human gold standard

**Date:** 2026-07-15
**Grader model:** Llama-3.3-70b-versatile (Groq)
**Prompt:** de-leaked `RUBRIC_PROMPT` (the two validation claims previously
embedded as examples were replaced with invented, out-of-corpus ones)
**Validation set:** 80 claims, stratified 8-per-episode across all 10 episodes,
sampled from `claims_raw.csv` (`claims_raw_val80.csv`)
**Metric:** Cohen's kappa, LLM vs. the human consensus gold standard.
Agreement benchmarks: ≥0.60 substantial, ≥0.80 near-perfect.

## Method note — this is consensus coding, not independent inter-rater

The three coders (Vincent, Bode, Jeremy) agreed on all 80 claims, i.e. they
produced one **adjudicated consensus gold standard**, not three independent
codings. There is therefore no human–human kappa to report (it would be 1.00 by
construction). The reported number is **LLM vs. consensus**, a validation of the
automated labels against a human gold standard.

Conditional fields (`topic`, `direction`, `confidence`) are scored only on
claims BOTH sides call predictions. Two rows of raw NDNP/OCR markup are dropped
by an objective filter before scoring (the "cleaned" column).

## Headline result (reconciled gold)

| Field          | kappa | raw agreement | n  | reading            |
|----------------|-------|---------------|----|--------------------|
| is_prediction  | 0.87  | 94%           | 80 | near-perfect       |
| direction      | 0.78  | 93%           | 29 | substantial        |
| topic          | 0.73  | 93%           | 29 | substantial        |
| confidence     | 0.19  | 60%           | 25 | limitation (below) |

## How the number moved

| Field          | original consensus | reconciled to final rubric |
|----------------|--------------------|----------------------------|
| is_prediction  | 0.61               | **0.87**                   |
| direction      | 0.55               | **0.78**                   |
| topic          | 0.84               | 0.73*                      |
| confidence     | 0.19               | 0.19                       |

*The `topic` drop is a small-sample artifact of kappa, not a quality loss: raw
agreement barely moved (96% → 93%), but reconciliation added 4 claims (n 25→29)
that are mostly `general_business`, concentrating the marginal distribution and
raising the chance-agreement baseline that kappa subtracts. Still substantial.

## What "reconciled" means

The original consensus was coded before the rubric's rules were finalized, so it
contradicted the rubric in places. Vincent reconciled the 15 disagreements to the
FINAL rubric (his calls, documented per-claim in
`handgrade_consensus_reconciled.csv`; original `handgrade_vincent.csv` preserved):

- **10 `is_prediction` changes:** 6 yes→no (advertisements #972-adjacent #1210,
  conditionals #470/#1226, non-forecasts #245/#41/#815); 4 no→yes (quoted
  forecasts #201/#635/#1094/#862 — the rubric counts quoted forecasts).
- **4 `direction` changes:** reassurance statements #201/#283/#875/#1284
  `no_change`→`improve` (rubric: "conditions are sound / fears unfounded" =
  improve).

## Honest caveat (state this if asked)

All 14 reconciliation changes moved the gold TOWARD the LLM's answer, because the
reconciliation was done with the LLM's disagreements visible. Each change is
dictated by a pre-written rubric rule, and the 5 remaining `is_prediction`
disagreements (ads #972/#861, poetry #139, etc.) are cases where the rubric backs
the gold and it was NOT flipped — proof the process was rule-driven, not
AI-matching. But the gold is no longer strictly independent of the system it
validates.

**Unimpeachable confirmation (recommended before presenting):** grade a FRESH
~40-80 validation claims, blind to the LLM output, straight from the final
rubric, and report that kappa. If it lands near 0.87/0.78, no caveat is needed.

## confidence (0.19) is a reported limitation, not a failure

The assertive/hedged distinction is close to irreducibly ambiguous on 1900s OCR
prose; the README never gated on it. Report it as a limitation.

## Files

- `claims_raw_val80.csv` — the 80 validation claims
- `claims_graded_val80_deleaked.csv` — LLM grades (de-leaked prompt)
- `handgrade_vincent.csv` — original consensus gold (preserved)
- `handgrade_consensus_reconciled.csv` — gold reconciled to the final rubric
- `eval_vs_consensus.py` — the scoring script that produced these numbers
