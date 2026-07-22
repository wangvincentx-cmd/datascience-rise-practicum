# Did Anyone See It Coming? A Machine-Graded Century of Economic Predictions, 1905–2009

**Authors:** Vincent Wang, Bode ___, Jonathan (Jeremy) Liu — Boston University RISE
*(author order / surnames / affiliations to confirm)*

> **Extended abstract — poster draft.** Numbers are the current committed results
> (`JeremysShit/`, `CHANGELOG.md`); the Narrative Economics strand is in progress
> (see Limitations). This is a working draft, not the final poster text.

## Motivation

Whether experts and the press can foresee economic turning points is an old
question with thin long-run evidence: the professional-forecasting record is
usually studied only back to the Survey of Professional Forecasters' 1968 start,
and newspaper forecasts have never been scored at scale because reading a
century of them by hand is infeasible. We use a large language model —
**validated against a human gold standard** — to extract and grade economic
predictions from a century of American newspapers, then score every prediction
against what actually happened. The organizing question is deliberately
information-time honest: *using only what was knowable when the prediction was
made, did anyone see it coming?*

## Data and methods

- **Corpus.** ≈4,100 candidate claims machine-extracted from Library of Congress
  full-text newspapers (1905–1963, 218 publishers) and the NYT Article Search
  API (1963–2009), spanning **19 episodes — crisis windows plus calm "placebo"
  controls**. After grading, **≈1,630 are genuine forward-looking predictions
  and ≈1,430 are scorable** (price claims before 1913 and employment claims
  before 1948 are left unscored, not guessed).
- **LLM grading, human-validated.** A six-model bake-off selected **gpt-4.1**,
  which agrees with a three-coder human gold standard at **Cohen's κ = 0.89**
  (is-this-a-prediction) and **0.90** (predicted direction) — near-perfect. The
  rubric was de-leaked and hardened; validation integrity (never editing human
  labels toward the model) is treated as a first-class constraint.
- **Ground truth.** NBER business-cycle chronology and FRED series (industrial
  production, CPI, unemployment). "No-change" bands are **human-calibrated**, and
  a Diebold–Mariano test is used for forecaster comparisons rather than a naïve
  bootstrap.
- **Leakage discipline / evaluation.** Every feature is information-time legal;
  splits are by episode, never random; and because the target is rare, accuracy
  is never the metric — hit rate, error direction, Brier, and permutation tests
  are.

## Results

**1. Forecast errors are asymmetrically, expensively optimistic.** Among
general-business predictions, misses are overwhelmingly *optimistic* (the paper
said "improve" and it didn't): **66% of crisis-window misses are optimistic vs
just 32% in calm controls** (binomial *p* = 4.5×10⁻¹⁰) — a clean crisis-vs-control
placebo split. The archetype is the **1929 Crash: 155 optimistic errors to 1
pessimistic** (10% hit rate). Under an outcome-severity-weighted "regret" loss,
error is concentrated in the highest-severity crises, and the asymmetry is
invariant to the cost weights. A permutation-tested model confirms the
mechanism: a claim's **own stated direction predicts whether it comes true**
(*p* = 0.0099).

**2. The press and the professionals are equally blind to turning points.**
Scored on identical ground truth, three independent benchmarks converge just
above a coin flip: **Survey of Professional Forecasters 54.1%**, **Livingston
economists 54.4%**, **Michigan households ≈55%**. Crucially, **the SPF issues a
"downturn" call in essentially 0% of surveys** — professionals almost never
forecast a contraction a year out (cf. Loungani's "failure to predict
recessions"). The optimism we find in the press is therefore not a lay-media
failing; it is shared with the experts. A Diebold–Mariano test finds **no
significant press-vs-economist difference** — we retire any "newspapers beat
economists" claim as unsupported at this sample size.

**3. A calibration correction.** Replacing the LLM's low-reliability confidence
label (κ = 0.19) with an **objective, reproducible hedging lexicon** (Hyland
2005) overturns an apparent "overconfidence" effect: on the objective measure,
assertive claims are **not** less accurate (0.516 vs hedged 0.485). Confident and
hedged forecasters were about equally right; the earlier effect was a label
artifact.

**4. Politics did not colour the forecasts.** Using each paper's hand-coded
political lean and the party controlling the White House, we test the classic
partisan-perceptual-bias hypothesis (Bartels 2002; Gerber & Huber 2010): did
papers forecast a rosier economy under their *own* party's president? We find
**no such effect** (aligned vs. opposed net optimism +0.59 vs. +0.57, Fisher
*p* = 1.00) — the optimism is broad, not partisan. (Descriptively, Socialist/left
papers were the least optimistic, but partisan samples are thin and
era-clustered; read as exploratory.)

**5. Narrative Economics (in progress).** We code each claim into Shiller's
perennial economic narratives ("new era," "fundamentally sound," "temporary
setback," …) to test whether *complacent* stories crowd out caution before the
worst crises. Infrastructure and a human-validation protocol are built; the
authoritative gpt-4.1 pass is pending (a lexical screen is too coarse to capture
framing — see Limitations).

## Limitations (stated as features, not hidden)

- **Post-1963 coverage is thin and crisis-windowed** (NYT headline/lead only),
  so pre/post-1963 and press-vs-SPF hit-rate comparisons are confounded by
  sampling; we report the sampling-robust prediction-mix contrast instead. A
  full-text ProQuest extension is under way to address this.
- **Band sensitivity.** Directional hit rates swing ~12–15 points across
  plausible no-change bands; every headline rate is reported with this
  sensitivity, and comparisons use Diebold–Mariano.
- **Residual label noise** (~3% of `is_prediction=yes` on spot-check) and one
  unvalidated feature (`voice`) are disclosed. The Narrative strand is not yet
  κ-validated.

## Conclusion

Across a machine-graded, human-validated century of American economic
predictions, forecasters — popular and professional alike — were **systematically
optimistic, most wrong exactly when it mattered, and structurally unable to call
turning points in advance.** The contribution is as much methodological as
substantive: a validated LLM-grading pipeline with strict information-time
discipline makes a century of qualitative forecasts quantitatively scorable, and
disciplined testing (permutation, Diebold–Mariano, calibrated bands) keeps the
honest negatives honest.

---

### Figures (candidates for the poster)
- `figures/fig_regret.png` — optimistic vs pessimistic errors by episode (centerpiece)
- `figures/fig_spf_benchmark.png` — professionals never forecast a downturn; the benchmark convergence
- `figures/fig_hit_by_episode.png` — directional accuracy by episode (1929 as the low)
- `figures/fig_optimism_gap.png` — predicted-improve vs actually-improved
- `figures/fig_hedging.png` — objective hedging vs the unreliable LLM label
- `figures/fig_optimism_timeline.png` — net optimism into the peak (supporting)
