# Did Anyone See It Coming? A Machine-Graded Century of Economic Predictions, 1905–2009

**Authors:** Vincent Wang, Bode ___, Jonathan (Jeremy) Liu — Boston University RISE
*(author order / surnames / affiliations to confirm)*

> **Extended abstract — poster draft.** Numbers are the current committed results
> (`JeremysShit/`, `CHANGELOG.md`); the Narrative Economics strand is kappa-
> validated at moderate confidence (see Result 5 and Limitations). This is a
> working draft, not the final poster text.

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
- **LLM grading, human-validated.** A seven-model bake-off selected **gpt-4.1**,
  which agrees with a three-coder human gold standard at **Cohen's κ = 0.89**
  (is-this-a-prediction) and **0.90** (predicted direction) — near-perfect. The
  rubric was de-leaked and hardened; validation integrity (never editing human
  labels toward the model) is treated as a first-class constraint.
- **Ground truth.** NBER business-cycle chronology and FRED series (industrial
  production, CPI, unemployment). "No-change" bands are **human-calibrated**, and
  a Diebold–Mariano test is used for forecaster comparisons rather than a naïve
  bootstrap.
- **Calm control windows** are selected by an explicit, pre-specified rule —
  mid-expansion, no NBER recession within 12 months of the window — not
  post-hoc: 1905/1925/1955 (LOC-era) and 1965/1995/2005 (NYT-era) all satisfy it.
- **Leakage discipline / evaluation.** Every feature is information-time legal;
  splits are by episode, never random; and because the target is rare, accuracy
  is never the metric — hit rate, error direction, Brier, and permutation tests
  are.

## Results

**1. Forecast errors lean optimistic, sharply so in crises — directionally
clear, not yet significant at the unit of analysis that matters.** Among
general-business predictions, misses skew *optimistic* (the paper said
"improve" and it didn't): pooling every claim, **66% of crisis-window misses
are optimistic vs just 32% in calm controls** (binomial *p* = 4.5×10⁻¹⁰). That
claim-level test treats each claim as an independent draw, which they are not
— claims inside one episode share wire-service copy and one macro reality, so
the p-value is anti-conservative. Re-tested three ways at the **episode**
level (13 crisis vs. 6 control episodes, the defensible independent unit) —
Mann–Whitney U on episode medians (p = 0.146), an **exact** cluster-
permutation test (episode as the randomization unit, all C(19,6) = 27,132
label assignments enumerated, claim-level pooled statistic so it doesn't
throw away episode size like Mann-Whitney does: observed gap **+34.3pp**,
exact one-sided **p = 0.161**), and an episode-cluster bootstrap on the gap's
size (95% CI **[−1.1pp, +63.9pp]**) — and all three agree: the gap does
**not** clear conventional significance, but the interval barely touches
zero and most of its mass is a large, real-looking effect. That convergence
across three different valid tests is itself informative: the non-
significance is a genuine power limit (19 episodes), not an artifact of which
test was picked. The archetype, **1929 Crash: 155 optimistic errors to 1
pessimistic** (10.6% hit rate, 95% CI [7.0%, 15.9%]), is real and large on its
own, but the century-wide asymmetry needs more crisis episodes — the pending
post-1963 ProQuest expansion is the natural next step — before it is a
confirmed result rather than a large, directionally consistent, underpowered
one.
Under an outcome-severity-weighted "regret" loss, error concentrates in the
highest-severity crises, and where the asymmetry holds it is invariant to the
cost weights. A permutation-tested model (LeaveOneGroupOut by episode, labels
shuffled 200×) separately confirms a related, better-powered mechanism: a
claim's **own stated direction predicts whether it comes true** (*p* = 0.0099).

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
label (κ = 0.17 for gpt-4.1, the grader used on the scored corpus) with an
**objective, reproducible hedging lexicon** (Hyland 2005) overturns an apparent
"overconfidence" effect: on the objective measure,
assertive claims are **not** less accurate (0.516 vs hedged 0.485). Confident and
hedged forecasters were about equally right; the earlier effect was a label
artifact.

**4. A claim's direction interacts with time and policy uncertainty — a lead,
not a settled effect.** Beyond the marginal effect Result 1's model already
uses, SHAP interaction values (LeaveOneGroupOut-by-episode CV, permutation-
tested at 200 shuffles) find **7 of 8 flagged feature pairs significant at
p = 0.005** (the maximum resolution at that many shuffles): a claim's own
predicted direction interacts with calendar year and with policy uncertainty
(EPU) at time of printing, and unified-government/president-party interact
with year. This does **not** by itself separate a genuine political-climate
effect from political climate simply proxying a long-run time trend (year and
EPU are themselves correlated across 120 years) — political-climate features
stay excluded from the model's *marginal* feature set (Result 1) on that
basis; this is a reported interaction lead, not a reversal of that call.

**5. Narrative Economics — a moderate-confidence reversal of the naive
hypothesis.** We code each claim into Shiller's perennial economic narratives
("new era," "sound fundamentals," "temporary setback," "panic/fear,"
"recovery/normalcy," "none") via an authoritative LLM pass over all 1,628
scored claims (kappa = 0.58 against a two-coder human consensus, n = 80 —
Landis-Koch "moderate," just under this project's 0.6 "substantial" bar;
confusion concentrates at close, related labels — e.g. sound_fundamentals vs
recovery_normalcy — not random noise or outright polarity flips). At that
confidence level, the data run **opposite the naive "complacency precedes a
crash" story**: `panic_fear` claims have the LOWEST hit rate of any narrative
(34.0%, n=285) and `new_era` claims the HIGHEST (64.7%, n=51); complacent
narratives overall hit MORE often than not (52.1% vs 46.3%). Complacent-
narrative share by episode also runs the "wrong" way for the classic
hypothesis: several calm-control periods (1965: 71.4%, 1925: 61.8%) show MORE
complacent framing than most crisis episodes, including 1929 itself (38.6%)
and 2008 (35.3%). Reported at moderate, not settled, confidence given the
kappa; a finer split between `sound_fundamentals` and `recovery_normalcy`
(the single biggest human/LLM confusion) is the natural next step.

## Limitations (stated as features, not hidden)

- **Post-1963 coverage is thin and crisis-windowed** (NYT headline/lead only),
  so pre/post-1963 and press-vs-SPF hit-rate comparisons are confounded by
  sampling; we report the sampling-robust prediction-mix contrast instead. A
  full-text ProQuest extension is under way to address this.
- **Band sensitivity.** Directional hit rates swing ~12–15 points across
  plausible no-change bands; every headline rate is reported with this
  sensitivity, and comparisons use Diebold–Mariano.
- **Residual label noise** (~3% of `is_prediction=yes` on spot-check) and one
  unvalidated feature (`voice`) are disclosed. The Narrative strand is
  κ-validated at moderate confidence (κ=0.58, n=80) — see Result 5.
- **Several angles were tested and ruled out, not hidden.** Forecaster
  disagreement (both per-claim and episode-level vs. crisis severity),
  publisher geography/financial-center proximity, and political-climate/
  publisher-lean features were all tested against the model under the same
  permutation-test validation gate the shipped features had to clear, and did
  not (disagreement measurably hurt one model; geography and political climate
  were indistinguishable from noise under leave-one-episode-out CV). Reported
  so the shipped feature set reads as filtered, not incomplete — code and data
  for all three remain in the repo (`disagreement.py`, `disagreement_severity.py`,
  `tier2_analysis.py`'s geography section, `political_climate.csv`).
- **Episode count limits statistical power, checked three ways, not just
  hedged once.** Only 19 episodes total (13 crisis, 6 control) means
  episode-level tests — the correct unit of analysis once within-episode
  correlation is accounted for — are underpowered; the optimism-asymmetry
  result (Result 1) is directionally consistent and the effect size is large
  (exact cluster-permutation gap +34.3pp, bootstrap 95% CI [−1.1pp, +63.9pp])
  but does not clear conventional significance at this n (exact p = 0.161).
  `results_by_episode.csv` now reports a 95% Wilson CI per episode; several
  episodes (1987 Crash n=14, 1990 Recession n=10, 1995 Calm n=15, 2008 GFC
  n=17) have wide enough intervals that individual episode rates should be
  read as illustrative, not precise.

## Conclusion

Across a machine-graded, human-validated century of American economic
predictions, forecasters — popular and professional alike — leaned
**optimistic, especially around the worst crises, and were structurally
unable to call turning points in advance** — a pattern that is clear and
large in specific episodes (1929 above all) and directionally consistent
century-wide, but not yet statistically confirmed at the episode level given
only 19 episodes. The contribution is as much methodological as substantive: a
validated LLM-grading pipeline with strict information-time discipline makes a
century of qualitative forecasts quantitatively scorable, and disciplined
testing (permutation, Diebold–Mariano, calibrated bands, and — new — testing
at the episode rather than claim level) keeps the honest negatives honest.

---

### Figures (candidates for the poster)
- `figures/fig_regret.png` — optimistic vs pessimistic errors by episode (centerpiece)
- `figures/fig_spf_benchmark.png` — professionals never forecast a downturn; the benchmark convergence
- `figures/fig_hit_by_episode.png` — directional accuracy by episode (1929 as the low)
- `figures/fig_optimism_gap.png` — predicted-improve vs actually-improved
- `figures/fig_hedging.png` — objective hedging vs the unreliable LLM label
- `figures/fig_optimism_timeline.png` — net optimism into the peak (supporting)
