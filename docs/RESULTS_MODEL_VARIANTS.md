# Model variants: penalty choice, the general-business subset, and what a pooled AUC gain actually means

Three questions asked of the AUC-0.647 model in `src/hit_predictor.py`, run
under its exact evaluation — leave-one-3-year-block-out, pooled out-of-fold
predictions, 22 blocks. Reproduce with:

```
python src/model_variants.py                   # all three parts
python src/model_variants.py --only penalty
python src/model_variants.py --only topic
python src/model_variants.py --only diagnose
```

Tables land in `data/models/model_variants_*.csv`.

**Everything tried is below, including the configurations that lost, and
including a conclusion this document reached and then had to withdraw.** A grid
where only the winner is reported is a grid that cannot be checked.

> **Read part C first if you read nothing else.** It is the reason parts A and B
> are written the way they are, and it changes how *every* pooled AUC in this
> project should be read — including the headline.

---

## A. Penalty: L2 vs L1 vs elastic net

### What was and was not varied

| knob | setting | why |
|---|---|---|
| loss | log-loss, fixed | changing it makes the model something other than logistic regression |
| optimiser | lbfgs (L2), saga (L1, elastic net) | **cannot improve the answer** — penalised log-loss is convex. saga appears only because lbfgs cannot fit a non-smooth penalty, and because saga is *proximal* it produces exact zeros where a gradient method like Adam would produce 1e-7 and destroy the sparsity that is the entire point |
| penalty | **varied** — L2, L1, elastic net (l1_ratio 0.5) | the one knob that changes the fit |
| C | **varied** — 0.005 … 1.0 | two orders of magnitude around the incumbent 0.5. Extended downward mid-analysis when 0.01 turned out to be the edge of the original grid |
| max_iter | 2000 (lbfgs), 5000 (saga) | comparing a converged fit to an unconverged one is not a comparison of penalties |
| features | spec 5, unchanged | a fair fight isolates the penalty |

Worth stating because it is a standing confusion: **Adam is an optimiser, not a
loss.** For L2 the optimiser is irrelevant by construction — one global minimum,
every solver finds it. For L1 the optimiser matters only in that the wrong
*family* of optimiser fails to produce exact zeros.

### All 21 configurations

`nonzero` counts coefficients surviving an in-sample fit on all rows — a
*description* of what the penalty keeps, not an evaluation. `folds won` counts
held-out blocks where the cell beat the incumbent (L2, C=0.5).

| penalty | C | ROC-AUC | PR-AUC | Brier | nonzero | folds won |
|---|---|---|---|---|---|---|
| L2 | 0.005 | 0.651 | 0.662 | 0.234 | 55/56 | 10/22 |
| L2 | 0.01 | 0.653 | 0.664 | 0.234 | 55/56 | 12/22 |
| L2 | 0.02 | 0.654 | 0.664 | 0.235 | 55/56 | 11/22 |
| L2 | 0.05 | 0.653 | 0.664 | 0.237 | 55/56 | 10/22 |
| L2 | 0.1 | 0.652 | 0.662 | 0.238 | 55/56 | 10/22 |
| **L2** | **0.5** | **0.647** | **0.652** | **0.241** | **55/56** | **— incumbent** |
| L2 | 1.0 | 0.645 | 0.648 | 0.242 | 55/56 | 10/22 |
| L1 | 0.005 | 0.655 | 0.645 | 0.232 | **12/56** | 10/22 |
| L1 | 0.01 | 0.651 | 0.653 | 0.234 | 17/56 | 11/22 |
| L1 | 0.02 | 0.656 | 0.660 | 0.233 | 24/56 | 9/22 |
| **L1** | **0.05** | **0.657** | 0.664 | 0.234 | **31/56** | 11/22 |
| L1 | 0.1 | 0.656 | 0.665 | 0.235 | 40/56 | 11/22 |
| L1 | 0.5 | 0.648 | 0.654 | 0.240 | 44/56 | **16/22** |
| L1 | 1.0 | 0.645 | 0.648 | 0.242 | 46/56 | 14/22 |
| elastic net | 0.005 | 0.655 | 0.655 | 0.232 | 17/56 | 11/22 |
| elastic net | 0.01 | 0.654 | 0.658 | 0.233 | 25/56 | 10/22 |
| elastic net | 0.02 | 0.656 | 0.662 | 0.233 | 31/56 | 11/22 |
| elastic net | 0.05 | 0.656 | 0.665 | 0.234 | 40/56 | 10/22 |
| elastic net | 0.1 | 0.655 | 0.664 | 0.236 | 42/56 | 13/22 |
| elastic net | 0.5 | 0.648 | 0.653 | 0.241 | 46/56 | 13/22 |
| elastic net | 1.0 | 0.645 | 0.648 | 0.242 | 48/56 | 11/22 |

### What this says

**1. Do not switch the penalty or C. The pooled gains are not what they look
like.** Twelve cells beat the incumbent's 0.647 on pooled AUC, some by as much
as +0.010. Part C shows those gains are **entirely cross-era** — the candidates
are level with the incumbent at ranking forecasts *within* an era, which is the
only thing this project claims to do. Reporting 0.657 as an improvement over
0.647 would be claiming a skill gain the fold data says does not exist.

**2. The incumbent C=0.5 is nonetheless the worst-regularised setting tried,
and it was never tuned on this corpus.** The documented grid search in
`model.py` was for the v1 TF-IDF model on 843 episodes. Every L2 cell with
C ≤ 0.1 beats it on all three pooled metrics, and the L2 optimum is *interior*
(rises to ~0.02, falls away below 0.003), so this is a plateau rather than a
boundary artifact. The one defensible reason to switch is **calibration**:
Brier improves from 0.241 to 0.234, and `RESULTS_MACRO.md` already has to
disclose that this model's probabilities are "too extreme at both ends... use
the output as a ranking, not as literal odds." Stronger shrinkage partly fixes
that. If the model is ever deployed as *odds* rather than a *ranking*, retune C.
For the poster, which uses it as a ranking, it changes nothing.

**3. The real prize is sparsity, and it is not an accuracy claim.** L1 at
C=0.05 matches the incumbent while zeroing 25 of 56 coefficients; at C=0.005 it
keeps **12 of 56** and is still pooled-level. The correct sentence is *"a lasso
fit deletes 25 of 56 coefficients without loss of accuracy"* — **not** "with a
gain."

**4. Lasso independently confirms the interaction finding.** Given the freedom
to delete any coefficient, L1 keeps **all four** direction × economy terms at
every C tried, and `x_dir_epu` is the largest coefficient in the model by a
factor of 1.5:

| surviving coefficient | value |
|---|---|
| `x_dir_epu` | +1.264 |
| `c_direction_no_change` | −0.869 |
| `x_dir_sign` | −0.463 |
| `x_dir_stock_ret6` | +0.392 |
| `c_topic_general_business` | +0.296 |
| `x_dir_stock_drawdown` | +0.287 |
| `c_topic_prices` | −0.247 |
| `m_unrate` | +0.241 |
| `x_dir_ip_accel` | +0.229 |

Full list: `data/models/model_variants_l1_survivors.csv`. **This is the one
result in part A worth putting on a poster**, because it does not depend on an
AUC comparison at all. A method whose entire job is deleting things refuses to
delete those four terms — `RESULTS_MACRO.md`'s central claim arriving by a
third independent route, after the stratified attribution and permutation
importance. It also re-raises the EPU caveat: the largest surviving coefficient
is the one built from newspaper text.

**A note on collinearity when reading that list.** Three macro columns are
exactly linearly dependent — `ip_accel = ip_growth_6m − ip_growth_12m`
(`macro_context.py:125`). L2 spreads weight across all three, so no single one
of those coefficients means what its name says; L1 picks one and zeroes the
rest. The honest caption for `x_dir_ip_accel` surviving is *"of three redundant
output measures, acceleration is the one that carries it"*, not "output growth
does not matter."

---

## B. General business only

61% of the corpus (8,713 of 14,251 claims), hit rate **0.561** against a pooled
0.513. Same spec, same CV, 22 blocks. `c_topic` is dropped — it is constant
inside the subset.

| model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| claim features only | **0.484** | 0.507 | 0.240 |
| economy only | 0.591 | 0.645 | 0.250 |
| claim + economy (additive) | 0.627 | 0.697 | 0.245 |
| + direction × economy | 0.665 | 0.725 | 0.237 |
| + direction × economy, L1 (C=0.1) | 0.669 | 0.732 | 0.232 |

### What this says

**1. Within one topic, how a forecast is written predicts nothing — it is
*below* chance (0.484).** In the pooled model, claim features alone scored
0.561. Removing topic variation drops that to 0.484. A meaningful part of the
pooled "wording matters" signal was the model learning **which topic is easy to
be right about** — a fact about the scoring series and its base rate, not about
forecasting or journalism. This is the cleanest single demonstration of the
poster's own thesis anywhere in the project, and it belongs next to the "no
smart paper" panel.

**2. The interaction structure survives the restriction.** It adds **+0.038
AUC** over the additive model here (0.627 → 0.665), against +0.066 pooled. On a
homogeneous topic, with the topic shortcut removed, essentially all of the
model's skill is the economy interacted with direction.

**3. Do not read 0.665 as "better than 0.647."** Different sample, different
base rate (0.561 vs 0.513), which alone changes what an AUC of a given size
means. The valid comparison is the ladder *within* this table.

**4. The L1 row is a sparsity result, not an accuracy result** — same caution as
part A.3. Survivors:
`data/models/model_variants_general_business_l1_survivors.csv`. The ordering
differs from the pooled fit: `c_direction_no_change` dominates at −2.05, and
`m_stock_drawdown` and `m_unrate` enter as *main* effects, which is worth noting
rather than smoothing over.

---

## C. What a pooled AUC gain actually means

**This section exists because an earlier draft of this document recommended
switching to C=0.02 on the strength of a +0.007 pooled AUC gain. That
recommendation was wrong and is withdrawn.** The diagnostic that caught it is
now `fold_decomposition()` in `src/model_variants.py`, run on every claimed gain
including the poster's own.

### The problem

Pooled out-of-fold AUC ranks all 14,251 predictions against each other,
**including claims from different eras**. So it rewards two separable abilities:

1. **within an era**, ranking hits above misses — forecast-level skill, the
   thing this project claims to measure
2. **across eras**, knowing 1929 was a worse year to be optimistic than 1959 —
   macro information, which a model gets for free simply by putting its scores
   on a consistent scale across decades

A model can add several points of pooled AUC while being exactly level on (1).
Per-fold deltas separate them.

### Every claimed gain, decomposed

| comparison | pooled | folds won | mean fold | **size-weighted fold** |
|---|---|---|---|---|
| L2 C=0.02 vs incumbent | +0.0068 | 11/22 | +0.0020 | **−0.0031** |
| L1 C=0.05 vs incumbent | +0.0103 | 11/22 | +0.0015 | **−0.0004** |
| **interacted vs additive (the headline)** | **+0.0659** | **17/22** | **+0.0257** | **+0.0468** |

`data/models/model_variants_fold_decomposition.csv`.

**The two penalty candidates are pure cross-era gains.** Both are within noise
of zero on the size-weighted per-fold delta, and both are marginally *negative*.
Their pooled improvement is calibration consistency across decades, not better
forecast discrimination. Neither should be reported as a better model.

**The headline is not.** The interaction block gains +0.047 size-weighted
per-fold and wins 17 of 22 blocks — a genuine within-era improvement that
survives exactly the test the penalty candidates fail. The poster's central
claim passes its own audit.

### One further observation

Mean per-fold AUC exceeds pooled AUC for every specification (interacted:
0.667 mean-fold vs 0.647 pooled). The model is *better* at ranking within an era
than the headline number suggests; cross-era score incomparability is dragging
the pooled figure down. That is a defensible reason to report the fold-level
number alongside the pooled one — but it has never been done in this project, so
0.647 should stay the quoted figure until that change is made deliberately
everywhere.

### The rule this establishes

**Any future pooled-AUC comparison in this project must be accompanied by the
size-weighted per-fold delta before it is called an improvement.** Fold-win
count alone is insufficient — it is a sign test that ignores magnitude, which is
why the headline (17/22) and a null (11/22) both needed the magnitude column to
be told apart properly.

---

## Summary for the methods page

- The optimiser is not a modelling choice here and was not varied; the penalty
  is, and was — 21 configurations.
- **No penalty or C change is adopted.** Every candidate's pooled gain is
  cross-era only; within-era they are level with the incumbent. 0.647 with L2
  `C=0.5` stands, now with a sensitivity analysis behind it rather than an
  unexplained default.
- **A lasso fit deletes 25 of 56 coefficients without loss of accuracy and
  keeps all four direction × economy terms** — an independent confirmation of
  the interaction finding that does not rest on an AUC comparison.
- **On general business alone, claim features score 0.484 — below chance.**
  Wording carries nothing once topic is held fixed.
- **The headline's +0.066 survives fold-level decomposition** (+0.047
  size-weighted, 17/22 blocks) where the penalty candidates do not.
