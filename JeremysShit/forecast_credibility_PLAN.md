# Forecast-credibility model — plan (pinned before building)

**Question:** given a forecast + the economy's state when it was made, what is the
probability the forecast comes true? A calibration/credibility model, not a
turning-point predictor (honest caveat: only ~8 recessions in the sample, so power
on downturn-calling is inherently limited — frame as calibration characterization).

## Data sources — LOCKED: US-only, three forecasters

| source | rows (gRGDP unit) | era | native extras |
|---|---|---|---|
| `greenbook` | ~1,804 (+1..+4q) / 2,735 (all horizons); 490 editions | 1967–2020 | own output gap/nowcast |
| `livingston` | ~few hundred (semiannual, ~160 rounds) | 1946–present | panel **disagreement** (Dispersion2.xlsx) |
| `spf` | thousands (individual microdata) + forecaster identity | 1968–present | dispersion; **Anxious Index** (P(GDP decline)) |

`newspaper` (already scored) can join later as a 4th source.

**Explicitly EXCLUDED (user decisions 2026-07-24):** Fed SEP (too recent, 2007+),
and ALL international sources (IMF WEO / OECD) — project stays US-scoped even
though international was the only real lever for turning-point power. Accepted
consequence: power on recession-calling stays limited (~8–12 US recessions);
frame as calibration, not turning-point prediction.

SPF files are directly curl-able (unlike Greenbook): `median_<var>_level.xlsx`,
`Individual_<VAR>.xlsx` (confirmed `Individual_RGDP.xlsx` = 683 KB),
`dispersion_<var>.xlsx` under the SPF `data-files/files/` path — a batch download
can be automated.

## Provenance — every row knows its origin (we are merging 3 different "who"s)

Core (harmonized, model-facing): `source` (greenbook|livingston|spf),
`forecaster_id` (fed_staff | livingston_median | spf_forecaster_<id>),
`variable_canonical` (real_gdp | industrial_production | unemployment | …),
`forecast_date`, `horizon_q`, `pred_growth`, `pred_direction`,
`realized_direction`, `hit`.

Provenance (traceability, NEVER fed to the model): `source_file` (exact file),
`variable_native` (gRGDP / RGDPX / RGDP), `horizon_native` (+4Q / 12M), `era`.

Source-native extras stay namespaced (`livingston_dispersion`,
`greenbook_output_gap`, `spf_forecaster_track_record`) and enter ONLY the
enrichment runs, never the shared-feature head-to-head.

Safeguards: (1) a **crosswalk table** documenting variable-code mappings across
sources; (2) a **merge audit** printed every build — rows per source × variable
× era — to catch double-loads/overlap.

**Independence caveat (must be reported):** the three sources overlap in time and
variable and all forecast the same ~8–12 US recessions, so combined n is large but
NOT n independent episodes. Keep `source` explicit; report source-stratified.

## Three models, compared

1. **Greenbook-only**
2. **Livingston-only**
3. **Combined** (`source` included as a disclosed feature — part of its edge is
   just knowing who spoke)

### What makes the 3-way comparison fair (guardrails)
- **Shared feature set = the intersection.** Prediction features (direction,
  magnitude/extremeness, horizon, revision) + economic-state features computable
  from FRED at the forecast date (output-gap proxy, growth momentum, yield-curve
  slope, months-since-NBER-trough, EPU). Source-native extras (Livingston
  dispersion, Greenbook's own gap) are reported SEPARATELY as enrichments, never
  mixed into the head-to-head.
- **Two eras reported:** each source on its FULL era (its own reach) AND on the
  **1967–2020 overlap** (same episodes — the fair contrast; this is the headline).
- **Cross-source transfer (the payoff):** train on Greenbook → test on Livingston,
  and vice versa. Good transfer ⇒ credibility is a general property of forecasts,
  not a quirk of one forecaster.
- **Source-stratified reporting** on the combined model, so Livingston (small n)
  isn't drowned by Greenbook volume.

## Target & features

- **y = `hit`**: did the forecast's banded direction (improve/worsen/no_change)
  match realized INDPRO/NBER over its horizon — the SAME `realized_direction`
  rule/band the newspapers, SPF, and greenbook_benchmark use. (Regression variant:
  continuous forecast error.)
- **X (~14 shared):** ~5 prediction features + ~8 economic-state features + `source`.

## Models & evaluation

- **Logistic regression (L2)** — primary, interpretable, paper-facing.
- **Shallow gradient-boosted trees** (depth ≤3, few trees, early stopping) —
  secondary; Greenbook/combined only (overfits Livingston-only's small n).
  Catches nonlinearity/interactions (e.g. horizon × turning-point-proximity).
- **Baselines to beat:** base-rate (~55–65% hit); "trust the forecast's own
  extremeness/revision" heuristic.
- **Split by era/episode, NEVER random** (windows share an outcome).
- **Metrics:** PR-AUC (on the miss / downturn-called class) + ROC-AUC + calibration
  curve / Brier + permutation importances, all with CIs (small n).

## Leakage discipline
- Economic-state features must be **real-time as of the forecast date** (Greenbook
  ships its own real-time gap/nowcast; use FRED real-time vintages, not revised).
- Livingston forecasts index *levels* → derive growth from MedianGrowthRate.xlsx;
  carry over the notebook's CPI/IP/GDP **rebasing-artifact** cleanup.

## Build order
1. Adapt `greenbook_benchmark.py` loader to the **All-Column-Format** folder
   (`gRGDP` first; then `gIP`/`UNEMP`/`gPCPI`), produce `greenbook_scored.csv`.
2. `forecast_credibility.py`: build the shared feature table, `source`-tagged, from
   Greenbook (+ Livingston), train the 3 models, run the comparison + transfer.
