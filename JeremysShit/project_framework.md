# Has America Gotten Better at Predicting Its Own Future?
## Forecast accuracy across eras of American history, 1946–2026
### BU RISE Project Framework

---

## 1. Research Questions

**Primary:** Has US economic forecast accuracy improved across historical eras since 1946?

**Secondary (each one becomes a poster panel):**
1. **Learning:** Is mean absolute forecast error lower in recent eras than earlier ones?
2. **Beating naive:** Do economists outperform a "no-change" baseline forecast (predicting next year = this year)? If not, in which eras did expertise actually matter?
3. **Bias:** Do forecasters systematically over- or under-predict in certain eras (e.g., underestimating 1970s inflation, overestimating post-2008 recovery)?
4. **Shock recovery:** After major world events (Korea, oil shocks, Volcker, 9/11, 2008, COVID), how long until forecast error returns to that era's baseline?
5. **Disagreement as early warning:** Does high disagreement among economists (forecast dispersion) predict large errors — i.e., does the expert community "know when it doesn't know"?

**Hypothesis (falsifiable):** Forecast error declined over time as economic science matured — but error spikes at shocks never shrank, meaning we got better at predicting normal times, not turning points.

---

## 2. Data

| Dataset | Years | Role |
|---|---|---|
| Livingston Survey — medians.xlsx | 1946– | Backbone: median forecasts of CPI, unemployment, GDP/GNP, industrial production at 6- and 12-month horizons |
| Livingston Dispersion2.xlsx | 1946– | Economist disagreement (75th–25th percentile spread) |
| Livingston individualdata.xlsx | 1946– | Stretch: individual-level analysis |
| Survey of Professional Forecasters | 1968– | Cross-check that findings aren't Livingston-specific |
| FRED actuals (CPI, UNRATE, GDP) | 1946– | Independent "what actually happened" series |

Era definitions (justify these in methods — cite standard periodizations):
- Postwar boom (1946–1965)
- Vietnam & stagflation (1965–1982)
- Great Moderation (1982–2000)
- Terror & financial crisis (2000–2012)
- Polarization & pandemic (2012–2026)

---

## 3. Methods

1. Compute predicted vs. actual % change per survey per variable → forecast error, absolute error, signed error (bias)
2. **Naive baseline:** error of assuming "next 12 months = last 12 months"; skill score = 1 − (economist error / naive error)
3. Aggregate by era: mean/median absolute error with **bootstrap 95% confidence intervals** (resample surveys within era — this is the statistical rigor mentors want)
4. Rolling 10-survey mean absolute error for the timeline figure
5. Shock recovery: define shock = error > 2 era-standard-deviations; count surveys until error back under 1 SD
6. Disagreement test: scatter + correlation of dispersion at time t vs. absolute error of the forecast resolved at t+12m; lagged correlation

---

## 4. Poster Layout (standard 36×48" symposium poster — 7 figures fills it easily)

```
+---------------------------------------------------------------+
|  TITLE: Has America Gotten Better at Predicting Its Future?   |
|  80 Years of Economic Forecasts Across American Eras          |
+---------------+----------------------------+------------------+
| INTRO &       |  FIG 1 (HEADLINE, centered, |  METHODS         |
| HYPOTHESIS    |  large): Rolling forecast   |  - data pipeline |
| - why polisci |  error 1946-2026, era       |  diagram (FIG 2) |
|   cares       |  boundaries as vertical     |  - error formula |
| - Livingston  |  lines, shocks annotated    |  - bootstrap CIs |
|   history     |  (oil shock, 2008, COVID)   |                  |
+---------------+----------------------------+------------------+
| FIG 3: Bar chart, mean abs   | FIG 4: Economists vs. naive    |
| error by era with bootstrap  | baseline skill score by era    |
| CIs (CPI, unemployment, GDP  | ("when did expertise matter?") |
| side by side)                |                                |
+------------------------------+--------------------------------+
| FIG 5: Signed error (bias)   | FIG 6: Shock recovery — small  |
| heatmap: era x variable,     | multiples of error around each |
| red=overpredict,             | major event, aligned at t=0    |
| blue=underpredict            | ("the anatomy of a surprise")  |
+------------------------------+--------------------------------+
| FIG 7: Disagreement vs.      | CONCLUSIONS + LIMITATIONS      |
| error scatter — "do experts  | + FUTURE WORK (Metaculus era,  |
| know when they don't know?"  | other countries)               |
+------------------------------+--------------------------------+
```

Every figure answers one research question — no filler. Figure 6 (small multiples of shocks) and Figure 5 (bias heatmap) are the visually distinctive ones people stop for.

---

## 5. Six-Week Timeline

| Week | Goal | Deliverable |
|---|---|---|
| 1 | Download data, read documentation, load & clean in pandas | Clean merged DataFrame; column glossary |
| 2 | Forecast errors for CPI + unemployment + GDP; first timeline plot | Draft Figure 1 |
| 3 | Era aggregation, bootstrap CIs, naive baseline | Figures 3 & 4; answer to primary question |
| 4 | Bias heatmap + shock recovery analysis | Figures 5 & 6 |
| 5 | Dispersion analysis; SPF cross-check; robustness (do results hold if era boundaries shift ±3 years?) | Figure 7; robustness appendix |
| 6 | Poster assembly, writing, practice talk | Final poster |

Buffer built in: if Week 4 runs long, drop the SPF cross-check — the story stands without it.

---

## 6. Improvements Over the Basic Version

- **Naive baseline (biggest upgrade):** "economists were wrong" is boring; "economists barely beat guessing until 1985" is a finding
- **Bootstrap CIs:** turns bar charts into defensible statistics — the #1 question judges ask is "is that difference real?"
- **Shock anatomy figure:** aligning all crises at t=0 makes an original, memorable visual
- **Robustness check on era boundaries:** shows you know your era choices are a researcher decision, not a fact
- **Political framing throughout:** tie each era's predictability to its politics (institutional stability, policy regimes, polarization) — this keeps it a polisci project, not just an econ exercise

## 7. Limitations to State Honestly

- 2 surveys/year → ~40 observations per era; mitigated by pooling 3 variables and bootstrapping
- Economic forecasts as proxy for "predicting the future" generally — defend the choice: economic expectations are measured consistently for 80 years; political predictions aren't
- Survey population changed over decades (survivorship, composition)
