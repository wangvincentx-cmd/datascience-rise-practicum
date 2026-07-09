# Who Saw It Coming? Elections, the Economy, and a Century of American Predictions (1900–2010)

## BU RISE — Merged Project Plan v3 (2026-07-09)

This merges two independently built pipelines into one project with two arms:

- **Economy arm** (this folder, Jonathan): newspaper economic predictions around 7 crisis +
  3 calm control windows (1905–1958, LOC Chronicling America), scored against NBER/FRED,
  benchmarked head-to-head against Livingston Survey economists (1946–2010).
- **Election arm** (Vincent's four scripts, to be dropped into `election_arm/`): newspaper
  election forecasts for presidential cycles 1896–2008 (LOC + NYT Article Search API),
  scored against actual winners, with source-type comparison (editorial vs. reported poll
  vs. betting odds vs. correspondent).

One question unifies them: **when has American prediction worked, and when has it failed?**
Elections = clean binary outcomes, recurring on schedule. Economy = continuous outcomes,
crisis vs. calm. Both built the same four stages independently (download → extract → score
→ model), which is itself a methods point: two teams converged on the same design.

---

## 1. Current status

| Piece | State |
|---|---|
| Economy scraper (`newspaper_scraper.py`) | DONE — 1,324 candidate claims, 910 pages, all 10 windows |
| Economy grading (`grade_claims.py`) | READY — needs `DEEPSEEK_API_KEY`; κ tooling included |
| Economy scoring (`score_claims.py`) | DONE — first heuristic pass run (see §5) |
| Livingston notebook | VERIFIED runs locally; robustness cell fixed |
| Election arm scripts | DELIVERED 2026-07-09 into `election_arm/` — his 21 offline tests pass; Livingston benchmark bridged from our medians.xlsx and RUN (83.9% overall / 38.5% near peaks); EPU wired into his analyzer+model; his downloaders/extractor await NYT + Anthropic keys. Metric reconciliation table in `election_arm/README.md` |
| Offline tests (`test_offline.py`) | DONE — 25 checks passing |
| Model stage (`model.py`) | DONE — episode-grouped split + EPU feature; GB 61.9% vs 51.0% baseline, AUC 0.686 |
| Tier 2 (`tier2_analysis.py`) | DONE — EPU vs accuracy, geography, Michigan three-way benchmark |

## 2. What each arm adopts from the other

**Economy arm adopts (already implemented here):**
1. Offline test harness — mock API responses through the real parsing functions
2. Model stage — predict claim correctness from claim features; **grouped split by episode**
   (Vincent's leakage insight: claims sharing one event share one outcome, so random splits cheat)

**Election arm adopts (Vincent's to-do list):**
1. **Human validation:** export a 20% sample of LLM-extracted claims, two people grade
   independently, report Cohen's κ (≥0.7 target). Copy the pattern from `grade_claims.py`
   (`validation_sample.csv`, `--kappa`). This is the #1 judge question.
2. **Search transparency log:** record total hits vs. pages taken per query per cycle
   (see `search_log.csv` here) — the "sampled, not cherry-picked" artifact.
3. **Simpler OCR fetch (optional):** page resource JSON exposes `resource.fulltext_file`
   directly; the recursive JSON walk can stay as fallback.
4. **Calibration:** he already captures a hedged flag — score it as a probability
   (assertive 0.9 / hedged 0.7) and report Brier, matching the economy arm.

## 3. Unified claim schema (the integration contract)

Both arms must emit scored claims with these columns so `merged_analysis` can pool them.
Extra arm-specific columns are fine; these are the shared core:

| column | values | economy arm | election arm mapping |
|---|---|---|---|
| `domain` | `economy` \| `election` | constant | constant |
| `arm_source` | `loc` \| `nyt` | `loc` | per record |
| `window` | episode / cycle label | e.g. "1929 Crash" | e.g. "1948 cycle" |
| `kind` | `crisis` \| `control` \| `election` | crisis/control | `election` |
| `publisher`, `state`, `date` | | as scraped | as scraped |
| `quote` | claim text | OCR sentence | corrected claim text |
| `voice` | editorial \| quoted_expert \| quoted_official \| reported_poll \| betting_odds \| correspondent \| unclear | LLM rubric | his `source_type` maps in directly |
| `confidence` | `assertive` \| `hedged` | LLM rubric | from his hedged flag |
| `speaker_name` | name or `na` | LLM rubric | his attribution field |
| `predicted_label` | direction / winner | improve/worsen/… | candidate or party |
| `realized_label` | | from NBER/FRED | actual winner |
| `hit` | 0/1 | | |
| `brier` | float | | |

File convention: each arm writes `claims_scored.csv` in its folder; a merged
`all_claims_scored.csv` is just `pd.concat` on the shared columns.

**What Vincent hands over:** `download_loc.py`, `download_nyt.py`, `extract_predictions.py`,
`analyze.py`, `model.py`, `data/ground_truth.csv`, `test_offline.py` → into `election_arm/`,
plus a small exporter mapping his scored output to the schema above.

## 4. Merged analyses (the poster's third act)

With both arms in one table, pooled questions no single arm can answer:

1. **Domain difficulty:** were elections or the economy easier to predict? (hit rate by
   domain, controlling for era)
2. **Voice, pooled:** editorials vs. experts vs. polls/odds across *both* domains —
   double the sample for the "whom to trust" figure
3. **Calibration, pooled:** overconfidence as a general phenomenon vs. domain-specific
4. **The model:** one classifier on the pooled table (grouped split by window); feature
   importances answer "which factors made predictions accurate" — the team's third goal
5. **Publisher consistency:** were papers accurate on elections also accurate on the
   economy? (scatter of per-publisher hit rates, both domains, min 10 claims each)

## 5. First real numbers (heuristic grading — LLM grading will sharpen these)

From 403 scorable economy claims (2026-07-09):

- **1929 Crash:** 75% of claims predicted improvement; hit rate **24.5%** — the
  "prosperity around the corner" failure, visible even with keyword grading
- **1945 Reconversion:** only 17% predicted improvement (the predicted depression
  never came); hit rate 43% — wrong in the *opposite* direction
- **Crisis vs. control:** 47.2% vs. 46.8% overall — but crisis windows are far more
  optimistic (55.9% vs. 46.8% predicting improvement)
- **Head-to-head 1946–63:** newspapers 50.5% [CI 41.1–59.8] vs. Livingston economists
  54.4% — statistically indistinguishable so far
- Publisher extremes: Milwaukee Leader 60%, Toledo Union Journal 20% (small n; needs
  the bigger scrape + LLM grading before naming names on a poster)

## 6. Division of labor & timeline (weeks 3–6)

| Week | Jonathan (economy) | Vincent (elections) | Shared |
|---|---|---|---|
| 3 | LLM-grade corpus; κ double-coding; scale scrape to 100 pages/term | Drop scripts into `election_arm/`; add κ validation + search log; finish NYT multi-day pulls | agree final schema |
| 4 | Scoring + Tier-1 figures; Livingston freeze | Score cycles; source-type table | build `all_claims_scored.csv`; pooled voice/calibration |
| 5 | Robustness (bands, second LLM, era shifts) | Robustness (phrase list, depth confound check) | model on pooled table; publisher-consistency scatter |
| 6 | — | — | poster assembly + talk |

Fallback ladder unchanged: each arm independently survives the other being cut; the
Livingston analysis alone is still a complete poster.

## 7. Honest limitations (merged)

All of v2.1's, plus: NYT post-1963 text depth ≠ LOC full text (recall confound — flag,
don't hide); phrase lists bound recall in both arms; LLM extraction validated on a sample,
not exhaustively; pre-1936 "polls" in the election arm are straw polls/betting odds, which
is a feature (what prediction looked like before polling) if framed honestly.
