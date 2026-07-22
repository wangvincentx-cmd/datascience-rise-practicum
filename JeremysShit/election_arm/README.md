# Vincent's pipeline — integration status (2026-07-09)

Vincent's full two-arm pipeline lives here (his docs: [VINCENT_README.md](VINCENT_README.md)).
Integrated and verified against the economy arm in the parent folder.

## Done here already

- **All 21 of his offline tests pass** on this machine (`python test_offline.py`)
- **Livingston benchmark bridged and RUN** — his Stage 5 wanted a manual Fed download;
  `data/livingston_medians.csv` was generated from the parent folder's `medians.xlsx`
  (161 surveys, 1946–2026). Result, his NBER state-at-horizon metric:
  economists **83.9% overall** but **38.5% within 6 months of a business-cycle peak**
  (n=13) — experts excel in normal times, fail at turning points. Output:
  `data/scored_livingston.csv`.
- **EPU wired in** — `data/epu_monthly.csv` (historical policy uncertainty, 1900–2014,
  exported from `../tier2_analysis.py`). `analyze_economy.py` now merges it and prints
  an accuracy-by-EPU-tercile table; `model.py` picks it up as a feature automatically.
  On the parent arm, EPU was the #2 predictor of claim correctness.
- **`test_offline.py`'s LLM mock fixed (uncommitted, found 2026-07-22).** Since
  `extract_predictions.py` moved to calling OpenAI's REST `chat/completions`
  endpoint directly (`requests.post` + Bearer key, 2026-07-16), the test's old
  Anthropic-SDK-style `FakeClient(messages.create(...))` mock no longer matched
  the real call signature and was silently falling through to a live network
  call (401, no key in the test env) instead of testing anything. Replaced with
  a `requests.post`-level mock (`FakeResp`/`fake_post`). Needs a commit.

## Blocked on keys / people

- `download_loc.py` / `download_nyt.py` full runs → need `NYT_API_KEY`; NYT is
  multi-day (500 req/day)
- `extract_predictions.py` → needs `OPENAI_API_KEY` (test with `--limit 20` first;
  migrated off `ANTHROPIC_API_KEY` on 2026-07-16, see above)
- κ validation → two human graders after extraction

## Metric reconciliation (IMPORTANT for the writeup)

The two economy scorings are different metrics — never mix them in one chart unlabeled:

| | parent folder (`score_claims.py`) | here (`analyze_economy.py`) |
|---|---|---|
| question | direction of *change* over horizon | *state* (recession/expansion) at horizon |
| ground truth | FRED CPI/INDPRO/UNRATE magnitudes + NBER | NBER chronology only |
| typical hit rate | ~50% (3-way, banded) | ~80%+ (2-way, expansion-dominated) |
| Livingston result | 54.4% directional (1946–63) | 83.9% state / 38.5% near peaks |

Both are valid; the state metric is more readable for the poster headline, the
direction metric is stricter. Report both, labeled.

## Division of coverage (agreed)

- Parent economy arm: pre-1963 LOC full-text depth, FRED-magnitude scoring, EPU /
  geography / Michigan benchmarks, calibration
- This pipeline: elections 1896–2008, post-1963 economy windows via NYT (incl. the
  1987 negative case), NBER-state scoring, κ protocol, per-window models
