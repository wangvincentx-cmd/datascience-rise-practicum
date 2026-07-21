# ✅ RESOLVED — NYT post-1963 re-pull (closed 2026-07-21)

This file tracked the NYT post-1963 re-pull, capping, merge, grading, and
downstream rerun. All of it is done; see `../CHANGELOG.md`'s economy-arm
"Done so far" section for the full history (search "NYT under-sampling
fixed" and "Resumed the stalled NYT downloads"). Summary:

- All 9 post-1963 windows re-downloaded to real depth (gulf_1990=456,
  oil_1973=179, volcker_1980=165, gfc_2008=873, etc.) — `search_phrase()`'s
  pagination bug (trusted NYT's unreliable `meta.hits` field) fixed along
  the way.
- Capped at 150 claims/episode (`append_nyt_claims.py`'s `PER_WINDOW_CAP`,
  matched to the LOC episodes' own 87-212 scale) before merging, resolving
  the capping decision this file originally flagged as needing discussion.
- Merged, graded on `gpt-4.1`, and rescored end-to-end; `tier2_analysis.py`,
  `tier3_robustness.py`, `model.py`, `model_figures.py` all rerun against
  the unified corpus.
- The separate `claims_graded_expanded.csv` merge-or-hold-out question (a
  different LOC recall-audit rescrape, not the NYT re-pull) was also closed
  2026-07-19 — user said "remove any duplicates and merge and rerun."

**Security note — still needs your action:** this file previously had two
live NYT API keys pasted in plaintext, one matching `bill_arm/.env`'s live
key. They're gone from the working copy but still recoverable from git
history (commit `1e201bc`). **Rotate both at developer.nytimes.com** — a
working-tree edit doesn't undo the history exposure. Not confirmed done as
of 2026-07-21 (tracked in `CHANGELOG.md`'s "Needs to be done by you").
