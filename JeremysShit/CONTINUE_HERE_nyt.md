# ⏭️ CONTINUE HERE — NYT post-1963 re-pull (paused 2026-07-16 by user)

A broadened NYT download was **paused mid-run** at the user's request. Finish it,
then grade + score. Everything is resume-safe (dedupes on URL / resumes on
claim_id) — nothing below re-fetches or re-grades work already done.

Full context: see the "NYT coverage is severely under-sampled" bullet in
`../CHANGELOG.md`. This file is just the short resume checklist.

## Why the re-pull: fixing under-sampling
`download_nyt.py`'s `ECONOMY_PHRASES` was 9 rare crisis phrases → only 30 NYT
predictions for 1963-2010. Broadened to 16 LOC-aligned terms (`business outlook`,
`economic outlook`, `economic recovery`, `business recession`, …). Yield jumped
massively (~1,950 articles on disk vs 156 before).

## State at pause (article counts in election_arm/data/raw/nyt_economy_*.jsonl)
DONE with broadened terms: calm_1965=120, calm_1995=311, calm_2005=269,
crash_1987=265, dotcom_2001=881 (dotcom may be PARTIAL — was mid-window when paused).
STILL on old narrow counts (re-pull these): gfc_2008=53, gulf_1990=22,
oil_1973=12, volcker_1980=15.
Daily NYT quota used today: ~155/500 (plenty left).

## Steps to finish (run from JeremysShit/ unless noted)
1. Finish the download (resumes; re-hits every window but only appends new URLs):
   ```
   cd election_arm
   NYT_API_KEY=<key>  python download_nyt.py --arm economy
   cd ..
   ```
   Keys used this session (rotate after): mqcl4MPbUudLYuAM5NenZI4jDma0z4eV1EYu1lrGkx83G31Y
   (backup: olCws7rhEQnSbwZzCac8GWrgcJxkG1mvaIIGBnecKy0VAqFs)

2. **DECISION FIRST — cap the corpus before grading.** Broad terms are very
   uneven per window (dotcom_2001=881, calm_1995=311, but oil_1973/volcker still
   ~12-15). Grading everything costs OpenAI $ AND skews crisis/control balance.
   Recommend capping each window (e.g. random N per window) in
   `append_nyt_claims.py` before the merge, or subsample after. Discuss with user.

3. Merge → grade only new rows → re-score:
   ```
   python append_nyt_claims.py                     # idempotent, dedupes on page_url
   OPENAI_API_KEY=<key> python grade_claims.py --model gpt-4.1 \
       --base-url https://api.openai.com/v1 --sleep 0.35
   python score_claims.py
   ```
   Grader is gpt-4.1 (bake-off winner), NOT luna. ~19% of articles historically
   become predictions.

4. Then rerun the now-stale downstream (see CHANGELOG "Not done"):
   `tier2_analysis.py`, `model.py`, `model_figures.py`.

## Watch-outs
- NYT returns headline/lead only (no body text) → recall capped regardless of
  term count. Full-text depth needs library ProQuest, not more NYT calls.
- `claims_graded.csv` was DELETED earlier today and reconstructed from
  `claims_scored.csv`; don't delete it again — commit/back it up.
- Update CHANGELOG.md as you go (user's standing instruction).
