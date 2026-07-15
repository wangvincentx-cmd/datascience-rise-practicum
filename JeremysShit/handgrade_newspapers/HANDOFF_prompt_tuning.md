# Handoff — LLM grading prompt tuning (economy arm)

## Objective
User asked whether switching the LLM grader from Groq/Llama-3.3-70b to a frontier
model ("luna" = GPT-5.6) would improve agreement with human hand-grades. The kappa
harness (`kappa.py`) is the only arbiter. Goal: get the two gated fields
(`is_prediction`, `direction`) over the 0.60 "substantial" floor so the pipeline can
proceed past the decision gate.

## Answer reached
**It was a prompt problem, not a model problem.** Rewriting `RUBRIC_PROMPT` in
`grade_claims.py` lifted both gated fields over 0.60 on the *same* cheap Groq model.
No model switch was needed. GPT-5.6 remains untested and is not currently justified.

## Results — vs human anchor (Vincent+Bode, who agree κ=1.00 on all fields)
All runs: Llama-3.3-70b-versatile on Groq, same 80 validation claims.

| Field          | old prompt | v1 rewrite | v2 refined |
|----------------|-----------|-----------|-----------|
| is_prediction  | 0.46      | 0.57      | **0.61** ✅ |
| direction      | 0.34      | 0.56      | **0.66** ✅ |
| topic          | 1.00      | 0.78      | 0.84      |
| confidence     | 0.23      | 0.18      | 0.13 ❌    |

Note: old-prompt numbers were computed on the 54/80 IDs the pre-existing
`claims_graded.csv` happened to contain; v1/v2 are the full 80. Movement is real
regardless.

## Files created / modified this session (NONE committed)
- `grade_claims.py` — `RUBRIC_PROMPT` rewritten:
  - `is_prediction`: explicit exclusion list (ads, unreadable OCR, refuse-to-forecast,
    present/past, conditional arithmetic, metaphor, event announcements, non-economic)
    PLUS a "grade the forecast, not the packaging" rule (real forecast buried in OCR
    or beside ad copy still counts).
  - `direction`: reassurance ("nothing to cause uneasiness") = `improve`, not
    `no_change`; `no_change` only when explicitly flat; `unclear` never a default.
  - `confidence`: a rewrite was tried and REVERTED (it made things worse, 0.18→0.13).
    Current file has the ORIGINAL confidence wording.
- `claims_raw_val80.csv` — 80-claim subset of `claims_raw.csv` (fast ~4-min test runs
  instead of the full 1,324-claim ~55-min run).
- `claims_graded_newprompt.csv` — v1 prompt output on the 80.
- `claims_graded_v2.csv` — v2 prompt output on the 80 (produced with the REWRITTEN
  confidence wording that has since been reverted).
- `handgrade_newspapers/handgrade_claude.csv` — Claude's own independent grading of
  the 80 (used for an exploratory Claude-vs-Groq check; NOT ground truth).
- `validation_sample.csv` — overwritten as a side effect of `grade_claims.py` runs.
- The real full-corpus `claims_graded.csv` was NOT touched; it still reflects the OLD
  prompt.

## IMPORTANT caveats
1. **Validation leakage.** Two illustrative examples in the current prompt — the
   Slichter recovery forecast (claim 1048) and the "recession...inevitable" one
   (claim 861) — are drawn FROM the 80-claim validation set. This slightly inflates
   the reported kappa on those two claims. De-leak before publishing: swap them for
   sentences outside the 80, or redraw the validation sample.
2. **Current prompt state ≠ any saved output.** `claims_graded_v2.csv` was made with
   the rewritten confidence wording (now reverted). is_prediction/direction (0.61/0.66)
   still hold because the revert only touched confidence, but to get the canonical
   confidence number under the current prompt you must re-run.
3. **Confidence is ~irreducibly ambiguous** (0.13–0.23). Not gated by the README.
   Report as a stated limitation, don't chase it.
4. **`voice` field is unvalidated** (never hand-graded) — flag if it shows as an
   important model feature.
5. **Groq free-tier keys drain fast** — a single 80-claim run largely exhausts a
   fresh account's daily budget; subsequent runs throttle to 300–970s waits. The full
   1,324-claim regrade needs a key with real budget. Three throwaway keys were used
   and are exposed in the chat log — rotate/revoke.

## Recommended next steps
1. De-leak the two in-prompt examples (caveat 1).
2. Re-run v2 prompt on `claims_raw_val80.csv` to confirm the canonical numbers under
   the current (reverted-confidence) prompt:
   ```
   OPENAI_API_KEY=<groq_key> python grade_claims.py --claims claims_raw_val80.csv \
       --out claims_graded_v2_final.csv --model llama-3.3-70b-versatile \
       --base-url https://api.groq.com/openai/v1 --sleep 2.5 --overwrite
   python handgrade_newspapers/kappa.py \
       --graders handgrade_newspapers/handgrade_vincent.csv \
                 handgrade_newspapers/handgrade_BLANK_bode.csv \
       --graded claims_graded_v2_final.csv
   ```
3. If confirmed ≥0.60 on is_prediction+direction, regrade the FULL corpus (~55 min,
   funded key) to regenerate the real `claims_graded.csv`:
   ```
   OPENAI_API_KEY=<groq_key> python grade_claims.py --claims claims_raw.csv \
       --out claims_graded.csv --model llama-3.3-70b-versatile \
       --base-url https://api.groq.com/openai/v1 --sleep 2.5 --overwrite
   ```
4. Commit the prompt change + this handoff. Then proceed to the scoring stage.
