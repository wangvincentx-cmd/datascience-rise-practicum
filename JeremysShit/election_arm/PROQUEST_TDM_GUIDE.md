# ProQuest TDM Studio setup guide (economy/election arm)

**Audience:** an AI assistant (or person) helping a teammate stand up the ProQuest
full-text extraction pipeline for `JeremysShit/election_arm/`. Read this end-to-end
before running anything — most of it is failure modes we already hit, and the fixes.

---

## 0. What this is, in one paragraph

ProQuest TDM Studio is a locked-down cloud VM (AWS SageMaker Jupyter) that gives
full-text newspaper articles the NYT Article Search API can't (it returns only
headline+abstract). We use it to feed the **economy arm** (and could feed the
election arm) with rich `ocr_text`. Each newspaper article becomes one structured
economic-forecast record, extracted by an LLM, then scored against NBER recession
dates. The pipeline is the same contract as the `loc` and `nyt` sources — ProQuest
is just a third `source`.

---

## 1. THE constraint that shapes everything

**Full text cannot leave the VM.** ProQuest datasets are copyrighted; the export
mechanism only ships derived data, under a shared **~15 MB / rolling 7-day** cap.

Consequence — the pipeline is split at the text boundary:
- **Inside the VM:** parse XML → extract forecasts with an LLM → strip the text.
- **Leaves the VM:** only the label-only `pred_*.export.jsonl` (no `claim_text`).
- **On the Mac:** scoring and analysis run on those labels (they're text-free).

Anything that needs the article text — reading claims, kappa validation, the
text-feature model — **must run in the VM**. Only numbers/labels come out.

---

## 2. The scripts and what each does

| Script | Runs where | In → Out |
|---|---|---|
| `tdm_parse.py` | VM | ProQuest XML folder → `data/raw/proquest_{arm}_{window}.jsonl` |
| `extract_gpt.py` | VM | raw jsonl → `data/predictions/pred_proquest_economy_{window}.jsonl` (via in-VM GPT proxy) |
| `strip_for_export.py` | VM | pred jsonl → `pred_*.export.jsonl` (removes `claim_text`) |
| `run_all_economy.sh` | VM | batches parse→extract→strip over all 9 windows, bundles `proquest_exports.tar.gz` |
| `sample_claims.py` | VM | prints N random claims to eyeball extraction quality |
| `validate_kappa.py` | VM | draws a sample + computes Cohen's kappa (human validation) |
| `analyze_economy.py` | Mac | pred jsonl (text-free) → NBER scoring, crisis-vs-placebo table |
| `model.py` | VM (for text features) | `scored_economy.csv` → predicts `hit`, prints feature importances |

`extract_predictions.py` is the **non-ProQuest** extractor (loc/nyt sources, runs
outside the VM, now uses gpt-4.1 per the gold bake-off). `extract_gpt.py` is the
ProQuest-proxy twin; keep their `ECONOMY_PROMPT` in sync.

---

## 3. VM environment — the exact details

**Directories:**
- Scripts / working dir: `/home/ec2-user/SageMaker/election_arm/`
- Datasets land at: `/home/ec2-user/SageMaker/data/{dataset_name}/` (one XML per article)

**Python:** use the sample conda env — it has `openai`, `lxml`, `pandas`. The default
`python` intermittently lacks them.
```
/home/ec2-user/SageMaker/.conda/envs/sample-2025.12.578/bin/python
```
(The env name may differ on another workbench — find it with `conda env list` and test
`<path>/bin/python -c "import lxml, openai"`.)

**Installing packages:** `pip` is blocked (no internet). `conda` works via ProQuest's
internal mirror:
```
conda install -n sample-2025.12.578 lxml -y
```
Then verify in the *exact* interpreter the scripts call, not just `python`.

**The in-VM GPT proxy (the key enabler):** ProQuest ships an OpenAI-compatible proxy so
you can call an LLM *without* an external key and *without* internet. `extract_gpt.py`
auto-discovers it from ProQuest's sample notebook. One-time setup:
```
jupyter nbconvert --to script --stdout \
  ".../ProQuest TDM Studio Samples/GPT_Batch_Processing.ipynb" > gpt_sample.txt
```
Observed values (yours may vary — the script discovers them, don't hardcode):
- base_url: `https://agai-proxy.prod.int.tdmstudio.proquest.com/large-language-models-openai-compatible/`
- key file: `/home/ec2-user/SageMaker/.token/.agaitoken`
- model: `gpt_4o_mini`

**Getting scripts INTO the VM (it can't `git pull` — no internet):** paste them in via
the Jupyter *terminal*. Two reliable methods:
- Heredoc: `cat > file.py <<'PYEOF'` … `PYEOF`
- base64 one-liner (best for long files / flaky clipboards):
  on the Mac `base64 -i file.py | tr -d '\n'`, then in the VM
  `echo <blob> | base64 -d > /home/ec2-user/SageMaker/election_arm/file.py`

---

## 3.5 Notebook bootstrap: empty workbench → ready to run

This is the part that trips people up. Do it once per fresh workbench, in order.

**A. Transfer all scripts + config in ONE shot.** The VM can't `git pull`, so bundle
the needed files into a tarball, base64 it, and paste one command. On the **Mac**, from
`JeremysShit/election_arm`, generate the paste:
```
tar czf - tdm_parse.py extract_gpt.py strip_for_export.py run_all_economy.sh \
  sample_claims.py validate_kappa.py analyze_economy.py model.py \
  data/windows_economy.csv data/nber_recessions.csv data/proquest_datasets.csv \
  data/epu_monthly.csv | base64 | tr -d '\n'
```
Copy that blob, then in the **VM Jupyter terminal** paste ONE command:
```
mkdir -p /home/ec2-user/SageMaker/election_arm && cd /home/ec2-user/SageMaker/election_arm && \
echo <PASTE_BLOB_HERE> | base64 -d | tar xzf -
```
That recreates every script and the `data/` CSVs in the right layout. (Datasets and
prediction outputs are NOT transferred — datasets get built in the dashboard, §4;
predictions get generated in §5.)

**B. Confirm the Python env has the packages.** Find the sample env and test the exact
interpreter the scripts use:
```
conda env list                       # find the sample-* env name
/home/ec2-user/SageMaker/.conda/envs/<sample-env>/bin/python -c "import lxml, openai; print('ok')"
```
If it errors on `lxml`, install into that env (`conda`, not `pip` — no internet):
```
conda install -n <sample-env> lxml -y
```
Then set `run_all_economy.sh`'s `PY=` line to that interpreter's full path.

**C. Expose the GPT proxy** so `extract_gpt.py` can auto-discover it:
```
jupyter nbconvert --to script --stdout \
  ".../ProQuest TDM Studio Samples/GPT_Batch_Processing.ipynb" > gpt_sample.txt
```

**D. Smoke-test the proxy** (one throwaway call — proves the key/URL/model resolve and the
daily quota isn't already spent):
```
<sample-python> -c "
from extract_gpt import make_client
class A: sample='gpt_sample.txt'; base_url=key_file=model=None
client, model = make_client(A)
print(client.chat.completions.create(model=model, max_tokens=5,
    messages=[{'role':'user','content':'say ok'}]).choices[0].message.content)
"
```
Prints `ok` → the notebook is fully set up and ready. Errors with `day rate exceeded` →
setup is fine, you're just quota-capped; try again after it resets. Any other error →
check the discovered base_url/key path in `make_client`'s printout.

After A–D succeed, go to §4 (build a dataset) then §5 (run).

---

## 4. Building a dataset in the ProQuest dashboard

One ProQuest dataset = one window. Steps:

1. **Create New Dataset** → **Select Publication Titles**.
2. **Add multiple papers** — NYT alone yields only ~500 docs per window. Add WSJ,
   Washington Post, LA Times, Chicago Tribune, Boston Globe, USA Today, Christian
   Science Monitor to reach the low thousands. **Watch editions:** papers have separate
   *historical* vs *current* editions with different date ranges (e.g. "LA Times
   (1923–1995)" vs "LA Times (1996–)"); tick the one(s) covering the window's dates.
3. **Date range = the window's config dates** from `data/windows_economy.csv`. Every
   window is a fixed ~6–7 month band. **Do NOT widen it** — the placebo (calm) windows
   are also 7 months and comparability depends on equal widths.
4. **Query** (the economy forecast-catcher):
   ```
   (recession OR downturn OR depression OR recovery OR slump) NEAR/10 (predict* OR expect* OR forecast* OR outlook OR likely OR coming OR ahead OR fear*)
   ```
5. **Name = window id with underscores removed** — ProQuest strips underscores, and
   `run_all_economy.sh` derives the folder as `${window//_/}`. So `gfc_2008` → dataset
   `gfc2008` → folder `/home/ec2-user/SageMaker/data/gfc2008`.
6. **Build it** (~1 hr, ProQuest-side, independent of the VM terminal).
7. **Log provenance** in `data/proquest_datasets.csv`: window_id, source_papers,
   start/end date, query, doc_count. The READMEs *require* the source mix be disclosed.

The 9 post-1963 windows (`kind` in `windows_economy.csv`): `oil_1973 volcker_1980
crash_1987 gulf_1990 dotcom_2001 gfc_2008` (crises) and `calm_1965 calm_1995 calm_2005`
(placebos).

---

## 5. Running the pipeline

Verify first: `<sample-python> -c "import lxml, openai"` → `ok`. Confirm
`run_all_economy.sh`'s `PY=` line points at the sample env python (not bare `python`).

Then, from `election_arm`, **launched exactly once**:
```
PYTHONUNBUFFERED=1 nohup bash run_all_economy.sh > batch.log 2>&1 &
tail -f batch.log
```
- `nohup` → survives a closed browser (TDM keeps processes ~48h).
- `PYTHONUNBUFFERED=1` → the log updates live (see §6).
- Test a single window first with `extract_gpt.py --source proquest --window <w> --limit 10`.

Monitor with the **pred file line count**, not the log (see §6):
```
wc -l data/predictions/pred_proquest_economy_*.jsonl
```
Quality check anytime: `python sample_claims.py --n 10`.

---

## 6. Failure modes we hit (READ THIS — it's the whole point of the guide)

**Output buffering makes the log look frozen.** Under `nohup`, Python block-buffers
stdout, so `processed N` prints (and errors!) don't appear in `batch.log` until the
buffer flushes — the log looks stuck even while work happens. The *data file* flushes
per line, so **trust `wc -l` on the pred file, not log prints**. Always launch with
`PYTHONUNBUFFERED=1` so the log is truthful. (We debugged a "0 errors" for an hour that
was really just buffered errors.)

**The daily LLM quota.** Error `429 - "Application cost/day rate exceeded"`. It's a
per-day *cost* cap on the shared proxy. `extract_gpt.py` now detects it
(`RateLimitReached`), **stops cleanly** (exit 2), and does NOT mark the current article
done, so a rerun resumes exactly there. The batch runner stops the whole run on exit 2.
→ **Workflow is "run once a day until all 9 windows finish."** Check if the cap reset
without a big run by making one probe call (a 5-token "say ok"); if it 429s, wait.

**Failures silently recorded as `no_predictions` (fixed).** The old code wrote a failed
call as an empty result AND marked it done, permanently losing the article. Fixed:
non-quota failures are now left unmarked for retry; only genuinely-empty successful calls
write `no_predictions`.

**Never launch the batch twice.** Two concurrent runs fight over the rate-limited proxy
(halving throughput, constant backoffs) and double-process articles → duplicate claims.
Before launching, `ps -ef | grep -E "run_all_economy|extract_gpt" | grep -v grep` must be
empty. If duplicates happened, dedup by `page_id` before scoring.

**Azure content filter false positives.** Some OCR'd articles trip the proxy's
`jailbreak detected` filter (a 400). They yield no claims — a recall leak. Occasional is
fine; if frequent, note it as a caveat.

**`lxml` / wrong-python errors.** `ModuleNotFoundError: lxml` almost always means a script
ran under the default `python`. Use the full sample-env path everywhere; check
`run_all_economy.sh`'s `PY=` line.

**iCloud eviction on the Mac side.** This repo lives in an iCloud-synced folder. Symptoms:
ProQuest files "disappear" from disk, or the checkout silently switches to `main` (where
the ProQuest files don't exist — they're on branch `proquest-tdm-integration`). Fix:
`git checkout proquest-tdm-integration`. Also: save screenshots to `~/Downloads` (syncs
fast), not the iCloud repo folder.

**ProQuest strips underscores from dataset names** — folder is `gfc2008`, window id is
`gfc_2008`. The batch handles this; manual `tdm_parse.py` calls need the real folder in
`--dataset-dir` and the underscore id in `--window`.

**Coverage gaps from editions.** A window showing only a handful of papers usually means
you added *historical* editions that end before the window's dates — add the *current*
editions too.

---

## 7. Validation and scoring

**Kappa (in VM — needs `claim_text`):**
```
python validate_kappa.py sample --arm economy --source proquest   # two graders fill columns
python validate_kappa.py kappa  --arm economy                     # only the numbers leave
```
Do NOT export `validation_sample.csv` / `validation_disagreements.csv` — they contain text.

**Gold bake-off harmonization.** `main` added `JeremysShit/gold_extraction/` (16 gold
pages, `eval_extraction.py`) and validated gpt-4.1/gemini for the loc/nyt sources. Our
ProQuest arm is forced onto **gpt-4o-mini** (only proxy model; can't reach the OpenAI API
from the VM). To get a comparable quality number, run gpt-4o-mini over `gold_pages.jsonl`
(via the proxy, in-VM) and score with their `eval_extraction.py`. Disclose the model
difference — don't silently pool gpt-4o-mini labels with gpt-4.1 ones.

**Scoring (Mac, text-free):**
```
python analyze_economy.py     # NBER hit rates, Brier, crisis-vs-placebo, by voice/source
```
Needs `data/nber_recessions.csv` (present). Writes `data/scored_economy.csv`.

**Model (`model.py`) uses `claim_text` via TF-IDF** as its strongest feature, so run it in
the VM on un-stripped data for the real result; on the Mac export it's metadata-only.
For a *deployable* forecast-credibility model, **drop `window_kind`** (retrospective →
leaks the outcome) and read the held-out-window ROC-AUC to judge if skill generalizes.

---

## 8. Export → Mac → git

1. In VM: `run_all_economy.sh` bundles `data/predictions/proquest_exports.tar.gz`
   (label-only, well under the cap). Export it via ProQuest's `Export Instructions.ipynb`
   (`aws s3 cp` → emailed 2-hour download link).
2. Download to the Mac, unpack into `election_arm/data/predictions/`.
3. Commit on branch **`proquest-tdm-integration`** (the PR branch, not `main`). Push as
   the verified GitHub account (`bodeb-gif`).

---

## 9. Reference: leakage & metric rules (from CLAUDE.md / READMEs)

- **Split by window, never random** — claims in one episode share an outcome.
- **Accuracy is meaningless** (rare event) — report PR-AUC, ROC-AUC, Brier, hit rate.
- **Crisis vs placebo is the core control** — a signal must beat the calm-window baseline.
- **Label every result by `source` and era** — proquest (gpt-4o-mini full text) ≠
  nyt (gpt-4.1 headline+lead) ≠ loc (pre-1963 OCR). Never silently mix.
