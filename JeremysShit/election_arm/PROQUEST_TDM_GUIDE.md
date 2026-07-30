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
- **Leaves the VM:** only the label-only `pred_*.export.jsonl` (no `quote`).
- **On the Mac:** scoring and analysis run on those labels (they're text-free).

Anything that needs the article text — reading claims, kappa validation, the
text-feature model — **must run in the VM**. Only numbers/labels come out.

---

## 2. The scripts and what each does

| Script | Runs where | In → Out |
|---|---|---|
| `tdm_parse.py` | VM | ProQuest XML folder → `data/raw/proquest_{arm}_{window}.jsonl` |
| `extract_gpt.py` | VM | raw jsonl → `data/predictions/pred_proquest_economy_{window}.jsonl` (via in-VM GPT proxy) |
| `strip_for_export.py` | VM | pred jsonl → `pred_*.export.jsonl` (removes `quote`) |
| `adapt_proquest_claims.py` | Mac | `pred_*.export.jsonl` → `pred_proquestllm_economy_*.jsonl` (main's adapter: derives `predicted_state_at_horizon`, `hedged`) |
| `run_all_economy.sh` | VM | batches parse→extract→strip over all 9 windows, bundles `proquest_exports.tar.gz` |
| `run_corpus_economy.sh` | VM | the same, over the ~110 **year shards** of a 1900-2010 corpus (§4b) |
| `extraction_status.py` | VM or Mac | the pred/verified files on disk → per-window table: claims, no-prediction pages, verifier drops, pages still to do |
| `corpus_progress.py` | VM or Mac | the corpus rollup: per-decade counts, pages left, daily runs remaining (§4b) |
| `sample_claims.py` | VM | prints N random claims to eyeball extraction quality |
| `validate_kappa.py` | VM | draws a sample + computes Cohen's kappa (human validation) |
| `analyze_economy.py` | Mac | pred jsonl (text-free) → NBER scoring, crisis-vs-placebo table |
| `model.py` | VM (for text features) | `scored_economy.csv` → predicts `hit`, prints feature importances |

`extract_predictions.py` is the **non-ProQuest** extractor (loc/nyt sources, runs
outside the VM, now uses gpt-4.1 per the gold bake-off). `extract_gpt.py` is the
ProQuest-proxy twin — but it is no longer a copy of `extract_predictions.py`'s
`ECONOMY_PROMPT`. It now mirrors **main's `src/extract_llm.py`**; see below.

---

## 2a. Label schema v2 — what `extract_gpt.py` emits, and why it changed

`extract_gpt.py` emits **main's extraction schema**, not a private one, so
ProQuest claims are checked by the same machinery as the rest of the project:

| Consumer (on main) | Reads |
|---|---|
| `src/score_predictions.py` | `topic`, `direction`, `price_direction`, `unemployment_direction`, `scope`, `horizon_months`, `date` |
| `src/adapt_proquest_claims.py` | `scope`, `direction`, `confidence` → **derives** `predicted_state_at_horizon`, `hedged` |
| `validation/gold_extraction/eval_extraction.py` | `topic`, `direction`, `horizon_months`, `confidence`, `voice` |

Four things v1 did that `docs/SCORING.md` forbids, now fixed:

1. **No outcome leakage.** v1 put `Window: crash_1929` in the model's context —
   window ids name the outcome. Only newspaper and date are passed now. Window
   is still attached to the output *record* (the scorer needs it); it never
   reaches the prompt.
2. **The model no longer states the outcome.** v1 asked it for
   `predicted_state_at_horizon: recession | expansion`. It now reports only a
   direction; `adapt_proquest_claims.py` maps direction → recession/expansion
   by a fixed table. This is SCORING.md's one rule.
3. **No manufactured horizons.** v1's prompt said "use 6 if unstated", which
   inflates the RIGID stratum. The schema is `6 | 12 | "vague"`.
4. **Hallucination guard.** Quotes must be verbatim and are dropped unless their
   tokens really appear in the source text. This matters more here than on main:
   the in-VM proxy only offers gpt-4o-mini, the weakest extractor in the bake-off.

### Everything derived from the quote must be computed in-VM

The quote cannot leave the VM, so anything downstream computes *from* the quote
has to be computed here and exported as a number. Three such fields exist, and
each replaces a calculation main would otherwise do at scoring/modelling time:

| exported field | replaces, on main | if it were missing |
|---|---|---|
| `horizon_hint` | `resolve_horizon()` reading the quote's time language | every claim collapses to the neutral default; the RIGID stratum disappears |
| `quote_n_words` | `c_len` = `quote.split()` word count | 0 for every ProQuest row |
| `quote_has_number` | `c_has_number` = quote contains a digit | 0 for every ProQuest row |

The last two matter more than they look. They are not *missing* — they are
**wrong in a way perfectly correlated with the source**, so a model pooling LOC
and ProQuest claims can read `c_len == 0` as "this row is ProQuest" and learn
the data source instead of the forecast. `quote_n_words` is exported unclipped;
main's 80-word clip is a modelling choice and stays on main, applied identically
to both corpora.

### The modelling contract

`claim_features()` in main's `src/model_hit.py` and `src/hit_predictor.py` reads
exactly these columns. All are present post-export:

| model feature | source column | ProQuest post-export |
|---|---|---|
| `c_direction`, `c_topic`, `c_voice`, `c_scope` | `direction`, `topic`, `voice`, `scope` | ✅ |
| `c_confidence` / `c_hedged` | `confidence` | ✅ |
| `c_quoted` | `is_quoted_forecaster` | ✅ |
| `c_named` | `speaker_name` | ✅ |
| `c_has_number`, `c_len` | `quote` | ✅ **via `quote_has_number` / `quote_n_words`** |
| `c_horizon` | `horizon_used` (from the scorer) | ✅ **via `horizon_hint`** |

**main needs a three-file patch to read them** — it currently reaches for the
quote directly. `main_pooling.patch` in this folder does it (verified to apply
cleanly to `main`):
```
git checkout main && git apply JeremysShit/election_arm/main_pooling.patch
```
It teaches `score_predictions.resolve_horizon` to prefer `horizon_hint`, and both
`claim_features()` to prefer the precomputed numbers, each falling back to the
quote when it IS present — so LOC behaviour is unchanged. Without the patch
`model_hit.py` **crashes** on a ProQuest frame (`df.get("quote","")` returns a
scalar) and `hit_predictor.py` silently produces the 0/0 columns above.

**Columns ProQuest does not have:** `quote` (never exportable), and
`conditional_on` / `reasoning`. The latter two are on main's prompt but are not
read by any scorer or model, and both are free text that could not be exported
anyway; they are deliberately not in this prompt (main's own note flags their
recall impact as unverified, and this arm runs the weakest extractor). Pooling
into one DataFrame simply leaves them NaN for ProQuest rows.

**Migrating the windows already extracted — use `migrate_v2.py`.** Every window
extracted before this change is v1. Records are now stamped `schema_version`, and
`extract_gpt.py` **refuses to append v2 records to a v1 file** rather than
silently mixing two vocabularies into a file `analyze_economy.py` globs together.
`migrate_v2.py` does the whole migration in the VM:
```
python migrate_v2.py            # DRY RUN: inventory every file, show the plan
python migrate_v2.py --apply    # rename v1 files to *.v1.bak
python migrate_v2.py --restore  # undo
```
It refuses to run if the v2 scripts were not pasted in first (migrating against
the old extractor would burn a day of quota re-creating v1), skips files already
on v2, is safe to re-run, and prints the `run_all_economy.sh` line for the
windows that need redoing. It also counts the **fake `no_predictions`** rows the
pre-fix extractor wrote for FAILED calls — indistinguishable from real empties in
v1, and one more reason those windows are worth redoing.

**It renames; it never deletes — and that distinction matters.** The label-only
exports of all nine v1 windows (29,267 claims) are already committed on `main`
at `data/proquest/*.export.jsonl`, so the v1 LABELS are safe either way. But the
in-VM copies are the only ones that still contain `claim_text`, and article text
can never leave the VM. Human kappa validation, the TF-IDF text model and
`sample_claims.py` all need it; delete it and the only way back is to spend the
extraction quota again. Disk is free, the daily quota is not.

Only v2 lines are scorable by main. `--allow-mixed` exists but is not recommended.

**Measuring this model against the gold standard.** Now that the vocabulary
matches, gpt-4o-mini can be scored on main's gold pages — the disclosure gap
noted in the labelling-model comparison. 16 calls:
```
python extract_gpt.py --pages gold_pages.jsonl --out pred_gpt4omini_gold.jsonl
# then, on the Mac, against main's harness:
python validation/gold_extraction/eval_extraction.py \
    --pred pred_gpt4omini_gold.jsonl --name gpt-4o-mini
```

---

## 3. VM environment — the exact details

**Directories:**
- Scripts / working dir: `/home/ec2-user/SageMaker/election_arm/`
- Datasets land at: `/home/ec2-user/SageMaker/data/{dataset_name}/` (one XML per article)

**Python: don't guess the path — run `vm_doctor.py`.**
```
python vm_doctor.py        # plain python is fine; that is the point
```
It probes every interpreter and Jupyter kernel on the box, reports which have
`openai` / `lxml` / `pandas`, and prints the exact `export PY=...` line to use. It
also checks `gpt_sample.txt` and whether the discovered key file exists.

As of 2026-07-28 the answer on this workbench is:
```
/home/ec2-user/SageMaker/.conda/envs/sample-2025.12.578/bin/python3.12
```
**Note `python3.12`, not `python`.** The env name was never the problem — the
binary name was. Hardcoding `bin/python` is what produced a long run of
"No module named openai" against an interpreter that existed but was the wrong
one. The error is misleading: a wrong *path* says "no such file or directory",
so "no module named openai" makes you hunt for the package when the real fault
is the interpreter. `run_all_economy.sh` now honours an exported `$PY` and falls
back to the path above, so there is nothing to edit by hand.

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

`test_offline.py` is deliberately **not** in the bundle: it is the pre-flight gate and
runs on the **Mac**, before you ship anything. It covers the VM-side logic that has no
business failing in the VM — prompt hygiene (no window id reaches the model), the
hallucination guard, horizon inference, the schema-mixing refusal, and the export
tripwire — all without an API key or the `openai` SDK. Run it before every transfer:
```
python test_offline.py     # from JeremysShit/election_arm
```
If it fails, do not paste anything into the VM.

**B. Find the Python env name and confirm its packages.** The env name has a version
suffix that differs per workbench, so read it off `conda env list`:
```
conda env list
```
Pick the **`sample-*` env that is NOT `-r`** (the `-r` one is R, not Python) — e.g.
`sample-2025.12.578`. If `conda` isn't found, first run
`source /home/ec2-user/SageMaker/.conda/etc/profile.d/conda.sh`. Then test that the exact
interpreter the scripts call has both packages:
```
$PY -c "import lxml, openai; print('ok')"     # $PY from vm_doctor.py
```
Prints `ok` → good. Errors on `lxml` → install into *that* env (`conda`, not `pip` — no
internet), then re-test:
```
conda install -n <sample-env> lxml -y
```
Finally, point `run_all_economy.sh` at that interpreter:
```
# run_all_economy.sh now honours an exported $PY -- nothing to edit.
# Only needed if you want to bake a different default into the file:
sed -i "s|^PY=.*|PY=\"\${PY:-$PY}\"|" run_all_economy.sh
```

**C. Create `gpt_sample.txt` so `extract_gpt.py` can auto-discover the proxy.** It needs the
proxy's base_url, key-file path, and model — all present in ProQuest's sample notebook.
**Find that notebook by its contents** (the filename varies — searching by name often
"matched no files"):
```
grep -rl -i "openai-compatible\|agai-proxy\|base_url" /home/ec2-user/SageMaker --include="*.ipynb" 2>/dev/null
```
Convert whatever it prints to text (quote the path — it has spaces):
```
cd /home/ec2-user/SageMaker/election_arm
jupyter nbconvert --to script --stdout "<PATH_FROM_GREP>" > gpt_sample.txt
grep -E "base_url|open\(|model" gpt_sample.txt     # verify all three are present
```
*Fallback if no notebook is found* — you only need the three values. Confirm the key file
exists first (`find /home/ec2-user/SageMaker -iname "*token*" -o -iname "*agai*" 2>/dev/null`),
then hand-write it (adjust the key path to what `find` reported):
```
cat > gpt_sample.txt <<'EOF'
base_url = "https://agai-proxy.prod.int.tdmstudio.proquest.com/large-language-models-openai-compatible/"
gpt_api_key = open("/home/ec2-user/SageMaker/.token/.agaitoken").read()
model = "gpt_4o_mini"
EOF
```

**D. Smoke-test the proxy** (one throwaway call — proves the key/URL/model resolve and the
quota isn't already spent):
```
$PY -c "
from extract_gpt import make_client
class A: sample='gpt_sample.txt'; base_url=key_file=model=None
client, model = make_client(A)
print(client.chat.completions.create(model=model, max_tokens=5,
    messages=[{'role':'user','content':'say ok'}]).choices[0].message.content)
"
```
Prints `using proxy base_url=...` then `ok` → the notebook is fully set up. Errors with
`day rate exceeded` → setup is fine, you're just quota-capped; retry after reset. Any other
error → check the base_url/key path in `make_client`'s printout (the key path may differ on
this workbench).

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

## 4b. The CORPUS layout (one 1900-2010 dataset, ~146k articles)

The alternative to §4, and the current direction: **one** ProQuest dataset built from
a single query over **1900-2010**, instead of nine window-sized keyword datasets.

**Why.** With per-window datasets, the ProQuest query itself decides how many articles
each window contains — a crisis window matches more recession-words by construction — so
"crisis windows contain more forecasts" is partly a statement about the query, not the
press. A corpus spanning every year gives a real denominator (articles read per year, from
`corpus_progress.py`) and a continuous 1900-2010 series instead of 9 disjoint bands.

**What changes mechanically.** A 110-year dataset cannot be stamped with one window, so:

| per-window (§4) | corpus |
|---|---|
| `tdm_parse.py --window gfc_2008` | `tdm_parse.py --corpus` |
| window comes from the dataset | window derived per article **from its date** |
| `data/raw/proquest_economy_gfc_2008.jsonl` | `data/raw/proquest_economy_{YYYY}.jsonl` |
| `run_all_economy.sh` (9 windows) | `run_corpus_economy.sh` (~110 year shards) |
| `extraction_status.py` (9 rows) | `corpus_progress.py` (decade rollup + pages left) |

Most articles get a **null window** — the configured bands cover ~12 of 110 years. That is
expected, not a parse failure; those claims are scored continuously against NBER, and
`analyze_economy.py` now prints how many claims each table actually covers so the
crisis-vs-placebo row can't be mistaken for the whole corpus.

**Build it:** same as §4 steps 1-2 and 4-6, but the date range is the full span and the
name has no window in it (e.g. `econ19002010`). Watch editions harder than usual — over
110 years nearly every paper has multiple historical/current editions and you need all of
them, or whole decades come back thin.

**Budget honestly, before you start.** ~146k articles is ~30x the keyword windows.
Extraction is ~1.2-1.5 LLM calls per article (`extract_gpt.py` chunks at 8000 chars) and
verification adds one call per surviving claim. Against a daily cost cap that stopped an
earlier run at ~4.3k articles, **this is a weeks-long run of once-a-day invocations**, and
it consumes the shared proxy's daily budget for the duration. Every stage resumes per
shard, so the operating procedure is genuinely just "run the same command daily":

```
python tdm_parse.py --arm economy --corpus --dataset-dir <folder> --inspect   # tags first
python tdm_parse.py --arm economy --corpus --dataset-dir <folder>             # ~25 min, once
PYTHONUNBUFFERED=1 nohup bash run_corpus_economy.sh > batch.log 2>&1 &        # daily
python corpus_progress.py --rate 3000                                         # where am I
```

`run_corpus_economy.sh` runs **window years first** by default, so the crisis-vs-placebo
result lands in days rather than at the very end; `--chrono` opts out. It also refuses to
start if another run is live, and **does not** build an export tarball unless you pass
`--export` — the 15 MB allowance is a rolling 7-day budget and a weeks-long run would
otherwise spend it every day on partial data.

**Do not let both layouts into `data/predictions/` at once.** `analyze_economy.py` globs
`pred_*_economy_*.jsonl` with no window filter, so an article present in both the keyword
window dataset and the corpus is counted twice — and the old window files are still schema
v1, which the scorer cannot read at all. `tdm_parse.py --corpus` warns when it sees them;
move them aside before scoring:

```
mkdir -p data/predictions/keyword_v1
mv data/predictions/pred_proquest_economy_*_*.jsonl data/predictions/keyword_v1/
```

(The `*_*` glob matches window ids like `gulf_1990` and never a bare year.) This also
settles the v1 backfile question from earlier: under the corpus the ~4.3k v1 records are
superseded rather than migrated.

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

Monitor with the **files on disk**, not the log (see §6):
```
python extraction_status.py
```
Per window: claims, pages that came back with no prediction, claims the verifier
dropped, and pages still to do — so a quota stop shows up as a `left` column that
stops shrinking. (`wc -l data/predictions/pred_proquest_economy_*.jsonl` is the
crude version: it counts empties and v1 leftovers as if they were claims.)

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

**`ModuleNotFoundError: openai` / `lxml` — the wrong-interpreter trap.** Almost always a
script ran under the default `python`, not the sample env. The error is actively
misleading: a wrong *path* would say "no such file or directory", so "no module named
openai" sends you hunting for a missing package when the interpreter is the fault.
On this image the working binary is **`python3.12`, not `python`** — that one character
cost a long debugging detour. Don't guess:
```
python vm_doctor.py        # prints the exact `export PY=...` to use
```
`run_all_economy.sh` honours an exported `$PY`, so there is nothing to edit by hand.

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
