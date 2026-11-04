# Multilingual Pedagogy Benchmark: how this branch works

This branch (`multilingual_base`) runs and maintains the
[Fab AI Multilingual Pedagogy Benchmark](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/benchmarks?benchmarkType=multilingual_pedagogy&benchmark=EN&modelMode=default&price=10&dataset=&models=%5B%5D):
the pedagogy benchmark translated into other languages, run on many LLMs, and published on the
fab-ai.org leaderboard. **The leaderboard currently shows the CDPK benchmark only.** The pipeline
also supports SEND (translation, preparation, runs), so SEND could be added to the leaderboard
later, once its few-shot issue is fixed (see step 3).

This page explains the full process, from adding a new language to publishing its scores, with
the commands to run. All scripts below are run **from the repository root, on the command line**


---

## Overview

| Step | What | Script | Main output |
|---|---|---|---|
| 0 | One-time setup | `uv sync`, `dvc pull`, `.env` | |
| 1 | Register the language | edit `src/cdpk/language_prompts.py` | |
| 2 | Translate the English benchmark | `translate_benchmark.py` | `data/pedagogy_benchmark_full_datasets/pedagogy_benchmark_<lang>_<dataset>_cleaned.csv` |
| 3 | Split it per category + write the question configs | `prepare_cdpk_dataset_multilingual.py` | `data/<Slug>/CDPK_per_category/`, `configs/questions/CDPK_<Slug>_*.yaml` |
| 4 | Run the models | `run_pedagogy_benchmark_multilingual.py` | `data/cache_local/`, `data/results/<Slug>/` |
| 5 | Check the runs *(optional, recommended before publishing)* | `count_runs.py`, `count_failures.py` | tables in the terminal |
| 6 | Build the results tables | `create_results.py` | `data/results/cdpk_multilingual_model_performance*.csv` |
| 7 | Publish to the leaderboard | `format_data_web.py` (**WIP**) | `data/web/*.json`, Azure upload |
| 8 | Save and share | `merge_cache.py`, `dvc push`, `git push` | |

`<lang>` is the base language name used for translation (e.g. `swahili`). `<Slug>` is the folder
name given by `language_prompts.py` (e.g. `Swahili_ep`).

---

## Step 0: Setup (once)

```bash
uv sync                      # install dependencies (Python >= 3.13), versions pinned in uv.lock
git submodule update --init  # fab-benchmarks-configs: models.csv, providers.csv, custom_models.yaml
uv run dvc pull              # translated datasets, splits and cached model responses

# Only if you need a change pushed to fabdata-llm's dev branch after the commit pinned in uv.lock
# (e.g. a new model fails to be called); test it, then commit pyproject.toml and uv.lock together
uv sync --upgrade-package fabdata-llm
```

Copy [`.env.example`](../.env.example) to `.env` at the repository root and fill in the API keys of
the providers you will call (e.g. `GEMINI_API_KEY` for translation with Gemini, plus the keys of the
benchmarked models, and `AZURE_STORAGE_KEY` only to upload to the leaderboard).
Models are defined in `fab-benchmarks-configs/custom_models.yaml` and their metadata (provider,
prices, size, display name) in `fab-benchmarks-configs/models.csv`.

> **Note: the `fab-benchmarks-configs` submodule is still required.** The runner (step 4) registers
> the models from its `custom_models.yaml` and reads its `models.csv`; the results and leaderboard
> scripts (steps 6-7) also read `models.csv` and `providers.csv`.
>
> **Work in progress:** these 3 files (`custom_models.yaml`, `models.csv`, `providers.csv`) are now
> also stored on GCS. In the future, the scripts may download them from GCS instead, and the
> submodule would then no longer be used. Until that change is made, keep using the submodule and
> **update it to its latest version before running models**, so that newly added models are
> registered:
>
> ```bash
> # Update the submodule to the latest commit of fab-benchmarks-configs (new models, providers, prices)
> git submodule update --remote --merge
>
> # Then record the new submodule commit in this repo, so others run with the same model list
> git add fab-benchmarks-configs
> git commit -m "Update fab-benchmarks-configs"
> ```
>
> `git submodule update --init` alone (as above) only checks out the commit recorded in this repo,
> which may not include the latest models.

### What is tracked by DVC, and how to get it

Large data files are not stored in git but with [DVC](https://dvc.org/), on an Azure blob remote
(`azure://llm-benchmarks/dvc`, see `.dvc/config`). Git only stores small `.dvc` pointer files that
record which version of the data goes with each commit.

| What | Path | Pointer file(s) in git |
|---|---|---|
| Translated full datasets (English source, raw, `_cleaned`, `_retranslated`, `_reviewed` files) | `data/pedagogy_benchmark_full_datasets/` | `data/pedagogy_benchmark_full_datasets.dvc` (whole folder) |
| Shared cache of model responses | `data/cache/` | `data/cache.dvc` (whole folder) |
| Per-category dev/test splits of each language (step 3) | `data/<Slug>/CDPK_per_category/{dev,test}/*.csv` | one `.csv.dvc` per CSV, next to it |

Not tracked by DVC:
- **in git**: code, `configs/` (models lists and question configs), `data/few_shot_examples_idx_dict.json`,
  `data/translation_log.json`;
- **local only** (neither git nor DVC): `data/cache_local/` (new model responses, copied to
  `data/cache/` by `merge_cache.py`, step 8) and `data/results/` (rebuilt from the caches by steps 4
  and 6).

Splits only exist on DVC once someone has run `dvc add` on them (step 8).

To get the data:

```bash
uv run dvc pull                                         # everything tracked, at the version of the current commit
uv run dvc pull data/pedagogy_benchmark_full_datasets.dvc   # only the translated datasets
uv run dvc pull data/cache.dvc                          # only the shared cache of model responses
uv run dvc status                                       # local files that differ from the commit's version
```

Reading needs no credentials (the remote allows anonymous read); `dvc push` needs write access.
After `git pull` or `git checkout` of another commit, run `uv run dvc pull` again: the pointer files
may have changed. How to add new data to DVC is in step 8.

---

## Step 1: Register the language

Add the language to `LANGUAGE_PROMPTS` in `src/cdpk/language_prompts.py`. Two variants exist:

- **`<lang>_ep` (English Prompt, the one used on the leaderboard)**: instructions to the model are
  in English, few-shot examples and questions are in the target language. Copy an existing `_ep`
  entry (e.g. `swahili_ep`) and change `intro_ep`, `display_name` and `slug` (e.g. `Swahili_ep`).
- **`<lang>`**: instructions are also translated (`intro`, `instruction`, `final` in the target
  language). Optional; some languages only have the `_ep` variant.

Each variant writes to its own folders (`Swahili_ep/` vs `Swahili/`), so it is prepared and run
separately. The `slug` names every output folder and file of the language.

---

## Step 2: Translate the benchmark (`translate_benchmark.py`)

> **Human-reviewed translations: skip step 2.** If the translation comes from (or was reviewed by)
> human translators, do not run `translate_benchmark.py`. Put the file in
> `data/pedagogy_benchmark_full_datasets/` **in the format described in step 3**
> ([Input format for human-reviewed translations](#input-format-for-human-reviewed-translations)),
> then go to step 3.

Translates the English questions and answer options from Hugging Face
(`AI-for-Education/pedagogy-benchmark`, config `cdpk_main` for CDPK, `cdpk_send` for SEND) with an LLM.
Always use the **base language name** (e.g., `swahili`, never `swahili_ep`).

```bash
# 1. Estimate the cost (no API call)
uv run python scripts/translate_benchmark.py --mode estimate_cost --dataset cdpk --language swahili --model gemini-2.5-flash-preview-09-2025

# 2. Translate (optimized = one API call per question, ~5x cheaper than the default per-cell mode). Recommanded.
uv run python scripts/translate_benchmark.py --mode optimized --dataset cdpk --language swahili --model gemini-2.5-flash-preview-09-2025 --retranslate true

# 3. Only if step 2 reported missing cells (or ran with --retranslate false): fill them,
#    with the same --model as step 2 (the printed command already includes it)
uv run python scripts/translate_benchmark.py --mode retranslate --dataset cdpk --language swahili --model gemini-2.5-flash-preview-09-2025 --file pedagogy_benchmark_swahili_cdpk.csv
```

Missing cells are always filled with the same model and language as the rest of the file: if
`--model`, `--language` or `--translator-slug` differ from the file's entry in
`data/translation_log.json`, retranslate stops before any API call and prints the right command.
Without `--model`, the default model (`DEFAULT_MODEL` in the script) is used, so pass `--model`
whenever the file was translated with another model.

Files written in `data/pedagogy_benchmark_full_datasets/`:

| File | Meaning |
|---|---|
| `pedagogy_benchmark_swahili_cdpk.csv` | raw translation, may have missing cells |
| `..._cdpk_cleaned.csv` | **verified: every cell translated. The file to use in step 3.** Written as soon as a translation is complete, even when nothing had to be retranslated |
| `..._cdpk_retranslated.csv` | some cells failed again: rerun `--mode retranslate --file <this file>` |
| `..._cdpk_partial.csv` | snapshot saved during a run |

The script refuses to translate a language that already has files.

**Translation log (`data/translation_log.json`, tracked in git).** One entry per translation, keyed
by the raw file name without `.csv`:

| Field | Content |
|---|---|
| `language`, `translator_slug`, `dataset` | the `--language`, `--translator-slug` and `--dataset` used |
| `source` | English source: Hugging Face repo, config, split and `revision` (commit hash) |
| `translation` | `model`, `mode` (`optimized` / `translate`) and `date` |
| `retranslations` | one record per `--mode retranslate` run: cells missing before/after, input and output files (`model: null` = nothing missing, no API call) |
| `synthetic`, `note` | `true` = entry backfilled by hand (translation made outside the script), `note` says what was assumed |

How it is used:
- **Written** by `translate_benchmark.py` (main fields on translation, a `retranslations` record on
  each verify/retranslate run).
- **Checked** by `--mode retranslate` before any API call: if `--language`, `--translator-slug` or
  `--model` differ from the log, it stops and prints the right command; it reloads the English
  source at the logged `revision`, so cells are filled from the same text.
- **Read** by `translation_experiment_analysis.py` to label each translation with its model.
- **To look up**: which languages/datasets are translated, with which model and when, the exact CLI
  arguments to use for a file (step 2 `--language`/`--translator-slug`/`--model`; steps 3-4 use the
  `language_prompts.py` key `<language>[_<translator_slug>]_ep`), and how many cells failed.
- **For a translation made outside the script** (colleague, human translators): add an entry by hand
  with `"synthetic": true` and a `note`, so retranslation enforces the right model and language.

```bash
# List every translation with its model and date
uv run python -c "import json; [print(k, v['translation']['model'], v['translation']['date'], 'synthetic' if v.get('synthetic') else '') for k, v in json.load(open('data/translation_log.json', encoding='utf-8')).items()]"
```

**Translating a language again with another model** (e.g. to measure how much the translation
model changes the scores): add `--translator-slug <short model name>` (lowercase letters and
digits). The prompt still uses `--language`; only the file names change, so the original
translation is not touched:

```bash
uv run python scripts/translate_benchmark.py --mode optimized --dataset cdpk --language arabic --translator-slug argon4low --model gemini-4-argon-low --retranslate true
# -> pedagogy_benchmark_arabic_argon4low_cdpk.csv (and _cleaned)
```

Pass the same `--translator-slug` in retranslate mode. For steps 3 and 4, add a language entry
`arabic_argon4low_ep` (slug `Arabic_argon4low_ep`, with `'translator_slug': 'argon4low'`) in
`language_prompts.py`, as done for Arabic, Hausa and Luganda. Its results and cache then go to
their own folders, separate from `Arabic_ep`.

Once the models are run on both versions (step 4), compare the scores with
`translation_experiment_analysis.py`. It reads the answers from `data/cache_local`. It
compares the original translation with the new one on the same questions, and the full test
set (899) with the 300-question core subset (`data/final_300_original_question_indices.csv`).
It writes CSV tables and figures to `data/results/figures/translation_experiment/`:

```bash
uv run python scripts/translation_experiment_analysis.py --languages Arabic Hausa Luganda --translator-slugs argon4low
```

---

## Step 3: Prepare the splits and configs (`prepare_cdpk_dataset_multilingual.py`)

Splits the translated file per category into two sets, and writes one question config per category:

- **dev set** (3 questions per category): the few-shot examples. They are shown to the model, with
  their correct answers, at the start of every prompt, to show it the expected answer format
  (a single letter). They are **not scored**. They are the same 3 questions in every language
  (listed in `data/few_shot_examples_idx_dict.json`).
- **test set** (all the other questions of the category, 899 in total for CDPK): the questions the
  model is scored on.

Why: the runner (step 4) does not read the full translated file. It benchmarks one category at a
time, from a question config that tells it which test file to score and which dev file to take the
few-shot examples from.

```bash
# Check everything first (writes nothing)
uv run python scripts/prepare_cdpk_dataset_multilingual.py --dataset cdpk --file pedagogy_benchmark_swahili_cdpk_cleaned --language swahili_ep --dry-run

# Then write the files
uv run python scripts/prepare_cdpk_dataset_multilingual.py --dataset cdpk --file pedagogy_benchmark_swahili_cdpk_cleaned --language swahili_ep
```

It checks the file (columns, question ids in order, categories, no missing text, answer key
identical to English) and each split (few-shot examples, sizes and answers identical to the English
split, column layout expected by the runner), prints a table per step, and stops on the first
problem. It never overwrites existing outputs. Use `--use-question-id true` only for human-reviewed
files (`*_reviewed.csv`) where rows were removed. By default the few-shot examples are found by row
position (they are rows 0, 1, 2, 186, ... of the full file), which only works if no row was removed;
with `--use-question-id true` they are found by their `question_id` instead, and the answer key is
compared with English question by question rather than row by row.

Outputs: `data/Swahili_ep/CDPK_per_category/{test,dev}/*.csv` and
`configs/questions/CDPK_Swahili_ep_<category>.yaml`.

### Input format for human-reviewed translations

A translation that did not come from step 2 must be put in this format before running step 3,
otherwise the checks stop the script:

- **Location and name**: `data/pedagogy_benchmark_full_datasets/pedagogy_benchmark_<lang>_cdpk_reviewed.csv`
  (or `..._send_reviewed.csv`), with `<lang>` the base language name (e.g. `luganda`, no `_ep`). Any
  other name is refused (the accepted suffixes are `_cleaned`, `_retranslated` and `_reviewed`).
- **CSV, UTF-8, one row per question**, with these columns **in this order** (the runner reads the
  question, options and answer by column position):

  ```
  question_id,question,answer_a,answer_b,answer_c,answer_d,answer_e,answer_f,answer_g,correct_answer,category
  ```

  Extra columns (e.g. `pedagogical_subdomain`, `age_group`, `year`, `secondary_category`) are
  allowed **after** `category`. `answer_e/f/g` may be left empty or omitted (they are added empty).
  Same layout as the English file `pedagogy_benchmark_cdpk.csv`.
- **`question_id`**: the id of the English question (CDPK: 0-919; SEND: from 920), unique.
- **`question`, `answer_a` to `answer_d`**: translated, never empty.
- **`correct_answer`**: the English letter (`A`-`D`), unchanged: it is checked against English
  question by question.
- **`category`**: the English category name, unchanged: `Science`, `Literacy`, `Creative arts`,
  `Maths`, `Social studies`, `Technology`, `General` (CDPK) or `SEND`.
- **Keep the few-shot questions**: question_id 0-2 (Science), 186-188 (Literacy), 340-342
  (Creative arts), 490-492 (Maths), 619-621 (Social studies), 732-734 (Technology), 841-843
  (General), from `data/few_shot_examples_idx_dict.json`. Every language uses the same examples.
- **Rows may be removed** (questions rejected by the reviewers). In that case, run step 3 with
  `--use-question-id true`; with all the 920 rows in the original order, the default works.

```bash
uv run python scripts/prepare_cdpk_dataset_multilingual.py --dataset cdpk --file pedagogy_benchmark_luganda_cdpk_reviewed --language luganda_ep --use-question-id true --dry-run
```

A reviewed file with removed rows has fewer test questions than English, so its scores are not
directly comparable with the full-test-set languages (step 6 skips folders with a different
question set than English).

> **SEND: known issue.** The SEND dataset contains 2 duplicated questions: question_id 1047 is an
> exact copy of the few-shot example 920, and 1056 of the few-shot example 922. The duplicates were
> caught when the few-shot list was created, so `data/few_shot_examples_idx_dict.json` lists 5 SEND
> examples (`[0, 127, 1, 2, 136]`) instead of 3, and `--dataset send` is refused for now. To fix it:
> remove the 2 duplicates (1047 and 1056) from the SEND datasets, so the same question is never both
> a few-shot example and a test question, and set `"CDPK_send"` to `[0, 1, 2]`. Then prepare English
> SEND first (`--dataset send --language english --file pedagogy_benchmark_send`): it is the
> reference the other languages are checked against.

---

## Step 4: Run the models (`run_pedagogy_benchmark_multilingual.py`)

```bash
uv run python scripts/run_pedagogy_benchmark_multilingual.py --language swahili_ep --benchmark cdpk --models-config full_list_default_models_20260218
```

- `--models-config` is a file name in `configs/models/` (model id: display name). The leaderboard
  uses `full_list_default_models_20260218`. To test a few models, create a small one (e.g.
  `configs/models/my_test.yaml`).
- `--categories science maths` runs only some categories.
- Each model answer is cached in `data/cache_local/CDPK_<Slug>_<category>/resps_<model>.csv`.
  The runner reuses a cached file if it exists in `data/cache_local/` or `data/cache/`, so a
  rerun only calls the models that have no results yet. Several languages can run in parallel in
  separate terminals.
- Why two caches: `data/cache/` was first meant to be a **public, read-only** bucket (DVC remote) of
  our model responses, that anyone can `dvc pull` but not change. New runs are therefore written to
  `data/cache_local/` (not tracked by git or DVC), so local changes never conflict with that shared
  cache; when both have a file for the same model, the local one wins. New responses are copied to
  `data/cache/` afterwards with `merge_cache.py` (step 8).
- Results: `data/results/<Slug>/cdpk_results_{accuracy,bad_format,full}_<models-config>.csv`, and the
  terminal prints the ranking by overall accuracy.

---

## How the scores are computed: accuracy and bad format

For every question, the model's raw answer is turned into a letter (A to G) by `clean_resps` in
`src/cdpk/benchmark_answers.py`, using the patterns `REPAT` in `src/cdpk/language_prompts.py`
(the same for all languages): an answer starting with a letter, or ending with a letter on its own
line (with variants for reasoning models).

- **Bad format**: no letter could be extracted. This happens when the model answers in another way
  (explanation only, several letters, unexpected format), or when the API call failed
  (`success == False` in the cached file, which leaves an empty answer).
- **Accuracy** = questions where the extracted letter equals the correct answer ÷ **all** questions.
  A bad-format answer has no letter, so it **counts as wrong and stays in the total**: it lowers
  accuracy exactly like a wrong answer.
- **Bad format %** = bad-format questions ÷ all questions.
- **Overall** is computed on all the test questions of the benchmark pooled together (899 for CDPK),
  **not** as the average of the 7 category scores: large categories (Science, 183 questions) weigh
  more than small ones (General, 76).


---

## Step 5 (optional, recommended before publishing): Check the runs (`count_runs.py`, `count_failures.py`)

Use it before publishing, or when step 6 warns about a high bad format.

```bash
# Number of cached model files per language and category (add the new language to the folder list
# at the top of the script first)
uv run python scripts/count_runs.py --cache-dir data/cache_local

# Model calls that failed (success == False) for the new language
uv run python scripts/count_failures.py CDPK_Swahili_ep
```

A failed call is kept in the cached file and is **not retried automatically**. If a model has many
errors (bad format) in some categories:

1. **Inspect why.** Open its `data/cache_local/CDPK_<Slug>_<category>/resps_<model>.csv` and look
   at the rows without a readable answer:
   - empty answer with `success == False` (listed by `count_failures.py`): the API call failed
     (timeout, rate limit, provider error), a **technical** problem;
   - an answer that is there but has no clear letter (explanation only, several letters, answer in
     another format): this is **how the model behaves** in that language, a real result.
2. **If a rerun is fair** (the errors are technical failures, not the model's own answers), rerun the
   **whole category** for that model. Caveat: a rerun replaces **all** the model's answers in that
   category, not only the failed ones, so its score there can change for other reasons too (answers
   vary between calls). Never rerun only because a score
   looks low: rerunning some models and not others until they score better would bias the
   comparison. To rerun:
   1. Delete its `resps_<model>.csv` in `data/cache_local/CDPK_<Slug>_<category>/` for each category
      to rerun. If `merge_cache.py` already copied it to `data/cache/CDPK_<Slug>_<category>/`, delete
      that copy too: the runner uses any cached file it finds.
   2. Rerun those categories, with a models config listing only that model (e.g.
      `configs/models/rerun.yaml` containing `glm-5.1: GLM-5.1`):
      ```bash
      uv run python scripts/run_pedagogy_benchmark_multilingual.py --language swahili_ep --benchmark cdpk --models-config rerun --categories science maths
      ```
      Category names: `science literacy creative_arts maths social_studies technology general`.
   3. Run step 4 again **without `--categories`**. A run with `--categories` rewrites the results
      files of `data/results/<Slug>/` with those categories only; the full run rebuilds them with all
      categories (only the deleted files are called again, everything else comes from the cache).

> **Possible improvement: retry only the failed rows.** A retry-errors option could resend only the
> questions whose call failed (`success == False`) and keep all the other answers, instead of
> rerunning the whole category. This would remove the caveat above (successful answers would not
> change) and cost fewer API calls. It is not implemented yet.

---

## Step 6: Build the results tables (`create_results.py`)

```bash
uv run python scripts/create_results.py                     # replaces the existing output files
uv run python scripts/create_results.py --overwrite false   # stop if an output file already exists
uv run python scripts/create_results.py --models-list-yaml full_list_default_models_20260218
```

Replacing the output files is safe here: they are only computed from the results in
`data/results/<Slug>/` (step 4), which this script never changes, so running it again rebuilds them
identically from the same results. If a check fails, nothing is written and the previous files are
kept.

Reads the results of one models list (`MODELS_LIST_YAML` at the top of the script, or
`--models-list-yaml`) in every `data/results/<Slug>/` folder and writes
`data/results/cdpk_multilingual_model_performance.csv` (one row per model, language, prompt and
category) and `..._detailed.csv` (per-question lists). It checks that the result files are complete
and consistent, prints which folders were used or skipped, and skips folders without results for
that list or with a different question set than English.

**Bad format warning.** It prints the models above 5% bad format overall for a language
(`BAD_FORMAT_WARNING_THRESHOLD` at the top of the script), with the categories above 5%. They are
**kept** in the tables: unreadable answers count as wrong, which lowers the model's accuracy (see
[How the scores are computed](#how-the-scores-are-computed-accuracy-and-bad-format)). Decide per case:
- a small model that really answers badly in that language: keep it, it is a real result;
- failed API calls (check with `count_failures.py`, step 5): rerun the affected categories (step 5),
  then run step 6 again;
- otherwise, add the model to `MODELS_TO_EXCLUDE` at the top of the script to leave it out.

---

## Step 7: Publish to the leaderboard (`format_data_web.py`)

> ⚠️ **Work in progress.** This script uploads the JSON files directly to the Azure blob container.
> It must be aligned with the new **Ingest API** of the fab-ai.org leaderboard before it is used to
> publish; until then, check with the web team before uploading.

```bash
uv run python scripts/format_data_web.py                 # generate data/web/fabai_dev_*.json locally
uv run python scripts/format_data_web.py --upload dev    # ...and upload to Azure dev (then beta, prod)
```

Reads `data/results/cdpk_multilingual_model_performance.csv` (step 6) and
`fab-benchmarks-configs/{models,providers}.csv`, and writes the leaderboard files (config, provider
metadata, scores, cost and size frontiers). **A new language must be added to `LANG_IDS`,
`LANG_DISPLAY` and `LANG_FROM_CSV` at the top of the script**, otherwise its scores are ignored.
Models missing from `models.csv` are dropped (listed as a warning). Uploading requires
`AZURE_STORAGE_KEY`, and the current file is backed up before each upload.

---

## Step 8: Save and share

```bash
# Copy the new model responses from data/cache_local/ to the shared data/cache/ (only adds files,
# never overwrites)
uv run python scripts/merge_cache.py

# Record the new versions of everything tracked with DVC (this updates the .dvc pointer files):
# - always, after new model runs: the shared cache of model responses (whole folder, after
#   merge_cache.py)
uv run dvc add data/cache

# - only if a new language was added (steps 2 and 3):
#   the translated datasets (whole folder: raw, _cleaned, _retranslated files)
uv run dvc add data/pedagogy_benchmark_full_datasets
#   the per-category splits of the new language (one pointer file per CSV; --glob lets DVC expand
#   the wildcard, which PowerShell does not do)
uv run dvc add --glob "data/<Slug>/CDPK_per_category/*/*.csv"    # e.g. data/Swahili_ep/...

# Upload all of it to the Azure DVC remote (needs write access)
uv run dvc push

# Commit the code, configs/questions/*.yaml, the .dvc pointer files and data/translation_log.json
git add .
git commit -m "Add Swahili to the multilingual pedagogy benchmark"
git push
```

`data/results/` and `data/cache_local/` are not tracked by git.

---

## Other scripts

These are not part of the step-by-step process above: nothing in it depends on them.

| Script | Status |
|---|---|
| `create_figures.py` | **visualizations only**: figures from the step 6 tables, run in an interactive window (VS Code / Jupyter). Also saves `cdpk_multilingual_model_performance_detailed_exploded.csv` in `data/results/`; rerun it after step 6 to keep that file up to date |
| `create_multilingual_comparison_plots.py`, `pedagogy_vs_language_confound.py` | analysis and plots from the step 6 tables |
| `create_barplots.py` | bar plots of model accuracy per language, from the step 4 results in `data/results/<Slug>/` (`--overview` for one figure with all languages) |
| `translation_experiment_analysis.py` | compares two translations of the same language (see step 2, "Translating a language again with another model") |
| `estimate_cost.py` | estimates the API cost of running a models list on a dataset, before step 4 |
| `prepare_review_dataset.py` | builds an Excel file (English + translation side by side) for human reviewers of a translation |
| `analysis_human_vs_machine_translations.py`, `load_dataset_multingual_human_reviews_exp.py` | analysis of human-reviewed translations (Luganda) |
| `load_dataset.py`, `baseline_allmodels_general.py` | exploratory scratch scripts (interactive window), not needed for the pipeline |
| `run_pedagogy_benchmark.py` | **legacy** English-only runner, use `run_pedagogy_benchmark_multilingual.py` |
| `legacy/` | code used to build the original benchmark |
