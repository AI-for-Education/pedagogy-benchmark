# Add `--rerun` flag to multilingual pedagogy benchmark runner

## Context

Cache files at `data/cache_local/CDPK_{language}_{category}/resps_{model}.csv`
record every model response with a `success` boolean. `success==False` means the
API call raised an exception (network error, rate limit, provider outage), not
that the model answered incorrectly. The corresponding `resps` cell is an empty
string and is excluded from accuracy stats, but the row is permanently baked
into the cache — re-running the benchmark today is all-or-nothing per model, so
a single API hiccup that affected one question forces a full re-run to recover
that data point.

We want a `--rerun` flag on
[scripts/run_pedagogy_benchmark_multilingual.py](scripts/run_pedagogy_benchmark_multilingual.py)
that surgically re-issues API calls for `success==False` rows in existing cache
files, splices the new answers back into place, and leaves successful rows
untouched. Per the user, scope is **"patch failures only"**: only models with an
existing cache containing at least one failed row are touched; models with no
cache and models whose cache is fully successful are skipped entirely.

We also need **provenance tracking** in the cache CSV so it is later possible
to tell whether any given answer came from the original run or from a rerun
(and which rerun).

## Key code references found during exploration

| What | Where |
|---|---|
| Runner entry point | [scripts/run_pedagogy_benchmark_multilingual.py:73-161](scripts/run_pedagogy_benchmark_multilingual.py#L73-L161) — loops over categories, calls `run_benchmark` |
| `run_benchmark` (library) | [src/cdpk/benchmark_run.py:50-134](src/cdpk/benchmark_run.py#L50-L134) — loads YAML, iterates models, reads cache or calls `runner()` |
| Cache read (all-or-nothing) | [src/cdpk/benchmark_run.py:88-101](src/cdpk/benchmark_run.py#L88-L101) — no per-row skip logic exists |
| Cache write | [src/cdpk/benchmark_run.py:65-67](src/cdpk/benchmark_run.py#L65-L67) — `df_res.to_csv(outfile, index=False)` |
| `evaluate_model` | [src/cdpk/benchmark_answers.py:88-160](src/cdpk/benchmark_answers.py#L88-L160) — iterates `for rowi in range(len(example_rows), total_row)`, sets `success=False` on exception ([:156-159](src/cdpk/benchmark_answers.py#L156-L159)) |
| Cache path | `CACHE_LOCAL_DIR / questions_config / f"resps_{model}.csv"` |
| CSV columns | `answers, resps, success, Latency, TokensUsed, TokensUsedCompletion, TokensUsedReasoning` |

## Row alignment (critical to get right)

The cache CSV has **no question_id column** — rows are identified by position only.
Few-shot example rows are NOT written to the cache. The mapping is:

```
cache_csv_row[i]  ↔  test_df_row[len(example_rows) + i]
```

Any rerun implementation MUST respect this so spliced answers land on the right
questions.

## Provenance tracking (applies to both approaches)

Add two columns to every cache CSV:

- **`attempt`** — `int`, starts at `1` for original rows; bumped to `2, 3, …`
  whenever that row is re-issued by `--rerun`.
- **`rerun_at`** — ISO-8601 timestamp (string), `""` for original rows, set to
  `datetime.utcnow().isoformat()` whenever the row is patched.

Why two: `attempt` is the cheap numeric filter (`df[df.attempt > 1]`) and
`rerun_at` answers "when was this patched" without parsing the file's mtime.

Backwards compatibility: when reading an old CSV that lacks these columns, fill
`attempt=1` and `rerun_at=""` before any rerun logic runs, and write them back
on save. This way every CSV touched going forward carries provenance, and
untouched CSVs are migrated transparently the first time `--rerun` opens them.

A row that originated from a `success==False` original call and was later
patched by a successful rerun will show `success=True, attempt=2, rerun_at=<ts>`
— that combination is the unambiguous "answered via rerun" marker.

## Approach A — `rerun_failed` parameter inside `run_benchmark`

Modify `run_benchmark` in [src/cdpk/benchmark_run.py](src/cdpk/benchmark_run.py)
to accept `rerun_failed: bool = False`, and have the multilingual script
forward `opt.rerun` to it.

Flow inside `run_benchmark` for each model:

1. Read existing cache (unchanged from today, line 92).
2. **If `rerun_failed` is set:**
   - If the cache file does not exist → `continue` (skip this model; per the
     user's "patch failures only" choice, we do not start fresh runs).
   - Migrate columns: ensure `attempt` and `rerun_at` exist.
   - `failed_mask = df_res["success"] == False`
   - If `failed_mask.sum() == 0` → use cache as-is and continue (skip this
     model; nothing to patch).
   - Otherwise, build a subset
     `test_df_subset = pd.concat([df.iloc[example_rows], df.iloc[len(example_rows):][failed_mask.to_numpy()]], ignore_index=True)`
     — example rows preserved (`evaluate_model` needs them for few-shot) plus
     just the failed questions.
   - Call `evaluate_model(test_df_subset, config, model, language=language)` →
     produces `answers/resps/success/extra_fields` lists with length equal to
     `failed_mask.sum()`.
   - Splice back: for the positions where `failed_mask` is True, overwrite
     `resps, success, Latency, TokensUsed*`, set `attempt += 1`, set
     `rerun_at = now_iso`. (Leave `answers` alone — the ground-truth answer
     does not change.)
   - Write `df_res` back to the same cache file.
3. Continue with the existing post-processing (lines 103-128).

Pros:
- Library-level fix; any future caller of `run_benchmark` gets `--rerun` for free.
- Direct access to `test_df`, `config`, `evaluate_model`, cache path — no
  reconstruction.
- Single source of truth for "what does a cache row mean".

Cons:
- Touches `src/cdpk/` library code, which is shared / versioned more carefully
  than scripts.
- Slightly muddies `run_benchmark`'s single responsibility (it gains a
  conditional surgical-rerun branch on top of the existing all-or-nothing
  cache logic).

## Approach B — Pre-patch in the runner script

Keep `src/cdpk/` untouched. In
[scripts/run_pedagogy_benchmark_multilingual.py](scripts/run_pedagogy_benchmark_multilingual.py),
add a new function `rerun_failed_rows(cat_config_name, models_config, language)`
that runs **before** the existing call to `run_benchmark` for each category:

1. Resolve the models YAML the same way `load_from_config` does:
   `yaml.safe_load(ROOT / "configs/models" / f"{models_config}.yaml")`.
2. Resolve the test DataFrame and `config` the same way `load_from_config` does
   (we have to duplicate this — or factor it out and import it).
3. For each model in the YAML:
   - `cache_path = CACHE_LOCAL_DIR / cat_config_name / f"resps_{model}.csv"`
   - Skip if `cache_path` does not exist.
   - Read CSV, migrate `attempt`/`rerun_at` if absent.
   - If no `success==False` rows → skip.
   - Build `test_df_subset` exactly as in Approach A, call
     `evaluate_model(...)` directly, splice results back, bump `attempt`, set
     `rerun_at`, save CSV.
4. Then call `run_benchmark(...)` as usual — it will read the now-patched cache
   and proceed without re-issuing any calls (because `success=True` rows are
   considered cached).

Pros:
- Zero changes to `src/cdpk/`. All rerun logic is co-located with the CLI flag.
- Easier to add script-level features later (verbose rerun progress, dry-run
  mode listing what *would* be re-issued).

Cons:
- Duplicates the YAML/test-data loading dance that `load_from_config`
  performs.
- Reaches across the library boundary to call `evaluate_model` directly, so
  any future change to how cache rows are produced (e.g. a new column) has to
  be kept in sync in two places.
- Slightly more code overall.

## Recommended approach

**Approach A.** The provenance design (new columns + migration on read) is
already a library-level concern — every place that writes the cache has to
know about `attempt` and `rerun_at`, and `run_benchmark` is the only writer.
Locating the splice logic there keeps cache invariants in one file. The flag
plumbing in the script stays trivial:

```python
parser.add_argument("--rerun", action="store_true",
    help="Re-issue API calls for cached rows with success==False; skip "
         "models with no cache or no failures.")
...
category_df, summary_df, config, models_dict_PK = run_benchmark(
    questions_config=cat_config_name, models_config=config_models_PK,
    language=opt.language, rerun_failed=opt.rerun,
)
```

If we prefer to keep `src/cdpk/` frozen, Approach B is a clean fallback; the
migration/splicing logic is identical, just relocated.

## Files to modify (Approach A)

- [src/cdpk/benchmark_run.py](src/cdpk/benchmark_run.py)
  - Add `rerun_failed: bool = False` to `run_benchmark` signature ([:50-56](src/cdpk/benchmark_run.py#L50-L56)).
  - Insert the migrate + splice block between the cache-read and the
    `clean_resps/clean_answers` post-processing (between [:101](src/cdpk/benchmark_run.py#L101) and [:103](src/cdpk/benchmark_run.py#L103)).
  - Whenever `runner()` writes a fresh cache ([:65-67](src/cdpk/benchmark_run.py#L65-L67)), also stamp `attempt=1, rerun_at=""` on every row.
- [scripts/run_pedagogy_benchmark_multilingual.py](scripts/run_pedagogy_benchmark_multilingual.py)
  - Add `--rerun` CLI argument near [:178-181](scripts/run_pedagogy_benchmark_multilingual.py#L178-L181).
  - Forward `rerun_failed=opt.rerun` in the `run_benchmark` call at [:108-110](scripts/run_pedagogy_benchmark_multilingual.py#L108-L110).
  - Update the module docstring usage examples ([:13-31](scripts/run_pedagogy_benchmark_multilingual.py#L13-L31)).

No new files. No new dependencies.

## Verification

1. **Sanity-check a cache with known failures** — the currently open file
   [data/cache_local/CDPK_Nyankore_ep_maths/resps_glm-5.1.csv](data/cache_local/CDPK_Nyankore_ep_maths/resps_glm-5.1.csv)
   reportedly has `success=False` rows. Back it up:
   `Copy-Item data/cache_local/CDPK_Nyankore_ep_maths/resps_glm-5.1.csv resps_glm-5.1.csv.bak`.
2. **Dry inspection** —
   `python -c "import pandas as pd; d=pd.read_csv('data/cache_local/CDPK_Nyankore_ep_maths/resps_glm-5.1.csv'); print((~d.success).sum(), 'failed rows')"`.
3. **Run rerun on one category/language/model** — narrow the models YAML to
   just `glm-5.1` (or use an existing small config) and run:
   `uv run python scripts/run_pedagogy_benchmark_multilingual.py --language nyankore --benchmark cdpk --categories maths --models-config <slim> --rerun`.
4. **Verify**:
   - Same row count in cache CSV as before.
   - `(~d.success).sum()` is `<=` the original count (ideally 0 if the API was
     healthy).
   - All rows that flipped `success: False → True` also show `attempt=2` and a
     non-empty `rerun_at`.
   - Rows that were already `success=True` are byte-identical except for the
     new columns showing `attempt=1, rerun_at=""`.
5. **Negative case** — re-run with `--rerun` against a fully-successful cache:
   the script should report it skipped the model (no failures to patch) and
   not issue any API calls. Easiest check: watch `tqdm` output / network.
6. **No-cache case** — delete a cache file and run with `--rerun`: the model
   should be skipped (consistent with "patch failures only"), not started
   fresh.
