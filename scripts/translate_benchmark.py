"""
Translate a Pedagogy Benchmark dataset (CDPK or SEND) into a target language with an LLM.

CLI only:
    uv run python scripts/translate_benchmark.py --mode optimized --dataset cdpk --language french

The English source is downloaded from Hugging Face (AI-for-Education/pedagogy-benchmark,
config `cdpk_main` for CDPK, `cdpk_send` for SEND). Each run pins the dataset version
(commit hash) and records it, with the model, in data/translation_log.json.

Modes (--mode)
--------------
- translate (default): one API call per cell, 5 calls per row.
- optimized: one API call per row using a structured-output schema
  (`RowTranslationResult`) that returns all 5 columns at once. ~5x fewer calls.
- estimate_cost: dry-run that counts source words, converts to tokens via a
  configurable ratio, and prints projected $ cost for both per-cell and per-row
  modes side-by-side using prices from `fab-benchmarks-configs/models.csv`.
  Makes no API calls.
- retranslate: re-verify an already-translated CSV given with `--file`. Missing
  (None/NaN or whitespace-only) cells are retranslated from the same Hugging Face
  version as the original translation (read from the log), matched by question_id,
  and the result is saved as described in "Output files" below.

Options
-------
--dataset {cdpk,send}    Benchmark to translate (required).
--language NAME          Target language, base name only (e.g. french, never french_ep:
                         '_ep' is only for prepare_cdpk_dataset_multilingual.py and the
                         benchmark runner). Used in the prompt and in the file names.
--model MODEL_ID         Registered model in fab-benchmarks-configs/custom_models.yaml
                         (default: DEFAULT_MODEL). Must be in models.csv for estimate_cost.
--file NAME              File in data/pedagogy_benchmark_full_datasets/ (retranslate mode).
--retranslate {true,false}
                         translate/optimized modes only. Default false: if cells are missing
                         after the run, print the retranslate command. true: retranslate them
                         right away. A complete translation is saved as `_cleaned` either way.

Examples (PowerShell: end lines with a backtick, or write everything on one line):
    uv run python scripts/translate_benchmark.py --mode estimate_cost --dataset cdpk `
        --language pashto --model gemini-3.1-pro-preview
    uv run python scripts/translate_benchmark.py --mode retranslate --dataset cdpk `
        --language swahili_tz --file pedagogy_benchmark_swahili_tz_cdpk.csv

Output files
------------
All in data/pedagogy_benchmark_full_datasets/ ({dataset} is cdpk or send):
- `pedagogy_benchmark_{language}_{dataset}.csv`: raw translation, may have missing cells.
- `..._{dataset}_cleaned.csv`: verified, every cell translated. Always written once a
  file is verified complete, even when nothing had to be retranslated (it is then
  a copy of the raw file). This is the file to pass to
  `scripts/prepare_cdpk_dataset_multilingual.py`.
- `..._{dataset}_retranslated.csv`: retranslation done but some cells failed again.
  Not verified: retry with `--mode retranslate --file <this file>` until it
  produces the `_cleaned.csv` file.
- `..._{dataset}_partial.csv`: snapshot saved during a translation run.

Translation log
---------------
data/translation_log.json has one entry per translation (key: raw file name without
.csv) recording the English source (Hugging Face repo, config, split and commit hash),
the model, the mode and the date, plus one record per retranslation. Entries with
"synthetic": true were backfilled after the fact, not recorded at translation time.

Environment
-----------
Requires the API key of the model's provider, e.g. GEMINI_API_KEY for Gemini models,
as an environment variable or in a .env file at the project root.
"""
import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from fdllm import get_caller
from fdllm.llmtypes import LLMMessage
from fdllm.sysutils import register_models
from huggingface_hub import HfApi
from pydantic import BaseModel
from tqdm import tqdm

# Silence the per-call "non-text parts in the response: ['thought_signature']"
# warning emitted by google-genai for thinking models (e.g. Gemini 3 Pro) when
# using structured output. The thought-signature parts are reasoning metadata;
# the parsed JSON we consume is correct.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "pedagogy_benchmark_full_datasets"
LOG_FILE = ROOT / "data" / "translation_log.json"

HF_REPO = "AI-for-Education/pedagogy-benchmark"
HF_CONFIGS = {"cdpk": "cdpk_main", "send": "cdpk_send"}  # --dataset -> Hugging Face config
HF_SPLIT = "train"

DEFAULT_MODEL = 'gemini-3.1-pro-preview'  # Must match a key in custom_models.yaml

WORDS_PER_TOKEN = 0.75  # ~1.43 tokens/word; common English approximation
COLUMNS_TO_TRANSLATE = ['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d']
PER_CELL_PROMPT_TOKENS = 90   # approx token count of the per-cell prompt template
PER_ROW_PROMPT_TOKENS = 130   # approx token count of the per-row prompt template (5 labelled fields)


class TranslationResult(BaseModel):
    """Structured output schema for translation responses."""
    translated_text: str


class RowTranslationResult(BaseModel):
    """Structured output for translating all 5 CDPK columns in one call."""
    question: str
    answer_a: str
    answer_b: str
    answer_c: str
    answer_d: str


def _build_translation_prompt(
    items: dict[str, str] | str,
    source_language: str,
    target_language: str,
) -> str:
    """Build the translation prompt. Shared by per-cell and per-row modes.

    Keeps the existing Translate-Gemma-style instructions verbatim so per-cell
    and per-row paths produce comparable quality. When `items` is a string, the
    prompt asks for a single translation (per-cell mode). When `items` is a
    dict {field_name: text}, the prompt lists every field by name and asks the
    model to return one translation per field via the structured-output schema.
    """
    header = (
        f"You are a professional {source_language} to {target_language} "
        f"translator. Your goal is to accurately convey the meaning and "
        f"nuances of the original {source_language} text while adhering to {target_language} grammar, "
        f"vocabulary, and cultural sensitivities. Produce only the {target_language} "
        f"translation, without any additional explanations or commentary. Please translate "
        f"the following {source_language} text into {target_language}"
    )
    if isinstance(items, str):
        return f"{header}:\n\n\n{items}"
    fields_block = "\n\n".join(f"{k}: {v}" for k, v in items.items())
    return (
        f"{header}.\n\n"
        f"There are {len(items)} separate fields to translate. Return one translation "
        f"per field, using the same field names, in the structured output schema:\n\n\n"
        f"{fields_block}"
    )


def test_gemini_api(model_name: str):
    """One short call to check that the model and API key work before a full run."""
    caller = get_caller(model_name)
    msg = LLMMessage(Role="user", Message="Why is the sky blue?")
    response = caller.call(msg, max_tokens=None, temperature=0.0)

    return response.Message


# ==================== Hugging Face source ====================

def current_hf_revision() -> str:
    """Commit hash of the latest version of the Hugging Face dataset."""
    return HfApi().dataset_info(HF_REPO).sha


def load_source(dataset: str, revision: str) -> pd.DataFrame:
    """English source rows of `dataset`, at the exact Hugging Face version `revision`."""
    return pd.DataFrame(load_dataset(HF_REPO, HF_CONFIGS[dataset], split=HF_SPLIT, revision=revision))


# ==================== Translation log ====================

def _read_log() -> dict:
    return json.loads(LOG_FILE.read_text(encoding="utf-8")) if LOG_FILE.exists() else {}


def _update_log(base: str, fields: dict | None = None, retranslation: dict | None = None):
    """Add or update the log entry of one translation.

    fields: keys set on the entry (e.g. source, translation).
    retranslation: one record appended to the entry's "retranslations" list.
    """
    log = _read_log()
    entry = log.setdefault(base, {})
    entry.update(fields or {})
    if retranslation:
        entry.setdefault("retranslations", []).append(retranslation)
    LOG_FILE.write_text(json.dumps(dict(sorted(log.items())), indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")


# --- Core Translation Logic ---
def translate_text(
    text_to_translate: str,
    model_name: str,
    target_language: str,
    source_language: str = "English",
) -> str:
    """
    Translates a single string of text using fdllm with structured output.

    fdllm provides built-in retry logic (8 attempts, exponential backoff).
    Uses Pydantic structured output (TranslationResult) to ensure the model
    returns only the translation without stray explanations.

    Args:
        text_to_translate (str): The text to be translated.
        model_name (str): The registered model name in custom_models.yaml.
        target_language (str): The language to translate the text into.
        source_language (str): The source language of the text.

    Returns:
        str: The translated text, or None if translation fails.
    """
    if not isinstance(text_to_translate, str) or not text_to_translate.strip():
        return text_to_translate  # Return non-strings or empty strings as is

    # Add Modern Standard Arabic in parentheses for Arabic translations to help
    # the model understand the target language better
    if target_language.lower() == "arabic":
        target_language += " (Modern Standard Arabic)"
    # Tanzanian Swahili variant — use slug `swahili_tz` to keep the filename
    # distinct from the existing `swahili` translation.
    elif target_language.lower() == "swahili_tz":
        target_language = "Swahili (Tanzania)"

    # New prompt format from Translate Gemma paper (built via shared helper so
    # per-cell and per-row modes stay in lock-step).
    prompt = _build_translation_prompt(text_to_translate, source_language, target_language)

    try:
        caller = get_caller(model_name)
        msg = LLMMessage(Role="user", Message=prompt)
        response = caller.call(
            msg,
            max_tokens=None,
            temperature=0.0,
            response_schema=TranslationResult,
        )

        result = json.loads(response.Message)
        translated_text = result["translated_text"].strip()

        if translated_text:
            return translated_text
        else:
            print(f"Warning: API returned empty translation for '{text_to_translate[:50]}...'")
            return None
    except Exception as e:
        print(f"Translation failed for '{text_to_translate[:50]}...': {e}")
        return None


def translate_row(
    row: dict,
    model_name: str,
    target_language: str,
    source_language: str = "English",
) -> dict | None:
    """Translate all fields of a single row in one structured API call.

    Args:
        row: Mapping of {field_name: source_text} — typically the 5 CDPK columns.
        model_name: Registered fdllm model name.
        target_language: Language to translate into.
        source_language: Source language.

    Returns:
        Dict {field_name: translated_text} on success. If translation fails or
        returns an empty string for a field, that field is set to None so the
        verify-pass will retranslate it cell-wise.
    """
    # Skip rows where every field is empty / non-string
    payload = {
        k: v for k, v in row.items()
        if isinstance(v, str) and v.strip()
    }
    if not payload:
        return {k: row.get(k) for k in row}

    if target_language.lower() == "arabic":
        target_language += " (Modern Standard Arabic)"
    elif target_language.lower() == "swahili_tz":
        target_language = "Swahili (Tanzania)"

    prompt = _build_translation_prompt(payload, source_language, target_language)

    try:
        caller = get_caller(model_name)
        msg = LLMMessage(Role="user", Message=prompt)
        response = caller.call(
            msg,
            max_tokens=None,
            temperature=0.0,
            response_schema=RowTranslationResult,
        )
        result = json.loads(response.Message)

        translated = {}
        for k in row:
            if k in payload:
                val = (result.get(k) or "").strip()
                translated[k] = val if val else None
            else:
                # Field was empty/non-string in source — pass through unchanged
                translated[k] = row.get(k)
        return translated
    except Exception as e:
        first_key = next(iter(payload))
        snippet = str(payload[first_key])[:50]
        print(f"Row translation failed (e.g. '{snippet}...'): {e}")
        return {k: None if k in payload else row.get(k) for k in row}


def translate_dataframe(
    df: pd.DataFrame,
    model_name: str,
    columns_to_translate: list,
    target_language: str,
    partial_path: Path,
    source_language: str = "English"
) -> pd.DataFrame:
    """
    Translates specified columns of a pandas DataFrame to a target language.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_translate (list): A list of column names to be translated.
        target_language (str): The language to translate the columns into.
        partial_path (Path): Snapshot saved after each column.
        source_language (str, optional): The source language. Defaults to "English".

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns translated.
    """
    df_translated = df.copy()

    # Initialize tqdm for pandas
    tqdm.pandas()

    for col in columns_to_translate:
        if col not in df_translated.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue

        print(f"\nTranslating column: '{col}' to {target_language} using model '{model_name}'...")

        # Using .progress_apply to show a progress bar
        df_translated[col] = df_translated[col].progress_apply(
            lambda x: translate_text(x,
                                     model_name,
                                     target_language,
                                     source_language,
                                     )
        )
        # save after each column
        df_translated.to_csv(partial_path, index=False)

    return df_translated


def translate_dataframe_optimized(
    df: pd.DataFrame,
    model_name: str,
    columns_to_translate: list,
    target_language: str,
    partial_path: Path,
    source_language: str = "English",
    save_every: int = 50,
) -> pd.DataFrame:
    """One-API-call-per-row translation. ~5x fewer calls than translate_dataframe.

    Output CSV shape matches the per-cell path so the verify/retranslate logic
    works unchanged. Saves a snapshot to `partial_path` every `save_every`
    rows so a crash mid-run doesn't lose all progress.
    """
    df_translated = df.copy()
    missing_cols = [c for c in columns_to_translate if c not in df_translated.columns]
    if missing_cols:
        print(f"Warning: columns not in DataFrame, skipping: {missing_cols}")
        columns_to_translate = [c for c in columns_to_translate if c in df_translated.columns]
    if not columns_to_translate:
        return df_translated

    print(f"\nTranslating {len(df_translated)} rows to {target_language} using model '{model_name}' "
          f"(optimized: 1 call/row across columns {columns_to_translate})...")

    for i, (idx, row) in enumerate(tqdm(df_translated.iterrows(),
                                        total=len(df_translated),
                                        desc=f"Rows -> {target_language}")):
        source_row = {c: row[c] for c in columns_to_translate}
        translated = translate_row(
            source_row,
            model_name=model_name,
            target_language=target_language,
            source_language=source_language,
        )
        if translated is not None:
            for c in columns_to_translate:
                df_translated.at[idx, c] = translated.get(c)

        if save_every and (i + 1) % save_every == 0:
            df_translated.to_csv(partial_path, index=False)

    df_translated.to_csv(partial_path, index=False)
    return df_translated


def estimate_translation_cost(
    dataset: str,
    target_language: str,
    model_name: str,
    words_per_token: float = WORDS_PER_TOKEN,
    columns: list[str] = COLUMNS_TO_TRANSLATE,
) -> None:
    """Estimate the $ cost of translating `dataset` to `target_language` with `model_name`.

    Reports per-cell mode (current default: 5 API calls per row) and per-row
    optimized mode (1 call per row) side-by-side so the savings are obvious.
    No API calls are made.
    """
    df = load_source(dataset, current_hf_revision())
    n_rows = len(df)

    cell_words = 0
    for c in columns:
        if c in df.columns:
            cell_words += df[c].fillna("").astype(str).str.split().str.len().sum()
    content_tokens = cell_words / words_per_token

    cell_calls = n_rows * len(columns)
    cell_input_tokens = content_tokens + cell_calls * PER_CELL_PROMPT_TOKENS
    cell_output_tokens = content_tokens  # assume 1:1 translation length

    row_calls = n_rows
    row_input_tokens = content_tokens + row_calls * PER_ROW_PROMPT_TOKENS
    row_output_tokens = content_tokens

    models_csv = ROOT / "fab-benchmarks-configs" / "models.csv"
    registry = pd.read_csv(models_csv).set_index("model_id")
    if model_name not in registry.index:
        sys.exit(f"ERROR: {model_name!r} not found in {models_csv}. "
                 "Cannot estimate cost without per-million-token prices.")
    input_per_M = registry.loc[model_name, "input_cost"]
    output_per_M = registry.loc[model_name, "output_cost"]
    if pd.isna(input_per_M) or pd.isna(output_per_M):
        sys.exit(f"ERROR: {model_name} has no input/output prices in models.csv.")

    cell_in_cost = (cell_input_tokens / 1_000_000) * input_per_M
    cell_out_cost = (cell_output_tokens / 1_000_000) * output_per_M
    cell_total = cell_in_cost + cell_out_cost

    row_in_cost = (row_input_tokens / 1_000_000) * input_per_M
    row_out_cost = (row_output_tokens / 1_000_000) * output_per_M
    row_total = row_in_cost + row_out_cost

    savings_pct = 100 * (cell_total - row_total) / cell_total if cell_total else 0

    print("=" * 78)
    print("TRANSLATION COST ESTIMATE")
    print("=" * 78)
    print(f"Dataset:                {dataset.upper()} ({HF_CONFIGS[dataset]}, {n_rows} rows, {len(columns)} columns)")
    print(f"Target language:        {target_language}")
    print(f"Model:                  {model_name}")
    print(f"Pricing (per 1M tok):   input ${input_per_M:.4f}  /  output ${output_per_M:.4f}")
    print(f"Word->token ratio:      tokens = words / {words_per_token}  (~{1/words_per_token:.2f} tok/word)")
    print(f"Total source words:     {cell_words:,}")
    print(f"Content tokens (~):     {content_tokens:,.0f}")
    print()
    print(f"{'Mode':<22} {'Calls':>8} {'Input tok':>14} {'Output tok':>14} {'$ Input':>10} {'$ Output':>10} {'$ Total':>10}")
    print("-" * 78)
    print(f"{'per-cell (current)':<22} {cell_calls:>8,} {cell_input_tokens:>14,.0f} {cell_output_tokens:>14,.0f}"
          f" {cell_in_cost:>10.4f} {cell_out_cost:>10.4f} {cell_total:>10.4f}")
    print(f"{'per-row (optimized)':<22} {row_calls:>8,} {row_input_tokens:>14,.0f} {row_output_tokens:>14,.0f}"
          f" {row_in_cost:>10.4f} {row_out_cost:>10.4f} {row_total:>10.4f}")
    print("-" * 78)
    print(f"Optimized savings:      ${cell_total - row_total:.4f}  ({savings_pct:.1f}%)")
    print()
    print("Notes:")
    print("  - Output tokens assumed 1:1 with content tokens. Non-Latin scripts")
    print("    (Arabic/Dari/Pashto) often tokenize 1.5-3x worse — expect higher cost.")
    print("  - Excludes verify-pass retries (~2-5% of cells in current runs).")
    print("=" * 78)


def run_translation(dataset: str, target_lang: str, optimized: bool, model_name: str, base: str) -> bool:
    """Translate the whole dataset, save the raw file '<base>.csv' and log the run.

    Args:
        dataset: "cdpk" or "send".
        target_lang: The target language for translation.
        optimized: If True, translate all 5 columns per row in one API call.
            If False, translate one cell per API call.
        model_name: Registered model name.
        base: Raw output file name without .csv, e.g. pedagogy_benchmark_french_cdpk.

    Returns:
        True if the raw file was saved.
    """
    if not test_gemini_api(model_name=model_name):
        print("API test failed. Exiting script.")
        return False
    print(f"API test successful. Starting translation to {target_lang}...")

    # Pin the Hugging Face version so the source can be found again for retranslation
    revision = current_hf_revision()
    source_df = load_source(dataset, revision)
    if source_df.empty:
        print(f"Error: Loaded {dataset.upper()} dataset is empty.")
        return False
    print(f"Loaded {HF_CONFIGS[dataset]} (revision {revision}) with {source_df.shape[0]} rows "
          f"and {source_df.shape[1]} columns.")

    print(f"\nStarting translation for {target_lang} (optimized={optimized})...")
    translate_fn = translate_dataframe_optimized if optimized else translate_dataframe
    translated_df = translate_fn(
        source_df,
        model_name=model_name,
        columns_to_translate=COLUMNS_TO_TRANSLATE,
        target_language=target_lang,
        partial_path=DATA_DIR / f"{base}_partial.csv",
    )

    if translated_df.empty:
        print("\nTranslation resulted in an empty DataFrame. File not saved.")
        return False
    output_path = DATA_DIR / f"{base}.csv"
    translated_df.to_csv(output_path, index=False)
    print(f"\nTranslation complete. Translated dataset saved to: {output_path}")
    _update_log(base, fields={
        "language": target_lang,
        "dataset": dataset,
        "source": {"hf_repo": HF_REPO, "hf_config": HF_CONFIGS[dataset], "split": HF_SPLIT,
                   "revision": revision},
        "translation": {"model": model_name, "mode": "optimized" if optimized else "translate",
                        "date": date.today().isoformat()},
        "synthetic": False,
    })
    return True


# ==================== Verification ====================

def _missing_mask(df: pd.DataFrame, cols: list[str] = COLUMNS_TO_TRANSLATE) -> pd.DataFrame:
    """True where a translated cell is missing: None/NaN or whitespace-only.

    Same definition as the load check in prepare_cdpk_dataset_multilingual.py,
    so a file verified here is accepted there.
    """
    text = df[cols]
    return text.isna() | text.astype(str).apply(lambda col: col.str.strip() == "")


def _missing_ids(df: pd.DataFrame) -> list:
    """question_id (or row index) of rows with at least one missing cell."""
    rows = _missing_mask(df).any(axis=1)
    return (df.loc[rows, "question_id"] if "question_id" in df.columns else df.index[rows]).tolist()


def _base_stem(file_name: str) -> str:
    """Translation file stem without the verification suffixes, e.g.
    'pedagogy_benchmark_french_cdpk_retranslated.csv' -> 'pedagogy_benchmark_french_cdpk'."""
    return Path(file_name).stem.removesuffix("_cleaned").removesuffix("_retranslated")


def _save_verified(df: pd.DataFrame, source_file: str, dataset: str, target_language: str) -> Path | None:
    """Save a checked translation under a name that tells whether it is complete.

    - No missing cell: '<stem>_cleaned.csv'. `_cleaned` always means "verified, every
      cell translated", whether or not a retranslation was needed. This is the file
      to pass to prepare_cdpk_dataset_multilingual.py.
    - Missing cells left (a retried cell failed again): '<stem>_retranslated.csv',
      to pass again to `--mode retranslate`. Retrying a `_retranslated` file
      overwrites it, which only fills more cells.

    Returns the saved path, or None if nothing was saved.
    """
    base = _base_stem(source_file)
    missing_ids = _missing_ids(df)
    if not missing_ids:
        path = DATA_DIR / f"{base}_cleaned.csv"
        if path.exists():
            print(f"ERROR: {path} already exists, not overwritten.")
            return None
        df.to_csv(path, index=False)
        print(f"✅ Verified: no missing cell. Saved as {path}")
        return path

    path = DATA_DIR / f"{base}_retranslated.csv"
    if path.exists() and path.name != Path(source_file).name:
        print(f"ERROR: {path} already exists, not overwritten.")
        return None
    df.to_csv(path, index=False)
    print(f"⚠️ {int(_missing_mask(df).sum().sum())} cells still missing after retranslation "
          f"(question_id: {missing_ids[:20]}). Saved as {path}, NOT as _cleaned. Retry with:")
    print(f"  uv run python scripts/translate_benchmark.py --mode retranslate "
          f"--dataset {dataset} --language {target_language} --file {path.name}")
    return path


def retranslate_failed_translations(
    translated_df: pd.DataFrame,
    source_df: pd.DataFrame,
    columns_to_check: list[str],
    target_language: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Finds and re-translates missing cells in a translated DataFrame.

    Rows are matched with the English source by question_id, not by position, so a
    file with rows in another order (or a different source version) cannot pick up
    another question's text.

    Args:
        translated_df (pd.DataFrame): The DataFrame with missing translations.
        source_df (pd.DataFrame): The English source with the original text.
        columns_to_check (list[str]): Column names to check for missing values.
        target_language (str): The language to translate the text into.
        model_name (str): The name of the model to use for translation.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled in.
    """
    source_by_id = source_df.set_index("question_id")
    unknown = sorted(set(translated_df["question_id"]) - set(source_by_id.index))
    if unknown:
        raise ValueError(f"question_id not in the English source: {unknown[:20]}")

    # 1. Find all (index, column) locations with missing values
    none_locations = []
    for col in columns_to_check:
        if col in translated_df.columns:
            # Get indices where the column in translated_df is None, NaN or whitespace-only
            none_indices = translated_df[_missing_mask(translated_df, [col])[col]].index
            for idx in none_indices:
                none_locations.append((idx, col)) # Store as (index, column) tuples
        else:
            print(f"Warning: Column '{col}' not found in the translated DataFrame. Skipping.")

    if not none_locations:
        print("✅ No missing translations found. DataFrame is clean!")
        return translated_df

    print(f"🔍 Found {len(none_locations)} missing translations. Beginning re-translation process...")

    # 2. Iterate only over the identified locations and re-translate
    for idx, col in tqdm(none_locations, desc="Retrying translations"):
        original_text = source_by_id.at[translated_df.at[idx, "question_id"], col]

        # Only proceed if there's actual text to translate in the source
        if pd.notna(original_text) and isinstance(original_text, str) and original_text.strip():
            translated_df.at[idx, col] = translate_text(
                text_to_translate=original_text,
                model_name=model_name,
                target_language=target_language
            )

    print("✨ Finished re-translating missing values.")
    return translated_df


def _verify_and_retranslate(input_file: str, dataset: str, target_language: str, model_name: str):
    """Check `input_file` for missing cells, retranslate them if any, then save the
    result with _save_verified: '<stem>_cleaned.csv' when complete (even if nothing
    had to be retranslated), '<stem>_retranslated.csv' when cells are still missing.
    The source version comes from the translation log, so retranslated cells match
    the text the file was translated from.
    """
    input_path = DATA_DIR / input_file
    if not input_path.exists():
        sys.exit(f"ERROR: file not found: {input_path}")
    base = _base_stem(input_file)
    cleaned_path = DATA_DIR / f"{base}_cleaned.csv"
    if cleaned_path.exists():
        sys.exit(f"ERROR: {cleaned_path} already exists: this translation is already verified. "
                 "Rename/move/delete it to verify again.")

    print(f"\nVerifying translations in {input_path}")
    translated_df = pd.read_csv(input_path)
    n_missing = int(_missing_mask(translated_df).sum().sum())
    print(f"Shape: {translated_df.shape} | missing cells: {n_missing}")

    if n_missing:
        revision = _read_log().get(base, {}).get("source", {}).get("revision")
        if revision is None:
            revision = current_hf_revision()
            print(f"Warning: no source version for {base} in {LOG_FILE.name}; using the current "
                  f"Hugging Face version {revision} (rows are still matched by question_id).")
        print(f"Loading {HF_CONFIGS[dataset]} (revision {revision}) and retranslating to "
              f"{target_language} using model '{model_name}'...")
        translated_df = retranslate_failed_translations(
            translated_df=translated_df,
            source_df=load_source(dataset, revision),
            columns_to_check=COLUMNS_TO_TRANSLATE,
            target_language=target_language,
            model_name=model_name,
        )
    saved = _save_verified(translated_df, input_file, dataset, target_language)
    if saved is not None:
        _update_log(base, retranslation={
            "date": date.today().isoformat(),
            "model": model_name if n_missing else None,  # no API call when nothing was missing
            "input_file": input_path.name,
            "cells_missing_before": n_missing,
            "cells_missing_after": int(_missing_mask(translated_df).sum().sum()),
            "output_file": saved.name,
        })


def _report_missing_cells(output_file: str, dataset: str, target_language: str):
    """After a translate-mode run without --retranslate true, report how many cells
    are missing and show the exact command to fix them. A complete file is saved as
    '<stem>_cleaned.csv' right away (no API call needed).
    """
    output_path = DATA_DIR / output_file
    if not output_path.exists():
        print(f"\nWARNING: expected translated file not found: {output_path}")
        return
    df = pd.read_csv(output_path)
    n_missing = int(_missing_mask(df).sum().sum())
    print(f"\nMissing cells in {output_file}: {n_missing}")
    if n_missing == 0:
        print("Translation file is complete — no retranslate needed.")
        saved = _save_verified(df, output_file, dataset, target_language)
        if saved is not None:
            _update_log(_base_stem(output_file), retranslation={
                "date": date.today().isoformat(), "model": None, "input_file": output_file,
                "cells_missing_before": 0, "cells_missing_after": 0, "output_file": saved.name,
            })
    else:
        print("To fix the missing cells, run:")
        print(f"  uv run python scripts/translate_benchmark.py --mode retranslate "
              f"--dataset {dataset} --language {target_language} --file {output_file}")


# ==================== CLI ====================

def _parse_cli_args():
    p = argparse.ArgumentParser(
        description="Translate a Pedagogy Benchmark dataset (CDPK or SEND). "
                    "See the module docstring for details.")
    p.add_argument(
        "--mode",
        choices=["translate", "estimate_cost", "optimized", "retranslate"],
        default="translate",
        help="translate: per-cell (default). optimized: 1 API call per row. "
             "estimate_cost: dry-run cost estimate, no API calls. "
             "retranslate: re-verify an existing translated file (requires --file).",
    )
    p.add_argument("--dataset", required=True, choices=sorted(HF_CONFIGS),
                   help="Benchmark to translate: cdpk or send.")
    p.add_argument("--language", required=True,
                   help="Target language, base name only (e.g. french, not french_ep).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Registered model name in custom_models.yaml (default: {DEFAULT_MODEL}).")
    p.add_argument("--file", default=None,
                   help="File in data/pedagogy_benchmark_full_datasets/ to verify (retranslate mode).")
    p.add_argument("--retranslate", choices=["true", "false"], default="false", type=str.lower,
                   help="translate/optimized modes: retranslate missing cells right after the run "
                        "(default false: only print the command).")
    return p.parse_args()


def main():
    args = _parse_cli_args()
    target_language, model_name, dataset = args.language, args.model, args.dataset
    lang_slug = target_language.lower().replace(" ", "_")
    base = f"pedagogy_benchmark_{lang_slug}_{dataset}"

    if lang_slug.endswith("_ep"):
        # "_ep" (English Prompt) only selects the prompt language when preparing and
        # running the benchmark; translation always uses the base language. Translating
        # with "french_ep" would ask the model for a language called "french_ep" and write
        # a second set of files next to the existing French translation.
        sys.exit(f"ERROR: --language {target_language!r} ends with '_ep'. Use the base language "
                 f"({target_language[:-3]!r}) to translate; '_ep' is only for "
                 "prepare_cdpk_dataset_multilingual.py and the benchmark runner.")

    # Loaded here rather than at import, so importing this file has no side effect
    load_dotenv(ROOT / ".env", override=True)
    register_models(ROOT / "fab-benchmarks-configs" / "custom_models.yaml")

    if args.mode == "estimate_cost":
        estimate_translation_cost(dataset, target_language, model_name)
    elif args.mode == "retranslate":
        if not args.file:
            sys.exit("ERROR: --mode retranslate requires --file <name> "
                     "(file name in data/pedagogy_benchmark_full_datasets/).")
        if _base_stem(args.file) != base:
            sys.exit(f"ERROR: --file {args.file!r} does not match --dataset {dataset} and "
                     f"--language {target_language}: expected {base}.csv (or its _cleaned/_retranslated version).")
        _verify_and_retranslate(args.file, dataset, target_language, model_name)
    else:  # translate (default per-cell) or optimized (per-row)
        # Check every file of an earlier translation of this language before any API
        # call: only checking the raw file would let a full (paid) translation run when
        # just the _cleaned file was kept, and then refuse to save its verified copy.
        existing = [p for p in (DATA_DIR / f"{base}{suffix}.csv" for suffix in ("", "_cleaned", "_retranslated"))
                    if p.exists()]
        if existing:
            sys.exit("ERROR: this language has already been translated, no API call made. Existing files:\n  "
                     + "\n  ".join(str(p) for p in existing)
                     + "\nRename/move/delete them to translate again, or use "
                       "`--mode retranslate --file <name>` to fix missing cells.")
        if not run_translation(dataset, target_language, args.mode == "optimized", model_name, base):
            sys.exit(1)
        if args.retranslate == "true":
            _verify_and_retranslate(f"{base}.csv", dataset, target_language, model_name)
        else:
            _report_missing_cells(f"{base}.csv", dataset, target_language)


if __name__ == "__main__":
    main()


# ==================== TRANSLATION ERROR DETECTION ====================
# Common translation artifacts from Gemini, identified through manual inspection.
# Typically affects 2-5% of translations.
#TRANSLATION_ERROR_PATTERNS = {
#    "silent_thinking": {
#        "description": "Silent thinking / internal reasoning tokens leaked into the translation",
#        "keywords": ["think", "thinking", "THINKING", "silent", "sorry", "system", "user"],
#    },
#    "multiple_alternatives": {
#        "description": "Model provides 2-3 alternative translations instead of one",
#        "keywords": ["or alternatively", "alternatively", "could also be translated as", "another translation", "it could translate to"],
#    },
#    "word_repetition": {
#        "description": "A word or phrase repeated excessively (infinite loop artifact)",
#        "max_char_length": 2000,
#    },
#    "parenthetical_english": {
#        "description": "English words kept in parentheses alongside the translation",
#        "regex": r"\([A-Za-z]+\)",
#    },
#}
