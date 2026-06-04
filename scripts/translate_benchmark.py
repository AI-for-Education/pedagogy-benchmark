"""
Translate the Pedagogy Benchmark (CDPK) dataset into a target language using Gemini.

Modes
-----
- translate (default): one API call per cell, 5 calls per row.
- optimized: one API call per row using a structured-output schema
  (`RowTranslationResult`) that returns all 5 columns at once. ~5x fewer calls.
- estimate_cost: dry-run that counts source words, converts to tokens via a
  configurable ratio, and prints projected $ cost for both per-cell and per-row
  modes side-by-side using prices from `fab-benchmarks-configs/models.csv`.
  Makes no API calls.
- retranslate: standalone re-verification of an already-translated CSV. Pass
  `--file <name>` (a filename in data/pedagogy_benchmark_full_datasets/) and
  `--language <name>`; missing (None/NaN) cells are retranslated against the
  HF source and the result is saved with `_cleaned` appended to the stem.
  Use this to re-verify a translation later without re-running the full pass.

Usage
-----
1. **Interactive window (VS Code / Jupyter)**
   - Open this file in VS Code and run cells with "Run Cell" (Ctrl+Shift+Enter).
   - Edit the CONFIGURATION section below to set LANGUAGE, MODEL_NAME, OUTPUT_FILE,
     and RETRANSLATE before running. CLI flags are ignored when no `sys.argv` is set.

2. **CLI (terminal)**
   Default per-cell translation (current behavior):
       uv run python scripts/translate_benchmark.py --language Dari

   Optimized 1-call-per-row translation:
       uv run python scripts/translate_benchmark.py --mode optimized --language Dari

   Dry-run cost estimate (no API calls):
       uv run python scripts/translate_benchmark.py --mode estimate_cost \\
           --language Pashto --model gemini-3.1-pro-preview

   Re-verify an existing translation file (no full re-run):
       uv run python scripts/translate_benchmark.py --mode retranslate \\
           --language swahili_tz \\
           --file pedagogy_benchmark_swahili_tz_cdpk.csv

   Available CLI flags:
       --mode {translate, optimized, estimate_cost, retranslate}  (default: translate)
       --language <name>                              (overrides LANGUAGE constant)
       --model <model_id>                             (overrides MODEL_NAME constant)
       --file <name>                                  (filename for retranslate mode;
                                                       relative to DATA_DIR)
       --no-retranslate                               (skip post-translation retranslate pass)

Configuration
-------------
LANGUAGE : str
    Target language key (e.g. 'luganda', 'swahili', 'hausa', 'yoruba', 'nyankore').
    See `cdpk.language_prompts.list_available_languages()` for all options.
MODEL_NAME : str
    Registered model name in custom_models.yaml. Must also exist in
    fab-benchmarks-configs/models.csv for cost estimation.
OUTPUT_FILE : str or None
    Custom output CSV filename. Leave as None to auto-generate
    'pedagogy_benchmark_{language}_cdpk.csv'.
RETRANSLATE : bool
    If True, after the initial translation pass the script checks for missing
    (None/NaN) cells and retranslates them, saving a '_cleaned.csv' variant.
    Only applies to `translate` / `optimized` modes (the `retranslate` mode
    always retranslates — that's its purpose).

Environment
-----------
Requires a GEMINI_API_KEY environment variable (or set in a .env file at the
project root).
"""

# %%
import argparse
import json
import logging
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from dotenv import load_dotenv
import sys

from fdllm import get_caller
from fdllm.llmtypes import LLMMessage
from fdllm.sysutils import register_models

# Silence the per-call "non-text parts in the response: ['thought_signature']"
# warning emitted by google-genai for thinking models (e.g. Gemini 3 Pro) when
# using structured output. The thought-signature parts are reasoning metadata;
# the parsed JSON we consume is correct.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data" / "pedagogy_benchmark_full_datasets"

# Add parent directory to path to import cdpk module
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(override=True)

# Register models from the project's custom_models.yaml
custom_models_file = ROOT / "fab-benchmarks-configs" / "custom_models.yaml"
register_models(custom_models_file)


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


# ==================== CONFIGURATION ====================
# Available languages: english, luganda, swahili, hausa, yoruba, nyankore, etc.
# Use list_available_languages() to see all options
LANGUAGE = 'Dari'
OUTPUT_FILE = None  # Optional: Set to a filename like 'pedagogy_benchmark_luganda_cdpk.csv' or leave as None for default
RETRANSLATE = False  # Parameter to retranslate missing cells after initial translation pass
#MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'  # Must match a key in custom_models.yaml
MODEL_NAME = 'gemini-3.1-pro-preview'
# =======================================================

# %%
# Test API
def test_gemini_api(model_name: str):

    #model = genai.GenerativeModel(model_name)
    #response = model.generate_content("Why is the sky blue?")

    #response = client.models.generate_content(
    #    model=model_name,
    #    contents="Why is the sky blue?"
    #)

    caller = get_caller(model_name)
    msg = LLMMessage(Role="user", Message="Why is the sky blue?")
    response = caller.call(msg, max_tokens=None, temperature=0.0)

    return response.Message

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
        verify-pass at the bottom of the script will retranslate it cell-wise.
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
    source_language: str = "English"
) -> pd.DataFrame:
    """
    Translates specified columns of a pandas DataFrame to a target language.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_translate (list): A list of column names to be translated.
        target_language (str): The language to translate the columns into.
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
        temp_lang_slug = target_language.lower().replace(" ", "_")
        temp_output_filename = f"pedagogy_benchmark_{temp_lang_slug}_cdpk_partial.csv"
        temp_output_path = DATA_DIR / temp_output_filename
        df_translated.to_csv(temp_output_path, index=False)

    return df_translated


def translate_dataframe_optimized(
    df: pd.DataFrame,
    model_name: str,
    columns_to_translate: list,
    target_language: str,
    source_language: str = "English",
    save_every: int = 50,
) -> pd.DataFrame:
    """One-API-call-per-row translation. ~5x fewer calls than translate_dataframe.

    Output CSV shape matches the per-cell path so the verify/retranslate logic
    works unchanged. Saves a `..._cdpk_partial.csv` snapshot every `save_every`
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

    lang_slug = target_language.lower().replace(" ", "_")
    temp_output_path = DATA_DIR / f"pedagogy_benchmark_{lang_slug}_cdpk_partial.csv"

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
            df_translated.to_csv(temp_output_path, index=False)

    df_translated.to_csv(temp_output_path, index=False)
    return df_translated


# Test of pipeline

#sample_df = pd.DataFrame({
#            'ID': [1, 2, 3],
#            'Title': ['Hello', 'Goodbye', 'Thank you'],
#            'Comment': ['This is a test.', 'Another test.', 'Final test.'],
#            'Numeric': [100, 200, 300]
#        })
#
## Define expected output
#expected_df = pd.DataFrame({
#    'ID': [1, 2, 3],
#    'Title': ['Bonjour', 'Au revoir', 'Merci'],
#    'Comment': ['Ceci est un test.', 'Un autre test.', 'Test final.'],
#    'Numeric': [100, 200, 300]
#})
#
#result_df = translate_dataframe(
#            sample_df,
#            model_name='gemini-2.5-flash-preview-09-2025',
#            columns_to_translate=['Title', 'Comment'],
#            target_language='French'
#        )
#
## Assert that the resulting DataFrame is identical to the expected one
#pd.testing.assert_frame_equal(result_df, expected_df)

# %%
# Another test with simple sentence
#result = translate_text("Hello, how are you?", 
#                        MODEL_NAME, "Arabic")
#print(result)  # Should be clean translated text, no JSON wrapper or extra explanation



# %%
# Translate the Pedagogy Benchmark datasets

WORDS_PER_TOKEN = 0.75  # ~1.43 tokens/word; common English approximation
COLUMNS_TO_TRANSLATE = ['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d']
PER_CELL_PROMPT_TOKENS = 90   # approx token count of the per-cell prompt template
PER_ROW_PROMPT_TOKENS = 130   # approx token count of the per-row prompt template (5 labelled fields)


def estimate_translation_cost(
    target_language: str,
    model_name: str,
    words_per_token: float = WORDS_PER_TOKEN,
    columns: list[str] = COLUMNS_TO_TRANSLATE,
) -> None:
    """Estimate the $ cost of translating CDPK to `target_language` with `model_name`.

    Reports per-cell mode (current default: 5 API calls per row) and per-row
    optimized mode (1 call per row) side-by-side so the savings are obvious.
    No API calls are made.
    """
    df = load_dataset("AI-for-Education/pedagogy-benchmark", "cdpk_main", split="train")
    df = pd.DataFrame(df)
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
        print(f"\nERROR: {model_name!r} not found in {models_csv}.")
        print("Cannot estimate cost without per-million-token prices.")
        return
    input_per_M = registry.loc[model_name, "input_cost"]
    output_per_M = registry.loc[model_name, "output_cost"]
    if pd.isna(input_per_M) or pd.isna(output_per_M):
        print(f"\nERROR: {model_name} has no input/output prices in models.csv.")
        return

    def _cost(in_tok, out_tok):
        return (in_tok / 1_000_000) * input_per_M + (out_tok / 1_000_000) * output_per_M

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
    print(f"Dataset:                CDPK main ({n_rows} rows, {len(columns)} columns)")
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


# All the argparse code has been removed from inside it.
def main(target_lang: str, optimized: bool = False, model_name: str | None = None):
    """
    Main function to load, translate, and save the pedagogy benchmark dataset.

    Args:
        target_lang: The target language for translation.
        optimized: If True, translate all 5 columns per row in one API call.
            If False (default), translate one cell per API call.
        model_name: Registered model name. Falls back to MODEL_NAME constant.
    """
    model_name = model_name or MODEL_NAME
    api_test_response = test_gemini_api(model_name=model_name)

    if api_test_response:
        print(f"Gemini API test successful. Starting translation to {target_lang}...")
    else:
        print("API test failed. Exiting script.")
        return

    # Load dataset from Hugging Face Hub
    cdpk_dataset_hf = load_dataset("AI-for-Education/pedagogy-benchmark", "cdpk_main", split="train")
    cdpk_dataset_hf = pd.DataFrame(cdpk_dataset_hf)

    # Check dataset loaded correctly
    if cdpk_dataset_hf.empty:
        print("Error: Loaded CDPK dataset is empty.")
        return
    print(f"Loaded CDPK dataset with {cdpk_dataset_hf.shape[0]} rows and {cdpk_dataset_hf.shape[1]} columns.")

    # Translate dataset using the language provided
    print(f"\nStarting translation for {target_lang} (optimized={optimized})...")
    translate_fn = translate_dataframe_optimized if optimized else translate_dataframe
    cdpk_dataset_translated = translate_fn(
        cdpk_dataset_hf,
        model_name=model_name,
        columns_to_translate=['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d'],
        target_language=target_lang,
    )

    # Create a dynamic output filename
    lang_slug = target_lang.lower().replace(" ", "_")
    output_filename = f"pedagogy_benchmark_{lang_slug}_cdpk.csv"
    output_path = DATA_DIR / output_filename

    # Save translated dataset to CSV
    if not cdpk_dataset_translated.empty:
        cdpk_dataset_translated.to_csv(output_path, index=False)
        print(f"\nTranslation complete. Translated dataset saved to: {output_path}")
    else:
        print("\nTranslation resulted in an empty DataFrame. File not saved.")

def retranslate_failed_translations(
    translated_df: pd.DataFrame,
    original_df: pd.DataFrame,
    columns_to_check: list[str],
    target_language: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Finds and re-translates cells with None values in a translated DataFrame.

    Args:
        translated_df (pd.DataFrame): The DataFrame with missing (None) translations.
        original_df (pd.DataFrame): The original DataFrame with the source text.
        columns_to_check (list[str]): A list of column names to check for None values.
        target_language (str): The language to translate the text into.
        model_name (str): The name of the model to use for translation.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled in.
    """
    # Ensure the dataframes are aligned
    if not translated_df.index.equals(original_df.index):
        raise ValueError("Indices of translated and original DataFrames do not match.")

    # 1. Find all (index, column) locations with None values
    none_locations = []
    for col in columns_to_check:
        if col in translated_df.columns:
            # Get indices where the column in translated_df is None or NaN
            none_indices = translated_df[translated_df[col].isna()].index
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
        original_text = original_df.at[idx, col]

        # Only proceed if there's actual text to translate in the original df
        if pd.notna(original_text) and isinstance(original_text, str) and original_text.strip():
            new_translation = translate_text(
                text_to_translate=original_text,
                model_name=model_name,
                target_language=target_language
            )
            # Use .at for fast, label-based single cell access
            translated_df.at[idx, col] = new_translation
        else:
            # If the original text was also empty/None, leave the translated cell as is.
            pass

    print("✨ Finished re-translating missing values.")
    return translated_df

# %%
# CLI dispatch
def _parse_cli_args():
    """Parse CLI args. Uses parse_known_args so Jupyter's `-f kernel.json`
    flag doesn't crash interactive cell execution."""
    p = argparse.ArgumentParser(description="Translate CDPK to a target language.")
    p.add_argument(
        "--mode",
        choices=["translate", "estimate_cost", "optimized", "retranslate"],
        default="translate",
        help="translate: per-cell (current default). optimized: 1 API call per row. "
             "estimate_cost: dry-run cost estimate, no API calls. "
             "retranslate: re-verify an existing translated file (requires --file).",
    )
    p.add_argument("--language", default=None, help="Override LANGUAGE constant.")
    p.add_argument("--model", default=None, help="Override MODEL_NAME constant.")
    p.add_argument("--file", default=None,
                   help="Filename (relative to DATA_DIR) for retranslate mode. "
                        "Output is written to '<stem>_cleaned.csv' alongside it.")
    p.add_argument("--no-retranslate", action="store_true",
                   help="Skip the post-translation retranslate pass.")
    args, _ = p.parse_known_args()
    return args


def _run_verify(output_file: str, target_language: str, model_name: str):
    """Verify and retranslate missing cells from `output_file`."""
    print("\nVerifying translations...")
    translated_df = pd.read_csv(DATA_DIR / output_file)
    print(translated_df.shape)
    col_to_translate = ['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d']
    n_missing = translated_df[col_to_translate].isna().sum().sum()
    print(f"Number of missing translations found: {n_missing}\n")
    if n_missing == 0:
        return

    print("Retranslating failed translations...")
    cdpk_dataset_hf = load_dataset("AI-for-Education/pedagogy-benchmark", "cdpk_main", split="train")
    cdpk_dataset_hf = pd.DataFrame(cdpk_dataset_hf)
    cleaned_translated_df = retranslate_failed_translations(
        translated_df=translated_df,
        original_df=cdpk_dataset_hf,
        columns_to_check=col_to_translate,
        target_language=target_language,
        model_name=model_name,
    )
    cleaned_output = output_file.replace(".csv", "_cleaned.csv")
    cleaned_translated_df.to_csv(DATA_DIR / cleaned_output, index=False)
    print(f"Cleaned dataset saved to: {DATA_DIR / cleaned_output}")


def _run_retranslate_standalone(input_file: str, target_language: str, model_name: str):
    """Standalone retranslate: read `input_file` from DATA_DIR, retranslate
    missing (None/NaN) cells in the CDPK answer columns for `target_language`,
    and write to '<stem>_cleaned.csv' next to it.
    """
    input_path = DATA_DIR / input_file
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}")
        return

    output_path = input_path.with_name(input_path.stem + "_cleaned.csv")
    if output_path.exists():
        print(f"ERROR: output file already exists: {output_path}")
        print("Aborting to avoid overwriting. Rename/move/delete it and rerun, "
              "or point --file at the existing cleaned file to continue cleaning.")
        return

    print(f"\nRetranslate mode — source: {input_path}")
    translated_df = pd.read_csv(input_path)
    print(f"Shape: {translated_df.shape}")
    col_to_translate = ['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d']
    n_missing = translated_df[col_to_translate].isna().sum().sum()
    print(f"Missing cells: {n_missing}")
    if n_missing == 0:
        print("Nothing to retranslate.")
        return

    print(f"Loading HF source dataset and retranslating to {target_language} "
          f"using model '{model_name}'...")
    cdpk_dataset_hf = pd.DataFrame(
        load_dataset("AI-for-Education/pedagogy-benchmark", "cdpk_main", split="train")
    )
    cleaned_df = retranslate_failed_translations(
        translated_df=translated_df,
        original_df=cdpk_dataset_hf,
        columns_to_check=col_to_translate,
        target_language=target_language,
        model_name=model_name,
    )
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")


def _report_missing_cells(output_file: str, target_language: str):
    """After a translate-mode run when auto-retranslate is disabled, report how
    many cells are missing and show the exact command to fix them.
    """
    output_path = DATA_DIR / output_file
    if not output_path.exists():
        print(f"\nWARNING: expected translated file not found: {output_path}")
        return
    df = pd.read_csv(output_path)
    cols = [c for c in ['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d']
            if c in df.columns]
    n_missing = int(df[cols].isna().sum().sum())
    print(f"\nMissing cells in {output_file}: {n_missing}")
    if n_missing == 0:
        print("Translation file is complete — no retranslate needed.")
    else:
        print("RETRANSLATE is disabled. To fix the missing cells, run:")
        print(f"  uv run python scripts/translate_benchmark.py --mode retranslate \\")
        print(f"      --language {target_language} \\")
        print(f"      --file {output_file}")


_cli = _parse_cli_args()
MODE = _cli.mode
target_language = _cli.language or LANGUAGE
model_name = _cli.model or MODEL_NAME
retranslate_after = RETRANSLATE and not _cli.no_retranslate
lang_slug = target_language.lower().replace(" ", "_")
output_file = OUTPUT_FILE or f"pedagogy_benchmark_{lang_slug}_cdpk.csv"

if MODE == "estimate_cost":
    estimate_translation_cost(target_language, model_name)
elif MODE == "retranslate":
    if not _cli.file:
        print("ERROR: --mode retranslate requires --file <name> "
              "(filename relative to DATA_DIR).")
    else:
        _run_retranslate_standalone(_cli.file, target_language, model_name)
else:  # translate (default per-cell) or optimized (per-row)
    output_path = DATA_DIR / output_file
    if output_path.exists():
        print(f"ERROR: output file already exists: {output_path}")
        print("Translation appears to have already been completed for this "
              "language. Rename/move/delete the file and rerun, or use "
              "`--mode retranslate --file <name>` to fix missing cells.")
    else:
        optimized_mode = (MODE == "optimized")
        main(target_lang=target_language, optimized=optimized_mode, model_name=model_name)
        if retranslate_after:
            _run_verify(output_file, target_language, model_name)
        else:
            _report_missing_cells(output_file, target_language)


# %%
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
## %%