"""
Translate the Pedagogy Benchmark (CDPK) dataset into a target language using Gemini.

Usage
-----
1. **Interactive window (VS Code / Jupyter)**
   - Open this file in VS Code and run cells with "Run Cell" (Ctrl+Shift+Enter).
   - Edit the CONFIGURATION section below to set LANGUAGE, OUTPUT_FILE, and VERIFY
     before running.

2. **CLI (terminal)**
   - Run directly with:
       uv run python scripts/translate_benchmark.py
   - To change the target language or other options, edit the CONFIGURATION
     section at the top of the file before running.

Configuration
-------------
LANGUAGE : str
    Target language key (e.g. 'luganda', 'swahili', 'hausa', 'yoruba', 'nyankore').
    See `cdpk.language_prompts.list_available_languages()` for all options.
OUTPUT_FILE : str or None
    Custom output CSV filename. Leave as None to auto-generate
    'pedagogy_benchmark_{language}_cdpk.csv'.
VERIFY : bool
    If True, after the initial translation pass the script checks for missing
    (None/NaN) cells and retranslates them, saving a '_cleaned.csv' variant.

Environment
-----------
Requires a GEMINI_API_KEY environment variable (or set in a .env file at the
project root).
"""

# %%
import json
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


# ==================== CONFIGURATION ====================
# Available languages: english, luganda, swahili, hausa, yoruba, nyankore, etc.
# Use list_available_languages() to see all options
LANGUAGE = 'Dari'
OUTPUT_FILE = None  # Optional: Set to a filename like 'pedagogy_benchmark_luganda_cdpk.csv' or leave as None for default
VERIFY = True  # Set to True to verify and retranslate missing values after initial translation
MODEL_NAME = 'gemini-2.5-flash-preview-09-2025'  # Must match a key in custom_models.yaml
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

    #prompt = (
    #    f"Translate the following text from {source_language} to {target_language}. "
    #    "Do not add any extra explanations, introductory text, or quotation marks. "
    #    "Just return the raw translated text.\n\n"
    #    f"Text to translate: \"{text_to_translate}\""
    #)
    
    # New prompt format from Translate Gemma paper
    prompt = (
        f"You are a professional {source_language} to {target_language} "
        f"translator. Your goal is to accurately convey the meaning and "
        f"nuances of the original {source_language} text while adhering to {target_language} grammar, "
        f"vocabulary, and cultural sensitivities. Produce only the {target_language} "
        f"translation, without any additional explanations or commentary. Please translate "
        f"the following {source_language} text into {target_language}:\n\n\n{text_to_translate}"
    )

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

        print(f"\nTranslating column: '{col}' to {target_language}...")

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

# All the argparse code has been removed from inside it.
def main(target_lang: str):
    """
    Main function to load, translate, and save the pedagogy benchmark dataset.

    Args:
        target_lang (str): The target language for translation.
    """
    # Test API once
    # Note: 'gemini-2.5-pro' is not a valid public model name.
    # Use a valid one like 'gemini-1.5-pro-latest' or 'gemini-pro'.
    api_test_response = test_gemini_api(model_name=MODEL_NAME)

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
    print(f"\nStarting translation for {target_lang}...")
    cdpk_dataset_translated = translate_dataframe(
        cdpk_dataset_hf,
        #model_name='gemini-2.5-pro',
        model_name=MODEL_NAME,
        columns_to_translate=['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d'],
        target_language=target_lang
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
# Main execution block
# Get language configuration
target_language = LANGUAGE
lang_slug = LANGUAGE.lower().replace(" ", "_")

# Default output filename if not provided
if OUTPUT_FILE is None:
    output_file = f"pedagogy_benchmark_{lang_slug}_cdpk.csv"
else:
    output_file = OUTPUT_FILE

# Call the main function with the parsed language
main(target_lang=target_language)

# Verify and retranslate if requested
if VERIFY:
    print("\nVerifying translations...")
    translated_df = pd.read_csv(DATA_DIR / output_file)
    print(translated_df.shape)
    col_to_translate = ['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d']
    print(f"Number of missing translations found: {translated_df[col_to_translate].isna().sum().sum()}\n")

    if translated_df[col_to_translate].isna().sum().sum() > 0:
        print("Retranslating failed translations...")
        # Load dataset from Hugging Face Hub
        cdpk_dataset_hf = load_dataset("AI-for-Education/pedagogy-benchmark", "cdpk_main", split="train")
        cdpk_dataset_hf = pd.DataFrame(cdpk_dataset_hf)

        # Check for missing translations in the DataFrame
        cleaned_translated_df = retranslate_failed_translations(
            translated_df=translated_df,
            original_df=cdpk_dataset_hf,
            columns_to_check=col_to_translate,
            target_language=target_language,
            model_name=MODEL_NAME
        )
        # Save the cleaned DataFrame back to CSV
        cleaned_output = output_file.replace(".csv", "_cleaned.csv")
        cleaned_translated_df.to_csv(DATA_DIR / cleaned_output, index=False)
        print(f"Cleaned dataset saved to: {DATA_DIR / cleaned_output}")
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

# %%
# ==================== RETRANSLATE DARI CLEANED FILE ====================
dari_cleaned_file = "pedagogy_benchmark_dari_cdpk_cleaned.csv"
dari_translated_df = pd.read_csv(DATA_DIR / dari_cleaned_file)

col_to_check = ['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d']
n_missing = dari_translated_df[col_to_check].isna().sum().sum()
print(f"Missing translations in Dari cleaned file: {n_missing}")

if n_missing > 0:
    from datasets import load_dataset as _load_dataset
    cdpk_original = pd.DataFrame(
        _load_dataset("AI-for-Education/pedagogy-benchmark", "cdpk_main", split="train")
    )
    dari_cleaned_df = retranslate_failed_translations(
        translated_df=dari_translated_df,
        original_df=cdpk_original,
        columns_to_check=col_to_check,
        target_language="Dari",
        model_name=MODEL_NAME,
    )
    dari_cleaned_df.to_csv(DATA_DIR / "pedagogy_benchmark_dari_cdpk_cleaned_2.csv", index=False)
    print(f"Updated Dari cleaned file saved to: {DATA_DIR / 'pedagogy_benchmark_dari_cdpk_cleaned_2.csv'}")
else:
    print("No missing translations — Dari cleaned file is complete.")
# %%
