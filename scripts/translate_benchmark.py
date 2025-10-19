# %%
import os
import time
import pandas as pd
#import google.generativeai as genai
#from google.generativeai import types
from google import genai 
from google.genai import types
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from dotenv import load_dotenv
import argparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
#dotenv_path = ROOT / ".env"
#load_dotenv(dotenv_path, override=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

#genai.configure(api_key=gemini_api_key)
client = genai.Client(api_key=gemini_api_key)

# %%
# Test API
def test_gemini_api(model_name: str):

    #model = genai.GenerativeModel(model_name)
    #response = model.generate_content("Why is the sky blue?")

    response = client.models.generate_content(
        model=model_name,
        contents="Why is the sky blue?"
    )

    return response.text

# --- Core Translation Logic ---
def translate_text(
    text_to_translate: str,
    model_name: str,
    target_language: str,
    source_language: str = "English",
    retries: int = 3,
    delay: int = 2,
) -> str:
    """
    Translates a single string of text using the Gemini API.

    Args:
        text_to_translate (str): The text to be translated.
        target_language (str): The language to translate the text into.
        source_language (str): The source language of the text.
        retries (int): Number of times to retry if the API returns an empty string.
        delay (int): Seconds to wait between retries.

    Returns:
        str: The translated text, or None if translation fails.
    """
    if not isinstance(text_to_translate, str) or not text_to_translate.strip():
        return text_to_translate  # Return non-strings or empty strings as is

    prompt = (
        f"Translate the following text from {source_language} to {target_language}. "
        "Do not add any extra explanations, introductory text, or quotation marks. "
        "Just return the raw translated text.\n\n"
        f"Text to translate: \"{text_to_translate}\""
    )

    for attempt in range(retries):
        try:
            
            #model = genai.GenerativeModel(model_name)
            #response = model.generate_content(prompt)

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            
            translated_text = response.text.strip()

            if translated_text:  # If the response is not empty
                return translated_text
            else:
                # This case handles a successful API call that returns an empty string
                print(f"Warning: API returned an empty response for '{text_to_translate}'. Retrying ({attempt + 1}/{retries})...")
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed for '{text_to_translate}': {e}")
            # If this was the last attempt, we exit the loop and return None below
            if attempt == retries - 1:
                break
        
        # Wait before the next retry
        time.sleep(delay)  

    print(f"Error: Translation failed for '{text_to_translate}' after {retries} retries. Returning original text.")
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

# Translate the Pedagogy Benchmark datasets

# %%

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
    api_test_response = test_gemini_api(model_name='gemini-2.5-flash-preview-09-2025') 

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
        model_name='gemini-2.5-flash-preview-09-2025',
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
# Call the main function with the desired target language
main(target_lang="Luganda")

# %%
# Verify that the file does not have missing values
translated_df = pd.read_csv(DATA_DIR / "pedagogy_benchmark_luganda_cdpk.csv")
print(translated_df.shape)
print(f"Number of missing translations found: {translated_df[['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d']].isna().sum().sum()}\n")
translated_df.info()

# %%
# Retranslate any failed translations

# Load dataset from Hugging Face Hub
cdpk_dataset_hf = load_dataset("AI-for-Education/pedagogy-benchmark", "cdpk_main", split="train")
cdpk_dataset_hf = pd.DataFrame(cdpk_dataset_hf)

col_to_translate = ['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d']

# Check for missing translations in the DataFrame
cleaned_translated_df = retranslate_failed_translations(
    translated_df=translated_df,
    original_df=cdpk_dataset_hf,
    columns_to_check=col_to_translate,
    target_language="Luganda",
    model_name="gemini-2.5-flash-preview-09-2025"
)
# Save the cleaned DataFrame back to CSV
#cleaned_translated_df.to_csv(DATA_DIR / "pedagogy_benchmark_luganda_cdpk_cleaned.csv", index=False)

# %%
# recheck
cleaned_translated_df.info()


# %%
# run block only when script is executed directly
# This block only runs when you execute the file as a script.
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="Translate the pedagogy benchmark dataset using the Gemini API.")
#    parser.add_argument(
#        '-l', '--language',
#        type=str,
#        required=True,
#        help='The target language for translation (e.g., "Luganda", "Swahili").'
#    )
#    args = parser.parse_args()
#    
#    # Call the main function with the parsed language
#    main(target_lang=args.language)
## %%
