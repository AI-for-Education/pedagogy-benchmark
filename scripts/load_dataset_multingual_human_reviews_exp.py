# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import json

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# %%
# Load human reviewed translations from HF Hub

human_reviewed_dataset = load_dataset("CraneAILabs/pedagogy-luganda-reviewed", "default", split="train")

df = human_reviewed_dataset.to_pandas()

print(f"CDPK questions: {len(df)}")
print(f"CDPK columns: {df.columns.tolist()}")
df.head(5)

# %%
# Preprocessing
# 1. Remove some columns
cols_remove = ['id', 'english_question', 'english_answer_a', 'english_answer_b', 'english_answer_c', 'english_answer_d', 'translation_quality_rating', 'review_status', 'corrections', 'issues_tagged']
#cols_remove = ['id', 'luganda_question', 'luganda_answer_a', 'luganda_answer_b', 'luganda_answer_c', 'luganda_answer_d', 'translation_quality_rating', 'review_status', 'corrections', 'issues_tagged']

df = df.drop(columns=cols_remove)

# Split between CDPK and SEND
df_cdpk = df[df['split'] == 'cdpk_main'].drop(columns=['split']).reset_index(drop=True)
df_send = df[df['split'] == 'cdpk_send'].drop(columns=['split']).reset_index(drop=True)

print(f"CDPK questions: {len(df_cdpk)} rows")
print(f"SEND questions: {len(df_send)} rows")

# remove "luganda_" from columns names in both dataframes
df_cdpk.columns = [col.replace("luganda_", "") for col in df_cdpk.columns]
df_send.columns = [col.replace("luganda_", "") for col in df_send.columns]
#df_cdpk.columns = [col.replace("english_", "") for col in df_cdpk.columns]
#df_send.columns = [col.replace("english_", "") for col in df_send.columns]

# add columns 'answer_e', 'answer_f', 'answer_g' with NaN values to both dataframes
for col in ['answer_e', 'answer_f', 'answer_g']:
    df_cdpk[col] = np.nan
    df_send[col] = np.nan

# Reorder columns to have correct answer column in position 6 (7th column after answer D)
cols_order = ['question_id', 'question', 'answer_a', 'answer_b', 'answer_c', 'answer_d', 'answer_e', 'answer_f', 'answer_g', 'correct_answer', 'category', 'subdomain', 'age_group']
df_cdpk = df_cdpk[cols_order]
df_send = df_send[cols_order]

display(df_cdpk.head(2), df_send.head(2))

# %%
# Check if few shot examples have been human reviewed for each category
# import dict few_shot_examples_idx_dict.json
with open(DATA_DIR / "few_shot_examples_idx_dict.json", "r") as f:
    few_shot_examples_idx_dict = json.load(f)

total_few_shot = 0
for category in df_cdpk['category'].unique():
    category_name = category.lower().replace(" ", "_") if category != "General" else "gen_pk"
    few_shot_indices = few_shot_examples_idx_dict.get(f"CDPK_{category_name}", [])
    print(f"\nIndices for category {category}: {few_shot_indices}")
    subset_df = df_cdpk[df_cdpk['category'] == category]
    # Check if few shot indices are in column "question_id" of subset_df
    reviewed_few_shot = set(subset_df['question_id'].astype(int)).intersection(set(few_shot_indices))
    num_reviewed = len(reviewed_few_shot)
    total_few_shot += num_reviewed
    #print(f"Category: {category}, Found {num_reviewed} FS examples out of {len(few_shot_indices)}")
    if num_reviewed < len(few_shot_indices):
        print(f"Missing {len(few_shot_indices) - num_reviewed} few-shot examples that were not human reviewed.")
        #print(f"Indices available in dataset: {list(subset_df['question_id'].values)}")
    else:
        print(f"All {num_reviewed} few-shot examples have been human reviewed.")

print(f"\nTotal few-shot examples human reviewed across all categories: {total_few_shot}. Should be 21.")

# %% 
# Add the few shot examples back to the main dataframe, delete the ones that are already there
def add_few_shot_examples(df, df_orig, few_shot_examples_idx_dict):

    few_shot_idx = []
    for category in df_orig['category'].unique():
        if category == "SEND":
            continue
        category_dict = category.lower().replace(" ", "_") if category != "General" else "gen_pk"
        few_shot_idx.extend(few_shot_examples_idx_dict.get(f"CDPK_{category_dict}", []))

    # remove rows from df that are in few_shot_idx
    df_cleaned = df[~df['question_id'].astype(int).isin(few_shot_idx)].reset_index(drop=True)
    print(f"Removed {len(df) - len(df_cleaned)} few-shot examples from the main dataframe.")

    # Add few-shot examples from original dataframe
    few_shot_examples = df_orig[df_orig['question_id'].astype(int).isin(few_shot_idx)].reset_index(drop=True)
    df_new = pd.concat([df_cleaned, few_shot_examples], axis=0).reset_index(drop=True)
    print(f"Added {len(few_shot_examples)} few-shot examples from the original dataframe.")

    return df_new

df_cdpk_orig = pd.read_csv(DATA_DIR / "pedagogy_benchmark_luganda_cdpk_cleaned.csv")
print(f"Original CDPK dataset shape: {df_cdpk_orig.shape}")

df_cdpk = add_few_shot_examples(df_cdpk, df_cdpk_orig, few_shot_examples_idx_dict)
print(f"New CDPK dataset shape after adding few-shot examples: {df_cdpk.shape}")

# %%
# check missing values
print("CDPK missing values:")
print(df_cdpk[['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d', 'correct_answer']].isna().sum())

# %%
columns_check = [
    'question',
    'answer_a',
    'answer_b',
    'answer_c',
    'answer_d',
    'correct_answer',
    'category',
]

# Display rows with missing values in any of the important columns
missing_rows_cdpk = df_cdpk[df_cdpk[columns_check].isna().any(axis=1)]
missing_rows_send = df_send[df_send[columns_check].isna().any(axis=1)]

print(f"CDPK rows with missing values in important columns: {len(missing_rows_cdpk)}")
print(f"SEND rows with missing values in important columns: {len(missing_rows_send)}")

# %%
print(f"Categories with missing values in CDPK: {missing_rows_cdpk['category'].value_counts().to_dict()}")
print(f"Categories with missing values in SEND: {missing_rows_send['category'].value_counts().to_dict()}")

# %%
# save as csv files
df_cdpk.to_csv(f"{DATA_DIR}/pedagogy_benchmark_luganda_cdpk_reviewed.csv", index=False)
df_send.to_csv(f"{DATA_DIR}/pedagogy_benchmark_luganda_send_reviewed.csv", index=False)
#df_cdpk.to_csv(f"{DATA_DIR}/pedagogy_benchmark_english_cdpk_reviewed.csv", index=False)
#df_send.to_csv(f"{DATA_DIR}/pedagogy_benchmark_english_send_reviewed.csv", index=False)

#
# %%
