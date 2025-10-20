# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# %%

# Load datasets from HF Hub

cdpk_dataset = load_dataset("CraneAILabs/pedagogy-benchmark-multilingual", "luganda", split="cdpk_main")

send_dataset = load_dataset("CraneAILabs/pedagogy-benchmark-multilingual", "luganda", split="cdpk_send")

df_cdpk = cdpk_dataset.to_pandas()
df_send = send_dataset.to_pandas()

print(f"CDPK questions: {len(df_cdpk)}")
print(f"SEND questions: {len(df_send)}")

print(f"CDPK columns: {df_cdpk.columns.tolist()}")
print(f"SEND columns: {df_send.columns.tolist()}")

display(df_cdpk.head(2), df_send.head(2))

# %%
# check missing values
print("CDPK missing values:")
print(df_cdpk[['question', 'answer_a', 'answer_b', 'answer_c', 'answer_d']].isna().sum())

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
df_cdpk.to_csv(f"{DATA_DIR}/pedagogy_benchmark_luganda_cdpk.csv", index=False)
df_send.to_csv(f"{DATA_DIR}/pedagogy_benchmark_luganda_send.csv", index=False)

#
# %%
