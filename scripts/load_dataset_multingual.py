# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# %%
# Load both splits
dataset_dict = load_dataset("CraneAILabs/pedagogy-benchmark-multilingual", "default")
dataset_dict
# %%
# Convert to a pandas DataFrame
df = dataset_dict['train'].to_pandas()
print(df.shape)
df.head()

# %%
# Split dataset in half by language

# Luganda = first half
df_luganda = df.iloc[:len(df)//2].reset_index(drop=True)

# Nyankore = second half
df_nyankore = df.iloc[len(df)//2:].reset_index(drop=True)

print(df_luganda.shape)
print(df_nyankore.shape)

display(df_luganda.head(2), df_nyankore.head(2))

# %%
# save csv files
df_luganda.to_csv(f"{DATA_DIR}/pedagogy_benchmark_luganda.csv", index=False)
#df_nyankore.to_csv(f"{DATA_DIR}/pedagogy_benchmark_multilingual_nyankore.csv", index=False)


# %%
# Divide CDPK and SEND questions if needed

df_luganda_cdpk= df_luganda[~(df_luganda['category'] == 'SEND')].reset_index(drop=True)
df_luganda_send= df_luganda[df_luganda['category'] == 'SEND'].reset_index(drop=True)

df_nyankore_cdpk= df_nyankore[~(df_nyankore['category'] == 'SEND')].reset_index(drop=True)
df_nyankore_send= df_nyankore[df_nyankore['category'] == 'SEND'].reset_index(drop=True)

print(df_luganda_cdpk.shape, df_luganda_send.shape)
print(df_nyankore_cdpk.shape, df_nyankore_send.shape)

# %%
# save all files
df_luganda_cdpk.to_csv(f"{DATA_DIR}/pedagogy_benchmark_luganda_cdpk.csv", index=False)
df_luganda_send.to_csv(f"{DATA_DIR}/pedagogy_benchmark_luganda_send.csv", index=False)

# %%
## Load Luganda main questions
#dataset = load_dataset("CraneAILabs/pedagogy-benchmark-multilingual", "luganda", split="cdpk_main")
#
## Load Nyankore SEND questions
#dataset = load_dataset("CraneAILabs/pedagogy-benchmark-multilingual", "nyankore", split="cdpk_send")
#
## Access a question
#print(dataset[0]['question'])
#print(f"A) {dataset[0]['answer_a']}")
#print(f"B) {dataset[0]['answer_b']}")
#print(f"C) {dataset[0]['answer_c']}")
#print(f"D) {dataset[0]['answer_d']}")
#print(f"Correct: {dataset[0]['correct_answer']}")
#

# %%
