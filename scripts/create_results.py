# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import json
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
import tiktoken
#from transformers import AutoTokenizer # to run tokenizer for deepseek

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
CACHE_LOCAL_DIR = DATA_DIR / "cache_local"
FIGS_DIR = DATA_DIR / "results"
FAB_CONFIGS_DIR = ROOT / "fab-benchmarks-configs"

MODELS_TO_EXCUDE = [
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-preview-09-2025",
    "gpt-5-2025-08-07-medium",
]

CATEGORY_TO_PLOT = "science"  # options: "math", "reading", "science"

# %%
# import useful files
models_csv = pd.read_csv(FAB_CONFIGS_DIR / "models.csv")
providers_csv = pd.read_csv(FAB_CONFIGS_DIR / "providers.csv")

# %%
# Load datasets from HF Hub
cdpk_dataset = load_dataset("AI-for-Education/pedagogy-benchmark", "cdpk_main", split="train")
send_dataset = load_dataset("AI-for-Education/pedagogy-benchmark", "cdpk_send", split="train")

df_cdpk = cdpk_dataset.to_pandas()
df_send = send_dataset.to_pandas()

print("CDPK dataset shape:", df_cdpk.shape)
print("SEND dataset shape:", df_send.shape)

categories_cdpk_dict = df_cdpk['category'].value_counts().to_dict()
categories_send_dict = df_send['category'].value_counts().to_dict()

display(df_cdpk.head(1), df_send.head(1))

# %%

# Accuracy plots
# Load data from each language folder
language_list = [
    "Luganda",
    "Nyankore",
    "Swahili",
    "Hausa",
    "Yoruba",
    "English"
]

language_speakers_dict = {
  "English": 1457e6,
  "Swahili": 97e6,
  "Luganda": 6e6,
  "Hausa": 94e6,
  "Yoruba": 50e6,
  "Nyankore": 3e6
}

# import code from pedagogy benchmark code to clean resp files and create acc dataframes

REPAT = [
    r"^\s*([ABCDEFG])(?:[\.(?:\s*\n)]+.*)*$",
    r"^<think>[\s\S]*?</think>[\s\S]*?([ABCDEFG])$", # for deepseek R1
    r"^## Step 1[\s\S]*?([ABCDEFG])[\.\s]*$", # for llama 4
    r'[\s\S]*\n([A-G])"?$', # for claude 4
    r"^\s*([ABCDEFG])[,.]",  # generic pattern added
]
REQ = [re.compile(pat) for pat in REPAT]

def clean_resps(resp):
    if pd.isna(resp):
        return
    for req in REQ:
        match = req.match(resp)
        if match is not None:
            break
    if match is None:
        return
    groups = match.groups()
    return groups[0]


def clean_answers(ans):
    return ans.replace(" and", ",").strip()

def plot_heatmap(pivot_df, 
                 language_speakers_dict, 
                 title="TODO",
                 colorbarlabel="TODO",
                 reverse_cmap=False,
                 vmin=None,
                 vmax=None
                 ):

    # order by number of speakers: reorder index in descending order of speakers
    pivot_df = pivot_df[sorted(pivot_df.columns, key=lambda x: language_speakers_dict.get(x, 0), reverse=True)]

    # --- 2. Create the heatmap ---
    plt.figure(figsize=(11, 6))

    ax = sns.heatmap(
        pivot_df,
        annot=True,     # Show the latency values in the cells
        fmt=".1f",      # Format numbers to two decimal places
        annot_kws={'size': 14},       # Smaller latency text
        cbar_kws={"label": colorbarlabel},  # Color bar label
        cmap='RdYlGn_r' if reverse_cmap else 'RdYlGn', # Red-Yellow-Green (reversed) is intuitive: Red=Slow, Green=Fast
        linewidths=.7,  # Add lines between cells
        vmin=vmin,
        vmax=vmax   
    )

    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Language', fontsize=15)
    ax.set_ylabel('Model', fontsize=15)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=10)

    # --- Add number of speakers above each box ---
    for j, lang in enumerate(pivot_df.columns):
        speakers_millions = language_speakers_dict.get(lang, 0) / 1e6
        ax.text(
            j + 0.5, 0.0,  # position above cell
            f"{speakers_millions:.0f} M",
            ha='center', va='bottom',
            fontsize=13, color='grey'
        )

    plt.tight_layout()

def plot_avg_heatmap(pivot_df_avg, 
                     language_speakers_dict, 
                     title='TODO', 
                     colorbarlabel='TODO', 
                     reverse_cmap=False,
                     vmin=None,
                     vmax=None
                     ):

    # order by number of speakers: reorder index in descending order of speakers
    pivot_df_avg = pivot_df_avg[sorted(pivot_df_avg.columns, key=lambda x: language_speakers_dict.get(x, 0), reverse=True)]

    # --- 2. Create the heatmap ---
    plt.figure(figsize=(8, 1))

    ax = sns.heatmap(
        pivot_df_avg.loc[['Average Model']],
        annot=True,     # Show the latency values in the cells
        fmt=".1f",      # Format numbers to two decimal places
        annot_kws={'size': 14},       # Smaller latency text
        cbar_kws={"label": colorbarlabel},  # Color bar label
        cmap='RdYlGn_r' if reverse_cmap else 'RdYlGn', # Red-Yellow-Green (reversed) is intuitive: Red=Slow, Green=Fast
        linewidths=.7,  # Add lines between cells
        vmin=vmin,  # <-- Pass vmin to heatmap
        vmax=vmax   # <-- Pass vmax to heatmap
    )

    ax.set_title(title, fontsize=12, pad=20)
    ax.set_xlabel('Language', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)

    # --- Add number of speakers above each box ---
    for j, lang in enumerate(pivot_df_avg.columns):
        speakers_millions = language_speakers_dict.get(lang, 0) / 1e6
        ax.text(
            j + 0.5, 0.0,  # position above cell
            f"{speakers_millions:.0f} M",
            ha='center', va='bottom',
            fontsize=10, color='grey'
        )

    plt.tight_layout()

MODELS_METADATA_MAPPING = {
    "Claude Sonnet 4.5": {
        "model_id": "claude-sonnet-4-5-20250929",
        "size": "Large",
        "reasoning": True,
    },
    "Deepseek R1": {
        "model_id": "fw-deepseek-r1-0528",
        "size": "Large",
        "reasoning": True,
    },
    "GPT-5 (Medium)": {
        "model_id": "gpt-5-2025-08-07-medium",
        "size": "Large",
        "reasoning": True,
    },
    "Gemini-2.5 Pro": {
        "model_id": "gemini-2.5-pro-preview-06-05",
        "size": "Large",
        "reasoning": True,
    },
    "Gemini-2.5 Flash": {
        "model_id": "gemini-2.5-flash-preview-09-2025",
        "size": "Large",
        "reasoning": True,
    },
    "Gemini-2.5 Flash-Lite": {
        "model_id": "gemini-2.5-flash-lite-preview-09-2025",
        "size": "Medium",
        "reasoning": False,
    },
    "o4-Mini": {
        "model_id": "o4-mini-2025-04-16",
        "size": "Medium",
        "reasoning": True,
    },
    "Qwen-3 32B": {
        "model_id": "qwen-3-32b",
        "size": "Medium",
        "reasoning": True,
    },
    "Gemma-3 27B": {
        "model_id": "gemma-3-27b",
        "size": "Medium",
        "reasoning": False,
    },
    "Gemma-3 4B": {
        "model_id": "gemma-3-4b-it",
        "size": "Small",
        "reasoning": False,
    },
}

# %%

acc_rows = []

# Only take data from category = CATEGORY and without "reviewed" in the language name
model_folders = [f for f in os.listdir(CACHE_LOCAL_DIR) if f.startswith("CDPK_") and
                    (CATEGORY_TO_PLOT in f.lower()) and ("reviewed" not in f.lower())
                    ]

print(f"Processing {len(model_folders)} model folders...")

for model_folder in model_folders:
    # --- Robust Folder Name Parsing ---
    name_parts = model_folder.replace("CDPK_", "").split("_")
    
    # Check for and remove the english prompt flag
    if "ep" in name_parts:
        english_prompt = True
        name_parts.remove("ep")
    else:
        english_prompt = False
        
    # Assume category is the last part, and language is everything before it
    category = name_parts[-1].capitalize()
    language = "_".join(name_parts[:-1]).capitalize().replace("_new", "")
    # --- End of Parsing ---

    # Find all CSV files in the folder
    files = list((CACHE_LOCAL_DIR / model_folder).glob("*.csv"))

    for file in files:
        df_res = pd.read_csv(file)
        model_name = file.stem
        model = model_name.replace("resps_", "")

        if model in MODELS_TO_EXCUDE:
            continue  # skip these models as newer versions exist
        
        # process dataframe (assuming clean_resps and clean_answers are defined)
        df_res["resps"] = df_res["resps"].apply(clean_resps)
        df_res["answers"] = df_res["answers"].apply(clean_answers)

        acc_score = (df_res["resps"] == df_res["answers"]).mean()
        bad_format_score = df_res["resps"].isna().mean()

        # Append a dictionary for this specific result directly to the list
        row = {
            "category": category,
            "model": model,
            "language": language,
            "english_prompt": english_prompt,
            "accuracy": acc_score * 100,  # convert to percentage
            "bad_format": bad_format_score * 100,  # convert to percentage
            "Latency Mean": df_res['Latency'].mean(),
            "Latency Median": df_res['Latency'].median()
        }
        acc_rows.append(row)
        
# Convert the list of rows to a DataFrame
acc_df = pd.DataFrame(acc_rows)

# add metadata from models_csv
#acc_df['display_name'] = acc_df['model'].apply(lambda x: models_csv.loc[models_csv['model_id'] == x, 'display_name'].values[0] if x in models_csv['model_id'].values else x)
acc_df['provider'] = acc_df['model'].apply(lambda x: models_csv.loc[models_csv['model_id'] == x, 'provider'].values[0] if x in models_csv['model_id'].values else 'Unknown')

print("Accuracy DataFrame shape:", acc_df.shape)
acc_df.head()

# %%
# save acc_df to csv
acc_df.to_csv(FIGS_DIR / "cdpk_multilingual_model_performance.csv", index=False)











# %%
# Latency plots

all_dfs = []

for model_folder in model_folders:

    # --- Robust Folder Name Parsing ---
    name_parts = model_folder.replace("CDPK_", "").split("_")

    # Check for and remove the english prompt flag
    if "ep" in name_parts:
        english_prompt = True
        name_parts.remove("ep")
    else:
        english_prompt = False
        
    # Assume category is the last part, and language is everything before it
    category = name_parts[-1].capitalize()
    language = "_".join(name_parts[:-1]).capitalize().replace("_new", "")
    # --- End of Parsing ---

    # Find all CSV files in the folder
    files = list((CACHE_LOCAL_DIR / model_folder).glob("*.csv"))

    for file in files:
        df_res = pd.read_csv(file)
        model_name = file.stem
        model = model_name.replace("resps_", "")

        if model in MODELS_TO_EXCUDE:
            continue  # skip these models as newer versions exist
        
        total_latency = df_res['Latency'].sum()

        row = {
            'model': model,
            'question config': model_folder,
            'language': language,
            'english_prompt': english_prompt,
            'category': category,
            'total latency time (s)': total_latency
        }
        df_latency = pd.DataFrame([row])
        all_dfs.append(df_latency)

latency_df = pd.concat(all_dfs, ignore_index=True)

#latency_df['language'] = latency_df['question config'].apply(lambda x: x.split('_')[1])
#latency_df['category'] = latency_df['question config'].apply(lambda x: x.split('_')[-1].capitalize())
# add latency per question column, use iterrows
latency_df['latency per question (s)'] = latency_df.apply(lambda row: row['total latency time (s)'] / categories_cdpk_dict.get(row['category'], 1), axis=1)
# create latency per question by taking median, not mean
#latency_df['latency per question (s)'] = latency_df['latency per question (s)'].median()

# add column to see if prompt in English, True if "ep" in 3rd group of question config
#latency_df['english_prompt'] = latency_df['question config'].apply(lambda x: True if 'ep' in x.split('_')[2] else False)

# add column display_name from models_csv, provider from providers_csv
#latency_df['display_name'] = latency_df['model'].apply(lambda x: models_csv.loc[models_csv['model_id'] == x, 'display_name'].values[0] if x in models_csv['model_id'].values else x)
latency_df['provider'] = latency_df['model'].apply(lambda x: models_csv.loc[models_csv['model_id'] == x, 'provider'].values[0] if x in models_csv['model_id'].values else 'Unknown')

print("Latency DataFrame shape:", latency_df.shape)
print(latency_df['language'].value_counts())
print(latency_df['category'].value_counts())
latency_df.head()

# %%
# save acc_df to csv
latency_df.to_csv(FIGS_DIR / "cdpk_multilingual_model_latency.csv", index=False)





# %%

# Latency and tokens count deeper analysis
    

# -----------------------------------------------------------------------------
# 1. SETUP: Pattern & Tokenizers (Global Scope)
# -----------------------------------------------------------------------------

# Pattern to capture thinking blocks (including tags)
# Note: This is defined outside as requested
THINK_PATTERN = r'(<think>.*?</think>)'

# Load Tokenizer 1: DeepSeek (Transformers)
try:
    # Using the specific model from your snippet
    deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528")
    print("DeepSeek tokenizer loaded successfully.")
except (OSError, ValueError):
    print("Warning: DeepSeek tokenizer not found. Using fallback or verify path.")
    deepseek_tokenizer = None

# Load Tokenizer 2: OpenAI/TikToken (cl100k)
cl100k_encoder = tiktoken.get_encoding("cl100k_base")

# %%
# -----------------------------------------------------------------------------
# 2. UNIFIED FUNCTION
# -----------------------------------------------------------------------------

def count_tokens(text, tokenizer, tokenizer_name):
    """
    Unified function to count tokens for a given text string.
    
    Args:
        text (str): The text to count.
        tokenizer: The tokenizer object 
                   
    Returns:
        int: Number of tokens.
    """
    if pd.isna(text):
        return 0
    
    text = str(text)
    
    # ---------------------------------------------------------
    # Case A: Hugging Face Tokenizers (DeepSeek, Llama, etc.)
    # ---------------------------------------------------------
    if tokenizer_name == "deepseek":
        # add_special_tokens=False is CRITICAL to avoid inflating counts 
        # with invisible "Start of Sentence" tokens.
        return len(tokenizer.encode(text, add_special_tokens=False))
    
    # ---------------------------------------------------------
    # Case B: TikToken (OpenAI cl100k, p50k, etc.)
    # ---------------------------------------------------------
    elif tokenizer_name == "cl100k":
        # Tiktoken objects don't support 'add_special_tokens'
        return len(tokenizer.encode(text))
    else:
        raise ValueError(f"Unknown tokenizer type: {type(tokenizer)}")

def extract_and_count_think(text, tokenizer, tokenizer_name):
        if pd.isna(text): return 0
        # 1. Extract (Logic outside the count function)
        matches = re.findall(THINK_PATTERN, str(text), re.DOTALL | re.IGNORECASE)
        content_to_count = "".join(matches)
        # 2. Count
        return count_tokens(content_to_count, tokenizer, tokenizer_name)
    

acc_rows_detailed = []
model_folders = [f for f in os.listdir(CACHE_LOCAL_DIR) if f.startswith("CDPK_")]

def clean_list(list):
    """Replaces any NaN/nat with None in a list"""
    # very important when saving to csv later
    return [None if pd.isna(x) else x for x in list]

for model_folder in model_folders:
    # --- Robust Folder Name Parsing ---
    name_parts = model_folder.replace("CDPK_", "").split("_")
    
    # Check for and remove the english prompt flag
    if "ep" in name_parts:
        english_prompt = True
        name_parts.remove("ep")
    else:
        english_prompt = False
        
    # Assume category is the last part, and language is everything before it
    category = name_parts[-1].capitalize()
    language = "_".join(name_parts[:-1]).capitalize().replace("_new", "")
    # --- End of Parsing ---

    # Find all CSV files in the folder
    files = list((CACHE_LOCAL_DIR / model_folder).glob("*.csv"))

    for file in files:
        df_res = pd.read_csv(file)
        model_name = file.stem
        model = model_name.replace("resps_", "")

        if model in MODELS_TO_EXCUDE:
            continue  # skip these models as newer versions exist
        
        # Manually count tokens for Deepseek before cleaning responses
        if model == "fw-deepseek-r1-0528":
            tokenizer=deepseek_tokenizer
            tokenizer_name="deepseek"
            #tokenizer=cl100k_encoder
            #tokenizer_name="cl100k"
            
            # Reasoning tokens  (use <think>...</think> tags)
            tokens_used_reasoning = df_res['resps'].apply(lambda x:
                extract_and_count_think(x, tokenizer=tokenizer, tokenizer_name=tokenizer_name)
            ).values
            # Recount completion tokens using same tokenizer
            tokens_used_completion_new = df_res['resps'].apply(lambda x:
                count_tokens(x, tokenizer=tokenizer, tokenizer_name=tokenizer_name)
            ).values
            # Update original completion tokens = new - reasoning
            tokens_used_completion = tokens_used_completion_new - tokens_used_reasoning

            # print original answer
            #print(f"#####\nOriginal response example:\n{df_res['resps'].values[0]}\n######\n")
        else:
            tokens_used_completion = df_res['TokensUsedCompletion'].values
            tokens_used_reasoning = df_res['TokensUsedReasoning'].values
        # Tokens used do not change
        tokens_used = df_res['TokensUsed'].values

        # process dataframe (assuming clean_resps and clean_answers are defined)
        df_res["resps"] = df_res["resps"].apply(clean_resps)
        df_res["answers"] = df_res["answers"].apply(clean_answers)

        correct = (df_res["resps"] == df_res["answers"]).values
        bad_format = df_res["resps"].isna().values

        # print examples if completion tokens > 1
        #if model == "fw-deepseek-r1-0528":
        #   print(f"Model: {model}, Language: {language}, Category: {category}, English Prompt: {english_prompt}")
        #   for idx in high_token_indices[:3]:  # print first 3 examples
        #       #print(f"Example index: {idx}")
        #       print(f"Response: {df_res['resps'].iloc[idx]}")
        #       print(f"Answer: {df_res['answers'].iloc[idx]}")
        #       print(f"Total Tokens Used: {df_res['TokensUsed'].iloc[idx]}")
        #       print(f"Completion Tokens Used Original: {df_res['TokensUsedCompletion'].iloc[idx]}")
        #       print(f"Reasoning Tokens Used: {df_res['TokensUsedReasoning'].iloc[idx]}")
        #       print("\nAfter deepseek adaptation:")
        #       print(f"Total Tokens Used Adapted: {tokens_used[idx]}")
        #       print(f"Completion Tokens Used Adapted: {tokens_used_completion[idx]}")
        #       print(f"Reasoning Tokens Used: {tokens_used_reasoning[idx]}\n")
        #       break


        # Append a dictionary for this specific result directly to the list
        row = {
            "category": category,
            "model": model,
            "language": language,
            "english_prompt": english_prompt,
            "correct": clean_list(correct.tolist()),
            "bad_format": clean_list(bad_format.tolist()),
            "Latency": clean_list(df_res['Latency'].values.tolist()),
            "TokensUsed": clean_list(tokens_used.tolist()),
            "TokensUsedCompletion": clean_list(tokens_used_completion.tolist()),
            "TokensUsedReasoning": clean_list(tokens_used_reasoning.tolist())
        }
        acc_rows_detailed.append(row)
        
# Convert the list of rows to a DataFrame
acc_df_detailed = pd.DataFrame(acc_rows_detailed)

# add metadata from models_csv
#acc_df_detailed['display_name'] = acc_df_detailed['model'].apply(lambda x: models_csv.loc[models_csv['model_id'] == x, 'display_name'].values[0] if x in models_csv['model_id'].values else x)
acc_df_detailed['provider'] = acc_df_detailed['model'].apply(lambda x: models_csv.loc[models_csv['model_id'] == x, 'provider'].values[0] if x in models_csv['model_id'].values else 'Unknown')
print(acc_df_detailed.shape)
acc_df_detailed.head()


# %%
# save acc_df_detailed to csv
acc_df_detailed.to_csv(FIGS_DIR / "cdpk_multilingual_model_performance_detailed.csv", index=False)


# %%
# Check code
deepseek_df = acc_df_detailed[acc_df_detailed['model'] == 'fw-deepseek-r1-0528']

# plot all the values of TokensUsedCompletion where english_prompt is True
deepseek_df_ep = deepseek_df[deepseek_df['english_prompt'] == True].reset_index(drop=True)
deepseek_df_ep_exploded = deepseek_df_ep.explode(
    ['correct', 'bad_format', 'Latency', 'TokensUsed', 'TokensUsedCompletion', 'TokensUsedReasoning']
).reset_index(drop=True)
print(deepseek_df_ep_exploded.shape)
print(deepseek_df_ep_exploded['english_prompt'].value_counts())
deepseek_df_ep_exploded.head(2)

# %%
# check code
# plot 3 countplots: tokens used, tokens used completion, tokens used reasoning

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'DeepSeek R1 Token Usage Distribution (English Prompt) - {tokenizer_name} tokenizer', fontsize=16, y=1.03)

# Tokens Used
sns.histplot(data=deepseek_df_ep_exploded,x='TokensUsed',bins=30,color='#1f77b4',ax=axs[0])
axs[0].set_title('Total Tokens Used', fontsize=12)
axs[0].set_xlabel('Tokens Used', fontsize=10)
axs[0].set_ylabel('Count', fontsize=10)
axs[0].grid(True, linestyle='--', alpha=0.6)

# Tokens Used Completion
sns.histplot(data=deepseek_df_ep_exploded,x='TokensUsedCompletion',bins=30,color='#ff7f0e',ax=axs[1])
axs[1].set_title('Completion Tokens Used', fontsize=12)
axs[1].set_xlabel('Completion Tokens Used', fontsize=10)
axs[1].set_ylabel('Count', fontsize=10)
axs[1].grid(True, linestyle='--', alpha=0.6)

# Tokens Used Reasoning
sns.histplot(data=deepseek_df_ep_exploded,x='TokensUsedReasoning',bins=30,color='#2ca02c',ax=axs[2])
axs[2].set_title('Reasoning Tokens Used', fontsize=12)
axs[2].set_xlabel('Reasoning Tokens Used', fontsize=10)
axs[2].set_ylabel('Count', fontsize=10)
axs[2].grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# %%
# compare how both tokenizer tokenize "<think> This is a test. </think>"
#test_text = "</think>"# Test </think>"
#deepseek_tokens = deepseek_tokenizer.encode(test_text, add_special_tokens=False) if deepseek_tokenizer else []
#cl100k_tokens = cl100k_encoder.encode(test_text)
#print(f"Test Text: {test_text}")
#print(f"DeepSeek Tokens ({len(deepseek_tokens)}): {deepseek_tokens}")
#print(f"cl100k Tokens ({len(cl100k_tokens)}): {cl100k_tokens}")

