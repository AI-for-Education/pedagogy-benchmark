# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import json
from pathlib import Path
from collections import defaultdict
import tiktoken

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
CACHE_LOCAL_DIR = DATA_DIR / "cache_local"  # kept for DeepSeek reasoning token fallback
OUTPUT_DIR = DATA_DIR / "results"
FAB_CONFIGS_DIR = ROOT / "fab-benchmarks-configs"

MODELS_TO_EXCLUDE = [
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-preview-09-2025",
    "gpt-5-2025-08-07-medium",
    "fw-deepseek-r1-0528",
]

CATEGORY_TO_PLOT = "Overall"  # options: "Science", "Literacy", "Creative arts", "Maths", "Social studies", "Technology", "General", "Overall"

# %%
# import useful files
models_csv = pd.read_csv(FAB_CONFIGS_DIR / "models.csv")
providers_csv = pd.read_csv(FAB_CONFIGS_DIR / "providers.csv")

# %%
# Helper: discover populated language subfolders in data/results/
def discover_language_folders(results_dir):
    """Scan results_dir for populated language subfolders.
    Returns list of dicts: {path, language, english_prompt}"""
    folders = []
    for subfolder in sorted(results_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        acc_files = list(subfolder.glob("cdpk_results_accuracy_*.csv"))
        if not acc_files:
            continue  # skip empty folders
        name = subfolder.name
        if name.endswith("_ep"):
            language = name[:-3]
            english_prompt = True
        else:
            language = name
            english_prompt = False
        folders.append({
            "path": subfolder,
            "language": language,
            "english_prompt": english_prompt
        })
    return folders

def get_models_from_full_csv(full_df):
    """Extract model names from full CSV column names (pred_[model] pattern)."""
    return [col[5:] for col in full_df.columns if col.startswith("pred_")]

language_folders = discover_language_folders(RESULTS_DIR)
print(f"Found {len(language_folders)} populated language folders:")
for f in language_folders:
    print(f"  {f['language']} (english_prompt={f['english_prompt']})")

# %%

# Language configuration
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
        "model_id": "deepseek-r1-0528-fp8",
        "size": "Large",
        "reasoning": True,
    },
    "GPT-5.2 (Medium)": {
        "model_id": "gpt-5.2-2025-12-11-medium",
        "size": "Large",
        "reasoning": True,
    },
    "Gemini-3 Pro": {
        "model_id": "gemini-3-pro-preview",
        "size": "Large",
        "reasoning": True,
    },
    "Gemini-3 Flash": {
        "model_id": "gemini-3-flash-preview",
        "size": "Medium",
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
# Build acc_df from accuracy + bad_format CSVs and latency from full CSV in results/

acc_rows = []

print(f"Processing {len(language_folders)} language folders for acc_df...")

for folder_info in language_folders:
    folder_path = folder_info["path"]
    language = folder_info["language"]
    english_prompt = folder_info["english_prompt"]

    # Find the 3 CSV files
    acc_file = list(folder_path.glob("cdpk_results_accuracy_*.csv"))[0]
    bf_file = list(folder_path.glob("cdpk_results_bad_format_*.csv"))[0]
    full_file = list(folder_path.glob("cdpk_results_full_*.csv"))[0]

    # Read accuracy and bad_format CSVs (model x category matrices)
    acc_csv = pd.read_csv(acc_file, index_col=0)
    bf_csv = pd.read_csv(bf_file, index_col=0)

    # Melt from wide to long format
    acc_melted = acc_csv.reset_index().melt(
        id_vars=[acc_csv.index.name or "index"],
        var_name="category", value_name="accuracy"
    ).rename(columns={acc_csv.index.name or "index": "model"})

    bf_melted = bf_csv.reset_index().melt(
        id_vars=[bf_csv.index.name or "index"],
        var_name="category", value_name="bad_format"
    ).rename(columns={bf_csv.index.name or "index": "model"})

    # Merge accuracy and bad_format
    merged = pd.merge(acc_melted, bf_melted, on=["model", "category"])

    # Compute latency from full CSV
    full_df = pd.read_csv(full_file)
    models = get_models_from_full_csv(full_df)

    latency_rows = []
    for model in models:
        lat_col = f"Latency_{model}"
        if lat_col not in full_df.columns:
            continue
        # Per-category latency
        for cat, group in full_df.groupby("category"):
            latency_vals = group[lat_col].dropna()
            latency_rows.append({
                "model": model,
                "category": cat,
                "Latency Mean": latency_vals.mean() if len(latency_vals) > 0 else np.nan,
                "Latency Median": latency_vals.median() if len(latency_vals) > 0 else np.nan,
            })
        # Overall latency
        latency_all = full_df[lat_col].dropna()
        latency_rows.append({
            "model": model,
            "category": "Overall",
            "Latency Mean": latency_all.mean() if len(latency_all) > 0 else np.nan,
            "Latency Median": latency_all.median() if len(latency_all) > 0 else np.nan,
        })

    latency_summary = pd.DataFrame(latency_rows)

    # Merge latency into accuracy/bad_format
    merged = pd.merge(merged, latency_summary, on=["model", "category"], how="left")

    # Add language and english_prompt
    merged["language"] = language
    merged["english_prompt"] = english_prompt

    # Filter out excluded models
    merged = merged[~merged["model"].isin(MODELS_TO_EXCLUDE)]

    acc_rows.append(merged)

    # For English, duplicate with english_prompt=True (questions are already in English)
    if language == "English" and not english_prompt:
        merged_ep = merged.copy()
        merged_ep["english_prompt"] = True
        acc_rows.append(merged_ep)

# Concatenate all language folders
acc_df = pd.concat(acc_rows, ignore_index=True)

# Add provider metadata
acc_df['provider'] = acc_df['model'].apply(
    lambda x: models_csv.loc[models_csv['model_id'] == x, 'provider'].values[0]
    if x in models_csv['model_id'].values else 'Unknown'
)

print("Accuracy DataFrame shape:", acc_df.shape)
print("Categories:", acc_df['category'].unique())
acc_df.head()

# %%
# save acc_df to csv
acc_df.to_csv(OUTPUT_DIR / "cdpk_multilingual_model_performance.csv", index=False)




# %%
# Build acc_df_detailed from full CSVs in results/, 
# extracting per-question correctness, bad_format, latency, 
# and token usage for each model and category

def clean_list(lst):
    """Replaces any NaN/nat with None in a list"""
    return [None if pd.isna(x) else x for x in lst]

acc_rows_detailed = []

print(f"Processing {len(language_folders)} language folders for acc_df_detailed...")

for folder_info in language_folders:
    folder_path = folder_info["path"]
    language = folder_info["language"]
    english_prompt = folder_info["english_prompt"]

    full_file = list(folder_path.glob("cdpk_results_full_*.csv"))[0]
    full_df = pd.read_csv(full_file)
    models = get_models_from_full_csv(full_df)

    for model in models:
        if model in MODELS_TO_EXCLUDE:
            continue

        pred_col = f"pred_{model}"
        lat_col = f"Latency_{model}"
        tok_col = f"TokensUsed_{model}"
        tok_comp_col = f"TokensUsedCompletion_{model}"
        tok_reas_col = f"TokensUsedReasoning_{model}"

        # Check required columns exist
        if pred_col not in full_df.columns:
            continue

        # Per-category detailed data
        for cat, group in full_df.groupby("category"):
            preds = group[pred_col]
            correct_answers = group["correct_answer"]

            correct = (preds == correct_answers).values
            bad_format = preds.isna().values
            latency = group[lat_col].values if lat_col in group.columns else np.full(len(group), np.nan)
            tokens_used = group[tok_col].values if tok_col in group.columns else np.full(len(group), np.nan)
            tokens_completion = group[tok_comp_col].values if tok_comp_col in group.columns else np.full(len(group), np.nan)
            tokens_reasoning = group[tok_reas_col].values if tok_reas_col in group.columns else np.full(len(group), np.nan)

            row = {
                "category": cat,
                "model": model,
                "language": language,
                "english_prompt": english_prompt,
                "correct": clean_list(correct.tolist()),
                "bad_format": clean_list(bad_format.tolist()),
                "Latency": clean_list(latency.tolist()),
                "TokensUsed": clean_list(tokens_used.tolist()),
                "TokensUsedCompletion": clean_list(tokens_completion.tolist()),
                "TokensUsedReasoning": clean_list(tokens_reasoning.tolist()),
            }
            acc_rows_detailed.append(row)

        # Overall (all categories combined)
        preds_all = full_df[pred_col]
        correct_answers_all = full_df["correct_answer"]

        correct_all = (preds_all == correct_answers_all).values
        bad_format_all = preds_all.isna().values
        latency_all = full_df[lat_col].values if lat_col in full_df.columns else np.full(len(full_df), np.nan)
        tokens_used_all = full_df[tok_col].values if tok_col in full_df.columns else np.full(len(full_df), np.nan)
        tokens_comp_all = full_df[tok_comp_col].values if tok_comp_col in full_df.columns else np.full(len(full_df), np.nan)
        tokens_reas_all = full_df[tok_reas_col].values if tok_reas_col in full_df.columns else np.full(len(full_df), np.nan)

        row = {
            "category": "Overall",
            "model": model,
            "language": language,
            "english_prompt": english_prompt,
            "correct": clean_list(correct_all.tolist()),
            "bad_format": clean_list(bad_format_all.tolist()),
            "Latency": clean_list(latency_all.tolist()),
            "TokensUsed": clean_list(tokens_used_all.tolist()),
            "TokensUsedCompletion": clean_list(tokens_comp_all.tolist()),
            "TokensUsedReasoning": clean_list(tokens_reas_all.tolist()),
        }
        acc_rows_detailed.append(row)

# Convert the list of rows to a DataFrame
acc_df_detailed = pd.DataFrame(acc_rows_detailed)

# For English, duplicate with english_prompt=True (questions are already in English)
english_rows = acc_df_detailed[(acc_df_detailed['language'] == 'English') & (acc_df_detailed['english_prompt'] == False)].copy()
if len(english_rows) > 0:
    english_rows['english_prompt'] = True
    acc_df_detailed = pd.concat([acc_df_detailed, english_rows], ignore_index=True)

# Add provider metadata
acc_df_detailed['provider'] = acc_df_detailed['model'].apply(
    lambda x: models_csv.loc[models_csv['model_id'] == x, 'provider'].values[0]
    if x in models_csv['model_id'].values else 'Unknown'
)

print("Detailed DataFrame shape:", acc_df_detailed.shape)
print("Categories:", acc_df_detailed['category'].unique())
acc_df_detailed.head()

# %%
# save acc_df_detailed to csv
acc_df_detailed.to_csv(OUTPUT_DIR / "cdpk_multilingual_model_performance_detailed.csv", index=False)



# %%
