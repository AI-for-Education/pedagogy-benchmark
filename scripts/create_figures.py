# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import json
import ast
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "data" / "results"
FAB_CONFIGS_DIR = ROOT / "fab-benchmarks-configs"

CATEGORY_TO_PLOT = "Overall"  # options: "Science", "Literacy", "Creative arts", "Maths", "Social studies", "Technology", "General", "Overall"

# %%
# import useful files
models_csv = pd.read_csv(FAB_CONFIGS_DIR / "models.csv")
providers_csv = pd.read_csv(FAB_CONFIGS_DIR / "providers.csv")

# %%
################################
# Import results data
################################
# Accuracy dataframe
acc_df_all = pd.read_csv(RESULTS_DIR / "cdpk_multilingual_model_performance.csv")
print("Accuracy df shape (all categories):", acc_df_all.shape)
print("Available categories:", acc_df_all['category'].unique())

# Filter to selected category
acc_df = acc_df_all[acc_df_all['category'] == CATEGORY_TO_PLOT].reset_index(drop=True)
print(f"Accuracy df shape (category={CATEGORY_TO_PLOT}):", acc_df.shape)
acc_df.head()

# %%
# Results dataframe detailed
acc_df_detailed_all = pd.read_csv(RESULTS_DIR / "cdpk_multilingual_model_performance_detailed.csv")
cols_to_fix = ['correct', 'bad_format', 'Latency', 'TokensUsed', 'TokensUsedCompletion', 'TokensUsedReasoning']
# 2. Convert them from String -> List
for col in cols_to_fix:
    acc_df_detailed_all[col] = acc_df_detailed_all[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

acc_df_detailed = acc_df_detailed_all[acc_df_detailed_all['category'] == CATEGORY_TO_PLOT].reset_index(drop=True)
print(f"Detailed Accuracy df shape (category={CATEGORY_TO_PLOT}):", acc_df_detailed.shape)
acc_df_detailed.head(2)

# %%
# Test whether all dataframes imported have same models
if set(acc_df['model'].unique()) == set(acc_df_detailed['model'].unique()):
    print("✅ All dataframes have the same set of models.")
else:
    print("❌ Dataframes have different sets of models.")
    acc_models = set(acc_df['model'].unique())
    detailed_models = set(acc_df_detailed['model'].unique())

    diff_acc_detailed = acc_models.symmetric_difference(detailed_models)

    if diff_acc_detailed:
        print(" - Models differing between Accuracy and Detailed Accuracy dataframes:", diff_acc_detailed)
    

# %%
# Add display names to dataframes
new_models_names = {
    'claude-sonnet-4-5-20250929': 'Claude Sonnet 4.5',
    'deepseek-r1-0528-fp8': 'Deepseek R1',
    'gemini-2.5-flash-lite-preview-09-2025': 'Gemini-2.5 Flash-Lite',
    #'gemini-2.5-flash-preview-09-2025': 'Gemini-2.5 Flash',
    'gemini-3-flash-preview': 'Gemini-3 Flash',
    #'gemini-2.5-pro-preview-06-05': 'Gemini-2.5 Pro',
    "gemini-3-pro-preview": "Gemini-3 Pro",
    'gemma-3-27b': 'Gemma-3 27B',
    'gemma-3-4b-it': 'Gemma-3 4B',
    #'gpt-5-2025-08-07-medium': 'GPT-5',
    "gpt-5.2-2025-12-11-medium": "GPT-5.2 (Medium)",
    'o4-mini-2025-04-16': 'o4-Mini',
    'qwen-3-32b': 'Qwen-3 32B',
    #'hf-gemma-3-1b-it': 'Gemma-3 1B',
}

old_new_models_mapping = {
    'claude-sonnet-4-5-20250929': 'claude-sonnet-4-6',
    'deepseek-r1-0528-fp8': 'deepseek-r1-0528-fp8',
    'gemini-2.5-flash-lite-preview-09-2025': 'gemini-3.1-flash-lite-preview',
    #'gemini-2.5-flash-preview-09-2025': 'Gemini-2.5 Flash',
    'gemini-3-flash-preview': 'gemini-3-flash-preview',
    #'gemini-2.5-pro-preview-06-05': 'Gemini-2.5 Pro',
    "gemini-3-pro-preview": 'gemini-3.1-pro-preview',
    'gemma-3-27b': 'gemma-4-31b-it',
    'gemma-3-4b-it': 'hf-gemma-4-e4b-it-gguf-bf16',
    #'gpt-5-2025-08-07-medium': 'GPT-5',
    "gpt-5.2-2025-12-11-medium": 'gpt-5.4-2026-03-05-medium',
    'o4-mini-2025-04-16': 'o4-mini-2025-04-16',
    'qwen-3-32b': 'qwen3.5-27b'
}

new_models_display_names = {
    'claude-sonnet-4-6': 'Claude Sonnet 4.6',
    'deepseek-r1-0528-fp8': 'Deepseek R1',
    'gemini-3.1-flash-lite-preview': 'Gemini-3.1 Flash-Lite',
    'gemini-3-flash-preview': 'Gemini-3 Flash',
    'gemini-3.1-pro-preview': 'Gemini-3.1 Pro',
    'gemma-4-31b-it': 'Gemma-4 31B',
    'hf-gemma-4-e4b-it-gguf-bf16': 'Gemma-4 E4B',
    'gpt-5.4-2026-03-05-medium': 'GPT-5.4 (Medium)',
    'o4-mini-2025-04-16': 'o4-Mini',
    'qwen3.5-27b': 'Qwen-3.5 27B',
}

acc_df['display_name'] = acc_df['model'].map(new_models_names)
acc_df_detailed['display_name'] = acc_df_detailed['model'].map(new_models_names)

# %%
#------------------------------------------------------------------
# 2. THE TEST FUNCTION
# ------------------------------------------------------------------
def test_explode_readiness(df, cols_to_explode):
    print(f"--- Testing columns: {cols_to_explode} ---\n")
    
    # CHECK 1: Are they all lists?
    print("1. CHECKING TYPES...")
    all_lists = True
    for col in cols_to_explode:
        # Check if every cell in this column is a list
        is_list = df[col].apply(lambda x: isinstance(x, list))
        if not is_list.all():
            print(f"❌ FAIL: Column '{col}' contains non-list values.")
            print(df.loc[~is_list, col].head())
            all_lists = False
        else:
            print(f"✅ PASS: Column '{col}' contains only lists.")
    
    if not all_lists:
        print("\nSTOPPING: Fix types before checking lengths.\n")
        return

    # CHECK 2: Do lengths match row-by-row?
    print("\n2. CHECKING LENGTH CONSISTENCY...")
    
    # Calculate lengths for all target columns
    lengths = df[cols_to_explode].applymap(len)
    
    # Check if all columns in a row have equal value (variance is 0)
    # We transpose because .var() works column-wise by default
    mismatches = lengths.var(axis=1) > 0
    
    if mismatches.any():
        print(f"❌ FAIL: Found {mismatches.sum()} rows with mismatched lengths.")
        print("\nHere are the problematic rows (showing lengths):")
        print(lengths[mismatches])
        print("\nFix these rows before exploding!")
    else:
        print("✅ PASS: All rows have matching list lengths. Ready to explode!")

# ------------------------------------------------------------------
# 3. RUN THE TEST
# ------------------------------------------------------------------
cols = ['correct', 'bad_format', 'Latency']
test_explode_readiness(acc_df_detailed, cols)

# %%
# Accuracy plots
acc_df_langp = acc_df[acc_df['english_prompt'] == False]
acc_df_ep = acc_df[acc_df['english_prompt'] == True]

print("Accuracy df (Non-English Prompt):", acc_df_langp.shape)
print("Accuracy df (English Prompt):", acc_df_ep.shape)


# %%

# Accuracy plots
# Load data from each language folder
language_list = [
    "Luganda",
    "Nyankore",
    "Swahili",
    "Hausa",
    "Yoruba",
    "English",
    "Pashto",
    "Dari",
    "Arabic"
]

language_speakers_dict = {
  "English": 1457e6,
  "Swahili": 97e6,
  "Luganda": 6e6,
  "Hausa": 94e6,
  "Yoruba": 50e6,
  "Nyankore": 3e6,
  "Arabic": 335e6,
  "Pashto": 55e6,
  "Dari": 30e6
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
                 vmax=None,
                 save_fig=False,
                 languages=None,
                 ):

    # filter columns if a language subset is requested
    if languages is not None:
        pivot_df = pivot_df[[c for c in languages if c in pivot_df.columns]]

    # order by number of speakers: reorder index in descending order of speakers
    pivot_df = pivot_df[sorted(pivot_df.columns, key=lambda x: language_speakers_dict.get(x, 0), reverse=True)]

    # --- 2. Create the heatmap ---
    plt.figure(figsize=(12, 6))

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

    ax.set_title(title, fontsize=18, pad=42) 
    
    # 2. Add the explanatory subtitle
    # transform=ax.transAxes uses relative coordinates (0 to 1)
    # y=1.2 places it just above the grey numbers but below the main title
    ax.text(0.5, 1.08, 
            "Languages ordered by number of estimated speakers (grey values)", 
            transform=ax.transAxes, 
            ha='center', va='bottom', 
            fontsize=13, color='grey', style='italic') # italic looks nice for subtitles
    # --
    ax.set_xlabel('Language', fontsize=16)
    ax.set_ylabel('Model', fontsize=16)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=13)

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

    if save_fig:
        # bbox_inches='tight' fits the bounding box around all artists (text, titles, etc.)
        plt.savefig(RESULTS_DIR / "model_performance_heatmap.svg", format='svg', dpi=300, bbox_inches='tight')



LANGUAGE_GROUPS_TILES = [
    ("English",                                ["English"]),
    ("Asian-originating languages",    ["Arabic", "Pashto", "Dari"]),
    ("African-originating languages",  ["Swahili", "Hausa", "Yoruba",
                                                "Luganda", "Nyankore"]),
]


def _plot_grouped_language_tiles(values_by_lang,
                                 language_speakers_dict,
                                 title=None,
                                 vmin=35,
                                 vmax=85,
                                 reverse_cmap=False,
                                 save_fig=False,
                                 fname="grouped_language_tiles.svg",
                                 colorbarlabel="Accuracy (%)",
                                 row_label=None):
    """One-row tile plot: speaker count above, accuracy %, language name below.

    Tiles are split into LANGUAGE_GROUPS_TILES (English | Asian-originating
    | African-originating). Single-language groups (English) get no caption —
    the tile's language name is enough. Multi-language groups get an italic
    caption centred under their tiles. Colors follow the same RdYlGn ramp
    as the regular heatmaps.
    """
    cmap = plt.get_cmap("RdYlGn_r" if reverse_cmap else "RdYlGn")
    norm = Normalize(vmin=vmin, vmax=vmax)

    resolved_groups = []
    for label, langs in LANGUAGE_GROUPS_TILES:
        present = [l for l in langs if l in values_by_lang
                   and pd.notna(values_by_lang[l])]
        # Sort languages within each region by speaker count, descending —
        # matches the subtitle "ordered by number of estimated speakers
        # within each region".
        present.sort(key=lambda l: language_speakers_dict.get(l, 0),
                     reverse=True)
        if present:
            resolved_groups.append((label, present))
    if not resolved_groups:
        print("[WARN] grouped_language_tiles: no data to plot.")
        return

    total_tiles = sum(len(langs) for _, langs in resolved_groups)
    gap = 0.25  # gap (in tile-widths) between groups
    n_gaps = max(len(resolved_groups) - 1, 0)
    fig_width = max(8, total_tiles * 1.4 + n_gaps * gap)
    fig, ax = plt.subplots(figsize=(fig_width, 3.0))

    cursor = 0.0
    for group_label, langs in resolved_groups:
        group_x_start = cursor
        for j, lang in enumerate(langs):
            x = cursor + j
            val = values_by_lang[lang]
            color = cmap(norm(val))
            ax.add_patch(mpatches.Rectangle((x, 0), 1, 1,
                                            facecolor=color,
                                            edgecolor="white", linewidth=2))
            speakers_m = language_speakers_dict.get(lang, 0) / 1e6
            spk_text = (f"{speakers_m:,.0f}m" if speakers_m >= 1
                        else f"{speakers_m:.1f}m")
            ax.text(x + 0.5, 1.1, spk_text,
                    ha="center", va="bottom",
                    fontsize=11, color="#888888")
            ax.text(x + 0.5, 0.5, f"{val:.1f}",
                    ha="center", va="center",
                    fontsize=16,)
            # Short tick mark below the tile centre — mirrors the heatmap
            # axis ticks shown in the reference figure.
            ax.plot([x + 0.5, x + 0.5], [-0.02, -0.10],
                    color="#333333", linewidth=1.0, solid_capstyle="butt")
            ax.text(x + 0.5, -0.18, lang,
                    ha="center", va="top",
                    fontsize=12, color="#333333")
        group_x_end = cursor + len(langs)
        if len(langs) > 1:
            ax.text((group_x_start + group_x_end) / 2, -0.55, group_label,
                    ha="center", va="top",
                    fontsize=11, color="#555555", style="italic")
        cursor = group_x_end + gap

    ax.set_xlim(-0.6, cursor - gap + 0.2)
    ax.set_ylim(-0.85, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, pad=18)
    # Row label on the left of the tile row — mirrors the heatmap path's
    # y-tick label (e.g. "Average Model"), rotated 90° to read bottom-to-top.
    if row_label is not None:
        ax.text(-0.35, 0.5, str(row_label),
                rotation=90, ha="center", va="center",
                fontsize=10, color="#333333")
        # Short tick mark immediately left of the row, at its vertical centre.
        ax.plot([-0.02, -0.10], [0.5, 0.5],
                color="#333333", linewidth=1.0, solid_capstyle="butt")
    # Subtitle: same italic-grey treatment as the regular heatmap path.
    ax.text(0.5, 1.01,
            "Languages ordered by number of estimated speakers "
            "(grey values), within each region",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=10, color="grey", style="italic")

    # Colorbar mirroring the RdYlGn ramp used for tile fills.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=18)
    cbar.set_label(colorbarlabel, fontsize=11, color="#444444")
    cbar.ax.tick_params(labelsize=10, colors="#555555")
    # Drop the black box around the colorbar — match the clean reference.
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    if save_fig:
        plt.savefig(RESULTS_DIR / fname, format="svg", bbox_inches="tight")


def plot_avg_heatmap(pivot_df_avg,
                     language_speakers_dict,
                     title='TODO',
                     colorbarlabel='TODO',
                     reverse_cmap=False,
                     vmin=None,
                     vmax=None,
                     save_fig=False,
                     languages=None,
                     group_lang_regions=False,
                     ):

    # filter columns if a language subset is requested
    if languages is not None:
        pivot_df_avg = pivot_df_avg[[c for c in languages if c in pivot_df_avg.columns]]

    # When grouping by region, the language order is dictated by
    # LANGUAGE_GROUPS_TILES (English → Asian → African). Otherwise default to
    # speaker-count descending order as before.
    if group_lang_regions:
        ordered = [l for _, langs in LANGUAGE_GROUPS_TILES for l in langs
                   if l in pivot_df_avg.columns]
        pivot_df_avg = pivot_df_avg[ordered]
        _plot_grouped_language_tiles(
            values_by_lang=pivot_df_avg.iloc[0].to_dict(),
            language_speakers_dict=language_speakers_dict,
            title=title,
            vmin=vmin if vmin is not None else 0,
            vmax=vmax if vmax is not None else 100,
            reverse_cmap=reverse_cmap,
            save_fig=save_fig,
            colorbarlabel=colorbarlabel,
            row_label=pivot_df_avg.index[0],
        )
        return

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

    ax.set_title(title, fontsize=12, pad=35) 
    
    # 2. Add the explanatory subtitle
    # transform=ax.transAxes uses relative coordinates (0 to 1)
    # y=1.2 places it just above the grey numbers but below the main title
    ax.text(0.5, 1.35, 
            "Languages ordered by number of estimated speakers (grey values)", 
            transform=ax.transAxes, 
            ha='center', va='bottom', 
            fontsize=9, color='grey', style='italic') # italic looks nice for subtitles
    # --
    
    ax.set_xlabel('Language', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    # x ticks to be horizontal and smaller
    ax.tick_params(axis='x', labelsize=10, rotation=0)

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

    if save_fig:
        # bbox_inches='tight' fits the bounding box around all artists (text, titles, etc.)
        plt.savefig(RESULTS_DIR / "average_model_performance_heatmap.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(RESULTS_DIR / "average_model_performance_heatmap.svg", format='svg', dpi=300, bbox_inches='tight')

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
    #"GPT-5": {
    #    "model_id": "gpt-5-2025-08-07-medium",
    #    "size": "Large",
    #    "reasoning": True,
    #},
    "GPT-5.2": {
        "model_id": "gpt-5.2-2025-12-11-medium",
        "size": "Large",
        "reasoning": True,
    },
    #"Gemini-2.5 Pro": {
    #    "model_id": "gemini-2.5-pro-preview-06-05",
    #    "size": "Large",
    #    "reasoning": True,
    #},
    "Gemini-3 Pro": {
        "model_id": "gemini-3-pro-preview",
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
    "Gemini-3 Flash": {
        "model_id": "gemini-3-flash-preview",
        "size": "Medium",
        "reasoning": True,
    },
    "Gemma-3 4B": {
        "model_id": "gemma-3-4b-it",
        "size": "Small",
        "reasoning": False,
    },
}

# %%
# Accuracy Heatmaps
pivot_df_acc = acc_df_langp.pivot_table(
    index='display_name',
    columns='language',
    values='accuracy'
)

pivot_df_acc_ep = acc_df_ep.pivot_table(
    index='display_name',
    columns='language',
    values='accuracy'
)

# order index based on size in MODELS_METADATA_MAPPING
# order: "Large", "Medium", "Small" and score on English withing each size category if necessary
#pivot_df_acc['size cat'] = pivot_df_acc.index.map(lambda x: MODELS_METADATA_MAPPING.get(x, {}).get('size', 'Unknown'))
#pivot_df_acc = pivot_df_acc.sort_values(by=['size cat', 'English'], ascending=[True, False])
pivot_df_acc = pivot_df_acc.sort_values(by='English', ascending=False)
#pivot_df_acc = pivot_df_acc.drop(columns=['size cat'])

#pivot_df_acc_ep['size cat'] = pivot_df_acc_ep.index.map(lambda x: MODELS_METADATA_MAPPING.get(x, {}).get('size', 'Unknown'))
#pivot_df_acc_ep = pivot_df_acc_ep.sort_values(by=['size cat', 'English'], ascending=[True, False])
pivot_df_acc_ep = pivot_df_acc_ep.sort_values(by='English', ascending=False)
#pivot_df_acc_ep = pivot_df_acc_ep.drop(columns=['size cat'])

order_models_acc_english = pivot_df_acc.sort_values(by='English', ascending=False).index.tolist()

plot_heatmap(pivot_df_acc, 
             language_speakers_dict, 
             title='AI Model Performance Across Languages\n' \
    'Values represent accuracy (%)',
             colorbarlabel='Accuracy (%)',
             reverse_cmap=False,
             vmin=35,
             vmax=85,
             #languages=["English", "Dari", "Pashto", "Arabic"]
)
plot_heatmap(pivot_df_acc_ep, 
             language_speakers_dict, 
             title='AI Model Performance Across Languages - January 2026\n' \
    'Values represent accuracy (%)',
             colorbarlabel='Accuracy (%)',
             reverse_cmap=False,
             vmin=35,
             vmax=85,
             save_fig=True,
             #languages=["English", "Dari", "Pashto", "Arabic"]
)


# %%
# Plot averaged over models
pivot_df_acc_avg = pivot_df_acc.copy()
# add avg in index, average over all models
pivot_df_acc_avg.loc['Average Model'] = pivot_df_acc_avg.mean(axis=0)

pivot_df_acc_ep_avg = pivot_df_acc_ep.copy()
# add avg in index, average over all models
pivot_df_acc_ep_avg.loc['Average Model'] = pivot_df_acc_ep_avg.mean(axis=0)

plot_avg_heatmap(pivot_df_acc_avg,
                language_speakers_dict, 
                title='Average AI Model Performance Across Languages\n' \
        'Values represent accuracy (%)',
                colorbarlabel='Accuracy (%)',
                reverse_cmap=False,
                vmin=35,
                vmax=85,
                save_fig=True,
                #languages=["English", "Dari", "Pashto", "Arabic"]
                )

plot_avg_heatmap(pivot_df_acc_ep_avg,
                language_speakers_dict, 
                title='Average AI Model Performance Across Languages\n' \
        'Values represent accuracy (%)',
                colorbarlabel='Accuracy (%)',
                reverse_cmap=False,
                vmin=35,
                vmax=85,
                save_fig=True,
                #languages=["English", "Dari", "Pashto", "Arabic"]
                )

# %%
# NEW MODELS HEATMAP (only those models that have data across all selected languages)

# Build mapping from old display_name to new display_name
_old_display_to_new_display = {}
for old_id, new_id in old_new_models_mapping.items():
    old_display = new_models_names.get(old_id)
    new_display = new_models_display_names.get(new_id)
    if old_display is not None and new_display is not None:
        _old_display_to_new_display[old_display] = new_display

# Filter acc_df_ep for new model IDs
new_model_ids = list(old_new_models_mapping.values())
acc_df_ep_new = acc_df_ep[acc_df_ep['model'].isin(new_model_ids)].copy()
acc_df_ep_new['display_name'] = acc_df_ep_new['model'].map(new_models_display_names)

# Pivot to heatmap format
pivot_df_acc_ep_new = acc_df_ep_new.pivot_table(
    index='display_name',
    columns='language',
    values='accuracy'
)

# Reorder rows to match old model y-axis order via old_new_models_mapping
ordered_new_index = [_old_display_to_new_display[name] for name in pivot_df_acc_ep.index if name in _old_display_to_new_display]
pivot_df_acc_ep_new = pivot_df_acc_ep_new.reindex(ordered_new_index)

plot_heatmap(pivot_df_acc_ep_new,
             language_speakers_dict,
             title='AI Model Performance Across Languages - April 2026\n'
                   'Values represent accuracy (%)',
             colorbarlabel='Accuracy (%)',
             reverse_cmap=False,
             vmin=35,
             vmax=85,
)

# Average heatmap for new models
pivot_df_acc_ep_new_avg = pivot_df_acc_ep_new.copy()
pivot_df_acc_ep_new_avg.loc['Average Model'] = pivot_df_acc_ep_new_avg.mean(axis=0)

plot_avg_heatmap(pivot_df_acc_ep_new_avg,
                language_speakers_dict,
                title='Average AI Model Performance Across Languages (New Models)\n'
                      'Values represent accuracy (%)',
                colorbarlabel='Accuracy (%)',
                reverse_cmap=False,
                vmin=35,
                vmax=85,
)

# %%
# Heatmap of common models across a selected set of languages
COMMON_LANGUAGES = ["English", "Dari", "Pashto", "Arabic"]

# Keep only models (rows) that have data for ALL selected languages
_cols_present = [c for c in COMMON_LANGUAGES if c in pivot_df_acc_ep.columns]
pivot_df_acc_ep_common = pivot_df_acc_ep.dropna(subset=_cols_present, how='any')
pivot_df_acc_ep_common = pivot_df_acc_ep_common[_cols_present]
pivot_df_acc_ep_common = pivot_df_acc_ep_common.sort_values(by='English', ascending=False)
print(f"Models common across {COMMON_LANGUAGES} ({len(pivot_df_acc_ep_common)} models):", pivot_df_acc_ep_common.index.tolist())

plot_heatmap(pivot_df_acc_ep_common,
             language_speakers_dict,
             title='AI Model Performance — Common Models Across Languages\n'
                   'Values represent accuracy (%)',
             colorbarlabel='Accuracy (%)',
             reverse_cmap=False,
             vmin=35,
             vmax=85,
)

# Average heatmap across those common models
pivot_df_acc_ep_common_avg = pivot_df_acc_ep_common.copy()
pivot_df_acc_ep_common_avg.loc['Average Model'] = pivot_df_acc_ep_common_avg.mean(axis=0)

plot_avg_heatmap(pivot_df_acc_ep_common_avg,
                 language_speakers_dict,
                 title='Average AI Model Performance — Common Models Across Languages\n'
                       'Values represent accuracy (%)',
                 colorbarlabel='Accuracy (%)',
                 reverse_cmap=False,
                 vmin=35,
                 vmax=85,
)


# %%
# Average heatmap across ALL models in `acc_df_ep` (every model run with an
# English prompt). Each cell = mean accuracy across all models for
# that language.
#
# Filter: keep only models that have an accuracy value for every language
# in the data. Models with partial coverage are dropped so the per-language
# averages are all computed over the same model population (otherwise a
# language run by, say, only the strongest models would look artificially
# better than English).
all_languages = sorted(acc_df_ep['language'].unique())
n_languages = len(all_languages)

coverage = (
    acc_df_ep.dropna(subset=['accuracy'])
             .groupby('model')['language']
             .nunique()
)
fully_covered_models = coverage[coverage == n_languages].index.tolist()
dropped_models = sorted(set(coverage.index) - set(fully_covered_models))

acc_df_ep_full = acc_df_ep[acc_df_ep['model'].isin(fully_covered_models)]

print(f"[INFO] Average across all-language-covered models: "
      f"keeping {len(fully_covered_models)}/{len(coverage)} models "
      f"(needed coverage of all {n_languages} languages: {all_languages}).")
if dropped_models:
    print(f"[INFO] Dropped {len(dropped_models)} models with partial coverage: "
          f"{dropped_models}")

pivot_df_acc_ep_avg_all = (
    acc_df_ep_full
    .groupby('language', as_index=True)['accuracy']
    .mean()
    .to_frame()
    .T
)
pivot_df_acc_ep_avg_all.index = ['Average Model']

# do median
pivot_df_acc_ep_avg_all_median = (
    acc_df_ep_full
    .groupby('language', as_index=True)['accuracy']
    .median()
    .to_frame()
    .T
)
pivot_df_acc_ep_avg_all_median.index = ['Average Model']

# Sanity-check: should be one row, columns = the languages present in the data.
print(f"Avg-all pivot shape: {pivot_df_acc_ep_avg_all.shape}, "
      f"languages: {list(pivot_df_acc_ep_avg_all.columns)}")
print(f"Average accuracy per language across "
      f"{len(fully_covered_models)} fully-covered models:")
print(pivot_df_acc_ep_avg_all.round(1))

# %%
plot_avg_heatmap(pivot_df_acc_ep_avg_all,
                    language_speakers_dict,
                    title='Average AI Model Performance Across All Languages\n'
                        'Values represent accuracy (%)',
                    colorbarlabel='Accuracy (%)',
                    reverse_cmap=False,
                    vmin=35,
                    vmax=85,
                    save_fig=True,
                    #languages=["English", "Swahili", "Hausa", "Yoruba", "Luganda", "Nyankore"]
                    )

plot_avg_heatmap(pivot_df_acc_ep_avg_all_median,
                    language_speakers_dict,
                    title='Median AI Model Performance Across All Languages\n'
                        'Values represent accuracy (%)',
                    colorbarlabel='Accuracy (%)',
                    reverse_cmap=False,
                    vmin=35,
                    vmax=85,
                    save_fig=False,
                    #languages=["English", "Swahili", "Hausa", "Yoruba", "Luganda", "Nyankore"]
                    )

# %%
# Grouped-tile variant: English | Asian-originating | African-originating.
# Mean
plot_avg_heatmap(pivot_df_acc_ep_avg_all,
                 language_speakers_dict,
                 title='Average AI Model Performance Across Languages\n'
                        'Values represent accuracy (%)',
                 colorbarlabel='Accuracy (%)',
                 reverse_cmap=False,
                 vmin=35,
                 vmax=85,
                 save_fig=True,
                 group_lang_regions=True,
                 )


# Median
plot_avg_heatmap(pivot_df_acc_ep_avg_all_median,
                 language_speakers_dict,
                 title='Median AI Model Performance Across Languages',
                 colorbarlabel='Accuracy (%)',
                 reverse_cmap=False,
                 vmin=35,
                 vmax=85,
                 save_fig=False,
                 group_lang_regions=True,
                 )



# %%
# Bad format Heatmaps
pivot_df_bf = acc_df_langp.pivot_table(
    index='display_name',
    columns='language',
    values='bad_format'
)
pivot_df_bf_ep = acc_df_ep.pivot_table(
    index='display_name',
    columns='language',
    values='bad_format'
)

# order index based on size in MODELS_METADATA_MAPPING
# order: "Large", "Medium", "Small" and score on English withing each size category if necessary
#pivot_df_bf['size cat'] = pivot_df_bf.index.map(lambda x: MODELS_METADATA_MAPPING.get(x, {}).get('size', 'Unknown'))
#pivot_df_bf = pivot_df_bf.sort_values(by=['size cat', 'English'], ascending=[True, True])
#pivot_df_bf = pivot_df_bf.drop(columns=['size cat'])

#pivot_df_bf_ep['size cat'] = pivot_df_bf_ep.index.map(lambda x: MODELS_METADATA_MAPPING.get(x, {}).get('size', 'Unknown'))
#pivot_df_bf_ep = pivot_df_bf_ep.sort_values(by=['size cat', 'English'], ascending=[True, True])
#pivot_df_bf_ep = pivot_df_bf_ep.drop(columns=['size cat'])

# reindex by order_models_acc_english
pivot_df_bf = pivot_df_bf.reindex(order_models_acc_english)
pivot_df_bf_ep = pivot_df_bf_ep.reindex(order_models_acc_english)

plot_heatmap(pivot_df_bf, 
             language_speakers_dict, 
             title='Comparative Analysis of Model Response Formatting Across Languages\n' \
    'Values represent % badly formatted responses',
             colorbarlabel='% Badly Formatted Responses',
             reverse_cmap=True,
             vmin=0,
             vmax=60)

plot_heatmap(pivot_df_bf_ep, 
             language_speakers_dict, 
             title='Comparative Analysis of Model Response Formatting Across Languages (English Prompt)\n' \
    'Values represent % badly formatted responses',
             colorbarlabel='% Badly Formatted Responses',
             reverse_cmap=True,
             vmin=0,
             vmax=60)


# %%
# Plot averaged over models
pivot_df_bf_avg = pivot_df_bf.copy()
# add avg in index, average over all models
pivot_df_bf_avg.loc['Average Model'] = pivot_df_bf_avg.mean(axis=0)

pivot_df_bf_ep_avg = pivot_df_bf_ep.copy()
# add avg in index, average over all models
pivot_df_bf_ep_avg.loc['Average Model'] = pivot_df_bf_ep_avg.mean(axis=0)

plot_avg_heatmap(pivot_df_bf_avg,
                language_speakers_dict, 
                title='AI Model Performance Across Languages\n' \
        'Values represent % badly formatted responses',
                colorbarlabel='% Badly Formatted Responses',
                reverse_cmap=True,
                vmin=0,
                vmax=60)

plot_avg_heatmap(pivot_df_bf_ep_avg,
                language_speakers_dict, 
                title='AI Model Performance Across Languages (English Prompt)\n' \
        'Values represent % badly formatted responses',
                colorbarlabel='% Badly Formatted Responses',
                reverse_cmap=True,
                vmin=0,
                vmax=60)


# %%
# --- 1. Pivot the data to create a matrix ---

# %%
pivot_df_latency = acc_df_langp.pivot_table(
    index='display_name',
    columns='language',
    values='Latency Mean'
)

pivot_df_latency_ep = acc_df_ep.pivot_table(
    index='display_name',
    columns='language',
    values='Latency Mean'
)

# order index based on size in MODELS_METADATA_MAPPING
# order: "Large", "Medium", "Small" and score on English withing each size category if necessary
#pivot_df_latency['size cat'] = pivot_df_latency.index.map(lambda x: MODELS_METADATA_MAPPING.get(x, {}).get('size', 'Unknown'))
#pivot_df_latency = pivot_df_latency.sort_values(by=['size cat', 'English'], ascending=[True, True])
#pivot_df_latency = pivot_df_latency.drop(columns=['size cat'])

# Sort index by accuracy on English using previously computed order
pivot_df_latency = pivot_df_latency.reindex(order_models_acc_english)
pivot_df_latency_ep = pivot_df_latency_ep.reindex(order_models_acc_english)

#pivot_df_latency_ep['size cat'] = pivot_df_latency_ep.index.map(lambda x: MODELS_METADATA_MAPPING.get(x, {}).get('size', 'Unknown'))
#pivot_df_latency_ep = pivot_df_latency_ep.sort_values(by=['size cat', 'English'], ascending=[True, True])
#pivot_df_latency_ep = pivot_df_latency_ep.drop(columns=['size cat'])

plot_heatmap(pivot_df_latency, 
             language_speakers_dict, 
             title='AI Model Performance Across Languages\n' \
'Values represent latency in sec/question',
             colorbarlabel='Latency (s)',
             reverse_cmap=True,
             vmin=5,
             vmax=25
             )

plot_heatmap(pivot_df_latency_ep,
             language_speakers_dict,
             title='AI Model Performance Across Languages (English Prompt)\n' \
'Values represent latency in sec/question',
             colorbarlabel='Latency (s)',
             reverse_cmap=True,
             vmin=5,
             vmax=25
             )

# %%
# Heatmap averaged over models
pivot_df_latency_avg = pivot_df_latency.copy()
# add avg in index, average over all models
pivot_df_latency_avg.loc['Average Model'] = pivot_df_latency_avg.mean(axis=0)

pivot_df_latency_ep_avg = pivot_df_latency_ep.copy()
# add avg in index, average over all models
pivot_df_latency_ep_avg.loc['Average Model'] = pivot_df_latency_ep_avg.mean(axis=0)

plot_avg_heatmap(pivot_df_latency_avg,
                language_speakers_dict,
                title='Average Model Inference Speed Across Languages\n' \
        'Values represent latency in sec/question',
                colorbarlabel='Latency (s)',
                reverse_cmap=True,
                vmin=5,
                vmax=25
                )

plot_avg_heatmap(pivot_df_latency_ep_avg,
                language_speakers_dict,
                title='Average Model Inference Speed Across Languages (English Prompt)\n' \
        'Values represent latency in sec/question',
                colorbarlabel='Latency (s)',
                reverse_cmap=True,
                vmin=5,
                vmax=25
                )






# %%
# Explode dataframe on columns "correct", "bad_format", "Latency", "TokensUsed", "TokensUsedCompletion", "TokensUsedReasoning"

acc_df_detailed_ep = acc_df_detailed[acc_df_detailed['english_prompt'] == True].reset_index(drop=True)

acc_df_detailed_ep_exploded = acc_df_detailed_ep.explode(
    ['correct', 'bad_format', 'Latency', 'TokensUsed', 'TokensUsedCompletion', 'TokensUsedReasoning']
).reset_index(drop=True)
print(acc_df_detailed_ep_exploded.shape)
print(acc_df_detailed_ep_exploded['english_prompt'].value_counts())
acc_df_detailed_ep_exploded.head()

# %%
# explode columns of ep and non ep for statistical analysis
acc_df_detailed_exploded = acc_df_detailed.explode(
    ['correct', 'bad_format', 'Latency', 'TokensUsed', 'TokensUsedCompletion', 'TokensUsedReasoning']
).reset_index(drop=True)
print(acc_df_detailed_exploded.shape)
print(acc_df_detailed_exploded['english_prompt'].value_counts())
acc_df_detailed_exploded.head()

# %%
# save
acc_df_detailed_exploded.to_csv(RESULTS_DIR / "cdpk_multilingual_model_performance_detailed_exploded.csv", index=False)

# %%

token_cols = ['TokensUsed', 'TokensUsedCompletion', 'TokensUsedReasoning']

# The color palette for the 'hue'
# We map 'True' to green and 'False' to red
palette_colors = {True: '#2ca02c',  # A nice green
           False: '#d62728'} # A nice red


# make columns numeric float
for col in ['Latency'] + token_cols:
    acc_df_detailed_ep_exploded[col] = pd.to_numeric(acc_df_detailed_ep_exploded[col], errors='coerce')

# --- 3. Create the Subplots ---
# Create a figure with 1 row and 3 columns of axes
fig, axs = plt.subplots(1, 3, figsize=(13, 6), sharex=True, sharey=True)

# Set a main title for the entire figure
fig.suptitle('Latency vs. Token Usage (by Correctness)', fontsize=16, y=1.03)


# --- 4. Loop and Plot ---
# Loop through each axis (subplot) and its corresponding token column
for ax, y_col in zip(axs, token_cols):
    
    # --- NEW: Calculate Correlations First ---
    df_true = acc_df_detailed_ep_exploded[acc_df_detailed_ep_exploded['correct'] == True]
    df_false = acc_df_detailed_ep_exploded[acc_df_detailed_ep_exploded['correct'] == False]
    
    # Calculate Pearson's r for each subgroup
    corr_true = df_true[['Latency', y_col]].corr().loc['Latency', y_col]
    corr_false = df_false[['Latency', y_col]].corr().loc['Latency', y_col]
    
    # Print correlations to console
    print(f"\nCorrelations for {y_col}:")
    print(f"  - Correct (True):   {corr_true:.3f}")
    print(f"  - Incorrect (False): {corr_false:.3f}")


    # Use seaborn's scatterplot function
    # It handles the x, y, hue, and palette mapping automatically
    sns.scatterplot(
        data=acc_df_detailed_ep_exploded,
        x='Latency',
        y=y_col,
        hue='correct',     # Use the 'correct' column for color
        palette=palette_colors,   # Use our defined red/green palette
        ax=ax,             # Tell seaborn which subplot to draw on
        s=50,              # Marker size
        alpha=0.7,         # Marker transparency
        edgecolor='black', # Add a thin edge to markers
        linewidth=0.5
    )

    # We add this on top, with scatter=False so it only draws the line
    sns.regplot(
        data=df_true,
        x='Latency',
        y=y_col,
        ax=ax,
        color=palette_colors[True],
        scatter=False, # <-- This is the key!
        #line_kws={'linestyle': '--', 'linewidth': 2.5}
    )
    
    # --- NEW: Plot 3: Regression Line for 'False' (Incorrect) ---
    sns.regplot(
        data=df_false,
        x='Latency',
        y=y_col,
        ax=ax,
        color=palette_colors[False],
        scatter=False, # <-- This is the key!
        #line_kws={'linestyle': '--', 'linewidth': 2.5}
    )
    
    # --- 5. Customize Each Subplot ---
    ax.set_title(f'{y_col} vs. Latency', fontsize=12)
    ax.set_xlabel('Latency (seconds)', fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    
    # Show grid lines
    ax.grid(True, linestyle='--', alpha=0.6)

     # --- *** NEW: ADD TEXT ANNOTATION FOR CORRELATIONS *** ---
    # Format the string to be printed
    text_str = f"Corr (Correct): {corr_true:.2f}\nCorr (Incorrect): {corr_false:.2f}"
    
    # Add the text to the top-right corner of the subplot
    ax.text(
        0.95, 0.95, text_str,      # x, y coordinates (95% from left, 95% from top)
        transform=ax.transAxes, # Use axis-relative coordinates
        ha='right', va='top',   # Align to top-right of the (x,y) point
        fontsize=10,
        # Add a white box for better readability
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='grey')
    )
    # --- *** END OF NEW SECTION *** ---
    
    
    # Remove the automatic legend from each subplot
    # (We will create one shared legend)
    if ax.get_legend():
        ax.get_legend().remove()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# --- 6. Create One Shared Legend ---
# Get the handles and labels from the last plot (they are all the same)
handles, labels = axs[2].get_legend_handles_labels()

# Create a new list for labels, making 'True' and 'False' clearer
new_labels = ['Correct' if label == 'True' else 'Incorrect' for label in labels]

# Add one legend for the whole figure
fig.legend(
    handles,
    new_labels,
    title='Result',
    loc='upper right',
    bbox_to_anchor=(0.98, 0.75), # Position the legend outside the plots
    frameon=True,
    shadow=True
)

# Adjust the layout to prevent titles/labels from overlapping
plt.tight_layout()


# %%
def plot_stripplot_var_by_correctness(ax, df, xaxis, xaxis_unit, title="default", legend=True):
    # Define the order for the 'hue' (optional, but good practice)
    hue_order = [True, False]

    # Define the custom color palette as requested
    palette = {True: '#2ca02c',  # Green
            False: '#d62728'} # Red

    # --- 3. Create the Plot ---
    # We use `stripplot` which is designed for a categorical axis (like 'model')
    # and a numerical axis (like 'Latency')

    ax = sns.stripplot(
        data=df,
        x=xaxis,            # X-axis is numerical
        y='display_name',       # Y-axis is categorical
        hue='correct',          # Use 'correct' column for color
        hue_order=hue_order,    # Use our defined order
        palette=palette,        # Use our red/green palette
        dodge=True,             # <-- This is the key! It separates the hues.
        alpha=0.7,              # Make dots slightly transparent
        s=5,                    # Set marker size
        jitter=0.1,              # Add a little jitter to see overlapping points
        ax=ax
    )

    # --- 4. Customize the Plot ---
    if title == "default":
        ax.set_title(f'{xaxis} Distribution by Model and Correctness', fontsize=16)
    else:
        ax.set_title(title, fontsize=13)
    ax.set_xlabel(f'{xaxis} ({xaxis_unit})', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    # Add a vertical grid for easier reading of latency
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Improve the legend
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    new_labels = ['Correct' if label == 'True' else 'Incorrect' for label in labels]
    if legend:
        ax.legend(handles, new_labels, title='Result', loc='lower right', 
                bbox_to_anchor=(0.95, 0.1), # Position legend below plot
                frameon=True, ncol=1)
    else:
        # Legend off
        ax.get_legend().remove()

    return ax


# %%

fig, ax = plt.subplots(figsize=(10, 5))
ax = plot_stripplot_var_by_correctness(
    ax,
    acc_df_detailed_ep_exploded,
    xaxis='Latency',
    xaxis_unit='seconds'
)
plt.show()

# %%

# Order by reasoning to split between reasoning and non reasoning models from MODELS_METADATA_MAPPING
acc_df_detailed_ep_exploded['reasoning_model'] = acc_df_detailed_ep_exploded['display_name'].apply(
    lambda x: MODELS_METADATA_MAPPING.get(x, {}).get('reasoning', "Unknown")
)
acc_df_detailed_ep_exploded = acc_df_detailed_ep_exploded.sort_values(by='reasoning_model', ascending=False)

# print value counts of model wit Unknown reasoning
print("Reasoning Model Value Counts:")
print(acc_df_detailed_ep_exploded['reasoning_model'].value_counts())
print("\n")
print("Models with 'Unknown' reasoning:")
acc_df_detailed_ep_exploded[
    acc_df_detailed_ep_exploded['reasoning_model'] == "Unknown"
].head()

# %%
fig, axs = plt.subplots(1,3, figsize=(15, 5), sharex=True, sharey=True)
axs[0] = plot_stripplot_var_by_correctness(
    axs[0],
    acc_df_detailed_ep_exploded,
    xaxis='TokensUsed',
    xaxis_unit='# tokens',
    title='Total Tokens Used',
    legend=True
)
axs[1] = plot_stripplot_var_by_correctness(
    axs[1],
    acc_df_detailed_ep_exploded,
    xaxis='TokensUsedCompletion',
    xaxis_unit='# tokens',
    title='Tokens Used for Completion',
    legend=True
)
axs[2] = plot_stripplot_var_by_correctness(
    axs[2],
    acc_df_detailed_ep_exploded,
    xaxis='TokensUsedReasoning',
    xaxis_unit='# tokens',
    title='Tokens Used for Reasoning',
    legend=True
)
plt.suptitle('Token Usage vs. Model by Correctness', fontsize=18)
plt.tight_layout()


# %%
#--- 2. Statistical Analysis (Correlation Matrix) ---
print("-------------------------------------------------")
print("--- Correlation Analysis ---")
print("-------------------------------------------------")

# Select only the numerical columns you want to correlate
cols_to_analyze = ['Latency', 'TokensUsed', 'TokensUsedCompletion', 'TokensUsedReasoning']
corr_matrix = acc_df_detailed_ep_exploded[cols_to_analyze].corr()

# Print the full matrix
print("Full Correlation Matrix:")
print(corr_matrix)
print("\n")

# Print the correlations with 'Latency' specifically, sorted
print("Correlation with Latency:")
print(corr_matrix['Latency'].sort_values(ascending=False))
print("-------------------------------------------------")

corr_matrix.head()

# %%

# Deep dive into small models vs big models on latency
models_shortlist = [
    # large
    #"gemini-2.5-pro-preview-06-05",
    "gemini-3-pro-preview",
    #"gpt-5-2025-08-07-medium",
    # small
    "gemma-3-27b",
    "gemma-3-4b-it",
]

acc_df_detailed_ep_exploded_shortlist = acc_df_detailed_ep_exploded[
    acc_df_detailed_ep_exploded['model'].isin(models_shortlist)
].reset_index(drop=True)

print(f"Models shortlisted: {acc_df_detailed_ep_exploded_shortlist['display_name'].unique()}")
acc_df_detailed_ep_exploded_shortlist.head()

# %%
reasoning_shortlist = {
    #'claude-sonnet-4-5-20250929', # no reasoning tokens available
    #'gpt-5-2025-08-07-medium',
    'gpt-5.2-2025-12-11-medium',
    #'gemini-2.5-flash-preview-09-2025', 
    #'o4-mini-2025-04-16',
    #'gemini-2.5-pro-preview-06-05', 
    'gemini-3-pro-preview',
    'deepseek-r1-0528-fp8',
    #'qwen-3-32b', # missing reasoning tokens for swahili
}

acc_df_detailed_ep_exploded_reasoning = acc_df_detailed_ep_exploded[
    acc_df_detailed_ep_exploded['model'].isin(reasoning_shortlist)
].reset_index(drop=True)

# check
print(f"Number of category (should be 1): {acc_df_detailed_ep_exploded_reasoning['reasoning_model'].nunique()}")
print(f"Number of models shortlisted: {acc_df_detailed_ep_exploded_reasoning['display_name'].nunique()}")
print(f"Number of languages: {acc_df_detailed_ep_exploded_reasoning['language'].nunique()}")

df = acc_df_detailed_ep_exploded_reasoning

# order the languages by number of speakers from language_speakers_dict
language_order = [language for language in sorted(
    df['language'].unique(), 
    key=lambda x: language_speakers_dict.get(x, 0), 
    reverse=True
)]
print(f"Language order: {language_order}")
df['language'] = pd.Categorical(df['language'], categories=language_order, ordered=True)

# Ensure 'correct' is boolean just in case
df['correct'] = df['correct'].astype(bool)

# %%
# --------------------------------------------------------------------------------
# 2. PLOTTING CODE
# --------------------------------------------------------------------------------

# Set a nice style
sns.set_theme(style="whitegrid")

# Define custom colors: True -> Green, False -> Red
custom_palette = {True: "#2ca02c", False: "#d62728"}

# Create the Grid using relplot (Relation Plot) which is perfect for row/col grids
# We use 'display_name' for rows and 'language' for columns
g = sns.relplot(
    data=df,
    x='Latency',
    y='TokensUsedReasoning',
    row='display_name',  # Corresponds to "display_model" in your request
    col='language',      # Corresponds to "language" column
    hue='correct',       # Color by correctness
    palette=custom_palette,
    kind='scatter',
    height=3,          # Height of each subplot
    aspect=1,          # Width aspect ratio
    s=80,                # Size of dots
    alpha=0.5,           # Transparency to see overlapping dots
    hue_order=[False, True], # Ensure True comes before False in legend
    facet_kws={'sharex': True, 'sharey': True} # Share axes
)

# --------------------------------------------------------------------------------
# 3. CUSTOMIZING LABELS AND LEGEND
# --------------------------------------------------------------------------------
# Clean up titles (optional, removes "display_name = " prefix)
g.set_titles(row_template="{row_name}", col_template="{col_name}")

# Set Axis Labels
g.set_axis_labels("Latency (s)", "Reasoning Tokens")

# Customize the Legend
legend = getattr(g, "_legend", None)
if legend is not None:
    legend.remove() # Remove the default legend (it's often in a bad spot)
# We iterate over the legend text to replace "True" with "Correct" and "False" with "Incorrect"
g.fig.legend(
    title="Response",
    labels=["Correct", "Incorrect"], # Explicitly naming them based on hue_order
    loc="upper center",              # Place it at the top center
    bbox_to_anchor=(0.5, 0.96),      # Adjust (x, y) to sit just below the main title
    ncol=2,                          # Horizontal legend (2 columns) saves vertical space
    frameon=False                    # Remove box border for a cleaner look
)

# Adjust layout to prevent cutting off titles
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Latency vs Reasoning Tokens by Model and Language', fontsize=16)

plt.show()

# %%
# Set style
sns.set_theme(style="whitegrid")

# Define specific green/red colors
custom_palette = {True: "#2ca02c", False: "#d62728"}

# We use catplot (Categorical Plot) with kind='box'
g = sns.catplot(
    data=df,
    kind="box",
    x="language",             # X-axis: Compare languages side-by-side
    y="TokensUsedReasoning",  # Y-axis: How much reasoning?
    hue="correct",            # Split each language into Correct vs Incorrect
    row="display_name",       # Separate models into rows
    palette=custom_palette,
    height=2.5,             # Height of each subplot
    aspect=2,               # Wide aspect ratio to fit all languages
    sharex=True,             # Keep axes independent if needed
    sharey=True,
    showfliers=False,         # Hides extreme outliers to keep the boxes visible
    width=0.5,
    linewidth=1.0,
    legend=True,             # We'll customize the legend later
)

# 1. Clean Titles and Labels
g.set_titles(row_template="{row_name}", fontweight='bold', fontsize=13)
g.set_axis_labels("", "Reasoning Tokens")

g.fig.suptitle('The Cost of Thinking:\nReasoning Effort vs. Correctness by Language', y=1.04, fontsize=13, fontweight='bold')

plt.show()

# %%
# %%
# Plot to analyze latency on small models


df_small = acc_df_detailed_ep_exploded_shortlist

# reorder models as in models_shortlist
model_order = []
for model_id in models_shortlist:
    display_name = df_small[df_small['model'] == model_id]['display_name'].iloc[0]
    model_order.append(display_name)
df_small['display_name'] = pd.Categorical(df_small['display_name'], categories=model_order, ordered=True)

# order the languages by number of speakers from language_speakers_dict
language_order = [language for language in sorted(
    df['language'].unique(), 
    key=lambda x: language_speakers_dict.get(x, 0), 
    reverse=True
)]
print(f"Language order: {language_order}")
df_small['language'] = pd.Categorical(df_small['language'], categories=language_order, ordered=True)

print(f"Number of category (should be 1): {df_small['category'].nunique()}")
print(f"Number of models shortlisted: {df_small['display_name'].nunique()}")
print(f"Number of languages: {df_small['language'].nunique()}")


# %%
# Set style
sns.set_theme(style="whitegrid")

# Define specific green/red colors
custom_palette = {True: "#2ca02c", False: "#d62728"}

# We use catplot (Categorical Plot) with kind='box'
g = sns.catplot(
    data=df_small,
    kind="box",
    x="language",             # X-axis: Compare languages side-by-side
    y="Latency",              # Y-axis: Latency
    hue="correct",            # Split each language into Correct vs Incorrect
    row="display_name",       # Separate models into rows
    palette=custom_palette,
    height=2.5,             # Height of each subplot
    aspect=2,               # Wide aspect ratio to fit all languages
    sharex=True,             # Keep axes independent if needed
    sharey=False,
    showfliers=False,         # Hides extreme outliers to keep the boxes visible
    width=0.5,
    linewidth=1.0,
    legend=True,             # We'll customize the legend later
)

# 1. Clean Titles and Labels
g.set_titles(row_template="{row_name}", fontweight='bold', fontsize=13)
g.set_axis_labels("", "Latency (s)")

g.fig.suptitle('Latency Profiles:\nHow Model Size Impacts Response Time and Accuracy', y=1.04, fontsize=13, fontweight='bold')

plt.show()

# %%

# ---------------------------------------------------------
# 1. PREPARE THE DATA
# ---------------------------------------------------------

# C. Combine them into one plotting dataframe
df_combined = acc_df[['display_name', 'language', 'accuracy', 'english_prompt']].copy()

df_combined['Prompt Type'] = df_combined['english_prompt'].apply(
    lambda x: 'English' if x else 'Native'
)

# include only few models
models_to_include = [
    "Claude Sonnet 4.5",
    "GPT-5.2 (Medium)",
    "Gemini-3 Pro",
    #"Gemini-3 Flash",
    #"Gemini-2.5 Flash-Lite",
    #"Deepseek R1",
    #"o4-Mini",
    #"Qwen-3 32B",
    "Gemma-3 27B",
    "Gemma-3 4B",
]

df_combined = df_combined[
    df_combined['display_name'].isin(models_to_include)
].reset_index(drop=True)

df_native = df_combined[df_combined['english_prompt'] == False]
df_english = df_combined[df_combined['english_prompt'] == True] 

# %%
# ---------------------------------------------------------
# 2. CALCULATE "DODGED" X-COORDINATES (INVERTED)
# ---------------------------------------------------------
# Now X-Axis = Models, Offset = Languages
# Models are sorted from highest to lowest average accuracy on english category
models = df_combined[df_combined['language'] == 'English'].groupby('display_name')['accuracy'].mean().sort_values(ascending=False).index.tolist()
# sort languages from language_speakers_dict values
languages = sorted(
    df_combined['language'].unique(), 
    key=lambda x: language_speakers_dict.get(x, 0), 
    reverse=True
)

# Assign a base number to each MODEL (0, 1, 2...)
model_map = {model: i for i, model in enumerate(models)}

# Assign a small offset to each LANGUAGE
# We spread 6 languages across a width of 0.7 units
num_langs = len(languages)
offsets = np.linspace(-0.35, 0.35, num_langs)
lang_offset_map = dict(zip(languages, offsets))

# Apply these to create the plotting coordinates
df_combined['base_x'] = df_combined['display_name'].map(model_map)
df_combined['offset'] = df_combined['language'].map(lang_offset_map)
df_combined['plotting_x'] = df_combined['base_x'] + df_combined['offset']

# ---------------------------------------------------------
# 3. PLOTTING
# ---------------------------------------------------------

plt.figure(figsize=(18, 8)) 

# Define Colors for Languages & Markers for Prompts
colors = sns.color_palette("bright", n_colors=len(languages)) # 'bright' helps distinguish thin lines
lang_color_map = dict(zip(languages, colors))
markers = {"English": "o", "Native": "s"}

# A. Draw the Vertical Lines
# Loop: Model (Base X) -> Language (Offset)
for model in models:
    for lang in languages:
        subset = df_combined[
            (df_combined['language'] == lang) & 
            (df_combined['display_name'] == model)
        ]
        
        if len(subset) == 2:
            x_pos = subset['plotting_x'].iloc[0]
            y_min = subset['accuracy'].min()
            y_max = subset['accuracy'].max()
            
            plt.plot(
                [x_pos, x_pos], [y_min, y_max],
                color=lang_color_map[lang],
                alpha=0.6,
                linewidth=2,
                zorder=1
            )

# B. Draw the Dots
sns.scatterplot(
    data=df_combined,
    x="plotting_x",
    y="accuracy",
    hue="language",      # Color by Language now
    style="Prompt Type", # Shape by Prompt Type
    markers=markers,
    palette=lang_color_map,
    s=100,
    zorder=2
)

# ---------------------------------------------------------
# 4. VISUAL POLISH
# ---------------------------------------------------------
# X-Axis: Show Model Names
plt.xticks(
    ticks=range(len(models)), 
    labels=models, 
    rotation=45, 
    ha='right', 
    fontweight='bold'
)

# Y-Axis: Percentages
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
#plt.ylim(0, 100)

plt.xlabel("")
plt.ylabel("Accuracy")
plt.title('The Translation Gap: Comparing Consistency Across Models', fontsize=20, fontweight='bold', y=1.03)

# Legend Layout
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, frameon=False, title="Language & Prompt")

plt.tight_layout()

# %%
# Stacked bar plot english vs local language prompt accuracy difference

# ---------------------------------------------------------
# 1. PREPARE THE DATA
# ---------------------------------------------------------
# A. Melt both dataframes to long format
df_native = pivot_df_acc.reset_index().melt(
    id_vars=['display_name'], var_name='language', value_name='acc_native'
)
df_english = pivot_df_acc_ep.reset_index().melt(
    id_vars=['display_name'], var_name='language', value_name='acc_english'
)

# B. Merge them into a single DataFrame
#    This ensures every row has both the Native and English score
df_merged = pd.merge(df_native, df_english, on=['display_name', 'language'])

# rename some models
df_merged['display_name'] = df_merged['display_name'].replace({
    "GPT-5 (Medium)": "GPT-5",
    "Deepseek R1 (May '25)": "Deepseek R1",
})

# select subset models_to_include
df_merged = df_merged[
    df_merged['display_name'].isin(models_to_include)
].reset_index(drop=True)

# ---------------------------------------------------------
# 2. SORTING (Crucial for clean grouped bars)
# ---------------------------------------------------------
# Sort models by average Native Accuracy (Best to Worst)
sorted_models = df_merged[df_merged['language'] == "English"].groupby('display_name')['acc_english'].mean().sort_values(ascending=False).index.tolist()

# Sort languages by difficulty (Gap size)
df_merged['gap'] = df_merged['acc_english'] - df_merged['acc_native']
sorted_langs = df_merged.groupby('language')['gap'].mean().sort_values().index.tolist()
df_merged.head()

# %%
# ---------------------------------------------------------
# 3. PLOTTING THE "GHOST" BAR CHART
# ---------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(16, 8))

# Define the colors for the languages
palette = sns.color_palette("bright", n_colors=len(sorted_langs))

# LAYER 1: The "English Prompt" (The Ceiling)
# We plot this first with low opacity (alpha). This acts as the "background" bar.
sns.barplot(
    data=df_merged,
    x='display_name',
    y='acc_english',
    hue='language',
    hue_order=sorted_langs,
    order=sorted_models,
    palette=palette,
    alpha=0.3,      
    dodge=True, 
    edgecolor="gray",
    linestyle="--",
    linewidth=1,
    #label='_nolegend_' # Hide from legend to avoid duplicates
    legend=False
)

# LAYER 2: The "Native Prompt" (The Reality)
# We plot this on top with full opacity.
# Since the X, Hue, and Order are identical, they will overlap perfectly.
ax = sns.barplot(
    data=df_merged,
    x='display_name',
    y='acc_native',
    hue='language',
    hue_order=sorted_langs,
    order=sorted_models,
    palette=palette,
    alpha=1,     
    dodge=True,
    edgecolor="white",
    linewidth=0.5
)

# ---------------------------------------------------------
# 4. VISUAL POLISH
# ---------------------------------------------------------
# Y-Axis to Percentages
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
#plt.ylim(0, 105)

# Labels
plt.xlabel("")
plt.ylabel("Accuracy")
#plt.xticks(rotation=45, ha='right', fontweight='bold')
plt.xticks(rotation=0, fontweight='bold')
plt.title('Native vs. English Prompts\nQuantifying the Decrease in Model Accuracy Across Languages', fontsize=20, fontweight='bold', y=1.01)

# Fix the Legend
# We only need one legend (from the second plot).
# Let's move it outside to save space.
sns.move_legend(
    ax, "upper left",
    bbox_to_anchor=(1.0, 1.0),
    title="Language",
    frameon=True
)
# remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
# %%
