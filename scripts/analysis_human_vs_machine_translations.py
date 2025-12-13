# %%
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from fdllm.sysutils import register_models

from cdpk.benchmark_run import run_benchmark
from cdpk.benchmark_constants import ROOT
from cdpk.benchmark_utils import fulldf_accuracy_by_category

# %%
# import data
machine_translations_tag = "Luganda_ep_new"
human_translations_tag = "Luganda_ep_reviewed"

cdpk_dataset_reviewed = pd.read_csv(f"./../data/pedagogy_benchmark_luganda_cdpk_reviewed.csv")
print(f"CDPK reviewed dataset shape: {cdpk_dataset_reviewed.shape}")
cdpk_dataset_reviewed.head()

# open fs examples dictionary
with open("./../data/few_shot_examples_idx_dict.json", "r") as f:
    few_shot_examples_idx_dict = json.load(f)

# Create a dictionary of category -> question_id
category_question_ids_dict = {}
for category in cdpk_dataset_reviewed['category'].unique():
    subset_df = cdpk_dataset_reviewed[cdpk_dataset_reviewed['category'] == category]
    question_ids = subset_df['question_id'].astype(int).tolist()
    #category_question_ids_dict[category] = question_ids

    # remove fs examples from the list
    cat_dict_name = category.lower().replace(" ", "_") if category != "General" else "gen_pk"
    few_shot_indices = few_shot_examples_idx_dict.get(f"CDPK_{cat_dict_name}", [])
    question_ids_no_fs = [qid for qid in question_ids if qid not in few_shot_indices]
    print(f"Removed {len(question_ids) - len(question_ids_no_fs)} few-shot examples from category {category}.")
    category_question_ids_dict[category] = question_ids_no_fs

print("Category to question IDs mapping:")
print(json.dumps(category_question_ids_dict, indent=2))


# %%
# Reproduce full results dataframe with questions and preds for each model

load_dotenv(override=True)

def create_results_dataframes(benchmark, version, model_config_file, output_folder=None):

    QUESTIONS_LIST_DICT = {
        "cdpk": [
            f"CDPK_{version}_science",
            f"CDPK_{version}_literacy",
            f"CDPK_{version}_creative_arts",
            f"CDPK_{version}_maths",
            f"CDPK_{version}_social_studies",
            f"CDPK_{version}_technology",
            f"CDPK_{version}_general",
        ],
        #"send": [f"CDPK_{language}_send"],
    }
    
    #if opt.models_config is None:
    #    if opt.benchmark == "cdpk":
    #        #opt.models_config = "cdpk_online_leaderboard"
    #        opt.models_config = "full_list_20251015_small"
    #    elif opt.benchmark == "send":
    #        opt.models_config = "send_online_leaderboard"
    #    else:
    #        raise NotImplementedError(f"'benchmark' must be one of 'send' or 'cdpk'")
    #config_models_PK = opt.models_config
    config_models_PK = model_config_file
    #
    if output_folder is None:
        res_dir = ROOT / "data" / "results"
    else:
        res_dir = Path(output_folder)
    res_dir.mkdir(exist_ok=True, parents=True)

    #
    category_df_list = []
    accuracies_df = pd.DataFrame()
    bad_format_df = pd.DataFrame()
    length_per_category = {}
    # summary_df_chile = pd.DataFrame()
    for cat_config_name in QUESTIONS_LIST_DICT[benchmark]:
        print(f"Running {benchmark} benchmark for {cat_config_name}")
        category_df, summary_df, config, models_dict_PK = run_benchmark(
            questions_config=cat_config_name, models_config=config_models_PK
        )
        if benchmark == "send":
            category_df["category"] = "SEND"
        if len(category_df["category"].unique()) > 1:
            print("More than one category in the dataframe")
            break
        else:
            category_name = category_df["category"].unique()[0]
            print(f"Category: {category_name}")

        # append result
        category_df_list.append(category_df)
        # concatenate accuracies and bad_format for each category and give name to the columns
        accuracies_df[category_name] = summary_df.loc["accuracy"] * 100
        bad_format_df[category_name] = summary_df.loc["bad_format"] * 100

        # get the length of the dataframe for each category (removing few-shot examples)
        length_per_category[category_name] = len(category_df) - len(
            config["example_rows"]
        )

    full_df = pd.concat(category_df_list, axis=0, ignore_index=True)
    col_pred = [col for col in full_df.columns if col.startswith("pred_")][0]
    full_df = full_df[full_df[col_pred] != "Few-shot example"]
    full_df = full_df.reset_index().rename(columns={"index": "question_id"})
    if benchmark == "send":
        full_df["question_id"] = full_df["question_id"] + 920
    accuracy_overall = fulldf_accuracy_by_category(
        fulldf=full_df, models_dict=models_dict_PK, bad_format_threshold=None
    )
    accuracies_df["Overall"] = accuracy_overall["Accuracy"]
    bad_format_df["Overall"] = accuracy_overall["Bad Format"]

    ###### save results
    acc_file = res_dir / f"{version}_{benchmark}_results_accuracy_{model_config_file}.csv"
    bf_file = res_dir / f"{version}_{benchmark}_results_bad_format_{model_config_file}.csv"
    full_file = res_dir / f"{version}_{benchmark}_results_full_{model_config_file}.csv"
    accuracies_df.to_csv(acc_file)
    bad_format_df.to_csv(bf_file)
    full_df.to_csv(full_file, index=False)
    accuracies_df.index = accuracies_df.index.map(models_dict_PK)
    print(accuracies_df[["Overall"]].sort_values("Overall", ascending=False).to_markdown())

    return models_dict_PK

# %%
models_dict_PK = create_results_dataframes(
    benchmark="cdpk",
    version=human_translations_tag,
    model_config_file="full_list_20251015_small",
    output_folder="./../data/results/",
)

# %%
models_dict_PK = create_results_dataframes(
    benchmark="cdpk",
    version=machine_translations_tag,
    model_config_file="full_list_20251015_small",
    output_folder="./../data/results/",
)

# %%
full_df_humans = pd.read_csv(f"./../data/results/Luganda_ep_reviewed_cdpk_results_full_full_list_20251015_small.csv")
print(full_df_humans.shape)

full_df_machine = pd.read_csv(f"./../data/results/Luganda_ep_new_cdpk_results_full_full_list_20251015_small.csv")
print(full_df_machine.shape)

# %%
# check if there is any cells with "Few-shot example" in pred_ columns
col_pred = [col for col in full_df_humans.columns if col.startswith("pred_")][0]
print(full_df_humans[full_df_humans[col_pred] == "Few-shot example"].shape)
print(full_df_machine[full_df_machine[col_pred] == "Few-shot example"].shape)
print(f"Columns in full_df_humans:  {full_df_humans.columns.tolist()}")
print(f"Columns in full_df_machine: {full_df_machine.columns.tolist()}")
display(full_df_humans.head(2), full_df_machine.head(2))


    
# %%
# only keep the rows of full_df_machine that are in full_df_humans based on question_id for comparison

full_df_machine_subset = full_df_machine[full_df_machine['question_id'].isin(full_df_humans['question_id_original'].values)].reset_index(drop=True)
print(f"Dataset of machine translations shape: {full_df_machine_subset.shape}")
print(f"Dataset of human rev. translat. shape: {full_df_humans.shape}")

# move question_id_original to 2nd column
cols = full_df_humans.columns.tolist()
cols.remove('question_id_original')
cols.insert(1, 'question_id_original')
full_df_humans = full_df_humans[cols]

# print different columns
print(full_df_machine_subset.columns.difference(full_df_humans.columns))
print(full_df_humans.columns.difference(full_df_machine_subset.columns))

display(full_df_machine_subset.head(2), full_df_humans.head(2))

# %%
# Check: print side by side 3 rows of full_df_humans and full_df_machine_subset for the same question_id_original
#for qid in full_df_humans['question_id_original'].unique()[:3]:
#
#    human_row = full_df_humans[full_df_humans['question_id_original'] == qid]
#    machine_row = full_df_machine_subset[full_df_machine_subset['question_id'] == qid]
#    print(f"Question ID: {qid}")
#    print("Human Reviewed Translation:")
#    display(human_row)
#    print("Machine Translation:")
#    display(machine_row)

# %%
# compare performance of each model between human reviewed translations and machine translations

models_list = [col.replace("pred_", "") for col in full_df_humans.columns if col.startswith("pred_")]
print(f"{len(models_list)} models to compare: {models_list}")

def summarize_accuracies(df, per_category=False):
    accuracy_df = pd.DataFrame()
    bad_format_df = pd.DataFrame()
    if per_category:

        for cat in df['category'].unique():
            subset_df = df[df['category'] == cat]
            accuracy_overall = fulldf_accuracy_by_category(
                fulldf=subset_df, 
                models_dict=models_dict_PK,
                bad_format_threshold=None
            )
            accuracy_df[cat] = accuracy_overall['Accuracy']
            bad_format_df[cat] = accuracy_overall['Bad Format']

    else:
        accuracy_overall = fulldf_accuracy_by_category(
            fulldf=df, 
            models_dict=models_dict_PK,
            bad_format_threshold=None
        )
        accuracy_df["Overall"] = accuracy_overall['Accuracy']
        bad_format_df["Overall"] = accuracy_overall['Bad Format']
    
    return accuracy_df, bad_format_df

accuracy_humans, bad_format_humans = summarize_accuracies(full_df_humans)
accuracy_machine, bad_format_machine = summarize_accuracies(full_df_machine_subset)
print("Accuracy - Human Reviewed Translations:")
display(accuracy_humans.head(10))
print("Accuracy - Machine Translations:")
display(accuracy_machine.head(10))

# %%
# per category
accuracy_humans_cat, bad_format_humans_cat = summarize_accuracies(full_df_humans, per_category=True)
accuracy_machine_cat, bad_format_machine_cat = summarize_accuracies(full_df_machine_subset, per_category=True)

print("Accuracy per category - Human Reviewed Translations:")
display(accuracy_humans_cat.head(10))
print("Accuracy per category - Machine Translations:")
display(accuracy_machine_cat.head(10))

# %%
# merge accuracy dataframes and remove first layer of columns
accuracy_humans['Translation'] = 'Humans'
accuracy_machine['Translation'] = 'LLM'
accuracy_overall_merged = pd.concat(
    [accuracy_humans, accuracy_machine],
    axis=0,
).reset_index()

# add display name
models_csv = pd.read_csv(ROOT / "data" / "models.csv")
accuracy_overall_merged['display_name'] = accuracy_overall_merged['Model'].apply(
    lambda x: models_csv[models_csv['model_id'] == x]['display_name'].values[0]
    if len(models_csv[models_csv['model_id'] == x]['display_name'].values) == 1
    else x
)
# order by Overall accuracy of LLM translations
accuracy_overall_merged = accuracy_overall_merged.sort_values(
    by=['Overall', 'Translation'],
    ascending=[False, True]
).reset_index(drop=True)

accuracy_overall_merged.head()
# %%
# plot overall accuracy comparison

plt.figure(figsize=(8, 4)) # Increased width slightly to make room for text

# 1. Capture the axes object (ax) when creating the plot
ax = sns.barplot(
    data=accuracy_overall_merged,
    y='display_name',
    x='Overall',
    hue='Translation',
    palette={'Humans': 'skyblue', 'LLM': 'salmon'},
)

# 2. Iterate over the labels actually plotted on the Y-axis
# This ensures we match the visual order, even if Seaborn sorted them differently
y_labels = [t.get_text() for t in ax.get_yticklabels()]

for i, model_name in enumerate(y_labels):
    # Filter the dataframe for the current model
    model_data = accuracy_overall_merged[accuracy_overall_merged['display_name'] == model_name]
    
    # Get the specific values for LLM and Humans
    # (Using .values[0] safely extracts the number)
    try:
        val_llm = model_data[model_data['Translation'] == 'LLM']['Overall'].values[0]
        val_human = model_data[model_data['Translation'] == 'Humans']['Overall'].values[0]
        
        # Calculate the difference
        diff = val_llm - val_human
        
        # Determine placement: Place it to the right of the longer bar
        max_val = max(val_llm, val_human)
        
        # Add the text annotation
        # x = max_val + offset (e.g., 1 or 2 units)
        # y = i (the index of the tick mark)
        ax.text(
            x=max_val + 1, 
            y=i, 
            s=f"{diff:+.1f}%",  # Format with sign (e.g., +2.5% or -1.2%)
            va='center', 
            fontsize=9, 
            fontweight='bold',
            color='green' if diff > 0 else 'red'
        )
    except IndexError:
        continue # Skip if data is missing for a pair

ax.xaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)
ax.set_axisbelow(True)
plt.xticks(rotation=0, ha='right')
plt.xlabel('Accuracy (%)')
plt.ylabel('')
plt.title('Overall Accuracy Comparison: LLM vs Human Reviewed Translations')
plt.legend(title='Translation')
# remove top and right spines
sns.despine()
plt.tight_layout()
plt.show()

# %%
# Prepare data for per category plot
accuracy_humans_cat_melted = accuracy_humans_cat.reset_index().melt(id_vars=['Model'], var_name='Category', value_name='Accuracy')
accuracy_machine_cat_melted = accuracy_machine_cat.reset_index().melt(id_vars=['Model'], var_name='Category', value_name='Accuracy')
accuracy_humans_cat_melted['Translation'] = 'Humans'
accuracy_machine_cat_melted['Translation'] = 'LLM'

accuracy_cat_merged = pd.concat(
    [accuracy_humans_cat_melted, accuracy_machine_cat_melted],
    axis=0,
).reset_index(drop=True)

accuracy_cat_merged['display_name'] = accuracy_cat_merged['Model'].apply(
    lambda x: models_csv[models_csv['model_id'] == x]['display_name'].values[0]
    if len(models_csv[models_csv['model_id'] == x]['display_name'].values) == 1
    else x
)
accuracy_cat_merged.head()

# %%
# subplots per category 1x7 grid
fig, axs = plt.subplots(1, 7, figsize=(20, 5), sharey=True, sharex=True)

global_models_order = accuracy_cat_merged.sort_values(
    by=['Accuracy'],
    ascending=False
)['display_name'].unique().tolist()

for i, cat in enumerate(accuracy_cat_merged['Category'].unique()):

    # Use .copy() to avoid SettingWithCopyWarning
    acc_category_subset = accuracy_cat_merged[accuracy_cat_merged['Category'] == cat].copy()
    
    # set global order
    acc_category_subset['display_name'] = pd.Categorical(
        acc_category_subset['display_name'],
        categories=global_models_order,
        ordered=True
    )

    sns.barplot(
        data=acc_category_subset,
        x='display_name',
        y='Accuracy',
        hue='Translation',
        ax=axs[i],
        palette={'Humans': 'skyblue', 'LLM': 'salmon'},
    )

    # --- START OF NEW LOGIC ---
    # Get the list of models exactly as they appear on the X-axis
    # (This ensures we align the calculation with the correct bar)
    models_on_axis = [label.get_text() for label in axs[i].get_xticklabels()]

    for j, model_name in enumerate(models_on_axis):
        # Filter the subset for the current model
        model_data = acc_category_subset[acc_category_subset['display_name'] == model_name]

        try:
            # Extract values safely
            val_llm = model_data[model_data['Translation'] == 'LLM']['Accuracy'].values[0]
            val_human = model_data[model_data['Translation'] == 'Humans']['Accuracy'].values[0]

            diff = val_llm - val_human
            max_val = max(val_llm, val_human)

            # Add text annotation
            # x = j (the index of the bar group)
            # y = max_val + offset (e.g., 2% higher)
            axs[i].text(
                x=j,
                y=max_val + 2, 
                s=f"{diff:+.1f}%",
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=90,  # Vertical text prevents overlapping in tight grids
                color='green' if diff > 0 else 'red',
                fontweight='bold'
            )
        except (IndexError, KeyError):
            continue
    # --- END OF NEW LOGIC ---

    axs[i].set_title(cat, fontsize=12, fontweight='bold')
    axs[i].set_xlabel('')
    axs[i].set_ylabel('Accuracy (%)')
    
    # It is safer to not force set_xticks if using sharex with Categorical data, 
    # but if you need to enforce rotation:
    axs[i].set_xticks(axs[i].get_xticks())  # Ensure ticks are set
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90, ha='right', fontsize=8)

plt.suptitle('Accuracy Comparison per Category: LLM vs Human Reviewed Translations', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
# %%
