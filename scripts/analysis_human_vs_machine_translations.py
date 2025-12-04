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
#test
model = "qwen-3-32b"
category = "science"
answers_machine = pd.read_csv(f"./../data/cache_local/CDPK_{machine_translations_tag}_{category}/resps_{model}.csv")
print(answers_machine.shape)
answers_machine.head()
# %%
answers_human = pd.read_csv(f"./../data/cache_local/CDPK_{human_translations_tag}_{category}/resps_{model}.csv")
print(answers_human.shape)
answers_human.head()

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
#
full_df_humans = pd.read_csv(f"./../data/results/Luganda_ep_reviewed_cdpk_results_full_full_list_20251015_small.csv")
print(full_df_humans.shape)

full_df_machine = pd.read_csv(f"./../data/results/Luganda_ep_new_cdpk_results_full_full_list_20251015_small.csv")
print(full_df_machine.shape)

# %%
# check if there is any cells with "Few-shot example" in pred_ columns
col_pred = [col for col in full_df_humans.columns if col.startswith("pred_")][0]
print(full_df_humans[full_df_humans[col_pred] == "Few-shot example"].shape)
print(full_df_machine[full_df_machine[col_pred] == "Few-shot example"].shape)

display(full_df_humans.head(2), full_df_machine.head(2))

# %%
# add the original question_id from cdpk reviewed dataset by matching on question text
full_df_humans['question_id_original'] = full_df_humans['question'].apply(
    lambda x: cdpk_dataset_reviewed[cdpk_dataset_reviewed['question'] == x]['question_id'].values[0]
    if len(cdpk_dataset_reviewed[cdpk_dataset_reviewed['question'] == x]['question_id'].values) == 1
    else None
)

# place new column in 2n position
full_df_humans = full_df_humans[full_df_humans.columns[:1].tolist() + ['question_id_original'] + full_df_humans.columns[1:-1].tolist()]


# %%
# check
for i, idx in enumerate(full_df_humans['question_id_original'].values):
    #print question column of both dataframe
    print(f"Index: {idx}")
    print(f"Question human reviewed: {full_df_humans[full_df_humans['question_id_original'] == idx]['question'].values}")
    print(f"Question machine transl: {full_df_machine[full_df_machine['question_id'] == idx]['question'].values}\n")

    if i == 3:
        break

    
# %%
# only keep the rows of full_df_machine that are in full_df_humans based on question_id for comparison

full_df_machine_subset = full_df_machine[full_df_machine['question_id'].isin(full_df_humans['question_id_original'].values)].reset_index(drop=True)
print(f"Dataset of machine translations shape: {full_df_machine_subset.shape}")
print(f"Dataset of human rev. translat. shape: {full_df_humans.shape}")

# print different columns
print(full_df_machine_subset.columns.difference(full_df_humans.columns))
print(full_df_humans.columns.difference(full_df_machine_subset.columns))

display(full_df_machine_subset.head(2), full_df_humans.head(2))

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

plt.figure(figsize=(12, 6))
sns.barplot(
    data=accuracy_overall_merged,
    x='display_name',
    y='Overall',
    hue='Translation'
)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Accuracy (%)')
plt.title('Overall Accuracy Comparison: Human Reviewed vs Machine Translations')
plt.legend(title='Translation Type')
plt.tight_layout()

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
# subplots per category 1x5 grid
fig, axs = plt.subplots(1, 5, figsize=(15,5), sharey=True)

for i, cat in enumerate(accuracy_cat_merged['Category'].unique()):

    acc_category_subset = accuracy_cat_merged[accuracy_cat_merged['Category'] == cat]

    sns.barplot(
        data=acc_category_subset,
        x='display_name',
        y='Accuracy',
        hue='Translation',
        ax=axs[i]
    )
    
    axs[i].set_title(cat, fontsize=12, fontweight='bold')
    # rotate x labels to 45
    # set number of x ticks to 5
    axs[i].set_xticks(acc_category_subset['display_name'].unique())
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.tight_layout()

# %%
