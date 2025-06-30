# %%
import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import json
from tqdm import tqdm
from fuzzywuzzy import fuzz
import yaml

from cdpk.benchmark_utils import create_subcsv_cdpk, process_csv_and_update_yaml, write_custom_yaml, get_few_shot_examples

homepath = Path.home()

from dotenv import load_dotenv
from fdllm.sysutils import register_models
from cdpk.benchmark_run import run_benchmark
from cdpk.benchmark_constants import ROOT

load_dotenv(override=True)
BENCHMARKS_DIR = Path(os.getenv("BENCHMARKS_DIR", "./"))

register_models(ROOT / "custom_models.yaml")

# %%
#def create_subsample_dataset(df, N_samples, by_category=True):
#    df_subsample = pd.DataFrame(columns=df.columns)
#    if by_category:
#        # check if there are enough samples per category
#        N_samples_per_category = N_samples // len(df["Category"].unique())
#        if N_samples_per_category > df["Category"].value_counts().min():
#            print("Not enough samples per category")
#            return None
#        for category in df["Category"].unique():
#            df_category = df[df["Category"] == category]
#            df_subsample = pd.concat([df_subsample, df_category.sample(N_samples_per_category, random_state=42)])
#    else:
#        df_subsample = df.sample(N_samples, random_state=42)
#    return df_subsample
#
#N_samples_exp = 210
#cdpk_dataset_subsample = create_subsample_dataset(cdpk_dataset, N_samples_exp, by_category=False)
#print(cdpk_dataset_subsample.shape)
#print(cdpk_dataset_subsample["Category"].value_counts())

# %%
# save as csv
#cdpk_dataset_subsample.to_csv(f"data/Chile/var experiments/cdpk_dataset_20241202_subsampled.csv", index=False)


# %%
######################################################################
# Correct answer shuffled
def shuffle_correct_answer_position(row, new_position):
    row_output = row.copy()
    # outputs the same row but with the correct answer in a different position
    old_position = row["Correct answer"]
    correct_answer = row[f"Answer {old_position}"]
    row_output[f"Answer {old_position}"] = row[f"Answer {new_position}"]
    row_output[f"Answer {new_position}"] = correct_answer
    row_output["Correct answer"] = new_position
    return row_output


# %%
# - for each category, read the csv test file and dev file
# - sample the test file (uniformly across categories or not) and create new csv in new folder
# - create a yaml file for each category (be careful with names: CDPK_dupli, CDPK_correct_ans_shuffled etc.)


# %%
# 1) Duplicate rows
# loop through each csv in folder data/Chile/CDPK_per_category/test
def create_csv_rows_duplicated(input_folder_test_files, output_folder, N_duplicates_per_mcqs, uniform_sampling_across_cat=False, perct_samples_to_keep=None):

    # create output folder if not exists
    if not os.path.exists(output_folder):
        print(f"Creating folder {output_folder}")
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder_test_files):
        if filename.endswith(".csv"):
            df_category = pd.read_csv(os.path.join(input_folder_test_files, filename))
            
            if perct_samples_to_keep is None:
                subsample = df_category
            else:
                # subsample
                if uniform_sampling_across_cat == False:
                    # Keep perct_samples_to_keep of the samples in each category
                    N_samples = int(perct_samples_to_keep * df_category.shape[0])
                    subsample = df_category.sample(N_samples, random_state=42).reset_index(drop=True)
                else:
                    # take the same number of samples in each category
                    folder_path = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile"
                    cdpk_dataset = pd.read_csv(folder_path / "cdpk_dataset_20241202.csv")
                    N_samples = int(perct_samples_to_keep * cdpk_dataset.shape[0]/cdpk_dataset.Category.nunique())
                    subsample = df_category.sample(min(len(df_category), N_samples), random_state=42).reset_index(drop=True)
                
                print(f"Subsampled {subsample.shape[0]} MCQs from {filename}")

            # duplicate rows
            df_rows_duplicated = pd.DataFrame(columns=subsample.columns)
            for _, row in subsample.iterrows():
                for _ in range(N_duplicates_per_mcqs):
                    df_rows_duplicated.loc[len(df_rows_duplicated)] = row
            
            # save as csv
            print(f"Total number of MCQs after duplication: {df_rows_duplicated.shape[0]}")
            new_filename = filename.replace("CDPK", "CDPK_dupli_rows")
            df_rows_duplicated.to_csv(os.path.join(output_folder, new_filename), index=False)
            print(f"Saved {new_filename}\n")

def create_new_yaml_files(csv_folder, output_folder, yaml_template, prefix="CDPK_dupli_rows"):
    # create output folder if not exists
    if not os.path.exists(output_folder):
        print(f"Creating folder {output_folder}")
        os.makedirs(output_folder)
    
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            # extract category by extracting the string between prefix and _test.csv
            category = re.search(f"{prefix}_(.*)_test", filename).group(1)

            new_yaml = yaml_template.copy()
            new_yaml["test_file"] = os.path.join(csv_folder.replace("data/", ""), filename)
            new_yaml["example_file"] = yaml_template['example_file'].replace("_category_", f"_{category}_")

            # define where to save the yaml file
            output_file_path = os.path.join(output_folder, filename.replace("_test.csv", ".yaml"))

            write_custom_yaml(new_yaml, output_file_path)
            print(f'Generated: {output_file_path}')       
            

# %%
create_csv_rows_duplicated(
    input_folder_test_files = "data/Chile/CDPK_per_category/test",
    output_folder = "data/Chile/CDPK_per_category/test_dupli_rows",
    N_duplicates_per_mcqs = 20,
    uniform_sampling_across_cat=False,
    #perct_samples_to_keep=0.2
)

# %%
yaml_template_cdpk_original = {
    "test_file": "",
    "test_header": 0,
    "example_file": "Chile\CDPK_per_category\dev\CDPK_category_dev.csv",
    "example_header": 0,
    "choices": ["A", "B", "C", "D", "E", "F", "G"],
    "choice_cols": [2, 3, 4, 5, 6, 7, 8],
    "answer_col": 9,
    "question_col": 1,
    "example_rows": [0, 1, 2]
}

create_new_yaml_files(
    csv_folder = "data/Chile/CDPK_per_category/test_dupli_rows",
    output_folder = "data/Chile/CDPK_per_category/yaml_dupli_rows",
    yaml_template = yaml_template_cdpk_original,
    prefix="CDPK_dupli_rows"
)




# %%
# 2) Correct answer shuffled
def create_csv_correct_ans_shuffled(input_folder_test_files, output_folder, uniform_sampling_across_cat=False, perct_samples_to_keep=None):

    # create output folder if not exists
    if not os.path.exists(output_folder):
        print(f"Creating folder {output_folder}")
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder_test_files):
        if filename.endswith(".csv"):
            df_category = pd.read_csv(os.path.join(input_folder_test_files, filename))
            
            if perct_samples_to_keep is None:
                subsample = df_category
            else:
                # subsample
                if uniform_sampling_across_cat == False:
                    # Keep perct_samples_to_keep of the samples in each category
                    N_samples = int(perct_samples_to_keep * df_category.shape[0])
                    subsample = df_category.sample(N_samples, random_state=42).reset_index(drop=True)
                else:
                    # take the same number of samples in each category
                    folder_path = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile"
                    cdpk_dataset = pd.read_csv(folder_path / "cdpk_dataset_20241202.csv")
                    N_samples = int(perct_samples_to_keep * cdpk_dataset.shape[0]/cdpk_dataset.Category.nunique())
                    subsample = df_category.sample(min(len(df_category), N_samples), random_state=42).reset_index(drop=True)
                
                print(f"Subsampled {subsample.shape[0]} MCQs from {filename}")

            # Shuffle correct answer position
            df_ans_shuffled = pd.DataFrame(columns=subsample.columns)
            for _, row in subsample.iterrows():
                # save original correct answer
                original_correct_answer = row["Correct answer"]
                df_ans_shuffled.loc[len(df_ans_shuffled)] = row
                for new_position in ['A', 'B', 'C', 'D']:
                    if new_position == original_correct_answer:
                        continue
                    else:
                        new_row = shuffle_correct_answer_position(row, new_position)
                        df_ans_shuffled.loc[len(df_ans_shuffled)] = new_row

            # save as csv
            print(f"Total number of MCQs after shuffling correct answer: {df_ans_shuffled.shape[0]}")
            new_filename = filename.replace("CDPK", "CDPK_ans_shuffled")
            df_ans_shuffled.to_csv(os.path.join(output_folder, new_filename), index=False)
            print(f"Saved {new_filename}\n")


# %%
create_csv_correct_ans_shuffled(
    input_folder_test_files = "data/Chile/CDPK_per_category/test",
    output_folder = "data/Chile/CDPK_per_category/test_ans_shuffled",
    #uniform_sampling_across_cat=False,
    #perct_samples_to_keep=0.2
)

# %%
yaml_template_cdpk_original = {
    "test_file": "",
    "test_header": 0,
    "example_file": "Chile\CDPK_per_category\dev\CDPK_category_dev.csv",
    "example_header": 0,
    "choices": ["A", "B", "C", "D", "E", "F", "G"],
    "choice_cols": [2, 3, 4, 5, 6, 7, 8],
    "answer_col": 9,
    "question_col": 1,
    "example_rows": [0, 1, 2]
}

create_new_yaml_files(
    csv_folder = "data/Chile/CDPK_per_category/test_ans_shuffled",
    output_folder = "data/Chile/CDPK_per_category/yaml_ans_shuffled",
    yaml_template = yaml_template_cdpk_original,
    prefix="CDPK_ans_shuffled"
)



# %%
######################################################################
# Random fews shot examples

# todo





# %%
######################################################################
# Analyze results
#DIR_CACHE = "data\cache"
#DIR_CONFIGS = "\configs"
#
## get name of files in configs / questions folder
#def get_names_files(folder, pattern="CDPK_"):
#    files = []
#    for filename in os.listdir(folder):
#        if filename.startswith(pattern):
#            files.append(filename.replace(".yaml", ""))
#    return files

#cdpk_cat = get_names_files("./configs/questions", pattern="CDPK_dupli_rows")
#print(cdpk_cat)

def collect_full_df_var_experiment(experiment="", config_models_PK="fast_cheap"):
    config_models_PK = config_models_PK

    if (experiment == "") or (experiment == "dupli_rows_") or (experiment == "ans_shuffled_"):
        pass
    else:
        print("Experiment name should be either 'dupli_rows_' or 'ans_shuffled_'")
        return None

    CDPK_list_dupli_rows = [
        f"CDPK_{experiment}science",
        f"CDPK_{experiment}literacy",
        f"CDPK_{experiment}creative_arts",
        f"CDPK_{experiment}maths",
        f"CDPK_{experiment}social_studies",
        f"CDPK_{experiment}technology",
        f"CDPK_{experiment}gen_pk",
    ]
    #
    full_df_chile = pd.DataFrame()
    accuracies_df = pd.DataFrame()
    bad_format_df = pd.DataFrame()
    length_per_category = {}
    # summary_df_chile = pd.DataFrame()
    for cat_config_name in CDPK_list_dupli_rows:
        print(f"Running CDPK benchmark for {cat_config_name}")
        full_df, summary_df, config, models_dict_PK = run_benchmark(
            questions_config=cat_config_name, models_config=config_models_PK
        )
        if len(full_df["Category"].unique()) > 1:
            print("More than one category in the dataframe")
            break
        else:
            category_name = full_df["Category"].unique()[0]
            print(f"Category: {category_name}")

        # concatenate
        full_df_chile = pd.concat([full_df_chile, full_df], ignore_index=True)
        # concatenate accuracies and bad_format for each category and give name to the columns
        accuracies_df[category_name] = summary_df.loc["accuracy"]
        bad_format_df[category_name] = summary_df.loc["bad_format"]

        # get the length of the dataframe for each category (removing few-shot examples)
        length_per_category[category_name] = len(full_df) - len(config["example_rows"])

    col_pred = [col for col in full_df_chile.columns if col.startswith("pred_")][0]
    full_df_chile_wo_fs = full_df_chile[full_df_chile[col_pred] != "Few-shot example"]

    print(f"Number of MCQs in the full dataframe: {len(full_df_chile)}")
    print(f"Number of MCQs in the full dataframe without few-shot examples: {len(full_df_chile_wo_fs)}")

    return full_df_chile, full_df_chile_wo_fs, accuracies_df, bad_format_df, length_per_category, models_dict_PK

# %%
# 1) Collect results for Duplicated rows experiment (20 runs per MCQ)
_, full_df_chile_wo_fs_dupli_rows, _, _, _, models_dict_PK = collect_full_df_var_experiment(experiment="dupli_rows_", 
                                                                            config_models_PK="fast_cheap")
print(full_df_chile_wo_fs_dupli_rows.shape)

# %%
# Format data to plot results of experiment with duplicated rows

df_dupli_rows = pd.DataFrame(columns = ["Category", "Model", "Accuracy", "Run", "N_samples"])
N_dupli_per_mcqs = 20

for cat in full_df_chile_wo_fs_dupli_rows["Category"].unique():
    df_cat = full_df_chile_wo_fs_dupli_rows[full_df_chile_wo_fs_dupli_rows["Category"] == cat].reset_index(drop=True)
    for model in [col for col in full_df_chile_wo_fs_dupli_rows.columns if col.startswith("pred_")]:
        for run in range(N_dupli_per_mcqs):
            # gather question of run N
            #print(len(df_cat)/N_dupli_per_mcqs)
            indices, step = np.linspace(run, len(df_cat)-1, num=int(len(df_cat)/N_dupli_per_mcqs), dtype=int, retstep=True)       
            df_run = df_cat.iloc[indices]
            # compute accuracy and bad format
            accuracy = (df_run["Correct answer"] == df_run[model]).mean()
            # update dataframe
            df_dupli_rows.loc[len(df_dupli_rows)] = [cat, model, accuracy, run, len(df_run)]
            #print(indices)

df_dupli_rows["Accuracy"] = df_dupli_rows["Accuracy"] * 100
print(df_dupli_rows.shape)
df_dupli_rows.head()

# %%
fig, ax = plt.subplots(figsize=(13, 6))
# barplot
ax = sns.barplot(x="Model", 
                 y="Accuracy", 
                 data=df_dupli_rows, 
                 hue = "Category",
                 #order = list(models_dict_PK.values()),
                 #errorbar='sd'
                 errorbar = lambda x: (x.min(), x.max()))
# boxplot
#ax = sns.boxplot(x="Model", 
#                 y="Accuracy", 
#                 data=df_dupli_rows, 
#                 hue = "Category", 
#                 showmeans=True, 
#                 meanprops={"marker":"x","markerfacecolor":"black", "markeredgecolor":"black", "markersize":"5"})

renamed_models = []
for model in models_dict_PK.values():
    if model == "Claude-3.5 Sonnet Oct. 24":
        renamed_models.append("Claude-3.5 Sonnet\nOct. 24     ")
    else:
        renamed_models.append(model)
ax.set_xticklabels(renamed_models, rotation=45, fontsize=14, ha = 'right')
ax.set_ylabel("Accuracy (%)", fontsize=14)
ax.set_xlabel("", fontsize=14)
ax.set_ylim(0, 95)
ax.set_title("CDPK Benchmark Performance (N=20 runs/MCQ)", fontsize=18, fontweight='bold')
#ax.legend(title="Category", bbox_to_anchor=(1.01, 0.75), loc='upper left')

# Create and store the first legend
legend1 = ax.legend(title="Category", bbox_to_anchor=(1, 0.8), loc='upper left')
# Add a second legend for error bars
legend_elements = [Line2D([0], [0], color='black', lw=1, label='min/max')]
# Add the first legend back as an artist so it stays
ax.add_artist(legend1)
# Now create the second legend
ax.legend(handles=legend_elements, title="Error bars", bbox_to_anchor=(1.05, 0.25), loc='upper left')
#plt.tight_layout()
plt.show()
# %%






# %%
# 2) Collect results for Shuffling correct answer position experiment
_, full_df_chile_wo_fs_ans_shuffled, _, _, _, models_dict_PK = collect_full_df_var_experiment(experiment="ans_shuffled_", 
                                                                            config_models_PK="fast_cheap")
print(full_df_chile_wo_fs_ans_shuffled.shape)

# %%
# Format data to plot results of experiment with correct answer shuffled

df_ans_shuffled = pd.DataFrame(columns = ["Category", "Model", "Accuracy", "Correct answer", "N_samples"])

for cat in full_df_chile_wo_fs_ans_shuffled["Category"].unique():
    df_cat = full_df_chile_wo_fs_ans_shuffled[full_df_chile_wo_fs_ans_shuffled["Category"] == cat].reset_index(drop=True)
    for model in [col for col in full_df_chile_wo_fs_ans_shuffled.columns if col.startswith("pred_")]:
        for pos in ['A', 'B', 'C', 'D']:
            df_pos = df_cat[df_cat["Correct answer"] == pos]
            accuracy = (df_pos["Correct answer"] == df_pos[model]).mean()
            df_ans_shuffled.loc[len(df_ans_shuffled)] = [cat, model, accuracy, pos, len(df_pos)]

df_ans_shuffled["Accuracy"] = df_ans_shuffled["Accuracy"] * 100
print(df_ans_shuffled.shape)
df_ans_shuffled.head()

# %%
fig, ax = plt.subplots(figsize=(13, 6))
# barplot
hue = "Correct answer"
#hue = "Category"
type_error = "minmax"
type_error = "sd"

if type_error == "minmax":
    errorbar = lambda x: (x.min(), x.max())
    label_errorbar = "min/max"
else:
    errorbar = "se"
    label_errorbar = "std err"
ax = sns.barplot(x="Model", 
                 y="Accuracy", 
                 data=df_ans_shuffled, 
                 hue = hue,
                 #order = list(models_dict_PK.values()),
                 errorbar = errorbar)
#for i in range(df_ans_shuffled[hue].nunique()):
#    ax.bar_label(ax.containers[i], fontsize = 11, fmt="%d", label_type="center")
# boxplot
#ax = sns.boxplot(x="Model", 
#                 y="Accuracy", 
#                 data=df_ans_shuffled, 
#                 hue = "Category", 
#                 showmeans=True, 
#                 meanprops={"marker":"x","markerfacecolor":"black", "markeredgecolor":"black", "markersize":"5"})

#renamed_models = df_ans_shuffled.Model.unique()
renamed_models = []
for model in models_dict_PK.values():
    if model == "Claude-3.5 Sonnet Oct. 24":
        renamed_models.append("Claude-3.5 Sonnet\nOct. 24     ")
    else:
        renamed_models.append(model)
ax.set_xticklabels(renamed_models, rotation=45, fontsize=14, ha = 'right')
ax.set_ylabel("Accuracy (%)", fontsize=14)
ax.set_xlabel("", fontsize=14)
ax.set_ylim(0, 95)
ax.set_title("CDPK Benchmark Performance (Correct answer shuffled)", fontsize=18, fontweight='bold')
#ax.legend(title="Category", bbox_to_anchor=(1.01, 0.75), loc='upper left')

# Create and store the first legend
legend1 = ax.legend(title=hue, bbox_to_anchor=(1, 0.8), loc='upper left')
# Add a second legend for error bars
legend_elements = [Line2D([0], [0], color='black', lw=1, label=label_errorbar)]
# Add the first legend back as an artist so it stays
ax.add_artist(legend1)
# Now create the second legend
ax.legend(handles=legend_elements, title="Error bars", bbox_to_anchor=(1.05, 0.25), loc='upper left')
#plt.tight_layout()
plt.show()



# %%
# compare to original CDPK benchmark
_, full_df_chile_wo_fs, accuracies_df_original, _, _, models_dict_PK = collect_full_df_var_experiment(experiment="", 
                                                                            config_models_PK="fast_cheap")
print(full_df_chile_wo_fs.shape, accuracies_df_original.shape)

# %%
df_ans_shuffled_avg = pd.DataFrame(df_ans_shuffled.groupby(["Category", "Model"])['Accuracy'].apply(lambda x: x.mean())).reset_index().rename(columns = {"Accuracy": "Accuracy (Mean 4)"})
df_ans_shuffled_avg.head()

# %%
# process dataframe for reliability 4/4

df_process = full_df_chile_wo_fs_ans_shuffled.groupby(["Question"]).apply(lambda x: pd.Series({
    "Correct answer": x["Correct answer"].values,
    "Category": x["Category"].unique()[0] if len(x["Category"].unique()) == 1 else "Multiple",
    **{f"pred_{model}": x[f"pred_{model}"].values for model in models_dict_PK}
})).reset_index()
# check
#print(test.Category.value_counts())
# check full_df_chile_wo_fs_ans_shuffled and test have same values in column "Question", without considering the order
print(set(full_df_chile_wo_fs_ans_shuffled["Question"]) == set(df_process["Question"]))
print(df_process.shape)
df_process.head()

# %%
df_reliability_4_4 = pd.DataFrame(columns = ["Category", "Model", "Accuracy (4/4)"])

for cat in df_process["Category"].unique():
    df_cat = df_process[df_process["Category"] == cat].reset_index(drop=True)
    for model in models_dict_PK:
        acc4_cat = []
        for i, row in df_cat.iterrows():
            acc4_cat.append((row["Correct answer"] == row[f"pred_{model}"]).all())
        df_reliability_4_4.loc[len(df_reliability_4_4)] = [cat, model, np.mean(acc4_cat)*100]
print(df_reliability_4_4.shape)
df_reliability_4_4.head()

# %%
# append the original acc in new column "Accuracy"
for i, row in df_ans_shuffled_avg.iterrows():
    model = row["Model"].replace("pred_", "")
    df_ans_shuffled_avg.loc[i, "Accuracy (1/4)"] = accuracies_df_original.loc[model, row["Category"]]*100
    df_ans_shuffled_avg.loc[i, "Accuracy (4/4)"] = df_reliability_4_4[(df_reliability_4_4["Model"] == model) & (df_reliability_4_4["Category"] == row["Category"])]["Accuracy (4/4)"].values[0]
df_ans_shuffled_avg.head()

# %%
# melt df
df_ans_shuffled_avg_melted =  pd.melt(
        df_ans_shuffled_avg,
        id_vars=['Category', 'Model'],          # Columns to keep fixed
        value_vars=['Accuracy (1/4)', 'Accuracy (4/4)', 'Accuracy (Mean 4)'],  # Columns to "melt" into rows
        var_name='Acc Type',                    # New column name for variable
        value_name='Acc Value'                  # New column name for the numeric value
    )
df_ans_shuffled_avg_melted.head()

# %%
# Plot difference mean accuracy when shuffling correct answer position and original results
fig, ax = plt.subplots(figsize=(16, 7))
type_error = "minmax"
#type_error = "sd"

if type_error == "minmax":
    errorbar = lambda x: (x.min(), x.max())
    label_errorbar = "min/max"
else:
    errorbar = "se"
    label_errorbar = "std err"
ax = sns.barplot(x="Model", 
                 y="Acc Value", 
                 data=df_ans_shuffled_avg_melted, 
                 hue = "Acc Type",
                 errorbar = errorbar)
for i in range(df_ans_shuffled_avg_melted["Acc Type"].nunique()):
    ax.bar_label(ax.containers[i], fontsize=11, label_type="center", labels=[round(val) for val in ax.containers[i].datavalues])

renamed_models = []
#for model in models_dict_PK.values():
#    if model == "Claude-3.5 Sonnet Oct. 24":
#        renamed_models.append("Claude-3.5 Sonnet\nOct. 24     ")
#    else:
#        renamed_models.append(model)
for model in df_ans_shuffled_avg_melted.Model.unique():
    model_name = model.replace("pred_", "")
    model_name_clean = models_dict_PK[model_name]
    if model_name_clean == "pred_claude-3-5-sonnet-20241022":
        renamed_models.append("Claude-3.5 Sonnet\nOct. 24     ")
    else:
        renamed_models.append(model_name_clean)
ax.set_xticklabels(labels = renamed_models, rotation=45, fontsize=14, ha = 'right')
ax.set_ylabel("Accuracy (%)", fontsize=14)
ax.set_xlabel("", fontsize=14)
ax.set_ylim(0, 95)
ax.set_title("CDPK Benchmark Performance\nCorrect Answer Position Experiment", fontsize=18, fontweight='bold')
#ax.legend(title="Category", bbox_to_anchor=(1.01, 0.75), loc='upper left')

# Create and store the first legend
legend1 = ax.legend(title="Reliability", bbox_to_anchor=(1, 0.8), loc='upper left')
# Add a second legend for error bars
legend_elements = [Line2D([0], [0], color='black', lw=1, label=label_errorbar)]
# Add the first legend back as an artist so it stays
ax.add_artist(legend1)
# Now create the second legend
ax.legend(handles=legend_elements, title="Error bars", bbox_to_anchor=(1.05, 0.25), loc='upper left')
#plt.tight_layout()
plt.show()










# %%
letter = "B"
model = "pred_claude-3-5-sonnet-20241022"

df_letter_model = full_df_chile_wo_fs_ans_shuffled[full_df_chile_wo_fs.columns[:15].tolist() + [model]]
df_letter_model = df_letter_model[df_letter_model["Correct answer"] == letter]
print(df_letter_model.shape)
df_letter_model.head()
# %%
#compute accuracy for each category
df_letter_model.groupby("Category")[model].apply(lambda x: (x == letter).mean())
# %%
df_letter_model.groupby("Category")[model].value_counts()
# %%
# compare to df_ans_shuffled
df_ans_shuffled[(df_ans_shuffled["Model"] == model) & (df_ans_shuffled["Correct answer"] == letter)].head(7)

# %%
df_ans_shuffled[df_ans_shuffled["Model"] == model].head()
# %%
