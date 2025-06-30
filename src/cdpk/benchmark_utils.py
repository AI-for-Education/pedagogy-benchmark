from pathlib import Path
import os

import numpy as np
import pandas as pd

from .benchmark_run import run_benchmark
from .benchmark_constants import ROOT


def collect_results_mmlu(models_config):

    accuracies_mmlu = pd.DataFrame()
    bad_format_mmlu = pd.DataFrame()
    nquestion_mmlu = pd.DataFrame()
    full_df_mmlu_list = []
    # collect results from MMLU
    for mmlu_questions_config in (ROOT / "configs/questions").rglob("MMLU*"):
        # if mmlu_questions_config.stem == "MMLU_virology":
        #     continue
        # get the name of the config file
        mmlu_cat_name = mmlu_questions_config.stem
        # print(mmlu_cat_name)

        full_df_mmlu, summary_df_mmlu, config_mmlu, models_mmlu = run_benchmark(
            questions_config=mmlu_cat_name, models_config=models_config
        )
        # full_df = pd.concat([full_df, full_df_mmlu])
        accuracies_mmlu[mmlu_cat_name] = summary_df_mmlu.loc["accuracy"]
        bad_format_mmlu[mmlu_cat_name] = summary_df_mmlu.loc["bad_format"]
        nquestion_mmlu[mmlu_cat_name] = pd.Series(
            index=summary_df_mmlu.columns, data=len(full_df_mmlu.iloc[5:])
        )
        full_df_mmlu["Category"] = mmlu_cat_name
        full_df_mmlu_list.append(
            full_df_mmlu.iloc[5:].rename(columns={5: "Correct answer"})
        )

    full_df_mmlu = pd.concat(full_df_mmlu_list, axis=0, ignore_index=True)

    return accuracies_mmlu, bad_format_mmlu, nquestion_mmlu, full_df_mmlu, models_mmlu


def fulldf_accuracy_by_category(
    fulldf, models_dict, category=None, column="Category", bad_format_threshold=0.05
):
    if category is None:
        category = sorted(fulldf[column].dropna().unique())
    elif not isinstance(category, list):
        category = [category]

    fulldf = pd.concat(
        [fulldf[fulldf[column] == cat] for cat in category],
        axis=0,
        ignore_index=True,
    )

    accuracies_detailed = {}
    bad_format_detailed = {}
    nquestions_detailed = {}

    for model in models_dict.keys():
        accuracies_detailed[model] = (
            fulldf[f"pred_{model}"] == fulldf["Correct answer"]
        ).mean()
        bad_format_detailed[model] = fulldf[f"pred_{model}"].isna().mean()
        nquestions_detailed[model] = len(fulldf)
    accuracies_detailed = pd.Series(accuracies_detailed) * 100
    bad_format_detailed = pd.Series(bad_format_detailed) * 100
    nquestions_detailed = pd.Series(nquestions_detailed)

    accuracies_detailed = pd.concat(
        [accuracies_detailed, bad_format_detailed, nquestions_detailed], axis=1
    ).rename(
        columns={i: nm for i, nm in enumerate(["Accuracy", "Bad Format", "N"])}
    )
    accuracies_detailed.index.name = "Model"

    if bad_format_threshold is not None:
        # remove lines with values above the bad format threshold
        models_bad_format = accuracies_detailed.loc[
            accuracies_detailed["Bad Format"] >= bad_format_threshold*100
        ].index
        accuracies_detailed = accuracies_detailed.drop(models_bad_format)
        if len(models_bad_format) > 0:
            print(
                f"Models with bad format above {bad_format_threshold * 100:.0f}%: {models_bad_format.tolist()}"
            )

    return accuracies_detailed

def create_subcsv_cdpk(df, folder_path, csv_filename_prefix, categories, levels):

    # check if dir exists and create it if not
    if not os.path.exists(folder_path):
        print(f"Creating folder {folder_path}, test and dev folders")
        os.makedirs(folder_path)
        os.makedirs(os.path.join(folder_path, "test"))
        os.makedirs(os.path.join(folder_path, "dev"))

    # check categories in the dataset
    for category in categories:
        if category not in df["Category"].unique():
            print(f"Category {category} not in the dataset!")
            raise ValueError
        else:
            if levels is None:
                sub_df = df[df["Category"] == category].reset_index(drop=True)
                #save_path = Path(folder_path) / f"CDPK_{category.replace(' PCK', '').replace(' ', '_').lower()}.csv"

                # split df into df_test and df_few_shot
                idx_few_shot = get_few_shot_examples(sub_df, n_examples=3)
                df_test = sub_df.drop(index=idx_few_shot).reset_index(drop=True)
                df_few_shot = sub_df.loc[idx_few_shot].reset_index(drop=True)

                path_test_file = Path(folder_path) / "test" / f"{csv_filename_prefix}_{category.replace(' PCK', '').replace(' ', '_').lower()}_test.csv"
                path_few_shot_file = Path(folder_path) / "dev" / f"{csv_filename_prefix}_{category.replace(' PCK', '').replace(' ', '_').lower()}_dev.csv"

                df_test.to_csv(path_test_file, index=False)
                df_few_shot.to_csv(path_few_shot_file, index=False)
                print(f"Saved {category} test and dev files!")
            else:
                print("Levels division not implemented yet!")
                return None
                #for level in levels:
                #    if level not in df["Age Group"].unique():
                #        print(f"Level {level} not in the dataset!")
                #        raise ValueError
                #    else:
                #        if df[(df["Category"] == category) & (df["Age Group"] == level)].shape[0] == 0:
                #            print(f"    No MCQs for {category} - {level}")
                #            break
                #        else:
                #            sub_df = df[(df["Category"] == category) & (df["Age Group"] == level)].reset_index(drop=True)
                #            save_path = Path(folder_path) / f"{csv_filename_prefix}_{level.lower()}_{category.replace(' PCK', '').lower()}.csv"


def write_custom_yaml(data, filepath):
    # Manually construct the YAML output as a string to ensure exact formatting
    #choices_str = "[" + ", ".join(f'"{choice}"' for choice in data["choices"]) + "]"
    choices_str = "[" + ", ".join(f'{choice}' for choice in data["choices"]) + "]" # without "" around letters
    choice_cols_str = "[" + ", ".join(map(str, data["choice_cols"])) + "]"
    example_rows_str = "[" + ", ".join(map(str, data["example_rows"])) + "]"

    test_file_path = data['test_file'].replace('/', '\\')
    example_file_path = data['example_file'].replace('/', '\\')

    yaml_content = (
        f"test_file: {test_file_path}\n"
        f"test_header: {data['test_header']}\n"
        f"example_file: {example_file_path}\n"
        f"example_header: {data['example_header']}\n"
        f"choices: {choices_str}\n"
        f"choice_cols: {choice_cols_str}\n"
        f"answer_col: {data['answer_col']}\n"
        f"question_col: {data['question_col']}\n"
        f"example_rows: {example_rows_str}"
    )

    # Write to file
    with open(filepath, 'w') as file:
        file.write(yaml_content)

def get_few_shot_examples(df, n_examples):
    # check that n_examples is a multiple of 3
    if n_examples % 3 != 0:
        print("n_examples must be a multiple of 3!")
        return None
    # Take n_examples from each age group
    primary_list = ["Primary", "Primary, Secondary", "Pre-primary, Primary", "All"]
    secondary_list = ["Secondary", "Primary, Secondary", "All"]
    preprimary_list = ["Pre-primary", "Pre-primary, Primary", "All"]

    df_preprimary = df[df["Age Group"].isin(preprimary_list)]
    df_primary = df[df["Age Group"].isin(primary_list)]
    df_secondary = df[df["Age Group"].isin(secondary_list)]

    # check if there are enough examples in each age group
    print("Number of examples in each age group:\n"
          f"Pre-primary: {df_preprimary.shape[0]} samples,\n"
          f"Primary: {df_primary.shape[0]} samples,\n"
          f"Secondary: {df_secondary.shape[0]} samples,\n")

    idx_few_shot = []
    for df_age in [df_preprimary, df_primary, df_secondary]:
        # select random n_examples/3 from each age group while answering the indices selected are unique
        idx_few_shot.extend(df_age.sample(n=n_examples//3, random_state = 42).index.tolist())
        # check that we did not sample the same example twice
        while len(set(idx_few_shot)) < len(idx_few_shot):
            print("Duplicates found! Re-sampling...")
            # remove the last example(s) added
            idx_few_shot = idx_few_shot[:int(-n_examples//3)]
            idx_few_shot.extend(df_age.sample(n=n_examples//3, random_state = 42).index.tolist())
    # check
    #display(df.loc[idx_few_shot, ["Question", "Age Group"]])
    return idx_few_shot
        


def process_csv_and_update_yaml(input_folder, output_folder, yaml_template):
    """
    Reads each CSV file in the input folder, updates the 'test_file' field in the YAML file,
    and saves the updated YAML in the specified output folder.

    Args:
        input_folder (str): Path to the folder containing CSV files.
        output_folder (str): Path to the folder where updated YAML files will be saved.
        yaml_template (dict): Template for the YAML file as a Python dictionary.
    """
    #default_path_test_file = r"Chile\CDPK_per_category"

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # update the input folder by adding /test
    input_folder_test = os.path.join(input_folder, "test")
    input_folder_dev = os.path.join(input_folder, "dev")

    # Iterate over all CSV files in the folder
    for filename in os.listdir(input_folder_test):
        if filename.endswith('.csv'):
            test_file = os.path.join(input_folder_test, filename)
            example_file = os.path.join(input_folder_dev, filename.replace("test", "dev"))

            # Set the test_file and example_file fields
            yaml_content = yaml_template.copy()
            # remove first folder in path data/
            yaml_content["test_file"] = test_file.replace("data/", "")
            yaml_content["example_file"] = example_file.replace("data/", "")

            # read example csv file to get number of examples
            df = pd.read_csv(os.path.join(input_folder_dev, filename.replace("test", "dev")))
            # get index of few shot examples
            #idx_few_shot = get_few_shot_examples(df, n_examples=3)
            #yaml_content["example_rows"] = idx_few_shot
            yaml_content["example_rows"] = np.arange(len(df)).tolist()

            
            # define where to save the yaml file
            output_file_path = os.path.join(output_folder, filename.replace("_test.csv", ".yaml"))

            # Write to YAML file with customed function to respect formatting
            write_custom_yaml(yaml_content, output_file_path)
            print(f'Generated: {output_file_path}')