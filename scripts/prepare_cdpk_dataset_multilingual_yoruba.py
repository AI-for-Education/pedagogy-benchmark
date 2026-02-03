# %%
import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import yaml

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# %%
# Load dataset
#######################

cdpk_dataset = pd.read_csv(DATA_DIR / "pedagogy_benchmark_yoruba_cdpk_cleaned.csv")
print(cdpk_dataset.shape)
cdpk_dataset.head(1)

# %%
# - format columns: 
#   - add source, add Answer E, add Answer F, add Answer G (before Correct answer)
#   - remove question number
cdpk_dataset.insert(0, "Source", "Pedagogy Benchmark Yoruba") # insert source
cdpk_dataset.drop(columns = ["question_id"], inplace=True) # remove question number
#cdpk_dataset.insert(6, "Answer E", None)
#cdpk_dataset.insert(7, "Answer F", None)
#cdpk_dataset.insert(8, "Answer G", None)


# %%
# Last check before saving
print(cdpk_dataset.columns.tolist())
print(cdpk_dataset.shape)
cdpk_dataset.head(2)


#########################################################################################
# %%
## SEND mcqs ##
# get SEND mcqs
send_dataset = pd.read_csv(DATA_DIR / "pedagogy_benchmark_luganda_send.csv")
print(send_dataset.shape)
send_dataset.head(1)


# %%
### Format columns
send_dataset.insert(0, "Source", "SEND Benchmark Luganda") # insert source
send_dataset.drop(columns = ["question_id"], inplace=True) # remove question number
#send_dataset.insert(6, "Answer E", None)
#send_dataset.insert(7, "Answer F", None)
#send_dataset.insert(8, "Answer G", None)
# %%
# Last check before saving
print(send_dataset.columns.tolist())
print(send_dataset.shape)
send_dataset.head(2)

# %%
# add date of today to the name
#today = pd.Timestamp.today().strftime("%Y%m%d")
#df_send_3.to_csv(folder_path / f"cdpk_dataset_send_{today}.csv", index=False)
#df_send_3.to_csv(f"./data/Chile/cdpk_dataset_send_{today}.csv", index=False)


#########################################################################################

## Create config files with FS examples per category ##

cdpk_categories = cdpk_dataset["category"].unique()
# %%
# Read json file
with open(DATA_DIR / "few_shot_examples_idx_dict.json", "r") as f:
    few_shot_examples_idx_dict = json.load(f)

renamed_keys_dict = {
    "CDPK_science": "Science",
    "CDPK_literacy": "Literacy",
    "CDPK_creative_arts": "Creative arts",
    "CDPK_maths": "Maths",
    "CDPK_social_studies": "Social studies",
    "CDPK_technology": "Technology",
    "CDPK_gen_pk": "General",
    "CDPK_send": "SEND",
}

# invert the dictionary
few_shot_examples_idx_dict = {
    renamed_keys_dict[key]: value for key, value in few_shot_examples_idx_dict.items()
}
#print(json.dumps(few_shot_examples_idx_dict, indent=2))

def create_subcsv_cdpk(df, folder_path, categories, levels, suffix=""):

    # Create folder if it does not exist
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    Path(folder_path + "/test").mkdir(parents=True, exist_ok=True)
    Path(folder_path + "/dev").mkdir(parents=True, exist_ok=True)

    # check categories in the dataset
    for category in categories:
        if category not in df["category"].unique():
            print(f"Category {category} not in the dataset!")
            raise ValueError
        else:
            if levels is None:
                #save_path = Path(folder_path) / f"CDPK_{category.replace(' PCK', '').replace(' ', '_').lower()}.csv"

                # split df into df_test and df_few_shot
                #idx_few_shot = get_few_shot_examples_new(sub_df, n_examples=3)
                idx_few_shot = few_shot_examples_idx_dict[category]

                df_few_shot = df.loc[idx_few_shot].reset_index(drop=True)
                # Remove the few shot examples from the main df, as the indices refer to the original df
                df_wo_fs = df.drop(index=idx_few_shot).reset_index(drop=True)
                # Get the sub_df for the current category, which does not have the few-shot examples
                sub_df = df_wo_fs[df_wo_fs["category"] == category].reset_index(drop=True)
                df_test = sub_df

                # Test
                if df_few_shot['category'].nunique() != 1 or df_few_shot['category'].unique()[0] != category:
                    print("Few-shot examples do not contain the right category!")
                    raise ValueError
                if df_test['category'].nunique() != 1 or df_test['category'].unique()[0] != category:
                    print("Test examples do not contain the right category!")
                    print(df_test['category'].value_counts())
                    raise ValueError

                path_test_file = Path(folder_path) / "test" / f"CDPK_{suffix}_{category.replace(' PCK', '').replace(' ', '_').lower()}_test.csv"
                path_few_shot_file = Path(folder_path) / "dev" / f"CDPK_{suffix}_{category.replace(' PCK', '').replace(' ', '_').lower()}_dev.csv"

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
                #            save_path = Path(folder_path) / f"CDPK_{level.lower()}_{category.replace(' PCK', '').lower()}.csv"


def write_custom_yaml(data, filepath):
    # Manually construct the YAML output as a string to ensure exact formatting
    #choices_str = "[" + ", ".join(f'"{choice}"' for choice in data["choices"]) + "]"
    choices_str = "[" + ", ".join(f'{choice}' for choice in data["choices"]) + "]" # without "" around letters
    choice_cols_str = "[" + ", ".join(map(str, data["choice_cols"])) + "]"
    example_rows_str = "[" + ", ".join(map(str, data["example_rows"])) + "]"

    yaml_content = (
        f"test_file: {data['test_file'].replace('/', '\\')}\n"
        f"test_header: {data['test_header']}\n"
        f"example_file: {data['example_file'].replace('/', '\\')}\n"
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

#def get_few_shot_examples(df, n_examples):
#    # check that n_examples is a multiple of 3
#    if n_examples % 3 != 0:
#        print("n_examples must be a multiple of 3!")
#        return None
#    # Take n_examples from each age group
#    primary_list = ["Primary", "Primary, Secondary", "Pre-primary, Primary", "All"]
#    secondary_list = ["Secondary", "Primary, Secondary", "All"]
#    preprimary_list = ["Pre-primary", "Pre-primary, Primary", "All"]
#
#    df_preprimary = df[df["age_group"].isin(preprimary_list)]
#    df_primary = df[df["age_group"].isin(primary_list)]
#    df_secondary = df[df["age_group"].isin(secondary_list)]
#
#    # check if there are enough examples in each age group
#    print("Number of examples in each age group:\n"
#          f"Pre-primary: {df_preprimary.shape[0]} samples,\n"
#          f"Primary: {df_primary.shape[0]} samples,\n"
#          f"Secondary: {df_secondary.shape[0]} samples,\n")
#
#    idx_few_shot = []
#    for df_age in [df_preprimary, df_primary, df_secondary]:
#        # select random n_examples/3 from each age group while answering the indices selected are unique
#        idx_few_shot.extend(df_age.sample(n=n_examples//3).index.tolist())
#        # check that we did not sample the same example twice
#        while len(set(idx_few_shot)) < len(idx_few_shot):
#            print("Duplicates found! Re-sampling...")
#            # remove the last example(s) added
#            idx_few_shot = idx_few_shot[:int(-n_examples//3)]
#            idx_few_shot.extend(df_age.sample(n=n_examples//3).index.tolist())
#    # check
#    #print(df.loc[idx_few_shot, ["question", "age_group"]])
#    return idx_few_shot



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
            # Make sure path starts from language folder
            yaml_content["test_file"] = test_file.split("data/")[-1]
            yaml_content["example_file"] = example_file.split("data/")[-1]

            # read example csv file to get number of examples
            df = pd.read_csv(os.path.join(input_folder_dev, filename.replace("test", "dev")))
            yaml_content["example_rows"] = np.arange(len(df)).tolist()

            # define where to save the yaml file
            output_file_path = os.path.join(output_folder, filename.replace("_test.csv", ".yaml"))

            # Write to YAML file with customed function to respect formatting
            write_custom_yaml(yaml_content, output_file_path)
            print(f'Generated: {output_file_path}')

# %%
for i in few_shot_examples_idx_dict['Science']:
    print(f"Index: {i}")
    print(f"Category: {cdpk_dataset.loc[i, 'category']}")
    print(f"Question: {cdpk_dataset.loc[i, 'question']}")
    print(f"Age Group: {cdpk_dataset.loc[i, 'age_group']}")
    print(f"Correct answer: {cdpk_dataset.loc[i, 'correct_answer']}")
    print("-----")
# %%

language = "Yoruba_new"

create_subcsv_cdpk(cdpk_dataset, 
                   folder_path = f"./../data/{language}/CDPK_per_category",
                   categories = ["Science", "Literacy", "Creative arts", "Maths", "Social studies", "Technology", "General"],
                   levels = None,
                   suffix=language
                     )
# %%
yaml_template = {
    "test_file": "",
    "test_header": 0,
    "example_file": "",
    "example_header": 0,
    "choices": ["A", "B", "C", "D", "E", "F", "G"],
    "choice_cols": [2, 3, 4, 5, 6, 7, 8],
    "answer_col": 9,
    "question_col": 1,
    "example_rows": ""
}


process_csv_and_update_yaml(input_folder = f"./../data/{language}/CDPK_per_category", 
                            output_folder = f"./../configs/questions",
                            yaml_template = yaml_template,
                            )


# %%
