# %%
import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from fuzzywuzzy import fuzz
import yaml

homepath = Path.home()

# %%
# load dataset
#######################
#### SET PATH TO FOLDER CONTAINING RAW DATA
folder_path = ""

cdpk_dataset = pd.read_excel(folder_path / "merged_edited_mcqs_2017_2023.xlsx")
print(cdpk_dataset.shape)
cdpk_dataset.head(1)


# checks:
# - check the missing values
# - remove religion questions
# - remove special education questions
# - rename subdomains
# - check no duplicates
# - Clear typos

# - format columns: 
#   - add source, add Answer E, add Answer F, add Answer G (before Correct answer)
#   - remove question number
#########################################################################################

### Check the missing values
# %%
### Check no missing values
if cdpk_dataset.isnull().values.any():
    print(cdpk_dataset.info())
    # print value counts of origin of the file of missing values
    #print(cdpk_dataset[cdpk_dataset.isnull().any(axis=1)]["origin of the file"].value_counts())

def plot_value_counts(df, columns, height = 5):
    fig, axs = plt.subplots(len(columns), 1, figsize=(12, height*len(columns)))
    for i, column in enumerate(columns):
        sns.countplot(data=df, y=column, ax=axs[i], color = 'coral', order = df[column].value_counts().index)
        axs[i].set_ylabel(column, fontsize=18)
        axs[i].set_xlabel("Number of MCQs", fontsize=16)
        axs[i].grid(axis='x', linestyle='--', alpha=0.6)

    #plt.suptitle("Value counts of MCQs features", fontsize=18)
    plt.gca().set_axisbelow(True)
    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.show()

# %%

### Remove religion questions
def remove_religion_questions(df, show_removed_rows=False):
    df_cleared = df.copy()
    # Remove rows with "re" or "rc" in the column "origin of the file"
    df_cleared = df_cleared[~df_cleared["origin of the file"].str.contains("re|rc")]
    print(f"Number of rows removed (about religion): {df.shape[0] - df_cleared.shape[0]}")
    if show_removed_rows:
        print(f"    Origins of the file of removed rows: {df[df['origin of the file'].str.contains('re|rc')]['origin of the file'].value_counts()}")
    # check
    #print(df[df["origin of the file"].str.contains("re|rc")]['origin of the file'].value_counts().sum())
    #print(df.shape[0] - df_cleared.shape[0])
    return df_cleared.reset_index(drop=True)

### Remove special education questions
def remove_special_education_questions(df, drop_altern_col=False, show_removed_rows=False):
    df_wo_send = df.copy()
    # Remove rows with "_dea", "_tel", "_parvtel", "_defint", "_defvis", "_defaud", "_defmot", "_deftea", "_dismult" in the column "origin of the file"
    # 2021 and 2023 files are ee_ and not ed_ (use of general pattern in filtering)
    df_wo_send = df_wo_send[~df_wo_send["origin of the file"].str.contains("_dea|_tel|_parvtel|_defint|_defvis|_defaud|_defmot|_deftea|_dismult")]
    print(f"Number of rows removed (about Special Education): {df.shape[0] - df_wo_send.shape[0]}")

    # check
    idx_removed = df[df["origin of the file"].str.contains("_dea|_tel|_parvtel|_defint|_defvis|_defaud|_defmot|_deftea|_dismult")].index
    if show_removed_rows:
        print(f"   Origins of the file of removed rows: {df.loc[idx_removed, 'origin of the file'].value_counts()}")

    #print(df.loc[idx_removed, 'origin of the file'].value_counts().sum())
    #print(df.shape[0] - df_wo_send.shape[0])

    if drop_altern_col:
        if len(df_wo_send["Alternative Category"].value_counts()) > 1:
            print("Warning: There are values in the column 'Alternative Category'")
        df_wo_send.drop(columns=['Alternative Category', 'Alternative Subdomain'], inplace=True)

    return df_wo_send.reset_index(drop=True)

cdpk_dataset_2 = remove_religion_questions(cdpk_dataset, show_removed_rows=False)
cdpk_dataset_3 = remove_special_education_questions(cdpk_dataset_2, drop_altern_col=True, show_removed_rows=False)

# %%

### Subdomains update
def update_subdomains(df):
    with open("subdomains_update.json", "r") as file:
        subdomains_update = json.load(file)

    df_out = df.copy()
    # iterate over the subdomains_update and update the subdomains with pd.replace
    for tuple in subdomains_update['subdomain_mapping']:
        old = tuple['Old subdomain']
        new = tuple['New subdomain']
        df_out['Subdomain'] = df_out['Subdomain'].replace(old, new)
    return df_out

cdpk_dataset_4 = update_subdomains(cdpk_dataset_3)

#plot_value_counts(cdpk_dataset_4, ["Subdomain", "Year"])

# %%
### Check no duplicates
def flag_fuzzy_duplicates(df, columns, new_column, ratio_threshold, add_versioning=False, add_fuzzy_ratio=False):
    
    df_out = df.copy()

    df_out[new_column] = None
    if add_versioning == True:
        df_out['Duplicate version'] = None
    if add_fuzzy_ratio == True:
        df_out['Fuzzy ratio'] = None

    for i in tqdm(range(len(df_out) - 1)):
        for j in range(i + 1, len(df_out)):
            fuzz_ratios = []
            for col in columns:
                score = fuzz.ratio(df_out.loc[i, col], df_out.loc[j, col])
                fuzz_ratios.append(score)
            avg_fuzz_ratio = sum(fuzz_ratios) / len(fuzz_ratios)
            if avg_fuzz_ratio > ratio_threshold:
                #df.loc[i, 'Duplicate fuzzy'] = True 
                df_out.loc[j, new_column] = True
                if add_versioning == True:
                    df_out.loc[j, 'Duplicate version'] = "duplicate"
                    df_out.loc[i, 'Duplicate version'] = "original" if df_out.loc[i, 'Duplicate version'] != "duplicate" else df_out.loc[i, 'Duplicate version']
                if add_fuzzy_ratio == True:
                    df_out.loc[j, 'Fuzzy ratio'] = avg_fuzz_ratio

    df_out[new_column] = df_out[new_column].astype('bool').fillna(False)

    if df_out["Duplicated Question fuzzy"].sum() > 0:
        print("There are duplicates in the dataset!")
        for i, row in df_out[df_out["Duplicated Question fuzzy"]].iterrows():
            print(f"Duplicate identified at index {i} with fuzzy ratio {row['Fuzzy ratio']}")
    else:
        print("There are no duplicates in the dataset!")

    return df_out

fuzzy_ratio = 80
cdpk_dataset_fuzzy = flag_fuzzy_duplicates(cdpk_dataset_4, 
                                           columns=["Question"], 
                                           new_column="Duplicated Question fuzzy", 
                                           ratio_threshold=fuzzy_ratio, 
                                           add_versioning=False, 
                                           add_fuzzy_ratio=True)


# %%
# remove duplicates if any
if cdpk_dataset_fuzzy["Duplicated Question fuzzy"].sum() > 0:
    cdpk_dataset_5 = cdpk_dataset_fuzzy[~cdpk_dataset_fuzzy["Duplicated Question fuzzy"]].reset_index(drop=True)
    for col in ["Duplicated Question fuzzy", "Fuzzy ratio", "Duplicate version"]:
        if col in cdpk_dataset_5.columns:
            cdpk_dataset_5.drop(columns=[col], inplace=True)
else:
    cdpk_dataset_5 = cdpk_dataset_4.copy()
    for col in ["Duplicated Question fuzzy", "Fuzzy ratio", "Duplicate version"]:
        if col in cdpk_dataset_5.columns:
            cdpk_dataset_5.drop(columns=[col], inplace=True)

print(f"Number of rows removed (about duplicates): {cdpk_dataset_4.shape[0] - cdpk_dataset_5.shape[0]}")

# %%
### Clear typos
# 1. typos in subdomain columns
def clear_typos(df):
    # in subdomain columns
    df_out = df.copy()
    df_out["Subdomain"] = df_out["Subdomain"].str.replace("Education Theories", "Education theories")
    df_out["Subdomain"] = df_out["Subdomain"].str.replace("Classroom Management", "Classroom management")
    df_out["Subdomain"] = df_out["Subdomain"].str.replace("’", "'")
    if "Alternative Subdomain" in df_out.columns:
        df_out["Alternative Subdomain"] = df_out["Alternative Subdomain"].str.replace("Education Theories", "Education theories")
        df_out["Alternative Subdomain"] = df_out["Alternative Subdomain"].str.replace("Classroom Management", "Classroom management")
        df_out["Alternative Subdomain"] = df_out["Alternative Subdomain"].str.replace("’", "'")

    # in Age Group column 
    df_out["Age Group"] = df_out["Age Group"].str.replace("Pre-Primary", "Pre-primary")
    df_out["Age Group"] = df_out["Age Group"].str.replace("Pre-Primary, Primary", "Pre-primary, Primary")
    # in correct answer column
    df_out["Correct answer"] = df_out["Correct answer"].str.strip()

    # in Category column
    df_out["Category"] = df_out["Category"].str.replace("Technical Professional PCK", "Technology PCK")
    return df_out

cdpk_dataset_6 = clear_typos(cdpk_dataset_5)


# %%
### Format columns
cdpk_dataset_6.insert(0, "Source", "Chile ECEP 17-23") # insert source
cdpk_dataset_6.drop(columns = ["Question number"], inplace=True) # remove question number
cdpk_dataset_6.insert(6, "Answer E", None)
cdpk_dataset_6.insert(7, "Answer F", None)
cdpk_dataset_6.insert(8, "Answer G", None)

# %%
plot_value_counts(cdpk_dataset_6, ["Correct answer", "Category", "Subdomain", "Age Group", "Year"])
print(cdpk_dataset_6.info())

# %%
# Last check before saving
print(cdpk_dataset_6.columns.tolist())
print(cdpk_dataset_6.shape)
cdpk_dataset_6.head(2)

# %%
# add date of today to the name
today = pd.Timestamp.today().strftime("%Y%m%d")
#cdpk_dataset_6.to_csv(folder_path / f"cdpk_dataset_{today}.csv", index=False)
#cdpk_dataset_6.to_csv(f"./data/Chile/cdpk_dataset_{today}.csv", index=False)







#########################################################################################

## SEND mcqs ##
# %%
print(cdpk_dataset.shape)

# get SEND mcqs
df_send = cdpk_dataset[cdpk_dataset["origin of the file"].str.contains("_dea|_tel|_parvtel|_defint|_defvis|_defaud|_defmot|_deftea|_dismult")].reset_index(drop=True)
print(df_send.shape)
df_send.head(1)


# %%
# run preprocessing
df_send_2 = update_subdomains(df_send)
df_send_test = flag_fuzzy_duplicates(df_send_2, 
                                  columns=["Question"], 
                                  new_column="Duplicated Question fuzzy", 
                                  ratio_threshold=80, 
                                  add_versioning=False, 
                                  add_fuzzy_ratio=True)
# %%
df_send_3 = clear_typos(df_send_2)

#plot_value_counts(df_send_3, ["Correct answer", "Category", "Subdomain", "Age Group", "Year"])

# %%
### Format columns
df_send_3.insert(0, "Source", "Chile ECEP 17-23 SEND") # insert source
df_send_3.drop(columns = ["Question number"], inplace=True) # remove question number
df_send_3.insert(6, "Answer E", None)
df_send_3.insert(7, "Answer F", None)
df_send_3.insert(8, "Answer G", None)
# %%
# Last check before saving
print(df_send_3.columns.tolist())
print(df_send_3.shape)
df_send_3.head(2)

# %%
# add date of today to the name
today = pd.Timestamp.today().strftime("%Y%m%d")
#df_send_3.to_csv(folder_path / f"cdpk_dataset_send_{today}.csv", index=False)
#df_send_3.to_csv(f"./data/Chile/cdpk_dataset_send_{today}.csv", index=False)




# %%
#########################################################################################
# last update with the new dataset

cdpk_dataset = pd.read_csv(folder_path / "cdpk_dataset_20241128.csv")
hard_questions = pd.read_excel(folder_path / "hard_questions_cdpk_17_20_5m_correct_edited.xlsx")
print(cdpk_dataset.shape, hard_questions.shape)
cdpk_dataset.head(1)

# %%
# for all Questions in hard_questions, identify the corresponding Question in cdpk_dataset and remove it
def remove_hard_questions(df, hard_questions):
    df_out = df.copy()
    df_to_remove = hard_questions[hard_questions['Need to remove question'].isin(["Yes", "No", "Probably"])]
    print(f"Number of hard questions to remove: {df_to_remove.shape[0]}")
    for i, row in df_to_remove.iterrows():
        question = row["Question"]
        idx = df_out[df_out["Question"] == question].index
        if len(idx) == 1:
            df_out.drop(index=idx, inplace=True)
        elif len(idx) > 1:
            print(f"Multiple questions found in the dataset for index {i}")
            print(question)
            print(df_out.loc[idx, "origin of the file"])
        else:
            print(f"Question in index {i} not found in the dataset!")
            print(question)
            print(row['origin of the file'])
    # check all questions have been removed
    print(f"Number of hard questions removed: {df.shape[0] - df_out.shape[0]}")
    return df_out.reset_index(drop=True)

cdpk_dataset_new = remove_hard_questions(cdpk_dataset, hard_questions)
print(cdpk_dataset_new.shape)

# %%
# Remove Category PE PCK
print(f"Number of rows removed (about PE PCK): {len(cdpk_dataset_new[cdpk_dataset_new['Category'] == 'PE PCK'])}")
cdpk_dataset_new = cdpk_dataset_new[cdpk_dataset_new["Category"] != "PE PCK"].reset_index(drop=True)
# Combine categories
cdpk_dataset_new['Category'] = cdpk_dataset_new['Category'].apply(lambda x: 'Creative arts' if (x == 'Art PCK') 
                                                                  or (x == 'Music PCK') 
                                                                  else x)
cdpk_dataset_new['Category'] = cdpk_dataset_new['Category'].apply(lambda x: 'Social studies' if (x == 'History PCK') 
                                                                  or (x == 'Geography PCK')
                                                                  or (x == 'Philosophy PCK')
                                                                  or (x == 'Psychology PCK')
                                                                  or (x == 'Social studies PCK')
                                                                  else x)
print(cdpk_dataset_new['Category'].value_counts())

# %%
#  Save the dataset
# add date of today to the name
today = pd.Timestamp.today().strftime("%Y%m%d")
#cdpk_dataset_new.to_csv(folder_path / f"cdpk_dataset_{today}.csv", index=False)
#cdpk_dataset_new.to_csv(f"./data/Chile/cdpk_dataset_{today}.csv", index=False)


#########################################################################################

## Create config files with FS examples per category ##
# %%
cdpk_dataset = pd.read_csv(folder_path / "cdpk_dataset_20241202.csv")
print(cdpk_dataset.shape)
cdpk_dataset.head(1)

#cdpk_categories = cdpk_dataset["Category"].unique()
# %%

def create_subcsv_cdpk(df, folder_path, categories, levels):
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

                path_test_file = Path(folder_path) / "test" / f"CDPK_{category.replace(' PCK', '').replace(' ', '_').lower()}_test.csv"
                path_few_shot_file = Path(folder_path) / "dev" / f"CDPK_{category.replace(' PCK', '').replace(' ', '_').lower()}_dev.csv"

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
        idx_few_shot.extend(df_age.sample(n=n_examples//3).index.tolist())
        # check that we did not sample the same example twice
        while len(set(idx_few_shot)) < len(idx_few_shot):
            print("Duplicates found! Re-sampling...")
            # remove the last example(s) added
            idx_few_shot = idx_few_shot[:int(-n_examples//3)]
            idx_few_shot.extend(df_age.sample(n=n_examples//3).index.tolist())
    # check
    print(df.loc[idx_few_shot, ["Question", "Age Group"]])
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
    default_path_test_file = r"Chile\CDPK_per_category"

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

# %%
create_subcsv_cdpk(cdpk_dataset, 
                   folder_path = "data/Chile/CDPK_per_category",
                   categories = ["Science PCK", "Literacy PCK", "Creative arts", "Maths PCK", "Social studies", "Technology PCK", "Gen PK"],
                   levels = None
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


process_csv_and_update_yaml(input_folder = "data/Chile/CDPK_per_category", 
                            output_folder = "data/Chile/CDPK_per_category",
                            yaml_template = yaml_template
                            )


# %%
