# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyfonts import load_font
import seaborn as sns
import json
import re
import os
from pathlib import Path
from tqdm import tqdm
from fuzzywuzzy import fuzz
from dotenv import load_dotenv

load_dotenv(override=True)
BENCHMARKS_DIR = Path(os.getenv("BENCHMARKS_DIR", "./"))

def flag_exact_duplicates(df, columns, new_column, keep='first'):
    '''
    Flags exact duplicates in a DataFrame based on the specified columns with pd.duplicated() function.

    Parameters:
    df (pd.DataFrame): DataFrame to be checked for duplicates.
    columns (list): List of columns to be used for checking duplicates.
    new_column (str): Name of the new column to be added to the DataFrame to flag duplicates (True if duplicate, False if not).
    keep: If False, mark all duplicates as True. 
          If "first", mark all duplicates as True except for the first occurrence. 
          If "last", mark all duplicates as True except for the last occurrence.
    '''
    df[new_column] = df.duplicated(subset=columns, keep=keep)
    return df

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
    return df_out


    
def duplicates_flagging_optimized(df, ratio_threshold, add_versioning=False, add_fuzzy_ratio=False):

    # initialize the columns
    df['Duplicated Question'] = False
    df['Duplicated MCQ'] = False
    df['Duplicated Question fuzzy'] = False
    df['Duplicated MCQ fuzzy'] = False
    df['Duplicated fuzzy combined'] = False

    #1. Check for exact duplicates on Question and all MCQs (fast)
    print(f"Running exact duplicates flagging on {len(df)} rows")
    df = flag_exact_duplicates(df, ["Question"], 'Duplicated Question', keep="first")
    print(f"Running exact duplicates flagging on {len(df[~df['Duplicated Question']])} rows")
    df = flag_exact_duplicates(df, ["Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"], 'Duplicated MCQ', keep="first")

    #2. Check for fuzzy duplicates on Question for the ones that have not been flagged
    print(f"Running fuzzy duplicates flagging on {len(df[(~df['Duplicated Question']) | (~df['Duplicated MCQ'])])} rows")
    columns = ["Question"]
    for i in tqdm(df[(~df['Duplicated Question']) | (~df['Duplicated MCQ'])].index.tolist()[:-1]):
        for j in df[(~df['Duplicated Question']) | (~df['Duplicated MCQ'])].index.tolist()[1:]:
            fuzz_ratios = []
            for col in columns:
                score = fuzz.ratio(df.loc[i, col], df.loc[j, col])
                #print(score)
                fuzz_ratios.append(score)
            avg_fuzz_ratio = sum(fuzz_ratios) / len(fuzz_ratios)
            if avg_fuzz_ratio > ratio_threshold:
                df.loc[j, "Question"] = True

                if add_versioning == True:
                    df.loc[j, 'Duplicate version'] = "duplicate"
                    df.loc[i, 'Duplicate version'] = "original" if df.loc[i, 'Duplicate version'] != "duplicate" else df.loc[i, 'Duplicate version']
                if add_fuzzy_ratio == True:
                    df.loc[j, 'Fuzzy ratio'] = avg_fuzz_ratio
    return df


# %%
# Load the data

path_SP_all_2017 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2017" / "merged_SP_ecep_2017_all.xlsx"
path_SP_all_2018 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2018" / "merged_SP_ecep_2018_all.xlsx"
path_SP_all_2019 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2019" / "merged_SP_ecep_2019_all.xlsx"
path_SP_all_2020 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2020" / "merged_SP_ecep_2020_all.xlsx"
path_SP_all_2021 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2021" / "merged_SP_ecep_2021_all.xlsx"
path_SP_all_2023 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2023" / "merged_SP_ecep_2023_all.xlsx"

ecep_2017_SP_all = pd.read_excel(path_SP_all_2017)
ecep_2018_SP_all = pd.read_excel(path_SP_all_2018)
ecep_2019_SP_all = pd.read_excel(path_SP_all_2019)
ecep_2020_SP_all = pd.read_excel(path_SP_all_2020)
ecep_2021_SP_all = pd.read_excel(path_SP_all_2021)
ecep_2023_SP_all = pd.read_excel(path_SP_all_2023)

# remove the spaces in column "Correct answer"
ecep_2017_SP_all["Correct answer"] = ecep_2017_SP_all["Correct answer"].str.strip()
ecep_2018_SP_all["Correct answer"] = ecep_2018_SP_all["Correct answer"].str.strip()
ecep_2019_SP_all["Correct answer"] = ecep_2019_SP_all["Correct answer"].str.strip()
ecep_2020_SP_all["Correct answer"] = ecep_2020_SP_all["Correct answer"].str.strip()
ecep_2021_SP_all["Correct answer"] = ecep_2021_SP_all["Correct answer"].str.strip()
ecep_2023_SP_all["Correct answer"] = ecep_2023_SP_all["Correct answer"].str.strip()


print(ecep_2017_SP_all.shape)
print(ecep_2018_SP_all.shape)
print(ecep_2019_SP_all.shape)
print(ecep_2020_SP_all.shape)
print(ecep_2021_SP_all.shape)
print(ecep_2023_SP_all.shape)

# check nan or empty values in each dataframe
print(ecep_2017_SP_all.isnull().sum().tolist())
print(ecep_2018_SP_all.isnull().sum().tolist())
print(ecep_2019_SP_all.isnull().sum().tolist())
print(ecep_2020_SP_all.isnull().sum().tolist())
print(ecep_2021_SP_all.isnull().sum().tolist())
print(ecep_2023_SP_all.isnull().sum().tolist())

# %%
# add year and merge all years
ecep_2017_SP_all['Year'] = 2017
ecep_2018_SP_all['Year'] = 2018
ecep_2019_SP_all['Year'] = 2019
ecep_2020_SP_all['Year'] = 2020
ecep_2021_SP_all['Year'] = 2021
ecep_2023_SP_all['Year'] = 2023

ecep_SP_all = pd.concat([ecep_2017_SP_all, 
                         ecep_2018_SP_all, 
                         ecep_2019_SP_all, 
                         ecep_2020_SP_all,
                         ecep_2021_SP_all,
                         ecep_2023_SP_all
                         ]).reset_index(drop=True)

# remove rows with nan values if any in columns "Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"
print(f"Number of MCQs before NaN removal: {ecep_SP_all.shape[0]}")
ecep_SP_all.dropna(subset=['Question', 'Answer A', 'Answer B', 'Answer C', 'Answer D', 'Correct answer'], inplace=True)
print(f"Number of MCQs after NaN removal: {ecep_SP_all.shape[0]}")

print(ecep_SP_all.shape)
ecep_SP_all.head(2)
# %%
# save as chilean_mcqs_all_original_SP.xlsx
#ecep_SP_all.to_excel('./data/Chile/chilean_mcqs_all_original_SP.xlsx', index=False)

# %%
# Flag  INTRA-year duplicates
dict_df_year = {
    "2017": ecep_2017_SP_all,
    "2018": ecep_2018_SP_all,
    "2019": ecep_2019_SP_all,
    "2020": ecep_2020_SP_all,
    "2021": ecep_2021_SP_all,
    "2023": ecep_2023_SP_all
}

# %%
#dict_df_year_intra = dict_df_year.copy()
dict_df_year_intra = {year: df.copy() for year, df in dict_df_year.items()}

for year, df in dict_df_year_intra.items():
    df = flag_exact_duplicates(df, ["Question"], 'Duplicated Question')
    df = flag_exact_duplicates(df, ["Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"], 'Duplicated MCQ')
    df = flag_fuzzy_duplicates(df, ["Question"], 'Duplicated Question fuzzy', ratio_threshold = 70, add_versioning=False, add_fuzzy_ratio=False)
    df = flag_fuzzy_duplicates(df, ["Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"], 'Duplicated MCQ fuzzy', ratio_threshold = 70, add_versioning=False, add_fuzzy_ratio=False)
    df['Duplicated fuzzy combined'] = df['Duplicated Question fuzzy'] | df['Duplicated MCQ fuzzy']
    dict_df_year_intra[year] = df

# add year to all dataframes
for year, df in dict_df_year_intra.items():
    df['Year'] = year

# merge all years with results from intra-year duplicates
ecep_SP_all_intra = pd.concat([dict_df_year_intra["2017"], 
                               dict_df_year_intra["2018"], 
                               dict_df_year_intra["2019"], 
                               dict_df_year_intra["2020"],
                               dict_df_year_intra["2021"],
                               dict_df_year_intra["2023"]
                               ]).reset_index(drop=True)

print(ecep_SP_all_intra.shape)
ecep_SP_all_intra.head(2)

# %% 
ecep_SP_all_grouped = ecep_SP_all_intra.groupby(['Year'])[['Duplicated MCQ', 'Duplicated Question', 'Duplicated MCQ fuzzy', 'Duplicated Question fuzzy', 'Duplicated fuzzy combined']].apply(lambda x: x.mean() * 100)
ecep_SP_all_melted = ecep_SP_all_grouped.reset_index().melt(id_vars='Year', var_name='Type of Duplication', value_name='Percentage')

plt.figure(figsize=(12, 5))
sns.barplot(x='Type of Duplication', y='Percentage', hue='Year', data=ecep_SP_all_melted, palette='Set2')
plt.title('Percentage of Duplicates Per Year', fontsize=16)
plt.ylabel('Percentage of Duplicates (%)')
plt.grid(axis='y')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

ecep_SP_all_melted.head(15)



# %%
# Flag  INTER-year duplicates

dict_df_year_inter = {year: df.copy() for year, df in dict_df_year.items()}

for year, df in dict_df_year_inter.items():
    print(df.columns.tolist())

# %%

# 2. Merge all years and drop uncessary columns
ecep_SP_all_inter = pd.concat([dict_df_year_inter["2017"], 
                               dict_df_year_inter["2018"], 
                               dict_df_year_inter["2019"], 
                               dict_df_year_inter["2020"],
                               dict_df_year_inter["2021"],
                               dict_df_year_inter["2023"]
                               ]).reset_index(drop=True)

keep_columns = ['Question number', 'Question', 'Answer A', 'Answer B', 'Answer C', 'Answer D', 'Correct answer', 'origin of the file', 'Year']    
for col in ecep_SP_all_inter.columns:
    if col not in keep_columns:
        ecep_SP_all_inter.drop(col, axis=1, inplace=True)

print(ecep_SP_all_inter['Year'].value_counts())
print(ecep_SP_all_inter.shape)
ecep_SP_all_inter.head(2)

# %%

# 3. Flag INTER-year duplicates
ecep_SP_all_inter = flag_exact_duplicates(ecep_SP_all_inter, ["Question"], 'Duplicated Question')
ecep_SP_all_inter = flag_exact_duplicates(ecep_SP_all_inter, ["Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"], 'Duplicated MCQ')
ecep_SP_all_inter = flag_fuzzy_duplicates(ecep_SP_all_inter, ["Question"], 'Duplicated Question fuzzy', ratio_threshold = 70, add_versioning=False, add_fuzzy_ratio=False)
ecep_SP_all_inter = flag_fuzzy_duplicates(ecep_SP_all_inter, ["Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"], 'Duplicated MCQ fuzzy', ratio_threshold = 70, add_versioning=False, add_fuzzy_ratio=False)
ecep_SP_all_inter['Duplicated fuzzy combined'] = ecep_SP_all_inter['Duplicated Question fuzzy'] | ecep_SP_all_inter['Duplicated MCQ fuzzy']

print(ecep_SP_all_inter.shape)
ecep_SP_all_inter.head(2)


# %%
duplicates_between_all_years = ecep_SP_all_inter.groupby("Year")[["Duplicated Question", "Duplicated MCQ", "Duplicated Question fuzzy", "Duplicated MCQ fuzzy", "Duplicated fuzzy combined"]].apply(lambda x: x.mean() * 100)
duplicates_between_all_years = duplicates_between_all_years.reset_index().melt(id_vars="Year", var_name="Type of Duplication", value_name="Percentage")

plt.figure(figsize=(12, 6))
sns.barplot(x='Type of Duplication', y='Percentage', hue='Year', data=duplicates_between_all_years, palette='Set2')
plt.title('Annual Duplicate Percentage (Both Within and Across Years)', fontsize=16)
plt.ylabel('Percentage of Duplicates (%)')
plt.grid(axis='y')
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

duplicates_between_all_years.head(15)

#  %%
print(ecep_SP_all_inter['Duplicated Question fuzzy'].value_counts())


# %%
# save the results to laptop dropbox
#ecep_SP_all_inter.to_excel('./data/Chile/ecep_SP_all_with_dupl_flagged_2017_2023.xlsx', index=False)
#ecep_SP_all_inter.head()


# %%
# Flag duplicates in english version

# first load all english mcqs together
# Load the data
path_EN_all_2017 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2017" / "merged_EN_ecep_2017_all.xlsx"
path_EN_all_2018 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2018" / "merged_EN_ecep_2018_all.xlsx"
path_EN_all_2019 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2019" / "merged_EN_ecep_2019_all.xlsx"
path_EN_all_2020 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2020" / "merged_EN_ecep_2020_all.xlsx"
path_EN_all_2021 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2021" / "merged_EN_ecep_2021_all.xlsx"
path_EN_all_2023 = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ECEP 2023" / "merged_EN_ecep_2023_all.xlsx"

ecep_2017_EN_all = pd.read_excel(path_EN_all_2017)
ecep_2018_EN_all = pd.read_excel(path_EN_all_2018)
ecep_2019_EN_all = pd.read_excel(path_EN_all_2019)
ecep_2020_EN_all = pd.read_excel(path_EN_all_2020)
ecep_2021_EN_all = pd.read_excel(path_EN_all_2021)
ecep_2023_EN_all = pd.read_excel(path_EN_all_2023)

# remove the spaces in column "Correct answer"
ecep_2017_EN_all["Correct answer"] = ecep_2017_EN_all["Correct answer"].str.strip()
ecep_2018_EN_all["Correct answer"] = ecep_2018_EN_all["Correct answer"].str.strip()
ecep_2019_EN_all["Correct answer"] = ecep_2019_EN_all["Correct answer"].str.strip()
ecep_2020_EN_all["Correct answer"] = ecep_2020_EN_all["Correct answer"].str.strip()
ecep_2021_EN_all["Correct answer"] = ecep_2021_EN_all["Correct answer"].str.strip()
ecep_2023_EN_all["Correct answer"] = ecep_2023_EN_all["Correct answer"].str.strip()


print(ecep_2017_EN_all.shape)
print(ecep_2018_EN_all.shape)
print(ecep_2019_EN_all.shape)
print(ecep_2020_EN_all.shape)
print(ecep_2021_EN_all.shape)
print(ecep_2023_EN_all.shape)

# check nan or empty values in each dataframe
print(ecep_2017_EN_all.isnull().sum().tolist())
print(ecep_2018_EN_all.isnull().sum().tolist())
print(ecep_2019_EN_all.isnull().sum().tolist())
print(ecep_2020_EN_all.isnull().sum().tolist())
print(ecep_2021_EN_all.isnull().sum().tolist())
print(ecep_2023_EN_all.isnull().sum().tolist())

# %%
ecep_2023_SP_all[ecep_2023_SP_all['Answer D'].isna()].head()

# %%
# add year and merge all years
ecep_2017_EN_all['Year'] = 2017
ecep_2018_EN_all['Year'] = 2018
ecep_2019_EN_all['Year'] = 2019
ecep_2020_EN_all['Year'] = 2020
ecep_2021_EN_all['Year'] = 2021
ecep_2023_EN_all['Year'] = 2023

ecep_EN_all = pd.concat([ecep_2017_EN_all, 
                         ecep_2018_EN_all,
                         ecep_2019_EN_all,
                         ecep_2020_EN_all,
                         ecep_2021_EN_all,
                         ecep_2023_EN_all
                         ]).reset_index(drop=True)

# remove rows with nan values in columns "Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"
print(f"Number of MCQs before NaN removal: {ecep_EN_all.shape[0]}")
ecep_EN_all.dropna(subset=['Question', 'Answer A', 'Answer B', 'Answer C', 'Answer D', 'Correct answer'], inplace=True)
print(f"Number of MCQs after NaN removal: {ecep_EN_all.shape[0]}")

# remove categories "Category", "Subdomain", "Age Group"
ecep_EN_all.drop(['Category', 'Subdomain', 'Age Group'], axis=1, inplace=True)
print(ecep_EN_all.shape)
ecep_EN_all.head(2)
# %%
#save as chilean_mcqs_all_original_SP.xlsx
#ecep_EN_all.to_excel('./data/Chile/chilean_mcqs_all_original_EN.xlsx', index=False)

# %%

path_SP_dupli_flagged = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "ecep_SP_all_with_dupl_flagged_2017_2023.xlsx"

merged_all_years_SP_dupl_flagged = pd.read_excel(path_SP_dupli_flagged)
print(merged_all_years_SP_dupl_flagged.shape)

# check same values and same order in column "Question number"
print((merged_all_years_SP_dupl_flagged['Question number'] == ecep_EN_all['Question number']).all())
display(merged_all_years_SP_dupl_flagged.head(2), ecep_EN_all.head(2))

# %%
# append column "Duplicated fuzzy combined" to the english version
ecep_EN_all['Duplicated fuzzy combined'] = merged_all_years_SP_dupl_flagged['Duplicated fuzzy combined']
print(ecep_EN_all.shape)
ecep_EN_all.head(2)
# %%

new_set_21_23 = ecep_EN_all[ecep_EN_all['Year'].isin([2021, 2023])].reset_index(drop=True)
# remove duplicates
new_set_21_23 = new_set_21_23[~new_set_21_23['Duplicated fuzzy combined']].reset_index(drop=True)
# remove columns starting with "Duplicated"
new_set_21_23 = new_set_21_23[[col for col in new_set_21_23.columns if not col.startswith('Duplicated')]]
print(new_set_21_23.shape)
print(new_set_21_23['Year'].value_counts())
new_set_21_23.head(2)

# save
#new_set_21_23.to_excel('./data/Chile/new_set_mcqs_2021_2023_new.xlsx', index=False)

# %%
# count the special needs questions: has "ED_" in "origin of the file"
test = new_set_21_23.copy()
test['Special needs'] = test['origin of the file'].str.contains("_ee_")
print(test['Special needs'].value_counts())



















# %%

# Check the duplicates

def check_duplicates(df, idx_check, columns, ratio_threshold):
    idx_duplicates = [idx_check]
    # add fuzz ratio column
    df['Fuzzy ratio'] = None

    # check if there are duplicates with fuzzy
    for i in range(len(df)):
        if i == idx_check:
            continue
        fuzz_ratios = []
        for col in columns:
            score = fuzz.ratio(df.loc[idx_check, col], df.loc[i, col])
            fuzz_ratios.append(score)
        avg_fuzz_ratio = sum(fuzz_ratios) / len(fuzz_ratios)
        if avg_fuzz_ratio > ratio_threshold:
            idx_duplicates.append(i)
            df.loc[i, 'Fuzzy ratio'] = avg_fuzz_ratio

    # concat original row with the duplicates
    duplicates = df.loc[idx_duplicates]
    return duplicates

# %%

# Load the data with all MCQS with duplicates flagged
df_check = pd.read_excel('./data/Chile/ecep_SP_all_with_dupl_flagged_2017_2023.xlsx')
print(df_check.shape)
print(df_check['Year'].value_counts())
df_check.head(2)
# %%
duplicates_fuzzy = pd.DataFrame(df_check[df_check["Duplicated fuzzy combined"] == True])
print(f"Number of duplicates to check: {duplicates_fuzzy.shape[0]}")
# %%
# create a column "checked" to store if the duplicates have been checked
duplicates_fuzzy["checked"] = False

counter = 0
for i, row in tqdm(duplicates_fuzzy.iterrows()):
    # add check to see the examples
    fuzzy_th = 80
    duplicates = check_duplicates(df_check, i, ["Question"], fuzzy_th)
    if len(duplicates) > 4:
        print(f"Number of duplicates: {len(duplicates)}")
        for idx in duplicates.index:
            print(duplicates.loc[idx, "Question"])
        display(duplicates.head())
        break

    # check first for fuzzy duplicate in question for faster processing
    if len(check_duplicates(df_check, i, ["Question"], 70)) > 1:
        duplicates_fuzzy.loc[i, "checked"] = True
    # check for duplicates in all columns if no duplicates in question
    elif len(check_duplicates(df_check, i, ["Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"], 70)) > 1:
        duplicates_fuzzy.loc[i, "checked"] = True
    counter += 1
    #if counter % 200 == 0:
        #print(f"Checked {counter} duplicates")

# They should all be to True in column "checked"
print(duplicates_fuzzy["checked"].value_counts())

# %%

# compute the number of duplicates on all MCQs extracted with different thresholds
fuzzy_th_list = [60, 70, 80, 90]

ecep_SP_all_fuzzy_test = pd.read_excel('./data/Chile/chilean_mcqs_all_original_EN.xlsx')
print(ecep_SP_all_fuzzy_test.shape)
# drop columns starting with "Duplicated" if any
ecep_SP_all_fuzzy_test = ecep_SP_all_fuzzy_test[[col for col in ecep_SP_all_fuzzy_test.columns if not col.startswith('Duplicated')]]
ecep_SP_all_fuzzy_test.head(2)

# %%
# took 1h20 to run!!
for fuzzy_th in fuzzy_th_list:
    # compute the number of fuzzy duplicates on question and all MCQs
    ecep_SP_all_fuzzy_test = flag_fuzzy_duplicates(ecep_SP_all_fuzzy_test, 
                                                   ["Question"], 
                                                   f'Duplicated Question fuzzy {fuzzy_th}', 
                                                   ratio_threshold = fuzzy_th, 
                                                   add_versioning=False, 
                                                   add_fuzzy_ratio=False)
    print(f"Percentage of duplicates with threshold {fuzzy_th} on Question: {ecep_SP_all_fuzzy_test[f'Duplicated Question fuzzy {fuzzy_th}'].mean() * 100:.2f}%")
    #ecep_SP_all_fuzzy_test = flag_fuzzy_duplicates(ecep_SP_all_fuzzy_test, ["Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"], 'Duplicated MCQ fuzzy', ratio_threshold = fuzzy_th, add_versioning=False, add_fuzzy_ratio=False)
    #ecep_SP_all_fuzzy_test[f'Duplicated fuzzy combined {fuzzy_th}'] = ecep_SP_all_fuzzy_test['Duplicated Question fuzzy'] | ecep_SP_all_fuzzy_test['Duplicated MCQ fuzzy']
    # drop columns "Duplicate Question fuzzy" and "Duplicate MCQ fuzzy"
    #ecep_SP_all_fuzzy_test.drop([f'Duplicated Question fuzzy {fuzzy_th}'], axis=1, inplace=True)

# %%
# save results of fuzzy ratio experiment
print(ecep_SP_all_fuzzy_test.shape)
ecep_SP_all_fuzzy_test.head(2)
#ecep_SP_all_fuzzy_test.to_excel('./data/Chile/ecep_SP_all_fuzzy_ratio_experiment.xlsx', index=False)






# %%

# load results if needed
ecep_SP_all_fuzzy_test = pd.read_excel('./data/Chile/ecep_SP_all_fuzzy_ratio_experiment.xlsx')
print(ecep_SP_all_fuzzy_test.shape)
# analyze the results
# count the proportion of duplicates per year
fuzzy_th_list = [60, 70, 80, 90]
results_fuzzy_th = pd.DataFrame({
    "Fuzzy threshold": fuzzy_th_list,
    "Number of duplicates": [ecep_SP_all_fuzzy_test[f'Duplicated Question fuzzy {fuzzy_th}'].sum() for fuzzy_th in fuzzy_th_list]
})
results_fuzzy_th["Proportion of duplicates"] = results_fuzzy_th["Number of duplicates"] / ecep_SP_all_fuzzy_test.shape[0] * 100

# plot the results
plt.figure(figsize=(5, 3))
sns.barplot(x='Fuzzy threshold', y='Proportion of duplicates', data=results_fuzzy_th, palette='Set2', hue = 'Fuzzy threshold', legend=False)
plt.ylabel('Proportion of Duplicates (%)')
plt.title('Proportion of Duplicates for different thresholds')
plt.ylim(0, 60)
# on top of each bar, write the number of MCQs
for i in range(results_fuzzy_th.shape[0]):
    plt.text(i, results_fuzzy_th.loc[i, "Proportion of duplicates"] + 0.5, results_fuzzy_th.loc[i, "Number of duplicates"], ha='center')
plt.show()

results_fuzzy_th.head()


# %%
# organize in pairs the first occurrence of the duplicates in the original dataframe and the duplicates
def flag_fuzzy_duplicates_by_pair(df, columns, ratio_threshold_min, ratio_threshold_max):
    
    df_out = pd.DataFrame(columns = ['original_index', 'duplicate_index', 'original_question', 'duplicate_question', 'Fuzzy ratios', "original_year", "duplicate_year", "original_origin", "duplicate_origin"])

    for i in tqdm(range(len(df) - 1)):
        duplicates_idx_list = []
        duplicates_question_list = []
        fuzz_ratios_list = []
        year_list = []
        origin_list = []
        for j in range(i + 1, len(df)):
            fuzz_ratios = []
            for col in columns:
                score = fuzz.ratio(df.loc[i, col], df.loc[j, col])
                fuzz_ratios.append(score)
            avg_fuzz_ratio = sum(fuzz_ratios) / len(fuzz_ratios)
            if avg_fuzz_ratio >= ratio_threshold_min and avg_fuzz_ratio < ratio_threshold_max:
                
                # save the pair of duplicates index, questions and fuzzy ratios
                duplicates_idx_list.append(j)
                duplicates_question_list.append(df.loc[j, 'Question'])
                fuzz_ratios_list.append(avg_fuzz_ratio)
                year_list.append(df.loc[j, 'Year'])
                origin_list.append(df.loc[j, 'origin of the file'])

        if len(duplicates_idx_list) > 0:
            # append all elements to the dataframe
            df_out.loc[len(df_out)] = [i, 
                                       duplicates_idx_list, 
                                       df.loc[i, 'Question'], 
                                       duplicates_question_list, 
                                       fuzz_ratios_list,
                                       df.loc[i, 'Year'],
                                       year_list,
                                       df.loc[i, 'origin of the file'],
                                       origin_list]

    return df_out

# load all original MCQs of SP
ecep_EN_all = pd.read_excel('./data/Chile/chilean_mcqs_all_original_SP.xlsx')
print(ecep_EN_all.shape)

duplicates_pairs = flag_fuzzy_duplicates_by_pair(ecep_EN_all, ["Question"], 70, 80)
duplicates_pairs.head()

# %%
# organize in a json format
duplicates_pairs_json = [
    {
        "original_question": row['original_question'],
        "data": {
            "index": int(row['original_index']),
            "year": int(row['original_year']),
            "origin": row['original_origin']
        },
        "duplicated questions": [
            {
                "duplicate_question": question, 
                "data": {
                    "index": int(idx),
                    "year": int(year),
                    "origin": origin,
                    "Fuzzy ratio": float(ratio)
                }
        }
            for idx, question, ratio, year, origin in zip(row['duplicate_index'], 
                                                          row['duplicate_question'], 
                                                          row['Fuzzy ratios'], 
                                                          row['duplicate_year'], 
                                                          row['duplicate_origin'])
        ]
    } for i, row in duplicates_pairs.iterrows()
]


# print json format
print(json.dumps(duplicates_pairs_json, indent=4))
# %%
# save json format
#with open('./data/Chile/duplicates_pairs_70_80.json', 'w') as f:
#    json.dump(duplicates_pairs_json, f)





# %%
# read the edited mcqs and see if there are duplicates between 70 and 90 fuzzy ratio that have been wrongly flagged
# load the edited mcqs
path_edited_mcqs = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "merged_edited_mcqs_2017_2023.xlsx"

edited_mcqs = pd.read_excel(path_edited_mcqs)
# load all original MCQs of EN
ecep_EN_all = pd.read_excel('./data/Chile/chilean_mcqs_all_original_EN.xlsx')
print(edited_mcqs.shape, ecep_EN_all.shape)
edited_mcqs.head(2)
# %%
# for each question in the edited mcqs, check if there are duplicates with fuzzy ratio between 70 and 90 in the file ecep_EN_all

candidates = pd.DataFrame(columns = ['original edited mcq', 'candidates', 'candidates_idx', 'fuzzy ratios', 'original_year', 'candidates_year', 'original_origin', 'candidates_origin'])

for i, row in tqdm(edited_mcqs.iterrows()):
    # check if the question has a duplicate in the range of fuzzy ratio between 70 and 90 in the original MCQs
    fuzz_ratios = []
    candidates_idx = []
    candidates_questions = []
    candidates_year = []
    candidates_origin = []

    for j, row_j in ecep_EN_all.iterrows():
        score = fuzz.ratio(row['Question'], row_j['Question'])
        if score >= 70 and score <= 90:
            fuzz_ratios.append(score)
            candidates_idx.append(j)
            candidates_questions.append(row_j['Question'])
            candidates_year.append(row_j['Year'])
            candidates_origin.append(row_j['origin of the file'])
    
    if len(candidates_idx) > 0:
        candidates.loc[len(candidates)] = [row['Question'], candidates_questions, candidates_idx, fuzz_ratios, row['Year'], candidates_year, row['origin of the file'], candidates_origin]
    
    #if i == 50:
    #    break

# %%
print(candidates.shape)
print(f"Number of potential candidates: {candidates['candidates'].apply(lambda x: len(x)).sum()}")
print(f"Number of unique potential candidates: {candidates['candidates_idx'].explode().nunique()}")
candidates.head()

# save excel file
#candidates.to_excel('./data/Chile/candidates_70_90_fuzzy_ratio.xlsx', index=False)

# %%
# organize in a json format
candidates_json = [
    {
        "cdpk sample": row['original edited mcq'],
        "data": {
            "year": int(row['original_year']),
            "origin": row['original_origin']
        },
        "cdpk candidates": [
            {
                "candidate": question,
                "data": {
                    "year": int(year),
                    "idx": int(idx),
                    "origin": origin,
                    "Fuzzy ratio": float(ratio)
                }
        }
            for idx, question, ratio, year, origin in zip(row['candidates_idx'],
                                                     row['candidates'],
                                                     row['fuzzy ratios'], 
                                                     row['candidates_year'],
                                                     row['candidates_origin'])
        ]
    } for i, row in candidates.iterrows()
]


# print json format
print(json.dumps(candidates_json, indent=4))

# save json
#with open('./data/Chile/candidates_70_90_fuzzy_ratio.json', 'w') as f:
#    json.dump(candidates_json, f, indent=4)

# %%
# save filtered candidates in an excel file
fp_duplicates_idx = [4672, 87, 2479, 6155, 2463, 3266, 2472, 3127, 7693, 7726, 8739, 9960, 9096, 5519, 7369, 738, 2551, 3433, 4128, 7912, 9116, 1332,  4400, 4858, 5312, 5324, 6097, 7916, 7991, 765, 1285, 3212, 3011, 998, 5304, 9418, 9567, 9780]
# print length 
print(len(fp_duplicates_idx))
print(len(set(fp_duplicates_idx)))
for elem in set(fp_duplicates_idx):
    # print elem if appears more than once
    if fp_duplicates_idx.count(elem) > 1:
        print(elem)


# %%
candidates_excel = ecep_EN_all.loc[fp_duplicates_idx]
print(candidates_excel.shape)
candidates_excel.head()
#candidates_excel.to_excel('./data/Chile/fp_duplicates.xlsx', index=False)















# %%

# Check duplicates in the final edited file
path_edited_mcqs = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "merged_edited_mcqs_2017_2023.xlsx"
merged_edited_mcqs = pd.read_excel(path_edited_mcqs)
print(merged_edited_mcqs.shape)
print(merged_edited_mcqs['Year'].value_counts())
merged_edited_mcqs.head(2)


# %%
merged_edited_mcqs_dupli_flagged = merged_edited_mcqs.copy()
merged_edited_mcqs_dupli_flagged = flag_exact_duplicates(merged_edited_mcqs_dupli_flagged, ["Question"], 'Duplicated Question')
merged_edited_mcqs_dupli_flagged = flag_exact_duplicates(merged_edited_mcqs_dupli_flagged, ["Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"], 'Duplicated MCQ')

merged_edited_mcqs_dupli_flagged = flag_fuzzy_duplicates(df = merged_edited_mcqs_dupli_flagged,
                                                         columns =  ["Question"], 
                                                         new_column = 'Duplicated Question fuzzy', 
                                                         ratio_threshold=70, 
                                                         add_versioning=False, 
                                                         add_fuzzy_ratio=False)

merged_edited_mcqs_dupli_flagged = flag_fuzzy_duplicates(df = merged_edited_mcqs_dupli_flagged,
                                                         columns =  ["Question", "Answer A", "Answer B", "Answer C", "Answer D", "Correct answer"], 
                                                         new_column = 'Duplicated MCQ fuzzy', 
                                                         ratio_threshold=70, 
                                                         add_versioning=False, 
                                                         add_fuzzy_ratio=False)

merged_edited_mcqs_dupli_flagged['Duplicated fuzzy combined'] = merged_edited_mcqs_dupli_flagged['Duplicated Question fuzzy'] | merged_edited_mcqs_dupli_flagged['Duplicated MCQ fuzzy']

print(merged_edited_mcqs_dupli_flagged.shape)
print(merged_edited_mcqs_dupli_flagged['Duplicated fuzzy combined'].value_counts())
merged_edited_mcqs_dupli_flagged[merged_edited_mcqs_dupli_flagged['Duplicated fuzzy combined'] == True].head()


# %%
merged_edited_mcqs_wo_dupli = merged_edited_mcqs_dupli_flagged[~merged_edited_mcqs_dupli_flagged['Duplicated fuzzy combined']].reset_index(drop=True)
print(f"Number of remaining MCQs after removing duplicates: {merged_edited_mcqs_wo_dupli.shape[0]}")

# remove columns starting with "Duplicated"
merged_edited_mcqs_wo_dupli = merged_edited_mcqs_wo_dupli[[col for col in merged_edited_mcqs_wo_dupli.columns if not col.startswith('Duplicated')]]
merged_edited_mcqs_wo_dupli.head(2)

# save file in dropbox and local computer
print(merged_edited_mcqs_wo_dupli.shape)
#merged_edited_mcqs_wo_dupli.to_excel(Path(str(homepath)+ "\\Fab Inc Dropbox\\Fab Inc BMGF AI\\Jacobs Foundation working folder\\Benchmarks\\Developing MCQs\\Teaching Exam Questions\\processed files\\Chile") / "merged_edited_mcqs_2017_2023.xlsx", index=False)
#merged_edited_mcqs_wo_dupli.to_excel("./data/Chile/merged_edited_mcqs_2017_2023.xlsx", index=False)







# %%
# remove the duplicates
edited_all_finale = merged_edited_mcqs_dupli_flagged[~merged_edited_mcqs_dupli_flagged['Duplicated fuzzy combined']].reset_index(drop=True)
# remove columns starting with "Duplicated"
edited_all_finale = edited_all_finale[[col for col in edited_all_finale.columns if not col.startswith('Duplicated')]]
print(edited_all_finale.shape)

#edited_all_finale.to_excel('./data/Chile/merged_edited_mcqs_wo_dupli.xlsx', index=False)

# %%
#duplicates_between_all_years = edited_all_finale.groupby("Year")[["Duplicated Question", "Duplicated MCQ", "Duplicated Question fuzzy", "Duplicated MCQ fuzzy", "Duplicated fuzzy combined"]].apply(lambda x: x.mean() * 100)
#duplicates_between_all_years = duplicates_between_all_years.reset_index().melt(id_vars="Year", var_name="Type of Duplication", value_name="Percentage")
#
#plt.figure(figsize=(12, 6))
#sns.barplot(x='Type of Duplication', y='Percentage', hue='Year', data=duplicates_between_all_years, palette='Set2')
#plt.title('Annual Duplicate Percentage (Both Within and Across Years)', fontsize=16)
#plt.ylabel('Percentage of Duplicates (%)')
#plt.grid(axis='y')
#plt.ylim(0, 100)
#plt.tight_layout()
#plt.show()

