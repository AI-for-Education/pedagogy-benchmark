# %%
import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyfonts import load_font
from matplotlib.lines import Line2D
from pathlib import Path
import json
from tqdm import tqdm

from fuzzywuzzy import fuzz
from tqdm import tqdm
from dotenv import load_dotenv

from cdpk.benchmark_constants import ROOT

load_dotenv(override=True)
BENCHMARKS_DIR = Path(os.getenv("BENCHMARKS_DIR", "./"))

BASEDIR = ROOT.parent
DATA_DIR = BASEDIR / "results teachers ECEP"


# %%
######## CDPK dataset ########
# import results cdpk to get distribution of categories
df_cdpk = pd.read_excel(BASEDIR / "LLM_benchmark" / "cdpk_results_all.xlsx")
print(df_cdpk.shape)
df_cdpk.head(2)

# %%
df_cdpk_clean = df_cdpk.copy()
df_cdpk_clean.drop(columns=[col for col in df_cdpk_clean.columns if col.startswith("pred_")], inplace = True)
df_cdpk_clean['origin short'] = df_cdpk_clean['origin of the file'].apply(lambda x: " ".join(x.split("EN_")[1].split('_')[:2]))
df_cdpk_clean['origin short'] = df_cdpk_clean['origin short'].apply(lambda x: re.sub(r'\d+', '', x).replace(" ", "_").upper())
df_cdpk_clean['origin short'] = df_cdpk_clean['origin short'].apply(lambda x: x.replace(".XLSX", ""))
print(df_cdpk_clean['origin short'].value_counts())
df_cdpk_clean.head(2)

# %%
df_cdpk_origin = df_cdpk[['origin of the file', 'Category', 'Year']].copy()
df_cdpk_origin['origin short'] = df_cdpk_origin['origin of the file'].apply(lambda x: " ".join(x.split("EN_")[1].split('_')[:2]))
df_cdpk_origin['origin short'] = df_cdpk_origin['origin short'].apply(lambda x: re.sub(r'\d+', '', x).replace(" ", "_").upper())
df_cdpk_origin['origin short'] = df_cdpk_origin['origin short'].apply(lambda x: x.replace(".XLSX", ""))

cat_cdpk_count = pd.DataFrame(df_cdpk_origin['origin short'].value_counts()).reset_index()
cat_cdpk_count_per_year = df_cdpk_origin.groupby('Year')['origin short'].value_counts().reset_index()
print(cat_cdpk_count.head(), cat_cdpk_count_per_year.head())

#############################################################################

# %%
######## Teachers' results dataset ########
kwargs = dict(
    on_bad_lines='skip',
    sep=';',
)

### Data overall teachers' results

# 2016 & 2022 not in cdpk
#df_2016 = pd.read_csv(DATA_DIR / "ECEP-2016" / "20190206_Resultados_ECEP_2016_Docentes_PUBL.csv", **kwargs)
df_2017 = pd.read_csv(DATA_DIR / "ECEP-2017" / "20190207_Resultados_ECEP_2017_Preguntas_Cerradas_PUBL.csv", **kwargs)
df_2018 = pd.read_csv(DATA_DIR / "ECEP-2018" / "20200827_Resultados_ECEP_2018_Preguntas_Cerrada_PUBL.csv", **kwargs)
df_2019 = pd.read_csv(DATA_DIR / "ECEP-2019" / "20211109_Resultados_ECEP_2019_Docentes_PUBL.csv", **kwargs)
df_2020 = pd.read_csv(DATA_DIR / "ECEP-2020" / "20211109_Resultados_ECEP_2020_Docentes_PUBL.csv", **kwargs)
df_2021 = pd.read_csv(DATA_DIR / "ECEP-2021" / "20230228_Resultados_ECEP_2021_Docentes_PUBL.csv", **kwargs)
#df_2022 = pd.read_csv(DATA_DIR / "ECEP-2022" / "240221_Resultados_ECEP_2022_Docentes_PUBL.csv",   **kwargs)

all_dfs = {"2017": df_2017, 
           "2018": df_2018, 
           "2019": df_2019, 
           "2020": df_2020, 
           "2021": df_2021,
           }
for year, df in all_dfs.items():
    print(f"{year}: {df.shape}, columns: {df.columns.tolist()}")

def clean_columns(df):
    # clean the last columns PTJE_PC, PTJE_FINAL 
    df["PTJE_PC_processed"] = df["PTJE_PC"].apply(lambda x: x.split(",")[0] if len(x.split(",")) >= 1 else np.nan)
    df["PTJE_PC_score"] = df["PTJE_PC"].apply(lambda x: x.split(",")[1] if len(x.split(",")) == 2 else np.nan)
    df['PTJE_FINAL_processed'] = df['PTJE_FINAL'].apply(lambda x: x.split(",")[0] if len(x.split(",")) >= 1 else np.nan)
    df['PTJE_FINAL_score'] = df['PTJE_FINAL'].apply(lambda x: x.split(",")[1] if len(x.split(",")) == 2 else np.nan)
    return df

# Custom cleaning for 2018
df_2018_clean = df_2018.copy()
df_2018_clean['PROM_DOM_DOCENTE'] = df_2018_clean['PROM_DOM_DOCENTE'].apply(lambda x: x.replace(",", "."))

df_2019 = clean_columns(df_2019)
df_2020 = clean_columns(df_2020)
df_2021 = clean_columns(df_2021)

# removing rows with nan in columns with score
col_nans = ['PTJE_PC_score', 'PTJE_FINAL_score']
df_2019_cleaned = df_2019.dropna(subset=col_nans).reset_index(drop=True)
df_2020_cleaned = df_2020.dropna(subset=col_nans).reset_index(drop=True)
df_2021_cleaned = df_2021.dropna(subset=col_nans).reset_index(drop=True)

all_dfs_cleaned = {
    "2017": df_2017,
    "2018": df_2018_clean,
    "2019": df_2019_cleaned,
    "2020": df_2020_cleaned,
    "2021": df_2021_cleaned,
}
for year, df in all_dfs_cleaned.items():
    print(f"{year}: {df.shape}, {all_dfs[year].shape[0] - df.shape[0]} rows removed")

score_per_test_all_years = pd.DataFrame()
col_score_1 = 'PROM_DOM_DOCENTE'
col_score_2 = 'PTJE_FINAL_score'

for year, df in all_dfs_cleaned.items():
    if year in ['2017', '2018']:
        col_score = col_score_1
    else:
        col_score = col_score_2

    df[col_score] = df[col_score].astype(float)

    cat_prueba_count = df.groupby('CAT_PRUEBA').apply(lambda x: pd.Series({
        'Count': x['MRUN'].nunique(), # set nunique to count unique teachers (several lines per teacher in 2017 and 2018)
        'Score mean': x[col_score].mean(),
    }), include_groups = False).sort_values(by='Count', ascending=False).reset_index()
    # insert column year in position 0
    cat_prueba_count.insert(0, 'Year', year)
    score_per_test_all_years = pd.concat([score_per_test_all_years, cat_prueba_count]).reset_index(drop=True)

# clean column CAT_PRUEBA to merge with cdpk later
score_per_test_all_years.insert(2, 'CAT_PRUEBA_clean', score_per_test_all_years['CAT_PRUEBA'].apply(lambda x: x.replace("-", "_").split("(")[0].strip()))

print(score_per_test_all_years.shape)
print(score_per_test_all_years['Year'].value_counts())
score_per_test_all_years.head()


# %%
### Data Pedagogy teachers' results

df_2017_per_domain = pd.read_csv(DATA_DIR / "ECEP-2017" / "20190207_Resultados_ECEP_2017_Preguntas_Cerradas_PUBL.csv", **kwargs)
df_2018_per_domain = pd.read_csv(DATA_DIR / "ECEP-2018" / "20200827_Resultados_ECEP_2018_Preguntas_Cerrada_PUBL.csv", **kwargs)
df_2019_per_domain = pd.read_csv(DATA_DIR / "ECEP-2019" / "20211109_Resultados_ECEP_2019_Dominio_PUBL.csv", **kwargs)
df_2020_per_domain = pd.read_csv(DATA_DIR / "ECEP-2020" / "20211109_Resultados_ECEP_2020_Dominio_PUBL.csv", **kwargs)
df_2021_per_domain = pd.read_csv(DATA_DIR / "ECEP-2021" / "20230228_Resultados_ECEP_2021_Dominio_PUBL.csv", **kwargs)
#df_2022_per_domain = pd.read_csv(DATA_DIR / "ECEP-2022" / "240221_Resultados_ECEP_2022_Dominio_PUBL.csv", **kwargs)

all_dfs_per_domain = {
    "2017": df_2017_per_domain,
    "2018": df_2018_per_domain,
    "2019": df_2019_per_domain,
    "2020": df_2020_per_domain,
    "2021": df_2021_per_domain,
}

# clean columns of 2018, replace , by .
df_2018_per_domain['PROM_DOM_DOCENTE'] = df_2018_per_domain['PROM_DOM_DOCENTE'].apply(lambda x: x.replace(",", "."))
df_2018_per_domain['PROM_DOM_NACIONAL'] = df_2018_per_domain['PROM_DOM_NACIONAL'].apply(lambda x: x.replace(",", "."))

print("All tests per domain:")
for year, df in all_dfs_per_domain.items():
    print(f"{year}: {df.shape}, columns: {df.columns.tolist()}")

def retrieve_tests_pedagogy(df):
    df_out = df.copy()
    df_out = df_out[df['GLOSA_DOMINIO'].str.contains("ENSEÑANZA")].reset_index(drop=True)
    return df_out

df_2017_pedagogy = retrieve_tests_pedagogy(df_2017_per_domain)
df_2018_pedagogy = retrieve_tests_pedagogy(df_2018_per_domain)
df_2019_pedagogy = retrieve_tests_pedagogy(df_2019_per_domain)
df_2020_pedagogy = retrieve_tests_pedagogy(df_2020_per_domain)
df_2021_pedagogy = retrieve_tests_pedagogy(df_2021_per_domain)

all_dfs_pedagogy = {
    "2017": df_2017_pedagogy,
    "2018": df_2018_pedagogy,
    "2019": df_2019_pedagogy,
    "2020": df_2020_pedagogy,
    "2021": df_2021_pedagogy,
}

print("\nAll tests pedagogy:")
for year, df in all_dfs_pedagogy.items():
    print(f"{year}: {df.shape}, columns: {df.columns.tolist()}")

score_per_test_all_years_pedagogy = pd.DataFrame()
col_score = 'PROM_DOM_DOCENTE'

for year, df in all_dfs_pedagogy.items():
    df[col_score] = df[col_score].astype(float)

    cat_prueba_count = df.groupby('CAT_PRUEBA').apply(lambda x: pd.Series({
        'Count': x['MRUN'].nunique(),
        'Score mean': x[col_score].mean(),
    }), include_groups = False).sort_values(by='Count', ascending=False).reset_index()
    # insert column year in position 0
    cat_prueba_count.insert(0, 'Year', year)
    score_per_test_all_years_pedagogy = pd.concat([score_per_test_all_years_pedagogy, cat_prueba_count]).reset_index(drop=True)

# clean column CAT_PRUEBA to merge with cdpk later
score_per_test_all_years_pedagogy.insert(2, 'CAT_PRUEBA_clean', score_per_test_all_years_pedagogy['CAT_PRUEBA'].apply(lambda x: x.replace("-", "_").split("(")[0].strip()))

print(score_per_test_all_years_pedagogy.shape)
print(score_per_test_all_years_pedagogy['Year'].value_counts())
score_per_test_all_years_pedagogy.head()

# %%
# Plot CDPK vs teachers' results per test and per year
# %%
fig, axs = plt.subplots(1,3, figsize=(20, 15), sharey=True)
sns.countplot(data = df_cdpk_clean, 
              y = 'origin short', 
              hue = 'Year', 
              palette='tab10',
              ax = axs[0])
axs[0].set_title("CDPK dataset", fontsize=20)
axs[0].tick_params(axis='y', labelsize=14)
sns.barplot(data = score_per_test_all_years[score_per_test_all_years['CAT_PRUEBA_clean'].isin(df_cdpk_clean['origin short'].unique())], 
            y = 'CAT_PRUEBA_clean', 
            x = 'Count', 
            hue = 'Year', 
            ax = axs[1])
axs[1].set_title("Teachers' results", fontsize=20)
sns.barplot(data = score_per_test_all_years_pedagogy[score_per_test_all_years_pedagogy['CAT_PRUEBA_clean'].isin(df_cdpk_clean['origin short'].unique())], 
            y = 'CAT_PRUEBA_clean', 
            x = 'Count', 
            hue = 'Year', 
            ax = axs[2])
axs[2].set_title("Teachers' results (Pedagogy only)", fontsize=20)
for i in range(3):
    axs[i].set_ylabel("Test", fontsize=20)
    axs[i].tick_params(axis='x', labelsize=15)
    axs[i].set_xlabel("Number of tests", fontsize=20)
plt.tight_layout()
plt.show()


# %%
# compute score as weighted average of categories
#def compute_score(cat_cdpk_count, score_per_test_all_years, year = None, verbose = False):
#
#    score = 0
#    N_total = 0
#    N_teacher_test_used = 0
#    tests_missing = []
#    for i, row in cat_cdpk_count.iterrows():
#        test = row['origin short']
#        N_samples = row['count']
#        # look for score in score_per_test_all_years
#        if year is not None:
#            score_per_test = score_per_test_all_years[score_per_test_all_years['Year'] == year]
#            if test in score_per_test['CAT_PRUEBA_clean'].values:
#                score += N_samples * score_per_test[score_per_test['CAT_PRUEBA_clean'] == test]['Score mean'].values[0]
#                N_total += N_samples
#            else:
#            #    print(f"Test {test} not found in {year} results, {N_samples} samples missed")
#                tests_missing.append(test)
#        else:
#            if test in score_per_test_all_years['CAT_PRUEBA_clean'].values:
#                score += N_samples * score_per_test_all_years[score_per_test_all_years['CAT_PRUEBA_clean'] == test]['Score mean'].mean()
#                N_total += N_samples
#            else:
#                #print(f"Test {test} not found in results, {N_samples} samples missed")
#                tests_missing.append(test)
#
#    # print number of items not matched
#    print(f"Number of items not matched: {cat_cdpk_count['count'].sum() - N_total}, ({round((cat_cdpk_count['count'].sum() - N_total) / cat_cdpk_count['count'].sum() * 100,2)} %)\nMissing Tests: {tests_missing}")
#    #print(f"   Tests missing: {tests_missing}")
#            
#    score_teachers = score / N_total
#    return score_teachers.round(3)



# %% 
# Approach 1: compute score for teachers in CDPK dataset (match per test)
# Assign score directly to each question

df_cdpk_teachers = df_cdpk_clean.copy()
df_cdpk_teachers['Teachers score est. 1'] = None
N_tests_not_found = 0
missing_tests = []

for i, row in df_cdpk_teachers.iterrows():
    test = row['origin short']
    year = row['Year']
    # Matching on test and year simultaneously
    #tmp = score_per_test_all_years[(score_per_test_all_years['Year'] == str(year)) & (score_per_test_all_years['CAT_PRUEBA_clean'] == test)]
    #if len(tmp) == 1:
    #    df_cdpk_teachers.at[i, 'Teachers score est. 1'] = tmp['Score mean'].astype(float).values[0]
    # Averaging across all years
    if len(score_per_test_all_years[score_per_test_all_years['CAT_PRUEBA_clean'] == test]) > 0:
        df_cdpk_teachers.at[i, 'Teachers score est. 1'] = score_per_test_all_years[score_per_test_all_years['CAT_PRUEBA_clean'] == test]['Score mean'].mean()
    else:
        N_tests_not_found += 1
        #missing_tests.append((test, year))
        missing_tests.append(test)

# %%
print(f"Number of items not matched: {N_tests_not_found}, ({np.round(N_tests_not_found / df_cdpk_teachers.shape[0] * 100,2)} %)\nMissing tests: {set(missing_tests)}")
print(f"Estimated score of teachers on CDPK dataset: {np.round(df_cdpk_teachers['Teachers score est. 1'].mean(), 3)} %")




# %%
# Approach 2: compute score for teachers in CDPK dataset considering duplicates
# import all Chilean questions extracted
path_ecep_EN_all_dropbox = BENCHMARKS_DIR / "Developing MCQs" / "Teaching Exam Questions" / "processed files" / "Chile" / "chilean_mcqs_all_original_EN.xlsx"
#path_ecep_EN_all_pc = Path(BASEDIR / "LLM_benchmark" / "data" / "Chile" / "chilean_mcqs_all_original_EN.xlsx")
ecep_EN_all = pd.read_excel(path_ecep_EN_all_dropbox)
print(ecep_EN_all.shape)
ecep_EN_all.head(2)

# %%
def clean_origin_of_file_row(name):
    name_clean = " ".join(name.split("EN_")[1].replace(".xlsx", "").split('_')[0:2])
    year = "20" + name_clean[-2:]
    name_clean = re.sub(r'\d+', '', name_clean).replace(" ", "_").upper()
    return (name_clean, year)

df_cdpk_origin_duplicates = df_cdpk_clean.copy()
df_cdpk_origin_duplicates['origin all'] = None

for i, row in tqdm(df_cdpk_origin_duplicates.iterrows()):
    # check if the question has a duplicate in the range of fuzzy ratio above 70 in the cdpk dataset
    all_origins = [clean_origin_of_file_row(row['origin of the file'])]

    for j, row_j in ecep_EN_all.iterrows():
        score = fuzz.ratio(row['Question'], row_j['Question'])
        if score >= 70:
            all_origins.append(clean_origin_of_file_row(row_j['origin of the file']))
    
    df_cdpk_origin_duplicates.at[i, 'origin all'] = all_origins

df_cdpk_origin_duplicates.head(3)

# %%
# Assign score directly to each question
df_cdpk_teachers['Teachers score est. 1 (with dupli)'] = None
missing_tests = []
for i, row in df_cdpk_origin_duplicates.iterrows():
    # compute the estimated score for each question
    teachers_score = 0
    N_matches = 0
    for test, year in row['origin all']:
        # Matching on test and year simultaneously
        #tmp = score_per_test_all_years[(score_per_test_all_years['Year'] == str(year)) & (score_per_test_all_years['CAT_PRUEBA_clean'] == test)]
        #if len(tmp) == 1:
        #    teachers_score += tmp['Score mean'].values[0]
        #    N_matches += 1
        # Average across all years
        if len(score_per_test_all_years[score_per_test_all_years['CAT_PRUEBA_clean'] == test]) > 0:
            teachers_score += score_per_test_all_years[score_per_test_all_years['CAT_PRUEBA_clean'] == test]['Score mean'].mean()
            N_matches += 1
        else:
            #missing_tests.append((test, year))
            missing_tests.append(test)

    df_cdpk_teachers.loc[i, 'Teachers score est. 1 (with dupli)'] = teachers_score / N_matches if N_matches > 0 else np.nan

# %%
print(f"Number of items not matched: {df_cdpk_teachers['Teachers score est. 1 (with dupli)'].isna().sum()}, ({np.round(df_cdpk_teachers['Teachers score est. 1 (with dupli)'].isna().sum() / df_cdpk_teachers.shape[0] * 100,2)} %)\nMissing tests: {set(missing_tests)}")
print(f"Estimated score of teachers on CDPK dataset (considering duplicates): {df_cdpk_teachers['Teachers score est. 1 (with dupli)'].mean().round(3)} %")







# %%
# Approach 3: pedagogy questions only
df_cdpk_teachers['Teachers score est. (pedagogy only)'] = None
missing_tests = []

for i, row in df_cdpk_teachers.iterrows():
    test = row['origin short']
    year = row['Year']
    # Matching on test and year simultaneously
    #tmp = score_per_test_all_years_pedagogy[(score_per_test_all_years_pedagogy['Year'] == str(year)) & (score_per_test_all_years_pedagogy['CAT_PRUEBA_clean'] == test)]
    #if len(tmp) == 1:
    #    df_cdpk_teachers.at[i, 'Teachers score est. (pedagogy only)'] = tmp['Score mean'].astype(float).values[0]
    # Averaging across all years
    if len(score_per_test_all_years_pedagogy[score_per_test_all_years_pedagogy['CAT_PRUEBA_clean'] == test]) > 0:
        df_cdpk_teachers.at[i, 'Teachers score est. (pedagogy only)'] = score_per_test_all_years_pedagogy[score_per_test_all_years_pedagogy['CAT_PRUEBA_clean'] == test]['Score mean'].mean()
    else:
        N_tests_not_found += 1
        #missing_tests.append((test, year))
        missing_tests.append(test)

# %%
print(f"Number of items not matched: {df_cdpk_teachers['Teachers score est. (pedagogy only)'].isna().sum()}, ({np.round(df_cdpk_teachers['Teachers score est. (pedagogy only)'].isna().sum() / df_cdpk_teachers.shape[0] * 100,2)} %)\nMissing tests: {set(missing_tests)}")
print(f"Estimated score of teachers on CDPK dataset (considering pedagogy exams only): {np.round(df_cdpk_teachers['Teachers score est. (pedagogy only)'].mean(), 3)} %")




# %%
# Approach 4: pedagogy questions with duplicates
# Assign score directly to each question
df_cdpk_teachers['Teachers score est. (pedagogy only with dupli)'] = None
missing_tests = []
for i, row in df_cdpk_origin_duplicates.iterrows():
    # compute the estimated score for each question
    teachers_score = 0
    N_matches = 0
    for tuple in row['origin all']:
        test = tuple[0]
        year = tuple[1]
        # Matching on test and year simultaneously
        #tmp = score_per_test_all_years_pedagogy[(score_per_test_all_years_pedagogy['Year'] == year) & (score_per_test_all_years_pedagogy['CAT_PRUEBA_clean'] == test)]
        #if len(tmp) == 1:
        #    teachers_score += tmp['Score mean'].values[0]
        #    N_matches += 1
        # Average across all years
        if len(score_per_test_all_years_pedagogy[score_per_test_all_years_pedagogy['CAT_PRUEBA_clean'] == test]) > 0:
            teachers_score += score_per_test_all_years_pedagogy[score_per_test_all_years_pedagogy['CAT_PRUEBA_clean'] == test]['Score mean'].mean()
            N_matches += 1
        else:
            #missing_tests.append(tuple)
            missing_tests.append(test)

    df_cdpk_teachers.loc[i, 'Teachers score est. (pedagogy only with dupli)'] = teachers_score / N_matches if N_matches > 0 else np.nan

# %%
print(f"Number of items not matched: {df_cdpk_teachers['Teachers score est. (pedagogy only with dupli)'].isna().sum()}, ({np.round(df_cdpk_teachers['Teachers score est. (pedagogy only with dupli)'].isna().sum() / df_cdpk_teachers.shape[0] * 100,2)} %)\nMissing tests: {set(missing_tests)}")
print(f"Estimated score of teachers on CDPK dataset (considering pedagogy exams and duplicates): {df_cdpk_teachers['Teachers score est. (pedagogy only with dupli)'].mean().round(3)} %")

# %%
