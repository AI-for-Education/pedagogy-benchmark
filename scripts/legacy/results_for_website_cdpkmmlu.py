# %%
import pandas as pd
import json
from pathlib import Path
from io import StringIO
import os

from matplotlib.colors import get_named_colors_mapping
from pathlib import Path
from dotenv import load_dotenv
from fdllm.sysutils import register_models
import numpy as np
from azure.storage.blob import ContentSettings

from cdpk.benchmark_run import run_benchmark
from cdpk.benchmark_utils import collect_results_mmlu, fulldf_accuracy_by_category
from cdpk.benchmark_constants import (
    PROVIDER_MODEL_MAPPING,
    PROVIDER_COLOR_MAPPING,
    MODEL_COST_MAPPING,
    MODEL_OPEN_MAPPING,
    MODEL_SIZE_MAPPING,
    MODEL_URL_MAPPING,
    PROVIDER_SVG_MAPPING,
    ROOT
)
# from cdpk.benchmark_blob import get_blob_client, upload_blob_chunked

load_dotenv(override=True)


register_models(ROOT / "custom_models.yaml")

named_colors_mapping = get_named_colors_mapping()
provider_color_mapping_hex = {
    provider: named_colors_mapping[color]
    for provider, color in PROVIDER_COLOR_MAPPING.items()
}

OUTPUTDIR = ""

INTERMEDIATEDIR = ""

if not OUTPUTDIR or INTERMEDIATEDIR:
    raise ValueError("You must set OTPUTDIR and INTERMEDIATEDIR before running")


STORAGE_KEY = os.environ.get("SAS_TOKEN", None)


# %%
# Set to True to update the website data
update_website = True
# %%
# Collect results for Chilean PK benchmark
config_models_PK = "full_list_20250106"

CDPK_list = [
    "CDPK_science",
    "CDPK_literacy",
    "CDPK_creative_arts",
    "CDPK_maths",
    "CDPK_social_studies",
    "CDPK_technology",
    "CDPK_gen_pk",
]
#
full_df_chile = pd.DataFrame()
accuracies_df = pd.DataFrame()
bad_format_df = pd.DataFrame()
length_per_category = {}
# summary_df_chile = pd.DataFrame()
for cat_config_name in CDPK_list:
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
full_df_chile = full_df_chile[full_df_chile[col_pred] != "Few-shot example"]

# %%
# Collect results for MMLU
config_models_MMLU = "full_list_20250106"
accuracies_mmlu, bad_format_mmlu, nquestions_mmlu, full_df_mmlu, models_mmlu = (
    collect_results_mmlu(config_models_MMLU)
)
print(f"Number of models in the MMLU benchmark: {len(accuracies_mmlu)}")
print(accuracies_mmlu.shape, bad_format_mmlu.shape)

# # %%
# from datetime import datetime, timezone

# INTERMEDIATEDIR.mkdir(exist_ok=True, parents=True)

# datestr = datetime.now(timezone.utc).strftime(r"%Y%m%d")
# full_df_chile.to_csv(INTERMEDIATEDIR / f"singleq_CDPK_{datestr}.csv", index=False)
# full_df_mmlu.to_csv(INTERMEDIATEDIR / f"singleq_MMLU_{datestr}.csv", index=False)


# %%
# read json file with categories mapping
with open(ROOT / "chile_mmlu_matching_dict_new.json", "r") as file:
    chile_mmlu_matching_dict = json.load(file)

cdpk_categories = list(chile_mmlu_matching_dict.keys())

print(cdpk_categories)

# %%
### CDPK data
cdpk_all_categories = sorted(full_df_chile.Category.dropna().unique())
accuracies_cdpk_categories = {
    cat: fulldf_accuracy_by_category(
        fulldf=full_df_chile,
        models_dict=models_dict_PK,
        category=cat,
        bad_format_threshold=None,
    )
    for cat in cdpk_all_categories
}

accuracies_cdpk_overall = fulldf_accuracy_by_category(
    fulldf=full_df_chile, models_dict=models_dict_PK, bad_format_threshold=None
)


def is_primary(x):
    if pd.isna(x):
        return False
    else:
        return ("Primary" in x and not "Pre-Primary" in x) or x == "All"


def is_secondary(x):
    if pd.isna(x):
        return False
    else:
        return "Secondary" in x or x == "All"


full_df_chile["Primary"] = full_df_chile["Age Group"].apply(is_primary)
full_df_chile["Secondary"] = full_df_chile["Age Group"].apply(is_secondary)

accuracies_cdpk_levels = {
    level: fulldf_accuracy_by_category(
        fulldf=full_df_chile,
        models_dict=models_dict_PK,
        category=1,
        column=level,
        bad_format_threshold=None,
    )
    for level in ["Primary", "Secondary"]
}

# %%
#### MMLU data
mmlu_all_categories = sorted(full_df_mmlu.Category.dropna().unique())
accuracies_mmlu_categories = {
    cat: fulldf_accuracy_by_category(
        fulldf=full_df_mmlu,
        models_dict=models_mmlu,
        category=cat,
        bad_format_threshold=None,
    )
    for cat in mmlu_all_categories
}

accuracies_mmlu_overall = fulldf_accuracy_by_category(
    fulldf=full_df_mmlu, models_dict=models_mmlu, bad_format_threshold=None
)

level_map = {
    "Primary": ["Elementary"],
    "Secondary": ["High School"],
    "Tertiary": ["College", "Other", "Professional"],
}
level_cats = {
    level: [
        f"MMLU_{val['file'].split('.')[0]}"
        for mmluvals_ in chile_mmlu_matching_dict.values()
        for val in mmluvals_
        if val["level"] in mmlu_level
    ]
    for level, mmlu_level in level_map.items()
}

accuracies_mmlu_levels = {
    level: fulldf_accuracy_by_category(
        fulldf=full_df_mmlu,
        models_dict=models_dict_PK,
        category=level_cats[level],
        bad_format_threshold=None,
    )
    for level in level_map
}

# %%
#### comparison data
accuracies_cdpk_categories_comparable = {}
accuracies_mmlu_categories_comparable = {}

for cdpk_cat, mmlu_match_list in chile_mmlu_matching_dict.items():

    *cat, suff = cdpk_cat.split()
    if suff == "PCK":
        cat = " ".join(cat)
    else:
        cat = cdpk_cat

    mmlu_cats = [
        f"MMLU_{mmlu_match['file'].split('.')[0]}"
        for mmlu_match in mmlu_match_list
        if mmlu_match["level"] in ["Elementary", "High School"]
    ]

    if not mmlu_cats:
        continue

    print(cat)
    print(mmlu_cats)

    accuracies_cdpk_categories_comparable[cat] = fulldf_accuracy_by_category(
        fulldf=full_df_chile,
        models_dict=models_dict_PK,
        category=cdpk_cat,
        bad_format_threshold=None,
    )
    accuracies_mmlu_categories_comparable[cat] = fulldf_accuracy_by_category(
        fulldf=full_df_mmlu,
        models_dict=models_dict_PK,
        category=mmlu_cats,
        bad_format_threshold=None,
    )

# %%
### save metadata
metadata = {
    provider: {
        "metadata": {
            "name": provider,
            "displayName": provider,
            "color": provider_color_mapping_hex[provider],
            "logo": PROVIDER_SVG_MAPPING[provider],
        },
        "models": [
            {"name": model, "displayName": models_dict_PK[model]}
            for model in model_list if model in models_dict_PK
        ],
    }
    for provider, model_list in PROVIDER_MODEL_MAPPING.items()
}

# save metadata as json with indent 4
#with open("leaderboard_metadata_test.json", "w") as f:
#    json.dump(metadata, f, indent=4)

# metadata = {
#     "provider_model_mapping": PROVIDER_MODEL_MAPPING,
#     "provider_color_mapping": provider_color_mapping_hex,
#     "model_display_name_mapping": models_dict_PK,
# }
# with open(OUTPUTDIR / "leaderboard_metadata.json", "w") as f:
#     json.dump(metadata, f, indent=2)

# if update_website:
#     blob = get_blob_client("leaderboard_metadata.json", credential=STORAGE_KEY)
#     backup_metadata = json.loads(blob.download_blob().readall())
#     with StringIO(json.dumps(metadata, indent=2, allow_nan=False)) as blobdata:
#         upload_blob_chunked(
#             blob,
#             blobdata,
#             content_settings=ContentSettings(content_type="application/json"),
#         )


# %%
### format data
def calculate_cost(model, combined=False):
    rate = MODEL_COST_MAPPING.get(model)
    if rate is None:
        return
    else:
        if combined:
            return float((np.array(rate) * np.array([0.8, 0.2])).sum())
        else:
            return float((np.array(rate) * np.array([1.0, 0.0])).sum())


def model_meta(model):
    return {
        "model": model,
        "cost": calculate_cost(model),
        "size": MODEL_SIZE_MAPPING.get(model),
        "open": MODEL_OPEN_MAPPING.get(model),
        "website": MODEL_URL_MAPPING.get(model),
        "provider": model_provider_mapping[model],
        "displayName": models_dict_PK[model],
    }


def res_data(res):
    return {
        "accuracy": res["Accuracy"],
        "bad_format": res["Bad Format"],
        "n": int(res["N"]),
    }


model_provider_mapping = {
    model: [
        provider
        for provider, models in PROVIDER_MODEL_MAPPING.items()
        if model in models
    ]
    for model in models_dict_PK
}
for model, providers in model_provider_mapping.items():
    if len(providers) != 1:
        raise
    else:
        model_provider_mapping[model] = providers[0]

flat_res_overall = []
for domain, domain_df in zip(
    ["CDPK", "MMLU"],
    [accuracies_cdpk_overall, accuracies_mmlu_overall],
):
    for model, res in domain_df.iterrows():
        resobj = {
            "benchmark_domain": domain,
            "subject": None,
            "level": None,
            **model_meta(model),
            **res_data(res),
        }
        flat_res_overall.append(resobj)

# out_file_subjects_overall = OUTPUTDIR / "leaderboard_data_overall.json"
# with open(out_file_subjects_overall, "w") as f:
#     json.dump(flat_res_overall, f, indent=2)

flat_res_subjects_comparable = []
for domain, domain_dict in zip(
    ["CDPK", "MMLU"],
    [accuracies_cdpk_categories_comparable, accuracies_mmlu_categories_comparable],
):
    for subject, df in domain_dict.items():
        for model, res in df.iterrows():
            resobj = {
                "benchmark_domain": domain,
                "subject": subject,
                "level": None,
                **model_meta(model),
                **res_data(res),
            }
            flat_res_subjects_comparable.append(resobj)

# out_file_subjects_comparable = OUTPUTDIR / "leaderboard_data_subjects_comparable.json"
# with open(out_file_subjects_comparable, "w") as f:
#     json.dump(flat_res_subjects_comparable, f, indent=2)

flat_res_levels_comparable = []
for domain, domain_dict in zip(
    ["CDPK", "MMLU"],
    [accuracies_cdpk_levels, accuracies_mmlu_levels],
):
    for level, df in domain_dict.items():
        for model, res in df.iterrows():
            resobj = {
                "benchmark_domain": domain,
                "subject": None,
                "level": level,
                **model_meta(model),
                **res_data(res),
            }
            flat_res_levels_comparable.append(resobj)

out_file_comparable = OUTPUTDIR / "leaderboard_data_comparable.json"
# with open(out_file_comparable, "w") as f:
#     json.dump(
#         flat_res_overall + flat_res_subjects_comparable + flat_res_levels_comparable,
#         f,
#         indent=2,
#     )

flat_res_full = (
    flat_res_overall + flat_res_subjects_comparable + flat_res_levels_comparable
)

# if update_website:
#     blob = get_blob_client("leaderboard_data_comparable.json", credential=STORAGE_KEY)
#     backup_data_comparable = json.loads(blob.download_blob().readall())
#     with StringIO(json.dumps(flat_res_full, indent=2, allow_nan=False)) as blobdata:
#         upload_blob_chunked(
#             blob,
#             blobdata,
#             content_settings=ContentSettings(content_type="application/json"),
#         )


flat_res_subjects_not_comparable = []
for domain, domain_dict in zip(
    ["CDPK", "MMLU"],
    [accuracies_cdpk_categories, accuracies_mmlu_categories],
):
    for subject, df in domain_dict.items():
        for model, res in df.iterrows():
            resobj = {
                "benchmark_domain": domain,
                "subject": subject,
                "level": None,
                **model_meta(model),
                **res_data(res),
            }
            flat_res_subjects_not_comparable.append(resobj)

out_file_subjects_not_comparable = (
    OUTPUTDIR / "leaderboard_data_subjects_not_comparable.json"
)
# %%
# Save metadata and results as json files in the repo so we can track history
with open(DATADIR / "web" / "metadata_leaderboard.json", "w") as f:
    json.dump(metadata, f, indent=2, allow_nan=False)

with open(DATADIR / "web" / "flat_res_full.json", "w") as f:
     json.dump(flat_res_full, f, indent=2, allow_nan=False)

print("Web data updated, please DVC add and commit")


# %%

# {"CDPK": "Pedagogical Knowledge", "MMLU": "Content Knowledge"}

# {
#     "benchmark domain":
#         [
#             {
#                 "benchmark domain": str
#                 "subject": str | None
#                 "model": str,
#                 "accuracy": float,
#                 "bad_format": float,
#                 "n": int
#             }
#         ]
#         {
#             "subject": {
#                 "model": {"accuracy": float, "bad_format": float, "n": int}
#             }
#         }
# }

# [
#     {
#         "full_name": str,
#         "display_name": str,
#         "provider": str
#     }
# ]

