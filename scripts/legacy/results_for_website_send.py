# %%
import json
import pandas as pd
from cdpk.benchmark_utils import ROOT as PROJECT_ROOT
from matplotlib import pyplot as plt

import matplotlib.patches as patches
import matplotlib.lines as mlines
from pyfonts import load_font
from matplotlib import font_manager
from matplotlib.colors import get_named_colors_mapping

from pathlib import Path
from dotenv import load_dotenv
from fdllm.sysutils import register_models

from cdpk.benchmark_run import run_benchmark
from cdpk.benchmark_utils import fulldf_accuracy_by_category
from cdpk.benchmark_constants import (
    PROVIDER_MODEL_MAPPING,
    PROVIDER_COLOR_MAPPING,
    MODEL_COST_MAPPING,
    MODEL_OPEN_MAPPING,
    MODEL_SIZE_MAPPING,
    MODEL_URL_MAPPING,
    PROVIDER_SVG_MAPPING,
)
from io import StringIO
from azure.storage.blob import ContentSettings

HERE = Path(__file__).resolve().parent
ROOT = HERE
dotenv_path = ROOT / ".env"
load_dotenv(dotenv_path, override=True)

try:
    register_models(Path.home() / ".fdllm" / "custom_models.yaml")
except:
    pass

DATA_DIR = PROJECT_ROOT / "data"
SEND_DATA_DIR = PROJECT_ROOT / "data" / "send_results"

STORAGE_KEY = os.environ.get("SAS_TOKEN", None)

# %%
update_website = True

# %%
config = "CDPK_send"
full_df_send, summary_df_send, config, models_dict_PK = run_benchmark(
        questions_config=config, models_config="full_list_20250106_send"
    )
print(full_df_send.shape, summary_df_send.shape)
print(summary_df_send.head(), full_df_send.head(2))

# %%
# Filter out few shot examples and ambiguous MCQs form 2nd annotation
#full_df_send_wo_fs = full_df_send.loc[3:].reset_index(drop=True)
col_pred = [col for col in full_df_send.columns if col.startswith("pred_")][0]
full_df_send_wo_fs = full_df_send[full_df_send[col_pred] != "Few-shot example"].reset_index(drop=True)
print(f"Number of MCQs after removing few-shot examples: {full_df_send_wo_fs.shape[0]}")

# Filter out questions that have been annotated
#df_hard_questions_annotated = pd.read_excel("./../data/Chile/SEND/hard_questions_cdpk_send_annotated.xlsx")
#print(f"Number of hard questions annotated: {df_hard_questions_annotated.shape[0]}")
#print(df_hard_questions_annotated["Reason for difficulty"].value_counts())

#questions_to_remove = df_hard_questions_annotated[df_hard_questions_annotated["Reason for difficulty"] != "Difficult content"].reset_index(drop=True)
#print(f"Number of questions to remove: {questions_to_remove.shape[0]}")

# remove those rows from full_df_send_wo_fs
#full_df_send_wo_fs_filtered = full_df_send_wo_fs.drop(questions_to_remove.index).reset_index(drop=True)
# Only keep the rows which Question are in CDPK_send_test_cleaned_after_removal
#full_df_send_wo_fs_filtered = full_df_send_wo_fs[
#    full_df_send_wo_fs["Question"].isin(CDPK_send_test_cleaned_after_removal["Question"])
#].reset_index(drop=True)
#print(f"Number of questions in final set (should be {len(CDPK_send_test_cleaned_after_removal)}): {full_df_send_wo_fs_filtered.shape[0]}")

accuracies_send = fulldf_accuracy_by_category(
    fulldf=full_df_send_wo_fs, models_dict=models_dict_PK, bad_format_threshold=None
)
accuracies_send = accuracies_send.rename(columns={"Accuracy": "all_categories"}) / 100
accuracies_send = accuracies_send.reset_index()
accuracies_send.head()

# %%
# Save SEND without few shot examples and ambiguous MCQs form 2nd annotation and remove all columns starting with "pred_"
#df_send_cleaned = full_df_send_wo_fs_filtered.drop(columns=[col for col in full_df_send_wo_fs_filtered.columns if col.startswith("pred_")], errors='ignore')
#print(df_send_cleaned.shape)
#df_send_cleaned.head()

# save
#df_send_cleaned.to_csv(SEND_DATA_DIR / "CDPK_send_test_cleaned.csv", index=False)

# %%
accuracies_send.to_csv(SEND_DATA_DIR / "accuracies_send.csv", index=False)

# %%
acc_df = accuracies_send.copy()
acc_df = acc_df.rename(columns={"Model": "model_id"})
models_df = pd.read_csv(DATA_DIR / "models.csv")

res_df = acc_df.merge(
    models_df[
        [
            "model_id",
            "provider",
            "open",
            "display_name",
            "url",
            "input_cost",
            "output_cost",
        ]
    ],
    on="model_id",
)

# %%
results = []
for _, row in res_df.iterrows():
    results.append(
        {
            "benchmark_domain": "SEND",
            "subject": None,
            "model": row["model_id"],
            "provider": row["provider"],
            "open": row["open"],
            "displayName": row["display_name"],
            "website": row["url"],
            "cost": None if pd.isna(row["input_cost"]) else row["input_cost"],
            "accuracy": row["all_categories"] * 100.0,
            "bad_format": row["Bad Format"] * 100.0,
        }
    )

# %%
# # Update website
# if update_website:
#     blob = get_blob_client("send_leaderboard_data.json", credential=STORAGE_KEY)
#     backup_data_comparable = json.loads(blob.download_blob().readall())
#     with StringIO(json.dumps(results, indent=2, allow_nan=False)) as blobdata:
#         upload_blob_chunked(
#             blob,
#             blobdata,
#             content_settings=ContentSettings(content_type="application/json"),
#         )

# %%
# save
(SEND_DATA_DIR / "send_leaderboard_data.json").write_text(
    json.dumps(results, indent=2, allow_nan=False)
)


# %%
### save metadata

named_colors_mapping = get_named_colors_mapping()
provider_color_mapping_hex = {
    provider: named_colors_mapping[color]
    for provider, color in PROVIDER_COLOR_MAPPING.items()
}

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

# %%
# upload website
# if update_website:
#     blob = get_blob_client("send_leaderboard_metadata.json", credential=STORAGE_KEY)
#     backup_metadata = json.loads(blob.download_blob().readall())
#     with StringIO(json.dumps(metadata, indent=2, allow_nan=False)) as blobdata:
#         upload_blob_chunked(
#             blob,
#             blobdata,
#             content_settings=ContentSettings(content_type="application/json"),
#         )


# %%
# save metadata
(SEND_DATA_DIR / "send_leaderboard_metadata.json").write_text(
    json.dumps(metadata, indent=2, allow_nan=False)
)

# %%
# Check
# count number of models in metadata, name of model
#model_count = 0
#for provider, models in PROVIDER_MODEL_MAPPING.items():
#    # count number of models in models
#        model_count += len(metadata[provider]["models"])
#print(f"Number of models in metadata: {model_count}")



# %%
# Plot results
# load font
FONT_REGULAR = load_font(
    font_url="https://github.com/google/fonts/blob/main/ofl/poppins/Poppins-Regular.ttf?raw=true"
)
FONT_BOLD = load_font(
    font_url="https://github.com/google/fonts/blob/main/ofl/poppins/Poppins-Bold.ttf?raw=true"
)

# build the color mapping from PROVIDER_COLOR_MAPPING and PROVIDER_MODEL_MAPPING
color_mapping = {}
for provider, models in PROVIDER_MODEL_MAPPING.items():
    color_mapping[PROVIDER_COLOR_MAPPING[provider]] = models
#print(json.dumps(color_mapping, indent=4))

company_to_color_dict = PROVIDER_COLOR_MAPPING

def plot_benchmark_df(
    accuracies,
    category,
    models_dict,
    colormapping,
    benchmark_name,
    folder_name,
    category_in_title=True,
    save_fig=False,
):

    fig, ax = plt.subplots(figsize=(30, 8))
    fig.tight_layout(pad=1.8)

    # Get the accuracy and bad format values, sorted by accuracy
    accuracy_values = accuracies[category].sort_values(ascending=False) * 100
    model_names_ordered = accuracy_values.index.tolist()

    # Plot the accuracy bars
    # create the color mapping based on the order of the models, the color is in the key of the dictionary colormapping
    model_colors = {}
    for color, models in colormapping.items():
        for model in models:
            model_colors[model] = color

    # Create a list of colors based on model_names_ordered
    bar_colors = []
    for model in model_names_ordered:
        # Append the corresponding color for the model, or a default color if not found
        bar_colors.append(model_colors.get(model, "grey"))
    bar_plot = ax.bar(
        model_names_ordered, accuracy_values, label="Accuracy", color=bar_colors
    )

    # Add accuracy value inside the blue bars
    for p in bar_plot:
        ax.annotate(
            f"{p.get_height():.0f}",  # Accuracy percentage
            (
                p.get_x() + p.get_width() / 2.0,
                p.get_height() + 3,
            ),  # Position in the middle of the accuracy bar
            ha="center",  # Horizontal alignment
            va="center",  # Vertical alignment
            fontsize=18,
            color="black",
        )
    model_names_short = [models_dict[model] for model in model_names_ordered]

    # Set labels and title
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.axhline(y=100, color="white", linewidth=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Accuracy (%)", fontsize=28, font=FONT_BOLD)
    if category_in_title:
        title = f"{benchmark_name} Performance - {category}"
    else:
        title = f"{benchmark_name} Performance"
    
    ax.set_title(
        title,
        font=FONT_BOLD,
        fontsize=30,
    )
    ax.set_ylim(0, 100)  # Keep the y-limit within 100%
    ax.set_xticks(range(len(accuracy_values.index)))
    ax.set_xticklabels(
        model_names_short, rotation=35, font=FONT_REGULAR, fontsize=18, ha="right"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 0.75))

    legend_elements = [
        mlines.Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="None",
            markersize=13,
            label=company,
        )
        for company, color in company_to_color_dict.items()
    ]
    ax.legend(
        handles=legend_elements,
        title="",
        loc="upper right",
        bbox_to_anchor=(1.03, 1.05),
        fontsize=13,
    )

    plt.tight_layout()
    plt.show()

    if save_fig:
        outfile = (
            ROOT
            / "data"
            / "plots"
            / folder_name
            / f"{category}_benchmark_finale.svg"
        )
        print(outfile)
        outfile.parent.mkdir(exist_ok=True, parents=True)
        # save fig as svg
        fig.savefig(outfile, format="svg", dpi=1200)

    return ax

# %%
# reset Model column in index to plot
accuracies_send_filtered_plot = accuracies_send.set_index("Model")

plot_benchmark_df(
    accuracies=accuracies_send_filtered_plot,
    category="all_categories",
    models_dict=models_dict_PK,
    colormapping=color_mapping,
    save_fig=False,
    category_in_title=False,
    benchmark_name="Pedagogical Knowledge Benchmark (SEND)",
    folder_name="Pedagogical Knowledge Benchmark paper",
)


# %%
# go from the csv in data/cache/MMLU_...
# open first file, count number of rows in each csv file in the directory, and return the total number of rows across all files.

import os
from pathlib import Path

DATA_DIR = PROJECT_ROOT / "data" / "cache"
def count_rows_in_csv_files(directory):
    total_rows = 0
    n_cat = 0
    for folder in os.listdir(directory):
        folder_path = Path(directory) / folder
        # if it's a directory and starts with "MMLU_"
        if folder_path.is_dir() and folder.startswith("MMLU_"):
            csv_files = list(folder_path.glob("*.csv"))
            for csv_file in csv_files:
                print(f"Processing file: {csv_file}")  # Debugging line to see which file is being processed
                df = pd.read_csv(csv_file)
                total_rows += len(df)
                n_cat += 1
                break
    return total_rows, n_cat

total_rows, mmlu_cat = count_rows_in_csv_files(DATA_DIR)

print(f"Number of MMLU categories processed: {mmlu_cat}")
print(f"Total number of MCQs from MMLU: {total_rows}")
# %%
