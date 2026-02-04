from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from fdllm.sysutils import register_models

from cdpk.benchmark_run import run_benchmark
from cdpk.benchmark_constants import ROOT
from cdpk.benchmark_utils import fulldf_accuracy_by_category

load_dotenv(override=True)

QUESTIONS_LIST_DICT = {
    "cdpk": [
        "CDPK_science",
        "CDPK_literacy",
        "CDPK_creative_arts",
        "CDPK_maths",
        "CDPK_social_studies",
        "CDPK_technology",
        "CDPK_gen_pk",
    ],
    "send": ["CDPK_send"],
}

def main(opt):
    if opt.models_config is None:
        if opt.benchmark == "cdpk":
            opt.models_config = "cdpk_online_leaderboard"
        elif opt.benchmark == "send":
            opt.models_config = "send_online_leaderboard"
        else:
            raise NotImplementedError(f"'benchmark' must be one of 'send' or 'cdpk'")
    config_models_PK = opt.models_config
    #
    if opt.output_folder is None:
        res_dir = ROOT / "data" / "results"
    else:
        res_dir = Path(opt.output_folder)
    res_dir.mkdir(exist_ok=True, parents=True)
    #
    category_df_list = []
    accuracies_df = pd.DataFrame()
    bad_format_df = pd.DataFrame()
    length_per_category = {}
    # summary_df_chile = pd.DataFrame()
    for cat_config_name in QUESTIONS_LIST_DICT[opt.benchmark]:
        print(f"Running {opt.benchmark} benchmark for {cat_config_name}")
        category_df, summary_df, config, models_dict_PK = run_benchmark(
            questions_config=cat_config_name, models_config=config_models_PK
        )
        if opt.benchmark == "send":
            category_df["Category"] = "SEND"
        if len(category_df["Category"].unique()) > 1:
            print("More than one category in the dataframe")
            break
        else:
            category_name = category_df["Category"].unique()[0]
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
    if opt.benchmark == "send":
        full_df["question_id"] = full_df["question_id"] + 920
    accuracy_overall = fulldf_accuracy_by_category(
        fulldf=full_df, models_dict=models_dict_PK, bad_format_threshold=None
    )
    accuracies_df["Overall"] = accuracy_overall["Accuracy"]
    bad_format_df["Overall"] = accuracy_overall["Bad Format"]

    ###### save results
    acc_file = res_dir / f"{opt.benchmark}_results_accuracy_{opt.models_config}.csv"
    bf_file = res_dir / f"{opt.benchmark}_results_bad_format_{opt.models_config}.csv"
    full_file = res_dir / f"{opt.benchmark}_results_full_{opt.models_config}.csv"
    accuracies_df.to_csv(acc_file)
    bad_format_df.to_csv(bf_file)
    full_df.to_csv(full_file, index=False)
    accuracies_df.index = accuracies_df.index.map(models_dict_PK)
    print(accuracies_df[["Overall"]].sort_values("Overall", ascending=False).to_markdown())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--benchmark", required=True, type=str, choices=["cdpk", "send"]
    )
    parser.add_argument(
        "--models-config", required=False, type=str, default=None
    )
    parser.add_argument("--output-folder", required=False, type=str, default=None)
    parser.add_argument("--custom-models-file", required=False, type=str, default=None)
    opt = parser.parse_args()

    ### register custom models
    if opt.custom_models_file is None:
        custom_models_file = ROOT / "custom_models.yaml"
    else:
        custom_models_file = Path(opt.custom_models_file)
    if not custom_models_file.exists():
        raise FileNotFoundError(f"File {str(custom_models_file)} does not exist")
    register_models(custom_models_file)
    #

    main(opt)
