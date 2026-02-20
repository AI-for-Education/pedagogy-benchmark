# %%
"""
Multilingual Pedagogy Benchmark Runner

This script runs the CDPK pedagogy benchmark across different languages and categories.

PARALLEL EXECUTION:
-------------------
This script is designed to run safely in parallel across multiple terminals, with each
terminal running a different language. Results are automatically organized by language
and category to prevent conflicts.

USAGE EXAMPLES:
--------------
Single language:
    uv run python scripts/run_pedagogy_benchmark_multilingual.py \\
        --language english \\
        --benchmark cdpk \\
        --models-config full_list_20251015_small

Parallel execution (run these in separate terminals):
    Terminal 1: uv run python scripts/run_pedagogy_benchmark_multilingual.py --language english --benchmark cdpk
    Terminal 2: uv run python scripts/run_pedagogy_benchmark_multilingual.py --language luganda --benchmark cdpk
    Terminal 3: uv run python scripts/run_pedagogy_benchmark_multilingual.py --language swahili --benchmark cdpk

Specific categories only:
    uv run python scripts/run_pedagogy_benchmark_multilingual.py \\
        --language english \\
        --benchmark cdpk \\
        --categories science maths literacy

OUTPUT STRUCTURE:
----------------
Results are saved to language-specific directories:
    data/results/English/
    data/results/Luganda/
    data/results/Swahili/
    etc.

Each directory contains:
    - cdpk_results_accuracy_{models_config}.csv  (accuracy per model)
    - cdpk_results_bad_format_{models_config}.csv (format errors per model)
    - cdpk_results_full_{models_config}.csv      (complete results)

CACHING:
--------
Results are cached per model and question configuration. If you re-run the same
language+category+model combination, it will use cached results instead of calling
the API again. Cache files are stored in:
    data/cache_local/CDPK_{language}_{category}/resps_{model}.csv
"""
# %%
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Optional
import sys

import pandas as pd
from dotenv import load_dotenv
from fdllm.sysutils import register_models

from cdpk.benchmark_run import run_benchmark
from cdpk.benchmark_constants import ROOT
from cdpk.benchmark_utils import fulldf_accuracy_by_category

# Add parent directory to path to import cdpk module
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cdpk.language_prompts import get_language_config, list_available_languages

load_dotenv(override=True)


def main(opt):
    # Get language configuration
    config = get_language_config(opt.language)
    language_slug = config['slug']

    # Build questions list dynamically based on language and categories
    QUESTIONS_LIST_DICT = {
        "cdpk": [f"CDPK_{language_slug}_{cat}" for cat in opt.categories],
        "send": [f"CDPK_{language_slug}_send"],
    }

    if opt.models_config is None:
        if opt.benchmark == "cdpk":
            #opt.models_config = "cdpk_online_leaderboard"
            opt.models_config = "full_list_20251015_small"
            #opt.models_config = "full_list_default_models_20260218"
        elif opt.benchmark == "send":
            opt.models_config = "send_online_leaderboard"
        else:
            raise NotImplementedError(f"'benchmark' must be one of 'send' or 'cdpk'")
    config_models_PK = opt.models_config
    
    if opt.output_folder is None:
        res_dir = ROOT / "data" / "results" / language_slug
    else:
        res_dir = Path(opt.output_folder) / language_slug
    res_dir.mkdir(exist_ok=True, parents=True)
    
    category_df_list = []
    accuracies_df = pd.DataFrame()
    bad_format_df = pd.DataFrame()
    length_per_category = {}
    # summary_df_chile = pd.DataFrame()
    for cat_config_name in QUESTIONS_LIST_DICT[opt.benchmark]:
        print(f"Running {opt.benchmark} benchmark for {cat_config_name}")
        category_df, summary_df, config, models_dict_PK = run_benchmark(
            questions_config=cat_config_name, models_config=config_models_PK, language=opt.language
        )
        if opt.benchmark == "send":
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
        "--language", required=True, type=str, choices=list_available_languages(),
        help='Target language for benchmarking'
    )
    parser.add_argument(
        "--benchmark", required=True, type=str, choices=["cdpk", "send"]
    )
    parser.add_argument(
        "--categories", nargs='+',
        default=["science", "literacy", "creative_arts", "maths", "social_studies", "technology", "general"],
        help='Categories to run (space-separated list)'
    )
    parser.add_argument(
        "--models-config", required=False, type=str, default=None,
    )
    parser.add_argument("--output-folder", required=False, type=str, default=None)
    parser.add_argument("--custom-models-file", required=False, type=str, default=None)
    opt = parser.parse_args()

    ### register custom models
    if opt.custom_models_file is None:
        custom_models_file = ROOT / "fab-benchmarks-configs" / "custom_models.yaml"
    else:
        custom_models_file = Path(opt.custom_models_file)
    if not custom_models_file.exists():
        raise FileNotFoundError(f"File {str(custom_models_file)} does not exist")
    register_models(custom_models_file)
    #

    main(opt)

# %%
