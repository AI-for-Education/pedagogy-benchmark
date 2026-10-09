"""
Build the multilingual results tables from the benchmark runs in data/results/.

Reads, for every language folder in data/results/ (e.g. Hausa_ep/), the files written by
`scripts/run_pedagogy_benchmark_multilingual.py` for one models list (configs/models/<name>.yaml,
MODELS_LIST_YAML below or --models-list-yaml):
    cdpk_results_accuracy_<name>.csv    accuracy (%) per model x category
    cdpk_results_bad_format_<name>.csv  % unparseable answers per model x category
    cdpk_results_full_<name>.csv        one row per question, pred_/Latency_/TokensUsed* per model

and writes, in data/results/:
    cdpk_multilingual_model_performance.csv           one row per (model, language, prompt, category):
                                                      accuracy, bad_format, latency mean/median, provider
    cdpk_multilingual_model_performance_detailed.csv  same keys, with per-question lists (correct,
                                                      bad_format, Latency, TokensUsed*), read back with
                                                      ast.literal_eval by create_figures.py and token_analysis.py

CLI only:
    uv run python scripts/create_results.py                       # replaces existing output files
    uv run python scripts/create_results.py --overwrite false     # stop if an output file exists
    uv run python scripts/create_results.py --models-list-yaml full_list_20251015_small

Folders
-------
- "<Language>_ep" = English prompt (english_prompt True), "<Language>" = prompt in the language.
- English results are also copied with english_prompt True (its questions are already in English).
- Folders without results for the models list are listed and skipped. Folders whose questions
  differ from the English run (e.g. French_Core300_ep, a 300-question subset) are skipped: their
  scores are not comparable with the full benchmark.

Checks (the script stops at the first failure, before writing anything)
------
R1  every accuracy file has its bad_format and full files (same config name)
R2  accuracy and bad_format files list the same models and categories
R3  the full file has the category and correct_answer columns and a pred_ column per model
R4  accuracy and bad_format recomputed from the per-question predictions match the summary files
R5  final table: one row per (model, language, english_prompt, category), every model has all
    categories, accuracy and bad_format between 0 and 100
Models missing from fab-benchmarks-configs/models.csv get provider "Unknown" (warning, not an error).

Output files are never half-written: each is written to a temporary file, then renamed.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results"
MODELS_CONFIGS_DIR = ROOT / "configs" / "models"

# Models list whose results are read (configs/models/<name>.yaml); --models-list-yaml overrides it
MODELS_LIST_YAML = "full_list_default_models_20260218"
MODELS_CSV = ROOT / "fab-benchmarks-configs" / "models.csv"
SUMMARY_FILE = RESULTS_DIR / "cdpk_multilingual_model_performance.csv"
DETAILED_FILE = RESULTS_DIR / "cdpk_multilingual_model_performance_detailed.csv"

REFERENCE_FOLDER = "English"  # its question set is the one every language must match
NON_LANGUAGE_FOLDERS = {"figures"}  # folders of data/results/ that are not language results
CATEGORIES = ["Science", "Literacy", "Creative arts", "Maths", "Social studies", "Technology", "General"]
ALL_CATEGORIES = [*CATEGORIES, "Overall"]

# Models left out of both tables
MODELS_TO_EXCLUDE = [
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-preview-09-2025",
    "gpt-5-2025-08-07-medium",
    "fw-deepseek-r1-0528",
]

# Warn (but keep the model) when its overall bad format is above this %, for a language. Unreadable
# answers count as wrong, so a high value usually means failed API calls rather than a weak model:
# investigate with count_failures.py, then rerun the affected categories or add the model to
# MODELS_TO_EXCLUDE (see scripts/README.md, steps 5 and 6).
BAD_FORMAT_WARNING_THRESHOLD = 5

# Per-question columns of the full file, per model (column name = f"{prefix}_{model}")
PER_QUESTION_COLUMNS = ["Latency", "TokensUsed", "TokensUsedCompletion", "TokensUsedReasoning"]


class ResultsError(Exception):
    """A check failed: printed as a single ERROR line by main()."""


# ==================== Discovery ====================

def discover_result_sets(results_dir):
    """Find every language folder and its result sets (one per models config).

    Returns a list of {folder, language, english_prompt, configs: [config name]}, sorted by
    folder name. Folders without any accuracy file are kept with an empty configs list.
    """
    folders = []
    for folder in sorted(p for p in results_dir.iterdir() if p.is_dir() and p.name not in NON_LANGUAGE_FOLDERS):
        configs = sorted(f.name.removeprefix("cdpk_results_accuracy_").removesuffix(".csv")
                         for f in folder.glob("cdpk_results_accuracy_*.csv"))
        folders.append({
            "folder": folder,
            "language": folder.name.removesuffix("_ep"),
            "english_prompt": folder.name.endswith("_ep"),
            "configs": configs,
        })
    return folders


def result_files(folder, config):
    """The accuracy, bad_format and full files of one result set (R1: all must exist)."""
    files = {kind: folder / f"cdpk_results_{kind}_{config}.csv" for kind in ("accuracy", "bad_format", "full")}
    missing = [f.name for f in files.values() if not f.exists()]
    if missing:
        raise ResultsError(f"{folder.name}: result set {config!r} is incomplete, missing {missing}")
    return files


def models_in_full(full_df):
    """Model names from the pred_<model> columns of a full file."""
    return [col.removeprefix("pred_") for col in full_df.columns if col.startswith("pred_")]


# ==================== Checks ====================

def load_result_set(folder, config):
    """Load and check one result set. Returns (accuracy, bad_format, full) DataFrames.

    accuracy and bad_format keep only the category columns: the accuracy file also has model
    metadata columns (provider, size, ...) that must not be read as categories.
    """
    files = result_files(folder, config)
    label = f"{folder.name}/{config}"
    acc = pd.read_csv(files["accuracy"], index_col=0)
    bf = pd.read_csv(files["bad_format"], index_col=0)
    full = pd.read_csv(files["full"])

    # R2: same models and categories in accuracy and bad_format
    missing_cats = [c for c in ALL_CATEGORIES if c not in acc.columns or c not in bf.columns]
    if missing_cats:
        raise ResultsError(f"{label}: categories {missing_cats} missing from the accuracy or bad_format file")
    acc, bf = acc[ALL_CATEGORIES], bf[ALL_CATEGORIES]
    if set(acc.index) != set(bf.index):
        raise ResultsError(f"{label}: accuracy and bad_format files list different models: "
                           f"{sorted(set(acc.index) ^ set(bf.index))}")

    # R3: per-question columns needed to rebuild the tables
    for col in ("category", "correct_answer"):
        if col not in full.columns:
            raise ResultsError(f"{label}: full file has no {col!r} column")
    no_pred = sorted(set(acc.index) - set(models_in_full(full)))
    if no_pred:
        raise ResultsError(f"{label}: no pred_ column in the full file for models {no_pred}")

    # R4: the summary files agree with the per-question predictions
    for model in acc.index:
        preds = full[f"pred_{model}"]
        correct = (preds == full["correct_answer"]).groupby(full["category"]).mean() * 100
        bad = preds.isna().groupby(full["category"]).mean() * 100
        correct["Overall"] = (preds == full["correct_answer"]).mean() * 100
        bad["Overall"] = preds.isna().mean() * 100
        for name, recomputed, summary in (("accuracy", correct, acc), ("bad_format", bad, bf)):
            diff = (recomputed.reindex(ALL_CATEGORIES) - summary.loc[model]).abs().max()
            if not diff < 1e-6:
                raise ResultsError(f"{label}: {name} of {model} recomputed from the full file differs from "
                                   f"the {name} file (max difference {diff:.4f} points)")
    return acc, bf, full


def question_counts(full_df):
    """Number of questions per category, used to compare question sets between folders."""
    return full_df.groupby("category").size().to_dict()


# ==================== Tables ====================

def latency_summary(full_df, models):
    """Mean and median latency per model and category (and Overall)."""
    rows = []
    for model in models:
        lat_col = f"Latency_{model}"
        if lat_col not in full_df.columns:
            continue
        for cat, group in [*full_df.groupby("category"), ("Overall", full_df)]:
            values = group[lat_col].dropna()
            rows.append({
                "model": model,
                "category": cat,
                "Latency Mean": values.mean() if len(values) else np.nan,
                "Latency Median": values.median() if len(values) else np.nan,
            })
    return pd.DataFrame(rows, columns=["model", "category", "Latency Mean", "Latency Median"])


def clean_list(values):
    """Replace NaN/NaT with None, so the list is written as a Python literal."""
    return [None if pd.isna(x) else x for x in values]


def detailed_rows(full_df, models, language, english_prompt):
    """One row per (model, category, Overall) with per-question lists."""
    rows = []
    for model in models:
        for cat, group in [*full_df.groupby("category"), ("Overall", full_df)]:
            preds = group[f"pred_{model}"]
            row = {
                "category": cat,
                "model": model,
                "language": language,
                "english_prompt": english_prompt,
                "correct": clean_list((preds == group["correct_answer"]).tolist()),
                "bad_format": clean_list(preds.isna().tolist()),
            }
            for prefix in PER_QUESTION_COLUMNS:
                col = f"{prefix}_{model}"
                row[prefix] = clean_list(group[col].tolist() if col in group.columns else [np.nan] * len(group))
            rows.append(row)
    return rows


def build_tables(folders, models_list, models_to_exclude):
    """Build the summary and detailed tables from the results of `models_list` in every folder.

    Returns (summary_df, detailed_df, report) where report has one row per folder.
    """
    reference = next((f for f in folders if f["folder"].name == REFERENCE_FOLDER), None)
    if reference is None or models_list not in reference["configs"]:
        raise ResultsError(f"No results for {models_list} in data/results/{REFERENCE_FOLDER}/: "
                           "needed as the reference question set")
    reference_counts = question_counts(load_result_set(reference["folder"], models_list)[2])

    summary_parts, detailed, report = [], [], []
    for info in folders:
        folder, language, english_prompt = info["folder"], info["language"], info["english_prompt"]
        if models_list not in info["configs"]:
            others = f" (has {', '.join(info['configs'])})" if info["configs"] else ""
            report.append({"Folder": folder.name, "Status": f"skipped: no results for this models list{others}"})
            continue

        acc, bf, full = load_result_set(folder, models_list)

        # Same questions as the English run, otherwise scores are not comparable
        counts = question_counts(full)
        if counts != reference_counts:
            report.append({"Folder": folder.name, "Status": f"skipped: {sum(counts.values())} questions, "
                                                             f"English has {sum(reference_counts.values())}"})
            continue

        models = [m for m in acc.index if m not in models_to_exclude]
        acc_long = acc.loc[models].rename_axis("model").reset_index().melt(
            id_vars="model", var_name="category", value_name="accuracy")
        bf_long = bf.loc[models].rename_axis("model").reset_index().melt(
            id_vars="model", var_name="category", value_name="bad_format")
        part = acc_long.merge(bf_long, on=["model", "category"]).merge(
            latency_summary(full, models), on=["model", "category"], how="left")
        part["language"], part["english_prompt"] = language, english_prompt
        summary_parts.append(part)
        detailed.extend(detailed_rows(full, models, language, english_prompt))
        report.append({"Folder": folder.name, "Status": "used", "Models": len(models),
                       "Questions": sum(counts.values())})

    summary_df = pd.concat(summary_parts, ignore_index=True)
    detailed_df = pd.DataFrame(detailed)

    # English questions are already in English: also count them as English-prompt results
    def with_english_prompt_copy(df):
        english = df[(df["language"] == REFERENCE_FOLDER) & ~df["english_prompt"]].copy()
        english["english_prompt"] = True
        return pd.concat([df, english], ignore_index=True)
    summary_df, detailed_df = with_english_prompt_copy(summary_df), with_english_prompt_copy(detailed_df)

    # Provider from models.csv
    providers = pd.read_csv(MODELS_CSV).set_index("model_id")["provider"]
    unknown = sorted(set(summary_df["model"]) - set(providers.index))
    if unknown:
        warnings.warn(f"{len(unknown)} models not in {MODELS_CSV.name}, provider set to 'Unknown': {unknown}")
    for df in (summary_df, detailed_df):
        df["provider"] = df["model"].map(providers).fillna("Unknown")

    return summary_df, detailed_df, pd.DataFrame(report)


def check_final_table(summary_df):
    """R5: one row per key, all categories per model, values in range."""
    keys = ["model", "language", "english_prompt", "category"]
    duplicated = summary_df[summary_df.duplicated(keys, keep=False)]
    if not duplicated.empty:
        raise ResultsError(f"{len(duplicated)} duplicated rows for {keys}, e.g.\n{duplicated[keys].head()}")
    n_cats = summary_df.groupby(["model", "language", "english_prompt"])["category"].nunique()
    incomplete = n_cats[n_cats != len(ALL_CATEGORIES)]
    if not incomplete.empty:
        raise ResultsError(f"{len(incomplete)} (model, language, prompt) without all {len(ALL_CATEGORIES)} "
                           f"categories, e.g. {incomplete.index[:5].tolist()}")
    for col in ("accuracy", "bad_format"):
        out_of_range = summary_df[~summary_df[col].between(0, 100)]
        if not out_of_range.empty:
            raise ResultsError(f"{len(out_of_range)} rows with {col} outside 0-100")


def warn_high_bad_format(summary_df, threshold=BAD_FORMAT_WARNING_THRESHOLD):
    """Print the (model, language) pairs above `threshold`% bad format overall, with the
    categories above it. Nothing is removed: deciding is left to the user."""
    rows = summary_df[~((summary_df["language"] == REFERENCE_FOLDER) & summary_df["english_prompt"])]  # English once
    overall = rows[(rows["category"] == "Overall") & (rows["bad_format"] > threshold)]
    if overall.empty:
        print(f"\nBad format: no model above {threshold}% overall.")
        return
    keys = ["model", "language", "english_prompt"]
    by_category = rows[(rows["category"] != "Overall") & (rows["bad_format"] > threshold)]
    worst = by_category.groupby(keys)["category"].apply(lambda cats: ", ".join(sorted(cats)))
    table = overall.set_index(keys)[["bad_format"]].round(1).join(worst.rename(f"categories > {threshold}%"))
    print(f"\nWARNING: {len(table)} (model, language) above {threshold}% bad format overall "
          "(kept in the tables; unreadable answers count as wrong):\n")
    print(table.sort_values("bad_format", ascending=False).reset_index().to_markdown(index=False))
    print("\nTo decide per case (scripts/README.md, steps 5 and 6): check failed API calls with "
          "`uv run python scripts/count_failures.py CDPK_<Slug>`, then rerun the affected categories, "
          "or add the model to MODELS_TO_EXCLUDE.")


# ==================== Writing ====================

def write_csv(df, path):
    """Write through a temporary file, so an interrupted run never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the multilingual results tables from data/results/. "
                    "See the module docstring for inputs, outputs and checks.")
    parser.add_argument("--models-list-yaml", default=MODELS_LIST_YAML,
                        help=f"Models list whose results are read: name of a file in configs/models/, "
                             f"with or without .yaml (default: {MODELS_LIST_YAML}).")
    parser.add_argument("--overwrite", choices=["true", "false"], default="true", type=str.lower,
                        help="true (default): replace existing output files. "
                             "false: stop if an output file already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    models_list = args.models_list_yaml.removesuffix(".yaml")
    if not (MODELS_CONFIGS_DIR / f"{models_list}.yaml").exists():
        sys.exit(f"ERROR: --models-list-yaml {models_list!r} does not exist in configs/models/. "
                 f"Available: {', '.join(sorted(p.stem for p in MODELS_CONFIGS_DIR.glob('*.yaml')))}")
    existing = [p for p in (SUMMARY_FILE, DETAILED_FILE) if p.exists()]
    if existing and args.overwrite == "false":
        sys.exit("ERROR: output files already exist, nothing was computed:\n  "
                 + "\n  ".join(str(p) for p in existing)
                 + "\nRun without --overwrite false (or with --overwrite true) to replace them.")

    try:
        summary_df, detailed_df, report = build_tables(discover_result_sets(RESULTS_DIR), models_list,
                                                       MODELS_TO_EXCLUDE)
        check_final_table(summary_df)
    except ResultsError as e:
        sys.exit(f"ERROR: {e}")

    # Folder table: _ep folders first, then the others, each in alphabetical order
    is_ep = report["Folder"].str.endswith("_ep")
    report.insert(1, "Prompt", np.where(is_ep, "English (ep)", "in the language"))
    report.loc[report["Folder"] == REFERENCE_FOLDER, "Prompt"] = "English (counted as both)"
    report = report.assign(_ep=~is_ep, _name=report["Folder"].str.lower()).sort_values(["_ep", "_name"])
    report = report.drop(columns=["_ep", "_name"])
    counts = ["Models", "Questions"]  # empty for skipped folders, shown as whole numbers
    report[counts] = report[counts].astype("Int64").astype(str).replace("<NA>", "")
    print(f"\nFolders (models list: {models_list})\n")
    print(report.to_markdown(index=False))
    print(f"\nExcluded models: {MODELS_TO_EXCLUDE}")
    warn_high_bad_format(summary_df)

    write_csv(summary_df, SUMMARY_FILE)
    write_csv(detailed_df, DETAILED_FILE)
    print(f"\nWritten ({'replaced' if existing else 'new'}):")
    for path, df in ((SUMMARY_FILE, summary_df), (DETAILED_FILE, detailed_df)):
        print(f"  {path.relative_to(ROOT)}  ({len(df)} rows x {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
