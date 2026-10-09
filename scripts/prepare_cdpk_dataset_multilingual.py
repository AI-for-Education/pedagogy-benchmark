"""
Prepare a translated benchmark dataset (CDPK or SEND) for the multilingual runner.

Takes one full translated dataset and writes the files that
`scripts/run_pedagogy_benchmark_multilingual.py` reads:
  - one test CSV and one dev (few-shot) CSV per category,
  - one question YAML config per category.

CLI only:
    uv run python scripts/prepare_cdpk_dataset_multilingual.py `
        --dataset cdpk `
        --file pedagogy_benchmark_french_cdpk_cleaned `
        --language french_ep

Run it from the repository root or anywhere else: all paths are resolved from
the repository root.

Required before running
-----------------------
1. A translated dataset in `data/pedagogy_benchmark_full_datasets/`, usually made
   with `scripts/translate_benchmark.py`. Use its `_cleaned.csv` file: it is written once the
   translation is verified complete, whether or not cells had to be retranslated.
   A `_retranslated.csv` file still has missing cells and is rejected. A raw file
   without a `_cleaned` version is accepted with a warning (it is still checked for
   missing cells); `--mode retranslate` on it creates the `_cleaned` file.
   The file must keep the original rows and `question_id` column of the English
   dataset (`pedagogy_benchmark_cdpk.csv` or `pedagogy_benchmark_send.csv`).
2. The language registered in `LANGUAGE_PROMPTS` in `src/cdpk/language_prompts.py`.
   Its `slug` names the output folder and files, e.g. `French_ep`.
3. The English reference split, used to check that every language has the same
   questions, few-shot examples and answer key as English:
     - CDPK: `data/English/CDPK_per_category/{test,dev}/`, already in the repo.
     - SEND: `data/English/SEND/{test,dev}/`, not created yet. Run this script once
       with `--language english --file pedagogy_benchmark_send` first. Until then
       SEND fails on purpose for other languages.
   KNOWN ISSUE: SEND cannot be prepared yet, because few_shot_examples_idx_dict.json
   lists 5 SEND few-shot examples instead of 3 (see the comment on DATASETS["send"]).
4. No previous output for this dataset and language: the script never overwrites
   files. Move or delete the old CSV and YAML files first to regenerate them.

Inputs
------
--dataset {cdpk,send}   Which benchmark the file contains (one per run).
--file NAME             File name in data/pedagogy_benchmark_full_datasets/, with or
                        without ".csv", e.g. pedagogy_benchmark_french_cdpk_cleaned.
--language KEY          Key from LANGUAGE_PROMPTS. Two variants exist for most languages:
                          - "french_ep": English Prompt. Instructions to the model are in
                            English; few-shot examples and questions are in French.
                          - "french": instructions are also in French. Needs the intro,
                            instruction and final prompts translated in
                            language_prompts.py (some languages only have the _ep variant).
                        Both read the same translated file but write to different folders
                        (French_ep/ vs French/), so run this script once per variant you
                        want to benchmark.
--use-question-id {true,false}
                        Default false: full files, rows in the original order. Use true for
                        human-reviewed files (`*_reviewed.csv`) where rows were removed, so
                        rows are no longer at their original positions: few-shot examples
                        and the English answer key are then looked up by question_id
                        instead of row position.
--dry-run               Run every check without writing any file.

What it does
------------
1. Pre-flight: input file exists, file name matches --dataset and --language, the
   few-shot list has 3 examples per category, and no output file exists yet.
2. Load checks on the full file (table "Dataset check"): required columns, question
   ids, categories, no missing text, answer key identical to English.
3. Formats columns: adds "Source" and empty answer_e/f/g if missing.
4. Splits each category into dev (the 3 few-shot examples listed in
   data/few_shot_examples_idx_dict.json, identical for every language) and test (the
   rest), checks each split (table per category), then writes the CSVs.
5. Writes one YAML config per category in configs/questions/.

Every check raises an error and nothing more is written as soon as one fails.
"""
import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FULL_DATASETS_DIR = DATA_DIR / "pedagogy_benchmark_full_datasets"
CONFIGS_DIR = ROOT / "configs" / "questions"
FEW_SHOT_FILE = DATA_DIR / "few_shot_examples_idx_dict.json"

# Add parent directory to path to import cdpk module
sys.path.insert(0, str(ROOT / "src"))
from cdpk.language_prompts import get_language_config, list_available_languages

REFERENCE_LANGUAGE = "english"
N_FEW_SHOT = 3  # few-shot examples per category, identical for every language

# What differs between the two benchmarks.
# id_offset: question_id of the first row. The few-shot indices in
# few_shot_examples_idx_dict.json are row positions, so question_id = id_offset + position.
DATASETS = {
    "cdpk": {
        "categories": ["Science", "Literacy", "Creative arts", "Maths", "Social studies", "Technology", "General"],
        "output_folder": "CDPK_per_category",
        "english_file": "pedagogy_benchmark_cdpk",
        "source_label": "Pedagogy Benchmark",
        "id_offset": 0,
    },
    # KNOWN ISSUE (SEND): the SEND dataset has 2 duplicated questions, question_id 1047 (row 127)
    # = few-shot example 920 (row 0) and 1056 (row 136) = 922 (row 2). The duplicates were caught
    # when the few-shot list was created, so few_shot_examples_idx_dict.json lists 5 few-shot
    # examples for SEND ("CDPK_send": [0, 127, 1, 2, 136]) instead of 3, and SEND is refused by
    # load_few_shot_positions for now. Fix: remove 1047 and 1056 from the SEND datasets, set
    # "CDPK_send" to [0, 1, 2], then prepare English SEND as the reference.
    "send": {
        "categories": ["SEND"],
        "output_folder": "SEND",
        "english_file": "pedagogy_benchmark_send",
        "source_label": "SEND Benchmark",
        "id_offset": 920,
    },
}

# Keys of few_shot_examples_idx_dict.json for each category
FEW_SHOT_KEYS = {
    "Science": "CDPK_science",
    "Literacy": "CDPK_literacy",
    "Creative arts": "CDPK_creative_arts",
    "Maths": "CDPK_maths",
    "Social studies": "CDPK_social_studies",
    "Technology": "CDPK_technology",
    "General": "CDPK_gen_pk",
    "SEND": "CDPK_send",
}

REQUIRED_TEXT_COLS = ["question", "answer_a", "answer_b", "answer_c", "answer_d"]
REQUIRED_COLS = ["question_id", *REQUIRED_TEXT_COLS, "correct_answer", "category"]

# Column positions written to the question YAML configs; also used to check that
# the saved CSVs follow this layout.
YAML_TEMPLATE = {
    "test_file": "",
    "test_header": 0,
    "example_file": "",
    "example_header": 0,
    "choices": ["A", "B", "C", "D", "E", "F", "G"],
    "choice_cols": [2, 3, 4, 5, 6, 7, 8],
    "answer_col": 9,
    "question_col": 1,
    "example_rows": "",
}


# ==================== Paths ====================

def category_slug(category):
    return category.replace(' PCK', '').replace(' ', '_').lower()


def split_file_paths(slug, dataset, category):
    """Test and dev CSV paths for one category, e.g. data/French_ep/CDPK_per_category/test/..."""
    folder = DATA_DIR / slug / DATASETS[dataset]["output_folder"]
    name = f"CDPK_{slug}_{category_slug(category)}"
    return folder / "test" / f"{name}_test.csv", folder / "dev" / f"{name}_dev.csv"


def yaml_config_path(slug, category):
    """Question config read by the runner, e.g. configs/questions/CDPK_French_ep_science.yaml."""
    return CONFIGS_DIR / f"CDPK_{slug}_{category_slug(category)}.yaml"


# ==================== Pre-flight checks ====================

def abort_if_files_exist(paths, what):
    """Abort before writing anything if any of the output files already exists."""
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        warnings.warn(
            f"{len(existing)} {what} already exist, nothing was written to avoid overwriting them:\n  "
            + "\n  ".join(existing)
            + "\nMove or delete them first if you want to regenerate them."
        )
        raise FileExistsError(f"{len(existing)} {what} already exist")


def check_file_matches_inputs(stem, dataset, language):
    """Catch a file passed with the wrong --dataset or --language.

    Translated files are named pedagogy_benchmark_{language}_{dataset}, plus a suffix
    _cleaned (verified by translate_benchmark.py), _retranslated (not verified) or
    _reviewed (human-reviewed subset); the English source files are pedagogy_benchmark_{dataset}.
    """
    if language == REFERENCE_LANGUAGE:
        if stem != DATASETS[dataset]["english_file"]:
            raise ValueError(f"For --language {language}, --file must be "
                             f"{DATASETS[dataset]['english_file']!r}, got {stem!r}")
        return
    # Exact name, not a substring search: "_swahili_" is also found in the Tanzanian
    # Swahili file pedagogy_benchmark_swahili_tz_cdpk
    base_language = language.removesuffix("_ep")
    pattern = rf"pedagogy_benchmark_{re.escape(base_language)}_{dataset}(_cleaned|_retranslated|_reviewed)?"
    if not re.fullmatch(pattern, stem):
        raise ValueError(f"--file {stem!r} does not match --dataset {dataset} and --language {language}: "
                         f"expected pedagogy_benchmark_{base_language}_{dataset}, optionally followed by "
                         "_cleaned, _retranslated or _reviewed")


def load_few_shot_positions(dataset):
    """Few-shot row positions per category, from few_shot_examples_idx_dict.json.

    Every language uses the same examples, so scores stay comparable. Each category
    must list exactly N_FEW_SHOT examples.
    """
    with open(FEW_SHOT_FILE) as f:
        few_shot = json.load(f)
    positions = {}
    for category in DATASETS[dataset]["categories"]:
        key = FEW_SHOT_KEYS[category]
        if key not in few_shot:
            raise KeyError(f"No few-shot examples for '{category}' ({key!r}) in {FEW_SHOT_FILE.name}")
        if len(few_shot[key]) != N_FEW_SHOT:
            warnings.warn(f"{FEW_SHOT_FILE.name} lists {len(few_shot[key])} few-shot examples for "
                          f"'{category}' ({key}: {few_shot[key]}), expected {N_FEW_SHOT}.")
            raise ValueError(f"Wrong number of few-shot examples for '{category}'")
        positions[category] = few_shot[key]
    return positions


# ==================== Load checks (full dataset) ====================

def check_no_missing_text(df, label):
    """Check that the question and options A-D are filled in every row.

    translate_benchmark.py only writes a `_cleaned.csv` file once every cell is
    translated (same definition of missing: None/NaN or whitespace-only), so a
    `_cleaned` file should pass. We still check because the input may not come from
    translate_benchmark.py (human-reviewed files, files shared by others), and
    `_cleaned` files written before it verified completeness could still have gaps.
    All questions have exactly options A-D, so E-G are not checked.
    Rows are reported by question_id, the id expected by the retranslate step.
    """
    text = df[REQUIRED_TEXT_COLS]
    missing = text.isna() | text.astype(str).apply(lambda col: col.str.strip() == "")
    if missing.any().any():
        rows = missing.any(axis=1)
        details = {col: int(n) for col, n in missing.sum().items() if n}
        warnings.warn(
            f"{label} has {int(rows.sum())} rows with missing text "
            f"(per column: {details}); question_id: {df.loc[rows, 'question_id'].tolist()[:20]}. "
            "Run `uv run python scripts/translate_benchmark.py --mode retranslate "
            f"--language <language without _ep> --file {label}` to fill them, then rerun this "
            "script on the _cleaned file it writes."
        )
        raise ValueError(f"Missing translations in {label}")


def check_answers_match_english(df, dataset, label):
    """Check that each question has the same correct answer as in English, by question_id.

    Works for full and reviewed (subset) files. A shifted, duplicated or wrong row
    would score models against another question's answer.
    """
    english = pd.read_csv(FULL_DATASETS_DIR / f"{DATASETS[dataset]['english_file']}.csv")
    english_answers = english.set_index("question_id")["correct_answer"]
    unknown = sorted(set(df["question_id"]) - set(english_answers.index))
    if unknown:
        warnings.warn(f"{label} has question_id not in the English dataset: {unknown[:20]}")
        raise ValueError(f"Unknown question_id in {label}")
    expected = english_answers.loc[df["question_id"]].to_numpy()
    differ = df.loc[df["correct_answer"].to_numpy() != expected, "question_id"].tolist()
    if differ:
        warnings.warn(f"{label}: {len(differ)} correct answers differ from English, question_id: {differ[:20]}")
        raise ValueError(f"Answer key mismatch with English in {label}")


def validate_dataset(df, dataset, language, use_question_id, label):
    """Check the full translated dataset right after loading.

    Runs before the split per category, so problems are reported against the input
    file, with the original question ids. Raises if a check fails, otherwise prints
    a summary table.
    """
    results = []
    categories = DATASETS[dataset]["categories"]
    id_offset = DATASETS[dataset]["id_offset"]

    # Check L1: the columns used by the rest of the script exist.
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        warnings.warn(f"{label} is missing columns {missing_cols}. Columns: {list(df.columns)}")
        raise ValueError(f"Missing columns in {label}")
    results.append(("Required columns", "pass", f"{len(REQUIRED_COLS)} columns found"))

    # Check L2: question ids are usable to find the few-shot examples.
    # With --use-question-id false the few-shot indices are row positions, so rows must be
    # in the original order. With it, rows may be missing and are looked up by
    # question_id, which must be unique.
    ids = df["question_id"]
    if use_question_id:
        duplicated = ids[ids.duplicated()].tolist()
        if duplicated:
            warnings.warn(f"{label} has duplicated question_id values: {duplicated[:20]}")
            raise ValueError(f"Duplicated question_id in {label}")
        results.append(("Question ids", "pass", f"{len(ids)} unique ids"))
    else:
        expected_ids = pd.Series(range(id_offset, id_offset + len(df)), index=df.index)
        if not ids.equals(expected_ids.astype(ids.dtype)):
            first = int((ids != expected_ids).idxmax())
            warnings.warn(
                f"{label}: question_id must be {id_offset}..{id_offset + len(df) - 1} in order, because "
                f"the few-shot indices are row positions; first mismatch at row {first} "
                f"(question_id {ids.iloc[first]}). Rows were reordered, removed or added. "
                "If rows were removed on purpose (reviewed file), use --use-question-id true."
            )
            raise ValueError(f"question_id not in original order in {label}")
        results.append(("Question ids", "pass", f"{id_offset}..{id_offset + len(df) - 1} in order"))

    # Check L3: the category names are exactly the expected ones.
    # A misspelled or extra category would otherwise only show up as an empty test set.
    found = set(df["category"].dropna().unique())
    unexpected, absent = sorted(found - set(categories)), sorted(set(categories) - found)
    if unexpected or absent:
        warnings.warn(f"{label} categories differ from expected {dataset.upper()} categories: "
                      f"unexpected {unexpected}, missing {absent}")
        raise ValueError(f"Unexpected categories in {label}")
    results.append(("Categories", "pass", f"{len(found)} expected categories"))

    # Check L4: no missing translations.
    # A cell left empty by a failed translation call would show models a question
    # with blank options and count their answer as wrong.
    check_no_missing_text(df, label)
    results.append(("No missing text", "pass", f"{', '.join(REQUIRED_TEXT_COLS)} filled"))

    # Check L5: every question has the same correct answer as in English.
    if language == REFERENCE_LANGUAGE:
        results.append(("Answer key matches English", "skipped", "reference language"))
    else:
        check_answers_match_english(df, dataset, label)
        results.append(("Answer key matches English", "pass", f"{len(df)} answers, by question_id"))

    print(f"\n{label}: {len(df)} questions loaded\n")
    print(pd.DataFrame(results, columns=["Dataset check", "Status", "Detail"]).to_markdown(index=False))


# ==================== Formatting ====================

def add_missing_answer_columns(df, optional_cols=("answer_e", "answer_f", "answer_g")):
    """Add empty optional answer columns right after the previous answer column.

    The question YAML configs read options A-G by position, so these columns must
    exist even though the questions only use options A-D. A translated file may
    only contain the translated columns, i.e. without answer_e/f/g.
    Options A-D are required and checked by validate_dataset, so they are never added.
    """
    added = []
    for col in optional_cols:
        if col not in df.columns:
            previous_col = f"answer_{chr(ord(col[-1]) - 1)}"  # answer_e -> answer_d
            df.insert(df.columns.get_loc(previous_col) + 1, col, None)
            added.append(col)
    print(f"\nAdded empty answer columns: {added}" if added else "\nAnswer columns A-G already present")
    return df


# ==================== Split checks (per category) ====================

def check_answers_match_reference(df_split, dataset, category, split):
    """Check that a dev or test split matches the English one.

    Compares the number of questions and the correct answers (same values, same
    order) with the English split file, so the saved files line up with English.
    """
    english_slug = get_language_config(REFERENCE_LANGUAGE)["slug"]
    test_file, dev_file = split_file_paths(english_slug, dataset, category)
    reference_file = dev_file if split == "dev" else test_file
    if not reference_file.exists():
        warnings.warn(
            f"English {split} file not found: {reference_file}. Prepare English first: "
            f"--dataset {dataset} --language {REFERENCE_LANGUAGE} --file {DATASETS[dataset]['english_file']}"
        )
        raise FileNotFoundError(reference_file)

    reference_answers = pd.read_csv(reference_file)["correct_answer"].tolist()
    new_answers = df_split["correct_answer"].tolist()
    if len(new_answers) != len(reference_answers):
        warnings.warn(
            f"{split} set for '{category}' has {len(new_answers)} questions, "
            f"English has {len(reference_answers)} ({reference_file.name})"
        )
        raise ValueError(f"{split} size mismatch with English for category '{category}'")
    mismatches = [i for i, (new, ref) in enumerate(zip(new_answers, reference_answers)) if new != ref]
    if mismatches:
        details = ", ".join(f"row {i}: {new_answers[i]} vs {reference_answers[i]}" for i in mismatches[:10])
        warnings.warn(
            f"{split} set for '{category}' does not match English: "
            f"{len(mismatches)} correct answers differ (first ones: {details}) in {reference_file.name}"
        )
        raise ValueError(f"{split} answers mismatch with English for category '{category}'")


def check_column_layout(df_split, category, split, column_layout=YAML_TEMPLATE):
    """Check that the columns sit where the question YAML config expects them.

    The runner reads question, options and answer by position, so a shifted
    column would silently score models against the wrong answer key.
    """
    columns = list(df_split.columns)
    expected = {column_layout["question_col"]: "question",
                column_layout["answer_col"]: "correct_answer"}
    expected.update({col: f"answer_{choice.lower()}"
                     for col, choice in zip(column_layout["choice_cols"], column_layout["choices"])})
    wrong = {pos: (columns[pos] if pos < len(columns) else None, name)
             for pos, name in expected.items()
             if pos >= len(columns) or columns[pos] != name}
    if wrong:
        # Report only the first misplaced column: an extra or missing column
        # shifts every column after it, which would otherwise flood the message.
        pos, (found, name) = min(wrong.items())
        warnings.warn(
            f"{split} set for '{category}' has an unexpected column layout: "
            f"position {pos} is {found!r}, expected {name!r} "
            f"({len(wrong)} positions wrong in total). Columns: {columns}"
        )
        raise ValueError(f"Column layout mismatch in {split} set for category '{category}'")


def split_category(df, category, few_shot_positions, id_offset, use_question_id):
    """Return (dev, test) rows of one category, without the question_id column.

    dev: the few-shot examples, in the order of few_shot_examples_idx_dict.json.
    test: the other questions of the category.
    """
    if use_question_id:
        few_shot_ids = [id_offset + pos for pos in few_shot_positions]
        by_id = df.set_index("question_id", drop=False)
        df_few_shot = by_id.loc[[i for i in few_shot_ids if i in by_id.index]]
        df_rest = df[~df["question_id"].isin(few_shot_ids)]
    else:
        df_few_shot = df.loc[few_shot_positions]
        df_rest = df.drop(index=few_shot_positions)
    df_test = df_rest[df_rest["category"] == category]
    return (df_few_shot.drop(columns=["question_id"]).reset_index(drop=True),
            df_test.drop(columns=["question_id"]).reset_index(drop=True))


def create_split_csvs(df, dataset, language, slug, few_shot_positions, use_question_id, dry_run):
    """Split each category into dev and test, check both, then save them.

    Returns {category: number of few-shot examples}, used for the YAML configs.
    """
    id_offset = DATASETS[dataset]["id_offset"]
    summary_rows, n_few_shot = [], {}

    for category in DATASETS[dataset]["categories"]:
        df_few_shot, df_test = split_category(df, category, few_shot_positions[category],
                                              id_offset, use_question_id)

        # ---- Sanity checks before saving the category files ----
        # These check the split and the output files. The input file itself is
        # checked by validate_dataset right after loading.
        checks = {}
        # Check 1: the few-shot rows are the N_FEW_SHOT examples of this category.
        # Few-shot indices are row positions (or question_ids with --use-question-id true).
        # A reordered file could make them land on another category's questions, and
        # a reviewed file may have removed a few-shot question.
        wrong_category = df_few_shot["category"].ne(category).any()
        if len(df_few_shot) != N_FEW_SHOT or wrong_category:
            warnings.warn(
                f"Few-shot examples for '{category}': expected {N_FEW_SHOT} rows of this category, "
                f"got {len(df_few_shot)} with categories {df_few_shot['category'].value_counts().to_dict()}"
            )
            raise ValueError(f"Wrong few-shot examples for category '{category}'")
        checks["Few-shot examples"] = "pass"
        # Check 2: the test set is not empty.
        # Misspelled categories are already caught by validate_dataset; this guards
        # against a category that only has few-shot examples.
        if df_test.empty:
            warnings.warn(f"No test questions found for category '{category}'")
            raise ValueError(f"Empty test set for category '{category}'")
        checks["Test not empty"] = "pass"
        # Check 3: dev and test files match the English ones (size and answer key, in order).
        # Load check L5 compares answers question by question; this one checks the
        # saved files line up with English. Reviewed files have fewer questions by
        # design, so only L5 applies to them.
        if language == REFERENCE_LANGUAGE:
            checks["Matches English"] = "skipped"
        elif use_question_id:
            checks["Matches English"] = "skipped (reviewed)"
        else:
            check_answers_match_reference(df_few_shot, dataset, category, "dev")
            check_answers_match_reference(df_test, dataset, category, "test")
            checks["Matches English"] = "pass"
        # Check 4: columns are at the positions written in the question YAML config.
        # The runner reads columns by position; an extra or reordered column would
        # make it read the wrong column as the answer key without any error.
        check_column_layout(df_few_shot, category, "dev")
        check_column_layout(df_test, category, "test")
        checks["Column layout"] = "pass"

        if not dry_run:
            test_file, dev_file = split_file_paths(slug, dataset, category)
            test_file.parent.mkdir(parents=True, exist_ok=True)
            dev_file.parent.mkdir(parents=True, exist_ok=True)
            df_test.to_csv(test_file, index=False)
            df_few_shot.to_csv(dev_file, index=False)
        n_few_shot[category] = len(df_few_shot)
        summary_rows.append({"Category": category, "Dev": len(df_few_shot), "Test": len(df_test),
                             **checks, "Saved": "no (dry run)" if dry_run else "yes"})

    summary = pd.DataFrame(summary_rows)
    folder = DATA_DIR / slug / DATASETS[dataset]["output_folder"]
    print(f"\n{slug} {dataset.upper()}: test and dev files "
          f"{'checked (dry run, not saved)' if dry_run else f'saved to {folder}'}\n")
    print(summary.to_markdown(index=False))
    print(f"\nTotal: {summary['Dev'].sum()} dev, {summary['Test'].sum()} test questions")
    return n_few_shot


# ==================== YAML configs ====================

def write_custom_yaml(data, filepath):
    # Manually construct the YAML output as a string to ensure exact formatting
    choices_str = "[" + ", ".join(f'{choice}' for choice in data["choices"]) + "]"  # without "" around letters
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

    with open(filepath, 'w') as file:
        file.write(yaml_content)


def write_yaml_configs(dataset, slug, n_few_shot):
    """Write one question config per category, pointing at the saved test and dev CSVs.

    CSV paths are written relative to data/, as expected by the runner.
    """
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    print()
    for category in DATASETS[dataset]["categories"]:
        test_file, dev_file = split_file_paths(slug, dataset, category)
        yaml_content = dict(YAML_TEMPLATE)
        yaml_content["test_file"] = test_file.relative_to(DATA_DIR).as_posix()
        yaml_content["example_file"] = dev_file.relative_to(DATA_DIR).as_posix()
        yaml_content["example_rows"] = list(range(n_few_shot[category]))
        output_file = yaml_config_path(slug, category)
        write_custom_yaml(yaml_content, output_file)
        print(f"Generated: {output_file.relative_to(ROOT)}")


# ==================== Main ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare test/dev CSVs and question YAML configs for one translated dataset. "
                    "See the module docstring for the required setup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  uv run python scripts/prepare_cdpk_dataset_multilingual.py "
               "--dataset cdpk --file pedagogy_benchmark_french_cdpk_cleaned --language french_ep",
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                        help="Benchmark contained in the file: cdpk or send (one per run).")
    parser.add_argument("--file", required=True,
                        help="File name in data/pedagogy_benchmark_full_datasets/, with or without .csv, "
                             "e.g. pedagogy_benchmark_french_cdpk_cleaned.")
    parser.add_argument("--language", required=True,
                        help="Key in LANGUAGE_PROMPTS (src/cdpk/language_prompts.py). '<lang>_ep' = "
                             "instructions in English, questions in <lang>; '<lang>' = instructions "
                             f"also in <lang>. Each writes to its own folder. "
                             f"Available: {', '.join(list_available_languages())}.")
    parser.add_argument("--use-question-id", choices=["true", "false"], default="false",
                        type=str.lower,  # also accept True/FALSE/...
                        help="false (default): full files, rows in the original order. "
                             "true: human-reviewed files where rows were removed; few-shot examples "
                             "and English answers are found by question_id instead of row position.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run every check without writing any file.")
    args = parser.parse_args()

    # Checked here rather than with argparse choices, to explain why a language is refused
    available = list_available_languages()
    if args.language not in available:
        # sys.exit instead of parser.error, which would print the usage line first
        sys.exit(f"ERROR: '{args.language}' is not a valid language: it does not exist in the LANGUAGE_PROMPTS "
                 f"dictionary of src/cdpk/language_prompts.py. Add it to LANGUAGE_PROMPTS first.\n"
                 f"Available languages: {', '.join(available)}")
    return args


def main():
    args = parse_args()
    dataset, language = args.dataset, args.language
    # Convert the text "true"/"false" to True/False: in Python, any non-empty text
    # (even "false") counts as True in an if, so the text can't be used directly
    if args.use_question_id == "true":
        use_question_id = True
    elif args.use_question_id == "false":
        use_question_id = False
    else:  # cannot happen today: argparse only accepts the choices of --use-question-id
        raise ValueError(f"--use-question-id must be true or false, got {args.use_question_id!r}")
    stem = args.file.removesuffix(".csv")
    input_file = FULL_DATASETS_DIR / f"{stem}.csv"
    language_config = get_language_config(language)
    slug = language_config["slug"]
    categories = DATASETS[dataset]["categories"]

    # ---- Pre-flight: nothing is read or written if these fail ----
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    check_file_matches_inputs(stem, dataset, language)
    # Hint when the file is not the verified version of a translate_benchmark.py translation
    # (see its "Output files"). Check L4 rejects any file with missing cells anyway.
    base = stem.removesuffix("_retranslated")
    if (FULL_DATASETS_DIR / f"{base}_cleaned.csv").exists():
        if not stem.endswith("_cleaned"):
            warnings.warn(f"{base}_cleaned.csv exists (verified by translate_benchmark.py): "
                          f"you probably want --file {base}_cleaned instead of {stem}.")
    elif language != REFERENCE_LANGUAGE and not stem.endswith(("_cleaned", "_reviewed")):
        warnings.warn(f"{input_file.name} has not been verified by translate_benchmark.py (no {base}_cleaned.csv). "
                      f"To create it (no API call if nothing is missing): `uv run python scripts/translate_benchmark.py "
                      f"--mode retranslate --language {language.removesuffix('_ep')} --file {input_file.name}`.")
    if not language.endswith("_ep") and language_config.get("intro") is None:
        warnings.warn(f"{language!r} has no translated prompts in language_prompts.py: the runner needs "
                      f"them for the non-_ep variant. Use '{language}_ep' or fill in its prompts.")
    few_shot_positions = load_few_shot_positions(dataset)
    if not args.dry_run:
        # Never overwrite: check every CSV and YAML output before writing any of them
        abort_if_files_exist(
            [p for category in categories for p in split_file_paths(slug, dataset, category)]
            + [yaml_config_path(slug, category) for category in categories],
            "output files (test/dev CSVs and YAML configs)",
        )

    print(f"Preparing {dataset.upper()} for {language!r} (folder {slug}) from {input_file.name}"
          f"{' (dry run)' if args.dry_run else ''}")

    # ---- Load and check the full dataset ----
    df = pd.read_csv(input_file)
    validate_dataset(df, dataset, language, use_question_id, input_file.name)

    # ---- Format columns ----
    df.insert(0, "Source", f"{DATASETS[dataset]['source_label']} {language_config['display_name']}")
    add_missing_answer_columns(df)

    # ---- Split per category, check and save ----
    n_few_shot = create_split_csvs(df, dataset, language, slug, few_shot_positions,
                                   use_question_id, args.dry_run)

    # ---- YAML configs ----
    if args.dry_run:
        print("\nDry run: no CSV or YAML file written.")
        return
    write_yaml_configs(dataset, slug, n_few_shot)
    print(f"\nDone. Run the benchmark with:\n  uv run python scripts/run_pedagogy_benchmark_multilingual.py "
          f"--language {language} --benchmark {dataset} --models-config <models_config>")


if __name__ == "__main__":
    main()
