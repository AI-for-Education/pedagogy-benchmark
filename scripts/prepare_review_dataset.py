"""
Prepare a bilingual review dataset by merging the English Pedagogy Benchmark
(CDPK) with a translated version, producing an Excel file with extra columns
for reviewers to record edits and the reasons for those edits.

Usage
-----
    uv run python scripts/prepare_review_dataset.py \
        --file2merge data/pedagogy_benchmark_full_datasets/pedagogy_benchmark_arabic_cdpk_cleaned.csv

The output .xlsx is written next to the input file.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_CSV = ROOT / "data" / "pedagogy_benchmark_full_datasets" / "pedagogy_benchmark_cdpk.csv"

TRANSLATED_FIELDS = ["question", "answer_a", "answer_b", "answer_c", "answer_d"]
DROP_COLUMNS = ["answer_e", "answer_f", "answer_g"]
MATCH_COLUMNS = [
    "question_id",
    "correct_answer",
    "category",
    "pedagogical_subdomain",
    "age_group",
    "year",
    "secondary_category",
]


def infer_language(csv_path: Path) -> str:
    """Infer the language from a filename like
    ``pedagogy_benchmark_<language>_cdpk[_cleaned].csv``.
    """
    stem = csv_path.stem
    match = re.match(r"pedagogy_benchmark_([a-zA-Z]+)_cdpk", stem)
    if not match:
        raise ValueError(
            f"Could not infer language from filename '{csv_path.name}'. "
            "Expected a name like 'pedagogy_benchmark_<language>_cdpk[_cleaned].csv'."
        )
    return match.group(1).lower()


def verify_alignment(base: pd.DataFrame, translated: pd.DataFrame) -> None:
    """Ensure the two dataframes describe the same items row-for-row on the
    non-translated metadata columns. Raises AssertionError on mismatch.
    """
    if len(base) != len(translated):
        raise AssertionError(
            f"Row count mismatch: base has {len(base)} rows, "
            f"translated has {len(translated)} rows."
        )

    shared = [c for c in MATCH_COLUMNS if c in base.columns and c in translated.columns]
    missing_in_translated = [
        c for c in MATCH_COLUMNS if c in base.columns and c not in translated.columns
    ]
    if missing_in_translated:
        raise AssertionError(
            f"Translated file is missing expected columns: {missing_in_translated}"
        )

    merged = base[shared].merge(
        translated[shared],
        on="question_id",
        how="outer",
        suffixes=("_base", "_translated"),
        indicator=True,
    )
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        ids = unmatched["question_id"].tolist()
        raise AssertionError(
            f"question_id values do not align between files. "
            f"{len(ids)} rows differ (e.g. {ids[:5]})."
        )

    for col in shared:
        if col == "question_id":
            continue
        left = merged[f"{col}_base"]
        right = merged[f"{col}_translated"]
        diff_mask = (left != right) & ~(left.isna() & right.isna())
        if diff_mask.any():
            bad = merged.loc[diff_mask, ["question_id", f"{col}_base", f"{col}_translated"]]
            raise AssertionError(
                f"Column '{col}' does not match between base and translated files.\n"
                f"First mismatches:\n{bad.head().to_string(index=False)}"
            )


def build_review_dataset(language_csv: Path, language: str) -> pd.DataFrame:
    base = pd.read_csv(BASE_CSV)
    translated = pd.read_csv(language_csv)

    verify_alignment(base, translated)

    base = base.drop(columns=[c for c in DROP_COLUMNS if c in base.columns])

    translated_subset = translated[["question_id", *TRANSLATED_FIELDS]].copy()
    rename_map = {field: f"{field}_{language}" for field in TRANSLATED_FIELDS}
    translated_subset = translated_subset.rename(columns=rename_map)

    merged = base.merge(translated_subset, on="question_id", how="left")

    insert_after = "answer_d"
    cols = list(merged.columns)
    insert_idx = cols.index(insert_after) + 1
    translated_cols = [rename_map[f] for f in TRANSLATED_FIELDS]
    remaining = [c for c in cols if c not in translated_cols]
    new_order = remaining[:insert_idx] + translated_cols + remaining[insert_idx:]
    merged = merged[new_order]

    for field in TRANSLATED_FIELDS:
        merged[f"{field}_{language}_edit"] = None
        merged[f"{field}_{language}_edit_reason"] = None

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file2merge",
        required=True,
        type=Path,
        help="Path to the translated Pedagogy Benchmark CSV to merge with the base.",
    )
    args = parser.parse_args()

    language_csv = args.file2merge.resolve()
    if not language_csv.is_file():
        raise FileNotFoundError(f"File to merge not found: {language_csv}")

    language = infer_language(language_csv)

    print(f"Merging base dataset: {BASE_CSV}")
    print(f"With translated dataset: {language_csv}")
    print(f"Using language suffix: '{language}'")

    merged = build_review_dataset(language_csv, language)

    output_path = language_csv.parent / f"pedagogy_benchmark_review_english_{language}.xlsx"
    merged.to_excel(output_path, index=False, engine="openpyxl")

    print(f"Wrote review dataset: {output_path}")
    print(f"Rows: {len(merged)} | Columns: {len(merged.columns)}")


if __name__ == "__main__":
    main()
