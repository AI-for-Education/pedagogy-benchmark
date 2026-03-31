"""
Count the number of model CSV files per language and category in cache folders.

Usage:
    # Default: scan African languages + English folders
    python scripts/count_runs.py

    # Use a different cache directory
    python scripts/count_runs.py --cache-dir data/other_cache
"""

import os
import argparse
from collections import defaultdict

# ── Folders to scan (edit this list to add/remove languages) ──────────────
LIST_FOLDERS = [
    "CDPK_Hausa_ep",
    "CDPK_Swahili_ep",
    "CDPK_Yoruba_ep",
    "CDPK_Nyankore_ep",
    "CDPK_Luganda_ep",
    "CDPK_English",
    "CDPK_Hausa",
    "CDPK_Swahili",
    "CDPK_Yoruba",
    "CDPK_Nyankore",
    "CDPK_Luganda",
]

CATEGORIES = [
    "creative_arts",
    "general",
    "literacy",
    "maths",
    "science",
    "social_studies",
    "technology",
]


def parse_folder_name(folder_name: str) -> tuple[str, str] | None:
    """Extract (language, category) from a folder name like CDPK_Hausa_ep_creative_arts."""
    for cat in CATEGORIES:
        suffix = f"_{cat}"
        if folder_name.endswith(suffix):
            lang = folder_name[: -len(suffix)].replace("CDPK_", "", 1)
            return lang, cat
    return None


def count_csv_files(folder_path: str) -> int:
    """Count .csv files directly inside a folder (non-recursive)."""
    return sum(1 for f in os.listdir(folder_path) if f.endswith(".csv"))


def build_expected_folders() -> list[str]:
    """Build the full list of expected folder names from LIST_FOLDERS x CATEGORIES."""
    folders = []
    for prefix in LIST_FOLDERS:
        for cat in CATEGORIES:
            folders.append(f"{prefix}_{cat}")
    return folders


def main():
    parser = argparse.ArgumentParser(
        description="Count model runs per language and category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "cache"),
        help="Path to cache directory (default: data/cache)",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)
    expected = build_expected_folders()

    # data[language][category] = csv_count
    data: dict[str, dict[str, int]] = defaultdict(dict)

    for folder_name in expected:
        full = os.path.join(cache_dir, folder_name)
        if not os.path.isdir(full):
            continue
        parsed = parse_folder_name(folder_name)
        if parsed is None:
            continue
        lang, cat = parsed
        data[lang][cat] = count_csv_files(full)

    if not data:
        print("No matching folders found.")
        return

    cat_list = sorted(CATEGORIES)
    # Sort: group by base language name, _ep variants after non-ep
    lang_list = sorted(data.keys(), key=lambda l: (l.replace("_ep", ""), "_ep" not in l))

    # Print table
    col_lang = "Language"
    col_cat = "Category"
    col_count = "Models Run"
    w_lang = max(len(col_lang), max(len(l) for l in lang_list))
    w_cat = max(len(col_cat), max(len(c) for c in cat_list))
    w_count = len(col_count)

    header = f"  {col_lang:<{w_lang}}  {col_cat:<{w_cat}}  {col_count}"
    sep = f"  {'-' * w_lang}  {'-' * w_cat}  {'-' * w_count}"

    print()
    print(header)
    print(sep)

    prev_lang = None
    for lang in lang_list:
        for cat in cat_list:
            count = data[lang].get(cat, 0)
            display_lang = lang if lang != prev_lang else ""
            print(f"  {display_lang:<{w_lang}}  {cat:<{w_cat}}  {count:>{w_count}}")
        prev_lang = lang
        print(sep)

    # Summary per language
    print()
    print(f"  {'Summary':<{w_lang}}  {'Total CSVs':<{w_cat}}  {'Avg/Cat':>{w_count}}")
    print(f"  {'-' * w_lang}  {'-' * w_cat}  {'-' * w_count}")
    for lang in lang_list:
        total = sum(data[lang].get(c, 0) for c in cat_list)
        n_cats = sum(1 for c in cat_list if data[lang].get(c, 0) > 0)
        avg = total / n_cats if n_cats else 0
        print(f"  {lang:<{w_lang}}  {total:<{w_cat}}  {avg:>{w_count}.1f}")
    print()


if __name__ == "__main__":
    main()
