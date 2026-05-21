"""
Count the number of model CSV files per language and category in cache folders.

Usage:
    # Default: scan African languages + English folders
    python scripts/count_runs.py

    # Use a different cache directory
    python scripts/count_runs.py --cache-dir data/cache_local

    # List models present (uncommented) in the LLM_benchmark reference list
    # but still commented out / missing in this repo's local list
    python scripts/count_runs.py --check_models
"""

import os
import argparse
from collections import defaultdict

# ── Default YAML paths for --check_models ────────────────────────────────
DEFAULT_LOCAL_YAML = os.path.join(
    os.path.dirname(__file__), "..", "configs", "models", "full_list_default_models_20260218.yaml"
)
DEFAULT_REF_YAML = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "LLM_benchmark", "configs", "models", "full_list_20250627.yaml",
)

# ── Folders to scan (edit this list to add/remove languages) ──────────────
LIST_FOLDERS = [
    "CDPK_Hausa_ep",
    "CDPK_Swahili_ep",
    "CDPK_Yoruba_ep",
    "CDPK_Nyankore_ep",
    "CDPK_Luganda_ep",
    "CDPK_Arabic_ep",
    "CDPK_Dari_ep",
    "CDPK_Pashto_ep",
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


def get_model_names(folder_path: str) -> set[str]:
    """Return the set of model IDs found in a folder (strip 'resps_' prefix and '.csv' suffix)."""
    models = set()
    for f in os.listdir(folder_path):
        if f.endswith(".csv") and f.startswith("resps_"):
            models.add(f[len("resps_"):-len(".csv")])
    return models


def build_expected_folders() -> list[str]:
    """Build the full list of expected folder names from LIST_FOLDERS x CATEGORIES."""
    folders = []
    for prefix in LIST_FOLDERS:
        for cat in CATEGORIES:
            folders.append(f"{prefix}_{cat}")
    return folders


def parse_model_yaml(path: str) -> tuple[dict[str, str], dict[str, str]]:
    """Parse a model list YAML.

    Returns (active, commented):
        active[model_id]   = display_name   for uncommented entries
        commented[model_id] = display_name  for entries commented out with '#'
    Display names have inline '# ...' comments stripped.
    """
    active: dict[str, str] = {}
    commented: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            is_commented = stripped.startswith("#")
            if is_commented:
                stripped = stripped.lstrip("#").strip()
            if not stripped or ":" not in stripped:
                continue
            key, _, value = stripped.partition(":")
            model_id = key.strip()
            if not model_id:
                continue
            display = value.split("#", 1)[0].strip()
            if is_commented:
                commented[model_id] = display
            else:
                active[model_id] = display
    return active, commented


def check_models() -> None:
    """Print models active in the LLM_benchmark reference YAML but not active locally."""
    local_yaml = DEFAULT_LOCAL_YAML
    ref_yaml = DEFAULT_REF_YAML
    if not os.path.isfile(ref_yaml):
        print(f"Reference YAML not found: {ref_yaml}")
        return
    if not os.path.isfile(local_yaml):
        print(f"Local YAML not found: {local_yaml}")
        return

    local_active, local_commented = parse_model_yaml(local_yaml)
    ref_active, _ = parse_model_yaml(ref_yaml)

    missing: list[tuple[str, str, str]] = []
    for model_id, display in ref_active.items():
        if model_id in local_active:
            continue
        status = "commented" if model_id in local_commented else "missing"
        missing.append((model_id, display, status))

    print()
    print(f"  Reference: {os.path.abspath(ref_yaml)}")
    print(f"  Local:     {os.path.abspath(local_yaml)}")
    print()
    print(f"  {len(ref_active)} active in reference, {len(local_active)} active in local.")
    print(f"  {len(missing)} models active in reference but not active in local:")
    print()

    if not missing:
        print("  (none — local list covers all active reference models)")
        return

    w_id = max(len("Model ID"), max(len(m) for m, _, _ in missing))
    w_name = max(len("Display Name"), max(len(d) for _, d, _ in missing))
    w_status = max(len("Status"), max(len(s) for _, _, s in missing))

    print(f"  {'Model ID':<{w_id}}  {'Display Name':<{w_name}}  {'Status':<{w_status}}")
    print(f"  {'-' * w_id}  {'-' * w_name}  {'-' * w_status}")
    for model_id, display, status in sorted(missing):
        print(f"  {model_id:<{w_id}}  {display:<{w_name}}  {status:<{w_status}}")
    print()


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
    parser.add_argument(
        "--check_models",
        action="store_true",
        help="List models active in the LLM_benchmark reference YAML that are not "
             "yet active in this repo's local YAML, then exit.",
    )
    args = parser.parse_args()

    if args.check_models:
        check_models()
        return

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

    # ── Model × Language presence matrix ─────────────────────────────────────
    # A model is "done" for a language only if it has a CSV in ALL 7 categories.
    # models_per_cat_lang[lang][cat] = set of model ids
    models_per_cat_lang: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for folder_name in expected:
        full = os.path.join(cache_dir, folder_name)
        if not os.path.isdir(full):
            continue
        parsed = parse_folder_name(folder_name)
        if parsed is None:
            continue
        lang, cat = parsed
        models_per_cat_lang[lang][cat] |= get_model_names(full)

    # For each (lang, model), count how many categories are missing
    # missing_count[lang][model] = number of categories where the model has no CSV
    missing_count: dict[str, dict[str, int]] = {}
    all_seen_models: set[str] = set()
    for lang, cat_models in models_per_cat_lang.items():
        all_cats = set(CATEGORIES)
        all_lang_models: set[str] = set()
        for s in cat_models.values():
            all_lang_models |= s
        all_seen_models |= all_lang_models
        missing_count[lang] = {
            model: sum(1 for cat in all_cats if model not in cat_models.get(cat, set()))
            for model in all_lang_models
        }

    all_models = sorted(all_seen_models)
    # Order languages as they appear in LIST_FOLDERS
    _folders_order = [f.replace("CDPK_", "", 1) for f in LIST_FOLDERS]
    matrix_langs = [l for l in _folders_order if l in models_per_cat_lang]

    if all_models:
        w_model = max(len("Model"), max(len(m) for m in all_models))
        col_w = max(len(l) for l in matrix_langs)

        header_parts = [f"{'Model':<{w_model}}"] + [f"{l:^{col_w}}" for l in matrix_langs]
        print("  " + "  ".join(header_parts))
        sep_parts = ["-" * w_model] + ["-" * col_w for _ in matrix_langs]
        print("  " + "  ".join(sep_parts))

        sep_line = "  " + "  ".join(sep_parts)
        for i, model in enumerate(all_models):
            if i > 0 and i % 5 == 0:
                print(sep_line)
            row_parts = [f"{model:<{w_model}}"]
            for lang in matrix_langs:
                n_missing = missing_count.get(lang, {}).get(model, len(CATEGORIES))
                val = "" if n_missing == 0 else str(n_missing)
                row_parts.append(f"{val:^{col_w}}")
            print("  " + "  ".join(row_parts))
        print()


if __name__ == "__main__":
    main()
