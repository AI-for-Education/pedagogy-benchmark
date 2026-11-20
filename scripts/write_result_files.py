#!/usr/bin/env python3
"""
Write the result files of the translated CDPK benchmarks, one benchmark per language.

Output, <out-root> defaulting to the repo's results/ (local only: upload them with
scripts/upload_result_files.py, to gs://fab-benchmarks-hub/metrics/):

    <out-root>/CDPK-sw/
        CDPK-sw_accuracy.csv             metrics (overall, per subject, per level)
        CDPK-sw_accuracy_metadata.json   metric metadata
        CDPK-sw_accuracy_frontier.json   price and size frontiers (overall rows)
    <out-root>/CDPK-ar/
        ... the same for every language in BENCHMARKS

Run from the repo root:

    uv run python scripts/write_result_files.py                    # every language
    uv run python scripts/write_result_files.py --benchmark CDPK-sw

English is not here. CDPK-en (and SEND-en) are published from LLM_benchmark
(scripts/write_result_files.py, branch feat/results-schema-metrics-frontiers), which
runs the leaderboard's English dataset and prompt. Merging LLM_benchmark and
pedagogy-benchmark into a single repo is left for later: see consolidation_plan.md.

This script is a port of LLM_benchmark's write_result_files.py, kept as close to it
as possible (same steps, same function names, same level rules) so both can move to
one shared package later. The differences: the benchmarks are languages, the test
CSVs use this repo's column names (question, answer_a..g, correct_answer, category,
age_group), and subjects keep the dataset's category names ("General", not "Gen PK").

First, models.csv and providers.csv are downloaded from the public bucket
gs://fab-open-configs (the configs shared by every benchmark) to the repo root,
overwriting the local copies only when they differ. The fab-benchmarks-configs
submodule's models.csv is not used: it still has the old price column names.

Steps, per language:
  1. metrics   the cached responses, data/cache/CDPK_<Language>_ep_<category>/
               resps_<model>.csv, scored with the regex parser (clean_resps)
               -> <benchmark>_accuracy.csv
  2. metadata  the metric's definition in leaderboard_config.json (repo root)
               -> <benchmark>_accuracy_metadata.json
  3. frontier  <benchmark>_accuracy.csv (read back from disk) + the models CSV
               (models.csv at the repo root)
               -> <benchmark>_accuracy_frontier.json

Only data/cache (the shared DVC cache) is read, never data/cache_local: the result
files must be reproducible by anyone who runs `dvc pull`. Run scripts/merge_cache.py
and push the cache before publishing new runs.

Only the English-prompt variant of each language (<Language>_ep, the one the
multilingual leaderboard uses) is published; native-prompt, translation-model
experiments (argon4low), reviewed subsets and Core-300 subsets are not. Only models
in the benchmark's model list (configs/models/<models_config>.yaml) are published.

Items: the test rows of the 7 per-category question configs (899 items). The few-shot
examples are in separate dev files; test items identical to one of them, and repeated
test items, would be dropped as in LLM_benchmark. The run stops if the count changes:
update expected_items and experiment_slug in BENCHMARKS when the question set changes
on purpose.

Accuracy follows the metric's is_bad_format_scored in the leaderboard config. A
response the regex parser cannot read (including a failed API call) is bad format:
counted as wrong when the flag is true, left out when false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdpk.benchmark_answers import clean_answers, clean_resps
from cdpk.benchmark_constants import CACHE_GLOBAL_DIR
from cdpk.benchmark_frontiers import build_frontier
from cdpk.benchmark_run import load_from_config
from cdpk.metrics import (
    build_metrics,
    canonical_subset,
    load_metric_definition,
    result_file_name,
)

# The shared DVC cache only (never data/cache_local).
CACHE_DIR = CACHE_GLOBAL_DIR
MODELS_CSV = ROOT / "models.csv"

# Configs shared by every benchmark, in a public bucket (anonymous reads over HTTPS).
# Synced to the repo root, where they are gitignored.
SHARED_CONFIG_URL = "https://storage.googleapis.com/fab-open-configs"
SHARED_CONFIGS = ("models.csv", "providers.csv")
CONFIG_FILE = ROOT / "leaderboard_config.json"

# The one metric these benchmarks publish; its definition is in the leaderboard config.
METRIC_SLUG = "accuracy"

# Suffixes of the per-category question configs, configs/questions/CDPK_<slug>_<cat>.yaml
CATEGORIES = (
    "science",
    "literacy",
    "creative_arts",
    "maths",
    "social_studies",
    "technology",
    "general",
)

# benchmark_slug -> language slug (data/<slug>/, CDPK_<slug>_<category>).
# Slugs use the ISO 639 code: 2-letter when one exists, else 3-letter (Dari: prs,
# Nyankore: nyn). Twi (tw), Xhosa (xh), Northern Sotho (nso) and French (fr) are
# translated but not published yet: too few models have been run on them.
LANGUAGES = {
    "CDPK-ar": "Arabic_ep",
    "CDPK-prs": "Dari_ep",
    "CDPK-ha": "Hausa_ep",
    "CDPK-lg": "Luganda_ep",
    "CDPK-nyn": "Nyankore_ep",
    "CDPK-ps": "Pashto_ep",
    "CDPK-sw": "Swahili_ep",
    "CDPK-yo": "Yoruba_ep",
}

# The models list of the multilingual leaderboard (as in create_results.py).
MODELS_CONFIG = "full_list_default_models_20260218"


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_slug: str
    # There is no experiment folder in data/cache, so the experiment is named here.
    experiment_slug: str
    question_configs: tuple[str, ...]
    models_config: str
    expected_items: int
    # Also write per-subject and per-level rows.
    subsets: bool = False


BENCHMARKS = {
    slug: BenchmarkConfig(
        benchmark_slug=slug,
        experiment_slug="final-899-items",
        question_configs=tuple(f"CDPK_{language}_{cat}" for cat in CATEGORIES),
        models_config=MODELS_CONFIG,
        expected_items=899,
        subsets=True,
    )
    for slug, language in LANGUAGES.items()
}


def sync_shared_configs() -> None:
    """Download SHARED_CONFIGS to the repo root, rewriting a file only if it differs."""
    for name in SHARED_CONFIGS:
        with urllib.request.urlopen(f"{SHARED_CONFIG_URL}/{name}", timeout=60) as response:
            remote = response.read()
        path = ROOT / name
        if path.exists() and path.read_bytes() == remote:
            print(f"{name}: up to date")
            continue
        print(f"{name}: {'updated' if path.exists() else 'downloaded'} from {SHARED_CONFIG_URL}/")
        path.write_bytes(remote)


def _show(path: Path) -> str:
    """Path relative to the repo root when inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


# ------------------------------------------------------------------------------ items


def is_primary(age_group) -> bool:
    """Same rule as CDPK-en (LLM_benchmark's write_result_files.py)."""
    if pd.isna(age_group):
        return False
    return ("Primary" in age_group and "Pre-Primary" not in age_group) or age_group == "All"


def is_secondary(age_group) -> bool:
    if pd.isna(age_group):
        return False
    return "Secondary" in age_group or age_group == "All"


def is_pre_primary(age_group) -> bool:
    """Same pattern ('All' counts in every level)."""
    if pd.isna(age_group):
        return False
    return "pre-primary" in age_group.lower() or age_group == "All"


# level name -> item flag column. Levels overlap: an item can be in several.
LEVELS = {"Pre-primary": "is_pre_primary", "Primary": "is_primary", "Secondary": "is_secondary"}


def subject_name(category: str) -> str:
    """The dataset's category, e.g. 'Maths', 'General'."""
    return category.strip()


def item_hash(row: pd.Series, columns: list[str]) -> str:
    text = "\x1f".join("" if pd.isna(row[c]) else str(row[c]).strip() for c in columns)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_items(questions_config: str, models_config: str) -> tuple[pd.DataFrame, dict, set]:
    """Test items of one question config, in the cache's row order.

    Returns the items (``position`` = row in the cache file), the model list, and the
    item ids of the few-shot examples shown in the prompt.
    """
    df, config, models = load_from_config(questions_config, models_config)
    n_examples = len(config["example_rows"] or [])
    id_columns = (
        ["question"]
        + [c for c in df.columns if str(c).startswith("answer_")]
        + ["correct_answer"]
    )
    ids = df.apply(item_hash, axis=1, columns=id_columns)
    few_shot_ids = set(ids.iloc[:n_examples])

    test = df.iloc[n_examples:].reset_index(drop=True)
    items = pd.DataFrame(
        {
            "item_id": ids.iloc[n_examples:].to_numpy(),
            "position": range(len(test)),
            "questions_config": questions_config,
            "gold": test["correct_answer"].astype(str).apply(clean_answers),
            "subject": test["category"].astype(str).apply(subject_name),
            "is_pre_primary": test["age_group"].apply(is_pre_primary),
            "is_primary": test["age_group"].apply(is_primary),
            "is_secondary": test["age_group"].apply(is_secondary),
        }
    )
    return items, models, few_shot_ids


def load_benchmark_items(cfg: BenchmarkConfig) -> tuple[pd.DataFrame, list[str]]:
    """All items of a benchmark, without duplicates, and its model list.

    Drops test items identical to a few-shot example and repeated test items (keeping
    the first), as LLM_benchmark does for CDPK-en and SEND-en.
    """
    parts, few_shot_ids, models = [], set(), None
    for qc in cfg.question_configs:
        items, models_dict, fs_ids = load_items(qc, cfg.models_config)
        parts.append(items)
        few_shot_ids |= fs_ids
        models = list(models_dict)
    items = pd.concat(parts, ignore_index=True)
    items["keep"] = ~items["item_id"].isin(few_shot_ids) & ~items["item_id"].duplicated()
    n_kept = int(items["keep"].sum())
    if n_kept != cfg.expected_items:
        raise ValueError(
            f"{cfg.benchmark_slug}: expected {cfg.expected_items} items, found {n_kept}. "
            "If the question set changed on purpose, update expected_items and the "
            "experiment_slug."
        )
    return items, models


# ----------------------------------------------------------------------------- scores


def read_responses(questions_config: str, model_id: str) -> pd.Series | None:
    """Raw responses of one model, from CACHE_DIR (data/cache) only."""
    path = CACHE_DIR / questions_config / f"resps_{model_id}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)["resps"]


def cached_models(cfg: BenchmarkConfig) -> set[str]:
    """Models with a cache file in any of the benchmark's question configs."""
    return {
        path.stem.removeprefix("resps_")
        for qc in cfg.question_configs
        for path in (CACHE_DIR / qc).glob("resps_*.csv")
    }


def score_items(
    cfg: BenchmarkConfig, items: pd.DataFrame, models: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """One row per (model, kept item): regex-parse each cached response and score it.

    A listed model must have a cache file for every question config: models with none
    are reported (not run yet), models with only some stop the run.

    Returns:
        (scores with item_score 0/1 and is_bad_format, listed models with no cache,
         cached models that are not listed).
    """
    rows, listed_not_cached, errors = [], [], []
    for model in models:
        resps = {qc: read_responses(qc, model) for qc in cfg.question_configs}
        missing = [qc for qc, r in resps.items() if r is None]
        if len(missing) == len(resps):
            listed_not_cached.append(model)
            continue
        if missing:
            errors.append(f"{model}: no cache file for {missing}")
            continue
        for qc, r in resps.items():
            n_items = int((items["questions_config"] == qc).sum())
            if len(r) != n_items:
                errors.append(f"{model}: {qc} cache has {len(r)} rows, expected {n_items}")
        if errors:
            continue
        model_items = items.copy()
        model_items["resp"] = [
            resps[qc].iloc[pos]
            for qc, pos in zip(model_items["questions_config"], model_items["position"])
        ]
        model_items["pred"] = model_items["resp"].apply(clean_resps)
        model_items["model_slug"] = model
        rows.append(model_items[model_items["keep"]])
    if errors:
        raise ValueError("Cannot score responses:\n" + "\n".join(f"- {e}" for e in errors))
    if not rows:
        raise SystemExit(f"ERROR: {cfg.benchmark_slug}: no listed model has cached responses")

    scores = pd.concat(rows, ignore_index=True)
    scores["is_bad_format"] = scores["pred"].isna()
    scores["item_score"] = (scores["pred"] == scores["gold"]).astype(int)
    cached_not_listed = sorted(cached_models(cfg) - set(models))
    return scores, listed_not_cached, cached_not_listed


def accuracy_from_scores(scores: pd.DataFrame, *, is_bad_format_scored: bool) -> float | None:
    """Accuracy in % of one model's scored items, rounded to 2 decimals like LK.

    Bad-format items count as wrong when is_bad_format_scored, else they are left
    out. None when no item counts.
    """
    is_scored = ~scores["is_bad_format"]
    n_correct = int(scores.loc[is_scored, "item_score"].sum())
    n_counted = len(scores) if is_bad_format_scored else int(is_scored.sum())
    return round(n_correct / n_counted * 100, 2) if n_counted else None


def accuracy_by_model(scores: pd.DataFrame, *, is_bad_format_scored: bool) -> dict[str, float]:
    """{model: accuracy} over the given scores; models with no counted item left out."""
    accuracy_of_model = {}
    for model, model_scores in scores.groupby("model_slug", sort=True):
        accuracy = accuracy_from_scores(model_scores, is_bad_format_scored=is_bad_format_scored)
        if accuracy is not None:
            accuracy_of_model[model] = accuracy
    return accuracy_of_model


def accuracy_by_subset(
    scores: pd.DataFrame, *, is_bad_format_scored: bool
) -> dict[str, dict[str, float]]:
    """{benchmark_subset: {model: accuracy}} for each subject, then each level."""
    by_subset = {}
    for subject in sorted(scores["subject"].unique()):
        by_subset[canonical_subset({"subject": subject})] = accuracy_by_model(
            scores[scores["subject"] == subject], is_bad_format_scored=is_bad_format_scored
        )
    for level, flag in LEVELS.items():
        level_scores = scores[scores[flag]]
        if len(level_scores):
            by_subset[canonical_subset({"level": level})] = accuracy_by_model(
                level_scores, is_bad_format_scored=is_bad_format_scored
            )
    return by_subset


# ------------------------------------------------------------------------------- main


def write_result_files(
    cfg: BenchmarkConfig,
    *,
    models_csv: Path,
    config_path: Path,
    out_root: Path,
) -> list[Path]:
    """Write the metrics, metadata and frontier files of one benchmark.

    Runs the three steps of the module docstring, writes the files to
    <out_root>/<benchmark_slug>/ and prints what was published and
    which models were left out.

    Returns:
        The paths of the metrics, metadata and frontier files.
    """
    slug = cfg.benchmark_slug
    print(f"\n== {slug} ({cfg.experiment_slug}) ==")
    metric_definition = load_metric_definition(config_path, slug, METRIC_SLUG)
    is_bad_format_scored = metric_definition["is_bad_format_scored"]

    items, models = load_benchmark_items(cfg)
    scores, listed_not_cached, cached_not_listed = score_items(cfg, items, models)
    metrics = build_metrics(
        accuracy_by_model(scores, is_bad_format_scored=is_bad_format_scored),
        benchmark_slug=slug,
        metric_slug=METRIC_SLUG,
        experiment_slug=cfg.experiment_slug,
        metric_value_by_subset=(
            accuracy_by_subset(scores, is_bad_format_scored=is_bad_format_scored)
            if cfg.subsets
            else None
        ),
    )

    outdir = out_root / slug
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path, metadata_path, frontier_path = (
        outdir / result_file_name(slug, METRIC_SLUG, suffix)
        for suffix in (".csv", "_metadata.json", "_frontier.json")
    )
    metrics.to_csv(metrics_path, index=False)
    metadata = {"benchmark_slug": slug, **metric_definition}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # the frontier is built from the metrics file just written, not from memory
    frontier = build_frontier(
        pd.read_csv(metrics_path),
        is_higher_better=metric_definition["is_higher_better"],
        models_csv=models_csv,
    )
    frontier_path.write_text(json.dumps(frontier, indent=2) + "\n", encoding="utf-8")

    n_models = scores["model_slug"].nunique()
    print(f"{slug}: {n_models} models, {int(items['keep'].sum())} items, {len(metrics)} rows")
    if listed_not_cached:
        print(f"  in {cfg.models_config}.yaml but not in the cache: {', '.join(listed_not_cached)}")
    if cached_not_listed:
        print(
            f"  in the cache but not in {cfg.models_config}.yaml (left out): "
            + ", ".join(cached_not_listed)
        )
    for axis_frontier in frontier["frontiers"]:
        print(f"  {axis_frontier['axis']} frontier: {len(axis_frontier['points'])} models")
    if frontier["models_without_metadata"]:
        print(
            f"  not in {Path(models_csv).name} (no frontier): "
            + ", ".join(frontier["models_without_metadata"])
        )
    paths = [metrics_path, metadata_path, frontier_path]
    for path in paths:
        print(f"  -> {_show(path)}")
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line and write the result files of each selected benchmark.

    `argv` defaults to the real command line; tests pass their own.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=list(BENCHMARKS),
        help="only write this benchmark (repeatable; default: all)",
    )
    parser.add_argument("--models-csv", type=Path, default=MODELS_CSV)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    sync_shared_configs()
    for slug in args.benchmark or list(BENCHMARKS):
        write_result_files(
            BENCHMARKS[slug],
            models_csv=args.models_csv,
            config_path=args.config,
            out_root=args.out_root,
        )


if __name__ == "__main__":
    main()
