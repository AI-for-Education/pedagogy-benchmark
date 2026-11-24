"""The result files of one (benchmark, metric), all named <benchmark>_<metric><suffix>:

    <benchmark>_<metric>.csv              metrics, one row per model and subset
                                          (build_metrics)
    <benchmark>_<metric>_metadata.json    the metric's definition, copied from the
                                          leaderboard config (load_metric_definition)
    <benchmark>_<metric>_frontiers.json    benchmark_frontiers.build_frontier

A local copy of qa-benchmarks-monorepo's fab_benchmark_publisher/metrics.py (branch
feat/benchmark-metrics-frontiers), via LLM_benchmark's src/metrics.py (branch
feat/results-schema-metrics-frontiers), identical to the LLM_benchmark copy so the three
can be replaced by one shared package later.
One difference with the package: build_metrics also takes per-subset values (CDPK subjects and
levels); the package writes overall rows only.

benchmark_slug, metric_slug and experiment_slug are the metrics file's composite key:
every row carries the same three. Each model has at most one row per
benchmark_subset; the overall result is the row with an empty benchmark_subset.

Only the key columns, model_slug, benchmark_subset and metric_value are filled for
now. The other columns are part of the schema but left empty until we decide how to
compute them (PLACEHOLDER_COLUMNS).

NOTHING HERE NAMES A BENCHMARK. Each benchmark's script turns its own outputs into
{model_slug: metric_value} and calls build_metrics.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

# The composite key of a metrics file: identical on every row of one file.
KEY_COLUMNS = ("benchmark_slug", "metric_slug", "experiment_slug")

# Not computed yet: written as empty cells so the file already has its final shape.
PLACEHOLDER_COLUMNS = (
    "metric_stderr",
    "metric_ci_low",
    "metric_ci_high",
    "n_items_scored",
    "n_items_bad_format",
    "n_repeats",
    "input_tokens_per_repeat_mean",
    "output_total_tokens_per_repeat_mean",
    "output_reason_tokens_per_repeat_mean",
    "output_non_reason_tokens_per_repeat_mean",
    "price_usd_per_repeat_mean",
    "price_usd_per_item_mean",
    "timestamp_metric",
)

# Every column of a metrics file, in the order they are written.
METRICS_COLUMNS = (
    *KEY_COLUMNS,
    "model_slug",
    "benchmark_subset",
    "metric_value",
    *PLACEHOLDER_COLUMNS,
)

# benchmark_subset value of the overall (whole-benchmark) row.
OVERALL_SUBSET = ""

# "_" separates the slugs in result file names, so a slug must not contain one.
SLUG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")


def result_file_name(benchmark_slug: str, metric_slug: str, suffix: str) -> str:
    """Name of one result file: <benchmark>_<metric><suffix>.

    The single place the naming convention is defined, e.g. ("CDPK-en", "accuracy",
    "_frontiers.json") -> "CDPK-en_accuracy_frontiers.json". Raises ValueError if a slug
    contains anything but letters, digits, '-' or '.', since a '_' would make the
    name ambiguous.
    """
    for slug in (benchmark_slug, metric_slug):
        if not SLUG_PATTERN.match(slug):
            raise ValueError(f"slug must be letters, digits, '-' or '.'; got {slug!r}")
    return f"{benchmark_slug}_{metric_slug}{suffix}"


def canonical_subset(subset: Mapping[str, str] | None) -> str:
    """benchmark_subset value of a subset: compact JSON with sorted keys.

    {} or None -> "" (the overall row); {"subject": "Maths"} -> '{"subject":"Maths"}'.
    One spelling per subset, so the same subset always gets the same key.
    """
    if not subset:
        return OVERALL_SUBSET
    return json.dumps(dict(subset), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_metrics(
    metric_value_by_model: dict[str, float],
    benchmark_slug: str,
    metric_slug: str,
    experiment_slug: str,
    metric_value_by_subset: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Build the metrics table written to <benchmark>_<metric>.csv.

    Args:
        metric_value_by_model: the benchmark's overall score per model,
            {model_slug: value}.
        benchmark_slug, metric_slug, experiment_slug: the file's composite key,
            repeated on every row.
        metric_value_by_subset: optional per-subset scores,
            {benchmark_subset: {model_slug: value}}, benchmark_subset being a
            canonical_subset string. Not in the package (overall rows only there).

    Returns:
        The overall row of each model, then each subset's rows, models sorted within
        each block, with METRICS_COLUMNS as columns and every PLACEHOLDER_COLUMNS
        cell left empty.
    """
    blocks = {OVERALL_SUBSET: metric_value_by_model, **(metric_value_by_subset or {})}
    rows = [
        {
            "benchmark_slug": benchmark_slug,
            "metric_slug": metric_slug,
            "experiment_slug": experiment_slug,
            "model_slug": model_slug,
            "benchmark_subset": subset,
            "metric_value": float(metric_value),
            **dict.fromkeys(PLACEHOLDER_COLUMNS),
        }
        for subset, values in blocks.items()
        for model_slug, metric_value in sorted(values.items())
    ]
    return pd.DataFrame(rows, columns=list(METRICS_COLUMNS))


def load_metric_definition(
    config_path: str | Path, benchmark_slug: str, metric_slug: str
) -> dict:
    """Read one metric's definition from the leaderboard config.

    The config lists each benchmark with a "metrics" array; the matching entry holds
    metric_slug, display_name, display_description, unit, is_higher_better and
    is_bad_format_scored. It is copied as-is into the metadata file and tells the
    caller how to compute the metric and which way its frontier goes.

    Raises ValueError if the config has no such metric for that benchmark.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for benchmark in config["benchmarks"]:
        if benchmark["benchmark_domain"] == benchmark_slug:
            for metric in benchmark["metrics"]:
                if metric["metric_slug"] == metric_slug:
                    return metric
    raise ValueError(
        f"{config_path} defines no metric '{metric_slug}' for '{benchmark_slug}'"
    )
