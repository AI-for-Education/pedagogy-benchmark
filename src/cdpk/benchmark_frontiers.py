"""The content of the frontier file, <benchmark>_<metric>_frontier.json: one frontier
per axis over a metrics table's overall rows:

  price   blended price in USD per million tokens, every priced model
  size    parameter count, open-weight models only

both read from the models CSV passed to build_frontier (via load_models). Each
frontier is the raw Pareto frontier, stored as plain points.

A local copy of qa-benchmarks-monorepo's fab_benchmark_publisher/benchmark_frontiers.py
(branch feat/benchmark-metrics-frontiers), via LLM_benchmark's src/benchmark_frontiers.py
(branch feat/results-schema-metrics-frontiers). load_models is a minimal copy of the
package's leaderboard.load_models (there is no leaderboard.py here).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from .frontier import compute_frontier
from .metrics import KEY_COLUMNS, OVERALL_SUBSET

# ------------------------------------------------------------------- model metadata

# Model prices, USD per million tokens. The first two are models.csv's column names;
# the blended price is derived from them in load_models.
INPUT_PRICE = "input_price_usd_per_mtok"
OUTPUT_PRICE = "output_price_usd_per_mtok"
BLENDED_PRICE = "blended_price_usd_per_mtok"
PRICE_COLUMNS = (INPUT_PRICE, OUTPUT_PRICE)


def _float_or_none(value: str | None) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def load_models(models_csv: str | Path) -> dict[str, dict]:
    """model_id -> {open, size, input/output/blended price} from models.csv.

    The blended price is the 3:1 input:output weighted average, as on the CDPK
    leaderboard. Raises ValueError if the CSV has no price columns: every price
    frontier would otherwise come out empty without notice.
    """
    with open(models_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in PRICE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{models_csv} has no {', '.join(missing)} column(s)")
        models: dict[str, dict] = {}
        for row in reader:
            inp = _float_or_none(row.get(INPUT_PRICE))
            out = _float_or_none(row.get(OUTPUT_PRICE))
            models[row["model_id"].strip()] = {
                "open": row["open"].strip().lower() == "true",
                "size": _float_or_none(row.get("size")),
                INPUT_PRICE: inp,
                OUTPUT_PRICE: out,
                BLENDED_PRICE: round((inp * 3 + out) / 4, 6)
                if inp is not None and out is not None
                else None,
            }
    return models


# ------------------------------------------------------------------------- frontier

# Frontier axes: axis name -> the load_models field used as its x value.
X_FIELD_BY_AXIS = {"price": BLENDED_PRICE, "size": "size"}


def _frontier_points(
    metric_value_by_model: dict[str, float],
    axis: str,
    model_metadata: dict[str, dict],
    *,
    is_higher_better: bool,
) -> list[dict]:
    """Compute one axis's frontier.

    Only models that can be placed on the axis are candidates: listed in the models
    CSV, with a positive x value, and open-weight for the size axis.

    Returns:
        The frontier models in order of increasing x, as
        [{"model_slug", "x", "y"}, ...]; empty if no model can be placed.
    """
    x_field = X_FIELD_BY_AXIS[axis]
    placeable = [
        {"model_slug": model_slug, "x": float(metadata[x_field]), "y": metric_value}
        for model_slug, metric_value in sorted(metric_value_by_model.items())
        if (metadata := model_metadata.get(model_slug)) is not None
        and (axis != "size" or metadata["open"])
        and metadata[x_field] is not None
        and metadata[x_field] > 0
    ]
    if not placeable:
        return []
    candidates = pd.DataFrame(placeable)
    # compute_frontier keeps whatever is largest so far; flip the sign for metrics
    # where smaller is better.
    candidates["score"] = candidates["y"] if is_higher_better else -candidates["y"]
    frontier = compute_frontier(
        candidates, acc_col="score", sorting_col="x", smooth=False
    )
    return [
        {"model_slug": point["model_slug"], "x": point["x"], "y": point["y"]}
        for _, point in frontier.sort_values("x").iterrows()
    ]


def build_frontier(
    metrics: pd.DataFrame, *, is_higher_better: bool, models_csv: str | Path
) -> dict[str, Any]:
    """Build the content of <benchmark>_<metric>_frontier.json from a metrics table.

    Args:
        metrics: a metrics table (build_metrics, or the CSV read back). Only its
            overall rows are used; an empty benchmark_subset read back as NaN counts
            as overall.
        is_higher_better: the metric's direction, from its definition.
        models_csv: the models CSV giving each model's price, size and open weights.

    Returns:
        A dict with the metrics file's key (benchmark_slug, metric_slug,
        experiment_slug), is_higher_better, models_without_metadata (models absent
        from the models CSV) and frontiers: one {"axis", "x", "points"} entry per
        X_FIELD_BY_AXIS axis.
    """
    model_metadata = load_models(models_csv)
    overall_rows = metrics[
        metrics["benchmark_subset"].fillna(OVERALL_SUBSET) == OVERALL_SUBSET
    ]
    metric_value_by_model = dict(
        zip(overall_rows["model_slug"], overall_rows["metric_value"].astype(float))
    )
    key_row = metrics.iloc[0]
    return {
        **{column: str(key_row[column]) for column in KEY_COLUMNS},
        "is_higher_better": is_higher_better,
        "models_without_metadata": sorted(
            set(metric_value_by_model) - set(model_metadata)
        ),
        "frontiers": [
            {
                "axis": axis,
                "x": x_field,
                "points": _frontier_points(
                    metric_value_by_model,
                    axis,
                    model_metadata,
                    is_higher_better=is_higher_better,
                ),
            }
            for axis, x_field in X_FIELD_BY_AXIS.items()
        ],
    }
