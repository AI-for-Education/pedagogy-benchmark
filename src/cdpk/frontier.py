"""Pareto frontier of a set of models.

A local copy of qa-benchmarks-monorepo's fab_benchmark_publisher/frontier.py
compute_frontier (branch feat/benchmark-metrics-frontiers), via LLM_benchmark's
src/frontier.py, same signature, so it can
be replaced by the package later. Only the non-smooth branch is copied: the result
files use the raw frontier (smooth=False). The package's frontier.py also holds the
website helpers (create_default_models_list, ...), which are not needed here.
"""

from __future__ import annotations

import pandas as pd


def compute_frontier(data, sorting_col, acc_col="accuracy", smooth=None):
    """
    Computes the value frontier from a set of models.

    1. Sorts models by sorting_col.
    2. Finds the Pareto frontier (models where accuracy is the highest seen so far).
    3. smooth=True (concave hull) is not copied here and raises NotImplementedError.

    Args:
        data: DataFrame containing model data
        acc_col: Column name for accuracy metric (default: 'accuracy')
        sorting_col: Column name for the independent variable
        smooth: must be falsy here

    Returns:
        DataFrame containing frontier models
    """
    if smooth:
        raise NotImplementedError("smooth=True is only in fab_benchmark_publisher")
    data = data.sort_values(sorting_col, ascending=True)
    frontier, best = [], -1e9

    for _, row in data.iterrows():
        if row[acc_col] > best:
            frontier.append(row)
            best = row[acc_col]

    if not frontier:
        return pd.DataFrame(columns=data.columns)

    df_return = pd.DataFrame(frontier)

    # if rows have the same sorting_col value, keep the one with highest accuracy
    df_return = df_return.sort_values(sorting_col).drop_duplicates(
        sorting_col, keep="last"
    )
    return df_return.reset_index(drop=True)
