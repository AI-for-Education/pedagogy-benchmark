"""metrics.py and benchmark_frontiers.py (mirrors the monorepo's fab_benchmark_publisher test_metrics.py)."""

import json

import pandas as pd
import pytest

from conftest import ROOT
from cdpk.benchmark_frontiers import BLENDED_PRICE, build_frontier, load_models
from cdpk.metrics import (
    METRICS_COLUMNS,
    PLACEHOLDER_COLUMNS,
    build_metrics,
    canonical_subset,
    load_metric_definition,
    result_file_name,
)

VALUES = {"a": 75.0, "b": 50.0, "c": 60.0, "d": 80.0}


def metrics(**kw) -> pd.DataFrame:
    return build_metrics(VALUES, "CDPK-en", "accuracy", "final-4-items", **kw)


@pytest.fixture
def models_csv(tmp_path):
    # a: closed, no size; b: open 7B, cheapest; c: open 70B; d: not in the CSV
    path = tmp_path / "models.csv"
    pd.DataFrame(
        {
            "model_id": ["a", "b", "c"],
            "open": [False, True, True],
            "size": [None, 7e9, 70e9],
            "input_price_usd_per_mtok": [1.0, 0.1, 0.5],
            "output_price_usd_per_mtok": [4.0, 0.2, 0.5],
        }
    ).to_csv(path, index=False)
    return path


def test_build_metrics_fills_only_key_and_value_columns() -> None:
    m = metrics()
    assert list(m.columns) == list(METRICS_COLUMNS)
    assert m["model_slug"].tolist() == ["a", "b", "c", "d"]
    assert (m["benchmark_subset"] == "").all()
    assert m[list(PLACEHOLDER_COLUMNS)].isna().all().all()


def test_subset_rows_follow_the_overall_rows() -> None:
    maths = canonical_subset({"subject": "Maths"})
    m = metrics(metric_value_by_subset={maths: {"b": 40.0, "a": 100.0}})
    assert m["benchmark_subset"].tolist() == [""] * 4 + [maths] * 2
    assert m["model_slug"].tolist()[4:] == ["a", "b"]
    assert m[list(PLACEHOLDER_COLUMNS)].isna().all().all()


def test_canonical_subset() -> None:
    assert canonical_subset(None) == ""
    assert canonical_subset({"b": "2", "a": "1"}) == '{"a":"1","b":"2"}'


def test_file_names_have_no_experiment_slug() -> None:
    assert result_file_name("CDPK-en", "accuracy", ".csv") == "CDPK-en_accuracy.csv"
    assert result_file_name("SEND-en", "accuracy", "_frontier.json") == "SEND-en_accuracy_frontier.json"
    with pytest.raises(ValueError):
        result_file_name("bad_slug", "accuracy", ".csv")


def test_metric_definition_comes_from_the_config(tmp_path) -> None:
    config = {
        "cluster_name": "c",
        "benchmarks": [
            {"benchmark_domain": "X", "display_name": "X", "metrics": [{"metric_slug": "m"}]}
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert load_metric_definition(path, "X", "m") == {"metric_slug": "m"}
    with pytest.raises(ValueError, match="defines no metric"):
        load_metric_definition(path, "X", "other")


REPO_BENCHMARKS = [
    b["benchmark_domain"]
    for b in json.loads((ROOT / "leaderboard_config.json").read_text(encoding="utf-8"))["benchmarks"]
]


@pytest.mark.parametrize("benchmark", REPO_BENCHMARKS)
def test_repo_config_defines_accuracy(benchmark) -> None:
    metric = load_metric_definition(ROOT / "leaderboard_config.json", benchmark, "accuracy")
    assert metric["is_higher_better"] is True
    assert metric["is_bad_format_scored"] is True


def test_load_models_blends_prices_3_to_1(models_csv) -> None:
    models = load_models(models_csv)
    assert models["a"][BLENDED_PRICE] == pytest.approx(1.75)  # (3*1 + 4) / 4
    assert models["a"]["size"] is None and models["b"]["open"] is True


def frontier_models(frontier: dict) -> dict[str, list[str]]:
    return {f["axis"]: [p["model_slug"] for p in f["points"]] for f in frontier["frontiers"]}


def test_price_and_size_frontiers(models_csv) -> None:
    f = build_frontier(metrics(), is_higher_better=True, models_csv=models_csv)
    # price: b 0.125 (50), c 0.5 (60), a 1.75 (75)
    assert frontier_models(f) == {"price": ["b", "c", "a"], "size": ["b", "c"]}
    assert f["models_without_metadata"] == ["d"]
    assert {k: f[k] for k in ("benchmark_slug", "metric_slug", "experiment_slug")} == {
        "benchmark_slug": "CDPK-en",
        "metric_slug": "accuracy",
        "experiment_slug": "final-4-items",
    }
    assert [x["x"] for x in f["frontiers"]] == [BLENDED_PRICE, "size"]


def test_frontier_uses_overall_rows_of_a_metrics_csv(tmp_path, models_csv) -> None:
    path = tmp_path / "m.csv"
    maths = canonical_subset({"subject": "Maths"})
    metrics(metric_value_by_subset={maths: {"b": 100.0}}).to_csv(path, index=False)
    f = build_frontier(pd.read_csv(path), is_higher_better=True, models_csv=models_csv)
    assert frontier_models(f)["price"] == ["b", "c", "a"]


def test_lower_is_better_flips_the_frontier(models_csv) -> None:
    f = build_frontier(metrics(), is_higher_better=False, models_csv=models_csv)
    assert frontier_models(f) == {"price": ["b"], "size": ["b"]}
