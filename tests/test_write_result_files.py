"""The multilingual CDPK script on a tiny hand-made benchmark (no real data needed).

Ported from LLM_benchmark's tests/test_write_result_files.py, with this repo's
column names and category names ("General" stays "General").
"""

import dataclasses
import json

import pandas as pd
import pytest

from conftest import ROOT, load_script
from cdpk.benchmark_constants import CACHE_GLOBAL_DIR, CACHE_LOCAL_DIR
from cdpk.metrics import SLUG_PATTERN

wrf = load_script("write_result_files")

QUESTIONS = pd.DataFrame(
    {
        "Source": ["s"] * 6,
        "question": ["ex1", "q1", "q2", "ex2", "ex1", "q4"],
        "answer_a": ["x"] * 6,
        "answer_b": ["y"] * 6,
        "correct_answer": ["A", "A", "B", "A", "A", "B"],
        "category": ["Maths"] * 3 + ["General"] * 3,
        "age_group": ["All", "Primary", "Secondary", "All", "All", "Pre-primary"],
    }
)
# Row 0 of each config is its few-shot example. Config c2's first test item is a copy
# of c1's example ex1, so it is dropped: the benchmark has 3 items (q1, q2, q4).
CONFIG_ROWS = {"c1": [0, 1, 2], "c2": [3, 4, 5]}
MODELS = {"good": "Good", "bad": "Bad", "notrun": "Not run"}
CACHE = {  # raw responses for the test rows of each config
    "good": {"c1": ["A", "B"], "c2": ["A", "B"]},
    "bad": {"c1": ["A", "I think C"], "c2": ["B", "A"]},
    "unlisted": {"c1": ["A", "B"], "c2": ["A", "B"]},
}
METRIC = {
    "metric_slug": "accuracy",
    "display_name": "Accuracy",
    "display_description": "d",
    "unit": "%",
    "is_higher_better": True,
    "is_bad_format_scored": True,
}


def write_config(
    path, metrics=(METRIC,), is_bad_format_scored=True, subsets=("subject", "level")
):
    metrics = [{**m, "is_bad_format_scored": is_bad_format_scored} for m in metrics]
    benchmark = {
        "benchmark_domain": "TEST",
        "display_name": "Test",
        "benchmark_subsets": list(subsets),
        "metrics": metrics,
    }
    config = {"cluster_name": "multilingual-pedagogy", "benchmarks": [benchmark]}
    path.write_text(json.dumps(config))


@pytest.fixture
def env(tmp_path, monkeypatch):
    def fake_load_from_config(questions_config, models_config):
        df = QUESTIONS.iloc[CONFIG_ROWS[questions_config]].reset_index(drop=True)
        return df, {"example_rows": [0]}, MODELS

    for model, configs in CACHE.items():
        for qc, resps in configs.items():
            d = tmp_path / "cache" / qc
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"answers": ["?"] * len(resps), "resps": resps}).to_csv(
                d / f"resps_{model}.csv", index=False
            )
    pd.DataFrame(
        {
            "model_id": ["good", "bad"],
            "open": [True, False],
            "size": [1e9, None],
            "input_price_usd_per_mtok": [1.0, 0.1],
            "output_price_usd_per_mtok": [1.0, 0.1],
        }
    ).to_csv(tmp_path / "models.csv", index=False)
    write_config(tmp_path / "config.json")
    monkeypatch.setattr(wrf, "load_from_config", fake_load_from_config)
    monkeypatch.setattr(wrf, "CACHE_DIR", tmp_path / "cache")

    cfg = wrf.BenchmarkConfig(
        benchmark_slug="TEST",
        experiment_slug="final-3-items",
        question_configs=("c1", "c2"),
        models_config="m",
        expected_items=3,
    )
    return cfg, tmp_path


def run(cfg, tmp):
    return wrf.write_result_files(
        cfg,
        models_csv=tmp / "models.csv",
        config_path=tmp / "config.json",
        out_root=tmp / "results",
    )


def read_metrics(tmp):
    path = tmp / "results" / "TEST" / "TEST_accuracy.csv"
    return pd.read_csv(path, keep_default_na=False, na_values={"metric_value": [""]})


def test_writes_the_three_files(env):
    cfg, tmp = env
    paths = run(cfg, tmp)
    assert [p.name for p in paths] == [
        "TEST_accuracy.csv",
        "TEST_accuracy_metadata.json",
        "TEST_accuracy_frontiers.json",
    ]
    assert json.loads(paths[1].read_text()) == {
        "benchmark_slug": "TEST",
        "benchmark_subsets": ["subject", "level"],
        **METRIC,
    }
    frontier = json.loads(paths[2].read_text())
    assert frontier["experiment_slug"] == "final-3-items"
    # overall, then each subject, then each level, each with price and size
    assert [(f["benchmark_subset"], f["axis"]) for f in frontier["frontiers"]][::2] == [
        ({}, "price"),
        ({"subject": "General"}, "price"),
        ({"subject": "Maths"}, "price"),
        ({"level": "Pre-primary"}, "price"),
        ({"level": "Primary"}, "price"),
        ({"level": "Secondary"}, "price"),
    ]
    assert [f["axis"] for f in frontier["frontiers"]][1::2] == ["size"] * 6


def test_config_subsets_choose_the_rows_and_frontiers(env):
    cfg, tmp = env
    write_config(tmp / "config.json", subsets=("level",))
    paths = run(cfg, tmp)
    assert set(read_metrics(tmp)["benchmark_subset"]) == {
        "",
        '{"level":"Pre-primary"}',
        '{"level":"Primary"}',
        '{"level":"Secondary"}',
    }
    assert json.loads(paths[1].read_text())["benchmark_subsets"] == ["level"]
    write_config(tmp / "config.json", subsets=())
    paths = run(cfg, tmp)
    assert set(read_metrics(tmp)["benchmark_subset"]) == {""}
    frontier = json.loads(paths[2].read_text())
    assert [f["benchmark_subset"] for f in frontier["frontiers"]] == [{}, {}]


def test_unknown_subset_key_is_an_error(env):
    cfg, tmp = env
    write_config(tmp / "config.json", subsets=("country",))
    with pytest.raises(ValueError, match="unknown key"):
        run(cfg, tmp)


def test_metrics_come_from_the_cache_for_listed_models(env, capsys):
    cfg, tmp = env
    run(cfg, tmp)
    m = read_metrics(tmp).set_index(["model_slug", "benchmark_subset"])
    assert m.loc[("good", ""), "metric_value"] == pytest.approx(100.0)
    # bad: q1 right, q2 unparseable (bad format counts as wrong), q4 wrong
    assert m.loc[("bad", ""), "metric_value"] == pytest.approx(33.33)
    # subjects keep the dataset's names; q1 is Primary, q2 Secondary, q4 Pre-primary
    assert m.loc[("bad", '{"subject":"Maths"}'), "metric_value"] == pytest.approx(50.0)
    assert m.loc[("bad", '{"subject":"General"}'), "metric_value"] == pytest.approx(0.0)
    assert m.loc[("bad", '{"level":"Primary"}'), "metric_value"] == pytest.approx(100.0)
    assert m.loc[("bad", '{"level":"Secondary"}'), "metric_value"] == pytest.approx(0.0)
    assert m.loc[("bad", '{"level":"Pre-primary"}'), "metric_value"] == pytest.approx(0.0)
    assert set(m.index.get_level_values(0)) == {"good", "bad"}
    assert (read_metrics(tmp)["n_items_scored"] == "").all()  # placeholder, empty
    out = capsys.readouterr().out
    assert "in m.yaml but not in the cache: notrun" in out
    assert "in the cache but not in m.yaml (left out): unlisted" in out


def test_config_flag_drives_the_published_accuracy(env):
    cfg, tmp = env
    write_config(tmp / "config.json", is_bad_format_scored=False)
    run(cfg, tmp)
    m = read_metrics(tmp).set_index(["model_slug", "benchmark_subset"])
    # bad format left out: q1 right, q4 wrong
    assert m.loc[("bad", ""), "metric_value"] == pytest.approx(50.0)


def test_metric_missing_from_the_config_is_reported(env):
    cfg, tmp = env
    write_config(tmp / "config.json", metrics=())
    with pytest.raises(ValueError, match="defines no metric 'accuracy' for 'TEST'"):
        run(cfg, tmp)


def test_partial_cache_is_an_error(env):
    cfg, tmp = env
    (tmp / "cache" / "c2" / "resps_good.csv").unlink()
    with pytest.raises(ValueError, match="no cache file"):
        run(cfg, tmp)


def test_unexpected_item_count_stops(env):
    cfg, _ = env
    with pytest.raises(ValueError, match="expected 4 items"):
        wrf.load_benchmark_items(dataclasses.replace(cfg, expected_items=4))


def test_level_rules_match_cdpk_en():
    assert wrf.is_primary("Primary") and wrf.is_primary("All")
    assert wrf.is_primary("Pre-primary, Primary")
    assert not wrf.is_primary("Pre-primary") and not wrf.is_primary(None)
    assert wrf.is_secondary("Primary, Secondary") and not wrf.is_secondary("Primary")
    assert wrf.is_pre_primary("Pre-primary, Primary") and wrf.is_pre_primary("All")
    assert not wrf.is_pre_primary("Primary")


# ---------------------------------------------------------- the repo's real setup


def test_reads_the_shared_cache_only():
    assert wrf.CACHE_DIR == CACHE_GLOBAL_DIR
    assert wrf.CACHE_DIR != CACHE_LOCAL_DIR


def test_benchmarks_match_the_leaderboard_config():
    config = json.loads((ROOT / "leaderboard_config.json").read_text(encoding="utf-8"))
    assert [b["benchmark_domain"] for b in config["benchmarks"]] == list(wrf.BENCHMARKS)
    assert "CDPK-en" not in wrf.BENCHMARKS  # published from LLM_benchmark for now


@pytest.mark.parametrize("cfg", wrf.BENCHMARKS.values(), ids=list(wrf.BENCHMARKS))
def test_each_language_has_valid_slugs_and_question_configs(cfg):
    assert SLUG_PATTERN.match(cfg.benchmark_slug)
    assert SLUG_PATTERN.match(cfg.experiment_slug)
    assert len(cfg.question_configs) == 7
    for qc in cfg.question_configs:
        assert "_ep_" in qc
        assert (ROOT / "configs" / "questions" / f"{qc}.yaml").is_file(), qc
    assert (ROOT / "configs" / "models" / f"{cfg.models_config}.yaml").is_file()
