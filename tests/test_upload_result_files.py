"""upload_result_files.py with a fake bucket (no network)."""

import datetime as dt
import json

import pytest

from conftest import load_script

urf = load_script("upload_result_files")

SUFFIXES = (".csv", "_metadata.json", "_frontiers.json")


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def exists(self):
        return self.name in self.bucket.objects

    def upload_from_filename(self, path, content_type):
        self.bucket.objects[self.name] = (open(path).read(), content_type)


class FakeBucket:
    name = "fab-benchmarks-hub"

    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return FakeBlob(self, name)


@pytest.fixture
def results(tmp_path):
    config = {
        "cluster_name": "c",
        "benchmarks": [
            {"benchmark_domain": b, "display_name": b, "metrics": [{"metric_slug": "accuracy"}]}
            for b in ("CDPK-ar", "CDPK-sw")
        ]
        + [{"benchmark_domain": "DA-en", "display_name": "DA", "metrics": []}],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    for b in ("CDPK-ar", "CDPK-sw"):
        (tmp_path / b).mkdir()
        for s in SUFFIXES:
            (tmp_path / b / f"{b}_accuracy{s}").write_text(f"{b}{s}")
    return tmp_path


def plan(tmp, date="2026-11-20"):
    return urf.planned_uploads(urf.benchmark_metrics(tmp / "config.json"), tmp, date)


def test_objects_go_to_a_dated_folder_per_benchmark_metric(results):
    names = [name for _, name, _ in plan(results)]
    assert names[:3] == [
        "metrics/CDPK-ar_accuracy/dt=2026-11-20/CDPK-ar_accuracy.csv",
        "metrics/CDPK-ar_accuracy/dt=2026-11-20/CDPK-ar_accuracy_metadata.json",
        "metrics/CDPK-ar_accuracy/dt=2026-11-20/CDPK-ar_accuracy_frontiers.json",
    ]
    assert len(names) == 6  # DA-en has no metric yet


def test_bucket_folder_sets_the_path_inside_the_bucket(results):
    pairs = urf.benchmark_metrics(results / "config.json")
    names = [n for _, n, _ in urf.planned_uploads(pairs, results, "2026-11-20", "/metrics/v2/")]
    assert names[0] == "metrics/v2/CDPK-ar_accuracy/dt=2026-11-20/CDPK-ar_accuracy.csv"
    names = [n for _, n, _ in urf.planned_uploads(pairs, results, "2026-11-20", "")]
    assert names[0] == "CDPK-ar_accuracy/dt=2026-11-20/CDPK-ar_accuracy.csv"


def test_missing_file_stops_before_any_upload(results):
    (results / "CDPK-sw" / "CDPK-sw_accuracy_frontiers.json").unlink()
    with pytest.raises(FileNotFoundError, match="CDPK-sw_accuracy_frontiers.json"):
        plan(results)


def test_second_upload_same_day_overwrites_with_a_warning(results, capsys):
    bucket = FakeBucket()
    assert urf.upload(plan(results), bucket) == []
    assert "WARNING" not in capsys.readouterr().out
    assert bucket.objects[
        "metrics/CDPK-sw_accuracy/dt=2026-11-20/CDPK-sw_accuracy.csv"
    ] == ("CDPK-sw.csv", "text/csv")

    overwritten = urf.upload(plan(results), bucket)
    assert len(overwritten) == 6
    assert "WARNING: overwriting gs://fab-benchmarks-hub/metrics/CDPK-ar_accuracy" in (
        capsys.readouterr().out
    )
    # another day is a new folder: nothing overwritten
    assert urf.upload(plan(results, "2026-11-21"), bucket) == []
    assert len(bucket.objects) == 12


def test_dry_run_uploads_nothing(results, capsys):
    urf.main(
        ["--dry-run", "--benchmark", "CDPK-sw", "--date", "2026-11-20",
         "--config", str(results / "config.json"), "--results-dir", str(results)]
    )
    out = capsys.readouterr().out
    assert "gs://fab-benchmarks-hub/metrics/CDPK-sw_accuracy/dt=2026-11-20/" in out
    assert "CDPK-ar" not in out
    assert "dry run: 3 files" in out


def test_default_date_is_today_in_utc(results, capsys):
    urf.main(["--dry-run", "--config", str(results / "config.json"),
              "--results-dir", str(results)])
    today_utc = dt.datetime.now(dt.timezone.utc).date().isoformat()
    assert f"dt={today_utc}" in capsys.readouterr().out
