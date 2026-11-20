#!/usr/bin/env python3
"""
Upload the result files written by write_result_files.py to GCS.

Each (benchmark, metric) goes to a dated folder, so earlier uploads are kept:

    gs://fab-benchmarks-hub/metrics/<benchmark>_<metric>/dt=YYYY-MM-DD/
        <benchmark>_<metric>.csv
        <benchmark>_<metric>_metadata.json
        <benchmark>_<metric>_frontier.json

e.g. metrics/CDPK-sw_accuracy/dt=2026-11-20/CDPK-sw_accuracy.csv. The date is
today's in UTC unless --date is given. Uploading twice on the same day
overwrites that day's files, with a warning for each one overwritten.

The benchmarks and metrics to upload are the ones in leaderboard_config.json, read
from results/<benchmark>/. Every file is checked to exist before anything
is uploaded, so a folder never ends up with a mix of new and old files.

Run from the repo root:

    uv run python scripts/upload_result_files.py --dry-run          # print, upload nothing
    uv run python scripts/upload_result_files.py                    # every language
    uv run python scripts/upload_result_files.py --benchmark CDPK-sw

A copy of LLM_benchmark's scripts/upload_result_files.py (branch
feat/results-schema-metrics-frontiers), itself a copy of qa-benchmarks-monorepo's LK
upload script, to be merged into one shared package later. Each language's GCS folder,
metrics/CDPK-<code>_accuracy/, is listed in leaderboard_config.json.

Credentials: Google's Application Default Credentials, i.e. either
`gcloud auth application-default login`, or GOOGLE_APPLICATION_CREDENTIALS set to a
service-account key file (in the environment or in .env). The account needs write
access to the bucket ("Storage Object Admin", to allow overwriting). With gcloud
credentials, also run once, to silence the "quota project" warning:

    gcloud auth application-default set-quota-project fab-benchmarks-485014
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cdpk.metrics import result_file_name

BUCKET = "fab-benchmarks-hub"
# Google Cloud project ID ("fab-benchmarks" is its display name), used by the storage
# client for quota and billing.
PROJECT_ID = "fab-benchmarks-485014"
# Path inside the bucket, from its root, under which the <benchmark>_<metric>/ folders
# go. May contain '/' (e.g. "metrics/v2"); no leading or trailing '/'.
BUCKET_FOLDER = "metrics"
CONFIG_FILE = ROOT / "leaderboard_config.json"
RESULTS_DIR = ROOT / "results"

# The files of one (benchmark, metric), by suffix, with their content type.
FILE_SUFFIXES = {
    ".csv": "text/csv",
    "_metadata.json": "application/json",
    "_frontier.json": "application/json",
}


def benchmark_metrics(config_path: Path) -> list[tuple[str, str]]:
    """(benchmark_slug, metric_slug) of every metric in the leaderboard config."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return [
        (benchmark["benchmark_domain"], metric["metric_slug"])
        for benchmark in config["benchmarks"]
        for metric in benchmark["metrics"]
    ]


def planned_uploads(
    pairs: Sequence[tuple[str, str]],
    results_dir: Path,
    date: str,
    bucket_folder: str = BUCKET_FOLDER,
) -> list[tuple[Path, str, str]]:
    """(local file, object name, content type) of every file to upload.

    Object names are <bucket_folder>/<benchmark>_<metric>/dt=<date>/<file>.
    Raises FileNotFoundError, listing every missing file, before anything is uploaded.
    """
    bucket_folder = bucket_folder.strip("/")
    uploads, missing = [], []
    for benchmark_slug, metric_slug in pairs:
        folder = f"{result_file_name(benchmark_slug, metric_slug, '')}/dt={date}"
        if bucket_folder:
            folder = f"{bucket_folder}/{folder}"
        for suffix, content_type in FILE_SUFFIXES.items():
            name = result_file_name(benchmark_slug, metric_slug, suffix)
            path = results_dir / benchmark_slug / name
            if not path.is_file():
                missing.append(str(path))
            uploads.append((path, f"{folder}/{name}", content_type))
    if missing:
        raise FileNotFoundError(
            "Missing result files (run scripts/write_result_files.py first):\n"
            + "\n".join(f"- {m}" for m in missing)
        )
    return uploads


def upload(uploads: Sequence[tuple[Path, str, str]], bucket) -> list[str]:
    """Upload every file to `bucket` (a google.cloud.storage Bucket).

    Returns the object names that already existed and were overwritten.
    """
    overwritten = []
    for path, object_name, content_type in uploads:
        blob = bucket.blob(object_name)
        if blob.exists():
            overwritten.append(object_name)
            print(f"WARNING: overwriting gs://{bucket.name}/{object_name}")
        blob.upload_from_filename(str(path), content_type=content_type)
        print(f"{path} -> gs://{bucket.name}/{object_name}")
    return overwritten


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        help="only upload this benchmark (repeatable; default: all in the config)",
    )
    parser.add_argument(
        "--date",
        type=dt.date.fromisoformat,
        default=None,
        help="date of the dt= folder, YYYY-MM-DD (default: today in UTC)",
    )
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument(
        "--bucket-folder",
        default=BUCKET_FOLDER,
        help=f"path inside the bucket to upload under (default: {BUCKET_FOLDER})",
    )
    parser.add_argument("--project-id", default=PROJECT_ID, help="Google Cloud project ID")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="print, upload nothing")
    args = parser.parse_args(argv)
    date = (args.date or dt.datetime.now(dt.timezone.utc).date()).isoformat()

    pairs = benchmark_metrics(args.config)
    if args.benchmark:
        unknown = set(args.benchmark) - {b for b, _ in pairs}
        if unknown:
            parser.error(f"--benchmark: not in {args.config.name}: {', '.join(sorted(unknown))}")
        pairs = [(b, m) for b, m in pairs if b in args.benchmark]
    uploads = planned_uploads(pairs, args.results_dir, date, args.bucket_folder)

    if args.dry_run:
        for path, object_name, _ in uploads:
            print(f"{path} -> gs://{args.bucket}/{object_name}")
        print(f"dry run: {len(uploads)} files, nothing uploaded")
        return

    from dotenv import load_dotenv
    from google.cloud import storage

    load_dotenv(ROOT / ".env")
    bucket = storage.Client(project=args.project_id).bucket(args.bucket)
    overwritten = upload(uploads, bucket)
    print(f"uploaded {len(uploads)} files to gs://{args.bucket}/{args.bucket_folder.strip('/')}/")
    if overwritten:
        print(f"WARNING: {len(overwritten)} files already existed and were overwritten")


if __name__ == "__main__":
    main()
