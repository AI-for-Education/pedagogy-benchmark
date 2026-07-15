#!/usr/bin/env python3
"""
format_data_web.py

Generate multilingual leaderboard JSON files consumed by the fab-ai.org web
leaderboard.

Inputs:
  fab-benchmarks-configs/models.csv    – model metadata (price, size, open, urls…)
  fab-benchmarks-configs/providers.csv – provider colors + SVG logos
  data/results/cdpk_multilingual_model_performance.csv – benchmark accuracy data

Outputs (data/web/, where <env> is dev/beta/prod — see --upload):
  fabai_<env>_leaderboard_configmeta_multilingual.json
  fabai_<env>_leaderboard_providermeta_multilingual.json
  fabai_<env>_leaderboard_data_multilingual.json
  fabai_<env>_leaderboard_data_costfrontier_multilingual.json
  fabai_<env>_leaderboard_data_sizefrontier_multilingual.json

Usage:
  python scripts/format_data_web.py                 # generate locally only (env=dev)
  python scripts/format_data_web.py --upload dev    # ...and upload to Azure dev
  python scripts/format_data_web.py --upload beta   # ...and upload to Azure beta
  python scripts/format_data_web.py --upload prod   # ...and upload to Azure prod

--upload dev|beta|prod publishes the generated files to that environment on the
Azure blob container (requires the AZURE_STORAGE_KEY env var). Without it, files
are written locally only. The chosen env is also the fabai_<env>_ filename prefix.
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from cloudpathlib import AzureBlobClient, CloudPath

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
MODELS_CSV    = ROOT / "fab-benchmarks-configs" / "models.csv"
PROVIDERS_CSV = ROOT / "fab-benchmarks-configs" / "providers.csv"
BENCH_CSV     = ROOT / "data" / "results" / "cdpk_multilingual_model_performance.csv"
OUT_DIR       = ROOT / "data" / "web"
OUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(override=True)

# ── Azure upload target (CLI) ─────────────────────────────────────────────────
# Default: generate locally only (no upload). Pass --upload dev|beta|prod to also
# publish the generated files to that environment on the Azure container.
_parser = argparse.ArgumentParser(description="Generate multilingual leaderboard JSON files.")
_parser.add_argument(
    "--upload",
    choices=("dev", "beta", "prod"),
    default=None,
    help="upload the generated files to this Azure environment (default: no upload)",
)
UPLOAD_TO = _parser.parse_args().upload
LOCAL_ENV = UPLOAD_TO or "dev"     # local filename prefix; blob uses the same name

AZURE_ACCOUNT_URL = "https://fabdatastorage.blob.core.windows.net"
AZURE_CONTAINER   = "az://ai-for-ed-public/"

container = None
if UPLOAD_TO:
    key = os.getenv("AZURE_STORAGE_KEY")
    if not key:
        raise SystemExit("AZURE_STORAGE_KEY env var is required to --upload.")
    client = AzureBlobClient(AZURE_ACCOUNT_URL, credential=key)
    container = CloudPath(AZURE_CONTAINER, client=client)

# ── Language config ───────────────────────────────────────────────────────────
LANG_IDS = ["EN", "AR", "SW", "HA", "PS", "YO", "DA", "LG", "NY"]

LANG_DISPLAY = {
    "EN": "English", "AR": "Arabic",  "SW": "Kiswahili", "HA": "Hausa",
    "PS": "Pashto",  "YO": "Yoruba",  "DA": "Dari",      "LG": "Luganda",
    "NY": "Nyankore",
}

# NB: keys here must match the raw language names in the benchmark results CSV
# (which still says "Swahili"); the display relabel to "Kiswahili" lives in
# LANG_DISPLAY only.
LANG_FROM_CSV = {
    "English":  "EN", "Arabic":  "AR", "Swahili": "SW", "Hausa": "HA",
    "Pashto":   "PS", "Yoruba":  "YO", "Dari":    "DA", "Luganda": "LG",
    "Nyankore": "NY",
}


# ── 1. Read providers.csv ─────────────────────────────────────────────────────
# Columns: provider, color, svg
providers_db: dict[str, dict] = {}  # name → {color, svg}
with open(PROVIDERS_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        name = row["provider"].strip()
        providers_db[name] = {
            "color": row["color"].strip(),
            "svg":   row["svg"].strip(),
        }


# ── 2. Read models.csv ────────────────────────────────────────────────────────
# Columns: model_id, provider, open, url, release_date, size, size_active,
#           input_cost, output_cost, display_name, is_reasoning, is_vision
def _float_or_none(s: str):
    s = s.strip()
    return float(s) if s else None

models_db: dict[str, dict] = {}         # model_id → metadata
models_row_index: dict[str, int] = {}   # model_id → 0-based row number in models.csv
with open(MODELS_CSV, newline="", encoding="utf-8") as f:
    for i, row in enumerate(csv.DictReader(f)):
        mid = row["model_id"].strip()
        models_row_index[mid] = i        # positional ID, mirrors merge_datasets.py
        inp  = _float_or_none(row.get("input_cost",  ""))
        out  = _float_or_none(row.get("output_cost", ""))
        size = _float_or_none(row.get("size", ""))
        models_db[mid] = {
            "provider":     row["provider"].strip(),
            "open":         row["open"].strip().lower() == "true",
            "url":          row.get("url", "").strip() or None,
            "release_date": row.get("release_date", "").strip() or None,
            "size":         size,                           # total params (actual count)
            "input_cost":   inp,
            "output_cost":  out,
            # blended = 3:1 input:output weighted average
            "blended_price": round((inp * 3 + out) / 4, 6)
                             if inp is not None and out is not None else None,
            "display_name": row.get("display_name", "").strip() or mid,
        }


# ── 3. Read benchmark CSV (Overall category only) ────────────────────────────
bench_model_provider: dict[str, str] = {}   # model_name → provider (from bench CSV)
data_map: dict[tuple, dict] = {}            # (model, lang_id) → {accuracy, bad_format}

with open(BENCH_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["category"].strip() != "Overall":
            continue
        model    = row["model"].strip()
        lang_id  = LANG_FROM_CSV.get(row["language"].strip())
        provider = row["provider"].strip().rstrip("\r")
        if lang_id is None:
            continue
        bench_model_provider.setdefault(model, provider)
        data_map[(model, lang_id)] = {
            "accuracy":   round(float(row["accuracy"]),   2),
            "bad_format": round(float(row["bad_format"]), 4),
        }

# All model names that appear in the benchmark data
bench_model_names: list[str] = sorted(bench_model_provider.keys())


# ── Keep benchmark models present in models.csv, assign positional IDs ────────
# ID = row index in models.csv / providers.csv, mirroring core (merge_datasets.py).
unresolved = [n for n in bench_model_names if n not in models_row_index]
if unresolved:
    print("WARNING: no models.csv match — dropped:", ", ".join(unresolved))
bench_model_names = [n for n in bench_model_names if n in models_row_index]

def model_meta(model_name: str) -> dict:
    return models_db.get(model_name, {})

def get_provider(model_name: str) -> str:
    return model_meta(model_name).get("provider") or bench_model_provider.get(model_name, "Unknown")

# Positional IDs: row index in providers.csv / models.csv
provider_id_map: dict[str, str] = {n: str(i) for i, n in enumerate(providers_db)}
for j, p in enumerate(sorted({get_provider(m) for m in bench_model_names} - set(provider_id_map))):
    provider_id_map[p] = str(len(providers_db) + j)
all_provider_names = list(provider_id_map)

model_id_map: dict[str, str] = {n: str(models_row_index[n]) for n in bench_model_names}


# ── File 2 — leaderboard_providermeta ────────────────────────────────────────
provider_models: dict[str, dict] = defaultdict(dict)

for model_name in bench_model_names:
    provider_name = get_provider(model_name)
    pid  = provider_id_map[provider_name]
    mid  = model_id_map[model_name]
    meta = model_meta(model_name)

    provider_models[pid][mid] = {
        "id":            mid,
        "name":          model_name,
        "displayName":   meta.get("display_name", model_name),
        "input_price":   meta.get("input_cost"),
        "output_price":  meta.get("output_cost"),
        "blended_price": meta.get("blended_price"),
        "size":          meta.get("size"),
        "open":          meta.get("open", False),
        "website":       meta.get("url"),
        "release_date":  meta.get("release_date"),
    }

PROVIDER_META: dict[str, dict] = {}
for provider_name in all_provider_names:
    pid = provider_id_map[provider_name]
    cfg = providers_db.get(provider_name, {"color": "#888888", "svg": ""})
    PROVIDER_META[pid] = {
        "metadata": {
            "id":          pid,
            "name":        provider_name,
            "displayName": provider_name,
            "color":       cfg["color"],
            "logo":        cfg["svg"],
        },
        "models": provider_models.get(pid, {}),
    }


# ── File 3 — leaderboard_data ─────────────────────────────────────────────────
DATA: list[dict] = []
for model_name in bench_model_names:
    mid = model_id_map[model_name]
    pid = provider_id_map[get_provider(model_name)]
    for lang_id in LANG_IDS:
        entry = data_map.get((model_name, lang_id))
        if entry is None:
            continue
        DATA.append({
            "benchmark_domain": lang_id,
            "modelId":         mid,
            "providerId":      pid,
            "accuracy":        entry["accuracy"],
            "bad_format":      entry["bad_format"],
            "n":               None,
            "categories":      [],
        })


# ── Default-model selection config (ported from core frontier_analysis.py) ────
# Models always forced into the default list, regardless of frontier/family
# logic. Each must be a model_id in models.csv with benchmark data.
MANUAL_DEFAULT_MODELS = [
    "claude-fable-5",
]

# TEMPORARY (2026-07): when True, the provider/family dedup also drops older
# same-family models that sit on a cost/size frontier — i.e. it relaxes
# frontier-protection to cut per-provider repetition (esp. Google). Trade-off:
# some drawn frontier lines will have no datapoint on the default view.
# Set to False to restore core behaviour (keep frontier-optimal models even if
# it means repeating a provider/family).
DROP_FRONTIER_PROTECTED_DUPES = True

# Per-provider family filters for default-list deduplication. Values are ordered
# regex lists (re.search), most specific first. A model is assigned to the first
# pattern it matches. Per (provider, family): always keep the most recent model
# (by release_date); if >= 2 of a family are present, drop the older ones unless
# they sit on a raw (non-smooth) Pareto frontier.
PROVIDER_FAMILY_FILTERS = {
    "Anthropic": [r"claude.*-sonnet", r"claude.*-opus", r"claude.*-haiku"],
    "Google":    [r"gemini.*-flash-lite", r"gemini.*-flash", r"gemini.*-pro"],
    "OpenAI": [
        r"gpt-5.*-(nano|luna)(-|$)", # nano tier: "luna" is OpenAI's post-rename name for nano
        r"gpt-5.*-(mini|terra)(-|$)", # mini tier: "terra" is OpenAI's post-rename name for mini
        r"gpt-5",
        r"^o.*mini",
        r"^o",
    ],
    "xAI":      [r"grok.*fast", r"grok"],
    "AI21labs": [r"jamba.*large", r"jamba.*mini"],
    "Z.ai":     [r"glm.*v", r"glm"],
    "Alibaba": [
        r"qwen.*flash",
        r"qwen.*plus",
        r"qwen.*max",
        r"qwen.*-a3b",
        r"qwen.*-a\d{2,}b",
        r"qwen",
    ],
    "DeepSeek-AI": [r"deepseek.*-r", r"deepseek"],
    "Mistral": [
        r"magistral",
        r"ministral",
        r"mistral.*small",
        r"mistral.*medium",
        r"mistral.*large",
    ],
    "Meta":   [r"llama-4|llama.*-\d{3}b", r"llama.*-\d{2}b", r"llama"],
    "Xiaomi": [r"mimo.*pro", r"mimo"],
}


# ── Frontier + default-list functions (ported from core frontier_analysis.py) ─
def compute_frontier(data, acc_col="accuracy", sorting_col="cost", smooth=None):
    """
    Pareto frontier (running-max accuracy vs sorting_col), optionally
    concave-smoothed in log-x space.

    The DRAWN frontier lines call this with smooth=False (raw Pareto) so the
    rendered step function never sits below a real model. The default-model
    layer peeling (compute_layers) calls it with smooth=True.
    """
    data = data.sort_values(sorting_col, ascending=True)
    frontier, best = [], -1e9
    for _, row in data.iterrows():
        if row[acc_col] > best:
            frontier.append(row)
            best = row[acc_col]

    if not frontier:
        return pd.DataFrame(columns=data.columns)

    df_return = pd.DataFrame(frontier)
    # if rows share a cost/size, keep the one with highest accuracy
    df_return = df_return.sort_values(sorting_col).drop_duplicates(sorting_col, keep="last")

    if not smooth or len(df_return) < 3:
        return df_return.reset_index(drop=True)

    # Enforce concavity in log-x space (slope must be non-increasing L→R)
    df_return["log_var"] = np.log(df_return[sorting_col].astype(float) + 1e-9)
    points = list(zip(df_return.index, df_return["log_var"], df_return[acc_col]))
    concave_indices = [points[0][0], points[1][0]]
    for i in range(2, len(points)):
        p_new_idx, x_new, y_new = points[i]
        while len(concave_indices) >= 2:
            p_mid_idx, p_prev_idx = concave_indices[-1], concave_indices[-2]
            x_mid, y_mid   = df_return.loc[p_mid_idx][["log_var", acc_col]]
            x_prev, y_prev = df_return.loc[p_prev_idx][["log_var", acc_col]]
            slope1 = (y_mid - y_prev) / (x_mid - x_prev) if (x_mid - x_prev) != 0 else np.inf
            slope2 = (y_new - y_mid) / (x_new - x_mid) if (x_new - x_mid) != 0 else np.inf
            if slope2 > slope1:            # convex bump → discard mid
                concave_indices.pop()
            else:
                break
        concave_indices.append(p_new_idx)

    df_return = df_return.loc[concave_indices].drop(columns=["log_var"])
    return df_return.reset_index(drop=True)


def compute_layers(df, frontier_var="cost", acc_col="accuracy", smooth=True, max_models=None):
    """Successive concave frontier layers ("peeling"): compute the frontier,
    remove those points, repeat until no >=3-point concave frontier remains."""
    all_frontiers = []
    remaining_df = df.copy().reset_index(drop=True)
    max_iterations = len(df) + 1
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            break
        frontier = compute_frontier(remaining_df, acc_col=acc_col, sorting_col=frontier_var, smooth=smooth)
        if len(frontier) < 3:
            break
        all_frontiers.append(frontier)

        key_cols = [frontier_var, acc_col]
        temp_frontier = frontier[key_cols].copy()
        temp_frontier["_is_frontier"] = True
        merged = remaining_df.merge(temp_frontier, on=key_cols, how="left")
        remaining_df = merged[merged["_is_frontier"].isnull()].drop(columns=["_is_frontier"])
        if remaining_df.empty:
            break

    if not all_frontiers:
        frontier = compute_frontier(remaining_df, acc_col=acc_col, sorting_col=frontier_var, smooth=smooth)
        all_frontiers.append(frontier)
        return all_frontiers

    max_models_collected = sum(len(layer) for layer in all_frontiers)
    reduction_iteration_count = 0
    max_reduction_iterations = len(all_frontiers) + 1
    while max_models is not None and max_models_collected > max_models:
        reduction_iteration_count += 1
        if reduction_iteration_count > max_reduction_iterations:
            break
        if len(all_frontiers) <= 1:
            break
        indices = np.arange(0, len(all_frontiers), 2)
        all_frontiers = [all_frontiers[i] for i in indices]
        max_models_collected = sum(len(layer) for layer in all_frontiers)

    return all_frontiers


def create_default_models_list(
    benchmark_dfs,
    cost_var="cost",
    max_models=10,
    add_missing_models=False,
    filter_by_provider_family=False,
    drop_frontier_protected_dupes=False,
):
    """Frontier-derived default list: union of concave frontier layers (cost
    open+closed, size open-only), plus raw-Pareto frontier models that were
    missed (add_missing_models), minus provider/family duplicates
    (filter_by_provider_family), plus MANUAL_DEFAULT_MODELS.

    drop_frontier_protected_dupes: if True, the provider/family dedup drops older
    same-family models even when they sit on a raw frontier (relaxes frontier
    protection). If False (default), frontier models are kept — core behaviour.

    Returns (selected_models, {benchmark: [models present in that benchmark]})."""
    selected_models_set = set()
    non_smooth_frontier_set = set()   # raw-Pareto frontier models

    for benchmark_name, df in benchmark_dfs.items():
        if "open" not in df.columns or "model" not in df.columns:
            continue
        for open_status in [True, False]:
            subset_df = df[df["open"] == open_status].copy()
            if subset_df.empty:
                continue

            if cost_var in subset_df.columns:
                for layer_df in compute_layers(subset_df, frontier_var=cost_var,
                                               acc_col="accuracy", smooth=True,
                                               max_models=max_models):
                    selected_models_set.update(layer_df["model"].tolist())
                cost_frontier = compute_frontier(subset_df, acc_col="accuracy",
                                                 sorting_col=cost_var, smooth=False)
                non_smooth_frontier_set.update(cost_frontier["model"].tolist())

            if open_status and "size" in subset_df.columns:
                for layer_df in compute_layers(subset_df, frontier_var="size",
                                               acc_col="accuracy", smooth=True,
                                               max_models=max_models):
                    selected_models_set.update(layer_df["model"].tolist())
                size_frontier = compute_frontier(subset_df, acc_col="accuracy",
                                                 sorting_col="size", smooth=False)
                non_smooth_frontier_set.update(size_frontier["model"].tolist())

    selected_models = sorted(selected_models_set)
    print(f"  default-list: {len(selected_models)} models on all frontier layers.")

    if add_missing_models:
        missing = [m for m in non_smooth_frontier_set if m not in selected_models]
        if missing:
            print(f"  default-list: adding {len(missing)} missing raw-frontier models.")
            selected_models = sorted(set(selected_models) | set(missing))

    if filter_by_provider_family:
        models_df = pd.read_csv(MODELS_CSV)
        models_df["release_date_parsed"] = pd.to_datetime(models_df["release_date"], format="%m/%Y")
        selected_models_set = set(selected_models)

        models_with_data = set()
        for df in benchmark_dfs.values():
            models_with_data.update(df["model"].values)

        for provider, families in PROVIDER_FAMILY_FILTERS.items():
            provider_df = models_df[models_df["provider"] == provider]

            def _first_match(model_id):
                return next((fk for fk in families if re.search(fk, model_id)), None)

            for family_keyword in families:
                family_df = provider_df[
                    provider_df["model_id"].apply(lambda m: _first_match(m) == family_keyword)
                ].sort_values("release_date_parsed")
                if family_df.empty:
                    continue

                family_with_data = family_df[family_df["model_id"].isin(models_with_data)]
                most_recent = (family_with_data.iloc[-1]["model_id"]
                               if not family_with_data.empty
                               else family_df.iloc[-1]["model_id"])
                family_model_ids = family_df["model_id"].tolist()

                if most_recent not in selected_models_set:
                    print(f"  default-list: adding recent {provider}/{family_keyword}: {most_recent}")
                    selected_models_set.add(most_recent)

                present = [m for m in family_model_ids if m in selected_models_set]
                if len(present) >= 2:
                    to_remove = [
                        m for m in present
                        if m != most_recent
                        and (drop_frontier_protected_dupes or m not in non_smooth_frontier_set)
                    ]
                    if to_remove:
                        print(f"  default-list: {provider}/{family_keyword} dropping {to_remove}")
                        selected_models_set -= set(to_remove)

        selected_models = sorted(selected_models_set)
        print(f"  default-list: {len(selected_models)} models after family dedup.")

    if MANUAL_DEFAULT_MODELS:
        models_with_data = set()
        for df in benchmark_dfs.values():
            models_with_data.update(df["model"].values)
        for model in MANUAL_DEFAULT_MODELS:
            if model not in models_with_data:
                print(f"  WARNING: manual default '{model}' has no benchmark data.")
        selected_models = sorted(set(selected_models) | set(MANUAL_DEFAULT_MODELS))

    benchmark_default_models_dict = {
        name: [m for m in selected_models if m in df["model"].values]
        for name, df in benchmark_dfs.items()
    }
    return selected_models, benchmark_default_models_dict


def step_segments(frontier: list[dict]) -> list[dict]:
    """
    Convert frontier points to step-line segments for the web chart renderer.
    Before each new point, insert a horizontal segment at the previous y-value
    so the chart draws a staircase rather than diagonal lines.
    """
    segs: list[dict] = []
    for i, p in enumerate(frontier):
        if i > 0:
            segs.append({
                "x":          p["x"],
                "y":          round(frontier[i - 1]["y"], 3),
                "providerId": p["providerId"],
            })
        segs.append({
            "x":          p["x"],
            "y":          round(p["y"], 3),
            "providerId": p["providerId"],
        })
    return segs


# ── Per-language benchmark DataFrames (one "benchmark" per language) ──────────
_rows = []
for model_name in bench_model_names:
    meta = model_meta(model_name)
    prov = get_provider(model_name)
    for lang_id in LANG_IDS:
        entry = data_map.get((model_name, lang_id))
        if entry is None:
            continue
        _rows.append({
            "benchmark":    lang_id,
            "model":        model_name,
            "accuracy":     entry["accuracy"],
            "cost":         meta.get("blended_price"),
            "size":         meta.get("size"),
            "open":         bool(meta.get("open", False)),
            "provider":     prov,
            "release_date": meta.get("release_date"),
            "modelId":      model_id_map[model_name],
            "providerId":   provider_id_map[prov],
        })
BENCH_DF = pd.DataFrame(_rows)
benchmark_dfs = {lid: BENCH_DF[BENCH_DF["benchmark"] == lid].copy() for lid in LANG_IDS}


# ── File 1 — leaderboard_configmeta ──────────────────────────────────────────
# default_models via the full core pipeline: frontier layers + missing raw
# frontier models + provider/family dedup + manual list.
_, benchmark_default_models = create_default_models_list(
    benchmark_dfs,
    cost_var="cost",
    max_models=10,
    add_missing_models=True,
    filter_by_provider_family=True,
    drop_frontier_protected_dupes=DROP_FRONTIER_PROTECTED_DUPES,
)

CONFIG_META = {
    "base_options": {
        "benchmarks": [{"id": lid, "displayName": LANG_DISPLAY[lid]} for lid in LANG_IDS],
        "axes": {"x": "blended_price", "y": "accuracy"},
    },
    "benchmarks": {
        lid: {
            "displayName":     LANG_DISPLAY[lid],
            "categories":      [],
            "default_models":  [model_id_map[m] for m in benchmark_default_models.get(lid, [])],
            "disabled_models": [],
        }
        for lid in LANG_IDS
    },
    "axes": {
        "x": {"blended_price": "log", "size": "log"},
        "y": {"accuracy": "linear"},
    },
}


# ── Files 4 & 5 — cost / size frontier lines ─────────────────────────────────
# Raw Pareto (smooth=False), matching core: no model can sit above the step.
# Cost frontier = all models; size frontier = open-source models only.
def frontier_line_segments(df_subset, sorting_col):
    fr = compute_frontier(df_subset, acc_col="accuracy", sorting_col=sorting_col, smooth=False)
    pts = [
        {
            "x":          float(row[sorting_col]),
            "y":          float(row["accuracy"]),
            "providerId": row["providerId"],
        }
        for _, row in fr.iterrows()
    ]
    return step_segments(pts)

COST_FRONTIER = []
for lang_id in LANG_IDS:
    df = benchmark_dfs[lang_id]
    sub = df[df["cost"].notnull() & (df["cost"] > 0)]
    COST_FRONTIER.append({
        "benchmark_domain": lang_id,
        "lineSegments":     frontier_line_segments(sub, "cost"),
        "categories":       [],
    })

SIZE_FRONTIER = []
for lang_id in LANG_IDS:
    df = benchmark_dfs[lang_id]
    sub = df[df["open"] & df["size"].notnull()]
    segs = frontier_line_segments(sub, "size")
    if not segs:
        continue
    SIZE_FRONTIER.append({
        "benchmark_domain": lang_id,
        "lineSegments":     segs,
        "categories":       [],
    })


# ── Write outputs ─────────────────────────────────────────────────────────────
# Base names; the env prefix (fabai_<LOCAL_ENV>_) is added on write + upload.
OUTPUTS = {
    "leaderboard_configmeta_multilingual.json":        CONFIG_META,
    "leaderboard_providermeta_multilingual.json":      PROVIDER_META,
    "leaderboard_data_multilingual.json":              DATA,
    "leaderboard_data_costfrontier_multilingual.json": COST_FRONTIER,
    "leaderboard_data_sizefrontier_multilingual.json": SIZE_FRONTIER,
}

written_paths = []
for base, payload in OUTPUTS.items():
    out_path = OUT_DIR / f"fabai_{LOCAL_ENV}_{base}"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    written_paths.append(out_path)
    size_kb = out_path.stat().st_size // 1024
    print(f"  {out_path.relative_to(ROOT)}  ({size_kb} KB)")

cost_segs = sum(len(e["lineSegments"]) for e in COST_FRONTIER)
size_segs = sum(len(e["lineSegments"]) for e in SIZE_FRONTIER)
print(
    f"\nDone. {len(all_provider_names)} providers | "
    f"{len(bench_model_names)} models | "
    f"{len(DATA)} data records | "
    f"{cost_segs} cost-frontier segs | "
    f"{size_segs} size-frontier segs"
)


# ── Upload to Azure Blob Storage (only when --upload was passed) ──────────────
if container is not None:
    print(f"\nUploading to {AZURE_CONTAINER} ({UPLOAD_TO})...")
    for out_path in written_paths:
        blob = container / out_path.name
        if blob.exists():                       # back up the current blob first
            blob.download_to(OUT_DIR / f"backup_{out_path.name}")
        blob.upload_from(out_path)
        print(f"  uploaded {out_path.name}")
else:
    print("\nNo --upload target - generated locally only (no Azure upload).")
