"""
create_multilingual_comparison_plots.py

Builds a per-question (exploded) DataFrame directly from the raw
``cdpk_results_full_*.csv`` files in ``data/results/<lang>/`` (same logic as
``scripts/create_results.py``), restricted to ``_ep`` language folders plus
``English``, and produces the following figures (CLI flag → SVG):

  bar          → multilingual_default_models_barplot.svg
                 Per-language accuracy dots per default-list model, ordered
                 by English accuracy.
  steps        → best_accuracy_over_time_by_size.svg
                 Best-accuracy-to-date step plot, one subplot per language,
                 line per size bucket.
  steps_by_lang → best_accuracy_over_time_by_language.svg
                 Same data, transposed: one subplot per size bucket,
                 line per language.
  scatter      → accuracy_cost_value_frontier.svg
                 Accuracy vs blended cost scatter with temporal Pareto
                 frontiers, one subplot per language.
  hardware     → accessibility_frontier_by_hardware.svg
                 + accessibility_frontier_by_hardware_fabstyle.svg
                 FT-style accessibility frontier per hardware tier
                 (default + Fab-brand variant).
  tokens       → tokens_per_language_reasoning_models.svg
                 Stacked completion + reasoning median tokens per language,
                 one subplot per FOCUS_TOKEN_MODELS entry, with overhead
                 ratio vs English baseline.
  tokens_grouped → tokens_grouped_barplot_by_model.svg
                 Single-axes grouped+stacked barplot: one cluster of bars
                 per FOCUS_TOKEN_MODELS entry on the x-axis, one bar per
                 language within each cluster, stacked completion (solid)
                 + reasoning (hatched) sharing the language colour.
  latency      → latency_per_language_reasoning_models.svg
                 Median latency per language, same focus-model layout as
                 the tokens plot.
  beeswarm     → beeswarm_accuracy_per_language.svg
                 + beeswarm_accuracy_per_language_by_provider.svg
                 + beeswarm_accuracy_per_language_by_hardware.svg
                 One swarm per language, one dot per model; three colour
                 variants (by language, by provider, by hardware tier).
                 Median across models drawn as a black tick over each swarm.
                 Restricted to models with full language coverage.
  all          → all of the above.

Usage:
    python scripts/create_multilingual_comparison_plots.py
    python scripts/create_multilingual_comparison_plots.py --plots bar steps
    python scripts/create_multilingual_comparison_plots.py --out-dir /tmp/figs
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results"
CONFIGS_DIR = ROOT / "fab-benchmarks-configs"
FIGURES_DIR = RESULTS_DIR / "figures" / "web"

DEFAULT_MODELS_YAML = ROOT / "configs" / "models" / "full_list_default_models_20260218.yaml"

MODELS_TO_EXCLUDE = [
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-preview-09-2025",
    "gpt-5-2025-08-07-medium",
    "fw-deepseek-r1-0528",
]

# Re-deployed models keep the same weights as an "original" entry in
# models.csv but get a fresh model_id when redeployed. To avoid duplicating
# rows in models.csv, list the alias here and the script will reuse the
# original's metadata (release_date, size, costs, provider, ...) at lookup.
MODEL_ALIASES = {
    # new_id (in benchmark results) -> original_id (in models.csv)
    "llama-3.1-8b-instruct":         "fw-llama-v3p1-8b-instruct",
    "llama-3.2-11b-vision-instruct": "fw-llama-v3p2-11b-vision-instruct",
    "qwen-2.5-7b-instruct":          "fw-qwen2p5-7b-instruct",
}

LANGUAGE_SPEAKERS = {
    "English": 1457e6,
    "Arabic": 335e6,
    "Swahili": 97e6,
    "Hausa": 94e6,
    "Pashto": 55e6,
    "Yoruba": 50e6,
    "Dari": 30e6,
    "Luganda": 6e6,
    "Nyankore": 3e6,
}

LANGUAGES = sorted(LANGUAGE_SPEAKERS, key=LANGUAGE_SPEAKERS.get, reverse=True)

# Storyline ordering used by the beeswarm plot — English, then example
# Asian-originating languages, then example African-originating languages.
# Within each group, languages are sorted by speakers (descending).
LANGUAGE_GROUPS: list[tuple[str, list[str]]] = [
    ("English",                       ["English"]),
    ("Asian-originating",     ["Arabic", "Pashto", "Dari"]),
    ("African-originating",   ["Swahili", "Hausa", "Yoruba",
                                       "Luganda", "Nyankore"]),
]
LANGUAGES_GROUPED: list[str] = [l for _, langs in LANGUAGE_GROUPS for l in langs]


def _build_speaker_gradient_colors(
    speakers: dict[str, float],
    cmap_name: str = "Blues",
    lo: float = 0.30,
    hi: float = 0.95,
) -> dict[str, tuple]:
    """Map each language to a colormap shade scaled by log10(speakers).

    Log scale because the speaker range (~3M → ~1.5B) spans nearly three
    decades; a linear mapping would collapse every non-English language into
    the pale end. The output range is compressed to ``[lo, hi]`` so the
    lightest shade is still visible on white and the darkest is not pure
    black.
    """
    log_vals = {lang: math.log10(max(n, 1.0)) for lang, n in speakers.items()}
    lo_val, hi_val = min(log_vals.values()), max(log_vals.values())
    span = hi_val - lo_val or 1.0
    cmap = plt.get_cmap(cmap_name)
    return {
        lang: cmap(lo + (hi - lo) * (log_vals[lang] - lo_val) / span)
        for lang in speakers
    }


# Language → shade-of-blue, darker = more speakers (log-scaled).
LANGUAGE_COLORS_BY_SPEAKERS = _build_speaker_gradient_colors(LANGUAGE_SPEAKERS)

# Hardware tiers — see llm-hardware-tier-methodology.md.
# Each tuple: (label, RAM cap in GB, storage cap in GB). Ordered smallest →
# largest. Caps are inclusive upper bounds; assignment picks the *smallest*
# tier where both constraints fit.
HARDWARE_TIERS = [
    ("Mobile",      4,        32),
    ("Tablet",      12,       256),
    ("Workstation", 35,       2048),
    ("Cloud",       math.inf, math.inf),
]

# Quantization + overhead assumptions per the methodology doc:
# INT8 (1 byte / param) for all tiers, plus 25% RAM overhead for KV cache,
# activations, and framework.
BYTES_PER_PARAM = 1
RAM_OVERHEAD = 0.25

FRONTIER_PERIODS = [
    ("Before Jan 2025", None, pd.Timestamp("2025-01-01")),
    ("Jan-Aug 2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-09-01")),
    ("Sep-Dec 2025", pd.Timestamp("2025-09-01"), pd.Timestamp("2026-01-01")),
    ("Since Jan 2026", pd.Timestamp("2026-01-01"), None),
]

FRONTIER_COLORS = {
    "Before Jan 2025": "#d62728",
    "Jan-Aug 2025": "#f4b400",
    "Sep-Dec 2025": "#2ca02c",
    "Since Jan 2026": "#1f77b4",
}

BUCKET_COLORS = {
    "Mobile":      "#56B4E9",  # sky blue
    "Tablet":      "#009E73",  # bluish green
    "Workstation": "#E69F00",  # orange
    "Cloud":       "#CC3311",  # vermillion / red
}

BUCKET_LINESTYLES = {
    "Mobile":      ":",
    "Tablet":      "--",
    "Workstation": "-.",
    "Cloud":       "-",
}

# Most-recent reasoning model per provider that has benchmark data today.
# TODO: swap to newer models once benchmarked — Anthropic / Z.ai / Moonshot
# currently have no reasoning model with data.
#   Anthropic   -> claude-opus-4-7
#   OpenAI      -> gpt-5.5-2026-04-23-medium
#   xAI         -> grok-4.20-0309-reasoning
#   Z.ai        -> glm-5.1
#   Moonshot    -> kimi-k2.6
#   Alibaba     -> qwen3.6-plus
FOCUS_TOKEN_MODELS = [
    "gemini-3.1-pro-preview",         # Google,    02/2026
    "gpt-5.5-2026-04-23-medium",      # OpenAI,    04/2026
    #"claude-opus-4-6",              # Anthropic, 04/2026
    "qwen3.6-35b-a3b",                # Alibaba,   04/2026
    "grok-4.3",                       # xAI,       04/2026
    #"deepseek-r1-0528-fp8",          # DeepSeek,  05/2025
    "kimi-k2.5",                      # Moonshot,  01/2026
    "glm-5.1",                        # Z.ai,      04/202
]


def apply_clean_style(ax, title=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_facecolor("white")
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=10)


def load_models_metadata() -> pd.DataFrame:
    models = pd.read_csv(CONFIGS_DIR / "models.csv")

    required_cols = {"model_id", "size", "size_active", "input_cost",
                     "output_cost", "release_date", "display_name",
                     "provider", "open"}
    missing = required_cols - set(models.columns)
    if missing:
        sys.exit(f"[ERROR] models.csv missing required columns: {sorted(missing)}")

    models["release_date"] = pd.to_datetime(
        models["release_date"], format="%m/%Y", errors="coerce"
    )
    models["size"] = pd.to_numeric(models["size"], errors="coerce")
    models["size_active"] = pd.to_numeric(models["size_active"], errors="coerce")
    models["input_cost"] = pd.to_numeric(models["input_cost"], errors="coerce")
    models["output_cost"] = pd.to_numeric(models["output_cost"], errors="coerce")
    models["blended_cost"] = (3 * models["input_cost"] + models["output_cost"]) / 4

    # Synthesize alias rows: each MODEL_ALIASES entry inherits the original
    # model's metadata under the new model_id, so the downstream merge picks
    # it up without needing an extra row in models.csv.
    alias_rows = []
    for new_id, original_id in MODEL_ALIASES.items():
        match = models[models["model_id"] == original_id]
        if len(match) == 0:
            print(f"[WARN] alias '{new_id}' -> '{original_id}': original "
                  f"not found in models.csv; metadata will be missing.")
            continue
        if len(match) > 1:
            print(f"[WARN] alias '{new_id}' -> '{original_id}': "
                  f"{len(match)} matches in models.csv; using first.")
        row = match.iloc[0].copy()
        row["model_id"] = new_id
        alias_rows.append(row)
    if alias_rows:
        models = pd.concat([models, pd.DataFrame(alias_rows)],
                           ignore_index=True)
    return models


def _is_closed_weights(open_val) -> bool:
    """Robust check that the `open` column says False (handles bool / str / NaN)."""
    if open_val is False:
        return True
    if isinstance(open_val, str):
        return open_val.strip().lower() == "false"
    return False


def hardware_tier_for_model(
    size: float,
    size_active: float | None = None,
    is_open: bool | None = None,
    model_id: str | None = None,
) -> str | None:
    """Assign the smallest hardware tier where a model can run inference.

    Implements ``llm-hardware-tier-methodology.md``: INT8 (1 byte/param) for
    all tiers, 25% RAM overhead. MoE models use ``size_active`` for RAM but
    full ``size`` for storage.

    Returns the tier label (``Mobile``/``Tablet``/``Workstation``/``Cloud``)
    or ``None`` when the row should be excluded (open-weights model with
    unpublished size).
    """
    if pd.isna(size):
        # Closed-weights model: served only via vendor APIs → Cloud.
        if _is_closed_weights(is_open):
            return "Cloud"
        # Open-weights (or unknown openness) with no published size: exclude.
        if model_id is not None:
            print(f"[WARN] no size for '{model_id}' (open={is_open}); "
                  f"excluding from hardware-tier plots")
        return None

    # MoE: active params drive RAM, total params drive storage.
    if size_active is not None and not pd.isna(size_active):
        params_ram: float = float(size_active)
    else:
        params_ram = float(size)

    ram_gb = params_ram * BYTES_PER_PARAM * (1 + RAM_OVERHEAD) / 1e9
    storage_gb = float(size) * BYTES_PER_PARAM / 1e9

    for label, ram_cap, storage_cap in HARDWARE_TIERS:
        if ram_gb <= ram_cap and storage_gb <= storage_cap:
            return label
    return "Cloud"  # defensive — Cloud has inf caps, so should be unreachable


def load_default_models() -> dict[str, str]:
    with DEFAULT_MODELS_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_provider_colors() -> dict[str, str]:
    """Return {provider: color} from fab-benchmarks-configs/providers.csv."""
    path = CONFIGS_DIR / "providers.csv"
    df = pd.read_csv(path, usecols=["provider", "color"])
    df = df.dropna(subset=["provider", "color"])
    return dict(zip(df["provider"].astype(str), df["color"].astype(str)))


def discover_language_folders() -> list[dict]:
    """Return folder info for `_ep` language folders + the English folder."""
    folders = []
    for sub in sorted(RESULTS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        if not list(sub.glob("cdpk_results_full_*.csv")):
            continue
        name = sub.name
        if name.endswith("_ep"):
            folders.append({"path": sub, "language": name[:-3]})
        elif name == "English":
            folders.append({"path": sub, "language": "English"})
    return folders


def _models_from_full_df(full_df: pd.DataFrame) -> list[str]:
    return [c[5:] for c in full_df.columns if c.startswith("pred_")]


def build_exploded_df() -> pd.DataFrame:
    """Reconstruct the per-question DataFrame for _ep + English only.

    Mirrors the Overall-category branch of ``create_results.py`` but skips the
    list-of-arrays intermediate step by emitting one row per question directly.
    """
    rows = []
    for info in discover_language_folders():
        full_file = max(info["path"].glob("cdpk_results_full_*.csv"),
                        key=lambda p: p.stat().st_mtime)
        full_df = pd.read_csv(full_file)
        if "correct_answer" not in full_df.columns:
            sys.exit(f"[ERROR] {full_file}: missing required column 'correct_answer'")
        correct_answer = full_df["correct_answer"]

        for model in _models_from_full_df(full_df):
            if model in MODELS_TO_EXCLUDE:
                continue
            pred_col = f"pred_{model}"
            if pred_col not in full_df.columns:
                continue
            preds = full_df[pred_col]
            sub = pd.DataFrame({
                "model": model,
                "language": info["language"],
                "correct": (preds == correct_answer),
                "bad_format": preds.isna(),
            })
            for src, dst in [
                (f"Latency_{model}", "Latency"),
                (f"TokensUsed_{model}", "TokensUsed"),
                (f"TokensUsedCompletion_{model}", "TokensUsedCompletion"),
                (f"TokensUsedReasoning_{model}", "TokensUsedReasoning"),
            ]:
                sub[dst] = full_df[src].values if src in full_df.columns else np.nan
            rows.append(sub)
    if not rows:
        sys.exit("[ERROR] No _ep or English folders with full CSVs found.")
    return pd.concat(rows, ignore_index=True)


def load_data(exploded: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build exploded df, aggregate to per-(model, language) accuracy, join meta."""
    if exploded is None:
        exploded = build_exploded_df()
    agg = (
        exploded.groupby(["model", "language"], as_index=False)
        .agg(accuracy=("correct", lambda s: s.mean() * 100),
             n_samples=("correct", "size"))
    )
    meta = load_models_metadata()
    n_before = len(agg)
    merged = agg.merge(
        meta[[
            "model_id", "provider", "open", "release_date", "size", "size_active",
            "input_cost", "output_cost", "blended_cost", "display_name",
        ]],
        left_on="model", right_on="model_id", how="left",
    )
    if len(merged) != n_before:
        sys.exit(f"[ERROR] merge with models.csv changed row count: "
                 f"{n_before} -> {len(merged)} (likely duplicate model_id)")
    unmatched = merged.loc[merged["model_id"].isna(), "model"].unique()
    if len(unmatched):
        print(f"[WARN] {len(unmatched)} models in results not found in "
              f"models.csv: {sorted(unmatched)}")

    merged["hardware_tier"] = merged.apply(
        lambda r: hardware_tier_for_model(
            r["size"], r.get("size_active"), r.get("open"), r["model"],
        ),
        axis=1,
    )
    return merged


def pivot_accuracy(agg: pd.DataFrame) -> pd.DataFrame:
    """Return model x language accuracy pivot with an 'All' average column.

    Warns when any model has partial language coverage, since the 'All'
    average is then taken over fewer languages and is not directly
    comparable to fully-covered models.
    """
    pivot = agg.pivot(index="model", columns="language", values="accuracy")
    coverage = pivot[LANGUAGES].notna().sum(axis=1)
    sparse = coverage[coverage < len(LANGUAGES)]
    if len(sparse):
        print(f"[WARN] {len(sparse)} models have partial language coverage and are excluded from the plot")
        for m, n in sparse.items():
            print(f"       {m}: {n}/{len(LANGUAGES)}")
    pivot["All"] = pivot[LANGUAGES].mean(axis=1, skipna=True)
    return pivot


# ---------------------------------------------------------------------------
# Plot 1: multilingual default-models barplot
# ---------------------------------------------------------------------------

def plot_default_models_barplot(agg: pd.DataFrame, out_dir: Path,
                                trace_languages: list[str] | None = None) -> Path:
    default_map = load_default_models()
    pivot = pivot_accuracy(agg)

    models_in_order = [m for m in default_map if m in pivot.index]
    if not models_in_order:
        sys.exit("[ERROR] No default-list models found in CSV.")

    pivot = pivot.loc[models_in_order]

    # Keep only models that have an accuracy score for every language —
    # partial-coverage models distort the per-model min/max range line and
    # make the 'All (avg)' tick an apples-to-oranges comparison.
    full_coverage = pivot[LANGUAGES].notna().all(axis=1)
    excluded = pivot.index[~full_coverage].tolist()

    pivot = pivot.loc[full_coverage]
    if pivot.empty:
        sys.exit("[ERROR] No default-list models have full language coverage.")

    pivot = pivot.sort_values("English", ascending=False, na_position="last")

    lang_colors = LANGUAGE_COLORS_BY_SPEAKERS

    n_models = len(pivot)
    x = np.arange(n_models)

    fig_w = max(14, n_models * 0.3)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    lang_matrix = pivot[LANGUAGES].to_numpy(dtype=float)  # shape (n_models, n_langs)

    # Vertical range line per model (min -> max across languages)
    with np.errstate(invalid="ignore"):
        row_min = np.nanmin(lang_matrix, axis=1)
        row_max = np.nanmax(lang_matrix, axis=1)
    for i in range(n_models):
        if np.isnan(row_min[i]):
            continue
        ax.vlines(i, row_min[i], row_max[i], color="#cccccc",
                  linewidth=1.5, zorder=2)

    # One colored marker per language at each model's x position. Optionally
    # also a dotted line through each model's points for selected languages.
    trace_set = set(trace_languages or [])
    for j, lang in enumerate(LANGUAGES):
        vals = lang_matrix[:, j]
        if lang in trace_set:
            mask = ~np.isnan(vals)
            ax.plot(x[mask], vals[mask], linestyle=":", linewidth=1.2,
                    color=lang_colors[lang], zorder=3)
        ax.scatter(x, vals, s=55, color=lang_colors[lang],
                   label=lang, edgecolor="white", linewidth=0.6, zorder=4)

    # Black horizontal tick marking the mean-across-languages accuracy
    all_vals = pivot["All"].to_numpy(dtype=float)
    ax.scatter(x, all_vals, s=110, color="#000000", marker="_",
               linewidths=2.2, label="All (avg)", zorder=5)

    # Random-guess baseline (4-option multiple choice)
    ax.axhline(25, color="#888888", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(-0.55, 25, "Random guess", ha="left", va="bottom",
            fontsize=11, color="#888888", style="italic")

    display_labels = [default_map.get(m, m) for m in pivot.index]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="right",
                       fontsize=10, color="#333333")
    ax.set_ylabel("Accuracy (%)", fontsize=15, color="#333333")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.tick_params(axis="y", labelsize=14, colors="#333333")
    ax.set_xlim(-0.6, n_models - 0.4)
    apply_clean_style(ax)

    # Vertical legend on the right, ordered by number of speakers (LANGUAGES order)
    handles = [
        mlines.Line2D([], [], color=lang_colors[lang], marker="o",
                      linestyle="", markersize=9, markeredgecolor="white",
                      markeredgewidth=0.6, label=lang)
        for lang in LANGUAGES
    ]
    handles.append(
        mlines.Line2D([], [], color="#000000", marker="_", linestyle="",
                      markersize=14, markeredgewidth=2.2, label="All (avg)")
    )
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False,
              fontsize=13, title="Language", title_fontsize=14,
              handletextpad=0.4)

    #fig.suptitle("Multilingual Pedagogy Benchmark Accuracy",
    fig.suptitle("While the best AI models close the language gap, " \
    "mid-tier and small models keep it open",
                 fontsize=18, fontweight="bold", y=0.96)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    out_path = out_dir / "multilingual_default_models_barplot.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 2: best accuracy over time per size bucket, one subplot per language
# ---------------------------------------------------------------------------

def plot_best_accuracy_over_time_by_size(agg: pd.DataFrame, out_dir: Path) -> Path:
    data = agg.dropna(subset=["release_date"]).copy()

    n = len(LANGUAGES)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 5),
                             sharex=True, sharey=True)

    for ax, lang in zip(axes, LANGUAGES):
        lang_df = data[data["language"] == lang]
        for bucket_label, _ram_cap, _storage_cap in HARDWARE_TIERS:
            sub = lang_df[lang_df["hardware_tier"] == bucket_label]
            sub = sub.sort_values("release_date")
            if sub.empty:
                continue
            running_best = sub["accuracy"].cummax()
            ax.step(sub["release_date"], running_best, where="post",
                    color=BUCKET_COLORS[bucket_label],
                    linestyle=BUCKET_LINESTYLES[bucket_label],
                    marker="o", markersize=4,
                    linewidth=1.6, label=bucket_label, zorder=3)
        apply_clean_style(ax)
        ax.set_title(lang, fontsize=16, fontweight="bold", pad=26)
        speakers_m = LANGUAGE_SPEAKERS.get(lang, 0) / 1e6
        ax.text(0.5, 1.01, f"{speakers_m:.0f} M speakers",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, color="#888888", style="italic")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="both", labelsize=13)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            #tick.set_ha("right")
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("Accuracy (%)", fontsize=15)

    handles = [
        mlines.Line2D([], [], color=BUCKET_COLORS[lbl],
                      linestyle=BUCKET_LINESTYLES[lbl],
                      marker="o", markersize=6, linewidth=2.0,
                      label=(f"{lbl} (+ closed proprietary)" if lbl == "Cloud" else lbl))
        for lbl, _, _ in HARDWARE_TIERS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(HARDWARE_TIERS),
               frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.95))

    fig.suptitle("Best Accuracy over Time by Model Size Bucket - Pedagogy Benchmark",
                 fontsize=17, fontweight="bold", y=1.07)
    fig.text(0.5, 1.0, "Languages ordered by number of speakers",
             ha="center", va="top", fontsize=16,
             color="#888888", style="italic")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path = out_dir / "best_accuracy_over_time_by_size.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 2b: best accuracy over time per language, one subplot per size bucket
# ---------------------------------------------------------------------------

def plot_best_accuracy_over_time_by_language(agg: pd.DataFrame, out_dir: Path) -> Path:
    data = agg.dropna(subset=["release_date"]).copy()

    palette = sns.color_palette("bright", n_colors=len(LANGUAGES))
    lang_colors = dict(zip(LANGUAGES, palette))

    n = len(HARDWARE_TIERS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5),
                             sharex=True, sharey=True)

    for ax, (bucket_label, _ram_cap, _storage_cap) in zip(axes, HARDWARE_TIERS):
        bucket_df = data[data["hardware_tier"] == bucket_label]
        for lang in LANGUAGES:
            sub = bucket_df[bucket_df["language"] == lang]
            sub = sub.sort_values("release_date")
            if sub.empty:
                continue
            running_best = sub["accuracy"].cummax()
            ax.step(sub["release_date"], running_best, where="post",
                    color=lang_colors[lang],
                    marker="o", markersize=4,
                    linewidth=1.6, label=lang, zorder=3)
        apply_clean_style(ax)
        title = f"{bucket_label} (+ closed proprietary)" if bucket_label == "Cloud" else bucket_label
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="both", labelsize=13)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("Accuracy (%)", fontsize=15)

    handles = [
        mlines.Line2D([], [], color=lang_colors[lang], marker="o",
                      markersize=6, linewidth=2.0, label=lang)
        for lang in LANGUAGES
    ]
    fig.legend(handles=handles, loc="center left",
               bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=False,
               fontsize=13, title="Language", title_fontsize=14,
               handletextpad=0.5)

    fig.suptitle("Best Accuracy over Time by Language - Pedagogy Benchmark",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path = out_dir / "best_accuracy_over_time_by_language.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 2c: FT-style Accessibility Frontier (narrative companion to 2b)
# ---------------------------------------------------------------------------

HARDWARE_LABELS = {
    "Mobile":      ("Mobile (≤4GB RAM)",         "Basic Android / Edge"),
    "Tablet":      ("Tablet (4-12GB RAM)",       "High-end mobile / Tablets"),
    "Workstation": ("Workstation (12-35GB RAM)", "Laptops / Local servers"),
    "Cloud":       ("Cloud (>35GB RAM)",         "Cloud / Data centers only"),
}

PROTAGONIST_LANGS = {
    # "Local tier" — both warm reds/oranges so they read as one group.
    "Luganda":  {"color": "#f57c00", "linewidth": 2.8, "linestyle": "-", "zorder": 6, "marker": "o"},
    "Nyankore": {"color": "#c62828", "linewidth": 2.8, "linestyle": "-", "zorder": 6, "marker": "o"},
}
REFERENCE_LANGS = {
    # "Global tier" — unified cool palette (English dark grey, Arabic + Swahili
    # shades of blue). Arabic/Swahili get small markers; English stays smooth.
    "English":  {"color": "#222222", "linewidth": 1.6, "linestyle": "-", "zorder": 5, "marker": None},
    "Arabic":   {"color": "#1565c0", "linewidth": 1.5, "linestyle": "-", "zorder": 5, "marker": "o"},
    "Swahili":  {"color": "#4fc3f7", "linewidth": 1.5, "linestyle": "-", "zorder": 5, "marker": "o"},
}
BACKDROP_STYLE = {"color": "#cfcfcf", "linewidth": 0.8, "linestyle": ":",
                  "zorder": 3, "marker": None, "alpha": 0.6}

PAPER_BG = "#fffcf9"

UTILITY_BAND_DEFAULT = {"facecolor": "#eaf5ea", "alpha": 0.6}
UTILITY_TEXT_DEFAULT = "#2e7d32"

# Fab brand palette variant (primary blue, burnt-orange alarm, teal secondary,
# reference greys, pastel-purple pass/fail zone, light grey paper).
PROTAGONIST_LANGS_FAB = {
    "Luganda":  {"color": "#005CA2", "linewidth": 2.8, "linestyle": "-", "zorder": 6, "marker": "o"},
    "Nyankore": {"color": "#C45421", "linewidth": 2.8, "linestyle": "-", "zorder": 6, "marker": "o"},
}
REFERENCE_LANGS_FAB = {
    "English":  {"color": "#666666", "linewidth": 1.6, "linestyle": "-", "zorder": 5, "marker": None},
    "Arabic":   {"color": "#DE8430", "linewidth": 1.5, "linestyle": "-", "zorder": 5, "marker": "o"},
    "Swahili":  {"color": "#999999", "linewidth": 1.5, "linestyle": "-", "zorder": 5, "marker": "o"},
}
BACKDROP_STYLE_FAB = {"color": "#cccccc", "linewidth": 0.8, "linestyle": ":",
                      "zorder": 3, "marker": None, "alpha": 0.6}
PAPER_BG_FAB = "#F5F5F5"
UTILITY_BAND_FAB = {"facecolor": "#A58CDB", "alpha": 0.18}
UTILITY_TEXT_FAB = "#6648b8"

UTILITY_THRESHOLD = 70

# Hardware plot: how often to resample the running-best curve (in months).
# The x-axis tick stride is independent (hard-coded to every 6 months).
HARDWARE_SAMPLE_INTERVAL_MONTHS = 3


def _language_style(lang: str, prot: dict, ref: dict, backdrop: dict) -> dict:
    if lang in prot:
        return prot[lang]
    if lang in ref:
        return ref[lang]
    return backdrop


def _spread_label_ys(endpoints: list[tuple[str, pd.Timestamp, float, dict]],
                     min_gap: float = 3.2, y_min: float = 0.0,
                     y_max: float = 100.0) -> list[float]:
    """Compute non-overlapping y-positions for labels.

    Two-pass: bottom-up push to enforce min_gap, then top-down cap at y_max.
    Returns label y-positions in the same order as `endpoints`.
    """
    order = sorted(range(len(endpoints)), key=lambda i: endpoints[i][2])
    ys_sorted = [endpoints[i][2] for i in order]
    # Bottom-up: push each label at least min_gap above the previous
    for i in range(1, len(ys_sorted)):
        if ys_sorted[i] - ys_sorted[i - 1] < min_gap:
            ys_sorted[i] = ys_sorted[i - 1] + min_gap
    # Top-down: cap the top label at y_max, pull down as needed
    if ys_sorted and ys_sorted[-1] > y_max:
        ys_sorted[-1] = y_max
        for i in range(len(ys_sorted) - 2, -1, -1):
            if ys_sorted[i + 1] - ys_sorted[i] < min_gap:
                ys_sorted[i] = ys_sorted[i + 1] - min_gap
    # Clamp bottom
    if ys_sorted and ys_sorted[0] < y_min:
        ys_sorted[0] = y_min
        for i in range(1, len(ys_sorted)):
            if ys_sorted[i] - ys_sorted[i - 1] < min_gap:
                ys_sorted[i] = ys_sorted[i - 1] + min_gap
    # Reorder back to original endpoint order
    result = [0.0] * len(endpoints)
    for sorted_pos, orig_idx in enumerate(order):
        result[orig_idx] = ys_sorted[sorted_pos]
    return result


def _running_best_on_grid(
    sub: pd.DataFrame, x_min: pd.Timestamp, x_max: pd.Timestamp,
    interval_months: int,
) -> tuple[list[pd.Timestamp], np.ndarray] | tuple[None, None]:
    """Resample per-release running-best accuracy onto a month-stride grid.

    Deduplicates same-month releases by max, then forward-fills the cummax
    onto a grid with a configurable month interval between `x_min` and
    `x_max`. Returns only grid points where data exists.
    """
    if sub.empty:
        return None, None
    series = pd.Series(sub["accuracy"].values,
                       index=pd.DatetimeIndex(sub["release_date"].values))
    series = series.groupby(level=0).max().sort_index()
    running = series.cummax()
    grid = pd.date_range(start=x_min, end=x_max,
                         freq=f"{interval_months}MS")
    combined = running.reindex(running.index.union(grid)).ffill()
    resampled = combined.reindex(grid)
    mask = resampled.notna()
    return list(resampled.index[mask]), resampled.to_numpy()[mask]


def plot_accessibility_frontier(agg: pd.DataFrame, out_dir: Path,
                                fab_style: bool = False) -> Path:
    if fab_style:
        prot = PROTAGONIST_LANGS_FAB
        ref = REFERENCE_LANGS_FAB
        backdrop = BACKDROP_STYLE_FAB
        paper_bg = PAPER_BG_FAB
        util_band = UTILITY_BAND_FAB
        util_text_color = UTILITY_TEXT_FAB
    else:
        prot = PROTAGONIST_LANGS
        ref = REFERENCE_LANGS
        backdrop = BACKDROP_STYLE
        paper_bg = PAPER_BG
        util_band = UTILITY_BAND_DEFAULT
        util_text_color = UTILITY_TEXT_DEFAULT

    data = agg.dropna(subset=["release_date"]).copy()

    # `x_min` is where the data resampling grid starts (i.e. the leftmost
    # data point sits exactly here). `x_min_display` is where the axis
    # *visually* starts, with extra padding so the leftmost markers don't
    # get half-clipped against the y-axis.
    x_min = pd.Timestamp("2024-04-01")
    last_data_ts = pd.Timestamp(data["release_date"].max())
    x_max = max(
        last_data_ts + pd.DateOffset(months=1),
        pd.Timestamp("2026-08-01"),  # keep Jul 26 tick visible even with no data yet
    )
    x_min_display = x_min - pd.DateOffset(months=2)
    # Any grid point beyond this timestamp is "future" — drawn as a dotted
    # continuation to signal that it's a forward-fill of the last known value.
    future_cutoff = last_data_ts + pd.DateOffset(months=1)

    other_langs = [l for l in LANGUAGES
                   if l not in prot and l not in ref]
    # Hide backdrop ("other") languages so only protagonists + references show.
    # To restore the grey backdrop curves, set this to `other_langs`.
    #visible_other_langs: list[str] = []  # use `other_langs` to re-enable
    visible_other_langs = [l for l in other_langs if l in data["language"].unique()]

    # Hardware plot subplot order. Two storytelling options:
    #   * left-to-right: cloud → mobile (descending hardware capability)
    #   * left-to-right: mobile → cloud (ascending hardware capability)
    # Currently using mobile → cloud, which mirrors HARDWARE_TIERS' natural
    # smallest → largest ordering.
    #tiers_for_plot = list(reversed(HARDWARE_TIERS))  # cloud → mobile
    tiers_for_plot = HARDWARE_TIERS  # mobile → cloud

    n = len(tiers_for_plot)
    fig, axes = plt.subplots(1, n, figsize=(14, 6),
                             sharex=True, sharey=True)

    # Draw order: backdrop first, then references, then protagonists on top
    draw_order = (visible_other_langs
                  + list(ref.keys())
                  + list(prot.keys()))

    endpoints_per_ax: list[list[tuple[str, pd.Timestamp, float, dict]]] = []

    for ax, (bucket_label, _ram_cap, _storage_cap) in zip(axes, tiers_for_plot):
        bucket_df = data[data["hardware_tier"] == bucket_label]

        # Utility zone — pass/fail band above the threshold
        #ax.axhspan(UTILITY_THRESHOLD, 100,
        #           facecolor=util_band["facecolor"],
        #           alpha=util_band["alpha"], zorder=1)

        endpoints: list[tuple[str, pd.Timestamp, float, dict]] = []

        for lang in draw_order:
            sub = bucket_df[bucket_df["language"] == lang].sort_values("release_date")
            if sub.empty:
                continue
            grid_dates, grid_vals = _running_best_on_grid(
                sub, x_min, x_max, HARDWARE_SAMPLE_INTERVAL_MONTHS)
            if grid_dates is None or grid_vals is None or len(grid_dates) == 0:
                continue
            style = _language_style(lang, prot, ref, backdrop)

            # Split into historical vs. future segments at `future_cutoff`
            split_idx = next(
                (i for i, d in enumerate(grid_dates) if d > future_cutoff),
                len(grid_dates),
            )
            hist_dates = grid_dates[:split_idx]
            hist_vals = grid_vals[:split_idx]
            future_dates = grid_dates[split_idx:]
            future_vals = grid_vals[split_idx:]

            # Historical part keeps the language's normal style + markers
            if hist_dates:
                ax.step(hist_dates, hist_vals, where="post",
                        color=style["color"], linewidth=style["linewidth"],
                        linestyle=style["linestyle"],
                        marker=style["marker"] if style["marker"] else None,
                        markersize=3.5 if style["marker"] else 0,
                        alpha=style.get("alpha", 1.0),
                        zorder=style["zorder"])

            # Future part: dotted, no markers, connected to last historical point
            if len(future_dates) > 0:
                if hist_dates:
                    future_dates_full = [hist_dates[-1]] + list(future_dates)
                    future_vals_full = np.concatenate(
                        ([hist_vals[-1]], future_vals))
                else:
                    future_dates_full = list(future_dates)
                    future_vals_full = future_vals
                ax.step(future_dates_full, future_vals_full, where="post",
                        color=style["color"], linewidth=style["linewidth"],
                        linestyle=":",
                        alpha=style.get("alpha", 1.0) * 0.9,
                        zorder=style["zorder"])

            # End label anchor is the last plotted point (future if any, else historical)
            last_x = (future_dates[-1] if len(future_dates) > 0
                      else hist_dates[-1])
            last_y = (float(future_vals[-1]) if len(future_vals) > 0
                      else float(hist_vals[-1]))
            endpoints.append((lang, last_x, last_y, style))

        endpoints_per_ax.append(endpoints)

        apply_clean_style(ax)
        hw_title, hw_subtitle = HARDWARE_LABELS[bucket_label]
        ax.set_title(hw_title, fontsize=13, fontweight="bold", pad=26,
                     loc="left")
        ax.text(0.0, 1.01, hw_subtitle,
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, color="#888888", style="italic")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        # Minor ticks: fill the midpoints (Apr, Oct) between the labelled ones
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 10]))
        ax.tick_params(axis="both", labelsize=12)
        ax.tick_params(axis="x", which="major", length=5, color="#888888")
        ax.tick_params(axis="x", which="minor", length=3, color="#aaaaaa")
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_xlim(x_min_display, x_max)
        ax.set_ylim(0, 100)
        # Paper background on the axes (overrides the white set by apply_clean_style)
        ax.set_facecolor(paper_bg)

    # Figure background + vertical y-axis label on the first subplot
    fig.patch.set_facecolor(paper_bg)
    axes[0].set_ylabel("Accuracy Pedagogy Benchmark (%)",
                       fontsize=14, color="#333333")

    # Utility threshold label on the first subplot only
    #axes[3].text(x_min, UTILITY_THRESHOLD + 1.5, "Utility threshold (70%)",
    #             ha="left", va="bottom", fontsize=10,
    #             color=util_text_color, style="italic")

    # Direct end-of-line labels, spread vertically to avoid overlap.
    # Currently labelling every subplot. To restore "rightmost subplot only"
    # behaviour (cleaner look for crowded plots), uncomment the gating block
    # below.
    last_idx = len(axes) - 1
    for i, (ax, endpoints) in enumerate(zip(axes, endpoints_per_ax)):
        # if i != last_idx:  # uncomment to label only the rightmost subplot
        #     continue
        if not endpoints:
            continue
        y_labels = _spread_label_ys(endpoints, min_gap=3.2,
                                    y_min=0.0, y_max=100.0)
        for (lang, x_last, y_last, style), y_lbl in zip(endpoints, y_labels):
            is_prot = lang in prot
            is_ref = lang in ref
            # Thin leader from the actual endpoint to the label position,
            # only when the label had to be nudged noticeably.
            if abs(y_lbl - y_last) > 1.5:
                ax.plot([x_last, x_last], [y_last, y_lbl],
                        color=style["color"], linewidth=0.6,
                        alpha=0.5, zorder=style["zorder"] - 0.1,
                        clip_on=False)
            ax.annotate(
                lang,
                xy=(x_last, y_lbl),
                xytext=(6, 0),
                textcoords="offset points",
                va="center", ha="left",
                fontsize=10 if is_prot else (9 if is_ref else 8),
                color=style["color"],
                alpha=1.0 if (is_prot or is_ref) else 0.75,
                fontweight=("bold" if is_prot else "normal"),
                clip_on=False,
            )

    # ------------------------------------------------------------------
    # Narrative callouts: one editorial line per subplot, bottom-centered.
    # ------------------------------------------------------------------
    subplot_points = {
        "Mobile":      "Two years, no real progress\nfor local languages.",
        "Tablet":      "Swahili pulls ahead,\nLuganda and Nyankore stall.",
        "Workstation": "First real climb for local languages,\nstill short of what's needed.",
        "Cloud":       "Progress is being made,\nbut only for the largest models",
    }
    for ax, (bucket_label, _ram_cap, _storage_cap) in zip(axes, tiers_for_plot):
        ax.text(0.02, 0.03, subplot_points[bucket_label],
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color="#444444", style="italic",
                fontweight="bold", linespacing=1.25)

    # Reserve the top ~18% of the figure for the suptitle + subtitle so they
    # don't collide with the subplot titles (which have pad=26 and are
    # accompanied by an italic hardware-subtitle just above each axis).
    # tight_layout must run *before* we read axis positions for left alignment.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    left_x = axes[0].get_position().x0

    # Title + subtitle (FT "double-edged sword" framing), left-aligned with
    # the leftmost subplot's y-axis.
    fig.suptitle(
        #"AI models can now support local languages, but not on the devices teachers actually use in the Global South",
        "AI models are making progress on African languages, but too slowly at optimising for the devices teachers use",
        fontsize=14, fontweight="bold",
        x=left_x, y=0.97, ha="left",
    )
    fig.text(
        left_x, 0.90,
        # Old (FT-style narrative subtitle):
        # "Arabic and Swahili models work well on tablets; Luganda and Nyankore still require"
        # " running in the cloud — out of reach without reliable internet"
        "Each line represents the best accuracy reached, over time, by any model that fits in a given"
        " hardware tier — for one language.",
        ha="left", va="top", fontsize=12, color="#666666", #style="italic",
    )
    # Source line at the bottom of the figure
    #fig.text(
    #    0.5, -0.02,
    #    "Source: Multilingual Pedagogy Benchmark, AI-for-Education.org, "
    #    "April 2026.  Lines show the highest accuracy reached by any model "
    #    "in each hardware tier, sampled every two months. Overall accuracy, "
    #    "English-prompt variants.",
    #    ha="center", va="top",
    #    fontsize=8, color="#888888", style="italic",
    #)
    suffix = "_fabstyle" if fab_style else ""
    out_path = out_dir / f"accessibility_frontier_by_hardware{suffix}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight",
                facecolor=paper_bg)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 3: accuracy vs cost scatter with temporal Pareto frontiers
# ---------------------------------------------------------------------------

def pareto_frontier(costs: np.ndarray, accs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Upper-left Pareto frontier on (cost asc, accuracy max-so-far)."""
    order = np.argsort(costs)
    costs_s = costs[order]
    accs_s = accs[order]
    keep_cost, keep_acc = [], []
    best = -np.inf
    for c, a in zip(costs_s, accs_s):
        if a > best:
            keep_cost.append(c)
            keep_acc.append(a)
            best = a
    return np.array(keep_cost), np.array(keep_acc)


def plot_accuracy_cost_with_frontiers(agg: pd.DataFrame, out_dir: Path) -> Path:
    data = agg.dropna(subset=["release_date", "blended_cost"]).copy()
    data = data[data["blended_cost"] > 0]

    n = len(LANGUAGES)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 5),
                             sharex=True, sharey=True)

    for ax, lang in zip(axes, LANGUAGES):
        lang_df = data[data["language"] == lang]
        if lang_df.empty:
            apply_clean_style(ax, title=lang)
            continue

        ax.scatter(lang_df["blended_cost"], lang_df["accuracy"],
                   color="#9e9e9e", alpha=0.45, s=22, zorder=2,
                   label="All Models")

        for label, _start, end in FRONTIER_PERIODS:
            # Cumulative frontier: all models released before the period end.
            cum_df = lang_df if end is None else lang_df[lang_df["release_date"] < end]
            if cum_df.empty:
                continue
            fc, fa = pareto_frontier(cum_df["blended_cost"].to_numpy(),
                                     cum_df["accuracy"].to_numpy())
            if len(fc) == 0:
                continue
            ax.step(fc, fa, where="post",
                    color=FRONTIER_COLORS[label],
                    linestyle="--", linewidth=1.8,
                    marker="o", markersize=5,
                    label=label, zorder=4)

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda v, _: f"{v:g}")
        )
        ax.xaxis.set_minor_formatter(mtick.NullFormatter())
        ax.set_ylim(0, 100)
        ax.set_xlabel("Price ($/1M tokens)", fontsize=13)
        apply_clean_style(ax)
        ax.set_title(lang, fontsize=16, fontweight="bold", pad=26)
        speakers_m = LANGUAGE_SPEAKERS.get(lang, 0) / 1e6
        ax.text(0.5, 1.01, f"{speakers_m:.0f} M speakers",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, color="#888888", style="italic")
        ax.tick_params(axis="both", labelsize=13)

    axes[0].set_ylabel("Accuracy (%)", fontsize=15)

    handles = [
        mlines.Line2D([], [], color="#9e9e9e", marker="o", linestyle="",
                      markersize=7, alpha=0.7, label="All Models")
    ] + [
        mlines.Line2D([], [], color=FRONTIER_COLORS[lbl], marker="o",
                      markersize=7, linewidth=2.2, label=lbl)
        for lbl, _, _ in FRONTIER_PERIODS
    ]
    fig.legend(handles=handles, loc="upper center",
               ncol=len(FRONTIER_PERIODS) + 1, frameon=False, fontsize=14,
               bbox_to_anchor=(0.5, 0.95))

    fig.suptitle("Value Frontier over Time - Pedagogy Benchmark Accuracy vs Price",
                 fontsize=17, fontweight="bold", y=1.07)
    fig.text(0.5, 1.0, "Languages ordered by number of speakers",
             ha="center", va="top", fontsize=16,
             color="#888888", style="italic")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path = out_dir / "accuracy_cost_value_frontier.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 4: per-language token usage for a small focus set of reasoning models
# (overhead vs English baseline + completion/reasoning stack)
# ---------------------------------------------------------------------------

TOKEN_COMPLETION_COLOR = "#005CA2"  # Fab Blue
TOKEN_REASONING_COLOR = "#C45421"  # Fab Burnt Orange


def aggregate_token_usage(exploded: pd.DataFrame,
                          focus_models: list[str]) -> pd.DataFrame:
    """Per-(model, language) median completion + reasoning tokens for the focus set."""
    df = exploded[exploded["model"].isin(focus_models)].copy()
    for col in ("TokensUsedCompletion", "TokensUsedReasoning"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = (
        df.groupby(["model", "language"], as_index=False)
        .agg(median_completion=("TokensUsedCompletion", "median"),
             median_reasoning=("TokensUsedReasoning", "median"))
    )
    return grouped


def plot_tokens_per_language(exploded: pd.DataFrame, out_dir: Path) -> Path:
    meta = load_models_metadata().set_index("model_id")["display_name"]

    present = [m for m in FOCUS_TOKEN_MODELS
               if m in exploded["model"].unique()]
    missing = [m for m in FOCUS_TOKEN_MODELS if m not in present]
    for m in missing:
        print(f"[WARN] focus token model not found in data: {m}")
    if not present:
        sys.exit("[ERROR] No focus token models found in data.")

    tokens = aggregate_token_usage(exploded, present)

    n = len(present)
    n_cols = n
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.4 * n_cols, 4.5),
                             sharey=False, squeeze=False)
    axes_flat = list(axes.flatten())

    x = np.arange(len(LANGUAGES))
    bar_width = 0.7

    for ax, model in zip(axes_flat, present):
        sub = tokens[tokens["model"] == model].set_index("language")

        comp = np.array([sub["median_completion"].get(l, np.nan)
                         for l in LANGUAGES], dtype=float)
        reas = np.array([sub["median_reasoning"].get(l, np.nan)
                         for l in LANGUAGES], dtype=float)
        totals = np.nan_to_num(comp, nan=0.0) + np.nan_to_num(reas, nan=0.0)

        # Per-subplot y-max — pad 18% above the tallest bar for overhead annotations.
        y_max_data = float(totals.max())
        y_max = y_max_data * 1.18 if y_max_data > 0 else 1.0

        ax.bar(x, np.nan_to_num(comp, nan=0.0), width=bar_width,
               color=TOKEN_COMPLETION_COLOR, zorder=3)
        ax.bar(x, np.nan_to_num(reas, nan=0.0), width=bar_width,
               bottom=np.nan_to_num(comp, nan=0.0),
               color=TOKEN_REASONING_COLOR, zorder=3)

        english_total = (totals[LANGUAGES.index("English")]
                         if "English" in LANGUAGES else np.nan)
        if english_total and not np.isnan(english_total):
            ax.axhline(english_total, color="#888888", linestyle=":",
                       linewidth=1.0, zorder=2)

        for xi, total in zip(x, totals):
            if total <= 0:
                continue
            if english_total and english_total > 0:
                ratio = total / english_total
                label = f"{ratio:.1f}×"
            else:
                label = ""
            ax.text(xi, float(total) + y_max * 0.02, label,
                    ha="center", va="bottom", fontsize=8,
                    color="#333333", zorder=5)

        apply_clean_style(ax)
        title = meta.get(model, model)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8,
                     loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(LANGUAGES, rotation=45, ha="right",
                           fontsize=9, color="#333333")
        ax.set_ylim(0, y_max)
        ax.tick_params(axis="y", labelsize=10, colors="#333333")

    # Hide any unused axes (when n is odd)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Each subplot has its own y-scale; label the leftmost column of each
    # row so the unit is unambiguous without crowding every panel.
    for row in range(n_rows):
        axes[row, 0].set_ylabel("Median tokens / question",
                                fontsize=12, color="#333333")

    handles = [
        mlines.Line2D([], [], color=TOKEN_COMPLETION_COLOR, marker="s",
                      linestyle="", markersize=10, label="Completion tokens"),
        mlines.Line2D([], [], color=TOKEN_REASONING_COLOR, marker="s",
                      linestyle="", markersize=10, label="Reasoning tokens"),
        mlines.Line2D([], [], color="#888888", linestyle=":",
                      linewidth=1.0, label="English baseline"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.96))

    fig.suptitle(
        "Reasoning models pay a hidden token tax in non-English languages",
        fontsize=15, fontweight="bold", y=1.0,
    )
    #fig.text(
    #    0.5, 0.97,
    #    "Mean tokens per question, stacked into output (completion) "
    #    "and hidden reasoning. Annotation = total tokens relative to "
    #    "the model's own English baseline.",
    #    ha="center", va="top", fontsize=10, color="#666666", style="italic",
    #)
    #fig.text(
    #    0.5, -0.01,
    #    "Source: Multilingual Pedagogy Benchmark, AI-for-Education.org. "
    #    "English-prompt variants (_ep) only. Languages ordered by number "
    #    "of speakers.",
    #    ha="center", va="top", fontsize=8, color="#888888", style="italic",
    #)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path = out_dir / "tokens_per_language_reasoning_models.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 4b: tokens-by-(model, language) — grouped-and-stacked bar plot.
# One cluster of N bars per FOCUS_TOKEN_MODELS entry on the x-axis, one bar
# per language; each bar is a stack of completion (solid) on the bottom and
# reasoning (hatched) on top, sharing the language colour.
# ---------------------------------------------------------------------------

def plot_tokens_grouped_barplot(exploded: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped-and-stacked token-usage barplot for the focus reasoning models.

    Each model gets one cluster of bars on the x-axis (one bar per language),
    each bar a stack of median completion + reasoning tokens.
    """
    meta = load_models_metadata().set_index("model_id")["display_name"]

    available = set(exploded["model"].unique())
    present = [m for m in FOCUS_TOKEN_MODELS if m in available]
    missing = [m for m in FOCUS_TOKEN_MODELS if m not in available]
    for m in missing:
        print(f"[WARN] tokens-grouped model not found in data: {m}")
    if not present:
        sys.exit("[ERROR] No tokens-grouped models found in data.")

    tokens = aggregate_token_usage(exploded, present)

    palette = sns.color_palette("bright", n_colors=len(LANGUAGES))
    lang_colors = dict(zip(LANGUAGES, palette))

    n_models = len(present)
    n_langs = len(LANGUAGES)
    # Each model occupies 1 unit on the x-axis. Reserve 85% for bars and
    # 15% for inter-cluster whitespace; split the 85% evenly across N langs.
    cluster_width = 0.85
    bar_width = cluster_width / n_langs
    x_offsets = np.linspace(
        -cluster_width / 2 + bar_width / 2,
         cluster_width / 2 - bar_width / 2,
         n_langs,
    )

    fig_w = max(12, n_models * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, 7.0))

    for i, model in enumerate(present):
        sub = tokens[tokens["model"] == model].set_index("language")
        for j, lang in enumerate(LANGUAGES):
            comp = sub["median_completion"].get(lang, np.nan)
            reas = sub["median_reasoning"].get(lang, np.nan)
            comp_val = 0.0 if pd.isna(comp) else float(comp)
            reas_val = 0.0 if pd.isna(reas) else float(reas)
            xpos = i + x_offsets[j]
            # Solid completion bar at the bottom of the stack.
            ax.bar(xpos, comp_val, width=bar_width,
                   color=lang_colors[lang],
                   edgecolor="white", linewidth=0.3, zorder=3)
            # Hatched reasoning bar on top — same colour, "///" pattern.
            if reas_val > 0:
                ax.bar(xpos, reas_val, width=bar_width, bottom=comp_val,
                       color=lang_colors[lang],
                       hatch="///", edgecolor="white", linewidth=0.4,
                       zorder=3)

    # X-axis labels: model display names.
    display_labels = [meta.get(m, m) for m in present]
    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10,
                       color="#333333")
    ax.set_xlim(-0.6, n_models - 0.4)

    ax.set_ylabel("Median tokens / question", fontsize=13, color="#333333")
    ax.tick_params(axis="y", labelsize=11, colors="#333333")
    apply_clean_style(ax)

    # Two legends side-by-side on the right: language colours + hatch key.
    lang_handles = [
        mpatches.Patch(facecolor=lang_colors[l], edgecolor="white", label=l)
        for l in LANGUAGES
    ]
    hatch_handles = [
        mpatches.Patch(facecolor="#cccccc", edgecolor="white",
                       label="Completion tokens"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="white",
                       hatch="///", label="Reasoning tokens"),
    ]
    leg1 = ax.legend(handles=lang_handles, loc="upper left",
                     bbox_to_anchor=(1.01, 1.0), frameon=False,
                     title="Language", fontsize=10, title_fontsize=11,
                     handletextpad=0.5)
    ax.add_artist(leg1)
    ax.legend(handles=hatch_handles, loc="upper left",
              bbox_to_anchor=(1.01, 0.50), frameon=False,
              title="Token type", fontsize=10, title_fontsize=11,
              handletextpad=0.5)

    fig.suptitle(
        "Token usage by model and language — completion + reasoning",
        fontsize=15, fontweight="bold", y=1.00,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.88, 0.96))
    out_path = out_dir / "tokens_grouped_barplot_by_model.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot 5: per-language latency for the same focus set as the tokens plot.
# Designed to be stacked directly under the tokens figure so the visual
# correlation between reasoning-token volume and latency is immediate.
# ---------------------------------------------------------------------------

LATENCY_BAR_COLOR = "#444444"  # neutral dark grey, distinct from tokens palette


def aggregate_latency(exploded: pd.DataFrame,
                      focus_models: list[str]) -> pd.DataFrame:
    """Per-(model, language) median latency for the focus set."""
    df = exploded[exploded["model"].isin(focus_models)].copy()
    df["Latency"] = pd.to_numeric(df["Latency"], errors="coerce")
    return (
        df.groupby(["model", "language"], as_index=False)
        .agg(median_latency=("Latency", "median"))
    )


def plot_latency_per_language(exploded: pd.DataFrame, out_dir: Path) -> Path:
    meta = load_models_metadata().set_index("model_id")["display_name"]

    present = [m for m in FOCUS_TOKEN_MODELS
               if m in exploded["model"].unique()]
    missing = [m for m in FOCUS_TOKEN_MODELS if m not in present]
    for m in missing:
        print(f"[WARN] focus token model not found in data: {m}")
    if not present:
        sys.exit("[ERROR] No focus token models found in data.")

    lat = aggregate_latency(exploded, present)

    n = len(present)
    n_cols = max(1, math.ceil(n / 2))
    n_rows = 2 if n > 1 else 1
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.4 * n_cols, 4.25 * n_rows),
                             sharey=False, squeeze=False)
    axes_flat = list(axes.flatten())

    x = np.arange(len(LANGUAGES))
    bar_width = 0.7

    for ax, model in zip(axes_flat, present):
        sub = lat[lat["model"] == model].set_index("language")
        vals = np.array([sub["median_latency"].get(l, np.nan)
                         for l in LANGUAGES], dtype=float)
        plotted = np.nan_to_num(vals, nan=0.0)

        y_max = float(plotted.max()) * 1.18 if plotted.max() > 0 else 1.0

        ax.bar(x, plotted, width=bar_width,
               color=LATENCY_BAR_COLOR, zorder=3)

        english_val = (vals[LANGUAGES.index("English")]
                       if "English" in LANGUAGES else np.nan)
        if english_val and not np.isnan(english_val):
            ax.axhline(english_val, color="#888888", linestyle=":",
                       linewidth=1.0, zorder=2)

        for xi, total in zip(x, plotted):
            if total <= 0:
                continue
            if english_val and english_val > 0:
                label = f"{total / english_val:.1f}×"
            else:
                label = ""
            ax.text(xi, total + y_max * 0.012, label,
                    ha="center", va="bottom", fontsize=8,
                    color="#333333")

        apply_clean_style(ax)
        title = meta.get(model, model)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8,
                     loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(LANGUAGES, rotation=45, ha="right",
                           fontsize=9, color="#333333")
        ax.set_ylim(0, y_max)
        ax.tick_params(axis="y", labelsize=10, colors="#333333")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    for row in range(n_rows):
        axes[row, 0].set_ylabel("Median latency / question (s)",
                                fontsize=12, color="#333333")

    handles = [
        mlines.Line2D([], [], color=LATENCY_BAR_COLOR, marker="s",
                      linestyle="", markersize=10, label="Median latency"),
        mlines.Line2D([], [], color="#888888", linestyle=":",
                      linewidth=1.0, label="English baseline"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.96))

    fig.suptitle(
        "Latency in non-English languages exceeds usable UX thresholds",
        fontsize=15, fontweight="bold", y=1.0,
    )
    #fig.text(
    #    0.5, 0.97,
    #    "Median wall-clock latency per question (seconds). Annotation = "
    #    "latency relative to the model's own English baseline.",
    #    ha="center", va="top", fontsize=10, color="#666666", style="italic",
    #)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path = out_dir / "latency_per_language_reasoning_models.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_path



# ---------------------------------------------------------------------------
# Plot: per-language accuracy beeswarm (one dot per model)
# ---------------------------------------------------------------------------

def plot_beeswarm_accuracy(agg: pd.DataFrame, out_dir: Path,
                           color_by: str = "language") -> Path:
    """Beeswarm of per-model accuracy, one swarm per language.

    ``color_by`` selects the dot palette:
      * ``"language"``       — each swarm uses its own LANGUAGE_COLORS_BY_SPEAKERS shade
      * ``"provider"``       — dots coloured by model provider (tab20 palette)
      * ``"hardware_tier"``  — dots coloured by BUCKET_COLORS

    Restricted to models with an accuracy score in every language so the
    per-language mean tick is computed over the same set of models for each
    swarm (otherwise low-coverage languages would have artificially noisy
    means).
    """
    pivot = pivot_accuracy(agg)
    full_coverage = pivot[LANGUAGES].notna().all(axis=1)
    excluded = pivot.index[~full_coverage].tolist()
    if excluded:
        print(f"[INFO] excluding {len(excluded)} model(s) from beeswarm plot "
              f"(missing at least one language)")
    kept = pivot.index[full_coverage]
    if len(kept) == 0:
        sys.exit("[ERROR] No models have full language coverage for beeswarm.")

    sub = (agg[agg["model"].isin(kept) & agg["language"].isin(LANGUAGES)]
           [["model", "language", "accuracy", "provider", "hardware_tier"]]
           .copy())

    # Resolve palette, hue order (data) and legend order (display) per variant.
    if color_by == "language":
        palette = {l: LANGUAGE_COLORS_BY_SPEAKERS[l] for l in LANGUAGES}
        hue_order = LANGUAGES
        legend_order = hue_order  # unused — language variant builds its own legend
        legend_title = None
    elif color_by == "provider":
        sub["provider"] = sub["provider"].fillna("Unknown")
        providers = sorted(p for p in sub["provider"].unique() if p != "Unknown")
        # Brand colours from providers.csv; fall back to tab20 for any provider
        # missing from the CSV so we never crash on a new provider.
        brand = load_provider_colors()
        fallback = sns.color_palette("tab20", n_colors=max(len(providers), 1))
        palette = {
            p: brand.get(p, fallback[i % len(fallback)])
            for i, p in enumerate(providers)
        }
        if "Unknown" in sub["provider"].values:
            palette["Unknown"] = (0.6, 0.6, 0.6)
            providers = providers + ["Unknown"]
        hue_order = providers
        legend_order = providers
        legend_title = "Provider"
    elif color_by == "hardware_tier":
        sub["hardware_tier"] = sub["hardware_tier"].fillna("Unknown")
        palette = dict(BUCKET_COLORS)
        tier_order = [t[0] for t in HARDWARE_TIERS]
        has_unknown = "Unknown" in sub["hardware_tier"].values
        if has_unknown:
            palette["Unknown"] = (0.6, 0.6, 0.6)
        hue_order = tier_order + (["Unknown"] if has_unknown else [])
        # Legend reads top-to-bottom: Cloud → Workstation → Tablet → Mobile,
        # with any Unknown bucket pinned at the bottom.
        legend_order = list(reversed(tier_order)) + (["Unknown"] if has_unknown else [])
        legend_title = "Hardware tier"
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    # Storyline ordering: English | Asian-originating | African-originating.
    lang_order = LANGUAGES_GROUPED
    sub = sub[sub["language"].isin(lang_order)]

    fig_w = max(12, len(lang_order) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    # Artists outside the axes bounds (right-side legend, suptitle, speaker
    # labels below the x-axis) get clipped by `bbox_inches="tight"` unless
    # explicitly passed via `bbox_extra_artists`. We collect them here.
    extra_artists: list = []

    sns.swarmplot(
        data=sub, x="language", y="accuracy", order=lang_order,
        hue=color_by, palette=palette, hue_order=hue_order, legend=False,
        size=5, alpha=0.9, edgecolor="white", linewidth=0.5,
        ax=ax, zorder=3,
    )

    # Per-language median (black cross over each swarm).
    medians = sub.groupby("language")["accuracy"].median()
    x_idx = {lang: i for i, lang in enumerate(lang_order)}
    for lang, m in medians.items():
        if lang in x_idx:
            ax.scatter([x_idx[lang]], [m], s=140, color="#000000",
                       marker="X", linewidths=1.4, edgecolor="white",
                       zorder=5)

    # Random-guess baseline (4-option multiple choice)
    ax.axhline(25, color="#888888", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(-0.75, 25, "Random guess", ha="left", va="bottom",
            fontsize=10, color="#888888", style="italic")

    # Vertical separators between language groups + group labels below the axis.
    boundaries: list[float] = []  # x-positions of separator lines
    group_spans: list[tuple[str, float, float]] = []  # (label, x_start, x_end)
    cursor = 0
    for group_label, group_langs in LANGUAGE_GROUPS:
        present = [l for l in group_langs if l in x_idx]
        if not present:
            continue
        x_start = x_idx[present[0]]
        x_end = x_idx[present[-1]]
        group_spans.append((group_label, x_start, x_end))
        if cursor > 0:
            boundaries.append(x_start - 0.5)
        cursor += 1
    for xb in boundaries:
        ax.axvline(xb, color="#cccccc", linewidth=1.0, linestyle="-",
                   zorder=0)
    # Group labels at the top of the plot, just below the 100% line.
    # Single-language groups (e.g. English) get no label — the tick label is enough.
    for label, x_start, x_end in group_spans:
        if x_start == x_end:
            continue
        ax.text((x_start + x_end) / 2, 0.97, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=11, color="#555555",
                style="italic")

    apply_clean_style(ax)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel("Accuracy (%)", fontsize=14, color="#333333")
    ax.set_xlabel("")
    # X-tick labels: language name (upright). Italic speaker count sits below
    # as a separate text element so the two lines can have different styles.
    ax.set_xticks(range(len(lang_order)))
    ax.set_xticklabels(lang_order)
    ax.tick_params(axis="x", labelsize=12, colors="#333333")
    ax.tick_params(axis="y", labelsize=12, colors="#333333")
    for i, l in enumerate(lang_order):
        t = ax.text(i, -0.08, f"{LANGUAGE_SPEAKERS.get(l, 0) / 1e6:.0f}M",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=10, color="#666666",
                    style="italic")
        extra_artists.append(t)

    median_handle = mlines.Line2D([], [], color="#000000", marker="X",
                                  linestyle="", markersize=11,
                                  markeredgewidth=1.4, markeredgecolor="white",
                                  label="Median across models")

    if color_by == "language":
        # x-axis already names the languages; only show the median tick legend.
        ax.legend(handles=[
            mlines.Line2D([], [], color="#777777", marker="o", linestyle="",
                          markersize=8, markeredgecolor="white",
                          markeredgewidth=0.5, label="One model"),
            median_handle,
        ], loc="lower left", frameon=False, fontsize=11)

        # Colorbar on the right explaining the speaker-count → blue-shade mapping.
        # Re-create the same truncated Blues colormap and log-norm that
        # _build_speaker_gradient_colors uses so the bar matches the dots exactly.
        truncated = mcolors.LinearSegmentedColormap.from_list(
            "BluesTrunc", plt.get_cmap("Blues")(np.linspace(0.30, 0.95, 256))
        )
        vmin = min(LANGUAGE_SPEAKERS.values())
        vmax = max(LANGUAGE_SPEAKERS.values())
        sm = mcm.ScalarMappable(
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
            cmap=truncated,
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.85)
        ticks_M = [3, 10, 30, 100, 300, 1000]
        cbar.set_ticks([t * 1e6 for t in ticks_M])
        cbar.set_ticklabels([f"{t}M" for t in ticks_M])
        cbar.ax.tick_params(labelsize=10, colors="#555555")
        cbar.set_label("Speakers (log scale)", fontsize=11, color="#444444")
        cbar.outline.set_visible(False)
    else:
        # For the hardware tier variant, append "(N)" to each label where N is
        # the number of distinct models in that tier (within the kept set).
        if color_by == "hardware_tier":
            tier_counts = (sub.drop_duplicates("model")
                              .groupby("hardware_tier").size().to_dict())
            def _legend_label(c: str) -> str:
                return f"{c} ({tier_counts.get(c, 0)})"
        else:
            def _legend_label(c: str) -> str:
                return str(c)
        cat_handles = [
            mlines.Line2D([], [], color=palette[c], marker="o", linestyle="",
                          markersize=9, markeredgecolor="white",
                          markeredgewidth=0.5, label=_legend_label(c))
            for c in legend_order
        ]
        # Categorical legend on the right (providers or hardware tiers).
        # `bbox_inches="tight"` ignores legends added via add_artist, so we
        # also stash this in `extra_artists` and pass it to savefig — otherwise
        # the legend renders past the SVG viewBox and gets clipped.
        cat_legend = ax.legend(handles=cat_handles, loc="center left",
                               bbox_to_anchor=(1.01, 0.5), frameon=False,
                               fontsize=10, title=legend_title,
                               title_fontsize=11, handletextpad=0.5)
        ax.add_artist(cat_legend)
        extra_artists.append(cat_legend)
        # Marker-meaning legend at the bottom-left, same as the language variant.
        one_model_handle = mlines.Line2D([], [], color="#777777", marker="o",
                                         linestyle="", markersize=8,
                                         markeredgecolor="white",
                                         markeredgewidth=0.5, label="One model")
        ax.legend(handles=[one_model_handle, median_handle], loc="lower left",
                  frameon=False, fontsize=11)

    suptitle = fig.suptitle(
        "Multilingual Pedagogy Benchmark Performance",
        fontsize=17, fontweight="bold",
    )
    extra_artists.append(suptitle)
    suffix = {
        "language": "",
        "provider": "_by_provider",
        "hardware_tier": "_by_hardware",
    }[color_by]
    out_path = out_dir / f"beeswarm_accuracy_per_language{suffix}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight",
                bbox_extra_artists=extra_artists)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR,
                        help="Where to write the SVG outputs (default: %(default)s)")
    parser.add_argument("--plots", nargs="+",
                        choices=["bar", "steps", "steps_by_lang", "scatter",
                                 "hardware", "tokens", "tokens_grouped",
                                 "latency", "beeswarm", "all"],
                        default=["all"],
                        help="Which plots to render (default: all)")
    parser.add_argument(
        "--trace-languages", nargs="+", choices=LANGUAGES + ["all"],
        default=[],
        help="Languages for which to draw a dotted line connecting each "
             "model's dots on the bar plot (default: none). Use 'all' for "
             "every language.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.plots)
    if "all" in selected:
        selected = {"bar", "steps", "steps_by_lang", "scatter", "hardware",
                    "tokens", "tokens_grouped", "latency", "beeswarm"}

    trace_languages = (
        list(LANGUAGES) if "all" in args.trace_languages else args.trace_languages
    )

    exploded = build_exploded_df()
    agg = load_data(exploded)

    if "bar" in selected:
        print(f"[INFO] wrote {plot_default_models_barplot(agg, args.out_dir, trace_languages=trace_languages)}")
    if "steps" in selected:
        print(f"[INFO] wrote {plot_best_accuracy_over_time_by_size(agg, args.out_dir)}")
    if "steps_by_lang" in selected:
        print(f"[INFO] wrote {plot_best_accuracy_over_time_by_language(agg, args.out_dir)}")
    if "scatter" in selected:
        print(f"[INFO] wrote {plot_accuracy_cost_with_frontiers(agg, args.out_dir)}")
    if "hardware" in selected:
        print(f"[INFO] wrote {plot_accessibility_frontier(agg, args.out_dir)}")
        print(f"[INFO] wrote {plot_accessibility_frontier(agg, args.out_dir, fab_style=True)}")
    if "tokens" in selected:
        print(f"[INFO] wrote {plot_tokens_per_language(exploded, args.out_dir)}")
    if "tokens_grouped" in selected:
        print(f"[INFO] wrote {plot_tokens_grouped_barplot(exploded, args.out_dir)}")
    if "latency" in selected:
        print(f"[INFO] wrote {plot_latency_per_language(exploded, args.out_dir)}")
    if "beeswarm" in selected:
        print(f"[INFO] wrote {plot_beeswarm_accuracy(agg, args.out_dir, color_by='language')}")
        print(f"[INFO] wrote {plot_beeswarm_accuracy(agg, args.out_dir, color_by='provider')}")
        print(f"[INFO] wrote {plot_beeswarm_accuracy(agg, args.out_dir, color_by='hardware_tier')}")
if __name__ == "__main__":
    main()
