"""
create_barplots.py

CLI script to plot model performance bar plots for a given language benchmark.

Usage:
    python scripts/create_barplots.py Arabic_ep
    python scripts/create_barplots.py Dari_ep
    python scripts/create_barplots.py English
    python scripts/create_barplots.py Hausa_ep --out-dir path/to/output

The language argument is the name of the results sub-folder
(e.g. Arabic_ep, Dari_ep, English, Hausa_ep, …).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "data" / "results"
CONFIGS_DIR = ROOT / "fab-benchmarks-configs"

DEFAULT_COLOR = "#888888"

CATEGORY_ORDER = [
    "Science",
    "Maths",
    "Literacy",
    "Technology",
    "Social studies",
    "Creative arts",
    "General",
]

BAR_WIDTH = 0.65


def apply_barplot_style(ax, title):
    """Apply the shared clean barplot style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)


def provider_legend_handles(models, model_provider_map, provider_color_map):
    providers_seen = {model_provider_map.get(m, "") for m in models}
    return [
        mlines.Line2D([], [], marker="o", color="w",
                      markerfacecolor=provider_color_map.get(p, DEFAULT_COLOR),
                      markersize=9, label=p)
        for p in sorted(providers_seen) if p
    ]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_configs():
    """Return (display_name_map, model_provider_map, provider_color_map)."""
    models_df = pd.read_csv(CONFIGS_DIR / "models.csv")
    providers_df = pd.read_csv(CONFIGS_DIR / "providers.csv")[["provider", "color"]]

    display_name_map = dict(zip(models_df["model_id"], models_df["display_name"]))
    model_provider_map = dict(zip(models_df["model_id"], models_df["provider"]))
    provider_color_map = dict(zip(providers_df["provider"], providers_df["color"]))

    return display_name_map, model_provider_map, provider_color_map


def get_lang_label(lang_arg: str) -> str:
    if lang_arg.endswith("_ep"):
        return f"{lang_arg[:-3]} (English prompt)"
    return lang_arg


def find_result_file(lang_arg: str, kind: str) -> Path:
    """Find cdpk_results_<kind>_*.csv in data/results/<lang_arg>/."""
    folder = RESULTS_DIR / lang_arg
    if not folder.exists():
        sys.exit(f"[ERROR] Results folder not found: {folder}")
    matches = sorted(folder.glob(f"cdpk_results_{kind}_*.csv"))
    if not matches:
        sys.exit(f"[ERROR] No '{kind}' CSV found in {folder}")
    return matches[-1]  # most recent by filename


def load_result_csv(path: Path) -> pd.DataFrame:
    """Load accuracy/bad_format CSV (first col = model_id index, rest = categories)."""
    df = pd.read_csv(path, index_col=0)
    df.index.name = "model_id"
    return df


# ---------------------------------------------------------------------------
# Labelling / colouring helpers
# ---------------------------------------------------------------------------

DISPLAY_NAME_OVERRIDES = {
    "hf-gemma-3-1b-it": "Gemma-3 1B",
}


def make_display_labels(models, display_name_map):
    return [DISPLAY_NAME_OVERRIDES.get(m) or display_name_map.get(m, m) for m in models]


def make_bar_colors(models, model_provider_map, provider_color_map):
    return [
        provider_color_map.get(model_provider_map.get(m, ""), DEFAULT_COLOR)
        for m in models
    ]




# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _draw_barplot(series, models, x_labels, colors, ylabel, title,
                  model_provider_map, provider_color_map, label_fmt, y_pad,
                  out_path):
    """Shared rendering for accuracy and bad-format barplots."""
    values = series.values
    n = len(models)

    _, ax = plt.subplots(figsize=(max(18, n * 0.52), 6))

    bars = ax.bar(range(n), values, width=BAR_WIDTH,
                  color=colors, edgecolor="none", zorder=3)

    # Integer value labels above each bar
    for bar, val in zip(bars, values):
        if val > 0 or label_fmt != "{:.0f}":
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_pad,
                label_fmt.format(val),
                ha="center", va="bottom", fontsize=12, fontweight="bold",
                rotation=0, color="#333333",
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=12, color="#333333")
    ax.set_ylabel(ylabel, fontsize=16, color="#333333")
    ax.set_ylim(0, min(values.max() + 14, 102) if ylabel.startswith("Acc") else max(6, values.max() + 5))
    ax.tick_params(axis="y", labelsize=13, labelcolor="#333333")

    apply_barplot_style(ax, title)

    ax.legend(
        handles=provider_legend_handles(models, model_provider_map, provider_color_map),
        title="", title_fontsize=9,
        bbox_to_anchor=(1.01, 0.75), loc="upper left", fontsize=12,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_accuracy_barplot(acc_df, label, lang_arg,
                          display_name_map, model_provider_map, provider_color_map,
                          out_dir):
    """Bar plot of Overall accuracy per model, sorted highest → lowest."""
    series = acc_df["Overall"].dropna().sort_values(ascending=False)
    models = series.index.tolist()
    x_labels = make_display_labels(models, display_name_map)
    colors = make_bar_colors(models, model_provider_map, provider_color_map)
    out_path = out_dir / f"accuracy_barplot_{lang_arg}.svg"
    _draw_barplot(series, models, x_labels, colors,
                  ylabel="Accuracy (%)",
                  title=f"Pedagogical Knowledge Benchmark Performance — {label}",
                  model_provider_map=model_provider_map,
                  provider_color_map=provider_color_map,
                  label_fmt="{:.0f}",
                  y_pad=0.5,
                  out_path=out_path)


def plot_bad_format_barplot(bf_df, label, lang_arg,
                            display_name_map, model_provider_map, provider_color_map,
                            out_dir):
    """Bar plot of Overall bad-format % per model, sorted highest → lowest."""
    series = bf_df["Overall"].dropna().sort_values(ascending=False)
    models = series.index.tolist()
    x_labels = make_display_labels(models, display_name_map)
    colors = make_bar_colors(models, model_provider_map, provider_color_map)
    out_path = out_dir / f"bad_format_barplot_{lang_arg}.svg"
    _draw_barplot(series, models, x_labels, colors,
                  ylabel="Bad Format (%)",
                  title=f"Bad Format Rate per Model — {label}",
                  model_provider_map=model_provider_map,
                  provider_color_map=provider_color_map,
                  label_fmt="{:.1f}",
                  y_pad=0.2,
                  out_path=out_path)


def plot_category_heatmap(acc_df, label, lang_arg, display_name_map, out_dir):
    """Heatmap: model (y, sorted by Overall desc) × category (x)."""
    # Column order: predefined first, then any extras, Overall excluded
    cat_cols = [c for c in CATEGORY_ORDER if c in acc_df.columns]
    cat_cols += [c for c in acc_df.columns if c not in CATEGORY_ORDER and c != "Overall"]

    # Sort models top → bottom by Overall accuracy
    models_sorted = acc_df["Overall"].dropna().sort_values(ascending=False).index.tolist()

    pivot = acc_df.reindex(index=models_sorted)[cat_cols]
    pivot.index = make_display_labels(models_sorted, display_name_map)

    n_models, n_cats = pivot.shape
    fig, ax = plt.subplots(figsize=(max(13, n_cats * 1.5), max(6, n_models * 0.25)))

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 9},
        cmap="RdYlGn",
        linewidths=0.5,
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)"},
    )

    ax.set_title(f"Accuracy by Category — {label}", fontsize=14, pad=10)
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    #plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    out_path = out_dir / f"category_heatmap_{lang_arg}.svg"
    plt.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy barplots and category heatmap for a language benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "language",
        help="Results sub-folder name, e.g. 'Arabic_ep', 'Dari_ep', 'English', 'Hausa_ep'.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "figures"),
        help="Directory to save figures (default: data/results/figures/).",
    )
    args = parser.parse_args()

    lang_arg = args.language
    label = get_lang_label(lang_arg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Language folder : {lang_arg}  |  label = '{label}'")
    print(f"Output dir      : {out_dir}\n")

    display_name_map, model_provider_map, provider_color_map = load_configs()

    acc_path = find_result_file(lang_arg, "accuracy")
    bf_path = find_result_file(lang_arg, "bad_format")
    print(f"Accuracy file   : {acc_path.name}")
    print(f"Bad format file : {bf_path.name}\n")

    acc_df = load_result_csv(acc_path)
    bf_df = load_result_csv(bf_path)

    plot_accuracy_barplot(acc_df, label, lang_arg,
                          display_name_map, model_provider_map, provider_color_map, out_dir)
    plot_bad_format_barplot(bf_df, label, lang_arg,
                            display_name_map, model_provider_map, provider_color_map, out_dir)
    plot_category_heatmap(acc_df, label, lang_arg, display_name_map, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
