"""
translation_experiment_analysis.py

Translation-model experiment: compares model accuracy
  1. depending on the model used to translate the items: the original `<Lang>_ep`
     translations (Gemini 2.5 Flash) vs the `<Lang>_<slug>_ep` retranslations
     (e.g. argon4low = gemini-4-argon-low), see `--translator-slug` in translate_benchmark.py;
  2. depending on the number of items: the full test set (899 questions) vs the 300-question
     core subset listed in data/final_300_original_question_indices.csv.

Scores are computed per question from the cache files
(data/cache_local/CDPK_<Slug>_<category>/resps_<model>.csv), so no rerun is needed. Cache
files have no question id: row i is row i of the category's test CSV, i.e. the category's
rows of the English source minus its 3 few-shot rows, in source order (split_category in
prepare_cdpk_dataset_multilingual.py). Each cache file is checked against the English answer
key before use, and skipped with a warning if it does not match (truncated or old run) or if
more than 5% of its API calls failed (success = False: errors, not model answers).

Translators are compared on the same questions: for each model and language, only the
categories run on every translation are used. Models with fewer than 7 such categories are
kept and marked "partial" (n_categories in the tables, "*" in the figures).

Outputs (in --out-dir):
  scores.csv            accuracy per model, language, translator, subset (full / core300)
  delta_translator.csv  retranslation - original, paired bootstrap 95% CI, flipped questions
  delta_subset.csv      core300 - full
  rank_stability.csv    Kendall tau between model rankings (complete models only)
  figures (CLI flag -> PNG):
    dumbbell  -> accuracy_by_translator.png         accuracy per translator, one panel per language
    dumbbell_core300 -> accuracy_by_translator_core300.png  same, on the core300 subset
    delta     -> delta_translator_ci.png            retranslation - original with 95% CI
    scatter   -> original_vs_retranslation_scatter.png
    subset    -> full_vs_core300_scatter.png        full vs core300 accuracy, all models and languages
    subset_delta -> delta_core300.png               core300 - full per model (English as reference)
    heatmap   -> heatmap_translation_experiment.png all scores

Usage:
    uv run python scripts/translation_experiment_analysis.py
    uv run python scripts/translation_experiment_analysis.py --languages Arabic Hausa --plots delta subset
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cdpk.benchmark_answers import clean_answers, clean_resps
from cdpk.benchmark_constants import CACHE_GLOBAL_DIR, CACHE_LOCAL_DIR
from prepare_cdpk_dataset_multilingual import DATASETS, FULL_DATASETS_DIR, category_slug, load_few_shot_positions

DATA_DIR = ROOT / "data"
LOG_FILE = DATA_DIR / "translation_log.json"
MODELS_DIR = ROOT / "configs" / "models"
DATASET = "cdpk"
N_CATEGORIES = len(DATASETS[DATASET]["categories"])

ORIGINAL = "original"  # <Lang>_ep, the first translation
SOURCE = "source"      # English, only shown as a reference in the core300 - full plot
SUBSETS = ["full", "core300"]
MAX_FAILED_CALLS = 0.05  # cache files with more failed API calls are skipped (not model answers)
N_BOOT = 2000
SEED = 0
PLOTS = ["dumbbell", "dumbbell_core300", "delta", "scatter", "subset", "subset_delta", "heatmap"]

# Okabe-Ito, colourblind-safe; translators keep their colour whatever the CLI order
TRANSLATOR_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
ENGLISH_COLOR = "#7f7f7f"
LANGUAGE_MARKERS = ["o", "s", "^", "D", "v", "P"]


# ==================== Loading ====================

def cache_slug(language, translator):
    """Slug of the question configs, e.g. Arabic_ep or Arabic_argon4low_ep."""
    if translator == SOURCE:
        return language
    if translator == ORIGINAL:
        return f"{language}_ep"
    return f"{language}_{translator}_ep"


def translator_labels(languages, translators):
    """Translation model of each translator, from translation_log.json (slug if not logged)."""
    log = json.loads(LOG_FILE.read_text(encoding="utf-8")) if LOG_FILE.exists() else {}
    labels = {}
    for t in translators:
        suffix = "" if t == ORIGINAL else f"_{t}"
        models = {log.get(f"pedagogy_benchmark_{lang.lower()}{suffix}_{DATASET}", {})
                  .get("translation", {}).get("model") for lang in languages} - {None}
        labels[t] = f"{t} ({', '.join(sorted(models))})" if models else t
    return labels


def question_ids_by_category():
    """Question ids and answers of each category's test set, in the order of the test CSVs."""
    english = pd.read_csv(FULL_DATASETS_DIR / f"{DATASETS[DATASET]['english_file']}.csv")
    offset = DATASETS[DATASET]["id_offset"]
    few_shot_ids = {offset + pos for positions in load_few_shot_positions(DATASET).values() for pos in positions}
    test = english[~english["question_id"].isin(few_shot_ids)]
    return {category_slug(c): test.loc[test["category"] == c, ["question_id", "correct_answer"]].reset_index(drop=True)
            for c in DATASETS[DATASET]["categories"]}


def load_subset_ids(subset_file):
    ids = pd.read_csv(subset_file)["original_question_id"]
    if ids.duplicated().any():
        raise ValueError(f"{subset_file} has duplicated question ids")
    return set(ids)


def load_answers(models, plan, qids, subset_ids):
    """One row per (model, language, translator, category, question_id) with correct / bad_format.

    plan: {language: [translators]}. Like the runner, the local cache overrides the global one.
    """
    frames = []
    for language, translators in plan.items():
        for translator in translators:
            slug = cache_slug(language, translator)
            for category, ref in qids.items():
                for model in models:
                    name = Path(f"CDPK_{slug}_{category}") / f"resps_{model}.csv"
                    found = [d / name for d in (CACHE_LOCAL_DIR, CACHE_GLOBAL_DIR) if (d / name).exists()]
                    if not found:
                        continue
                    df = pd.read_csv(found[0])
                    answers = df["answers"].astype(str).map(clean_answers)
                    expected = ref["correct_answer"].astype(str).map(clean_answers)
                    if len(df) != len(ref) or (answers.to_numpy() != expected.to_numpy()).any():
                        warnings.warn(f"Skipped {found[0]}: {len(df)} rows, expected {len(ref)}, "
                                      "or answers that differ from the English answer key")
                        continue
                    failed = ~df["success"].astype(bool) if "success" in df else pd.Series(False, index=df.index)
                    if failed.mean() > MAX_FAILED_CALLS:
                        warnings.warn(f"Skipped {found[0]}: {failed.mean():.0%} of the API calls failed")
                        continue
                    resps = df["resps"].map(clean_resps)
                    frames.append(pd.DataFrame({
                        "model": model, "language": language, "translator": translator,
                        "category": category, "question_id": ref["question_id"].to_numpy(),
                        "correct": (resps == answers).to_numpy(),
                        "bad_format": resps.isna().to_numpy(),
                    }))
    if not frames:
        sys.exit("No cache file found for these models and languages.")
    answers = pd.concat(frames, ignore_index=True)
    answers["in_core300"] = answers["question_id"].isin(subset_ids)
    return answers


def keep_paired_categories(answers, plan):
    """Keep, per model and language, the categories run on every translator of that language."""
    n_runs = answers.groupby(["model", "language", "category"])["translator"].nunique().reset_index()
    required = n_runs["language"].map({lang: len(ts) for lang, ts in plan.items()})
    paired = n_runs.loc[n_runs["translator"] == required, ["model", "language", "category"]]
    coverage = paired.groupby(["model", "language"]).size().rename("n_categories").reset_index()
    coverage["complete"] = coverage["n_categories"] == N_CATEGORIES
    return answers.merge(paired, on=["model", "language", "category"]), coverage


# ==================== Scores and statistics ====================

def subset_rows(answers, subset):
    return answers if subset == "full" else answers[answers["in_core300"]]


def compute_scores(answers, coverage):
    scores = []
    for subset in SUBSETS:
        s = (subset_rows(answers, subset).groupby(["model", "language", "translator"])
             .agg(accuracy=("correct", "mean"), bad_format=("bad_format", "mean"), n_questions=("correct", "size"))
             .reset_index())
        s[["accuracy", "bad_format"]] *= 100
        s["subset"] = subset
        scores.append(s)
    return pd.concat(scores, ignore_index=True).merge(coverage, on=["model", "language"])


def bootstrap_ci(values, rng):
    """95% CI of the mean of `values`, resampling its rows (values: n x k, mean of col 0 - col 1 if k = 2)."""
    idx = rng.integers(0, len(values), (N_BOOT, len(values)))
    return np.percentile(values[idx].mean(axis=1) * 100, [2.5, 97.5])


def delta_translator(answers, coverage, plan):
    """Retranslation - original accuracy, paired over questions."""
    rng = np.random.default_rng(SEED)
    rows = []
    for (model, language), df in answers.groupby(["model", "language"]):
        for translator in plan[language]:
            if translator == ORIGINAL:
                continue
            for subset in SUBSETS:
                wide = subset_rows(df, subset).pivot(index="question_id", columns="translator", values="correct")
                new, old = wide[translator].astype(int).to_numpy(), wide[ORIGINAL].astype(int).to_numpy()
                low, high = bootstrap_ci(new - old, rng)
                rows.append({"model": model, "language": language, "translator": translator, "subset": subset,
                             "delta": (new - old).mean() * 100, "ci_low": low, "ci_high": high,
                             "n_questions": len(wide), "right_to_wrong": int(((old == 1) & (new == 0)).sum()),
                             "wrong_to_right": int(((old == 0) & (new == 1)).sum())})
    return pd.DataFrame(rows).merge(coverage, on=["model", "language"]) if rows else pd.DataFrame()


def delta_subset(answers, coverage):
    """Core300 - full accuracy. No CI: both question sets are fixed."""
    rows = []
    for (model, language, translator), df in answers.groupby(["model", "language", "translator"]):
        core = df["in_core300"]
        if not core.any():
            continue
        rows.append({"model": model, "language": language, "translator": translator,
                     "delta": (df.loc[core, "correct"].mean() - df["correct"].mean()) * 100,
                     "n_full": len(df), "n_core300": int(core.sum())})
    return pd.DataFrame(rows).merge(coverage, on=["model", "language"])


def rank_stability(scores):
    """Kendall tau between rankings: original vs each retranslation, and full vs core300."""
    complete = scores[scores["complete"]]
    rows = []

    def add(comparison, language, condition, wide, a, b):
        wide = wide[[a, b]].dropna() if {a, b} <= set(wide.columns) else wide.iloc[:0]
        tau, p = kendalltau(wide[a], wide[b]) if len(wide) >= 3 else (np.nan, np.nan)
        rows.append({"comparison": comparison, "language": language, "condition": condition,
                     "n_models": len(wide), "kendall_tau": tau, "p_value": p})

    for (language, subset), df in complete.groupby(["language", "subset"]):
        wide = df.pivot(index="model", columns="translator", values="accuracy")
        for translator in wide.columns.difference([ORIGINAL]):
            add(f"{ORIGINAL} vs {translator}", language, subset, wide, ORIGINAL, translator)
    for (language, translator), df in complete.groupby(["language", "translator"]):
        add("full vs core300", language, translator,
            df.pivot(index="model", columns="subset", values="accuracy"), "full", "core300")
    return pd.DataFrame(rows)


# ==================== Figures ====================

def model_order(scores):
    """Models sorted by mean original full-set accuracy, best first."""
    full = scores[scores["subset"] == "full"]
    mean = full[full["translator"] == ORIGINAL].groupby("model")["accuracy"].mean()
    others = sorted(set(full["model"]) - set(mean.index))
    return list(mean.sort_values(ascending=False).index) + others


def style_model_ticks(ax, models, names):
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([names.get(m, m) for m in models])
    ax.invert_yaxis()


PARTIAL_NOTE = f"*: partial run, fewer than {N_CATEGORIES} categories in common (n/{N_CATEGORIES} shown)."


def footnote(fig, text):
    """Note at the bottom left, only if there is something to explain; returns the bottom margin."""
    if not text:
        return 0
    fig.text(0.01, 0.005, text, fontsize=8, color="#555555", ha="left", va="bottom")
    return 0.03


def finish_layout(fig, title, handles, note):
    """Centred title, legend stacked at the top right, panels starting right below the legend."""
    legend = fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, y=0.995, va="top", fontsize=14, fontweight="bold")
    bottom = footnote(fig, note)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    # Move the panels up until their titles sit just below the legend
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    legend_bottom = legend.get_window_extent().transformed(to_fig).y0
    titles_top = max(ax.title.get_window_extent().transformed(to_fig).y1 for ax in fig.axes)
    fig.subplots_adjust(top=fig.subplotpars.top + legend_bottom - titles_top - 0.01)


def plot_dumbbell(scores, ctx, subset="full"):
    languages, models = ctx["languages"], ctx["models"]
    full = scores[scores["subset"] == subset]
    fig, axes = plt.subplots(1, len(languages), figsize=(4.6 * len(languages) + 2, 0.3 * len(models) + 2),
                             sharey=True, sharex=True, squeeze=False)
    for ax, language in zip(axes[0], languages):
        df = full[full["language"] == language].set_index(["model", "translator"])
        for y, model in enumerate(models):
            points = [(t, df.loc[(model, t)]) for t in ctx["plan"][language] if (model, t) in df.index]
            if not points:
                continue
            xs = [p["accuracy"] for _, p in points]
            ax.plot([min(xs), max(xs)], [y, y], color="#bbbbbb", linewidth=2, zorder=1)
            for t, p in points:
                ax.scatter(p["accuracy"], y, s=55, color=ctx["colors"][t], edgecolor="white",
                           linewidth=1, zorder=3)
            if not points[0][1]["complete"]:
                ax.text(max(xs) + 0.8, y, f"* {points[0][1]['n_categories']}/{N_CATEGORIES}",
                        fontsize=7, color="#555555", va="center")
        ax.set_title(language, fontsize=12, fontweight="bold")
        ax.set_xlabel("Accuracy (%), full set" if subset == "full" else f"Accuracy (%), {subset} subset")
        ax.grid(axis="x", color="#e5e5e5")
        ax.set_axisbelow(True)
    style_model_ticks(axes[0][0], models, ctx["names"])
    handles = [mlines.Line2D([], [], marker="o", linestyle="", color=ctx["colors"][t], label=ctx["labels"][t])
               for t in ctx["translators"]]
    partial = not full.loc[full["language"].isin(languages), "complete"].all()
    title = "Accuracy depending on the translation model"
    if subset != "full":
        title += f" ({subset} subset)"
    finish_layout(fig, title, handles, PARTIAL_NOTE if partial else "")
    return fig


def plot_delta_ci(deltas, ctx, title, xlabel, group_col, styles):
    """Forest plot, one panel per language; `styles` maps each value of group_col to (color, filled)."""
    languages = [lang for lang in ctx["languages"] if lang in set(deltas["language"])]
    models = [m for m in ctx["models"] if m in set(deltas["model"])]
    fig, axes = plt.subplots(1, len(languages), figsize=(4.2 * len(languages) + 2, 0.38 * len(models) + 2),
                             sharey=True, sharex=True, squeeze=False)
    offsets = dict(zip(styles, np.linspace(-0.2, 0.2, len(styles)) if len(styles) > 1 else [0]))
    for ax, language in zip(axes[0], languages):
        df = deltas[deltas["language"] == language]
        ax.axvline(0, color="#444444", linewidth=1)
        for _, r in df.iterrows():
            if r["model"] not in models:
                continue
            color, filled = styles[r[group_col]]
            y = models.index(r["model"]) + offsets[r[group_col]]
            ax.errorbar(r["delta"], y, xerr=[[r["delta"] - r["ci_low"]], [r["ci_high"] - r["delta"]]],
                        fmt="o", color=color, markerfacecolor=color if filled else "white",
                        markersize=6, elinewidth=1.5, capsize=0)
            if not r["complete"]:
                ax.text(r["ci_high"] + 0.5, y, f"* {r['n_categories']}/{N_CATEGORIES}", fontsize=7,
                        color="#555555", va="center")
        ax.set_title(language, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#e5e5e5")
        ax.set_axisbelow(True)
    style_model_ticks(axes[0][0], models, ctx["names"])
    handles = [mlines.Line2D([], [], marker="o", color=c, markerfacecolor=c if f else "white", label=label)
               for label, (c, f) in styles.items()]
    finish_layout(fig, title, handles, "" if deltas["complete"].all() else PARTIAL_NOTE)
    return fig


def plot_delta(d_translator, ctx):
    if d_translator.empty:
        return None
    d = d_translator.copy()
    d["group"] = d["translator"] + ", " + d["subset"]
    styles = {f"{t}, {s}": (ctx["colors"][t], s == "full")
              for t in ctx["translators"] if t != ORIGINAL for s in SUBSETS}
    return plot_delta_ci(d, ctx, "Accuracy change with the retranslated items (95% CI, paired bootstrap)",
                         "Retranslation - original (points)", "group", styles)


def plot_scatter(scores, rank, ctx):
    full = scores[scores["subset"] == "full"]
    retranslations = [t for t in ctx["translators"] if t != ORIGINAL]
    languages = ctx["languages"]
    if not retranslations:
        return None
    fig, axes = plt.subplots(1, len(languages), figsize=(5 * len(languages), 5.2), squeeze=False)
    for ax, language in zip(axes[0], languages):
        wide = full[full["language"] == language].pivot(index="model", columns="translator", values="accuracy")
        complete = full[full["language"] == language].groupby("model")["complete"].first()
        titles = [language]
        for t in retranslations:
            if t not in wide.columns or ORIGINAL not in wide.columns:
                continue
            pts = wide[[ORIGINAL, t]].dropna()
            ax.scatter(pts[ORIGINAL], pts[t], s=45, color=ctx["colors"][t], edgecolor="white", zorder=3)
            for model, (x, y) in pts.iterrows():
                star = "" if complete[model] else "*"
                ax.annotate(ctx["names"].get(model, model) + star, (x, y), fontsize=6.5, color="#333333",
                            xytext=(3, 2), textcoords="offset points")
            r = rank[(rank["comparison"] == f"{ORIGINAL} vs {t}") & (rank["language"] == language)
                     & (rank["condition"] == "full")]
            if not r.empty and not np.isnan(r["kendall_tau"].iloc[0]):
                titles.append(f"Kendall τ = {r['kendall_tau'].iloc[0]:.2f} (n = {r['n_models'].iloc[0]})")
        lims = [np.nanmin(wide.to_numpy()) - 3, np.nanmax(wide.to_numpy()) + 3] if wide.size else [0, 100]
        ax.plot(lims, lims, color="#888888", linestyle="--", linewidth=1, zorder=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_title("\n".join(titles), fontsize=11)
        ax.set_xlabel(f"Accuracy (%), {ctx['labels'][ORIGINAL]}", fontsize=9)
        ax.set_ylabel(f"Accuracy (%), {' / '.join(ctx['labels'][t] for t in retranslations)}", fontsize=9)
        ax.grid(color="#eeeeee")
    fig.suptitle("Original vs retranslated items, full set (above the dashed line: higher on the retranslation)",
                 fontsize=13, fontweight="bold")
    partial = not full.loc[full["language"].isin(languages), "complete"].all()
    bottom = footnote(fig, "*: partial run. τ computed on complete models only." if partial else "")
    fig.tight_layout(rect=(0, bottom + 0.02 if bottom else 0, 1, 0.95))
    return fig


def plot_subset_scatter(scores, rank, ctx):
    wide = scores.pivot_table(index=["model", "language", "translator"], columns="subset",
                              values="accuracy").dropna().reset_index()
    languages = ctx["languages"]
    fig, ax = plt.subplots(figsize=(7, 7))
    markers = dict(zip(languages, LANGUAGE_MARKERS))
    for _, r in wide.iterrows():
        ax.scatter(r["full"], r["core300"], s=35, marker=markers[r["language"]],
                   color=ctx["colors"][r["translator"]], edgecolor="white", alpha=0.9)
    lims = [wide[["full", "core300"]].min().min() - 3, wide[["full", "core300"]].max().max() + 3]
    ax.plot(lims, lims, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Accuracy (%), full set (899)")
    ax.set_ylabel("Accuracy (%), core300")
    taus = sorted({round(t, 2) for t in rank.loc[rank["comparison"] == "full vs core300", "kendall_tau"].dropna()})
    tau_text = "" if not taus else f"; Kendall τ {taus[0]:.2f}" + (f" to {taus[-1]:.2f}" if len(taus) > 1 else "")
    ax.set_title("Accuracy on the full set vs the core300 subset\n(one point per model, language and translation"
                 + tau_text + ")", fontsize=11)
    ax.grid(color="#eeeeee")
    handles = [mlines.Line2D([], [], marker="o", linestyle="", color=ctx["colors"][t], label=ctx["labels"][t])
               for t in ctx["translators"]]
    handles += [mlines.Line2D([], [], marker=markers[lang], linestyle="", color="#555555", label=lang)
                for lang in languages]
    ax.legend(handles=handles, fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def plot_subset_delta(d_subset, d_english, ctx):
    """Core300 - full per language, one row per model with the translations joined by a grey line,
    and English added on the right as a reference."""
    colors = ctx["colors"] | {SOURCE: ENGLISH_COLOR}
    languages = [lang for lang in ctx["languages"] if lang in set(d_subset["language"])]
    if not d_english.empty:
        languages.append("English")
    d = pd.concat([d_subset, d_english], ignore_index=True)
    models = [m for m in ctx["models"] if m in set(d["model"])]
    fig, axes = plt.subplots(1, len(languages), figsize=(4.2 * len(languages) + 2, 0.3 * len(models) + 2),
                             sharey=True, sharex=True, squeeze=False)
    for ax, language in zip(axes[0], languages):
        df = d[d["language"] == language]
        ax.axvline(0, color="#444444", linewidth=1)
        for model, points in df.groupby("model"):
            if model not in models:
                continue
            y = models.index(model)
            ax.plot([points["delta"].min(), points["delta"].max()], [y, y], color="#bbbbbb", linewidth=2, zorder=1)
            for _, r in points.iterrows():
                ax.scatter(r["delta"], y, s=55, color=colors[r["translator"]], edgecolor="white",
                           linewidth=1, zorder=3)
            if not points["complete"].iloc[0]:
                ax.text(points["delta"].max() + 0.6, y, f"* {points['n_categories'].iloc[0]}/{N_CATEGORIES}",
                        fontsize=7, color="#555555", va="center")
        ax.set_title(language, fontsize=12, fontweight="bold")
        ax.set_xlabel("Core300 - full (points)")
        ax.grid(axis="x", color="#e5e5e5")
        ax.set_axisbelow(True)
    style_model_ticks(axes[0][0], models, ctx["names"])
    labels = ctx["labels"] | {SOURCE: "English (source)"}
    shown = [*ctx["translators"], *([SOURCE] if not d_english.empty else [])]
    handles = [mlines.Line2D([], [], marker="o", linestyle="", color=colors[t], label=labels[t]) for t in shown]
    finish_layout(fig, "Accuracy on the 300-question core subset vs the full set", handles,
                  "" if d["complete"].all() else PARTIAL_NOTE)
    return fig


def plot_heatmap(scores, ctx):
    columns = [(lang, t, s) for lang in ctx["languages"] for t in ctx["plan"][lang] for s in SUBSETS]
    acc = scores.pivot_table(index="model", columns=["language", "translator", "subset"], values="accuracy")
    acc = acc.reindex(index=ctx["models"], columns=pd.MultiIndex.from_tuples(columns))
    complete = (scores.pivot_table(index="model", columns=["language", "translator", "subset"],
                                   values="complete", aggfunc="first")
                .reindex(index=ctx["models"], columns=acc.columns))
    annot = acc.map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    annot = annot.where(complete.fillna(True).astype(bool), annot + "*")
    fig, ax = plt.subplots(figsize=(0.75 * len(columns) + 3, 0.42 * len(ctx["models"]) + 2.5))
    ax.set_facecolor("#d9d9d9")
    values = acc.to_numpy(dtype=float)
    sns.heatmap(acc, annot=annot, fmt="", cmap="RdYlGn", vmin=np.nanpercentile(values, 5),
                vmax=np.nanpercentile(values, 95), linewidths=1, linecolor="white",
                cbar_kws={"label": "Accuracy (%)"}, annot_kws={"fontsize": 7.5}, ax=ax)
    ax.set_xticklabels([f"{lang}\n{t}\n{s}" for lang, t, s in columns],
                       rotation=0, fontsize=7.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels([ctx["names"].get(m, m) for m in ctx["models"]], rotation=0)
    ax.set_title("Accuracy by language, translation model and question set (grey: not run)",
                 fontsize=13, fontweight="bold", pad=12)
    bottom = footnote(fig, "" if complete.fillna(True).astype(bool).all().all() else PARTIAL_NOTE)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig


# ==================== Main ====================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", nargs="+", default=["Arabic", "Hausa", "Luganda"],
                   help="Language slugs without _ep, as in data/cache_local (default: Arabic Hausa Luganda)")
    p.add_argument("--translator-slugs", nargs="+", default=["argon4low"],
                   help="Retranslations compared with the original <Lang>_ep (default: argon4low)")
    p.add_argument("--models-config", default="translation_model_experiment",
                   help="Model list in configs/models/ (default: translation_model_experiment)")
    p.add_argument("--subset-file", type=Path, default=DATA_DIR / "final_300_original_question_indices.csv",
                   help="CSV with an original_question_id column (0-based English question_id)")
    p.add_argument("--out-dir", type=Path, default=DATA_DIR / "results" / "figures" / "translation_experiment")
    p.add_argument("--plots", nargs="+", default=["all"], choices=["all", *PLOTS])
    return p.parse_args()


def main():
    args = parse_args()
    names = yaml.safe_load((MODELS_DIR / f"{args.models_config}.yaml").read_text(encoding="utf-8"))
    models = list(names)
    translators = [ORIGINAL, *args.translator_slugs]
    plan = {lang: translators for lang in args.languages}

    subset_ids = load_subset_ids(args.subset_file)
    answers = load_answers(models, plan, question_ids_by_category(), subset_ids)
    answers, coverage = keep_paired_categories(answers, plan)

    scores = compute_scores(answers, coverage)
    d_translator = delta_translator(answers, coverage, plan)
    d_subset = delta_subset(answers, coverage)
    rank = rank_stability(scores)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("scores", scores), ("delta_translator", d_translator),
                     ("delta_subset", d_subset), ("rank_stability", rank)]:
        df.to_csv(args.out_dir / f"{name}.csv", index=False)

    print(f"Paired categories per model (out of {N_CATEGORIES}):")
    print(coverage.pivot(index="model", columns="language", values="n_categories")
          .reindex(models).dropna(how="all").fillna(0).astype(int).to_markdown())
    if not d_translator.empty:
        print("\nRetranslation - original (points, 95% CI):")
        d = d_translator.assign(ci=lambda x: x.apply(lambda r: f"{r.delta:+.1f} [{r.ci_low:+.1f}, {r.ci_high:+.1f}]",
                                                    axis=1))
        print(d.pivot(index="model", columns=["language", "subset"], values="ci").reindex(models)
              .dropna(how="all").to_markdown())
    print("\nRank stability (Kendall tau, complete models):")
    print(rank.round(3).to_markdown(index=False))

    ctx = {"languages": args.languages, "plan": plan, "translators": translators,
           "names": names, "models": model_order(scores), "labels": translator_labels(args.languages, translators),
           "colors": dict(zip(translators, TRANSLATOR_COLORS))}

    def english_delta_subset():
        english_plan = {"English": [SOURCE]}
        english, english_coverage = keep_paired_categories(
            load_answers(models, english_plan, question_ids_by_category(), subset_ids), english_plan)
        return delta_subset(english, english_coverage)

    figures = {
        "dumbbell": ("accuracy_by_translator.png", lambda: plot_dumbbell(scores, ctx)),
        "dumbbell_core300": ("accuracy_by_translator_core300.png", lambda: plot_dumbbell(scores, ctx, "core300")),
        "delta": ("delta_translator_ci.png", lambda: plot_delta(d_translator, ctx)),
        "scatter": ("original_vs_retranslation_scatter.png", lambda: plot_scatter(scores, rank, ctx)),
        "subset": ("full_vs_core300_scatter.png", lambda: plot_subset_scatter(scores, rank, ctx)),
        "subset_delta": ("delta_core300.png", lambda: plot_subset_delta(d_subset, english_delta_subset(), ctx)),
        "heatmap": ("heatmap_translation_experiment.png", lambda: plot_heatmap(scores, ctx)),
    }
    selected = PLOTS if "all" in args.plots else args.plots
    for key in selected:
        filename, make = figures[key]
        fig = make()
        if fig is None:
            print(f"Skipped {key}: no retranslation scores")
            continue
        fig.savefig(args.out_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {args.out_dir / filename}")


if __name__ == "__main__":
    main()
