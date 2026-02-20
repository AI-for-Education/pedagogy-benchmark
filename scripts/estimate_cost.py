"""Estimate API cost of running models on a benchmark dataset.

Replicates the actual prompt structure used in the benchmark:
  1. Fixed template text (intro + instruction + final)
  2. N few-shot examples (question + choices + answer letter)
  3. The actual question to answer (question + choices)

Each API call's input = fixed_text + few_shot_examples + question_with_choices.
Words are converted to tokens via a configurable ratio, then multiplied by
per-model costs from the models registry.

Usage examples:
    # Basic usage with defaults (3 few-shot examples, MCQ output)
    python scripts/estimate_cost.py \
        --dataset data/pedagogy_benchmark_full_datasets/pedagogy_benchmark_cdpk.csv \
        --models-config configs/models/full_list_default_models_20260218.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_CSV = ROOT / "fab-benchmarks-configs" / "models.csv"

# Fixed prompt template text (from language_prompts.py english config + benchmark_answers.py)
DEFAULT_PROMPT_TEMPLATE = {
    "intro": "The following are example multiple choice questions (with answers).",
    "instruction": "Answer the following real question using same answer format:",
    "final": (
        "Only answer the real question.\n"
        "Only provide the letter for your answer.\n"
        "Stop exactly after the letter."
    ),
}

CHOICE_LABELS = ["A", "B", "C", "D"]
WORD_TOKEN_RATIO = 0.75  # Approximate tokens per word (varies by language/model)
MEAN_REASONING_TOKENS = 300  # Extra output tokens for reasoning models (chain-of-thought)


def count_words(text) -> int:
    """Count words in a string. Returns 0 for NaN / empty values."""
    if pd.isna(text):
        return 0
    return len(str(text).split())


def format_question_words(row, question_col: str, choice_cols: list[str]) -> int:
    """Count words in a formatted question block: question text + 'A. answer_a' etc."""
    words = count_words(row[question_col])
    for _, col in zip(CHOICE_LABELS, choice_cols):
        if not pd.isna(row[col]):
            # "A. <answer_text>" — the label itself is ~1 word
            words += 1 + count_words(row[col])
    return words


def format_example_words(row, question_col: str, choice_cols: list[str]) -> int:
    """Count words in a few-shot example: question + choices + answer letter."""
    # Same as a question block + 1 word for the answer letter (e.g. "D")
    return format_question_words(row, question_col, choice_cols) + 1


def load_models_registry(models_csv: Path) -> pd.DataFrame:
    """Load the models registry CSV and index by model_id."""
    df = pd.read_csv(models_csv)
    df = df.set_index("model_id")
    return df


def load_model_list(config_path: Path) -> dict[str, str]:
    """Load model IDs from a YAML config file. Returns {model_id: display_name}."""
    with open(config_path) as f:
        models = yaml.safe_load(f)
    return models


def estimate_costs(
    dataset_path: Path,
    question_col: str,
    choice_cols: list[str],
    num_examples: int,
    system_prompt: str,
    output_tokens: int,
    word_token_ratio: float,
    reasoning_tokens: int,
    models_config_path: Path | None,
    models_csv_path: Path,
):
    # ── Load dataset ──────────────────────────────────────────────────
    df = pd.read_csv(dataset_path)
    n_questions = len(df)

    # Validate columns exist
    all_cols = [question_col] + choice_cols
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        print(f"Error: columns not found in dataset: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # ── Fixed template words (same for every API call) ────────────────
    template = DEFAULT_PROMPT_TEMPLATE
    if system_prompt:
        fixed_words = count_words(system_prompt)
    else:
        fixed_words = (
            count_words(template["intro"])
            + count_words(template["instruction"])
            + count_words(template["final"])
        )

    # ── Per-row word counts (question + choices) ──────────────────────
    df["_question_words"] = df.apply(
        lambda row: format_question_words(row, question_col, choice_cols), axis=1
    )

    # ── Few-shot example words ────────────────────────────────────────
    # In practice, examples come from a dev set. We approximate by using
    # the average formatted-example size from this dataset.
    df["_example_words"] = df.apply(
        lambda row: format_example_words(row, question_col, choice_cols), axis=1
    )
    avg_example_words = df["_example_words"].mean()
    fewshot_words = avg_example_words * num_examples  # per API call

    # ── Total input words ─────────────────────────────────────────────
    # Each API call = fixed_text + few_shot_examples + question_with_choices
    df["_total_input_words"] = fixed_words + fewshot_words + df["_question_words"]
    total_input_words = df["_total_input_words"].sum()
    avg_input_words = total_input_words / n_questions

    # Convert to tokens
    total_input_tokens = total_input_words * word_token_ratio
    total_output_tokens = output_tokens * n_questions

    # ── Load model costs ──────────────────────────────────────────────
    registry = load_models_registry(models_csv_path)

    if models_config_path:
        model_names = load_model_list(models_config_path)  # {model_id: display_name}
    else:
        # Use all models from registry; fall back to model_id as display name
        model_names = {mid: mid for mid in registry.index.tolist()}

    # ── Compute per-model costs ───────────────────────────────────────
    results = []
    for model_id, yaml_name in model_names.items():
        if model_id not in registry.index:
            results.append(
                {
                    "model_id": model_id,
                    "display_name": yaml_name,
                    "input_cost_per_M": None,
                    "output_cost_per_M": None,
                    "estimated_input_cost": None,
                    "estimated_output_cost": None,
                    "estimated_total_cost": None,
                    "note": "not found in models.csv",
                }
            )
            continue

        row = registry.loc[model_id]
        input_cost_per_M = row.get("input_cost")   # $ per 1M tokens
        output_cost_per_M = row.get("output_cost")  # $ per 1M tokens
        is_reasoning = row.get("is_reasoning", False)

        if pd.isna(input_cost_per_M) or pd.isna(output_cost_per_M):
            results.append(
                {
                    "model_id": model_id,
                    "display_name": yaml_name,
                    "is_reasoning": bool(is_reasoning),
                    "input_cost_per_M": input_cost_per_M if not pd.isna(input_cost_per_M) else None,
                    "output_cost_per_M": output_cost_per_M if not pd.isna(output_cost_per_M) else None,
                    "estimated_input_cost": None,
                    "estimated_output_cost": None,
                    "estimated_total_cost": None,
                    "note": "missing cost data",
                }
            )
            continue

        # Reasoning models produce extra chain-of-thought tokens
        model_output_tokens = total_output_tokens
        if is_reasoning:
            model_output_tokens += reasoning_tokens * n_questions

        est_input = (total_input_tokens / 1_000_000) * input_cost_per_M
        est_output = (model_output_tokens / 1_000_000) * output_cost_per_M
        est_total = est_input + est_output

        results.append(
            {
                "model_id": model_id,
                "display_name": yaml_name,
                "is_reasoning": bool(is_reasoning),
                "input_cost_per_M": input_cost_per_M,
                "output_cost_per_M": output_cost_per_M,
                "estimated_input_cost": round(est_input, 4),
                "estimated_output_cost": round(est_output, 4),
                "estimated_total_cost": round(est_total, 4),
                "note": "reasoning" if is_reasoning else "",
            }
        )

    results_df = pd.DataFrame(results)

    # ── Print summary ─────────────────────────────────────────────────
    print("=" * 70)
    print("BENCHMARK COST ESTIMATION")
    print("=" * 70)
    print(f"Dataset:              {dataset_path}")
    print(f"Number of questions:  {n_questions}")
    print(f"Question column:      {question_col}")
    print(f"Choice columns:       {choice_cols}")
    print()
    print("Prompt structure per API call:")
    print(f"  Fixed template:     {fixed_words} words")
    print(f"  Few-shot examples:  {num_examples} x ~{avg_example_words:.0f} words = ~{fewshot_words:.0f} words")
    print(f"  Avg question block: {df['_question_words'].mean():.0f} words")
    print(f"  Avg total input:    {avg_input_words:.0f} words/call")
    print()
    print(f"Word-to-token ratio:  {word_token_ratio}")
    print(f"Total input tokens:   {total_input_tokens:,.0f}")
    print(f"Output tokens/q:      {output_tokens} (+ {reasoning_tokens} reasoning for reasoning models)")
    print(f"Total output tokens:  {total_output_tokens:,} (non-reasoning) / {(output_tokens + reasoning_tokens) * n_questions:,} (reasoning)")
    print(f"Number of models:     {len(model_names)}")
    print("=" * 70)

    # Print per-model table
    valid = results_df[results_df["estimated_total_cost"].notna()].copy()
    invalid = results_df[results_df["estimated_total_cost"].isna()]

    if not valid.empty:
        valid = valid.sort_values("estimated_total_cost", ascending=True)
        print(f"\n{'Model':<40} {'Input $':>10} {'Output $':>10} {'Total $':>10}")
        print("-" * 70)
        for _, r in valid.iterrows():
            print(
                f"{r['display_name']:<40} "
                f"${r['estimated_input_cost']:>9.4f} "
                f"${r['estimated_output_cost']:>9.4f} "
                f"${r['estimated_total_cost']:>9.4f}"
            )
        print("-" * 70)
        grand_total = valid["estimated_total_cost"].sum()
        print(f"{'TOTAL (all models)':<40} {'':>10} {'':>10} ${grand_total:>9.4f}")

    if not invalid.empty:
        print(f"\nModels with missing cost data ({len(invalid)}):")
        for _, r in invalid.iterrows():
            print(f"  - {r['display_name']} ({r['model_id']}): {r['note']}")

    print()
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Estimate API cost of running models on a benchmark dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the benchmark CSV dataset.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["question", "answer_a", "answer_b", "answer_c", "answer_d",
                 "answer_e", "answer_f", "answer_g"],
        help=(
            "CSV columns to use. First column is the question, rest are choices. "
            "(default: question answer_a..answer_g)"
        ),
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of few-shot examples included in each prompt (default: 3).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help=(
            "Custom system prompt text (or path to a .txt file). "
            "If not provided, uses the default English benchmark template."
        ),
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=2,
        help="Estimated number of output tokens per question (default: 2 for MCQ).",
    )
    parser.add_argument(
        "--word-token-ratio",
        type=float,
        default=WORD_TOKEN_RATIO,
        help="Estimated tokens per word (default: 0.75).",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default=None,
        help=(
            "Path to a YAML models config file to select which models to estimate. "
            "If not provided, all models in models.csv are used."
        ),
    )
    parser.add_argument(
        "--models-csv",
        type=str,
        default=str(MODELS_CSV),
        help=f"Path to the models registry CSV (default: {MODELS_CSV}).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save results as CSV.",
    )

    args = parser.parse_args()

    # Resolve dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}")
        sys.exit(1)

    # Resolve models config
    models_config_path = None
    if args.models_config:
        models_config_path = Path(args.models_config)
        if not models_config_path.is_absolute():
            models_config_path = ROOT / models_config_path
        if not models_config_path.exists():
            print(f"Error: models config not found: {models_config_path}")
            sys.exit(1)

    # Handle system prompt: if it's a file path, read it
    system_prompt = args.system_prompt
    prompt_path = Path(system_prompt) if system_prompt else None
    if prompt_path and prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding="utf-8")

    # Split columns: first = question, rest = choices
    question_col = args.columns[0]
    choice_cols = args.columns[1:]

    models_csv_path = Path(args.models_csv)

    results_df = estimate_costs(
        dataset_path=dataset_path,
        question_col=question_col,
        choice_cols=choice_cols,
        num_examples=args.num_examples,
        system_prompt=system_prompt,
        output_tokens=args.output_tokens,
        word_token_ratio=args.word_token_ratio,
        reasoning_tokens=MEAN_REASONING_TOKENS,
        models_config_path=models_config_path,
        models_csv_path=models_csv_path,
    )

    if args.output_csv:
        output_path = Path(args.output_csv)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
