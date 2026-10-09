"""Translate the local 920-row CDPK dataset with the Google Gemini API.

The runner translates one complete multiple-choice item per request, checkpoints
after every row, and resumes from an existing output without retranslating valid
rows. It intentionally does not invoke any benchmark/evaluation code.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data/pedagogy_benchmark_full_datasets/pedagogy_benchmark_cdpk.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data/pedagogy_benchmark_full_datasets"
TRANSLATED_COLUMNS = ("question", "answer_a", "answer_b", "answer_c", "answer_d")
DEFAULT_MODEL = "gemini-3.1-pro-preview"
PROMPT_VERSION = "cdpk-context-v1"


class TranslatedItem(BaseModel):
    question: str
    answer_a: str
    answer_b: str
    answer_c: str
    answer_d: str


class TranslatedField(BaseModel):
    translated_text: str


def prompt_for(row: pd.Series, language: str) -> str:
    source = {column: str(row[column]) for column in TRANSLATED_COLUMNS}
    return (
        f"You are a professional English-to-{language} translator. Translate this "
        "teacher-education multiple-choice item accurately and naturally. Preserve "
        "the pedagogical meaning, names, quoted text, numbering, and relationship "
        "between the question and answer choices. Do not answer the question, add "
        "explanations, or leave English commentary. Return every field in the "
        f"requested JSON schema.\n\nEnglish item:\n{json.dumps(source, ensure_ascii=False)}"
    )


def generation_config(
    schema: type[BaseModel],
    thinking_budget: int,
    max_output_tokens: int | None,
) -> types.GenerateContentConfig:
    config: dict[str, Any] = {
        "temperature": 0,
        "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
        "response_mime_type": "application/json",
        "response_schema": schema,
    }
    if max_output_tokens is not None:
        config["max_output_tokens"] = max_output_tokens
    return types.GenerateContentConfig(**config)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def word_tokens(value: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", value.casefold(), flags=re.UNICODE)


def validate_candidate(source: str, candidate: str) -> None:
    """Reject obvious generation failures without attempting linguistic review."""
    if not candidate.strip():
        raise ValueError("Gemini returned an empty translation")

    source_words = word_tokens(source)
    candidate_words = word_tokens(candidate)
    if len(source_words) >= 4 and normalize_text(source) == normalize_text(candidate):
        raise ValueError("Gemini returned the untranslated English source")

    if len(candidate_words) >= 20:
        most_common_word = max(candidate_words.count(word) for word in set(candidate_words))
        dominant_word_ratio = most_common_word / len(candidate_words)
        bigrams = list(zip(candidate_words, candidate_words[1:]))
        most_common_bigram = max(
            (bigrams.count(bigram) for bigram in set(bigrams)), default=0
        )
        repeated_bigram_ratio = most_common_bigram / len(bigrams) if bigrams else 0
        if dominant_word_ratio >= 0.55 or repeated_bigram_ratio >= 0.30:
            raise ValueError("Gemini returned obviously repetitive text")

    if len(source_words) >= 50 and len(candidate_words) < 15:
        raise ValueError("Gemini returned an implausibly short translation")


def translate_row(
    client: genai.Client,
    row: pd.Series,
    language: str,
    model: str,
    thinking_budget: int,
    max_output_tokens: int | None,
) -> dict[str, str]:
    response = client.models.generate_content(
        model=model,
        contents=prompt_for(row, language),
        config=generation_config(
            TranslatedItem, thinking_budget, max_output_tokens
        ),
    )
    parsed = response.parsed
    if parsed is None:
        parsed = TranslatedItem.model_validate_json(response.text)
    if isinstance(parsed, BaseModel):
        result = parsed.model_dump()
    else:
        result = TranslatedItem.model_validate(parsed).model_dump()
    for column, value in result.items():
        validate_candidate(str(row[column]), value)
    return result


def translate_field(
    client: genai.Client,
    row: pd.Series,
    column: str,
    language: str,
    model: str,
    thinking_budget: int,
    max_output_tokens: int | None,
) -> str:
    if column == "question":
        context = f"English question:\n{row[column]}"
    else:
        context = (
            f"English question (context only; do not return it):\n{row['question']}\n\n"
            f"English {column} to translate:\n{row[column]}"
        )
    prompt = (
        f"Translate the requested text from English into {language}. Preserve its exact "
        "meaning, names, numbering, and relationship to the multiple-choice question. "
        "Return only the translation in the requested JSON schema; do not answer the "
        f"question or add commentary.\n\n{context}"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=generation_config(
            TranslatedField, thinking_budget, max_output_tokens
        ),
    )
    parsed = response.parsed
    if parsed is None:
        parsed = TranslatedField.model_validate_json(response.text)
    if isinstance(parsed, BaseModel):
        value = parsed.translated_text
    else:
        value = TranslatedField.model_validate(parsed).translated_text
    value = value.strip()
    validate_candidate(str(row[column]), value)
    return value


def is_complete(row: pd.Series) -> bool:
    return all(pd.notna(row[column]) and str(row[column]).strip() for column in TRANSLATED_COLUMNS)


def atomic_write(df: pd.DataFrame, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    temporary.replace(output)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify_language(language: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", language.casefold()).strip("_")
    if not slug:
        raise ValueError(f"Could not create an output slug for language: {language!r}")
    return slug


def output_path_for(
    output_dir: Path,
    language: str,
    limit: int | None,
    output_slug: str | None,
) -> Path:
    slug = output_slug or slugify_language(language)
    suffix = f"_smoke_{limit}" if limit is not None else ""
    return output_dir / f"pedagogy_benchmark_{slug}_cdpk{suffix}.csv"


def manifest_path_for(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def atomic_write_json(data: dict[str, Any], output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def write_manifest(
    *,
    output: Path,
    source_path: Path,
    language: str,
    language_code: str | None,
    model: str,
    row_count: int,
    workers: int,
    retries: int,
    mode: str,
    thinking_budget: int,
    max_output_tokens: int | None,
) -> Path:
    output_sha256 = sha256_file(output)
    slug = output.stem.removeprefix("pedagogy_benchmark_").split("_cdpk", 1)[0]
    manifest = {
        "release_id": f"cdpk-{slug}-{output_sha256[:12]}",
        "benchmark": "CDPK",
        "source_file": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "output_file": str(output.resolve()),
        "output_sha256": output_sha256,
        "language": language,
        "language_code": language_code,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "translation_mode": mode,
        "temperature": 0,
        "thinking_budget": thinking_budget,
        "max_output_tokens": max_output_tokens,
        "workers": workers,
        "retries": retries,
        "row_count": row_count,
        "translated_columns": list(TRANSLATED_COLUMNS),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = manifest_path_for(output)
    atomic_write_json(manifest, manifest_path)
    return manifest_path


def validate_completed_dataset(source: pd.DataFrame, translated: pd.DataFrame) -> None:
    if len(source) != len(translated):
        raise ValueError(
            f"Output row count does not match source: {len(translated)} vs {len(source)}"
        )
    if source["question_id"].astype(str).tolist() != translated[
        "question_id"
    ].astype(str).tolist():
        raise ValueError("Output question_id values do not align with the source")

    issues: list[str] = []
    for index in translated.index:
        for column in TRANSLATED_COLUMNS:
            try:
                value = translated.at[index, column]
                if pd.isna(value):
                    raise ValueError("translation is missing")
                validate_candidate(str(source.at[index, column]), str(value))
            except ValueError as error:
                issues.append(
                    f"question_id={source.at[index, 'question_id']} "
                    f"column={column}: {error}"
                )
    if issues:
        preview = "\n".join(issues[:20])
        raise ValueError(
            f"Translation validation found {len(issues)} issue(s):\n{preview}"
        )


def run_language(
    client: genai.Client,
    source: pd.DataFrame,
    language: str,
    model: str,
    output_dir: Path,
    limit: int | None,
    retries: int,
    workers: int,
    field_by_field: bool,
    auto_fallback: bool,
    output_slug: str | None,
    thinking_budget: int,
    max_output_tokens: int | None,
) -> Path:
    output = output_path_for(output_dir, language, limit, output_slug)
    selected = source.head(limit).copy() if limit is not None else source.copy()

    if output.exists():
        translated = pd.read_csv(output)
        if translated["question_id"].astype(str).tolist() != selected[
            "question_id"
        ].astype(str).tolist():
            raise ValueError(f"Existing output does not align with source: {output}")
    else:
        translated = selected.copy()
        translated.loc[:, list(TRANSLATED_COLUMNS)] = pd.NA

    pending = [index for index, row in translated.iterrows() if not is_complete(row)]
    print(f"{language}: {len(pending)} of {len(translated)} rows pending -> {output}", flush=True)

    def translate_fields_with_retries(index: int) -> tuple[int, dict[str, str]]:
        result: dict[str, str] = {}
        for column in TRANSLATED_COLUMNS:
            for attempt in range(1, retries + 1):
                try:
                    result[column] = translate_field(
                        client,
                        selected.loc[index],
                        column,
                        language,
                        model,
                        thinking_budget,
                        max_output_tokens,
                    )
                    break
                except Exception as error:
                    if attempt == retries:
                        raise RuntimeError(
                            f"{language} question_id={selected.at[index, 'question_id']} "
                            f"column={column} failed after {retries} attempts"
                        ) from error
                    delay = min(60, 2 ** attempt + random.random())
                    print(
                        f"{language}: {column} attempt {attempt}/{retries} failed; "
                        f"retrying in {delay:.1f}s: {error}",
                        flush=True,
                    )
                    time.sleep(delay)
        return index, result

    def translate_with_retries(index: int) -> tuple[int, dict[str, str]]:
        if field_by_field:
            return translate_fields_with_retries(index)

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                result = translate_row(
                    client,
                    selected.loc[index],
                    language,
                    model,
                    thinking_budget,
                    max_output_tokens,
                )
                return index, result
            except Exception as error:
                last_error = error
                if attempt == retries:
                    break
                delay = min(60, 2 ** attempt + random.random())
                print(f"{language}: attempt {attempt}/{retries} failed; retrying in {delay:.1f}s: {error}", flush=True)
                time.sleep(delay)

        if auto_fallback:
            print(
                f"{language}: question_id={selected.at[index, 'question_id']} "
                "falling back to field-by-field translation",
                flush=True,
            )
            return translate_fields_with_retries(index)
        raise RuntimeError(
            f"{language} question_id={selected.at[index, 'question_id']} "
            f"failed after {retries} attempts"
        ) from last_error

    completed = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(translate_with_retries, index) for index in pending]
        for future in as_completed(futures):
            try:
                index, result = future.result()
            except Exception as error:
                # A permanently failing row must not prevent successful sibling
                # requests from being checkpointed. It remains blank and will be
                # picked up automatically on the next invocation.
                failed.append(str(error))
                print(f"{language}: leaving row pending for next resume: {error}", flush=True)
                continue
            for column, value in result.items():
                translated.at[index, column] = value
            atomic_write(translated, output)
            completed += 1
            print(
                f"{language}: completed {completed}/{len(pending)} "
                f"(question_id={selected.at[index, 'question_id']})",
                flush=True,
            )

    missing = sum(
        1
        for index in translated.index
        for column in TRANSLATED_COLUMNS
        if pd.isna(translated.at[index, column])
        or not str(translated.at[index, column]).strip()
    )
    if missing:
        atomic_write(translated, output)
        raise RuntimeError(
            f"{output} still contains {missing} missing translated cells; "
            f"{len(failed)} rows exhausted their retries and remain resumable"
        )
    validate_completed_dataset(selected, translated)
    atomic_write(translated, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["Mende", "Temne", "Krio"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-slug",
        help="Override the filename language slug (only valid with one language)",
    )
    parser.add_argument(
        "--language-code",
        help="BCP-47 or project language code recorded in the manifest",
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Translate only the first N rows (smoke test)")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--thinking-budget", type=int, default=128)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Explicit Gemini output-token limit; provider default when omitted",
    )
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--field-by-field",
        action="store_true",
        help="Use five shorter requests per row for languages prone to long-response timeouts",
    )
    parser.add_argument(
        "--no-auto-fallback",
        action="store_true",
        help="Do not fall back to field-by-field mode after full-row retries fail",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file, override=True)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY was not found in {args.env_file}")
    source = pd.read_csv(args.input)
    required = {"question_id", *TRANSLATED_COLUMNS}
    missing_columns = required.difference(source.columns)
    if missing_columns:
        raise ValueError(f"Input is missing columns: {sorted(missing_columns)}")
    if source["question_id"].duplicated().any():
        raise ValueError("Input contains duplicate question_id values")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.retries < 1:
        raise ValueError("--retries must be at least 1")
    if args.thinking_budget < 0:
        raise ValueError("--thinking-budget cannot be negative")
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        raise ValueError("--max-output-tokens must be at least 1")
    if args.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be at least 1")
    if args.output_slug and len(args.languages) != 1:
        raise ValueError("--output-slug requires exactly one language")
    if args.language_code and len(args.languages) != 1:
        raise ValueError("--language-code requires exactly one language")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=args.timeout_seconds * 1000),
    )
    for language in args.languages:
        output = run_language(
            client,
            source,
            language,
            args.model,
            args.output_dir,
            args.limit,
            args.retries,
            args.workers,
            args.field_by_field,
            not args.no_auto_fallback,
            args.output_slug,
            args.thinking_budget,
            args.max_output_tokens,
        )
        manifest_path = write_manifest(
            output=output,
            source_path=args.input,
            language=language,
            language_code=args.language_code,
            model=args.model,
            row_count=len(source.head(args.limit)) if args.limit is not None else len(source),
            workers=args.workers,
            retries=args.retries,
            mode="field-by-field" if args.field_by_field else "row-with-field-fallback",
            thinking_budget=args.thinking_budget,
            max_output_tokens=args.max_output_tokens,
        )
        print(f"Complete: {output}", flush=True)
        print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
