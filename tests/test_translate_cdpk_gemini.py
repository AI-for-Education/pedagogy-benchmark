from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import translate_cdpk_gemini as translator


def source_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question_id": 10,
                "question": "Which teaching method best supports this learner?",
                "answer_a": "Use the first method",
                "answer_b": "Use the second method",
                "answer_c": "Use the third method",
                "answer_d": "Use the fourth method",
                "correct_answer": "A",
                "category": "General",
            },
            {
                "question_id": 20,
                "question": "How should the teacher assess this activity?",
                "answer_a": "Observe the learners",
                "answer_b": "Ignore the activity",
                "answer_c": "Repeat the instructions",
                "answer_d": "Cancel the lesson",
                "correct_answer": "A",
                "category": "General",
            },
        ]
    )


def arabic_result(prefix: str) -> dict[str, str]:
    return {
        "question": f"{prefix} كيف ينبغي للمعلم تنفيذ هذا النشاط؟",
        "answer_a": f"{prefix} استخدام الطريقة الأولى",
        "answer_b": f"{prefix} استخدام الطريقة الثانية",
        "answer_c": f"{prefix} استخدام الطريقة الثالثة",
        "answer_d": f"{prefix} استخدام الطريقة الرابعة",
    }


class TranslateCdpkGeminiTests(unittest.TestCase):
    def test_slug_and_smoke_output_are_deterministic(self) -> None:
        output = translator.output_path_for(
            Path("/tmp/results"),
            "Modern Standard Arabic",
            2,
            "arabic",
        )
        self.assertEqual(
            output.name, "pedagogy_benchmark_arabic_cdpk_smoke_2.csv"
        )

    def test_quality_validator_rejects_unchanged_and_repetitive_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "untranslated English"):
            translator.validate_candidate(
                "The teacher should observe every learner carefully.",
                "The teacher should observe every learner carefully.",
            )
        with self.assertRaisesRegex(ValueError, "repetitive"):
            translator.validate_candidate(
                "This is a sufficiently long source sentence for the validation test.",
                "كلمة " * 25,
            )

    def test_resume_translates_only_incomplete_rows(self) -> None:
        source = source_dataframe()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            output = translator.output_path_for(
                output_dir, "Modern Standard Arabic", None, "arabic"
            )
            existing = source.copy()
            for column, value in arabic_result("الأول").items():
                existing.at[0, column] = value
            existing.loc[1, list(translator.TRANSLATED_COLUMNS)] = pd.NA
            existing.to_csv(output, index=False)

            called_ids: list[int] = []

            def fake_translate_row(client, row, *args, **kwargs):
                called_ids.append(int(row["question_id"]))
                return arabic_result("الثاني")

            with patch.object(translator, "translate_row", side_effect=fake_translate_row):
                translator.run_language(
                    object(), source, "Modern Standard Arabic", "test-model",
                    output_dir, None, 1, 1, False, True, "arabic", 128, 4096,
                )

            self.assertEqual(called_ids, [20])
            completed = pd.read_csv(output)
            self.assertTrue(all(translator.is_complete(row) for _, row in completed.iterrows()))
            self.assertEqual(completed["question_id"].tolist(), [10, 20])

    def test_full_row_failure_falls_back_to_all_five_fields(self) -> None:
        source = source_dataframe().head(1)
        values = arabic_result("بديل")
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(translator, "translate_row", side_effect=RuntimeError("row failed")),
                patch.object(
                    translator,
                    "translate_field",
                    side_effect=lambda client, row, column, *args: values[column],
                ) as field_call,
            ):
                output = translator.run_language(
                    object(), source, "Modern Standard Arabic", "test-model",
                    Path(directory), None, 1, 1, False, True, "arabic", 128, 4096,
                )

            self.assertEqual(field_call.call_count, 5)
            completed = pd.read_csv(output)
            self.assertTrue(translator.is_complete(completed.iloc[0]))

    def test_successful_rows_remain_checkpointed_when_another_row_fails(self) -> None:
        source = source_dataframe()

        def fake_translate_row(client, row, *args, **kwargs):
            if int(row["question_id"]) == 20:
                raise RuntimeError("simulated permanent failure")
            return arabic_result("محفوظ")

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with patch.object(translator, "translate_row", side_effect=fake_translate_row):
                with self.assertRaisesRegex(RuntimeError, "resumable"):
                    translator.run_language(
                        object(), source, "Modern Standard Arabic", "test-model",
                        output_dir, None, 1, 1, False, False, "arabic", 128, 4096,
                    )

            saved = pd.read_csv(
                translator.output_path_for(
                    output_dir, "Modern Standard Arabic", None, "arabic"
                )
            )
            self.assertTrue(translator.is_complete(saved.iloc[0]))
            self.assertFalse(translator.is_complete(saved.iloc[1]))

    def test_existing_output_must_have_matching_question_ids(self) -> None:
        source = source_dataframe()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            output = translator.output_path_for(
                output_dir, "Modern Standard Arabic", None, "arabic"
            )
            mismatched = source.copy()
            mismatched["question_id"] = [999, 1000]
            mismatched.to_csv(output, index=False)
            with self.assertRaisesRegex(ValueError, "does not align"):
                translator.run_language(
                    object(), source, "Modern Standard Arabic", "test-model",
                    output_dir, None, 1, 1, False, True, "arabic", 128, 4096,
                )

    def test_manifest_records_checksums_and_generation_parameters(self) -> None:
        source = source_dataframe()
        translated = source.copy()
        for index in translated.index:
            for column, value in arabic_result(str(index)).items():
                translated.at[index, column] = value

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source_path = directory_path / "source.csv"
            output_path = directory_path / "pedagogy_benchmark_arabic_cdpk.csv"
            source.to_csv(source_path, index=False)
            translated.to_csv(output_path, index=False)

            manifest_path = translator.write_manifest(
                output=output_path,
                source_path=source_path,
                language="Modern Standard Arabic",
                language_code="ar",
                model="gemini-test",
                row_count=2,
                workers=2,
                retries=6,
                mode="row-with-field-fallback",
                thinking_budget=128,
                max_output_tokens=8192,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["language_code"], "ar")
            self.assertEqual(manifest["row_count"], 2)
            self.assertEqual(manifest["prompt_version"], translator.PROMPT_VERSION)
            self.assertEqual(manifest["source_sha256"], translator.sha256_file(source_path))
            self.assertEqual(manifest["output_sha256"], translator.sha256_file(output_path))


if __name__ == "__main__":
    unittest.main()
