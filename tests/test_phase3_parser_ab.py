import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phase3_mask_edit.benchmark.models import BenchmarkIntent, BenchmarkPrompt
from phase3_mask_edit.benchmark.prompts import semantic_diff_for_intent
from phase3_mask_edit.cli.run_mask_edit_parser_ab import (
    INSTRUCTION_MODE,
    PROMPT_MODE,
    _recompute_existing_results,
    _result_row,
    _write_results,
    evaluate_parser_sample,
)


def _intent() -> BenchmarkIntent:
    return BenchmarkIntent(
        sample_id="sample",
        organ="breast",
        profile="BCSS",
        image_path=None,
        mask_path="/definitely/missing/mask.png",
        primitive="tumor_burden_increase",
        strength="mild",
        region_hint={},
        source_labels=("Stroma",),
        target_label="Tumor",
        expected_direction="increase",
        expected_area_bucket=(0.01, 0.05),
        seed=13,
    )


class ParserAbTests(unittest.TestCase):
    def test_evaluation_is_parser_only_and_does_not_read_mask(self) -> None:
        intent = _intent()
        prompt = BenchmarkPrompt(
            sample_id=intent.sample_id,
            old_prompt="No tumor is present.",
            new_prompt="A mild tumor focus is now present.",
            instruction="Mildly increase tumor burden.",
        )
        parsed = semantic_diff_for_intent(intent)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch(
            "phase3_mask_edit.cli.run_mask_edit_parser_ab.parse_prompts_with_api",
            return_value=parsed,
        ):
            row = evaluate_parser_sample(
                intent,
                prompt,
                mode=PROMPT_MODE,
                model="test-model",
                api_base_url="https://example.invalid/v1",
                api_key_env="TEST_KEY",
                output_dir=Path(tmpdir),
            )

        self.assertEqual(row["status"], "completed")
        self.assertTrue(row["primitive_exact"])
        self.assertEqual(row["parsed_primitives"], [intent.primitive])

    def test_extra_primitive_fails_strict_exact_match(self) -> None:
        intent = _intent()
        row = _result_row(
            intent,
            mode=PROMPT_MODE,
            model="test-model",
            status="completed",
            expected=semantic_diff_for_intent(intent),
            parsed=semantic_diff_for_intent(intent),
            parsed_primitives=[intent.primitive, "stromal_desmoplasia"],
            error="",
            output_dir=Path("/tmp/sample"),
        )
        self.assertFalse(row["primitive_exact"])
        self.assertEqual(row["parsed_primitive"], "")

    def test_recompute_uses_saved_json_without_api_calls(self) -> None:
        intent = BenchmarkIntent(
            **{
                **_intent().__dict__,
                "sample_id": "benign-atrophy",
                "profile": "PANDA",
                "primitive": "benign_atrophy",
                "expected_direction": "transition",
            }
        )
        prompt = BenchmarkPrompt(
            sample_id=intent.sample_id,
            old_prompt="Benign prostatic epithelium is present.",
            new_prompt="Stromal tissue replaces the benign glands.",
            instruction=(
                "Replace benign prostatic epithelium with stromal tissue without "
                "epithelial glands."
            ),
        )
        parsed = semantic_diff_for_intent(intent)
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            result_path = output / "parser_eval_results.csv"
            stale_row = _result_row(
                intent,
                mode=INSTRUCTION_MODE,
                model="test-model",
                status="completed",
                expected=parsed,
                parsed=parsed,
                parsed_primitives=[],
                error="",
                output_dir=output / "samples",
            )
            _write_results([stale_row], result_path)

            report = _recompute_existing_results(
                [intent],
                {intent.sample_id: prompt},
                modes=[INSTRUCTION_MODE],
                model="test-model",
                results_path=result_path,
                output_dir=output,
            )

        self.assertEqual(
            report["modes"][INSTRUCTION_MODE]["primitive_exact_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
