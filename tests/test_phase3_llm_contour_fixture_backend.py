import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from phase3_mask_edit.backends.fixture_contour import (
    STATUS_PROPOSAL_REJECTED,
    STATUS_VALIDATED,
    STATUS_VALIDATION_FAILED,
    execute_fixture_contour_backend,
)
from phase3_mask_edit.backends.llm_contour import PROJECTION_MODE_HARD_V1
from phase3_mask_edit.cli.run_llm_contour_fixture import main as run_fixture_main
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import (
    load_change_region,
    load_id_mask,
    load_metadata,
    save_id_mask,
)


WORKSPACE_TMP = Path(".tmp_phase3_llm_contour_fixture_tests")


class LLMContourFixtureBackendTests(unittest.TestCase):
    def setUp(self):
        self.schema = MaskProfileSchema.from_reference_profile("BCSS")
        self.recipe = load_recipe("phase3_mask_edit/recipes/generic.yaml")
        self.mask = _synthetic_bcss_mask()

    def test_stromal_immune_fixture_executes_and_validates(self):
        intent = EditIntent(
            primitive="stromal_immune_infiltration",
            strength="mild",
            reference_profile="BCSS",
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
        )

        result = execute_fixture_contour_backend(
            old_mask=self.mask,
            fixture_path="tests/fixtures/llm_contour_stromal_immune_bcss.json",
            schema=self.schema,
            intent=intent,
            primitive_config=_primitive(self.recipe, "stromal_immune_infiltration"),
        )

        self.assertEqual(result.status, STATUS_VALIDATED)
        self.assertIsNotNone(result.edit_result)
        _assert_final_diff_labels(
            self,
            old_mask=self.mask,
            target_mask=result.edit_result.target_mask,
            allowed_source_ids={2},
            target_id=4,
        )
        tumor = self.mask == 1
        np.testing.assert_array_equal(
            result.edit_result.target_mask[tumor],
            self.mask[tumor],
        )

    def test_necrosis_fixture_executes_and_validates(self):
        intent = EditIntent(
            primitive="necrosis_appearance",
            strength="mild",
            reference_profile="BCSS",
            source_labels=("Tumor",),
            target_label="Necrosis",
        )

        result = execute_fixture_contour_backend(
            old_mask=self.mask,
            fixture_path="tests/fixtures/llm_contour_necrosis_bcss.json",
            schema=self.schema,
            intent=intent,
            primitive_config=_primitive(self.recipe, "necrosis_appearance"),
        )

        self.assertEqual(result.status, STATUS_VALIDATED)
        self.assertIsNotNone(result.edit_result)
        _assert_final_diff_labels(
            self,
            old_mask=self.mask,
            target_mask=result.edit_result.target_mask,
            allowed_source_ids={1},
            target_id=3,
        )
        stroma = self.mask == 2
        np.testing.assert_array_equal(
            result.edit_result.target_mask[stroma],
            self.mask[stroma],
        )

    def test_validation_failed_status_is_separate_from_execution(self):
        fixture_path = _write_fixture(
            _proposal(
                primitive="stromal_immune_infiltration",
                target_label="Immune infiltrate",
                source_labels=["Stroma"],
                points=[[1, 1], [6, 1], [6, 6], [1, 6]],
            )
        )
        intent = EditIntent(
            primitive="stromal_immune_infiltration",
            strength="mild",
            reference_profile="BCSS",
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
        )

        result = execute_fixture_contour_backend(
            old_mask=self.mask,
            fixture_path=fixture_path,
            schema=self.schema,
            intent=intent,
            primitive_config=_primitive(self.recipe, "stromal_immune_infiltration"),
            projection_mode=PROJECTION_MODE_HARD_V1,
        )

        self.assertEqual(result.status, STATUS_VALIDATION_FAILED)
        self.assertIsNotNone(result.edit_result)
        self.assertIsNotNone(result.validation)
        self.assertFalse(result.validation.passed)
        failed_names = {check.name for check in result.validation.failed_checks}
        self.assertIn("change_area_nonempty", failed_names)
        self.assertIn("change_area_within_range", failed_names)

    def test_proposal_rejected_status_for_bad_schema_payload(self):
        fixture_path = _write_fixture(
            _proposal(
                primitive="stromal_immune_infiltration",
                target_label="Immune infiltrate",
                source_labels=["Stroma"],
                points=[[1, 1], [99, 1], [1, 8]],
            )
        )
        intent = EditIntent(
            primitive="stromal_immune_infiltration",
            strength="mild",
            reference_profile="BCSS",
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
        )

        result = execute_fixture_contour_backend(
            old_mask=self.mask,
            fixture_path=fixture_path,
            schema=self.schema,
            intent=intent,
            primitive_config=_primitive(self.recipe, "stromal_immune_infiltration"),
        )

        self.assertEqual(result.status, STATUS_PROPOSAL_REJECTED)
        self.assertIsNone(result.edit_result)
        self.assertIn("outside mask bounds", result.error)

    def test_artifacts_are_saved_and_consistent_for_validated_run(self):
        intent = EditIntent(
            primitive="stromal_immune_infiltration",
            strength="mild",
            reference_profile="BCSS",
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
        )
        WORKSPACE_TMP.mkdir(exist_ok=True)
        tmp = WORKSPACE_TMP / f"artifacts_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True)
        try:
            result = execute_fixture_contour_backend(
                old_mask=self.mask,
                fixture_path="tests/fixtures/llm_contour_stromal_immune_bcss.json",
                schema=self.schema,
                intent=intent,
                primitive_config=_primitive(self.recipe, "stromal_immune_infiltration"),
                output_dir=tmp,
            )

            self.assertEqual(result.status, STATUS_VALIDATED)
            for name in (
                "execution_summary.json",
                "validation.json",
                "target_mask.png",
                "change_region.png",
                "rasterized_region.png",
                "projected_region.png",
                "source_mask_llm_rgb_grid.png",
            ):
                self.assertTrue((Path(tmp) / name).exists(), name)

            target_mask = load_id_mask(tmp / "target_mask.png")
            _assert_final_diff_labels(
                self,
                old_mask=self.mask,
                target_mask=target_mask,
                allowed_source_ids={2},
                target_id=4,
            )

            rasterized = load_change_region(tmp / "rasterized_region.png")
            projected = load_change_region(tmp / "projected_region.png")
            rasterized_pixels = int(np.count_nonzero(rasterized))
            projected_pixels = int(np.count_nonzero(projected))

            summary = load_metadata(tmp / "execution_summary.json")
            ops_log = summary["edit_result"]["ops_log"]
            self.assertEqual(ops_log["projection_mode"], "organic_v2")
            self.assertEqual(ops_log["projection_backend"], "organic_score_projection_v2")
            self.assertEqual(ops_log["candidate_pixels"], rasterized_pixels)
            self.assertEqual(ops_log["projected_pixels"], projected_pixels)
            self.assertEqual(ops_log["selected_pixels"], projected_pixels)
            self.assertTrue(summary["validation"]["passed"])

            validation = load_metadata(tmp / "validation.json")
            self.assertTrue(validation["passed"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cli_runs_fixture_backend(self):
        WORKSPACE_TMP.mkdir(exist_ok=True)
        tmp = WORKSPACE_TMP / f"cli_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True)
        try:
            mask_path = tmp / "source_mask.png"
            output_dir = tmp / "run"
            save_id_mask(self.mask, mask_path)

            code = run_fixture_main(
                [
                    "--profile",
                    "BCSS",
                    "--primitive",
                    "necrosis_appearance",
                    "--strength",
                    "mild",
                    "--mask",
                    str(mask_path),
                    "--fixture",
                    "tests/fixtures/llm_contour_necrosis_bcss.json",
                    "--output",
                    str(output_dir),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((output_dir / "execution_summary.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cli_compare_mode_writes_v1_and_v2_artifacts(self):
        WORKSPACE_TMP.mkdir(exist_ok=True)
        tmp = WORKSPACE_TMP / f"compare_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True)
        try:
            mask_path = tmp / "source_mask.png"
            output_dir = tmp / "run"
            save_id_mask(self.mask, mask_path)

            code = run_fixture_main(
                [
                    "--profile",
                    "BCSS",
                    "--primitive",
                    "stromal_immune_infiltration",
                    "--strength",
                    "mild",
                    "--mask",
                    str(mask_path),
                    "--fixture",
                    "tests/fixtures/llm_contour_stromal_immune_bcss.json",
                    "--output",
                    str(output_dir),
                    "--projection-mode",
                    "compare_v1_v2",
                    "--organic-seed",
                    "11",
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((output_dir / "v1_hard_projection" / "summary.json").exists())
            self.assertTrue((output_dir / "organic_v2" / "summary.json").exists())
            v2 = load_metadata(output_dir / "organic_v2" / "summary.json")
            self.assertEqual(
                v2["edit_result"]["ops_log"]["projection_backend"],
                "organic_score_projection_v2",
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _synthetic_bcss_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[8:56, 8:56] = 2
    mask[18:46, 18:46] = 1
    return mask


def _primitive(recipe, name):
    for primitive in recipe["primitives"]:
        if primitive["name"] == name:
            return primitive
    raise AssertionError(f"missing primitive {name}")


def _assert_final_diff_labels(
    case,
    *,
    old_mask: np.ndarray,
    target_mask: np.ndarray,
    allowed_source_ids: set[int],
    target_id: int,
) -> None:
    diff = old_mask != target_mask
    case.assertGreater(int(np.count_nonzero(diff)), 0)
    changed_old_labels = set(np.unique(old_mask[diff]).astype(int).tolist())
    changed_new_labels = set(np.unique(target_mask[diff]).astype(int).tolist())
    case.assertLessEqual(changed_old_labels, allowed_source_ids)
    case.assertEqual(changed_new_labels, {target_id})


def _proposal(*, primitive, target_label, source_labels, points):
    return {
        "schema_version": "0.1",
        "backend": "llm_contour_proposal",
        "primitive": primitive,
        "reference_profile": "BCSS",
        "target_label": target_label,
        "coordinate_system": {
            "origin": "top_left",
            "point_format": "[x, y]",
            "x_axis": "horizontal_column_right",
            "y_axis": "vertical_row_down",
            "width": 64,
            "height": 64,
        },
        "regions": [
            {
                "region_id": "r1",
                "type": "polygon",
                "source_labels": source_labels,
                "points": points,
                "confidence": 0.8,
            }
        ],
    }


def _write_fixture(payload):
    WORKSPACE_TMP.mkdir(exist_ok=True)
    path = WORKSPACE_TMP / f"fixture_{uuid.uuid4().hex}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


if __name__ == "__main__":
    unittest.main()
