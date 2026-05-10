import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit.backends.fixture_contour import (
    STATUS_PROPOSAL_REJECTED,
    STATUS_VALIDATED,
    STATUS_VALIDATION_FAILED,
    execute_fixture_contour_backend,
)
from phase3_mask_edit.cli.run_llm_contour_fixture import main as run_fixture_main
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import save_id_mask


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
        changed = result.edit_result.change_region
        self.assertGreater(int(np.count_nonzero(changed)), 0)
        self.assertTrue(np.all(self.mask[changed] == 2))
        self.assertTrue(np.all(result.edit_result.target_mask[changed] == 4))

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
        changed = result.edit_result.change_region
        self.assertGreater(int(np.count_nonzero(changed)), 0)
        self.assertTrue(np.all(self.mask[changed] == 1))
        self.assertTrue(np.all(result.edit_result.target_mask[changed] == 3))

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
        )

        self.assertEqual(result.status, STATUS_VALIDATION_FAILED)
        self.assertIsNotNone(result.edit_result)
        self.assertIsNotNone(result.validation)
        self.assertFalse(result.validation.passed)

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

    def test_artifacts_are_saved_for_validated_run(self):
        intent = EditIntent(
            primitive="necrosis_appearance",
            strength="mild",
            reference_profile="BCSS",
            source_labels=("Tumor",),
            target_label="Necrosis",
        )
        WORKSPACE_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=WORKSPACE_TMP) as tmp:
            result = execute_fixture_contour_backend(
                old_mask=self.mask,
                fixture_path="tests/fixtures/llm_contour_necrosis_bcss.json",
                schema=self.schema,
                intent=intent,
                primitive_config=_primitive(self.recipe, "necrosis_appearance"),
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

    def test_cli_runs_fixture_backend(self):
        WORKSPACE_TMP.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=WORKSPACE_TMP) as tmp:
            mask_path = Path(tmp) / "source_mask.png"
            output_dir = Path(tmp) / "run"
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
    handle = tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        delete=False,
        dir=WORKSPACE_TMP,
    )
    with handle:
        json.dump(payload, handle)
    return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
