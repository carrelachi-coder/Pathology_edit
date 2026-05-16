import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np

from phase3_mask_edit.backends.llm_contour import (
    PROJECTION_MODE_ORGANIC_V2,
    validate_contour_proposal,
)
from phase3_mask_edit.cli.run_organic_projection_stage2_experiment import (
    main as run_stage2_main,
    run_organic_v2_smoke,
    run_weight_decay_seed_grid,
    summarize_grid_trends,
)
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import load_metadata, save_id_mask, save_metadata


WORKSPACE_TMP = Path(".tmp_phase3_organic_stage2_tests")


class OrganicProjectionStage2ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.schema = MaskProfileSchema.from_reference_profile("BCSS")
        self.recipe = load_recipe("phase3_mask_edit/recipes/generic.yaml")
        self.primitive_config = _primitive(self.recipe, "stromal_immune_infiltration")

    def test_reduced_grid_reports_distance_and_area_metrics(self):
        mask = _synthetic_bcss_mask()
        raw_candidate = np.zeros_like(mask, dtype=bool)
        raw_candidate[6:18, 6:18] = True

        rows = run_weight_decay_seed_grid(
            mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config=self.primitive_config,
            weight_combos=(
                ("balanced", 0.45, 0.45, 0.10),
                ("spatial", 0.25, 0.65, 0.10),
            ),
            decays=(16.0, 48.0),
            seeds=(0, 1),
            template_sigma=3.0,
            noise_sigma=18.0,
            noise_amplitude=0.18,
        )

        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["validation_passed"] for row in rows))
        self.assertTrue(all(row["label_safe"] for row in rows))
        self.assertTrue(all(row["selected_to_target_ratio"] == 1.0 for row in rows))
        self.assertTrue(
            all(row["mean_dist_to_tumor_boundary_px"] is not None for row in rows)
        )

        trend = summarize_grid_trends(rows)
        self.assertEqual(trend["grid_rows"], 8)
        self.assertEqual(len(trend["groups"]), 4)
        self.assertEqual(len(trend["decay_trend_by_weight_combo"]), 2)

    def test_smoke_reports_organic_v2_metrics(self):
        mask = _synthetic_bcss_mask()
        payload = _proposal(points=[[20, 20], [44, 20], [44, 44], [20, 44]])
        proposal = validate_contour_proposal(
            payload,
            schema=self.schema,
            mask_shape=tuple(mask.shape),
            primitive="stromal_immune_infiltration",
            reference_profile="BCSS",
            target_label="Immune infiltrate",
            allowed_source_labels=("Stroma",),
        )

        smoke = run_organic_v2_smoke(
            mask,
            proposal,
            schema=self.schema,
            primitive_config=self.primitive_config,
            organic_seed=3,
        )

        self.assertTrue(smoke[PROJECTION_MODE_ORGANIC_V2]["validation_passed"])
        self.assertGreater(smoke[PROJECTION_MODE_ORGANIC_V2]["selected_pixels"], 0)
        self.assertTrue(smoke[PROJECTION_MODE_ORGANIC_V2]["metrics"]["label_safe"])

    def test_cli_writes_smoke_grid_and_trend_outputs(self):
        WORKSPACE_TMP.mkdir(exist_ok=True)
        tmp = WORKSPACE_TMP / f"cli_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True)
        try:
            mask_path = tmp / "source_mask.png"
            fixture_path = tmp / "proposal.json"
            output_dir = tmp / "experiment"
            save_id_mask(_synthetic_bcss_mask(), mask_path)
            save_metadata(
                _proposal(points=[[20, 20], [44, 20], [44, 44], [20, 44]]),
                fixture_path,
            )

            code = run_stage2_main(
                [
                    "--mask",
                    str(mask_path),
                    "--fixture",
                    str(fixture_path),
                    "--output",
                    str(output_dir),
                    "--decay-px",
                    "16,48",
                    "--seed",
                    "0,1",
                    "--weight-combo",
                    "balanced,0.45,0.45,0.10",
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue((output_dir / "smoke_summary.json").exists())
            self.assertTrue((output_dir / "grid_results.json").exists())
            self.assertTrue((output_dir / "grid_results.csv").exists())
            self.assertTrue((output_dir / "trend_summary.json").exists())
            grid = load_metadata(output_dir / "grid_results.json")
            self.assertEqual(len(grid["rows"]), 4)
            trend = load_metadata(output_dir / "trend_summary.json")
            self.assertEqual(trend["grid_rows"], 4)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _synthetic_bcss_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[8:56, 8:56] = 2
    mask[20:44, 20:44] = 1
    return mask


def _primitive(recipe, name):
    for primitive in recipe["primitives"]:
        if primitive["name"] == name:
            return primitive
    raise AssertionError(f"missing primitive {name}")


def _proposal(*, points):
    return {
        "schema_version": "0.1",
        "backend": "llm_contour_proposal",
        "primitive": "stromal_immune_infiltration",
        "reference_profile": "BCSS",
        "target_label": "Immune infiltrate",
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
                "source_labels": ["Stroma"],
                "points": points,
                "confidence": 0.8,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
