import unittest
from pathlib import Path

import numpy as np

from dataset_config.unified_labels import UNIFIED_COLOR_MAP
from phase3_mask_edit.backends.llm_contour import (
    CONTOUR_PROPOSAL_BACKEND,
    CONTOUR_PROPOSAL_SCHEMA_VERSION,
    ContourProposalValidationError,
    load_contour_proposal_json,
    rasterize_contour_proposal,
    rasterize_polygon,
    validate_contour_proposal,
)
from phase3_mask_edit.backends.llm_preview import (
    add_coordinate_grid_overlay,
    id_mask_to_llm_preview_rgb,
    llm_palette_legend,
)
from phase3_mask_edit.core.labels import MaskProfileSchema


class LLMContourProposalTests(unittest.TestCase):
    def setUp(self):
        self.schema = MaskProfileSchema.from_reference_profile("BCSS")
        self.mask_shape = (64, 64)

    def test_valid_polygon_rasterizes_to_binary_region(self):
        proposal = validate_contour_proposal(
            self._proposal(),
            schema=self.schema,
            mask_shape=self.mask_shape,
            primitive="stromal_immune_infiltration",
            allowed_source_labels=("Stroma",),
            target_label="Immune infiltrate",
        )

        region = rasterize_contour_proposal(proposal)

        self.assertEqual(region.shape, self.mask_shape)
        self.assertEqual(region.dtype, np.bool_)
        self.assertGreater(int(np.count_nonzero(region)), 0)
        self.assertTrue(region[20, 20])

    def test_loads_llm_json_file(self):
        path = Path("tests/fixtures/llm_contour_response.json")

        payload = load_contour_proposal_json(path)

        self.assertEqual(payload["backend"], CONTOUR_PROPOSAL_BACKEND)

    def test_out_of_bounds_point_is_rejected(self):
        payload = self._proposal(points=[[10, 10], [65, 10], [10, 30]])

        with self.assertRaisesRegex(ContourProposalValidationError, "outside mask bounds"):
            validate_contour_proposal(
                payload,
                schema=self.schema,
                mask_shape=self.mask_shape,
            )

    def test_too_few_points_are_rejected(self):
        payload = self._proposal(points=[[10, 10], [20, 10]])

        with self.assertRaisesRegex(ContourProposalValidationError, "at least 3 points"):
            validate_contour_proposal(
                payload,
                schema=self.schema,
                mask_shape=self.mask_shape,
            )

    def test_unknown_source_label_is_rejected(self):
        payload = self._proposal(source_labels=["Imaginary tissue"])

        with self.assertRaisesRegex(ContourProposalValidationError, "unknown label"):
            validate_contour_proposal(
                payload,
                schema=self.schema,
                mask_shape=self.mask_shape,
            )

    def test_unknown_target_label_is_rejected(self):
        payload = self._proposal(target_label="Imaginary tissue")

        with self.assertRaisesRegex(ContourProposalValidationError, "unknown label"):
            validate_contour_proposal(
                payload,
                schema=self.schema,
                mask_shape=self.mask_shape,
            )

    def test_coordinate_shape_mismatch_is_rejected(self):
        payload = self._proposal()
        payload["coordinate_system"]["width"] = 128

        with self.assertRaisesRegex(ContourProposalValidationError, "coordinate_system.width"):
            validate_contour_proposal(
                payload,
                schema=self.schema,
                mask_shape=self.mask_shape,
            )

    def test_region_and_point_limits_are_enforced(self):
        too_many_regions = self._proposal()
        too_many_regions["regions"] = [
            dict(too_many_regions["regions"][0], region_id=f"r{i}") for i in range(3)
        ]

        with self.assertRaisesRegex(ContourProposalValidationError, "maximum is 2"):
            validate_contour_proposal(
                too_many_regions,
                schema=self.schema,
                mask_shape=self.mask_shape,
                max_regions=2,
            )

        too_many_points = self._proposal(
            points=[[10 + i, 10] for i in range(6)]
        )
        with self.assertRaisesRegex(ContourProposalValidationError, "maximum is 5"):
            validate_contour_proposal(
                too_many_points,
                schema=self.schema,
                mask_shape=self.mask_shape,
                max_points_per_region=5,
            )

    def test_allowed_source_labels_are_enforced(self):
        payload = self._proposal(source_labels=["Stroma"])

        with self.assertRaisesRegex(ContourProposalValidationError, "not allowed"):
            validate_contour_proposal(
                payload,
                schema=self.schema,
                mask_shape=self.mask_shape,
                allowed_source_labels=("Tumor",),
            )

    def test_standalone_rasterize_polygon_validates_bounds(self):
        with self.assertRaisesRegex(ContourProposalValidationError, "outside mask bounds"):
            rasterize_polygon(
                [[0, 0], [63, 0], [64, 63]],
                mask_shape=self.mask_shape,
            )

    def _proposal(
        self,
        *,
        points=None,
        source_labels=None,
        target_label="Immune infiltrate",
    ):
        return {
            "schema_version": CONTOUR_PROPOSAL_SCHEMA_VERSION,
            "backend": CONTOUR_PROPOSAL_BACKEND,
            "primitive": "stromal_immune_infiltration",
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
                    "source_labels": source_labels or ["Stroma"],
                    "points": points or [[10, 10], [30, 10], [30, 30], [10, 30]],
                    "confidence": 0.8,
                }
            ],
        }


class LLMPreviewTests(unittest.TestCase):
    def test_preview_uses_dataset_config_palette_with_coarse_tumor_parent(self):
        mask = np.array([[0, 1, 14], [2, 3, 4]], dtype=np.int64)

        rgb = id_mask_to_llm_preview_rgb(mask)

        self.assertEqual(rgb.shape, (2, 3, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        np.testing.assert_array_equal(rgb[0, 0], UNIFIED_COLOR_MAP[0])
        np.testing.assert_array_equal(rgb[0, 1], UNIFIED_COLOR_MAP[1])
        np.testing.assert_array_equal(rgb[0, 2], UNIFIED_COLOR_MAP[1])
        np.testing.assert_array_equal(rgb[1, 0], UNIFIED_COLOR_MAP[2])

    def test_palette_legend_exposes_dataset_config_colors(self):
        legend = llm_palette_legend()

        self.assertEqual(legend["Tumor"], UNIFIED_COLOR_MAP[1])
        self.assertEqual(legend["Stroma"], UNIFIED_COLOR_MAP[2])
        self.assertEqual(legend["Immune infiltrate"], UNIFIED_COLOR_MAP[4])

    def test_grid_overlay_preserves_shape_and_draws_grid(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)

        overlay = add_coordinate_grid_overlay(rgb, grid_spacing_px=16)

        self.assertEqual(overlay.shape, rgb.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertGreater(int(np.count_nonzero(overlay)), 0)
        self.assertTrue(np.any(overlay[:, 16] != rgb[:, 16]))


if __name__ == "__main__":
    unittest.main()
