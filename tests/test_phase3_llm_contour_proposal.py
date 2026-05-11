import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

from dataset_config.unified_labels import UNIFIED_COLOR_MAP
from phase3_mask_edit.tools.generate_llm_contour_preview import _load_mask_auto
from phase3_mask_edit.backends.llm_contour import (
    CONTOUR_PROPOSAL_BACKEND,
    CONTOUR_PROPOSAL_SCHEMA_VERSION,
    ContourProposalValidationError,
    PROJECTION_MODE_HARD_V1,
    PROJECTION_MODE_COMPARE_V1_V2,
    PROJECTION_MODE_ORGANIC_V2,
    execute_contour_proposal_write,
    load_contour_proposal_json,
    rasterize_contour_proposal,
    rasterize_polygon,
    smooth_candidate_region,
    validate_contour_proposal,
)
from phase3_mask_edit.backends.organic_projection import (
    ORGANIC_PROJECTION_BACKEND,
    apply_organic_projected_label_write,
)
from phase3_mask_edit.backends.proposal_execution import apply_projected_label_write
from phase3_mask_edit.backends.llm_prompt import build_repair_feedback
from phase3_mask_edit.backends.llm_preview import (
    add_coordinate_grid_overlay,
    id_mask_to_llm_preview_rgb,
    llm_palette_legend,
)
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.validation import validate_edit_result
from phase3_mask_edit.generic.tumor_burden import PrimitiveEditResult


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

    def test_fixture_triangle_rasterizes_expected_inside_and_outside_pixels(self):
        region = rasterize_polygon(
            [[8, 8], [40, 8], [8, 40]],
            mask_shape=self.mask_shape,
        )

        self.assertTrue(region[12, 12])
        self.assertTrue(region[8, 24])
        self.assertFalse(region[40, 40])
        self.assertFalse(region[5, 5])

    def test_fixture_concave_polygon_rasterizes_without_filling_notch(self):
        region = rasterize_polygon(
            [[8, 8], [40, 8], [40, 20], [22, 20], [22, 40], [8, 40]],
            mask_shape=self.mask_shape,
        )

        self.assertTrue(region[12, 12])
        self.assertTrue(region[30, 12])
        self.assertFalse(region[30, 30])
        self.assertFalse(region[45, 45])

    def test_fixture_edge_touching_polygon_rasterizes_at_mask_border(self):
        region = rasterize_polygon(
            [[0, 0], [20, 0], [20, 20], [0, 20]],
            mask_shape=self.mask_shape,
        )

        self.assertTrue(region[0, 0])
        self.assertTrue(region[10, 10])
        self.assertFalse(region[21, 21])

    def test_fixture_tiny_polygon_rasterizes_to_small_nonempty_region(self):
        region = rasterize_polygon(
            [[10, 10], [12, 10], [10, 12]],
            mask_shape=self.mask_shape,
        )

        self.assertGreater(int(np.count_nonzero(region)), 0)
        self.assertLessEqual(int(np.count_nonzero(region)), 6)
        self.assertTrue(region[10, 10])

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

    def test_wrong_schema_version_is_rejected(self):
        payload = self._proposal()
        payload["schema_version"] = "999"

        with self.assertRaisesRegex(ContourProposalValidationError, "schema_version"):
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

    def test_known_optional_v2_fields_are_allowed_by_strict_validator(self):
        payload = self._proposal()
        payload["template_role"] = "coarse_template"
        payload["placement_relation"] = "tumor_adjacent_stroma"
        payload["shape_hints"] = ["patchy", "irregular_boundary"]
        payload["regions"][0]["source_component_ids"] = ["source_1"]
        payload["regions"][0]["adjacency_side"] = "tumor_adjacent_stroma"
        payload["regions"][0]["template_role"] = "coarse_template"

        proposal = validate_contour_proposal(
            payload,
            schema=self.schema,
            mask_shape=self.mask_shape,
            allowed_source_labels=("Stroma",),
        )

        self.assertEqual(proposal.raw_payload["template_role"], "coarse_template")
        self.assertEqual(
            proposal.raw_payload["regions"][0]["source_component_ids"],
            ["source_1"],
        )

    def test_unknown_payload_fields_are_rejected_even_with_v2_allowlist(self):
        payload = self._proposal()
        payload["surprise_field"] = True

        with self.assertRaisesRegex(ContourProposalValidationError, "unknown field"):
            validate_contour_proposal(
                payload,
                schema=self.schema,
                mask_shape=self.mask_shape,
            )

    def test_invalid_template_role_is_rejected(self):
        payload = self._proposal()
        payload["template_role"] = "final_mask"

        with self.assertRaisesRegex(ContourProposalValidationError, "template_role"):
            validate_contour_proposal(
                payload,
                schema=self.schema,
                mask_shape=self.mask_shape,
            )

    def test_standalone_rasterize_polygon_validates_bounds(self):
        with self.assertRaisesRegex(ContourProposalValidationError, "outside mask bounds"):
            rasterize_polygon(
                [[0, 0], [63, 0], [64, 63]],
                mask_shape=self.mask_shape,
            )

    def test_projected_write_only_changes_legal_source_label_pixels(self):
        old_mask = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 2, 2, 0],
                [0, 1, 1, 2, 2, 0],
                [0, 3, 3, 2, 2, 0],
                [0, 3, 3, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        candidate = np.ones_like(old_mask, dtype=bool)

        result = apply_projected_label_write(
            old_mask,
            candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            backend=CONTOUR_PROPOSAL_BACKEND,
        )

        self.assertEqual(result.selected_pixels, 8)
        self.assertTrue(np.all(result.target_mask[old_mask == 2] == 4))
        tumor_coords = np.argwhere(old_mask == 1)
        np.testing.assert_array_equal(
            result.target_mask[tumor_coords[:, 0], tumor_coords[:, 1]],
            old_mask[tumor_coords[:, 0], tumor_coords[:, 1]],
        )
        self.assertTrue(np.all(result.target_mask[old_mask == 3] == 3))
        self.assertTrue(np.all(result.target_mask[old_mask == 0] == 0))
        self.assertTrue(np.all(old_mask[result.change_region] == 2))

    def test_projection_retained_fraction_tracks_large_illegal_polygon(self):
        old_mask = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 2, 2, 0],
                [0, 1, 1, 2, 2, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        candidate = np.ones_like(old_mask, dtype=bool)

        result = apply_projected_label_write(
            old_mask,
            candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            backend=CONTOUR_PROPOSAL_BACKEND,
        )

        self.assertEqual(result.ops_log["candidate_pixels"], 36)
        self.assertEqual(result.ops_log["projected_pixels"], 4)
        self.assertEqual(result.ops_log["source_projected_pixels"], 4)
        self.assertGreater(
            result.ops_log["candidate_pixels"],
            result.ops_log["projected_pixels"],
        )
        self.assertAlmostEqual(result.ops_log["projection_retained_fraction"], 4 / 36)

    def test_projected_write_removes_preserve_and_forbidden_labels(self):
        old_mask = np.array(
            [
                [2, 2, 5],
                [2, 5, 1],
                [1, 2, 2],
            ],
            dtype=np.int64,
        )
        candidate = np.ones_like(old_mask, dtype=bool)

        result = apply_projected_label_write(
            old_mask,
            candidate,
            schema=self.schema,
            source_labels=("Stroma", "Blood vessel", "Tumor"),
            target_label="Immune infiltrate",
            preserve_labels=("Blood vessel",),
            forbidden_labels=("Tumor",),
        )

        self.assertEqual(result.selected_pixels, 5)
        self.assertTrue(np.all(result.target_mask[old_mask == 2] == 4))
        self.assertTrue(np.all(result.target_mask[old_mask == 5] == 5))
        tumor_coords = np.argwhere(old_mask == 1)
        np.testing.assert_array_equal(
            result.target_mask[tumor_coords[:, 0], tumor_coords[:, 1]],
            old_mask[tumor_coords[:, 0], tumor_coords[:, 1]],
        )

    def test_contour_write_projects_each_region_with_its_own_source_labels(self):
        old_mask = np.zeros(self.mask_shape, dtype=np.int64)
        old_mask[10:31, 10:21] = 2
        old_mask[10:31, 21:31] = 1
        old_mask[10:31, 34:55] = 1
        payload = self._proposal()
        payload["regions"] = [
            {
                "region_id": "stroma_region",
                "type": "polygon",
                "source_labels": ["Stroma"],
                "points": [[10, 10], [30, 10], [30, 30], [10, 30]],
            },
            {
                "region_id": "tumor_region",
                "type": "polygon",
                "source_labels": ["Tumor"],
                "points": [[34, 10], [54, 10], [54, 30], [34, 30]],
            },
        ]
        proposal = validate_contour_proposal(
            payload,
            schema=self.schema,
            mask_shape=self.mask_shape,
            allowed_source_labels=("Stroma", "Tumor"),
        )

        result = execute_contour_proposal_write(old_mask, proposal, schema=self.schema)

        self.assertGreater(result.selected_pixels, 0)
        self.assertEqual(result.ops_log["projection_mode"], PROJECTION_MODE_HARD_V1)
        self.assertEqual(
            result.ops_log["projection_fallback_reason"],
            "organic_v2_mvp_requires_uniform_region_source_labels",
        )
        self.assertTrue(np.all(result.target_mask[10:31, 10:21] == 4))
        region_a_tumor_coords = np.argwhere(old_mask[10:31, 21:31] == 1)
        region_a_tumor_rows = region_a_tumor_coords[:, 0] + 10
        region_a_tumor_cols = region_a_tumor_coords[:, 1] + 21
        np.testing.assert_array_equal(
            result.target_mask[region_a_tumor_rows, region_a_tumor_cols],
            old_mask[region_a_tumor_rows, region_a_tumor_cols],
        )
        self.assertTrue(np.all(result.target_mask[10:31, 34:55] == 4))
        self.assertEqual(len(result.ops_log["region_projection"]), 2)
        region_logs = {item["region_id"]: item for item in result.ops_log["region_projection"]}
        self.assertGreater(
            region_logs["stroma_region"]["candidate_pixels"],
            region_logs["stroma_region"]["projected_pixels"],
        )

    def test_empty_projection_is_left_for_validation_feedback(self):
        old_mask = np.ones(self.mask_shape, dtype=np.int64)
        proposal = validate_contour_proposal(
            self._proposal(),
            schema=self.schema,
            mask_shape=self.mask_shape,
            allowed_source_labels=("Stroma",),
        )

        result = execute_contour_proposal_write(
            old_mask,
            proposal,
            schema=self.schema,
            projection_mode=PROJECTION_MODE_HARD_V1,
        )

        self.assertEqual(result.selected_pixels, 0)
        self.assertIn("proposal_projected_region_empty", result.warnings)
        validation = validate_edit_result(
            src_mask=old_mask,
            target_mask=result.target_mask,
            change_region=result.change_region,
            schema=self.schema,
            primitive_config={
                "name": "stromal_immune_infiltration",
                "required_tissue_labels": ["Stroma"],
                "parameter_ranges": {},
                "validation_rules": ["immune_area_must_increase"],
            },
            changed_area_fraction=result.changed_area_fraction,
        )
        self.assertFalse(validation.passed)
        self.assertFalse(
            next(c for c in validation.checks if c.name == "change_area_nonempty").passed
        )

    def test_smooth_candidate_region_preserves_nonempty_region(self):
        candidate = np.zeros((64, 64), dtype=bool)
        candidate[20:40, 20:40] = True
        candidate[19, 19] = True
        candidate[40, 40] = True

        smoothed = smooth_candidate_region(candidate)

        self.assertEqual(smoothed.shape, candidate.shape)
        self.assertEqual(smoothed.dtype, np.bool_)
        self.assertGreater(int(np.count_nonzero(smoothed)), 0)
        self.assertTrue(smoothed[30, 30])

    def test_contour_write_smoothing_still_projects_to_source_labels(self):
        old_mask = np.zeros(self.mask_shape, dtype=np.int64)
        old_mask[8:56, 8:56] = 2
        old_mask[20:44, 20:44] = 1
        proposal = validate_contour_proposal(
            self._proposal(points=[[10, 10], [54, 10], [54, 54], [10, 54]]),
            schema=self.schema,
            mask_shape=self.mask_shape,
            allowed_source_labels=("Stroma",),
        )

        result = execute_contour_proposal_write(
            old_mask,
            proposal,
            schema=self.schema,
            projection_mode=PROJECTION_MODE_HARD_V1,
        )

        self.assertGreater(result.selected_pixels, 0)
        changed_old_labels = set(np.unique(old_mask[result.change_region]).astype(int).tolist())
        self.assertEqual(changed_old_labels, {2})

    def test_contour_write_defaults_to_organic_v2(self):
        old_mask = np.zeros(self.mask_shape, dtype=np.int64)
        old_mask[8:56, 8:56] = 2
        old_mask[20:44, 20:44] = 1
        proposal = validate_contour_proposal(
            self._proposal(points=[[10, 10], [24, 10], [24, 24], [10, 24]]),
            schema=self.schema,
            mask_shape=self.mask_shape,
            allowed_source_labels=("Stroma",),
        )

        result = execute_contour_proposal_write(
            old_mask,
            proposal,
            schema=self.schema,
            primitive_config={
                "name": "stromal_immune_infiltration",
                "parameter_ranges": {
                    "immune_area_delta_fraction": {"mild": [0.08, 0.14]},
                },
            },
        )

        self.assertEqual(result.ops_log["projection_mode"], PROJECTION_MODE_ORGANIC_V2)
        self.assertEqual(result.ops_log["projection_backend"], ORGANIC_PROJECTION_BACKEND)
        self.assertTrue(np.all(old_mask[result.change_region] == 2))

    def test_compare_mode_is_rejected_by_single_write_executor(self):
        proposal = validate_contour_proposal(
            self._proposal(),
            schema=self.schema,
            mask_shape=self.mask_shape,
            allowed_source_labels=("Stroma",),
        )

        with self.assertRaisesRegex(ContourProposalValidationError, "orchestration"):
            execute_contour_proposal_write(
                np.zeros(self.mask_shape, dtype=np.int64),
                proposal,
                schema=self.schema,
                projection_mode=PROJECTION_MODE_COMPARE_V1_V2,
            )

    def test_validation_detects_projected_area_too_small(self):
        old_mask = np.zeros((20, 20), dtype=np.int64)
        old_mask[:, :] = 2
        candidate = np.zeros_like(old_mask, dtype=bool)
        candidate[1, 1] = True

        result = apply_projected_label_write(
            old_mask,
            candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
        )
        validation = validate_edit_result(
            src_mask=old_mask,
            target_mask=result.target_mask,
            change_region=result.change_region,
            schema=self.schema,
            primitive_config={
                "name": "stromal_immune_infiltration",
                "required_tissue_labels": ["Stroma"],
                "parameter_ranges": {
                    "immune_area_delta_fraction": {"mild": [0.08, 0.14]},
                    "max_changed_area_fraction": 0.40,
                },
                "validation_rules": ["immune_area_must_increase"],
            },
            changed_area_fraction=result.changed_area_fraction,
        )

        self.assertFalse(validation.passed)
        self.assertFalse(
            next(c for c in validation.checks if c.name == "change_area_within_range").passed
        )

    def test_organic_projection_selects_area_inside_legal_stroma(self):
        old_mask = np.zeros((64, 64), dtype=np.int64)
        old_mask[6:58, 6:58] = 2
        old_mask[22:42, 22:42] = 1
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[10:20, 10:20] = True

        primitive_config = {
            "name": "stromal_immune_infiltration",
            "parameter_ranges": {
                "immune_area_delta_fraction": {"mild": [0.08, 0.14]},
                "peritumoral_falloff_radius_px": 32,
            },
        }
        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config=primitive_config,
            seed=7,
        )

        self.assertEqual(result.ops_log["backend"], ORGANIC_PROJECTION_BACKEND)
        self.assertGreater(result.selected_pixels, int(np.count_nonzero(raw_candidate & (old_mask == 2))))
        self.assertTrue(np.all(old_mask[result.change_region] == 2))
        self.assertTrue(np.all(result.target_mask[result.change_region] == 4))
        validation = validate_edit_result(
            src_mask=old_mask,
            target_mask=result.target_mask,
            change_region=result.change_region,
            schema=self.schema,
            primitive_config={
                "name": "stromal_immune_infiltration",
                "required_tissue_labels": ["Stroma"],
                "parameter_ranges": {
                    "immune_area_delta_fraction": {"mild": [0.08, 0.14]},
                    "max_changed_area_fraction": 0.40,
                },
                "validation_rules": ["immune_area_must_increase"],
            },
            changed_area_fraction=result.changed_area_fraction,
        )
        self.assertTrue(validation.passed)

    def test_organic_projection_rejects_template_with_no_legal_overlap(self):
        old_mask = np.zeros((64, 64), dtype=np.int64)
        old_mask[8:56, 8:56] = 2
        old_mask[24:40, 24:40] = 1
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[25:35, 25:35] = True

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config={
                "name": "stromal_immune_infiltration",
                "parameter_ranges": {
                    "immune_area_delta_fraction": {"mild": [0.08, 0.14]},
                },
            },
            seed=5,
            target_pixels=100,
        )

        self.assertEqual(result.selected_pixels, 0)
        self.assertEqual(result.ops_log["raw_candidate_legal_overlap_pixels"], 0)
        self.assertEqual(result.ops_log["selected_raw_template_iou"], 0.0)
        self.assertIn("organic_projection_template_no_legal_overlap", result.warnings)
        np.testing.assert_array_equal(result.target_mask, old_mask)

    def test_organic_projection_mixed_label_matrix_only_writes_source(self):
        old_mask = np.zeros((12, 12), dtype=np.int64)
        old_mask[1:5, 1:5] = 2
        old_mask[1:5, 5:8] = 4
        old_mask[5:8, 1:5] = 5
        old_mask[5:8, 5:9] = 1
        old_mask[8:11, 1:5] = 3
        raw_candidate = np.ones_like(old_mask, dtype=bool)

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma", "Blood vessel", "Tumor"),
            target_label="Immune infiltrate",
            preserve_labels=("Blood vessel",),
            forbidden_labels=("Tumor",),
            primitive_config={"name": "unknown_future_primitive"},
            seed=2,
            target_pixels=12,
        )

        self.assertEqual(result.selected_pixels, 12)
        self.assertEqual(result.ops_log["raw_candidate_pixels"], int(raw_candidate.size))
        self.assertEqual(result.ops_log["legal_domain_pixels"], 16)
        self.assertEqual(result.ops_log["raw_candidate_legal_overlap_pixels"], 16)
        self.assertAlmostEqual(
            result.ops_log["template_overlap_with_legal_domain"],
            16 / raw_candidate.size,
        )
        self.assertTrue(np.all(old_mask[result.change_region] == 2))
        self.assertTrue(np.all(result.target_mask[result.change_region] == 4))
        self.assertTrue(np.all(result.target_mask[old_mask == 5] == 5))
        self.assertTrue(np.all(result.target_mask[old_mask == 1] == 1))
        self.assertTrue(np.all(result.target_mask[old_mask == 0] == 0))
        self.assertTrue(np.all(result.target_mask[old_mask == 3] == 3))

    def test_organic_projection_low_overlap_regression_uses_score_top_k(self):
        old_mask = np.zeros((80, 80), dtype=np.int64)
        old_mask[8:72, 8:72] = 2
        old_mask[30:50, 30:50] = 1
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[:, 0:10] = True
        raw_candidate[0:8, 10:50] = True
        raw_overlap = int(np.count_nonzero(raw_candidate & (old_mask == 2)))
        raw_pixels = int(np.count_nonzero(raw_candidate))
        target_pixels = 900

        hard = apply_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
        )
        organic = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config={
                "name": "stromal_immune_infiltration",
                "parameter_ranges": {
                    "immune_area_delta_fraction": {"mild": [0.08, 0.14]},
                    "organic_min_component_fraction": 0.0,
                    "peritumoral_falloff_radius_px": 32,
                },
            },
            seed=17,
            target_pixels=target_pixels,
        )

        self.assertGreaterEqual(int(np.count_nonzero(old_mask == 2)), 3 * target_pixels)
        self.assertGreaterEqual(raw_overlap / raw_pixels, 0.10)
        self.assertLessEqual(raw_overlap / raw_pixels, 0.15)
        self.assertLess(hard.selected_pixels, target_pixels)
        self.assertEqual(organic.selected_pixels, target_pixels)
        self.assertGreater(organic.selected_pixels, raw_overlap)
        self.assertEqual(
            organic.ops_log["raw_candidate_legal_overlap_pixels"],
            raw_overlap,
        )
        self.assertAlmostEqual(
            organic.ops_log["template_overlap_with_legal_domain"],
            raw_overlap / raw_pixels,
        )
        self.assertIn("score_terms", organic.ops_log)
        self.assertEqual(organic.ops_log["legal_domain_pixels"], int(np.count_nonzero(old_mask == 2)))
        self.assertEqual(organic.ops_log["target_pixels"], target_pixels)
        self.assertGreater(organic.ops_log["selected_raw_template_union_pixels"], 0)

    def test_organic_projection_area_shortfall_logs_legal_domain_insufficient(self):
        old_mask = np.zeros((20, 20), dtype=np.int64)
        old_mask[4:8, 4:8] = 2
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[3:9, 3:9] = True

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config={
                "name": "stromal_immune_infiltration",
                "parameter_ranges": {
                    "organic_min_component_fraction": 0.0,
                    "organic_min_template_legal_overlap_fraction": 0.0,
                },
            },
            seed=6,
            target_pixels=40,
        )

        self.assertEqual(result.ops_log["legal_domain_pixels"], 16)
        self.assertEqual(result.ops_log["target_pixels"], 40)
        self.assertEqual(result.selected_pixels, 16)
        self.assertEqual(result.ops_log["selected_pixels"], 16)
        self.assertEqual(result.ops_log["area_shortfall"], 24)
        self.assertIn("organic_projection_area_shortfall", result.warnings)
        self.assertTrue(np.all(old_mask[result.change_region] == 2))

    def test_organic_projection_prefers_template_neighborhood_before_spillover(self):
        old_mask = np.zeros((96, 96), dtype=np.int64)
        old_mask[8:88, 8:88] = 2
        old_mask[34:62, 54:82] = 1
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[24:48, 12:36] = True
        target_pixels = 360

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config={
                "name": "stromal_immune_infiltration",
                "parameter_ranges": {
                    "immune_area_delta_fraction": {"mild": [0.08, 0.14]},
                    "organic_score_weights": {
                        "template": 0.20,
                        "spatial": 0.75,
                        "noise": 0.05,
                    },
                    "organic_template_neighborhood_radius_px": 12,
                    "organic_template_spillover_fraction": 0.15,
                    "organic_min_template_legal_overlap_fraction": 0.01,
                    "organic_min_selected_template_iou": 0.0,
                    "organic_min_component_fraction": 0.0,
                    "peritumoral_falloff_radius_px": 16,
                },
            },
            seed=4,
            target_pixels=target_pixels,
        )

        policy = result.ops_log["selection_policy"]
        self.assertEqual(policy["name"], "template_neighborhood_constrained_top_k")
        self.assertEqual(result.selected_pixels, target_pixels)
        self.assertGreaterEqual(
            policy["selected_inside_primary_zone_pixels"],
            int(round(target_pixels * 0.85)),
        )
        self.assertLessEqual(
            policy["selected_outside_primary_zone_pixels"],
            int(round(target_pixels * 0.15)),
        )
        self.assertTrue(np.all(old_mask[result.change_region] == 2))

    def test_stromal_immune_policy_prefers_peritumoral_stroma(self):
        old_mask = np.zeros((80, 80), dtype=np.int64)
        old_mask[:, :] = 2
        old_mask[24:56, 8:24] = 1
        raw_candidate = old_mask == 2

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config={
                "name": "stromal_immune_infiltration",
                "parameter_ranges": {
                    "organic_score_weights": {
                        "template": 0.0,
                        "spatial": 1.0,
                        "noise": 0.0,
                    },
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 128,
                    "organic_template_spillover_fraction": 0.0,
                    "peritumoral_falloff_radius_px": 10,
                },
            },
            seed=1,
            target_pixels=220,
        )

        self.assertEqual(
            result.ops_log["component_policy"]["policy_name"],
            "stromal_immune_peritumoral",
        )
        self.assertEqual(result.selected_pixels, 220)
        self.assertTrue(np.all(old_mask[result.change_region] == 2))
        selected_cols = np.argwhere(result.change_region)[:, 1]
        all_stroma_cols = np.argwhere(old_mask == 2)[:, 1]
        self.assertLess(float(np.mean(selected_cols)), float(np.mean(all_stroma_cols)))
        self.assertLessEqual(int(np.max(selected_cols)), 38)

    def test_stromal_immune_policy_uses_existing_immune_adjacency(self):
        old_mask = np.zeros((64, 64), dtype=np.int64)
        old_mask[:, :] = 2
        old_mask[28:36, 4:12] = 4
        raw_candidate = old_mask == 2

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config={
                "name": "stromal_immune_infiltration",
                "parameter_ranges": {
                    "organic_score_weights": {
                        "template": 0.0,
                        "spatial": 1.0,
                        "noise": 0.0,
                    },
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 128,
                    "organic_template_spillover_fraction": 0.0,
                    "immune_neighbor_radius_px": 8,
                },
            },
            seed=1,
            target_pixels=80,
        )

        params = result.ops_log["component_policy"]["params"]
        self.assertTrue(params["used_existing_immune_adjacency"])
        self.assertTrue(np.all(old_mask[result.change_region] == 2))
        selected_cols = np.argwhere(result.change_region)[:, 1]
        self.assertLess(float(np.mean(selected_cols)), 18.0)

    def test_stromal_immune_policy_penalizes_necrosis_adjacency(self):
        old_mask = np.zeros((64, 64), dtype=np.int64)
        old_mask[:, :] = 2
        old_mask[20:44, 20:44] = 1
        old_mask[20:44, 10:18] = 3
        raw_candidate = old_mask == 2

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config={
                "name": "stromal_immune_infiltration",
                "parameter_ranges": {
                    "organic_score_weights": {
                        "template": 0.0,
                        "spatial": 1.0,
                        "noise": 0.0,
                    },
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 128,
                    "organic_template_spillover_fraction": 0.0,
                    "peritumoral_falloff_radius_px": 8,
                    "necrosis_adjacency_penalty_radius_px": 12,
                    "necrosis_adjacency_penalty_weight": 1.5,
                },
            },
            seed=1,
            target_pixels=100,
        )

        params = result.ops_log["component_policy"]["params"]
        self.assertTrue(params["used_necrosis_adjacency_penalty"])
        self.assertTrue(np.all(old_mask[result.change_region] == 2))
        selected_cols = np.argwhere(result.change_region)[:, 1]
        self.assertGreater(float(np.mean(selected_cols)), 42.0)

    def test_organic_projection_necrosis_policy_caps_area_like_validation(self):
        old_mask = np.zeros((80, 80), dtype=np.int64)
        old_mask[8:72, 8:72] = 2
        old_mask[20:60, 20:60] = 1
        tumor_coords = np.argwhere(old_mask == 1)
        existing_count = int(round(tumor_coords.shape[0] * 0.30))
        old_mask[tumor_coords[:existing_count, 0], tumor_coords[:existing_count, 1]] = 3
        raw_candidate = np.ones_like(old_mask, dtype=bool)
        original_tumor_pixels = int(np.count_nonzero(old_mask == 1))
        existing_necrosis_pixels = int(np.count_nonzero(old_mask == 3))
        max_fraction = 0.60
        expected_remaining = int(round(original_tumor_pixels * max_fraction)) - existing_necrosis_pixels

        primitive_config = {
            "name": "necrosis_appearance",
            "required_tissue_labels": ["Tumor"],
            "parameter_ranges": {
                "target_changed_area_fraction": {"mild": [0.08, 0.14]},
                "max_necrosis_fraction_of_tumor": max_fraction,
                "organic_min_component_fraction": 0.0,
            },
            "validation_rules": [
                "necrosis_area_must_increase",
                "new_necrosis_must_be_inside_original_tumor",
            ],
        }
        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Necrosis",
            primitive_config=primitive_config,
            seed=9,
            target_pixels=500,
        )

        self.assertEqual(result.selected_pixels, expected_remaining)
        self.assertTrue(np.all(old_mask[result.change_region] == 1))
        policy = result.ops_log["component_policy"]
        self.assertEqual(policy["policy_name"], "necrosis_intratumoral_hypoxic")
        self.assertEqual(
            policy["params"]["necrosis_denominator_policy"],
            "original_tumor_fine_ids_only_matches_validation",
        )
        self.assertEqual(policy["params"]["tumor_pixels"], original_tumor_pixels)
        self.assertEqual(
            policy["params"]["existing_necrosis_pixels"],
            existing_necrosis_pixels,
        )
        validation = validate_edit_result(
            src_mask=old_mask,
            target_mask=result.target_mask,
            change_region=result.change_region,
            schema=self.schema,
            primitive_config=primitive_config,
            changed_area_fraction=result.changed_area_fraction,
        )
        self.assertTrue(validation.passed)

    def test_necrosis_policy_prefers_tumor_interior_over_outer_boundary(self):
        old_mask = np.zeros((72, 72), dtype=np.int64)
        old_mask[8:64, 8:64] = 2
        old_mask[12:60, 12:60] = 1
        raw_candidate = old_mask == 1

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Necrosis",
            primitive_config={
                "name": "necrosis_appearance",
                "parameter_ranges": {
                    "organic_score_weights": {
                        "template": 0.0,
                        "spatial": 1.0,
                        "noise": 0.0,
                    },
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 128,
                    "organic_template_spillover_fraction": 0.0,
                    "tumor_boundary_margin_radius_px": 18,
                    "max_necrosis_fraction_of_tumor": 1.0,
                },
            },
            seed=4,
            target_pixels=180,
        )

        rows, cols = np.where(result.change_region)
        self.assertEqual(
            result.ops_log["component_policy"]["policy_name"],
            "necrosis_intratumoral_hypoxic",
        )
        self.assertEqual(result.selected_pixels, 180)
        self.assertTrue(np.all(old_mask[result.change_region] == 1))
        self.assertGreaterEqual(int(rows.min()), 20)
        self.assertLessEqual(int(rows.max()), 51)
        self.assertGreaterEqual(int(cols.min()), 20)
        self.assertLessEqual(int(cols.max()), 51)

    def test_necrosis_policy_uses_existing_necrosis_adjacency(self):
        old_mask = np.zeros((72, 72), dtype=np.int64)
        old_mask[8:64, 8:64] = 2
        old_mask[12:60, 12:60] = 1
        old_mask[30:42, 14:22] = 3
        raw_candidate = old_mask == 1

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Necrosis",
            primitive_config={
                "name": "necrosis_appearance",
                "parameter_ranges": {
                    "organic_score_weights": {
                        "template": 0.0,
                        "spatial": 1.0,
                        "noise": 0.0,
                    },
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 128,
                    "organic_template_spillover_fraction": 0.0,
                    "tumor_boundary_margin_radius_px": 200,
                    "necrosis_neighbor_radius_px": 10,
                    "max_necrosis_fraction_of_tumor": 1.0,
                },
            },
            seed=1,
            target_pixels=120,
        )

        params = result.ops_log["component_policy"]["params"]
        self.assertTrue(params["used_existing_necrosis_neighborhood"])
        self.assertTrue(np.all(old_mask[result.change_region] == 1))
        selected_cols = np.argwhere(result.change_region)[:, 1]
        self.assertLess(float(np.mean(selected_cols)), 30.0)

    def test_necrosis_policy_avoids_blood_vessel_neighborhood(self):
        old_mask = np.zeros((72, 72), dtype=np.int64)
        old_mask[8:64, 8:64] = 2
        old_mask[12:60, 12:60] = 1
        old_mask[28:44, 14:22] = 6
        raw_candidate = old_mask == 1

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Necrosis",
            primitive_config={
                "name": "necrosis_appearance",
                "parameter_ranges": {
                    "organic_score_weights": {
                        "template": 0.0,
                        "spatial": 1.0,
                        "noise": 0.0,
                    },
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 128,
                    "organic_template_spillover_fraction": 0.0,
                    "tumor_boundary_margin_radius_px": 200,
                    "vessel_avoidance_radius_px": 24,
                    "vessel_avoidance_weight": 1.0,
                    "max_necrosis_fraction_of_tumor": 1.0,
                },
            },
            seed=1,
            target_pixels=120,
        )

        params = result.ops_log["component_policy"]["params"]
        self.assertTrue(params["used_vessel_avoidance"])
        self.assertTrue(np.all(old_mask[result.change_region] == 1))
        selected_cols = np.argwhere(result.change_region)[:, 1]
        self.assertGreater(float(np.mean(selected_cols)), 38.0)

    def test_organic_projection_cleanup_refill_is_single_pass(self):
        old_mask = np.zeros((32, 32), dtype=np.int64)
        old_mask[2:30, 2:30] = 2
        old_mask[12:20, 12:20] = 1
        raw_candidate = np.ones_like(old_mask, dtype=bool)

        primitive_config = {
            "name": "stromal_immune_infiltration",
            "parameter_ranges": {
                "immune_area_delta_fraction": {"mild": [0.08, 0.14]},
                "organic_min_component_fraction": 0.50,
                "organic_fill_holes_max_area_px": 0,
            },
        }
        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config=primitive_config,
            seed=2,
            target_pixels=20,
        )

        self.assertEqual(result.ops_log["cleanup_iteration_limit"], 1)
        self.assertTrue(result.ops_log["cleanup_single_pass"])
        self.assertEqual(result.ops_log["target_pixels"], 20)
        self.assertLessEqual(result.selected_pixels, 20)
        self.assertEqual(
            result.ops_log["post_cleanup_pixels"],
            result.selected_pixels,
        )
        self.assertTrue(np.all(old_mask[result.change_region] == 2))

    def test_intratumoral_immune_policy_uses_spot_policy_cleanup_floor(self):
        old_mask = np.zeros((48, 48), dtype=np.int64)
        old_mask[6:42, 6:42] = 1
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[20:24, 20:24] = True

        primitive_config = {
            "name": "intratumoral_immune_infiltration",
            "required_tissue_labels": ["Tumor", "Immune infiltrate"],
            "spatial_pattern": {
                "region": "inside_tumor",
                "spot_policy": {
                    "max_total_area_fraction_of_tumor": 0.30,
                    "min_spot_area_px": 12,
                    "max_spot_area_px": 256,
                    "max_spots_per_patch": 32,
                },
            },
            "parameter_ranges": {
                "target_changed_area_fraction": {"mild": [0.05, 0.10]},
                "max_changed_area_fraction": 0.30,
                "organic_min_component_fraction": 0.50,
                "organic_fill_holes_max_area_px": 0,
                "organic_template_neighborhood_radius_px": 16,
                "organic_template_spillover_fraction": 0.15,
                "organic_min_template_legal_overlap_fraction": 0.0,
                "organic_min_selected_template_iou": 0.0,
            },
            "validation_rules": [
                "new_immune_must_be_inside_original_tumor",
                "tumor_body_must_remain",
            ],
        }
        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Immune infiltrate",
            primitive_config=primitive_config,
            seed=11,
            target_pixels=12,
        )

        self.assertEqual(
            result.ops_log["component_policy"]["policy_name"],
            "intratumoral_immune_til_spots",
        )
        self.assertEqual(result.ops_log["cleanup_min_component_policy"], "spot_policy.min_spot_area_px")
        self.assertEqual(result.ops_log["cleanup_min_component_pixels"], 12)
        self.assertEqual(result.selected_pixels, 12)
        self.assertTrue(np.all(old_mask[result.change_region] == 1))
        self.assertTrue(np.all(result.target_mask[result.change_region] == 4))

    def test_intratumoral_immune_policy_area_and_label_safety_contract(self):
        old_mask = np.zeros((72, 72), dtype=np.int64)
        old_mask[8:64, 8:64] = 1
        old_mask[24:40, 24:40] = 3
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[18:28, 18:28] = True
        raw_candidate[26:50, 26:50] = True

        primitive_config = {
            "name": "intratumoral_immune_infiltration",
            "required_tissue_labels": ["Tumor", "Immune infiltrate"],
            "spatial_pattern": {
                "region": "inside_tumor",
                "spot_policy": {
                    "max_total_area_fraction_of_tumor": 0.30,
                    "min_spot_area_px": 12,
                    "max_spot_area_px": 256,
                    "max_spots_per_patch": 32,
                },
            },
            "parameter_ranges": {
                "target_changed_area_fraction": {"mild": [0.05, 0.10]},
                "max_changed_area_fraction": 0.30,
                "organic_min_component_fraction": 0.0,
                "organic_fill_holes_max_area_px": 0,
                "organic_template_neighborhood_radius_px": 20,
                "organic_template_spillover_fraction": 0.15,
                "organic_min_template_legal_overlap_fraction": 0.0,
                "organic_min_selected_template_iou": 0.0,
                "organic_score_weights": {
                    "template": 0.35,
                    "spatial": 0.55,
                    "noise": 0.10,
                },
            },
            "validation_rules": [
                "new_immune_must_be_inside_original_tumor",
                "tumor_body_must_remain",
                "no_bias_only_edit_without_immune_tissue_label",
            ],
        }
        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Immune infiltrate",
            primitive_config=primitive_config,
            seed=3,
            target_pixels=180,
        )

        self.assertEqual(
            result.ops_log["component_policy"]["policy_name"],
            "intratumoral_immune_til_spots",
        )
        self.assertEqual(result.ops_log["projection_backend"], ORGANIC_PROJECTION_BACKEND)
        self.assertEqual(result.selected_pixels, 180)
        self.assertEqual(result.ops_log["target_pixels"], 180)
        self.assertEqual(result.ops_log["legal_domain_pixels"], int(np.count_nonzero(old_mask == 1)))
        self.assertTrue(np.all(old_mask[result.change_region] == 1))
        self.assertTrue(np.all(result.target_mask[result.change_region] == 4))
        self.assertNotIn("organic_projection_area_shortfall", result.warnings)

    def test_intratumoral_immune_policy_uses_existing_immune_neighborhood(self):
        old_mask = np.zeros((72, 72), dtype=np.int64)
        old_mask[8:64, 8:64] = 1
        old_mask[30:42, 12:20] = 4
        raw_candidate = old_mask == 1

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Immune infiltrate",
            primitive_config={
                "name": "intratumoral_immune_infiltration",
                "required_tissue_labels": ["Tumor", "Immune infiltrate"],
                "spatial_pattern": {
                    "region": "inside_tumor",
                    "spot_policy": {
                        "max_total_area_fraction_of_tumor": 0.60,
                        "min_spot_area_px": 1,
                    },
                },
                "parameter_ranges": {
                    "target_changed_area_fraction": {"mild": [0.05, 0.10]},
                    "organic_score_weights": {
                        "template": 0.0,
                        "spatial": 1.0,
                        "noise": 0.0,
                    },
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 128,
                    "organic_template_spillover_fraction": 0.0,
                    "tumor_boundary_margin_radius_px": 200,
                    "immune_neighbor_radius_px": 10,
                    "max_changed_area_fraction": 0.60,
                },
            },
            seed=1,
            target_pixels=120,
        )

        params = result.ops_log["component_policy"]["params"]
        self.assertTrue(params["used_existing_immune_neighborhood"])
        self.assertEqual(result.selected_pixels, 120)
        self.assertTrue(np.all(old_mask[result.change_region] == 1))
        selected_cols = np.argwhere(result.change_region)[:, 1]
        self.assertLess(float(np.mean(selected_cols)), 30.0)

    def test_intratumoral_immune_cap_ignores_existing_stromal_immune(self):
        old_mask = np.zeros((80, 80), dtype=np.int64)
        old_mask[4:76, 4:76] = 2
        old_mask[12:68, 12:68] = 1
        old_mask[4:12, 4:76] = 4
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[24:56, 24:56] = True

        primitive_config = {
            "name": "intratumoral_immune_infiltration",
            "required_tissue_labels": ["Tumor", "Immune infiltrate"],
            "spatial_pattern": {
                "region": "inside_tumor",
                "spot_policy": {
                    "max_total_area_fraction_of_tumor": 0.30,
                    "min_spot_area_px": 12,
                    "max_spot_area_px": 256,
                    "max_spots_per_patch": 32,
                },
            },
            "parameter_ranges": {
                "target_changed_area_fraction": {"mild": [0.05, 0.10]},
                "max_changed_area_fraction": 0.30,
                "organic_min_component_fraction": 0.0,
                "organic_fill_holes_max_area_px": 0,
                "organic_template_neighborhood_radius_px": 24,
                "organic_template_spillover_fraction": 0.15,
                "organic_min_template_legal_overlap_fraction": 0.0,
                "organic_min_selected_template_iou": 0.0,
            },
            "validation_rules": ["new_immune_must_be_inside_original_tumor"],
        }
        tumor_pixels = int(np.count_nonzero(old_mask == 1))
        stromal_immune_pixels = int(np.count_nonzero(old_mask == 4))
        target_pixels = int(round(tumor_pixels * 0.20))

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Immune infiltrate",
            primitive_config=primitive_config,
            seed=13,
            target_pixels=target_pixels,
        )

        params = result.ops_log["component_policy"]["params"]
        self.assertEqual(params["existing_immune_pixels_total"], stromal_immune_pixels)
        self.assertEqual(params["existing_intratumoral_immune_pixels"], 0)
        self.assertEqual(
            params["intratumoral_immune_cap_policy"],
            "per_edit_new_pixels_only",
        )
        self.assertTrue(params["existing_immune_pixels_not_subtracted_from_cap"])
        self.assertEqual(
            params["remaining_allowed_intratumoral_immune_pixels"],
            int(round(tumor_pixels * 0.30)),
        )
        self.assertEqual(result.selected_pixels, target_pixels)
        self.assertTrue(np.all(old_mask[result.change_region] == 1))

    def test_organic_projection_generic_policy_is_label_safe_and_logged(self):
        old_mask = np.zeros((24, 24), dtype=np.int64)
        old_mask[2:22, 2:12] = 2
        old_mask[2:22, 12:22] = 1
        raw_candidate = np.ones_like(old_mask, dtype=bool)

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config={"name": "unknown_future_primitive"},
            seed=1,
            target_pixels=30,
        )

        self.assertEqual(
            result.ops_log["component_policy"]["policy_name"],
            "generic_label_safe",
        )
        self.assertIn("organic_projection_generic_policy_used", result.warnings)
        self.assertTrue(np.all(old_mask[result.change_region] == 2))

    def test_stromal_desmoplasia_policy_contract(self):
        old_mask = np.zeros((72, 72), dtype=np.int64)
        old_mask[6:66, 6:66] = 7
        old_mask[24:48, 24:48] = 1
        ring = np.zeros_like(old_mask, dtype=bool)
        ring[18:54, 18:54] = True
        old_mask[ring & (old_mask != 1)] = 2
        old_mask[14:20, 40:48] = 4
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[10:60, 10:60] = True

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Other tissue", "Normal epithelium", "Immune infiltrate"),
            target_label="Stroma",
            primitive_config={
                "name": "stromal_desmoplasia",
                "mask_operation": {
                    "primary_sources": ["Other tissue", "Normal epithelium"],
                    "secondary_sources": ["Immune infiltrate"],
                    "target": "Stroma",
                    "forbid_sources": ["Tumor"],
                },
                "spatial_pattern": {
                    "immune_to_stroma_constraints": {
                        "max_fraction_of_total_desmoplasia_delta": 0.30,
                        "require_direct_stroma_adjacency": True,
                    },
                },
                "parameter_ranges": {
                    "stroma_area_delta_fraction": {"mild": [0.08, 0.14]},
                    "max_distance_from_tumor_px": 20,
                    "organic_min_template_legal_overlap_fraction": 0.0,
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 64,
                    "organic_template_spillover_fraction": 0.0,
                },
            },
            seed=5,
            target_pixels=120,
        )

        self.assertEqual(
            result.ops_log["component_policy"]["policy_name"],
            "stromal_desmoplasia_peritumoral_stroma_expansion",
        )
        self.assertEqual(result.selected_pixels, 120)
        self.assertTrue(np.all(old_mask[result.change_region] != 1))
        self.assertTrue(np.all(result.target_mask[result.change_region] == 2))
        self.assertTrue(np.all(np.isin(old_mask[result.change_region], (4, 7))))
        dist_to_tumor = ndimage.distance_transform_edt(old_mask != 1)
        self.assertLessEqual(float(dist_to_tumor[result.change_region].max()), 20.0)

    def test_stromal_desmoplasia_target_pixels_are_stroma_relative(self):
        old_mask = np.zeros((80, 80), dtype=np.int64)
        old_mask[4:76, 4:76] = 7
        old_mask[28:52, 28:52] = 1
        old_mask[20:60, 20:60][old_mask[20:60, 20:60] != 1] = 2
        raw_candidate = np.ones_like(old_mask, dtype=bool)
        stroma_pixels = int(np.count_nonzero(old_mask == 2))

        result = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Other tissue",),
            target_label="Stroma",
            primitive_config={
                "name": "stromal_desmoplasia",
                "parameter_ranges": {
                    "stroma_area_delta_fraction": {"mild": [0.10, 0.20]},
                    "max_distance_from_tumor_px": 40,
                    "organic_min_template_legal_overlap_fraction": 0.0,
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 128,
                    "organic_template_spillover_fraction": 0.0,
                },
                "spatial_pattern": {
                    "immune_to_stroma_constraints": {
                        "max_fraction_of_total_desmoplasia_delta": 0.30,
                        "require_direct_stroma_adjacency": True,
                    },
                },
            },
            seed=1,
            target_pixels=None,
        )

        self.assertEqual(result.ops_log["target_pixels"], int(np.ceil(stroma_pixels * 0.15)))
        self.assertEqual(result.selected_pixels, result.ops_log["target_pixels"])

    def test_organic_projection_same_seed_is_bit_identical(self):
        old_mask = np.zeros((64, 64), dtype=np.int64)
        old_mask[6:58, 6:58] = 2
        old_mask[20:44, 20:44] = 1
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[8:56, 8:56] = True
        primitive_config = {
            "name": "stromal_immune_infiltration",
            "parameter_ranges": {
                "organic_min_component_fraction": 0.0,
                "organic_noise_sigma_px": 8,
                "organic_noise_amplitude": 0.5,
                "organic_score_weights": {
                    "template": 0.30,
                    "spatial": 0.30,
                    "noise": 0.40,
                },
            },
        }

        first = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config=primitive_config,
            seed=123,
            target_pixels=300,
        )
        second = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config=primitive_config,
            seed=123,
            target_pixels=300,
        )
        different_seed = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Stroma",),
            target_label="Immune infiltrate",
            primitive_config=primitive_config,
            seed=124,
            target_pixels=300,
        )

        np.testing.assert_array_equal(first.change_region, second.change_region)
        np.testing.assert_array_equal(first.target_mask, second.target_mask)
        self.assertEqual(first.selected_pixels, 300)
        self.assertEqual(different_seed.selected_pixels, 300)
        self.assertTrue(np.all(old_mask[different_seed.change_region] == 2))

    def test_repair_feedback_distinguishes_organic_projection_failure_reasons(self):
        legal_small = self._synthetic_edit_result(
            selected_pixels=16,
            warnings=("organic_projection_area_shortfall",),
            ops_log={
                "legal_domain_pixels": 16,
                "target_pixels": 40,
                "area_shortfall": 24,
                "template_overlap_with_legal_domain": 0.50,
                "projection_mode": PROJECTION_MODE_ORGANIC_V2,
                "projection_backend": ORGANIC_PROJECTION_BACKEND,
            },
        )
        cleanup_shortfall = self._synthetic_edit_result(
            selected_pixels=35,
            warnings=("organic_projection_area_shortfall",),
            ops_log={
                "legal_domain_pixels": 120,
                "target_pixels": 40,
                "area_shortfall": 5,
                "template_overlap_with_legal_domain": 0.50,
                "projection_mode": PROJECTION_MODE_ORGANIC_V2,
                "projection_backend": ORGANIC_PROJECTION_BACKEND,
            },
        )
        low_overlap = self._synthetic_edit_result(
            selected_pixels=40,
            warnings=(),
            ops_log={
                "legal_domain_pixels": 120,
                "target_pixels": 40,
                "area_shortfall": 0,
                "template_overlap_with_legal_domain": 0.02,
                "projection_mode": PROJECTION_MODE_ORGANIC_V2,
                "projection_backend": ORGANIC_PROJECTION_BACKEND,
            },
        )

        legal_feedback = build_repair_feedback(
            status="validation_failed",
            attempt_index=1,
            edit_result=legal_small,
        )
        cleanup_feedback = build_repair_feedback(
            status="validation_failed",
            attempt_index=1,
            edit_result=cleanup_shortfall,
        )
        overlap_feedback = build_repair_feedback(
            status="validation_failed",
            attempt_index=1,
            edit_result=low_overlap,
        )

        self.assertEqual(
            legal_feedback["projection"]["top_failed_reason"],
            "legal_domain_too_small",
        )
        self.assertEqual(
            cleanup_feedback["projection"]["top_failed_reason"],
            "projector_area_shortfall_after_cleanup",
        )
        self.assertEqual(
            overlap_feedback["projection"]["top_failed_reason"],
            "template_overlap_with_legal_domain_too_low",
        )
        self.assertNotIn("repair_instruction", cleanup_feedback)
        self.assertNotIn("vertex", str(cleanup_feedback).lower())
        self.assertNotIn("pixel-perfect", str(cleanup_feedback).lower())

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

    def _synthetic_edit_result(
        self,
        *,
        selected_pixels: int,
        warnings: tuple[str, ...],
        ops_log: dict,
    ) -> PrimitiveEditResult:
        target_mask = np.zeros((4, 4), dtype=np.int64)
        change_region = np.zeros_like(target_mask, dtype=bool)
        if selected_pixels:
            change_region[0, 0] = True
        return PrimitiveEditResult(
            target_mask=target_mask,
            change_region=change_region,
            changed_area_fraction=selected_pixels / max(int(target_mask.size), 1),
            selected_pixels=selected_pixels,
            warnings=warnings,
            ops_log=ops_log,
        )


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
        rgb = np.zeros((192, 192, 3), dtype=np.uint8)

        overlay = add_coordinate_grid_overlay(rgb, grid_spacing_px=64)

        self.assertEqual(overlay.shape, rgb.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertGreater(int(np.count_nonzero(overlay)), 0)
        self.assertTrue(np.any(overlay[:, 64] != rgb[:, 64]))
        self.assertTrue(np.any(overlay[64, :] != rgb[64, :]))

    def test_preview_tool_auto_loads_rgb_mask_without_turning_unknown_ids_white(self):
        mask = _load_mask_auto(
            Path("phase3_mask_edit/previews/tumor_burden/BCSS_mild_real0_tb15_increase_src.png")
        )

        rgb = id_mask_to_llm_preview_rgb(mask)
        unique_colors = np.unique(rgb.reshape(-1, 3), axis=0)

        self.assertGreater(len(unique_colors), 1)
        self.assertFalse(np.all(rgb == 255))



if __name__ == "__main__":
    unittest.main()
