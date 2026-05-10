import unittest
from pathlib import Path

import numpy as np

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
from phase3_mask_edit.backends.llm_preview import (
    add_coordinate_grid_overlay,
    id_mask_to_llm_preview_rgb,
    llm_palette_legend,
)
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.validation import validate_edit_result


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
