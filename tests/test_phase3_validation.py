"""Tests for phase3_mask_edit/core/validation.py."""

import unittest

import numpy as np

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.validation import (
    ValidationCheck,
    ValidationResult,
    validate_edit_result,
)


def _bcss_schema() -> MaskProfileSchema:
    return MaskProfileSchema.from_reference_profile("BCSS")


def _tumor_burden_increase_config() -> dict:
    return {
        "name": "tumor_burden_increase",
        "required_tissue_labels": ["Tumor"],
        "parameter_ranges": {
            "target_area_delta_fraction": {
                "mild": [0.08, 0.14],
                "moderate": [0.14, 0.24],
                "significant": [0.24, 0.40],
                "xlarge_deid": [0.40, 0.60],
            },
        },
        "validation_rules": [
            "tumor_area_must_increase",
            "new_tumor_must_touch_or_neighbor_original_tumor",
            "changed_area_within_parameter_range",
        ],
    }


def _tumor_burden_decrease_config() -> dict:
    return {
        "name": "tumor_burden_decrease",
        "required_tissue_labels": ["Tumor"],
        "parameter_ranges": {
            "target_area_decrease_fraction": {
                "mild": [0.08, 0.14],
                "moderate": [0.14, 0.24],
            },
        },
        "validation_rules": [
            "tumor_area_must_decrease",
            "released_region_must_not_be_background",
            "changed_area_within_parameter_range",
        ],
    }


def _boundary_pushing_config() -> dict:
    return {
        "name": "boundary_pushing_remodel",
        "required_tissue_labels": ["Tumor"],
        "parameter_ranges": {
            "target_changed_area_fraction": {
                "mild": [0.08, 0.14],
                "moderate": [0.14, 0.20],
            },
            "max_changed_area_fraction": 0.30,
            "max_abs_tumor_area_delta_fraction": 0.02,
        },
        "validation_rules": [
            "tumor_area_change_must_remain_small",
            "tumor_must_not_fragment_or_disappear",
        ],
    }


class GlobalValidationTests(unittest.TestCase):
    def test_change_area_nonempty_passes(self):
        schema = _bcss_schema()
        src = np.ones((5, 5), dtype=np.int64)
        tgt = np.ones((5, 5), dtype=np.int64)
        change = np.zeros((5, 5), dtype=bool)
        change[2, 2] = True

        result = validate_edit_result(
            src, tgt, change, schema, {"name": "test"}, 0.04,
        )
        check = next(c for c in result.checks if c.name == "change_area_nonempty")
        self.assertTrue(check.passed)

    def test_change_area_nonempty_fails_when_empty(self):
        schema = _bcss_schema()
        src = np.ones((5, 5), dtype=np.int64)
        tgt = np.ones((5, 5), dtype=np.int64)
        change = np.zeros((5, 5), dtype=bool)

        result = validate_edit_result(
            src, tgt, change, schema, {"name": "test"}, 0.0,
        )
        check = next(c for c in result.checks if c.name == "change_area_nonempty")
        self.assertFalse(check.passed)

    def test_change_area_within_range_passes(self):
        schema = _bcss_schema()
        src = np.ones((5, 5), dtype=np.int64)
        tgt = np.ones((5, 5), dtype=np.int64)
        change = np.zeros((5, 5), dtype=bool)
        change[:2, :] = True
        config = {"name": "test", "parameter_ranges": {}}

        result = validate_edit_result(
            src, tgt, change, schema, config, 0.10,
        )
        check = next(c for c in result.checks if c.name == "change_area_within_range")
        self.assertTrue(check.passed)

    def test_change_area_out_of_range_fails(self):
        schema = _bcss_schema()
        src = np.ones((5, 5), dtype=np.int64)
        tgt = np.ones((5, 5), dtype=np.int64)
        change = np.ones((5, 5), dtype=bool)
        config = {
            "name": "test",
            "parameter_ranges": {"max_changed_area_fraction": 0.30},
        }

        result = validate_edit_result(
            src, tgt, change, schema, config, 1.0,
        )
        check = next(c for c in result.checks if c.name == "change_area_within_range")
        self.assertFalse(check.passed)

    def test_label_legality_passes_for_known_ids(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = np.array([[1, 2, 0]], dtype=np.int64)
        change = np.array([[False, True, False]], dtype=bool)

        result = validate_edit_result(
            src, tgt, change, schema, {"name": "test"}, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "label_legality")
        self.assertTrue(check.passed)

    def test_label_legality_fails_for_unknown_id(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = np.array([[1, 99, 0]], dtype=np.int64)
        change = np.array([[False, True, False]], dtype=bool)

        result = validate_edit_result(
            src, tgt, change, schema, {"name": "test"}, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "label_legality")
        self.assertFalse(check.passed)

    def test_no_background_leakage_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = np.array([[1, 2, 0]], dtype=np.int64)
        change = np.array([[False, True, False]], dtype=bool)

        result = validate_edit_result(
            src, tgt, change, schema, {"name": "test"}, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "no_background_leakage")
        self.assertTrue(check.passed)

    def test_no_background_leakage_fails_when_change_creates_bg(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 7]], dtype=np.int64)
        tgt = np.array([[1, 0, 7]], dtype=np.int64)
        change = np.array([[False, True, False]], dtype=bool)

        result = validate_edit_result(
            src, tgt, change, schema, {"name": "test"}, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "no_background_leakage")
        self.assertFalse(check.passed)

    def test_required_labels_present_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = src.copy()
        change = np.array([[False, True, False]], dtype=bool)
        config = {"name": "test", "required_tissue_labels": ["Tumor", "Stroma"]}

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "required_labels_present")
        self.assertTrue(check.passed)

    def test_required_labels_present_fails_when_missing(self):
        schema = _bcss_schema()
        src = np.array([[2, 2, 0]], dtype=np.int64)
        tgt = src.copy()
        change = np.zeros((1, 3), dtype=bool)
        config = {"name": "test", "required_tissue_labels": ["Tumor"]}

        result = validate_edit_result(
            src, tgt, change, schema, config, 0.0,
        )
        check = next(c for c in result.checks if c.name == "required_labels_present")
        self.assertFalse(check.passed)


class TumorBurdenIncreaseGuardTests(unittest.TestCase):
    def test_tumor_area_must_increase_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0], [1, 2, 0]], dtype=np.int64)
        tgt = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.int64)
        change = np.array([[False, True, False], [False, False, False]], dtype=bool)
        config = _tumor_burden_increase_config()

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 6,
        )
        check = next(c for c in result.checks if c.name == "tumor_area_must_increase")
        self.assertTrue(check.passed)

    def test_tumor_area_must_increase_fails_when_no_growth(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0], [1, 2, 0]], dtype=np.int64)
        tgt = src.copy()
        change = np.zeros((2, 3), dtype=bool)
        config = _tumor_burden_increase_config()

        result = validate_edit_result(
            src, tgt, change, schema, config, 0.0,
        )
        check = next(c for c in result.checks if c.name == "tumor_area_must_increase")
        self.assertFalse(check.passed)

    def test_new_tumor_touches_original_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 2], [1, 2, 2]], dtype=np.int64)
        tgt = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.int64)
        change = np.array([[False, True, False], [False, False, False]], dtype=bool)
        config = _tumor_burden_increase_config()

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 6,
        )
        check = next(c for c in result.checks if c.name == "new_tumor_must_touch_or_neighbor_original_tumor")
        self.assertTrue(check.passed)


class TumorBurdenDecreaseGuardTests(unittest.TestCase):
    def test_tumor_area_must_decrease_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0], [1, 2, 0]], dtype=np.int64)
        tgt = np.array([[2, 2, 0], [1, 2, 0]], dtype=np.int64)
        change = np.array([[True, False, False], [False, False, False]], dtype=bool)
        config = _tumor_burden_decrease_config()

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 6,
        )
        check = next(c for c in result.checks if c.name == "tumor_area_must_decrease")
        self.assertTrue(check.passed)

    def test_released_region_not_background_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0], [1, 2, 0]], dtype=np.int64)
        tgt = np.array([[2, 2, 0], [1, 2, 0]], dtype=np.int64)
        change = np.array([[True, False, False], [False, False, False]], dtype=bool)
        config = _tumor_burden_decrease_config()

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 6,
        )
        check = next(c for c in result.checks if c.name == "released_region_must_not_be_background")
        self.assertTrue(check.passed)

    def test_released_region_not_background_fails(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0], [1, 2, 0]], dtype=np.int64)
        tgt = np.array([[0, 2, 0], [1, 2, 0]], dtype=np.int64)
        change = np.array([[True, False, False], [False, False, False]], dtype=bool)
        config = _tumor_burden_decrease_config()

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 6,
        )
        check = next(c for c in result.checks if c.name == "released_region_must_not_be_background")
        self.assertFalse(check.passed)


class BoundaryPushingRemodelGuardTests(unittest.TestCase):
    def test_tumor_area_change_small_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 1, 2, 2, 0]], dtype=np.int64)
        tgt = np.array([[1, 2, 1, 2, 0]], dtype=np.int64)
        change = np.array([[False, True, True, False, False]], dtype=bool)
        config = _boundary_pushing_config()

        result = validate_edit_result(
            src, tgt, change, schema, config, 2 / 5,
        )
        check = next(c for c in result.checks if c.name == "tumor_area_change_must_remain_small")
        self.assertTrue(check.passed)

    def test_tumor_must_not_fragment_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 1, 2, 0, 0]], dtype=np.int64)
        tgt = np.array([[1, 2, 2, 0, 0]], dtype=np.int64)
        change = np.array([[False, True, False, False, False]], dtype=bool)
        config = _boundary_pushing_config()

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 5,
        )
        check = next(c for c in result.checks if c.name == "tumor_must_not_fragment_or_disappear")
        self.assertTrue(check.passed)

    def test_tumor_disappears_fails(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = np.array([[2, 2, 0]], dtype=np.int64)
        change = np.array([[True, False, False]], dtype=bool)
        config = _boundary_pushing_config()

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "tumor_must_not_fragment_or_disappear")
        self.assertFalse(check.passed)


class NecrosisGuardTests(unittest.TestCase):
    def test_necrosis_area_must_increase_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 1, 1, 0]], dtype=np.int64)
        tgt = np.array([[1, 3, 1, 0]], dtype=np.int64)
        change = np.array([[False, True, False, False]], dtype=bool)
        config = {
            "name": "necrosis_appearance",
            "required_tissue_labels": ["Tumor"],
            "parameter_ranges": {},
            "validation_rules": ["necrosis_area_must_increase", "new_necrosis_must_be_inside_original_tumor"],
        }

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 4,
        )
        nec_check = next(c for c in result.checks if c.name == "necrosis_area_must_increase")
        self.assertTrue(nec_check.passed)
        inside_check = next(c for c in result.checks if c.name == "new_necrosis_must_be_inside_original_tumor")
        self.assertTrue(inside_check.passed)

    def test_necrosis_outside_tumor_fails(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = np.array([[1, 3, 0]], dtype=np.int64)
        change = np.array([[False, True, False]], dtype=bool)
        config = {
            "name": "necrosis_appearance",
            "required_tissue_labels": ["Tumor"],
            "parameter_ranges": {},
            "validation_rules": ["new_necrosis_must_be_inside_original_tumor"],
        }

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "new_necrosis_must_be_inside_original_tumor")
        self.assertFalse(check.passed)


class ImmuneGuardTests(unittest.TestCase):
    def test_immune_area_must_increase_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = np.array([[1, 4, 0]], dtype=np.int64)
        change = np.array([[False, True, False]], dtype=bool)
        config = {
            "name": "stromal_immune_infiltration",
            "required_tissue_labels": ["Tumor", "Stroma", "Immune infiltrate"],
            "parameter_ranges": {},
            "validation_rules": ["immune_area_must_increase", "new_immune_must_be_mainly_outside_tumor"],
        }

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 3,
        )
        inc_check = next(c for c in result.checks if c.name == "immune_area_must_increase")
        self.assertTrue(inc_check.passed)
        outside_check = next(c for c in result.checks if c.name == "new_immune_must_be_mainly_outside_tumor")
        self.assertTrue(outside_check.passed)

    def test_immune_area_must_decrease_passes(self):
        schema = _bcss_schema()
        src = np.array([[4, 2, 0]], dtype=np.int64)
        tgt = np.array([[2, 2, 0]], dtype=np.int64)
        change = np.array([[True, False, False]], dtype=bool)
        config = {
            "name": "immune_infiltration_decrease",
            "required_tissue_labels": ["Immune infiltrate"],
            "parameter_ranges": {},
            "validation_rules": ["immune_area_must_decrease", "no_background_holes"],
        }

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "immune_area_must_decrease")
        self.assertTrue(check.passed)
        bg_check = next(c for c in result.checks if c.name == "no_background_holes")
        self.assertTrue(bg_check.passed)

    def test_intratumoral_immune_inside_tumor_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 1, 0]], dtype=np.int64)
        tgt = np.array([[4, 1, 0]], dtype=np.int64)
        change = np.array([[True, False, False]], dtype=bool)
        config = {
            "name": "intratumoral_immune_infiltration",
            "required_tissue_labels": ["Tumor", "Immune infiltrate"],
            "parameter_ranges": {},
            "validation_rules": ["new_immune_must_be_inside_original_tumor"],
        }

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "new_immune_must_be_inside_original_tumor")
        self.assertTrue(check.passed)

    def test_stromal_immune_outside_tumor_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = np.array([[1, 4, 0]], dtype=np.int64)
        change = np.array([[False, True, False]], dtype=bool)
        config = {
            "name": "stromal_immune_infiltration",
            "required_tissue_labels": ["Tumor", "Stroma", "Immune infiltrate"],
            "parameter_ranges": {},
            "validation_rules": ["new_immune_must_be_mainly_outside_tumor"],
        }

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "new_immune_must_be_mainly_outside_tumor")
        self.assertTrue(check.passed)


class StromaDesmoplasiaGuardTests(unittest.TestCase):
    def test_stroma_area_increase_passes(self):
        schema = _bcss_schema()
        src = np.array([[1, 7, 0]], dtype=np.int64)
        tgt = np.array([[1, 2, 0]], dtype=np.int64)
        change = np.array([[False, True, False]], dtype=bool)
        config = {
            "name": "stromal_desmoplasia",
            "required_tissue_labels": ["Tumor", "Stroma"],
            "parameter_ranges": {},
            "validation_rules": [
                "stroma_area_or_generation_region_must_increase",
                "tumor_area_must_remain_stable",
                "change_region_must_be_outside_tumor",
            ],
        }

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 3,
        )
        str_check = next(c for c in result.checks if c.name == "stroma_area_or_generation_region_must_increase")
        self.assertTrue(str_check.passed)
        tumor_check = next(c for c in result.checks if c.name == "tumor_area_must_remain_stable")
        self.assertTrue(tumor_check.passed)
        outside_check = next(c for c in result.checks if c.name == "change_region_must_be_outside_tumor")
        self.assertTrue(outside_check.passed)

    def test_change_region_inside_tumor_fails(self):
        schema = _bcss_schema()
        src = np.array([[1, 7, 0]], dtype=np.int64)
        tgt = np.array([[2, 7, 0]], dtype=np.int64)
        change = np.array([[True, False, False]], dtype=bool)
        config = {
            "name": "stromal_desmoplasia",
            "required_tissue_labels": ["Tumor", "Stroma"],
            "parameter_ranges": {},
            "validation_rules": ["change_region_must_be_outside_tumor"],
        }

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 3,
        )
        check = next(c for c in result.checks if c.name == "change_region_must_be_outside_tumor")
        self.assertFalse(check.passed)


class ValidationResultTests(unittest.TestCase):
    def test_failed_checks_property(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = np.array([[0, 2, 0]], dtype=np.int64)
        change = np.array([[True, False, False]], dtype=bool)

        result = validate_edit_result(
            src, tgt, change, schema, {"name": "test"}, 1 / 3,
        )
        self.assertFalse(result.passed)
        self.assertTrue(len(result.failed_checks) > 0)
        self.assertTrue(len(result.warnings) > 0)

    def test_all_checks_pass_yields_passed_result(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = np.array([[1, 1, 0]], dtype=np.int64)
        change = np.array([[False, True, False]], dtype=bool)
        config = _tumor_burden_increase_config()

        result = validate_edit_result(
            src, tgt, change, schema, config, 1 / 3,
        )
        self.assertTrue(result.passed)
        self.assertEqual(len(result.failed_checks), 0)


class UnknownGuardRuleTests(unittest.TestCase):
    def test_unknown_rule_is_skipped_as_passing(self):
        schema = _bcss_schema()
        src = np.array([[1, 2, 0]], dtype=np.int64)
        tgt = src.copy()
        change = np.zeros((1, 3), dtype=bool)
        config = {"name": "test", "parameter_ranges": {}, "validation_rules": ["future_unknown_rule"]}

        result = validate_edit_result(
            src, tgt, change, schema, config, 0.0,
        )
        check = next(c for c in result.checks if c.name == "future_unknown_rule")
        self.assertTrue(check.passed)
        self.assertIn("skipped", check.detail)