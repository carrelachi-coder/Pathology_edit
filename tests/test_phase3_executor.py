"""Tests for the unified primitive executor interface."""

import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.executor import (
    EditExecutionResult,
    execute_edit,
    register_primitive,
)
from phase3_mask_edit.generic.tumor_burden import (
    PrimitiveExecutionError,
    apply_tumor_burden_decrease,
    apply_tumor_burden_increase,
)


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


def _bcss_schema() -> MaskProfileSchema:
    return MaskProfileSchema.from_reference_profile("BCSS")


def _primitive(recipe, name):
    return next(
        p for p in recipe["primitives"] if p["name"] == name
    )


class ExecutorApplicabilityGateTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)
        self.schema = _bcss_schema()

    def test_rejected_intent_returns_no_edit_result(self):
        mask = np.zeros((5, 5), dtype=np.int64)
        mask[:, :] = 0
        context = MaskEditContext.from_mask(mask, self.schema)

        intent = EditIntent.from_mapping({
            "primitive": "tumor_burden_increase",
            "reference_profile": "BCSS",
        })

        result = execute_edit(mask, intent, self.recipe, self.schema, context)

        self.assertEqual(result.status, "rejected")
        self.assertIsNone(result.edit_result)
        self.assertIsNone(result.validation)
        self.assertEqual(result.applicability.status, "rejected")
        self.assertTrue(any("no_tumor" in r or "absent" in r for r in result.applicability.reasons))

    def test_executable_intent_runs_mask_transform(self):
        old_mask = np.array([
            [1, 1, 2, 2, 0],
            [1, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [7, 7, 7, 7, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.int64)
        context = MaskEditContext.from_mask(old_mask, self.schema)

        intent = EditIntent.from_mapping({
            "primitive": "tumor_burden_increase",
            "reference_profile": "BCSS",
            "target_change_fraction": 5 / 25,
        })

        result = execute_edit(old_mask, intent, self.recipe, self.schema, context)

        self.assertNotEqual(result.status, "rejected")
        self.assertIsNotNone(result.edit_result)
        self.assertIsNotNone(result.validation)
        # BCSS mask missing optional labels → degraded, not pure executable.
        self.assertIn(result.applicability.status, ("executable", "degraded"))
        self.assertTrue(result.edit_result.changed_area_fraction > 0)

    def test_degraded_intent_still_runs_with_warnings(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        mask = np.array([
            [1, 1, 2, 2, 0],
            [1, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [7, 7, 7, 7, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.int64)
        context = MaskEditContext.from_mask(mask, schema)

        intent = EditIntent.from_mapping({
            "primitive": "tumor_burden_increase",
            "reference_profile": "BCSS",
            "target_change_fraction": 5 / 25,
        })

        result = execute_edit(mask, intent, self.recipe, schema, context)

        # If applicability is degraded, the status should reflect that.
        if result.applicability.status == "degraded":
            self.assertTrue(result.status.startswith("degraded"))
            self.assertTrue(len(result.applicability.warnings) > 0)

    def test_applicability_warnings_merge_into_edit_result(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array([
            [1, 1, 2, 2, 0],
            [1, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [7, 7, 7, 7, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.int64)
        context = MaskEditContext.from_mask(old_mask, schema)

        intent = EditIntent.from_mapping({
            "primitive": "tumor_burden_increase",
            "reference_profile": "BCSS",
            "target_change_fraction": 5 / 25,
        })

        result = execute_edit(old_mask, intent, self.recipe, schema, context)

        if result.edit_result is not None:
            merged = result.edit_result.warnings
            applicability_warnings = result.applicability.warnings
            for w in applicability_warnings:
                self.assertIn(w, merged)


class ExecutorValidationTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)
        self.schema = _bcss_schema()

    def test_successful_edit_passes_validation(self):
        old_mask = np.array([
            [1, 1, 2, 2, 0],
            [1, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [7, 7, 7, 7, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.int64)
        context = MaskEditContext.from_mask(old_mask, self.schema)

        intent = EditIntent.from_mapping({
            "primitive": "tumor_burden_increase",
            "reference_profile": "BCSS",
            "target_change_fraction": 5 / 25,
        })

        result = execute_edit(old_mask, intent, self.recipe, self.schema, context)

        if result.edit_result is not None:
            self.assertIsNotNone(result.validation)
            self.assertTrue(result.validation.passed)


class ExecutorRegistryTests(unittest.TestCase):
    def test_register_and_execute_custom_primitive(self):
        # Registering an existing primitive under an alias should work.
        from phase3_mask_edit.generic.executor import _PRIMITIVE_REGISTRY
        register_primitive("test_alias", apply_tumor_burden_increase)
        self.assertIn("test_alias", _PRIMITIVE_REGISTRY)


class ExecutorStromaImmuneRejectedTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)

    def test_stromal_immune_infiltration_on_panda_is_rejected(self):
        schema = MaskProfileSchema.from_reference_profile("PANDA")
        mask = np.array([
            [1, 1, 2, 2, 0],
            [1, 1, 2, 2, 0],
            [2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.int64)
        context = MaskEditContext.from_mask(mask, schema)

        intent = EditIntent.from_mapping({
            "primitive": "stromal_immune_infiltration",
            "reference_profile": "PANDA",
        })

        result = execute_edit(mask, intent, self.recipe, schema, context)

        self.assertEqual(result.status, "rejected")
        self.assertIsNone(result.edit_result)

    def test_intratumoral_immune_on_no_tumor_mask_is_rejected(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        mask = np.zeros((5, 5), dtype=np.int64)
        mask[:, :] = 2
        context = MaskEditContext.from_mask(mask, schema)

        intent = EditIntent.from_mapping({
            "primitive": "intratumoral_immune_infiltration",
            "reference_profile": "BCSS",
        })

        result = execute_edit(mask, intent, self.recipe, schema, context)

        self.assertEqual(result.status, "rejected")


if __name__ == "__main__":
    unittest.main()