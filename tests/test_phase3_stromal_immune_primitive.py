"""Tests for phase3_mask_edit/generic/immune.py."""

import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.executor import execute_edit
from phase3_mask_edit.generic.immune import apply_stromal_immune_infiltration
from phase3_mask_edit.generic.tumor_burden import PrimitiveExecutionError


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


def _primitive(recipe, name):
    return next(p for p in recipe["primitives"] if p["name"] == name)


def _stromal_mask(size=256, *, tumor_radius=42, existing_immune=False):
    mask = np.zeros((size, size), dtype=np.int64)
    mask[24:-24, 24:-24] = 2
    yy, xx = np.mgrid[:size, :size]
    tumor = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= tumor_radius**2
    mask[tumor] = 1
    if existing_immune:
        mask[56:72, 56:72] = 4
    return mask


class Phase3StromalImmunePrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)
        self.primitive = _primitive(self.recipe, "stromal_immune_infiltration")

    def test_stromal_immune_replaces_only_original_stroma(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _stromal_mask()
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.08,
                "parameters": {
                    "min_stromal_immune_component_area_px": 4,
                    "max_stromal_immune_components": 8,
                },
                "seed": 7,
            }
        )

        result = apply_stromal_immune_infiltration(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertGreater(result.selected_pixels, 0)
        self.assertTrue(np.all(result.change_region <= (old_mask == 2)))
        self.assertTrue(np.all(result.target_mask[result.change_region] == 4))
        self.assertTrue(np.all(result.target_mask[old_mask == 1] == 1))
        self.assertTrue(np.all(result.target_mask[old_mask == 0] == 0))
        self.assertEqual(
            result.ops_log["spatial"]["target_area_reference"],
            "stroma_plus_immune",
        )

    def test_stromal_immune_uses_soft_peritumoral_priority_without_hard_limit(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        size = 256
        old_mask = np.zeros((size, size), dtype=np.int64)
        old_mask[16:240, 16:240] = 2
        old_mask[92:164, 92:164] = 1
        tumor = old_mask == 1
        dist_to_tumor = ndimage.distance_transform_edt(~tumor)
        near_stroma = (old_mask == 2) & (dist_to_tumor <= 64)
        # Make near-tumor stroma insufficient so far stroma must remain eligible.
        old_mask[near_stroma] = 0
        old_mask[84:172, 84:172] = 2
        old_mask[92:164, 92:164] = 1
        dist_to_tumor = ndimage.distance_transform_edt(~(old_mask == 1))

        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.18,
                "parameters": {
                    "max_stromal_immune_components": 20,
                    "min_stromal_immune_component_area_px": 1,
                    "peritumoral_falloff_radius_px": 96,
                },
                "seed": 3,
            }
        )

        result = apply_stromal_immune_infiltration(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertIsNone(result.ops_log["spatial"]["hard_distance_limit_px"])
        self.assertTrue(np.any(dist_to_tumor[result.change_region] > 64))
        self.assertTrue(result.ops_log["spatial"]["used_soft_peritumoral_priority"])
        self.assertEqual(result.ops_log["spatial"]["tumor_mode"], "small")

    def test_stromal_immune_does_not_enter_tumor(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _stromal_mask(tumor_radius=64)
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.10,
                "seed": 11,
            }
        )

        result = apply_stromal_immune_infiltration(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertFalse(np.any(result.change_region & (old_mask == 1)))

    def test_stromal_immune_rejects_no_stroma(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.zeros((64, 64), dtype=np.int64)
        old_mask[16:48, 16:48] = 1
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.10,
            }
        )

        with self.assertRaisesRegex(PrimitiveExecutionError, "no_stroma"):
            apply_stromal_immune_infiltration(
                old_mask,
                schema,
                context,
                self.primitive,
                intent,
            )

    def test_stromal_immune_rejects_no_immune_label(self):
        schema = MaskProfileSchema.from_reference_profile("PANDA")
        old_mask = np.full((64, 64), 2, dtype=np.int64)
        old_mask[16:48, 16:48] = 8
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "PANDA",
                "target_change_fraction": 0.10,
            }
        )

        with self.assertRaisesRegex(PrimitiveExecutionError, "no_immune_label"):
            apply_stromal_immune_infiltration(
                old_mask,
                schema,
                context,
                self.primitive,
                intent,
            )

    def test_executor_runs_stromal_immune_infiltration(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _stromal_mask()
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.12,
                "seed": 5,
            }
        )

        result = execute_edit(old_mask, intent, self.recipe, schema, context)

        self.assertIn(result.status, ("executed_validated", "degraded_executed"))
        self.assertIsNotNone(result.edit_result)
        self.assertIsNotNone(result.validation)
        self.assertTrue(result.validation.passed)
        self.assertGreater(result.edit_result.selected_pixels, 0)

    def test_stromal_immune_baseline_zero_immune_validates(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _stromal_mask(existing_immune=False)
        self.assertFalse(np.any(old_mask == 4))
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.10,
                "seed": 13,
            }
        )

        result = execute_edit(old_mask, intent, self.recipe, schema, context)

        self.assertIsNotNone(result.validation)
        self.assertTrue(result.validation.passed)
        immune_check = next(
            c for c in result.validation.checks if c.name == "immune_area_must_increase"
        )
        self.assertTrue(immune_check.passed)
        self.assertIn("0 ->", immune_check.detail)

    def test_stromal_immune_no_tumor_random_patchy(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.zeros((256, 256), dtype=np.int64)
        old_mask[24:-24, 24:-24] = 2
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.10,
                "parameters": {
                    "max_stromal_immune_components": 12,
                    "min_stromal_immune_component_area_px": 4,
                },
                "seed": 17,
            }
        )

        result = apply_stromal_immune_infiltration(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertGreater(result.selected_pixels, 0)
        self.assertTrue(np.all(result.change_region <= (old_mask == 2)))
        self.assertEqual(result.ops_log["spatial"]["tumor_mode"], "none")
        self.assertFalse(
            result.ops_log["spatial"]["used_soft_peritumoral_priority"]
        )
        self.assertLess(
            result.ops_log["spatial"]["selected_components"],
            result.selected_pixels,
        )

    def test_stromal_immune_small_tumor_blends_priority_and_noise(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.zeros((256, 256), dtype=np.int64)
        old_mask[16:240, 16:240] = 2
        old_mask[28:44, 28:44] = 1
        tumor = old_mask == 1
        dist_to_tumor = ndimage.distance_transform_edt(~tumor)
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.16,
                "parameters": {
                    "max_stromal_immune_components": 20,
                    "min_stromal_immune_component_area_px": 1,
                },
                "seed": 19,
            }
        )

        result = apply_stromal_immune_infiltration(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertEqual(result.ops_log["spatial"]["tumor_mode"], "small")
        self.assertTrue(np.any(dist_to_tumor[result.change_region] <= 64))
        self.assertTrue(np.any(dist_to_tumor[result.change_region] > 96))
        self.assertIn("smooth_noise", result.ops_log["spatial"]["active_weights"])
        self.assertGreater(
            result.ops_log["spatial"]["active_weights"]["smooth_noise"],
            result.ops_log["spatial"]["active_weights"]["peritumoral_proximity"],
        )


if __name__ == "__main__":
    unittest.main()
