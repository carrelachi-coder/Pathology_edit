"""Tests for phase3_mask_edit/generic/desmoplasia.py."""

import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.desmoplasia import apply_stromal_desmoplasia
from phase3_mask_edit.generic.executor import execute_edit
from phase3_mask_edit.generic.tumor_burden import PrimitiveExecutionError


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


def _primitive(recipe, name):
    return next(p for p in recipe["primitives"] if p["name"] == name)


def _desmoplasia_mask(size=128, *, with_immune=False):
    mask = np.zeros((size, size), dtype=np.int64)
    mask[12:-12, 12:-12] = 7
    yy, xx = np.mgrid[:size, :size]
    tumor = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= 22**2
    stroma_ring = (
        (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= 34**2
    ) & ~tumor
    mask[tumor] = 1
    mask[stroma_ring] = 2
    if with_immune:
        mask[42:50, 78:86] = 4
        mask[10:20, 96:106] = 4
    return mask


class Phase3StromalDesmoplasiaPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)
        self.primitive = _primitive(self.recipe, "stromal_desmoplasia")
        self.schema = MaskProfileSchema.from_reference_profile("BCSS")

    def test_stromal_desmoplasia_expands_stroma_without_entering_tumor(self):
        old_mask = _desmoplasia_mask()
        context = MaskEditContext.from_mask(old_mask, self.schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_desmoplasia",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.10,
                "parameters": {"min_desmoplasia_component_area_px": 1},
                "seed": 5,
            }
        )

        result = apply_stromal_desmoplasia(
            old_mask,
            self.schema,
            context,
            self.primitive,
            intent,
        )

        self.assertGreater(result.selected_pixels, 0)
        self.assertTrue(np.all(old_mask[result.change_region] != 1))
        self.assertTrue(np.all(result.target_mask[result.change_region] == 2))
        self.assertGreater(np.count_nonzero(result.target_mask == 2), np.count_nonzero(old_mask == 2))
        self.assertEqual(result.ops_log["spatial"]["method"], "peritumoral_stroma_expansion_score_field")
        dist_to_tumor = ndimage.distance_transform_edt(old_mask != 1)
        self.assertLessEqual(
            float(dist_to_tumor[result.change_region].max()),
            self.primitive["parameter_ranges"]["max_distance_from_tumor_px"],
        )

    def test_stromal_desmoplasia_consumes_only_stroma_adjacent_immune(self):
        old_mask = _desmoplasia_mask(with_immune=True)
        context = MaskEditContext.from_mask(old_mask, self.schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_desmoplasia",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.20,
                "parameters": {"min_desmoplasia_component_area_px": 1},
                "seed": 7,
            }
        )

        result = apply_stromal_desmoplasia(
            old_mask,
            self.schema,
            context,
            self.primitive,
            intent,
        )

        consumed_immune = result.change_region & (old_mask == 4)
        far_immune = np.zeros_like(old_mask, dtype=bool)
        far_immune[10:20, 96:106] = True
        self.assertFalse(np.any(consumed_immune & far_immune))
        self.assertLessEqual(
            result.ops_log["spatial"]["selected_immune_pixels"] / result.selected_pixels,
            0.30,
        )

    def test_executor_runs_stromal_desmoplasia_and_validates(self):
        old_mask = _desmoplasia_mask()
        context = MaskEditContext.from_mask(old_mask, self.schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_desmoplasia",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.10,
                "parameters": {"min_desmoplasia_component_area_px": 1},
                "seed": 9,
            }
        )

        result = execute_edit(old_mask, intent, self.recipe, self.schema, context)

        self.assertIn(result.status, ("executed_validated", "degraded_executed"))
        self.assertIsNotNone(result.edit_result)
        self.assertIsNotNone(result.validation)
        self.assertTrue(result.validation.passed)

    def test_stromal_desmoplasia_rejects_no_tumor(self):
        old_mask = np.full((32, 32), 2, dtype=np.int64)
        context = MaskEditContext.from_mask(old_mask, self.schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_desmoplasia",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.10,
            }
        )

        with self.assertRaisesRegex(PrimitiveExecutionError, "no_tumor"):
            apply_stromal_desmoplasia(
                old_mask,
                self.schema,
                context,
                self.primitive,
                intent,
            )


if __name__ == "__main__":
    unittest.main()
