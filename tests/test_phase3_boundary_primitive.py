"""Tests for phase3_mask_edit/generic/boundary.py — pushing remodel."""

import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.boundary import (
    apply_boundary_pushing_remodel,
)
from phase3_mask_edit.generic.tumor_burden import PrimitiveExecutionError


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


def _primitive(recipe, name):
    return next(
        p for p in recipe["primitives"] if p["name"] == name
    )


class Phase3BoundaryPushingRemodelTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)
        self.primitive = _primitive(self.recipe, "boundary_pushing_remodel")

    def test_boundary_pushing_remodel_removes_spike_and_fills_notch(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 1, 2, 2, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 2, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 2, 2, 1, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "boundary_pushing_remodel",
                "reference_profile": "BCSS",
                "target_change_fraction": 4 / 49,
                "parameters": {
                    "smooth_radius": 1,
                    "min_component_area_px": 1,
                    "max_abs_tumor_area_delta_fraction": 4 / 49,
                },
            }
        )

        result = apply_boundary_pushing_remodel(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertEqual(result.target_mask[1, 3], 2)
        self.assertEqual(result.target_mask[5, 3], 2)
        self.assertEqual(result.target_mask[3, 3], 1)
        self.assertLessEqual(
            abs(np.count_nonzero(result.target_mask == 1) - np.count_nonzero(old_mask == 1)),
            4,
        )
        self.assertTrue(np.all(result.target_mask[old_mask == 0] == 0))
        self.assertEqual(result.ops_log["primitive"], "boundary_pushing_remodel")

    def test_boundary_pushing_remodel_does_not_fill_necrosis_or_background_holes(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [2, 2, 2, 2, 2, 2, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 3, 1, 1, 2],
                [2, 1, 1, 0, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 2, 2, 2, 2, 2, 2],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "boundary_pushing_remodel",
                "reference_profile": "BCSS",
                "target_change_fraction": 8 / 49,
                "parameters": {
                    "smooth_radius": 1,
                    "min_component_area_px": 1,
                },
            }
        )

        result = apply_boundary_pushing_remodel(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertEqual(result.target_mask[2, 3], 3)
        self.assertEqual(result.target_mask[3, 3], 0)
        self.assertFalse(result.change_region[2, 3])
        self.assertFalse(result.change_region[3, 3])

    def test_boundary_pushing_remodel_uses_recipe_defaults_for_radius_and_delta_limit(self):
        self.assertEqual(
            self.primitive["parameter_ranges"]["default_smooth_radius_px"],
            18,
        )
        self.assertEqual(
            self.primitive["parameter_ranges"]["max_abs_tumor_area_delta_fraction"],
            0.02,
        )

        schema = MaskProfileSchema.from_reference_profile("BCSS")
        size = 160
        old_mask = np.full((size, size), 2, dtype=np.int64)
        yy, xx = np.mgrid[:size, :size]
        center = size // 2
        radius = size * 0.28
        tumor = (yy - center) ** 2 + (xx - center) ** 2 < radius * radius
        tumor[center - 4 : center + 4, int(center + radius) - 2 : int(center + radius) + 28] = True
        tumor[
            int(center - radius) - 24 : int(center - radius) + 3,
            center - 4 : center + 4,
        ] = True
        old_mask[tumor] = 1
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "boundary_pushing_remodel",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.03,
                "parameters": {"min_component_area_px": 20},
            }
        )

        result = apply_boundary_pushing_remodel(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertEqual(result.ops_log["spatial"]["smooth_radius"], 18)
        self.assertEqual(
            result.ops_log["spatial"]["max_abs_tumor_delta_fraction"],
            0.02,
        )


if __name__ == "__main__":
    unittest.main()