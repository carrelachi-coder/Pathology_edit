import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.tumor_burden import (
    PrimitiveExecutionError,
    apply_tumor_burden_decrease,
    apply_tumor_burden_increase,
)


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


def _primitive(recipe, name):
    return next(
        primitive for primitive in recipe["primitives"] if primitive["name"] == name
    )


class Phase3TumorBurdenPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)
        self.increase_primitive = _primitive(self.recipe, "tumor_burden_increase")
        self.decrease_primitive = _primitive(self.recipe, "tumor_burden_decrease")

    def test_tumor_burden_increase_expands_tumor_into_boundary_candidates(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [1, 1, 2, 2, 0],
                [1, 1, 2, 2, 0],
                [2, 2, 2, 2, 0],
                [7, 7, 7, 7, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "BCSS",
                "target_change_fraction": 5 / 25,
            }
        )

        result = apply_tumor_burden_increase(
            old_mask,
            schema,
            context,
            self.increase_primitive,
            intent,
        )

        expected_change = np.zeros_like(old_mask, dtype=bool)
        expected_change[0, 2] = True
        expected_change[1, 2] = True
        expected_change[2, 0:3] = True

        np.testing.assert_array_equal(result.change_region, expected_change)
        self.assertTrue(np.all(result.target_mask[expected_change] == 1))
        self.assertTrue(np.all(result.target_mask[old_mask == 0] == 0))
        self.assertGreater(
            int(np.count_nonzero(result.target_mask == 1)),
            int(np.count_nonzero(old_mask == 1)),
        )
        self.assertEqual(result.changed_area_fraction, 5 / 25)
        self.assertEqual(result.ops_log["primitive"], "tumor_burden_increase")
        self.assertEqual(result.ops_log["candidate_labels"], ["Stroma"])

    def test_tumor_burden_increase_honors_preserve_labels(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [1, 1, 2, 5, 0],
                [1, 1, 5, 5, 0],
                [2, 5, 5, 5, 0],
                [7, 7, 7, 7, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "BCSS",
                "target_change_fraction": 3 / 25,
                "preserve_labels": ["Stroma"],
            }
        )

        result = apply_tumor_burden_increase(
            old_mask,
            schema,
            context,
            self.increase_primitive,
            intent,
        )

        self.assertTrue(np.all(result.target_mask[old_mask == 2] == 2))
        self.assertTrue(np.all(result.change_region <= (old_mask == 5)))
        self.assertEqual(result.ops_log["candidate_labels"], ["Normal epithelium"])

    def test_tumor_burden_increase_raises_when_no_editable_candidate_region_exists(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [1, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "BCSS",
                "target_change_fraction": 1 / 9,
            }
        )

        with self.assertRaisesRegex(PrimitiveExecutionError, "no editable"):
            apply_tumor_burden_increase(
                old_mask,
                schema,
                context,
                self.increase_primitive,
                intent,
            )

    def test_tumor_burden_increase_reports_only_necrosis_available(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [1, 1, 3],
                [1, 3, 3],
                [3, 3, 3],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "BCSS",
                "target_change_fraction": 1 / 9,
            }
        )

        with self.assertRaisesRegex(
            PrimitiveExecutionError,
            "no_editable_non_tumor_tissue_only_necrosis_available",
        ):
            apply_tumor_burden_increase(
                old_mask,
                schema,
                context,
                self.increase_primitive,
                intent,
            )

    def test_tumor_burden_decrease_shrinks_boundary_and_backfills_with_stroma(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [2, 2, 2, 2, 2],
                [2, 1, 1, 1, 2],
                [2, 1, 1, 1, 2],
                [2, 1, 1, 1, 2],
                [2, 2, 2, 2, 2],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_decrease",
                "reference_profile": "BCSS",
                "target_change_fraction": 8 / 25,
                "parameters": {"smooth_boundary": False},
            }
        )

        result = apply_tumor_burden_decrease(
            old_mask,
            schema,
            context,
            self.decrease_primitive,
            intent,
        )

        expected_change = np.zeros_like(old_mask, dtype=bool)
        expected_change[1:4, 1:4] = True
        expected_change[2, 2] = False

        np.testing.assert_array_equal(result.change_region, expected_change)
        self.assertEqual(result.target_mask[2, 2], 1)
        self.assertTrue(np.all(result.target_mask[expected_change] == 2))
        self.assertEqual(result.changed_area_fraction, 8 / 25)
        self.assertEqual(result.ops_log["primitive"], "tumor_burden_decrease")
        self.assertEqual(result.ops_log["backfill_labels"], ["Stroma"])

    def test_tumor_burden_decrease_does_not_shrink_background_facing_boundary(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_decrease",
                "reference_profile": "BCSS",
                "target_change_fraction": 3 / 25,
                "parameters": {"smooth_boundary": False},
            }
        )

        result = apply_tumor_burden_decrease(
            old_mask,
            schema,
            context,
            self.decrease_primitive,
            intent,
        )

        expected_change = np.zeros_like(old_mask, dtype=bool)
        expected_change[2, 2] = True

        np.testing.assert_array_equal(result.change_region, expected_change)
        self.assertTrue(np.all(result.target_mask[1, 1:4] == 1))
        self.assertEqual(result.target_mask[2, 2], 2)
        self.assertTrue(result.ops_log["spatial"]["target_area_shortfall"])

    def test_tumor_burden_decrease_allows_shrink_near_internal_background_hole(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [2, 1, 1, 1, 1, 1, 0],
                [2, 1, 1, 1, 1, 1, 0],
                [2, 1, 1, 0, 1, 1, 0],
                [2, 1, 1, 1, 1, 1, 0],
                [2, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_decrease",
                "reference_profile": "BCSS",
                "target_change_fraction": 6 / 49,
                "parameters": {"smooth_boundary": False},
            }
        )

        result = apply_tumor_burden_decrease(
            old_mask,
            schema,
            context,
            self.decrease_primitive,
            intent,
        )

        internal_hole_neighbors = [(2, 3), (3, 2), (3, 4), (4, 3)]
        self.assertTrue(
            any(result.change_region[row, col] for row, col in internal_hole_neighbors)
        )
        self.assertEqual(result.target_mask[3, 3], 0)

    def test_tumor_burden_decrease_preserves_tumor_rim_around_necrosis(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [2, 2, 2, 2, 2, 2, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 3, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 2, 2, 2, 2, 2, 2],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_decrease",
                "reference_profile": "BCSS",
                "target_change_fraction": 16 / 49,
                "parameters": {"smooth_boundary": False},
            }
        )

        result = apply_tumor_burden_decrease(
            old_mask,
            schema,
            context,
            self.decrease_primitive,
            intent,
        )

        necrosis_neighbors = [(2, 3), (3, 2), (3, 4), (4, 3)]
        for row, col in necrosis_neighbors:
            self.assertEqual(result.target_mask[row, col], 1)
            self.assertFalse(result.change_region[row, col])
        self.assertTrue(np.any(result.change_region))
        self.assertEqual(result.target_mask[3, 3], 3)

    def test_tumor_burden_decrease_applies_boundary_smoothing_by_default(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [2, 2, 2, 2, 2, 2, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 1, 1, 1, 1, 1, 2],
                [2, 2, 2, 2, 2, 2, 2],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_decrease",
                "reference_profile": "BCSS",
                "target_change_fraction": 16 / 49,
            }
        )

        result = apply_tumor_burden_decrease(
            old_mask,
            schema,
            context,
            self.decrease_primitive,
            intent,
        )

        self.assertTrue(result.ops_log["spatial"]["smoothing_applied"])
        self.assertEqual(result.ops_log["spatial"]["smoothing_method"], "gaussian_threshold")
        self.assertGreater(result.ops_log["spatial"]["smoothing_radius"], 0)
        self.assertTrue(np.all(result.change_region <= (old_mask == 1)))

    def test_tumor_burden_decrease_fails_without_valid_backfill_tissue(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_decrease",
                "reference_profile": "BCSS",
                "target_change_fraction": 1 / 9,
            }
        )

        with self.assertRaisesRegex(PrimitiveExecutionError, "no_valid_backfill_tissue"):
            apply_tumor_burden_decrease(
                old_mask,
                schema,
                context,
                self.decrease_primitive,
                intent,
            )

    def test_tumor_burden_decrease_fails_when_default_mode_would_delete_tumor(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.array(
            [
                [2, 2, 2],
                [2, 1, 2],
                [2, 2, 2],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_decrease",
                "reference_profile": "BCSS",
                "target_change_fraction": 1 / 9,
            }
        )

        with self.assertRaisesRegex(PrimitiveExecutionError, "shrink_would_delete_tumor"):
            apply_tumor_burden_decrease(
                old_mask,
                schema,
                context,
                self.decrease_primitive,
                intent,
            )


if __name__ == "__main__":
    unittest.main()
