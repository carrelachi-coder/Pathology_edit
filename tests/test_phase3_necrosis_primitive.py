"""Tests for phase3_mask_edit/generic/necrosis.py."""

import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.executor import execute_edit
from phase3_mask_edit.generic.necrosis import (
    _edge_aware_tumor_interior_distance,
    apply_necrosis_appearance,
    apply_necrosis_resolution,
)
from phase3_mask_edit.generic.tumor_burden import PrimitiveExecutionError


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


def _primitive(recipe, name):
    return next(p for p in recipe["primitives"] if p["name"] == name)


def _large_tumor_mask(size=512, *, with_necrosis=False, with_vessel=False):
    mask = np.zeros((size, size), dtype=np.int64)
    mask[48:-48, 48:-48] = 2
    yy, xx = np.mgrid[:size, :size]
    tumor = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= (size * 0.34) ** 2
    mask[tumor] = 1
    if with_necrosis:
        nec = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= 18**2
        mask[nec] = 3
    if with_vessel:
        mask[250:262, 72:210] = 6
    return mask


class Phase3NecrosisPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)
        self.primitive = _primitive(self.recipe, "necrosis_appearance")
        self.resolution_primitive = _primitive(self.recipe, "necrosis_resolution")

    def test_necrosis_appearance_replaces_only_original_tumor(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _large_tumor_mask(with_necrosis=False)
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.04,
                "parameters": {"min_necrosis_component_area_px": 4},
                "seed": 1,
            }
        )

        result = apply_necrosis_appearance(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertGreater(result.selected_pixels, 0)
        self.assertTrue(np.all(result.change_region <= (old_mask == 1)))
        self.assertTrue(np.all(result.target_mask[result.change_region] == 3))
        self.assertTrue(np.all(result.target_mask[old_mask == 0] == 0))
        self.assertTrue(np.all(result.target_mask[old_mask == 2] == 2))
        self.assertGreaterEqual(result.ops_log["spatial"]["selected_components"], 1)
        self.assertEqual(result.ops_log["spatial"]["max_components"], 2)
        filled = ndimage.binary_fill_holes(result.change_region) & (old_mask == 1)
        self.assertEqual(
            int(np.count_nonzero(filled & ~result.change_region)),
            0,
        )

    def test_necrosis_appearance_uses_existing_necrosis_neighborhood_when_present(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _large_tumor_mask(with_necrosis=True)
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.03,
                "parameters": {"necrosis_score_noise_weight": 0.0},
            }
        )

        result = apply_necrosis_appearance(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertTrue(
            result.ops_log["spatial"]["used_existing_necrosis_neighborhood"]
        )
        necrosis_boundary_distance = ndimage.distance_transform_edt(old_mask != 3)
        selected_distance = float(necrosis_boundary_distance[result.change_region].mean())
        rows, cols = np.where(old_mask == 1)
        bbox_diagonal = float(
            np.hypot(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1)
        )
        self.assertLess(selected_distance, max(64.0, 0.25 * bbox_diagonal))

    def test_necrosis_appearance_falls_back_without_existing_necrosis(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _large_tumor_mask(with_necrosis=False)
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.03,
                "parameters": {"necrosis_score_noise_weight": 0.0},
            }
        )

        result = apply_necrosis_appearance(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertGreater(result.selected_pixels, 0)
        self.assertFalse(
            result.ops_log["spatial"]["used_existing_necrosis_neighborhood"]
        )
        self.assertEqual(result.ops_log["spatial"]["selected_components"], 1)
        self.assertIn(
            "tumor_interior_far_from_outer_boundary",
            result.ops_log["spatial"]["active_weights"],
        )
        self.assertEqual(result.ops_log["spatial"]["max_components"], 2)

    def test_necrosis_appearance_avoids_boundary_when_interior_available(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _large_tumor_mask(with_necrosis=False)
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.02,
                "parameters": {"necrosis_score_noise_weight": 0.0},
            }
        )

        result = apply_necrosis_appearance(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        interior_distance = ndimage.distance_transform_edt(old_mask == 1)
        self.assertGreater(float(interior_distance[result.change_region].mean()), 20.0)

    def test_edge_aware_tumor_distance_does_not_treat_patch_edge_as_tumor_edge(self):
        source_tumor = np.zeros((96, 96), dtype=bool)
        source_tumor[:72, :] = True

        false_padded = ndimage.distance_transform_edt(
            np.pad(source_tumor, 32, mode="constant", constant_values=False)
        )[32:-32, 32:-32]
        edge_aware = _edge_aware_tumor_interior_distance(
            source_tumor,
            pad_width=32,
        )

        self.assertLess(false_padded[0, 48], 2.0)
        self.assertGreater(edge_aware[0, 48], 60.0)
        self.assertGreater(edge_aware[12, 48], false_padded[12, 48])

    def test_necrosis_appearance_can_span_multiple_tumor_components(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.zeros((256, 256), dtype=np.int64)
        old_mask[16:240, 16:240] = 2
        yy, xx = np.mgrid[:256, :256]
        left = (yy - 128) ** 2 + (xx - 78) ** 2 <= 46**2
        right = (yy - 128) ** 2 + (xx - 178) ** 2 <= 46**2
        old_mask[left | right] = 1
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.08,
                "parameters": {
                    "necrosis_score_noise_weight": 0.25,
                    "min_necrosis_component_area_px": 4,
                    "max_necrosis_components": 2,
                },
                "seed": 23,
            }
        )

        result = apply_necrosis_appearance(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertTrue(np.any(result.change_region & left))
        self.assertTrue(np.any(result.change_region & right))
        self.assertGreaterEqual(result.ops_log["spatial"]["selected_components"], 2)

    def test_necrosis_appearance_extends_existing_necrosis_without_overwriting_it(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _large_tumor_mask(with_necrosis=True)
        existing = old_mask == 3
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.04,
                "parameters": {"min_necrosis_component_area_px": 4},
                "seed": 29,
            }
        )

        result = apply_necrosis_appearance(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        self.assertFalse(np.any(result.change_region & existing))
        self.assertTrue(np.all(result.target_mask[existing] == 3))
        self.assertGreater(np.count_nonzero(result.target_mask == 3), np.count_nonzero(existing))

    def test_necrosis_appearance_can_touch_patch_edge(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.full((160, 160), 2, dtype=np.int64)
        old_mask[:120, 20:140] = 1
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.10,
                "parameters": {
                    "necrosis_score_noise_weight": 0.0,
                    "min_necrosis_component_area_px": 16,
                    "max_necrosis_components": 1,
                },
                "seed": 31,
            }
        )

        result = apply_necrosis_appearance(
            old_mask,
            schema,
            context,
            self.primitive,
            intent,
        )

        edge_pixels = np.zeros_like(result.change_region, dtype=bool)
        edge_pixels[0, :] = True
        edge_pixels[-1, :] = True
        edge_pixels[:, 0] = True
        edge_pixels[:, -1] = True
        self.assertTrue(np.any(result.change_region & edge_pixels))

    def test_necrosis_appearance_respects_max_necrosis_fraction_of_tumor(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _large_tumor_mask(with_necrosis=False)
        tumor_coords = np.argwhere(old_mask == 1)
        old_mask[
            tumor_coords[: int(tumor_coords.shape[0] * 0.61), 0],
            tumor_coords[: int(tumor_coords.shape[0] * 0.61), 1],
        ] = 3
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.01,
            }
        )

        with self.assertRaisesRegex(
            PrimitiveExecutionError, "necrosis_fraction_limit_reached"
        ):
            apply_necrosis_appearance(
                old_mask,
                schema,
                context,
                self.primitive,
                intent,
            )

    def test_necrosis_appearance_rejects_profile_without_necrosis_label(self):
        schema = MaskProfileSchema.from_reference_profile("PANDA")
        old_mask = np.full((64, 64), 2, dtype=np.int64)
        old_mask[16:48, 16:48] = 8
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "PANDA",
                "target_change_fraction": 0.05,
            }
        )

        with self.assertRaisesRegex(PrimitiveExecutionError, "no_necrosis_label"):
            apply_necrosis_appearance(
                old_mask,
                schema,
                context,
                self.primitive,
                intent,
            )

    def test_executor_runs_necrosis_appearance(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _large_tumor_mask(with_necrosis=True)
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.08,
                "parameters": {"necrosis_score_noise_weight": 0.0},
            }
        )

        result = execute_edit(old_mask, intent, self.recipe, schema, context)

        self.assertIn(result.status, ("executed_validated", "degraded_executed"))
        self.assertIsNotNone(result.edit_result)
        self.assertIsNotNone(result.validation)
        self.assertTrue(result.validation.passed)
        self.assertGreater(result.edit_result.selected_pixels, 0)

    def test_necrosis_resolution_backfills_to_stroma(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.full((96, 96), 2, dtype=np.int64)
        old_mask[12:84, 12:48] = 1
        old_mask[36:60, 36:60] = 3
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_resolution",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.50,
                "parameters": {
                    "min_necrosis_resolution_component_area_px": 1,
                    "necrosis_resolution_noise_weight": 0.0,
                },
            }
        )

        result = apply_necrosis_resolution(
            old_mask,
            schema,
            context,
            self.resolution_primitive,
            intent,
        )

        self.assertGreater(result.selected_pixels, 0)
        self.assertTrue(np.all(result.change_region <= (old_mask == 3)))
        self.assertFalse(np.any(result.target_mask[result.change_region] == 3))
        self.assertTrue(np.all(result.target_mask[result.change_region] == 2))
        self.assertLess(
            np.count_nonzero(result.target_mask == 3),
            np.count_nonzero(old_mask == 3),
        )
        self.assertEqual(result.ops_log["spatial"]["backfill_labels"], ["Stroma"])

    def test_necrosis_resolution_rejects_tumor_only_backfill(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.full((96, 96), 1, dtype=np.int64)
        old_mask[32:64, 32:64] = 3
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_resolution",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.25,
                "parameters": {
                    "min_necrosis_resolution_component_area_px": 1,
                    "necrosis_resolution_noise_weight": 0.0,
                },
            }
        )

        with self.assertRaisesRegex(PrimitiveExecutionError, "no_valid_backfill_tissue"):
            apply_necrosis_resolution(
                old_mask,
                schema,
                context,
                self.resolution_primitive,
                intent,
            )

    def test_necrosis_resolution_does_not_treat_patch_background_as_necrosis_edge(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = np.zeros((64, 64), dtype=np.int64)
        old_mask[16:48, 0:20] = 3
        old_mask[16:48, 20:34] = 2
        old_mask[16:48, 34:56] = 1
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_resolution",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.10,
                "parameters": {
                    "min_necrosis_resolution_component_area_px": 1,
                    "necrosis_resolution_noise_weight": 0.0,
                },
            }
        )

        result = apply_necrosis_resolution(
            old_mask,
            schema,
            context,
            self.resolution_primitive,
            intent,
        )

        self.assertGreater(result.selected_pixels, 0)
        self.assertTrue(np.all(result.target_mask[result.change_region] == 2))
        selected_cols = np.argwhere(result.change_region)[:, 1]
        self.assertGreaterEqual(int(selected_cols.min()), 17)
        self.assertGreater(float(np.mean(selected_cols)), 17.0)

    def test_executor_runs_necrosis_resolution_and_validates(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        old_mask = _large_tumor_mask(with_necrosis=True)
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_resolution",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.10,
                "parameters": {
                    "min_necrosis_resolution_component_area_px": 1,
                    "necrosis_resolution_noise_weight": 0.0,
                },
            }
        )

        result = execute_edit(old_mask, intent, self.recipe, schema, context)

        self.assertIn(result.status, ("executed_validated", "degraded_executed"))
        self.assertIsNotNone(result.edit_result)
        self.assertIsNotNone(result.validation)
        self.assertTrue(result.validation.passed)
        self.assertGreater(result.edit_result.selected_pixels, 0)


if __name__ == "__main__":
    unittest.main()
