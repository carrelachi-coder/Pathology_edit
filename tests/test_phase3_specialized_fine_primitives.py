import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.executor import execute_edit


class SpecializedFinePrimitiveRecipeTests(unittest.TestCase):
    def test_fine_dataset_recipes_expand_specialized_primitives(self):
        expected = {
            "bcss.yaml": {"dcis_invasion", "angioinvasion_emphasis"},
            "panda.yaml": {
                "gleason_upgrade_3to4",
                "gleason_upgrade_4to5",
                "gleason_downgrade_4to3",
                "benign_to_gleason3",
                "benign_atrophy",
            },
            "glas.yaml": {
                "normal_to_adenomatous",
                "adenoma_to_carcinoma",
                "grade_upgrade",
                "treatment_dedifferentiation",
            },
        }

        for filename, names in expected.items():
            with self.subTest(filename=filename):
                recipe = load_recipe(Path("phase3_mask_edit/recipes") / filename)
                primitive_names = {primitive["name"] for primitive in recipe["primitives"]}
                self.assertTrue(names.issubset(primitive_names))
                self.assertEqual(set(recipe["specialized_strategies"]), names)

    def test_coarse_dataset_recipes_do_not_add_fine_specials(self):
        for filename in ("ignite.yaml", "puma.yaml", "orca.yaml"):
            with self.subTest(filename=filename):
                recipe = load_recipe(Path("phase3_mask_edit/recipes") / filename)
                self.assertEqual(recipe["specialized_strategies"], [])


class SpecializedFinePrimitiveExecutionTests(unittest.TestCase):
    def test_panda_gleason_upgrade_3to4_runs_in_place(self):
        recipe = load_recipe("phase3_mask_edit/recipes/panda.yaml")
        schema = MaskProfileSchema.from_reference_profile("PANDA")
        old_mask = np.array(
            [
                [8, 8, 8, 2, 0],
                [8, 8, 8, 2, 0],
                [8, 8, 9, 2, 0],
                [5, 5, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "gleason_upgrade_3to4",
                "reference_profile": "PANDA",
                "target_change_fraction": 0.5,
            }
        )

        result = execute_edit(old_mask, intent, recipe, schema, context)

        self.assertEqual(result.status, "executed_validated")
        self.assertIsNotNone(result.edit_result)
        target = result.edit_result.target_mask
        self.assertLess(np.count_nonzero(target == 8), np.count_nonzero(old_mask == 8))
        self.assertGreater(np.count_nonzero(target == 9), np.count_nonzero(old_mask == 9))
        self.assertTrue(np.all(old_mask[result.edit_result.change_region] == 8))

    def test_missing_source_fine_id_is_rejected(self):
        recipe = load_recipe("phase3_mask_edit/recipes/panda.yaml")
        schema = MaskProfileSchema.from_reference_profile("PANDA")
        old_mask = np.array(
            [
                [9, 9, 9, 2, 0],
                [9, 9, 9, 2, 0],
                [5, 5, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "gleason_upgrade_3to4",
                "reference_profile": "PANDA",
            }
        )

        result = execute_edit(old_mask, intent, recipe, schema, context)

        self.assertEqual(result.status, "rejected")
        self.assertTrue(
            any("source_fine_id_absent" in reason for reason in result.applicability.reasons)
        )


if __name__ == "__main__":
    unittest.main()
