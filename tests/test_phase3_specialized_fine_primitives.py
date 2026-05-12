import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.parser.semantic_diff import DEFAULT_SEMANTIC_DIFF
from phase3_mask_edit.cli.edit_from_intents import execute_intents_on_mask
from phase3_mask_edit.generic.executor import execute_edit
from phase3_mask_edit.rules.semantic_to_intent import plan_edit_intents


class SpecializedFinePrimitiveRecipeTests(unittest.TestCase):
    def test_fine_dataset_recipes_expand_specialized_primitives(self):
        expected = {
            "bcss.yaml": set(),
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
                primitives = {
                    primitive["name"]: primitive for primitive in recipe["primitives"]
                }
                primitive_names = set(primitives)
                self.assertTrue(names.issubset(primitive_names))
                self.assertEqual(set(recipe["specialized_strategies"]), names)
                for name in names:
                    self.assertEqual(primitives[name]["execution_strategy"], "id_transition")

    def test_generic_primitives_default_to_geometric_organic_strategy(self):
        recipe = load_recipe("phase3_mask_edit/recipes/generic.yaml")
        primitive = next(
            primitive
            for primitive in recipe["primitives"]
            if primitive["name"] == "stromal_immune_infiltration"
        )
        self.assertEqual(primitive["execution_strategy"], "geometric_organic")

    def test_coarse_dataset_recipes_do_not_add_fine_specials(self):
        for filename in ("ignite.yaml", "puma.yaml", "orca.yaml"):
            with self.subTest(filename=filename):
                recipe = load_recipe(Path("phase3_mask_edit/recipes") / filename)
                self.assertEqual(recipe["specialized_strategies"], [])


class SpecializedFinePrimitiveExecutionTests(unittest.TestCase):
    def test_planned_panda_grade_special_executes_with_default_dataset_recipe(self):
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
        diff = {
            **DEFAULT_SEMANTIC_DIFF,
            "tumor_change": {
                **DEFAULT_SEMANTIC_DIFF["tumor_change"],
                "growth": "none",
                "grade_change": "upgrade",
            },
        }

        plan = plan_edit_intents(
            diff,
            reference_profile="PANDA",
            old_mask=old_mask,
            old_prompt="Gleason pattern 3 adenocarcinoma.",
            new_prompt="Gleason pattern 4 adenocarcinoma.",
        )

        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.intents[0].primitive, "gleason_upgrade_3to4")
        self.assertEqual(plan.items[0].status, "planned")
        result = execute_intents_on_mask(
            old_mask,
            plan.intents,
            reference_profile="PANDA",
        )
        self.assertEqual(result.status, "executed")
        self.assertLess(np.count_nonzero(result.target_mask == 8), np.count_nonzero(old_mask == 8))
        self.assertGreater(np.count_nonzero(result.target_mask == 9), np.count_nonzero(old_mask == 9))

    def test_panda_gleason_upgrade_3to4_runs_in_place(self):
        recipe = load_recipe("phase3_mask_edit/recipes/panda.yaml")
        schema = MaskProfileSchema.from_reference_profile("PANDA")
        old_mask = np.array(
            [
                [8, 8, 2, 8, 8],
                [8, 8, 2, 8, 8],
                [2, 2, 2, 2, 2],
                [5, 5, 2, 9, 9],
                [0, 0, 2, 9, 9],
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
        ops = result.edit_result.ops_log
        self.assertLess(np.count_nonzero(target == 8), np.count_nonzero(old_mask == 8))
        self.assertGreater(np.count_nonzero(target == 9), np.count_nonzero(old_mask == 9))
        self.assertTrue(np.all(old_mask[result.edit_result.change_region] == 8))
        self.assertEqual(ops["selection_unit"], "connected_component")
        self.assertEqual(ops["selection_policy"], "whole_source_components")
        self.assertEqual(ops["execution_strategy"], "id_transition")
        self.assertEqual(
            ops["target_change_fraction_semantics"],
            "source_fine_id_relative_relabel_fraction",
        )
        self.assertEqual(ops["target_change_fraction_denominator"], "source_fine_id_pixels")
        self.assertEqual(ops["candidate_pixels"], 8)
        self.assertEqual(ops["target_pixels"], 4)
        self.assertEqual(ops["selected_pixels"], 4)
        self.assertEqual(ops["selected_component_areas"], [4])
        self.assertAlmostEqual(ops["source_relative_fraction"], 4 / 8)
        self.assertAlmostEqual(ops["changed_area_fraction"], 4 / old_mask.size)

        changed = result.edit_result.change_region
        self.assertTrue(
            np.array_equal(changed[:2, :2], np.ones((2, 2), dtype=bool))
            or np.array_equal(changed[:2, 3:], np.ones((2, 2), dtype=bool))
        )
        self.assertFalse(np.any(changed[:2, :2]) and np.any(changed[:2, 3:]))

    def test_glas_grade_upgrade_uses_whole_connected_components(self):
        recipe = load_recipe("phase3_mask_edit/recipes/glas.yaml")
        schema = MaskProfileSchema.from_reference_profile("GlaS")
        old_mask = np.array(
            [
                [12, 12, 2, 12, 12],
                [12, 12, 2, 12, 12],
                [2, 2, 2, 2, 2],
                [11, 11, 2, 13, 13],
                [0, 0, 2, 13, 13],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(old_mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "grade_upgrade",
                "reference_profile": "GlaS",
                "target_change_fraction": 0.5,
            }
        )

        result = execute_edit(old_mask, intent, recipe, schema, context)

        self.assertEqual(result.status, "executed_validated")
        self.assertIsNotNone(result.edit_result)
        target = result.edit_result.target_mask
        ops = result.edit_result.ops_log
        self.assertLess(np.count_nonzero(target == 12), np.count_nonzero(old_mask == 12))
        self.assertGreater(np.count_nonzero(target == 13), np.count_nonzero(old_mask == 13))
        self.assertTrue(np.all(old_mask[result.edit_result.change_region] == 12))
        self.assertEqual(ops["selection_unit"], "connected_component")
        self.assertEqual(ops["selection_policy"], "whole_source_components")
        self.assertEqual(ops["candidate_pixels"], 8)
        self.assertEqual(ops["target_pixels"], 4)
        self.assertEqual(ops["selected_pixels"], 4)
        self.assertEqual(ops["selected_component_areas"], [4])

        changed = result.edit_result.change_region
        self.assertTrue(
            np.array_equal(changed[:2, :2], np.ones((2, 2), dtype=bool))
            or np.array_equal(changed[:2, 3:], np.ones((2, 2), dtype=bool))
        )
        self.assertFalse(np.any(changed[:2, :2]) and np.any(changed[:2, 3:]))

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
