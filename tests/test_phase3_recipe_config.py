import copy
import unittest
from pathlib import Path

from phase3_mask_edit.core.config import (
    RecipeValidationError,
    load_recipe,
    validate_recipe_schema,
)


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


class Phase3RecipeConfigTests(unittest.TestCase):
    def test_generic_recipe_loads_with_frozen_stage1_shape(self):
        recipe = load_recipe(GENERIC_RECIPE)

        self.assertEqual(recipe["schema_version"], 1)
        self.assertEqual(len(recipe["primitives"]), 9)
        self.assertEqual(len(recipe["composite_recipes"]), 5)

        primitive_names = {primitive["name"] for primitive in recipe["primitives"]}
        self.assertEqual(
            primitive_names,
            {
                "tumor_burden_increase",
                "tumor_burden_decrease",
                "boundary_pushing_remodel",
                "necrosis_appearance",
                "necrosis_resolution",
                "stromal_immune_infiltration",
                "intratumoral_immune_infiltration",
                "immune_infiltration_decrease",
                "stromal_desmoplasia",
            },
        )

    def test_validator_rejects_missing_frozen_primitive_field(self):
        recipe = load_recipe(GENERIC_RECIPE)
        mutated = copy.deepcopy(recipe)
        del mutated["primitives"][0]["required_context"]

        with self.assertRaisesRegex(RecipeValidationError, "required_context"):
            validate_recipe_schema(mutated)

    def test_validator_rejects_unknown_tissue_label(self):
        recipe = load_recipe(GENERIC_RECIPE)
        mutated = copy.deepcopy(recipe)
        mutated["primitives"][0]["required_tissue_labels"].append("Mystery tissue")

        with self.assertRaisesRegex(RecipeValidationError, "Mystery tissue"):
            validate_recipe_schema(mutated)

    def test_validator_rejects_invalid_parameter_interval(self):
        recipe = load_recipe(GENERIC_RECIPE)
        mutated = copy.deepcopy(recipe)
        mutated["primitives"][0]["parameter_ranges"]["target_area_delta_fraction"][
            "mild"
        ] = [0.20, 0.10]

        with self.assertRaisesRegex(RecipeValidationError, "lower < upper"):
            validate_recipe_schema(mutated)

    def test_validator_rejects_boundary_primitive_with_xlarge_deid_bucket(self):
        recipe = load_recipe(GENERIC_RECIPE)
        mutated = copy.deepcopy(recipe)
        boundary = next(
            primitive
            for primitive in mutated["primitives"]
            if primitive["name"] == "boundary_pushing_remodel"
        )
        boundary["parameter_ranges"]["target_changed_area_fraction"][
            "xlarge_deid"
        ] = [0.40, 0.50]

        with self.assertRaisesRegex(RecipeValidationError, "xlarge_deid"):
            validate_recipe_schema(mutated)

    def test_validator_rejects_unknown_composite_primitive_reference(self):
        recipe = load_recipe(GENERIC_RECIPE)
        mutated = copy.deepcopy(recipe)
        mutated["composite_recipes"][0]["primitives"].append("not_a_primitive")

        with self.assertRaisesRegex(RecipeValidationError, "not_a_primitive"):
            validate_recipe_schema(mutated)

    def test_validator_rejects_unknown_use_overlap_guard_reference(self):
        recipe = load_recipe(GENERIC_RECIPE)
        mutated = copy.deepcopy(recipe)
        mutated["primitives"][0][
            "overlap_guard"
        ] = "use_not_a_primitive_for_this_operation"

        with self.assertRaisesRegex(RecipeValidationError, "not_a_primitive"):
            validate_recipe_schema(mutated)


if __name__ == "__main__":
    unittest.main()
