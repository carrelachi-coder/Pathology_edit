import unittest

import numpy as np

from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.generic.executor import execute_edit


class StromaIncreasePrimitiveTests(unittest.TestCase):
    def test_generic_stroma_increase_needs_no_tumor(self):
        schema = MaskProfileSchema.from_reference_profile("GlaS")
        mask = np.full((256, 256), 5, dtype=np.int64)
        mask[:, :128] = 2
        intent = EditIntent(
            primitive="stroma_increase",
            strength="moderate",
            reference_profile="GlaS",
        )

        result = execute_edit(
            mask,
            intent,
            load_recipe(default_recipe_path_for_profile("GlaS")),
            schema,
            MaskEditContext.from_mask(mask, schema),
        )

        self.assertEqual(result.status, "executed_validated")
        self.assertGreater(result.edit_result.selected_pixels, 0)
        self.assertFalse(np.any(mask == 1))
        self.assertTrue(np.all(mask[result.edit_result.change_region] == 5))
        self.assertTrue(
            np.all(result.edit_result.target_mask[result.edit_result.change_region] == 2)
        )

    def test_desmoplasia_still_requires_tumor(self):
        schema = MaskProfileSchema.from_reference_profile("GlaS")
        mask = np.full((64, 64), 5, dtype=np.int64)
        mask[:, :32] = 2
        intent = EditIntent(
            primitive="stromal_desmoplasia",
            strength="moderate",
            reference_profile="GlaS",
        )

        result = execute_edit(
            mask,
            intent,
            load_recipe(default_recipe_path_for_profile("GlaS")),
            schema,
            MaskEditContext.from_mask(mask, schema),
        )

        self.assertEqual(result.status, "rejected")
        self.assertIn(
            "required_context_label_absent_in_mask:Tumor",
            result.applicability.reasons,
        )
