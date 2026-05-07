import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit.core.applicability import assess_edit_applicability
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


class Phase3ApplicabilityTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)

    def test_tumor_burden_increase_on_orca_tumor_other_mask_is_executable(self):
        schema = MaskProfileSchema.from_reference_profile("ORCA")
        context = MaskEditContext.from_mask(
            np.array(
                [
                    [1, 1, 7],
                    [1, 7, 7],
                    [0, 7, 7],
                ],
                dtype=np.int64,
            ),
            schema,
        )
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "ORCA",
            }
        )

        decision = assess_edit_applicability(intent, self.recipe, schema, context)

        self.assertEqual(decision.status, "executable")
        self.assertEqual(decision.reasons, ())

    def test_stromal_immune_infiltration_on_panda_rejected_without_immune_profile_label(self):
        schema = MaskProfileSchema.from_reference_profile("PANDA")
        context = MaskEditContext.from_mask(
            np.array(
                [
                    [8, 8, 2],
                    [8, 2, 2],
                    [5, 5, 2],
                ],
                dtype=np.int64,
            ),
            schema,
        )
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_immune_infiltration",
                "reference_profile": "PANDA",
            }
        )

        decision = assess_edit_applicability(intent, self.recipe, schema, context)

        self.assertEqual(decision.status, "rejected")
        self.assertIn(
            "target_label_not_writable:Immune infiltrate",
            decision.reasons,
        )

    def test_necrosis_appearance_on_bcss_without_blood_vessel_is_degraded(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        context = MaskEditContext.from_mask(
            np.array(
                [
                    [1, 1, 1, 2],
                    [1, 1, 2, 2],
                    [1, 2, 2, 2],
                    [0, 2, 2, 2],
                ],
                dtype=np.int64,
            ),
            schema,
        )
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "reference_profile": "BCSS",
            }
        )

        decision = assess_edit_applicability(intent, self.recipe, schema, context)

        self.assertEqual(decision.status, "degraded")
        self.assertIn("optional_label_absent_in_mask:Blood vessel", decision.warnings)

    def test_intratumoral_immune_infiltration_without_tumor_is_rejected(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        context = MaskEditContext.from_mask(
            np.array(
                [
                    [2, 2, 4],
                    [2, 4, 4],
                    [0, 2, 2],
                ],
                dtype=np.int64,
            ),
            schema,
        )
        intent = EditIntent.from_mapping(
            {
                "primitive": "intratumoral_immune_infiltration",
                "reference_profile": "BCSS",
            }
        )

        decision = assess_edit_applicability(intent, self.recipe, schema, context)

        self.assertEqual(decision.status, "rejected")
        self.assertIn("required_context_label_absent_in_mask:Tumor", decision.reasons)

    def test_profile_mismatch_is_rejected(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        context = MaskEditContext.from_mask(
            np.array([[1, 2], [1, 2]], dtype=np.int64),
            schema,
        )
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "ORCA",
            }
        )

        decision = assess_edit_applicability(intent, self.recipe, schema, context)

        self.assertEqual(decision.status, "rejected")
        self.assertIn("reference_profile_mismatch:intent=ORCA,schema=BCSS", decision.reasons)


if __name__ == "__main__":
    unittest.main()
