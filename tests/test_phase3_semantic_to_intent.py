import json
import unittest

import numpy as np

from phase3_mask_edit.parser.semantic_diff import DEFAULT_SEMANTIC_DIFF
from phase3_mask_edit.rules.semantic_to_intent import (
    plan_edit_intents,
    semantic_diff_to_intents,
)


def semantic_diff_with(**updates):
    payload = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
    for section, fields in updates.items():
        payload[section].update(fields)
    return payload


class Phase3SemanticToIntentTests(unittest.TestCase):
    def test_no_change_returns_no_intents(self):
        intents = semantic_diff_to_intents(
            DEFAULT_SEMANTIC_DIFF,
            reference_profile="BCSS",
        )

        self.assertEqual(intents, [])

    def test_necrosis_add_maps_to_mild_necrosis_appearance(self):
        diff = semantic_diff_with(necrosis_change={"action": "add", "extent": "focal"})

        intents = semantic_diff_to_intents(diff, reference_profile="BCSS")

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "necrosis_appearance")
        self.assertEqual(intents[0].strength, "mild")
        self.assertEqual(intents[0].reference_profile, "BCSS")

    def test_necrosis_increase_extensive_maps_to_significant(self):
        diff = semantic_diff_with(
            necrosis_change={"action": "increase", "extent": "extensive"}
        )

        intents = semantic_diff_to_intents(diff, reference_profile="BCSS")

        self.assertEqual(intents[0].primitive, "necrosis_appearance")
        self.assertEqual(intents[0].strength, "significant")

    def test_necrosis_decrease_maps_to_necrosis_resolution(self):
        diff = semantic_diff_with(
            necrosis_change={"action": "decrease", "extent": "moderate"}
        )

        intents = semantic_diff_to_intents(diff, reference_profile="BCSS")

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "necrosis_resolution")
        self.assertEqual(intents[0].strength, "moderate")

    def test_necrosis_remove_maps_to_significant_resolution(self):
        diff = semantic_diff_with(
            necrosis_change={"action": "remove", "extent": "focal"}
        )

        intents = semantic_diff_to_intents(diff, reference_profile="BCSS")

        self.assertEqual(intents[0].primitive, "necrosis_resolution")
        self.assertEqual(intents[0].strength, "significant")

    def test_immune_increase_maps_to_stromal_immune(self):
        diff = semantic_diff_with(
            lymphocyte_change={
                "infiltration": "increase",
                "degree": "significant",
            }
        )

        intents = semantic_diff_to_intents(diff, reference_profile="BCSS")

        self.assertEqual(intents[0].primitive, "stromal_immune_infiltration")
        self.assertEqual(intents[0].strength, "significant")

    def test_immune_decrease_maps_to_immune_decrease_primitive(self):
        diff = semantic_diff_with(
            lymphocyte_change={
                "infiltration": "decrease",
                "degree": "moderate",
            }
        )

        intents = semantic_diff_to_intents(diff, reference_profile="BCSS")

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "immune_infiltration_decrease")
        self.assertEqual(intents[0].strength, "moderate")

    def test_tumor_increase_and_decrease_map_to_tumor_primitives(self):
        increase = semantic_diff_with(
            tumor_change={"growth": "increase", "degree": "moderate"}
        )
        decrease = semantic_diff_with(
            tumor_change={"growth": "decrease", "degree": "moderate"}
        )

        increase_intents = semantic_diff_to_intents(increase, reference_profile="BCSS")
        decrease_intents = semantic_diff_to_intents(decrease, reference_profile="BCSS")

        self.assertEqual(increase_intents[0].primitive, "tumor_burden_increase")
        self.assertEqual(decrease_intents[0].primitive, "tumor_burden_decrease")

    def test_multiple_intents_are_returned_in_execution_order(self):
        diff = semantic_diff_with(
            lymphocyte_change={"infiltration": "increase", "degree": "moderate"},
            necrosis_change={"action": "add", "extent": "focal"},
            tumor_change={"growth": "increase", "degree": "mild"},
        )

        intents = semantic_diff_to_intents(diff, reference_profile="BCSS")

        self.assertEqual(
            [intent.primitive for intent in intents],
            [
                "tumor_burden_increase",
                "necrosis_appearance",
                "stromal_immune_infiltration",
            ],
        )

    def test_stroma_density_increase_maps_to_desmoplasia(self):
        diff = semantic_diff_with(
            stroma_change={"density": "increase", "degree": "moderate"}
        )

        plan = plan_edit_intents(diff, reference_profile="BCSS")

        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.intents[0].primitive, "stromal_desmoplasia")
        self.assertEqual(plan.intents[0].strength, "moderate")
        self.assertEqual(plan.unsupported_changes, ())

    def test_grade_only_change_without_supported_special_emits_warning_only(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "none", "grade_change": "upgrade"}
        )

        plan = plan_edit_intents(diff, reference_profile="IGNITE")

        self.assertEqual(plan.intents, ())
        self.assertEqual(plan.unsupported_changes[0].field, "tumor_change.grade_change")

    def test_panda_grade_upgrade_maps_to_gleason_special(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "none", "grade_change": "upgrade"}
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="PANDA",
            old_prompt="Prostate adenocarcinoma with Gleason pattern 3.",
            new_prompt="Prostate adenocarcinoma upgraded to Gleason pattern 4.",
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "gleason_upgrade_3to4")
        self.assertEqual(intents[0].reference_profile, "PANDA")

    def test_panda_pattern_5_maps_to_gleason_4to5_special(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "none", "grade_change": "upgrade"}
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="PANDA",
            old_prompt="Prostate adenocarcinoma with Gleason pattern 4.",
            new_prompt="Prostate adenocarcinoma with new Gleason pattern 5.",
        )

        self.assertEqual(intents[0].primitive, "gleason_upgrade_4to5")

    def test_glas_grade_upgrade_maps_to_grade_special(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "none", "grade_change": "upgrade"}
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="GlaS",
            old_prompt="Moderately differentiated colorectal carcinoma.",
            new_prompt="Poorly differentiated high grade colorectal carcinoma.",
        )

        self.assertEqual(intents[0].primitive, "grade_upgrade")

    def test_bcss_dcis_invasion_maps_to_special(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "none", "grade_change": "upgrade"}
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="BCSS",
            old_prompt="Breast lesion with DCIS.",
            new_prompt="DCIS becomes invasive carcinoma.",
        )

        self.assertEqual(intents[0].primitive, "dcis_invasion")

    def test_applicability_rejection_removes_intent_from_executable_list(self):
        diff = semantic_diff_with(
            lymphocyte_change={"infiltration": "increase", "degree": "moderate"}
        )
        old_mask = np.array(
            [
                [8, 8, 2],
                [8, 2, 2],
                [5, 5, 2],
            ],
            dtype=np.int64,
        )

        plan = plan_edit_intents(diff, reference_profile="PANDA", old_mask=old_mask)

        self.assertEqual(plan.intents, ())
        self.assertEqual(plan.items[0].status, "rejected_by_applicability")
        self.assertIn(
            "target_label_not_writable:Immune infiltrate",
            plan.items[0].reasons,
        )

    def test_applicability_degraded_plan_keeps_intent(self):
        diff = semantic_diff_with(necrosis_change={"action": "add", "extent": "focal"})
        old_mask = np.array(
            [
                [1, 1, 1, 2],
                [1, 1, 2, 2],
                [1, 2, 2, 2],
                [0, 2, 2, 2],
            ],
            dtype=np.int64,
        )

        plan = plan_edit_intents(diff, reference_profile="BCSS", old_mask=old_mask)

        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.items[0].status, "degraded_planned")
        self.assertIn("optional_label_absent_in_mask:Blood vessel", plan.items[0].warnings)


if __name__ == "__main__":
    unittest.main()
