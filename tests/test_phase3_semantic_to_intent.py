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

    def test_immune_decrease_with_stroma_replacement_plans_desmoplasia_as_fallback(
        self,
    ):
        diff = semantic_diff_with(
            lymphocyte_change={
                "infiltration": "decrease",
                "degree": "significant",
            },
            stroma_change={"density": "increase", "degree": "significant"},
        )

        plan = plan_edit_intents(
            diff,
            reference_profile="BCSS",
            old_prompt=(
                "H&E stained breast cancer histopathology with dense lymphocytic "
                "immune infiltrate and sparse stroma."
            ),
            new_prompt=(
                "Decrease the lymphocytic immune infiltrate substantially and "
                "replace it with stromal tissue, keeping tumor burden stable."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in plan.intents],
            ["immune_infiltration_decrease"],
        )
        self.assertEqual(
            [item.primitive for item in plan.items],
            [
                "immune_infiltration_decrease",
                "stromal_desmoplasia",
            ],
        )
        self.assertEqual(plan.items[0].role, "primary")
        self.assertEqual(plan.items[1].role, "fallback")
        self.assertEqual(plan.items[1].fallback_for, "immune_infiltration_decrease")
        self.assertEqual(plan.items[1].status, "fallback_planned")
        self.assertEqual(
            plan.items[0].execution_group,
            "immune_decrease_stroma_replacement",
        )

    def test_necrosis_resolution_with_stroma_replacement_plans_desmoplasia_as_fallback(
        self,
    ):
        diff = semantic_diff_with(
            necrosis_change={
                "action": "remove",
                "extent": "extensive",
            },
            stroma_change={"density": "increase", "degree": "significant"},
        )

        plan = plan_edit_intents(
            diff,
            reference_profile="BCSS",
            new_prompt=(
                "Resolve most necrosis/debris and replace it with viable stroma; "
                "keep tumor burden unchanged."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in plan.intents],
            ["necrosis_resolution"],
        )
        self.assertEqual(
            [item.primitive for item in plan.items],
            ["necrosis_resolution", "stromal_desmoplasia"],
        )
        self.assertEqual(plan.items[0].role, "primary")
        self.assertEqual(plan.items[1].role, "fallback")
        self.assertEqual(plan.items[1].fallback_for, "necrosis_resolution")
        self.assertEqual(plan.items[1].status, "fallback_planned")
        self.assertEqual(
            plan.items[0].execution_group,
            "necrosis_resolution_stroma_replacement",
        )

    def test_tumor_decrease_with_stroma_replacement_plans_desmoplasia_as_fallback(self):
        diff = semantic_diff_with(
            tumor_change={
                "growth": "decrease",
                "degree": "moderate",
            },
            stroma_change={"density": "increase", "degree": "moderate"},
        )

        plan = plan_edit_intents(
            diff,
            reference_profile="BCSS",
            new_prompt=(
                "Reduce the tumor burden and replace the removed tumor with "
                "stromal tissue."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in plan.intents],
            ["tumor_burden_decrease"],
        )
        self.assertEqual(
            [item.primitive for item in plan.items],
            ["tumor_burden_decrease", "stromal_desmoplasia"],
        )
        self.assertEqual(plan.items[0].role, "primary")
        self.assertEqual(plan.items[1].role, "fallback")
        self.assertEqual(plan.items[1].fallback_for, "tumor_burden_decrease")
        self.assertEqual(plan.items[1].status, "fallback_planned")
        self.assertEqual(
            plan.items[0].execution_group,
            "tumor_decrease_stroma_replacement",
        )

    def test_contextual_stroma_after_necrosis_resolution_is_fallback(self):
        diff = semantic_diff_with(
            necrosis_change={"action": "decrease", "extent": "focal"},
            stroma_change={"density": "increase", "degree": "moderate"},
        )

        plan = plan_edit_intents(
            diff,
            reference_profile="BCSS",
            old_prompt="A large necrotic focus leaves scant viable stroma.",
            new_prompt=(
                "Minimal necrotic debris remains and viable collagenous stroma "
                "predominates centrally."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in plan.intents],
            ["necrosis_resolution"],
        )
        self.assertEqual(plan.items[1].role, "fallback")

    def test_implicit_stroma_backfill_remains_a_fallback_when_parser_suppresses_it(self):
        diff = semantic_diff_with(
            necrosis_change={"action": "decrease", "extent": "extensive"},
            stroma_change={"density": "none", "degree": "moderate"},
        )

        plan = plan_edit_intents(
            diff,
            reference_profile="BCSS",
            new_prompt=(
                "Markedly reduce necrotic tissue and restore viable stromal "
                "tissue only as backfill for the vacated area."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in plan.intents],
            ["necrosis_resolution"],
        )
        self.assertEqual(
            [item.primitive for item in plan.items],
            ["necrosis_resolution", "stromal_desmoplasia"],
        )
        self.assertEqual(plan.items[1].role, "fallback")
        self.assertEqual(plan.items[1].fallback_for, "necrosis_resolution")

    def test_independent_desmoplasia_remains_separate_after_immune_decrease(self):
        diff = semantic_diff_with(
            lymphocyte_change={"infiltration": "decrease", "degree": "moderate"},
            stroma_change={"density": "increase", "degree": "moderate"},
        )

        plan = plan_edit_intents(
            diff,
            reference_profile="BCSS",
            new_prompt=(
                "Decrease immune infiltrate and also increase the desmoplastic "
                "stromal reaction around tumor nests."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in plan.intents],
            ["immune_infiltration_decrease", "stromal_desmoplasia"],
        )
        self.assertTrue(all(item.role == "primary" for item in plan.items))

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

    def test_contextual_immune_adjective_is_not_a_second_tumor_edit(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "increase", "degree": "significant"},
            lymphocyte_change={
                "infiltration": "increase",
                "degree": "moderate",
                "location": "peritumoral",
            },
        )

        plan = plan_edit_intents(
            diff,
            reference_profile="BCSS",
            old_prompt=(
                "Sparse tumor nests are present with scattered immune infiltrate."
            ),
            new_prompt=(
                "Prominent tumor nests occupy the compartment. A conspicuous "
                "immune infiltrate is adjacent to tumor."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in plan.intents],
            ["tumor_burden_increase"],
        )
        self.assertEqual(
            plan.unsupported_changes[-1].field,
            "lymphocyte_change.infiltration",
        )

    def test_explicit_independent_immune_action_remains_a_second_tumor_edit(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "increase", "degree": "moderate"},
            lymphocyte_change={
                "infiltration": "increase",
                "degree": "moderate",
                "location": "intratumoral",
            },
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="BCSS",
            new_prompt="Increase tumor burden and increase immune cells within tumor.",
        )

        self.assertEqual(
            [intent.primitive for intent in intents],
            ["tumor_burden_increase", "intratumoral_immune_infiltration"],
        )

    def test_contextual_atypia_is_not_a_second_tumor_extent_edit(self):
        diff = semantic_diff_with(
            tumor_change={
                "growth": "decrease",
                "degree": "moderate",
                "grade_change": "downgrade",
            }
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="GlaS",
            old_prompt="Prominent tumor nests have moderate nuclear atypia.",
            new_prompt="Sparse tumor nests have mild nuclear atypia.",
        )

        self.assertEqual(
            [intent.primitive for intent in intents],
            ["tumor_burden_decrease"],
        )

    def test_necrosis_replacement_is_not_a_second_tumor_decrease(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "decrease", "degree": "significant"},
            necrosis_change={"action": "add", "extent": "extensive"},
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="IGNITE",
            new_prompt=(
                "A central necrotic area is present. Necrotic debris replaces viable "
                "tumor cells in the central compartment."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in intents],
            ["necrosis_appearance"],
        )

    def test_primary_stroma_decrease_overrides_contextual_epithelium_growth(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "increase", "degree": "moderate"}
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="PANDA",
            old_prompt=(
                "The central stromal compartment contains abundant fibrous stroma "
                "that is dense and well-developed."
            ),
            new_prompt=(
                "The central stromal compartment contains scant fibrous stroma. "
                "Epithelial elements predominate with limited stromal tissue."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in intents],
            ["stroma_decrease"],
        )

    def test_primary_stroma_increase_overrides_contextual_immune_adjective(self):
        diff = semantic_diff_with(
            lymphocyte_change={
                "infiltration": "increase",
                "degree": "mild",
                "location": "unspecified",
            }
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="IGNITE",
            old_prompt="The central compartment contains scant fibrous stroma.",
            new_prompt=(
                "The central compartment shows mild focal fibrous stroma. Subtle "
                "collagen deposition is present with scattered immune infiltrate."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in intents],
            ["stromal_desmoplasia"],
        )

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

    def test_tumor_growth_defers_secondary_immune_and_necrosis_decrease(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "increase", "degree": "significant"},
            lymphocyte_change={"infiltration": "decrease", "degree": "moderate"},
            necrosis_change={"action": "decrease", "extent": "focal"},
        )

        plan = plan_edit_intents(diff, reference_profile="BCSS")

        self.assertEqual(
            [intent.primitive for intent in plan.intents],
            ["tumor_burden_increase"],
        )
        self.assertEqual(
            [warning.field for warning in plan.unsupported_changes],
            ["necrosis_change.action", "lymphocyte_change.infiltration"],
        )

    def test_tumor_growth_on_tumor_necrosis_only_mask_uses_necrosis_resolution(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "increase", "degree": "moderate"}
        )
        old_mask = np.array(
            [
                [1, 1, 3],
                [1, 3, 3],
                [3, 3, 3],
            ],
            dtype=np.int64,
        )

        plan = plan_edit_intents(diff, reference_profile="BCSS", old_mask=old_mask)

        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.intents[0].primitive, "necrosis_resolution")
        self.assertEqual(plan.intents[0].strength, "moderate")
        self.assertEqual(plan.items[0].status, "degraded_planned")
        self.assertIn("optional_label_absent_in_mask:Stroma", plan.items[0].warnings)

    def test_tumor_decrease_on_tumor_necrosis_only_mask_uses_necrosis_appearance(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "decrease", "degree": "moderate"}
        )
        old_mask = np.array(
            [
                [1, 1, 3],
                [1, 3, 3],
                [3, 3, 3],
            ],
            dtype=np.int64,
        )

        plan = plan_edit_intents(diff, reference_profile="BCSS", old_mask=old_mask)

        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.intents[0].primitive, "necrosis_appearance")
        self.assertEqual(plan.intents[0].strength, "moderate")
        self.assertEqual(plan.items[0].status, "degraded_planned")
        self.assertIn("optional_label_absent_in_mask:Stroma", plan.items[0].warnings)

    def test_stroma_density_increase_maps_to_desmoplasia(self):
        diff = semantic_diff_with(
            stroma_change={"density": "increase", "degree": "moderate"}
        )

        plan = plan_edit_intents(diff, reference_profile="BCSS")

        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.intents[0].primitive, "stromal_desmoplasia")
        self.assertEqual(plan.intents[0].strength, "moderate")
        self.assertEqual(plan.unsupported_changes, ())

    def test_intratumoral_immune_text_selects_intratumoral_primitive(self):
        diff = semantic_diff_with(
            lymphocyte_change={"infiltration": "increase", "degree": "significant"}
        )
        old_mask = np.ones((8, 8), dtype=np.int64)

        plan = plan_edit_intents(
            diff,
            reference_profile="BCSS",
            old_mask=old_mask,
            new_prompt="Add significant intratumoral immune infiltrate inside tumor.",
        )

        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(
            plan.intents[0].primitive,
            "intratumoral_immune_infiltration",
        )

    def test_intratumoral_schema_location_selects_primitive_without_text_hint(self):
        diff = semantic_diff_with(
            lymphocyte_change={
                "infiltration": "increase",
                "degree": "moderate",
                "location": "intratumoral",
            }
        )

        intents = semantic_diff_to_intents(diff, reference_profile="BCSS")

        self.assertEqual(len(intents), 1)
        self.assertEqual(
            intents[0].primitive,
            "intratumoral_immune_infiltration",
        )

    def test_explicit_fine_transition_pairs_map_without_prompt_text(self):
        cases = (
            ("PANDA", "benign_epithelium", "gleason_pattern_3", "benign_to_gleason3"),
            ("PANDA", "benign_epithelium", "stromal_tissue", "benign_atrophy"),
            ("PANDA", "gleason_pattern_3", "gleason_pattern_4", "gleason_upgrade_3to4"),
            ("PANDA", "gleason_pattern_4", "gleason_pattern_5", "gleason_upgrade_4to5"),
            (
                "PANDA",
                "gleason_pattern_4",
                "gleason_pattern_3",
                "gleason_downgrade_4to3",
            ),
            ("GlaS", "normal_gland", "adenomatous_gland", "normal_to_adenomatous"),
            (
                "GlaS",
                "adenomatous_gland",
                "moderately_differentiated_carcinoma",
                "adenoma_to_carcinoma",
            ),
            (
                "GlaS",
                "moderately_differentiated_carcinoma",
                "poorly_differentiated_carcinoma",
                "grade_upgrade",
            ),
            (
                "GlaS",
                "poorly_differentiated_carcinoma",
                "moderately_differentiated_carcinoma",
                "treatment_dedifferentiation",
            ),
        )

        for profile, source, target, primitive in cases:
            with self.subTest(primitive=primitive):
                diff = semantic_diff_with(
                    transition_change={
                        "source_state": source,
                        "target_state": target,
                        "degree": "moderate",
                    }
                )
                intents = semantic_diff_to_intents(
                    diff,
                    reference_profile=profile,
                )

                self.assertEqual([item.primitive for item in intents], [primitive])

    def test_explicit_transition_suppresses_generic_growth(self):
        diff = semantic_diff_with(
            tumor_change={
                "growth": "increase",
                "degree": "moderate",
                "grade_change": "upgrade",
            },
            transition_change={
                "source_state": "adenomatous_gland",
                "target_state": "moderately_differentiated_carcinoma",
                "degree": "moderate",
            },
        )

        plan = plan_edit_intents(
            diff,
            reference_profile="GlaS",
            old_prompt="Adenomatous glands are present.",
            new_prompt="Moderately differentiated carcinoma is present.",
        )

        self.assertEqual(
            [intent.primitive for intent in plan.intents],
            ["adenoma_to_carcinoma"],
        )
        self.assertEqual(plan.unsupported_changes[0].field, "tumor_change.growth")

    def test_unsupported_transition_evidence_does_not_hide_tumor_growth(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "increase", "degree": "significant"},
            transition_change={
                "source_state": "normal_gland",
                "target_state": "adenomatous_gland",
                "degree": "moderate",
            },
        )

        plan = plan_edit_intents(
            diff,
            reference_profile="GlaS",
            old_prompt="Sparse tumor nests are present in stroma.",
            new_prompt="Abundant malignant tumor nests occupy the compartment.",
        )

        self.assertEqual(
            [intent.primitive for intent in plan.intents],
            ["tumor_burden_increase"],
        )
        self.assertEqual(plan.unsupported_changes[0].field, "transition_change")

    def test_transition_ignores_contextual_immune_adjective(self):
        diff = semantic_diff_with(
            lymphocyte_change={
                "infiltration": "increase",
                "degree": "mild",
                "location": "stromal",
            },
            transition_change={
                "source_state": "poorly_differentiated_carcinoma",
                "target_state": "moderately_differentiated_carcinoma",
                "degree": "mild",
            },
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="GlaS",
            old_prompt="Poorly differentiated carcinoma with scant stroma.",
            new_prompt=(
                "Moderately differentiated carcinoma is present. The surrounding "
                "stroma contains sparse lymphocytes."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in intents],
            ["treatment_dedifferentiation"],
        )

    def test_transition_ignores_contextual_desmoplastic_response(self):
        diff = semantic_diff_with(
            stroma_change={"density": "increase", "degree": "mild"},
            transition_change={
                "source_state": "benign_epithelium",
                "target_state": "gleason_pattern_3",
                "degree": "moderate",
            },
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="PANDA",
            old_prompt="Benign prostatic epithelium is present.",
            new_prompt=(
                "Gleason pattern 3 malignant glands are present. The surrounding "
                "stroma contains a mild desmoplastic response."
            ),
        )

        self.assertEqual(
            [intent.primitive for intent in intents],
            ["benign_to_gleason3"],
        )

    def test_panda_benign_atrophy_is_inferred_from_transition_text(self):
        old_mask = np.array([[5, 5, 2], [5, 2, 2], [0, 2, 2]], dtype=np.int64)

        plan = plan_edit_intents(
            DEFAULT_SEMANTIC_DIFF,
            reference_profile="PANDA",
            old_mask=old_mask,
            old_prompt="Normal prostate glandular epithelium is present.",
            new_prompt="Replace mild normal prostate epithelium with stromal tissue.",
        )

        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.intents[0].primitive, "benign_atrophy")
        self.assertEqual(plan.intents[0].strength, "mild")

    def test_panda_normal_prostate_epithelium_maps_to_benign_to_gleason3(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "none", "grade_change": "upgrade"}
        )

        plan = plan_edit_intents(
            diff,
            reference_profile="PANDA",
            new_prompt=(
                "Convert normal prostate epithelium into Gleason pattern 3 tumor glands."
            ),
        )

        self.assertEqual(len(plan.intents), 1)
        self.assertEqual(plan.intents[0].primitive, "benign_to_gleason3")

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

    def test_instruction_style_panda_gleason_3to4_maps_without_old_prompt(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "none", "grade_change": "upgrade"}
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="PANDA",
            new_prompt="upgrade prostate tumor from Gleason pattern 3 to pattern 4",
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "gleason_upgrade_3to4")

    def test_instruction_style_glas_normal_to_adenomatous_maps_without_old_prompt(self):
        diff = semantic_diff_with(
            tumor_change={"growth": "none", "grade_change": "upgrade"}
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="GlaS",
            new_prompt="convert normal colonic glands into adenomatous glands",
        )

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "normal_to_adenomatous")

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

    def test_glas_growth_and_grade_upgrade_plans_both_area_and_fine_transition(self):
        diff = semantic_diff_with(
            tumor_change={
                "growth": "increase",
                "degree": "moderate",
                "grade_change": "upgrade",
            }
        )

        intents = semantic_diff_to_intents(
            diff,
            reference_profile="GlaS",
            old_prompt="Adenomatous low-grade malignant glands.",
            new_prompt="Higher-grade colorectal adenocarcinoma appearance.",
        )

        self.assertEqual(
            [intent.primitive for intent in intents],
            ["tumor_burden_increase", "adenoma_to_carcinoma"],
        )

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
        self.assertIn(
            "optional_label_absent_in_mask:Blood vessel", plan.items[0].warnings
        )


if __name__ == "__main__":
    unittest.main()
