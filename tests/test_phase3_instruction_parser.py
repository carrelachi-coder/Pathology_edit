import unittest

from phase3_mask_edit.parser.instruction_parser import (
    INSTRUCTION_SYSTEM_PROMPT,
    InstructionParserError,
    parse_instruction_rule_based,
)
from phase3_mask_edit.rules.semantic_to_intent import semantic_diff_to_intents


class Phase3InstructionParserTests(unittest.TestCase):
    def test_english_tumor_more_moderate_maps_to_tumor_increase(self):
        semantic_diff = parse_instruction_rule_based("make the tumor moderately larger")

        self.assertEqual(semantic_diff["tumor_change"]["growth"], "increase")
        self.assertEqual(semantic_diff["tumor_change"]["degree"], "moderate")

        intents = semantic_diff_to_intents(semantic_diff, reference_profile="BCSS")
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "tumor_burden_increase")
        self.assertEqual(intents[0].strength, "moderate")

    def test_chinese_tumor_more_moderate_also_maps_to_tumor_increase(self):
        instruction = "\u5e0c\u671b\u80bf\u7624\u66f4\u591a\u4e00\u70b9\uff0c\u7a0b\u5ea6\u9002\u4e2d"
        semantic_diff = parse_instruction_rule_based(instruction)

        self.assertEqual(semantic_diff["tumor_change"]["growth"], "increase")
        self.assertEqual(semantic_diff["tumor_change"]["degree"], "moderate")

    def test_necrosis_decrease_maps_to_resolution(self):
        instruction = "\u574f\u6b7b\u51cf\u5c11\uff0c\u7a0b\u5ea6\u4e2d\u7b49"
        semantic_diff = parse_instruction_rule_based(instruction)

        intents = semantic_diff_to_intents(semantic_diff, reference_profile="BCSS")

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "necrosis_resolution")
        self.assertEqual(intents[0].strength, "moderate")

    def test_necrosis_larger_maps_to_appearance(self):
        semantic_diff = parse_instruction_rule_based(
            "make necrosis significantly larger"
        )

        self.assertEqual(semantic_diff["necrosis_change"]["action"], "increase")
        self.assertEqual(semantic_diff["necrosis_change"]["extent"], "extensive")

        intents = semantic_diff_to_intents(semantic_diff, reference_profile="BCSS")
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "necrosis_appearance")
        self.assertEqual(intents[0].strength, "significant")

    def test_chinese_necrosis_bigger_maps_to_appearance(self):
        instruction = "\u574f\u6b7b\u5927\u4e00\u70b9"
        semantic_diff = parse_instruction_rule_based(instruction)

        self.assertEqual(semantic_diff["necrosis_change"]["action"], "increase")
        self.assertEqual(semantic_diff["necrosis_change"]["extent"], "focal")

    def test_immune_decrease_stroma_replacement_does_not_parse_as_stroma_increase(self):
        semantic_diff = parse_instruction_rule_based(
            "decrease the lymphocytic immune infiltrate and replace it with stromal tissue"
        )

        self.assertEqual(
            semantic_diff["lymphocyte_change"]["infiltration"],
            "decrease",
        )
        self.assertEqual(semantic_diff["stroma_change"]["density"], "none")

    def test_necrosis_resolution_stroma_replacement_does_not_parse_as_stroma_increase(
        self,
    ):
        semantic_diff = parse_instruction_rule_based(
            "resolve most necrosis/debris and replace it with viable stroma; keep tumor burden unchanged"
        )

        self.assertIn(
            semantic_diff["necrosis_change"]["action"], {"decrease", "remove"}
        )
        self.assertEqual(semantic_diff["stroma_change"]["density"], "none")

    def test_tumor_decrease_stroma_replacement_does_not_parse_as_stroma_increase(self):
        semantic_diff = parse_instruction_rule_based(
            "reduce the tumor burden and replace the removed tumor with stromal tissue"
        )

        self.assertEqual(semantic_diff["tumor_change"]["growth"], "decrease")
        self.assertEqual(semantic_diff["stroma_change"]["density"], "none")

    def test_independent_desmoplasia_still_parses_as_stroma_increase(self):
        semantic_diff = parse_instruction_rule_based(
            "increase the desmoplastic stromal reaction around tumor nests"
        )

        self.assertEqual(semantic_diff["stroma_change"]["density"], "increase")

    def test_rule_based_intratumoral_location_is_explicit(self):
        semantic_diff = parse_instruction_rule_based(
            "add moderate immune infiltration inside tumor"
        )

        self.assertEqual(
            semantic_diff["lymphocyte_change"]["location"],
            "intratumoral",
        )

    def test_rule_based_gleason_transition_is_explicit(self):
        semantic_diff = parse_instruction_rule_based(
            "upgrade prostate tumor from Gleason pattern 3 to pattern 4"
        )

        self.assertEqual(
            semantic_diff["transition_change"],
            {
                "source_state": "gleason_pattern_3",
                "target_state": "gleason_pattern_4",
                "degree": "moderate",
            },
        )
        intents = semantic_diff_to_intents(semantic_diff, reference_profile="PANDA")
        self.assertEqual([item.primitive for item in intents], ["gleason_upgrade_3to4"])

    def test_api_instruction_prompt_mentions_special_fine_edits_without_chinese(self):
        self.assertFalse(any(ord(char) > 127 for char in INSTRUCTION_SYSTEM_PROMPT))
        self.assertIn(
            "gleason_pattern_3 -> gleason_pattern_4", INSTRUCTION_SYSTEM_PROMPT
        )
        self.assertIn("normal_gland -> adenomatous_gland", INSTRUCTION_SYSTEM_PROMPT)
        self.assertIn("lymphocyte_change.location", INSTRUCTION_SYSTEM_PROMPT)
        self.assertIn(
            "Replacement/backfill targets are not separate edits",
            INSTRUCTION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "An instruction may request two independent edits",
            INSTRUCTION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "Preserve/keep stable/leave unchanged clauses are constraints",
            INSTRUCTION_SYSTEM_PROMPT,
        )

    def test_unrecognized_instruction_raises(self):
        with self.assertRaises(InstructionParserError):
            parse_instruction_rule_based("make it look nicer")


if __name__ == "__main__":
    unittest.main()
