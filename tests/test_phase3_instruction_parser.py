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
        semantic_diff = parse_instruction_rule_based("make necrosis significantly larger")

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

    def test_api_instruction_prompt_mentions_special_fine_edits_without_chinese(self):
        self.assertFalse(any(ord(char) > 127 for char in INSTRUCTION_SYSTEM_PROMPT))
        self.assertIn("PANDA special edits", INSTRUCTION_SYSTEM_PROMPT)
        self.assertIn("GlaS special edits", INSTRUCTION_SYSTEM_PROMPT)
        self.assertIn("Gleason pattern 3 to 4", INSTRUCTION_SYSTEM_PROMPT)
        self.assertIn("normal gland becoming adenomatous", INSTRUCTION_SYSTEM_PROMPT)

    def test_unrecognized_instruction_raises(self):
        with self.assertRaises(InstructionParserError):
            parse_instruction_rule_based("make it look nicer")


if __name__ == "__main__":
    unittest.main()
