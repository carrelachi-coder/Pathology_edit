import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from phase3_mask_edit.parser.api_parser import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    ApiParserConfig,
    parse_prompts_with_api,
)
from phase3_mask_edit.parser.instruction_parser import (
    InstructionParserConfig,
    parse_instruction_with_api,
)
from phase3_mask_edit.parser.qwen_local_parser import canonicalize_qwen_response
from phase3_mask_edit.parser.semantic_diff import (
    DEFAULT_SEMANTIC_DIFF,
    SEMANTIC_DIFF_SCHEMA_VERSION,
    SemanticDiffValidationError,
    extract_json_object,
    load_semantic_diff,
    normalize_semantic_diff,
    save_semantic_diff,
    semantic_diff_response_format,
    validate_semantic_diff,
)


class Phase3SemanticDiffSchemaTests(unittest.TestCase):
    def test_default_semantic_diff_validates(self):
        payload = validate_semantic_diff(DEFAULT_SEMANTIC_DIFF)

        self.assertEqual(payload["schema_version"], SEMANTIC_DIFF_SCHEMA_VERSION)
        self.assertEqual(payload["tumor_change"]["growth"], "none")
        self.assertEqual(payload["lymphocyte_change"]["location"], "unspecified")
        self.assertEqual(payload["transition_change"]["source_state"], "none")

    def test_missing_schema_version_is_rejected(self):
        payload = dict(DEFAULT_SEMANTIC_DIFF)
        payload.pop("schema_version")

        with self.assertRaisesRegex(SemanticDiffValidationError, "schema_version"):
            validate_semantic_diff(payload)

    def test_invalid_nested_value_is_rejected(self):
        payload = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
        payload["necrosis_change"]["action"] = "invent"

        with self.assertRaisesRegex(
            SemanticDiffValidationError, "necrosis_change.action"
        ):
            validate_semantic_diff(payload)

    def test_invalid_transition_pair_is_rejected(self):
        payload = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
        payload["transition_change"]["source_state"] = "normal_gland"
        payload["transition_change"]["target_state"] = "none"

        with self.assertRaisesRegex(
            SemanticDiffValidationError, "supported exact pair"
        ):
            validate_semantic_diff(payload)

    def test_parser_normalization_can_fill_missing_fields(self):
        payload = {
            "schema_version": SEMANTIC_DIFF_SCHEMA_VERSION,
            "necrosis_change": {"action": "add"},
        }

        normalized = normalize_semantic_diff(payload, fill_missing=True)

        self.assertEqual(normalized["necrosis_change"]["action"], "add")
        self.assertEqual(normalized["necrosis_change"]["extent"], "focal")
        self.assertEqual(normalized["tumor_change"]["growth"], "none")

    def test_strict_normalization_does_not_fill_missing_fields(self):
        payload = {
            "schema_version": SEMANTIC_DIFF_SCHEMA_VERSION,
            "necrosis_change": {"action": "add"},
        }

        with self.assertRaisesRegex(SemanticDiffValidationError, "tumor_change"):
            normalize_semantic_diff(payload, fill_missing=False)

    def test_extra_fields_are_preserved_for_audit(self):
        payload = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
        payload["evidence"] = {"raw": "focal necrosis"}
        payload["tumor_change"]["evidence"] = "no growth statement"

        normalized = validate_semantic_diff(payload)

        self.assertEqual(normalized["evidence"]["raw"], "focal necrosis")
        self.assertEqual(normalized["tumor_change"]["evidence"], "no growth statement")

    def test_extract_json_object_from_fenced_response(self):
        response = '```json\n{"schema_version": "0.2"}\n```'

        extracted = extract_json_object(response)

        self.assertEqual(extracted["schema_version"], SEMANTIC_DIFF_SCHEMA_VERSION)

    def test_legacy_v01_payload_is_migrated_to_v02(self):
        legacy = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
        legacy["schema_version"] = "0.1"
        legacy["lymphocyte_change"].pop("location")
        legacy.pop("transition_change")

        normalized = validate_semantic_diff(legacy)

        self.assertEqual(normalized["schema_version"], "0.2")
        self.assertEqual(normalized["lymphocyte_change"]["location"], "unspecified")
        self.assertEqual(normalized["transition_change"]["target_state"], "none")

    def test_response_format_is_strict_json_schema(self):
        response_format = semantic_diff_response_format()

        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertIn("transition_change", schema["required"])
        self.assertIn(
            "location",
            schema["properties"]["lymphocyte_change"]["required"],
        )
        self.assertEqual(
            len(schema["properties"]["transition_change"]["anyOf"]),
            10,
        )

    @mock.patch.dict("os.environ", {"TEST_API_KEY": "secret"})
    def test_both_api_parsers_send_strict_json_schema(self):
        response = {
            "choices": [{"message": {"content": json.dumps(DEFAULT_SEMANTIC_DIFF)}}]
        }
        captured = []

        def fake_post(payload, **kwargs):
            captured.append(payload)
            return response

        with mock.patch(
            "phase3_mask_edit.parser.api_parser._post_chat_completion",
            side_effect=fake_post,
        ):
            parse_prompts_with_api(
                "old report",
                "new report",
                config=ApiParserConfig(model="test", api_key_env="TEST_API_KEY"),
            )
        with mock.patch(
            "phase3_mask_edit.parser.instruction_parser._post_chat_completion",
            side_effect=fake_post,
        ):
            parse_instruction_with_api(
                "increase intratumoral immune cells",
                config=InstructionParserConfig(
                    model="test", api_key_env="TEST_API_KEY"
                ),
            )

        self.assertEqual(len(captured), 2)
        self.assertTrue(
            all(
                payload["response_format"]["type"] == "json_schema"
                for payload in captured
            )
        )
        self.assertTrue(
            all(
                payload["response_format"]["json_schema"]["strict"]
                for payload in captured
            )
        )

    @mock.patch.dict("os.environ", {"TEST_API_KEY": "secret"})
    def test_both_api_parsers_include_planner_repair_feedback(self):
        response = {
            "choices": [{"message": {"content": json.dumps(DEFAULT_SEMANTIC_DIFF)}}]
        }
        captured = []

        def fake_post(payload, **kwargs):
            captured.append(payload)
            return response

        feedback = {
            "status": "planner_no_executable_intents",
            "planner_items": [],
        }
        with mock.patch(
            "phase3_mask_edit.parser.api_parser._post_chat_completion",
            side_effect=fake_post,
        ):
            parse_prompts_with_api(
                "old report",
                "new report",
                config=ApiParserConfig(model="test", api_key_env="TEST_API_KEY"),
                repair_feedback=feedback,
                previous_semantic_diff=DEFAULT_SEMANTIC_DIFF,
            )
        with mock.patch(
            "phase3_mask_edit.parser.instruction_parser._post_chat_completion",
            side_effect=fake_post,
        ):
            parse_instruction_with_api(
                "increase intratumoral immune cells",
                config=InstructionParserConfig(
                    model="test", api_key_env="TEST_API_KEY"
                ),
                repair_feedback=feedback,
                previous_semantic_diff=DEFAULT_SEMANTIC_DIFF,
            )

        self.assertEqual(len(captured), 2)
        for payload in captured:
            user_content = payload["messages"][-1]["content"]
            self.assertIn("PREVIOUS SEMANTIC DIFF", user_content)
            self.assertIn("DOWNSTREAM PLANNER FEEDBACK", user_content)
            self.assertIn("planner_no_executable_intents", user_content)
            self.assertIn("Do not invent", user_content)

    def test_save_and_load_semantic_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semantic_diff.json"
            save_semantic_diff(DEFAULT_SEMANTIC_DIFF, path)

            loaded = load_semantic_diff(path)

        self.assertEqual(loaded, DEFAULT_SEMANTIC_DIFF)

    def test_qwen_response_canonicalization_adds_schema_version(self):
        response = json.dumps(
            {
                "tumor_change": {
                    "growth": "none",
                    "degree": "mild",
                    "grade_change": "none",
                },
                "lymphocyte_change": {
                    "infiltration": "increase",
                    "degree": "moderate",
                },
                "necrosis_change": {
                    "action": "add",
                    "extent": "focal",
                },
                "stroma_change": {
                    "density": "none",
                    "degree": "moderate",
                },
            }
        )

        normalized = canonicalize_qwen_response(response)

        self.assertEqual(normalized["schema_version"], SEMANTIC_DIFF_SCHEMA_VERSION)
        self.assertEqual(normalized["lymphocyte_change"]["infiltration"], "increase")

    def test_phase3_prompt_keeps_legacy_few_shot_coverage(self):
        self.assertIn("CRITICAL RULES", SYSTEM_PROMPT)
        self.assertIn("reciprocal backfill", SYSTEM_PROMPT)
        self.assertIn(
            "Replacement/backfill targets are not separate edits", SYSTEM_PROMPT
        )
        self.assertGreaterEqual(len(FEW_SHOT_EXAMPLES), 9)
        self.assertGreaterEqual(len(FEW_SHOT_EXAMPLES), 11)

        outputs = [example[2] for example in FEW_SHOT_EXAMPLES]
        self.assertTrue(
            any(output["tumor_change"]["growth"] == "decrease" for output in outputs)
        )
        self.assertTrue(
            any(output["necrosis_change"]["action"] == "decrease" for output in outputs)
        )
        self.assertTrue(
            any(output["stroma_change"]["density"] == "increase" for output in outputs)
        )
        self.assertTrue(
            all(
                output["schema_version"] == SEMANTIC_DIFF_SCHEMA_VERSION
                for output in outputs
            )
        )
        self.assertTrue(
            any(
                output["transition_change"]["source_state"] != "none"
                for output in outputs
            )
        )
        self.assertTrue(
            any(
                "replace it with stromal tissue" in example[1]
                and example[2]["lymphocyte_change"]["infiltration"] == "decrease"
                and example[2]["stroma_change"]["density"] == "none"
                for example in FEW_SHOT_EXAMPLES
            )
        )


if __name__ == "__main__":
    unittest.main()
