import json
import tempfile
import unittest
from pathlib import Path

from phase3_mask_edit.parser.semantic_diff import (
    DEFAULT_SEMANTIC_DIFF,
    SEMANTIC_DIFF_SCHEMA_VERSION,
    SemanticDiffValidationError,
    extract_json_object,
    load_semantic_diff,
    normalize_semantic_diff,
    save_semantic_diff,
    validate_semantic_diff,
)


class Phase3SemanticDiffSchemaTests(unittest.TestCase):
    def test_default_semantic_diff_validates(self):
        payload = validate_semantic_diff(DEFAULT_SEMANTIC_DIFF)

        self.assertEqual(payload["schema_version"], SEMANTIC_DIFF_SCHEMA_VERSION)
        self.assertEqual(payload["tumor_change"]["growth"], "none")

    def test_missing_schema_version_is_rejected(self):
        payload = dict(DEFAULT_SEMANTIC_DIFF)
        payload.pop("schema_version")

        with self.assertRaisesRegex(SemanticDiffValidationError, "schema_version"):
            validate_semantic_diff(payload)

    def test_invalid_nested_value_is_rejected(self):
        payload = json.loads(json.dumps(DEFAULT_SEMANTIC_DIFF))
        payload["necrosis_change"]["action"] = "invent"

        with self.assertRaisesRegex(SemanticDiffValidationError, "necrosis_change.action"):
            validate_semantic_diff(payload)

    def test_parser_normalization_can_fill_missing_fields(self):
        payload = {
            "schema_version": "0.1",
            "necrosis_change": {"action": "add"},
        }

        normalized = normalize_semantic_diff(payload, fill_missing=True)

        self.assertEqual(normalized["necrosis_change"]["action"], "add")
        self.assertEqual(normalized["necrosis_change"]["extent"], "focal")
        self.assertEqual(normalized["tumor_change"]["growth"], "none")

    def test_strict_normalization_does_not_fill_missing_fields(self):
        payload = {
            "schema_version": "0.1",
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
        response = "```json\n{\"schema_version\": \"0.1\"}\n```"

        extracted = extract_json_object(response)

        self.assertEqual(extracted["schema_version"], "0.1")

    def test_save_and_load_semantic_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semantic_diff.json"
            save_semantic_diff(DEFAULT_SEMANTIC_DIFF, path)

            loaded = load_semantic_diff(path)

        self.assertEqual(loaded, DEFAULT_SEMANTIC_DIFF)


if __name__ == "__main__":
    unittest.main()
