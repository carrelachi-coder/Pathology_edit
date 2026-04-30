import json
import unittest
from pathlib import Path

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.intent import (
    EditIntent,
    IntentValidationError,
    resolve_reference_profile,
    validate_intent_against_recipe,
)


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


class Phase3EditIntentTests(unittest.TestCase):
    def test_minimal_payload_builds_intent_with_default_strength(self):
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "organ": "breast",
            }
        )

        self.assertEqual(intent.primitive, "tumor_burden_increase")
        self.assertEqual(intent.organ, "breast")
        self.assertIsNone(intent.reference_profile)
        self.assertEqual(intent.strength, "moderate")
        self.assertEqual(intent.source_labels, ())
        self.assertIsNone(intent.target_label)

    def test_reference_profile_is_optional_before_profile_resolution(self):
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
            }
        )

        self.assertIsNone(intent.reference_profile)
        self.assertIsNone(intent.organ)

    def test_dataset_field_is_rejected_to_avoid_provenance_confusion(self):
        with self.assertRaisesRegex(IntentValidationError, "reference_profile"):
            EditIntent.from_mapping(
                {
                    "primitive": "tumor_burden_increase",
                    "dataset": "BCSS",
                }
            )

    def test_semantic_domain_fields_are_preserved(self):
        intent = EditIntent.from_mapping(
            {
                "primitive": "stromal_desmoplasia",
                "organ": "pancreas",
                "cancer_type": "pancreatic ductal adenocarcinoma",
                "site": "primary tumor",
                "diagnosis": "desmoplastic adenocarcinoma",
            }
        )

        self.assertEqual(intent.organ, "pancreas")
        self.assertEqual(intent.cancer_type, "pancreatic ductal adenocarcinoma")
        self.assertEqual(intent.site, "primary tumor")
        self.assertEqual(intent.diagnosis, "desmoplastic adenocarcinoma")

    def test_explicit_reference_profile_is_preserved_and_resolved(self):
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "organ": "breast",
                "reference_profile": "BCSS",
            }
        )

        self.assertEqual(intent.reference_profile, "BCSS")
        self.assertEqual(resolve_reference_profile(intent), "BCSS")

    def test_organ_can_resolve_default_reference_profile(self):
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "organ": "breast",
            }
        )

        self.assertEqual(resolve_reference_profile(intent), "BCSS")

    def test_unrecognized_domain_cannot_resolve_reference_profile(self):
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "organ": "pancreas",
            }
        )

        with self.assertRaisesRegex(IntentValidationError, "reference_profile"):
            resolve_reference_profile(intent)

    def test_bucket_alias_is_normalized_to_strength(self):
        intent = EditIntent.from_mapping(
            {
                "primitive": "necrosis_appearance",
                "bucket": "significant",
            }
        )

        self.assertEqual(intent.strength, "significant")
        self.assertNotIn("bucket", intent.to_metadata())
        self.assertEqual(intent.to_metadata()["strength"], "significant")

    def test_invalid_strength_raises(self):
        with self.assertRaisesRegex(IntentValidationError, "strength"):
            EditIntent.from_mapping(
                {
                    "primitive": "tumor_burden_increase",
                    "strength": "huge",
                }
            )

    def test_invalid_target_change_fraction_raises(self):
        with self.assertRaisesRegex(IntentValidationError, "target_change_fraction"):
            EditIntent.from_mapping(
                {
                    "primitive": "tumor_burden_increase",
                    "target_change_fraction": 1.2,
                }
            )

    def test_unknown_primitive_rejected_against_recipe(self):
        recipe = load_recipe(GENERIC_RECIPE)
        intent = EditIntent.from_mapping(
            {
                "primitive": "not_a_primitive",
            }
        )

        with self.assertRaisesRegex(IntentValidationError, "not_a_primitive"):
            validate_intent_against_recipe(intent, recipe)

    def test_unknown_tissue_label_rejected_against_recipe(self):
        recipe = load_recipe(GENERIC_RECIPE)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "source_labels": ["Mystery tissue"],
            }
        )

        with self.assertRaisesRegex(IntentValidationError, "Mystery tissue"):
            validate_intent_against_recipe(intent, recipe)

    def test_strength_must_exist_for_primitive_parameter_ranges(self):
        recipe = load_recipe(GENERIC_RECIPE)
        intent = EditIntent.from_mapping(
            {
                "primitive": "boundary_infiltration",
                "strength": "xlarge_deid",
            }
        )

        with self.assertRaisesRegex(IntentValidationError, "xlarge_deid"):
            validate_intent_against_recipe(intent, recipe)

    def test_to_metadata_is_json_serializable(self):
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "organ": "breast",
                "cancer_type": "breast cancer",
                "reference_profile": "BCSS",
                "site": "primary tumor",
                "diagnosis": "invasive carcinoma",
                "strength": "significant",
                "target_change_fraction": 0.28,
                "source_labels": ["Stroma"],
                "target_label": "Tumor",
                "region_hint": {"relation": "tumor_boundary"},
                "parameters": {"allow_islands": False},
                "preserve_labels": ["Normal epithelium"],
                "forbidden_labels": ["Background"],
                "old_prompt": "low tumor burden",
                "new_prompt": "higher tumor burden with invasive margin",
                "prompt_diff": {"tumor_burden": "increase"},
                "seed": 42,
            }
        )

        metadata = intent.to_metadata()
        encoded = json.dumps(metadata, sort_keys=True)

        self.assertIn("tumor_burden_increase", encoded)
        self.assertEqual(metadata["organ"], "breast")
        self.assertEqual(metadata["reference_profile"], "BCSS")
        self.assertNotIn("dataset", metadata)
        self.assertEqual(metadata["source_labels"], ["Stroma"])
        self.assertEqual(metadata["preserve_labels"], ["Normal epithelium"])


if __name__ == "__main__":
    unittest.main()
