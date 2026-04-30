import unittest

from phase3_mask_edit.core.labels import (
    MaskProfileSchema,
    MaskProfileSchemaError,
)


class Phase3MaskProfileSchemaTests(unittest.TestCase):
    def test_all_reference_profiles_expose_expected_unified_labels(self):
        expected_labels = {
            "BCSS": {
                "Tumor",
                "Stroma",
                "Necrosis",
                "Immune infiltrate",
                "Normal epithelium",
                "Blood vessel",
                "Other tissue",
            },
            "PANDA": {"Tumor", "Stroma", "Normal epithelium"},
            "GlaS": {"Tumor", "Stroma", "Normal epithelium"},
            "IGNITE": {
                "Tumor",
                "Stroma",
                "Necrosis",
                "Immune infiltrate",
                "Normal epithelium",
                "Blood vessel",
                "Other tissue",
            },
            "PUMA": {
                "Tumor",
                "Stroma",
                "Necrosis",
                "Normal epithelium",
                "Blood vessel",
            },
            "ORCA": {"Tumor", "Other tissue"},
        }

        for profile, labels in expected_labels.items():
            with self.subTest(profile=profile):
                schema = MaskProfileSchema.from_reference_profile(profile)
                self.assertEqual(schema.readable_labels, labels)
                self.assertEqual(schema.writable_labels, labels)
                self.assertTrue(schema.resolve_fine_ids("Tumor"))

    def test_bcss_profile_exposes_rich_breast_tissue_labels(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")

        self.assertEqual(schema.reference_profile, "BCSS")
        self.assertEqual(schema.resolve_fine_ids("Tumor"), (1, 14, 15))
        self.assertEqual(schema.resolve_fine_ids("Stroma"), (2,))
        self.assertEqual(schema.resolve_fine_ids("Necrosis"), (3,))
        self.assertEqual(schema.resolve_fine_ids("Immune infiltrate"), (4,))
        self.assertEqual(schema.resolve_fine_ids("Normal epithelium"), (5,))
        self.assertEqual(schema.resolve_fine_ids("Blood vessel"), (6,))
        self.assertEqual(schema.resolve_fine_ids("Other tissue"), (7,))

        self.assertTrue(
            {
                "Tumor",
                "Stroma",
                "Necrosis",
                "Immune infiltrate",
                "Normal epithelium",
                "Blood vessel",
                "Other tissue",
            }.issubset(schema.readable_labels)
        )
        self.assertEqual(schema.choose_default_backfill_label(), "Stroma")

    def test_orca_profile_only_exposes_tumor_and_coarse_other_tissue(self):
        schema = MaskProfileSchema.from_reference_profile("ORCA")

        self.assertEqual(schema.reference_profile, "ORCA")
        self.assertEqual(schema.resolve_fine_ids("Tumor"), (1,))
        self.assertEqual(schema.resolve_fine_ids("Other tissue"), (7,))
        self.assertNotIn("Stroma", schema.readable_labels)
        self.assertNotIn("Necrosis", schema.readable_labels)
        self.assertNotIn("Immune infiltrate", schema.readable_labels)
        self.assertEqual(schema.choose_default_backfill_label(), "Other tissue")
        self.assertIn("Other tissue", schema.semantic_warnings)

    def test_unknown_reference_profile_raises_clear_error(self):
        with self.assertRaisesRegex(MaskProfileSchemaError, "Unknown reference_profile"):
            MaskProfileSchema.from_reference_profile("NOT_A_PROFILE")

    def test_unknown_label_resolution_raises_clear_error(self):
        schema = MaskProfileSchema.from_reference_profile("ORCA")

        with self.assertRaisesRegex(MaskProfileSchemaError, "Stroma"):
            schema.resolve_fine_ids("Stroma")

    def test_choose_default_backfill_can_exclude_labels(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")

        self.assertEqual(
            schema.choose_default_backfill_label(exclude_labels=("Stroma",)),
            "Other tissue",
        )


if __name__ == "__main__":
    unittest.main()
