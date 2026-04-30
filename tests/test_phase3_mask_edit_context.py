import unittest

import numpy as np

from phase3_mask_edit.core.context import MaskEditContext, MaskEditContextError
from phase3_mask_edit.core.labels import MaskProfileSchema


class Phase3MaskEditContextTests(unittest.TestCase):
    def test_context_summarizes_present_labels_and_area_fractions(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        mask = np.array(
            [
                [0, 1, 1, 2],
                [0, 1, 2, 2],
                [4, 4, 2, 3],
                [7, 7, 3, 3],
            ],
            dtype=np.int64,
        )

        context = MaskEditContext.from_mask(mask, schema)

        self.assertEqual(context.reference_profile, "BCSS")
        self.assertEqual(context.mask_shape, (4, 4))
        self.assertEqual(
            context.present_labels,
            frozenset({"Tumor", "Stroma", "Necrosis", "Immune infiltrate", "Other tissue"}),
        )
        self.assertAlmostEqual(context.label_area_fractions["Tumor"], 3 / 16)
        self.assertAlmostEqual(context.label_area_fractions["Stroma"], 4 / 16)
        self.assertAlmostEqual(context.label_area_fractions["Necrosis"], 3 / 16)
        self.assertAlmostEqual(context.fine_id_area_fractions[1], 3 / 16)

    def test_context_counts_four_connected_components_per_label(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        mask = np.array(
            [
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [2, 2, 0, 2],
                [0, 0, 0, 2],
            ],
            dtype=np.int64,
        )

        context = MaskEditContext.from_mask(mask, schema)

        self.assertEqual(context.component_counts["Tumor"], 2)
        self.assertEqual(context.component_counts["Stroma"], 2)

    def test_context_builds_symmetric_label_adjacency(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        mask = np.array(
            [
                [1, 1, 2],
                [1, 4, 2],
                [7, 4, 3],
            ],
            dtype=np.int64,
        )

        context = MaskEditContext.from_mask(mask, schema)

        self.assertIn("Stroma", context.adjacency["Tumor"])
        self.assertIn("Tumor", context.adjacency["Stroma"])
        self.assertIn("Immune infiltrate", context.adjacency["Tumor"])
        self.assertIn("Necrosis", context.adjacency["Immune infiltrate"])

    def test_orca_context_does_not_invent_stroma_from_other_tissue(self):
        schema = MaskProfileSchema.from_reference_profile("ORCA")
        mask = np.array(
            [
                [1, 1, 7],
                [1, 7, 7],
                [0, 7, 7],
            ],
            dtype=np.int64,
        )

        context = MaskEditContext.from_mask(mask, schema)

        self.assertEqual(context.present_labels, frozenset({"Tumor", "Other tissue"}))
        self.assertNotIn("Stroma", context.present_labels)
        self.assertIn("Other tissue", context.semantic_warnings)

    def test_unknown_non_skip_fine_id_is_mapped_to_other_tissue(self):
        schema = MaskProfileSchema.from_reference_profile("ORCA")
        mask = np.array(
            [
                [1, 2],
                [7, 0],
            ],
            dtype=np.int64,
        )

        context = MaskEditContext.from_mask(mask, schema)

        self.assertEqual(context.present_labels, frozenset({"Tumor", "Other tissue"}))
        self.assertAlmostEqual(context.label_area_fractions["Other tissue"], 2 / 4)
        self.assertEqual(context.normalized_mask.tolist(), [[1, 7], [7, 0]])
        self.assertIn("remapped_unknown_fine_ids_to_other_tissue:2", context.risk_flags)

    def test_context_requires_2d_mask(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")

        with self.assertRaisesRegex(MaskEditContextError, "2D"):
            MaskEditContext.from_mask(np.zeros((2, 2, 3), dtype=np.int64), schema)


if __name__ == "__main__":
    unittest.main()
