import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit.core.candidates import build_candidate_mask_by_priority
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


def _primitive(recipe, name):
    return next(
        primitive for primitive in recipe["primitives"] if primitive["name"] == name
    )


class Phase3CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)

    def test_tumor_burden_increase_uses_first_available_priority_label(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        mask = np.array(
            [
                [1, 1, 2, 5],
                [1, 2, 2, 5],
                [7, 7, 4, 4],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "BCSS",
            }
        )

        selection = build_candidate_mask_by_priority(
            context.normalized_mask,
            schema,
            _primitive(self.recipe, "tumor_burden_increase"),
            intent,
        )

        self.assertEqual(selection.included_labels, ("Stroma",))
        self.assertEqual(selection.candidate_mask.tolist(), (mask == 2).tolist())

    def test_source_labels_limit_candidates_even_when_priority_has_more_labels(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        mask = np.array(
            [
                [1, 1, 2, 5],
                [1, 2, 2, 5],
                [7, 7, 4, 4],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "BCSS",
                "source_labels": ["Normal epithelium"],
            }
        )

        selection = build_candidate_mask_by_priority(
            context.normalized_mask,
            schema,
            _primitive(self.recipe, "tumor_burden_increase"),
            intent,
        )

        self.assertEqual(selection.included_labels, ("Normal epithelium",))
        self.assertEqual(selection.candidate_mask.tolist(), (mask == 5).tolist())
        self.assertIn("source_label_filter_applied", selection.warnings)

    def test_target_change_fraction_progressively_adds_lower_priority_labels(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        mask = np.array(
            [
                [1, 1, 2, 5],
                [1, 7, 7, 5],
                [7, 7, 4, 4],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "BCSS",
                "target_change_fraction": 0.25,
            }
        )

        selection = build_candidate_mask_by_priority(
            context.normalized_mask,
            schema,
            _primitive(self.recipe, "tumor_burden_increase"),
            intent,
        )

        self.assertEqual(selection.included_labels, ("Stroma", "Normal epithelium"))
        self.assertEqual(
            selection.candidate_mask.tolist(),
            np.isin(mask, [2, 5]).tolist(),
        )

    def test_preserve_labels_exclude_a_priority_label_and_fall_back_to_next(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        mask = np.array(
            [
                [1, 1, 2, 5],
                [1, 2, 2, 5],
                [7, 7, 4, 4],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "BCSS",
                "preserve_labels": ["Stroma"],
            }
        )

        selection = build_candidate_mask_by_priority(
            context.normalized_mask,
            schema,
            _primitive(self.recipe, "tumor_burden_increase"),
            intent,
        )

        self.assertEqual(selection.included_labels, ("Normal epithelium",))
        self.assertEqual(selection.candidate_mask.tolist(), (mask == 5).tolist())
        self.assertIn("label_preserved:Stroma", selection.excluded_labels)

    def test_background_is_never_selected_even_if_present(self):
        schema = MaskProfileSchema.from_reference_profile("BCSS")
        mask = np.array(
            [
                [1, 1, 0, 0],
                [1, 7, 7, 0],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "BCSS",
            }
        )

        selection = build_candidate_mask_by_priority(
            context.normalized_mask,
            schema,
            _primitive(self.recipe, "tumor_burden_increase"),
            intent,
        )

        self.assertFalse(np.any(selection.candidate_mask & (mask == 0)))
        self.assertEqual(selection.included_labels, ("Other tissue",))

    def test_orca_can_use_coarse_other_tissue_with_warning(self):
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
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "ORCA",
            }
        )

        selection = build_candidate_mask_by_priority(
            context.normalized_mask,
            schema,
            _primitive(self.recipe, "tumor_burden_increase"),
            intent,
        )

        self.assertEqual(selection.included_labels, ("Other tissue",))
        self.assertEqual(selection.candidate_mask.tolist(), (mask == 7).tolist())
        self.assertIn("semantic_warning:Other tissue", selection.warnings)

    def test_unknown_fine_ids_remapped_to_other_tissue_remain_traceable(self):
        schema = MaskProfileSchema.from_reference_profile("ORCA")
        mask = np.array(
            [
                [1, 2],
                [7, 0],
            ],
            dtype=np.int64,
        )
        context = MaskEditContext.from_mask(mask, schema)
        intent = EditIntent.from_mapping(
            {
                "primitive": "tumor_burden_increase",
                "reference_profile": "ORCA",
            }
        )

        selection = build_candidate_mask_by_priority(
            context.normalized_mask,
            schema,
            _primitive(self.recipe, "tumor_burden_increase"),
            intent,
            context=context,
        )

        self.assertEqual(selection.included_labels, ("Other tissue",))
        self.assertEqual(selection.candidate_mask.tolist(), [[False, True], [True, False]])
        self.assertIn(
            "context_risk:remapped_unknown_fine_ids_to_other_tissue:2",
            selection.warnings,
        )


if __name__ == "__main__":
    unittest.main()
