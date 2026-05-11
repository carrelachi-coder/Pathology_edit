"""Contract tests for sequential Phase 3 primitive composition."""

import unittest

import numpy as np

from phase3_mask_edit.backends.organic_projection import apply_organic_projected_label_write
from phase3_mask_edit.core.labels import MaskProfileSchema


class Phase3ComposeContractTests(unittest.TestCase):
    def setUp(self):
        self.schema = MaskProfileSchema.from_reference_profile("BCSS")

    def test_necrosis_then_intratumoral_immune_uses_current_mask(self):
        old_mask = np.zeros((64, 64), dtype=np.int64)
        old_mask[8:56, 8:56] = 1
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[20:44, 20:44] = True

        necrosis = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Necrosis",
            primitive_config={
                "name": "necrosis_appearance",
                "parameter_ranges": {
                    "organic_min_template_legal_overlap_fraction": 0.0,
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 32,
                    "organic_template_spillover_fraction": 0.0,
                    "max_necrosis_fraction_of_tumor": 1.0,
                },
            },
            seed=1,
            target_pixels=160,
        )
        immune = apply_organic_projected_label_write(
            necrosis.target_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Immune infiltrate",
            primitive_config={
                "name": "intratumoral_immune_infiltration",
                "spatial_pattern": {
                    "spot_policy": {
                        "max_total_area_fraction_of_tumor": 1.0,
                        "min_spot_area_px": 1,
                    },
                },
                "parameter_ranges": {
                    "organic_min_template_legal_overlap_fraction": 0.0,
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 32,
                    "organic_template_spillover_fraction": 0.0,
                    "max_changed_area_fraction": 1.0,
                },
            },
            seed=2,
            target_pixels=120,
        )

        self.assertEqual(immune.selected_pixels, 120)
        self.assertFalse(np.any(immune.change_region & necrosis.change_region))
        self.assertTrue(np.all(necrosis.target_mask[immune.change_region] == 1))
        self.assertTrue(np.all(immune.target_mask[necrosis.change_region] == 3))

    def test_intratumoral_immune_then_necrosis_uses_current_mask(self):
        old_mask = np.zeros((64, 64), dtype=np.int64)
        old_mask[8:56, 8:56] = 1
        raw_candidate = np.zeros_like(old_mask, dtype=bool)
        raw_candidate[18:46, 18:46] = True

        immune = apply_organic_projected_label_write(
            old_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Immune infiltrate",
            primitive_config={
                "name": "intratumoral_immune_infiltration",
                "spatial_pattern": {
                    "spot_policy": {
                        "max_total_area_fraction_of_tumor": 1.0,
                        "min_spot_area_px": 1,
                    },
                },
                "parameter_ranges": {
                    "organic_min_template_legal_overlap_fraction": 0.0,
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 32,
                    "organic_template_spillover_fraction": 0.0,
                    "max_changed_area_fraction": 1.0,
                },
            },
            seed=3,
            target_pixels=140,
        )
        necrosis = apply_organic_projected_label_write(
            immune.target_mask,
            raw_candidate,
            schema=self.schema,
            source_labels=("Tumor",),
            target_label="Necrosis",
            primitive_config={
                "name": "necrosis_appearance",
                "parameter_ranges": {
                    "organic_min_template_legal_overlap_fraction": 0.0,
                    "organic_min_component_fraction": 0.0,
                    "organic_template_neighborhood_radius_px": 32,
                    "organic_template_spillover_fraction": 0.0,
                    "max_necrosis_fraction_of_tumor": 1.0,
                },
            },
            seed=4,
            target_pixels=120,
        )

        self.assertEqual(necrosis.selected_pixels, 120)
        self.assertFalse(np.any(necrosis.change_region & immune.change_region))
        self.assertTrue(np.all(immune.target_mask[necrosis.change_region] == 1))
        self.assertTrue(np.all(necrosis.target_mask[immune.change_region] == 4))


if __name__ == "__main__":
    unittest.main()
