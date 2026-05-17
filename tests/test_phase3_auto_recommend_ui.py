import sys
import types
import unittest
from pathlib import Path

import numpy as np


sys.modules.setdefault(
    "gradio",
    types.SimpleNamespace(Error=RuntimeError, update=lambda **kwargs: kwargs),
)

inpaint_pipeline = types.ModuleType("scripts.run_phase3_inpaint_pipeline")
for name in (
    "_build_target_nuclei",
    "_load_rgb_image",
    "_load_uint8_mask",
    "_run_generation_stage",
    "_save_compare_panel",
    "_save_pre_generation_artifacts",
    "_save_target_combined_mask",
    "_select_generation_mode",
    "_format_subprocess_error",
    "_validate_same_size",
):
    setattr(inpaint_pipeline, name, lambda *args, **kwargs: None)
inpaint_pipeline._change_area_fraction = lambda *args, **kwargs: 0.0
sys.modules.setdefault("scripts.run_phase3_inpaint_pipeline", inpaint_pipeline)

from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from scripts.phase3_end_to_end_ui import (
    _auto_feasible_strengths_for_primitive,
    _estimate_recommendation_capacity,
    _primitive_config,
    _with_default_contour_labels,
)
from phase3_mask_edit.core.context import MaskEditContext


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


class Phase3AutoRecommendUiTests(unittest.TestCase):
    def setUp(self):
        self.recipe = load_recipe(GENERIC_RECIPE)
        self.schema = MaskProfileSchema.from_reference_profile("BCSS")

    def test_tumor_decrease_not_feasible_without_backfill_tissue(self):
        mask = np.ones((32, 32), dtype=np.int64)
        primitive = _primitive_config(self.recipe, "tumor_burden_decrease")
        intent = _with_default_contour_labels(
            EditIntent(
                primitive="tumor_burden_decrease",
                strength="mild",
                reference_profile="BCSS",
            ),
            primitive,
            self.schema,
        )

        feasibility = _estimate_recommendation_capacity(
            mask, intent, primitive, self.schema
        )

        self.assertEqual(feasibility["status"], "capacity_failed")
        self.assertIn(
            "capacity failed: no backfill tissue pixels in current mask "
            "(Stroma, Other tissue, Normal epithelium, Immune infiltrate).",
            feasibility["validation_failed_checks"],
        )

    def test_tumor_increase_not_feasible_without_target_tissue(self):
        mask = np.ones((32, 32), dtype=np.int64)
        primitive = _primitive_config(self.recipe, "tumor_burden_increase")
        intent = _with_default_contour_labels(
            EditIntent(
                primitive="tumor_burden_increase",
                strength="mild",
                reference_profile="BCSS",
            ),
            primitive,
            self.schema,
        )

        feasibility = _estimate_recommendation_capacity(
            mask, intent, primitive, self.schema
        )

        self.assertEqual(feasibility["status"], "capacity_failed")
        self.assertIn(
            "capacity failed: no target tissue pixels in current mask "
            "(Stroma, Normal epithelium, Other tissue, Immune infiltrate).",
            feasibility["validation_failed_checks"],
        )

    def test_backfill_primitive_feasible_when_backfill_tissue_present(self):
        mask = np.ones((32, 32), dtype=np.int64)
        mask[:, 24:] = 2
        primitive = _primitive_config(self.recipe, "tumor_burden_decrease")

        strengths = _auto_feasible_strengths_for_primitive(
            mask,
            schema=self.schema,
            recipe=self.recipe,
            context=MaskEditContext.from_mask(mask, self.schema),
            primitive_config=primitive,
        )

        self.assertIn("mild", strengths)

    def test_necrosis_appearance_still_allows_absent_target_label(self):
        mask = np.ones((32, 32), dtype=np.int64)
        primitive = _primitive_config(self.recipe, "necrosis_appearance")

        strengths = _auto_feasible_strengths_for_primitive(
            mask,
            schema=self.schema,
            recipe=self.recipe,
            context=MaskEditContext.from_mask(mask, self.schema),
            primitive_config=primitive,
        )

        self.assertIn("mild", strengths)


if __name__ == "__main__":
    unittest.main()
