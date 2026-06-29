import tempfile
import unittest
from pathlib import Path

import numpy as np

from phase3_mask_edit.benchmark.intents import (
    BuildConfig,
    ProfileSource,
    build_benchmark_intents,
    estimate_capacity,
    inject_region_hint,
    primitive_config_by_name,
    recommend_region_hint,
)
from phase3_mask_edit.benchmark.metrics import evaluate_mask_edit
from phase3_mask_edit.benchmark.models import BenchmarkIntent, read_intents_jsonl, write_intents_jsonl
from phase3_mask_edit.benchmark.prompts import semantic_diff_for_intent, template_prompt_for_intent
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import save_id_mask


class MaskEditBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.schema = MaskProfileSchema.from_reference_profile("BCSS")
        self.recipe = load_recipe("phase3_mask_edit/recipes/generic.yaml")

    def test_capacity_and_region_hint_for_necrosis(self):
        mask = np.zeros((64, 64), dtype=np.int64)
        mask[8:48, 8:48] = 1
        mask[48:60, 8:48] = 2
        primitive = primitive_config_by_name(self.recipe, "necrosis_appearance")
        intent = EditIntent(primitive="necrosis_appearance", strength="mild", reference_profile="BCSS")

        capacity = estimate_capacity(mask, intent, primitive, self.schema)
        region_hint = recommend_region_hint(mask, self.schema, intent, primitive)

        self.assertEqual(capacity["status"], "executable")
        self.assertGreater(region_hint["area_pixels"], 0)
        self.assertIn("centroid_xy", region_hint)

    def test_jsonl_roundtrip_and_region_injection(self):
        gt = BenchmarkIntent(
            sample_id="s1",
            organ="breast",
            profile="BCSS",
            image_path=None,
            mask_path="mask.png",
            primitive="necrosis_appearance",
            strength="mild",
            region_hint={"location": "center", "centroid_xy": [3, 4]},
            source_labels=("Tumor",),
            target_label="Necrosis",
            expected_direction="increase",
            expected_area_bucket=(0.08, 0.14),
            seed=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intents.jsonl"
            write_intents_jsonl([gt], path)
            loaded = read_intents_jsonl(path)[0]

        intent = inject_region_hint(EditIntent(primitive="necrosis_appearance", reference_profile="BCSS"), loaded.region_hint)
        self.assertEqual(loaded.source_labels, ("Tumor",))
        self.assertEqual(intent.region_hint["location"], "center")

    def test_template_prompt_and_semantic_diff(self):
        gt = BenchmarkIntent(
            sample_id="s1",
            organ="breast",
            profile="BCSS",
            image_path=None,
            mask_path="mask.png",
            primitive="tumor_burden_decrease",
            strength="significant",
            region_hint={"location": "upper_left", "relation": "peripheral"},
            source_labels=("Tumor",),
            target_label="Stroma",
            expected_direction="decrease",
            expected_area_bucket=(0.24, 0.40),
            seed=1,
        )
        prompt = template_prompt_for_intent(gt)
        diff = semantic_diff_for_intent(gt)

        self.assertIn("decrease the tumor burden", prompt.instruction)
        self.assertEqual(diff["tumor_change"]["growth"], "decrease")
        self.assertEqual(diff["benchmark_gt"]["region_hint"]["location"], "upper_left")

    def test_metrics_detect_expected_change(self):
        source = np.ones((32, 32), dtype=np.int64)
        target = source.copy()
        target[8:16, 8:16] = 3
        gt = BenchmarkIntent(
            sample_id="s1",
            organ="breast",
            profile="BCSS",
            image_path=None,
            mask_path="mask.png",
            primitive="necrosis_appearance",
            strength="mild",
            region_hint={"bbox_xyxy": [6, 6, 18, 18], "centroid_xy": [12, 12]},
            source_labels=("Tumor",),
            target_label="Necrosis",
            expected_direction="increase",
            expected_area_bucket=(0.04, 0.20),
            seed=1,
        )

        metrics = evaluate_mask_edit(source, target, gt)

        self.assertTrue(metrics["class_ok"])
        self.assertTrue(metrics["direction_ok"])
        self.assertTrue(metrics["strength_ok"])
        self.assertTrue(metrics["location_ok"])

    def test_build_intents_from_temp_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask = np.zeros((96, 96), dtype=np.int64)
            mask[16:72, 16:72] = 1
            mask[72:88, 16:72] = 2
            mask_path = root / "BCSS" / "sample_mask.png"
            save_id_mask(mask, mask_path)
            config = BuildConfig(
                data_root=root,
                output_dir=root / "out",
                profiles=(ProfileSource("breast", "BCSS", ("BCSS/*mask.png",)),),
                patches_per_combo=1,
                strengths=("mild",),
                allowed_primitives=("necrosis_appearance",),
                seed=7,
            )

            intents, summary = build_benchmark_intents(config)

        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].primitive, "necrosis_appearance")
        self.assertFalse(summary["shortfalls"])


if __name__ == "__main__":
    unittest.main()
