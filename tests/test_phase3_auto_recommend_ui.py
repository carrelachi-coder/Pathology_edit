import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


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
    _build_online_agent_command,
    _estimate_recommendation_capacity,
    _large_patch_specs,
    _primitive_config,
    _profile_defaults,
    _run_segmentator_tissue_mask,
    _with_default_contour_labels,
)
from phase3_mask_edit.core.context import MaskEditContext


GENERIC_RECIPE = Path("phase3_mask_edit/recipes/generic.yaml")


class OnlineProductIntegrationTests(unittest.TestCase):
    def test_probnet_ui_defaults_match_frozen_sampling_policy(self):
        defaults = _profile_defaults("BCSS")

        self.assertTrue(defaults["probnet_ckpt"].endswith("best_epoch29_c29607f1b609accb.pt"))
        self.assertEqual(defaults["density_scale_json"], "")
        self.assertTrue(defaults["nuclei_library"].endswith("/BCSS"))

    def test_agentic_ui_delegates_to_release_driven_production_runner(self):
        state = {
            "profile": "BCSS",
            "output_dir": "/tmp/ui-run",
            "reference_image": "/tmp/source.png",
            "reference_tissue_mask": "/tmp/source_tissue.png",
            "reference_nuclei_mask": "/tmp/source_nuclei.png",
            "target_tissue_mask": "/tmp/target_tissue.png",
            "target_nuclei_mask": "/tmp/target_nuclei.png",
            "semantic_change_region": "/tmp/semantic.png",
            "change_region": "/tmp/generation.png",
            "verification_runtime": {
                "segmentator_env": "segmentator-env",
                "segmentator_release": "/tmp/segmentator-release.json",
                "segmentator_python": "/tmp/segmentator-python",
                "segmentator_device": "cuda:1",
                "cellvit_script": "/tmp/cellvit-wrapper.py",
                "cellvit_model": "/tmp/cellvit.pt",
                "cellvit_root": "/tmp/cellvit",
                "cellvit_device": "cuda:2",
            },
        }
        args = SimpleNamespace(
            pretrained_model_name_or_path="/tmp/flux",
            inpaint_checkpoint="/tmp/inpaint",
            cross_v1_checkpoint="/tmp/cross",
            pix2pix_checkpoint="/tmp/pix2pix_epoch26_step214895.pt",
            device="cuda:0",
            color_match="none",
        )

        command, output_dir = _build_online_agent_command(
            state=state,
            args=args,
            route_threshold=0.30,
        )

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertTrue(command[1].endswith("scripts/run_agentic_edit_workflow.py"))
        self.assertEqual(value("--segmentator-release"), "/tmp/segmentator-release.json")
        self.assertNotIn("--segmentator-checkpoint", command)
        self.assertEqual(value("--semantic-change-region"), "/tmp/semantic.png")
        self.assertEqual(value("--generation-change-region"), "/tmp/generation.png")
        self.assertEqual(value("--semantic-postprocess-mode"), "shadow")
        self.assertEqual(value("--cellvit-script"), "/tmp/cellvit-wrapper.py")
        self.assertEqual(value("--pix2pix-checkpoint"), "/tmp/pix2pix_epoch26_step214895.pt")
        self.assertEqual(output_dir, Path("/tmp/ui-run/agentic_generation"))

    def test_source_auto_segmentation_uses_c_line_release_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "segmentator-release.json"
            release.write_text("{}\n", encoding="utf-8")
            image = root / "source.png"
            Image.new("RGB", (8, 8), "white").save(image)
            captured = {}

            def fake_run(command, **kwargs):
                del kwargs
                captured["command"] = command
                output_dir = Path(command[command.index("--output-dir") + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(
                    np.full((8, 8), 9, dtype=np.uint8)
                ).save(output_dir / "fine_mask.png")
                return SimpleNamespace(stdout="", stderr="")

            with patch(
                "scripts.phase3_end_to_end_ui.subprocess.run",
                side_effect=fake_run,
            ):
                output = _run_segmentator_tissue_mask(
                    profile="BCSS",
                    image_path=image,
                    output_dir=root / "run",
                    conda_env="unused",
                    release=str(release),
                    python_executable="/tmp/segmentator-python",
                    device="cuda:1",
                )

            command = captured["command"]
            self.assertIn("--release", command)
            self.assertNotIn("--checkpoint", command)
            self.assertIn("--save-probabilities", command)
            self.assertIn("--save-entropy", command)
            self.assertEqual(
                int(np.asarray(Image.open(output))[0, 0]),
                9,
            )

    def test_large_patch_agent_keeps_semantic_and_generation_regions_separate(self):
        semantic = np.zeros((16, 16), dtype=bool)
        semantic[8, 8] = True
        generation = np.zeros((16, 16), dtype=bool)
        generation[7:10, 7:10] = True

        specs = _large_patch_specs(
            semantic_change_region=semantic,
            generation_change_region=generation,
            patch_size=16,
            write_margin=0,
            patch_stride=16,
        )

        self.assertEqual(len(specs), 1)
        self.assertEqual(
            int(np.count_nonzero(specs[0]["semantic_change_patch"])),
            1,
        )
        self.assertEqual(
            int(np.count_nonzero(specs[0]["generation_change_patch"])),
            9,
        )
        self.assertEqual(int(np.count_nonzero(specs[0]["write_mask"])), 9)


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
