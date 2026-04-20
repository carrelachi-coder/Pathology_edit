import json
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from controlnet_train import EditPipelineInputs as ExportedEditPipelineInputs
from controlnet_train import run_edit_pipeline as exported_run_edit_pipeline
from controlnet_train.cli.edit_pipeline import parse_args
from controlnet_train.inference import (
    EditPipelineInputs,
    EditPipelineResult,
    EditRoutingConfig,
    compute_change_region_mask,
    resolve_prompt,
    run_edit_pipeline,
    route_edit_request,
)

_TMP_ROOT = Path.cwd() / ".tmp_testdata"
_TMP_ROOT.mkdir(exist_ok=True)


def _write_rgb(path: Path, value: int) -> None:
    array = np.full((8, 8, 3), value, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _write_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


class RouterTests(unittest.TestCase):
    def test_compute_change_region_mask_marks_changed_pixels(self):
        reference = torch.tensor([[1, 1, 2], [1, 3, 3]], dtype=torch.int64)
        target = torch.tensor([[1, 4, 2], [1, 3, 5]], dtype=torch.int64)

        change_mask = compute_change_region_mask(reference, target)

        expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(change_mask, expected))

    def test_route_edit_request_reports_cross_and_changed_labels(self):
        reference = torch.tensor([[1, 1], [1, 1]], dtype=torch.int64)
        target = torch.tensor([[1, 2], [3, 1]], dtype=torch.int64)

        decision = route_edit_request(reference, target, config=EditRoutingConfig())

        self.assertEqual(decision.selected_mode, "cross")
        self.assertAlmostEqual(decision.change_ratio, 0.5)
        self.assertEqual(decision.changed_tissue_ids_from, [1])
        self.assertEqual(decision.changed_tissue_ids_to, [2, 3])

    def test_route_edit_request_defaults_middle_band_to_inpaint(self):
        reference = torch.zeros((4, 4), dtype=torch.int64)
        target = reference.clone()
        target[0, 0] = 1
        target[0, 1] = 1
        target[0, 2] = 1
        target[0, 3] = 1

        decision = route_edit_request(
            reference,
            target,
            config=EditRoutingConfig(t_inpaint=0.12, t_cross=0.30),
        )

        self.assertEqual(decision.selected_mode, "inpaint")
        self.assertAlmostEqual(decision.change_ratio, 0.25)

    def test_route_edit_request_rejects_invalid_threshold_order(self):
        with self.assertRaises(ValueError):
            route_edit_request(
                torch.zeros((2, 2), dtype=torch.int64),
                torch.zeros((2, 2), dtype=torch.int64),
                config=EditRoutingConfig(t_inpaint=0.4, t_cross=0.2),
            )


class PromptResolutionTests(unittest.TestCase):
    def test_resolve_prompt_prefers_explicit_prompt(self):
        prompt = resolve_prompt("custom prompt", "BCSS")
        self.assertEqual(prompt, "custom prompt")

    def test_resolve_prompt_uses_dataset_default_when_prompt_missing(self):
        prompt = resolve_prompt(None, "PANDA")
        self.assertIn("prostate", prompt.lower())

    def test_resolve_prompt_falls_back_to_generic_prompt(self):
        prompt = resolve_prompt(None, None)
        self.assertEqual(prompt, "H&E stained cancer histopathology at 40x magnification")


class PipelineTests(unittest.TestCase):
    def test_run_edit_pipeline_uses_inpaint_runner_and_writes_summary(self):
        tmpdir = _TMP_ROOT / f"phase54_{uuid.uuid4().hex}"
        try:
            reference_image = tmpdir / "reference.png"
            reference_tissue = tmpdir / "reference_tissue.png"
            reference_nuclei = tmpdir / "reference_nuclei.png"
            target_tissue = tmpdir / "target_tissue.png"
            target_nuclei = tmpdir / "target_nuclei.png"
            output_dir = tmpdir / "outputs"

            _write_rgb(reference_image, 32)
            _write_mask(reference_tissue, np.full((8, 8), 1, dtype=np.uint8))
            _write_mask(reference_nuclei, np.full((8, 8), 101, dtype=np.uint8))
            _write_mask(target_tissue, np.full((8, 8), 1, dtype=np.uint8))
            _write_mask(target_nuclei, np.full((8, 8), 101, dtype=np.uint8))

            calls = {}

            def fake_inpaint_runner(bundle, inputs, prompt, change_region_mask):
                calls["mode"] = "inpaint"
                calls["prompt"] = prompt
                calls["mask_sum"] = float(change_region_mask.sum().item())
                return Image.new("RGB", (8, 8), color=(12, 34, 56))

            def fake_cross_runner(bundle, inputs, prompt):
                raise AssertionError("cross runner should not be used for unchanged tissue masks")

            result = run_edit_pipeline(
                inputs=EditPipelineInputs(
                    reference_image=reference_image,
                    reference_tissue_mask=reference_tissue,
                    reference_nuclei_mask=reference_nuclei,
                    target_tissue_mask=target_tissue,
                    target_nuclei_mask=target_nuclei,
                    output_dir=output_dir,
                    dataset="PANDA",
                ),
                inpaint_bundle=object(),
                cross_bundle=object(),
                inpaint_runner=fake_inpaint_runner,
                cross_runner=fake_cross_runner,
            )

            self.assertIsInstance(result, EditPipelineResult)
            self.assertEqual(result.selected_mode, "inpaint")
            self.assertEqual(calls["mode"], "inpaint")
            self.assertEqual(calls["mask_sum"], 0.0)
            self.assertIn("prostate", result.prompt.lower())
            self.assertTrue((output_dir / "final.png").exists())
            self.assertTrue((output_dir / "change_region_mask.png").exists())
            summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf8"))
            self.assertEqual(summary["selected_mode"], "inpaint")
            self.assertIn("prostate", summary["prompt"].lower())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_edit_pipeline_uses_cross_runner_for_large_changes(self):
        tmpdir = _TMP_ROOT / f"phase54_{uuid.uuid4().hex}"
        try:
            reference_image = tmpdir / "reference.png"
            reference_tissue = tmpdir / "reference_tissue.png"
            reference_nuclei = tmpdir / "reference_nuclei.png"
            target_tissue = tmpdir / "target_tissue.png"
            target_nuclei = tmpdir / "target_nuclei.png"
            output_dir = tmpdir / "outputs"

            _write_rgb(reference_image, 32)
            _write_mask(reference_tissue, np.full((8, 8), 1, dtype=np.uint8))
            _write_mask(reference_nuclei, np.full((8, 8), 101, dtype=np.uint8))
            _write_mask(target_tissue, np.full((8, 8), 4, dtype=np.uint8))
            _write_mask(target_nuclei, np.full((8, 8), 101, dtype=np.uint8))

            calls = {}

            def fake_inpaint_runner(bundle, inputs, prompt, change_region_mask):
                raise AssertionError("inpaint runner should not be used for large tissue edits")

            def fake_cross_runner(bundle, inputs, prompt):
                calls["mode"] = "cross"
                calls["prompt"] = prompt
                return Image.new("RGB", (8, 8), color=(120, 90, 60))

            result = run_edit_pipeline(
                inputs=EditPipelineInputs(
                    reference_image=reference_image,
                    reference_tissue_mask=reference_tissue,
                    reference_nuclei_mask=reference_nuclei,
                    target_tissue_mask=target_tissue,
                    target_nuclei_mask=target_nuclei,
                    output_dir=output_dir,
                    prompt="custom pathology prompt",
                ),
                inpaint_bundle=object(),
                cross_bundle=object(),
                inpaint_runner=fake_inpaint_runner,
                cross_runner=fake_cross_runner,
            )

            self.assertEqual(result.selected_mode, "cross")
            self.assertEqual(result.prompt, "custom pathology prompt")
            self.assertEqual(calls["mode"], "cross")
            self.assertEqual(calls["prompt"], "custom pathology prompt")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_edit_pipeline_rejects_size_mismatch(self):
        tmpdir = _TMP_ROOT / f"phase54_{uuid.uuid4().hex}"
        try:
            reference_image = tmpdir / "reference.png"
            reference_tissue = tmpdir / "reference_tissue.png"
            reference_nuclei = tmpdir / "reference_nuclei.png"
            target_tissue = tmpdir / "target_tissue.png"
            target_nuclei = tmpdir / "target_nuclei.png"

            _write_rgb(reference_image, 32)
            _write_mask(reference_tissue, np.full((8, 8), 1, dtype=np.uint8))
            _write_mask(reference_nuclei, np.full((8, 8), 101, dtype=np.uint8))
            _write_mask(target_tissue, np.full((4, 4), 1, dtype=np.uint8))
            _write_mask(target_nuclei, np.full((8, 8), 101, dtype=np.uint8))

            with self.assertRaises(ValueError):
                run_edit_pipeline(
                    inputs=EditPipelineInputs(
                        reference_image=reference_image,
                        reference_tissue_mask=reference_tissue,
                        reference_nuclei_mask=reference_nuclei,
                        target_tissue_mask=target_tissue,
                        target_nuclei_mask=target_nuclei,
                        output_dir=tmpdir / "outputs",
                    ),
                    inpaint_bundle=object(),
                    cross_bundle=object(),
                    inpaint_runner=lambda *args, **kwargs: Image.new("RGB", (8, 8)),
                    cross_runner=lambda *args, **kwargs: Image.new("RGB", (8, 8)),
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class EditPipelineCliTests(unittest.TestCase):
    def test_parse_args_accepts_required_inputs_and_optional_dataset(self):
        args = parse_args(
            [
                "--reference-image",
                "ref.png",
                "--reference-tissue-mask",
                "ref_tissue.png",
                "--reference-nuclei-mask",
                "ref_nuclei.png",
                "--target-tissue-mask",
                "target_tissue.png",
                "--target-nuclei-mask",
                "target_nuclei.png",
                "--pretrained-model-name-or-path",
                "flux-dev",
                "--inpaint-checkpoint",
                "runs/inpaint",
                "--cross-checkpoint",
                "runs/cross",
                "--output-dir",
                "outputs",
                "--dataset",
                "BCSS",
                "--save-debug-artifacts",
            ]
        )

        self.assertEqual(args.dataset, "BCSS")
        self.assertTrue(args.save_debug_artifacts)
        self.assertIsNone(args.prompt)

    def test_parse_args_accepts_force_mode_and_prompt(self):
        args = parse_args(
            [
                "--reference-image",
                "ref.png",
                "--reference-tissue-mask",
                "ref_tissue.png",
                "--reference-nuclei-mask",
                "ref_nuclei.png",
                "--target-tissue-mask",
                "target_tissue.png",
                "--target-nuclei-mask",
                "target_nuclei.png",
                "--pretrained-model-name-or-path",
                "flux-dev",
                "--inpaint-checkpoint",
                "runs/inpaint",
                "--cross-checkpoint",
                "runs/cross",
                "--output-dir",
                "outputs",
                "--force-mode",
                "cross",
                "--prompt",
                "custom prompt",
            ]
        )

        self.assertEqual(args.force_mode, "cross")
        self.assertEqual(args.prompt, "custom prompt")


class ExportTests(unittest.TestCase):
    def test_top_level_package_exports_edit_pipeline_api(self):
        self.assertIs(ExportedEditPipelineInputs, EditPipelineInputs)
        self.assertIs(exported_run_edit_pipeline, run_edit_pipeline)


if __name__ == "__main__":
    unittest.main()
