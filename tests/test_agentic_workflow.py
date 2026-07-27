import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from controlnet_train.inference.agentic import (
    AgenticWorkflowConfig,
    GenerationArtifact,
    VerificationResult,
    run_agentic_workflow,
    verify_mask_fidelity,
)
from controlnet_train.inference.router import route_agentic_edit_request
from scripts.run_agentic_edit_workflow import main as run_agentic_cli


class AgenticRoutingTests(unittest.TestCase):
    def test_compact_local_edit_routes_to_inpaint(self):
        reference = np.ones((32, 32), dtype=np.uint8)
        target = reference.copy()
        target[10:14, 10:14] = 2

        decision = route_agentic_edit_request(reference, target)

        self.assertEqual(decision.primary_mode, "inpaint")
        self.assertEqual(decision.candidate_modes, ("inpaint", "cross"))
        self.assertEqual(decision.features.component_count, 1)

    def test_large_structural_edit_routes_to_production_cross(self):
        reference = np.ones((32, 32), dtype=np.uint8)
        target = reference.copy()
        target[:16] = 2

        decision = route_agentic_edit_request(reference, target)

        self.assertEqual(decision.primary_mode, "cross")
        self.assertGreaterEqual(decision.features.change_ratio_tissue, 0.30)


class AgenticWorkflowTests(unittest.TestCase):
    def test_failed_inpaint_falls_back_to_production_cross(self):
        reference = np.ones((16, 16), dtype=np.uint8)
        target = reference.copy()
        target[4:6, 4:6] = 2
        modes = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
                modes.append(mode)
                image_path = attempt_dir / "generated.png"
                Image.new("RGB", (16, 16), "white").save(image_path)
                return GenerationArtifact(mode=mode, image_path=image_path)

            def verify(artifact: GenerationArtifact) -> VerificationResult:
                passed = artifact.mode != "inpaint"
                return VerificationResult(
                    passed=passed,
                    score=0.9 if passed else 0.2,
                    metrics={"synthetic": 1.0 if passed else 0.0},
                    failed_checks=() if passed else ("synthetic",),
                )

            result = run_agentic_workflow(
                reference_tissue_mask=reference,
                target_tissue_mask=target,
                output_dir=root,
                generate=generate,
                verify=verify,
                config=AgenticWorkflowConfig(max_attempts=2),
            )

            self.assertEqual(result.status, "validated")
            self.assertEqual(modes, ["inpaint", "cross-v1-no-ip-pix2pix-v2"])
            self.assertTrue((root / "agentic_workflow.json").exists())

    def test_cross_off_target_failure_recovers_with_inpaint(self):
        reference = np.ones((16, 16), dtype=np.uint8)
        target = reference.copy()
        target[:8] = 2
        modes = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
                modes.append(mode)
                image_path = attempt_dir / "generated.png"
                Image.new("RGB", (16, 16), "white").save(image_path)
                return GenerationArtifact(mode=mode, image_path=image_path)

            def verify(artifact: GenerationArtifact) -> VerificationResult:
                passed = artifact.mode == "inpaint"
                return VerificationResult(
                    passed=passed,
                    score=0.9 if passed else 0.4,
                    metrics={"off_target_drift": 0.0 if passed else 0.2},
                    failed_checks=() if passed else ("off_target_drift",),
                )

            result = run_agentic_workflow(
                reference_tissue_mask=reference,
                target_tissue_mask=target,
                output_dir=root,
                generate=generate,
                verify=verify,
                config=AgenticWorkflowConfig(max_attempts=2),
            )

            self.assertEqual(result.status, "validated")
            self.assertEqual(modes, ["cross-v1-no-ip-pix2pix-v2", "inpaint"])
            self.assertIn("preservation_recovery", result.attempts[1].decision_reason)

    def test_verifier_error_keeps_generated_artifact_before_fallback(self):
        reference = np.ones((16, 16), dtype=np.uint8)
        target = reference.copy()
        target[4:6, 4:6] = 2

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
                image_path = attempt_dir / "generated.png"
                Image.new("RGB", (16, 16), "white").save(image_path)
                return GenerationArtifact(mode=mode, image_path=image_path)

            def verify(artifact: GenerationArtifact) -> VerificationResult:
                if artifact.mode == "inpaint":
                    raise RuntimeError("segmentator unavailable")
                return VerificationResult(True, 0.8, {"ok": 1.0})

            result = run_agentic_workflow(
                reference_tissue_mask=reference,
                target_tissue_mask=target,
                output_dir=root,
                generate=generate,
                verify=verify,
                config=AgenticWorkflowConfig(max_attempts=2),
            )

            self.assertIsNotNone(result.attempts[0].artifact)
            self.assertIn("verification failed", result.attempts[0].error)
            self.assertEqual(result.status, "validated")

    def test_mask_fidelity_checks_changed_and_preserved_regions(self):
        reference = np.ones((10, 10), dtype=np.uint8)
        target = reference.copy()
        target[2:5, 2:5] = 2
        change = target != reference
        predicted = target.copy()

        result = verify_mask_fidelity(
            reference_tissue_mask=reference,
            target_tissue_mask=target,
            predicted_tissue_mask=predicted,
            change_region=change,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["off_target_drift"], 0.0)

    def test_inpaint_can_record_off_target_drift_without_using_it_as_a_gate(self):
        reference = np.ones((10, 10), dtype=np.uint8)
        target = reference.copy()
        target[2:5, 2:5] = 2
        change = target != reference
        predicted = target.copy()
        predicted[0, :] = 3

        cross_result = verify_mask_fidelity(
            reference_tissue_mask=reference,
            target_tissue_mask=target,
            predicted_tissue_mask=predicted,
            change_region=change,
            enforce_off_target_drift=True,
        )
        inpaint_result = verify_mask_fidelity(
            reference_tissue_mask=reference,
            target_tissue_mask=target,
            predicted_tissue_mask=predicted,
            change_region=change,
            enforce_off_target_drift=False,
        )

        self.assertIn("off_target_drift", cross_result.failed_checks)
        self.assertNotIn("off_target_drift", inpaint_result.failed_checks)
        self.assertEqual(cross_result.score, inpaint_result.score)

    def test_standalone_cli_handles_noop_without_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "reference.png"
            tissue = root / "tissue.png"
            nuclei = root / "nuclei.png"
            Image.new("RGB", (8, 8), "white").save(image)
            Image.fromarray(np.ones((8, 8), dtype=np.uint8)).save(tissue)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(nuclei)
            pretrained = root / "flux"
            inpaint = root / "inpaint"
            cross = root / "cross"
            cellvit_root = root / "cellvit"
            for directory in (pretrained, inpaint, cross, cellvit_root):
                directory.mkdir()
            pix2pix = root / "pix2pix_model.pt"
            segmentator = root / "segmentator.pt"
            cellvit = root / "cellvit.pt"
            for checkpoint in (pix2pix, segmentator, cellvit):
                checkpoint.touch()
            output = root / "output"

            argv = [
                "--profile", "BCSS",
                "--reference-image", str(image),
                "--reference-tissue-mask", str(tissue),
                "--reference-nuclei-mask", str(nuclei),
                "--target-tissue-mask", str(tissue),
                "--target-nuclei-mask", str(nuclei),
                "--output", str(output),
                "--pretrained-model-name-or-path", str(pretrained),
                "--inpaint-checkpoint", str(inpaint),
                "--cross-v1-checkpoint", str(cross),
                "--pix2pix-checkpoint", str(pix2pix),
                "--segmentator-checkpoint", str(segmentator),
                "--cellvit-model", str(cellvit),
                "--cellvit-root", str(cellvit_root),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = run_agentic_cli(argv)

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "generated_image.png").exists())
            summary = (output / "pipeline_summary.json").read_text(encoding="utf-8")
            self.assertIn('"status": "noop"', summary)


if __name__ == "__main__":
    unittest.main()
