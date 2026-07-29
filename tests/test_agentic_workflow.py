import contextlib
import io
import json
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
from scripts.run_agentic_edit_workflow import (
    _generation_backend_mode,
    _load_and_validate_inputs,
    _validate_nuclei_generation_contract,
    build_parser as build_agentic_parser,
    main as run_agentic_cli,
)
from scripts.run_phase3_inpaint_pipeline import (
    _build_arg_parser as build_phase3_parser,
    _retain_complete_reference_cells,
)
from controlnet_train.cli.eval_controlnet_flux_cross_v1 import (
    build_parser as build_cross_eval_parser,
)


class AgenticRoutingTests(unittest.TestCase):
    def test_agent_cross_label_maps_to_production_backend(self):
        self.assertEqual(
            _generation_backend_mode("cross-v1-no-ip-pix2pix-v2"),
            "cross-v1",
        )
        self.assertEqual(_generation_backend_mode("inpaint"), "inpaint")

    def test_agent_validates_frozen_probnet_sampling_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "release.json"
            release.write_text(
                json.dumps(
                    {
                        "release_id": "test-release",
                        "nuclei_generation": {
                            "candidate_queue_policy": (
                                "stable_descending_probnet_score"
                            ),
                            "checkpoint_sha256": "abc123",
                        },
                    }
                ),
                encoding="utf-8",
            )
            log = root / "cell_fill_log.json"
            log.write_text(
                json.dumps(
                    {
                        "mode": "probnet",
                        "shape_sampling": {
                            "candidate_queue_policy": (
                                "stable_descending_probnet_score"
                            ),
                            "organ_specific_constraints": False,
                            "probnet_release": {"sha256": "abc123"},
                            "diagnostics_path": "/tmp/diagnostics.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "product_release": release,
                    "nuclei_generation_log": log,
                },
            )()

            result = _validate_nuclei_generation_contract(args)

            self.assertTrue(result["validated"])
            self.assertEqual(
                result["candidate_queue_policy"],
                "stable_descending_probnet_score",
            )

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
    def test_runner_exposes_release_driven_online_verifier_inputs(self):
        parser = build_agentic_parser()
        destinations = {action.dest for action in parser._actions}

        self.assertIn("segmentator_release", destinations)
        self.assertIn("cellvit_script", destinations)
        self.assertIn("semantic_change_region", destinations)
        self.assertIn("generation_change_region", destinations)
        segmentator_checkpoint = next(
            action
            for action in parser._actions
            if action.dest == "segmentator_checkpoint"
        )
        self.assertIsNone(segmentator_checkpoint.default)

    def test_cross_product_clis_do_not_expose_color_matching(self):
        for parser in (
            build_agentic_parser(),
            build_phase3_parser(),
            build_cross_eval_parser(),
        ):
            destinations = {action.dest for action in parser._actions}
            self.assertNotIn("color_match", destinations)

    def test_source_cell_retention_supports_profile_encoded_subtypes(self):
        source = np.zeros((12, 12), dtype=np.uint8)
        source[1:4, 1:4] = 101
        source[7:10, 7:10] = 103
        changed = np.zeros((12, 12), dtype=bool)
        changed[6:11, 6:11] = True

        retained, stats = _retain_complete_reference_cells(
            source,
            changed,
            policy="centroid",
        )

        self.assertEqual(stats["source_components"], 2)
        self.assertEqual(stats["kept_components"], 1)
        self.assertEqual(stats["deleted_components"], 1)
        self.assertTrue(np.all(retained[1:4, 1:4] == 101))
        self.assertEqual(int(np.count_nonzero(retained[7:10, 7:10])), 0)

    def test_cli_keeps_semantic_and_generation_regions_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "reference.png"
            source_tissue = root / "source_tissue.png"
            target_tissue = root / "target_tissue.png"
            nuclei = root / "nuclei.png"
            semantic_region = root / "semantic_region.png"
            generation_region = root / "generation_region.png"

            Image.new("RGB", (8, 8), "white").save(image)
            source = np.ones((8, 8), dtype=np.uint8)
            target = source.copy()
            target[3, 3] = 2
            semantic = np.zeros((8, 8), dtype=np.uint8)
            semantic[3, 3] = 255
            generation = np.zeros((8, 8), dtype=np.uint8)
            generation[2:5, 2:5] = 255
            Image.fromarray(source).save(source_tissue)
            Image.fromarray(target).save(target_tissue)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(nuclei)
            Image.fromarray(semantic).save(semantic_region)
            Image.fromarray(generation).save(generation_region)

            args = build_agentic_parser().parse_args(
                [
                    "--profile",
                    "BCSS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source_tissue),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--target-tissue-mask",
                    str(target_tissue),
                    "--target-nuclei-mask",
                    str(nuclei),
                    "--semantic-change-region",
                    str(semantic_region),
                    "--generation-change-region",
                    str(generation_region),
                    "--output",
                    str(root / "output"),
                ]
            )

            loaded = _load_and_validate_inputs(args)

            self.assertEqual(
                int(np.count_nonzero(loaded["semantic_change_region"])),
                1,
            )
            self.assertEqual(
                int(np.count_nonzero(loaded["generation_change_region"])),
                9,
            )

    def test_cli_rejects_generation_region_that_misses_semantic_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "reference.png"
            source_tissue = root / "source_tissue.png"
            target_tissue = root / "target_tissue.png"
            nuclei = root / "nuclei.png"
            semantic_region = root / "semantic_region.png"
            generation_region = root / "generation_region.png"

            Image.new("RGB", (8, 8), "white").save(image)
            source = np.ones((8, 8), dtype=np.uint8)
            target = source.copy()
            target[3, 3] = 2
            semantic = np.zeros((8, 8), dtype=np.uint8)
            semantic[3, 3] = 255
            Image.fromarray(source).save(source_tissue)
            Image.fromarray(target).save(target_tissue)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(nuclei)
            Image.fromarray(semantic).save(semantic_region)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(
                generation_region
            )

            args = build_agentic_parser().parse_args(
                [
                    "--profile",
                    "BCSS",
                    "--reference-image",
                    str(image),
                    "--reference-tissue-mask",
                    str(source_tissue),
                    "--reference-nuclei-mask",
                    str(nuclei),
                    "--target-tissue-mask",
                    str(target_tissue),
                    "--target-nuclei-mask",
                    str(nuclei),
                    "--semantic-change-region",
                    str(semantic_region),
                    "--generation-change-region",
                    str(generation_region),
                    "--output",
                    str(root / "output"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "must contain every semantic"):
                _load_and_validate_inputs(args)

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

    def test_g2_verifier_uses_source_prediction_and_penalizes_hallucinations(self):
        reference = np.ones((6, 6), dtype=np.uint8)
        target = reference.copy()
        target[2:4, 2:4] = 2
        change = target != reference
        source_prediction = reference.copy()
        source_prediction[0, 0] = 3
        predicted = target.copy()
        predicted[0, 0] = 3
        predicted[2, 2] = 4

        result = verify_mask_fidelity(
            reference_tissue_mask=reference,
            target_tissue_mask=target,
            predicted_tissue_mask=predicted,
            source_predicted_tissue_mask=source_prediction,
            change_region=change,
        )

        self.assertEqual(result.metrics["off_target_drift"], 0.0)
        self.assertGreater(result.metrics["target_gain_accuracy"], 0.0)
        self.assertLess(result.metrics["changed_region_macro_iou"], 0.5)

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
