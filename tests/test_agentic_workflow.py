import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from controlnet_train.inference.agentic import (
    AgenticWorkflowConfig,
    FidelityThresholds,
    GenerationArtifact,
    VerificationResult,
    run_agentic_workflow,
    verify_mask_fidelity,
)
from controlnet_train.inference.router import (
    AgenticRouteFeatures,
    AgenticRoutingDecision,
    route_agentic_edit_request,
)
from scripts.run_agentic_edit_workflow import (
    _generation_backend_mode,
    _load_and_validate_inputs,
    _prepare_verification_runtime,
    _selected_image_generation_provenance,
    _run_segmentator,
    _validate_image_generation_contract,
    _validate_frozen_nuclei_replay_compatibility,
    _validate_nuclei_generation_contract,
    _source_region_quality_or_abstain,
    build_parser as build_agentic_parser,
    main as run_agentic_cli,
)
from scripts.run_phase3_inpaint_pipeline import (
    _build_arg_parser as build_phase3_parser,
    _release_generation_model_caches,
    _retain_complete_reference_cells,
)
from scripts.run_cellvit_single_patch import (
    _detector_reported_zero_cells,
    cellvit_instance_counts_in_region,
    main as run_cellvit_single_patch,
)
from controlnet_train.cli.eval_controlnet_flux_cross_v1 import (
    build_parser as build_cross_eval_parser,
)


class AgenticRoutingTests(unittest.TestCase):
    def test_cellvit_zero_detection_signal_is_explicit(self):
        self.assertTrue(
            _detector_reported_zero_cells(
                "2026-08-02 [WARNING] - No cells have been extracted",
                "",
            )
        )
        self.assertFalse(_detector_reported_zero_cells("Finished processing", ""))

    def test_cellvit_explicit_empty_cells_json_writes_zero_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            model_path = root / "model.pth"
            cells_path = root / "empty_cells.json"
            output_path = root / "nuclei.png"
            Image.new("RGB", (12, 10), "white").save(image_path)
            model_path.write_bytes(b"checkpoint fixture")
            cells_path.write_text('{"cells": []}', encoding="utf-8")

            return_code = run_cellvit_single_patch(
                [
                    "--image",
                    str(image_path),
                    "--output-mask",
                    str(output_path),
                    "--model",
                    str(model_path),
                    "--cells-json",
                    str(cells_path),
                ]
            )

            mask = np.asarray(Image.open(output_path))
            summary = json.loads(
                output_path.with_suffix(".cellvit_single_patch.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(return_code, 0)
            self.assertFalse(np.any(mask))
            self.assertEqual(summary["detected_cell_count"], 0)
            self.assertTrue(summary["zero_detections"])

    def _frozen_nuclei_replay_fixture(self, root: Path):
        tissue = root / "target_mask.png"
        nuclei = root / "target_nuclei_mask.png"
        log = root / "cell_fill_log.json"
        tissue.write_bytes(b"frozen tissue")
        nuclei.write_bytes(b"frozen nuclei")
        log.write_text('{"mode":"probnet"}', encoding="utf-8")

        sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        tissue_sha256 = sha256(tissue)
        nuclei_sha256 = sha256(nuclei)
        manifest = root / "approved_nuclei_stage_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "nuclei",
                    "all_automatic_checks_passed": True,
                    "approval": {"status": "approved"},
                    "entries": [
                        {
                            "case_id": "case-1",
                            "approval": "approved",
                            "audit_passed": True,
                            "approved_target_nuclei_sha256": nuclei_sha256,
                            "parent_target_tissue_sha256": tissue_sha256,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        provenance = root / "approved_nuclei_provenance.json"
        provenance.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "approved_nuclei_reused",
                    "approved_nuclei_manifest": str(manifest),
                    "approved_entry_case_id": "case-1",
                    "approved_target_tissue_sha256": tissue_sha256,
                    "approved_target_nuclei_sha256": nuclei_sha256,
                    "asset_sha256": {
                        "target_nuclei": nuclei_sha256,
                        "cell_fill_log": sha256(log),
                    },
                    "tissue_stage_rerun": False,
                    "nuclei_stage_rerun": False,
                }
            ),
            encoding="utf-8",
        )
        args = SimpleNamespace(
            target_tissue_mask=tissue,
            target_nuclei_mask=nuclei,
        )
        expected = {
            "sampling_audit_policy": "audit-v3",
            "compatible_frozen_audit_policies": {
                "audit-v2": {
                    "scope": "hash_locked_approved_nuclei_replay_only",
                    "sampling_audit_max_attempts": 6,
                    "sampling_feedback_required": False,
                }
            },
        }
        return args, log, expected

    def test_hash_locked_approved_legacy_nuclei_replay_is_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, log, expected = self._frozen_nuclei_replay_fixture(Path(tmp))

            result = _validate_frozen_nuclei_replay_compatibility(
                args=args,
                log_path=log,
                actual_policy="audit-v2",
                expected=expected,
            )

        self.assertEqual(result["mode"], "hash_locked_approved_nuclei_replay")
        self.assertEqual(result["actual_policy"], "audit-v2")
        self.assertEqual(result["current_policy"], "audit-v3")

    def test_unapproved_legacy_nuclei_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, log, expected = self._frozen_nuclei_replay_fixture(Path(tmp))
            (Path(tmp) / "approved_nuclei_provenance.json").unlink()

            with self.assertRaisesRegex(ValueError, "approval provenance"):
                _validate_frozen_nuclei_replay_compatibility(
                    args=args,
                    log_path=log,
                    actual_policy="audit-v2",
                    expected=expected,
                )

    def test_tampered_approved_nuclei_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, log, expected = self._frozen_nuclei_replay_fixture(Path(tmp))
            args.target_nuclei_mask.write_bytes(b"tampered nuclei")

            with self.assertRaisesRegex(ValueError, "target nuclei hash"):
                _validate_frozen_nuclei_replay_compatibility(
                    args=args,
                    log_path=log,
                    actual_policy="audit-v2",
                    expected=expected,
                )

    def test_verification_releases_generation_models_before_evaluators(self):
        args = SimpleNamespace()
        calls = []
        with (
            unittest.mock.patch(
                "scripts.run_agentic_edit_workflow._validate_verification_runtime",
                side_effect=lambda value: calls.append(("validate", value)),
            ),
            unittest.mock.patch(
                "scripts.run_agentic_edit_workflow._release_generation_model_caches",
                side_effect=lambda: calls.append(("release", None)),
            ),
        ):
            _prepare_verification_runtime(args)

        self.assertEqual(calls, [("validate", args), ("release", None)])

    def test_source_region_without_dataset_native_labels_abstains(self):
        source = np.full((6, 7), 7, dtype=np.uint8)
        prediction = np.ones((6, 7), dtype=np.uint8)
        probabilities = np.zeros((8, 6, 7), dtype=np.float32)
        probabilities[1, ...] = 1.0

        result = _source_region_quality_or_abstain(
            level="coarse",
            source_mask=source,
            source_prediction=prediction,
            source_probabilities=probabilities,
            class_ids=(1,),
            region=np.ones((6, 7), dtype=bool),
        )

        self.assertFalse(result["available"])
        self.assertEqual(
            result["reason"], "no_dataset_native_evaluable_pixels"
        )
        self.assertEqual(result["interpretation"], "coarse_evaluator_abstained")

    def test_segmentator_uses_explicit_python_without_conda_path_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "segmentator"
            image_path = Path(tmp) / "image.png"
            image_path.touch()
            args = SimpleNamespace(
                segmentator_python=Path("/opt/segmentator/bin/python"),
                segmentator_env="unused-conda-env",
                segmentator_checkpoint=None,
                segmentator_release=Path("/tmp/segmentator-release.json"),
                segmentator_decoder="mask2former",
                segmentator_device="cuda:1",
                profile="BCSS",
            )
            captured = {}

            def fake_run(command, _log_path):
                captured["command"] = command
                output_dir.mkdir(parents=True, exist_ok=True)
                for name in (
                    "coarse_mask.png",
                    "coarse_probabilities.npz",
                    "entropy.npy",
                    "provenance.json",
                ):
                    (output_dir / name).touch()

            with unittest.mock.patch(
                "scripts.run_agentic_edit_workflow._run_logged",
                side_effect=fake_run,
            ):
                _run_segmentator(
                    args=args,
                    image_path=image_path,
                    output_dir=output_dir,
                )

            self.assertEqual(
                captured["command"][0],
                "/opt/segmentator/bin/python",
            )
            self.assertNotIn("conda", captured["command"])

    def test_agent_validates_frozen_g2_image_generation_contract(self):
        root = Path(__file__).resolve().parents[1]
        args = type(
            "Args",
            (),
            {
                "product_release": (
                    root
                    / "benchmark_configs"
                    / "releases"
                    / "online_agent_product_v1.json"
                ),
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "controlnet_conditioning_scale": 1.0,
                "torch_dtype": "bf16",
                "seed": 42,
                "t_inpaint": 0.12,
                "t_cross": 0.30,
                "max_attempts": 2,
                "semantic_postprocess_mode": "shadow",
                "segmentator_checkpoint": None,
                "segmentator_release": (
                    root
                    / "benchmark_configs"
                    / "releases"
                    / "segmentator_fine_legacy_anchor.json"
                ),
            },
        )()

        result = _validate_image_generation_contract(args)

        self.assertTrue(result["validated"])
        self.assertEqual(result["inference"]["num_inference_steps"], 28)
        self.assertEqual(result["inference"]["seed"], 42)
        self.assertEqual(
            result["segmentator_release_id"],
            "segmentator-fine-legacy-anchor-v1",
        )
        self.assertEqual(
            result["quality_evaluator"]["preservation_exclusion_region"],
            "full_generation_change_region",
        )

    def test_agent_rejects_a_release_with_the_wrong_preservation_region(self):
        root = Path(__file__).resolve().parents[1]
        source_release = (
            root
            / "benchmark_configs"
            / "releases"
            / "online_agent_product_v1.json"
        )
        source_segmentator = (
            root
            / "benchmark_configs"
            / "releases"
            / "segmentator_fine_legacy_anchor.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            release = json.loads(source_release.read_text(encoding="utf-8"))
            release["verification"]["evaluator"][
                "preservation_exclusion_region"
            ] = "semantic_change_region_only"
            release_path = Path(tmp) / "release.json"
            release_path.write_text(json.dumps(release), encoding="utf-8")
            args = SimpleNamespace(
                product_release=release_path,
                num_inference_steps=28,
                guidance_scale=3.5,
                controlnet_conditioning_scale=1.0,
                torch_dtype="bf16",
                seed=42,
                t_inpaint=0.12,
                t_cross=0.30,
                max_attempts=2,
                semantic_postprocess_mode="shadow",
                segmentator_checkpoint=None,
                segmentator_release=source_segmentator,
            )

            with self.assertRaisesRegex(
                ValueError,
                "preservation_exclusion_region",
            ):
                _validate_image_generation_contract(args)

    def test_backend_cache_release_clears_both_generation_bundles(self):
        from scripts import run_phase3_inpaint_pipeline as pipeline

        pipeline._INPAINT_BUNDLE_CACHE[("flux", "inpaint", "cuda")] = object()
        pipeline._CROSS_V1_NO_IP_CACHE[
            ("flux", "cross", "cuda", "bf16", 28, 3.5, 1.0)
        ] = object()

        with (
            unittest.mock.patch.object(pipeline.gc, "collect") as collect,
            unittest.mock.patch.object(
                pipeline.torch.cuda, "is_available", return_value=True
            ),
            unittest.mock.patch.object(
                pipeline.torch.cuda, "empty_cache"
            ) as empty_cache,
        ):
            _release_generation_model_caches()

        self.assertFalse(pipeline._INPAINT_BUNDLE_CACHE)
        self.assertFalse(pipeline._CROSS_V1_NO_IP_CACHE)
        collect.assert_called_once_with()
        empty_cache.assert_called_once_with()

    def test_agent_surfaces_selected_cross_low_stain_provenance(self):
        provenance = _selected_image_generation_provenance(
            {
                "selected_attempt": {
                    "attempt_index": 2,
                    "artifact": {
                        "mode": "cross-v1-no-ip-pix2pix-v2",
                        "metadata": {
                            "selected_mode": "cross-v1",
                            "cross_v1": {
                                "pix2pix_v2": {
                                    "cross_rgb_od_low_stain_protection": {
                                        "policy": "cross_rgb_od_low_stain_v1",
                                        "enabled": True,
                                        "applied": True,
                                        "protected_fraction_image": 0.35,
                                        "organ_specific_constraints": False,
                                    }
                                }
                            },
                        },
                    },
                }
            }
        )

        self.assertEqual(provenance["selected_attempt"], 2)
        self.assertEqual(provenance["selected_mode"], "cross-v1")
        protection = provenance["cross_rgb_od_low_stain_protection"]
        self.assertEqual(protection["status"], "applied")
        self.assertEqual(protection["protected_fraction_image"], 0.35)
        self.assertFalse(protection["organ_specific_constraints"])

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
                            "candidate_queue_policy": "probnet_odds_mass_without_replacement",
                            "candidate_quality_score": (
                                "gamma_times_logit_probnet_probability_plus_seeded_gumbel"
                            ),
                            "candidate_probability_mass_exponent": 3.0,
                            "candidate_diversity_score": "none_poisson_candidates_only",
                            "candidate_diversity_weight": 0.0,
                            "quota_coverage_spacing_scale": 0.0,
                            "quota_coverage_max_radius": 0.0,
                            "retry_tail_policy": (
                                "same_probnet_mass_permutation_then_component_"
                                "pixel_backfill_then_same_tissue_quota_"
                                "reassignment"
                            ),
                            "component_quota_reassignment_policy": (
                                "unplaceable_component_quota_to_same_tissue_"
                                "probnet_mass_tail"
                            ),
                            "gamma": 3.0,
                            "checkpoint_sha256": "abc123",
                            "count_policy": (
                                "pre_edit_source_tissue_density_or_target_prior_"
                                "calibrated_by_pre_edit_source_times_post_edit_"
                                "target_area"
                            ),
                            "type_quota_routing_policy": (
                                "prior_total_count_then_probnet_local_type_log_"
                                "pool_with_cumulative_posterior_balancing"
                            ),
                            "shape_policy": (
                                "component_local_same_class_reference_then_"
                                "component_calibrated_library"
                            ),
                            "nucleus_spacing_margin_px": 1,
                            "instance_connectivity_policy": (
                                "largest_8_connected_component_after_transform"
                            ),
                            "source_nucleus_erasure_policy": (
                                "complete_component_on_any_deletion_region_"
                                "intersection"
                            ),
                            "buffer_nucleus_policy": (
                                "retain_generation_buffer_only_nuclei_as_"
                                "placement_obstacles"
                            ),
                            "sampling_audit_policy": (
                                "probnet_patch_relative_count_type_spatial_v3"
                            ),
                            "sampling_audit_attempts": 3,
                            "sampling_feedback_policy": (
                                "reason_directed_gamma_then_seed_v1"
                            ),
                            "sampling_feedback_max_attempts": 3,
                            "sampling_feedback_gamma_down_factor": 0.75,
                            "sampling_feedback_gamma_up_factor": 4.0 / 3.0,
                            "sampling_feedback_gamma_min": 1.5,
                            "sampling_feedback_gamma_max": 5.0,
                            "sampling_feedback_concentration_z_threshold": 1.96,
                            "sampling_feedback_immutable_parameters": [
                                "target_count",
                                "tissue_and_component_allocation",
                                "deletion_and_generation_regions",
                                "shape_source_policy",
                                "nucleus_spacing_margin_px",
                            ],
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
                            "candidate_queue_policy": "probnet_odds_mass_without_replacement",
                            "candidate_quality_score": (
                                "gamma_times_logit_probnet_probability_plus_seeded_gumbel"
                            ),
                            "candidate_probability_mass_exponent": 3.0,
                            "candidate_diversity_score": "none_poisson_candidates_only",
                            "candidate_diversity_weight": 0.0,
                            "quota_coverage_spacing_scale": 0.0,
                            "quota_coverage_max_radius": 0.0,
                            "retry_tail_policy": (
                                "same_probnet_mass_permutation_then_component_"
                                "pixel_backfill_then_same_tissue_quota_"
                                "reassignment"
                            ),
                            "component_quota_reassignment_policy": (
                                "unplaceable_component_quota_to_same_tissue_"
                                "probnet_mass_tail"
                            ),
                            "gamma": 3.0,
                            "organ_specific_constraints": False,
                            "probnet_release": {"sha256": "abc123"},
                            "diagnostics_path": "/tmp/diagnostics.json",
                            "count_policy": (
                                "pre_edit_source_tissue_density_or_target_prior_"
                                "calibrated_by_pre_edit_source_times_post_edit_"
                                "target_area"
                            ),
                            "type_quota_routing_policy": (
                                "prior_total_count_then_probnet_local_type_log_"
                                "pool_with_cumulative_posterior_balancing"
                            ),
                            "shape_policy": (
                                "component_local_same_class_reference_then_"
                                "component_calibrated_library"
                            ),
                            "nucleus_spacing_margin_px": 1,
                            "instance_connectivity_policy": (
                                "largest_8_connected_component_after_transform"
                            ),
                            "source_nucleus_erasure_policy": (
                                "complete_component_on_any_deletion_region_"
                                "intersection"
                            ),
                            "buffer_nucleus_policy": (
                                "retain_generation_buffer_only_nuclei_as_"
                                "placement_obstacles"
                            ),
                            "sampling_audit": {
                                "policy": (
                                    "probnet_patch_relative_count_type_spatial_v3"
                                ),
                                "organ_specific_constraints": False,
                                "attempt_index": 0,
                                "sampling_gamma": 3.0,
                                "evaluation_gamma": 3.0,
                                "passed": True,
                                "score": 0.88,
                            },
                            "sampling_audit_attempts": [
                                {
                                    "attempt_index": 0,
                                    "seed": 42,
                                    "sampling_gamma": 3.0,
                                    "action": "initial_sample",
                                    "trigger_reasons": [],
                                    "stage": "sampling_audit",
                                    "passed": True,
                                }
                            ],
                            "sampling_audit_max_attempts": 3,
                            "sampling_feedback": {
                                "policy": "reason_directed_gamma_then_seed_v1",
                                "initial_gamma": 3.0,
                                "selected_gamma": 3.0,
                                "selected_seed": 42,
                                "max_attempts": 3,
                                "gamma_down_factor": 0.75,
                                "gamma_up_factor": 4.0 / 3.0,
                                "gamma_min": 1.5,
                                "gamma_max": 5.0,
                                "concentration_z_threshold": 1.96,
                                "immutable_parameters": [
                                    "target_count",
                                    "tissue_and_component_allocation",
                                    "deletion_and_generation_regions",
                                    "shape_source_policy",
                                    "nucleus_spacing_margin_px",
                                ],
                                "attempts": [
                                    {
                                        "attempt_index": 0,
                                        "seed": 42,
                                        "sampling_gamma": 3.0,
                                        "action": "initial_sample",
                                        "trigger_reasons": [],
                                        "stage": "sampling_audit",
                                        "passed": True,
                                    }
                                ],
                                "selected_attempt": 0,
                                "resampled": False,
                                "parameter_adjusted": False,
                            },
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
                "probnet_odds_mass_without_replacement",
            )
            self.assertEqual(result["gamma"], 3.0)
            self.assertEqual(result["candidate_probability_mass_exponent"], 3.0)
            self.assertEqual(result["candidate_diversity_weight"], 0.0)
            self.assertEqual(result["quota_coverage_spacing_scale"], 0.0)
            self.assertEqual(result["quota_coverage_max_radius"], 0.0)
            self.assertEqual(
                result["retry_tail_policy"],
                "same_probnet_mass_permutation_then_component_pixel_backfill_"
                "then_same_tissue_quota_reassignment",
            )
            self.assertEqual(
                result["component_quota_reassignment_policy"],
                "unplaceable_component_quota_to_same_tissue_probnet_mass_tail",
            )
            self.assertEqual(result["nucleus_spacing_margin_px"], 1)
            self.assertEqual(
                result["instance_connectivity_policy"],
                "largest_8_connected_component_after_transform",
            )
            self.assertTrue(result["sampling_audit"]["passed"])
            self.assertEqual(result["sampling_audit_max_attempts"], 3)
            self.assertEqual(
                result["sampling_feedback"]["policy"],
                "reason_directed_gamma_then_seed_v1",
            )
            self.assertEqual(
                result["shape_policy"],
                "component_local_same_class_reference_then_"
                "component_calibrated_library",
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
    def test_cross_only_route_cannot_fallback_to_inpaint(self):
        tissue = np.ones((16, 16), dtype=np.uint8)
        modes = []
        route = AgenticRoutingDecision(
            primary_mode="cross",
            candidate_modes=("cross",),
            confidence=0.90,
            reason="large generation support requires cross generation",
            features=AgenticRouteFeatures(
                change_ratio_image=0.65,
                change_ratio_tissue=0.65,
                component_count=1,
                largest_component_fraction=1.0,
                bbox_fraction=0.70,
                transition_count=0,
                changed_tissue_ids_from=(),
                changed_tissue_ids_to=(),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
                modes.append(mode)
                image_path = attempt_dir / "generated.png"
                Image.new("RGB", (16, 16), "white").save(image_path)
                return GenerationArtifact(mode=mode, image_path=image_path)

            result = run_agentic_workflow(
                reference_tissue_mask=tissue,
                target_tissue_mask=tissue,
                output_dir=tmp,
                generate=generate,
                verify=lambda _artifact: VerificationResult(
                    passed=False,
                    score=0.40,
                    quality_score=0.40,
                    evidence_coverage=0.20,
                    metrics={},
                    failed_checks=("evidence_coverage",),
                    scientific_status="evaluator_uncertain",
                ),
                routing_decision=route,
            )

        self.assertEqual(result.status, "evaluator_uncertain")
        self.assertEqual(modes, ["cross-v1-no-ip-pix2pix-v2"])
        self.assertEqual(result.selected_attempt.requested_mode, modes[0])

    def test_authoritative_joint_route_prevents_cell_only_noop(self):
        tissue = np.ones((16, 16), dtype=np.uint8)
        modes = []
        route = AgenticRoutingDecision(
            primary_mode="inpaint",
            candidate_modes=("inpaint", "cross"),
            confidence=0.90,
            reason="approved joint handoff",
            features=AgenticRouteFeatures(
                change_ratio_image=0.02,
                change_ratio_tissue=0.02,
                component_count=2,
                largest_component_fraction=0.5,
                bbox_fraction=0.1,
                transition_count=0,
                changed_tissue_ids_from=(),
                changed_tissue_ids_to=(),
            ),
        )

        with tempfile.TemporaryDirectory() as tmp:
            def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
                modes.append(mode)
                image_path = attempt_dir / "generated.png"
                Image.new("RGB", (16, 16), "white").save(image_path)
                return GenerationArtifact(mode=mode, image_path=image_path)

            result = run_agentic_workflow(
                reference_tissue_mask=tissue,
                target_tissue_mask=tissue,
                output_dir=tmp,
                generate=generate,
                verify=lambda _artifact: VerificationResult(
                    passed=True,
                    score=0.82,
                    metrics={},
                    scientific_status="validated",
                ),
                routing_decision=route,
            )

        self.assertEqual(result.status, "validated_first_pass")
        self.assertEqual(modes, ["inpaint"])

    def test_first_candidate_passes_without_running_alternate_backend(self):
        reference = np.ones((16, 16), dtype=np.uint8)
        target = reference.copy()
        target[4:6, 4:6] = 2
        modes = []

        with tempfile.TemporaryDirectory() as tmp:
            def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
                modes.append(mode)
                image_path = attempt_dir / "generated.png"
                Image.new("RGB", (16, 16), "white").save(image_path)
                return GenerationArtifact(mode=mode, image_path=image_path)

            result = run_agentic_workflow(
                reference_tissue_mask=reference,
                target_tissue_mask=target,
                output_dir=tmp,
                generate=generate,
                verify=lambda _artifact: VerificationResult(
                    passed=True,
                    score=0.80,
                    quality_score=0.80,
                    evidence_coverage=1.0,
                    component_scores={
                        "semantic": 0.80,
                        "preservation": 0.90,
                    },
                    metrics={},
                    scientific_status="validated",
                ),
            )

        self.assertEqual(result.status, "validated_first_pass")
        self.assertEqual(modes, ["inpaint"])

    def test_evaluator_abstention_still_runs_alternate_backend(self):
        reference = np.ones((16, 16), dtype=np.uint8)
        target = reference.copy()
        target[4:6, 4:6] = 2
        modes = []

        with tempfile.TemporaryDirectory() as tmp:
            def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
                modes.append(mode)
                image_path = attempt_dir / "generated.png"
                Image.new("RGB", (16, 16), "white").save(image_path)
                return GenerationArtifact(mode=mode, image_path=image_path)

            def verify(artifact: GenerationArtifact) -> VerificationResult:
                if artifact.mode == "inpaint":
                    return VerificationResult(
                        passed=False,
                        score=0.80,
                        quality_score=0.80,
                        evidence_coverage=0.50,
                        metrics={},
                        failed_checks=(
                            "semantic_evaluator_source_calibration",
                            "evidence_coverage",
                        ),
                        scientific_status="evaluator_uncertain",
                    )
                return VerificationResult(
                    passed=True,
                    score=0.80,
                    quality_score=0.80,
                    evidence_coverage=1.0,
                    metrics={},
                    scientific_status="validated",
                )

            result = run_agentic_workflow(
                reference_tissue_mask=reference,
                target_tissue_mask=target,
                output_dir=tmp,
                generate=generate,
                verify=verify,
            )

        self.assertEqual(result.status, "recovered")
        self.assertEqual(
            modes,
            ["inpaint", "cross-v1-no-ip-pix2pix-v2"],
        )
        self.assertIn(
            "evaluator_uncertainty_comparison",
            result.attempts[1].decision_reason,
        )

    def test_two_failed_candidates_use_frozen_selection_order(self):
        reference = np.ones((16, 16), dtype=np.uint8)
        target = reference.copy()
        target[4:6, 4:6] = 2

        with tempfile.TemporaryDirectory() as tmp:
            def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
                image_path = attempt_dir / "generated.png"
                Image.new("RGB", (16, 16), "white").save(image_path)
                return GenerationArtifact(mode=mode, image_path=image_path)

            def verify(artifact: GenerationArtifact) -> VerificationResult:
                cross = artifact.mode != "inpaint"
                return VerificationResult(
                    passed=False,
                    score=0.72 if cross else 0.61,
                    quality_score=0.72 if cross else 0.61,
                    evidence_coverage=1.0,
                    component_scores={
                        "semantic": 0.65 if cross else 0.80,
                        "preservation": 0.90,
                    },
                    metrics={},
                    failed_checks=("quality_score",),
                    scientific_status="needs_review",
                )

            result = run_agentic_workflow(
                reference_tissue_mask=reference,
                target_tissue_mask=target,
                output_dir=tmp,
                generate=generate,
                verify=verify,
            )

        self.assertEqual(result.status, "needs_review")
        self.assertEqual(
            result.selected_attempt.requested_mode,
            "cross-v1-no-ip-pix2pix-v2",
        )

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

    def test_cli_bounds_unrequested_generation_only_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "reference.png"
            source_tissue = root / "source_tissue.png"
            target_tissue = root / "target_tissue.png"
            nuclei = root / "nuclei.png"
            semantic_region = root / "semantic_region.png"
            generation_region = root / "generation_region.png"

            Image.new("RGB", (64, 64), "white").save(image)
            source = np.ones((64, 64), dtype=np.uint8)
            target = source.copy()
            target[24:40, 24:40] = 2
            semantic = source != target
            Image.fromarray(source).save(source_tissue)
            Image.fromarray(target).save(target_tissue)
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(nuclei)
            Image.fromarray(semantic.astype(np.uint8) * 255).save(semantic_region)
            Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(
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

            loaded = _load_and_validate_inputs(args)

            semantic_pixels = int(np.count_nonzero(semantic))
            self.assertEqual(
                int(np.count_nonzero(loaded["generation_change_region"])),
                semantic_pixels * 2,
            )
            self.assertTrue(
                loaded["generation_region_policy"]["capped"]
            )

    def test_cli_discovers_wider_generation_context_for_cord_primitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "reference.png"
            source_tissue = root / "source_tissue.png"
            target_tissue = root / "target_tissue.png"
            nuclei = root / "nuclei.png"
            semantic_region = root / "semantic_region.png"
            generation_region = root / "generation_region.png"

            Image.new("RGB", (64, 64), "white").save(image)
            source = np.ones((64, 64), dtype=np.uint8)
            target = source.copy()
            target[24:40, 24:40] = 2
            semantic = source != target
            Image.fromarray(source).save(source_tissue)
            Image.fromarray(target).save(target_tissue)
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(nuclei)
            Image.fromarray(semantic.astype(np.uint8) * 255).save(
                semantic_region
            )
            Image.fromarray(np.full((64, 64), 255, dtype=np.uint8)).save(
                generation_region
            )
            (root / "input_case_context.json").write_text(
                json.dumps(
                    {"primitive_id": "invasive-cord-formation-v1"}
                ),
                encoding="utf-8",
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

            loaded = _load_and_validate_inputs(args)

            semantic_pixels = int(np.count_nonzero(semantic))
            self.assertEqual(
                int(np.count_nonzero(loaded["generation_change_region"])),
                semantic_pixels * 5 // 2,
            )
            self.assertEqual(
                loaded["generation_region_policy"]["primitive_id"],
                "invasive-cord-formation-v1",
            )
            self.assertEqual(
                loaded["generation_region_policy"]["primitive_id_source"],
                "adjacent_case_context",
            )

    def test_cli_preserves_glas_whole_component_generation_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "reference.png"
            source_tissue = root / "source_tissue.png"
            target_tissue = root / "target_tissue.png"
            nuclei = root / "nuclei.png"
            semantic_region = root / "semantic_region.png"
            generation_region = root / "generation_region.png"

            Image.new("RGB", (64, 64), "white").save(image)
            source = np.full((64, 64), 2, dtype=np.uint8)
            source[8:56, 8:56] = 12
            target = source.copy()
            target[24:40, 24:40] = 2
            semantic = source != target
            generation = np.zeros((64, 64), dtype=np.uint8)
            generation[8:56, 8:56] = 255
            Image.fromarray(source).save(source_tissue)
            Image.fromarray(target).save(target_tissue)
            Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(nuclei)
            Image.fromarray(semantic.astype(np.uint8) * 255).save(semantic_region)
            Image.fromarray(generation).save(generation_region)

            args = build_agentic_parser().parse_args(
                [
                    "--profile",
                    "GlaS",
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
                int(np.count_nonzero(loaded["generation_change_region"])),
                48 * 48,
            )
            self.assertFalse(
                loaded["generation_region_policy"]["capped"]
            )

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

            self.assertEqual(result.status, "recovered")
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

            self.assertEqual(result.status, "recovered")
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
            self.assertEqual(result.status, "recovered")

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

    def test_g2_verifier_uses_source_prediction_and_target_supported_macro_iou(self):
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
        self.assertEqual(result.metrics["changed_region_macro_iou"], 0.75)

    def test_nuclei_gate_uses_instances_and_keeps_occupied_area_as_diagnostic(self):
        tissue = np.ones((16, 16), dtype=np.uint8)
        change = np.ones_like(tissue, dtype=bool)
        target_nuclei = np.zeros_like(tissue)
        predicted_nuclei = np.zeros_like(tissue)
        target_nuclei[1:6, 1:6] = 101
        target_nuclei[9:15, 9:15] = 101
        predicted_nuclei[2, 2] = 101
        predicted_nuclei[11, 11] = 101

        result = verify_mask_fidelity(
            reference_tissue_mask=tissue,
            target_tissue_mask=tissue,
            predicted_tissue_mask=tissue,
            change_region=change,
            target_nuclei_mask=target_nuclei,
            predicted_nuclei_mask=predicted_nuclei,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["nuclei_count_relative_error"], 0.0)
        self.assertEqual(result.metrics["nuclei_density_relative_error"], 0.0)
        self.assertGreater(
            result.metrics["nuclei_occupied_area_relative_error"],
            0.9,
        )

    def test_nuclei_detection_and_type_errors_are_separate(self):
        tissue = np.ones((16, 16), dtype=np.uint8)
        change = np.ones_like(tissue, dtype=bool)

        result = verify_mask_fidelity(
            reference_tissue_mask=tissue,
            target_tissue_mask=tissue,
            predicted_tissue_mask=tissue,
            change_region=change,
            target_nuclei_mask=np.zeros_like(tissue),
            predicted_nuclei_mask=np.zeros_like(tissue),
            target_nuclei_instance_counts={101: 9, 102: 21, 103: 20},
            predicted_nuclei_instance_counts={102: 24, 103: 25},
        )

        self.assertAlmostEqual(
            result.metrics["nuclei_detection_count_relative_error"],
            0.02,
        )
        self.assertAlmostEqual(
            result.metrics["nuclei_type_composition_tv_error"],
            0.18,
        )
        self.assertNotIn(
            "nuclei_detection_count_relative_error",
            result.failed_checks,
        )
        self.assertNotIn("nuclei_type_composition_error", result.failed_checks)

    def test_small_semantic_edit_uses_boundary_eroded_core(self):
        reference = np.ones((256, 256), dtype=np.uint8)
        target = reference.copy()
        target[80:128, 80:128] = 2
        change = target != reference
        predicted = reference.copy()
        predicted[84:124, 84:124] = 2

        result = verify_mask_fidelity(
            reference_tissue_mask=reference,
            target_tissue_mask=target,
            predicted_tissue_mask=predicted,
            change_region=change,
        )

        self.assertLess(result.metrics["changed_region_accuracy"], 0.70)
        self.assertEqual(result.metrics["semantic_small_region"], 1.0)
        self.assertEqual(result.metrics["semantic_gate_accuracy"], 1.0)
        self.assertNotIn("changed_region_accuracy", result.failed_checks)

    def test_cellvit_instance_counts_use_local_centroids_and_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "patch.png"
            cells_path = root / "cells.json"
            Image.new("RGB", (8, 8), "white").save(image_path)
            cells_path.write_text(
                json.dumps(
                    {
                        "wsi_metadata": {},
                        "cells": [
                            {
                                "type": 1,
                                "contour": [[1, 1], [3, 1], [3, 3], [1, 3]],
                            },
                            {
                                "type": 2,
                                "contour": [[5, 1], [7, 1], [7, 3], [5, 3]],
                            },
                            {
                                "type": 3,
                                "contour": [[1, 5], [3, 5], [3, 7], [1, 7]],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            region = np.zeros((8, 8), dtype=bool)
            region[:, :4] = True

            counts = cellvit_instance_counts_in_region(
                cells_path,
                image_path,
                region,
            )

        self.assertEqual(counts, {101: 1, 103: 1})

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
            cellvit = root / "cellvit.pt"
            for checkpoint in (pix2pix, cellvit):
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
