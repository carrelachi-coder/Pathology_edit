#!/usr/bin/env python3
"""Run the bounded pathology edit agent: route, generate, verify, and recover."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.inference.agentic import (
    AgenticWorkflowConfig,
    FidelityThresholds,
    GenerationArtifact,
    VerificationResult,
    run_agentic_workflow,
    semantic_mask_instance_counts,
    verify_mask_fidelity,
)
from controlnet_train.inference.router import (
    AgenticRoutingConfig,
    AgenticRoutingDecision,
)
from controlnet_train.inference.model_paths import (
    DEFAULT_CELLVIT_MODEL,
    DEFAULT_CELLVIT_PYTHON,
    DEFAULT_CELLVIT_ROOT,
    DEFAULT_CROSS_V1_CHECKPOINT,
    DEFAULT_INPAINT_CHECKPOINT,
    DEFAULT_PIX2PIX_CHECKPOINT,
    validate_frozen_cellvit_checkpoint,
    validate_production_controlnet_checkpoint,
    validate_production_pix2pix_checkpoint,
)
from phase3_mask_edit.core.mask_io import (
    load_change_region,
    load_id_mask,
    save_change_region,
    save_id_mask,
)
from phase3_mask_edit.core.gland_region import bound_generation_context_region
from phase3_mask_edit.audit import (
    OnlineAuditPolicy,
    OnlineSemanticAuditor,
    QualityPolicy,
    SemanticPrediction,
    dataset_native_metric_class_ids,
    evaluate_product_quality,
    profile_supports_fine,
    source_evaluator_quality,
    to_coarse_mask,
    write_generation_report,
)
from segmentator.release import load_segmentator_release
from scripts.run_phase3_inpaint_pipeline import (
    _load_rgb_image,
    _load_uint8_mask,
    _release_generation_model_caches,
    _run_generation_stage,
    _validate_same_size,
)
from scripts.run_cellvit_single_patch import cellvit_instance_counts_in_region


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED_MODEL = "/data/huggingface/FLUX.1-dev"
DEFAULT_SEGMENTATOR_RELEASE = (
    REPO_ROOT
    / "benchmark_configs"
    / "releases"
    / "segmentator_fine_legacy_anchor.json"
)
DEFAULT_ONLINE_PRODUCT_RELEASE = (
    REPO_ROOT
    / "benchmark_configs"
    / "releases"
    / "online_agent_product_v1.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded agentic pathology edit workflow over already prepared target "
            "tissue/nuclei masks. The workflow routes to inpaint or production cross-v1 "
            "+ pix2pix-v2, re-segments the output, and tries at most one fallback by default."
        )
    )
    parser.add_argument("--profile", required=True, help="Dataset/profile name, e.g. BCSS.")
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--reference-tissue-mask", required=True, type=Path)
    parser.add_argument("--reference-nuclei-mask", required=True, type=Path)
    parser.add_argument("--target-tissue-mask", required=True, type=Path)
    parser.add_argument("--target-nuclei-mask", required=True, type=Path)
    parser.add_argument(
        "--joint-generation-handoff",
        type=Path,
        help=(
            "Approved joint-generation-handoff-v2/v3 manifest. When supplied, "
            "its hash-locked generation support is authoritative for routing "
            "and is preserved without legacy context truncation."
        ),
    )
    parser.add_argument(
        "--nuclei-generation-log",
        type=Path,
        help=(
            "Cell-stage provenance JSON. When supplied, validate the target "
            "nuclei against the frozen online product sampling contract."
        ),
    )
    parser.add_argument(
        "--product-release",
        type=Path,
        default=DEFAULT_ONLINE_PRODUCT_RELEASE,
        help="Online product release manifest defining the nuclei sampling contract.",
    )
    semantic_region_group = parser.add_mutually_exclusive_group()
    semantic_region_group.add_argument(
        "--semantic-change-region",
        type=Path,
        help=(
            "Binary region used by the verifier; defaults to "
            "reference_tissue != target_tissue."
        ),
    )
    semantic_region_group.add_argument(
        "--change-region",
        type=Path,
        help=(
            "Deprecated alias for --semantic-change-region. Retained for "
            "backward compatibility."
        ),
    )
    parser.add_argument(
        "--generation-change-region",
        type=Path,
        help=(
            "Binary region erased/conditioned by the generator. Defaults to the "
            "semantic change region and may be a wider superset for GlaS or thin edits."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--pretrained-model-name-or-path", default=DEFAULT_PRETRAINED_MODEL)
    parser.add_argument("--inpaint-checkpoint", type=Path, default=Path(DEFAULT_INPAINT_CHECKPOINT))
    parser.add_argument("--cross-v1-checkpoint", type=Path, default=Path(DEFAULT_CROSS_V1_CHECKPOINT))
    parser.add_argument("--pix2pix-checkpoint", type=Path, default=Path(DEFAULT_PIX2PIX_CHECKPOINT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-source", choices=("metadata", "dataset"), default="dataset")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--t-inpaint",
        type=float,
        default=0.12,
        help=(
            "Maximum generation-support fraction for high-confidence Inpaint "
            "routing on an approved joint handoff."
        ),
    )
    parser.add_argument(
        "--t-cross",
        type=float,
        default=0.30,
        help=(
            "Legacy tissue-normalized threshold used by the generic image "
            "router when no approved joint handoff is supplied."
        ),
    )
    parser.add_argument(
        "--force-cross-generation-support-fraction",
        type=float,
        default=0.50,
        help=(
            "Generation-support fraction G at or above which an approved "
            "joint handoff is Cross-only."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--reuse-existing-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse a completed attempt image and generation metadata when an "
            "interrupted run resumes in the same output directory."
        ),
    )
    parser.add_argument(
        "--inject-verifier-failure-attempt",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--inject-verifier-failed-check",
        choices=(
            "changed_region_accuracy",
            "changed_region_macro_iou",
            "off_target_drift",
            "nuclei_count_relative_error",
            "nuclei_detection_count_relative_error",
            "nuclei_type_composition_error",
            "nuclei_density_relative_error",
        ),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--inject-verifier-failed-check-attempt",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--changed-region-accuracy-min", type=float, default=0.70)
    parser.add_argument("--changed-region-macro-iou-min", type=float, default=0.55)
    parser.add_argument("--off-target-drift-max", type=float, default=0.08)
    parser.add_argument("--quality-score-min", type=float, default=0.75)
    parser.add_argument("--evidence-coverage-min", type=float, default=0.80)
    parser.add_argument("--semantic-score-min", type=float, default=0.60)
    parser.add_argument("--source-boundary-f1-min", type=float, default=0.45)
    parser.add_argument("--boundary-support-min-pixels", type=int, default=256)
    parser.add_argument(
        "--nuclei-count-relative-error-max",
        "--nuclei-density-relative-error-max",
        dest="nuclei_count_relative_error_max",
        type=float,
        default=0.35,
        help=(
            "Maximum relative error in the total CellViT instance count. "
            "The legacy density flag is retained as a CLI alias."
        ),
    )
    parser.add_argument(
        "--nuclei-type-composition-error-max",
        type=float,
        default=0.35,
        help="Maximum total-variation error in typed nucleus proportions.",
    )
    parser.add_argument(
        "--nuclei-type-min-instances",
        type=int,
        default=10,
        help="Minimum target and detected nuclei needed to enforce the type gate.",
    )
    parser.add_argument(
        "--semantic-postprocess-mode",
        choices=("off", "shadow", "enforce"),
        default="shadow",
        help=(
            "Run conservative P1 off, as a non-decision shadow, or as the "
            "verification decision mask. Shadow is the safe product default."
        ),
    )

    parser.add_argument(
        "--segmentator-release",
        type=Path,
        default=DEFAULT_SEGMENTATOR_RELEASE,
        help="Frozen Segmentator release JSON/YAML used for strict G2 inference.",
    )
    parser.add_argument(
        "--segmentator-checkpoint",
        type=Path,
        default=None,
        help="Legacy checkpoint override; disables release-driven architecture reconstruction.",
    )
    parser.add_argument("--segmentator-env", default="pathology-segmentator-mmseg")
    parser.add_argument(
        "--segmentator-python",
        type=Path,
        help="Run segmentator with this Python instead of `conda run -n SEGMENTATOR_ENV`.",
    )
    parser.add_argument("--segmentator-decoder", choices=("upernet", "mask2former"), default="mask2former")
    parser.add_argument("--segmentator-device", default="cuda:1")
    parser.add_argument("--cellvit-model", type=Path, default=Path(DEFAULT_CELLVIT_MODEL))
    parser.add_argument("--cellvit-root", type=Path, default=Path(DEFAULT_CELLVIT_ROOT))
    parser.add_argument(
        "--cellvit-script",
        type=Path,
        default=REPO_ROOT / "scripts" / "run_cellvit_single_patch.py",
        help="CellViT wrapper used for generated-image nuclei verification.",
    )
    parser.add_argument(
        "--cellvit-launch-python",
        type=Path,
        default=Path(sys.executable),
        help="Python used to launch scripts/run_cellvit_single_patch.py.",
    )
    parser.add_argument(
        "--cellvit-python",
        type=Path,
        default=Path(DEFAULT_CELLVIT_PYTHON),
        help="Python used by the CellViT wrapper to run upstream CellViT code.",
    )
    parser.add_argument("--cellvit-gpu", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = _load_and_validate_inputs(args)
    joint_handoff = _validate_joint_generation_handoff(args, inputs)
    if joint_handoff is not None:
        args.prompt = joint_handoff["compiled_prompt"]
    nuclei_generation = _validate_nuclei_generation_contract(args)
    image_generation = _validate_image_generation_contract(args)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_change_region_path = save_change_region(
        inputs["semantic_change_region"],
        output_dir / "semantic_change_region.png",
    )
    generation_change_region_path = save_change_region(
        inputs["generation_change_region"],
        output_dir / "generation_change_region.png",
    )

    generation_args = _generation_namespace(args)
    thresholds = FidelityThresholds(
        changed_region_accuracy_min=args.changed_region_accuracy_min,
        changed_region_macro_iou_min=args.changed_region_macro_iou_min,
        off_target_drift_max=args.off_target_drift_max,
        quality_score_min=args.quality_score_min,
        evidence_coverage_min=args.evidence_coverage_min,
        semantic_score_min=args.semantic_score_min,
        source_boundary_f1_min=args.source_boundary_f1_min,
        boundary_support_min_pixels=args.boundary_support_min_pixels,
        nuclei_count_relative_error_max=args.nuclei_count_relative_error_max,
        nuclei_type_composition_error_max=(
            args.nuclei_type_composition_error_max
        ),
        nuclei_type_min_instances=args.nuclei_type_min_instances,
    )
    quality_policy = QualityPolicy(
        quality_score_min=thresholds.quality_score_min,
        evidence_coverage_min=thresholds.evidence_coverage_min,
        semantic_score_min=thresholds.semantic_score_min,
        source_region_accuracy_min=thresholds.changed_region_accuracy_min,
        source_macro_miou_min=thresholds.changed_region_macro_iou_min,
        source_transition_recall_min=thresholds.changed_region_accuracy_min,
        target_reference_min_pixels=256,
        target_reference_recall_min=thresholds.changed_region_macro_iou_min,
        source_to_target_confusion_max=(
            1.0 - thresholds.changed_region_accuracy_min
        ),
        semantic_core_min_pixels=thresholds.semantic_core_min_pixels,
        semantic_core_min_fraction=thresholds.semantic_core_min_fraction,
        source_boundary_f1_min=thresholds.source_boundary_f1_min,
        boundary_support_min_pixels=thresholds.boundary_support_min_pixels,
        off_target_drift_max=thresholds.off_target_drift_max,
        nuclei_detection_error_max=thresholds.nuclei_count_relative_error_max,
        nuclei_type_error_max=(
            thresholds.nuclei_type_composition_error_max
        ),
        nuclei_min_instances=thresholds.nuclei_type_min_instances,
    )
    quality_policy.validate()
    target_nuclei_instance_counts = semantic_mask_instance_counts(
        inputs["target_nuclei"],
        region=inputs["semantic_change_region"],
    )
    source_segmentator = None
    source_quality = None
    source_semantic_prediction = None
    source_nuclei_calibration = None
    semantic_auditor = OnlineSemanticAuditor(
        OnlineAuditPolicy(postprocess_mode=args.semantic_postprocess_mode)
    )
    if np.any(inputs["semantic_change_region"]):
        _validate_verification_runtime(args)
        source_segmentator = _run_segmentator(
            args=args,
            image_path=args.reference_image.resolve(),
            output_dir=output_dir / "source_verification",
        )
        source_semantic_prediction = _load_semantic_prediction(source_segmentator)
        source_quality = source_evaluator_quality(
            source_mask=inputs["reference_coarse_tissue"],
            source_prediction=source_semantic_prediction.mask,
            source_probabilities=source_semantic_prediction.probabilities,
            class_ids=dataset_native_metric_class_ids(
                args.profile, level="coarse"
            ),
        )
        _write_json(
            output_dir / "source_verification" / "evaluator_quality.json",
            source_quality,
        )
        changed_region_coarse_quality = _source_region_quality_or_abstain(
            level="coarse",
            source_mask=inputs["reference_coarse_tissue"],
            source_prediction=source_semantic_prediction.mask,
            source_probabilities=source_semantic_prediction.probabilities,
            class_ids=dataset_native_metric_class_ids(
                args.profile, level="coarse"
            ),
            region=inputs["semantic_change_region"],
        )
        changed_region_quality: dict[str, Any] = {
            "coarse": changed_region_coarse_quality
        }
        if profile_supports_fine(args.profile):
            source_fine_prediction = _load_fine_prediction(source_segmentator)
            if source_fine_prediction is None:
                raise RuntimeError(
                    f"{args.profile} requires source fine evaluator artifacts"
                )
            changed_region_quality["fine"] = (
                _source_region_quality_or_abstain(
                    level="fine",
                    source_mask=inputs["reference_tissue"],
                    source_prediction=source_fine_prediction.mask,
                    source_probabilities=source_fine_prediction.probabilities,
                    class_ids=dataset_native_metric_class_ids(
                        args.profile, level="fine"
                    ),
                    region=inputs["semantic_change_region"],
                )
            )
        _write_json(
            output_dir
            / "source_verification"
            / "changed_region_evaluator_quality.json",
            changed_region_quality,
        )
        source_cellvit_dir = output_dir / "source_nuclei_verification"
        source_predicted_nuclei_path = _run_cellvit(
            args=args,
            image_path=args.reference_image.resolve(),
            output_dir=source_cellvit_dir,
        )
        full_region = np.ones_like(
            inputs["semantic_change_region"],
            dtype=bool,
        )
        source_nuclei_calibration = {
            "changed_region": {
                "reference": semantic_mask_instance_counts(
                    inputs["reference_nuclei"],
                    region=inputs["semantic_change_region"],
                ),
                "predicted": _cellvit_counts_from_wrapper_summary(
                    predicted_nuclei_path=source_predicted_nuclei_path,
                    image_path=args.reference_image.resolve(),
                    region=inputs["semantic_change_region"],
                ),
            },
            "full_image": {
                "reference": semantic_mask_instance_counts(
                    inputs["reference_nuclei"],
                    region=full_region,
                ),
                "predicted": _cellvit_counts_from_wrapper_summary(
                    predicted_nuclei_path=source_predicted_nuclei_path,
                    image_path=args.reference_image.resolve(),
                    region=full_region,
                ),
            },
        }
        _write_json(
            source_cellvit_dir / "evaluator_calibration_counts.json",
            source_nuclei_calibration,
        )

    previous_generation_mode: str | None = None

    def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
        nonlocal previous_generation_mode
        generation_mode = _generation_backend_mode(mode)
        if (
            previous_generation_mode is not None
            and previous_generation_mode != generation_mode
        ):
            _release_generation_model_caches()
        previous_generation_mode = generation_mode
        _validate_generation_runtime(args, generation_mode)
        existing_image = attempt_dir / "generated_image.png"
        existing_metadata = attempt_dir / "generation_info.json"
        if (
            args.reuse_existing_generation
            and existing_image.is_file()
            and existing_metadata.is_file()
        ):
            metadata = json.loads(existing_metadata.read_text(encoding="utf-8"))
            metadata = {
                **metadata,
                "resumed_generation": True,
                "resumed_image_path": str(existing_image),
            }
            return GenerationArtifact(
                mode=mode,
                image_path=existing_image,
                metadata=metadata,
            )
        attempt_args = SimpleNamespace(**vars(generation_args))
        attempt_args.generation_mode = generation_mode
        image_path, metadata = _run_generation_stage(
            args=attempt_args,
            output_dir=attempt_dir,
            reference_image=inputs["reference_image"],
            change_region=inputs["generation_change_region"],
            target_tissue_path=args.target_tissue_mask.resolve(),
            target_nuclei_path=args.target_nuclei_mask.resolve(),
        )
        return GenerationArtifact(mode=mode, image_path=image_path, metadata=metadata)

    def verify(artifact: GenerationArtifact):
        _prepare_verification_runtime(args)
        verification_dir = artifact.image_path.parent / "verification"
        verification_dir.mkdir(parents=True, exist_ok=True)
        attempt_index = _attempt_index_from_path(artifact.image_path.parent)
        injection_marker = verification_dir / ".injected_failure_consumed"
        if (
            args.inject_verifier_failure_attempt == attempt_index
            and not injection_marker.exists()
        ):
            injection_marker.write_text(
                "intentional canary fault after generation\n", encoding="utf-8"
            )
            raise RuntimeError(
                f"injected verifier failure for attempt {attempt_index}"
            )
        predicted_tissue = _run_segmentator(
            args=args,
            image_path=artifact.image_path,
            output_dir=verification_dir,
        )
        predicted_nuclei_path = _run_cellvit(
            args=args,
            image_path=artifact.image_path,
            output_dir=verification_dir,
        )
        predicted_nuclei_instance_counts = _cellvit_counts_from_wrapper_summary(
            predicted_nuclei_path=predicted_nuclei_path,
            image_path=artifact.image_path,
            region=inputs["semantic_change_region"],
        )
        generated_semantic_prediction = _load_semantic_prediction(predicted_tissue)
        online_audit = semantic_auditor.audit(
            source_mask=inputs["reference_coarse_tissue"],
            target_mask=inputs["target_coarse_tissue"],
            source_prediction=source_semantic_prediction,
            generated_prediction=generated_semantic_prediction,
            class_ids=dataset_native_metric_class_ids(
                args.profile, level="coarse"
            ),
            semantic_change_region=inputs["semantic_change_region"],
            preservation_exclusion_region=inputs[
                "generation_change_region"
            ],
            **_fine_audit_inputs(
                args=args,
                inputs=inputs,
                source_artifacts=source_segmentator,
                generated_artifacts=predicted_tissue,
            ),
        )
        decision_tissue = online_audit.decision_mask
        raw_mask_path = save_id_mask(
            generated_semantic_prediction.mask,
            verification_dir / "coarse_mask_raw.png",
        )
        audited_mask_path = None
        p1_changed_path = None
        if online_audit.p1_result is not None:
            audited_mask_path = save_id_mask(
                online_audit.p1_result.audited_mask,
                verification_dir / "coarse_mask_p1.png",
            )
            p1_changed_path = save_change_region(
                online_audit.p1_result.changed_mask,
                verification_dir / "p1_changed_pixels.png",
            )
        online_audit_metadata = online_audit.to_metadata()
        online_audit_metadata["artifacts"] = {
            "raw_mask": str(raw_mask_path),
            "p1_mask": (
                None if audited_mask_path is None else str(audited_mask_path)
            ),
            "p1_changed_pixels": (
                None if p1_changed_path is None else str(p1_changed_path)
            ),
        }
        _write_json(
            verification_dir / "online_semantic_audit.json",
            online_audit_metadata,
        )
        result = verify_mask_fidelity(
            reference_tissue_mask=inputs["reference_coarse_tissue"],
            target_tissue_mask=inputs["target_coarse_tissue"],
            predicted_tissue_mask=decision_tissue,
            source_predicted_tissue_mask=source_semantic_prediction.mask,
            change_region=inputs["semantic_change_region"],
            target_nuclei_mask=inputs["target_nuclei"],
            predicted_nuclei_mask=_load_uint8_mask(predicted_nuclei_path),
            target_nuclei_instance_counts=target_nuclei_instance_counts,
            predicted_nuclei_instance_counts=predicted_nuclei_instance_counts,
            thresholds=thresholds,
            enforce_off_target_drift=True,
        )
        decision_metrics = (
            online_audit.p1_metrics
            if online_audit.decision_input == "p1_audited"
            else online_audit.raw_metrics
        )
        if decision_metrics is None or source_quality is None:
            raise RuntimeError("product quality evaluator inputs are incomplete")
        quality = evaluate_product_quality(
            coarse_metrics=decision_metrics,
            source_quality=online_audit.source_quality,
            base_metrics=result.metrics,
            source_nuclei_calibration=source_nuclei_calibration,
            target_nuclei_counts=target_nuclei_instance_counts,
            generated_nuclei_counts=predicted_nuclei_instance_counts,
            policy=quality_policy,
        )
        result = VerificationResult(
            passed=quality.passed,
            score=quality.quality_score,
            metrics=quality.metrics,
            failed_checks=quality.failed_checks,
            component_scores=quality.component_scores,
            applicability=quality.applicability,
            evidence_coverage=quality.evidence_coverage,
            quality_score=quality.quality_score,
            scientific_status=quality.scientific_status,
            reason_codes=quality.reason_codes,
        )
        if (
            args.inject_verifier_failed_check
            and attempt_index == args.inject_verifier_failed_check_attempt
        ):
            failed_checks = tuple(
                dict.fromkeys(
                    (
                        *result.failed_checks,
                        args.inject_verifier_failed_check,
                    )
                )
            )
            result = replace(
                result,
                passed=False,
                metrics={
                    **dict(result.metrics),
                    "injected_canary_failure": 1.0,
                },
                failed_checks=failed_checks,
                scientific_status="needs_review",
                reason_codes=tuple(
                    dict.fromkeys(
                        (*result.reason_codes, "injected_canary_failure")
                    )
                ),
            )
        _write_json(
            verification_dir / "verification.json",
            {
                "schema_version": result.schema_version,
                "passed": result.passed,
                "score": result.score,
                "quality_score": result.quality_score,
                "evidence_coverage": result.evidence_coverage,
                "component_scores": dict(result.component_scores),
                "applicability": dict(result.applicability),
                "scientific_status": result.scientific_status,
                "reason_codes": list(result.reason_codes),
                "metrics": dict(result.metrics),
                "failed_checks": list(result.failed_checks),
                "off_target_drift_enforced": True,
                "preservation_exclusion_region_policy": (
                    "full_generation_change_region"
                ),
                "preservation_exclusion_region": str(
                    generation_change_region_path
                ),
                "quality_policy": quality.to_metadata()["policy"],
                "source_segmentator": source_segmentator,
                "predicted_tissue": predicted_tissue,
                "predicted_nuclei_mask": str(predicted_nuclei_path),
                "target_nuclei_instance_counts": target_nuclei_instance_counts,
                "predicted_nuclei_instance_counts": predicted_nuclei_instance_counts,
                "source_nuclei_evaluator_calibration": (
                    source_nuclei_calibration
                ),
                "cellvit_release": args.cellvit_release,
                "online_semantic_audit": str(
                    verification_dir / "online_semantic_audit.json"
                ),
                "semantic_decision_input": online_audit.decision_input,
                "raw_audit_metrics": online_audit.raw_metrics,
                "p1_audit_metrics": online_audit.p1_metrics,
            },
        )
        return result

    workflow = run_agentic_workflow(
        reference_tissue_mask=inputs["reference_tissue"],
        target_tissue_mask=inputs["target_tissue"],
        output_dir=output_dir,
        generate=generate,
        verify=verify,
        config=AgenticWorkflowConfig(
            routing=AgenticRoutingConfig(t_inpaint=args.t_inpaint, t_cross=args.t_cross),
            max_attempts=args.max_attempts,
        ),
        routing_decision=(
            None if joint_handoff is None else joint_handoff["routing_decision"]
        ),
    )

    final_path = output_dir / "generated_image.png"
    if workflow.status == "noop":
        shutil.copy2(args.reference_image, final_path)
    elif workflow.selected_attempt and workflow.selected_attempt.artifact:
        shutil.copy2(workflow.selected_attempt.artifact.image_path, final_path)

    summary = workflow.to_metadata()
    summary["generated_image"] = str(final_path) if final_path.exists() else None
    summary["nuclei_generation"] = nuclei_generation
    summary["joint_generation_handoff"] = (
        None
        if joint_handoff is None
        else joint_handoff["summary"]
    )
    summary["image_generation_contract"] = image_generation
    summary["image_generation_provenance"] = (
        _selected_image_generation_provenance(summary)
    )
    report_json_path, report_markdown_path, generation_report = (
        write_generation_report(summary, output_dir=output_dir)
    )
    summary["generation_report"] = {
        "json": str(report_json_path),
        "markdown": str(report_markdown_path),
        "content": generation_report,
    }
    summary["online_self_audit"] = {
        "scope": "product_runtime",
        "benchmark_independent": True,
        "semantic_postprocess_mode": args.semantic_postprocess_mode,
        "verifier_policy_status": "frozen_engineering_policy",
        "policy_id": quality_policy.policy_id,
        "formal_validated": workflow.status
        in {"validated_first_pass", "recovered"},
        "engineering_status": workflow.status,
        "reason": (
            "Validation denotes a frozen automated engineering evaluator pass; "
            "it is not a clinical correctness claim."
        ),
    }
    summary["change_regions"] = {
        "semantic": str(semantic_change_region_path),
        "generation": str(generation_change_region_path),
        "semantic_pixels": int(
            np.count_nonzero(inputs["semantic_change_region"])
        ),
        "generation_pixels": int(
            np.count_nonzero(inputs["generation_change_region"])
        ),
        "semantic_matches_tissue_difference": bool(
            np.array_equal(
                inputs["semantic_change_region"],
                inputs["reference_tissue"] != inputs["target_tissue"],
            )
        ),
        "generation_context_policy": inputs["generation_region_policy"],
    }
    _write_json(output_dir / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return (
        0
        if workflow.status in {"validated_first_pass", "recovered", "noop"}
        else 2
    )


def _generation_backend_mode(agent_mode: str) -> str:
    if agent_mode == "inpaint":
        return "inpaint"
    if agent_mode in {
        "cross",
        "cross-v1",
        "cross-v1-no-ip-pix2pix-v2",
    }:
        return "cross-v1"
    raise ValueError(f"Unsupported agent generation mode: {agent_mode}")


def _prepare_verification_runtime(args: argparse.Namespace) -> None:
    _validate_verification_runtime(args)
    _release_generation_model_caches()


def _source_region_quality_or_abstain(
    *,
    level: str,
    source_mask: np.ndarray,
    source_prediction: np.ndarray,
    source_probabilities: np.ndarray,
    class_ids: Sequence[int],
    region: np.ndarray,
) -> dict[str, Any]:
    try:
        return source_evaluator_quality(
            source_mask=source_mask,
            source_prediction=source_prediction,
            source_probabilities=source_probabilities,
            class_ids=class_ids,
            region=region,
        )
    except ValueError as error:
        if "no evaluable pixels" not in str(error):
            raise
        return {
            "available": False,
            "reason": "no_dataset_native_evaluable_pixels",
            "interpretation": f"{level}_evaluator_abstained",
        }


def _selected_image_generation_provenance(
    workflow_summary: dict[str, Any],
) -> dict[str, Any]:
    selected = workflow_summary.get("selected_attempt") or {}
    artifact = selected.get("artifact") or {}
    metadata = artifact.get("metadata") or {}
    selected_mode = metadata.get("selected_mode") or artifact.get("mode")
    cross = metadata.get("cross_v1") or {}
    pix2pix = cross.get("pix2pix_v2") or {}
    protection = pix2pix.get("cross_rgb_od_low_stain_protection")
    if protection is None:
        protection = {
            "policy": "cross_rgb_od_low_stain_v1",
            "enabled": False,
            "applied": False,
            "status": (
                "not_applicable"
                if selected_mode not in {"cross", "cross-v1"}
                else "missing_from_selected_cross_artifact"
            ),
        }
    else:
        protection = {
            **protection,
            "status": (
                "applied"
                if protection.get("applied")
                else "enabled_no_supported_region"
            ),
        }
    return {
        "selected_attempt": selected.get("attempt_index"),
        "selected_mode": selected_mode,
        "cross_rgb_od_low_stain_protection": protection,
    }


def _validate_nuclei_generation_contract(
    args: argparse.Namespace,
) -> dict[str, Any]:
    release_path = Path(args.product_release)
    if not release_path.is_file():
        raise FileNotFoundError(f"Online product release not found: {release_path}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    expected = release["nuclei_generation"]
    result = {
        "product_release": str(release_path.resolve()),
        "release_id": release.get("release_id"),
        "expected_candidate_queue_policy": expected["candidate_queue_policy"],
        "expected_gamma": expected["gamma"],
        "expected_checkpoint_sha256": expected["checkpoint_sha256"],
        "expected_count_policy": expected["count_policy"],
        "expected_type_quota_routing_policy": expected[
            "type_quota_routing_policy"
        ],
        "expected_shape_policy": expected["shape_policy"],
    }
    if args.nuclei_generation_log is None:
        return {
            **result,
            "status": "not_provided_legacy_input",
            "validated": False,
        }

    log_path = Path(args.nuclei_generation_log)
    if not log_path.is_file():
        raise FileNotFoundError(f"Nuclei generation log not found: {log_path}")
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    if payload.get("mode") != "probnet":
        return {
            **result,
            "status": "non_probnet_cell_fill",
            "validated": False,
            "log": str(log_path.resolve()),
            "cell_fill_mode": payload.get("mode"),
        }

    sampling = payload.get("shape_sampling") or {}
    diagnostics = _probnet_diagnostics_contract(sampling)
    actual_policy = sampling.get("candidate_queue_policy")
    if actual_policy != expected["candidate_queue_policy"]:
        raise ValueError(
            "Target nuclei candidate queue policy does not match the online "
            f"product release: {actual_policy!r} != "
            f"{expected['candidate_queue_policy']!r}"
        )
    for field in (
        "candidate_quality_score",
        "candidate_diversity_score",
        "candidate_diversity_weight",
    ):
        actual_value = sampling.get(field, diagnostics.get(field))
        expected_value = expected.get(field)
        if not _contract_values_equal(actual_value, expected_value):
            raise ValueError(
                "Target nuclei spatial sampling does not match the online "
                f"product release for {field}: "
                f"{actual_value!r} != {expected_value!r}"
            )
    for field in (
        "quota_coverage_spacing_scale",
        "quota_coverage_max_radius",
        "retry_tail_policy",
        "component_quota_reassignment_policy",
        "count_policy",
        "type_quota_routing_policy",
        "shape_policy",
        "nucleus_spacing_margin_px",
        "instance_connectivity_policy",
        "source_nucleus_erasure_policy",
        "buffer_nucleus_policy",
    ):
        actual_value = sampling.get(field)
        expected_value = expected.get(field)
        if actual_value != expected_value:
            raise ValueError(
                "Target nuclei coverage contract does not match the online "
                f"product release for {field}: "
                f"{actual_value!r} != {expected_value!r}"
            )
    checkpoint = sampling.get("probnet_release") or {}
    actual_sha256 = checkpoint.get("sha256")
    if actual_sha256 != expected["checkpoint_sha256"]:
        raise ValueError(
            "Target nuclei ProbNet checkpoint does not match the online "
            f"product release: {actual_sha256!r} != "
            f"{expected['checkpoint_sha256']!r}"
        )
    if sampling.get("organ_specific_constraints") is not False:
        raise ValueError(
            "Target nuclei provenance must explicitly disable organ-specific "
            "placement constraints."
        )
    sampling_audit = sampling.get("sampling_audit") or diagnostics.get(
        "sampling_audit"
    )
    actual_audit_policy = sampling_audit.get("policy")
    audit_compatibility = None
    if actual_audit_policy != expected.get("sampling_audit_policy"):
        audit_compatibility = _validate_frozen_nuclei_replay_compatibility(
            args=args,
            log_path=log_path,
            actual_policy=actual_audit_policy,
            expected=expected,
        )
    if sampling_audit.get("organ_specific_constraints") is not False:
        raise ValueError(
            "Target nuclei sampling audit must be patch-relative and disable "
            "organ-specific constraints."
        )
    if sampling_audit.get("passed") is not True:
        raise ValueError(
            "Target nuclei did not pass the frozen count/type/spatial sampling audit."
        )
    sampling_feedback = sampling.get("sampling_feedback") or diagnostics.get(
        "sampling_feedback"
    )
    if audit_compatibility is None:
        _validate_nuclei_sampling_feedback(
            sampling=sampling,
            sampling_audit=sampling_audit,
            sampling_feedback=sampling_feedback,
            expected=expected,
            diagnostics=diagnostics,
        )
        expected_audit_attempts = expected.get("sampling_audit_attempts")
    else:
        expected_audit_attempts = audit_compatibility[
            "sampling_audit_max_attempts"
        ]
        if (
            not audit_compatibility["sampling_feedback_required"]
            and sampling_feedback is not None
        ):
            raise ValueError(
                "Frozen nuclei replay unexpectedly contains a feedback trace."
            )
    if not _contract_values_equal(
        sampling.get("sampling_audit_max_attempts"),
        expected_audit_attempts,
    ):
        raise ValueError(
            "Target nuclei sampling audit attempt budget does not match the "
            "applicable online product contract."
        )
    return {
        **result,
        "status": "validated",
        "validated": True,
        "log": str(log_path.resolve()),
        "candidate_queue_policy": actual_policy,
        "gamma": sampling.get("gamma", diagnostics.get("gamma")),
        "candidate_quality_score": sampling.get(
            "candidate_quality_score",
            diagnostics.get("candidate_quality_score"),
        ),
        "candidate_probability_mass_exponent": sampling.get(
            "candidate_probability_mass_exponent",
            diagnostics.get("candidate_probability_mass_exponent"),
        ),
        "candidate_diversity_score": sampling.get(
            "candidate_diversity_score",
            diagnostics.get("candidate_diversity_score"),
        ),
        "candidate_diversity_weight": sampling.get(
            "candidate_diversity_weight",
            diagnostics.get("candidate_diversity_weight"),
        ),
        "quota_coverage_spacing_scale": sampling[
            "quota_coverage_spacing_scale"
        ],
        "quota_coverage_max_radius": sampling["quota_coverage_max_radius"],
        "retry_tail_policy": sampling["retry_tail_policy"],
        "component_quota_reassignment_policy": sampling[
            "component_quota_reassignment_policy"
        ],
        "count_policy": sampling["count_policy"],
        "type_quota_routing_policy": sampling[
            "type_quota_routing_policy"
        ],
        "shape_policy": sampling["shape_policy"],
        "nucleus_spacing_margin_px": sampling[
            "nucleus_spacing_margin_px"
        ],
        "instance_connectivity_policy": sampling[
            "instance_connectivity_policy"
        ],
        "source_nucleus_erasure_policy": sampling[
            "source_nucleus_erasure_policy"
        ],
        "buffer_nucleus_policy": sampling["buffer_nucleus_policy"],
        "checkpoint_sha256": actual_sha256,
        "organ_specific_constraints": False,
        "diagnostics_path": sampling.get("diagnostics_path"),
        "accepted_center_probability_by_tissue": sampling.get(
            "accepted_center_probability_by_tissue"
        ),
        "sampling_audit": sampling_audit,
        "sampling_audit_attempts": sampling.get("sampling_audit_attempts"),
        "sampling_audit_max_attempts": sampling.get(
            "sampling_audit_max_attempts"
        ),
        "sampling_audit_selected_attempt": sampling.get(
            "sampling_audit_selected_attempt"
        ),
        "sampling_audit_resampled": sampling.get("sampling_audit_resampled"),
        "sampling_feedback": sampling_feedback,
        "sampling_audit_compatibility": audit_compatibility,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frozen_nuclei_replay_compatibility(
    *,
    args: argparse.Namespace,
    log_path: Path,
    actual_policy: Any,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    policies = expected.get("compatible_frozen_audit_policies") or {}
    compatibility = policies.get(actual_policy) if isinstance(policies, dict) else None
    expected_policy = expected.get("sampling_audit_policy")
    if not isinstance(compatibility, dict):
        raise ValueError(
            "Target nuclei sampling audit does not match the online product "
            f"release: {actual_policy!r} != {expected_policy!r}"
        )
    if compatibility.get("scope") != "hash_locked_approved_nuclei_replay_only":
        raise ValueError("Frozen nuclei replay compatibility has an invalid scope.")

    provenance_path = log_path.parent / "approved_nuclei_provenance.json"
    if not provenance_path.is_file():
        raise ValueError(
            "Legacy nuclei audit policies require hash-locked approval provenance."
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("schema_version") != 1
        or provenance.get("status") != "approved_nuclei_reused"
        or provenance.get("tissue_stage_rerun") is not False
        or provenance.get("nuclei_stage_rerun") is not False
    ):
        raise ValueError("Frozen nuclei approval provenance is not replay-only.")

    target_tissue_path = Path(args.target_tissue_mask)
    target_nuclei_path = Path(args.target_nuclei_mask)
    for label, path in (
        ("target tissue mask", target_tissue_path),
        ("target nuclei mask", target_nuclei_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen {label} not found: {path}")
    tissue_sha256 = _sha256_file(target_tissue_path)
    nuclei_sha256 = _sha256_file(target_nuclei_path)
    log_sha256 = _sha256_file(log_path)
    if tissue_sha256 != provenance.get("approved_target_tissue_sha256"):
        raise ValueError("Frozen target tissue hash does not match its approval.")
    if nuclei_sha256 != provenance.get("approved_target_nuclei_sha256"):
        raise ValueError("Frozen target nuclei hash does not match its approval.")
    asset_sha256 = provenance.get("asset_sha256") or {}
    if asset_sha256.get("target_nuclei") != nuclei_sha256:
        raise ValueError("Frozen target nuclei asset hash is inconsistent.")
    if asset_sha256.get("cell_fill_log") != log_sha256:
        raise ValueError("Frozen cell-fill provenance hash is inconsistent.")

    manifest_path = Path(str(provenance.get("approved_nuclei_manifest", "")))
    if not manifest_path.is_file():
        raise ValueError("Approved nuclei manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    approval = manifest.get("approval") or {}
    if (
        manifest.get("schema_version") != 1
        or manifest.get("stage") != "nuclei"
        or manifest.get("all_automatic_checks_passed") is not True
        or approval.get("status") != "approved"
    ):
        raise ValueError("Approved nuclei manifest is not a completed approval.")
    case_id = provenance.get("approved_entry_case_id")
    entries = [
        entry
        for entry in manifest.get("entries", [])
        if entry.get("case_id") == case_id
    ]
    if len(entries) != 1:
        raise ValueError("Approved nuclei manifest does not contain one matching case.")
    entry = entries[0]
    if (
        entry.get("approval") != "approved"
        or entry.get("audit_passed") is not True
        or entry.get("approved_target_nuclei_sha256") != nuclei_sha256
        or entry.get("parent_target_tissue_sha256") != tissue_sha256
    ):
        raise ValueError("Approved nuclei manifest entry does not match frozen assets.")

    max_attempts = compatibility.get("sampling_audit_max_attempts")
    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("Frozen nuclei compatibility has no valid attempt budget.")
    return {
        "mode": "hash_locked_approved_nuclei_replay",
        "scope": compatibility["scope"],
        "actual_policy": actual_policy,
        "current_policy": expected_policy,
        "sampling_audit_max_attempts": max_attempts,
        "sampling_feedback_required": bool(
            compatibility.get("sampling_feedback_required", False)
        ),
        "approval_provenance": str(provenance_path.resolve()),
        "approved_manifest": str(manifest_path.resolve()),
        "approved_case_id": case_id,
        "target_tissue_sha256": tissue_sha256,
        "target_nuclei_sha256": nuclei_sha256,
        "cell_fill_log_sha256": log_sha256,
    }


def _validate_nuclei_sampling_feedback(
    *,
    sampling: dict[str, Any],
    sampling_audit: dict[str, Any],
    sampling_feedback: dict[str, Any],
    expected: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    if not isinstance(sampling_feedback, dict):
        raise ValueError("Target nuclei feedback trace is missing.")
    if sampling_feedback.get("policy") != expected.get(
        "sampling_feedback_policy"
    ):
        raise ValueError(
            "Target nuclei feedback policy does not match the online product "
            "release."
        )
    field_pairs = (
        ("max_attempts", "sampling_feedback_max_attempts"),
        ("gamma_down_factor", "sampling_feedback_gamma_down_factor"),
        ("gamma_up_factor", "sampling_feedback_gamma_up_factor"),
        ("gamma_min", "sampling_feedback_gamma_min"),
        ("gamma_max", "sampling_feedback_gamma_max"),
        (
            "concentration_z_threshold",
            "sampling_feedback_concentration_z_threshold",
        ),
    )
    for actual_field, expected_field in field_pairs:
        if not _contract_values_equal(
            sampling_feedback.get(actual_field),
            expected.get(expected_field),
        ):
            raise ValueError(
                "Target nuclei feedback contract does not match the online "
                f"product release for {actual_field}."
            )
    if sampling_feedback.get("immutable_parameters") != expected.get(
        "sampling_feedback_immutable_parameters"
    ):
        raise ValueError(
            "Target nuclei feedback loop changed a frozen biological parameter."
        )
    initial_gamma = sampling_feedback.get("initial_gamma")
    selected_gamma = sampling_feedback.get("selected_gamma")
    if not _contract_values_equal(initial_gamma, expected.get("gamma")):
        raise ValueError("Target nuclei feedback loop used the wrong initial gamma.")
    try:
        selected_gamma_float = float(selected_gamma)
    except (TypeError, ValueError) as exc:
        raise ValueError("Target nuclei feedback loop omitted selected gamma.") from exc
    if not (
        float(expected["sampling_feedback_gamma_min"])
        <= selected_gamma_float
        <= float(expected["sampling_feedback_gamma_max"])
    ):
        raise ValueError("Target nuclei feedback selected gamma outside release bounds.")
    actual_gamma = sampling.get("gamma", diagnostics.get("gamma"))
    actual_exponent = sampling.get(
        "candidate_probability_mass_exponent",
        diagnostics.get("candidate_probability_mass_exponent"),
    )
    if not _contract_values_equal(actual_gamma, selected_gamma_float) or not (
        _contract_values_equal(actual_exponent, selected_gamma_float)
    ):
        raise ValueError(
            "Target nuclei selected gamma, diagnostics gamma and candidate "
            "probability exponent must agree."
        )
    if not _contract_values_equal(
        sampling_audit.get("evaluation_gamma"),
        expected.get("gamma"),
    ) or not _contract_values_equal(
        sampling_audit.get("sampling_gamma"),
        selected_gamma_float,
    ):
        raise ValueError(
            "Target nuclei audit must keep the frozen evaluation gamma while "
            "reporting the selected sampling gamma."
        )
    attempts = sampling_feedback.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("Target nuclei feedback loop must record every attempt.")
    if len(attempts) > int(expected["sampling_feedback_max_attempts"]):
        raise ValueError("Target nuclei feedback loop exceeded its attempt budget.")
    if attempts != sampling.get("sampling_audit_attempts"):
        raise ValueError(
            "Target nuclei feedback and sampling-audit attempt ledgers differ."
        )
    gamma_actions = [
        item
        for item in attempts
        if item.get("action") in {"decrease_gamma", "increase_gamma"}
    ]
    if len(gamma_actions) > 1:
        raise ValueError("Target nuclei feedback loop adjusted gamma more than once.")
    previous_gamma = float(initial_gamma)
    previous_failure_reasons: list[str] = []
    base_seed = None
    for position, item in enumerate(attempts):
        if int(item.get("attempt_index", -1)) != position:
            raise ValueError("Target nuclei feedback attempts are not contiguous.")
        if base_seed is None:
            base_seed = int(item.get("seed"))
        if int(item.get("seed", -1)) != base_seed + position:
            raise ValueError("Target nuclei feedback seeds are not deterministic.")
        action = item.get("action")
        if action not in {"initial_sample", "resample_seed", "decrease_gamma", "increase_gamma"}:
            raise ValueError(f"Unknown target nuclei feedback action: {action!r}")
        reasons = set(item.get("trigger_reasons") or ())
        if position == 0:
            if action != "initial_sample" or reasons:
                raise ValueError("Target nuclei feedback trace has an invalid first attempt.")
        elif reasons != set(previous_failure_reasons):
            raise ValueError(
                "Target nuclei feedback action is not linked to the preceding failure."
            )
        attempt_gamma = float(item.get("sampling_gamma"))
        if action == "decrease_gamma" and "PROBNET_OVERCONCENTRATED" not in reasons:
            raise ValueError("Gamma decrease lacks an overconcentration reason.")
        if action == "increase_gamma" and "PROBNET_UNDERFOLLOW" not in reasons:
            raise ValueError("Gamma increase lacks an under-follow reason.")
        if action == "decrease_gamma":
            expected_gamma = max(
                float(expected["sampling_feedback_gamma_min"]),
                previous_gamma * float(expected["sampling_feedback_gamma_down_factor"]),
            )
        elif action == "increase_gamma":
            expected_gamma = min(
                float(expected["sampling_feedback_gamma_max"]),
                previous_gamma * float(expected["sampling_feedback_gamma_up_factor"]),
            )
        else:
            expected_gamma = previous_gamma
        if not _contract_values_equal(attempt_gamma, expected_gamma):
            raise ValueError(
                "Target nuclei feedback gamma does not match its recorded action."
            )
        previous_gamma = attempt_gamma
        previous_failure_reasons = list(item.get("failure_reasons") or ())

    selected_attempt = int(sampling_feedback.get("selected_attempt", -1))
    selected_records = [
        item
        for item in attempts
        if int(item.get("attempt_index", -1)) == selected_attempt
        and item.get("stage") == "sampling_audit"
    ]
    if len(selected_records) != 1:
        raise ValueError("Target nuclei feedback selected attempt is not auditable.")
    selected_record = selected_records[0]
    if not _contract_values_equal(
        selected_record.get("sampling_gamma"), selected_gamma_float
    ) or int(selected_record.get("seed", -1)) != int(
        sampling_feedback.get("selected_seed", -2)
    ):
        raise ValueError("Target nuclei feedback selected record is inconsistent.")
    if selected_record.get("passed") is not True:
        raise ValueError("Target nuclei feedback selected a failed attempt.")
    if int(sampling_audit.get("attempt_index", -1)) != selected_attempt:
        raise ValueError("Target nuclei audit and feedback selected attempts differ.")


def _contract_values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(actual) - float(expected)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return actual == expected


def _probnet_diagnostics_contract(sampling: dict[str, Any]) -> dict[str, Any]:
    diagnostics_path = sampling.get("diagnostics_path")
    if not diagnostics_path:
        return {}
    path = Path(diagnostics_path)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        return {}
    first = payload[0]
    if not isinstance(first, dict):
        return {}
    result: dict[str, Any] = {
        "gamma": first.get("gamma"),
        "sampling_audit": first.get("sampling_audit"),
        "sampling_feedback": first.get("sampling_feedback"),
    }
    tissues = first.get("tissues") or {}
    for field in (
        "candidate_quality_score",
        "candidate_probability_mass_exponent",
        "candidate_diversity_score",
        "candidate_diversity_weight",
    ):
        values = {
            item[field]
            for item in tissues.values()
            if isinstance(item, dict) and item.get(field) is not None
        }
        if len(values) == 1:
            result[field] = next(iter(values))
    return result


def _validate_image_generation_contract(
    args: argparse.Namespace,
) -> dict[str, Any]:
    release_path = Path(args.product_release)
    if not release_path.is_file():
        raise FileNotFoundError(f"Online product release not found: {release_path}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    generation = release["image_generation"]
    expected_inference = generation["inference"]
    runtime_inference = {
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
        "torch_dtype": args.torch_dtype,
        "seed": args.seed,
    }
    for field, actual_value in runtime_inference.items():
        expected_value = expected_inference[field]
        if not _contract_values_equal(actual_value, expected_value):
            raise ValueError(
                "Image-generation inference does not match the online "
                f"product release for {field}: "
                f"{actual_value!r} != {expected_value!r}"
            )

    expected_routing = generation["routing"]
    runtime_routing = {
        "t_inpaint": args.t_inpaint,
        "t_cross": args.t_cross,
        "joint_force_cross_min_generation_support_fraction": (
            getattr(args, "force_cross_generation_support_fraction", 0.50)
        ),
        "max_generation_attempts": args.max_attempts,
    }
    for field, actual_value in runtime_routing.items():
        expected_value = expected_routing.get(
            field,
            0.50
            if field == "joint_force_cross_min_generation_support_fraction"
            else None,
        )
        if expected_value is None:
            raise ValueError(
                f"Online product release is missing routing field {field!r}"
            )
        if not _contract_values_equal(actual_value, expected_value):
            raise ValueError(
                "Image-generation routing does not match the online "
                f"product release for {field}: "
                f"{actual_value!r} != {expected_value!r}"
            )

    expected_verification = release["verification"]
    if args.semantic_postprocess_mode != expected_verification[
        "semantic_postprocess_mode"
    ]:
        raise ValueError(
            "Semantic verification mode does not match the online product "
            f"release: {args.semantic_postprocess_mode!r} != "
            f"{expected_verification['semantic_postprocess_mode']!r}"
        )
    if args.segmentator_checkpoint is not None:
        raise ValueError(
            "The online G2 product requires its release-driven "
            "Segmentator; a raw checkpoint override is not allowed."
        )
    segmentator_path = Path(args.segmentator_release)
    if not segmentator_path.is_file():
        raise FileNotFoundError(
            f"Segmentator release not found: {segmentator_path}"
        )
    segmentator = json.loads(segmentator_path.read_text(encoding="utf-8"))
    for actual_value, expected_field in (
        (segmentator.get("release_id"), "segmentator_release_id"),
        (segmentator.get("checkpoint_sha256"), "segmentator_checkpoint_sha256"),
    ):
        expected_value = expected_verification[expected_field]
        if actual_value != expected_value:
            raise ValueError(
                "Segmentator release does not match the online G2 product "
                f"for {expected_field}: "
                f"{actual_value!r} != {expected_value!r}"
            )
    expected_evaluator = expected_verification["evaluator"]
    evaluator_defaults = FidelityThresholds()
    runtime_evaluator = {
        "policy_id": QualityPolicy().policy_id,
        "schema_version": 2,
        "preservation_exclusion_region": (
            "full_generation_change_region"
        ),
        "quality_score_min": getattr(
            args, "quality_score_min", evaluator_defaults.quality_score_min
        ),
        "evidence_coverage_min": getattr(
            args,
            "evidence_coverage_min",
            evaluator_defaults.evidence_coverage_min,
        ),
        "relative_evidence_coverage_min": (
            QualityPolicy().relative_evidence_coverage_min
        ),
        "semantic_score_min": getattr(
            args, "semantic_score_min", evaluator_defaults.semantic_score_min
        ),
        "relative_semantic_score_min": (
            QualityPolicy().relative_semantic_score_min
        ),
        "relative_semantic_evidence_weight": (
            QualityPolicy().relative_semantic_evidence_weight
        ),
        "relative_semantic_direction_epsilon": (
            QualityPolicy().relative_semantic_direction_epsilon
        ),
        "source_region_accuracy_min": getattr(
            args,
            "changed_region_accuracy_min",
            evaluator_defaults.changed_region_accuracy_min,
        ),
        "source_macro_miou_min": getattr(
            args,
            "changed_region_macro_iou_min",
            evaluator_defaults.changed_region_macro_iou_min,
        ),
        "source_transition_recall_min": getattr(
            args,
            "changed_region_accuracy_min",
            evaluator_defaults.changed_region_accuracy_min,
        ),
        "target_reference_min_pixels": 256,
        "target_reference_recall_min": getattr(
            args,
            "changed_region_macro_iou_min",
            evaluator_defaults.changed_region_macro_iou_min,
        ),
        "source_to_target_confusion_max": (
            1.0
            - getattr(
                args,
                "changed_region_accuracy_min",
                evaluator_defaults.changed_region_accuracy_min,
            )
        ),
        "semantic_core_min_pixels": (
            evaluator_defaults.semantic_core_min_pixels
        ),
        "semantic_core_min_fraction": (
            evaluator_defaults.semantic_core_min_fraction
        ),
        "source_boundary_f1_4_min": getattr(
            args,
            "source_boundary_f1_min",
            evaluator_defaults.source_boundary_f1_min,
        ),
        "boundary_support_min_pixels": getattr(
            args,
            "boundary_support_min_pixels",
            evaluator_defaults.boundary_support_min_pixels,
        ),
        "off_target_drift_u_far_max": getattr(
            args,
            "off_target_drift_max",
            evaluator_defaults.off_target_drift_max,
        ),
        "nuclei_detection_error_max": getattr(
            args,
            "nuclei_count_relative_error_max",
            evaluator_defaults.nuclei_count_relative_error_max,
        ),
        "nuclei_type_error_max": getattr(
            args,
            "nuclei_type_composition_error_max",
            evaluator_defaults.nuclei_type_composition_error_max,
        ),
        "nuclei_min_instances": getattr(
            args,
            "nuclei_type_min_instances",
            evaluator_defaults.nuclei_type_min_instances,
        ),
        "component_weights": dict(QualityPolicy().component_weights),
    }
    for field, actual_value in runtime_evaluator.items():
        expected_value = expected_evaluator[field]
        if not _contract_values_equal(actual_value, expected_value):
            raise ValueError(
                "Quality evaluator does not match the online product release "
                f"for {field}: {actual_value!r} != {expected_value!r}"
            )
    return {
        "status": "validated",
        "validated": True,
        "product_release": str(release_path.resolve()),
        "release_id": release.get("release_id"),
        "inference": runtime_inference,
        "routing": runtime_routing,
        "semantic_postprocess_mode": args.semantic_postprocess_mode,
        "segmentator_release": str(segmentator_path.resolve()),
        "segmentator_release_id": segmentator.get("release_id"),
        "segmentator_checkpoint_sha256": segmentator.get(
            "checkpoint_sha256"
        ),
        "quality_evaluator": runtime_evaluator,
    }


def _load_and_validate_inputs(args: argparse.Namespace) -> dict[str, np.ndarray]:
    required_paths = {
        "reference image": args.reference_image,
        "reference tissue mask": args.reference_tissue_mask,
        "reference nuclei mask": args.reference_nuclei_mask,
        "target tissue mask": args.target_tissue_mask,
        "target nuclei mask": args.target_nuclei_mask,
    }
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not Path(path).exists()]
    semantic_change_region_path = (
        args.semantic_change_region
        if args.semantic_change_region is not None
        else args.change_region
    )
    if (
        semantic_change_region_path is not None
        and not semantic_change_region_path.exists()
    ):
        missing.append(f"semantic change region: {semantic_change_region_path}")
    if (
        args.generation_change_region is not None
        and not args.generation_change_region.exists()
    ):
        missing.append(
            f"generation change region: {args.generation_change_region}"
        )
    if missing:
        raise FileNotFoundError("Required runtime paths not found:\n" + "\n".join(missing))

    reference_image = _load_rgb_image(args.reference_image)
    reference_tissue = load_id_mask(args.reference_tissue_mask)
    reference_nuclei = _load_uint8_mask(args.reference_nuclei_mask)
    target_tissue = load_id_mask(args.target_tissue_mask)
    target_nuclei = _load_uint8_mask(args.target_nuclei_mask)
    for label, mask in (
        ("reference tissue mask", reference_tissue),
        ("reference nuclei mask", reference_nuclei),
        ("target tissue mask", target_tissue),
        ("target nuclei mask", target_nuclei),
    ):
        _validate_same_size(reference_image, mask, label)
    semantic_change_region = (
        load_change_region(semantic_change_region_path)
        if semantic_change_region_path is not None
        else reference_tissue != target_tissue
    )
    generation_change_region = (
        load_change_region(args.generation_change_region)
        if args.generation_change_region is not None
        else np.array(semantic_change_region, copy=True)
    )
    _validate_same_size(
        reference_image,
        semantic_change_region,
        "semantic change region",
    )
    _validate_same_size(
        reference_image,
        generation_change_region,
        "generation change region",
    )
    missing_generation_pixels = (
        np.asarray(semantic_change_region, dtype=bool)
        & ~np.asarray(generation_change_region, dtype=bool)
    )
    if np.any(missing_generation_pixels):
        raise ValueError(
            "generation change region must contain every semantic change pixel; "
            f"missing {int(np.count_nonzero(missing_generation_pixels))} pixels."
        )
    if args.joint_generation_handoff is not None:
        generation_region_policy = {
            "policy": "hash_locked_approved_joint_generation_support_v1",
            "capped": False,
            "generation_pixels": int(
                np.count_nonzero(generation_change_region)
            ),
            "handoff": str(args.joint_generation_handoff.resolve()),
        }
    elif str(args.profile).upper() == "GLAS":
        generation_region_policy = {
            "policy": "preserve_glas_whole_component_and_nucleus_buffer",
            "capped": False,
            "generation_pixels": int(
                np.count_nonzero(generation_change_region)
            ),
        }
    else:
        generation_change_region, generation_region_policy = (
            bound_generation_context_region(
                semantic_change_region,
                generation_change_region,
            )
        )
    return {
        "reference_image": reference_image,
        "reference_tissue": reference_tissue,
        "reference_nuclei": reference_nuclei,
        "target_tissue": target_tissue,
        "reference_coarse_tissue": to_coarse_mask(reference_tissue),
        "target_coarse_tissue": to_coarse_mask(target_tissue),
        "target_nuclei": target_nuclei,
        "semantic_change_region": np.asarray(
            semantic_change_region,
            dtype=bool,
        ),
        "generation_change_region": np.asarray(
            generation_change_region,
            dtype=bool,
        ),
        "generation_region_policy": generation_region_policy,
    }


def _validate_joint_generation_handoff(
    args: argparse.Namespace,
    inputs: Mapping[str, np.ndarray],
) -> dict[str, Any] | None:
    if args.joint_generation_handoff is None:
        return None

    from phase3_joint_edit_refine.generator_adapter import (
        JointGeneratorRoutingConfig,
        build_agentic_joint_route,
        build_frozen_generator_inputs,
    )

    frozen_inputs, _route, manifest = build_frozen_generator_inputs(
        args.joint_generation_handoff,
        output_dir=args.output,
        prompt=args.prompt,
        dataset=args.profile,
        routing_config=JointGeneratorRoutingConfig(
            inpaint_max_generation_support_fraction=args.t_inpaint,
            force_cross_min_generation_support_fraction=(
                getattr(args, "force_cross_generation_support_fraction", 0.50)
            ),
        ),
    )
    expected_paths = {
        "reference image": (args.reference_image, frozen_inputs.reference_image),
        "reference tissue mask": (
            args.reference_tissue_mask,
            frozen_inputs.reference_tissue_mask,
        ),
        "reference nuclei mask": (
            args.reference_nuclei_mask,
            frozen_inputs.reference_nuclei_mask,
        ),
        "target tissue mask": (
            args.target_tissue_mask,
            frozen_inputs.target_tissue_mask,
        ),
        "target nuclei mask": (
            args.target_nuclei_mask,
            frozen_inputs.target_nuclei_mask,
        ),
        "generation change region": (
            args.generation_change_region,
            frozen_inputs.generation_change_region,
        ),
    }
    for label, (actual, expected) in expected_paths.items():
        if actual is None or Path(actual).resolve() != Path(expected).resolve():
            raise ValueError(
                f"{label} does not match the approved joint handoff: "
                f"{actual!s} != {expected!s}"
            )
    semantic_path = (
        args.semantic_change_region
        if args.semantic_change_region is not None
        else args.change_region
    )
    expected_semantic = Path(manifest["paths"]["joint_change"])
    if (
        semantic_path is None
        or Path(semantic_path).resolve() != expected_semantic.resolve()
    ):
        raise ValueError(
            "semantic change region must be the approved joint-change mask: "
            f"{semantic_path!s} != {expected_semantic!s}"
        )

    routing_config = JointGeneratorRoutingConfig(
        inpaint_max_generation_support_fraction=args.t_inpaint,
        force_cross_min_generation_support_fraction=(
            getattr(args, "force_cross_generation_support_fraction", 0.50)
        ),
    )
    routing_decision: AgenticRoutingDecision = build_agentic_joint_route(
        manifest,
        joint_change_mask=inputs["semantic_change_region"],
        generation_support_mask=inputs["generation_change_region"],
        reference_tissue_mask=inputs["reference_tissue"],
        config=routing_config,
    )
    return {
        "routing_decision": routing_decision,
        "compiled_prompt": frozen_inputs.prompt,
        "summary": {
            "status": "validated",
            "manifest": str(args.joint_generation_handoff.resolve()),
            "schema_version": manifest["schema_version"],
            "case_id": manifest["case_id"],
            "candidate_id": manifest["candidate_id"],
            "executable_contract_id": manifest["executable_contract_id"],
            "routing_authority": "generation_support",
            "generation_support_authority": "hash_locked_handoff",
            "large_support_selection_policy": "cross_only",
            "compiled_render_prompt": frozen_inputs.prompt,
        },
    }


def _validate_generation_runtime(args: argparse.Namespace, mode: str) -> None:
    required = {"pretrained model": Path(args.pretrained_model_name_or_path)}
    if mode == "inpaint":
        required["inpaint checkpoint"] = args.inpaint_checkpoint
    else:
        required["cross-v1 checkpoint"] = args.cross_v1_checkpoint
        required["pix2pix-v2 checkpoint"] = args.pix2pix_checkpoint
    missing = [f"{label}: {path}" for label, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Generation runtime paths not found:\n" + "\n".join(missing))
    args.controlnet_release = validate_production_controlnet_checkpoint(
        (
            args.inpaint_checkpoint
            if mode == "inpaint"
            else args.cross_v1_checkpoint
        ),
        mode=mode,
    )
    if mode != "inpaint":
        args.pix2pix_release = validate_production_pix2pix_checkpoint(
            args.pix2pix_checkpoint
        )


def _validate_verification_runtime(args: argparse.Namespace) -> None:
    if args.segmentator_checkpoint is not None:
        segmentator_checkpoint = args.segmentator_checkpoint
        segmentator_runtime = {}
    else:
        release = load_segmentator_release(
            args.segmentator_release,
            verify_checkpoint=False,
        )
        segmentator_checkpoint = Path(release["checkpoint"])
        segmentator_runtime = {
            "segmentator release": args.segmentator_release,
        }
    required = {
        **segmentator_runtime,
        "segmentator checkpoint": segmentator_checkpoint,
        "CellViT model": args.cellvit_model,
        "CellViT root": args.cellvit_root,
        "CellViT wrapper": args.cellvit_script,
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Verification runtime paths not found:\n" + "\n".join(missing))
    args.cellvit_release = validate_frozen_cellvit_checkpoint(args.cellvit_model)


def _generation_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        profile=args.profile,
        reference_image=args.reference_image.resolve(),
        reference_tissue_mask=args.reference_tissue_mask.resolve(),
        reference_nuclei_mask=args.reference_nuclei_mask.resolve(),
        generation_mode="auto",
        cross_backend="cross-v1",
        route_threshold=args.t_cross,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        inpaint_checkpoint=args.inpaint_checkpoint.resolve(),
        cross_v1_checkpoint=args.cross_v1_checkpoint.resolve(),
        pix2pix_checkpoint=args.pix2pix_checkpoint.resolve(),
        device=args.device,
        prompt=args.prompt,
        prompt_source=args.prompt_source,
        torch_dtype=args.torch_dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        seed=args.seed,
    )


def _run_segmentator(
    *, args: argparse.Namespace, image_path: Path, output_dir: Path
) -> dict[str, str]:
    if args.segmentator_python:
        command = [str(args.segmentator_python)]
    else:
        command = ["conda", "run", "-n", args.segmentator_env, "python"]
    command.extend(
        [
            str(REPO_ROOT / "scripts" / "predict_segmentator_mask.py"),
            "--input",
            str(image_path),
            "--output-dir",
            str(output_dir),
            "--profile",
            args.profile,
            "--save-probabilities",
            "--save-entropy",
            "--save-fine-when-applicable",
            "--device",
            args.segmentator_device,
        ]
    )
    if args.segmentator_checkpoint is not None:
        command.extend(
            [
                "--checkpoint",
                str(args.segmentator_checkpoint),
                "--decoder",
                args.segmentator_decoder,
            ]
        )
    else:
        command.extend(["--release", str(args.segmentator_release)])
    _run_logged(command, output_dir / "segmentator.log")
    result = {
        "coarse_mask": str(output_dir / "coarse_mask.png"),
        "coarse_probabilities": str(
            output_dir / "coarse_probabilities.npz"
        ),
        "entropy": str(output_dir / "entropy.npy"),
        "provenance": str(output_dir / "provenance.json"),
    }
    missing = [path for path in result.values() if not Path(path).is_file()]
    if missing:
        raise RuntimeError(
            f"Segmentator completed without required outputs: {missing}"
        )
    fine_paths = {
        "fine_mask": output_dir / "fine_mask.png",
        "fine_probabilities": output_dir / "fine_probabilities.npz",
        "fine_entropy": output_dir / "fine_entropy.npy",
    }
    for name, path in fine_paths.items():
        if path.is_file():
            result[name] = str(path)
    return result


def _load_segmentator_probabilities(path: str | Path) -> np.ndarray:
    with np.load(path) as payload:
        if str(payload["layout"]) != "CHW":
            raise ValueError(f"unsupported Segmentator probability layout: {path}")
        return np.asarray(payload["probabilities"], dtype=np.float64)


def _load_semantic_prediction(
    artifacts: dict[str, str],
) -> SemanticPrediction:
    return SemanticPrediction(
        mask=load_id_mask(artifacts["coarse_mask"]),
        probabilities=_load_segmentator_probabilities(
            artifacts["coarse_probabilities"]
        ),
        entropy=np.load(artifacts["entropy"]),
    )


def _load_fine_prediction(
    artifacts: dict[str, str],
) -> SemanticPrediction | None:
    required = ("fine_mask", "fine_probabilities", "fine_entropy")
    if not all(name in artifacts for name in required):
        return None
    return SemanticPrediction(
        mask=load_id_mask(artifacts["fine_mask"]),
        probabilities=_load_segmentator_probabilities(
            artifacts["fine_probabilities"]
        ),
        entropy=np.load(artifacts["fine_entropy"]),
    )


def _fine_audit_inputs(
    *,
    args: argparse.Namespace,
    inputs: dict[str, np.ndarray],
    source_artifacts: dict[str, str],
    generated_artifacts: dict[str, str],
) -> dict[str, Any]:
    if not profile_supports_fine(args.profile):
        return {}
    source_prediction = _load_fine_prediction(source_artifacts)
    generated_prediction = _load_fine_prediction(generated_artifacts)
    if source_prediction is None or generated_prediction is None:
        raise RuntimeError(
            f"{args.profile} requires dataset-native fine audit artifacts"
        )
    return {
        "source_fine_mask": inputs["reference_tissue"],
        "target_fine_mask": inputs["target_tissue"],
        "source_fine_prediction": source_prediction,
        "generated_fine_prediction": generated_prediction,
        "fine_class_ids": dataset_native_metric_class_ids(
            args.profile, level="fine"
        ),
    }


def _attempt_index_from_path(path: Path) -> int:
    name = path.name
    if not name.startswith("attempt_"):
        return 0
    try:
        return int(name.split("_", 2)[1])
    except (IndexError, ValueError):
        return 0


def _run_cellvit(*, args: argparse.Namespace, image_path: Path, output_dir: Path) -> Path:
    output_path = output_dir / "predicted_nuclei_mask.png"
    command = [
        str(args.cellvit_launch_python),
        str(args.cellvit_script),
        "--image",
        str(image_path),
        "--output-mask",
        str(output_path),
        "--model",
        str(args.cellvit_model),
        "--cellvit-root",
        str(args.cellvit_root),
        "--cellvit-python",
        str(args.cellvit_python),
        "--gpu",
        str(args.cellvit_gpu),
    ]
    _run_logged(command, output_dir / "cellvit.log")
    if not output_path.exists():
        raise RuntimeError(f"CellViT completed without writing {output_path}")
    return output_path


def _cellvit_counts_from_wrapper_summary(
    *,
    predicted_nuclei_path: Path,
    image_path: Path,
    region: np.ndarray,
) -> dict[int, int]:
    summary_path = predicted_nuclei_path.with_suffix(".cellvit_single_patch.json")
    if not summary_path.is_file():
        raise RuntimeError(
            f"CellViT wrapper completed without writing instance provenance: {summary_path}"
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    cells_json = payload.get("cells_json")
    if not cells_json:
        raise RuntimeError(f"CellViT wrapper summary has no cells_json: {summary_path}")
    return cellvit_instance_counts_in_region(
        cells_json,
        image_path,
        region,
    )


def _run_logged(command: list[str], log_path: Path) -> None:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        log_path.write_text(_format_process(command, exc.stdout, exc.stderr), encoding="utf-8")
        raise RuntimeError(f"Command failed; see {log_path}") from exc
    log_path.write_text(_format_process(command, result.stdout, result.stderr), encoding="utf-8")


def _format_process(command: list[str], stdout: str | None, stderr: str | None) -> str:
    parts = ["command: " + " ".join(command)]
    if stdout and stdout.strip():
        parts.append("stdout:\n" + stdout.strip())
    if stderr and stderr.strip():
        parts.append("stderr:\n" + stderr.strip())
    return "\n\n".join(parts) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
