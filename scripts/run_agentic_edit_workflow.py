#!/usr/bin/env python3
"""Run the bounded pathology edit agent: route, generate, verify, and recover."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.inference.agentic import (
    AgenticWorkflowConfig,
    FidelityThresholds,
    GenerationArtifact,
    VerificationResult,
    run_agentic_workflow,
    verify_mask_fidelity,
)
from controlnet_train.inference.router import AgenticRoutingConfig
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
from phase3_mask_edit.audit import (
    OnlineAuditPolicy,
    OnlineSemanticAuditor,
    SemanticPrediction,
    dataset_native_metric_class_ids,
    profile_supports_fine,
    source_evaluator_quality,
    to_coarse_mask,
)
from segmentator.release import load_segmentator_release
from scripts.run_phase3_inpaint_pipeline import (
    _load_rgb_image,
    _load_uint8_mask,
    _run_generation_stage,
    _validate_same_size,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED_MODEL = "/data/huggingface/FLUX.1-dev"
DEFAULT_SEGMENTATOR_RELEASE = (
    REPO_ROOT
    / "benchmark_configs"
    / "releases"
    / "segmentator_fine_c_epoch2.json"
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
    parser.add_argument("--t-inpaint", type=float, default=0.12)
    parser.add_argument("--t-cross", type=float, default=0.30)
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
    parser.add_argument("--nuclei-density-relative-error-max", type=float, default=0.35)
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
    nuclei_generation = _validate_nuclei_generation_contract(args)
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
        nuclei_density_relative_error_max=args.nuclei_density_relative_error_max,
    )
    source_segmentator = None
    source_semantic_prediction = None
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

    def generate(mode: str, attempt_dir: Path) -> GenerationArtifact:
        generation_mode = _generation_backend_mode(mode)
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
        _validate_verification_runtime(args)
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
            thresholds=thresholds,
            enforce_off_target_drift=artifact.mode != "inpaint",
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
            result = VerificationResult(
                passed=False,
                score=result.score,
                metrics={
                    **dict(result.metrics),
                    "injected_canary_failure": 1.0,
                },
                failed_checks=failed_checks,
            )
        _write_json(
            verification_dir / "verification.json",
            {
                "passed": result.passed,
                "score": result.score,
                "metrics": dict(result.metrics),
                "failed_checks": list(result.failed_checks),
                "off_target_drift_enforced": artifact.mode != "inpaint",
                "source_segmentator": source_segmentator,
                "predicted_tissue": predicted_tissue,
                "predicted_nuclei_mask": str(predicted_nuclei_path),
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
    )

    final_path = output_dir / "generated_image.png"
    if workflow.status == "noop":
        shutil.copy2(args.reference_image, final_path)
    elif workflow.selected_attempt and workflow.selected_attempt.artifact:
        shutil.copy2(workflow.selected_attempt.artifact.image_path, final_path)

    summary = workflow.to_metadata()
    summary["generated_image"] = str(final_path) if final_path.exists() else None
    summary["nuclei_generation"] = nuclei_generation
    summary["image_generation_provenance"] = (
        _selected_image_generation_provenance(summary)
    )
    summary["online_self_audit"] = {
        "scope": "product_runtime",
        "benchmark_independent": True,
        "semantic_postprocess_mode": args.semantic_postprocess_mode,
        "verifier_policy_status": "pilot_not_formal",
        "formal_validated": False,
        "engineering_status": (
            "engineering_pass_uncalibrated"
            if workflow.status == "validated"
            else workflow.status
        ),
        "reason": (
            "Confidence, evaluator-clean, and acceptance thresholds require "
            "a separately frozen blinded calibration cohort."
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
    }
    _write_json(output_dir / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if workflow.status in {"validated", "noop"} else 2


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
        "expected_checkpoint_sha256": expected["checkpoint_sha256"],
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
    actual_policy = sampling.get("candidate_queue_policy")
    if actual_policy != expected["candidate_queue_policy"]:
        raise ValueError(
            "Target nuclei candidate queue policy does not match the online "
            f"product release: {actual_policy!r} != "
            f"{expected['candidate_queue_policy']!r}"
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
    return {
        **result,
        "status": "validated",
        "validated": True,
        "log": str(log_path.resolve()),
        "candidate_queue_policy": actual_policy,
        "checkpoint_sha256": actual_sha256,
        "organ_specific_constraints": False,
        "diagnostics_path": sampling.get("diagnostics_path"),
        "accepted_center_probability_by_tissue": sampling.get(
            "accepted_center_probability_by_tissue"
        ),
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
