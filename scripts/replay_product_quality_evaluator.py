#!/usr/bin/env python3
"""Replay the product quality evaluator from frozen generation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.inference.agentic import (
    FidelityThresholds,
    verify_mask_fidelity,
)
from phase3_mask_edit.audit import (
    QualityPolicy,
    dataset_native_metric_class_ids,
    evaluate_product_quality,
    source_evaluator_quality,
    source_relative_tissue_metrics,
    to_coarse_mask,
    write_generation_report,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Update canonical evaluator/report/final-image artifacts in place.",
    )
    args = parser.parse_args(argv)
    output_root = args.output.resolve()
    if args.apply and args.expected_count is None:
        raise ValueError(
            "--apply requires --expected-count so a partial cohort cannot be "
            "accepted accidentally."
        )

    cases = _load_cases(args.manifest)
    summaries = _discover_case_summaries(
        run_root=args.run_root,
        output_root=output_root,
    )
    if not summaries:
        raise FileNotFoundError(f"No agentic pipeline summaries below {args.run_root}")
    if args.expected_count is not None and len(summaries) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} cases, found {len(summaries)}."
        )
    discovered_ids = [_case_id(path) for path in summaries]
    missing = sorted(set(discovered_ids) - set(cases))
    if missing:
        raise KeyError(f"Cases absent from {args.manifest}: {missing[:10]}")

    args.output.mkdir(parents=True, exist_ok=True)
    preflight_results = []
    for summary_path in summaries:
        case_id = _case_id(summary_path)
        preflight_results.append(
            replay_case(
                summary_path=summary_path,
                case=cases[case_id],
                output_dir=args.output / case_id,
                apply=False,
            )
        )
    preflight_payload = _replay_summary(
        preflight_results,
        applied_in_place=False,
    )
    preflight_path = args.output / "quality_evaluator_v2_4_preflight.json"
    _write_json(preflight_path, preflight_payload)

    results = preflight_results
    if args.apply:
        expected_by_case = {
            str(item["case_id"]): item for item in preflight_results
        }
        results = []
        for summary_path in summaries:
            case_id = _case_id(summary_path)
            results.append(
                replay_case(
                    summary_path=summary_path,
                    case=cases[case_id],
                    output_dir=args.output / case_id,
                    apply=True,
                    expected_selection=expected_by_case[case_id],
                )
            )

    payload = {
        **_replay_summary(results, applied_in_place=bool(args.apply)),
        "preflight": {
            "completed": True,
            "case_count": len(preflight_results),
            "summary": str(preflight_path),
            "summary_sha256": _sha256(preflight_path),
        },
    }
    summary_name = "quality_evaluator_v2_4_replay_summary.json"
    _write_json(args.output / summary_name, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _replay_summary(
    results: Sequence[Mapping[str, Any]],
    *,
    applied_in_place: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "policy_id": QualityPolicy().policy_id,
        "artifact_replay_only": True,
        "generation_models_invoked": False,
        "applied_in_place": bool(applied_in_place),
        "case_count": len(results),
        "status_counts": _counts(results, "status"),
        "validated_count": sum(
            item["status"] in {"validated_first_pass", "recovered"}
            for item in results
        ),
        "evaluator_uncertain_count": sum(
            item["status"] == "evaluator_uncertain" for item in results
        ),
        "needs_review_count": sum(
            item["status"] == "needs_review" for item in results
        ),
        "results": list(results),
    }


def _discover_case_summaries(
    *,
    run_root: Path,
    output_root: Path,
) -> list[Path]:
    candidates = [
        path
        for path in run_root.rglob("agentic_generation/pipeline_summary.json")
        if output_root not in path.resolve().parents
    ]
    selected: dict[str, Path] = {}
    for path in sorted(candidates, key=lambda item: (item.stat().st_mtime_ns, str(item))):
        case_id = _case_id(path)
        current = selected.get(case_id)
        if current is None or _summary_preference(path) > _summary_preference(current):
            selected[case_id] = path
    return [selected[case_id] for case_id in sorted(selected)]


def _summary_preference(path: Path) -> tuple[int, int, str]:
    workflow = _read_json(path)
    selected = _mapping(workflow.get("selected_attempt"))
    artifact = _mapping(selected.get("artifact"))
    image_path = Path(str(artifact.get("image_path") or ""))
    replayable = bool(
        selected.get("attempt_index") is not None
        and image_path.is_file()
    )
    return (int(replayable), path.stat().st_mtime_ns, str(path))


def replay_case(
    *,
    summary_path: Path,
    case: Mapping[str, Any],
    output_dir: Path,
    apply: bool = False,
    expected_selection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workflow = _read_json(summary_path)
    agentic_dir = summary_path.parent
    instruction_dir = agentic_dir.parent
    run_inputs = _find_generation_inputs(agentic_dir, workflow)
    profile = str(case.get("profile") or case.get("dataset") or "")
    class_ids = dataset_native_metric_class_ids(profile, level="coarse")

    source_mask = to_coarse_mask(_load_mask(run_inputs["reference_tissue_mask"]))
    target_mask = to_coarse_mask(_load_mask(run_inputs["target_tissue_mask"]))
    semantic_region_path = agentic_dir / "semantic_change_region.png"
    generation_region_path = agentic_dir / "generation_change_region.png"
    semantic_region = _load_mask(semantic_region_path) > 0
    generation_region = _load_mask(generation_region_path) > 0
    if np.any(semantic_region & ~generation_region):
        raise RuntimeError(
            f"Generation region does not cover semantic region for {_case_id(summary_path)}"
        )

    source_dir = agentic_dir / "source_verification"
    source_prediction = _load_mask(source_dir / "coarse_mask.png")
    source_probabilities = _load_probabilities(
        source_dir / "coarse_probabilities.npz"
    )
    source_entropy = _optional_array(source_dir / "entropy.npy")
    source_quality = _frozen_source_quality(
        workflow=workflow,
        source_mask=source_mask,
        source_prediction=source_prediction,
        source_probabilities=source_probabilities,
        class_ids=class_ids,
    )
    source_nuclei_calibration = _optional_json(
        agentic_dir
        / "source_nuclei_verification"
        / "evaluator_calibration_counts.json"
    )
    if not source_nuclei_calibration:
        raise FileNotFoundError(
            f"Missing source CellViT calibration below {agentic_dir}"
        )
    target_nuclei_mask = _load_mask(run_inputs["target_nuclei_mask"])

    replay_records = []
    for attempt in workflow.get("attempts") or []:
        replay_records.append(
            _replay_attempt(
                attempt=attempt,
                source_mask=source_mask,
                target_mask=target_mask,
                semantic_region=semantic_region,
                generation_region=generation_region,
                generation_region_path=generation_region_path,
                target_nuclei_mask=target_nuclei_mask,
                source_prediction=source_prediction,
                source_probabilities=source_probabilities,
                source_entropy=source_entropy,
                source_quality=source_quality,
                source_nuclei_calibration=source_nuclei_calibration,
                class_ids=class_ids,
            )
        )
    replayed_attempts = [record["attempt"] for record in replay_records]
    selected, status = select_replayed_attempt(replayed_attempts)
    superseded_attempts = (
        replayed_attempts[1:] if status == "validated_first_pass" else []
    )
    canonical_attempts = (
        replayed_attempts[:1]
        if status == "validated_first_pass"
        else replayed_attempts
    )
    replayed_workflow = {
        **workflow,
        "status": status,
        "attempts": canonical_attempts,
        "selected_attempt": selected,
        "artifact_replay": {
            "source_pipeline_summary": str(summary_path),
            "policy_id": QualityPolicy().policy_id,
            "generation_models_invoked": False,
            "semantic_region_policy": "exact_source_target_tissue_difference",
            "preservation_exclusion_region_policy": (
                "full_generation_change_region"
            ),
            "source_nuclei_calibration": str(
                agentic_dir
                / "source_nuclei_verification"
                / "evaluator_calibration_counts.json"
            ),
            "superseded_historical_attempts": [
                _historical_attempt_record(item) for item in superseded_attempts
            ],
        },
    }
    selected_image = Path(str(_mapping(selected.get("artifact")).get("image_path")))
    if not selected_image.is_file():
        raise FileNotFoundError(f"Selected replay image is missing: {selected_image}")
    selected_sha256 = _sha256(selected_image)
    replayed_workflow["artifact_replay"][
        "selected_image_sha256"
    ] = selected_sha256
    replayed_workflow["image_generation_provenance"] = (
        _selected_image_generation_provenance(selected)
    )

    selection_signature = {
        "case_id": _case_id(summary_path),
        "status": status,
        "selected_attempt": selected.get("attempt_index"),
        "selected_image_sha256": selected_sha256,
    }
    if expected_selection is not None:
        expected_signature = {
            key: expected_selection.get(key) for key in selection_signature
        }
        if selection_signature != expected_signature:
            raise RuntimeError(
                "Replay selection changed after full-cohort preflight for "
                f"{selection_signature['case_id']}: "
                f"{selection_signature!r} != {expected_signature!r}"
            )
    contract = dict(replayed_workflow.get("image_generation_contract") or {})
    contract_evaluator = dict(contract.get("quality_evaluator") or {})
    contract_evaluator.update(
        {
            "policy_id": QualityPolicy().policy_id,
            "preservation_exclusion_region": (
                "full_generation_change_region"
            ),
        }
    )
    contract["quality_evaluator"] = contract_evaluator
    replayed_workflow["image_generation_contract"] = contract
    online_self_audit = dict(replayed_workflow.get("online_self_audit") or {})
    online_self_audit.update(
        {
            "policy_id": QualityPolicy().policy_id,
            "formal_validated": status
            in {"validated_first_pass", "recovered"},
            "engineering_status": status,
            "artifact_replay_only": True,
        }
    )
    replayed_workflow["online_self_audit"] = online_self_audit

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "evaluator_replay.json", replayed_workflow)
    report_json, report_markdown, report = write_generation_report(
        replayed_workflow,
        output_dir=output_dir,
    )

    if apply:
        _apply_replay(
            agentic_dir=agentic_dir,
            instruction_dir=instruction_dir,
            replayed_workflow=replayed_workflow,
            replay_records=replay_records,
            selected=selected,
            selected_image=selected_image,
            report=report,
        )
        if _sha256(agentic_dir / "generated_image.png") != selected_sha256:
            raise RuntimeError("Agentic final image hash does not match selection.")
        if _sha256(instruction_dir / "generated_image.png") != selected_sha256:
            raise RuntimeError("Instruction final image hash does not match selection.")

    selected_verification = _mapping(selected.get("verification"))
    return {
        **selection_signature,
        "profile": profile,
        "primitive": case.get("primitive"),
        "selected_model": selected.get("requested_mode"),
        "quality_score": selected_verification.get("quality_score"),
        "evidence_coverage": selected_verification.get("evidence_coverage"),
        "component_scores": selected_verification.get("component_scores"),
        "applicability": selected_verification.get("applicability"),
        "reason_codes": selected_verification.get("reason_codes"),
        "evaluator_replay": str(output_dir / "evaluator_replay.json"),
        "generation_report_json": str(report_json),
        "generation_report_markdown": str(report_markdown),
        "applied_in_place": bool(apply),
        "superseded_historical_attempt_count": len(superseded_attempts),
    }


def _replay_attempt(
    *,
    attempt: Mapping[str, Any],
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    semantic_region: np.ndarray,
    generation_region: np.ndarray,
    generation_region_path: Path,
    target_nuclei_mask: np.ndarray,
    source_prediction: np.ndarray,
    source_probabilities: np.ndarray,
    source_entropy: np.ndarray | None,
    source_quality: Mapping[str, Any],
    source_nuclei_calibration: Mapping[str, Any],
    class_ids: Sequence[int],
) -> dict[str, Any]:
    replayed = dict(attempt)
    artifact = _mapping(attempt.get("artifact"))
    image_path = artifact.get("image_path")
    if attempt.get("error") or not image_path:
        replayed["verification"] = None
        return {"attempt": replayed}

    verification_dir = Path(str(image_path)).parent / "verification"
    verification_path = verification_dir / "verification.json"
    if not verification_path.is_file():
        replayed["verification"] = None
        return {"attempt": replayed}
    old_verification = _read_json(verification_path)
    online_audit_path = Path(str(old_verification["online_semantic_audit"]))
    online_audit = _read_json(online_audit_path)
    raw_path = verification_dir / "coarse_mask_raw.png"
    if not raw_path.is_file():
        raw_path = verification_dir / "coarse_mask.png"
    raw_prediction = _load_mask(raw_path)
    p1_path = verification_dir / "coarse_mask_p1.png"
    generated_probabilities = _load_probabilities(
        verification_dir / "coarse_probabilities.npz"
    )
    generated_entropy = _optional_array(verification_dir / "entropy.npy")

    def metrics_for(prediction: np.ndarray) -> dict[str, Any]:
        return source_relative_tissue_metrics(
            source_mask=source_mask,
            target_mask=target_mask,
            source_prediction=source_prediction,
            generated_prediction=prediction,
            source_probabilities=source_probabilities,
            generated_probabilities=generated_probabilities,
            class_ids=class_ids,
            source_entropy=source_entropy,
            generated_entropy=generated_entropy,
            semantic_change_region=semantic_region,
            preservation_exclusion_region=generation_region,
        )

    raw_metrics = metrics_for(raw_prediction)
    p1_metrics = metrics_for(_load_mask(p1_path)) if p1_path.is_file() else None
    decision = str(old_verification.get("semantic_decision_input") or "raw")
    decision_prediction = (
        _load_mask(p1_path)
        if decision == "p1_audited" and p1_path.is_file()
        else raw_prediction
    )
    decision_metrics = p1_metrics if decision == "p1_audited" else raw_metrics
    if decision_metrics is None:
        raise RuntimeError(f"Missing decision metrics in {verification_dir}")

    target_counts = _mapping(old_verification.get("target_nuclei_instance_counts"))
    generated_counts = _mapping(
        old_verification.get("predicted_nuclei_instance_counts")
    )
    predicted_nuclei_path = Path(str(old_verification["predicted_nuclei_mask"]))
    base = verify_mask_fidelity(
        reference_tissue_mask=source_mask,
        target_tissue_mask=target_mask,
        predicted_tissue_mask=decision_prediction,
        source_predicted_tissue_mask=source_prediction,
        change_region=semantic_region,
        target_nuclei_mask=target_nuclei_mask,
        predicted_nuclei_mask=_load_mask(predicted_nuclei_path),
        target_nuclei_instance_counts=target_counts,
        predicted_nuclei_instance_counts=generated_counts,
        thresholds=FidelityThresholds(),
        enforce_off_target_drift=True,
    )
    quality = evaluate_product_quality(
        coarse_metrics=decision_metrics,
        source_quality=source_quality,
        base_metrics=base.metrics,
        source_nuclei_calibration=source_nuclei_calibration,
        target_nuclei_counts=target_counts,
        generated_nuclei_counts=generated_counts,
        policy=QualityPolicy(),
    )
    metadata = quality.to_metadata()
    compact = {
        key: metadata[key]
        for key in (
            "schema_version",
            "passed",
            "quality_score",
            "evidence_coverage",
            "component_scores",
            "applicability",
            "scientific_status",
            "reason_codes",
            "metrics",
            "failed_checks",
        )
    }
    compact["score"] = compact["quality_score"]
    replayed["verification"] = compact
    full_verification = {
        **old_verification,
        **compact,
        "score": compact["quality_score"],
        "off_target_drift_enforced": True,
        "preservation_exclusion_region_policy": (
            "full_generation_change_region"
        ),
        "preservation_exclusion_region": str(generation_region_path),
        "quality_policy": metadata["policy"],
        "source_nuclei_evaluator_calibration": source_nuclei_calibration,
        "raw_audit_metrics": raw_metrics,
        "p1_audit_metrics": p1_metrics,
        "evaluator_replay": {
            "policy_id": QualityPolicy().policy_id,
            "generation_models_invoked": False,
        },
    }
    online_audit_update = {
        **online_audit,
        "raw_metrics": raw_metrics,
        "p1_metrics": p1_metrics,
        "preservation_exclusion_region_policy": (
            "full_generation_change_region"
        ),
        "preservation_exclusion_region": str(generation_region_path),
    }
    return {
        "attempt": replayed,
        "verification_path": verification_path,
        "verification": full_verification,
        "online_audit_path": online_audit_path,
        "online_audit": online_audit_update,
    }


def select_replayed_attempt(
    attempts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    verified = [item for item in attempts if item.get("verification")]
    if not verified:
        raise RuntimeError("No replayable verified candidate exists.")
    first = attempts[0]
    if bool(_mapping(first.get("verification")).get("passed")):
        return dict(first), "validated_first_pass"
    for item in attempts[1:]:
        if bool(_mapping(item.get("verification")).get("passed")):
            return dict(item), "recovered"
    selected = max(verified, key=_selection_key)
    status = (
        "evaluator_uncertain"
        if _mapping(selected["verification"]).get("scientific_status")
        == "evaluator_uncertain"
        else "needs_review"
    )
    return dict(selected), status


def _selection_key(attempt: Mapping[str, Any]) -> tuple[float, ...]:
    verification = _mapping(attempt.get("verification"))
    components = _mapping(verification.get("component_scores"))
    return (
        float(bool(verification.get("passed"))),
        float(verification.get("quality_score") or 0.0),
        float(components.get("semantic") or 0.0),
        float(components.get("preservation") or 0.0),
        -float(attempt.get("attempt_index") or 0),
    )


def _apply_replay(
    *,
    agentic_dir: Path,
    instruction_dir: Path,
    replayed_workflow: dict[str, Any],
    replay_records: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    selected_image: Path,
    report: Mapping[str, Any],
) -> None:
    for record in replay_records:
        if "verification_path" not in record:
            continue
        _write_json(Path(record["verification_path"]), record["verification"])
        _write_json(Path(record["online_audit_path"]), record["online_audit"])

    final_agentic = agentic_dir / "generated_image.png"
    final_instruction = instruction_dir / "generated_image.png"
    _copy_atomic(selected_image, final_agentic)
    _copy_atomic(selected_image, final_instruction)
    report_json_path, report_markdown_path, canonical_report = (
        write_generation_report(replayed_workflow, output_dir=agentic_dir)
    )
    if dict(report) != canonical_report:
        raise RuntimeError("Replay report changed between dry output and apply.")

    replayed_workflow["generated_image"] = str(final_agentic)
    replayed_workflow["generation_report"] = {
        "json": str(report_json_path),
        "markdown": str(report_markdown_path),
        "content": canonical_report,
    }
    _write_json(agentic_dir / "pipeline_summary.json", replayed_workflow)
    _write_json(
        agentic_dir / "agentic_workflow.json",
        {
            key: replayed_workflow[key]
            for key in (
                "status",
                "route",
                "attempts",
                "selected_attempt",
                "output_dir",
                "artifact_replay",
            )
            if key in replayed_workflow
        },
    )
    _write_json(
        agentic_dir / "evaluator_replay_v2_1.json",
        replayed_workflow["artifact_replay"],
    )

    outer_path = instruction_dir / "pipeline_summary.json"
    outer = _read_json(outer_path)
    generation = dict(outer.get("generation") or {})
    selected_verification = _mapping(selected.get("verification"))
    selected_artifact = _mapping(selected.get("artifact"))
    selected_metadata = _mapping(selected_artifact.get("metadata"))
    selected_fields = {
        field: selected_metadata[field]
        for field in (
            "raw_generated_image",
            "controlnet_output_dir",
            "controlnet_release",
            "change_ratio",
            "semantic_change_ratio",
            "generation_change_ratio",
            "generation_change_region",
            "route_threshold",
            "prompt",
        )
        if field in selected_metadata
    }
    generation.update(
        {
            **selected_fields,
            "status": replayed_workflow["status"],
            "generated_image": str(final_instruction),
            "selected_mode": selected_metadata.get("selected_mode")
            or selected.get("requested_mode"),
            "quality_score": selected_verification.get("quality_score"),
            "evidence_coverage": selected_verification.get(
                "evidence_coverage"
            ),
            "component_scores": selected_verification.get(
                "component_scores"
            ),
            "evaluator_applicability": selected_verification.get(
                "applicability"
            ),
            "scientific_status": selected_verification.get(
                "scientific_status"
            ),
            "failed_checks": selected_verification.get("failed_checks"),
            "reason_codes": selected_verification.get("reason_codes"),
            "generation_report": replayed_workflow["generation_report"],
            "agentic_workflow": {
                key: replayed_workflow[key]
                for key in (
                    "status",
                    "route",
                    "attempts",
                    "selected_attempt",
                    "output_dir",
                    "artifact_replay",
                )
                if key in replayed_workflow
            },
            "evaluator_replay": replayed_workflow["artifact_replay"],
            "image_generation_provenance": replayed_workflow[
                "image_generation_provenance"
            ],
        }
    )
    outer["generation"] = generation
    _write_json(outer_path, outer)


def _selected_image_generation_provenance(
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = _mapping(selected.get("artifact"))
    metadata = _mapping(artifact.get("metadata"))
    selected_mode = metadata.get("selected_mode") or selected.get(
        "requested_mode"
    )
    cross = _mapping(metadata.get("cross_v1"))
    pix2pix = _mapping(cross.get("pix2pix_v2"))
    protection = pix2pix.get("cross_rgb_od_low_stain_protection")
    if not isinstance(protection, Mapping):
        protection = {
            "policy": "cross_rgb_od_low_stain_v1",
            "enabled": False,
            "applied": False,
            "status": (
                "not_applicable"
                if selected_mode not in {"cross", "cross-v1"}
                and "cross" not in str(selected_mode)
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


def _historical_attempt_record(attempt: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _mapping(attempt.get("artifact"))
    image_path = Path(str(artifact.get("image_path") or ""))
    return {
        "attempt_index": attempt.get("attempt_index"),
        "requested_mode": attempt.get("requested_mode"),
        "image_path": str(image_path) if image_path.is_file() else None,
        "image_sha256": _sha256(image_path) if image_path.is_file() else None,
        "verification": attempt.get("verification"),
        "reason": (
            "generated_under_previous_evaluator_after_a_first_candidate_"
            "that_passes_the_replayed_policy"
        ),
    }


def _frozen_source_quality(
    *,
    workflow: Mapping[str, Any],
    source_mask: np.ndarray,
    source_prediction: np.ndarray,
    source_probabilities: np.ndarray,
    class_ids: Sequence[int],
) -> Mapping[str, Any]:
    for attempt in workflow.get("attempts") or []:
        artifact = _mapping(attempt.get("artifact"))
        image_path = artifact.get("image_path")
        if not image_path:
            continue
        old = _optional_json(
            Path(str(image_path)).parent / "verification" / "verification.json"
        )
        audit_path = old.get("online_semantic_audit")
        if audit_path and Path(str(audit_path)).is_file():
            quality = _mapping(_read_json(Path(str(audit_path))).get("source_quality"))
            if quality:
                return quality
    return source_evaluator_quality(
        source_mask=source_mask,
        source_prediction=source_prediction,
        source_probabilities=source_probabilities,
        class_ids=class_ids,
    )


def _find_generation_inputs(
    agentic_dir: Path,
    workflow: Mapping[str, Any],
) -> dict[str, Any]:
    for attempt in workflow.get("attempts") or []:
        artifact = _mapping(attempt.get("artifact"))
        metadata = _mapping(artifact.get("metadata"))
        output_dir = metadata.get("controlnet_output_dir")
        if output_dir:
            summary = Path(str(output_dir)) / "run_summary.json"
            if summary.is_file():
                return _read_json(summary)
    summaries = sorted(agentic_dir.glob("attempt_*/**/run_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No generation run_summary below {agentic_dir}")
    return _read_json(summaries[0])


def _case_id(summary_path: Path) -> str:
    if summary_path.parent.name != "agentic_generation":
        raise ValueError(f"Unexpected agentic summary path: {summary_path}")
    return summary_path.parent.parent.parent.name


def _load_cases(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = _read_json(path)
        rows = list(payload.get("cases") or payload.get("rows") or [])
    return {
        str(row.get("case_id") or row.get("condition_id")): dict(row)
        for row in rows
    }


def _load_mask(path: str | Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    return array[..., 0] if array.ndim == 3 else array


def _load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
        class_ids = np.asarray(payload["class_ids"], dtype=np.int64)
    if probabilities.shape[0] == len(class_ids) and np.array_equal(
        class_ids, np.arange(len(class_ids))
    ):
        return probabilities
    channel_count = max(8, int(class_ids.max()) + 1)
    ordered = np.zeros((channel_count, *probabilities.shape[1:]), dtype=np.float64)
    for source_index, class_id in enumerate(class_ids):
        ordered[int(class_id)] = probabilities[source_index]
    return ordered


def _optional_array(path: Path) -> np.ndarray | None:
    return np.load(path) if path.is_file() else None


def _optional_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field))
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
