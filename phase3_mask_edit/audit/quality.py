"""Reliability-aware product quality scoring and generation reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

QUALITY_POLICY_ID = "online-quality-evaluator-v2.4"
QUALITY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class QualityPolicy:
    """Frozen engineering policy for online candidate evaluation."""

    policy_id: str = QUALITY_POLICY_ID
    quality_score_min: float = 0.75
    evidence_coverage_min: float = 0.80
    relative_evidence_coverage_min: float = 0.70
    semantic_score_min: float = 0.60
    relative_semantic_score_min: float = 0.60
    relative_semantic_evidence_weight: float = 0.35
    relative_semantic_direction_epsilon: float = 1e-4
    source_region_accuracy_min: float = 0.70
    source_macro_miou_min: float = 0.55
    source_transition_recall_min: float = 0.70
    target_reference_min_pixels: int = 256
    target_reference_recall_min: float = 0.55
    source_to_target_confusion_max: float = 0.30
    semantic_core_min_pixels: int = 256
    semantic_core_min_fraction: float = 0.20
    source_boundary_f1_min: float = 0.45
    boundary_support_min_pixels: int = 256
    off_target_drift_max: float = 0.08
    nuclei_detection_error_max: float = 0.35
    nuclei_type_error_max: float = 0.35
    nuclei_min_instances: int = 10
    component_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "semantic": 0.50,
            "preservation": 0.20,
            "boundary": 0.10,
            "nuclei_count": 0.12,
            "nuclei_type": 0.08,
        }
    )

    def validate(self) -> None:
        unit_fields = (
            self.quality_score_min,
            self.evidence_coverage_min,
            self.relative_evidence_coverage_min,
            self.semantic_score_min,
            self.relative_semantic_score_min,
            self.relative_semantic_evidence_weight,
            self.relative_semantic_direction_epsilon,
            self.source_region_accuracy_min,
            self.source_macro_miou_min,
            self.source_transition_recall_min,
            self.target_reference_recall_min,
            self.source_to_target_confusion_max,
            self.semantic_core_min_fraction,
            self.source_boundary_f1_min,
            self.off_target_drift_max,
            self.nuclei_detection_error_max,
            self.nuclei_type_error_max,
        )
        if any(value < 0.0 or value > 1.0 for value in unit_fields):
            raise ValueError("Quality policy thresholds must be in [0, 1].")
        if self.target_reference_min_pixels <= 0:
            raise ValueError("target_reference_min_pixels must be positive.")
        if self.semantic_core_min_pixels <= 0:
            raise ValueError("semantic_core_min_pixels must be positive.")
        expected = {
            "semantic",
            "preservation",
            "boundary",
            "nuclei_count",
            "nuclei_type",
        }
        if set(self.component_weights) != expected:
            raise ValueError("Quality policy component weights are incomplete.")
        if any(float(value) <= 0.0 for value in self.component_weights.values()):
            raise ValueError("Quality component weights must be positive.")
        if abs(sum(self.component_weights.values()) - 1.0) > 1e-9:
            raise ValueError("Quality component weights must sum to one.")
        if self.relative_semantic_evidence_weight > float(
            self.component_weights["semantic"]
        ):
            raise ValueError(
                "Relative semantic evidence cannot outweigh absolute semantic evidence."
            )
        if self.relative_semantic_evidence_weight <= 0.0:
            raise ValueError("Relative semantic evidence weight must be positive.")


@dataclass(frozen=True)
class QualityEvaluation:
    """Complete scoring result consumed by the agent and UI."""

    passed: bool
    quality_score: float
    evidence_coverage: float
    component_scores: Mapping[str, float]
    applicability: Mapping[str, bool]
    metrics: Mapping[str, float]
    failed_checks: tuple[str, ...]
    reason_codes: tuple[str, ...]
    scientific_status: str
    policy: QualityPolicy

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": QUALITY_SCHEMA_VERSION,
            "policy_id": self.policy.policy_id,
            "passed": self.passed,
            "quality_score": self.quality_score,
            "evidence_coverage": self.evidence_coverage,
            "component_scores": dict(self.component_scores),
            "applicability": dict(self.applicability),
            "metrics": dict(self.metrics),
            "failed_checks": list(self.failed_checks),
            "reason_codes": list(self.reason_codes),
            "scientific_status": self.scientific_status,
            "policy": {
                **asdict(self.policy),
                "component_weights": dict(self.policy.component_weights),
            },
            "validated_interpretation": (
                "frozen_engineering_evaluator_pass_not_clinical_correctness"
            ),
        }


def evaluate_product_quality(
    *,
    coarse_metrics: Mapping[str, Any],
    source_quality: Mapping[str, Any],
    base_metrics: Mapping[str, float],
    source_nuclei_calibration: Mapping[str, Any] | None,
    target_nuclei_counts: Mapping[int, int] | None,
    generated_nuclei_counts: Mapping[int, int] | None,
    policy: QualityPolicy | None = None,
) -> QualityEvaluation:
    """Combine only source-calibrated evidence into one product score."""

    policy = policy or QualityPolicy()
    policy.validate()
    weights = dict(policy.component_weights)
    metrics: dict[str, float] = {
        str(key): float(value)
        for key, value in base_metrics.items()
        if _is_number(value)
    }
    components: dict[str, float] = {}
    applicable = {
        "semantic": False,
        "semantic_absolute": False,
        "semantic_relative": False,
        "semantic_relative_bilateral": False,
        "semantic_relative_target_only": False,
        "semantic_relative_source_only": False,
        "preservation": False,
        "preservation_appearance_calibrated": False,
        "boundary": False,
        "boundary_absolute": False,
        "boundary_relative": False,
        "nuclei_count": False,
        "nuclei_type": False,
    }
    reasons: list[str] = []

    changed = _mapping(coarse_metrics.get("changed_region"))
    source_calibration = _mapping(
        coarse_metrics.get("source_evaluator_calibration")
    )
    transition = _mapping(
        coarse_metrics.get("transition_evaluator_calibration")
    )
    preservation = _mapping(coarse_metrics.get("preservation"))
    boundary = _mapping(coarse_metrics.get("boundary"))
    source_quality_metrics = _mapping(source_quality.get("metrics"))

    source_accuracy = _number(source_calibration.get("accuracy"), 0.0)
    source_macro = _number(source_calibration.get("macro_miou"), 0.0)
    source_recall = _number(transition.get("source_class_recall_min"), 0.0)
    target_available = bool(transition.get("target_reference_available"))
    target_recall = _number(
        transition.get("target_reference_recall_min"), 0.0
    )
    source_target_confusion = _number(
        transition.get("source_to_target_confusion_rate"), 1.0
    )
    scale_applicable = bool(
        base_metrics.get("semantic_scale_evaluator_applicable", 1.0)
    )
    transition_pixels = int(transition.get("transition_pixels") or 0)
    semantic_absolute_applicable = bool(
        transition_pixels > 0
        and source_accuracy >= policy.source_region_accuracy_min
        and source_macro >= policy.source_macro_miou_min
        and source_recall >= policy.source_transition_recall_min
        and target_available
        and target_recall >= policy.target_reference_recall_min
        and source_target_confusion <= policy.source_to_target_confusion_max
        and scale_applicable
    )
    target_direction_gain = _first_number(
        changed,
        "appearance_calibrated_target_probability_gain",
        "target_probability_gain",
    )
    source_direction_gain = _first_number(
        changed,
        "appearance_calibrated_source_probability_suppression",
        "source_probability_suppression",
    )
    margin_direction_gain = _first_number(
        changed,
        "appearance_calibrated_soft_margin_gain",
        "soft_margin_gain",
    )
    target_direction_available = target_direction_gain is not None
    source_direction_available = source_direction_gain is not None
    margin_direction_available = margin_direction_gain is not None
    target_direction_support = int(
        changed.get("target_direction_support_pixels")
        or (transition_pixels if target_direction_available else 0)
    )
    source_direction_support = int(
        changed.get("source_direction_support_pixels")
        or (transition_pixels if source_direction_available else 0)
    )
    margin_direction_support = int(
        changed.get("margin_direction_support_pixels")
        or (transition_pixels if margin_direction_available else 0)
    )
    target_direction_available = bool(
        target_direction_available and target_direction_support > 0
    )
    source_direction_available = bool(
        source_direction_available and source_direction_support > 0
    )
    margin_direction_available = bool(
        margin_direction_available and margin_direction_support > 0
    )
    direction_metrics_available = bool(
        target_direction_available or source_direction_available
    )
    metric_direction_epsilon = _number(
        changed.get("semantic_direction_epsilon"),
        policy.relative_semantic_direction_epsilon,
    )
    semantic_relative_applicable = bool(
        not semantic_absolute_applicable
        and scale_applicable
        and direction_metrics_available
        and abs(
            metric_direction_epsilon
            - policy.relative_semantic_direction_epsilon
        )
        <= 1e-12
    )
    semantic_applicable = bool(
        semantic_absolute_applicable or semantic_relative_applicable
    )
    semantic_mode = (
        "absolute"
        if semantic_absolute_applicable
        else "relative"
        if semantic_relative_applicable
        else "abstained"
    )
    applicable["semantic"] = semantic_applicable
    applicable["semantic_absolute"] = semantic_absolute_applicable
    applicable["semantic_relative"] = semantic_relative_applicable
    relative_bilateral = bool(
        semantic_relative_applicable
        and target_direction_available
        and source_direction_available
    )
    relative_target_only = bool(
        semantic_relative_applicable
        and target_direction_available
        and not source_direction_available
    )
    relative_source_only = bool(
        semantic_relative_applicable
        and source_direction_available
        and not target_direction_available
    )
    applicable["semantic_relative_bilateral"] = relative_bilateral
    applicable["semantic_relative_target_only"] = relative_target_only
    applicable["semantic_relative_source_only"] = relative_source_only
    metrics.update(
        {
            "semantic_source_region_accuracy": source_accuracy,
            "semantic_source_macro_miou": source_macro,
            "semantic_source_transition_recall_min": source_recall,
            "semantic_target_reference_available": float(target_available),
            "semantic_target_reference_recall_min": target_recall,
            "semantic_source_to_target_confusion": source_target_confusion,
            "semantic_transition_pixels": float(transition_pixels),
            "semantic_mode_absolute": float(semantic_mode == "absolute"),
            "semantic_mode_relative": float(semantic_mode == "relative"),
            "semantic_relative_bilateral": float(relative_bilateral),
            "semantic_relative_target_only": float(relative_target_only),
            "semantic_relative_source_only": float(relative_source_only),
            "semantic_relative_target_support_pixels": float(
                target_direction_support
            ),
            "semantic_relative_source_support_pixels": float(
                source_direction_support
            ),
            "semantic_relative_margin_support_pixels": float(
                margin_direction_support
            ),
            "semantic_direction_epsilon": metric_direction_epsilon,
        }
    )
    if semantic_absolute_applicable:
        target_accuracy = _number(
            base_metrics.get(
                "semantic_gate_accuracy", changed.get("accuracy")
            ),
            0.0,
        )
        target_macro = _number(
            changed.get("macro_miou"),
            0.0,
        )
        source_accuracy_baseline = _number(
            base_metrics.get(
                "semantic_gate_no_edit_accuracy",
                changed.get("no_edit_accuracy"),
            ),
            0.0,
        )
        accuracy_gain = _normalized_gain(
            target_accuracy, source_accuracy_baseline
        )
        generated_margin = _number(
            changed.get("soft_target_source_margin"), -1.0
        )
        source_margin = _number(
            changed.get("soft_no_edit_target_source_margin"), -1.0
        )
        margin_gain = _normalized_gain(generated_margin, source_margin)
        semantic_score = _clip01(
            0.35 * target_accuracy
            + 0.35 * target_macro
            + 0.15 * accuracy_gain
            + 0.15 * margin_gain
        )
        components["semantic"] = semantic_score
        metrics.update(
            {
                "semantic_target_accuracy": target_accuracy,
                "semantic_target_macro_miou": target_macro,
                "semantic_no_edit_accuracy": source_accuracy_baseline,
                "semantic_normalized_accuracy_gain": accuracy_gain,
                "semantic_generated_target_source_margin": generated_margin,
                "semantic_no_edit_target_source_margin": source_margin,
                "semantic_normalized_margin_gain": margin_gain,
                "semantic_evidence_weight": float(weights["semantic"]),
            }
        )
    elif semantic_relative_applicable:
        direction_terms: list[tuple[float, float]] = []
        target_direction = None
        source_direction = None
        margin_direction = None
        if target_direction_available:
            target_direction = _direction_score(
                float(target_direction_gain), metric_direction_epsilon
            )
            direction_terms.append((0.40, target_direction))
        if source_direction_available:
            source_direction = _direction_score(
                float(source_direction_gain), metric_direction_epsilon
            )
            direction_terms.append((0.40, source_direction))
        if margin_direction_available:
            margin_direction = _direction_score(
                float(margin_direction_gain), metric_direction_epsilon
            )
            direction_terms.append((0.20, margin_direction))
        active_direction_weight = sum(weight for weight, _ in direction_terms)
        semantic_score = _clip01(
            sum(weight * value for weight, value in direction_terms)
            / active_direction_weight
        )
        components["semantic"] = semantic_score
        metrics.update(
            {
                "semantic_relative_target_direction_fraction": (
                    target_direction if target_direction is not None else 0.0
                ),
                "semantic_relative_source_suppression_fraction": (
                    source_direction if source_direction is not None else 0.0
                ),
                "semantic_relative_margin_direction_fraction": (
                    margin_direction if margin_direction is not None else 0.0
                ),
                "semantic_relative_active_direction_weight": (
                    active_direction_weight
                ),
                "semantic_relative_target_probability_gain": _number(
                    target_direction_gain, 0.0
                ),
                "semantic_relative_source_probability_suppression": _number(
                    source_direction_gain, 0.0
                ),
                "semantic_relative_margin_gain": _number(
                    margin_direction_gain, 0.0
                ),
                "semantic_relative_uses_appearance_calibration": float(
                    any(
                        key in changed
                        for key in (
                            "appearance_calibrated_target_probability_gain",
                            "appearance_calibrated_source_probability_suppression",
                            "appearance_calibrated_soft_margin_gain",
                        )
                    )
                ),
                "semantic_evidence_weight": (
                    policy.relative_semantic_evidence_weight
                ),
            }
        )
    else:
        reasons.append("semantic_evaluator_unreliable")
        metrics["semantic_evidence_weight"] = 0.0

    raw_drift = _number(
        preservation.get(
            "prediction_relative_drift_U_far",
            base_metrics.get("off_target_drift"),
        ),
        1.0,
    )
    calibrated_drift = (
        float(preservation["appearance_calibrated_prediction_drift_U_far"])
        if bool(preservation.get("appearance_calibration_applicable"))
        and _is_number(
            preservation.get(
                "appearance_calibrated_prediction_drift_U_far"
            )
        )
        else None
    )
    drift = calibrated_drift if calibrated_drift is not None else 1.0
    u_far_pixels = int(
        _mapping(coarse_metrics.get("region_pixels")).get("U_far") or 0
    )
    applicable["preservation"] = bool(
        u_far_pixels > 0 and calibrated_drift is not None
    )
    applicable["preservation_appearance_calibrated"] = applicable[
        "preservation"
    ]
    metrics.update(
        {
            "preservation_raw_drift_u_far": raw_drift,
            "preservation_drift_u_far": drift,
            "preservation_appearance_calibration_coverage": _number(
                preservation.get("appearance_calibration_coverage_U_far"),
                0.0,
            ),
            "preservation_global_appearance_probability_shift_l1": _number(
                preservation.get("global_appearance_probability_shift_l1"),
                0.0,
            ),
        }
    )
    if applicable["preservation"]:
        components["preservation"] = _clip01(1.0 - drift)
    else:
        reasons.append("preservation_evaluator_unavailable")

    source_boundary_f1 = _number(
        source_quality_metrics.get("source_boundary_f1_4"), 0.0
    )
    boundary_pixels = int(
        _mapping(coarse_metrics.get("region_pixels")).get("B") or 0
    )
    boundary_absolute_applicable = bool(
        source_boundary_f1 >= policy.source_boundary_f1_min
        and boundary_pixels >= policy.boundary_support_min_pixels
    )
    boundary_target_gain = _first_number(
        boundary, "relative_inner_target_probability_gain"
    )
    boundary_source_gain = _first_number(
        boundary, "relative_inner_source_probability_suppression"
    )
    boundary_margin_gain = _first_number(
        boundary, "relative_inner_margin_gain"
    )
    boundary_inner_support = max(
        int(boundary.get("relative_inner_target_support_pixels") or 0),
        int(boundary.get("relative_inner_source_support_pixels") or 0),
    )
    boundary_relative_applicable = bool(
        not boundary_absolute_applicable
        and boundary_pixels >= policy.boundary_support_min_pixels
        and boundary_inner_support >= 64
        and bool(boundary.get("relative_outer_applicable"))
        and _is_number(boundary.get("relative_outer_drift"))
        and (boundary_target_gain is not None or boundary_source_gain is not None)
    )
    boundary_applicable = bool(
        boundary_absolute_applicable or boundary_relative_applicable
    )
    applicable["boundary"] = boundary_applicable
    applicable["boundary_absolute"] = boundary_absolute_applicable
    applicable["boundary_relative"] = boundary_relative_applicable
    metrics.update(
        {
            "boundary_source_f1_4": source_boundary_f1,
            "boundary_support_pixels": float(boundary_pixels),
            "boundary_inner_relative_support_pixels": float(
                boundary_inner_support
            ),
            "boundary_mode_absolute": float(boundary_absolute_applicable),
            "boundary_mode_relative": float(boundary_relative_applicable),
        }
    )
    if boundary_absolute_applicable:
        boundary_f1 = _number(boundary.get("class_aware_f1_4"), 0.0)
        inner_error = _number(
            preservation.get("inner_ring_target_error"), 1.0
        )
        outer_spillover = _number(
            preservation.get("appearance_calibrated_outer_ring_spillover")
            if bool(
                preservation.get(
                    "appearance_calibrated_outer_ring_applicable"
                )
            )
            else preservation.get("outer_ring_spillover"),
            1.0,
        )
        components["boundary"] = _clip01(
            0.50 * boundary_f1
            + 0.25 * (1.0 - inner_error)
            + 0.25 * (1.0 - outer_spillover)
        )
        metrics.update(
            {
                "boundary_f1_4": boundary_f1,
                "boundary_inner_target_error": inner_error,
                "boundary_outer_spillover": outer_spillover,
            }
        )
    elif boundary_relative_applicable:
        boundary_terms: list[tuple[float, float]] = []
        if boundary_target_gain is not None:
            boundary_terms.append(
                (
                    0.40,
                    _direction_score(
                        boundary_target_gain, metric_direction_epsilon
                    ),
                )
            )
        if boundary_source_gain is not None:
            boundary_terms.append(
                (
                    0.40,
                    _direction_score(
                        boundary_source_gain, metric_direction_epsilon
                    ),
                )
            )
        if boundary_margin_gain is not None:
            boundary_terms.append(
                (
                    0.20,
                    _direction_score(
                        boundary_margin_gain, metric_direction_epsilon
                    ),
                )
            )
        boundary_direction_weight = sum(
            weight for weight, _ in boundary_terms
        )
        boundary_direction_score = sum(
            weight * value for weight, value in boundary_terms
        ) / boundary_direction_weight
        boundary_outer_drift = _number(
            boundary.get("relative_outer_drift"), 1.0
        )
        components["boundary"] = _clip01(
            0.50 * boundary_direction_score
            + 0.50 * (1.0 - boundary_outer_drift)
        )
        metrics.update(
            {
                "boundary_relative_direction_score": (
                    boundary_direction_score
                ),
                "boundary_relative_outer_drift": boundary_outer_drift,
                "boundary_relative_target_gain": _number(
                    boundary_target_gain, 0.0
                ),
                "boundary_relative_source_suppression": _number(
                    boundary_source_gain, 0.0
                ),
                "boundary_relative_margin_gain": _number(
                    boundary_margin_gain, 0.0
                ),
            }
        )
    else:
        reasons.append("boundary_evaluator_unreliable")

    nuclei = _nuclei_scores(
        source_calibration=source_nuclei_calibration,
        target_counts=target_nuclei_counts,
        generated_counts=generated_nuclei_counts,
        policy=policy,
    )
    metrics.update(nuclei["metrics"])
    for name in ("nuclei_count", "nuclei_type"):
        applicable[name] = bool(nuclei["applicability"][name])
        if applicable[name]:
            components[name] = float(nuclei["scores"][name])
        else:
            reasons.append(f"{name}_evaluator_unreliable")

    effective_weights = dict(weights)
    if semantic_relative_applicable:
        effective_weights["semantic"] = (
            policy.relative_semantic_evidence_weight
        )
    active_weight = sum(effective_weights[name] for name in components)
    quality_score = (
        sum(
            effective_weights[name] * components[name]
            for name in components
        )
        / active_weight
        if active_weight
        else 0.0
    )
    evidence_coverage = float(active_weight)

    failed: list[str] = []
    if quality_score < policy.quality_score_min:
        failed.append("quality_score")
        reasons.append("quality_score_below_threshold")
    required_evidence_coverage = (
        policy.relative_evidence_coverage_min
        if semantic_relative_applicable
        else policy.evidence_coverage_min
    )
    metrics["evidence_coverage_required"] = required_evidence_coverage
    if evidence_coverage + 1e-9 < required_evidence_coverage:
        failed.append("evidence_coverage")
        reasons.append("insufficient_reliable_evidence")
    if not semantic_applicable:
        failed.append("semantic_evaluator_unavailable")
    elif semantic_absolute_applicable and (
        components["semantic"] < policy.semantic_score_min
    ):
        failed.append("semantic_score")
        reasons.append("changed_region_semantic_mismatch")
    elif semantic_relative_applicable and (
        components["semantic"] < policy.relative_semantic_score_min
    ):
        failed.append("relative_semantic_direction")
        reasons.append("changed_region_semantic_direction_mismatch")
    if applicable["preservation"] and drift > policy.off_target_drift_max:
        failed.append("off_target_drift")
        reasons.append("unedited_region_semantic_drift")
    if (
        applicable["nuclei_count"]
        and nuclei["metrics"]["nuclei_calibrated_count_error"]
        > policy.nuclei_detection_error_max
    ):
        failed.append("nuclei_detection_count_relative_error")
        reasons.append("nuclei_count_mismatch")
    if (
        applicable["nuclei_type"]
        and nuclei["metrics"]["nuclei_type_composition_tv_error"]
        > policy.nuclei_type_error_max
    ):
        failed.append("nuclei_type_composition_error")
        reasons.append("nuclei_type_composition_mismatch")

    failed_checks = tuple(dict.fromkeys(failed))
    reason_codes = tuple(dict.fromkeys(reasons))
    passed = not failed_checks
    if passed:
        status = "validated"
    elif (
        not semantic_applicable
        or evidence_coverage + 1e-9 < required_evidence_coverage
    ):
        status = "evaluator_uncertain"
    else:
        status = "needs_review"
    return QualityEvaluation(
        passed=passed,
        quality_score=float(quality_score),
        evidence_coverage=evidence_coverage,
        component_scores=components,
        applicability=applicable,
        metrics=metrics,
        failed_checks=failed_checks,
        reason_codes=reason_codes,
        scientific_status=status,
        policy=policy,
    )


def build_generation_report(workflow: Mapping[str, Any]) -> dict[str, Any]:
    """Create a deterministic human-readable report from workflow metadata."""

    attempts = list(workflow.get("attempts") or [])
    selected = _mapping(workflow.get("selected_attempt"))
    route = _mapping(workflow.get("route"))
    selected_index = selected.get("attempt_index")
    attempt_reports = []
    for attempt in attempts:
        verification = _mapping(attempt.get("verification"))
        verification_metrics = _mapping(verification.get("metrics"))
        semantic_mode = (
            "absolute"
            if bool(verification_metrics.get("semantic_mode_absolute"))
            else "relative"
            if bool(verification_metrics.get("semantic_mode_relative"))
            else "abstained"
        )
        semantic_variant = None
        if semantic_mode == "relative":
            if bool(
                verification_metrics.get("semantic_relative_target_only")
            ):
                semantic_variant = "target-only"
            elif bool(
                verification_metrics.get("semantic_relative_source_only")
            ):
                semantic_variant = "source-only"
            else:
                semantic_variant = "bilateral"
        reason_codes = list(verification.get("reason_codes") or [])
        attempt_reports.append(
            {
                "attempt_index": attempt.get("attempt_index"),
                "model": attempt.get("requested_mode"),
                "decision_reason": attempt.get("decision_reason"),
                "error": attempt.get("error"),
                "passed": verification.get("passed"),
                "quality_score": verification.get(
                    "quality_score", verification.get("score")
                ),
                "evidence_coverage": verification.get("evidence_coverage"),
                "semantic_mode": semantic_mode,
                "semantic_variant": semantic_variant,
                "component_scores": dict(
                    verification.get("component_scores") or {}
                ),
                "applicability": dict(
                    verification.get("applicability") or {}
                ),
                "failed_checks": list(
                    verification.get("failed_checks") or []
                ),
                "reason_codes": reason_codes,
                "assessment": [_reason_text(code) for code in reason_codes],
                "selected": attempt.get("attempt_index") == selected_index,
            }
        )
    selected_verification = _mapping(selected.get("verification"))
    selected_reasons = list(selected_verification.get("reason_codes") or [])
    if not selected_reasons and selected_verification.get("passed"):
        selected_reasons = ["engineering_quality_validated"]
    workflow_status = str(workflow.get("status") or "")
    if workflow_status == "validated_first_pass":
        selection_rationale = (
            "首次候选完整通过冻结 evaluator，未调用另一模型。"
        )
    elif workflow_status == "recovered":
        selection_rationale = (
            "首次候选未完整通过；另一模型通过冻结 evaluator，选为最终图。"
        )
    else:
        selection_rationale = (
            "两种模型均未完整通过；按 validated 状态、quality score、"
            "semantic score、preservation score、较早 attempt 的固定顺序"
            "选出最终候选。"
        )
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "policy_id": QUALITY_POLICY_ID,
        "workflow_status": workflow.get("status"),
        "initial_route": route.get("primary_mode"),
        "initial_route_reason": route.get("reason"),
        "alternate_model_triggered": len(attempts) > 1,
        "selected_attempt": selected_index,
        "selected_model": selected.get("requested_mode"),
        "selected_quality_score": selected_verification.get(
            "quality_score", selected_verification.get("score")
        ),
        "selected_scientific_status": selected_verification.get(
            "scientific_status"
        ),
        "selection_rationale": selection_rationale,
        "attempts": attempt_reports,
        "final_assessment": [_reason_text(code) for code in selected_reasons],
        "validated_interpretation": (
            "自动工程验证通过，不代表临床正确或人工病理复核通过。"
            if bool(selected_verification.get("passed"))
            else "最终候选未通过自动工程验证，必须保留人工复核状态。"
        ),
    }


def write_generation_report(
    workflow: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Write canonical JSON plus a concise Chinese Markdown report."""

    import json

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = build_generation_report(workflow)
    json_path = output / "generation_report.json"
    markdown_path = output / "generation_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# 生成质量报告",
        "",
        f"- 工作流状态：`{report['workflow_status']}`",
        f"- 首次路由：`{report['initial_route']}`",
        f"- 路由原因：{report['initial_route_reason']}",
        f"- 是否调用另一模型：{'是' if report['alternate_model_triggered'] else '否'}",
        f"- 最终模型：`{report['selected_model']}`",
        f"- 最终质量分：{_format_score(report['selected_quality_score'])}",
        f"- 选图依据：{report['selection_rationale']}",
        "",
        "## 候选",
    ]
    for attempt in report["attempts"]:
        lines.extend(
            [
                "",
                (
                    f"### Attempt {attempt['attempt_index']}："
                    f"`{attempt['model']}`"
                ),
                f"- 质量分：{_format_score(attempt['quality_score'])}",
                f"- 证据覆盖：{_format_score(attempt['evidence_coverage'])}",
                (
                    f"- Semantic 模式：`{attempt['semantic_mode']}`"
                    + (
                        f"（`{attempt['semantic_variant']}`）"
                        if attempt["semantic_variant"]
                        else ""
                    )
                ),
                f"- 是否通过：{'是' if attempt['passed'] else '否'}",
                (
                    "- 评价："
                    + (
                        "；".join(attempt["assessment"])
                        if attempt["assessment"]
                        else "未发现自动 evaluator 可确认的问题。"
                    )
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 最终结论",
            "",
            "；".join(report["final_assessment"])
            if report["final_assessment"]
            else "最终候选已按冻结规则选出。",
            "",
            report["validated_interpretation"],
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path, report


def _nuclei_scores(
    *,
    source_calibration: Mapping[str, Any] | None,
    target_counts: Mapping[int, int] | None,
    generated_counts: Mapping[int, int] | None,
    policy: QualityPolicy,
) -> dict[str, Any]:
    metrics: dict[str, float] = {
        "nuclei_calibrated_count_error": 1.0,
        "nuclei_type_composition_tv_error": 1.0,
    }
    applicability = {"nuclei_count": False, "nuclei_type": False}
    scores: dict[str, float] = {}
    if (
        source_calibration is None
        or target_counts is None
        or generated_counts is None
    ):
        return {
            "metrics": metrics,
            "applicability": applicability,
            "scores": scores,
        }

    changed = _mapping(source_calibration.get("changed_region"))
    full = _mapping(source_calibration.get("full_image"))
    changed_reference = _counts(changed.get("reference"))
    changed_predicted = _counts(changed.get("predicted"))
    full_reference = _counts(full.get("reference"))
    full_predicted = _counts(full.get("predicted"))
    target = _counts(target_counts)
    generated = _counts(generated_counts)
    changed_reference_total = sum(changed_reference.values())
    full_reference_total = sum(full_reference.values())
    changed_error = _count_error(changed_reference, changed_predicted)
    full_error = _count_error(full_reference, full_predicted)

    calibration_name = None
    calibration_reference_total = 0
    calibration_predicted_total = 0
    if (
        changed_reference_total >= policy.nuclei_min_instances
        and changed_error <= policy.nuclei_detection_error_max
    ):
        calibration_name = "changed_region"
        calibration_reference_total = changed_reference_total
        calibration_predicted_total = sum(changed_predicted.values())
    elif (
        full_reference_total >= policy.nuclei_min_instances
        and full_error <= policy.nuclei_detection_error_max
    ):
        calibration_name = "full_image"
        calibration_reference_total = full_reference_total
        calibration_predicted_total = sum(full_predicted.values())

    metrics.update(
        {
            "nuclei_source_changed_detection_error": changed_error,
            "nuclei_source_full_detection_error": full_error,
            "nuclei_target_instance_count": float(sum(target.values())),
            "nuclei_generated_instance_count": float(sum(generated.values())),
        }
    )
    if calibration_name is None:
        return {
            "metrics": metrics,
            "applicability": applicability,
            "scores": scores,
        }

    alpha = _clip(
        calibration_predicted_total / max(1, calibration_reference_total),
        0.5,
        1.5,
    )
    expected = alpha * sum(target.values())
    generated_total = sum(generated.values())
    calibrated_error = abs(generated_total - expected) / max(expected, 10.0)
    applicability["nuclei_count"] = True
    scores["nuclei_count"] = _clip01(1.0 - min(calibrated_error, 1.0))
    metrics.update(
        {
            "nuclei_detection_calibration_alpha": alpha,
            "nuclei_expected_detected_count": expected,
            "nuclei_calibrated_count_error": calibrated_error,
            "nuclei_detection_calibration_local": float(
                calibration_name == "changed_region"
            ),
        }
    )

    source_type_reference = (
        changed_reference
        if calibration_name == "changed_region"
        else full_reference
    )
    source_type_predicted = (
        changed_predicted
        if calibration_name == "changed_region"
        else full_predicted
    )
    source_type_error = _type_tv(
        source_type_reference,
        source_type_predicted,
    )
    generated_type_error = _type_tv(target, generated)
    metrics["nuclei_source_type_composition_tv_error"] = source_type_error
    metrics["nuclei_type_composition_tv_error"] = generated_type_error
    type_applicable = bool(
        source_type_error <= policy.nuclei_type_error_max
        and sum(target.values()) >= policy.nuclei_min_instances
        and sum(generated.values()) >= policy.nuclei_min_instances
    )
    applicability["nuclei_type"] = type_applicable
    if type_applicable:
        scores["nuclei_type"] = _clip01(1.0 - generated_type_error)
    return {
        "metrics": metrics,
        "applicability": applicability,
        "scores": scores,
    }


def _normalized_gain(generated: float, source: float) -> float:
    return _clip01((generated - source) / max(1e-6, 1.0 - source))


def _direction_score(value: float, epsilon: float) -> float:
    if value > epsilon:
        return 1.0
    if value < -epsilon:
        return 0.0
    return 0.5


def _first_number(
    values: Mapping[str, Any], *keys: str
) -> float | None:
    for key in keys:
        value = values.get(key)
        if _is_number(value):
            return float(value)
    return None


def _count_error(reference: Mapping[int, int], predicted: Mapping[int, int]) -> float:
    reference_total = sum(reference.values())
    predicted_total = sum(predicted.values())
    if reference_total == 0:
        return 0.0 if predicted_total == 0 else 1.0
    return abs(predicted_total - reference_total) / reference_total


def _type_tv(reference: Mapping[int, int], predicted: Mapping[int, int]) -> float:
    reference_total = sum(reference.values())
    predicted_total = sum(predicted.values())
    if reference_total == 0 or predicted_total == 0:
        return 0.0 if reference_total == predicted_total else 1.0
    return 0.5 * sum(
        abs(
            reference.get(label, 0) / reference_total
            - predicted.get(label, 0) / predicted_total
        )
        for label in set(reference) | set(predicted)
    )


def _reason_text(code: str) -> str:
    return {
        "engineering_quality_validated": (
            "编辑区域、未编辑区域和适用的细胞证据均通过冻结工程阈值。"
        ),
        "quality_score_below_threshold": (
            "综合生成质量未达到自动验证阈值。"
        ),
        "insufficient_reliable_evidence": (
            "本例可用的可靠 evaluator 证据不足，需要人工复核。"
        ),
        "semantic_evaluator_unreliable": (
            "Segmentator 在原图对应组织上的校准不足，语义分数未用于放行。"
        ),
        "changed_region_semantic_mismatch": (
            "编辑区域生成与目标语义要求不符或转化不充分。"
        ),
        "changed_region_semantic_direction_mismatch": (
            "编辑区域相对原图没有稳定地朝目标语义方向变化。"
        ),
        "unedited_region_semantic_drift": (
            "未编辑区域出现超过容许范围的语义漂移。"
        ),
        "boundary_evaluator_unreliable": (
            "Segmentator 在原图边界上的可靠性不足，边界分数仅作诊断。"
        ),
        "preservation_evaluator_unavailable": (
            "没有足够的未编辑远区用于评估语义保持。"
        ),
        "nuclei_count_evaluator_unreliable": (
            "CellViT 在原图上的核检测数量不可靠，核数分数未用于放行。"
        ),
        "nuclei_type_evaluator_unreliable": (
            "CellViT 在原图上的细胞判型不可靠，类型分数未用于放行。"
        ),
        "nuclei_count_mismatch": (
            "编辑区域的细胞数量与目标 condition 不一致。"
        ),
        "nuclei_type_composition_mismatch": (
            "编辑区域的细胞类型组成与目标 condition 不一致。"
        ),
    }.get(code, f"自动 evaluator 记录了问题：{code}。")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _counts(value: Any) -> dict[int, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        int(label): max(0, int(count))
        for label, count in value.items()
        if int(label) != 0
    }


def _number(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clip01(value: float) -> float:
    return _clip(value, 0.0, 1.0)


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def _format_score(value: Any) -> str:
    if value is None:
        return "不可用"
    return f"{float(value):.4f}"


__all__ = [
    "QUALITY_POLICY_ID",
    "QUALITY_SCHEMA_VERSION",
    "QualityEvaluation",
    "QualityPolicy",
    "build_generation_report",
    "evaluate_product_quality",
    "write_generation_report",
]
