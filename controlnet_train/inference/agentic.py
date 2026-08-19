"""Bounded agentic orchestration for pathology image generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .router import (
    AgenticRouteFeatures,
    AgenticRoutingConfig,
    AgenticRoutingDecision,
    route_agentic_edit_request,
)

PRODUCTION_CROSS_MODE = "cross-v1-no-ip-pix2pix-v2"


@dataclass(frozen=True)
class FidelityThresholds:
    changed_region_accuracy_min: float = 0.70
    changed_region_macro_iou_min: float = 0.55
    off_target_drift_max: float = 0.08
    quality_score_min: float = 0.75
    evidence_coverage_min: float = 0.80
    semantic_score_min: float = 0.60
    source_boundary_f1_min: float = 0.45
    boundary_support_min_pixels: int = 256
    nuclei_count_relative_error_max: float = 0.35
    nuclei_type_composition_error_max: float = 0.35
    nuclei_type_min_instances: int = 10
    semantic_boundary_tolerance_pixels: int = 4
    semantic_small_region_ratio_max: float = 0.05
    semantic_core_min_pixels: int = 256
    semantic_core_min_fraction: float = 0.20


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    score: float
    metrics: Mapping[str, float]
    failed_checks: tuple[str, ...] = ()
    schema_version: int = 2
    component_scores: Mapping[str, float] = field(default_factory=dict)
    applicability: Mapping[str, bool] = field(default_factory=dict)
    evidence_coverage: float = 0.0
    quality_score: float | None = None
    scientific_status: str = "not_evaluated"
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.quality_score is None:
            object.__setattr__(self, "quality_score", float(self.score))


@dataclass(frozen=True)
class GenerationArtifact:
    mode: str
    image_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgenticAttempt:
    attempt_index: int
    requested_mode: str
    decision_reason: str
    artifact: GenerationArtifact | None
    verification: VerificationResult | None
    error: str | None = None


@dataclass(frozen=True)
class RecoveryDecision:
    next_mode: str | None
    action: str
    reason: str


@dataclass(frozen=True)
class AgenticWorkflowConfig:
    routing: AgenticRoutingConfig = field(default_factory=AgenticRoutingConfig)
    max_attempts: int = 2
    cross_mode: str = PRODUCTION_CROSS_MODE


@dataclass(frozen=True)
class AgenticWorkflowResult:
    status: str
    route: AgenticRoutingDecision
    attempts: tuple[AgenticAttempt, ...]
    selected_attempt: AgenticAttempt | None
    output_dir: Path

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "route": _route_metadata(self.route),
            "attempts": [_attempt_metadata(item) for item in self.attempts],
            "selected_attempt": (
                None
                if self.selected_attempt is None
                else _attempt_metadata(self.selected_attempt)
            ),
            "output_dir": str(self.output_dir),
        }


Generator = Callable[[str, Path], GenerationArtifact]
Verifier = Callable[[GenerationArtifact], VerificationResult]
RecoveryPolicy = Callable[
    [AgenticRoutingDecision, tuple[AgenticAttempt, ...], AgenticWorkflowConfig],
    RecoveryDecision,
]


def run_agentic_workflow(
    *,
    reference_tissue_mask: np.ndarray,
    target_tissue_mask: np.ndarray,
    output_dir: str | Path,
    generate: Generator,
    verify: Verifier,
    config: AgenticWorkflowConfig | None = None,
    recovery_policy: RecoveryPolicy | None = None,
    routing_decision: AgenticRoutingDecision | None = None,
    routing_change_region: np.ndarray | None = None,
    routing_change_scope: str = "tissue",
    routing_cell_only_direction: str | None = None,
    routing_edit_primitive_id: str | None = None,
) -> AgenticWorkflowResult:
    """Route, generate, verify, and perform bounded failure-aware recovery."""

    config = config or AgenticWorkflowConfig()
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    route = routing_decision or route_agentic_edit_request(
        reference_tissue_mask,
        target_tissue_mask,
        config=config.routing,
        change_region=routing_change_region,
        change_scope=routing_change_scope,
        cell_only_direction=routing_cell_only_direction,
        edit_primitive_id=routing_edit_primitive_id,
    )
    if route.primary_mode == "noop":
        result = AgenticWorkflowResult(
            status="noop",
            route=route,
            attempts=(),
            selected_attempt=None,
            output_dir=output_dir,
        )
        _save_result(result)
        return result

    allowed_modes = tuple(
        config.cross_mode if mode == "cross" else mode
        for mode in route.candidate_modes
    )
    recovery_policy = recovery_policy or default_recovery_policy
    attempts: list[AgenticAttempt] = []
    mode = config.cross_mode if route.primary_mode == "cross" else route.primary_mode
    decision_reason = f"initial route: {route.reason}"
    for index in range(1, config.max_attempts + 1):
        if mode not in allowed_modes:
            break
        attempt_dir = output_dir / f"attempt_{index:02d}_{_safe_mode(mode)}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        try:
            artifact = generate(mode, attempt_dir)
        except Exception as exc:  # generation tool boundary
            attempt = AgenticAttempt(
                index,
                mode,
                decision_reason,
                None,
                None,
                error=f"generation failed: {exc}",
            )
        else:
            try:
                verification = verify(artifact)
            except Exception as exc:  # verifier tool boundary; keep generated artifact
                attempt = AgenticAttempt(
                    index,
                    mode,
                    decision_reason,
                    artifact,
                    None,
                    error=f"verification failed: {exc}",
                )
            else:
                attempt = AgenticAttempt(
                    index,
                    mode,
                    decision_reason,
                    artifact,
                    verification,
                )
        attempts.append(attempt)
        _save_running_state(route, attempts, output_dir)
        if attempt.verification is not None and attempt.verification.passed:
            result = AgenticWorkflowResult(
                status=(
                    "validated_first_pass"
                    if attempt.attempt_index == 1
                    else "recovered"
                ),
                route=route,
                attempts=tuple(attempts),
                selected_attempt=attempt,
                output_dir=output_dir,
            )
            _save_result(result)
            return result
        recovery = recovery_policy(route, tuple(attempts), config)
        if recovery.next_mode is None:
            break
        mode = recovery.next_mode
        decision_reason = f"{recovery.action}: {recovery.reason}"

    verified = [item for item in attempts if item.verification is not None]
    selected = max(verified, key=_attempt_selection_key) if verified else None
    status = (
        "evaluator_uncertain"
        if selected is not None
        and selected.verification is not None
        and selected.verification.scientific_status == "evaluator_uncertain"
        else "needs_review"
    )
    result = AgenticWorkflowResult(
        status=status,
        route=route,
        attempts=tuple(attempts),
        selected_attempt=selected,
        output_dir=output_dir,
    )
    _save_result(result)
    return result


def default_recovery_policy(
    route: AgenticRoutingDecision,
    attempts: tuple[AgenticAttempt, ...],
    config: AgenticWorkflowConfig,
) -> RecoveryDecision:
    """Choose one untried backend from structured failure evidence."""

    del route
    if not attempts:
        return RecoveryDecision(None, "stop", "no attempt is available to inspect")
    allowed = tuple(
        config.cross_mode if mode == "cross" else mode
        for mode in ("inpaint", "cross")
    )
    tried = {attempt.requested_mode for attempt in attempts}
    untried = [mode for mode in allowed if mode not in tried]
    if not untried:
        return RecoveryDecision(None, "stop", "all allowed generation backends were tried")

    last = attempts[-1]
    if last.error:
        return RecoveryDecision(
            untried[0],
            "tool_error_fallback",
            f"{last.requested_mode} raised an error; try the remaining backend",
        )

    failed = set(last.verification.failed_checks if last.verification else ())
    evaluator_abstentions = {
        "coarse_evaluator_source_calibration",
        "fine_evaluator_source_calibration",
        "nuclei_evaluator_source_detection_calibration",
        "semantic_evaluator_scale_calibration",
        "semantic_evaluator_source_calibration",
        "evidence_coverage",
    }
    if failed & evaluator_abstentions:
        return RecoveryDecision(
            untried[0],
            "evaluator_uncertainty_comparison",
            "the evaluator cannot auto-validate this case, so generate the "
            "remaining backend for a reliable-evidence comparison",
        )
    if "off_target_drift" in failed and "inpaint" in untried:
        return RecoveryDecision(
            "inpaint",
            "preservation_recovery",
            "off-target drift exceeded the limit, so use the local-preserving backend",
        )
    structure_failures = {
        "changed_region_accuracy",
        "changed_region_macro_iou",
        "nuclei_detection_count_relative_error",
        "nuclei_type_composition_error",
    }
    if failed & structure_failures and config.cross_mode in untried:
        return RecoveryDecision(
            config.cross_mode,
            "structure_recovery",
            "target structure or nuclei fidelity failed, so use production cross generation",
        )
    return RecoveryDecision(
        untried[0],
        "generic_backend_fallback",
        "verification failed without a more specific recovery rule",
    )


def verify_mask_fidelity(
    *,
    reference_tissue_mask: np.ndarray,
    target_tissue_mask: np.ndarray,
    predicted_tissue_mask: np.ndarray,
    source_predicted_tissue_mask: np.ndarray | None = None,
    change_region: np.ndarray,
    target_nuclei_mask: np.ndarray | None = None,
    predicted_nuclei_mask: np.ndarray | None = None,
    target_nuclei_instance_counts: Mapping[int, int] | None = None,
    predicted_nuclei_instance_counts: Mapping[int, int] | None = None,
    thresholds: FidelityThresholds | None = None,
    enforce_off_target_drift: bool = True,
) -> VerificationResult:
    """Deterministically score re-segmented output against target masks.

    The score is based only on target-region tissue and nuclei consistency so
    local and global backends remain comparable. Off-target drift is retained
    as a separate metric and can be enabled as a route-specific acceptance
    gate, notably for full-patch Cross-v1 generation.
    """

    thresholds = thresholds or FidelityThresholds()
    reference = np.asarray(reference_tissue_mask)
    target = np.asarray(target_tissue_mask)
    predicted = np.asarray(predicted_tissue_mask)
    source_predicted = (
        reference
        if source_predicted_tissue_mask is None
        else np.asarray(source_predicted_tissue_mask)
    )
    change = np.asarray(change_region, dtype=bool)
    if not (
        reference.shape
        == target.shape
        == predicted.shape
        == source_predicted.shape
        == change.shape
    ):
        raise ValueError("Tissue masks and change_region must have identical shapes.")
    changed_count = int(np.count_nonzero(change))
    changed_accuracy = (
        float(np.mean(predicted[change] == target[change])) if changed_count else 1.0
    )
    semantic_region, semantic_scale = _semantic_evaluation_region(
        change,
        thresholds=thresholds,
    )
    semantic_gate_accuracy = (
        float(np.mean(predicted[semantic_region] == target[semantic_region]))
        if np.any(semantic_region)
        else 1.0
    )
    semantic_gate_macro_iou = _macro_iou(
        target,
        predicted,
        region=semantic_region,
    )
    outside = ~change
    off_target_drift = (
        float(np.mean(predicted[outside] != source_predicted[outside]))
        if np.any(outside)
        else 0.0
    )
    macro_iou = _macro_iou(target, predicted, region=change)
    metrics: dict[str, float] = {
        "changed_region_accuracy": changed_accuracy,
        "changed_region_macro_iou": macro_iou,
        "semantic_gate_accuracy": semantic_gate_accuracy,
        "semantic_gate_macro_iou": semantic_gate_macro_iou,
        **semantic_scale,
        "off_target_drift": off_target_drift,
    }
    if source_predicted_tissue_mask is not None:
        source_relative_tissue_accuracy = (
            float(np.mean(predicted[change] == source_predicted[change]))
            if np.any(change)
            else 1.0
        )
        source_relative_tissue_weighted_iou = _prevalence_weighted_iou(
            source_predicted,
            predicted,
            region=change,
        )
        no_edit_accuracy = (
            float(np.mean(source_predicted[change] == target[change]))
            if changed_count
            else 1.0
        )
        semantic_gate_no_edit_accuracy = (
            float(
                np.mean(
                    source_predicted[semantic_region]
                    == target[semantic_region]
                )
            )
            if np.any(semantic_region)
            else 1.0
        )
        no_edit_macro_iou = _macro_iou(
            target, source_predicted, region=change
        )
        metrics.update(
            {
                "no_edit_changed_region_accuracy": no_edit_accuracy,
                "semantic_gate_no_edit_accuracy": (
                    semantic_gate_no_edit_accuracy
                ),
                "cell_only_source_relative_tissue_accuracy": (
                    source_relative_tissue_accuracy
                ),
                "cell_only_source_relative_tissue_weighted_iou": (
                    source_relative_tissue_weighted_iou
                ),
                "target_gain_accuracy": changed_accuracy - no_edit_accuracy,
                "no_edit_changed_region_macro_iou": no_edit_macro_iou,
                "target_gain_macro_iou": macro_iou - no_edit_macro_iou,
            }
        )
    failed = []
    if not bool(semantic_scale["semantic_scale_evaluator_applicable"]):
        failed.append("semantic_evaluator_scale_calibration")
    else:
        if semantic_gate_accuracy < thresholds.changed_region_accuracy_min:
            failed.append("changed_region_accuracy")
        if semantic_gate_macro_iou < thresholds.changed_region_macro_iou_min:
            failed.append("changed_region_macro_iou")
    if enforce_off_target_drift and off_target_drift > thresholds.off_target_drift_max:
        failed.append("off_target_drift")

    nuclei_error = None
    if target_nuclei_mask is not None and predicted_nuclei_mask is not None:
        target_nuclei = np.asarray(target_nuclei_mask)
        predicted_nuclei = np.asarray(predicted_nuclei_mask)
        if target_nuclei.shape != change.shape or predicted_nuclei.shape != change.shape:
            raise ValueError("Nuclei masks must match tissue mask shape.")
        occupied_area_error = _nuclei_occupied_area_relative_error(
            target_nuclei, predicted_nuclei, region=change
        )
        target_counts = (
            semantic_mask_instance_counts(target_nuclei, region=change)
            if target_nuclei_instance_counts is None
            else _normalize_instance_counts(target_nuclei_instance_counts)
        )
        predicted_counts = (
            semantic_mask_instance_counts(predicted_nuclei, region=change)
            if predicted_nuclei_instance_counts is None
            else _normalize_instance_counts(predicted_nuclei_instance_counts)
        )
        nuclei_error = _nuclei_detection_count_relative_error(
            target_counts,
            predicted_counts,
        )
        nuclei_type_error = _nuclei_type_composition_tv_error(
            target_counts,
            predicted_counts,
        )
        target_instance_count = int(sum(target_counts.values()))
        predicted_instance_count = int(sum(predicted_counts.values()))
        type_sample_sufficient = (
            target_instance_count >= thresholds.nuclei_type_min_instances
            and predicted_instance_count >= thresholds.nuclei_type_min_instances
        )
        metrics["nuclei_occupied_area_relative_error"] = occupied_area_error
        metrics["nuclei_detection_count_relative_error"] = nuclei_error
        metrics["nuclei_type_composition_tv_error"] = nuclei_type_error
        metrics["nuclei_type_sample_sufficient"] = float(type_sample_sufficient)
        # Compatibility aliases now refer to total instance detection, never
        # to a macro average of per-type relative errors.
        metrics["nuclei_count_relative_error"] = nuclei_error
        metrics["nuclei_density_relative_error"] = nuclei_error
        metrics["nuclei_target_instance_count"] = float(target_instance_count)
        metrics["nuclei_predicted_instance_count"] = float(predicted_instance_count)
        for label in sorted(set(target_counts) | set(predicted_counts)):
            target_count = target_counts.get(label, 0)
            predicted_count = predicted_counts.get(label, 0)
            metrics[f"nuclei_target_count_{label}"] = float(target_count)
            metrics[f"nuclei_predicted_count_{label}"] = float(predicted_count)
            metrics[f"nuclei_count_relative_error_{label}"] = float(
                1.0
                if target_count == 0 and predicted_count > 0
                else abs(predicted_count - target_count) / max(1, target_count)
            )
        if nuclei_error > thresholds.nuclei_count_relative_error_max:
            failed.append("nuclei_detection_count_relative_error")
        if (
            type_sample_sufficient
            and nuclei_type_error > thresholds.nuclei_type_composition_error_max
        ):
            failed.append("nuclei_type_composition_error")

    score = 0.55 * semantic_gate_accuracy + 0.45 * semantic_gate_macro_iou
    if nuclei_error is not None:
        score = 0.85 * score + 0.15 * (1.0 - min(1.0, nuclei_error))
    return VerificationResult(
        passed=not failed,
        score=float(score),
        metrics=metrics,
        failed_checks=tuple(failed),
    )


def measure_local_detail_retention(
    *,
    reference_image: np.ndarray,
    generated_image: np.ndarray,
    edit_region: np.ndarray,
    context_radius_px: int = 24,
    context_gap_px: int = 3,
    interior_erosion_px: int = 2,
    minimum_region_pixels: int = 256,
) -> dict[str, float]:
    """Measure edit-region focus relative to nearby unchanged context.

    Absolute high-frequency energy legitimately changes when cellular content
    changes. The ratio-of-ratios instead asks whether the edited region became
    disproportionately softer than its local context after generation.
    """

    from scipy import ndimage

    reference = np.asarray(reference_image, dtype=np.float32)
    generated = np.asarray(generated_image, dtype=np.float32)
    region = np.asarray(edit_region, dtype=bool)
    if reference.shape != generated.shape or reference.ndim != 3:
        raise ValueError(
            "reference and generated images must be same-shape RGB arrays."
        )
    if region.shape != reference.shape[:2]:
        raise ValueError("edit_region must match the image height and width.")
    if min(
        context_radius_px,
        context_gap_px,
        interior_erosion_px,
        minimum_region_pixels,
    ) < 0:
        raise ValueError("Local-detail measurement parameters must be non-negative.")
    if context_radius_px <= context_gap_px:
        raise ValueError("context_radius_px must exceed context_gap_px.")

    interior = (
        ndimage.binary_erosion(region, iterations=interior_erosion_px)
        if interior_erosion_px
        else region.copy()
    )
    if np.count_nonzero(interior) < minimum_region_pixels:
        interior = region.copy()
    context = ndimage.binary_dilation(
        region, iterations=context_radius_px
    ) & ~ndimage.binary_dilation(region, iterations=context_gap_px)
    applicable = bool(
        np.count_nonzero(interior) >= minimum_region_pixels
        and np.count_nonzero(context) >= minimum_region_pixels
    )
    if not applicable:
        return {
            "local_detail_evaluator_applicable": 0.0,
            "local_detail_edit_pixels": float(np.count_nonzero(interior)),
            "local_detail_context_pixels": float(np.count_nonzero(context)),
        }

    def energies(image: np.ndarray, selected: np.ndarray) -> tuple[float, float]:
        scale = 255.0 if float(np.nanmax(image)) > 1.5 else 1.0
        normalized = image / scale
        gray = (
            0.299 * normalized[..., 0]
            + 0.587 * normalized[..., 1]
            + 0.114 * normalized[..., 2]
        )
        laplacian = ndimage.laplace(gray)
        gradient = np.hypot(
            ndimage.sobel(gray, axis=0),
            ndimage.sobel(gray, axis=1),
        )
        return (
            float(np.mean(np.square(laplacian[selected]))),
            float(np.mean(np.square(gradient[selected]))),
        )

    source_edit_lap, source_edit_gradient = energies(reference, interior)
    source_context_lap, source_context_gradient = energies(reference, context)
    generated_edit_lap, generated_edit_gradient = energies(generated, interior)
    generated_context_lap, generated_context_gradient = energies(
        generated, context
    )
    epsilon = 1e-12
    if min(
        source_edit_lap,
        source_edit_gradient,
        source_context_lap,
        source_context_gradient,
        generated_context_lap,
        generated_context_gradient,
    ) <= epsilon:
        return {
            "local_detail_evaluator_applicable": 0.0,
            "local_detail_edit_pixels": float(np.count_nonzero(interior)),
            "local_detail_context_pixels": float(np.count_nonzero(context)),
            "local_detail_energy_floor": epsilon,
        }
    laplacian_retention = (
        generated_edit_lap / max(generated_context_lap, epsilon)
    ) / (source_edit_lap / max(source_context_lap, epsilon))
    gradient_retention = (
        generated_edit_gradient / max(generated_context_gradient, epsilon)
    ) / (source_edit_gradient / max(source_context_gradient, epsilon))
    return {
        "local_detail_evaluator_applicable": 1.0,
        "local_detail_edit_pixels": float(np.count_nonzero(interior)),
        "local_detail_context_pixels": float(np.count_nonzero(context)),
        "local_detail_laplacian_relative_retention": float(laplacian_retention),
        "local_detail_gradient_relative_retention": float(gradient_retention),
        "local_detail_minimum_relative_retention": float(
            min(laplacian_retention, gradient_retention)
        ),
    }


def _macro_iou(target: np.ndarray, predicted: np.ndarray, *, region: np.ndarray) -> float:
    if not np.any(region):
        return 1.0
    # Average only over labels supported by the target region. Prediction-only
    # labels already reduce the IoU of the intended target labels through false
    # negatives; adding a separate zero-IoU class would double-penalize even a
    # single stray pixel and make patch-level scores discontinuous.
    labels = np.unique(target[region])
    scores = []
    for label in labels:
        target_label = (target == label) & region
        predicted_label = (predicted == label) & region
        union = np.count_nonzero(target_label | predicted_label)
        if union:
            scores.append(np.count_nonzero(target_label & predicted_label) / union)
    return float(np.mean(scores)) if scores else 1.0


def _prevalence_weighted_iou(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    region: np.ndarray,
) -> float:
    """Measure source-relative label stability without rare-class domination."""

    if not np.any(region):
        return 1.0
    region_pixels = int(np.count_nonzero(region))
    score = 0.0
    for label in np.unique(reference[region]):
        reference_label = (reference == label) & region
        predicted_label = (predicted == label) & region
        union = np.count_nonzero(reference_label | predicted_label)
        prevalence = np.count_nonzero(reference_label) / region_pixels
        if union:
            score += prevalence * (
                np.count_nonzero(reference_label & predicted_label) / union
            )
    return float(score)


def _semantic_evaluation_region(
    change: np.ndarray,
    *,
    thresholds: FidelityThresholds,
) -> tuple[np.ndarray, dict[str, float]]:
    from scipy import ndimage

    selected = np.asarray(change, dtype=bool)
    changed_pixels = int(np.count_nonzero(selected))
    change_ratio = float(changed_pixels / selected.size)
    small_region = bool(
        changed_pixels > 0
        and change_ratio <= thresholds.semantic_small_region_ratio_max
    )
    core = (
        ndimage.binary_erosion(
            selected,
            iterations=thresholds.semantic_boundary_tolerance_pixels,
            border_value=0,
        )
        if small_region and thresholds.semantic_boundary_tolerance_pixels > 0
        else np.array(selected, copy=True)
    )
    core_pixels = int(np.count_nonzero(core))
    core_fraction = float(core_pixels / changed_pixels) if changed_pixels else 1.0
    scale_applicable = bool(
        not small_region
        or (
            core_pixels >= thresholds.semantic_core_min_pixels
            and core_fraction >= thresholds.semantic_core_min_fraction
        )
    )
    evaluation_region = core if small_region and scale_applicable else selected
    return evaluation_region, {
        "semantic_changed_pixels": float(changed_pixels),
        "semantic_change_ratio": change_ratio,
        "semantic_small_region": float(small_region),
        "semantic_boundary_tolerance_pixels": float(
            thresholds.semantic_boundary_tolerance_pixels
        ),
        "semantic_core_pixels": float(core_pixels),
        "semantic_core_fraction": core_fraction,
        "semantic_scale_evaluator_applicable": float(scale_applicable),
    }


def _nuclei_occupied_area_relative_error(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    region: np.ndarray,
) -> float:
    labels = sorted(int(value) for value in np.unique(target[region]) if int(value) != 0)
    if not labels:
        return 0.0 if not np.any(predicted[region] != 0) else 1.0
    errors = []
    for label in labels:
        target_count = int(np.count_nonzero((target == label) & region))
        predicted_count = int(np.count_nonzero((predicted == label) & region))
        errors.append(abs(predicted_count - target_count) / max(1, target_count))
    return float(np.mean(errors))


def semantic_mask_instance_counts(
    mask: np.ndarray,
    *,
    region: np.ndarray,
) -> dict[int, int]:
    """Count typed instances whose full-component centroid lies in ``region``."""

    from scipy import ndimage

    semantic = np.asarray(mask)
    selected_region = np.asarray(region, dtype=bool)
    if semantic.shape != selected_region.shape:
        raise ValueError("Nuclei mask and region must have identical shapes.")
    counts: dict[int, int] = {}
    structure = np.ones((3, 3), dtype=np.uint8)
    for value in np.unique(semantic):
        label_id = int(value)
        if label_id == 0:
            continue
        components, component_count = ndimage.label(
            semantic == label_id,
            structure=structure,
        )
        accepted = 0
        for component_id in range(1, component_count + 1):
            ys, xs = np.nonzero(components == component_id)
            if ys.size == 0:
                continue
            center_y = int(np.clip(np.rint(np.mean(ys)), 0, semantic.shape[0] - 1))
            center_x = int(np.clip(np.rint(np.mean(xs)), 0, semantic.shape[1] - 1))
            if selected_region[center_y, center_x]:
                accepted += 1
        if accepted:
            counts[label_id] = accepted
    return counts


def _normalize_instance_counts(counts: Mapping[int, int]) -> dict[int, int]:
    normalized: dict[int, int] = {}
    for label, count in counts.items():
        label_id = int(label)
        count_value = int(count)
        if label_id == 0 or count_value < 0:
            if count_value < 0:
                raise ValueError("Nuclei instance counts cannot be negative.")
            continue
        normalized[label_id] = count_value
    return normalized


def _nuclei_detection_count_relative_error(
    target_counts: Mapping[int, int],
    predicted_counts: Mapping[int, int],
) -> float:
    target_total = int(sum(target_counts.values()))
    predicted_total = int(sum(predicted_counts.values()))
    if target_total == 0:
        return 0.0 if predicted_total == 0 else 1.0
    return float(abs(predicted_total - target_total) / target_total)


def _nuclei_type_composition_tv_error(
    target_counts: Mapping[int, int],
    predicted_counts: Mapping[int, int],
) -> float:
    target_total = int(sum(target_counts.values()))
    predicted_total = int(sum(predicted_counts.values()))
    if target_total == 0 or predicted_total == 0:
        return 0.0 if target_total == predicted_total else 1.0
    labels = set(target_counts) | set(predicted_counts)
    return float(
        0.5
        * sum(
            abs(
                target_counts.get(label, 0) / target_total
                - predicted_counts.get(label, 0) / predicted_total
            )
            for label in labels
        )
    )


def _route_metadata(route: AgenticRoutingDecision) -> dict[str, Any]:
    payload = asdict(route)
    payload["features"] = asdict(route.features)
    return payload


def _attempt_metadata(attempt: AgenticAttempt) -> dict[str, Any]:
    return {
        "attempt_index": attempt.attempt_index,
        "requested_mode": attempt.requested_mode,
        "decision_reason": attempt.decision_reason,
        "error": attempt.error,
        "artifact": (
            None
            if attempt.artifact is None
            else {
                "mode": attempt.artifact.mode,
                "image_path": str(attempt.artifact.image_path),
                "metadata": dict(attempt.artifact.metadata),
            }
        ),
        "verification": (
            None
            if attempt.verification is None
            else {
                "schema_version": attempt.verification.schema_version,
                "passed": attempt.verification.passed,
                "score": attempt.verification.score,
                "quality_score": attempt.verification.quality_score,
                "evidence_coverage": attempt.verification.evidence_coverage,
                "component_scores": dict(
                    attempt.verification.component_scores
                ),
                "applicability": dict(attempt.verification.applicability),
                "scientific_status": attempt.verification.scientific_status,
                "reason_codes": list(attempt.verification.reason_codes),
                "metrics": dict(attempt.verification.metrics),
                "failed_checks": list(attempt.verification.failed_checks),
            }
        ),
    }


def _attempt_selection_key(attempt: AgenticAttempt) -> tuple[float, ...]:
    verification = attempt.verification
    if verification is None:
        return (0.0, 0.0, 0.0, 0.0, -float(attempt.attempt_index))
    components = verification.component_scores
    return (
        float(verification.passed),
        float(
            verification.quality_score
            if verification.quality_score is not None
            else verification.score
        ),
        float(components.get("semantic", 0.0)),
        float(components.get("preservation", 0.0)),
        -float(attempt.attempt_index),
    )


def _save_result(result: AgenticWorkflowResult) -> None:
    path = result.output_dir / "agentic_workflow.json"
    path.write_text(
        json.dumps(
            result.to_metadata(),
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def _save_running_state(
    route: AgenticRoutingDecision,
    attempts: list[AgenticAttempt],
    output_dir: Path,
) -> None:
    payload = {
        "status": "running",
        "route": _route_metadata(route),
        "attempts": [_attempt_metadata(item) for item in attempts],
        "selected_attempt": None,
        "output_dir": str(output_dir),
    }
    (output_dir / "agentic_workflow.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _safe_mode(mode: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in mode).strip("_")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
