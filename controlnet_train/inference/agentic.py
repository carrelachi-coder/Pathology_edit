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
    nuclei_density_relative_error_max: float = 0.35


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    score: float
    metrics: Mapping[str, float]
    failed_checks: tuple[str, ...] = ()


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
) -> AgenticWorkflowResult:
    """Route, generate, verify, and perform bounded failure-aware recovery."""

    config = config or AgenticWorkflowConfig()
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    route = route_agentic_edit_request(
        reference_tissue_mask,
        target_tissue_mask,
        config=config.routing,
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
                status="validated",
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
    selected = max(verified, key=lambda item: item.verification.score) if verified else None
    result = AgenticWorkflowResult(
        status="needs_review",
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
    if "off_target_drift" in failed and "inpaint" in untried:
        return RecoveryDecision(
            "inpaint",
            "preservation_recovery",
            "off-target drift exceeded the limit, so use the local-preserving backend",
        )
    structure_failures = {
        "changed_region_accuracy",
        "changed_region_macro_iou",
        "nuclei_density_relative_error",
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
    change_region: np.ndarray,
    target_nuclei_mask: np.ndarray | None = None,
    predicted_nuclei_mask: np.ndarray | None = None,
    thresholds: FidelityThresholds | None = None,
) -> VerificationResult:
    """Deterministically score re-segmented output against target masks."""

    thresholds = thresholds or FidelityThresholds()
    reference = np.asarray(reference_tissue_mask)
    target = np.asarray(target_tissue_mask)
    predicted = np.asarray(predicted_tissue_mask)
    change = np.asarray(change_region, dtype=bool)
    if not (reference.shape == target.shape == predicted.shape == change.shape):
        raise ValueError("Tissue masks and change_region must have identical shapes.")
    changed_count = int(np.count_nonzero(change))
    changed_accuracy = (
        float(np.mean(predicted[change] == target[change])) if changed_count else 1.0
    )
    outside = ~change
    off_target_drift = (
        float(np.mean(predicted[outside] != reference[outside]))
        if np.any(outside)
        else 0.0
    )
    macro_iou = _macro_iou(target, predicted, region=change)
    metrics: dict[str, float] = {
        "changed_region_accuracy": changed_accuracy,
        "changed_region_macro_iou": macro_iou,
        "off_target_drift": off_target_drift,
    }
    failed = []
    if changed_accuracy < thresholds.changed_region_accuracy_min:
        failed.append("changed_region_accuracy")
    if macro_iou < thresholds.changed_region_macro_iou_min:
        failed.append("changed_region_macro_iou")
    if off_target_drift > thresholds.off_target_drift_max:
        failed.append("off_target_drift")

    nuclei_error = None
    if target_nuclei_mask is not None and predicted_nuclei_mask is not None:
        target_nuclei = np.asarray(target_nuclei_mask)
        predicted_nuclei = np.asarray(predicted_nuclei_mask)
        if target_nuclei.shape != change.shape or predicted_nuclei.shape != change.shape:
            raise ValueError("Nuclei masks must match tissue mask shape.")
        nuclei_error = _nuclei_density_relative_error(
            target_nuclei, predicted_nuclei, region=change
        )
        metrics["nuclei_density_relative_error"] = nuclei_error
        if nuclei_error > thresholds.nuclei_density_relative_error_max:
            failed.append("nuclei_density_relative_error")

    score = (
        0.45 * changed_accuracy
        + 0.35 * macro_iou
        + 0.20 * (1.0 - min(1.0, off_target_drift))
    )
    if nuclei_error is not None:
        score = 0.85 * score + 0.15 * (1.0 - min(1.0, nuclei_error))
    return VerificationResult(
        passed=not failed,
        score=float(score),
        metrics=metrics,
        failed_checks=tuple(failed),
    )


def _macro_iou(target: np.ndarray, predicted: np.ndarray, *, region: np.ndarray) -> float:
    if not np.any(region):
        return 1.0
    labels = np.unique(target[region])
    scores = []
    for label in labels:
        target_label = (target == label) & region
        predicted_label = (predicted == label) & region
        union = np.count_nonzero(target_label | predicted_label)
        if union:
            scores.append(np.count_nonzero(target_label & predicted_label) / union)
    return float(np.mean(scores)) if scores else 1.0


def _nuclei_density_relative_error(
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
                "passed": attempt.verification.passed,
                "score": attempt.verification.score,
                "metrics": dict(attempt.verification.metrics),
                "failed_checks": list(attempt.verification.failed_checks),
            }
        ),
    }


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
