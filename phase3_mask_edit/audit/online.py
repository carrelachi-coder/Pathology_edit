"""Product-level online semantic auditor used by the editing agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from .metrics import (
    ConfidencePolicy,
    EvaluatorCleanPolicy,
    source_evaluator_quality,
    source_relative_tissue_metrics,
)
from .postprocess import (
    ConservativeP1Policy,
    P1Result,
    apply_conservative_p1,
)


PostprocessMode = Literal["off", "shadow", "enforce"]


@dataclass(frozen=True)
class SemanticPrediction:
    mask: np.ndarray
    probabilities: np.ndarray
    entropy: np.ndarray | None = None

    def validate(self) -> None:
        mask = np.asarray(self.mask)
        probabilities = np.asarray(self.probabilities)
        if mask.ndim != 2:
            raise ValueError("semantic prediction mask must be rank 2")
        if probabilities.ndim != 3 or probabilities.shape[1:] != mask.shape:
            raise ValueError("semantic probabilities must be CHW and match the mask")
        if self.entropy is not None and np.asarray(self.entropy).shape != mask.shape:
            raise ValueError("semantic entropy must match the mask")


@dataclass(frozen=True)
class OnlineAuditPolicy:
    policy_id: str = "online-semantic-audit-v1"
    postprocess_mode: PostprocessMode = "shadow"
    boundary_radius_pixels: int = 4
    confidence: ConfidencePolicy = field(default_factory=ConfidencePolicy)
    evaluator_clean: EvaluatorCleanPolicy = field(
        default_factory=EvaluatorCleanPolicy
    )
    p1: ConservativeP1Policy = field(default_factory=ConservativeP1Policy)

    def validate(self) -> None:
        if self.postprocess_mode not in {"off", "shadow", "enforce"}:
            raise ValueError(f"unsupported postprocess_mode: {self.postprocess_mode}")
        if self.boundary_radius_pixels < 1:
            raise ValueError("boundary_radius_pixels must be positive")
        self.p1.validate()


@dataclass(frozen=True)
class OnlineAuditResult:
    policy: OnlineAuditPolicy
    source_quality: dict[str, Any]
    raw_metrics: dict[str, Any]
    p1_metrics: dict[str, Any] | None
    fine_metrics: dict[str, Any] | None
    p1_result: P1Result | None
    raw_mask: np.ndarray
    decision_input: str

    @property
    def decision_mask(self) -> np.ndarray:
        if self.decision_input == "p1_audited":
            assert self.p1_result is not None
            return self.p1_result.audited_mask
        if self.p1_result is not None:
            return self.p1_result.raw_mask
        return self.raw_mask

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "audit_scope": "online_agent_product",
            "benchmark_independent": True,
            "policy": {
                **asdict(self.policy),
                "confidence": asdict(self.policy.confidence),
                "evaluator_clean": asdict(self.policy.evaluator_clean),
                "p1": asdict(self.policy.p1),
            },
            "decision_input": self.decision_input,
            "source_quality": self.source_quality,
            "raw_metrics": self.raw_metrics,
            "p1_metrics": self.p1_metrics,
            "fine_metrics": self.fine_metrics,
            "p1": (
                None if self.p1_result is None else self.p1_result.to_metadata()
            ),
        }


class OnlineSemanticAuditor:
    """Compute source-relative raw and optional P1-shadow audit evidence."""

    def __init__(self, policy: OnlineAuditPolicy | None = None) -> None:
        self.policy = policy or OnlineAuditPolicy()
        self.policy.validate()

    def audit(
        self,
        *,
        source_mask: np.ndarray,
        target_mask: np.ndarray,
        source_prediction: SemanticPrediction,
        generated_prediction: SemanticPrediction,
        class_ids: tuple[int, ...] = tuple(range(8)),
        semantic_change_region: np.ndarray | None = None,
        source_fine_mask: np.ndarray | None = None,
        target_fine_mask: np.ndarray | None = None,
        source_fine_prediction: SemanticPrediction | None = None,
        generated_fine_prediction: SemanticPrediction | None = None,
        fine_class_ids: tuple[int, ...] | None = None,
        ignore_index: int = 255,
    ) -> OnlineAuditResult:
        source_prediction.validate()
        generated_prediction.validate()
        source_quality = source_evaluator_quality(
            source_mask=source_mask,
            source_prediction=source_prediction.mask,
            source_probabilities=source_prediction.probabilities,
            class_ids=class_ids,
            ignore_index=ignore_index,
            policy=self.policy.evaluator_clean,
        )
        raw_metrics = self._metrics(
            source_mask=source_mask,
            target_mask=target_mask,
            source_prediction=source_prediction,
            generated_mask=generated_prediction.mask,
            generated_prediction=generated_prediction,
            class_ids=class_ids,
            ignore_index=ignore_index,
            semantic_change_region=semantic_change_region,
        )
        p1_result = None
        p1_metrics = None
        if self.policy.postprocess_mode != "off":
            p1_result = apply_conservative_p1(
                predicted_mask=generated_prediction.mask,
                probabilities=generated_prediction.probabilities,
                entropy=generated_prediction.entropy,
                source_mask=source_mask,
                target_mask=target_mask,
                source_prediction=source_prediction.mask,
                policy=self.policy.p1,
                ignore_index=ignore_index,
                semantic_change_region=semantic_change_region,
            )
            p1_metrics = self._metrics(
                source_mask=source_mask,
                target_mask=target_mask,
                source_prediction=source_prediction,
                generated_mask=p1_result.audited_mask,
                generated_prediction=generated_prediction,
                class_ids=class_ids,
                ignore_index=ignore_index,
                semantic_change_region=semantic_change_region,
            )
        fine_inputs = (
            source_fine_mask,
            target_fine_mask,
            source_fine_prediction,
            generated_fine_prediction,
            fine_class_ids,
        )
        fine_metrics = None
        if any(value is not None for value in fine_inputs):
            if not all(value is not None for value in fine_inputs):
                raise ValueError("fine audit inputs must be supplied together")
            source_fine_prediction.validate()
            generated_fine_prediction.validate()
            fine_metrics = source_relative_tissue_metrics(
                source_mask=source_fine_mask,
                target_mask=target_fine_mask,
                source_prediction=source_fine_prediction.mask,
                generated_prediction=generated_fine_prediction.mask,
                source_probabilities=source_fine_prediction.probabilities,
                generated_probabilities=generated_fine_prediction.probabilities,
                class_ids=fine_class_ids,
                ignore_index=ignore_index,
                boundary_radius=self.policy.boundary_radius_pixels,
                source_entropy=source_fine_prediction.entropy,
                generated_entropy=generated_fine_prediction.entropy,
                confidence_policy=self.policy.confidence,
                semantic_change_region=semantic_change_region,
            )
        decision_input = (
            "p1_audited"
            if self.policy.postprocess_mode == "enforce"
            else "raw"
        )
        return OnlineAuditResult(
            policy=self.policy,
            source_quality=source_quality,
            raw_metrics=raw_metrics,
            p1_metrics=p1_metrics,
            fine_metrics=fine_metrics,
            p1_result=p1_result,
            raw_mask=np.array(generated_prediction.mask, copy=True),
            decision_input=decision_input,
        )

    def _metrics(
        self,
        *,
        source_mask: np.ndarray,
        target_mask: np.ndarray,
        source_prediction: SemanticPrediction,
        generated_mask: np.ndarray,
        generated_prediction: SemanticPrediction,
        class_ids: tuple[int, ...],
        ignore_index: int,
        semantic_change_region: np.ndarray | None,
    ) -> dict[str, Any]:
        return source_relative_tissue_metrics(
            source_mask=source_mask,
            target_mask=target_mask,
            source_prediction=source_prediction.mask,
            generated_prediction=generated_mask,
            source_probabilities=source_prediction.probabilities,
            generated_probabilities=generated_prediction.probabilities,
            class_ids=class_ids,
            ignore_index=ignore_index,
            boundary_radius=self.policy.boundary_radius_pixels,
            source_entropy=source_prediction.entropy,
            generated_entropy=generated_prediction.entropy,
            confidence_policy=self.policy.confidence,
            semantic_change_region=semantic_change_region,
        )
