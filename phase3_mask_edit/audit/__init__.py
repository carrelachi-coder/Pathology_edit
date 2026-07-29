"""Online semantic self-audit primitives for pathology editing."""

from .metrics import (
    ConfidencePolicy,
    EvaluatorCleanPolicy,
    build_edit_regions,
    confidence_maps,
    normalized_entropy,
    source_evaluator_quality,
    source_relative_tissue_metrics,
)
from .labels import (
    dataset_native_metric_class_ids,
    profile_supports_fine,
    to_coarse_mask,
)
from .online import (
    OnlineAuditPolicy,
    OnlineAuditResult,
    OnlineSemanticAuditor,
    SemanticPrediction,
)
from .postprocess import (
    ConservativeP1Policy,
    P1Operation,
    P1Result,
    apply_conservative_p1,
)
from .visualization import build_online_audit_deck

__all__ = [
    "ConfidencePolicy",
    "ConservativeP1Policy",
    "EvaluatorCleanPolicy",
    "OnlineAuditPolicy",
    "OnlineAuditResult",
    "OnlineSemanticAuditor",
    "P1Operation",
    "P1Result",
    "SemanticPrediction",
    "apply_conservative_p1",
    "build_edit_regions",
    "build_online_audit_deck",
    "confidence_maps",
    "dataset_native_metric_class_ids",
    "normalized_entropy",
    "profile_supports_fine",
    "source_evaluator_quality",
    "source_relative_tissue_metrics",
    "to_coarse_mask",
]
