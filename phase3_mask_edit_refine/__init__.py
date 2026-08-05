"""Auditable, dual-axis pathology mask editing research pipeline.

This package is deliberately independent from :mod:`phase3_mask_edit`.  The
legacy package is only reached through explicit deterministic tool adapters.
"""

from .models import (
    AreaBudget,
    CaseContext,
    EditPlan,
    GateReport,
    ResolvedAreaContract,
    WorkflowResult,
)

__all__ = [
    "AreaBudget",
    "CaseContext",
    "EditPlan",
    "GateReport",
    "ResolvedAreaContract",
    "WorkflowResult",
]
