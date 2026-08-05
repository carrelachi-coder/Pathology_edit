"""Independent tissue--cell joint pathology edit pipeline.

The package deliberately does not replace or mutate ``phase3_mask_edit_refine``.
It consumes the mask editor and frozen nuclei tooling through explicit adapters
and approves only atomic tissue+nuclei conditions.
"""

from .models import (
    CellEditPlan,
    CouplingPlan,
    JointAreaBudget,
    JointCandidate,
    JointCaseContext,
    JointCondition,
    JointContractError,
    JointCriticResult,
    JointEditPlan,
    JointGateReport,
    JointWorkflowResult,
)

__all__ = [
    "CellEditPlan",
    "CouplingPlan",
    "JointAreaBudget",
    "JointCandidate",
    "JointCaseContext",
    "JointCondition",
    "JointContractError",
    "JointCriticResult",
    "JointEditPlan",
    "JointGateReport",
    "JointWorkflowResult",
]
