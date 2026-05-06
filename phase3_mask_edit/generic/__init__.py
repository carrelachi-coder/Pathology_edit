"""Generic mask editing strategies that must work on all six datasets."""

from phase3_mask_edit.generic.boundary import (
    apply_boundary_infiltration,
    apply_boundary_pushing_remodel,
)
from phase3_mask_edit.generic.executor import (
    EditExecutionResult,
    execute_edit,
    register_primitive,
)
from phase3_mask_edit.generic.tumor_burden import (
    PrimitiveEditResult,
    PrimitiveExecutionError,
    apply_tumor_burden_decrease,
    apply_tumor_burden_increase,
)

__all__ = [
    "EditExecutionResult",
    "PrimitiveEditResult",
    "PrimitiveExecutionError",
    "apply_boundary_infiltration",
    "apply_boundary_pushing_remodel",
    "apply_tumor_burden_decrease",
    "apply_tumor_burden_increase",
    "execute_edit",
    "register_primitive",
]
