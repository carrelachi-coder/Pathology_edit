"""Generic mask editing strategies that must work on all six datasets."""

from phase3_mask_edit.generic.boundary import (
    apply_boundary_pushing_remodel,
)
from phase3_mask_edit.generic.executor import (
    EditExecutionResult,
    execute_edit,
    register_primitive,
)
from phase3_mask_edit.generic.necrosis import (
    apply_necrosis_appearance,
)
from phase3_mask_edit.generic.immune import (
    apply_stromal_immune_infiltration,
)
from phase3_mask_edit.generic.desmoplasia import (
    apply_stromal_desmoplasia,
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
    "apply_boundary_pushing_remodel",
    "apply_necrosis_appearance",
    "apply_stromal_desmoplasia",
    "apply_stromal_immune_infiltration",
    "apply_tumor_burden_decrease",
    "apply_tumor_burden_increase",
    "execute_edit",
    "register_primitive",
]
