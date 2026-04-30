"""Generic mask editing strategies that must work on all six datasets."""

from phase3_mask_edit.generic.tumor_burden import (
    PrimitiveEditResult,
    PrimitiveExecutionError,
    apply_tumor_burden_increase,
)

__all__ = [
    "PrimitiveEditResult",
    "PrimitiveExecutionError",
    "apply_tumor_burden_increase",
]
