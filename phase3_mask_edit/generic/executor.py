"""Unified primitive executor: applicability gate + mask transform + validation.

Each Phase 3 primitive must follow this execution flow:
  1. Applicability gate: assess whether the intent can run on the current
     mask/schema.  Rejected intents never proceed to mask transform.
  2. Mask transform: call the primitive-specific algorithm.
  3. Post-execution validation: check the output against recipe rules.

The executor enforces this flow so that future primitives only need to
implement the mask-transform step; applicability and validation are
handled centrally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from phase3_mask_edit.core.applicability import (
    EditApplicabilityDecision,
    assess_edit_applicability,
)
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.validation import ValidationResult, validate_edit_result
from phase3_mask_edit.generic.tumor_burden import (
    PrimitiveEditResult,
    PrimitiveExecutionError,
    apply_tumor_burden_decrease,
    apply_tumor_burden_increase,
)
from phase3_mask_edit.generic.boundary import (
    apply_boundary_pushing_remodel,
)
from phase3_mask_edit.generic.necrosis import (
    apply_necrosis_appearance,
    apply_necrosis_resolution,
)
from phase3_mask_edit.generic.immune import (
    apply_stromal_immune_infiltration,
)
from phase3_mask_edit.generic.desmoplasia import (
    apply_stromal_desmoplasia,
)
from phase3_mask_edit.specialized.fine_transition import (
    apply_fine_label_transition,
)


# ── EditExecutionResult ────────────────────────────────────────────

@dataclass(frozen=True)
class EditExecutionResult:
    """Complete result from the unified primitive executor."""

    applicability: EditApplicabilityDecision
    edit_result: PrimitiveEditResult | None
    validation: ValidationResult | None
    status: str  # "rejected", "degraded_executed", "executed_validated", etc.


# ── primitive registry ─────────────────────────────────────────────

_PRIMITIVE_REGISTRY: dict[str, Callable] = {}


def register_primitive(name: str, fn: Callable) -> None:
    """Register a mask-transform function under a primitive name."""
    _PRIMITIVE_REGISTRY[name] = fn


def _auto_register() -> None:
    """Register all known V1 primitives on first call."""
    if _PRIMITIVE_REGISTRY:
        return
    register_primitive("tumor_burden_increase", apply_tumor_burden_increase)
    register_primitive("tumor_burden_decrease", apply_tumor_burden_decrease)
    register_primitive("boundary_pushing_remodel", apply_boundary_pushing_remodel)
    register_primitive("necrosis_appearance", apply_necrosis_appearance)
    register_primitive("necrosis_resolution", apply_necrosis_resolution)
    register_primitive("fine_label_transition", apply_fine_label_transition)
    register_primitive(
        "stromal_immune_infiltration",
        apply_stromal_immune_infiltration,
    )
    register_primitive("stromal_desmoplasia", apply_stromal_desmoplasia)


_EXECUTION_STRATEGY_REGISTRY: dict[str, Callable] = {
    "id_transition": apply_fine_label_transition,
}


# ── main executor ──────────────────────────────────────────────────

def execute_edit(
    old_mask: np.ndarray,
    intent: EditIntent,
    recipe: Mapping[str, Any],
    schema: MaskProfileSchema,
    context: MaskEditContext,
) -> EditExecutionResult:
    """Execute a Phase 3 primitive intent through the full pipeline.

    Steps:
      1. Applicability gate → if rejected, return with no edit_result.
      2. Find primitive config from recipe.
      3. Call the registered mask-transform function.
      4. Post-execution validation on the output.
    """

    _auto_register()

    # ── step 1: applicability ────────────────────────────────────
    applicability = assess_edit_applicability(intent, recipe, schema, context)
    if applicability.status == "rejected":
        return EditExecutionResult(
            applicability=applicability,
            edit_result=None,
            validation=None,
            status="rejected",
        )

    # ── step 2: primitive config ─────────────────────────────────
    primitive_config = _find_primitive_config(recipe, intent.primitive)
    if primitive_config is None:
        return EditExecutionResult(
            applicability=applicability,
            edit_result=None,
            validation=None,
            status="rejected",
        )

    # ── step 3: mask transform ───────────────────────────────────
    fn = _resolve_primitive_function(primitive_config, intent.primitive)
    if fn is None:
        raise PrimitiveExecutionError(
            f"No registered primitive function for {intent.primitive}."
        )

    try:
        edit_result = fn(old_mask, schema, context, primitive_config, intent)
    except PrimitiveExecutionError:
        return EditExecutionResult(
            applicability=applicability,
            edit_result=None,
            validation=None,
            status="execution_failed",
        )

    # Merge applicability warnings into edit_result warnings.
    merged_warnings = tuple(dict.fromkeys(
        applicability.warnings + edit_result.warnings
    ))
    edit_result = PrimitiveEditResult(
        target_mask=edit_result.target_mask,
        change_region=edit_result.change_region,
        changed_area_fraction=edit_result.changed_area_fraction,
        selected_pixels=edit_result.selected_pixels,
        warnings=merged_warnings,
        ops_log=edit_result.ops_log,
    )

    # ── step 4: post-execution validation ────────────────────────
    validation = validate_edit_result(
        src_mask=old_mask,
        target_mask=edit_result.target_mask,
        change_region=edit_result.change_region,
        schema=schema,
        primitive_config=primitive_config,
        changed_area_fraction=edit_result.changed_area_fraction,
    )

    status = "executed_validated" if validation.passed else "executed_with_validation_warnings"
    if applicability.status == "degraded":
        status = "degraded_executed" if validation.passed else "degraded_executed_with_validation_warnings"

    return EditExecutionResult(
        applicability=applicability,
        edit_result=edit_result,
        validation=validation,
        status=status,
    )


# ── helpers ────────────────────────────────────────────────────────

def _find_primitive_config(
    recipe: Mapping[str, Any], primitive_name: str
) -> Mapping[str, Any] | None:
    primitives = recipe.get("primitives", [])
    for primitive in primitives:
        if isinstance(primitive, Mapping) and primitive.get("name") == primitive_name:
            return primitive
    return None


def _resolve_primitive_function(
    primitive_config: Mapping[str, Any], primitive_name: str
) -> Callable | None:
    strategy = primitive_config.get("execution_strategy")
    if isinstance(strategy, str):
        fn = _EXECUTION_STRATEGY_REGISTRY.get(strategy)
        if fn is not None:
            return fn
    return _PRIMITIVE_REGISTRY.get(primitive_name)
