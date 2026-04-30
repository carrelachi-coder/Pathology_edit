"""Applicability decisions for Phase 3 edit intents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import (
    EditIntent,
    IntentValidationError,
    validate_intent_against_recipe,
)
from phase3_mask_edit.core.labels import MaskProfileSchema


ApplicabilityStatus = Literal["executable", "degraded", "rejected"]


@dataclass(frozen=True)
class EditApplicabilityDecision:
    """Decision explaining whether an intent can run on the current mask."""

    status: ApplicabilityStatus
    primitive: str
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    fallback_actions: tuple[str, ...] = ()


def assess_edit_applicability(
    intent: EditIntent,
    recipe: Mapping[str, Any],
    schema: MaskProfileSchema,
    context: MaskEditContext,
) -> EditApplicabilityDecision:
    """Assess schema and current-mask gates before any mask transform."""

    reasons: list[str] = []
    warnings: list[str] = list(context.risk_flags)
    fallback_actions: list[str] = []

    if intent.reference_profile and intent.reference_profile != schema.reference_profile:
        reasons.append(
            "reference_profile_mismatch:"
            f"intent={intent.reference_profile},schema={schema.reference_profile}"
        )

    try:
        validate_intent_against_recipe(intent, recipe)
    except IntentValidationError as exc:
        reasons.append(f"intent_invalid:{exc}")

    primitive = _primitive_by_name(recipe).get(intent.primitive)
    if primitive is None:
        return _decision(intent.primitive, reasons, warnings, fallback_actions)

    source_label = _operation_label(primitive, "source")
    target_label = _operation_label(primitive, "target")
    required_labels = tuple(primitive.get("required_tissue_labels", ()))
    optional_labels = tuple(primitive.get("optional_tissue_labels", ()))

    for label in required_labels:
        if label not in schema.readable_labels or label not in schema.writable_labels:
            reasons.append(f"required_label_not_readable_or_writable:{label}")

    if intent.target_label and intent.target_label not in schema.writable_labels:
        reasons.append(f"target_label_not_writable:{intent.target_label}")

    labels_that_must_exist = _labels_required_in_current_mask(
        required_labels=required_labels,
        source_label=source_label,
        target_label=target_label,
    )
    for label in labels_that_must_exist:
        if label in schema.readable_labels and label not in context.present_labels:
            reasons.append(f"required_context_label_absent_in_mask:{label}")

    for label in optional_labels:
        if label in schema.readable_labels and label not in context.present_labels:
            warnings.append(f"optional_label_absent_in_mask:{label}")
            if _optional_label_has_fallback(primitive, label):
                fallback_actions.append(f"use_fallback_without:{label}")

    return _decision(intent.primitive, reasons, warnings, fallback_actions)


def _decision(
    primitive: str,
    reasons: list[str],
    warnings: list[str],
    fallback_actions: list[str],
) -> EditApplicabilityDecision:
    if reasons:
        status: ApplicabilityStatus = "rejected"
    elif warnings or fallback_actions:
        status = "degraded"
    else:
        status = "executable"

    return EditApplicabilityDecision(
        status=status,
        primitive=primitive,
        reasons=tuple(dict.fromkeys(reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
        fallback_actions=tuple(dict.fromkeys(fallback_actions)),
    )


def _primitive_by_name(recipe: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        primitive["name"]: primitive
        for primitive in recipe.get("primitives", [])
        if isinstance(primitive, Mapping) and isinstance(primitive.get("name"), str)
    }


def _operation_label(primitive: Mapping[str, Any], key: str) -> str | None:
    mask_operation = primitive.get("mask_operation", {})
    value = mask_operation.get(key) if isinstance(mask_operation, Mapping) else None
    return value if isinstance(value, str) else None


def _labels_required_in_current_mask(
    *, required_labels: tuple[str, ...], source_label: str | None, target_label: str | None
) -> tuple[str, ...]:
    labels: list[str] = []
    for label in required_labels:
        # Required labels that are only the written target do not need to be
        # already present in the current mask. They must only be writable.
        if label == target_label and label != source_label:
            continue
        labels.append(label)
    if source_label and source_label not in labels:
        labels.append(source_label)
    return tuple(labels)


def _optional_label_has_fallback(primitive: Mapping[str, Any], label: str) -> bool:
    spatial_pattern = primitive.get("spatial_pattern", {})
    if not isinstance(spatial_pattern, Mapping):
        return False

    if label == "Blood vessel" and "candidate_weights" in spatial_pattern:
        return True
    if label == "Necrosis" and "candidate_weights_no_existing_necrosis" in spatial_pattern:
        return True
    return False
