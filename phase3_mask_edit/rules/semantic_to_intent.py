"""Map validated semantic diffs to Phase 3 EditIntent objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from phase3_mask_edit.core.applicability import assess_edit_applicability
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.parser.semantic_diff import validate_semantic_diff


PlanItemStatus = Literal[
    "planned",
    "degraded_planned",
    "rejected_by_applicability",
    "unsupported",
]

INTENT_ORDER = {
    "tumor_burden_increase": 10,
    "tumor_burden_decrease": 20,
    "necrosis_appearance": 30,
    "stromal_immune_infiltration": 40,
}


@dataclass(frozen=True)
class PlanningWarning:
    """A semantic change that Phase 3 cannot execute yet."""

    field: str
    value: str
    reason: str

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntentPlanItem:
    """One planned or rejected Phase 3 intent."""

    primitive: str
    strength: str
    status: PlanItemStatus
    intent: EditIntent | None = None
    applicability: str | None = None
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    fallback_actions: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "primitive": self.primitive,
            "strength": self.strength,
            "status": self.status,
            "applicability": self.applicability,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "fallback_actions": list(self.fallback_actions),
        }
        if self.intent is not None:
            metadata["intent"] = self.intent.to_metadata()
        return metadata


@dataclass(frozen=True)
class IntentPlanningResult:
    """Full prompt-planning result for serialization."""

    semantic_diff: dict[str, Any]
    reference_profile: str
    items: tuple[IntentPlanItem, ...]
    unsupported_changes: tuple[PlanningWarning, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def intents(self) -> tuple[EditIntent, ...]:
        return tuple(item.intent for item in self.items if item.intent is not None)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "semantic_diff": self.semantic_diff,
            "reference_profile": self.reference_profile,
            "intents": [intent.to_metadata() for intent in self.intents],
            "items": [item.to_metadata() for item in self.items],
            "unsupported_changes": [
                warning.to_metadata() for warning in self.unsupported_changes
            ],
            "metadata": dict(self.metadata),
        }


def semantic_diff_to_intents(
    semantic_diff: Mapping[str, Any],
    *,
    reference_profile: str,
    old_prompt: str | None = None,
    new_prompt: str | None = None,
) -> list[EditIntent]:
    """Return executable/planned intents without applicability gating."""

    result = plan_edit_intents(
        semantic_diff,
        reference_profile=reference_profile,
        old_prompt=old_prompt,
        new_prompt=new_prompt,
    )
    return list(result.intents)


def plan_edit_intents(
    semantic_diff: Mapping[str, Any],
    *,
    reference_profile: str,
    old_mask: np.ndarray | None = None,
    recipe: Mapping[str, Any] | None = None,
    old_prompt: str | None = None,
    new_prompt: str | None = None,
) -> IntentPlanningResult:
    """Map a validated semantic diff into ordered Phase 3 intent plan items."""

    validated = validate_semantic_diff(semantic_diff)
    if not reference_profile:
        raise ValueError("reference_profile is required.")

    raw_items, unsupported = _raw_intent_specs(
        validated,
        reference_profile=reference_profile,
        old_prompt=old_prompt,
        new_prompt=new_prompt,
    )
    raw_items.sort(key=lambda item: INTENT_ORDER.get(item["primitive"], 999))

    if old_mask is None:
        items = tuple(
            IntentPlanItem(
                primitive=payload["primitive"],
                strength=payload["strength"],
                status="planned",
                intent=EditIntent.from_mapping(payload),
            )
            for payload in raw_items
        )
        return IntentPlanningResult(
            semantic_diff=validated,
            reference_profile=reference_profile,
            items=items,
            unsupported_changes=tuple(unsupported),
            metadata={
                "applicability_checked": False,
                "intent_order": _intent_order_names(items),
            },
        )

    if recipe is None:
        recipe = load_recipe("phase3_mask_edit/recipes/generic.yaml")

    schema = MaskProfileSchema.from_reference_profile(reference_profile)
    context = MaskEditContext.from_mask(old_mask, schema)
    gated_items: list[IntentPlanItem] = []
    for payload in raw_items:
        intent = EditIntent.from_mapping(payload)
        decision = assess_edit_applicability(intent, recipe, schema, context)
        if decision.status == "rejected":
            status: PlanItemStatus = "rejected_by_applicability"
            item_intent = None
        elif decision.status == "degraded":
            status = "degraded_planned"
            item_intent = intent
        else:
            status = "planned"
            item_intent = intent
        gated_items.append(
            IntentPlanItem(
                primitive=intent.primitive,
                strength=intent.strength,
                status=status,
                intent=item_intent,
                applicability=decision.status,
                reasons=decision.reasons,
                warnings=decision.warnings,
                fallback_actions=decision.fallback_actions,
            )
        )

    items = tuple(gated_items)
    return IntentPlanningResult(
        semantic_diff=validated,
        reference_profile=reference_profile,
        items=items,
        unsupported_changes=tuple(unsupported),
        metadata={
            "applicability_checked": True,
            "intent_order": _intent_order_names(items),
            "present_labels": sorted(context.present_labels),
            "risk_flags": list(context.risk_flags),
        },
    )


def _raw_intent_specs(
    semantic_diff: Mapping[str, Any],
    *,
    reference_profile: str,
    old_prompt: str | None,
    new_prompt: str | None,
) -> tuple[list[dict[str, Any]], list[PlanningWarning]]:
    prompt_diff = {"semantic_diff": semantic_diff}
    raw_items: list[dict[str, Any]] = []
    unsupported: list[PlanningWarning] = []

    tumor_change = semantic_diff["tumor_change"]
    tumor_growth = tumor_change["growth"]
    if tumor_growth == "increase":
        raw_items.append(
            _intent_payload(
                "tumor_burden_increase",
                _strength_from_degree(tumor_change["degree"]),
                reference_profile,
                old_prompt,
                new_prompt,
                prompt_diff,
            )
        )
    elif tumor_growth == "decrease":
        raw_items.append(
            _intent_payload(
                "tumor_burden_decrease",
                _strength_from_degree(tumor_change["degree"]),
                reference_profile,
                old_prompt,
                new_prompt,
                prompt_diff,
            )
        )

    if tumor_change["grade_change"] != "none" and tumor_growth == "none":
        unsupported.append(
            PlanningWarning(
                field="tumor_change.grade_change",
                value=tumor_change["grade_change"],
                reason="Phase3 grade-only/cell-only primitive is not implemented yet.",
            )
        )

    necrosis_change = semantic_diff["necrosis_change"]
    necrosis_action = necrosis_change["action"]
    if necrosis_action in {"add", "increase"}:
        raw_items.append(
            _intent_payload(
                "necrosis_appearance",
                _strength_from_necrosis_extent(necrosis_change["extent"]),
                reference_profile,
                old_prompt,
                new_prompt,
                prompt_diff,
            )
        )
    elif necrosis_action in {"decrease", "remove"}:
        unsupported.append(
            PlanningWarning(
                field="necrosis_change.action",
                value=necrosis_action,
                reason="Phase3 necrosis decrease/resolution primitive is not implemented yet.",
            )
        )

    lymphocyte_change = semantic_diff["lymphocyte_change"]
    infiltration = lymphocyte_change["infiltration"]
    if infiltration == "increase":
        raw_items.append(
            _intent_payload(
                "stromal_immune_infiltration",
                _strength_from_degree(lymphocyte_change["degree"]),
                reference_profile,
                old_prompt,
                new_prompt,
                prompt_diff,
            )
        )
    elif infiltration == "decrease":
        unsupported.append(
            PlanningWarning(
                field="lymphocyte_change.infiltration",
                value=infiltration,
                reason="Phase3 immune decrease primitive is not implemented yet.",
            )
        )

    stroma_change = semantic_diff["stroma_change"]
    if stroma_change["density"] != "none":
        unsupported.append(
            PlanningWarning(
                field="stroma_change.density",
                value=stroma_change["density"],
                reason="Phase3 stromal fibrosis/desmoplasia primitive is not implemented yet.",
            )
        )

    return raw_items, unsupported


def _intent_payload(
    primitive: str,
    strength: str,
    reference_profile: str,
    old_prompt: str | None,
    new_prompt: str | None,
    prompt_diff: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "primitive": primitive,
        "strength": strength,
        "reference_profile": reference_profile,
        "old_prompt": old_prompt,
        "new_prompt": new_prompt,
        "prompt_diff": dict(prompt_diff),
    }


def _strength_from_degree(value: str) -> str:
    if value in {"mild", "moderate", "significant"}:
        return value
    return "moderate"


def _strength_from_necrosis_extent(value: str) -> str:
    if value == "focal":
        return "mild"
    if value == "extensive":
        return "significant"
    return "moderate"


def _intent_order_names(items: tuple[IntentPlanItem, ...]) -> list[str]:
    return [item.primitive for item in items if item.intent is not None]
