"""Map validated semantic diffs to Phase 3 EditIntent objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from phase3_mask_edit.core.applicability import assess_edit_applicability
from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.parser.semantic_diff import validate_semantic_diff


PlanItemStatus = Literal[
    "planned",
    "degraded_planned",
    "fallback_planned",
    "rejected_by_applicability",
    "unsupported",
]

INTENT_ORDER = {
    "tumor_burden_increase": 10,
    "tumor_burden_decrease": 20,
    "benign_to_gleason3": 27,
    "normal_to_adenomatous": 27,
    "gleason_upgrade_3to4": 28,
    "adenoma_to_carcinoma": 28,
    "gleason_upgrade_4to5": 29,
    "grade_upgrade": 29,
    "gleason_downgrade_4to3": 30,
    "treatment_dedifferentiation": 30,
    "benign_atrophy": 31,
    "necrosis_appearance": 30,
    "necrosis_resolution": 35,
    "stromal_immune_infiltration": 40,
    "immune_infiltration_decrease": 45,
    "stromal_desmoplasia": 50,
    "stroma_decrease": 55,
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
    execution_group: str | None = None
    role: str = "primary"
    fallback_for: str | None = None
    planning_note: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        metadata = {
            "primitive": self.primitive,
            "strength": self.strength,
            "status": self.status,
            "applicability": self.applicability,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "fallback_actions": list(self.fallback_actions),
            "execution_group": self.execution_group,
            "role": self.role,
            "fallback_for": self.fallback_for,
            "planning_note": self.planning_note,
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
        return tuple(
            item.intent
            for item in self.items
            if item.intent is not None and item.role != "fallback"
        )

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

    context = None
    if old_mask is not None:
        schema = MaskProfileSchema.from_reference_profile(reference_profile)
        context = MaskEditContext.from_mask(old_mask, schema)

    raw_items, unsupported = _raw_intent_specs(
        validated,
        reference_profile=reference_profile,
        old_prompt=old_prompt,
        new_prompt=new_prompt,
        context=context,
    )
    raw_items.sort(key=lambda item: INTENT_ORDER.get(item["primitive"], 999))

    if old_mask is None:
        items = tuple(
            IntentPlanItem(
                primitive=payload["primitive"],
                strength=payload["strength"],
                status=_planned_status_for_payload(payload, degraded=False),
                intent=EditIntent.from_mapping(payload),
                execution_group=_payload_text(payload, "_execution_group"),
                role=_payload_text(payload, "_role") or "primary",
                fallback_for=_payload_text(payload, "_fallback_for"),
                planning_note=_payload_text(payload, "_planning_note"),
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
        recipe = load_recipe(default_recipe_path_for_profile(reference_profile))

    assert context is not None
    gated_items: list[IntentPlanItem] = []
    for payload in raw_items:
        intent = EditIntent.from_mapping(payload)
        decision = assess_edit_applicability(intent, recipe, schema, context)
        if decision.status == "rejected":
            status: PlanItemStatus = "rejected_by_applicability"
            item_intent = None
        else:
            status = _planned_status_for_payload(
                payload,
                degraded=decision.status == "degraded",
            )
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
                execution_group=_payload_text(payload, "_execution_group"),
                role=_payload_text(payload, "_role") or "primary",
                fallback_for=_payload_text(payload, "_fallback_for"),
                planning_note=_payload_text(payload, "_planning_note"),
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
    context: MaskEditContext | None,
) -> tuple[list[dict[str, Any]], list[PlanningWarning]]:
    prompt_diff = {"semantic_diff": semantic_diff}
    raw_items: list[dict[str, Any]] = []
    unsupported: list[PlanningWarning] = []

    tumor_change = semantic_diff["tumor_change"]
    tumor_growth = tumor_change["growth"]
    if tumor_growth == "increase":
        primitive, warning = _select_tumor_growth_primitive(context)
        if primitive is None:
            unsupported.append(
                PlanningWarning(
                    field="tumor_change.growth",
                    value=tumor_growth,
                    reason=(
                        "No feasible primitive can realize tumor growth from the "
                        "current mask composition."
                    ),
                )
            )
        else:
            raw_items.append(
                _intent_payload(
                    primitive,
                    _strength_from_degree(tumor_change["degree"]),
                    reference_profile,
                    old_prompt,
                    new_prompt,
                    prompt_diff,
                )
            )
            if warning is not None:
                unsupported.append(warning)
    elif tumor_growth == "decrease":
        primitive, warning = _select_tumor_decrease_primitive(context)
        if primitive is None:
            unsupported.append(
                PlanningWarning(
                    field="tumor_change.growth",
                    value=tumor_growth,
                    reason=(
                        "No feasible primitive can realize tumor decrease from the "
                        "current mask composition."
                    ),
                )
            )
        else:
            raw_items.append(
                _intent_payload(
                    primitive,
                    _strength_from_degree(tumor_change["degree"]),
                    reference_profile,
                    old_prompt,
                    new_prompt,
                    prompt_diff,
                )
            )
            if warning is not None:
                unsupported.append(warning)

    if tumor_change["grade_change"] != "none":
        special_payload = _specialized_grade_payload(
            tumor_change["grade_change"],
            reference_profile=reference_profile,
            old_prompt=old_prompt,
            new_prompt=new_prompt,
            prompt_diff=prompt_diff,
        )
        if special_payload is None:
            unsupported.append(
                PlanningWarning(
                    field="tumor_change.grade_change",
                    value=tumor_change["grade_change"],
                    reason=(
                        "No dataset-specialized fine-ID transition could be inferred "
                        "from the reference profile and prompt wording."
                    ),
                )
            )
        else:
            raw_items.append(special_payload)

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
        if tumor_growth == "increase":
            unsupported.append(
                PlanningWarning(
                    field="necrosis_change.action",
                    value=necrosis_action,
                    reason=(
                        "Deferred because tumor_burden_increase can consume or "
                        "replace minor necrotic/debris regions as a secondary effect."
                    ),
                )
            )
        else:
            raw_items.append(
                _intent_payload(
                    "necrosis_resolution",
                    _strength_from_necrosis_resolution(
                        necrosis_action,
                        necrosis_change["extent"],
                    ),
                    reference_profile,
                    old_prompt,
                    new_prompt,
                    prompt_diff,
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
        if tumor_growth == "increase":
            unsupported.append(
                PlanningWarning(
                    field="lymphocyte_change.infiltration",
                    value=infiltration,
                    reason=(
                        "Deferred because lymphocyte reduction is already implied "
                        "by tumor_burden_increase replacing immune-rich tissue."
                    ),
                )
            )
        else:
            raw_items.append(
                _intent_payload(
                    "immune_infiltration_decrease",
                    _strength_from_degree(lymphocyte_change["degree"]),
                    reference_profile,
                    old_prompt,
                    new_prompt,
                    prompt_diff,
                )
            )

    stroma_change = semantic_diff["stroma_change"]
    if stroma_change["density"] == "increase":
        payload = _intent_payload(
            "stromal_desmoplasia",
            _strength_from_degree(stroma_change["degree"]),
            reference_profile,
            old_prompt,
            new_prompt,
            prompt_diff,
        )
        if _stroma_increase_is_immune_replacement_fallback(
            semantic_diff,
            old_prompt=old_prompt,
            new_prompt=new_prompt,
        ):
            payload = _fallback_payload(
                payload,
                group="immune_decrease_stroma_replacement",
                fallback_for="immune_infiltration_decrease",
                note=(
                    "Stroma increase was interpreted as the replacement/backfill "
                    "target for immune decrease, not a separate desmoplasia edit."
                ),
            )
            _mark_primary_payload(
                raw_items,
                primitive="immune_infiltration_decrease",
                group="immune_decrease_stroma_replacement",
                note=(
                    "Primary realization for immune decrease with stromal "
                    "replacement/backfill."
                ),
            )
            unsupported.append(
                PlanningWarning(
                    field="stroma_change.density",
                    value="increase",
                    reason=(
                        "Treated as a fallback for immune_infiltration_decrease "
                        "because the text describes stromal replacement/backfill "
                        "for the same immune-decrease request."
                    ),
                )
            )
        raw_items.append(payload)
    elif stroma_change["density"] != "none":
        raw_items.append(
            _intent_payload(
                "stroma_decrease",
                _strength_from_degree(stroma_change["degree"]),
                reference_profile,
                old_prompt,
                new_prompt,
                prompt_diff,
            )
        )

    return raw_items, unsupported


def _select_tumor_growth_primitive(
    context: MaskEditContext | None,
) -> tuple[str | None, PlanningWarning | None]:
    if context is None:
        return "tumor_burden_increase", None

    present = set(context.present_labels)
    has_editable_non_tumor = bool(
        present & {"Stroma", "Normal epithelium", "Other tissue", "Immune infiltrate"}
    )
    has_necrosis = "Necrosis" in present
    if has_editable_non_tumor:
        return "tumor_burden_increase", None
    if has_necrosis:
        return (
            "necrosis_resolution",
            PlanningWarning(
                field="tumor_change.growth",
                value="increase",
                reason=(
                    "Mapped tumor growth to necrosis_resolution because the mask "
                    "contains necrosis but lacks editable non-tumor source tissue."
                ),
            ),
        )
    return None, None


def _select_tumor_decrease_primitive(
    context: MaskEditContext | None,
) -> tuple[str | None, PlanningWarning | None]:
    if context is None:
        return "tumor_burden_decrease", None

    present = set(context.present_labels)
    has_editable_backfill = bool(
        present & {"Stroma", "Normal epithelium", "Other tissue", "Immune infiltrate"}
    )
    has_necrosis = "Necrosis" in present
    if has_editable_backfill:
        return "tumor_burden_decrease", None
    if has_necrosis:
        return (
            "necrosis_appearance",
            PlanningWarning(
                field="tumor_change.growth",
                value="decrease",
                reason=(
                    "Mapped tumor decrease to necrosis_appearance because the mask "
                    "contains necrosis but lacks editable backfill tissue."
                ),
            ),
        )
    return None, None


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


def _fallback_payload(
    payload: Mapping[str, Any],
    *,
    group: str,
    fallback_for: str,
    note: str,
) -> dict[str, Any]:
    updated = dict(payload)
    updated["_execution_group"] = group
    updated["_role"] = "fallback"
    updated["_fallback_for"] = fallback_for
    updated["_planning_note"] = note
    return updated


def _mark_primary_payload(
    raw_items: list[dict[str, Any]],
    *,
    primitive: str,
    group: str,
    note: str,
) -> None:
    for item in reversed(raw_items):
        if item.get("primitive") != primitive:
            continue
        item.setdefault("_execution_group", group)
        item.setdefault("_role", "primary")
        item.setdefault("_planning_note", note)
        return


def _planned_status_for_payload(
    payload: Mapping[str, Any],
    *,
    degraded: bool,
) -> PlanItemStatus:
    if payload.get("_role") == "fallback":
        return "fallback_planned"
    if degraded:
        return "degraded_planned"
    return "planned"


def _payload_text(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _stroma_increase_is_immune_replacement_fallback(
    semantic_diff: Mapping[str, Any],
    *,
    old_prompt: str | None,
    new_prompt: str | None,
) -> bool:
    lymphocyte_change = semantic_diff.get("lymphocyte_change", {})
    stroma_change = semantic_diff.get("stroma_change", {})
    if not isinstance(lymphocyte_change, Mapping) or not isinstance(
        stroma_change, Mapping
    ):
        return False
    if lymphocyte_change.get("infiltration") != "decrease":
        return False
    if stroma_change.get("density") != "increase":
        return False

    text = _normalize_text(new_prompt) or _normalize_text(
        f"{old_prompt or ''} {new_prompt or ''}"
    )
    if not text:
        return False
    if _contains_independent_stroma_edit(text):
        return False
    return (
        _contains_any(text, _IMMUNE_REPLACEMENT_TERMS)
        and _contains_any(text, _IMMUNE_TERMS)
        and _contains_any(text, _STROMA_TERMS)
    )


_IMMUNE_TERMS = (
    "immune",
    "lymphocyte",
    "lymphocytic",
    "til",
    "inflammatory",
    "inflammation",
)

_STROMA_TERMS = (
    "stroma",
    "stromal",
    "connective",
    "fibrous tissue",
)

_IMMUNE_REPLACEMENT_TERMS = (
    "replace",
    "replaced",
    "replacement",
    "backfill",
    "backfilled",
    "fill with",
    "filled with",
    "convert",
    "converted",
    "conversion",
    "turn into",
    "turned into",
)

_INDEPENDENT_STROMA_EDIT_TERMS = (
    "desmoplasia",
    "desmoplastic",
    "stromal response",
    "stromal reaction",
    "fibrosis",
    "fibrotic",
    "collagenous",
    "dense stroma",
    "peritumoral stroma",
)


def _contains_independent_stroma_edit(text: str) -> bool:
    return _contains_any(text, _INDEPENDENT_STROMA_EDIT_TERMS)


def _specialized_grade_payload(
    grade_change: str,
    *,
    reference_profile: str,
    old_prompt: str | None,
    new_prompt: str | None,
    prompt_diff: Mapping[str, Any],
) -> dict[str, Any] | None:
    primitive = _specialized_grade_primitive(
        grade_change,
        reference_profile=reference_profile,
        old_prompt=old_prompt,
        new_prompt=new_prompt,
    )
    if primitive is None:
        return None
    return _intent_payload(
        primitive,
        _strength_from_grade_prompt(old_prompt, new_prompt),
        reference_profile,
        old_prompt,
        new_prompt,
        prompt_diff,
    )


def _specialized_grade_primitive(
    grade_change: str,
    *,
    reference_profile: str,
    old_prompt: str | None,
    new_prompt: str | None,
) -> str | None:
    profile = reference_profile.upper()
    old_text = _normalize_text(old_prompt)
    new_text = _normalize_text(new_prompt)
    combined = f"{old_text} {new_text}".strip()

    if profile == "PANDA":
        if grade_change == "downgrade":
            return "gleason_downgrade_4to3"
        if _mentions_benign_to_gleason3(old_text, new_text) or (
            _contains_any(combined, ("benign", "normal epithelium"))
            and _contains_any(
                combined,
                ("gleason 3", "pattern 3", "low grade malignant"),
            )
        ):
            return "benign_to_gleason3"
        if _mentions_single_prompt_transition(combined, "4", "5"):
            return "gleason_upgrade_4to5"
        if _mentions_single_prompt_transition(combined, "3", "4"):
            return "gleason_upgrade_3to4"
        if _mentions_benign_to_gleason3(old_text, new_text):
            return "benign_to_gleason3"
        if _mentions_gleason5(new_text) or _mentions_transition(
            old_text, new_text, "4", "5"
        ):
            return "gleason_upgrade_4to5"
        if (
            _mentions_gleason4(new_text)
            or _mentions_transition(old_text, new_text, "3", "4")
            or "gleason" in combined
        ):
            return "gleason_upgrade_3to4"
        return "gleason_upgrade_3to4"

    if profile == "GLAS":
        if grade_change == "downgrade":
            return "treatment_dedifferentiation"
        if (
            "normal" in old_text
            and _contains_any(new_text, ("adenoma", "adenomatous"))
        ) or (
            "normal" in combined
            and _contains_any(combined, ("adenoma", "adenomatous"))
        ):
            return "normal_to_adenomatous"
        if (
            _contains_any(old_text, ("adenoma", "adenomatous"))
            and _contains_any(
                new_text,
                (
                    "carcinoma",
                    "moderately differentiated",
                    "moderate differentiation",
                ),
            )
        ) or (
            _contains_any(combined, ("adenoma", "adenomatous"))
            and _contains_any(
                combined,
                (
                    "carcinoma",
                    "moderately differentiated",
                    "moderate differentiation",
                ),
            )
        ):
            return "adenoma_to_carcinoma"
        if _contains_any(
            new_text,
            ("poorly differentiated", "poor differentiation", "high grade"),
        ) or _contains_any(
            combined,
            ("poorly differentiated", "poor differentiation", "high grade"),
        ):
            return "grade_upgrade"
        if _contains_any(combined, ("adenoma", "adenomatous")):
            return "adenoma_to_carcinoma"
        return "grade_upgrade"

    return None


def _strength_from_grade_prompt(old_prompt: str | None, new_prompt: str | None) -> str:
    text = _normalize_text(f"{old_prompt or ''} {new_prompt or ''}")
    if _contains_any(
        text, ("extensive", "marked", "significant", "predominant", "widespread")
    ):
        return "significant"
    if _contains_any(text, ("focal", "small", "limited", "mild")):
        return "mild"
    return "moderate"


def _normalize_text(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", " ").replace("_", " ")


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _mentions_gleason4(text: str) -> bool:
    return _contains_any(text, ("gleason 4", "pattern 4", "grade group 4"))


def _mentions_gleason5(text: str) -> bool:
    return _contains_any(text, ("gleason 5", "pattern 5", "grade group 5"))


def _mentions_transition(old_text: str, new_text: str, source: str, target: str) -> bool:
    return (
        _contains_any(old_text, (f"gleason {source}", f"pattern {source}"))
        and _contains_any(new_text, (f"gleason {target}", f"pattern {target}"))
    )


def _mentions_single_prompt_transition(text: str, source: str, target: str) -> bool:
    source_terms = (f"gleason {source}", f"pattern {source}")
    target_terms = (f"gleason {target}", f"pattern {target}")
    transition_terms = (
        f"{source} to {target}",
        f"{source}->{target}",
        f"{source} -> {target}",
        f"{source}to{target}",
    )
    return (
        _contains_any(text, source_terms)
        and _contains_any(text, target_terms)
        and _contains_any(text, transition_terms + ("upgrade", "upgraded", "convert"))
    )


def _mentions_benign_to_gleason3(old_text: str, new_text: str) -> bool:
    return _contains_any(old_text, ("benign", "normal epithelium")) and _contains_any(
        new_text,
        ("gleason 3", "pattern 3", "low grade malignant"),
    )




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


def _strength_from_necrosis_resolution(action: str, extent: str) -> str:
    if action == "remove":
        return "significant"
    return _strength_from_necrosis_extent(extent)


def _intent_order_names(items: tuple[IntentPlanItem, ...]) -> list[str]:
    return [item.primitive for item in items if item.intent is not None]
