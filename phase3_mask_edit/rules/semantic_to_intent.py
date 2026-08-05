"""Map validated semantic diffs to Phase 3 EditIntent objects."""

from __future__ import annotations

import re
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
    "intratumoral_immune_infiltration": 42,
    "immune_infiltration_decrease": 45,
    "stromal_desmoplasia": 50,
    "stroma_increase": 50,
    "stroma_decrease": 55,
}

TRANSITION_PRIMITIVES: dict[str, dict[tuple[str, str], str]] = {
    "PANDA": {
        ("benign_epithelium", "gleason_pattern_3"): "benign_to_gleason3",
        ("benign_epithelium", "stromal_tissue"): "benign_atrophy",
        ("gleason_pattern_3", "gleason_pattern_4"): "gleason_upgrade_3to4",
        ("gleason_pattern_4", "gleason_pattern_5"): "gleason_upgrade_4to5",
        ("gleason_pattern_4", "gleason_pattern_3"): "gleason_downgrade_4to3",
    },
    "GLAS": {
        ("normal_gland", "adenomatous_gland"): "normal_to_adenomatous",
        (
            "adenomatous_gland",
            "moderately_differentiated_carcinoma",
        ): "adenoma_to_carcinoma",
        (
            "moderately_differentiated_carcinoma",
            "poorly_differentiated_carcinoma",
        ): "grade_upgrade",
        (
            "poorly_differentiated_carcinoma",
            "moderately_differentiated_carcinoma",
        ): "treatment_dedifferentiation",
    },
}

RETIRED_LOCAL_GRADE_PRIMITIVES = frozenset(
    {
        "gleason_upgrade_3to4",
        "gleason_upgrade_4to5",
        "gleason_downgrade_4to3",
        "grade_upgrade",
        "treatment_dedifferentiation",
    }
)


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

    schema = MaskProfileSchema.from_reference_profile(reference_profile)
    reference_profile = schema.reference_profile
    context = None
    if old_mask is not None:
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

    transition_change = semantic_diff["transition_change"]
    transition_pair = (
        transition_change["source_state"],
        transition_change["target_state"],
    )
    reported_transition = transition_pair != ("none", "none")
    transition_primitive = TRANSITION_PRIMITIVES.get(reference_profile.upper(), {}).get(
        transition_pair
    )
    transition_has_evidence = _transition_pair_has_text_evidence(
        transition_pair,
        old_prompt=old_prompt,
        new_prompt=new_prompt,
    )
    has_recognized_transition = bool(
        reported_transition
        and transition_primitive
        and transition_has_evidence
    )
    has_explicit_transition = bool(
        has_recognized_transition
        and transition_primitive not in RETIRED_LOCAL_GRADE_PRIMITIVES
    )
    if reported_transition:
        if transition_primitive is None:
            unsupported.append(
                PlanningWarning(
                    field="transition_change",
                    value=f"{transition_pair[0]}->{transition_pair[1]}",
                    reason=(
                        "The explicit fine transition is not supported for "
                        f"reference profile {reference_profile}."
                    ),
                )
            )
        elif not transition_has_evidence:
            unsupported.append(
                PlanningWarning(
                    field="transition_change",
                    value=f"{transition_pair[0]}->{transition_pair[1]}",
                    reason=(
                        "Ignored the model transition because the source/target "
                        "phenotypes are not supported by the prompt text."
                    ),
                )
            )
        elif transition_primitive in RETIRED_LOCAL_GRADE_PRIMITIVES:
            unsupported.append(
                PlanningWarning(
                    field="transition_change",
                    value=f"{transition_pair[0]}->{transition_pair[1]}",
                    reason=(
                        "Localized histologic-grade transformation is not supported "
                        "by the current product. Grade remains an audit and "
                        "preservation attribute, not a generation target."
                    ),
                )
            )
        else:
            raw_items.append(
                _intent_payload(
                    transition_primitive,
                    _strength_from_degree(transition_change["degree"]),
                    reference_profile,
                    old_prompt,
                    new_prompt,
                    prompt_diff,
                )
            )
    else:
        specialized_non_grade = _specialized_non_grade_payload(
            reference_profile=reference_profile,
            old_prompt=old_prompt,
            new_prompt=new_prompt,
            prompt_diff=prompt_diff,
        )
        if specialized_non_grade is not None:
            raw_items.append(specialized_non_grade)

    stroma_change = semantic_diff["stroma_change"]
    reported_stroma_density = stroma_change["density"]
    inferred_stroma_density = (
        _infer_primary_stroma_density_change(old_prompt, new_prompt)
        if reported_stroma_density == "none"
        else "none"
    )
    stroma_density = (
        inferred_stroma_density
        if inferred_stroma_density != "none"
        else reported_stroma_density
    )

    tumor_change = semantic_diff["tumor_change"]
    reported_tumor_growth = tumor_change["growth"]
    necrosis_action = semantic_diff["necrosis_change"]["action"]
    suppress_transition_growth = (
        has_recognized_transition
        and reported_tumor_growth != "none"
        and bool(old_prompt or new_prompt)
        and not _contains_independent_tumor_extent_change(new_prompt)
    )
    suppress_necrosis_replacement_growth = (
        reported_tumor_growth == "decrease"
        and necrosis_action in {"add", "increase"}
        and bool(new_prompt)
        and not _contains_independent_tumor_extent_change(new_prompt)
    )
    suppress_stroma_primary_growth = (
        reported_tumor_growth != "none" and inferred_stroma_density != "none"
    )
    tumor_growth = (
        "none"
        if (
            suppress_transition_growth
            or suppress_necrosis_replacement_growth
            or suppress_stroma_primary_growth
        )
        else reported_tumor_growth
    )
    if suppress_transition_growth:
        unsupported.append(
            PlanningWarning(
                field="tumor_change.growth",
                value=reported_tumor_growth,
                reason=(
                    "Ignored generic tumor growth because an exact fine-grained "
                    "phenotype transition is the primary edit."
                ),
            )
        )
    if suppress_necrosis_replacement_growth:
        unsupported.append(
            PlanningWarning(
                field="tumor_change.growth",
                value=reported_tumor_growth,
                reason=(
                    "Ignored reciprocal viable-tumor displacement because "
                    "necrosis appearance is the primary edit."
                ),
            )
        )
    if suppress_stroma_primary_growth:
        unsupported.append(
            PlanningWarning(
                field="tumor_change.growth",
                value=reported_tumor_growth,
                reason=(
                    "Ignored contextual epithelial prominence because stromal "
                    "density is the primary changed subject."
                ),
            )
        )
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

    grade_change = tumor_change["grade_change"]
    suppress_contextual_grade = (
        grade_change != "none"
        and tumor_growth != "none"
        and bool(new_prompt)
        and not _contains_independent_grade_edit(new_prompt)
    )
    if suppress_contextual_grade:
        unsupported.append(
            PlanningWarning(
                field="tumor_change.grade_change",
                value=grade_change,
                reason=(
                    "Ignored a contextual atypia descriptor because tumor extent "
                    "is the primary edit and no independent grade action is stated."
                ),
            )
        )
    elif grade_change != "none" and not has_recognized_transition:
        special_payload = _specialized_grade_payload(
            grade_change,
            reference_profile=reference_profile,
            old_prompt=old_prompt,
            new_prompt=new_prompt,
            prompt_diff=prompt_diff,
        )
        if (
            special_payload is not None
            and special_payload["primitive"] not in RETIRED_LOCAL_GRADE_PRIMITIVES
        ):
            raw_items.append(special_payload)
        else:
            unsupported.append(
                PlanningWarning(
                    field="tumor_change.grade_change",
                    value=grade_change,
                    reason=(
                        "Localized histologic-grade transformation is not supported "
                        "by the current product. The request must abstain rather than "
                        "approximate grade using tissue or nucleus composition."
                    ),
                )
            )

    necrosis_change = semantic_diff["necrosis_change"]
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
    suppress_contextual_immune = (
        (
            tumor_growth != "none"
            or has_recognized_transition
            or (stroma_density != "none" and _stroma_is_primary_subject(new_prompt))
        )
        and infiltration != "none"
        and bool(new_prompt)
        and not _contains_independent_immune_edit(new_prompt)
    )
    if suppress_contextual_immune:
        unsupported.append(
            PlanningWarning(
                field="lymphocyte_change.infiltration",
                value=infiltration,
                reason=(
                    "Ignored a contextual immune descriptor because tumor extent "
                    "is the primary edit and no independent immune action is stated."
                ),
            )
        )
    elif infiltration == "increase":
        raw_items.append(
            _intent_payload(
                _immune_increase_primitive(
                    lymphocyte_change["location"], old_prompt, new_prompt
                ),
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

    suppress_transition_stroma = (
        has_recognized_transition
        and stroma_density != "none"
        and bool(new_prompt)
        and not _contains_independent_stroma_action(new_prompt)
    )
    if suppress_transition_stroma:
        unsupported.append(
            PlanningWarning(
                field="stroma_change.density",
                value=stroma_density,
                reason=(
                    "Ignored a contextual stromal reaction because the exact "
                    "phenotype transition is the primary edit."
                ),
            )
        )
    elif stroma_density == "increase":
        primary_stroma_primitive = (
            "stromal_desmoplasia"
            if _contains_independent_stroma_edit(
                _normalize_text(new_prompt or old_prompt)
            )
            else "stroma_increase"
        )
        payload = _intent_payload(
            primary_stroma_primitive,
            _strength_from_degree(stroma_change["degree"]),
            reference_profile,
            old_prompt,
            new_prompt,
            prompt_diff,
        )
        if (
            _stroma_increase_is_immune_replacement_fallback(
                semantic_diff,
                old_prompt=old_prompt,
                new_prompt=new_prompt,
            )
            or _stroma_increase_is_necrosis_replacement_fallback(
                semantic_diff,
                old_prompt=old_prompt,
                new_prompt=new_prompt,
            )
            or _stroma_increase_is_tumor_replacement_fallback(
                semantic_diff,
                old_prompt=old_prompt,
                new_prompt=new_prompt,
            )
        ):
            payload = _intent_payload(
                "stromal_desmoplasia",
                _strength_from_degree(stroma_change["degree"]),
                reference_profile,
                old_prompt,
                new_prompt,
                prompt_diff,
            )
            primary_primitive = (
                "immune_infiltration_decrease"
                if semantic_diff["lymphocyte_change"]["infiltration"] == "decrease"
                else (
                    "necrosis_resolution"
                    if semantic_diff["necrosis_change"]["action"]
                    in {"decrease", "remove"}
                    else "tumor_burden_decrease"
                )
            )
            group = (
                "immune_decrease_stroma_replacement"
                if primary_primitive == "immune_infiltration_decrease"
                else (
                    "necrosis_resolution_stroma_replacement"
                    if primary_primitive == "necrosis_resolution"
                    else "tumor_decrease_stroma_replacement"
                )
            )
            payload = _fallback_payload(
                payload,
                group=group,
                fallback_for=primary_primitive,
                note=(
                    "Stroma increase was interpreted as the replacement/backfill "
                    f"target for {primary_primitive}, not a separate desmoplasia edit."
                ),
            )
            _mark_primary_payload(
                raw_items,
                primitive=primary_primitive,
                group=group,
                note=(
                    f"Primary realization for {primary_primitive} with stromal "
                    "replacement/backfill."
                ),
            )
            unsupported.append(
                PlanningWarning(
                    field="stroma_change.density",
                    value="increase",
                    reason=(
                        f"Treated as a fallback for {primary_primitive} "
                        "because the text describes stromal replacement/backfill "
                        "for the same source-tissue replacement request."
                    ),
                )
            )
        raw_items.append(payload)
    elif stroma_density != "none":
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

    implicit_fallback = _implicit_stroma_replacement_fallback(
        semantic_diff,
        old_prompt=old_prompt,
        new_prompt=new_prompt,
    )
    if stroma_density == "none" and implicit_fallback is not None:
        primary_primitive, group = implicit_fallback
        payload = _fallback_payload(
            _intent_payload(
                "stromal_desmoplasia",
                "moderate",
                reference_profile,
                old_prompt,
                new_prompt,
                prompt_diff,
            ),
            group=group,
            fallback_for=primary_primitive,
            note=(
                "Stromal replacement/backfill was retained as a fallback "
                f"realization for {primary_primitive}, not a separate primary edit."
            ),
        )
        _mark_primary_payload(
            raw_items,
            primitive=primary_primitive,
            group=group,
            note=(
                f"Primary realization for {primary_primitive} with stromal "
                "replacement/backfill."
            ),
        )
        raw_items.append(payload)

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
    return _contains_any(text, _IMMUNE_TERMS) and _contains_any(text, _STROMA_TERMS)


def _stroma_increase_is_necrosis_replacement_fallback(
    semantic_diff: Mapping[str, Any],
    *,
    old_prompt: str | None,
    new_prompt: str | None,
) -> bool:
    necrosis_change = semantic_diff.get("necrosis_change", {})
    stroma_change = semantic_diff.get("stroma_change", {})
    if not isinstance(necrosis_change, Mapping) or not isinstance(
        stroma_change, Mapping
    ):
        return False
    if necrosis_change.get("action") not in {"decrease", "remove"}:
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
    return _contains_any(text, _NECROSIS_TERMS) and _contains_any(text, _STROMA_TERMS)


def _stroma_increase_is_tumor_replacement_fallback(
    semantic_diff: Mapping[str, Any],
    *,
    old_prompt: str | None,
    new_prompt: str | None,
) -> bool:
    tumor_change = semantic_diff.get("tumor_change", {})
    stroma_change = semantic_diff.get("stroma_change", {})
    if not isinstance(tumor_change, Mapping) or not isinstance(stroma_change, Mapping):
        return False
    if tumor_change.get("growth") != "decrease":
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
    return _contains_any(text, _TUMOR_TERMS) and _contains_any(text, _STROMA_TERMS)


def _implicit_stroma_replacement_fallback(
    semantic_diff: Mapping[str, Any],
    *,
    old_prompt: str | None,
    new_prompt: str | None,
) -> tuple[str, str] | None:
    """Retain an explicit stromal backfill as a non-primary planner item."""

    stroma_change = semantic_diff.get("stroma_change", {})
    if not isinstance(stroma_change, Mapping) or stroma_change.get("density") != "none":
        return None
    text = _normalize_text(new_prompt) or _normalize_text(
        f"{old_prompt or ''} {new_prompt or ''}"
    )
    if (
        not text
        or _contains_independent_stroma_edit(text)
        or not _contains_any(text, _STROMA_REPLACEMENT_TERMS)
        or not _contains_any(text, _STROMA_TERMS)
    ):
        return None

    lymphocyte_change = semantic_diff.get("lymphocyte_change", {})
    if (
        isinstance(lymphocyte_change, Mapping)
        and lymphocyte_change.get("infiltration") == "decrease"
        and _contains_any(text, _IMMUNE_TERMS)
    ):
        return (
            "immune_infiltration_decrease",
            "immune_decrease_stroma_replacement",
        )

    necrosis_change = semantic_diff.get("necrosis_change", {})
    if (
        isinstance(necrosis_change, Mapping)
        and necrosis_change.get("action") in {"decrease", "remove"}
        and _contains_any(text, _NECROSIS_TERMS)
    ):
        return ("necrosis_resolution", "necrosis_resolution_stroma_replacement")

    tumor_change = semantic_diff.get("tumor_change", {})
    if (
        isinstance(tumor_change, Mapping)
        and tumor_change.get("growth") == "decrease"
        and _contains_any(text, _TUMOR_TERMS)
    ):
        return ("tumor_burden_decrease", "tumor_decrease_stroma_replacement")
    return None


_IMMUNE_TERMS = (
    "immune",
    "lymphocyte",
    "lymphocytic",
    "til",
    "inflammatory",
    "inflammation",
)

_NECROSIS_TERMS = (
    "necrosis",
    "necrotic",
    "debris",
    "dead tissue",
    "dead-looking",
    "dead looking",
)

_TUMOR_TERMS = (
    "tumor",
    "tumour",
    "cancer",
    "carcinoma",
    "neoplasm",
    "malignant tissue",
    "tumor burden",
    "tumour burden",
)

_STROMA_TERMS = (
    "stroma",
    "stromal",
    "connective",
    "fibrous tissue",
)

_STROMA_REPLACEMENT_TERMS = (
    "replace",
    "replacement",
    "backfill",
    "back-fill",
    "restore with stroma",
    "restore viable stroma",
    "fill the vacated",
    "fill the removed",
)

_INDEPENDENT_STROMA_EDIT_TERMS = (
    "desmoplasia",
    "desmoplastic",
    "stromal response",
    "stromal reaction",
    "peritumoral stroma",
)


def _contains_independent_stroma_edit(text: str) -> bool:
    return _contains_any(text, _INDEPENDENT_STROMA_EDIT_TERMS)


def _contains_independent_stroma_action(text: str | None) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(normalized, _INDEPENDENT_STROMA_ACTION_TERMS)


_INDEPENDENT_STROMA_ACTION_TERMS = (
    "also increase stroma",
    "also increase stromal",
    "also add stroma",
    "also add desmoplasia",
    "increase desmoplasia",
    "increase stromal reaction",
    "add desmoplasia",
    "independently increase stroma",
)


def _stroma_is_primary_subject(text: str | None) -> bool:
    first_sentence = _normalize_text(text).split(".", 1)[0]
    return (
        bool(first_sentence)
        and _contains_any(first_sentence, _STROMA_TERMS)
        and not _contains_any(
            first_sentence,
            _TUMOR_TERMS + _IMMUNE_TERMS + _NECROSIS_TERMS,
        )
    )


def _infer_primary_stroma_density_change(
    old_prompt: str | None,
    new_prompt: str | None,
) -> str:
    if not _stroma_is_primary_subject(new_prompt):
        return "none"
    old_text = _normalize_text(old_prompt)
    new_text = _normalize_text(new_prompt)
    if not _contains_any(old_text, _STROMA_TERMS) or not _contains_any(
        new_text, _STROMA_TERMS
    ):
        return "none"
    old_low = _contains_any(old_text, _LOW_STROMA_TERMS)
    old_high = _contains_any(old_text, _HIGH_STROMA_TERMS)
    new_low = _contains_any(new_text, _LOW_STROMA_TERMS)
    new_high = _contains_any(new_text, _HIGH_STROMA_TERMS)
    if old_low and new_high:
        return "increase"
    if old_high and new_low:
        return "decrease"
    return "none"


_LOW_STROMA_TERMS = (
    "scant stroma",
    "scant fibrous stroma",
    "minimal stroma",
    "minimal fibrous stroma",
    "limited stromal tissue",
    "sparse stroma",
)

_HIGH_STROMA_TERMS = (
    "abundant stroma",
    "abundant fibrous stroma",
    "dense stroma",
    "dense fibrous stroma",
    "prominent stroma",
    "prominent stromal",
    "well-developed",
    "collagen deposition",
    "desmoplasia",
    "desmoplastic",
)


def _contains_independent_tumor_extent_change(text: str | None) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(normalized, _INDEPENDENT_TUMOR_EXTENT_TERMS)


_INDEPENDENT_TUMOR_EXTENT_TERMS = (
    "more tumor",
    "greater tumor",
    "increased tumor",
    "larger tumor",
    "expanded tumor",
    "tumor expansion",
    "tumor burden",
    "occupying more",
    "occupies more",
    "larger area",
    "increased area",
    "greater extent",
    "decrease tumor",
    "decreased tumor",
    "reduce tumor",
    "reduced tumor",
    "less tumor",
    "smaller tumor",
    "tumor regression",
    "shrink tumor",
    "shrinking tumor",
)


def _contains_independent_grade_edit(text: str | None) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(normalized, _INDEPENDENT_GRADE_EDIT_TERMS)


_INDEPENDENT_GRADE_EDIT_TERMS = (
    "grade",
    "differentiated",
    "differentiation",
    "dedifferentiation",
    "gleason",
    "upgrade",
    "downgrade",
)


def _transition_pair_has_text_evidence(
    pair: tuple[str, str],
    *,
    old_prompt: str | None,
    new_prompt: str | None,
) -> bool:
    if pair == ("none", "none") or not (old_prompt or new_prompt):
        return True
    source_state, target_state = pair
    source_text = _normalize_text(f"{old_prompt or ''} {new_prompt or ''}")
    target_text = _normalize_text(new_prompt)
    return _contains_any(
        source_text, _TRANSITION_STATE_TERMS[source_state]
    ) and _contains_any(target_text, _TRANSITION_STATE_TERMS[target_state])


_TRANSITION_STATE_TERMS: dict[str, tuple[str, ...]] = {
    "none": ("none",),
    "benign_epithelium": (
        "benign epithelium",
        "benign prostatic epithelium",
        "normal epithelium",
        "benign gland",
    ),
    "stromal_tissue": ("stromal tissue", "stroma"),
    "gleason_pattern_3": ("gleason pattern 3", "gleason 3", "pattern 3"),
    "gleason_pattern_4": ("gleason pattern 4", "gleason 4", "pattern 4"),
    "gleason_pattern_5": ("gleason pattern 5", "gleason 5", "pattern 5"),
    "normal_gland": (
        "normal gland",
        "normal colonic gland",
        "normal colonic glands",
        "normal colorectal gland",
        "normal colorectal glands",
    ),
    "adenomatous_gland": ("adenomatous gland", "adenoma", "adenomatous"),
    "moderately_differentiated_carcinoma": (
        "moderately differentiated carcinoma",
        "moderately differentiated colorectal carcinoma",
    ),
    "poorly_differentiated_carcinoma": (
        "poorly differentiated carcinoma",
        "poorly differentiated colorectal carcinoma",
    ),
}


def _contains_independent_immune_edit(text: str | None) -> bool:
    normalized = _normalize_text(text)
    return _contains_any(normalized, _INDEPENDENT_IMMUNE_EDIT_TERMS)


_INDEPENDENT_IMMUNE_EDIT_TERMS = (
    "increase immune",
    "increased immune",
    "add immune",
    "added immune",
    "more immune",
    "increase lymphocyte",
    "increased lymphocyte",
    "add lymphocyte",
    "more lymphocyte",
    "stronger immune",
    "brisk til",
    "dense til",
    "decrease immune",
    "decreased immune",
    "reduce immune",
    "reduced immune",
    "less immune",
    "decrease lymphocyte",
    "reduce lymphocyte",
    "sparse til",
)


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


def _specialized_non_grade_payload(
    *,
    reference_profile: str,
    old_prompt: str | None,
    new_prompt: str | None,
    prompt_diff: Mapping[str, Any],
) -> dict[str, Any] | None:
    if reference_profile.upper() != "PANDA":
        return None
    old_text = _normalize_text(old_prompt)
    new_text = _normalize_text(new_prompt)
    combined = f"{old_text} {new_text}".strip()
    has_normal_epithelium = _mentions_normal_or_benign_epithelium(combined)
    has_stromal_target = _contains_any(
        new_text,
        ("stroma", "stromal tissue", "fibrous connective tissue"),
    )
    has_replacement_signal = _contains_any(
        new_text,
        (
            "replace",
            "convert",
            "without epithelial",
            "no epithelial",
            "epithelial structures are not prominent",
        ),
    )
    if not (has_normal_epithelium and has_stromal_target and has_replacement_signal):
        return None
    return _intent_payload(
        "benign_atrophy",
        _strength_from_grade_prompt(old_prompt, new_prompt),
        reference_profile,
        old_prompt,
        new_prompt,
        prompt_diff,
    )


def _immune_increase_primitive(
    location: str,
    old_prompt: str | None,
    new_prompt: str | None,
) -> str:
    if location == "intratumoral":
        return "intratumoral_immune_infiltration"
    if location in {"stromal", "peritumoral"}:
        return "stromal_immune_infiltration"
    text = _normalize_text(f"{old_prompt or ''} {new_prompt or ''}")
    if _contains_any(
        text,
        (
            "intratumoral",
            "inside tumor",
            "within tumor",
            "among tumor",
            "interspersed among tumor",
            "central tumor compartment",
        ),
    ):
        return "intratumoral_immune_infiltration"
    return "stromal_immune_infiltration"


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
            _mentions_normal_or_benign_epithelium(combined)
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
            "normal" in old_text and _contains_any(new_text, ("adenoma", "adenomatous"))
        ) or (
            "normal" in combined and _contains_any(combined, ("adenoma", "adenomatous"))
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


def _mentions_normal_or_benign_epithelium(text: str) -> bool:
    return bool(re.search(r"\b(?:normal|benign)(?:\s+\w+){0,3}\s+epitheli", text))


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _mentions_gleason4(text: str) -> bool:
    return _contains_any(text, ("gleason 4", "pattern 4", "grade group 4"))


def _mentions_gleason5(text: str) -> bool:
    return _contains_any(text, ("gleason 5", "pattern 5", "grade group 5"))


def _mentions_transition(
    old_text: str, new_text: str, source: str, target: str
) -> bool:
    return _contains_any(
        old_text, (f"gleason {source}", f"pattern {source}")
    ) and _contains_any(new_text, (f"gleason {target}", f"pattern {target}"))


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
