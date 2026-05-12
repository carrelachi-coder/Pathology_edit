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
    "rejected_by_applicability",
    "unsupported",
]

INTENT_ORDER = {
    "tumor_burden_increase": 10,
    "tumor_burden_decrease": 20,
    "dcis_invasion": 25,
    "angioinvasion_emphasis": 26,
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
        recipe = load_recipe(default_recipe_path_for_profile(reference_profile))

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
        raw_items.append(
            _intent_payload(
                "stromal_desmoplasia",
                _strength_from_degree(stroma_change["degree"]),
                reference_profile,
                old_prompt,
                new_prompt,
                prompt_diff,
            )
        )
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
        if "normal" in old_text and _contains_any(
            new_text, ("adenoma", "adenomatous")
        ):
            return "normal_to_adenomatous"
        if _contains_any(old_text, ("adenoma", "adenomatous")) and _contains_any(
            new_text,
            ("carcinoma", "moderately differentiated", "moderate differentiation"),
        ):
            return "adenoma_to_carcinoma"
        if _contains_any(
            new_text,
            ("poorly differentiated", "poor differentiation", "high grade"),
        ):
            return "grade_upgrade"
        if _contains_any(combined, ("adenoma", "adenomatous")):
            return "adenoma_to_carcinoma"
        return "grade_upgrade"

    if profile == "BCSS":
        if _contains_any(
            new_text, ("angioinvasion", "vascular invasion", "lymphovascular")
        ):
            return "angioinvasion_emphasis"
        if "dcis" in old_text and _contains_any(new_text, ("invasive", "invasion")):
            return "dcis_invasion"
        if "dcis" in combined and _contains_any(new_text, ("invasive", "invasion")):
            return "dcis_invasion"

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
