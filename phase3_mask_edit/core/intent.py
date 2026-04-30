"""Structured edit intents for Phase 3 mask execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


VALID_STRENGTHS = frozenset({"mild", "moderate", "significant", "xlarge_deid"})

REFERENCE_PROFILE_BY_DOMAIN = {
    "breast": "BCSS",
    "breast cancer": "BCSS",
    "prostate": "PANDA",
    "prostate cancer": "PANDA",
    "colorectal": "GlaS",
    "colon": "GlaS",
    "colorectal cancer": "GlaS",
    "lung": "IGNITE",
    "lung cancer": "IGNITE",
    "melanoma": "PUMA",
    "skin": "PUMA",
    "skin cancer": "PUMA",
    "oral": "ORCA",
    "oral_scc": "ORCA",
    "oral squamous cell carcinoma": "ORCA",
    "head_neck": "ORCA",
    "head and neck": "ORCA",
}


class IntentValidationError(ValueError):
    """Raised when a rule-engine payload cannot become a valid edit intent."""


@dataclass(frozen=True)
class EditIntent:
    """One mask edit request emitted by the LLM parser / rule engine."""

    primitive: str
    strength: str = "moderate"
    reference_profile: str | None = None
    organ: str | None = None
    cancer_type: str | None = None
    site: str | None = None
    diagnosis: str | None = None
    target_change_fraction: float | None = None
    source_labels: tuple[str, ...] = ()
    target_label: str | None = None
    region_hint: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    preserve_labels: tuple[str, ...] = ()
    forbidden_labels: tuple[str, ...] = ()
    old_prompt: str | None = None
    new_prompt: str | None = None
    prompt_diff: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EditIntent":
        """Build an intent from a parser/rule-engine mapping."""

        if not isinstance(payload, Mapping):
            raise IntentValidationError("EditIntent payload must be a mapping.")
        if "dataset" in payload:
            raise IntentValidationError(
                "Use reference_profile instead of dataset in EditIntent; "
                "reference_profile means the training mask profile used for editing, "
                "not benchmark image provenance."
            )

        primitive = _required_string(payload, "primitive")
        strength = payload.get("strength", payload.get("bucket", "moderate"))
        if not isinstance(strength, str) or strength not in VALID_STRENGTHS:
            raise IntentValidationError(
                f"Invalid strength {strength!r}; expected one of {sorted(VALID_STRENGTHS)}."
            )

        target_change_fraction = payload.get("target_change_fraction")
        if target_change_fraction is not None:
            if not isinstance(target_change_fraction, (int, float)):
                raise IntentValidationError("target_change_fraction must be numeric.")
            target_change_fraction = float(target_change_fraction)
            if not 0.0 <= target_change_fraction <= 1.0:
                raise IntentValidationError("target_change_fraction must be in [0, 1].")

        target_label = payload.get("target_label")
        if target_label is not None and not isinstance(target_label, str):
            raise IntentValidationError("target_label must be a string when provided.")

        seed = payload.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise IntentValidationError("seed must be an integer when provided.")

        return cls(
            primitive=primitive,
            strength=strength,
            reference_profile=_optional_string(
                payload.get("reference_profile", payload.get("mask_profile")),
                "reference_profile",
            ),
            organ=_optional_string(payload.get("organ"), "organ"),
            cancer_type=_optional_string(payload.get("cancer_type"), "cancer_type"),
            site=_optional_string(payload.get("site"), "site"),
            diagnosis=_optional_string(payload.get("diagnosis"), "diagnosis"),
            target_change_fraction=target_change_fraction,
            source_labels=_string_tuple(payload.get("source_labels", ()), "source_labels"),
            target_label=target_label,
            region_hint=_dict_value(payload.get("region_hint", {}), "region_hint"),
            parameters=_dict_value(payload.get("parameters", {}), "parameters"),
            preserve_labels=_string_tuple(
                payload.get("preserve_labels", ()), "preserve_labels"
            ),
            forbidden_labels=_string_tuple(
                payload.get("forbidden_labels", ()), "forbidden_labels"
            ),
            old_prompt=_optional_string(payload.get("old_prompt"), "old_prompt"),
            new_prompt=_optional_string(payload.get("new_prompt"), "new_prompt"),
            prompt_diff=_dict_value(payload.get("prompt_diff", {}), "prompt_diff"),
            seed=seed,
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-safe representation for ops logs and metadata."""

        metadata = asdict(self)
        metadata["source_labels"] = list(self.source_labels)
        metadata["preserve_labels"] = list(self.preserve_labels)
        metadata["forbidden_labels"] = list(self.forbidden_labels)
        return metadata


def resolve_reference_profile(intent: EditIntent) -> str:
    """Resolve the training mask profile an edit should use."""

    if intent.reference_profile:
        return intent.reference_profile

    for domain_value in (intent.organ, intent.cancer_type, intent.diagnosis):
        if not domain_value:
            continue
        normalized = domain_value.strip().lower().replace("-", " ").replace("_", " ")
        direct_match = REFERENCE_PROFILE_BY_DOMAIN.get(normalized)
        if direct_match:
            return direct_match
        for key, profile in REFERENCE_PROFILE_BY_DOMAIN.items():
            if key.replace("_", " ") in normalized:
                return profile

    raise IntentValidationError(
        "reference_profile could not be resolved from reference_profile, organ, "
        "cancer_type, or diagnosis."
    )


def validate_intent_against_recipe(
    intent: EditIntent, recipe: Mapping[str, Any]
) -> None:
    """Validate an intent against a loaded recipe schema."""

    primitives = {
        primitive["name"]: primitive
        for primitive in recipe.get("primitives", [])
        if isinstance(primitive, Mapping) and isinstance(primitive.get("name"), str)
    }
    if intent.primitive not in primitives:
        raise IntentValidationError(f"Unknown primitive: {intent.primitive}")

    primitive = primitives[intent.primitive]
    if not _primitive_supports_strength(primitive, intent.strength):
        raise IntentValidationError(
            f"Primitive {intent.primitive} does not support strength {intent.strength}."
        )

    label_space = recipe.get("label_space", {})
    tissue_labels = set(label_space.get("tissue", []))
    for context, labels in (
        ("source_labels", intent.source_labels),
        ("preserve_labels", intent.preserve_labels),
        ("forbidden_labels", intent.forbidden_labels),
    ):
        for label in labels:
            if label not in tissue_labels:
                raise IntentValidationError(f"{context} contains unknown label: {label}")

    if intent.target_label is not None and intent.target_label not in tissue_labels:
        raise IntentValidationError(
            f"target_label contains unknown label: {intent.target_label}"
        )


def _primitive_supports_strength(primitive: Mapping[str, Any], strength: str) -> bool:
    parameter_ranges = primitive.get("parameter_ranges", {})
    if not isinstance(parameter_ranges, Mapping):
        return False
    return _mapping_contains_key(parameter_ranges, strength)


def _mapping_contains_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        if key in value:
            return True
        return any(_mapping_contains_key(nested, key) for nested in value.values())
    return False


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise IntentValidationError(f"{key} is required and must be a non-empty string.")
    return value


def _optional_string(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise IntentValidationError(f"{key} must be a string when provided.")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise IntentValidationError(f"{key} must be a string or list of strings.")

    labels = tuple(value)
    if not all(isinstance(label, str) for label in labels):
        raise IntentValidationError(f"{key} must contain only strings.")
    return labels


def _dict_value(value: Any, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise IntentValidationError(f"{key} must be a mapping when provided.")
    return dict(value)
