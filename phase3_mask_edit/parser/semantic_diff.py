"""Versioned semantic-diff schema for Phase 3 prompt parsing."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


SEMANTIC_DIFF_SCHEMA_VERSION = "0.2"
LEGACY_SEMANTIC_DIFF_SCHEMA_VERSIONS = frozenset({"0.1"})

IMMUNE_LOCATIONS = (
    "unspecified",
    "stromal",
    "intratumoral",
    "peritumoral",
)

TRANSITION_STATES = (
    "none",
    "benign_epithelium",
    "stromal_tissue",
    "gleason_pattern_3",
    "gleason_pattern_4",
    "gleason_pattern_5",
    "normal_gland",
    "adenomatous_gland",
    "moderately_differentiated_carcinoma",
    "poorly_differentiated_carcinoma",
)

FINE_TRANSITION_PAIRS = (
    ("none", "none"),
    ("benign_epithelium", "gleason_pattern_3"),
    ("benign_epithelium", "stromal_tissue"),
    ("gleason_pattern_3", "gleason_pattern_4"),
    ("gleason_pattern_4", "gleason_pattern_5"),
    ("gleason_pattern_4", "gleason_pattern_3"),
    ("normal_gland", "adenomatous_gland"),
    (
        "adenomatous_gland",
        "moderately_differentiated_carcinoma",
    ),
    (
        "moderately_differentiated_carcinoma",
        "poorly_differentiated_carcinoma",
    ),
    (
        "poorly_differentiated_carcinoma",
        "moderately_differentiated_carcinoma",
    ),
)

DEFAULT_SEMANTIC_DIFF: dict[str, Any] = {
    "schema_version": SEMANTIC_DIFF_SCHEMA_VERSION,
    "tumor_change": {
        "growth": "none",
        "degree": "mild",
        "grade_change": "none",
    },
    "lymphocyte_change": {
        "infiltration": "none",
        "degree": "mild",
        "location": "unspecified",
    },
    "necrosis_change": {
        "action": "none",
        "extent": "focal",
    },
    "stroma_change": {
        "density": "none",
        "degree": "moderate",
    },
    "transition_change": {
        "source_state": "none",
        "target_state": "none",
        "degree": "moderate",
    },
}

VALID_VALUES: dict[str, frozenset[str]] = {
    "tumor_change.growth": frozenset({"none", "increase", "decrease"}),
    "tumor_change.degree": frozenset({"mild", "moderate", "significant"}),
    "tumor_change.grade_change": frozenset({"none", "upgrade", "downgrade"}),
    "lymphocyte_change.infiltration": frozenset({"none", "increase", "decrease"}),
    "lymphocyte_change.degree": frozenset({"mild", "moderate", "significant"}),
    "lymphocyte_change.location": frozenset(IMMUNE_LOCATIONS),
    "necrosis_change.action": frozenset(
        {"none", "add", "increase", "decrease", "remove"}
    ),
    "necrosis_change.extent": frozenset({"focal", "moderate", "extensive"}),
    "stroma_change.density": frozenset({"none", "increase", "decrease"}),
    "stroma_change.degree": frozenset({"mild", "moderate", "significant"}),
    "transition_change.source_state": frozenset(TRANSITION_STATES),
    "transition_change.target_state": frozenset(TRANSITION_STATES),
    "transition_change.degree": frozenset({"mild", "moderate", "significant"}),
}


class SemanticDiffValidationError(ValueError):
    """Raised when a semantic-diff payload violates the Phase 3 schema."""


def normalize_semantic_diff(
    payload: Mapping[str, Any], *, fill_missing: bool = False
) -> dict[str, Any]:
    """Return a validated semantic diff, optionally filling parser omissions.

    Rule mapping should call this with ``fill_missing=False`` so incomplete
    planner inputs fail loudly. Parser adapters may use ``fill_missing=True``
    to canonicalize imperfect model JSON before passing it downstream.
    """

    if not isinstance(payload, Mapping):
        raise SemanticDiffValidationError("semantic_diff must be a mapping.")

    payload = _upgrade_legacy_payload(payload)

    if fill_missing:
        normalized = deepcopy(DEFAULT_SEMANTIC_DIFF)
        _deep_update(normalized, payload)
    else:
        normalized = dict(payload)

    return validate_semantic_diff(normalized)


def validate_semantic_diff(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and JSON-normalize a semantic-diff payload."""

    if not isinstance(payload, Mapping):
        raise SemanticDiffValidationError("semantic_diff must be a mapping.")

    payload = _upgrade_legacy_payload(payload)
    version = payload.get("schema_version")
    if version != SEMANTIC_DIFF_SCHEMA_VERSION:
        raise SemanticDiffValidationError(
            "semantic_diff.schema_version must be "
            f"{SEMANTIC_DIFF_SCHEMA_VERSION!r}, got {version!r}."
        )

    result: dict[str, Any] = {"schema_version": version}
    for section, defaults in DEFAULT_SEMANTIC_DIFF.items():
        if section == "schema_version":
            continue
        value = payload.get(section)
        if not isinstance(value, Mapping):
            raise SemanticDiffValidationError(f"{section} must be a mapping.")
        result[section] = {}
        for field in defaults:
            field_value = value.get(field)
            full_key = f"{section}.{field}"
            allowed = VALID_VALUES[full_key]
            if field_value not in allowed:
                raise SemanticDiffValidationError(
                    f"{full_key} must be one of {sorted(allowed)}, got "
                    f"{field_value!r}."
                )
            result[section][field] = field_value

        for extra_key, extra_value in value.items():
            if extra_key not in defaults:
                result[section][extra_key] = extra_value

    for extra_key, extra_value in payload.items():
        if extra_key not in DEFAULT_SEMANTIC_DIFF:
            result[extra_key] = extra_value

    transition_pair = (
        result["transition_change"]["source_state"],
        result["transition_change"]["target_state"],
    )
    if transition_pair not in FINE_TRANSITION_PAIRS:
        raise SemanticDiffValidationError(
            "transition_change source/target must be one supported exact pair, got "
            f"{transition_pair[0]!r} -> {transition_pair[1]!r}."
        )

    return _json_safe(result)


def semantic_diff_json_schema() -> dict[str, Any]:
    """Return the strict JSON Schema used by API parser adapters."""

    def enum_field(values: frozenset[str]) -> dict[str, Any]:
        return {"type": "string", "enum": sorted(values)}

    properties: dict[str, Any] = {
        "schema_version": {
            "type": "string",
            "enum": [SEMANTIC_DIFF_SCHEMA_VERSION],
        }
    }
    for section, defaults in DEFAULT_SEMANTIC_DIFF.items():
        if section == "schema_version":
            continue
        section_properties = {
            field: enum_field(VALID_VALUES[f"{section}.{field}"]) for field in defaults
        }
        section_schema: dict[str, Any] = {
            "type": "object",
            "properties": section_properties,
            "required": list(defaults),
            "additionalProperties": False,
        }
        if section == "transition_change":
            section_schema["anyOf"] = [
                {
                    "type": "object",
                    "properties": {
                        "source_state": {"type": "string", "enum": [source]},
                        "target_state": {"type": "string", "enum": [target]},
                        "degree": enum_field(VALID_VALUES["transition_change.degree"]),
                    },
                    "required": list(defaults),
                    "additionalProperties": False,
                }
                for source, target in FINE_TRANSITION_PAIRS
            ]
        properties[section] = section_schema
    return {
        "type": "object",
        "properties": properties,
        "required": list(DEFAULT_SEMANTIC_DIFF),
        "additionalProperties": False,
    }


def semantic_diff_response_format() -> dict[str, Any]:
    """Return a strict Chat Completions response_format payload."""

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "phase3_semantic_diff_v0_2",
            "strict": True,
            "schema": semantic_diff_json_schema(),
        },
    }


def load_semantic_diff(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a semantic-diff JSON file."""

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return validate_semantic_diff(payload)


def save_semantic_diff(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and save semantic-diff JSON."""

    normalized = validate_semantic_diff(payload)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(normalized, stream, indent=2, ensure_ascii=False)
    return output_path


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response string."""

    if not isinstance(text, str):
        raise SemanticDiffValidationError("model response must be a string.")

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fence(stripped)

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise SemanticDiffValidationError("model response did not contain JSON.")
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SemanticDiffValidationError(
                "model response JSON was invalid."
            ) from exc

    if not isinstance(payload, dict):
        raise SemanticDiffValidationError("model response JSON must be an object.")
    return payload


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _upgrade_legacy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    upgraded = deepcopy(dict(payload))
    version = upgraded.get("schema_version")
    if version not in LEGACY_SEMANTIC_DIFF_SCHEMA_VERSIONS:
        return upgraded
    upgraded["schema_version"] = SEMANTIC_DIFF_SCHEMA_VERSION
    lymphocyte_change = upgraded.get("lymphocyte_change")
    if isinstance(lymphocyte_change, Mapping):
        upgraded["lymphocyte_change"] = dict(lymphocyte_change)
        upgraded["lymphocyte_change"].setdefault("location", "unspecified")
    upgraded.setdefault(
        "transition_change",
        deepcopy(DEFAULT_SEMANTIC_DIFF["transition_change"]),
    )
    return upgraded


def _json_safe(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
