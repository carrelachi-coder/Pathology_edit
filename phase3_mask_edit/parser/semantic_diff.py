"""Versioned semantic-diff schema for Phase 3 prompt parsing."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


SEMANTIC_DIFF_SCHEMA_VERSION = "0.1"

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
    },
    "necrosis_change": {
        "action": "none",
        "extent": "focal",
    },
    "stroma_change": {
        "density": "none",
        "degree": "moderate",
    },
}

VALID_VALUES: dict[str, frozenset[str]] = {
    "tumor_change.growth": frozenset({"none", "increase", "decrease"}),
    "tumor_change.degree": frozenset({"mild", "moderate", "significant"}),
    "tumor_change.grade_change": frozenset({"none", "upgrade", "downgrade"}),
    "lymphocyte_change.infiltration": frozenset({"none", "increase", "decrease"}),
    "lymphocyte_change.degree": frozenset({"mild", "moderate", "significant"}),
    "necrosis_change.action": frozenset(
        {"none", "add", "increase", "decrease", "remove"}
    ),
    "necrosis_change.extent": frozenset({"focal", "moderate", "extensive"}),
    "stroma_change.density": frozenset({"none", "increase", "decrease"}),
    "stroma_change.degree": frozenset({"mild", "moderate", "significant"}),
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

    return _json_safe(result)


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
            raise SemanticDiffValidationError("model response JSON was invalid.") from exc

    if not isinstance(payload, dict):
        raise SemanticDiffValidationError("model response JSON must be an object.")
    return payload


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _json_safe(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
