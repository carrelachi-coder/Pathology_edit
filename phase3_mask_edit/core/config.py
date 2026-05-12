"""Recipe loading and Stage 1 schema validation for Phase 3 mask edits."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from phase3_mask_edit.specialized.catalog import specialized_primitives_for


FROZEN_PRIMITIVE_FIELDS = frozenset(
    {
        "name",
        "version",
        "execution_strategy",
        "pathology_meaning",
        "required_tissue_labels",
        "required_context",
        "optional_tissue_labels",
        "mask_operation",
        "spatial_pattern",
        "parameter_ranges",
        "probnet_bias",
        "validation_rules",
        "expected_failure_cases",
        "overlap_guard",
    }
)

XLARGE_DEID_ALLOWED_PRIMITIVES = frozenset(
    {
        "tumor_burden_increase",
        "tumor_burden_decrease",
        "necrosis_appearance",
        "necrosis_resolution",
        "immune_infiltration_decrease",
        "stromal_desmoplasia",
        "stroma_decrease",
        "stromal_reduction",
    }
)

FUTURE_OVERLAP_GUARD_REFERENCES = frozenset(
    {
        "necrosis_expansion",
        "peritumoral_inflammation",
    }
)

TISSUE_LABEL_KEYS = frozenset(
    {
        "source",
        "target",
        "target_priority",
        "backfill_priority",
        "preserve_source_subtype",
        "forbid_targets",
        "primary_sources",
        "secondary_sources",
        "forbid_sources",
    }
)


class RecipeValidationError(ValueError):
    """Raised when a Phase 3 recipe violates the frozen Stage 1 schema."""


def load_recipe(path: str | Path) -> dict[str, Any]:
    """Load a YAML recipe and validate it against the Stage 1 schema."""

    recipe_path = Path(path)
    with recipe_path.open("r", encoding="utf-8") as stream:
        recipe = yaml.safe_load(stream)

    recipe = expand_recipe(recipe, base_path=recipe_path)
    recipe = normalize_recipe_execution_strategy(recipe)
    validate_recipe_schema(recipe)
    return recipe


def normalize_recipe_execution_strategy(recipe: dict[str, Any]) -> dict[str, Any]:
    """Fill default execution strategies for older generic recipe entries."""

    if not isinstance(recipe, dict):
        return recipe
    primitives = recipe.get("primitives")
    if not isinstance(primitives, list):
        return recipe
    normalized = deepcopy(recipe)
    for primitive in normalized.get("primitives", []):
        if not isinstance(primitive, dict):
            continue
        primitive.setdefault(
            "execution_strategy",
            _default_execution_strategy(primitive),
        )
    return normalized


def _default_execution_strategy(primitive: dict[str, Any]) -> str:
    mask_operation = primitive.get("mask_operation", {})
    if isinstance(mask_operation, dict) and mask_operation.get("type") == "fine_label_transition":
        return "id_transition"
    return "geometric_organic"


def expand_recipe(recipe: dict[str, Any], *, base_path: Path | None = None) -> dict[str, Any]:
    """Expand dataset wrapper recipes into executable recipes."""

    if not isinstance(recipe, dict) or "primitives" in recipe:
        return recipe
    if not recipe.get("include_generic"):
        return recipe

    dataset = recipe.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        return recipe

    generic_path = (
        base_path.parent / "generic.yaml"
        if base_path is not None
        else Path("phase3_mask_edit/recipes/generic.yaml")
    )
    with generic_path.open("r", encoding="utf-8") as stream:
        expanded = yaml.safe_load(stream)

    expanded = deepcopy(expanded)
    expanded["dataset"] = dataset
    expanded["primitive_set"] = (
        f"{expanded.get('primitive_set', 'phase3')}_{dataset.lower()}_specialized"
    )
    expanded.setdefault("metadata", {})
    expanded["metadata"]["dataset_recipe"] = dataset

    strategy_names = recipe.get("strategies", [])
    if strategy_names is None:
        strategy_names = []
    if not isinstance(strategy_names, list):
        raise RecipeValidationError("dataset recipe strategies must be a list.")

    specialized = specialized_primitives_for(dataset)
    if strategy_names:
        wanted = set(strategy_names)
        specialized = [
            primitive for primitive in specialized if primitive["name"] in wanted
        ]
        missing = wanted - {primitive["name"] for primitive in specialized}
        if missing:
            raise RecipeValidationError(
                f"Unknown specialized strategies for {dataset}: "
                f"{', '.join(sorted(missing))}"
            )

    expanded["primitives"].extend(specialized)
    expanded["specialized_strategies"] = [
        primitive["name"] for primitive in specialized
    ]
    return expanded


def validate_recipe_schema(recipe: dict[str, Any]) -> None:
    """Validate a parsed Phase 3 recipe.

    The validator intentionally checks only the Stage 1 contract. Primitive
    algorithms will add stronger spatial checks once mask operations exist.
    """

    if not isinstance(recipe, dict):
        raise RecipeValidationError("Recipe root must be a mapping.")

    label_space = _require_mapping(recipe, "label_space", "recipe")
    tissue_labels = set(_require_list(label_space, "tissue", "label_space"))
    cell_labels = set(_require_list(label_space, "cells", "label_space"))
    max_changed_area = _read_max_changed_area(recipe)

    primitives = _require_list(recipe, "primitives", "recipe")
    primitive_names = _validate_primitives(
        primitives=primitives,
        tissue_labels=tissue_labels,
        cell_labels=cell_labels,
        max_changed_area=max_changed_area,
    )

    _validate_overlap_guards(primitives, primitive_names, recipe)
    _validate_composite_recipes(recipe, primitive_names)


def _validate_primitives(
    *,
    primitives: list[Any],
    tissue_labels: set[str],
    cell_labels: set[str],
    max_changed_area: float,
) -> set[str]:
    primitive_names: set[str] = set()

    for index, primitive in enumerate(primitives):
        if not isinstance(primitive, dict):
            raise RecipeValidationError(f"Primitive #{index} must be a mapping.")

        name = primitive.get("name", f"#{index}")
        missing = sorted(FROZEN_PRIMITIVE_FIELDS - set(primitive))
        if missing:
            raise RecipeValidationError(
                f"Primitive {name} missing required field(s): {', '.join(missing)}"
            )

        if not isinstance(primitive["name"], str) or not primitive["name"]:
            raise RecipeValidationError(f"Primitive #{index} has invalid name.")
        if primitive["name"] in primitive_names:
            raise RecipeValidationError(f"Duplicate primitive name: {primitive['name']}")
        primitive_names.add(primitive["name"])

        _validate_label_list(
            primitive["required_tissue_labels"],
            tissue_labels,
            f"{primitive['name']}.required_tissue_labels",
        )
        _validate_label_list(
            primitive["optional_tissue_labels"],
            tissue_labels,
            f"{primitive['name']}.optional_tissue_labels",
        )
        _validate_tissue_labels_in_operation(
            primitive["mask_operation"],
            tissue_labels,
            f"{primitive['name']}.mask_operation",
        )
        _validate_probnet_bias(
            primitive["probnet_bias"],
            tissue_labels=tissue_labels,
            cell_labels=cell_labels,
            context=f"{primitive['name']}.probnet_bias",
        )
        _validate_parameter_ranges(
            primitive["parameter_ranges"],
            primitive_name=primitive["name"],
            max_changed_area=max_changed_area,
        )

    return primitive_names


def _validate_label_list(labels: Any, legal_labels: set[str], context: str) -> None:
    if labels is None:
        return
    if not isinstance(labels, list):
        raise RecipeValidationError(f"{context} must be a list.")

    for label in labels:
        if label not in legal_labels:
            raise RecipeValidationError(f"{context} contains unknown label: {label}")


def _validate_tissue_labels_in_operation(
    value: Any, legal_tissue_labels: set[str], context: str
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            nested_context = f"{context}.{key}"
            if key in TISSUE_LABEL_KEYS:
                _validate_tissue_label_value(nested, legal_tissue_labels, nested_context)
            elif isinstance(nested, (dict, list)):
                _validate_tissue_labels_in_operation(
                    nested, legal_tissue_labels, nested_context
                )
    elif isinstance(value, list):
        for item_index, item in enumerate(value):
            _validate_tissue_labels_in_operation(
                item, legal_tissue_labels, f"{context}[{item_index}]"
            )


def _validate_tissue_label_value(
    value: Any, legal_tissue_labels: set[str], context: str
) -> None:
    if isinstance(value, str):
        if value not in legal_tissue_labels and value not in {"nearest_boundary_tumor_label"}:
            raise RecipeValidationError(f"{context} contains unknown tissue label: {value}")
        return

    if isinstance(value, list):
        _validate_label_list(value, legal_tissue_labels, context)


def _validate_probnet_bias(
    value: Any,
    *,
    tissue_labels: set[str],
    cell_labels: set[str],
    context: str,
) -> None:
    if not isinstance(value, dict):
        raise RecipeValidationError(f"{context} must be a mapping.")

    legal_top_level = cell_labels | tissue_labels | {"default", "treatment_response_recipe"}
    for key in value:
        if key not in legal_top_level:
            raise RecipeValidationError(f"{context} contains unknown label: {key}")


def _validate_parameter_ranges(
    parameter_ranges: Any, *, primitive_name: str, max_changed_area: float
) -> None:
    if not isinstance(parameter_ranges, dict):
        raise RecipeValidationError(f"{primitive_name}.parameter_ranges must be a mapping.")

    _walk_parameter_ranges(
        parameter_ranges,
        primitive_name=primitive_name,
        max_changed_area=max_changed_area,
        path=f"{primitive_name}.parameter_ranges",
    )


def _walk_parameter_ranges(
    value: Any, *, primitive_name: str, max_changed_area: float, path: str
) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "xlarge_deid" and primitive_name not in XLARGE_DEID_ALLOWED_PRIMITIVES:
                raise RecipeValidationError(
                    f"{primitive_name} cannot define xlarge_deid parameter bucket."
                )
            _walk_parameter_ranges(
                nested,
                primitive_name=primitive_name,
                max_changed_area=max_changed_area,
                path=f"{path}.{key}",
            )
        return

    if isinstance(value, list):
        if len(value) != 2 or not all(isinstance(item, (int, float)) for item in value):
            raise RecipeValidationError(f"{path} must be a numeric [lower, upper] interval.")
        lower, upper = float(value[0]), float(value[1])
        if not 0 <= lower < upper <= max_changed_area:
            raise RecipeValidationError(
                f"{path} must satisfy 0 <= lower < upper <= {max_changed_area}."
            )


def _validate_composite_recipes(
    recipe: dict[str, Any], primitive_names: set[str]
) -> None:
    composite_recipes = recipe.get("composite_recipes", [])
    if not isinstance(composite_recipes, list):
        raise RecipeValidationError("recipe.composite_recipes must be a list.")

    for index, composite in enumerate(composite_recipes):
        if not isinstance(composite, dict):
            raise RecipeValidationError(f"Composite recipe #{index} must be a mapping.")

        name = composite.get("name", f"#{index}")
        references = composite.get("primitives")
        if not isinstance(references, list) or not references:
            raise RecipeValidationError(f"Composite recipe {name} needs primitives list.")

        for primitive_name in references:
            if primitive_name not in primitive_names:
                raise RecipeValidationError(
                    f"Composite recipe {name} references unknown primitive: {primitive_name}"
                )


def _validate_overlap_guards(
    primitives: list[Any], primitive_names: set[str], recipe: dict[str, Any]
) -> None:
    future_names = _roadmap_candidate_names(recipe) | FUTURE_OVERLAP_GUARD_REFERENCES
    legal_references = primitive_names | future_names

    for primitive in primitives:
        guard_texts = _flatten_guard_text(primitive["overlap_guard"])
        for guard_text in guard_texts:
            if not guard_text.startswith("use_"):
                continue
            if not any(reference in guard_text for reference in legal_references):
                raise RecipeValidationError(
                    f"{primitive['name']}.overlap_guard references unknown primitive: "
                    f"{guard_text.removeprefix('use_')}"
                )


def _flatten_guard_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_flatten_guard_text(item))
        return texts
    if isinstance(value, dict):
        texts = []
        for key, nested in value.items():
            texts.extend(_flatten_guard_text(key))
            texts.extend(_flatten_guard_text(nested))
        return texts
    return []


def _roadmap_candidate_names(recipe: dict[str, Any]) -> set[str]:
    roadmap = recipe.get("roadmap")
    if not isinstance(roadmap, dict):
        return set()

    candidate_names: set[str] = set()
    for value in roadmap.values():
        if not isinstance(value, list):
            continue
        for candidate in value:
            if isinstance(candidate, dict) and isinstance(candidate.get("name"), str):
                candidate_names.add(candidate["name"])
    return candidate_names


def _read_max_changed_area(recipe: dict[str, Any]) -> float:
    defaults = _require_mapping(recipe, "defaults", "recipe")
    validation = _require_mapping(defaults, "validation", "defaults")
    max_changed_area = validation.get("max_changed_area_fraction")
    if not isinstance(max_changed_area, (int, float)):
        raise RecipeValidationError(
            "defaults.validation.max_changed_area_fraction must be numeric."
        )
    return float(max_changed_area)


def _require_mapping(value: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise RecipeValidationError(f"{context}.{key} must be a mapping.")
    return nested


def _require_list(value: dict[str, Any], key: str, context: str) -> list[Any]:
    nested = value.get(key)
    if not isinstance(nested, list):
        raise RecipeValidationError(f"{context}.{key} must be a list.")
    return nested
