"""Run a fixture LLM-contour proposal against one Phase 3 mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from phase3_mask_edit.backends.fixture_contour import (
    STATUS_VALIDATED,
    execute_fixture_contour_backend,
)
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import load_id_mask


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    recipe = load_recipe(args.recipe)
    primitive_config = _primitive_config(recipe, args.primitive)
    intent = _build_intent(
        primitive_config,
        profile=args.profile,
        strength=args.strength,
        source_labels=args.source_label,
        target_label=args.target_label,
        preserve_labels=args.preserve_label,
        forbidden_labels=args.forbidden_label,
    )

    schema = MaskProfileSchema.from_reference_profile(args.profile)
    mask = load_id_mask(args.mask)
    result = execute_fixture_contour_backend(
        old_mask=mask,
        fixture_path=args.fixture,
        schema=schema,
        intent=intent,
        primitive_config=primitive_config,
        output_dir=args.output,
        max_regions=args.max_regions,
        max_points_per_region=args.max_points_per_region,
    )

    if args.print_summary:
        print(json.dumps(result.to_metadata(), indent=2, ensure_ascii=False))

    return 0 if result.status == STATUS_VALIDATED or args.allow_failed_validation else 1


def _build_intent(
    primitive_config: Mapping[str, Any],
    *,
    profile: str,
    strength: str,
    source_labels: list[str] | None,
    target_label: str | None,
    preserve_labels: list[str],
    forbidden_labels: list[str],
) -> EditIntent:
    operation = primitive_config.get("mask_operation", {})
    if not isinstance(operation, Mapping):
        operation = {}

    resolved_source = tuple(source_labels or _labels_from_operation(operation.get("source")))
    resolved_target = target_label or _string_or_none(operation.get("target"))
    if not resolved_source:
        raise ValueError(
            f"Primitive {primitive_config.get('name')} needs --source-label "
            "because mask_operation.source is not a label."
        )
    if not resolved_target:
        raise ValueError(
            f"Primitive {primitive_config.get('name')} needs --target-label "
            "because mask_operation.target is not a label."
        )

    return EditIntent(
        primitive=str(primitive_config["name"]),
        strength=strength,
        reference_profile=profile,
        source_labels=resolved_source,
        target_label=resolved_target,
        preserve_labels=tuple(preserve_labels),
        forbidden_labels=tuple(forbidden_labels),
    )


def _primitive_config(recipe: Mapping[str, Any], primitive_name: str) -> Mapping[str, Any]:
    for primitive in recipe.get("primitives", []):
        if isinstance(primitive, Mapping) and primitive.get("name") == primitive_name:
            return primitive
    raise ValueError(f"Unknown primitive: {primitive_name}")


def _labels_from_operation(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return []


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a saved llm_contour_proposal JSON fixture through Phase 3."
    )
    parser.add_argument("--profile", required=True, help="Reference profile, e.g. BCSS.")
    parser.add_argument("--primitive", required=True, help="Primitive name.")
    parser.add_argument("--strength", default="mild")
    parser.add_argument("--mask", required=True, type=Path, help="Source id mask PNG.")
    parser.add_argument("--fixture", required=True, type=Path, help="Contour JSON fixture.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("phase3_mask_edit/recipes/generic.yaml"),
    )
    parser.add_argument("--source-label", action="append")
    parser.add_argument("--target-label")
    parser.add_argument("--preserve-label", action="append", default=[])
    parser.add_argument("--forbidden-label", action="append", default=[])
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--max-points-per-region", type=int, default=64)
    parser.add_argument("--allow-failed-validation", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
