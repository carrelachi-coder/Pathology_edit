"""Run an API-backed text-only LLM contour proposal agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from phase3_mask_edit.backends.fixture_contour import STATUS_VALIDATED
from phase3_mask_edit.backends.llm_contour import (
    DEFAULT_PROJECTION_MODE,
    PROJECTION_MODE_HARD_V1,
    PROJECTION_MODE_ORGANIC_V2,
)
from phase3_mask_edit.backends.llm_agent import (
    FixtureContourProvider,
    OpenAICompatibleMultimodalContourProvider,
    OpenAICompatibleTextContourProvider,
    execute_llm_contour_agent,
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

    if args.provider == "api-text":
        if not args.api_model:
            parser.error("--api-model is required when --provider api-text is used")
        provider = OpenAICompatibleTextContourProvider(
            model=args.api_model,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            timeout_sec=args.api_timeout_sec,
            temperature=args.api_temperature,
        )
    elif args.provider == "api-multimodal":
        if not args.api_model:
            parser.error("--api-model is required when --provider api-multimodal is used")
        provider = OpenAICompatibleMultimodalContourProvider(
            model=args.api_model,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            timeout_sec=args.api_timeout_sec,
            temperature=args.api_temperature,
            image_detail=args.api_image_detail,
        )
    elif args.provider == "fixture":
        if not args.fixture:
            parser.error("--fixture is required when --provider fixture is used")
        provider = FixtureContourProvider(args.fixture)
    else:  # pragma: no cover - argparse choices guard this.
        parser.error(f"Unsupported provider: {args.provider}")

    schema = MaskProfileSchema.from_reference_profile(args.profile)
    mask = load_id_mask(args.mask)
    result = execute_llm_contour_agent(
        old_mask=mask,
        schema=schema,
        intent=intent,
        primitive_config=primitive_config,
        provider=provider,
        output_dir=args.output,
        max_attempts=args.max_attempts,
        max_regions=args.max_regions,
        max_points_per_region=args.max_points_per_region,
        projection_mode=args.projection_mode,
        organic_seed=args.organic_seed,
    )

    if args.print_summary:
        print(json.dumps(result.to_metadata(), indent=2, ensure_ascii=False))

    return 0 if result.status == STATUS_VALIDATED or args.allow_failed else 1


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

    resolved_source = tuple(
        source_labels or _default_source_labels(primitive_config, operation)
    )
    resolved_target = target_label or _string_or_none(operation.get("target"))
    if not resolved_target and primitive_config.get("name") == "tumor_burden_increase":
        resolved_target = "Tumor"
    if not resolved_target and primitive_config.get("name") in {
        "immune_infiltration_decrease",
        "stroma_decrease",
        "stromal_reduction",
        "tumor_burden_decrease",
    }:
        resolved_target = _first_backfill_label(operation)
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


def _default_source_labels(
    primitive_config: Mapping[str, Any],
    operation: Mapping[str, Any],
) -> list[str]:
    primitive_name = primitive_config.get("name")
    if primitive_name == "tumor_burden_increase":
        return _labels_from_operation(operation.get("target_priority"))
    labels = _labels_from_operation(operation.get("source"))
    if labels:
        return labels
    labels.extend(_labels_from_operation(operation.get("primary_sources")))
    labels.extend(_labels_from_operation(operation.get("secondary_sources")))
    return list(dict.fromkeys(labels))


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _first_backfill_label(operation: Mapping[str, Any]) -> str | None:
    priority = operation.get("backfill_priority", ())
    if isinstance(priority, list):
        for label in priority:
            if isinstance(label, str):
                return label
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an API LLM contour proposal through Phase 3."
    )
    parser.add_argument("--profile", required=True, help="Reference profile, e.g. BCSS.")
    parser.add_argument("--primitive", required=True, help="Primitive name.")
    parser.add_argument("--strength", default="mild")
    parser.add_argument("--mask", required=True, type=Path, help="Source id mask PNG.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("phase3_mask_edit/recipes/generic.yaml"),
    )
    parser.add_argument(
        "--provider",
        choices=("api-text", "api-multimodal", "fixture"),
        default="api-text",
    )
    parser.add_argument("--fixture", type=Path, help="Fixture JSON for provider=fixture.")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-model", help="Model name, e.g. gpt-4o.")
    parser.add_argument("--api-timeout-sec", type=float, default=180.0)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--api-image-detail", choices=("low", "high", "auto"), default="high")
    parser.add_argument("--source-label", action="append")
    parser.add_argument("--target-label")
    parser.add_argument("--preserve-label", action="append", default=[])
    parser.add_argument("--forbidden-label", action="append", default=[])
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--max-points-per-region", type=int, default=64)
    parser.add_argument(
        "--projection-mode",
        choices=(PROJECTION_MODE_HARD_V1, PROJECTION_MODE_ORGANIC_V2),
        default=DEFAULT_PROJECTION_MODE,
    )
    parser.add_argument("--organic-seed", type=int, default=0)
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
