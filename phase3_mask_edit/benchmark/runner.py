"""Batch runner for mask-edit semantic benchmark intents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from phase3_mask_edit.backends.llm_agent import (
    FixtureContourProvider,
    OpenAICompatibleMultimodalContourProvider,
    OpenAICompatibleTextContourProvider,
    STATUS_VALIDATED,
    execute_llm_contour_agent,
)
from phase3_mask_edit.backends.llm_contour import PROJECTION_MODE_ORGANIC_V2
from phase3_mask_edit.benchmark.intents import inject_region_hint, primitive_config_by_name, source_target_labels_for_primitive
from phase3_mask_edit.benchmark.metrics import evaluate_mask_edit, row_for_eval
from phase3_mask_edit.benchmark.models import BenchmarkIntent, BenchmarkPrompt
from phase3_mask_edit.benchmark.prompts import semantic_diff_for_intent
from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import load_id_mask, save_change_region, save_id_mask, save_metadata, save_rgb_mask
from phase3_mask_edit.parser.api_parser import ApiParserConfig, parse_prompts_with_api
from phase3_mask_edit.parser.instruction_parser import InstructionParserConfig, parse_instruction
from phase3_mask_edit.parser.semantic_diff import save_semantic_diff
from phase3_mask_edit.rules.semantic_to_intent import plan_edit_intents


PROMPT_MODE = "prompt"
INSTRUCTION_MODE = "instruction"
GT_MODE = "gt"


def run_benchmark_sample(
    intent: BenchmarkIntent,
    prompt: BenchmarkPrompt | None = None,
    *,
    mode: str,
    output_dir: str | Path,
    prompt_parser: str = "api",
    instruction_parser: str = "api",
    parser_api_base_url: str = "https://api.openai.com/v1",
    parser_api_key_env: str = "OPENAI_API_KEY",
    parser_model: str = "",
    contour_provider: str = "fixture",
    contour_api_base_url: str = "https://api.openai.com/v1",
    contour_api_key_env: str = "OPENAI_API_KEY",
    contour_model: str = "",
    contour_fixture: str | Path | None = None,
    api_image_detail: str = "high",
    max_attempts: int = 4,
    max_regions: int = 8,
    max_points_per_region: int = 64,
) -> dict[str, Any]:
    sample_dir = Path(output_dir) / intent.sample_id / mode
    sample_dir.mkdir(parents=True, exist_ok=True)
    source_mask = load_id_mask(intent.mask_path)
    schema = MaskProfileSchema.from_reference_profile(intent.profile)
    recipe = load_recipe(default_recipe_path_for_profile(intent.profile))
    try:
        if mode == GT_MODE:
            semantic_diff = _gt_record_for_intent(intent)
            save_metadata(semantic_diff, sample_dir / "gt_intent.json")
            planned_intent = _select_or_override_planned_intent(
                intent,
                _intent_from_gt(intent),
                recipe,
                schema,
                allow_gt_override=True,
            )
            save_metadata(
                {
                    "mode": GT_MODE,
                    "planner_bypassed": True,
                    "intent": planned_intent.to_metadata(),
                },
                sample_dir / "planning_summary.json",
            )
        else:
            if prompt is None:
                raise RuntimeError(f"prompt is required for mode={mode}")
            semantic_diff = _resolve_semantic_diff(
                intent,
                prompt,
                mode=mode,
                prompt_parser=prompt_parser,
                instruction_parser=instruction_parser,
                api_base_url=parser_api_base_url,
                api_key_env=parser_api_key_env,
                parser_model=parser_model,
                output_dir=sample_dir,
            )
            save_semantic_diff(semantic_diff, sample_dir / "semantic_diff.json")
            plan = plan_edit_intents(
                semantic_diff,
                reference_profile=intent.profile,
                old_prompt=prompt.old_prompt,
                new_prompt=prompt.new_prompt if mode == PROMPT_MODE else prompt.instruction,
                old_mask=source_mask,
                recipe=recipe,
            )
            executable = [item for item in getattr(plan, "items", ()) if getattr(item, "intent", None) is not None]
            allow_gt_override = _uses_gt_parser(
                mode=mode,
                prompt_parser=prompt_parser,
                instruction_parser=instruction_parser,
            )
            if executable:
                planned_base = executable[0].intent
            elif allow_gt_override:
                planned_base = _intent_from_gt(intent)
            else:
                raise RuntimeError("planner produced no executable intents")
            planned_intent = _select_or_override_planned_intent(
                intent,
                planned_base,
                recipe,
                schema,
                allow_gt_override=allow_gt_override,
            )
        primitive_config = primitive_config_by_name(recipe, planned_intent.primitive)
        source_labels, target_label = _execution_labels(
            primitive_config,
            schema,
            planned_intent,
            prefer_intent_labels=mode == GT_MODE,
        )
        provider = _build_contour_provider(
            contour_provider,
            api_base_url=contour_api_base_url,
            api_key_env=contour_api_key_env,
            model=contour_model,
            fixture=contour_fixture,
            api_image_detail=api_image_detail,
        )
        result = execute_llm_contour_agent(
            old_mask=source_mask,
            schema=schema,
            intent=planned_intent,
            primitive_config=primitive_config,
            provider=provider,
            output_dir=sample_dir / "phase3_mask_edit",
            allowed_source_labels=source_labels,
            max_attempts=max_attempts,
            max_regions=max_regions,
            max_points_per_region=max_points_per_region,
            projection_mode=PROJECTION_MODE_ORGANIC_V2,
            organic_seed=intent.seed,
        )
        if mode != GT_MODE:
            save_metadata(plan.to_metadata(), sample_dir / "planning_summary.json")
        if result.status != STATUS_VALIDATED or result.edit_result is None:
            raise RuntimeError(result.error or f"contour execution failed: {result.status}")
        target_mask = result.edit_result.target_mask
        change_region = source_mask != target_mask
        save_id_mask(source_mask, sample_dir / "source_mask.png")
        save_id_mask(target_mask, sample_dir / "target_mask.png")
        save_rgb_mask(source_mask, sample_dir / "source_mask_rgb.png")
        save_rgb_mask(target_mask, sample_dir / "target_mask_rgb.png")
        save_change_region(change_region, sample_dir / "change_region.png")
        metrics = evaluate_mask_edit(source_mask, target_mask, intent)
        save_metadata(metrics, sample_dir / "metrics.json")
        return row_for_eval(
            sample_id=intent.sample_id,
            mode=mode,
            status="completed",
            parsed_semantic_diff=semantic_diff,
            planned_primitive=planned_intent.primitive,
            metrics=metrics,
            output_dir=str(sample_dir),
            organ=intent.organ,
            profile=intent.profile,
            primitive=intent.primitive,
            strength=intent.strength,
        )
    except Exception as exc:
        save_metadata({"status": "failed", "error": str(exc)}, sample_dir / "failure.json")
        return row_for_eval(
            sample_id=intent.sample_id,
            mode=mode,
            status="failed",
            parsed_semantic_diff=None,
            planned_primitive=None,
            metrics=None,
            error=str(exc),
            output_dir=str(sample_dir),
            organ=intent.organ,
            profile=intent.profile,
            primitive=intent.primitive,
            strength=intent.strength,
        )


def _resolve_semantic_diff(
    intent: BenchmarkIntent,
    prompt: BenchmarkPrompt,
    *,
    mode: str,
    prompt_parser: str,
    instruction_parser: str,
    api_base_url: str,
    api_key_env: str,
    parser_model: str,
    output_dir: Path,
) -> dict[str, Any]:
    if mode == PROMPT_MODE:
        if prompt_parser == "gt":
            return semantic_diff_for_intent(intent)
        if prompt_parser == "api":
            if not parser_model:
                raise RuntimeError("parser_model is required for prompt_parser=api")
            return parse_prompts_with_api(
                prompt.old_prompt,
                prompt.new_prompt,
                config=ApiParserConfig(
                    model=parser_model,
                    api_base_url=api_base_url,
                    api_key_env=api_key_env,
                    debug_dir=str(output_dir / "api_parser_debug"),
                ),
            )
        raise RuntimeError(f"Unsupported prompt_parser: {prompt_parser}")
    if mode == INSTRUCTION_MODE:
        if instruction_parser == "gt":
            return semantic_diff_for_intent(intent)
        if instruction_parser == "rule-based":
            return parse_instruction(prompt.instruction, mode="rule-based")
        if instruction_parser == "api":
            if not parser_model:
                raise RuntimeError("parser_model is required for instruction_parser=api")
            return parse_instruction(
                prompt.instruction,
                mode="api",
                config=InstructionParserConfig(
                    model=parser_model,
                    api_base_url=api_base_url,
                    api_key_env=api_key_env,
                    debug_dir=str(output_dir / "instruction_parser_debug"),
                ),
            )
        raise RuntimeError(f"Unsupported instruction_parser: {instruction_parser}")
    raise RuntimeError(f"Unsupported benchmark mode: {mode}")


def _select_or_override_planned_intent(
    gt: BenchmarkIntent,
    planned: EditIntent,
    recipe: Mapping[str, Any],
    schema: MaskProfileSchema,
    *,
    allow_gt_override: bool,
) -> EditIntent:
    if allow_gt_override and (planned.primitive != gt.primitive or planned.strength != gt.strength):
        payload = planned.to_metadata()
        payload["primitive"] = gt.primitive
        payload["strength"] = gt.strength
        payload["reference_profile"] = gt.profile
        payload["source_labels"] = list(gt.source_labels)
        payload["target_label"] = gt.target_label
        planned = EditIntent.from_mapping(payload)
    planned = inject_region_hint(planned, gt.region_hint)
    primitive_config = primitive_config_by_name(recipe, planned.primitive)
    source_labels, target_label = source_target_labels_for_primitive(primitive_config, schema)
    if allow_gt_override and gt.primitive != "tumor_burden_increase":
        source_labels = tuple(gt.source_labels) or source_labels
        target_label = gt.target_label or target_label
    payload = planned.to_metadata()
    if source_labels:
        payload["source_labels"] = list(source_labels)
    if target_label:
        payload["target_label"] = target_label
    anchor_labels = _anchor_labels_from_gt(gt)
    if anchor_labels:
        parameters = dict(payload.get("parameters") or {})
        parameters["anchor_labels"] = list(anchor_labels)
        payload["parameters"] = parameters
        region_hint = dict(payload.get("region_hint") or {})
        region_hint.setdefault("anchor_labels", list(anchor_labels))
        payload["region_hint"] = region_hint
    payload["seed"] = gt.seed
    return EditIntent.from_mapping(payload)


def _execution_labels(
    primitive_config: Mapping[str, Any],
    schema: MaskProfileSchema,
    intent: EditIntent,
    *,
    prefer_intent_labels: bool,
) -> tuple[tuple[str, ...], str | None]:
    recipe_source_labels, recipe_target_label = source_target_labels_for_primitive(primitive_config, schema)
    intent_source_labels = tuple(intent.source_labels)
    intent_target_label = intent.target_label
    if primitive_config.get("name") == "tumor_burden_increase":
        return recipe_source_labels or intent_source_labels, recipe_target_label or intent_target_label
    if prefer_intent_labels:
        return intent_source_labels or recipe_source_labels, intent_target_label or recipe_target_label
    return recipe_source_labels or intent_source_labels, recipe_target_label or intent_target_label


def _anchor_labels_from_gt(gt: BenchmarkIntent) -> tuple[str, ...]:
    metadata_labels = gt.metadata.get("anchor_labels") if isinstance(gt.metadata, Mapping) else None
    if isinstance(metadata_labels, (list, tuple)):
        return tuple(str(label) for label in metadata_labels if str(label))
    region_labels = gt.region_hint.get("anchor_labels") if isinstance(gt.region_hint, Mapping) else None
    if isinstance(region_labels, (list, tuple)):
        return tuple(str(label) for label in region_labels if str(label))
    if gt.primitive == "tumor_burden_increase":
        return ("Tumor",)
    return ()


def _uses_gt_parser(*, mode: str, prompt_parser: str, instruction_parser: str) -> bool:
    return (mode == PROMPT_MODE and prompt_parser == "gt") or (
        mode == INSTRUCTION_MODE and instruction_parser == "gt"
    )


def _gt_record_for_intent(gt: BenchmarkIntent) -> dict[str, Any]:
    return {
        "sample_id": gt.sample_id,
        "organ": gt.organ,
        "profile": gt.profile,
        "mask_path": gt.mask_path,
        "image_path": gt.image_path,
        "primitive": gt.primitive,
        "strength": gt.strength,
        "region_hint": gt.region_hint,
        "source_labels": list(gt.source_labels),
        "target_label": gt.target_label,
        "expected_direction": gt.expected_direction,
        "expected_area_bucket": list(gt.expected_area_bucket) if gt.expected_area_bucket else None,
        "seed": gt.seed,
        "specialized": gt.specialized,
        "metadata": gt.metadata,
    }


def _intent_from_gt(gt: BenchmarkIntent) -> EditIntent:
    return EditIntent(
        primitive=gt.primitive,
        strength=gt.strength,
        reference_profile=gt.profile,
        source_labels=gt.source_labels,
        target_label=gt.target_label,
        region_hint=dict(gt.region_hint),
        seed=gt.seed,
    )


def _build_contour_provider(
    provider: str,
    *,
    api_base_url: str,
    api_key_env: str,
    model: str,
    fixture: str | Path | None,
    api_image_detail: str,
):
    if provider == "fixture":
        if fixture is None:
            raise RuntimeError("contour_fixture is required when contour_provider=fixture")
        return FixtureContourProvider(fixture)
    if provider == "api-text":
        if not model:
            raise RuntimeError("contour_model is required when contour_provider=api-text")
        return OpenAICompatibleTextContourProvider(model=model, api_base_url=api_base_url, api_key_env=api_key_env)
    if provider == "api-vision":
        if not model:
            raise RuntimeError("contour_model is required when contour_provider=api-vision")
        return OpenAICompatibleMultimodalContourProvider(
            model=model,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
            image_detail=api_image_detail,
        )
    raise RuntimeError(f"Unsupported contour_provider: {provider}")
