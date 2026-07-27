"""Batch runner for mask-edit semantic benchmark intents."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from phase3_mask_edit.backends.llm_agent import (
    FixtureContourProvider,
    LLMContourAgentResult,
    OpenAICompatibleMultimodalContourProvider,
    OpenAICompatibleTextContourProvider,
    STATUS_VALIDATED,
    execute_llm_contour_agent,
)
from phase3_mask_edit.backends.llm_contour import PROJECTION_MODE_ORGANIC_V2
from phase3_mask_edit.benchmark.intents import (
    inject_region_hint,
    primitive_config_by_name,
    source_target_labels_for_primitive,
)
from phase3_mask_edit.benchmark.metrics import evaluate_mask_edit, row_for_eval
from phase3_mask_edit.benchmark.models import BenchmarkIntent, BenchmarkPrompt
from phase3_mask_edit.benchmark.prompts import semantic_diff_for_intent
from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import (
    load_id_mask,
    save_change_region,
    save_id_mask,
    save_metadata,
    save_rgb_mask,
)
from phase3_mask_edit.parser.api_parser import ApiParserConfig, parse_prompts_with_api
from phase3_mask_edit.parser.instruction_parser import (
    InstructionParserConfig,
    parse_instruction,
)
from phase3_mask_edit.parser.semantic_diff import save_semantic_diff
from phase3_mask_edit.rules.semantic_to_intent import (
    IntentPlanningResult,
    plan_edit_intents,
)


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
    max_attempts: int = 10,
    semantic_repair_attempts: int = 2,
    max_regions: int = 8,
    max_points_per_region: int = 64,
    coordinate_tolerance_px: float = 16.0,
) -> dict[str, Any]:
    sample_dir = Path(output_dir) / intent.sample_id / mode
    sample_dir.mkdir(parents=True, exist_ok=True)
    source_mask = load_id_mask(intent.mask_path)
    schema = MaskProfileSchema.from_reference_profile(intent.profile)
    recipe = load_recipe(default_recipe_path_for_profile(intent.profile))
    semantic_diff: dict[str, Any] | None = None
    planned_intent: EditIntent | None = None
    contour_agentic = _empty_contour_agentic_metrics(max_attempts=max_attempts)
    semantic_agentic = _empty_semantic_agentic_metrics()
    failure_stage = ""
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
            failure_stage = "semantic_planning"
            semantic_diff, plan, semantic_agentic = _resolve_semantic_plan_with_repair(
                intent,
                prompt,
                mode=mode,
                prompt_parser=prompt_parser,
                instruction_parser=instruction_parser,
                api_base_url=parser_api_base_url,
                api_key_env=parser_api_key_env,
                parser_model=parser_model,
                output_dir=sample_dir,
                source_mask=source_mask,
                recipe=recipe,
                max_repair_attempts=semantic_repair_attempts,
            )
            save_semantic_diff(semantic_diff, sample_dir / "semantic_diff.json")
            save_metadata(plan.to_metadata(), sample_dir / "planning_summary.json")
            executable = [
                item
                for item in getattr(plan, "items", ())
                if getattr(item, "intent", None) is not None
            ]
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
        primitive_config = _benchmark_primitive_config(
            primitive_config_by_name(recipe, planned_intent.primitive)
        )
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
        if coordinate_tolerance_px > 0:
            provider = _BoundaryTolerantContourProvider(
                provider,
                mask_shape=tuple(source_mask.shape),
                tolerance_px=coordinate_tolerance_px,
            )
        failure_stage = "contour"
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
        contour_agentic = _contour_agentic_metrics(
            result,
            max_attempts=max_attempts,
        )
        if result.status != STATUS_VALIDATED or result.edit_result is None:
            raise RuntimeError(
                result.error or f"contour execution failed: {result.status}"
            )
        target_mask = result.edit_result.target_mask
        change_region = source_mask != target_mask
        save_id_mask(source_mask, sample_dir / "source_mask.png")
        save_id_mask(target_mask, sample_dir / "target_mask.png")
        save_rgb_mask(source_mask, sample_dir / "source_mask_rgb.png")
        save_rgb_mask(target_mask, sample_dir / "target_mask_rgb.png")
        save_change_region(change_region, sample_dir / "change_region.png")
        metrics = evaluate_mask_edit(source_mask, target_mask, intent, mode=mode)
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
            source_dataset=intent.source_dataset,
            wsi_id=intent.wsi_id,
            patient_id=intent.patient_id,
            ordinal_group_id=intent.ordinal_group_id,
            contour_agentic=contour_agentic,
            semantic_agentic=semantic_agentic,
            failure_stage="",
        )
    except Exception as exc:
        if (
            failure_stage == "contour"
            and not contour_agentic["terminal_failure_reason"]
        ):
            contour_agentic["terminal_failure_reason"] = str(exc)
        if (
            failure_stage == "semantic_planning"
            and not semantic_agentic["semantic_terminal_failure_reason"]
        ):
            semantic_agentic["semantic_terminal_failure_reason"] = str(exc)
        save_metadata(
            {"status": "failed", "error": str(exc)}, sample_dir / "failure.json"
        )
        return row_for_eval(
            sample_id=intent.sample_id,
            mode=mode,
            status="failed",
            parsed_semantic_diff=semantic_diff,
            planned_primitive=(planned_intent.primitive if planned_intent else None),
            metrics=None,
            error=str(exc),
            output_dir=str(sample_dir),
            organ=intent.organ,
            profile=intent.profile,
            primitive=intent.primitive,
            strength=intent.strength,
            source_dataset=intent.source_dataset,
            wsi_id=intent.wsi_id,
            patient_id=intent.patient_id,
            ordinal_group_id=intent.ordinal_group_id,
            contour_agentic=contour_agentic,
            semantic_agentic=semantic_agentic,
            failure_stage=failure_stage,
        )


def _resolve_semantic_plan_with_repair(
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
    source_mask: np.ndarray,
    recipe: Mapping[str, Any],
    max_repair_attempts: int,
) -> tuple[dict[str, Any], IntentPlanningResult, dict[str, Any]]:
    repair_supported = (
        mode == PROMPT_MODE
        and prompt_parser == "api"
        or mode == INSTRUCTION_MODE
        and instruction_parser == "api"
    )
    total_attempts = 1 + max(0, max_repair_attempts) if repair_supported else 1
    repair_feedback: dict[str, Any] | None = None
    previous_semantic_diff: dict[str, Any] | None = None
    trace_attempts: list[dict[str, Any]] = []
    final_semantic_diff: dict[str, Any] | None = None
    final_plan: IntentPlanningResult | None = None

    for attempt_index in range(1, total_attempts + 1):
        attempt_dir = output_dir / "semantic_attempts" / f"attempt_{attempt_index:03d}"
        semantic_diff = _resolve_semantic_diff(
            intent,
            prompt,
            mode=mode,
            prompt_parser=prompt_parser,
            instruction_parser=instruction_parser,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
            parser_model=parser_model,
            output_dir=attempt_dir,
            repair_feedback=repair_feedback,
            previous_semantic_diff=previous_semantic_diff,
        )
        plan = plan_edit_intents(
            semantic_diff,
            reference_profile=intent.profile,
            old_prompt=prompt.old_prompt,
            new_prompt=(
                prompt.new_prompt if mode == PROMPT_MODE else prompt.instruction
            ),
            old_mask=source_mask,
            recipe=recipe,
        )
        executable = [
            item
            for item in getattr(plan, "items", ())
            if getattr(item, "intent", None) is not None
        ]
        status = "planned" if executable else "planner_no_executable_intents"
        attempt_metadata = {
            "attempt_index": attempt_index,
            "status": status,
            "repair_feedback": repair_feedback,
            "semantic_diff": semantic_diff,
            "planner": plan.to_metadata(),
        }
        save_metadata(attempt_metadata, attempt_dir / "attempt_summary.json")
        trace_attempts.append(attempt_metadata)
        final_semantic_diff = semantic_diff
        final_plan = plan
        if executable:
            break
        previous_semantic_diff = semantic_diff
        repair_feedback = _semantic_planner_repair_feedback(
            plan,
            attempt_index=attempt_index,
        )

    assert final_semantic_diff is not None
    assert final_plan is not None
    semantic_agentic = _semantic_agentic_metrics(trace_attempts)
    save_metadata(
        {
            **semantic_agentic,
            "attempts": trace_attempts,
        },
        output_dir / "semantic_planning_trace.json",
    )
    return final_semantic_diff, final_plan, semantic_agentic


def _semantic_planner_repair_feedback(
    plan: IntentPlanningResult,
    *,
    attempt_index: int,
) -> dict[str, Any]:
    metadata = plan.to_metadata()
    return {
        "status": "planner_no_executable_intents",
        "attempt": attempt_index,
        "reference_profile": plan.reference_profile,
        "planner_items": metadata.get("items", []),
        "unsupported_changes": metadata.get("unsupported_changes", []),
        "planner_context": metadata.get("metadata", {}),
        "instruction": (
            "Recompare the original inputs for a missed or misclassified explicit "
            "primary edit. Report pairs intentionally use standalone, non-comparative "
            "absolute states; for example, 'mild focal stromal tissue' versus 'scant "
            "stromal tissue' supports a stromal decrease even though neither report "
            "uses comparative wording. Correct only a change supported by the original "
            "reports or instruction, and do not invent an edit."
        ),
    }


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
    repair_feedback: Mapping[str, Any] | None = None,
    previous_semantic_diff: Mapping[str, Any] | None = None,
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
                repair_feedback=repair_feedback,
                previous_semantic_diff=previous_semantic_diff,
            )
        raise RuntimeError(f"Unsupported prompt_parser: {prompt_parser}")
    if mode == INSTRUCTION_MODE:
        if instruction_parser == "gt":
            return semantic_diff_for_intent(intent)
        if instruction_parser == "rule-based":
            return parse_instruction(prompt.instruction, mode="rule-based")
        if instruction_parser == "api":
            if not parser_model:
                raise RuntimeError(
                    "parser_model is required for instruction_parser=api"
                )
            return parse_instruction(
                prompt.instruction,
                mode="api",
                config=InstructionParserConfig(
                    model=parser_model,
                    api_base_url=api_base_url,
                    api_key_env=api_key_env,
                    debug_dir=str(output_dir / "instruction_parser_debug"),
                ),
                repair_feedback=repair_feedback,
                previous_semantic_diff=previous_semantic_diff,
            )
        raise RuntimeError(f"Unsupported instruction_parser: {instruction_parser}")
    raise RuntimeError(f"Unsupported benchmark mode: {mode}")


def _empty_contour_agentic_metrics(*, max_attempts: int) -> dict[str, Any]:
    return {
        "attempt_count": 0,
        "first_attempt_status": "",
        "final_attempt_status": "",
        "replanned": False,
        "repair_success": False,
        "terminal_failure_reason": "",
        "cumulative_success_at_k": {
            str(index): False for index in range(1, max(0, max_attempts) + 1)
        },
    }


def _contour_agentic_metrics(
    result: LLMContourAgentResult,
    *,
    max_attempts: int,
) -> dict[str, Any]:
    attempts = list(result.attempts)
    successful_attempt = next(
        (
            attempt.attempt_index
            for attempt in attempts
            if attempt.status == STATUS_VALIDATED
        ),
        None,
    )
    final_status = result.final_attempt.status if result.final_attempt else ""
    return {
        "attempt_count": len(attempts),
        "first_attempt_status": attempts[0].status if attempts else "",
        "final_attempt_status": final_status,
        "replanned": len(attempts) > 1,
        "repair_success": len(attempts) > 1 and result.status == STATUS_VALIDATED,
        "terminal_failure_reason": (
            ""
            if result.status == STATUS_VALIDATED
            else result.error or final_status or result.status
        ),
        "cumulative_success_at_k": {
            str(index): successful_attempt is not None and successful_attempt <= index
            for index in range(1, max(0, max_attempts) + 1)
        },
    }


def _empty_semantic_agentic_metrics() -> dict[str, Any]:
    return {
        "semantic_attempt_count": 0,
        "semantic_first_attempt_status": "",
        "semantic_final_attempt_status": "",
        "semantic_replanned": False,
        "semantic_repair_success": False,
        "semantic_terminal_failure_reason": "",
    }


def _semantic_agentic_metrics(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [str(item.get("status") or "") for item in attempts]
    final_status = statuses[-1] if statuses else ""
    return {
        "semantic_attempt_count": len(attempts),
        "semantic_first_attempt_status": statuses[0] if statuses else "",
        "semantic_final_attempt_status": final_status,
        "semantic_replanned": len(attempts) > 1,
        "semantic_repair_success": len(attempts) > 1 and final_status == "planned",
        "semantic_terminal_failure_reason": (
            "" if final_status == "planned" else final_status
        ),
    }


def _select_or_override_planned_intent(
    gt: BenchmarkIntent,
    planned: EditIntent,
    recipe: Mapping[str, Any],
    schema: MaskProfileSchema,
    *,
    allow_gt_override: bool,
) -> EditIntent:
    if allow_gt_override and (
        planned.primitive != gt.primitive or planned.strength != gt.strength
    ):
        payload = planned.to_metadata()
        payload["primitive"] = gt.primitive
        payload["strength"] = gt.strength
        payload["reference_profile"] = gt.profile
        payload["source_labels"] = list(gt.source_labels)
        payload["target_label"] = gt.target_label
        planned = EditIntent.from_mapping(payload)
    planned = inject_region_hint(planned, gt.region_hint)
    primitive_config = primitive_config_by_name(recipe, planned.primitive)
    source_labels, target_label = source_target_labels_for_primitive(
        primitive_config, schema
    )
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
    recipe_source_labels, recipe_target_label = source_target_labels_for_primitive(
        primitive_config, schema
    )
    intent_source_labels = tuple(intent.source_labels)
    intent_target_label = intent.target_label
    if primitive_config.get("name") == "tumor_burden_increase":
        return (
            recipe_source_labels or intent_source_labels,
            recipe_target_label or intent_target_label,
        )
    if prefer_intent_labels:
        return (
            intent_source_labels or recipe_source_labels,
            intent_target_label or recipe_target_label,
        )
    return (
        recipe_source_labels or intent_source_labels,
        recipe_target_label or intent_target_label,
    )


def _benchmark_primitive_config(
    primitive_config: Mapping[str, Any],
) -> dict[str, Any]:
    strict_config = copy.deepcopy(dict(primitive_config))
    if strict_config.get("name") == "necrosis_appearance":
        ranges = dict(strict_config.get("parameter_ranges") or {})
        ranges["necrosis_intrusion_closing_radius_px"] = 0
        strict_config["parameter_ranges"] = ranges
    return strict_config


def _anchor_labels_from_gt(gt: BenchmarkIntent) -> tuple[str, ...]:
    metadata_labels = (
        gt.metadata.get("anchor_labels") if isinstance(gt.metadata, Mapping) else None
    )
    if isinstance(metadata_labels, (list, tuple)):
        return tuple(str(label) for label in metadata_labels if str(label))
    region_labels = (
        gt.region_hint.get("anchor_labels")
        if isinstance(gt.region_hint, Mapping)
        else None
    )
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
        "expected_area_bucket": list(gt.expected_area_bucket)
        if gt.expected_area_bucket
        else None,
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
            raise RuntimeError(
                "contour_fixture is required when contour_provider=fixture"
            )
        return FixtureContourProvider(fixture)
    if provider == "api-text":
        if not model:
            raise RuntimeError(
                "contour_model is required when contour_provider=api-text"
            )
        return OpenAICompatibleTextContourProvider(
            model=model, api_base_url=api_base_url, api_key_env=api_key_env
        )
    if provider == "api-vision":
        if not model:
            raise RuntimeError(
                "contour_model is required when contour_provider=api-vision"
            )
        return OpenAICompatibleMultimodalContourProvider(
            model=model,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
            image_detail=api_image_detail,
        )
    raise RuntimeError(f"Unsupported contour_provider: {provider}")


class _BoundaryTolerantContourProvider:
    """Clamp only small LLM coordinate overshoots before strict validation."""

    def __init__(
        self, delegate: Any, *, mask_shape: tuple[int, int], tolerance_px: float
    ) -> None:
        self.delegate = delegate
        self.mask_shape = mask_shape
        self.tolerance_px = max(0.0, float(tolerance_px))
        self.name = f"{getattr(delegate, 'name', 'provider')}_boundary_tolerant"

    def propose(self, request: Any) -> Mapping[str, Any]:
        payload = copy.deepcopy(dict(self.delegate.propose(request)))
        height, width = self.mask_shape
        regions = payload.get("regions")
        if not isinstance(regions, list):
            return payload
        for region in regions:
            if not isinstance(region, dict) or not isinstance(
                region.get("points"), list
            ):
                continue
            region["points"] = [
                _clamp_near_boundary_point(
                    point, width=width, height=height, tolerance=self.tolerance_px
                )
                for point in region["points"]
            ]
        return payload


def _clamp_near_boundary_point(
    point: Any, *, width: int, height: int, tolerance: float
) -> Any:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return point
    x, y = point
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return point
    max_x = float(width - 1)
    max_y = float(height - 1)
    if -tolerance <= float(x) <= max_x + tolerance:
        x = min(max(float(x), 0.0), max_x)
    if -tolerance <= float(y) <= max_y + tolerance:
        y = min(max(float(y), 0.0), max_y)
    return [x, y]
