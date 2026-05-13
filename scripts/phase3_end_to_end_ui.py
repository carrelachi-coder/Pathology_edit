#!/usr/bin/env python3
"""Local step-by-step UI for the Phase3 -> Phase4 -> Phase5 edit chain."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - exercised by users launching the UI.
    raise SystemExit(
        "Gradio is required for the local UI. Install it in this environment with `pip install gradio`."
    ) from exc

from phase3_mask_edit.backends.llm_agent import (
    FixtureContourProvider,
    OpenAICompatibleMultimodalContourProvider,
    OpenAICompatibleTextContourProvider,
    execute_llm_contour_agent,
)
from phase3_mask_edit.backends.llm_contour import PROJECTION_MODE_ORGANIC_V2
from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import (
    load_change_region,
    load_id_mask,
    save_change_region,
    save_id_mask,
    save_metadata,
)
from phase3_mask_edit.parser.api_parser import ApiParserConfig, parse_prompts_with_api
from phase3_mask_edit.parser.qwen_local_parser import (
    QwenLocalParserConfig,
    parse_prompts_with_qwen_local,
)
from phase3_mask_edit.parser.semantic_diff import save_semantic_diff
from phase3_mask_edit.rules.semantic_to_intent import plan_edit_intents
from scripts.run_phase3_inpaint_pipeline import (
    _build_target_nuclei,
    _change_area_fraction,
    _load_rgb_image,
    _load_uint8_mask,
    _run_generation_stage,
    _save_compare_panel,
    _save_pre_generation_artifacts,
    _save_target_combined_mask,
    _select_generation_mode,
    _format_subprocess_error,
    _validate_same_size,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "phase3_end_to_end_ui"


def _file_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return Path(value)
    name = getattr(value, "name", None)
    return Path(name) if name else None


def _copy_input(value: Any, output_dir: Path, filename: str) -> Path:
    source = _file_path(value)
    if source is None:
        raise gr.Error(f"Missing input: {filename}")
    target = output_dir / "inputs" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _make_args(state: dict[str, Any], **overrides: Any) -> SimpleNamespace:
    defaults = {
        "profile": state.get("profile", "BCSS"),
        "reference_image": Path(state["reference_image"]),
        "reference_tissue_mask": Path(state["reference_tissue_mask"]),
        "reference_nuclei_mask": Path(state["reference_nuclei_mask"]),
        "target_tissue_mask": Path(state["target_tissue_mask"]) if state.get("target_tissue_mask") else None,
        "change_region": Path(state["change_region"]) if state.get("change_region") else None,
        "output": Path(state["output_dir"]),
        "continue_on_failure": False,
        "cell_fill_mode": "preserve",
        "crossing_cell_policy": "delete",
        "probnet_ckpt": None,
        "nuclei_library": None,
        "probnet_device": "auto",
        "probnet_gamma_values": "1.0",
        "density_scale_json": None,
        "generation_mode": "dry-run",
        "route_threshold": 0.35,
        "pretrained_model_name_or_path": None,
        "inpaint_checkpoint": None,
        "cross_v1_checkpoint": None,
        "uni_checkpoint": None,
        "device": "cuda",
        "prompt": None,
        "print_summary": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _contour_failure_message(result: Any) -> str:
    lines = [f"Contour stage finished with status {result.status}."]
    if result.error:
        lines.append(f"Error: {result.error}")

    final_attempt = getattr(result, "final_attempt", None)
    if final_attempt is not None:
        lines.append(f"Final attempt status: {final_attempt.status}")
        if final_attempt.error:
            lines.append(f"Final attempt error: {final_attempt.error}")
        if final_attempt.validation is not None:
            failed = [
                f"{check.name}: {check.detail}"
                for check in final_attempt.validation.failed_checks
            ]
            if failed:
                lines.append("Failed validation checks:")
                lines.extend(f"- {item}" for item in failed)
            warnings = list(final_attempt.validation.warnings)
            if warnings:
                lines.append("Validation warnings:")
                lines.extend(f"- {warning}" for warning in warnings)
        if final_attempt.repair_feedback:
            lines.append("Repair feedback:")
            lines.append(_json_text(final_attempt.repair_feedback))
        if final_attempt.artifact_paths:
            lines.append("Attempt artifacts:")
            for name, path in final_attempt.artifact_paths.items():
                lines.append(f"- {name}: {path}")

    if getattr(result, "artifact_paths", None):
        lines.append("Run artifacts:")
        for name, path in result.artifact_paths.items():
            lines.append(f"- {name}: {path}")
    return "\n".join(lines)


class _NoOpEditResult:
    def __init__(self, target_mask: np.ndarray) -> None:
        self.target_mask = np.array(target_mask, copy=True)


class _SkippedPromptResult:
    status = "skipped_no_source_region"
    error = None
    final_attempt = None
    validation = None
    projection_mode = PROJECTION_MODE_ORGANIC_V2

    def __init__(
        self,
        *,
        source_mask: np.ndarray,
        target_mask: np.ndarray,
        attempts: list[dict[str, Any]],
        artifact_paths: dict[str, str],
    ) -> None:
        self.source_mask = np.array(source_mask, copy=True)
        self.attempts = tuple(attempts)
        self.artifact_paths = dict(artifact_paths)
        self._edit_result = _NoOpEditResult(target_mask)

    @property
    def edit_result(self) -> _NoOpEditResult:
        return self._edit_result

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error": self.error,
            "projection_mode": self.projection_mode,
            "attempts": list(self.attempts),
            "artifact_paths": dict(self.artifact_paths),
        }


def load_inputs(
    profile: str,
    source_image,
    source_tissue_mask,
    source_cell_mask,
    cellvit_command: str,
    output_root: str,
) -> tuple[dict[str, Any], str, str | None, str | None]:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root or DEFAULT_OUTPUT_ROOT) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = _copy_input(source_image, output_dir, "source_image.png")
    tissue_path = _copy_input(source_tissue_mask, output_dir, "source_tissue_mask.png")
    nuclei_path = _file_path(source_cell_mask)
    if nuclei_path is None:
        command = (cellvit_command or "").strip()
        if not command:
            raise gr.Error("Upload a CellViT source cell mask, or provide a CellViT command template.")
        nuclei_path = output_dir / "inputs" / "source_cell_mask.png"
        nuclei_path.parent.mkdir(parents=True, exist_ok=True)
        formatted = command.format(image=str(image_path), output=str(nuclei_path))
        subprocess.run(formatted, shell=True, cwd=REPO_ROOT, check=True)
        if not nuclei_path.exists():
            raise gr.Error(f"CellViT command finished but did not write {nuclei_path}")
    else:
        nuclei_path = _copy_input(source_cell_mask, output_dir, "source_cell_mask.png")

    image = _load_rgb_image(image_path)
    tissue = load_id_mask(tissue_path)
    nuclei = _load_uint8_mask(nuclei_path)
    _validate_same_size(image, tissue, "source_tissue_mask")
    _validate_same_size(image, nuclei, "source_cell_mask")

    state = {
        "profile": profile,
        "output_dir": str(output_dir),
        "reference_image": str(image_path),
        "reference_tissue_mask": str(tissue_path),
        "reference_nuclei_mask": str(nuclei_path),
    }
    source_rgb = str(_save_pre_generation_artifacts(
        output_dir=output_dir,
        reference_image=image,
        reference_tissue=tissue,
        target_tissue=tissue,
        change_region=np.zeros(tissue.shape, dtype=bool),
    )["source_mask_rgb"])
    return state, _json_text({"status": "loaded", "output_dir": str(output_dir)}), str(image_path), source_rgb


def run_tissue_stage(
    state: dict[str, Any],
    old_prompt: str,
    new_prompt: str,
    parser: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    qwen_model_path: str,
    no_few_shot: bool,
    primitive: str,
    source_labels: str,
    target_label: str,
    strength: str,
    provider: str,
    api_image_detail: str,
    fixture_file,
    max_attempts: int,
    max_regions: int,
    max_points_per_region: int,
    organic_seed: int,
    continue_on_failure: bool,
) -> tuple[dict[str, Any], str, str, str]:
    if not state:
        raise gr.Error("Load inputs first.")
    output_dir = Path(state["output_dir"])
    reference_image = _load_rgb_image(state["reference_image"])
    reference_tissue = load_id_mask(state["reference_tissue_mask"])
    schema = MaskProfileSchema.from_reference_profile(state["profile"])
    recipe = load_recipe(default_recipe_path_for_profile(state["profile"]))
    try:
        if old_prompt.strip() and new_prompt.strip():
            semantic_diff, parser_info = _resolve_prompt_semantic_diff(
                old_prompt=old_prompt,
                new_prompt=new_prompt,
                parser=parser,
                api_base_url=api_base_url,
                api_key_env=api_key_env,
                api_model=api_model,
                qwen_model_path=qwen_model_path,
                no_few_shot=no_few_shot,
                output_dir=output_dir,
            )
            plan = plan_edit_intents(
                semantic_diff,
                reference_profile=state["profile"],
                old_mask=reference_tissue,
                old_prompt=old_prompt,
                new_prompt=new_prompt,
            )
            save_semantic_diff(semantic_diff, output_dir / "phase3_mask_edit" / "semantic_diff.json")
            save_metadata(
                plan.to_metadata(),
                output_dir / "phase3_mask_edit" / "planning_summary.json",
            )
            provider_instance = _build_contour_provider(
                provider=provider,
                api_base_url=api_base_url,
                api_key_env=api_key_env,
                api_model=api_model,
                api_image_detail=api_image_detail,
                fixture_file=fixture_file,
            )
            current_mask = np.array(reference_tissue, copy=True)
            last_result = None
            last_edit_result = None
            attempt_logs: list[dict[str, Any]] = []
            for intent in plan.intents:
                primitive_config = _primitive_config(recipe, intent.primitive)
                intent = _with_default_contour_labels(intent, primitive_config, schema)
                source_summary = _source_region_summary(
                    current_mask,
                    schema,
                    intent,
                    primitive_config,
                )
                if source_summary["source_pixels"] == 0:
                    attempt_logs.append(
                        {
                            "primitive": intent.primitive,
                            "status": "skipped_no_source_region",
                            "projection_mode": PROJECTION_MODE_ORGANIC_V2,
                            "source_labels": source_summary["source_labels"],
                            "missing_source_labels": source_summary.get(
                                "missing_source_labels",
                                [],
                            ),
                            "source_pixels": 0,
                            "error": (
                                "Skipped because prior edits left no pixels for "
                                f"source labels {source_summary['source_labels']}."
                            ),
                            "artifact_paths": {},
                        }
                    )
                    continue
                result = execute_llm_contour_agent(
                    old_mask=current_mask,
                    schema=schema,
                    intent=intent,
                    primitive_config=primitive_config,
                    provider=provider_instance,
                    output_dir=output_dir / "phase3_mask_edit" / "llm_contour" / intent.primitive,
                    projection_mode=PROJECTION_MODE_ORGANIC_V2,
                    organic_seed=organic_seed,
                    max_attempts=max_attempts,
                    max_regions=max_regions,
                    max_points_per_region=max_points_per_region,
                )
                attempt_logs.append(
                    {
                        "primitive": intent.primitive,
                        "status": result.status,
                        "projection_mode": PROJECTION_MODE_ORGANIC_V2,
                        "error": result.error,
                        "artifact_paths": result.artifact_paths,
                    }
                )
                last_result = result
                if result.edit_result is None:
                    if not continue_on_failure:
                        break
                    continue
                last_edit_result = result.edit_result
                current_mask = np.array(result.edit_result.target_mask, copy=True)
                if result.status != "validated" and not continue_on_failure:
                    break
            if last_result is None:
                result = _SkippedPromptResult(
                    source_mask=reference_tissue,
                    target_mask=current_mask,
                    attempts=attempt_logs,
                    artifact_paths={},
                )
                phase3_info = {
                    "mode": "prompt_to_contour",
                    "parser": parser_info,
                    "semantic_diff": semantic_diff,
                    "plan": plan.to_metadata(),
                    "attempts": attempt_logs,
                    "projection_mode": PROJECTION_MODE_ORGANIC_V2,
                    "status": "all_intents_skipped",
                }
                save_metadata(
                    phase3_info,
                    output_dir / "phase3_mask_edit" / "execution_summary.json",
                )
                if not attempt_logs:
                    raise gr.Error("Prompt-driven contour planning produced no intents.")
            elif last_edit_result is None:
                raise gr.Error(_contour_failure_message(last_result))
            else:
                result = last_result
                phase3_info = {
                    "mode": "prompt_to_contour",
                    "parser": parser_info,
                    "semantic_diff": semantic_diff,
                    "plan": plan.to_metadata(),
                    "attempts": attempt_logs,
                    "projection_mode": PROJECTION_MODE_ORGANIC_V2,
                }
        else:
            primitive_config = _primitive_config(recipe, primitive)
            intent = _build_contour_intent(
                primitive_config,
                profile=state["profile"],
                strength=strength,
                source_labels=source_labels,
                target_label=target_label,
            )
            provider_instance = _build_contour_provider(
                provider=provider,
                api_base_url=api_base_url,
                api_key_env=api_key_env,
                api_model=api_model,
                api_image_detail=api_image_detail,
                fixture_file=fixture_file,
            )
            result = execute_llm_contour_agent(
                old_mask=reference_tissue,
                schema=schema,
                intent=intent,
                primitive_config=primitive_config,
                provider=provider_instance,
                output_dir=output_dir / "phase3_mask_edit" / "llm_contour",
                projection_mode=PROJECTION_MODE_ORGANIC_V2,
                organic_seed=organic_seed,
                max_attempts=max_attempts,
                max_regions=max_regions,
                max_points_per_region=max_points_per_region,
            )
            phase3_info = result.to_metadata()
    except Exception as exc:
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc

    if result.edit_result is None:
        raise gr.Error(_contour_failure_message(result))
    if (
        result.status not in {"validated", "skipped_no_source_region"}
        and not continue_on_failure
    ):
        raise gr.Error(_contour_failure_message(result))

    target_tissue = result.edit_result.target_mask
    target_path = save_id_mask(target_tissue, output_dir / "target_mask.png")
    if phase3_info.get("mode") == "prompt_to_contour":
        phase3_info = {**phase3_info, "result": result.to_metadata()}
    else:
        phase3_info = result.to_metadata()

    _validate_same_size(reference_image, target_tissue, "target_tissue_mask")
    change_region = reference_tissue != target_tissue
    stage_paths = _save_pre_generation_artifacts(
        output_dir=output_dir,
        reference_image=_load_rgb_image(state["reference_image"]),
        reference_tissue=reference_tissue,
        target_tissue=target_tissue,
        change_region=change_region,
    )
    state.update(
        {
            "target_tissue_mask": str(target_path),
            "change_region": stage_paths["change_region"],
            "phase3": phase3_info,
        }
    )
    info = {
        "status": "tissue_done",
        "projection_mode": PROJECTION_MODE_ORGANIC_V2,
        "primitive": primitive,
        "prompt_mode": bool(old_prompt.strip() and new_prompt.strip()),
        "changed_area_fraction": _change_area_fraction(change_region),
        "target_tissue_mask": str(target_path),
        "change_region": stage_paths["change_region"],
    }
    return state, _json_text(info), stage_paths["target_mask_rgb"], stage_paths["change_region"]


def run_cell_stage(
    state: dict[str, Any],
    cell_fill_mode: str,
    crossing_cell_policy: str,
    probnet_ckpt: str,
    nuclei_library: str,
    density_scale_json: str,
    probnet_device: str,
    gamma_values: str,
) -> tuple[dict[str, Any], str, str, str, str]:
    if not state or not state.get("target_tissue_mask") or not state.get("change_region"):
        raise gr.Error("Run the tissue stage first.")
    output_dir = Path(state["output_dir"])
    target_tissue = load_id_mask(state["target_tissue_mask"])
    reference_nuclei = _load_uint8_mask(state["reference_nuclei_mask"])
    change_region = load_change_region(state["change_region"])
    args = _make_args(
        state,
        cell_fill_mode=cell_fill_mode,
        crossing_cell_policy=crossing_cell_policy,
        probnet_ckpt=Path(probnet_ckpt) if probnet_ckpt else None,
        nuclei_library=Path(nuclei_library) if nuclei_library else None,
        density_scale_json=Path(density_scale_json) if density_scale_json else None,
        probnet_device=probnet_device,
        probnet_gamma_values=gamma_values or "1.0",
    )
    try:
        target_nuclei, cell_info = _build_target_nuclei(
            args, reference_nuclei, target_tissue, change_region, output_dir
        )
    except subprocess.CalledProcessError as exc:
        raise gr.Error(_format_subprocess_error(exc, label="ProbNet cell fill")) from exc
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc
    target_nuclei_path = save_id_mask(target_nuclei, output_dir / "target_nuclei_mask.png")
    combined_path = _save_target_combined_mask(
        output_dir / "target_combined_mask.png",
        target_tissue=target_tissue,
        target_nuclei=target_nuclei,
    )
    cell_info["target_nuclei_mask"] = str(target_nuclei_path)
    (output_dir / "cell_fill_log.json").write_text(_json_text(cell_info), encoding="utf-8")
    state.update(
        {
            "target_nuclei_mask": str(target_nuclei_path),
            "cell_fill": cell_info,
            "target_combined_mask": str(combined_path),
        }
    )
    return (
        state,
        _json_text(cell_info),
        str(output_dir / "retained_nuclei_mask.png"),
        str(output_dir / "new_nuclei_mask.png"),
        str(combined_path),
    )


def run_generation_stage(
    state: dict[str, Any],
    generation_mode: str,
    route_threshold: float,
    model_path: str,
    inpaint_checkpoint: str,
    cross_v1_checkpoint: str,
    uni_checkpoint: str,
    device: str,
    prompt: str,
) -> tuple[dict[str, Any], str, str, str]:
    if not state or not state.get("target_nuclei_mask"):
        raise gr.Error("Run the cell-mask stage first.")
    output_dir = Path(state["output_dir"])
    reference_image = _load_rgb_image(state["reference_image"])
    change_region = load_change_region(state["change_region"])
    args = _make_args(
        state,
        generation_mode=generation_mode,
        route_threshold=route_threshold,
        pretrained_model_name_or_path=model_path or None,
        inpaint_checkpoint=Path(inpaint_checkpoint) if inpaint_checkpoint else None,
        cross_v1_checkpoint=Path(cross_v1_checkpoint) if cross_v1_checkpoint else None,
        uni_checkpoint=Path(uni_checkpoint) if uni_checkpoint else None,
        device=device,
        prompt=prompt or None,
    )
    generated_path, generation_info = _run_generation_stage(
        args=args,
        output_dir=output_dir,
        reference_image=reference_image,
        change_region=change_region,
        target_tissue_path=Path(state["target_tissue_mask"]),
        target_nuclei_path=Path(state["target_nuclei_mask"]),
    )
    panel_path = _save_compare_panel(
        output_dir / "compare_panel.png",
        reference_image=reference_image,
        erased_image=np.asarray(Image.open(output_dir / "erased_image.png").convert("RGB")),
        target_mask_rgb=np.asarray(Image.open(output_dir / "target_mask_rgb.png").convert("RGB")),
        generated_image=np.asarray(Image.open(generated_path).convert("RGB")),
        title=f"{generation_info['selected_mode']} / change={generation_info['change_ratio']:.3f}",
    )
    summary = {
        "status": "completed",
        "output_dir": str(output_dir),
        "phase3": state.get("phase3"),
        "cell_fill": state.get("cell_fill"),
        "generation": generation_info,
        "artifacts": {
            "target_tissue_mask": state["target_tissue_mask"],
            "change_region": state["change_region"],
            "target_nuclei_mask": state["target_nuclei_mask"],
            "generated_image": str(generated_path),
            "compare_panel": str(panel_path),
        },
    }
    (output_dir / "pipeline_summary.json").write_text(_json_text(summary), encoding="utf-8")
    state["generation"] = generation_info
    return state, _json_text(summary), str(generated_path), str(panel_path)


def preview_route(state: dict[str, Any], threshold: float) -> str:
    if not state or not state.get("change_region"):
        return "Run the tissue stage first."
    change_region = load_change_region(state["change_region"])
    ratio = _change_area_fraction(change_region)
    selected = _select_generation_mode("auto", ratio, threshold)
    return f"change_region = {ratio:.2%}; auto route = {selected} (threshold {threshold:.0%})"


def _primitive_config(recipe: dict[str, Any], primitive_name: str) -> dict[str, Any]:
    for primitive in recipe.get("primitives", []):
        if isinstance(primitive, dict) and primitive.get("name") == primitive_name:
            return primitive
    raise gr.Error(f"Unknown primitive: {primitive_name}")


def _source_region_summary(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    intent: EditIntent,
    primitive_config: dict[str, Any],
) -> dict[str, Any]:
    labels = tuple(intent.source_labels)
    if not labels:
        operation = primitive_config.get("mask_operation", {})
        operation = operation if isinstance(operation, dict) else {}
        labels = tuple(_default_contour_sources(primitive_config, operation))
    if not labels:
        required = primitive_config.get("required_tissue_labels", ())
        if isinstance(required, list) and all(isinstance(item, str) for item in required):
            labels = tuple(required)
    if not labels:
        return {
            "source_labels": [],
            "source_pixels": int(np.count_nonzero(mask)),
        }

    source_mask = np.zeros(mask.shape, dtype=bool)
    resolved_labels: list[str] = []
    missing_labels: list[str] = []
    for label in labels:
        try:
            fine_ids = schema.resolve_fine_ids(label)
        except Exception:
            missing_labels.append(label)
            continue
        source_mask |= np.isin(mask, fine_ids)
        resolved_labels.append(label)

    return {
        "source_labels": resolved_labels,
        "missing_source_labels": missing_labels,
        "source_pixels": int(np.count_nonzero(source_mask)),
    }


def _build_contour_intent(
    primitive_config: dict[str, Any],
    *,
    profile: str,
    strength: str,
    source_labels: str,
    target_label: str,
) -> EditIntent:
    operation = primitive_config.get("mask_operation", {})
    source = _split_csv(source_labels) or _default_contour_sources(primitive_config, operation)
    target = target_label.strip() if target_label else ""
    if not target:
        schema = MaskProfileSchema.from_reference_profile(profile)
        target = _default_contour_target(primitive_config, operation, schema=schema)
    if not source:
        raise gr.Error("Please provide at least one source label.")
    if not target:
        raise gr.Error("Please provide a target label.")
    return EditIntent(
        primitive=str(primitive_config["name"]),
        strength=strength,
        reference_profile=profile,
        source_labels=tuple(source),
        target_label=target,
    )


def _with_default_contour_labels(
    intent: EditIntent,
    primitive_config: dict[str, Any],
    schema: MaskProfileSchema,
) -> EditIntent:
    operation = primitive_config.get("mask_operation", {})
    operation = operation if isinstance(operation, dict) else {}
    source = tuple(intent.source_labels) or tuple(
        _default_contour_sources(primitive_config, operation)
    )
    target = intent.target_label or _default_contour_target(
        primitive_config,
        operation,
        schema=schema,
    )
    if source == tuple(intent.source_labels) and target == intent.target_label:
        return intent
    payload = intent.to_metadata()
    payload["source_labels"] = list(source)
    payload["target_label"] = target
    return EditIntent.from_mapping(payload)


def _build_contour_provider(
    *,
    provider: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    api_image_detail: str,
    fixture_file,
):
    if provider == "api-text":
        if not api_model:
            raise gr.Error("--api model is required for api-text.")
        return OpenAICompatibleTextContourProvider(
            model=api_model,
            api_base_url=(api_base_url or "https://api.openai.com/v1").rstrip("/"),
            api_key_env=api_key_env or "OPENAI_API_KEY",
        )
    if provider == "api-multimodal":
        if not api_model:
            raise gr.Error("--api model is required for api-multimodal.")
        return OpenAICompatibleMultimodalContourProvider(
            model=api_model,
            api_base_url=(api_base_url or "https://api.openai.com/v1").rstrip("/"),
            api_key_env=api_key_env or "OPENAI_API_KEY",
            image_detail=api_image_detail,
        )
    if provider == "fixture":
        fixture_path = _file_path(fixture_file)
        if fixture_path is None:
            raise gr.Error("Upload a contour fixture JSON when using fixture provider.")
        return FixtureContourProvider(fixture_path)
    raise gr.Error(f"Unsupported contour provider: {provider}")


def _resolve_prompt_semantic_diff(
    *,
    old_prompt: str,
    new_prompt: str,
    parser: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    qwen_model_path: str,
    no_few_shot: bool,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if parser == "api":
        if not api_model:
            raise gr.Error("api model is required for prompt parsing.")
        config = ApiParserConfig(
            model=api_model,
            api_base_url=(api_base_url or "https://api.openai.com/v1").rstrip("/"),
            api_key_env=api_key_env or "OPENAI_API_KEY",
            debug_dir=str(output_dir / "phase3_mask_edit" / "api_parser_debug"),
            use_few_shot=not no_few_shot,
        )
        return parse_prompts_with_api(old_prompt, new_prompt, config=config), {
            "mode": "api",
            "api_base_url": config.api_base_url,
            "api_key_env": config.api_key_env,
            "api_model": api_model,
            "use_few_shot": not no_few_shot,
        }
    if parser == "qwen-local":
        if not qwen_model_path:
            raise gr.Error("qwen model path is required for prompt parsing.")
        config = QwenLocalParserConfig(
            model_path=qwen_model_path,
            device="cuda",
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=not no_few_shot,
            use_few_shot=not no_few_shot,
        )
        return parse_prompts_with_qwen_local(old_prompt, new_prompt, config=config), {
            "mode": "qwen-local",
            "model_path": qwen_model_path,
            "use_few_shot": not no_few_shot,
        }
    raise gr.Error(f"Unsupported parser: {parser}")


def _split_csv(value: str) -> list[str]:
    labels = [part.strip() for part in value.split(",")]
    return [label for label in labels if label]


def _default_contour_sources(
    primitive_config: dict[str, Any],
    operation: dict[str, Any],
) -> list[str]:
    if primitive_config.get("name") == "tumor_burden_increase":
        return _labels_from_operation(operation.get("target_priority"))
    labels = _labels_from_operation(operation.get("source"))
    if labels:
        return labels
    labels.extend(_labels_from_operation(operation.get("primary_sources")))
    labels.extend(_labels_from_operation(operation.get("secondary_sources")))
    return list(dict.fromkeys(labels))


def _default_contour_target(
    primitive_config: dict[str, Any],
    operation: dict[str, Any],
    *,
    schema: MaskProfileSchema | None = None,
) -> str:
    target = operation.get("target")
    if isinstance(target, str):
        return target
    if primitive_config.get("name") == "tumor_burden_increase":
        return "Tumor"
    priority = operation.get("backfill_priority", ())
    if isinstance(priority, list):
        for label in priority:
            if not isinstance(label, str):
                continue
            if schema is None or label in schema.writable_labels:
                return label
    return ""


def _labels_from_operation(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return []


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Pathology Edit Pipeline") as demo:
        gr.Markdown("## Pathology edit pipeline")
        state = gr.State({})

        with gr.Row():
            profile = gr.Dropdown(["BCSS", "PANDA", "GlaS", "IGNITE", "PUMA", "ORCA"], value="BCSS", label="profile")
            output_root = gr.Textbox(value=str(DEFAULT_OUTPUT_ROOT), label="output root")
        with gr.Row():
            source_image = gr.File(label="src_image", file_types=["image"], type="filepath")
            source_tissue = gr.File(label="src_tissue_mask", file_types=["image"], type="filepath")
            source_cell = gr.File(label="src_cell_mask / CellViT output", file_types=["image"], type="filepath")
        cellvit_command = gr.Textbox(
            label="optional CellViT command template",
            placeholder="python scripts/run_cellvit_single_patch.py --image {image} --output-mask {output} --model D:\\path\\to\\CellViT-SAM-H-x40-AMP-001.pth",
        )
        load_button = gr.Button("1. Load inputs")
        load_log = gr.Code(label="load log", language="json")
        with gr.Row():
            src_image_preview = gr.Image(label="source image")
            src_tissue_preview = gr.Image(label="source tissue")

        gr.Markdown("### Tissue mask edit")
        with gr.Row():
            old_prompt = gr.Textbox(label="src prompt", lines=3)
            new_prompt = gr.Textbox(label="new prompt", lines=3)
        with gr.Row():
            parser = gr.Radio(["api", "qwen-local"], value="api", label="parser")
            no_few_shot = gr.Checkbox(value=False, label="no few shot")
        with gr.Row():
            api_model = gr.Textbox(label="api model", placeholder="gpt-4o")
            qwen_model_path = gr.Textbox(label="qwen model path")
        with gr.Row():
            api_base_url = gr.Textbox(value="https://api.openai.com/v1", label="api base url")
            api_key_env = gr.Textbox(value="OPENAI_API_KEY", label="api key env")
        continue_on_failure = gr.Checkbox(value=False, label="continue on Phase3 failure")
        tissue_button = gr.Button("2. Run prompt-driven organic v2 contour edit")
        tissue_log = gr.Code(label="tissue log", language="json")
        with gr.Row():
            target_tissue_preview = gr.Image(label="target tissue")
            change_region_preview = gr.Image(label="change region")
        with gr.Accordion("Advanced overrides", open=False):
            with gr.Row():
                primitive = gr.Dropdown(
                    [
                        "stromal_immune_infiltration",
                        "necrosis_appearance",
                        "tumor_burden_increase",
                        "tumor_burden_decrease",
                        "immune_infiltration_decrease",
                        "stromal_desmoplasia",
                        "stroma_decrease",
                        "stromal_reduction",
                    ],
                    value="stromal_immune_infiltration",
                    label="primitive fallback",
                )
                strength = gr.Radio(["mild", "moderate", "significant"], value="mild", label="strength")
            with gr.Row():
                source_labels = gr.Textbox(label="source labels fallback", placeholder="Stroma")
                target_label = gr.Textbox(label="target label fallback", placeholder="Immune infiltrate")
            with gr.Row():
                provider = gr.Radio(["api-text", "api-multimodal", "fixture"], value="api-multimodal", label="contour provider")
                api_image_detail = gr.Radio(["low", "high", "auto"], value="high", label="image detail")
            with gr.Row():
                fixture_file = gr.File(label="contour fixture JSON", file_types=[".json"], type="filepath")
            with gr.Row():
                max_attempts = gr.Slider(1, 8, value=4, step=1, label="max attempts")
                max_regions = gr.Slider(1, 8, value=8, step=1, label="max regions")
                max_points_per_region = gr.Slider(8, 128, value=64, step=1, label="max points / region")
            organic_seed = gr.Number(value=0, precision=0, label="organic seed")

        gr.Markdown("### Cell mask synthesis")
        with gr.Row():
            cell_fill = gr.Radio(["probnet", "blank", "preserve"], value="probnet", label="cell fill")
            crossing_policy = gr.Radio(["delete", "majority", "keep"], value="delete", label="crossing source-cell policy")
        with gr.Row():
            probnet_ckpt = gr.Textbox(label="ProbNet checkpoint")
            nuclei_library = gr.Textbox(label="nuclei library directory")
            density_scale_json = gr.Textbox(label="density scale JSON")
        with gr.Row():
            probnet_device = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="ProbNet device")
            gamma_values = gr.Textbox(value="1.0", label="gamma values")
        cell_button = gr.Button("3. Build target cell mask")
        cell_log = gr.Code(label="cell log", language="json")
        with gr.Row():
            retained_preview = gr.Image(label="retained source cells")
            new_cells_preview = gr.Image(label="new cells")
            combined_preview = gr.Image(label="target tissue + cells")

        gr.Markdown("### Image generation")
        with gr.Row():
            generation_mode = gr.Radio(["dry-run", "auto", "inpaint", "cross-v1"], value="dry-run", label="generation mode")
            route_threshold = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="inpaint if change > threshold")
        route_button = gr.Button("Preview route")
        route_log = gr.Textbox(label="route")
        with gr.Row():
            model_path = gr.Textbox(label="pretrained FLUX/model path")
            device = gr.Textbox(value="cuda", label="device")
        with gr.Row():
            inpaint_checkpoint = gr.Textbox(label="inpaint checkpoint")
            cross_v1_checkpoint = gr.Textbox(label="cross-v1 checkpoint")
            uni_checkpoint = gr.Textbox(label="UNI checkpoint")
        generation_prompt = gr.Textbox(label="generation prompt", lines=2)
        generate_button = gr.Button("4. Route + generate")
        generation_log = gr.Code(label="summary", language="json")
        with gr.Row():
            generated_preview = gr.Image(label="generated image")
            panel_preview = gr.Image(label="compare panel")

        load_button.click(
            load_inputs,
            inputs=[profile, source_image, source_tissue, source_cell, cellvit_command, output_root],
            outputs=[state, load_log, src_image_preview, src_tissue_preview],
        )
        tissue_button.click(
            run_tissue_stage,
            inputs=[
                state,
                old_prompt,
                new_prompt,
                parser,
                api_base_url,
                api_key_env,
                api_model,
                qwen_model_path,
                no_few_shot,
                primitive,
                source_labels,
                target_label,
                strength,
                provider,
                api_image_detail,
                fixture_file,
                max_attempts,
                max_regions,
                max_points_per_region,
                organic_seed,
                continue_on_failure,
            ],
            outputs=[state, tissue_log, target_tissue_preview, change_region_preview],
        )
        cell_button.click(
            run_cell_stage,
            inputs=[
                state,
                cell_fill,
                crossing_policy,
                probnet_ckpt,
                nuclei_library,
                density_scale_json,
                probnet_device,
                gamma_values,
            ],
            outputs=[state, cell_log, retained_preview, new_cells_preview, combined_preview],
        )
        route_button.click(preview_route, inputs=[state, route_threshold], outputs=[route_log])
        generate_button.click(
            run_generation_stage,
            inputs=[
                state,
                generation_mode,
                route_threshold,
                model_path,
                inpaint_checkpoint,
                cross_v1_checkpoint,
                uni_checkpoint,
                device,
                generation_prompt,
            ],
            outputs=[state, generation_log, generated_preview, panel_preview],
        )
    return demo


def main() -> None:
    build_ui().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
