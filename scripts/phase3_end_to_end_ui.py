#!/usr/bin/env python3
"""Local step-by-step UI for the Phase3 -> Phase4 -> Phase5 edit chain."""

from __future__ import annotations

import json
import os
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
from phase3_mask_edit.backends.llm_contour import (
    CONTOUR_PROPOSAL_BACKEND,
    CONTOUR_PROPOSAL_SCHEMA_VERSION,
    PROJECTION_MODE_ORGANIC_V2,
    ContourProposalValidationError,
    execute_contour_proposal_write,
    load_contour_proposal_json,
    validate_contour_proposal,
)
from phase3_mask_edit.backends.proposal_execution import apply_projected_label_write
from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.applicability import assess_edit_applicability
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import (
    load_change_region,
    load_id_mask,
    save_change_region,
    save_id_mask,
    save_metadata,
)
from phase3_mask_edit.generic.executor import execute_edit
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
from scripts.run_cellvit_single_patch import DEFAULT_CELLVIT_ROOT


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "phase3_end_to_end_ui"
DEFAULT_API_MODEL = "gpt-4o-all"
DEFAULT_API_BASE_URL = "https://api.cursorai.art/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_QWEN_DEVICE = "cuda:0"
DEFAULT_CELLVIT_SCRIPT = REPO_ROOT / "scripts" / "run_cellvit_single_patch.py"
DEFAULT_CELLVIT_MODEL = r"D:\path\to\CellViT-SAM-H-x40-AMP-001.pth"
DEFAULT_CELLVIT_DEVICE = "cuda:0"
DEFAULT_PROBNET_CHECKPOINT = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/checkpoints/best.pt"
DEFAULT_NUCLEI_LIBRARY_TEMPLATE = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/nuclei_library/{profile}"
DEFAULT_DENSITY_SCALE_TEMPLATE = (
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/configs/"
    "density_scale_{profile_lower}.json"
)
DEFAULT_PRETRAINED_MODEL = "/data/huggingface/FLUX.1-dev"
DEFAULT_INPAINT_CHECKPOINT = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_inpaint_all"
DEFAULT_CROSS_CHECKPOINT = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_cross"
DEFAULT_CROSS_V1_CHECKPOINT = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_cross_v1/checkpoint-40000"
DEFAULT_UNI_CHECKPOINT = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/UNI-2h/pytorch_model.bin"
CUDA_DEVICE_CHOICES = []
PROBNET_DEVICE_CHOICES = ["auto", *CUDA_DEVICE_CHOICES, "cpu"]
GENERATION_DEVICE_CHOICES = ["cuda", *CUDA_DEVICE_CHOICES, "cpu"]
EDIT_MODE_PROMPT = "prompt"
EDIT_MODE_MANUAL_CONTOUR = "manual_contour"
EDIT_MODE_AUTO_RECOMMEND = "auto_recommend"
EDIT_MODE_CHOICES = [
    EDIT_MODE_PROMPT,
    EDIT_MODE_MANUAL_CONTOUR,
    EDIT_MODE_AUTO_RECOMMEND,
]
_STRONGEST_FIRST = {"xlarge_deid": 3, "significant": 2, "moderate": 1, "mild": 0}
AUTO_RECOMMEND_LIMIT = 6


def _detect_visible_cuda_device_choices() -> list[str]:
    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        count = _detect_nvidia_smi_gpu_count()
    return [f"cuda:{idx}" for idx in range(count)] or ["cuda:0"]


def _detect_nvidia_smi_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return 0
    return sum(1 for line in result.stdout.splitlines() if line.strip())


CUDA_DEVICE_CHOICES = _detect_visible_cuda_device_choices()
PROBNET_DEVICE_CHOICES = ["auto", *CUDA_DEVICE_CHOICES, "cpu"]
GENERATION_DEVICE_CHOICES = ["cuda", *CUDA_DEVICE_CHOICES, "cpu"]


def _canonical_profile(profile: str) -> str:
    if profile == "GlaS":
        return "GlaS"
    return (profile or "BCSS").upper()


def _profile_defaults(profile: str) -> dict[str, str]:
    profile_name = _canonical_profile(profile)
    return {
        "probnet_ckpt": DEFAULT_PROBNET_CHECKPOINT,
        "nuclei_library": DEFAULT_NUCLEI_LIBRARY_TEMPLATE.format(profile=profile_name),
        "density_scale_json": DEFAULT_DENSITY_SCALE_TEMPLATE.format(
            profile=profile_name,
            profile_lower=profile_name.lower(),
        ),
    }


def _defaulted_text(value: str | None, default: str) -> str:
    return (value or "").strip() or default


def _cuda_index(device: str | None) -> int:
    text = (device or DEFAULT_CELLVIT_DEVICE).strip().lower()
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    if text == "cuda":
        return 0
    return int(text)


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
        "cross_backend": "cross-v1",
        "route_threshold": 0.35,
        "pretrained_model_name_or_path": None,
        "inpaint_checkpoint": None,
        "cross_checkpoint": None,
        "cross_v1_checkpoint": None,
        "uni_checkpoint": None,
        "device": "cuda",
        "prompt": None,
        "prompt_source": "dataset",
        "torch_dtype": "bf16",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "controlnet_conditioning_scale": 1.0,
        "color_match": "lab",
        "print_summary": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _manual_contour_editor_value(source_mask_rgb: str | None) -> dict[str, Any] | None:
    if not source_mask_rgb:
        return None
    return {"background": source_mask_rgb, "layers": [], "composite": None}


def _ordered_schema_labels(schema: MaskProfileSchema) -> list[str]:
    return [label for label in schema.label_to_fine_ids if label in schema.readable_labels]


def _manual_target_label_choices(schema: MaskProfileSchema) -> list[str]:
    return [label for label in _ordered_schema_labels(schema) if label != "Background"]


def _default_manual_target_label(schema: MaskProfileSchema) -> str:
    choices = _manual_target_label_choices(schema)
    if "Tumor" in choices:
        return "Tumor"
    return choices[0] if choices else ""


def _image_to_array(image: Any) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return np.asarray(image)
    if isinstance(image, (str, Path)):
        try:
            return np.asarray(Image.open(image))
        except Exception:
            return None
    try:
        return np.asarray(image)
    except Exception:
        return None


def _resize_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    resized = pil.resize((shape[1], shape[0]), Image.NEAREST)
    return (np.asarray(resized) > 0).astype(bool)


def _editor_value_to_binary_mask(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    if not isinstance(value, dict):
        arr = _image_to_array(value)
        if arr is None:
            return np.zeros(shape, dtype=bool)
        if arr.ndim == 2:
            return _resize_binary_mask(arr > 0, shape)
        if arr.ndim == 3:
            return _resize_binary_mask(np.any(arr[..., :3] != 0, axis=2), shape)
        return np.zeros(shape, dtype=bool)

    background = _image_to_array(value.get("background"))
    composite = _image_to_array(value.get("composite"))
    layers = value.get("layers", [])
    candidate = np.zeros(shape, dtype=bool)

    def _merge_from_array(arr: np.ndarray | None) -> None:
        nonlocal candidate
        if arr is None:
            return
        if arr.ndim == 2:
            mask = arr > 0
        elif arr.ndim == 3 and arr.shape[2] == 4:
            mask = arr[..., 3] > 0
            if background is not None and background.shape[:2] == arr.shape[:2]:
                base = background[..., :3] if background.ndim == 3 else background
                mask |= np.any(arr[..., :3] != base[..., :3], axis=2)
        elif arr.ndim == 3:
            mask = np.any(arr[..., :3] != 0, axis=2)
            if background is not None and background.shape[:2] == arr.shape[:2]:
                base = background[..., :3] if background.ndim == 3 else background
                mask |= np.any(arr[..., :3] != base[..., :3], axis=2)
        else:
            return
        candidate |= _resize_binary_mask(mask, shape)

    _merge_from_array(composite)
    if isinstance(layers, list):
        for layer in layers:
            _merge_from_array(_image_to_array(layer))

    if not np.any(candidate) and composite is not None and background is not None:
        comp = _image_to_array(composite)
        base = _image_to_array(background)
        if comp is not None and base is not None and comp.ndim == 3 and base.ndim == 3:
            diff = np.any(comp[..., :3] != base[..., :3], axis=2)
            candidate |= _resize_binary_mask(diff, shape)

    return candidate


def _manual_source_labels(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    candidate_region: np.ndarray,
    target_label: str,
) -> tuple[str, ...]:
    labels: list[str] = []
    region_ids = np.unique(mask[candidate_region])
    for fine_id in region_ids:
        if int(fine_id) in schema.skip_fine_ids:
            continue
        label = None
        for name, fine_ids in schema.label_to_fine_ids.items():
            if int(fine_id) in fine_ids:
                label = name
                break
        if label and label != target_label and label not in labels:
            labels.append(label)
    return tuple(labels)


def _refresh_edit_mode_panels(
    state: dict[str, Any],
    edit_mode: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Return panel visibility and auto-recommendation UI state."""

    prompt_visible = gr.update(visible=edit_mode == EDIT_MODE_PROMPT)
    manual_visible = gr.update(visible=edit_mode == EDIT_MODE_MANUAL_CONTOUR)
    auto_visible = gr.update(visible=edit_mode == EDIT_MODE_AUTO_RECOMMEND)
    tissue_button_visible = gr.update(visible=edit_mode != EDIT_MODE_AUTO_RECOMMEND)
    auto_execute_visible = gr.update(visible=edit_mode == EDIT_MODE_AUTO_RECOMMEND)
    manual_editor_value = gr.update()
    manual_target_update = gr.update()
    auto_choices_update = gr.update(choices=[], value=[])
    auto_summary = gr.update(value="")

    if not state:
        return (
            state,
            prompt_visible,
            manual_visible,
            auto_visible,
            manual_editor_value,
            manual_target_update,
            auto_choices_update,
            auto_summary,
            tissue_button_visible,
            auto_execute_visible,
        )

    if edit_mode == EDIT_MODE_MANUAL_CONTOUR:
        schema = MaskProfileSchema.from_reference_profile(state.get("profile", "BCSS"))
        background_rgb = state.get("target_mask_rgb") or state.get("source_mask_rgb")
        manual_editor_value = gr.update(value=_manual_contour_editor_value(background_rgb))
        manual_target_update = gr.update(
            choices=_manual_target_label_choices(schema),
            value=_default_manual_target_label(schema),
        )
    else:
        manual_target_update = gr.update(choices=[], value=None)

    if edit_mode == EDIT_MODE_AUTO_RECOMMEND:
        auto_pool = _build_auto_recommendation_pool(state)
        state["auto_recommendations"] = auto_pool
        auto_choices = [(item["label"], item["key"]) for item in auto_pool]
        auto_choices_update = gr.update(choices=auto_choices, value=[])
        auto_summary = gr.update(value=_format_auto_recommendation_summary(auto_pool))
    else:
        state.pop("auto_recommendations", None)

    return (
        state,
        prompt_visible,
        manual_visible,
        auto_visible,
        manual_editor_value,
        manual_target_update,
        auto_choices_update,
        auto_summary,
        tissue_button_visible,
        auto_execute_visible,
    )


def _build_auto_recommendation_pool(state: dict[str, Any]) -> list[dict[str, Any]]:
    profile = state.get("profile", "BCSS")
    tissue_path = state.get("target_tissue_mask") or state.get("reference_tissue_mask")
    if not tissue_path:
        return []
    reference_tissue = load_id_mask(tissue_path)
    schema = MaskProfileSchema.from_reference_profile(profile)
    recipe = load_recipe(default_recipe_path_for_profile(profile))
    context = MaskEditContext.from_mask(reference_tissue, schema)
    recommendations = _recommend_edit_intents(reference_tissue, schema, recipe, context)
    pool: list[dict[str, Any]] = []
    for rank, (intent, primitive_config) in enumerate(recommendations[:AUTO_RECOMMEND_LIMIT], start=1):
        decision = assess_edit_applicability(intent, recipe, schema, context)
        key = _recommendation_key(intent)
        pool.append(
            {
                "key": key,
                "label": f"{rank}. {intent.primitive} / {intent.strength}",
                "primitive": intent.primitive,
                "strength": intent.strength,
                "reference_profile": intent.reference_profile,
                "source_labels": list(intent.source_labels),
                "target_label": intent.target_label,
                "score": _recommendation_score(primitive_config, context, decision),
                "status": decision.status,
                "reasons": list(decision.reasons),
                "warnings": list(decision.warnings),
            }
        )
    return pool


def _format_auto_recommendation_summary(pool: list[dict[str, Any]]) -> str:
    if not pool:
        return "No applicable primitive/strength combinations were found for this mask."
    lines = [f"{len(pool)} recommendations found."]
    for item in pool:
        source = ", ".join(item.get("source_labels", [])) or "-"
        target = item.get("target_label") or "-"
        lines.append(
            f"{item['label']} | score={item['score']} | status={item['status']}"
        )
        lines.append(f"source: {source} -> target: {target}")
    return "\n".join(lines)


def _recommendation_key(intent: EditIntent) -> str:
    return f"{intent.primitive}|{intent.strength}"


def _selected_recommendations_to_intents(
    state: dict[str, Any],
    selected_keys: list[str],
) -> list[tuple[EditIntent, dict[str, Any]]]:
    recommendations = state.get("auto_recommendations") or _build_auto_recommendation_pool(state)
    selected = {key for key in selected_keys if key}
    if not selected:
        return []

    profile = state.get("profile", "BCSS")
    schema = MaskProfileSchema.from_reference_profile(profile)
    recipe = load_recipe(default_recipe_path_for_profile(profile))

    intents: list[tuple[EditIntent, dict[str, Any]]] = []
    for item in recommendations:
        if item.get("key") not in selected:
            continue
        primitive_config = _primitive_config(recipe, str(item["primitive"]))
        payload = {
            "primitive": item["primitive"],
            "strength": item["strength"],
            "reference_profile": profile,
            "source_labels": item.get("source_labels", []),
            "target_label": item.get("target_label"),
        }
        intent = EditIntent.from_mapping(payload)
        intent = _with_default_contour_labels(intent, primitive_config, schema)
        intents.append((intent, primitive_config))
    return intents


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
    cellvit_script: str,
    cellvit_model: str,
    cellvit_root: str,
    cellvit_device: str,
) -> tuple[dict[str, Any], str, str | None, str | None]:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = _copy_input(source_image, output_dir, "source_image.png")
    tissue_path = _copy_input(source_tissue_mask, output_dir, "source_tissue_mask.png")
    nuclei_path = _file_path(source_cell_mask)
    if nuclei_path is None:
        nuclei_path = output_dir / "inputs" / "source_cell_mask.png"
        nuclei_path.parent.mkdir(parents=True, exist_ok=True)
        script_path = Path(_defaulted_text(cellvit_script, str(DEFAULT_CELLVIT_SCRIPT)))
        model_path = Path(_defaulted_text(cellvit_model, DEFAULT_CELLVIT_MODEL))
        root_path = Path(_defaulted_text(cellvit_root, str(DEFAULT_CELLVIT_ROOT)))
        command = [
            sys.executable,
            str(script_path),
            "--image",
            str(image_path),
            "--output-mask",
            str(nuclei_path),
            "--model",
            str(model_path),
            "--cellvit-root",
            str(root_path),
            "--gpu",
            str(_cuda_index(cellvit_device)),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            log_path = output_dir / "cellvit_error.log"
            log_path.write_text(_format_subprocess_error(exc, label="CellViT"), encoding="utf-8")
            raise gr.Error(_format_subprocess_error(exc, label="CellViT")) from exc
        log_text = "\n".join(
            part for part in [(result.stdout or "").strip(), (result.stderr or "").strip()] if part
        )
        if log_text:
            (output_dir / "cellvit.log").write_text(log_text, encoding="utf-8")
        if not nuclei_path.exists():
            raise gr.Error(f"CellViT finished but did not write {nuclei_path}")
    else:
        nuclei_path = _copy_input(source_cell_mask, output_dir, "source_cell_mask.png")

    image = _load_rgb_image(image_path)
    tissue = load_id_mask(tissue_path)
    nuclei = _load_uint8_mask(nuclei_path)
    _validate_same_size(image, tissue, "source_tissue_mask")
    _validate_same_size(image, nuclei, "source_cell_mask")

    source_rgb = str(
        _save_pre_generation_artifacts(
            output_dir=output_dir,
            reference_image=image,
            reference_tissue=tissue,
            target_tissue=tissue,
            change_region=np.zeros(tissue.shape, dtype=bool),
        )["source_mask_rgb"]
    )

    state = {
        "profile": profile,
        "output_dir": str(output_dir),
        "reference_image": str(image_path),
        "reference_tissue_mask": str(tissue_path),
        "reference_nuclei_mask": str(nuclei_path),
        "source_mask_rgb": str(source_rgb),
        "target_mask_rgb": str(source_rgb),
    }
    return state, _json_text({"status": "loaded", "output_dir": str(output_dir)}), str(image_path), source_rgb


def run_tissue_stage(
    state: dict[str, Any],
    edit_mode: str,
    old_prompt: str,
    new_prompt: str,
    manual_contour_editor,
    auto_recommendation_keys,
    parser: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    qwen_model_path: str,
    qwen_device: str,
    no_few_shot: bool,
    source_labels: str,
    target_label: str,
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
    reference_tissue = load_id_mask(state.get("target_tissue_mask") or state["reference_tissue_mask"])
    schema = MaskProfileSchema.from_reference_profile(state["profile"])
    recipe = load_recipe(default_recipe_path_for_profile(state["profile"]))
    try:
        if edit_mode == EDIT_MODE_MANUAL_CONTOUR:
            result, phase3_info = _run_manual_contour_stage(
                reference_tissue=reference_tissue,
                schema=schema,
                manual_contour_editor=manual_contour_editor,
                manual_target_label=target_label,
                output_dir=output_dir,
                source_labels=source_labels,
                primitive="manual_contour",
            )
        elif edit_mode == EDIT_MODE_AUTO_RECOMMEND:
            result, phase3_info = _execute_selected_recommendations(
                state=state,
                reference_tissue=reference_tissue,
                schema=schema,
                recipe=recipe,
                output_dir=output_dir,
                selected_keys=list(auto_recommendation_keys or []),
            )
        elif old_prompt.strip() and new_prompt.strip():
            semantic_diff, parser_info = _resolve_prompt_semantic_diff(
                old_prompt=old_prompt,
                new_prompt=new_prompt,
                parser=parser,
                api_base_url=api_base_url,
                api_key_env=api_key_env,
                api_model=api_model,
                qwen_model_path=qwen_model_path,
                qwen_device=qwen_device,
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
                api_model=_defaulted_text(api_model, DEFAULT_API_MODEL),
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
            raise gr.Error(
                "Prompt mode requires both old and new prompts. "
                "Choose manual contour or auto recommend for non-prompt edits."
            )
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
    elif phase3_info.get("mode") in {EDIT_MODE_MANUAL_CONTOUR, EDIT_MODE_AUTO_RECOMMEND}:
        phase3_info = _merge_phase3_info(phase3_info, result)
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
            "target_mask_rgb": stage_paths["target_mask_rgb"],
            "change_region": stage_paths["change_region"],
            "phase3": phase3_info,
        }
    )
    info = {
        "status": "tissue_done",
        "projection_mode": PROJECTION_MODE_ORGANIC_V2,
        "primitive": phase3_info.get("primitive", primitive),
        "edit_mode": edit_mode,
        "prompt_mode": edit_mode == EDIT_MODE_PROMPT and bool(old_prompt.strip() and new_prompt.strip()),
        "changed_area_fraction": _change_area_fraction(change_region),
        "target_tissue_mask": str(target_path),
        "change_region": stage_paths["change_region"],
    }
    return state, _json_text(info), stage_paths["target_mask_rgb"], stage_paths["change_region"]


def _run_manual_contour_stage(
    *,
    reference_tissue: np.ndarray,
    schema: MaskProfileSchema,
    manual_contour_editor,
    manual_target_label: str,
    output_dir: Path,
    source_labels: str,
    primitive: str,
) -> tuple[Any, dict[str, Any]]:
    candidate_region = _editor_value_to_binary_mask(manual_contour_editor, reference_tissue.shape)
    if not np.any(candidate_region):
        raise gr.Error("Draw a contour on the mask first.")

    target_label = manual_target_label.strip()
    if not target_label:
        target_label = _default_manual_target_label(schema)
    if target_label not in schema.writable_labels:
        raise gr.Error(f"Target label {target_label!r} is not writable for {schema.reference_profile}.")

    inferred_source_labels = _manual_source_labels(
        reference_tissue,
        schema,
        candidate_region,
        target_label,
    )
    override_source_labels = tuple(_filter_schema_labels(_split_csv(source_labels), schema))
    if override_source_labels:
        inferred_source_labels = tuple(
            label for label in override_source_labels if label != target_label
        )
    if not inferred_source_labels:
        raise gr.Error(
            "Could not infer source labels from the drawn contour. "
            "Draw over a visible source label or provide source labels."
        )

    edit_result = apply_projected_label_write(
        reference_tissue,
        candidate_region,
        schema=schema,
        source_labels=inferred_source_labels,
        target_label=target_label,
        backend="manual_contour",
        raw_payload={
            "primitive": primitive,
            "mode": EDIT_MODE_MANUAL_CONTOUR,
            "manual_target_label": target_label,
            "source_labels": list(inferred_source_labels),
        },
    )
    phase3_info = {
        "mode": EDIT_MODE_MANUAL_CONTOUR,
        "primitive": primitive,
        "reference_profile": schema.reference_profile,
        "target_label": target_label,
        "source_labels": list(inferred_source_labels),
        "selected_pixels": int(np.count_nonzero(edit_result.change_region)),
        "manual_contour": True,
        "projection_mode": "manual_projected_label_write",
        "result": {
            "ops_log": edit_result.ops_log,
            "warnings": list(edit_result.warnings),
        },
    }
    save_metadata(phase3_info, output_dir / "phase3_mask_edit" / "execution_summary.json")
    result = SimpleNamespace(
        status="validated",
        edit_result=edit_result,
        error=None,
        final_attempt=None,
        validation=None,
        projection_mode="manual_projected_label_write",
        to_metadata=lambda: dict(phase3_info),
    )
    return result, phase3_info


def _load_manual_contour_payload(manual_contour_json: str, manual_contour_file) -> dict[str, Any]:
    if manual_contour_file is not None:
        path = _file_path(manual_contour_file)
        if path is not None:
            return load_contour_proposal_json(path)
    text = (manual_contour_json or "").strip()
    if not text:
        raise gr.Error("Provide a manual contour JSON or upload a contour file.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Invalid contour JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise gr.Error("Manual contour JSON must be an object.")
    payload.setdefault("schema_version", CONTOUR_PROPOSAL_SCHEMA_VERSION)
    payload.setdefault("backend", CONTOUR_PROPOSAL_BACKEND)
    return payload


def _run_auto_recommend_stage(
    *,
    state: dict[str, Any],
    reference_tissue: np.ndarray,
    schema: MaskProfileSchema,
    recipe: dict[str, Any],
    output_dir: Path,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    context = MaskEditContext.from_mask(reference_tissue, schema)
    recommendations = _recommend_edit_intents(reference_tissue, schema, recipe, context)
    if not recommendations:
        raise gr.Error(
            f"No applicable primitive/strength combinations found for {schema.reference_profile}."
        )
    pool: list[dict[str, Any]] = []
    for rank, (intent, primitive_config) in enumerate(recommendations[:AUTO_RECOMMEND_LIMIT], start=1):
        decision = assess_edit_applicability(intent, recipe, schema, context)
        pool.append(
            {
                "rank": rank,
                "key": _recommendation_key(intent),
                "primitive": intent.primitive,
                "strength": intent.strength,
                "reference_profile": intent.reference_profile,
                "source_labels": list(intent.source_labels),
                "target_label": intent.target_label,
                "score": _recommendation_score(primitive_config, context, decision),
                "status": decision.status,
                "reasons": list(decision.reasons),
                "warnings": list(decision.warnings),
            }
        )
    result = SimpleNamespace(
        status="recommendations_ready",
        edit_result=None,
        error=None,
        final_attempt=None,
        validation=None,
        projection_mode="recommendation",
        to_metadata=lambda: {"mode": EDIT_MODE_AUTO_RECOMMEND, "recommendations": pool},
    )
    phase3_info = {
        "mode": EDIT_MODE_AUTO_RECOMMEND,
        "recommendations": pool,
        "projection_mode": "recommendation",
    }
    save_metadata(phase3_info, output_dir / "phase3_mask_edit" / "execution_summary.json")
    return result, phase3_info, pool


def _execute_selected_recommendations(
    *,
    state: dict[str, Any],
    reference_tissue: np.ndarray,
    schema: MaskProfileSchema,
    recipe: dict[str, Any],
    output_dir: Path,
    selected_keys: list[str],
) -> tuple[Any, dict[str, Any]]:
    recommendations = _build_auto_recommendation_pool(state)
    if not recommendations:
        raise gr.Error("No auto-recommendations available. Re-run the auto-recommend panel.")
    selected_intents = _selected_recommendations_to_intents(state, selected_keys)
    if not selected_intents:
        raise gr.Error("Select at least one recommendation before running auto recommend.")

    current_mask = np.array(reference_tissue, copy=True)
    attempt_logs: list[dict[str, Any]] = []
    last_result = None
    for intent, primitive_config in selected_intents[:AUTO_RECOMMEND_LIMIT]:
        decision = assess_edit_applicability(intent, recipe, schema, MaskEditContext.from_mask(current_mask, schema))
        if decision.status == "rejected":
            attempt_logs.append(
                {
                    "primitive": intent.primitive,
                    "strength": intent.strength,
                    "status": "rejected",
                    "reasons": list(decision.reasons),
                }
            )
            continue
        result = execute_edit(
            current_mask,
            intent,
            recipe,
            schema,
            MaskEditContext.from_mask(current_mask, schema),
        )
        attempt_logs.append(
            {
                "primitive": intent.primitive,
                "strength": intent.strength,
                "status": result.status,
                "reasons": list(result.applicability.reasons),
                "warnings": list(result.applicability.warnings),
                "primitive_config": primitive_config.get("name"),
            }
        )
        last_result = result
        if result.edit_result is not None:
            current_mask = np.array(result.edit_result.target_mask, copy=True)

    if last_result is None or last_result.edit_result is None:
        raise gr.Error("Selected recommendations could not be executed.")

    phase3_info = {
        "mode": EDIT_MODE_AUTO_RECOMMEND,
        "recommendations": recommendations,
        "selected_keys": list(selected_keys),
        "attempts": attempt_logs,
        "primitive": last_result.edit_result.ops_log.get("primitive", last_result.applicability.primitive),
        "projection_mode": "primitive_executor",
    }
    save_metadata(phase3_info, output_dir / "phase3_mask_edit" / "execution_summary.json")
    return last_result, phase3_info


def _recommend_edit_intents(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    recipe: dict[str, Any],
    context: MaskEditContext,
) -> list[tuple[EditIntent, dict[str, Any]]]:
    candidates: list[tuple[int, int, EditIntent, dict[str, Any]]] = []
    for primitive_config in recipe.get("primitives", []):
        if not isinstance(primitive_config, dict):
            continue
        primitive_name = primitive_config.get("name")
        if not isinstance(primitive_name, str):
            continue
        strengths = _primitive_strengths(primitive_config)
        if not strengths:
            strengths = ("moderate",)
        for strength in strengths:
            intent = EditIntent(
                primitive=primitive_name,
                strength=strength,
                reference_profile=schema.reference_profile,
            )
            intent = _with_default_contour_labels(intent, primitive_config, schema)
            decision = assess_edit_applicability(intent, recipe, schema, context)
            if decision.status == "rejected":
                continue
            score = _recommendation_score(primitive_config, context, decision)
            candidates.append((score, _STRONGEST_FIRST.get(strength, 0), intent, primitive_config))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2].primitive))
    return [(intent, primitive_config) for _, _, intent, primitive_config in candidates]


def _primitive_strengths(primitive_config: dict[str, Any]) -> tuple[str, ...]:
    ranges = primitive_config.get("parameter_ranges", {})
    if not isinstance(ranges, dict):
        return ()
    strengths = [key for key in ("mild", "moderate", "significant", "xlarge_deid") if key in ranges]
    return tuple(strengths)


def _recommendation_score(
    primitive_config: dict[str, Any],
    context: MaskEditContext,
    decision,
) -> int:
    score = 0
    if decision.status == "degraded":
        score += 1
    mask_operation = primitive_config.get("mask_operation", {})
    if isinstance(mask_operation, dict):
        source = mask_operation.get("source")
        if isinstance(source, str) and source in context.present_labels:
            score += 3
        target = mask_operation.get("target")
        if isinstance(target, str) and target in context.present_labels:
            score += 1
    score += min(4, len(decision.warnings))
    score += min(4, len(decision.fallback_actions))
    return score


def _merge_phase3_info(base: dict[str, Any], result: Any) -> dict[str, Any]:
    phase3_info = dict(base)
    if hasattr(result, "to_metadata"):
        try:
            phase3_info["result"] = result.to_metadata()
        except Exception:
            pass
    return phase3_info


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
    profile_defaults = _profile_defaults(state.get("profile", "BCSS"))
    args = _make_args(
        state,
        cell_fill_mode=cell_fill_mode,
        crossing_cell_policy=crossing_cell_policy,
        probnet_ckpt=Path(_defaulted_text(probnet_ckpt, profile_defaults["probnet_ckpt"])),
        nuclei_library=Path(_defaulted_text(nuclei_library, profile_defaults["nuclei_library"])),
        density_scale_json=Path(_defaulted_text(density_scale_json, profile_defaults["density_scale_json"])),
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
    cross_backend: str,
    route_threshold: float,
    model_path: str,
    inpaint_checkpoint: str,
    cross_checkpoint: str,
    cross_v1_checkpoint: str,
    uni_checkpoint: str,
    device: str,
) -> tuple[dict[str, Any], str, str, str]:
    if not state or not state.get("target_nuclei_mask"):
        raise gr.Error("Run the cell-mask stage first.")
    output_dir = Path(state["output_dir"])
    reference_image = _load_rgb_image(state["reference_image"])
    change_region = load_change_region(state["change_region"])
    args = _make_args(
        state,
        generation_mode=generation_mode,
        cross_backend=cross_backend,
        route_threshold=route_threshold,
        pretrained_model_name_or_path=_defaulted_text(model_path, DEFAULT_PRETRAINED_MODEL),
        inpaint_checkpoint=Path(_defaulted_text(inpaint_checkpoint, DEFAULT_INPAINT_CHECKPOINT)),
        cross_checkpoint=Path(_defaulted_text(cross_checkpoint, DEFAULT_CROSS_CHECKPOINT)),
        cross_v1_checkpoint=Path(_defaulted_text(cross_v1_checkpoint, DEFAULT_CROSS_V1_CHECKPOINT)),
        uni_checkpoint=Path(_defaulted_text(uni_checkpoint, DEFAULT_UNI_CHECKPOINT)),
        device=device or GENERATION_DEVICE_CHOICES[0],
        prompt=None,
    )
    change_ratio = _change_area_fraction(change_region)
    selected_mode = _select_generation_mode(
        generation_mode,
        change_ratio,
        route_threshold,
        cross_backend=cross_backend,
    )
    _validate_generation_paths(args, selected_mode)
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
        prompt=str(generation_info.get("prompt", "")),
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


def _validate_generation_paths(args: SimpleNamespace, selected_mode: str) -> None:
    required_paths: dict[str, Path] = {}
    if selected_mode == "inpaint":
        required_paths["inpaint checkpoint"] = Path(args.inpaint_checkpoint)
    elif selected_mode == "cross-v0":
        required_paths["cross-v0 checkpoint"] = Path(args.cross_checkpoint)
    elif selected_mode == "cross-v1":
        required_paths["cross-v1 checkpoint"] = Path(args.cross_v1_checkpoint)
        required_paths["UNI checkpoint"] = Path(args.uni_checkpoint)

    missing = [f"{label}: {path}" for label, path in required_paths.items() if not path.exists()]
    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise gr.Error(
            f"Selected generation mode '{selected_mode}' is missing required files:\n"
            f"{detail}\n"
            "Update the corresponding path in Advanced generation inputs, or choose dry-run."
        )


def preview_route(state: dict[str, Any], threshold: float, cross_backend: str) -> str:
    if not state or not state.get("change_region"):
        return "Run the tissue stage first."
    change_region = load_change_region(state["change_region"])
    ratio = _change_area_fraction(change_region)
    selected = _select_generation_mode("auto", ratio, threshold, cross_backend=cross_backend)
    return f"change_region = {ratio:.2%}; auto route = {selected} (threshold {threshold:.0%})"


def check_cuda_memory() -> str:
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            query,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return "nvidia-smi not found. CUDA memory cannot be checked on this machine."
    except subprocess.TimeoutExpired:
        return "nvidia-smi timed out while checking CUDA memory."
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        return f"nvidia-smi failed: {detail or exc}"

    lines: list[str] = []
    visible_devices = _cuda_visible_devices_summary()
    if visible_devices:
        lines.append(visible_devices)
    for raw_line in result.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 4:
            continue
        index, name, free_mib, total_mib = parts
        lines.append(f"physical GPU {index}  {free_mib} / {total_mib} MiB free  {name}")
    return "\n".join(lines) if lines else "No CUDA GPU memory rows returned by nvidia-smi."


def _cuda_visible_devices_summary() -> str:
    try:
        import torch

        torch_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        torch_count = 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    parts = [f"PyTorch visible devices: cuda:0..cuda:{torch_count - 1}" if torch_count else "PyTorch visible devices: none"]
    if visible is not None:
        parts.append(f"CUDA_VISIBLE_DEVICES={visible}")
    parts.append("nvidia-smi rows below use physical GPU indexes")
    return "; ".join(parts)


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
    schema = MaskProfileSchema.from_reference_profile(profile)
    operation = primitive_config.get("mask_operation", {})
    source = _filter_schema_labels(
        _split_csv(source_labels) or _default_contour_sources(primitive_config, operation),
        schema,
    )
    target = _schema_label_or_none(target_label.strip() if target_label else "", schema) or ""
    if not target:
        target = _default_contour_target(primitive_config, operation, schema=schema)
    target = _schema_label_or_none(target, schema) or ""
    if not source:
        raise gr.Error(
            "Please provide at least one source label available in "
            f"profile {schema.reference_profile}. Readable labels are: "
            f"{', '.join(sorted(schema.readable_labels))}."
        )
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
    source = tuple(
        _filter_schema_labels(
            tuple(intent.source_labels)
            or tuple(_default_contour_sources(primitive_config, operation)),
            schema,
        )
    )
    target = _schema_label_or_none(
        intent.target_label
        or _default_contour_target(
            primitive_config,
            operation,
            schema=schema,
        ),
        schema,
    )
    preserve = tuple(_filter_schema_labels(intent.preserve_labels, schema))
    forbidden = tuple(_filter_schema_labels(intent.forbidden_labels, schema))
    if not target:
        target = schema.choose_default_backfill_label(exclude_labels=source)
    if not source:
        raise gr.Error(
            "No valid source labels remain for "
            f"profile {schema.reference_profile} and primitive {intent.primitive}. "
            f"Readable labels are: {', '.join(sorted(schema.readable_labels))}."
        )
    if (
        source == list(intent.source_labels)
        and target == intent.target_label
        and preserve == tuple(intent.preserve_labels)
        and forbidden == tuple(intent.forbidden_labels)
    ):
        return intent
    payload = intent.to_metadata()
    payload["source_labels"] = list(source)
    payload["target_label"] = target
    payload["preserve_labels"] = list(preserve)
    payload["forbidden_labels"] = list(forbidden)
    return EditIntent.from_mapping(payload)


def _filter_schema_labels(labels: tuple[str, ...] | list[str], schema: MaskProfileSchema) -> list[str]:
    return [label for label in labels if label in schema.readable_labels]


def _schema_label_or_none(label: str | None, schema: MaskProfileSchema) -> str | None:
    if label and label in schema.writable_labels:
        return label
    return None


def _build_contour_provider(
    *,
    provider: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    api_image_detail: str,
    fixture_file,
):
    api_model = _defaulted_text(api_model, DEFAULT_API_MODEL)
    api_base_url = _defaulted_text(api_base_url, DEFAULT_API_BASE_URL).rstrip("/")
    api_key_env = _defaulted_text(api_key_env, DEFAULT_API_KEY_ENV)
    if provider == "api-text":
        return OpenAICompatibleTextContourProvider(
            model=api_model,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
        )
    if provider == "api-multimodal":
        return OpenAICompatibleMultimodalContourProvider(
            model=api_model,
            api_base_url=api_base_url,
            api_key_env=api_key_env,
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
    qwen_device: str = DEFAULT_QWEN_DEVICE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if parser == "api":
        api_model = _defaulted_text(api_model, DEFAULT_API_MODEL)
        config = ApiParserConfig(
            model=api_model,
            api_base_url=_defaulted_text(api_base_url, DEFAULT_API_BASE_URL).rstrip("/"),
            api_key_env=_defaulted_text(api_key_env, DEFAULT_API_KEY_ENV),
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
            device=qwen_device or DEFAULT_QWEN_DEVICE,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=not no_few_shot,
            use_few_shot=not no_few_shot,
        )
        return parse_prompts_with_qwen_local(old_prompt, new_prompt, config=config), {
            "mode": "qwen-local",
            "model_path": qwen_model_path,
            "device": config.device,
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
        with gr.Row():
            source_image = gr.File(label="src_image", file_types=["image"], type="filepath")
            source_tissue = gr.File(label="src_tissue_mask", file_types=["image"], type="filepath")
            source_cell = gr.File(label="src_cell_mask / CellViT output", file_types=["image"], type="filepath")
        load_button = gr.Button("1. Load inputs")
        load_log = gr.Code(label="load log", language="json")
        with gr.Row():
            src_image_preview = gr.Image(label="source image")
            src_tissue_preview = gr.Image(label="source tissue")

        gr.Markdown("### Tissue mask edit")
        edit_mode = gr.Radio(
            EDIT_MODE_CHOICES,
            value=EDIT_MODE_PROMPT,
            label="edit mode",
        )
        prompt_panel = gr.Column(visible=True)
        with prompt_panel:
            with gr.Row():
                old_prompt = gr.Textbox(label="src prompt", lines=3)
                new_prompt = gr.Textbox(label="new prompt", lines=3)
            with gr.Row():
                parser = gr.Radio(["api", "qwen-local"], value="api", label="parser")
                no_few_shot = gr.Checkbox(value=False, label="no few shot")
            with gr.Row():
                api_model = gr.Textbox(value=DEFAULT_API_MODEL, label="api model")
                qwen_model_path = gr.Textbox(label="qwen model path")
            with gr.Accordion("Advanced parser inputs", open=False):
                with gr.Row():
                    api_base_url = gr.Textbox(value=DEFAULT_API_BASE_URL, label="api base url")
                    api_key_env = gr.Textbox(value=DEFAULT_API_KEY_ENV, label="api key env")
                with gr.Row():
                    qwen_device = gr.Dropdown(CUDA_DEVICE_CHOICES, value=DEFAULT_QWEN_DEVICE, label="qwen device")
                    cuda_memory_button = gr.Button("Check CUDA memory")
                cuda_memory_log = gr.Textbox(label="CUDA memory", lines=8, interactive=False)
                with gr.Row():
                    cellvit_script = gr.Textbox(value=str(DEFAULT_CELLVIT_SCRIPT), label="CellViT runner script")
                    cellvit_model = gr.Textbox(value=DEFAULT_CELLVIT_MODEL, label="CellViT model")
                with gr.Row():
                    cellvit_root = gr.Textbox(value=str(DEFAULT_CELLVIT_ROOT), label="CellViT source root")
                    cellvit_device = gr.Dropdown(CUDA_DEVICE_CHOICES, value=DEFAULT_CELLVIT_DEVICE, label="CellViT device")

        manual_panel = gr.Column(visible=False)
        with manual_panel:
            manual_editor = gr.ImageEditor(
                label="draw contour on mask",
                sources=["upload", "clipboard"],
                type="numpy",
                value=None,
            )
            with gr.Row():
                target_label = gr.Dropdown([], value=None, label="target label")
                source_labels = gr.Textbox(label="source labels", placeholder="Stroma")
            with gr.Row():
                primitive = gr.Dropdown(
                    [
                        "manual_contour",
                        "stromal_immune_infiltration",
                        "necrosis_appearance",
                        "tumor_burden_increase",
                        "tumor_burden_decrease",
                        "immune_infiltration_decrease",
                        "stromal_desmoplasia",
                        "stroma_decrease",
                        "stromal_reduction",
                    ],
                    value="manual_contour",
                    label="manual primitive tag",
                )

        auto_panel = gr.Column(visible=False)
        with auto_panel:
            auto_choices = gr.CheckboxGroup(label="recommendations", choices=[])
            auto_summary = gr.Textbox(label="recommendation summary", lines=6, interactive=False)

        continue_on_failure = gr.Checkbox(value=False, label="continue on Phase3 failure")
        tissue_button = gr.Button("2. Run tissue-stage edit")
        tissue_log = gr.Code(label="tissue log", language="json")
        with gr.Row():
            target_tissue_preview = gr.Image(label="target tissue")
            change_region_preview = gr.Image(label="change region")
        with gr.Accordion("Advanced overrides", open=False):
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

        auto_execute_button = gr.Button("Run selected recommendations", visible=False)

        gr.Markdown("### Cell mask synthesis")
        with gr.Row():
            cell_fill = gr.Radio(["probnet", "blank", "preserve"], value="probnet", label="cell fill")
            crossing_policy = gr.Radio(["delete", "majority", "keep"], value="delete", label="crossing source-cell policy")
        profile_default_values = _profile_defaults("BCSS")
        with gr.Accordion("Advanced ProbNet inputs", open=False):
            with gr.Row():
                probnet_ckpt = gr.Textbox(value=profile_default_values["probnet_ckpt"], label="ProbNet checkpoint")
                nuclei_library = gr.Textbox(value=profile_default_values["nuclei_library"], label="nuclei library directory")
                density_scale_json = gr.Textbox(value=profile_default_values["density_scale_json"], label="density scale JSON")
            probnet_device = gr.Dropdown(PROBNET_DEVICE_CHOICES, value="auto", label="ProbNet device")
            gamma_values = gr.Textbox(value="1.0", label="gamma values")
        cell_button = gr.Button("3. Build target cell mask")
        cell_log = gr.Code(label="cell log", language="json")
        with gr.Row():
            retained_preview = gr.Image(label="retained source cells")
            new_cells_preview = gr.Image(label="new cells")
            combined_preview = gr.Image(label="target tissue + cells")

        gr.Markdown("### Image generation")
        with gr.Row():
            generation_mode = gr.Radio(["dry-run", "auto", "inpaint", "cross-v0", "cross-v1"], value="dry-run", label="generation mode")
            cross_backend = gr.Radio(["cross-v0", "cross-v1"], value="cross-v1", label="auto cross backend")
            route_threshold = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="inpaint if change > threshold")
        route_button = gr.Button("Preview route")
        route_log = gr.Textbox(label="route")
        with gr.Accordion("Advanced generation inputs", open=False):
            with gr.Row():
                model_path = gr.Textbox(value=DEFAULT_PRETRAINED_MODEL, label="pretrained FLUX/model path")
                device = gr.Dropdown(GENERATION_DEVICE_CHOICES, value=GENERATION_DEVICE_CHOICES[0], label="device")
            with gr.Row():
                inpaint_checkpoint = gr.Textbox(value=DEFAULT_INPAINT_CHECKPOINT, label="inpaint checkpoint")
                cross_checkpoint = gr.Textbox(value=DEFAULT_CROSS_CHECKPOINT, label="cross-v0 checkpoint")
                cross_v1_checkpoint = gr.Textbox(value=DEFAULT_CROSS_V1_CHECKPOINT, label="cross-v1 checkpoint")
            with gr.Row():
                uni_checkpoint = gr.Textbox(value=DEFAULT_UNI_CHECKPOINT, label="UNI checkpoint")
        generate_button = gr.Button("4. Route + generate")
        generation_log = gr.Code(label="summary", language="json")
        with gr.Row():
            generated_preview = gr.Image(label="generated image")
            panel_preview = gr.Image(label="compare panel")

        load_button.click(
            load_inputs,
            inputs=[
                profile,
                source_image,
                source_tissue,
                source_cell,
                cellvit_script,
                cellvit_model,
                cellvit_root,
                cellvit_device,
            ],
            outputs=[state, load_log, src_image_preview, src_tissue_preview],
        ).then(
            _refresh_edit_mode_panels,
            inputs=[state, edit_mode],
            outputs=[
                state,
                prompt_panel,
                manual_panel,
                auto_panel,
                manual_editor,
                target_label,
                auto_choices,
                auto_summary,
                tissue_button,
                auto_execute_button,
            ],
        )
        edit_mode.change(
            _refresh_edit_mode_panels,
            inputs=[state, edit_mode],
            outputs=[
                state,
                prompt_panel,
                manual_panel,
                auto_panel,
                manual_editor,
                target_label,
                auto_choices,
                auto_summary,
                tissue_button,
                auto_execute_button,
            ],
        )
        profile.change(
            lambda value: tuple(_profile_defaults(value).values()),
            inputs=[profile],
            outputs=[probnet_ckpt, nuclei_library, density_scale_json],
        )
        cuda_memory_button.click(check_cuda_memory, inputs=[], outputs=[cuda_memory_log])
        tissue_button.click(
            run_tissue_stage,
            inputs=[
                state,
                edit_mode,
                old_prompt,
                new_prompt,
                manual_editor,
                auto_choices,
                parser,
                api_base_url,
                api_key_env,
                api_model,
                qwen_model_path,
                qwen_device,
                no_few_shot,
                source_labels,
                target_label,
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
        ).then(
            _refresh_edit_mode_panels,
            inputs=[state, edit_mode],
            outputs=[
                state,
                prompt_panel,
                manual_panel,
                auto_panel,
                manual_editor,
                target_label,
                auto_choices,
                auto_summary,
                tissue_button,
                auto_execute_button,
            ],
        )
        auto_execute_button.click(
            _run_auto_selected_from_ui,
            inputs=[state, auto_choices],
            outputs=[state, tissue_log, target_tissue_preview, change_region_preview],
        ).then(
            _refresh_edit_mode_panels,
            inputs=[state, edit_mode],
            outputs=[
                state,
                prompt_panel,
                manual_panel,
                auto_panel,
                manual_editor,
                target_label,
                auto_choices,
                auto_summary,
                tissue_button,
                auto_execute_button,
            ],
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
        route_button.click(preview_route, inputs=[state, route_threshold, cross_backend], outputs=[route_log])
        generate_button.click(
            run_generation_stage,
            inputs=[
                state,
                generation_mode,
                cross_backend,
                route_threshold,
                model_path,
                inpaint_checkpoint,
                cross_checkpoint,
                cross_v1_checkpoint,
                uni_checkpoint,
                device,
            ],
            outputs=[state, generation_log, generated_preview, panel_preview],
        )
    return demo


def _run_auto_selected_from_ui(
    state: dict[str, Any],
    auto_recommendation_keys,
) -> tuple[dict[str, Any], str, str, str]:
    if not state:
        raise gr.Error("Load inputs first.")
    output_dir = Path(state["output_dir"])
    reference_tissue = load_id_mask(state.get("target_tissue_mask") or state["reference_tissue_mask"])
    schema = MaskProfileSchema.from_reference_profile(state["profile"])
    recipe = load_recipe(default_recipe_path_for_profile(state["profile"]))
    result, phase3_info = _execute_selected_recommendations(
        state=state,
        reference_tissue=reference_tissue,
        schema=schema,
        recipe=recipe,
        output_dir=output_dir,
        selected_keys=list(auto_recommendation_keys or []),
    )
    target_tissue = result.edit_result.target_mask
    target_path = save_id_mask(target_tissue, output_dir / "target_mask.png")
    _validate_same_size(_load_rgb_image(state["reference_image"]), target_tissue, "target_tissue_mask")
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
            "target_mask_rgb": stage_paths["target_mask_rgb"],
            "change_region": stage_paths["change_region"],
            "phase3": phase3_info,
        }
    )
    info = {
        "status": "tissue_done",
        "edit_mode": EDIT_MODE_AUTO_RECOMMEND,
        "target_tissue_mask": str(target_path),
        "change_region": stage_paths["change_region"],
    }
    return state, _json_text(info), stage_paths["target_mask_rgb"], stage_paths["change_region"]


def main() -> None:
    build_ui().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
