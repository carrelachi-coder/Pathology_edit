#!/usr/bin/env python3
"""Local step-by-step UI for the Phase3 -> Phase4 -> Phase5 edit chain."""

from __future__ import annotations

import base64
import html as html_lib
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
from PIL import ImageDraw

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
    STATUS_VALIDATED,
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
from phase3_mask_edit.backends.organic_projection import apply_organic_projected_label_write
from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.applicability import assess_edit_applicability
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import (
    id_to_rgb,
    load_change_region,
    load_id_mask,
    save_change_region,
    save_id_mask,
    save_metadata,
)
from phase3_mask_edit.parser.api_parser import ApiParserConfig, parse_prompts_with_api
from phase3_mask_edit.parser.instruction_parser import (
    InstructionParserConfig,
    parse_instruction,
)
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
EDIT_MODE_INSTRUCTION = "instruction"
EDIT_MODE_MANUAL_CONTOUR = "manual_contour"
EDIT_MODE_AUTO_RECOMMEND = "auto_recommend"
EDIT_MODE_CHOICES = [
    EDIT_MODE_PROMPT,
    EDIT_MODE_INSTRUCTION,
    EDIT_MODE_MANUAL_CONTOUR,
    EDIT_MODE_AUTO_RECOMMEND,
]
AUTO_RECOMMEND_DEFAULT_PRIMITIVE = "tumor_burden_increase"
AUTO_RECOMMEND_DEFAULT_STRENGTH = "moderate"
MANUAL_CONTOUR_MAX_COMPONENTS = 24
MANUAL_CONTOUR_MAX_POINTS = 32


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


def _manual_contour_editor_value(
    background_rgb: str | None,
    components: list[dict[str, Any]] | None,
    shape: tuple[int, int] | None = None,
    *,
    target_label: str | None = None,
    target_color: tuple[int, int, int] | None = None,
) -> str:
    return _manual_contour_editor_html(
        background_rgb,
        components,
        shape=shape,
        target_label=target_label,
        target_color=target_color,
    )


def _manual_contour_payload_value(
    background_rgb: str | None,
    components: list[dict[str, Any]] | None,
    shape: tuple[int, int] | None = None,
    *,
    target_label: str | None = None,
    target_color: tuple[int, int, int] | None = None,
) -> str:
    return json.dumps(
        {
            "background": _background_data_uri(background_rgb),
            "components": components or [],
            "height": int(shape[0]) if shape else None,
            "width": int(shape[1]) if shape else None,
            "target_label": target_label or "",
            "target_color": list(target_color or (59, 130, 246)),
            "dirty": False,
        },
        ensure_ascii=False,
    )


def _encode_data_uri(path: str | Path) -> str:
    data = Path(path).read_bytes()
    mime = "image/png"
    if str(path).lower().endswith((".jpg", ".jpeg")):
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _background_data_uri(value: str | Path | None) -> str:
    if not value:
        return ""
    text = str(value)
    if text.startswith("data:"):
        return text
    return _encode_data_uri(value)


def _rgb_array_data_uri(rgb: np.ndarray) -> str:
    import io

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _label_color(schema: MaskProfileSchema, label: str | None) -> tuple[int, int, int]:
    if label:
        fine_ids = schema.label_to_fine_ids.get(label, ())
        for fine_id in fine_ids:
            rgb = id_to_rgb(np.asarray([[fine_id]], dtype=np.int64))[0, 0]
            return tuple(int(channel) for channel in rgb)
    return (59, 130, 246)


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _simplify_closed_contour(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    try:
        from skimage.measure import approximate_polygon
    except Exception:
        step = max(1, int(np.ceil(len(points) / max_points)))
        indices = np.arange(0, len(points), step, dtype=int)[:max_points]
        return points[indices]

    tolerance = 0.75
    simplified = np.asarray(points, dtype=float)
    for _ in range(12):
        candidate = np.asarray(approximate_polygon(points, tolerance=tolerance), dtype=float)
        if len(candidate) >= 2 and np.allclose(candidate[0], candidate[-1]):
            candidate = candidate[:-1]
        if 3 <= len(candidate) <= max_points:
            simplified = candidate
            break
        if len(candidate) < 3:
            break
        simplified = candidate
        tolerance *= 1.35
    if len(simplified) > max_points:
        indices = np.linspace(0, len(simplified) - 1, num=max_points, endpoint=False, dtype=int)
        simplified = simplified[indices]
    return simplified


def _extract_manual_contour_components(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    label_filter: str | None = None,
    max_components: int = MANUAL_CONTOUR_MAX_COMPONENTS,
    max_points: int = MANUAL_CONTOUR_MAX_POINTS,
) -> list[dict[str, Any]]:
    from scipy import ndimage
    from skimage import measure

    components: list[dict[str, Any]] = []
    for label_name in _ordered_schema_labels(schema):
        if label_name == "Background":
            continue
        if label_filter and label_name != label_filter:
            continue
        fine_ids = schema.label_to_fine_ids.get(label_name, ())
        label_mask = np.isin(mask, fine_ids)
        if not np.any(label_mask):
            continue
        labeled, count = ndimage.label(label_mask)
        for component_id in range(1, count + 1):
            component = labeled == component_id
            area = int(np.count_nonzero(component))
            if area == 0:
                continue
            ys, xs = np.where(component)
            padded_component = np.pad(component.astype(float), pad_width=1, mode="constant", constant_values=0)
            contours = measure.find_contours(padded_component, level=0.5)
            if not contours:
                continue
            contour = max(contours, key=len)
            contour_xy = np.asarray(
                [[float(point[1] - 1.0), float(point[0] - 1.0)] for point in contour],
                dtype=float,
            )
            contour_xy[:, 0] = np.clip(contour_xy[:, 0], 0.0, float(mask.shape[1] - 1))
            contour_xy[:, 1] = np.clip(contour_xy[:, 1], 0.0, float(mask.shape[0] - 1))
            if len(contour_xy) < 3:
                continue
            if _polygon_area(contour_xy) < 0:
                contour_xy = contour_xy[::-1]
            contour_xy = _simplify_closed_contour(contour_xy, max_points=max_points)
            components.append(
                {
                    "component_id": f"{label_name}_{component_id}",
                    "label": label_name,
                    "area_px": area,
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                    "points": contour_xy.round(2).tolist(),
                }
            )
    components.sort(key=lambda item: int(item["area_px"]), reverse=True)
    return components[:max_components]


def _manual_contour_editor_html(
    background_rgb: str | None,
    components: list[dict[str, Any]] | None,
    *,
    shape: tuple[int, int] | None = None,
    target_label: str | None = None,
    target_color: tuple[int, int, int] | None = None,
) -> str:
    payload = {
        "background": _background_data_uri(background_rgb),
        "components": components or [],
        "height": int(shape[0]) if shape else 768,
        "width": int(shape[1]) if shape else 1024,
        "target_label": target_label or "",
        "target_color": list(target_color or (59, 130, 246)),
    }
    srcdoc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #fff; font-family: sans-serif; }}
    .wrap {{ display: flex; flex-direction: column; gap: 8px; padding: 8px; box-sizing: border-box; width: 100%; height: 100%; }}
    .bar {{ display:flex; justify-content:space-between; align-items:center; }}
    .grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:12px; flex:1; min-height:0; }}
    .pane {{ display:flex; flex-direction:column; gap:6px; min-width:0; min-height:0; }}
    .title {{ font-size: 13px; color:#374151; }}
    .stage {{ position: relative; width:100%; aspect-ratio: {payload["width"]} / {payload["height"]}; max-height:100%; border: 1px solid #d1d5db; overflow: hidden; background: #f8fafc; }}
    img, svg, canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }}
    svg {{ touch-action: none; }}
    button {{ border:1px solid #d1d5db; background:#fff; border-radius:6px; padding:4px 8px; cursor:pointer; }}
    button.active {{ background:#f97316; color:#fff; border-color:#f97316; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <div>Drag contour points</div>
      <button id="drawModeButton" type="button">Add shape</button>
      <div id="status"></div>
    </div>
    <div class="grid">
      <div class="pane">
        <div class="title">edit mask contour</div>
        <div class="stage">
          <img id="bg" />
          <svg id="svg" viewBox="0 0 {payload["width"]} {payload["height"]}" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
      </div>
      <div class="pane">
        <div class="title">live color-mask preview</div>
        <div class="stage">
          <canvas id="preview" width="{payload["width"]}" height="{payload["height"]}"></canvas>
        </div>
      </div>
    </div>
  </div>
  <script>
    const seed = {json.dumps(payload, ensure_ascii=False)};
    const status = document.getElementById('status');
    const svg = document.getElementById('svg');
    const bg = document.getElementById('bg');
    const drawModeButton = document.getElementById('drawModeButton');
    const preview = document.getElementById('preview');
    const previewCtx = preview.getContext('2d');
    let payloadBox = null;
    let backgroundLoaded = false;
    const state = {{
      components: (seed.components || []).map((component) => ({{
        ...component,
        points: (component.points || []).map((point) => [Number(point[0]), Number(point[1])]),
      }})),
      active: null,
      drawing: null,
      drawMode: false,
      dirty: false,
    }};
    const ns = 'http://www.w3.org/2000/svg';
    const targetColor = seed.target_color || [59, 130, 246];
    const targetFill = `rgba(${{targetColor[0]}},${{targetColor[1]}},${{targetColor[2]}},0.24)`;
    const targetStroke = `rgb(${{targetColor[0]}},${{targetColor[1]}},${{targetColor[2]}})`;
    if (seed.background) {{
      bg.onload = () => {{
        backgroundLoaded = true;
        renderPreview();
      }};
      bg.src = seed.background;
    }}

    function resolvePayloadBox() {{
      const parentDocument = window.parent.document;
      if (payloadBox && parentDocument.contains(payloadBox)) return payloadBox;
      payloadBox = parentDocument.querySelector(
        '#manualContourPayload textarea, #manualContourPayload input, textarea#manualContourPayload, input#manualContourPayload'
      );
      return payloadBox;
    }}

    function emit() {{
      const parentWindow = window.parent;
      const box = resolvePayloadBox();
      if (!box) {{
        status.textContent = 'payload bridge missing';
        return;
      }}
      const nextValue = JSON.stringify({{
        background: seed.background,
        components: state.components,
        height: seed.height,
        width: seed.width,
        target_label: seed.target_label,
        target_color: seed.target_color,
        dirty: state.dirty,
      }});
      const proto = box.tagName && box.tagName.toLowerCase() === 'textarea'
        ? parentWindow.HTMLTextAreaElement.prototype
        : parentWindow.HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
      if (setter) setter.call(box, nextValue);
      else box.value = nextValue;
      box.dispatchEvent(new parentWindow.Event('input', {{ bubbles: true }}));
      box.dispatchEvent(new parentWindow.Event('change', {{ bubbles: true }}));
    }}

    function render() {{
      svg.innerHTML = '';
      if (state.drawing && state.drawing.length) {{
        const path = document.createElementNS(ns, 'polyline');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', '#f97316');
        path.setAttribute('stroke-width', '2');
        path.setAttribute('points', state.drawing.map((p) => `${{p[0]}},${{p[1]}}`).join(' '));
        svg.appendChild(path);
      }}
      state.components.forEach((component, compIndex) => {{
        if (!component.points || component.points.length < 3) return;
        const polygon = document.createElementNS(ns, 'polygon');
        polygon.setAttribute('fill', targetFill);
        polygon.setAttribute('stroke', targetStroke);
        polygon.setAttribute('stroke-width', '2');
        polygon.setAttribute('points', component.points.map((p) => `${{p[0]}},${{p[1]}}`).join(' '));
        svg.appendChild(polygon);
        component.points.forEach((point, pointIndex) => {{
          const handle = document.createElementNS(ns, 'circle');
          handle.setAttribute('cx', point[0]);
          handle.setAttribute('cy', point[1]);
          handle.setAttribute('r', '5');
          handle.setAttribute('fill', state.active && state.active.compIndex === compIndex && state.active.pointIndex === pointIndex ? '#f97316' : '#fff');
          handle.setAttribute('stroke', '#111827');
          handle.setAttribute('stroke-width', '1.5');
          handle.style.cursor = 'move';
          handle.addEventListener('pointerdown', (event) => {{
            event.preventDefault();
            state.active = {{ compIndex, pointIndex }};
            handle.setPointerCapture(event.pointerId);
          }});
          svg.appendChild(handle);
        }});
      }});
      status.textContent = state.components.length ? `${{seed.target_label || 'selected label'}}: ${{state.components.length}} contour(s)` : 'click to draw a new contour';
      drawModeButton.classList.toggle('active', state.drawMode);
      renderPreview();
      emit();
    }}

    function toSvgPoint(event) {{
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const transformed = point.matrixTransform(svg.getScreenCTM().inverse());
      const x = transformed.x;
      const y = transformed.y;
      return [Math.max(0, Math.min(seed.width, x)), Math.max(0, Math.min(seed.height, y))];
    }}

    function renderPreview() {{
      previewCtx.clearRect(0, 0, seed.width, seed.height);
      if (backgroundLoaded) previewCtx.drawImage(bg, 0, 0, seed.width, seed.height);
      previewCtx.fillStyle = `rgb(${{targetColor[0]}},${{targetColor[1]}},${{targetColor[2]}})`;
      state.components.forEach((component) => {{
        if (!component.points || component.points.length < 3) return;
        previewCtx.beginPath();
        component.points.forEach((point, index) => {{
          if (index === 0) previewCtx.moveTo(point[0], point[1]);
          else previewCtx.lineTo(point[0], point[1]);
        }});
        previewCtx.closePath();
        previewCtx.fill();
      }});
    }}

    function addDrawnContour() {{
      if (!state.drawing || state.drawing.length < 3) {{
        state.drawing = null;
        render();
        return;
      }}
      state.components.push({{
        component_id: `${{seed.target_label || 'manual'}}_new_${{Date.now()}}`,
        label: seed.target_label || '',
        area_px: 0,
        bbox: [],
        centroid: [],
        points: state.drawing,
      }});
      state.drawing = null;
      state.drawMode = false;
      state.dirty = true;
      render();
    }}

    drawModeButton.addEventListener('click', () => {{
      state.drawMode = !state.drawMode;
      state.drawing = null;
      render();
    }});

    svg.addEventListener('pointermove', (event) => {{
      const [x, y] = toSvgPoint(event);
      if (state.active) {{
        state.components[state.active.compIndex].points[state.active.pointIndex] = [x, y];
        state.dirty = true;
        render();
      }} else if (state.drawing) {{
        const last = state.drawing[state.drawing.length - 1];
        if (!last || Math.hypot(last[0] - x, last[1] - y) >= 2) {{
          state.drawing.push([x, y]);
          state.dirty = true;
          render();
        }}
      }}
    }});
    svg.addEventListener('pointerdown', (event) => {{
      if (event.target && event.target.tagName && event.target.tagName.toLowerCase() === 'circle') return;
      if (!state.drawMode) return;
      event.preventDefault();
      state.drawing = [toSvgPoint(event)];
      svg.setPointerCapture(event.pointerId);
    }});
    svg.addEventListener('pointerup', () => {{
      if (state.active) state.active = null;
      else addDrawnContour();
    }});
    svg.addEventListener('pointerleave', () => {{
      state.active = null;
      if (state.drawing) addDrawnContour();
    }});
    render();
    let retries = 0;
    const timer = setInterval(() => {{
      retries += 1;
      if (resolvePayloadBox()) {{
        emit();
        clearInterval(timer);
      }} else if (retries >= 50) {{
        clearInterval(timer);
      }}
    }}, 100);
  </script>
</body>
</html>
"""
    return f'<iframe style="width:100%;height:760px;border:0;" srcdoc="{html_lib.escape(srcdoc, quote=True)}"></iframe>'


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
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.zeros(shape, dtype=bool)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
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

    if isinstance(value.get("components"), list):
        candidate = np.zeros(shape, dtype=bool)
        for component in value["components"]:
            points = component.get("points") if isinstance(component, dict) else None
            if not isinstance(points, list) or len(points) < 3:
                continue
            polygon = []
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                polygon.append((float(point[0]), float(point[1])))
            if len(polygon) < 3:
                continue
            image = Image.new("L", (shape[1], shape[0]), 0)
            draw = ImageDraw.Draw(image)
            draw.polygon(polygon, fill=1)
            candidate |= np.asarray(image, dtype=np.uint8).astype(bool)
        return candidate

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
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def _manual_editor_updates_for_label(
    state: dict[str, Any],
    target_label: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not state:
        return gr.update(value=""), gr.update(value="")

    schema = MaskProfileSchema.from_reference_profile(state.get("profile", "BCSS"))
    selected_label = target_label if target_label in schema.writable_labels else _default_manual_target_label(schema)
    state["manual_selected_label"] = selected_label
    tissue_mask_path = state.get("manual_working_tissue_mask") or state.get("target_tissue_mask") or state.get("reference_tissue_mask")
    if not tissue_mask_path:
        return gr.update(value=""), gr.update(value="")

    tissue_mask = load_id_mask(tissue_mask_path)
    background_uri = _rgb_array_data_uri(id_to_rgb(tissue_mask))
    components = _extract_manual_contour_components(
        tissue_mask,
        schema=schema,
        label_filter=selected_label,
    )
    state["manual_contour_components"] = components
    color = _label_color(schema, selected_label)
    return (
        gr.update(
            value=_manual_contour_editor_value(
                background_uri,
                components,
                tissue_mask.shape,
                target_label=selected_label,
                target_color=color,
            )
        ),
        gr.update(
            value=_manual_contour_payload_value(
                background_uri,
                components,
                tissue_mask.shape,
                target_label=selected_label,
                target_color=color,
            )
        ),
    )


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
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Return panel visibility and auto-recommendation UI state."""

    prompt_visible = gr.update(visible=edit_mode == EDIT_MODE_PROMPT)
    instruction_visible = gr.update(visible=edit_mode == EDIT_MODE_INSTRUCTION)
    manual_visible = gr.update(visible=edit_mode == EDIT_MODE_MANUAL_CONTOUR)
    auto_visible = gr.update(visible=edit_mode == EDIT_MODE_AUTO_RECOMMEND)
    tissue_button_visible = gr.update(
        visible=edit_mode in {EDIT_MODE_PROMPT, EDIT_MODE_INSTRUCTION}
    )
    auto_execute_visible = gr.update(visible=edit_mode == EDIT_MODE_AUTO_RECOMMEND)
    manual_editor_value = gr.update(value="")
    manual_contour_payload_value = gr.update(value="")
    manual_target_update = gr.update()
    auto_primitive_update = gr.update(choices=[], value=None)
    auto_strength_update = gr.update(choices=[], value=None)
    auto_summary = gr.update(value="")

    if not state:
        return (
            state,
            prompt_visible,
            instruction_visible,
            manual_visible,
            auto_visible,
            manual_editor_value,
            manual_contour_payload_value,
            manual_target_update,
            auto_primitive_update,
            auto_strength_update,
            auto_summary,
            tissue_button_visible,
            auto_execute_visible,
        )

    if edit_mode == EDIT_MODE_MANUAL_CONTOUR:
        schema = MaskProfileSchema.from_reference_profile(state.get("profile", "BCSS"))
        state_label = state.get("manual_selected_label")
        selected_label = state_label if state_label in schema.writable_labels else _default_manual_target_label(schema)
        tissue_mask_path = state.get("manual_working_tissue_mask") or state.get("target_tissue_mask") or state.get("reference_tissue_mask")
        tissue_shape = None
        components: list[dict[str, Any]] = []
        background_rgb = ""
        if tissue_mask_path:
            tissue_mask = load_id_mask(tissue_mask_path)
            tissue_shape = tissue_mask.shape
            background_rgb = _rgb_array_data_uri(id_to_rgb(tissue_mask))
            components = _extract_manual_contour_components(
                tissue_mask,
                schema=schema,
                label_filter=selected_label,
            )
        state["manual_contour_components"] = components
        target_color = _label_color(schema, selected_label)
        manual_editor_value = gr.update(
            value=_manual_contour_editor_value(
                background_rgb,
                components,
                tissue_shape,
                target_label=selected_label,
                target_color=target_color,
            )
        )
        manual_contour_payload_value = gr.update(
            value=_manual_contour_payload_value(
                background_rgb,
                components,
                tissue_shape,
                target_label=selected_label,
                target_color=target_color,
            )
        )
        manual_target_update = gr.update(
            choices=_manual_target_label_choices(schema),
            value=selected_label,
        )
    else:
        manual_target_update = gr.update(choices=[], value=None)

    if edit_mode == EDIT_MODE_AUTO_RECOMMEND:
        primitive_choices = _auto_primitive_choices_for_state(state)
        primitive_value = _auto_selected_primitive(state, primitive_choices)
        strength_choices = _auto_strength_choices_for_state(state, primitive_value)
        strength_value = _auto_selected_strength(state, strength_choices)
        state["auto_selected_primitive"] = primitive_value
        state["auto_selected_strength"] = strength_value
        auto_primitive_update = gr.update(
            choices=primitive_choices,
            value=primitive_value,
        )
        auto_strength_update = gr.update(
            choices=strength_choices,
            value=strength_value,
        )
        auto_summary = gr.update(
            value=_format_auto_selection_summary(state, primitive_value, strength_value)
        )
    else:
        state.pop("auto_selected_primitive", None)
        state.pop("auto_selected_strength", None)

    return (
        state,
        prompt_visible,
        instruction_visible,
        manual_visible,
        auto_visible,
        manual_editor_value,
        manual_contour_payload_value,
        manual_target_update,
        auto_primitive_update,
        auto_strength_update,
        auto_summary,
        tissue_button_visible,
        auto_execute_visible,
    )


def _auto_recipe_for_state(state: dict[str, Any]) -> dict[str, Any]:
    profile = state.get("profile", "BCSS")
    return load_recipe(default_recipe_path_for_profile(profile))


def _auto_primitive_choices_for_state(state: dict[str, Any]) -> list[str]:
    profile = state.get("profile", "BCSS")
    tissue_path = state.get("target_tissue_mask") or state.get("reference_tissue_mask")
    if not tissue_path:
        return []
    reference_tissue = load_id_mask(tissue_path)
    schema = MaskProfileSchema.from_reference_profile(profile)
    recipe = load_recipe(default_recipe_path_for_profile(profile))
    context = MaskEditContext.from_mask(reference_tissue, schema)
    choices: list[str] = []
    for primitive_config in recipe.get("primitives", []):
        if not isinstance(primitive_config, dict):
            continue
        primitive_name = primitive_config.get("name")
        if not isinstance(primitive_name, str):
            continue
        if _auto_feasible_strengths_for_primitive(
            reference_tissue,
            schema=schema,
            recipe=recipe,
            context=context,
            primitive_config=primitive_config,
        ):
            choices.append(primitive_name)
    return choices


def _auto_strength_choices_for_state(
    state: dict[str, Any],
    primitive_name: str | None,
) -> list[str]:
    if not primitive_name:
        return []
    profile = state.get("profile", "BCSS")
    tissue_path = state.get("target_tissue_mask") or state.get("reference_tissue_mask")
    if not tissue_path:
        return []
    reference_tissue = load_id_mask(tissue_path)
    schema = MaskProfileSchema.from_reference_profile(profile)
    recipe = load_recipe(default_recipe_path_for_profile(profile))
    try:
        primitive_config = _primitive_config(recipe, primitive_name)
    except gr.Error:
        return []
    context = MaskEditContext.from_mask(reference_tissue, schema)
    return list(
        _auto_feasible_strengths_for_primitive(
            reference_tissue,
            schema=schema,
            recipe=recipe,
            context=context,
            primitive_config=primitive_config,
        )
    )


def _auto_feasible_strengths_for_primitive(
    mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    recipe: dict[str, Any],
    context: MaskEditContext,
    primitive_config: dict[str, Any],
) -> tuple[str, ...]:
    primitive_name = primitive_config.get("name")
    if not isinstance(primitive_name, str):
        return ()
    strengths = _primitive_strengths(primitive_config) or (
        AUTO_RECOMMEND_DEFAULT_STRENGTH,
    )
    feasible: list[str] = []
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
        feasibility = _estimate_recommendation_capacity(
            mask,
            intent,
            primitive_config,
            schema,
        )
        if feasibility.get("status") == "executable":
            feasible.append(strength)
    return tuple(feasible)


def _auto_selected_primitive(
    state: dict[str, Any],
    choices: list[str],
) -> str | None:
    current = state.get("auto_selected_primitive")
    if isinstance(current, str) and current in choices:
        return current
    if AUTO_RECOMMEND_DEFAULT_PRIMITIVE in choices:
        return AUTO_RECOMMEND_DEFAULT_PRIMITIVE
    return choices[0] if choices else None


def _auto_selected_strength(
    state: dict[str, Any],
    choices: list[str],
) -> str | None:
    current = state.get("auto_selected_strength")
    if isinstance(current, str) and current in choices:
        return current
    if AUTO_RECOMMEND_DEFAULT_STRENGTH in choices:
        return AUTO_RECOMMEND_DEFAULT_STRENGTH
    return choices[0] if choices else None


def _refresh_auto_strength_options(
    state: dict[str, Any],
    primitive_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not state:
        return state, gr.update(choices=[], value=None), gr.update(value="")
    primitive_value = primitive_name if isinstance(primitive_name, str) else None
    state["auto_selected_primitive"] = primitive_value
    strength_choices = _auto_strength_choices_for_state(state, primitive_value)
    strength_value = _auto_selected_strength(state, strength_choices)
    state["auto_selected_strength"] = strength_value
    return (
        state,
        gr.update(choices=strength_choices, value=strength_value),
        gr.update(value=_format_auto_selection_summary(state, primitive_value, strength_value)),
    )


def _refresh_auto_selection_summary(
    state: dict[str, Any],
    primitive_name: str | None,
    strength: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not state:
        return state, gr.update(value="")
    state["auto_selected_primitive"] = primitive_name
    state["auto_selected_strength"] = strength
    return (
        state,
        gr.update(value=_format_auto_selection_summary(state, primitive_name, strength)),
    )


def _format_auto_selection_summary(
    state: dict[str, Any],
    primitive_name: str | None,
    strength: str | None,
) -> str:
    if not primitive_name or not strength:
        return "No feasible primitive/strength options for the current mask."
    profile = state.get("profile", "BCSS")
    tissue_path = state.get("target_tissue_mask") or state.get("reference_tissue_mask")
    if not tissue_path:
        return f"Selected: {primitive_name} / {strength}"
    reference_tissue = load_id_mask(tissue_path)
    schema = MaskProfileSchema.from_reference_profile(profile)
    recipe = load_recipe(default_recipe_path_for_profile(profile))
    primitive_config = _primitive_config(recipe, primitive_name)
    intent = EditIntent(
        primitive=primitive_name,
        strength=strength,
        reference_profile=schema.reference_profile,
    )
    intent = _with_default_contour_labels(intent, primitive_config, schema)
    context = MaskEditContext.from_mask(reference_tissue, schema)
    decision = assess_edit_applicability(intent, recipe, schema, context)
    feasibility = _estimate_recommendation_capacity(
        reference_tissue, intent, primitive_config, schema
    )
    source = ", ".join(intent.source_labels) or "-"
    target = intent.target_label or "-"
    changed = feasibility.get("changed_area_fraction")
    changed_text = f"{float(changed):.3f}" if isinstance(changed, (int, float)) else "-"
    lines = [
        f"Selected: {primitive_name} / {strength}",
        f"status={decision.status} | changed={changed_text}",
        f"source: {source} -> target: {target}",
    ]
    notes = list(decision.reasons) + list(decision.warnings) + list(feasibility.get("notes", []))
    failed = feasibility.get("validation_failed_checks") or []
    notes.extend(str(item) for item in failed)
    if notes:
        lines.append(f"notes: {'; '.join(str(note) for note in notes)}")
    return "\n".join(lines)


def _auto_selection_to_intent(
    state: dict[str, Any],
    primitive_name: str | None,
    strength: str | None,
) -> tuple[EditIntent, dict[str, Any]]:
    if not primitive_name or not strength:
        raise gr.Error("Select both primitive and strength before running auto recommend.")
    profile = state.get("profile", "BCSS")
    schema = MaskProfileSchema.from_reference_profile(profile)
    recipe = load_recipe(default_recipe_path_for_profile(profile))
    primitive_config = _primitive_config(recipe, primitive_name)
    tissue_path = state.get("target_tissue_mask") or state.get("reference_tissue_mask")
    if not tissue_path:
        raise gr.Error("Load inputs first.")
    reference_tissue = load_id_mask(tissue_path)
    feasible_strengths = _auto_feasible_strengths_for_primitive(
        reference_tissue,
        schema=schema,
        recipe=recipe,
        context=MaskEditContext.from_mask(reference_tissue, schema),
        primitive_config=primitive_config,
    )
    if strength not in feasible_strengths:
        if feasible_strengths:
            available = ", ".join(feasible_strengths)
            raise gr.Error(
                f"Primitive {primitive_name} is not feasible with strength {strength}; "
                f"available strengths: {available}."
            )
        raise gr.Error(f"Primitive {primitive_name} is not feasible for the current mask.")
    payload = {
        "primitive": primitive_name,
        "strength": strength,
        "reference_profile": profile,
    }
    intent = EditIntent.from_mapping(payload)
    intent = _with_default_contour_labels(intent, primitive_config, schema)
    return intent, primitive_config


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
    instruction_text: str,
    instruction_parser: str,
    manual_contour_editor,
    manual_contour_payload,
    auto_primitive: str | None,
    auto_strength: str | None,
    parser: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    parser_api_model: str,
    qwen_model_path: str,
    qwen_device: str,
    no_few_shot: bool,
    source_labels: str,
    target_label: str,
    provider: str,
    contour_api_base_url: str,
    contour_api_key_env: str,
    contour_api_model: str,
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
                manual_contour_payload=manual_contour_payload,
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
                primitive=auto_primitive,
                strength=auto_strength,
                provider=provider,
                api_base_url=contour_api_base_url,
                api_key_env=contour_api_key_env,
                api_model=contour_api_model,
                api_image_detail=api_image_detail,
                fixture_file=fixture_file,
                max_attempts=max_attempts,
                max_regions=max_regions,
                max_points_per_region=max_points_per_region,
                organic_seed=organic_seed,
            )
        elif edit_mode == EDIT_MODE_INSTRUCTION:
            semantic_diff, parser_info = _resolve_instruction_semantic_diff(
                instruction=instruction_text,
                parser=instruction_parser,
                api_base_url=api_base_url,
                api_key_env=api_key_env,
                api_model=parser_api_model,
                output_dir=output_dir,
            )
            plan = plan_edit_intents(
                semantic_diff,
                reference_profile=state["profile"],
                old_mask=reference_tissue,
                new_prompt=instruction_text,
            )
            result, phase3_info = _execute_planned_intents(
                reference_tissue=reference_tissue,
                schema=schema,
                recipe=recipe,
                output_dir=output_dir,
                plan=plan,
                semantic_diff=semantic_diff,
                parser_info=parser_info,
                execution_mode=EDIT_MODE_INSTRUCTION,
                provider=provider,
                api_base_url=contour_api_base_url,
                api_key_env=contour_api_key_env,
                api_model=contour_api_model,
                api_image_detail=api_image_detail,
                fixture_file=fixture_file,
                max_attempts=max_attempts,
                max_regions=max_regions,
                max_points_per_region=max_points_per_region,
                organic_seed=organic_seed,
                continue_on_failure=continue_on_failure,
            )
        elif edit_mode == EDIT_MODE_PROMPT and old_prompt.strip() and new_prompt.strip():
            semantic_diff, parser_info = _resolve_prompt_semantic_diff(
                old_prompt=old_prompt,
                new_prompt=new_prompt,
                parser=parser,
                api_base_url=api_base_url,
                api_key_env=api_key_env,
                api_model=parser_api_model,
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
            result, phase3_info = _execute_planned_intents(
                reference_tissue=reference_tissue,
                schema=schema,
                recipe=recipe,
                output_dir=output_dir,
                plan=plan,
                semantic_diff=semantic_diff,
                parser_info=parser_info,
                execution_mode="prompt_to_contour",
                provider=provider,
                api_base_url=contour_api_base_url,
                api_key_env=contour_api_key_env,
                api_model=contour_api_model,
                api_image_detail=api_image_detail,
                fixture_file=fixture_file,
                max_attempts=max_attempts,
                max_regions=max_regions,
                max_points_per_region=max_points_per_region,
                organic_seed=organic_seed,
                continue_on_failure=continue_on_failure,
            )
        else:
            raise gr.Error(
                "Prompt mode requires both old and new prompts. Instruction mode "
                "requires one edit instruction. Choose manual contour or auto "
                "recommend for direct non-prompt edits."
            )
    except Exception as exc:
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc

    if result.edit_result is None:
        raise gr.Error(_contour_failure_message(result))
    if (
        result.status not in {
            "validated",
            "skipped_no_source_region",
            "executed_validated",
            "degraded_executed",
        }
        and not continue_on_failure
    ):
        raise gr.Error(_contour_failure_message(result))

    target_tissue = result.edit_result.target_mask
    target_path = save_id_mask(target_tissue, output_dir / "target_mask.png")
    if phase3_info.get("mode") in {"prompt_to_contour", EDIT_MODE_INSTRUCTION}:
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
        "primitive": phase3_info.get(
            "primitive",
            phase3_info.get("selected_primitive", auto_primitive or ""),
        ),
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
    manual_contour_payload,
    manual_target_label: str,
    output_dir: Path,
    source_labels: str,
    primitive: str,
) -> tuple[Any, dict[str, Any]]:
    candidate_region = _editor_value_to_binary_mask(manual_contour_payload, reference_tissue.shape)
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
        inferred_source_labels = override_source_labels
    if not inferred_source_labels:
        raise gr.Error(
            "Could not infer source labels from the drawn contour. "
            "Draw over a visible source label or provide source labels."
        )

    primitive_config = _primitive_config(load_recipe(default_recipe_path_for_profile(schema.reference_profile)), primitive)
    edit_result = apply_organic_projected_label_write(
        reference_tissue,
        candidate_region,
        schema=schema,
        source_labels=inferred_source_labels,
        target_label=target_label,
        primitive_config=primitive_config,
    )
    edit_result.ops_log["manual_payload"] = {
        "primitive": primitive,
        "mode": EDIT_MODE_MANUAL_CONTOUR,
        "manual_target_label": target_label,
        "source_labels": list(inferred_source_labels),
    }
    true_change_region = edit_result.target_mask != reference_tissue
    if not np.array_equal(true_change_region, edit_result.change_region):
        edit_result = type(edit_result)(
            target_mask=edit_result.target_mask,
            change_region=true_change_region,
            changed_area_fraction=float(np.count_nonzero(true_change_region)) / int(true_change_region.size),
            selected_pixels=int(np.count_nonzero(true_change_region)),
            warnings=edit_result.warnings,
            ops_log={
                **edit_result.ops_log,
                "projected_pixels_including_existing_target": int(np.count_nonzero(edit_result.change_region)),
                "selected_pixels": int(np.count_nonzero(true_change_region)),
                "changed_area_fraction": float(np.count_nonzero(true_change_region)) / int(true_change_region.size),
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
        "manual_contour_points": _json_safe(manual_contour_payload),
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


def _execute_planned_intents(
    *,
    reference_tissue: np.ndarray,
    schema: MaskProfileSchema,
    recipe: dict[str, Any],
    output_dir: Path,
    plan,
    semantic_diff: dict[str, Any],
    parser_info: dict[str, Any],
    execution_mode: str,
    provider: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    api_image_detail: str,
    fixture_file,
    max_attempts: int,
    max_regions: int,
    max_points_per_region: int,
    organic_seed: int,
    continue_on_failure: bool,
) -> tuple[Any, dict[str, Any]]:
    save_semantic_diff(
        semantic_diff,
        output_dir / "phase3_mask_edit" / "semantic_diff.json",
    )
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
            output_dir=(
                output_dir
                / "phase3_mask_edit"
                / execution_mode
                / intent.primitive
            ),
            projection_mode=PROJECTION_MODE_ORGANIC_V2,
            organic_seed=organic_seed,
            max_attempts=max_attempts,
            max_regions=max_regions,
            max_points_per_region=max_points_per_region,
        )
        attempt_logs.append(
            {
                "primitive": intent.primitive,
                "strength": intent.strength,
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
            "mode": execution_mode,
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
            raise gr.Error("Instruction planning produced no executable intents.")
        return result, phase3_info

    if last_edit_result is None:
        raise gr.Error(_contour_failure_message(last_result))

    phase3_info = {
        "mode": execution_mode,
        "parser": parser_info,
        "semantic_diff": semantic_diff,
        "plan": plan.to_metadata(),
        "attempts": attempt_logs,
        "projection_mode": PROJECTION_MODE_ORGANIC_V2,
    }
    return last_result, phase3_info


def _save_manual_current_label(
    state: dict[str, Any],
    manual_contour_payload,
    source_labels: str,
    target_label: str,
) -> tuple[dict[str, Any], dict[str, Any], str, str, str]:
    state, log, working_rgb_path = _save_manual_current_label_state(
        state,
        manual_contour_payload,
        source_labels,
        target_label,
    )
    editor_update, payload_update = _manual_editor_updates_for_label(state, target_label)
    return state, editor_update, payload_update, _json_text(log), working_rgb_path


def _save_manual_current_label_state(
    state: dict[str, Any],
    manual_contour_payload,
    source_labels: str,
    target_label: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    if not state:
        raise gr.Error("Load inputs first.")
    output_dir = Path(state["output_dir"])
    schema = MaskProfileSchema.from_reference_profile(state["profile"])
    base_mask = load_id_mask(state.get("manual_working_tissue_mask") or state.get("target_tissue_mask") or state["reference_tissue_mask"])
    if not (isinstance(manual_contour_payload, str) and manual_contour_payload.strip()):
        components = state.get("manual_contour_components") or []
        if components:
            manual_contour_payload = {
                "components": components,
                "target_label": target_label,
                "dirty": False,
            }
    result, phase3_info = _run_manual_contour_stage(
        reference_tissue=base_mask,
        schema=schema,
        manual_contour_editor=None,
        manual_contour_payload=manual_contour_payload,
        manual_target_label=target_label,
        output_dir=output_dir,
        source_labels=source_labels,
        primitive="manual_contour",
    )
    working_mask = result.edit_result.target_mask
    working_path = save_id_mask(working_mask, output_dir / "manual_working_tissue_mask.png")
    working_rgb_path = output_dir / "manual_working_tissue_rgb.png"
    Image.fromarray(id_to_rgb(working_mask), mode="RGB").save(working_rgb_path)
    manual_steps = list(state.get("manual_steps", []))
    manual_steps.append(
        {
            "target_label": target_label,
            "source_labels": phase3_info.get("source_labels", []),
            "selected_pixels": phase3_info.get("selected_pixels", 0),
        }
    )
    state.update(
        {
            "manual_working_tissue_mask": str(working_path),
            "manual_working_tissue_rgb": str(working_rgb_path),
            "manual_selected_label": target_label,
            "manual_steps": manual_steps,
        }
    )
    log = {
        "status": "manual_label_saved",
        "target_label": target_label,
        "working_tissue_mask": str(working_path),
        "selected_pixels": phase3_info.get("selected_pixels", 0),
        "steps": manual_steps,
    }
    return state, log, str(working_rgb_path)


def _manual_switch_label_saving_current(
    state: dict[str, Any],
    manual_contour_payload,
    source_labels: str,
    next_label: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    if not state:
        return state, gr.update(value=""), gr.update(value=""), "", None

    previous_label = state.get("manual_selected_label")
    log: dict[str, Any] = {"status": "manual_label_switched", "from": previous_label, "to": next_label}
    preview_path: str | None = None
    payload_components = []
    payload_dirty = False
    if isinstance(manual_contour_payload, str) and manual_contour_payload.strip():
        try:
            payload_data = json.loads(manual_contour_payload)
            if isinstance(payload_data, dict):
                payload_components = payload_data.get("components", [])
                payload_dirty = bool(payload_data.get("dirty"))
        except json.JSONDecodeError:
            payload_components = []

    if previous_label and previous_label in MaskProfileSchema.from_reference_profile(state["profile"]).writable_labels and payload_components and payload_dirty:
        state, save_log, preview_path = _save_manual_current_label_state(
            state,
            manual_contour_payload,
            source_labels,
            previous_label,
        )
        log["saved_previous"] = save_log

    editor_update, payload_update = _manual_editor_updates_for_label(state, next_label)
    state["manual_selected_label"] = next_label
    return state, editor_update, payload_update, _json_text(log), preview_path


def _finalize_manual_tissue_stage(
    state: dict[str, Any],
    manual_contour_payload=None,
    source_labels: str = "",
    target_label: str | None = None,
) -> tuple[dict[str, Any], str, str, str]:
    if not state:
        raise gr.Error("Load inputs first.")
    if manual_contour_payload and target_label:
        try:
            payload_data = json.loads(manual_contour_payload) if isinstance(manual_contour_payload, str) else {}
        except json.JSONDecodeError:
            payload_data = {}
        if isinstance(payload_data, dict) and payload_data.get("dirty"):
            state, _, _ = _save_manual_current_label_state(
                state,
                manual_contour_payload,
                source_labels,
                target_label,
            )
    if not state.get("manual_working_tissue_mask"):
        raise gr.Error("Save at least one manual tissue edit first.")

    output_dir = Path(state["output_dir"])
    reference_image = _load_rgb_image(state["reference_image"])
    reference_tissue = load_id_mask(state["reference_tissue_mask"])
    target_tissue = load_id_mask(state["manual_working_tissue_mask"])
    _validate_same_size(reference_image, target_tissue, "target_tissue_mask")
    target_path = save_id_mask(target_tissue, output_dir / "target_mask.png")
    change_region = reference_tissue != target_tissue
    stage_paths = _save_pre_generation_artifacts(
        output_dir=output_dir,
        reference_image=reference_image,
        reference_tissue=reference_tissue,
        target_tissue=target_tissue,
        change_region=change_region,
    )
    phase3_info = {
        "mode": EDIT_MODE_MANUAL_CONTOUR,
        "primitive": "manual_contour",
        "reference_profile": state.get("profile", "BCSS"),
        "projection_mode": "manual_stepwise_projected_label_write",
        "manual_steps": list(state.get("manual_steps", [])),
        "selected_pixels": int(np.count_nonzero(change_region)),
    }
    save_metadata(phase3_info, output_dir / "phase3_mask_edit" / "execution_summary.json")
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
        "edit_mode": EDIT_MODE_MANUAL_CONTOUR,
        "target_tissue_mask": str(target_path),
        "change_region": stage_paths["change_region"],
        "changed_area_fraction": _change_area_fraction(change_region),
        "manual_steps": phase3_info["manual_steps"],
    }
    return state, _json_text(info), stage_paths["target_mask_rgb"], stage_paths["change_region"]


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


def _execute_selected_recommendations(
    *,
    state: dict[str, Any],
    reference_tissue: np.ndarray,
    schema: MaskProfileSchema,
    recipe: dict[str, Any],
    output_dir: Path,
    primitive: str | None,
    strength: str | None,
    provider: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    api_image_detail: str,
    fixture_file,
    max_attempts: int,
    max_regions: int,
    max_points_per_region: int,
    organic_seed: int,
) -> tuple[Any, dict[str, Any]]:
    intent, primitive_config = _auto_selection_to_intent(state, primitive, strength)

    current_mask = np.array(reference_tissue, copy=True)
    attempt_logs: list[dict[str, Any]] = []
    last_success = None
    provider_instance = _build_contour_provider(
        provider=provider,
        api_base_url=api_base_url,
        api_key_env=api_key_env,
        api_model=_defaulted_text(api_model, DEFAULT_API_MODEL),
        api_image_detail=api_image_detail,
        fixture_file=fixture_file,
    )
    decision = assess_edit_applicability(
        intent,
        recipe,
        schema,
        MaskEditContext.from_mask(current_mask, schema),
    )
    if decision.status == "rejected":
        attempt_logs.append(
            {
                "primitive": intent.primitive,
                "strength": intent.strength,
                "status": "rejected",
                "reasons": list(decision.reasons),
            }
        )
    else:
        result = execute_llm_contour_agent(
            old_mask=current_mask,
            schema=schema,
            intent=intent,
            primitive_config=primitive_config,
            provider=provider_instance,
            output_dir=output_dir / "phase3_mask_edit" / "auto_recommend" / intent.primitive,
            projection_mode=PROJECTION_MODE_ORGANIC_V2,
            organic_seed=organic_seed,
            max_attempts=max_attempts,
            max_regions=max_regions,
            max_points_per_region=max_points_per_region,
        )
        if result.status == STATUS_VALIDATED and result.edit_result is not None:
            last_success = result
            current_mask = np.array(result.edit_result.target_mask, copy=True)
        attempt_logs.append(
            {
                "primitive": intent.primitive,
                "strength": intent.strength,
                "status": result.status,
                "projection_mode": PROJECTION_MODE_ORGANIC_V2,
                "error": result.error,
                "artifact_paths": result.artifact_paths,
                "validation_failed_checks": _validation_failed_checks(result.validation),
                "primitive_config": primitive_config.get("name"),
            }
        )

    if last_success is None or last_success.edit_result is None:
        raise gr.Error(_auto_recommend_execution_failure_message(attempt_logs))

    phase3_info = {
        "mode": EDIT_MODE_AUTO_RECOMMEND,
        "selected_primitive": intent.primitive,
        "selected_strength": intent.strength,
        "attempts": attempt_logs,
        "primitive": last_success.edit_result.ops_log.get("primitive", ""),
        "projection_mode": PROJECTION_MODE_ORGANIC_V2,
    }
    save_metadata(phase3_info, output_dir / "phase3_mask_edit" / "execution_summary.json")
    return last_success, phase3_info


def _primitive_strengths(primitive_config: dict[str, Any]) -> tuple[str, ...]:
    ranges = primitive_config.get("parameter_ranges", {})
    if not isinstance(ranges, dict):
        return ()
    strengths: list[str] = []
    for key in ("mild", "moderate", "significant", "xlarge_deid"):
        if key in ranges:
            strengths.append(key)
            continue
        if any(isinstance(value, dict) and key in value for value in ranges.values()):
            strengths.append(key)
    return tuple(strengths)


def _estimate_recommendation_capacity(
    mask: np.ndarray,
    intent: EditIntent,
    primitive_config: dict[str, Any],
    schema: MaskProfileSchema,
) -> dict[str, Any]:
    interval = _strength_interval(primitive_config, intent.strength)
    denominator = _strength_denominator_pixels(mask, primitive_config, schema)
    legal_pixels = _recommendation_legal_pixels(mask, primitive_config, schema)
    dependency_failures = _recommendation_dependency_failures(
        mask, primitive_config, schema
    )
    failed: list[str] = []
    if denominator <= 0:
        failed.append("capacity failed: no denominator pixels in current mask.")
    if legal_pixels <= 0:
        failed.append("capacity failed: no legal source pixels in current mask.")
    failed.extend(dependency_failures)
    if interval is not None and denominator > 0:
        lower, upper = interval
        lower_pixels = int(np.ceil(denominator * lower))
        upper_pixels = int(np.floor(denominator * upper))
        if legal_pixels < lower_pixels:
            failed.append(
                f"capacity failed: legal_pixels={legal_pixels} below "
                f"{intent.strength} minimum {lower_pixels}."
            )
    else:
        lower_pixels = 1 if denominator > 0 else 0
        upper_pixels = legal_pixels
    target_pixels = max(1, int(round((lower_pixels + max(upper_pixels, lower_pixels)) / 2))) if legal_pixels > 0 else 0
    achievable_pixels = min(target_pixels, legal_pixels)
    fraction = achievable_pixels / denominator if denominator > 0 else None
    return {
        "status": "executable" if not failed else "capacity_failed",
        "validation_passed": not failed,
        "validation_failed_checks": failed,
        "changed_area_fraction": fraction,
        "strength_fraction": fraction,
        "strength_range": list(interval) if interval is not None else None,
        "selected_pixels": int(achievable_pixels),
        "legal_pixels": int(legal_pixels),
        "denominator_pixels": int(denominator),
        "notes": ["static_mask_capacity_estimate_only"],
    }


def _strength_fit_summary(
    mask: np.ndarray,
    change_region: np.ndarray,
    intent: EditIntent,
    primitive_config: dict[str, Any],
    schema: MaskProfileSchema,
) -> dict[str, Any]:
    interval = _strength_interval(primitive_config, intent.strength)
    if interval is None:
        return {"fits": True, "fraction": None, "range": None, "detail": ""}
    denominator = _strength_denominator_pixels(mask, primitive_config, schema)
    if denominator <= 0:
        return {
            "fits": False,
            "fraction": None,
            "range": list(interval),
            "detail": "strength feasibility failed: no denominator pixels in current mask.",
        }
    fraction = int(np.count_nonzero(change_region)) / denominator
    lower, upper = interval
    fits = lower <= fraction <= upper
    return {
        "fits": fits,
        "fraction": float(fraction),
        "range": [float(lower), float(upper)],
        "detail": (
            f"strength feasibility failed: achieved_fraction={fraction:.4f} "
            f"outside {intent.strength} range [{lower:.2f}, {upper:.2f}]"
        ),
    }


def _strength_interval(
    primitive_config: dict[str, Any],
    strength: str,
) -> tuple[float, float] | None:
    ranges = primitive_config.get("parameter_ranges", {})
    if not isinstance(ranges, dict):
        return None
    for key in (
        "target_changed_area_fraction",
        "target_area_delta_fraction",
        "target_area_decrease_fraction",
        "necrosis_area_decrease_fraction",
        "immune_area_delta_fraction",
        "immune_area_decrease_fraction",
        "stroma_area_delta_fraction",
        "stroma_area_decrease_fraction",
        "source_area_transition_fraction",
    ):
        value = ranges.get(key)
        if not isinstance(value, dict):
            continue
        interval = value.get(strength)
        if (
            isinstance(interval, list)
            and len(interval) == 2
            and all(isinstance(item, (int, float)) for item in interval)
        ):
            return float(interval[0]), float(interval[1])
    return None


def _strength_denominator_pixels(
    mask: np.ndarray,
    primitive_config: dict[str, Any],
    schema: MaskProfileSchema,
) -> int:
    name = primitive_config.get("name")
    if name in {"tumor_burden_increase", "tumor_burden_decrease"}:
        return int(np.count_nonzero(np.isin(mask, schema.tumor_fine_ids)))
    if name in {"necrosis_appearance", "intratumoral_immune_infiltration"}:
        return int(np.count_nonzero(np.isin(mask, schema.tumor_fine_ids)))
    if name == "necrosis_resolution":
        return int(np.count_nonzero(_safe_schema_label_mask(mask, schema, "Necrosis")))
    if name in {"immune_infiltration_decrease"}:
        return int(np.count_nonzero(_safe_schema_label_mask(mask, schema, "Immune infiltrate")))
    if name == "stromal_immune_infiltration":
        stroma = _safe_schema_label_mask(mask, schema, "Stroma")
        immune = _safe_schema_label_mask(mask, schema, "Immune infiltrate")
        return int(np.count_nonzero(stroma | immune))
    if name in {"stromal_desmoplasia", "stroma_decrease", "stromal_reduction"}:
        return int(np.count_nonzero(_safe_schema_label_mask(mask, schema, "Stroma")))
    if name in {"grade_upgrade", "adenoma_to_carcinoma", "gleason_upgrade_3to4", "gleason_upgrade_4to5", "benign_to_gleason3"}:
        source_ids = primitive_config.get("mask_operation", {}).get("source_fine_ids", ())
        if isinstance(source_ids, int):
            source_ids = (source_ids,)
        if isinstance(source_ids, (list, tuple)):
            return int(np.count_nonzero(np.isin(mask, tuple(source_ids))))
    return int(mask.size)


def _recommendation_legal_pixels(
    mask: np.ndarray,
    primitive_config: dict[str, Any],
    schema: MaskProfileSchema,
) -> int:
    name = primitive_config.get("name")
    operation = primitive_config.get("mask_operation", {})
    operation = operation if isinstance(operation, dict) else {}
    if name in {"necrosis_appearance", "intratumoral_immune_infiltration"}:
        target = operation.get("target")
        target_mask = (
            _safe_schema_label_mask(mask, schema, target)
            if isinstance(target, str)
            else np.zeros(mask.shape, dtype=bool)
        )
        return int(np.count_nonzero(np.isin(mask, schema.tumor_fine_ids) & ~target_mask))
    if name == "tumor_burden_increase":
        if not np.any(np.isin(mask, schema.tumor_fine_ids)):
            return 0
        labels = _labels_from_operation(operation.get("target_priority"))
        legal = np.zeros(mask.shape, dtype=bool)
        for label in labels:
            legal |= _safe_schema_label_mask(mask, schema, label)
        legal &= ~_safe_schema_label_mask(mask, schema, "Necrosis")
        return int(np.count_nonzero(legal))
    if name in {
        "tumor_burden_decrease",
        "necrosis_resolution",
        "immune_infiltration_decrease",
        "stroma_decrease",
        "stromal_reduction",
    }:
        source = operation.get("source")
        if isinstance(source, str):
            return int(np.count_nonzero(_safe_schema_label_mask(mask, schema, source)))
    if name == "stromal_immune_infiltration":
        return int(np.count_nonzero(_safe_schema_label_mask(mask, schema, "Stroma")))
    if name == "stromal_desmoplasia":
        sources = [
            *_labels_from_operation(operation.get("primary_sources")),
            *_labels_from_operation(operation.get("secondary_sources")),
        ]
        legal = np.zeros(mask.shape, dtype=bool)
        for label in sources:
            legal |= _safe_schema_label_mask(mask, schema, label)
        return int(np.count_nonzero(legal))
    source_ids = operation.get("source_fine_ids")
    if isinstance(source_ids, int):
        return int(np.count_nonzero(mask == source_ids))
    if isinstance(source_ids, (list, tuple)):
        return int(np.count_nonzero(np.isin(mask, tuple(source_ids))))
    source = operation.get("source")
    if isinstance(source, str):
        return int(np.count_nonzero(_safe_schema_label_mask(mask, schema, source)))
    return int(np.count_nonzero(~np.isin(mask, tuple(schema.skip_fine_ids))))


def _recommendation_dependency_failures(
    mask: np.ndarray,
    primitive_config: dict[str, Any],
    schema: MaskProfileSchema,
) -> list[str]:
    """Check current-mask target/backfill tissue needed by auto recommendations."""

    failures: list[str] = []
    for kind, labels in _recommendation_required_context_labels(
        primitive_config, schema
    ):
        if not labels:
            failures.append(f"capacity failed: no configured {kind} labels.")
            continue
        if _schema_labels_pixel_count(mask, schema, labels) > 0:
            continue
        failures.append(
            "capacity failed: no "
            f"{kind} pixels in current mask ({', '.join(labels)})."
        )
    return failures


def _recommendation_required_context_labels(
    primitive_config: dict[str, Any],
    schema: MaskProfileSchema,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    operation = primitive_config.get("mask_operation", {})
    operation = operation if isinstance(operation, dict) else {}
    required_context = primitive_config.get("required_context", ())
    required_context = required_context if isinstance(required_context, list) else ()
    required: list[tuple[str, tuple[str, ...]]] = []
    name = primitive_config.get("name")
    if name == "tumor_burden_increase":
        labels = tuple(
            _filter_schema_labels(
                _labels_from_operation(operation.get("target_priority")),
                schema,
            )
        )
        required.append(("target tissue", labels))
    if "valid_backfill_tissue" in required_context:
        labels = tuple(
            _filter_schema_labels(
                _labels_from_operation(operation.get("backfill_priority")),
                schema,
            )
        )
        required.append(("backfill tissue", labels))
    return tuple(required)


def _schema_labels_pixel_count(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    labels: tuple[str, ...],
) -> int:
    pixels = 0
    for label in labels:
        pixels += int(np.count_nonzero(_safe_schema_label_mask(mask, schema, label)))
    return pixels


def _safe_schema_label_mask(
    mask: np.ndarray,
    schema: MaskProfileSchema,
    label: str,
) -> np.ndarray:
    if label not in schema.readable_labels:
        return np.zeros(mask.shape, dtype=bool)
    return np.isin(mask, schema.resolve_fine_ids(label))


def _validation_failed_checks(validation: Any) -> list[str]:
    if validation is None:
        return []
    return [
        f"{check.name}: {check.detail}"
        for check in getattr(validation, "failed_checks", ())
    ]


def _auto_recommend_execution_failure_message(attempt_logs: list[dict[str, Any]]) -> str:
    if not attempt_logs:
        return "Selected primitive/strength could not be executed: no attempts were made."
    lines = ["Selected primitive/strength could not be executed."]
    for attempt in attempt_logs:
        lines.append(
            f"- {attempt.get('primitive')} / {attempt.get('strength')}: {attempt.get('status')}"
        )
        details = (
            attempt.get("reasons")
            or ([attempt["error"]] if attempt.get("error") else [])
            or attempt.get("validation_failed_checks")
            or attempt.get("warnings")
            or []
        )
        for detail in details[:4]:
            lines.append(f"  {detail}")
    return "\n".join(lines)


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


def _resolve_instruction_semantic_diff(
    *,
    instruction: str,
    parser: str,
    api_base_url: str,
    api_key_env: str,
    api_model: str,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    instruction = (instruction or "").strip()
    if not instruction:
        raise gr.Error("Instruction mode requires one edit instruction.")
    if parser == "rule-based":
        return parse_instruction(instruction, mode="rule-based"), {
            "mode": "instruction_rule_based",
        }
    if parser == "api":
        config = InstructionParserConfig(
            model=_defaulted_text(api_model, DEFAULT_API_MODEL),
            api_base_url=_defaulted_text(api_base_url, DEFAULT_API_BASE_URL).rstrip("/"),
            api_key_env=_defaulted_text(api_key_env, DEFAULT_API_KEY_ENV),
            timeout_sec=60.0,
            temperature=0.0,
            debug_dir=str(output_dir / "phase3_mask_edit" / "instruction_parser_debug"),
        )
        return parse_instruction(instruction, mode="api", config=config), {
            "mode": "instruction_api",
            "api_base_url": config.api_base_url,
            "api_key_env": config.api_key_env,
            "api_model": config.model,
        }
    raise gr.Error(f"Unsupported instruction parser: {parser}")


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
    with gr.Blocks(
        title="Pathology Edit Pipeline",
        css="#manualContourPayload {display:none !important;}",
    ) as demo:
        gr.Markdown("## Pathology edit pipeline")
        state = gr.State({})

        with gr.Row():
            profile = gr.Dropdown(["BCSS", "PANDA", "GlaS", "IGNITE", "PUMA", "ORCA"], value="BCSS", label="profile")
        gr.Markdown("### Tissue mask edit")
        edit_mode = gr.Radio(
            EDIT_MODE_CHOICES,
            value=EDIT_MODE_PROMPT,
            label="edit mode",
        )
        with gr.Row():
            source_image = gr.File(label="src_image", file_types=["image"], type="filepath")
            source_tissue = gr.File(label="src_tissue_mask", file_types=["image"], type="filepath")
            source_cell = gr.File(label="src_cell_mask / CellViT output", file_types=["image"], type="filepath")
        load_button = gr.Button("1. Load inputs")
        load_log = gr.Code(label="load log", language="json")
        with gr.Row():
            src_image_preview = gr.Image(label="source image")
            src_tissue_preview = gr.Image(label="source tissue")

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

        instruction_panel = gr.Column(visible=False)
        with instruction_panel:
            instruction_text = gr.Textbox(
                label="edit instruction",
                lines=3,
                placeholder="Example: make the tumor moderately larger",
            )
            instruction_parser = gr.Radio(
                ["rule-based", "api"],
                value="rule-based",
                label="instruction parser",
            )

        manual_panel = gr.Column(visible=False)
        with manual_panel:
            manual_editor = gr.HTML(label="manual contour editor")
            manual_contour_payload = gr.Textbox(
                value="",
                label="manual contour payload",
                elem_id="manualContourPayload",
            )
            with gr.Row():
                target_label = gr.Dropdown([], value=None, label="target label")
                source_labels = gr.Textbox(label="source labels", placeholder="Stroma")
            with gr.Row():
                manual_save_label_button = gr.Button("Save current tissue edit")
                manual_finalize_button = gr.Button("Save full manual mask")
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
            with gr.Row():
                auto_primitive = gr.Dropdown([], value=None, label="primitive")
                auto_strength = gr.Dropdown([], value=None, label="strength")
            auto_summary = gr.Textbox(label="selection summary", lines=6, interactive=False)

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
                parser_api_model = gr.Textbox(value=DEFAULT_API_MODEL, label="parser api model")
                contour_api_model = gr.Textbox(value=DEFAULT_API_MODEL, label="contour api model")
            with gr.Row():
                contour_api_base_url = gr.Textbox(value=DEFAULT_API_BASE_URL, label="contour api base url")
                contour_api_key_env = gr.Textbox(value=DEFAULT_API_KEY_ENV, label="contour api key env")
            with gr.Row():
                fixture_file = gr.File(label="contour fixture JSON", file_types=[".json"], type="filepath")
            with gr.Row():
                max_attempts = gr.Slider(1, 8, value=4, step=1, label="max attempts")
                max_regions = gr.Slider(1, 8, value=8, step=1, label="max regions")
                max_points_per_region = gr.Slider(8, 128, value=64, step=1, label="max points / region")
            organic_seed = gr.Number(value=0, precision=0, label="organic seed")

        auto_execute_button = gr.Button("Run selected primitive", visible=False)

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
                instruction_panel,
                manual_panel,
                auto_panel,
                manual_editor,
                manual_contour_payload,
                target_label,
                auto_primitive,
                auto_strength,
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
                instruction_panel,
                manual_panel,
                auto_panel,
                manual_editor,
                manual_contour_payload,
                target_label,
                auto_primitive,
                auto_strength,
                auto_summary,
                tissue_button,
                auto_execute_button,
            ],
        )
        auto_primitive.change(
            _refresh_auto_strength_options,
            inputs=[state, auto_primitive],
            outputs=[state, auto_strength, auto_summary],
        )
        auto_strength.change(
            _refresh_auto_selection_summary,
            inputs=[state, auto_primitive, auto_strength],
            outputs=[state, auto_summary],
        )
        profile.change(
            lambda value: tuple(_profile_defaults(value).values()),
            inputs=[profile],
            outputs=[probnet_ckpt, nuclei_library, density_scale_json],
        )
        target_label.change(
            _manual_switch_label_saving_current,
            inputs=[state, manual_contour_payload, source_labels, target_label],
            outputs=[state, manual_editor, manual_contour_payload, tissue_log, target_tissue_preview],
        )
        manual_save_label_button.click(
            _save_manual_current_label,
            inputs=[state, manual_contour_payload, source_labels, target_label],
            outputs=[state, manual_editor, manual_contour_payload, tissue_log, target_tissue_preview],
        )
        manual_finalize_button.click(
            _finalize_manual_tissue_stage,
            inputs=[state, manual_contour_payload, source_labels, target_label],
            outputs=[state, tissue_log, target_tissue_preview, change_region_preview],
        ).then(
            _refresh_edit_mode_panels,
            inputs=[state, edit_mode],
            outputs=[
                state,
                prompt_panel,
                instruction_panel,
                manual_panel,
                auto_panel,
                manual_editor,
                manual_contour_payload,
                target_label,
                auto_primitive,
                auto_strength,
                auto_summary,
                tissue_button,
                auto_execute_button,
            ],
        )
        cuda_memory_button.click(check_cuda_memory, inputs=[], outputs=[cuda_memory_log])
        tissue_button.click(
            run_tissue_stage,
            inputs=[
                state,
                edit_mode,
                old_prompt,
                new_prompt,
                instruction_text,
                instruction_parser,
                manual_editor,
                manual_contour_payload,
                auto_primitive,
                auto_strength,
                parser,
                api_base_url,
                api_key_env,
                api_model,
                parser_api_model,
                qwen_model_path,
                qwen_device,
                no_few_shot,
                source_labels,
                target_label,
                provider,
                contour_api_base_url,
                contour_api_key_env,
                contour_api_model,
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
                instruction_panel,
                manual_panel,
                auto_panel,
                manual_editor,
                manual_contour_payload,
                target_label,
                auto_primitive,
                auto_strength,
                auto_summary,
                tissue_button,
                auto_execute_button,
            ],
        )
        auto_execute_button.click(
            _run_auto_selected_from_ui,
            inputs=[
                state,
                auto_primitive,
                auto_strength,
                provider,
                contour_api_base_url,
                contour_api_key_env,
                contour_api_model,
                api_image_detail,
                fixture_file,
                max_attempts,
                max_regions,
                max_points_per_region,
                organic_seed,
            ],
            outputs=[state, tissue_log, target_tissue_preview, change_region_preview],
        ).then(
            _refresh_edit_mode_panels,
            inputs=[state, edit_mode],
            outputs=[
                state,
                prompt_panel,
                instruction_panel,
                manual_panel,
                auto_panel,
                manual_editor,
                manual_contour_payload,
                target_label,
                auto_primitive,
                auto_strength,
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
    auto_primitive: str | None,
    auto_strength: str | None,
    provider: str,
    api_base_url: str,
    api_key_env: str,
    contour_api_model: str,
    api_image_detail: str,
    fixture_file,
    max_attempts: int,
    max_regions: int,
    max_points_per_region: int,
    organic_seed: int,
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
        primitive=auto_primitive,
        strength=auto_strength,
        provider=provider,
        api_base_url=api_base_url,
        api_key_env=api_key_env,
        api_model=contour_api_model,
        api_image_detail=api_image_detail,
        fixture_file=fixture_file,
        max_attempts=max_attempts,
        max_regions=max_regions,
        max_points_per_region=max_points_per_region,
        organic_seed=organic_seed,
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
