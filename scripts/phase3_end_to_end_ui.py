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

from phase3_mask_edit.core.mask_io import load_change_region, load_id_mask, save_change_region, save_id_mask
from scripts.run_phase3_inpaint_pipeline import (
    _build_target_nuclei,
    _change_area_fraction,
    _load_rgb_image,
    _load_uint8_mask,
    _read_text_arg,
    _resolve_semantic_diff,
    _run_generation_stage,
    _run_phase3_semantic_stage,
    _save_compare_panel,
    _save_pre_generation_artifacts,
    _save_target_combined_mask,
    _select_generation_mode,
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
        "mode": "prompt",
        "profile": state.get("profile", "BCSS"),
        "reference_image": Path(state["reference_image"]),
        "reference_tissue_mask": Path(state["reference_tissue_mask"]),
        "reference_nuclei_mask": Path(state["reference_nuclei_mask"]),
        "target_tissue_mask": Path(state["target_tissue_mask"]) if state.get("target_tissue_mask") else None,
        "change_region": Path(state["change_region"]) if state.get("change_region") else None,
        "semantic_diff": None,
        "old_prompt": None,
        "new_prompt": None,
        "old_prompt_file": None,
        "new_prompt_file": None,
        "parser": "api",
        "api_base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "api_model": None,
        "api_timeout_sec": 60.0,
        "api_temperature": 0.0,
        "qwen_model_path": None,
        "qwen_device": "cuda",
        "qwen_max_new_tokens": 256,
        "qwen_temperature": 0.1,
        "qwen_top_p": 0.9,
        "qwen_greedy": False,
        "no_few_shot": False,
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
    mode: str,
    old_prompt: str,
    new_prompt: str,
    parser: str,
    api_model: str,
    qwen_model_path: str,
    semantic_diff_file,
    target_tissue_file,
    continue_on_failure: bool,
) -> tuple[dict[str, Any], str, str, str]:
    if not state:
        raise gr.Error("Load inputs first.")
    output_dir = Path(state["output_dir"])
    reference_image = _load_rgb_image(state["reference_image"])
    reference_tissue = load_id_mask(state["reference_tissue_mask"])

    if mode == "target mask":
        target_path = _copy_input(target_tissue_file, output_dir, "target_tissue_mask.png")
        target_tissue = load_id_mask(target_path)
        phase3_info = {"mode": "target_mask_upload", "target_tissue_mask": str(target_path)}
    else:
        diff_path = _file_path(semantic_diff_file)
        args = _make_args(
            state,
            mode="diff" if mode == "semantic diff" else "prompt",
            semantic_diff=diff_path,
            old_prompt=old_prompt or None,
            new_prompt=new_prompt or None,
            parser=parser,
            api_model=api_model or None,
            qwen_model_path=qwen_model_path or None,
            continue_on_failure=continue_on_failure,
        )
        semantic_diff, parser_info = _resolve_semantic_diff(args)
        target_tissue, phase3_info = _run_phase3_semantic_stage(
            args,
            reference_tissue,
            output_dir,
            semantic_diff=semantic_diff,
            parser_info=parser_info,
        )
        target_path = save_id_mask(target_tissue, output_dir / "target_mask.png")

    _validate_same_size(reference_image, target_tissue, "target_tissue_mask")
    change_region = reference_tissue != target_tissue
    stage_paths = _save_pre_generation_artifacts(
        output_dir=output_dir,
        reference_image=reference_image,
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
    target_nuclei, cell_info = _build_target_nuclei(args, reference_nuclei, target_tissue, change_region, output_dir)
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
            tissue_mode = gr.Radio(["prompt", "semantic diff", "target mask"], value="prompt", label="mode")
            parser = gr.Radio(["api", "qwen-local", "fixture"], value="api", label="parser")
        with gr.Row():
            old_prompt = gr.Textbox(label="src_prompt", lines=3)
            new_prompt = gr.Textbox(label="new_prompt", lines=3)
        with gr.Row():
            api_model = gr.Textbox(label="api model")
            qwen_model_path = gr.Textbox(label="qwen model path")
        with gr.Row():
            semantic_diff = gr.File(label="semantic_diff JSON", file_types=[".json"], type="filepath")
            target_tissue = gr.File(label="target_tissue_mask", file_types=["image"], type="filepath")
        continue_on_failure = gr.Checkbox(value=False, label="continue on Phase3 failure")
        tissue_button = gr.Button("2. Run LLM parser + organic v2 tissue edit")
        tissue_log = gr.Code(label="tissue log", language="json")
        with gr.Row():
            target_tissue_preview = gr.Image(label="target tissue")
            change_region_preview = gr.Image(label="change region")

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
                tissue_mode,
                old_prompt,
                new_prompt,
                parser,
                api_model,
                qwen_model_path,
                semantic_diff,
                target_tissue,
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
