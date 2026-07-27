#!/usr/bin/env python3
"""Inpaint-first Phase3 end-to-end pipeline wrapper.

This script intentionally keeps the first integration thin:

  - ``--mode gen`` starts from an already edited tissue mask.
  - ``--mode diff`` starts from a semantic_diff JSON and executes Phase3
    mask edits before generation.
  - ``--mode prompt`` starts from old/new pathology prompts, parses a
    semantic_diff JSON, then follows the diff path.

Both modes can run with ``--generation-mode dry-run`` so the local machine can
validate artifact layout without real data, GPU checkpoints, or model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controlnet_train.inference import (
    DEFAULT_CROSS_V1_CHECKPOINT,
    DEFAULT_INPAINT_CHECKPOINT,
    DEFAULT_PIX2PIX_CHECKPOINT,
    DEFAULT_PROBNET_CHECKPOINT,
    EditPipelineInputs,
    EditRoutingConfig,
    load_inpaint_bundle,
    run_edit_pipeline,
    run_inpaint_bundle,
)
from controlnet_train.data.common import default_prompt_for_dataset
from inpaint_cells.data import iter_class_components
from inpaint_cells.sampling_policy import widen_locally_thin_mask
from phase3_mask_edit.core.mask_io import (
    load_change_region,
    load_id_mask,
    save_change_region,
    save_id_mask,
    save_metadata,
    save_rgb_mask,
)
from phase3_mask_edit.core.gland_region import glas_whole_gland_generation_region
from phase3_mask_edit.parser.api_parser import ApiParserConfig, parse_prompts_with_api
from phase3_mask_edit.parser.qwen_local_parser import (
    QwenLocalParserConfig,
    parse_prompts_with_qwen_local,
)
from phase3_mask_edit.parser.semantic_diff import load_semantic_diff, save_semantic_diff
from phase3_mask_edit.rules.semantic_to_intent import plan_edit_intents


_INPAINT_BUNDLE_CACHE: dict[tuple[str, str, str], Any] = {}
_CROSS_V1_NO_IP_CACHE: dict[tuple[str, str, str, str, int, float, float], Any] = {}


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_image = _load_rgb_image(args.reference_image)
    reference_tissue = load_id_mask(args.reference_tissue_mask)
    reference_nuclei = _load_uint8_mask(args.reference_nuclei_mask)
    _validate_same_size(reference_image, reference_tissue, "reference_tissue_mask")
    _validate_same_size(reference_image, reference_nuclei, "reference_nuclei_mask")

    phase3_info: dict[str, Any] | None = None
    if args.mode in {"diff", "prompt"}:
        semantic_diff, parser_info = _resolve_semantic_diff(args)
        target_tissue, phase3_info = _run_phase3_semantic_stage(
            args,
            reference_tissue,
            output_dir,
            semantic_diff=semantic_diff,
            parser_info=parser_info,
        )
    else:
        if args.target_tissue_mask is None:
            raise SystemExit("--target-tissue-mask is required with --mode gen")
        target_tissue = load_id_mask(args.target_tissue_mask)

    _validate_same_size(reference_image, target_tissue, "target_tissue_mask")

    if args.change_region:
        semantic_change_region = load_change_region(args.change_region)
    else:
        semantic_change_region = reference_tissue != target_tissue
    _validate_same_size(reference_image, semantic_change_region, "change_region")
    change_region, gland_structure_policy = glas_whole_gland_generation_region(
        reference_tissue,
        target_tissue,
        semantic_change_region,
        profile=args.profile,
    )
    if args.cell_fill_mode == "probnet" and np.any(change_region):
        change_region = widen_locally_thin_mask(
            change_region,
            (reference_tissue != 0) | (target_tissue != 0),
            minimum_width=args.minimum_mask_width,
        )

    stage_paths = _save_pre_generation_artifacts(
        output_dir=output_dir,
        reference_image=reference_image,
        reference_tissue=reference_tissue,
        target_tissue=target_tissue,
        semantic_change_region=semantic_change_region,
        change_region=change_region,
    )

    target_nuclei, cell_info = _build_target_nuclei(
        args,
        reference_nuclei,
        reference_tissue,
        target_tissue,
        change_region,
        output_dir,
    )
    target_nuclei_path = save_id_mask(target_nuclei, output_dir / "target_nuclei_mask.png")
    cell_info["target_nuclei_mask"] = str(target_nuclei_path)
    cell_info["gland_structure_policy"] = gland_structure_policy
    save_metadata(cell_info, output_dir / "cell_fill_log.json")
    target_combined_path = _save_target_combined_mask(
        output_dir / "target_combined_mask.png",
        target_tissue=target_tissue,
        target_nuclei=target_nuclei,
    )

    generated_path, generation_info = _run_generation_stage(
        args=args,
        output_dir=output_dir,
        reference_image=reference_image,
        change_region=change_region,
        target_tissue_path=Path(stage_paths["target_mask"]),
        target_nuclei_path=target_nuclei_path,
    )

    panel_path = _save_compare_panel(
        output_dir / "compare_panel.png",
        reference_image=reference_image,
        erased_image=np.asarray(Image.open(stage_paths["erased_image"]).convert("RGB")),
        target_mask_rgb=np.asarray(Image.open(stage_paths["target_mask_rgb"]).convert("RGB")),
        generated_image=np.asarray(Image.open(generated_path).convert("RGB")),
        title=f"Phase3 {args.mode} / {generation_info['generation_mode']}",
        prompt=str(generation_info.get("prompt") or ""),
    )

    summary = {
        "status": "completed",
        "mode": args.mode,
        "profile": args.profile,
        "generation_mode": generation_info["generation_mode"],
        "cell_fill_mode": args.cell_fill_mode,
        "changed_pixels": int(np.count_nonzero(semantic_change_region)),
        "changed_area_fraction": (
            float(np.count_nonzero(semantic_change_region))
            / int(semantic_change_region.size)
            if semantic_change_region.size
            else 0.0
        ),
        "generation_region_pixels": int(np.count_nonzero(change_region)),
        "generation_region_area_fraction": _change_area_fraction(change_region),
        "gland_structure_policy": gland_structure_policy,
        "inputs": {
            "reference_image": str(args.reference_image),
            "reference_tissue_mask": str(args.reference_tissue_mask),
            "reference_nuclei_mask": str(args.reference_nuclei_mask),
            "target_tissue_mask": str(args.target_tissue_mask) if args.target_tissue_mask else None,
            "semantic_diff": str(args.semantic_diff) if args.semantic_diff else None,
            "old_prompt": _read_text_arg(args.old_prompt, args.old_prompt_file),
            "new_prompt": _read_text_arg(args.new_prompt, args.new_prompt_file),
        },
        "phase3": phase3_info,
        "cell_fill": cell_info,
        "generation": generation_info,
        "artifacts": {
            **stage_paths,
            "target_nuclei_mask": str(target_nuclei_path),
            "target_combined_mask": str(target_combined_path),
            "generated_image": str(generated_path),
            "compare_panel": str(panel_path),
            "generation_info": str(output_dir / "generation_info.json"),
            "cell_fill_log": str(output_dir / "cell_fill_log.json"),
        },
    }
    save_metadata(summary, output_dir / "pipeline_summary.json")

    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _resolve_semantic_diff(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.mode == "diff":
        if args.semantic_diff is None:
            raise SystemExit("--semantic-diff is required with --mode diff")
        return load_semantic_diff(args.semantic_diff), {
            "mode": "semantic_diff_file",
            "path": str(args.semantic_diff),
        }

    old_prompt = _read_text_arg(args.old_prompt, args.old_prompt_file)
    new_prompt = _read_text_arg(args.new_prompt, args.new_prompt_file)
    if old_prompt is None or new_prompt is None:
        raise SystemExit("--old-prompt/--new-prompt or prompt files are required with --mode prompt")

    if args.parser == "fixture":
        if args.semantic_diff is None:
            raise SystemExit("--semantic-diff is required with --mode prompt --parser fixture")
        return load_semantic_diff(args.semantic_diff), {
            "mode": "fixture",
            "path": str(args.semantic_diff),
        }

    if args.parser == "api":
        if not args.api_model:
            raise SystemExit("--api-model is required with --mode prompt --parser api")
        config = ApiParserConfig(
            model=args.api_model,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            timeout_sec=args.api_timeout_sec,
            temperature=args.api_temperature,
            use_few_shot=not args.no_few_shot,
            debug_dir=str(args.output / "phase3_mask_edit" / "api_parser_debug"),
        )
        return parse_prompts_with_api(old_prompt, new_prompt, config=config), {
            "mode": "api",
            "api_base_url": args.api_base_url,
            "api_key_env": args.api_key_env,
            "api_model": args.api_model,
            "use_few_shot": not args.no_few_shot,
        }

    if not args.qwen_model_path:
        raise SystemExit("--qwen-model-path is required with --mode prompt --parser qwen-local")
    config = QwenLocalParserConfig(
        model_path=args.qwen_model_path,
        device=args.qwen_device,
        max_new_tokens=args.qwen_max_new_tokens,
        temperature=args.qwen_temperature,
        top_p=args.qwen_top_p,
        do_sample=not args.qwen_greedy,
        use_few_shot=not args.no_few_shot,
    )
    return parse_prompts_with_qwen_local(old_prompt, new_prompt, config=config), {
        "mode": "qwen-local",
        "model_path": args.qwen_model_path,
        "device": args.qwen_device,
        "max_new_tokens": args.qwen_max_new_tokens,
        "temperature": args.qwen_temperature,
        "top_p": args.qwen_top_p,
        "do_sample": not args.qwen_greedy,
        "use_few_shot": not args.no_few_shot,
    }


def _run_phase3_semantic_stage(
    args: argparse.Namespace,
    reference_tissue: np.ndarray,
    output_dir: Path,
    *,
    semantic_diff: dict[str, Any],
    parser_info: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    diff_dir = output_dir / "phase3_mask_edit"
    diff_dir.mkdir(parents=True, exist_ok=True)
    save_semantic_diff(semantic_diff, diff_dir / "semantic_diff.json")
    old_prompt = _read_text_arg(args.old_prompt, args.old_prompt_file)
    new_prompt = _read_text_arg(args.new_prompt, args.new_prompt_file)
    plan = plan_edit_intents(
        semantic_diff,
        reference_profile=args.profile,
        old_mask=reference_tissue,
        old_prompt=old_prompt,
        new_prompt=new_prompt,
    )
    save_metadata(
        {"intents": [intent.to_metadata() for intent in plan.intents]},
        diff_dir / "edit_intents.json",
    )
    planning_summary = plan.to_metadata()
    planning_summary["parser"] = parser_info
    save_metadata(planning_summary, diff_dir / "planning_summary.json")

    raise RuntimeError(
        "The diff/prompt modes used the retired non-LLM deterministic primitive "
        "executor. Run organic_v2 mask editing first through the LLM contour UI/API, "
        "then call this pipeline with --mode gen and --target-tissue-mask."
    )

    return execution.target_mask, {
        "semantic_diff": str(diff_dir / "semantic_diff.json"),
        "edit_intents": str(diff_dir / "edit_intents.json"),
        "planning_summary": str(diff_dir / "planning_summary.json"),
        "mask_edit_dir": str(mask_edit_dir),
        "parser": parser_info,
        "execution": execution.to_metadata(),
    }


def _save_pre_generation_artifacts(
    *,
    output_dir: Path,
    reference_image: np.ndarray,
    reference_tissue: np.ndarray,
    target_tissue: np.ndarray,
    semantic_change_region: np.ndarray,
    change_region: np.ndarray,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    paths["reference_image"] = str(_save_rgb_array(reference_image, output_dir / "reference_image.png"))
    paths["source_mask"] = str(save_id_mask(reference_tissue, output_dir / "source_mask.png"))
    paths["target_mask"] = str(save_id_mask(target_tissue, output_dir / "target_mask.png"))
    paths["source_mask_rgb"] = str(save_rgb_mask(reference_tissue, output_dir / "source_mask_rgb.png"))
    paths["target_mask_rgb"] = str(save_rgb_mask(target_tissue, output_dir / "target_mask_rgb.png"))
    paths["semantic_change_region"] = str(
        save_change_region(
            semantic_change_region,
            output_dir / "semantic_change_region.png",
        )
    )
    paths["change_region"] = str(
        save_change_region(change_region, output_dir / "change_region.png")
    )
    paths["generation_change_region"] = paths["change_region"]

    erased = reference_image.copy()
    erased[np.asarray(change_region, dtype=bool)] = np.array([128, 128, 128], dtype=np.uint8)
    paths["erased_image"] = str(_save_rgb_array(erased, output_dir / "erased_image.png"))
    return paths


def _build_target_nuclei(
    args: argparse.Namespace,
    reference_nuclei: np.ndarray,
    reference_tissue: np.ndarray,
    target_tissue: np.ndarray,
    change_region: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    requested_policy = args.crossing_cell_policy
    effective_policy = "centroid" if args.cell_fill_mode == "probnet" else requested_policy
    retained, integrity_info = _retain_complete_reference_cells(
        reference_nuclei,
        change_region,
        policy=effective_policy,
    )
    integrity_info["requested_policy"] = requested_policy
    if requested_policy != effective_policy:
        integrity_info["policy_override"] = (
            "probnet_fill_uses_complete_component_centroid_erasure"
        )
    save_id_mask(retained, output_dir / "retained_nuclei_mask.png")

    if args.cell_fill_mode == "preserve":
        target = np.array(retained, copy=True)
        new = np.zeros_like(reference_nuclei, dtype=np.uint8)
        status = "preserved_reference_nuclei"
        shape_sampling = None
    elif args.cell_fill_mode == "blank":
        target = retained
        new = np.zeros_like(reference_nuclei, dtype=np.uint8)
        status = "blanked_change_region"
        shape_sampling = None
    else:
        target, status, shape_sampling = _run_probnet_cell_fill(
            args,
            reference_tissue,
            target_tissue,
            retained,
            reference_nuclei,
            change_region,
            output_dir,
        )
        new = np.array(target, copy=True)
        new[~np.asarray(change_region, dtype=bool)] = 0

    save_id_mask(new, output_dir / "new_nuclei_mask.png")
    return target, {
        "mode": args.cell_fill_mode,
        "status": status,
        "changed_pixels": int(np.count_nonzero(change_region)),
        "source_cell_integrity": integrity_info,
        "shape_sampling": shape_sampling,
        "retained_nuclei_mask": str(output_dir / "retained_nuclei_mask.png"),
        "new_nuclei_mask": str(output_dir / "new_nuclei_mask.png"),
        "target_combined_mask": str(output_dir / "target_combined_mask.png"),
    }


def _retain_complete_reference_cells(
    reference_nuclei: np.ndarray,
    change_region: np.ndarray,
    *,
    policy: str = "delete",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retain source nuclei as whole connected components.

    Any source cell touching the changed region is handled as a complete
    component, so the target mask never contains clipped source-cell fragments.
    """
    if policy not in {"delete", "keep", "majority", "centroid"}:
        raise ValueError(f"Unsupported crossing-cell-policy: {policy}")

    from scipy import ndimage

    source = np.asarray(reference_nuclei, dtype=np.uint8)
    changed = np.asarray(change_region, dtype=bool)
    labeled, count = ndimage.label(source > 0)
    retained = np.zeros_like(source, dtype=np.uint8)
    stats = {
        "policy": policy,
        "source_components": int(count),
        "kept_components": 0,
        "deleted_components": 0,
        "crossing_components": 0,
        "inside_change_components": 0,
        "outside_change_components": 0,
    }

    if policy == "centroid":
        stats["source_components"] = 0
        height, width = changed.shape
        for _, component, (centroid_y, centroid_x) in iter_class_components(source):
            stats["source_components"] += 1
            touches_change = bool(np.any(component & changed))
            touches_unchanged = bool(np.any(component & ~changed))
            if touches_change and touches_unchanged:
                stats["crossing_components"] += 1
            elif touches_change:
                stats["inside_change_components"] += 1
            else:
                stats["outside_change_components"] += 1
            row = int(np.clip(round(centroid_y), 0, height - 1))
            col = int(np.clip(round(centroid_x), 0, width - 1))
            if changed[row, col]:
                stats["deleted_components"] += 1
            else:
                retained[component] = source[component]
                stats["kept_components"] += 1
        return retained, stats

    for component_id in range(1, count + 1):
        component = labeled == component_id
        touches_change = bool(np.any(component & changed))
        touches_unchanged = bool(np.any(component & ~changed))

        keep = False
        if touches_change and touches_unchanged:
            stats["crossing_components"] += 1
            if policy == "keep":
                keep = True
            elif policy == "majority":
                keep = int(np.count_nonzero(component & ~changed)) >= int(np.count_nonzero(component & changed))
            else:
                keep = False
        elif touches_change:
            stats["inside_change_components"] += 1
            keep = policy == "keep"
        else:
            stats["outside_change_components"] += 1
            keep = True

        if keep:
            retained[component] = source[component]
            stats["kept_components"] += 1
        else:
            stats["deleted_components"] += 1

    return retained, stats


def _run_probnet_cell_fill(
    args: argparse.Namespace,
    reference_tissue: np.ndarray,
    target_tissue: np.ndarray,
    retained_nuclei: np.ndarray,
    full_reference_nuclei: np.ndarray,
    change_region: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, str, dict[str, Any] | None]:
    missing = [
        name for name, value in {
            "--probnet-ckpt": args.probnet_ckpt,
            "--nuclei-library": args.nuclei_library,
        }.items()
        if value is None
    ]
    if missing:
        raise SystemExit(f"{', '.join(missing)} required with --cell-fill-mode probnet")

    cell_dir = output_dir / "probnet_cell_fill"
    cell_dir.mkdir(parents=True, exist_ok=True)
    source_tissue = save_id_mask(
        reference_tissue,
        cell_dir / "reference_tissue.png",
    )
    input_tissue = save_id_mask(target_tissue, cell_dir / "input_tissue.png")
    input_nuclei = save_id_mask(retained_nuclei, cell_dir / "input_nuclei.png")
    reference_nuclei_shapes = save_id_mask(
        full_reference_nuclei,
        cell_dir / "reference_nuclei_shapes.png",
    )
    edit_region = save_change_region(change_region, cell_dir / "edit_region.png")
    output_nuclei = cell_dir / "target_nuclei.png"
    probnet_device, probnet_env, visible_device = _normalize_probnet_device(args.probnet_device)

    cmd = [
        sys.executable,
        "inpaint_cells/generate.py",
        "--dataset",
        args.profile,
        "--ckpt",
        str(args.probnet_ckpt),
        "--library",
        str(args.nuclei_library),
        "--input-tissue",
        str(input_tissue),
        "--reference-tissue",
        str(source_tissue),
        "--input-nuclei",
        str(input_nuclei),
        "--reference-nuclei-shapes",
        str(reference_nuclei_shapes),
        "--edit-region",
        str(edit_region),
        "--output",
        str(output_nuclei),
        "--device",
        probnet_device,
        "--gamma-values",
        args.probnet_gamma_values,
        "--minimum-mask-width",
        str(args.minimum_mask_width),
        "--no-widen-edit-region",
        "--reference-shape-min-area",
        str(getattr(args, "reference_shape_min_area", 8)),
        "--reference-shape-max-area-ratio",
        str(getattr(args, "reference_shape_max_area_ratio", 0.0)),
        "--library-size-min-scale",
        str(getattr(args, "library_size_min_scale", 0.5)),
        "--library-size-max-scale",
        str(getattr(args, "library_size_max_scale", 2.0)),
        "--library-size-log-area-jitter",
        str(getattr(args, "library_size_log_area_jitter", 0.05)),
    ]
    if getattr(args, "include_border_reference_shapes", False):
        cmd.append("--include-border-reference-shapes")
    if getattr(args, "disable_library_size_calibration", False):
        cmd.append("--disable-library-size-calibration")
    if args.density_scale_json:
        cmd.extend(["--density-scale-json", str(args.density_scale_json)])
    display_cmd = (
        f"CUDA_VISIBLE_DEVICES={visible_device} " + " ".join(map(str, cmd))
        if visible_device is not None
        else " ".join(map(str, cmd))
    )

    try:
        subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            env=probnet_env,
        )
    except subprocess.CalledProcessError as exc:
        details = _format_subprocess_error(exc, label="ProbNet generate.py")
        if visible_device is not None:
            details = details.replace("Command: " + str(cmd), "Command: " + display_cmd)
        raise RuntimeError(details) from exc
    diagnostics_path = output_nuclei.with_suffix(".diagnostics.json")
    shape_sampling = None
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics:
            first = diagnostics[0]
            shape_sampling = {
                "reference_pool": first.get("reference_pool"),
                "placed_by_shape_source": first.get("placed_by_shape_source"),
                "sampling": first.get("shape_sampling"),
                "diagnostics_path": str(diagnostics_path),
            }
    return _load_uint8_mask(output_nuclei), "probnet_generated", shape_sampling


def _normalize_probnet_device(device: str | None) -> tuple[str, dict[str, str] | None, str | None]:
    """Map cuda:N UI choices onto generate.py's auto/cuda/cpu CLI contract."""

    value = (device or "auto").strip().lower()
    if value in {"auto", "cuda", "cpu"}:
        return value, None, None
    if value.startswith("cuda:"):
        index = value.split(":", 1)[1]
        if index.isdigit():
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = index
            return "cuda", env, index
    raise ValueError(
        f"Unsupported ProbNet device {device!r}; choose auto, cuda, cpu, or cuda:<index>."
    )


def _run_generation_stage(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    reference_image: np.ndarray,
    change_region: np.ndarray,
    target_tissue_path: Path,
    target_nuclei_path: Path,
) -> tuple[Path, dict[str, Any]]:
    if args.generation_mode == "dry-run":
        erased = reference_image.copy()
        erased[np.asarray(change_region, dtype=bool)] = np.array([128, 128, 128], dtype=np.uint8)
        generated = _save_rgb_array(erased, output_dir / "generated_image.png")
        change_ratio = _change_area_fraction(change_region)
        selected_mode = _select_generation_mode(
            args.generation_mode,
            change_ratio,
            args.route_threshold,
            cross_backend=getattr(args, "cross_backend", "cross-v1"),
        )
        info = {
            "generation_mode": "dry-run",
            "status": "skipped_model_generation",
            "generated_image": str(generated),
            "selected_mode": selected_mode,
            "change_ratio": change_ratio,
            "route_threshold": args.route_threshold,
        }
        save_metadata(info, output_dir / "generation_info.json")
        return generated, info

    change_ratio = _change_area_fraction(change_region)
    selected_mode = _select_generation_mode(
        args.generation_mode,
        change_ratio,
        args.route_threshold,
        cross_backend=getattr(args, "cross_backend", "cross-v1"),
    )
    required = {
        "--pretrained-model-name-or-path": args.pretrained_model_name_or_path,
    }
    if selected_mode == "inpaint":
        required["--inpaint-checkpoint"] = args.inpaint_checkpoint
    else:
        required["--cross-v1-checkpoint"] = args.cross_v1_checkpoint
        required["--pix2pix-checkpoint"] = args.pix2pix_checkpoint
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"{', '.join(missing)} required with --generation-mode {args.generation_mode}")

    if selected_mode == "cross-v1":
        prompt = _resolve_cross_v1_prompt(
            prompt_override=args.prompt,
            prompt_source=getattr(args, "prompt_source", "dataset"),
            dataset=args.profile,
        )
        image, raw_image, no_ip_info = _run_cross_v1_no_ip_generation(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            checkpoint_path=args.cross_v1_checkpoint,
            pix2pix_checkpoint_path=getattr(args, "pix2pix_checkpoint", None),
            reference_image_path=args.reference_image,
            reference_tissue_mask_path=args.reference_tissue_mask,
            reference_nuclei_mask_path=args.reference_nuclei_mask,
            target_tissue_mask_path=target_tissue_path,
            target_nuclei_mask_path=target_nuclei_path,
            prompt=prompt,
            output_dir=output_dir / "controlnet_cross_v1_no_ip",
            device=args.device,
            torch_dtype=_parse_torch_dtype(getattr(args, "torch_dtype", "bf16")),
            num_inference_steps=getattr(args, "num_inference_steps", 28),
            guidance_scale=getattr(args, "guidance_scale", 3.5),
            controlnet_conditioning_scale=getattr(args, "controlnet_conditioning_scale", 1.0),
            seed=getattr(args, "seed", 42),
        )
        generated = output_dir / "generated_image.png"
        raw_generated = output_dir / "generated_image_raw.png"
        raw_image.save(raw_generated)
        image.save(generated)
        info = {
            "generation_mode": args.generation_mode,
            "status": "generated",
            "generated_image": str(generated),
            "raw_generated_image": str(raw_generated),
            "controlnet_output_dir": str(output_dir / "controlnet_cross_v1_no_ip"),
            "selected_mode": selected_mode,
            "change_ratio": change_ratio,
            "route_threshold": args.route_threshold,
            "prompt": prompt,
            "color_match": {"method": "none", "applied": False, "reference": str(args.reference_image)},
            "cross_v1": no_ip_info,
        }
        save_metadata(info, output_dir / "generation_info.json")
        return generated, info

    controlnet_dir = output_dir / f"controlnet_{selected_mode.replace('-', '_')}"
    with torch.inference_mode():
        prompt = _resolve_cross_v1_prompt(
            prompt_override=args.prompt,
            prompt_source=getattr(args, "prompt_source", "dataset"),
            dataset=args.profile,
        )
        result = run_edit_pipeline(
            inputs=EditPipelineInputs(
                reference_image=args.reference_image,
                reference_tissue_mask=args.reference_tissue_mask,
                reference_nuclei_mask=args.reference_nuclei_mask,
                target_tissue_mask=target_tissue_path,
                target_nuclei_mask=target_nuclei_path,
                output_dir=controlnet_dir,
                prompt=prompt,
                dataset=args.profile,
                force_mode=selected_mode if selected_mode == "inpaint" else "cross",
                save_debug_artifacts=True,
            ),
            inpaint_bundle=(
                _cached_inpaint_bundle(
                    pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                    checkpoint_path=args.inpaint_checkpoint,
                    device=args.device,
                )
                if selected_mode == "inpaint"
                else object()
            ),
            cross_bundle=object(),
            inpaint_runner=run_inpaint_bundle,
            cross_runner=_production_cross_is_handled_before_inpaint,
            routing_config=EditRoutingConfig(t_inpaint=args.route_threshold, t_cross=args.route_threshold),
        )
    generated = output_dir / "generated_image.png"
    raw_generated = output_dir / "generated_image_raw.png"
    result.image.save(raw_generated)
    result.image.save(generated)
    info = {
        "generation_mode": args.generation_mode,
        "status": "generated",
        "generated_image": str(generated),
        "raw_generated_image": str(raw_generated),
        "controlnet_output_dir": str(controlnet_dir),
        "selected_mode": selected_mode,
        "change_ratio": result.change_ratio,
        "route_threshold": args.route_threshold,
        "prompt": result.prompt,
        "color_match": {"method": "none", "applied": False},
    }
    save_metadata(info, output_dir / "generation_info.json")
    return generated, info


def _cached_inpaint_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
):
    key = (
        str(pretrained_model_name_or_path),
        str(checkpoint_path),
        str(device),
    )
    if key not in _INPAINT_BUNDLE_CACHE:
        _INPAINT_BUNDLE_CACHE[key] = load_inpaint_bundle(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    return _INPAINT_BUNDLE_CACHE[key]


def _production_cross_is_handled_before_inpaint(*_args, **_kwargs):
    raise RuntimeError("Production cross-v1 must be handled before the inpaint-only pipeline.")


def _cached_cross_v1_no_ip_components(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
    torch_dtype: torch.dtype,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
):
    key = (
        str(pretrained_model_name_or_path),
        str(checkpoint_path),
        str(device),
        str(torch_dtype),
        int(num_inference_steps),
        float(guidance_scale),
        float(controlnet_conditioning_scale),
    )
    if key not in _CROSS_V1_NO_IP_CACHE:
        from controlnet_train.inference.pipeline_cross_v1 import (
            _load_cross_v1_control_spec,
            _load_flux_controlnet_pipeline,
            _validate_checkpoint_dir,
        )
        from scripts.generate_cross_v1_no_ip_strict import _load_condition_modules_no_ref

        checkpoint = _validate_checkpoint_dir(checkpoint_path)
        control_spec = _load_cross_v1_control_spec(checkpoint)
        pipe, controlnet = _load_flux_controlnet_pipeline(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            checkpoint_path=checkpoint,
            packed_channels=control_spec.packed_channels,
            device=device,
            torch_dtype=torch_dtype,
        )
        modules = _load_condition_modules_no_ref(
            checkpoint_path=checkpoint,
            device=device,
            torch_dtype=torch_dtype,
        )
        _CROSS_V1_NO_IP_CACHE[key] = (pipe, controlnet, modules, control_spec)
    return _CROSS_V1_NO_IP_CACHE[key]


@torch.inference_mode()
def _run_cross_v1_no_ip_generation(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    pix2pix_checkpoint_path: str | Path,
    reference_image_path: str | Path,
    reference_tissue_mask_path: str | Path,
    reference_nuclei_mask_path: str | Path,
    target_tissue_mask_path: str | Path,
    target_nuclei_mask_path: str | Path,
    prompt: str,
    output_dir: Path,
    device: str,
    torch_dtype: torch.dtype,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    from controlnet_train.data.common import load_nuclei_mask, load_tissue_mask
    from controlnet_train.inference.pipeline_cross_v1 import _sample_with_flux_controlnet
    from controlnet_train.modules.cross_v1_conditioning import build_cross_v1_condition
    from scripts.generate_cross_v1_no_ip_strict import _run_pix2pix_transfer

    output_dir.mkdir(parents=True, exist_ok=True)
    pipe, controlnet, modules, control_spec = _cached_cross_v1_no_ip_components(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint_path,
        device=device,
        torch_dtype=torch_dtype,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )

    target_tissue_mask = load_tissue_mask(target_tissue_mask_path)
    target_nuclei_mask = load_nuclei_mask(target_nuclei_mask_path)
    target_tissue_feat = modules["tissue_downsampler"](
        modules["hte"](target_tissue_mask.unsqueeze(0).to(device=device))
    ).to(dtype=torch_dtype)
    target_nuclei_feat = modules["nuclei_encoder"](
        target_nuclei_mask.unsqueeze(0).to(device=device)
    ).to(dtype=torch_dtype)

    reference_tissue_feat = None
    reference_nuclei_feat = None
    stage1_reference_mask_mode = "target"
    if control_spec.spatial_mode in {"reference_target", "reference_target_delta"}:
        reference_tissue_feat = target_tissue_feat
        reference_nuclei_feat = target_nuclei_feat

    control_tensor = build_cross_v1_condition(
        reference_tissue_feat=reference_tissue_feat,
        reference_nuclei_feat=reference_nuclei_feat,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
        spatial_mode=control_spec.spatial_mode,
    )
    output_size = tuple(int(v) for v in target_tissue_mask.shape[-2:])
    stage1_image = _sample_with_flux_controlnet(
        pipe=pipe,
        controlnet=controlnet,
        prompt=prompt,
        control_tensor=control_tensor,
        output_size=output_size,
        device=device,
        torch_dtype=torch_dtype,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        joint_attention_kwargs=None,
        seed=seed,
    )
    stage1_path = output_dir / "stage1_no_ip.png"
    stage1_image.save(stage1_path)

    pix2pix_output_path = output_dir / "stage2_pix2pix.png"
    record = {
        "reference_image": str(reference_image_path),
        "reference_tissue_mask": str(reference_tissue_mask_path),
        "reference_nuclei_mask": str(reference_nuclei_mask_path),
        "target_tissue_mask": str(target_tissue_mask_path),
        "target_nuclei_mask": str(target_nuclei_mask_path),
    }
    pix2pix_info = _run_pix2pix_transfer(
        i0_image=stage1_image,
        record=record,
        checkpoint_path=pix2pix_checkpoint_path,
        output_path=pix2pix_output_path,
        device=device,
        torch_dtype=torch_dtype,
        image_size=int(output_size[0]),
    )
    final_image = Image.open(pix2pix_output_path).convert("RGB")

    info = {
        "backend": "cross-v1-no-ip-pix2pix-v2",
        "loads_ip_adapter": False,
        "loads_uni": False,
        "checkpoint": str(checkpoint_path),
        "stage1_no_ip_image": str(stage1_path),
        "stage1_reference_mask_mode": stage1_reference_mask_mode,
        "pix2pix_checkpoint": str(pix2pix_checkpoint_path),
        "pix2pix_output": str(pix2pix_output_path),
        "pix2pix_v2": pix2pix_info,
        "controlnet_conditioning_scale": float(controlnet_conditioning_scale),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "seed": int(seed),
    }
    return final_image, stage1_image, info


def _change_area_fraction(change_region: np.ndarray) -> float:
    changed = np.asarray(change_region, dtype=bool)
    return float(np.count_nonzero(changed)) / int(changed.size) if changed.size else 0.0


def _select_generation_mode(
    generation_mode: str,
    change_ratio: float,
    threshold: float,
    *,
    cross_backend: str = "cross-v1",
) -> str:
    if generation_mode == "auto":
        return "inpaint" if change_ratio < threshold else cross_backend
    if generation_mode in {"inpaint", "cross-v1"}:
        return generation_mode
    return "dry-run"
def _save_compare_panel(
    path: Path,
    *,
    reference_image: np.ndarray,
    erased_image: np.ndarray,
    target_mask_rgb: np.ndarray,
    generated_image: np.ndarray,
    title: str,
    prompt: str,
) -> Path:
    panels = [
        ("Reference", reference_image),
        ("Erased", erased_image),
        ("Target mask", target_mask_rgb),
        ("Generated", generated_image),
    ]
    h, w = reference_image.shape[:2]
    header_h = 52
    footer_h = 26
    gap = 4
    out = Image.new("RGB", (w * len(panels) + gap * (len(panels) - 1), h + header_h + footer_h), (245, 245, 245))
    draw = ImageDraw.Draw(out)
    for idx, (label, array) in enumerate(panels):
        x = idx * (w + gap)
        out.paste(Image.fromarray(array).resize((w, h)), (x, header_h))
        draw.text((x + 5, 7), label, fill=(0, 0, 0))
    draw.text((5, 24), title[:180], fill=(0, 0, 0))
    draw.text((5, 38), f"prompt: {prompt[:220]}", fill=(0, 0, 0))
    draw.text((5, h + header_h + 6), title[:180], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path


def _match_image_color_to_reference(
    *,
    source: np.ndarray,
    reference: np.ndarray,
    method: str,
) -> np.ndarray:
    if method == "lab":
        return _mean_std_transfer_pil_lab(source=source, reference=reference)
    raise ValueError(f"Unsupported color match method: {method}")


def _mean_std_transfer_pil_lab(*, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    from skimage.color import lab2rgb, rgb2lab

    source_rgb = np.asarray(source, dtype=np.float32) / 255.0
    reference_rgb = np.asarray(reference, dtype=np.float32) / 255.0
    source_lab = rgb2lab(source_rgb).astype(np.float32)
    reference_lab = rgb2lab(reference_rgb).astype(np.float32)
    source_mask = _tissue_mask_from_rgb(source_rgb)
    reference_mask = _tissue_mask_from_rgb(reference_rgb)

    if not np.any(source_mask) or not np.any(reference_mask):
        return np.asarray(source, dtype=np.uint8)

    matched_lab = source_lab.copy()
    for channel in range(3):
        source_values = source_lab[..., channel][source_mask]
        reference_values = reference_lab[..., channel][reference_mask]
        source_std = float(source_values.std())
        reference_std = float(reference_values.std())
        matched_lab[..., channel][source_mask] = (
            (source_values - float(source_values.mean()))
            * (reference_std / max(source_std, 1e-6))
            + float(reference_values.mean())
        )

    matched_rgb = np.clip(lab2rgb(matched_lab), 0.0, 1.0)
    output = source_rgb.copy()
    output[source_mask] = matched_rgb[source_mask]
    return (output * 255.0).round().astype(np.uint8)


def _tissue_mask_from_rgb(rgb_float: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    return rgb_float.mean(axis=-1) < threshold


def _save_target_combined_mask(
    path: Path,
    *,
    target_tissue: np.ndarray,
    target_nuclei: np.ndarray,
) -> Path:
    from phase3_mask_edit.core.mask_io import id_to_rgb

    combined = id_to_rgb(target_tissue)
    combined[np.asarray(target_nuclei) > 0] = np.array([255, 255, 255], dtype=np.uint8)
    return _save_rgb_array(combined, path)


def _format_subprocess_error(exc: subprocess.CalledProcessError, *, label: str) -> str:
    parts = [f"{label} failed with exit code {exc.returncode}."]
    if exc.cmd:
        parts.append(f"Command: {exc.cmd!r}")
    stdout = (exc.stdout or "").strip()
    stderr = (exc.stderr or "").strip()
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n".join(parts)


def _load_rgb_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_uint8_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _save_rgb_array(array: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(p)
    return p


def _validate_same_size(image: np.ndarray, mask: np.ndarray, label: str) -> None:
    expected = tuple(int(v) for v in image.shape[:2])
    actual = tuple(int(v) for v in mask.shape[:2])
    if actual != expected:
        raise ValueError(f"{label} must match reference image size {expected}, got {actual}.")


def _read_text_arg(value: str | None, path: Path | None) -> str | None:
    if value is not None and path is not None:
        raise SystemExit("Provide either direct prompt text or prompt file, not both.")
    if path is None:
        return value
    return path.read_text(encoding="utf-8").strip()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase3 end-to-end tissue/cell/edit pipeline.")
    parser.add_argument("--mode", choices=("gen", "diff", "prompt"), required=True)
    parser.add_argument("--profile", required=True, help="Reference profile, e.g. BCSS.")
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--reference-tissue-mask", required=True, type=Path)
    parser.add_argument("--reference-nuclei-mask", required=True, type=Path)
    parser.add_argument("--target-tissue-mask", type=Path, help="Edited tissue mask for --mode gen.")
    parser.add_argument("--change-region", type=Path, help="Optional explicit change-region mask.")
    parser.add_argument(
        "--semantic-diff",
        type=Path,
        help="semantic_diff JSON for --mode diff, or fixture input for --mode prompt --parser fixture.",
    )
    parser.add_argument("--old-prompt", default=None)
    parser.add_argument("--new-prompt", default=None)
    parser.add_argument("--old-prompt-file", type=Path)
    parser.add_argument("--new-prompt-file", type=Path)
    parser.add_argument(
        "--parser",
        choices=("api", "qwen-local", "fixture"),
        default="api",
        help="Prompt parser used by --mode prompt.",
    )
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-model")
    parser.add_argument("--api-timeout-sec", type=float, default=60.0)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-model-path")
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=256)
    parser.add_argument("--qwen-temperature", type=float, default=0.1)
    parser.add_argument("--qwen-top-p", type=float, default=0.9)
    parser.add_argument("--qwen-greedy", action="store_true")
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--continue-on-failure", action="store_true")

    parser.add_argument(
        "--cell-fill-mode",
        choices=("preserve", "blank", "probnet"),
        default="preserve",
        help="How to build target nuclei before ControlNet generation.",
    )
    parser.add_argument(
        "--crossing-cell-policy",
        choices=("delete", "keep", "majority"),
        default="delete",
        help="How to handle source nuclei components touching both changed and unchanged regions.",
    )
    parser.add_argument("--probnet-ckpt", type=Path, default=Path(DEFAULT_PROBNET_CHECKPOINT))
    parser.add_argument("--nuclei-library", type=Path)
    parser.add_argument("--probnet-device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--probnet-gamma-values", default="1.5")
    parser.add_argument("--minimum-mask-width", type=int, default=33)
    parser.add_argument("--density-scale-json", type=Path)
    parser.add_argument("--reference-shape-min-area", type=int, default=8)
    parser.add_argument("--reference-shape-max-area-ratio", type=float, default=0.0)
    parser.add_argument("--include-border-reference-shapes", action="store_true")
    parser.add_argument("--disable-library-size-calibration", action="store_true")
    parser.add_argument("--library-size-min-scale", type=float, default=0.5)
    parser.add_argument("--library-size-max-scale", type=float, default=2.0)
    parser.add_argument("--library-size-log-area-jitter", type=float, default=0.05)

    parser.add_argument(
        "--generation-mode",
        choices=("dry-run", "inpaint", "cross-v1", "auto"),
        default="dry-run",
        help="dry-run skips generation; auto uses local inpaint below the threshold and production cross-v1 above it.",
    )
    parser.add_argument("--route-threshold", type=float, default=0.35)
    parser.add_argument(
        "--cross-backend",
        choices=("cross-v1",),
        default="cross-v1",
        help="Cross model used by --generation-mode auto.",
    )
    parser.add_argument("--pretrained-model-name-or-path")
    parser.add_argument("--inpaint-checkpoint", type=Path, default=Path(DEFAULT_INPAINT_CHECKPOINT))
    parser.add_argument("--cross-v1-checkpoint", type=Path, default=Path(DEFAULT_CROSS_V1_CHECKPOINT))
    parser.add_argument("--pix2pix-checkpoint", type=Path, default=Path(DEFAULT_PIX2PIX_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-source", choices=("metadata", "dataset"), default="dataset")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--color-match",
        choices=("none", "lab"),
        default="lab",
        help="Postprocess cross-v1 output to match reference stain/color statistics.",
    )
    parser.add_argument("--print-summary", action="store_true")
    return parser


def _parse_torch_dtype(name: str) -> torch.dtype:
    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return dtype_by_name[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {name!r}") from exc


def _resolve_cross_v1_prompt(*, prompt_override: str | None, prompt_source: str, dataset: str) -> str:
    if prompt_override:
        return prompt_override
    if prompt_source == "dataset":
        return default_prompt_for_dataset(dataset)
    return default_prompt_for_dataset(dataset)


if __name__ == "__main__":
    raise SystemExit(main())
