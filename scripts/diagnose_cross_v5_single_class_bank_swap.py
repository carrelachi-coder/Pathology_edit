#!/usr/bin/env python
"""Cross V5 single-class TissueBank swap diagnostic.

Inference-only check for the core V5 hypothesis:
fixed target geometry + fixed seed + only TissueBank[class_id] swapped should
change appearance mostly inside that target class and leave other classes stable.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_config import COARSE_LABELS, FINE_TO_PARENT, NUM_COARSE  # noqa: E402
from controlnet_train.data.common import load_image_tensor, load_nuclei_mask, load_tissue_mask  # noqa: E402
from controlnet_train.modules.cross_v5_conditioning import (  # noqa: E402
    CrossV5GeometryControlSpec,
    CrossV5PriorPrototypeBank,
    CrossV5RefBankBuilder,
    CrossV5SpatialAdaLNModulator,
    CrossV5TissueBank,
    build_cross_v5_geometry_control_condition,
    build_cross_v5_hed_stat_prototypes,
    build_cross_v5_spatial_structure_tokens,
)
from controlnet_train.training.conditioning import patch_controlnet_x_embedder  # noqa: E402
from controlnet_train.training.cross_v5_flux_adapters import (  # noqa: E402
    CROSS_V5_ADALN_SCALE_KEY,
    CROSS_V5_BANK_KEY,
    CROSS_V5_FALLBACK_PROTOTYPES_KEY,
    CROSS_V5_TARGET_CLASS_IDS_KEY,
    CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY,
    install_cross_v5_flux_adaln_adapters,
)
from controlnet_train.inference.pipeline_cross_v3 import _calculate_shift  # noqa: E402


CROSS_V5_PROMPT = "histopathology image"


@dataclass
class CrossV5SwapBundle:
    pretrained_model_name_or_path: str | Path
    checkpoint_path: Path
    device: str
    torch_dtype: torch.dtype
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    flux_pipeline: Any
    controlnet: Any
    modules: dict[str, torch.nn.Module] = field(default_factory=dict)
    control_spec: CrossV5GeometryControlSpec = field(default_factory=CrossV5GeometryControlSpec)
    adaln_double_blocks: str = "last"


@dataclass(frozen=True)
class SwapSelection:
    target_record: dict
    swap_record: dict
    class_id: int
    score: float
    target_fraction: float
    ref_a_fraction: float
    ref_b_fraction: float
    hed_distance: float
    same_target: bool
    cross_case: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Cross V5 single-class TissueBank swap.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V5 checkpoint dir, e.g. checkpoint-6000.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_train/val.json.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--min-target-class-fraction", type=float, default=0.06)
    parser.add_argument("--min-reference-class-fraction", type=float, default=0.03)
    parser.add_argument("--candidate-limit", type=int, default=2000)
    parser.add_argument("--classes", default="1,2,3,4,5,6,7", help="Coarse class IDs to consider.")
    parser.add_argument("--prefer-same-target", action="store_true", default=True)
    parser.add_argument("--allow-different-target", action="store_true")
    parser.add_argument("--adaln-double-blocks", default=None, help="Override saved/launch AdaLN double block indices.")
    parser.add_argument(
        "--adaln-scales",
        default="1,3,10",
        help=(
            "Comma-separated inference-time multipliers for Cross V5 AdaLN gamma/beta delta. "
            "Use an empty string to skip the scale sweep."
        ),
    )
    parser.add_argument("--thumbnail-size", type=int, default=192)
    parser.add_argument("--save-individual-images", action="store_true")
    parser.add_argument(
        "--dry-run-selection",
        action="store_true",
        help="Only auto-select swap samples and write selected_samples.json; do not load FLUX or run inference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)
    records = read_cross_metadata(args.metadata)
    args.adaln_scales = parse_float_list(args.adaln_scales)
    selections = select_swap_samples(
        records,
        num_samples=args.num_samples,
        class_ids=parse_class_ids(args.classes),
        min_target_fraction=args.min_target_class_fraction,
        min_reference_fraction=args.min_reference_class_fraction,
        candidate_limit=args.candidate_limit,
        prefer_same_target=bool(args.prefer_same_target),
        allow_different_target=bool(args.allow_different_target),
        rng=rng,
    )
    if not selections:
        raise SystemExit("No V5 single-class swap candidates found. Lower min fractions or allow different target.")

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    write_selection_summary(selections, output_dir / "selected_samples.json")
    if args.dry_run_selection:
        print(f"wrote selected Cross V5 bank-swap samples to {output_dir / 'selected_samples.json'}")
        return 0

    bundle = load_cross_v5_swap_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        torch_dtype=resolve_dtype(args.torch_dtype),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        adaln_double_blocks=args.adaln_double_blocks,
    )

    rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for index, selection in enumerate(selections):
        row, panel_path = run_selection(bundle, selection, samples_dir=samples_dir, index=index, args=args)
        rows.append(row)
        panel_paths.append(panel_path)
        print(
            f"[{index + 1}/{len(selections)}] class={selection.class_id} "
            f"{class_name(selection.class_id)} leakage={row['leakage_ratio']:.3f} "
            f"target_delta={row['swap_delta_target_class']:.5f} "
            f"other_max={row['swap_delta_other_class_max']:.5f} panel={panel_path}",
            flush=True,
        )

    summary = summarize_rows(rows)
    payload = {
        "checkpoint": str(Path(args.checkpoint)),
        "metadata": str(Path(args.metadata)),
        "num_samples": len(rows),
        "summary": summary,
        "rows": rows,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf8")
    make_overview(panel_paths, output_dir / "overview.png", thumbnail_width=args.thumbnail_size * 4)
    print(f"wrote Cross V5 bank-swap diagnostics to {output_dir}")
    return 0


def read_cross_metadata(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    records = payload.get("pairs", payload) if isinstance(payload, dict) else payload
    return [dict(record) for record in records]


def write_selection_summary(selections: list[SwapSelection], path: Path) -> None:
    rows = []
    for index, selection in enumerate(selections):
        record = selection.target_record
        swap_record = selection.swap_record
        rows.append(
            {
                "index": index,
                "sample_id": str(record.get("sample_id") or Path(record["target_image"]).stem),
                "reference_a_sample_id": str(record.get("reference_sample_id") or Path(record["reference_image"]).stem),
                "reference_b_sample_id": str(
                    swap_record.get("reference_sample_id") or Path(swap_record["reference_image"]).stem
                ),
                "dataset": record.get("dataset", ""),
                "case_id": record.get("case_id", ""),
                "class_id": selection.class_id,
                "class_name": class_name(selection.class_id),
                "selection_score": selection.score,
                "selection_hed_distance": selection.hed_distance,
                "target_class_fraction": selection.target_fraction,
                "reference_a_class_fraction": selection.ref_a_fraction,
                "reference_b_class_fraction": selection.ref_b_fraction,
                "same_target_swap": selection.same_target,
                "cross_case_swap": selection.cross_case,
                "target_image": record.get("target_image", ""),
                "reference_a_image": record.get("reference_image", ""),
                "reference_b_image": swap_record.get("reference_image", ""),
            }
        )
    path.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf8")


def parse_class_ids(value: str) -> list[int]:
    ids = []
    for chunk in str(value or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        ids.append(int(chunk))
    return sorted({class_id for class_id in ids if 0 <= class_id < NUM_COARSE})


def select_swap_samples(
    records: list[dict],
    *,
    num_samples: int,
    class_ids: list[int],
    min_target_fraction: float,
    min_reference_fraction: float,
    candidate_limit: int,
    prefer_same_target: bool,
    allow_different_target: bool,
    rng: random.Random,
) -> list[SwapSelection]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    records_for_stats = shuffled[: max(1, int(candidate_limit))]
    cache: dict[str, dict[str, Any]] = {}
    ref_pool: list[tuple[dict, dict[str, Any]]] = []
    for record in records_for_stats:
        try:
            ref_pool.append((record, record_reference_stats(record, cache)))
        except Exception:
            continue

    candidates: list[SwapSelection] = []
    by_target: dict[tuple[str, str, str], list[dict]] = {}
    for record in records_for_stats:
        by_target.setdefault(record_target_key(record), []).append(record)

    for record in records_for_stats:
        try:
            target_stats = record_target_stats(record, cache)
            ref_a_stats = record_reference_stats(record, cache)
        except Exception:
            continue
        class_order = list(class_ids)
        rng.shuffle(class_order)
        local_pool = by_target.get(record_target_key(record), []) if prefer_same_target else []
        if not local_pool and allow_different_target:
            local_pool = records_for_stats
        elif allow_different_target:
            local_pool = local_pool + records_for_stats
        for class_id in class_order:
            target_fraction = float(target_stats["fractions"].get(class_id, 0.0))
            ref_a_fraction = float(ref_a_stats["fractions"].get(class_id, 0.0))
            if target_fraction < min_target_fraction or ref_a_fraction < min_reference_fraction:
                continue
            swap = best_swap_for_class(
                record,
                class_id,
                ref_a_stats,
                local_pool=local_pool,
                ref_pool=ref_pool,
                cache=cache,
                min_reference_fraction=min_reference_fraction,
            )
            if swap is None:
                continue
            swap_record, ref_b_stats, hed_distance, same_target = swap
            ref_b_fraction = float(ref_b_stats["fractions"].get(class_id, 0.0))
            cross_case = str(swap_record.get("case_id", "")) != str(record.get("case_id", ""))
            score = hed_distance * math.sqrt(max(target_fraction, 1e-6)) * math.sqrt(max(ref_b_fraction, 1e-6))
            if not same_target:
                score *= 0.85
            if cross_case:
                score *= 1.1
            candidates.append(
                SwapSelection(
                    target_record=record,
                    swap_record=swap_record,
                    class_id=class_id,
                    score=float(score),
                    target_fraction=target_fraction,
                    ref_a_fraction=ref_a_fraction,
                    ref_b_fraction=ref_b_fraction,
                    hed_distance=float(hed_distance),
                    same_target=same_target,
                    cross_case=cross_case,
                )
            )
            break

    candidates.sort(key=lambda item: item.score, reverse=True)
    selected: list[SwapSelection] = []
    used_targets: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = record_target_key(candidate.target_record)
        if key in used_targets:
            continue
        selected.append(candidate)
        used_targets.add(key)
        if len(selected) >= int(num_samples):
            break
    return selected


def best_swap_for_class(
    record: dict,
    class_id: int,
    ref_a_stats: dict[str, Any],
    *,
    local_pool: list[dict],
    ref_pool: list[tuple[dict, dict[str, Any]]],
    cache: dict[str, dict[str, Any]],
    min_reference_fraction: float,
) -> tuple[dict, dict[str, Any], float, bool] | None:
    ref_a_path = str(record.get("reference_image", ""))
    ref_a_proto = ref_a_stats["hed"][class_id]
    best: tuple[dict, dict[str, Any], float, bool] | None = None
    best_distance = -1.0

    pools: list[tuple[bool, list[tuple[dict, dict[str, Any]]]]] = []
    local_items: list[tuple[dict, dict[str, Any]]] = []
    for candidate in local_pool:
        try:
            local_items.append((candidate, record_reference_stats(candidate, cache)))
        except Exception:
            continue
    pools.append((True, local_items))
    pools.append((False, ref_pool))

    for same_target, pool in pools:
        for candidate, stats in pool:
            if str(candidate.get("reference_image", "")) == ref_a_path:
                continue
            if float(stats["fractions"].get(class_id, 0.0)) < min_reference_fraction:
                continue
            distance = float(torch.linalg.vector_norm(ref_a_proto - stats["hed"][class_id]).item())
            if distance > best_distance:
                best = (candidate, stats, distance, same_target and record_target_key(candidate) == record_target_key(record))
                best_distance = distance
        if best is not None and best[3]:
            return best
    return best


def record_target_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("dataset", "")),
        str(record.get("case_id", "")),
        str(record.get("sample_id", "")),
    )


def record_reference_stats(record: Mapping[str, Any], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    image_path = str(record["reference_image"])
    mask_path = str(record["reference_tissue_mask"])
    key = f"ref::{image_path}::{mask_path}"
    if key not in cache:
        cache[key] = image_mask_stats(image_path, mask_path)
    return cache[key]


def record_target_stats(record: Mapping[str, Any], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    image_path = str(record["target_image"])
    mask_path = str(record["target_tissue_mask"])
    key = f"target::{image_path}::{mask_path}"
    if key not in cache:
        cache[key] = image_mask_stats(image_path, mask_path)
    return cache[key]


def image_mask_stats(image_path: str | Path, tissue_mask_path: str | Path) -> dict[str, Any]:
    image = load_image_tensor(image_path).unsqueeze(0)
    tissue = fine_to_coarse(load_tissue_mask(tissue_mask_path).unsqueeze(0))
    hed = build_cross_v5_hed_stat_prototypes(
        reference_image=image,
        reference_class_ids=tissue,
        num_classes=NUM_COARSE,
    )[0].cpu()
    counts = torch.bincount(tissue.reshape(-1).cpu(), minlength=NUM_COARSE).float()
    fractions = counts / counts.sum().clamp_min(1.0)
    return {
        "hed": hed,
        "fractions": {class_id: float(fractions[class_id].item()) for class_id in range(NUM_COARSE)},
    }


def load_cross_v5_swap_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
    torch_dtype: torch.dtype,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    adaln_double_blocks: str | None,
) -> CrossV5SwapBundle:
    from diffusers import FluxControlNetModel, FluxControlNetPipeline

    checkpoint = validate_checkpoint_dir(checkpoint_path)
    device = resolve_device(device)
    if device == "cpu" and torch_dtype in {torch.float16, torch.bfloat16}:
        torch_dtype = torch.float32
    state = _torch_load_weights(checkpoint / "phase5_conditioning.pt")
    control_spec = load_cross_v5_control_spec(state)
    controlnet_config = FluxControlNetModel.load_config(checkpoint)
    controlnet = FluxControlNetModel.from_config(controlnet_config)
    patch_controlnet_x_embedder(controlnet, control_spec.packed_channels)
    controlnet.load_state_dict(_load_diffusers_model_state_dict(checkpoint), strict=True)
    controlnet.to(dtype=torch_dtype)

    pipe = FluxControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    modules = load_cross_v5_condition_modules(
        state=state,
        transformer=pipe.transformer,
        control_spec=control_spec,
        device=device,
        torch_dtype=torch_dtype,
    )
    blocks = adaln_double_blocks or str(
        ((state.get("cross_v5_reference_bank_spec") or {}).get("adaln_double_blocks") or "last")
    )
    double_indices = parse_block_indices(blocks, len(getattr(pipe.transformer, "transformer_blocks", []) or []))
    install_cross_v5_flux_adaln_adapters(
        transformer=pipe.transformer,
        modulator=modules["cross_v5_adaln_modulator"],
        double_block_indices=double_indices,
        single_block_indices=(),
        require_nonzero_gamma=False,
        require_conditioning=True,
    )
    return CrossV5SwapBundle(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        device=device,
        torch_dtype=torch_dtype,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        flux_pipeline=pipe,
        controlnet=controlnet,
        modules=modules,
        control_spec=control_spec,
        adaln_double_blocks=blocks,
    )


def validate_checkpoint_dir(path: str | Path) -> Path:
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(f"Missing ControlNet config.json under checkpoint: {checkpoint}")
    if not (checkpoint / "phase5_conditioning.pt").exists():
        raise FileNotFoundError(f"Missing phase5_conditioning.pt under checkpoint: {checkpoint}")
    return checkpoint


def load_cross_v5_control_spec(state: dict) -> CrossV5GeometryControlSpec:
    saved = state.get("cross_v5_control_spec") or {}
    return CrossV5GeometryControlSpec(geometry_channels=int(saved.get("geometry_channels", 4)))


def load_cross_v5_condition_modules(
    *,
    state: dict,
    transformer: torch.nn.Module,
    control_spec: CrossV5GeometryControlSpec,
    device: str,
    torch_dtype: torch.dtype,
) -> dict[str, torch.nn.Module]:
    spec = state.get("cross_v5_reference_bank_spec") or {}
    builder_state = state.get("cross_v5_ref_bank_builder", {})
    prior_state = state["cross_v5_prior_bank"]
    mod_state = state["cross_v5_adaln_modulator"]
    prototype_dim = int(prior_state["prototypes"].shape[-1])
    hidden_dim = int(mod_state.get("mlp.2.weight", torch.empty(0, 0)).shape[0] // 2) or infer_transformer_hidden_dim(transformer)
    mlp_hidden_dim = int(mod_state.get("mlp.0.weight", torch.empty(0, 0)).shape[0]) or None
    structure_dim = NUM_COARSE + int(control_spec.raw_channels) + 2
    builder = CrossV5RefBankBuilder(
        num_classes=NUM_COARSE,
        local_tokens_per_class=int(spec.get("local_tokens_per_class", 4)),
        prototype_source=str(spec.get("prototype_source", "hed_stats")),
        hed_channels=int(spec.get("hed_channels", 2)),
        include_hed_covariance=bool(spec.get("include_hed_covariance", False)),
        texture_stat_kind=infer_texture_stat_kind(spec, prototype_dim=prototype_dim),
    )
    prior = CrossV5PriorPrototypeBank(num_classes=NUM_COARSE, prototype_dim=prototype_dim)
    modulator = CrossV5SpatialAdaLNModulator(
        hidden_dim=hidden_dim,
        prototype_dim=prototype_dim,
        structure_dim=structure_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        use_internal_norm=False,
    )
    modules = {
        "cross_v5_ref_bank_builder": builder,
        "cross_v5_prior_bank": prior,
        "cross_v5_adaln_modulator": modulator,
    }
    if builder_state:
        builder.load_state_dict(builder_state, strict=False)
    prior.load_state_dict(prior_state, strict=True)
    modulator.load_state_dict(mod_state, strict=True)
    for module in modules.values():
        module.to(device=device, dtype=torch_dtype)
        module.eval()
        module.requires_grad_(False)
    return modules


def infer_texture_stat_kind(spec: Mapping[str, Any], *, prototype_dim: int) -> str:
    saved = str(spec.get("texture_stat_kind", "") or "").strip().lower()
    if saved:
        return saved
    source = str(spec.get("prototype_source", "hed_stats") or "hed_stats").lower()
    if "texture" not in source:
        return "var"
    hed_channels = int(spec.get("hed_channels", 2) or 2)
    hed_dim = 2 * hed_channels + (1 if bool(spec.get("include_hed_covariance", False)) else 0)
    texture_dim = max(0, int(prototype_dim) - hed_dim)
    return "mean_var" if texture_dim >= 6 else "var"


def infer_transformer_hidden_dim(transformer: torch.nn.Module) -> int:
    for block in getattr(transformer, "transformer_blocks", []) or []:
        ff = getattr(block, "ff", None)
        net = getattr(ff, "net", None)
        if net is not None:
            for layer in net:
                if isinstance(layer, torch.nn.Linear):
                    return int(layer.in_features)
        norm = getattr(block, "norm2", None)
        normalized_shape = getattr(norm, "normalized_shape", None)
        if normalized_shape:
            return int(normalized_shape[0])
    config = getattr(transformer, "config", {})
    for key in ("inner_dim", "hidden_size", "joint_attention_dim"):
        if key in config:
            return int(config[key])
    raise ValueError("Could not infer FLUX transformer hidden dim for V5 AdaLN modulator.")


@torch.inference_mode()
def run_selection(
    bundle: CrossV5SwapBundle,
    selection: SwapSelection,
    *,
    samples_dir: Path,
    index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path]:
    record = selection.target_record
    swap_record = selection.swap_record
    sample_id = str(record.get("sample_id") or Path(record["target_image"]).stem)
    ref_a_id = str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)
    ref_b_id = str(swap_record.get("reference_sample_id") or Path(swap_record["reference_image"]).stem)
    class_id = int(selection.class_id)
    sample_dir = samples_dir / f"{index:04d}_{safe_name(sample_id)}__class_{class_id}_{safe_name(class_name(class_id))}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_sample_tensors(record)
    ref_b_image = load_image_tensor(swap_record["reference_image"])
    ref_b_tissue = load_tissue_mask(swap_record["reference_tissue_mask"])
    bank_a = build_bank(bundle, tensors["reference_image"], tensors["reference_tissue_mask"], tensors["target_image"])
    bank_b = build_bank(bundle, ref_b_image, ref_b_tissue, tensors["target_image"])
    bank_single = swap_bank_class(bank_a, bank_b, class_id)

    normal_scale = 1.0
    normal = sample_with_bank(bundle, tensors=tensors, bank=bank_a, seed=args.seed, adaln_scale=normal_scale)
    scale_images = run_adaln_scale_sweep(
        bundle,
        tensors=tensors,
        bank=bank_a,
        seed=args.seed,
        scales=args.adaln_scales,
        normal=normal,
        normal_scale=normal_scale,
    )
    single = sample_with_bank(bundle, tensors=tensors, bank=bank_single, seed=args.seed, adaln_scale=normal_scale)
    full = sample_with_bank(bundle, tensors=tensors, bank=bank_b, seed=args.seed, adaln_scale=normal_scale)

    normal_path = sample_dir / "normal_ref_a.png"
    single_path = sample_dir / f"single_class_swap_{class_id}.png"
    full_path = sample_dir / "full_ref_b.png"
    normal.save(normal_path)
    single.save(single_path)
    full.save(full_path)
    scale_paths: dict[str, str] = {}
    for scale, image in scale_images.items():
        path = sample_dir / f"adaln_scale_{scale_label(scale)}x.png"
        image.save(path)
        scale_paths[scale_label(scale)] = str(path)
    if args.save_individual_images:
        tensor_to_pil(tensors["target_image"]).save(sample_dir / "target.png")
        tensor_to_pil(tensors["reference_image"]).save(sample_dir / "reference_a.png")
        tensor_to_pil(ref_b_image).save(sample_dir / "reference_b.png")
        mask_to_rgb(fine_to_coarse(tensors["target_tissue_mask"].unsqueeze(0))[0]).save(sample_dir / "target_tissue_coarse.png")

    metrics = compute_swap_metrics(
        normal=normal,
        single=single,
        full=full,
        target_tissue=tensors["target_tissue_mask"],
        class_id=class_id,
    )
    scale_metrics = compute_adaln_scale_metrics(
        baseline=normal,
        scale_images=scale_images,
        target_tissue=tensors["target_tissue_mask"],
        class_id=class_id,
    )
    row = {
        "index": index,
        "sample_id": sample_id,
        "reference_a_sample_id": ref_a_id,
        "reference_b_sample_id": ref_b_id,
        "dataset": record.get("dataset", ""),
        "case_id": record.get("case_id", ""),
        "class_id": class_id,
        "class_name": class_name(class_id),
        "selection_score": selection.score,
        "selection_hed_distance": selection.hed_distance,
        "target_class_fraction": selection.target_fraction,
        "reference_a_class_fraction": selection.ref_a_fraction,
        "reference_b_class_fraction": selection.ref_b_fraction,
        "same_target_swap": selection.same_target,
        "cross_case_swap": selection.cross_case,
        "normal_path": str(normal_path),
        "single_class_swap_path": str(single_path),
        "full_swap_path": str(full_path),
        "adaln_scale_paths": scale_paths,
        "adaln_scale_metrics": scale_metrics,
        **metrics,
    }
    (sample_dir / "metrics.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf8")
    panel = make_panel(
        [
            ("target", tensor_to_pil(tensors["target_image"])),
            ("ref A", tensor_to_pil(tensors["reference_image"])),
            ("ref B", tensor_to_pil(ref_b_image)),
            ("target mask", mask_to_rgb(fine_to_coarse(tensors["target_tissue_mask"].unsqueeze(0))[0])),
            ("normal A", normal),
            (f"swap {class_id}", single),
            ("full B", full),
            ("|swap-normal|", diff_heatmap(normal, single)),
            *[(f"AdaLN {scale_label(scale)}x", image) for scale, image in scale_images.items()],
            *[
                (f"|{scale_label(scale)}x-1x|", diff_heatmap(normal, image))
                for scale, image in scale_images.items()
                if abs(float(scale) - normal_scale) > 1e-8
            ],
        ],
        width=args.thumbnail_size,
        title=(
            f"{sample_id} | class {class_id} {class_name(class_id)} | "
            f"leak {row['leakage_ratio']:.2f} | scale {summarize_scale_metrics(scale_metrics)}"
        ),
    )
    panel_path = sample_dir / "panel_single_class_bank_swap.png"
    panel.save(panel_path)
    return row, panel_path


def load_sample_tensors(record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "target_image": load_image_tensor(record["target_image"]),
        "target_tissue_mask": load_tissue_mask(record["target_tissue_mask"]),
        "target_nuclei_mask": load_nuclei_mask(record["target_nuclei_mask"], remap=True),
        "reference_image": load_image_tensor(record["reference_image"]),
        "reference_tissue_mask": load_tissue_mask(record["reference_tissue_mask"]),
    }


def build_bank(
    bundle: CrossV5SwapBundle,
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    target_image: torch.Tensor,
) -> CrossV5TissueBank:
    device = torch.device(bundle.device)
    token_height = int(target_image.shape[1] // 16)
    token_width = int(target_image.shape[2] // 16)
    reference_image_b = reference_image.unsqueeze(0).to(device=device, dtype=torch.float32)
    reference_coarse = fine_to_coarse(reference_tissue_mask.unsqueeze(0)).to(device=device)
    reference_tokens = build_reference_texture_tokens(
        reference_image_b,
        token_height=token_height,
        token_width=token_width,
        dtype=bundle.torch_dtype,
    )
    return bundle.modules["cross_v5_ref_bank_builder"](
        reference_tokens=reference_tokens,
        reference_image=reference_image_b,
        reference_class_ids=reference_coarse,
        token_height=token_height,
        token_width=token_width,
    )


def swap_bank_class(bank_a: CrossV5TissueBank, bank_b: CrossV5TissueBank, class_id: int) -> CrossV5TissueBank:
    prototypes = bank_a.prototypes.clone()
    local_tokens = bank_a.local_tokens.clone()
    class_present = bank_a.class_present.clone()
    class_mass = bank_a.class_mass.clone()
    prototypes[:, class_id] = bank_b.prototypes[:, class_id]
    local_tokens[:, class_id] = bank_b.local_tokens[:, class_id]
    class_present[:, class_id] = bank_b.class_present[:, class_id]
    class_mass[:, class_id] = bank_b.class_mass[:, class_id]
    return CrossV5TissueBank(
        prototypes=prototypes,
        local_tokens=local_tokens,
        class_present=class_present,
        class_mass=class_mass,
        token_class_ids=bank_a.token_class_ids,
        token_class_confidence=bank_a.token_class_confidence,
    )


@torch.inference_mode()
def sample_with_bank(
    bundle: CrossV5SwapBundle,
    *,
    tensors: dict[str, torch.Tensor],
    bank: CrossV5TissueBank,
    seed: int,
    adaln_scale: float = 1.0,
) -> Image.Image:
    from diffusers import FluxControlNetPipeline
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

    pipe = bundle.flux_pipeline
    controlnet = bundle.controlnet
    device = torch.device(bundle.device)
    height = int(tensors["target_image"].shape[1])
    width = int(tensors["target_image"].shape[2])
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[CROSS_V5_PROMPT],
        prompt_2=[CROSS_V5_PROMPT],
        device=device,
    )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator=torch.Generator(device=device).manual_seed(int(seed)),
        latents=None,
    )
    latent_h = int(height // pipe.vae_scale_factor)
    latent_w = int(width // pipe.vae_scale_factor)
    token_height = latent_h // 2
    token_width = latent_w // 2
    control_tensor = build_cross_v5_geometry_control_condition(
        target_tissue_mask=tensors["target_tissue_mask"].unsqueeze(0).to(device=device),
        target_nuclei_mask=tensors["target_nuclei_mask"].unsqueeze(0).to(device=device),
        output_height=latent_h,
        output_width=latent_w,
    ).to(device=device, dtype=bundle.torch_dtype)
    control_image = FluxControlNetPipeline._pack_latents(
        control_tensor,
        1,
        control_tensor.shape[1],
        control_tensor.shape[2],
        control_tensor.shape[3],
    )
    target_coarse = fine_to_coarse(tensors["target_tissue_mask"].unsqueeze(0)).to(device=device)
    target_class_ids = token_class_ids(target_coarse, token_height=token_height, token_width=token_width)
    target_structure_tokens = build_cross_v5_spatial_structure_tokens(
        class_ids=target_coarse,
        num_classes=NUM_COARSE,
        token_height=token_height,
        token_width=token_width,
        geometry_maps=control_tensor.float(),
    ).to(device=device, dtype=bundle.torch_dtype)
    fallback_prototypes = bundle.modules["cross_v5_prior_bank"](1).to(device=device, dtype=bundle.torch_dtype)
    joint_attention_kwargs = {
        CROSS_V5_TARGET_CLASS_IDS_KEY: target_class_ids.to(device=device),
        CROSS_V5_TARGET_STRUCTURE_TOKENS_KEY: target_structure_tokens,
        CROSS_V5_BANK_KEY: move_bank(bank, device=device, dtype=bundle.torch_dtype),
        CROSS_V5_FALLBACK_PROTOTYPES_KEY: fallback_prototypes,
        CROSS_V5_ADALN_SCALE_KEY: float(adaln_scale),
    }
    sigmas = np.linspace(1.0, 1 / bundle.num_inference_steps, bundle.num_inference_steps)
    mu = _calculate_shift(
        image_seq_len=latents.shape[1],
        base_seq_len=pipe.scheduler.config.get("base_image_seq_len", 256),
        max_seq_len=pipe.scheduler.config.get("max_image_seq_len", 4096),
        base_shift=pipe.scheduler.config.get("base_shift", 0.5),
        max_shift=pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, bundle.num_inference_steps, device, sigmas=sigmas, mu=mu)
    controlnet_blocks_repeat = False if getattr(controlnet, "input_hint_block", None) is None else True
    for timestep in timesteps:
        expanded_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
        guidance = None
        if controlnet.config.guidance_embeds:
            guidance = torch.tensor([bundle.guidance_scale], device=device).expand(latents.shape[0])
        controlnet_block_samples, controlnet_single_block_samples = controlnet(
            hidden_states=latents,
            controlnet_cond=control_image,
            controlnet_mode=None,
            conditioning_scale=bundle.controlnet_conditioning_scale,
            timestep=expanded_timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        transformer_guidance = None
        if pipe.transformer.config.guidance_embeds:
            transformer_guidance = torch.tensor([bundle.guidance_scale], device=device).expand(latents.shape[0])
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=expanded_timestep / 1000,
            guidance=transformer_guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )[0]
        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(dtype=bundle.torch_dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def run_adaln_scale_sweep(
    bundle: CrossV5SwapBundle,
    *,
    tensors: dict[str, torch.Tensor],
    bank: CrossV5TissueBank,
    seed: int,
    scales: list[float],
    normal: Image.Image,
    normal_scale: float = 1.0,
) -> dict[float, Image.Image]:
    images: dict[float, Image.Image] = {}
    for scale in scales:
        scale = float(scale)
        if abs(scale - float(normal_scale)) <= 1e-8:
            images[scale] = normal
        else:
            images[scale] = sample_with_bank(
                bundle,
                tensors=tensors,
                bank=bank,
                seed=seed,
                adaln_scale=scale,
            )
    return images


def build_reference_texture_tokens(reference_image: torch.Tensor, *, token_height: int, token_width: int, dtype: torch.dtype) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(reference_image.float(), output_size=(int(token_height), int(token_width)))
    return pooled.permute(0, 2, 3, 1).reshape(reference_image.shape[0], int(token_height) * int(token_width), -1).to(dtype=dtype)


def token_class_ids(class_ids: torch.Tensor, *, token_height: int, token_width: int) -> torch.Tensor:
    one_hot = F.one_hot(class_ids.long(), num_classes=NUM_COARSE).permute(0, 3, 1, 2).float()
    pooled = F.adaptive_avg_pool2d(one_hot, output_size=(int(token_height), int(token_width)))
    return pooled.argmax(dim=1).reshape(class_ids.shape[0], int(token_height) * int(token_width)).long()


def fine_to_coarse(tissue_ids: torch.Tensor) -> torch.Tensor:
    if tissue_ids.ndim == 4 and tissue_ids.shape[1] == 1:
        tissue_ids = tissue_ids[:, 0]
    max_key = max(int(key) for key in FINE_TO_PARENT)
    lookup = torch.as_tensor([int(FINE_TO_PARENT.get(index, 0)) for index in range(max_key + 1)], device=tissue_ids.device)
    return lookup[tissue_ids.long().clamp(0, len(lookup) - 1)]


def move_bank(bank: CrossV5TissueBank, *, device: torch.device, dtype: torch.dtype) -> CrossV5TissueBank:
    return CrossV5TissueBank(
        prototypes=bank.prototypes.to(device=device, dtype=dtype),
        local_tokens=bank.local_tokens.to(device=device, dtype=dtype),
        class_present=bank.class_present.to(device=device),
        class_mass=bank.class_mass.to(device=device, dtype=dtype),
        token_class_ids=bank.token_class_ids.to(device=device),
        token_class_confidence=bank.token_class_confidence.to(device=device, dtype=dtype),
    )


def compute_swap_metrics(*, normal: Image.Image, single: Image.Image, full: Image.Image, target_tissue: torch.Tensor, class_id: int) -> dict[str, Any]:
    normal_arr = pil_to_float(normal)
    single_arr = pil_to_float(single)
    full_arr = pil_to_float(full)
    diff = np.abs(single_arr - normal_arr).mean(axis=2)
    full_diff = np.abs(full_arr - normal_arr).mean(axis=2)
    coarse = fine_to_coarse(target_tissue.unsqueeze(0))[0].cpu().numpy()
    if coarse.shape != diff.shape:
        coarse = np.asarray(Image.fromarray(coarse.astype(np.uint8)).resize((diff.shape[1], diff.shape[0]), Image.Resampling.NEAREST))
    per_class = {}
    for cid in range(NUM_COARSE):
        mask = coarse == cid
        per_class[str(cid)] = float(diff[mask].mean()) if np.any(mask) else 0.0
    target_delta = per_class[str(class_id)]
    other_values = [value for key, value in per_class.items() if int(key) != class_id]
    other_max = max(other_values) if other_values else 0.0
    full_target = float(full_diff[coarse == class_id].mean()) if np.any(coarse == class_id) else 0.0
    return {
        "swap_delta_target_class": float(target_delta),
        "swap_delta_other_class_max": float(other_max),
        "swap_delta_all": float(diff.mean()),
        "full_swap_delta_target_class": float(full_target),
        "leakage_ratio": float(target_delta / max(other_max, 1e-8)),
        "per_class_swap_delta": per_class,
    }


def compute_adaln_scale_metrics(
    *,
    baseline: Image.Image,
    scale_images: Mapping[float, Image.Image],
    target_tissue: torch.Tensor,
    class_id: int,
) -> dict[str, dict[str, Any]]:
    base_arr = pil_to_float(baseline)
    coarse = fine_to_coarse(target_tissue.unsqueeze(0))[0].cpu().numpy()
    if coarse.shape != base_arr.shape[:2]:
        coarse = np.asarray(
            Image.fromarray(coarse.astype(np.uint8)).resize(
                (base_arr.shape[1], base_arr.shape[0]),
                Image.Resampling.NEAREST,
            )
        )
    result: dict[str, dict[str, Any]] = {}
    for scale, image in scale_images.items():
        diff = np.abs(pil_to_float(image) - base_arr).mean(axis=2)
        per_class = {}
        for cid in range(NUM_COARSE):
            mask = coarse == cid
            per_class[str(cid)] = float(diff[mask].mean()) if np.any(mask) else 0.0
        target_delta = per_class[str(class_id)]
        other_values = [value for key, value in per_class.items() if int(key) != int(class_id)]
        other_max = max(other_values) if other_values else 0.0
        result[scale_label(scale)] = {
            "delta_all_vs_1x": float(diff.mean()),
            "delta_target_class_vs_1x": float(target_delta),
            "delta_other_class_max_vs_1x": float(other_max),
            "target_to_other_ratio": float(target_delta / max(other_max, 1e-8)),
            "per_class_delta_vs_1x": per_class,
        }
    return result


def pil_to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB")).astype(np.float32) / 255.0


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB")


def mask_to_rgb(mask: torch.Tensor) -> Image.Image:
    colors = np.array([[30, 30, 30], [180, 60, 60], [60, 150, 60], [60, 60, 180], [140, 60, 180], [180, 180, 80], [60, 140, 100], [170, 170, 170]], dtype=np.uint8)
    arr = mask.detach().cpu().long().clamp(0, NUM_COARSE - 1).numpy()
    return Image.fromarray(colors[arr], mode="RGB")


def diff_heatmap(a: Image.Image, b: Image.Image) -> Image.Image:
    diff = np.abs(pil_to_float(a) - pil_to_float(b)).mean(axis=2)
    if float(diff.max()) > 0:
        diff = diff / float(diff.max())
    heat = np.zeros(diff.shape + (3,), dtype=np.uint8)
    heat[..., 0] = (diff * 255).astype(np.uint8)
    heat[..., 1] = (np.sqrt(diff) * 180).astype(np.uint8)
    heat[..., 2] = ((1.0 - diff) * 40).astype(np.uint8)
    return Image.fromarray(heat, mode="RGB")


def make_panel(items: list[tuple[str, Image.Image]], *, width: int, title: str) -> Image.Image:
    thumbs = []
    label_h = 24
    title_h = 30
    for label, image in items:
        thumb = image.convert("RGB").resize((width, width), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, width + label_h), "white")
        canvas.paste(thumb, (0, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), label, fill=(0, 0, 0))
        thumbs.append(canvas)
    cols = 4
    rows = int(math.ceil(len(thumbs) / cols))
    panel = Image.new("RGB", (cols * width, title_h + rows * (width + label_h)), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title, fill=(0, 0, 0))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * width
        y = title_h + (idx // cols) * (width + label_h)
        panel.paste(thumb, (x, y))
    return panel


def make_overview(panel_paths: list[Path], output_path: Path, *, thumbnail_width: int) -> None:
    if not panel_paths:
        return
    thumbs = []
    for path in panel_paths:
        image = Image.open(path).convert("RGB")
        scale = thumbnail_width / image.width
        thumbs.append(image.resize((thumbnail_width, max(1, int(image.height * scale))), Image.Resampling.LANCZOS))
    cols = 1
    total_h = sum(img.height for img in thumbs)
    overview = Image.new("RGB", (thumbnail_width * cols, total_h), "white")
    y = 0
    for img in thumbs:
        overview.paste(img, (0, y))
        y += img.height
    overview.save(output_path)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = ["leakage_ratio", "swap_delta_target_class", "swap_delta_other_class_max", "swap_delta_all"]
    return {f"{key}_mean": float(np.mean([float(row[key]) for row in rows])) for key in keys}


def summarize_scale_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> str:
    if not metrics:
        return "off"
    chunks = []
    for label in sorted(metrics, key=lambda item: float(item.replace("p", "."))):
        chunks.append(f"{label}x={float(metrics[label].get('delta_all_vs_1x', 0.0)):.4f}")
    return ",".join(chunks)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))[:80]


def scale_label(value: float) -> str:
    return ("%g" % float(value)).replace("-", "m").replace(".", "p")


def parse_float_list(value: str | list[float] | tuple[float, ...] | None) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        values = []
        for chunk in str(value).split(","):
            chunk = chunk.strip()
            if chunk:
                values.append(float(chunk))
    result = []
    for number in values:
        if not math.isfinite(number):
            raise ValueError(f"AdaLN scale must be finite, got {number!r}.")
        if number not in result:
            result.append(float(number))
    return result


def class_name(class_id: int) -> str:
    return str(COARSE_LABELS.get(int(class_id), f"class_{class_id}"))


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def parse_block_indices(value: str, total: int) -> tuple[int, ...]:
    value = str(value or "last").strip().lower()
    if total <= 0:
        return ()
    if value in {"last", "-1"}:
        return (total - 1,)
    if value in {"all", "*"}:
        return tuple(range(total))
    result = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = int(chunk)
        if idx < 0:
            idx = total + idx
        if idx < 0 or idx >= total:
            raise ValueError(f"Block index {chunk!r} resolves to {idx}, outside [0,{total}).")
        if idx not in result:
            result.append(idx)
    return tuple(result)


def _load_diffusers_model_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    safetensors_indexes = sorted(checkpoint_path.glob("diffusion_pytorch_model*.safetensors.index.json"))
    bin_indexes = sorted(checkpoint_path.glob("diffusion_pytorch_model*.bin.index.json"))
    if safetensors_indexes:
        return _load_sharded_diffusers_state_dict(safetensors_indexes[0])
    if bin_indexes:
        return _load_sharded_diffusers_state_dict(bin_indexes[0])
    weight_candidates = [
        *sorted(checkpoint_path.glob("diffusion_pytorch_model*.safetensors")),
        *sorted(checkpoint_path.glob("diffusion_pytorch_model*.bin")),
        checkpoint_path / "pytorch_model.bin",
        checkpoint_path / "model.safetensors",
    ]
    for weight_path in weight_candidates:
        if weight_path.exists():
            return _load_single_diffusers_weight_file(weight_path)
    raise FileNotFoundError(f"No diffusers ControlNet weights found under: {checkpoint_path}")


def _load_sharded_diffusers_state_dict(index_path: Path) -> dict[str, torch.Tensor]:
    payload = json.loads(index_path.read_text(encoding="utf8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid diffusers weight index file: {index_path}")
    state_dict: dict[str, torch.Tensor] = {}
    for filename in sorted(set(weight_map.values())):
        state_dict.update(_load_single_diffusers_weight_file(index_path.parent / filename))
    return state_dict


def _load_single_diffusers_weight_file(weight_path: Path) -> dict[str, torch.Tensor]:
    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(weight_path)
    return _torch_load_weights(weight_path)


def _torch_load_weights(weight_path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location="cpu")


if __name__ == "__main__":
    raise SystemExit(main())
