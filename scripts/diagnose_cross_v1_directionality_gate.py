"""Run a lightweight descriptor-based Cross V1 directionality gate.

This is the cheap generation-side gate:

* generate only the paired reference arm;
* compare the generated tumor-region descriptor to cached real paired and
  alternate reference descriptors;
* reuse the same generated image for same-dataset and different-dataset
  alternate comparisons.

It intentionally does not generate alternate/zero arms. Those belong to the
separate L1/edge-sensitivity instrument.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_cross_v1_generation_gate import (  # noqa: E402
    ALTERNATE_MODES,
    PROMPT_MODES,
    build_manifest_row,
    parse_indices,
    parse_mode_selection,
    parse_scales,
    read_records,
    record_sample_id,
    reference_case_id,
    reference_sample_id,
    resolve_prompt,
    safe_name,
    select_gate_records,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cheap descriptor directionality gate for Cross V1.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Eval-ready Cross V1 checkpoint dir.")
    parser.add_argument("--uni-checkpoint-path", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-manifest", default=None)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--record-indices", default="")
    parser.add_argument("--selection-seed", type=int, default=20260611)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--scales", default="1.0")
    parser.add_argument("--alternate-mode", choices=("same_dataset", "different_dataset", "both"), default="both")
    parser.add_argument("--prompt-mode", choices=("dataset", "empty", "both"), default="both")
    parser.add_argument("--tumor-label", type=int, default=1)
    parser.add_argument("--min-tumor-fraction", type=float, default=0.02)
    parser.add_argument("--min-tumor-tokens", type=int, default=1)
    parser.add_argument(
        "--feature-stage",
        choices=("uni", "projected", "encoder_hid_proj"),
        default="uni",
        help="Descriptor space for generated/ref tumor-region comparison.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--descriptor-dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--prompt-source", choices=("metadata", "dataset"), default="dataset")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--source-latent-init-strength", type=float, default=0.0)
    parser.add_argument("--thumbnail-size", type=int, default=192)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Write the paired/alternate manifest without loading the model.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata_path = Path(args.metadata)
    records = read_records(metadata_path)
    scales = parse_scales(args.scales)
    alternate_modes = parse_mode_selection(args.alternate_mode, ALTERNATE_MODES)
    prompt_modes = parse_mode_selection(args.prompt_mode, PROMPT_MODES)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.selection_manifest:
        selected = select_records_from_manifest(
            Path(args.selection_manifest),
            records,
            alternate_modes=alternate_modes,
        )
    else:
        selected = select_gate_records(
            records,
            record_indices=parse_indices(args.record_indices),
            num_samples=args.num_samples,
            seed=args.selection_seed,
            tumor_label=args.tumor_label,
            min_tumor_fraction=args.min_tumor_fraction,
            alternate_modes=alternate_modes,
        )

    manifest = [
        build_manifest_row(index, paired, alternates)
        for index, paired, alternates in selected
    ]
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    total_generations = len(selected) * len(prompt_modes) * len(scales)
    print(
        f"selected {len(selected)} probes; planned paired generations={total_generations} "
        f"({len(selected)} probes x {len(prompt_modes)} prompt x {len(scales)} scale)",
        flush=True,
    )
    if args.selection_only:
        return 0

    from controlnet_train.data.common import load_image_tensor, load_nuclei_mask, load_tissue_mask
    from controlnet_train.inference.pipeline_cross_v1 import (
        load_cross_v1_bundle,
        run_cross_v1_bundle,
        set_ip_adapter_scale,
    )
    from controlnet_train.modules.reference_image_encoder import resize_mask_to_token_labels

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    bundle = load_cross_v1_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        uni_checkpoint_path=args.uni_checkpoint_path,
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        ip_adapter_scale=scales[0],
    )
    descriptor_dtype = dtype_by_name[args.descriptor_dtype]

    descriptor_cache: dict[tuple[str, str], torch.Tensor] = {}
    metric_rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    generation_index = 0

    for probe_index, (metadata_index, paired, alternates) in enumerate(selected):
        sample_id = record_sample_id(paired)
        sample_dir = output_dir / f"{probe_index:03d}_{safe_name(sample_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        target_pil = Image.open(paired["target_image"]).convert("RGB")
        paired_ref_pil = Image.open(paired["reference_image"]).convert("RGB")
        paired_ref_tensor = load_image_tensor(paired["reference_image"])
        target_tissue = load_tissue_mask(paired["target_tissue_mask"])
        target_nuclei = load_nuclei_mask(paired["target_nuclei_mask"])
        paired_ref_tissue = load_tissue_mask(paired["reference_tissue_mask"])
        paired_ref_nuclei = load_nuclei_mask(paired["reference_nuclei_mask"])

        paired_ref_embedding = cached_descriptor(
            descriptor_cache,
            key=("ref", str(paired["reference_image"])),
            image=paired_ref_tensor,
            tissue_mask=paired_ref_tissue,
            bundle=bundle,
            resize_mask_to_token_labels=resize_mask_to_token_labels,
            feature_stage=args.feature_stage,
            tumor_label=args.tumor_label,
            min_tumor_tokens=args.min_tumor_tokens,
            descriptor_dtype=descriptor_dtype,
        )
        target_embedding = cached_descriptor(
            descriptor_cache,
            key=("target", str(paired["target_image"])),
            image=load_image_tensor(paired["target_image"]),
            tissue_mask=target_tissue,
            bundle=bundle,
            resize_mask_to_token_labels=resize_mask_to_token_labels,
            feature_stage=args.feature_stage,
            tumor_label=args.tumor_label,
            min_tumor_tokens=args.min_tumor_tokens,
            descriptor_dtype=descriptor_dtype,
        )
        alternate_embeddings: dict[str, torch.Tensor] = {}
        alternate_pils: dict[str, Image.Image] = {}
        for alternate_mode, alternate in alternates.items():
            alternate_ref_tensor = load_image_tensor(alternate["reference_image"])
            alternate_ref_tissue = load_tissue_mask(alternate["reference_tissue_mask"])
            alternate_embeddings[alternate_mode] = cached_descriptor(
                descriptor_cache,
                key=("ref", str(alternate["reference_image"])),
                image=alternate_ref_tensor,
                tissue_mask=alternate_ref_tissue,
                bundle=bundle,
                resize_mask_to_token_labels=resize_mask_to_token_labels,
                feature_stage=args.feature_stage,
                tumor_label=args.tumor_label,
                min_tumor_tokens=args.min_tumor_tokens,
                descriptor_dtype=descriptor_dtype,
            )
            alternate_pils[alternate_mode] = Image.open(alternate["reference_image"]).convert("RGB")

        target_pil.save(sample_dir / "target.png")
        paired_ref_pil.save(sample_dir / "reference_paired.png")
        for mode, image in alternate_pils.items():
            image.save(sample_dir / f"reference_{mode}.png")

        for prompt_mode in prompt_modes:
            prompt = "" if prompt_mode == "empty" else resolve_prompt(args, paired)
            prompt_dir = sample_dir / f"prompt_{prompt_mode}"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            predictions: dict[float, Image.Image] = {}
            for scale in scales:
                generation_index += 1
                set_ip_adapter_scale(bundle.flux_pipeline.transformer, scale)
                print(
                    f"[{generation_index}/{total_generations}] {sample_id} "
                    f"prompt={prompt_mode} scale={scale:g} variant=paired",
                    flush=True,
                )
                prediction = run_cross_v1_bundle(
                    bundle,
                    reference_image=paired_ref_tensor,
                    reference_tissue_mask=paired_ref_tissue,
                    reference_nuclei_mask=paired_ref_nuclei,
                    target_tissue_mask=target_tissue,
                    target_nuclei_mask=target_nuclei,
                    prompt=prompt,
                    source_latent_init_strength=args.source_latent_init_strength,
                    seed=args.generation_seed,
                )
                predictions[scale] = prediction
                prediction.save(prompt_dir / f"prediction_scale_{scale:g}_paired.png")
                generated_embedding = descriptor_from_pil(
                    prediction,
                    tissue_mask=target_tissue,
                    bundle=bundle,
                    resize_mask_to_token_labels=resize_mask_to_token_labels,
                    feature_stage=args.feature_stage,
                    tumor_label=args.tumor_label,
                    min_tumor_tokens=args.min_tumor_tokens,
                    descriptor_dtype=descriptor_dtype,
                )
                target_cosine = cosine(generated_embedding, target_embedding)
                paired_cosine = cosine(generated_embedding, paired_ref_embedding)
                for alternate_mode, alternate in sorted(alternates.items()):
                    alternate_cosine = cosine(generated_embedding, alternate_embeddings[alternate_mode])
                    margin = paired_cosine - alternate_cosine
                    metric_rows.append(
                        {
                            "metadata_index": metadata_index,
                            "sample_id": sample_id,
                            "dataset": paired.get("dataset", ""),
                            "paired_reference_sample_id": reference_sample_id(paired),
                            "alternate_reference_sample_id": reference_sample_id(alternate),
                            "paired_reference_case_id": reference_case_id(paired),
                            "alternate_reference_case_id": reference_case_id(alternate),
                            "alternate_mode": alternate_mode,
                            "prompt_mode": prompt_mode,
                            "scale": scale,
                            "variant": "paired_descriptor",
                            "feature_stage": args.feature_stage,
                            "target_cosine": target_cosine,
                            "paired_ref_cosine": paired_cosine,
                            "alternate_ref_cosine": alternate_cosine,
                            "cosine_margin": margin,
                            "paired_advantage": margin,
                            "paired_win": margin > 0.0,
                            "generation_seed": args.generation_seed,
                        }
                    )
            panel = make_directionality_panel(
                sample_id=sample_id,
                target=target_pil,
                paired_reference=paired_ref_pil,
                alternates=alternate_pils,
                predictions=predictions,
                scales=scales,
                thumbnail_size=args.thumbnail_size,
            )
            panel_path = prompt_dir / "directionality_gate.png"
            panel.save(panel_path)
            panel_paths.append(panel_path)

    write_csv(output_dir / "directionality_metrics.csv", metric_rows)
    summary = summarize_directionality_metrics(metric_rows)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "generation_seed": int(args.generation_seed),
            "selection_seed": int(args.selection_seed),
            "scales": scales,
            "alternate_modes": alternate_modes,
            "prompt_modes": prompt_modes,
            "num_samples": len(selected),
            "planned_generations": total_generations,
            "actual_generations": generation_index,
            "feature_stage": args.feature_stage,
            "semantics": {
                "generated_arm": "paired image + paired labels only",
                "win": "cos(generated_tumor, paired_ref_tumor) > cos(generated_tumor, alternate_ref_tumor)",
                "alternate_modes": "same/different alternates are post-processing comparisons sharing the same generated image",
            },
        }
    )
    (output_dir / "directionality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths and len(panel_paths) <= 48:
        make_overview(panel_paths).save(output_dir / "directionality_overview.png")
    elif panel_paths:
        print(f"Skipping overview panel because {len(panel_paths)} panels were generated.")
    print(f"wrote lightweight directionality gate to {output_dir}")
    return 0


def select_records_from_manifest(
    manifest_path: Path,
    records: list[dict[str, Any]],
    *,
    alternate_modes: list[str],
) -> list[tuple[int, dict[str, Any], dict[str, dict[str, Any]]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    by_target_ref = {
        (path_key(row.get("target_image")), path_key(row.get("reference_image"))): (index, row)
        for index, row in enumerate(records)
        if row.get("target_image") and row.get("reference_image")
    }
    by_ref: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("reference_image"):
            by_ref[path_key(row.get("reference_image"))].append(row)
    selected = []
    for item in manifest:
        target_key = path_key(item.get("target_image"))
        paired_ref_key = path_key(item.get("paired_reference_image"))
        found = by_target_ref.get((target_key, paired_ref_key))
        if found is None:
            raise ValueError(
                f"manifest probe not found in metadata: target={item.get('target_image')} "
                f"paired_ref={item.get('paired_reference_image')}"
            )
        metadata_index, paired = found
        alternates: dict[str, dict[str, Any]] = {}
        for mode in alternate_modes:
            payload = (item.get("alternates") or {}).get(mode)
            if not payload:
                continue
            candidates = by_ref.get(path_key(payload.get("reference_image"))) or []
            if not candidates:
                raise ValueError(f"manifest alternate not found in metadata: {payload}")
            alternates[mode] = candidates[0]
        selected.append((int(item.get("metadata_index", metadata_index)), paired, alternates))
    return selected


def path_key(value: Any) -> str:
    if value is None:
        return ""
    return str(Path(str(value).replace("\\", "/")).expanduser())


def cached_descriptor(
    cache: dict[tuple[str, str], torch.Tensor],
    *,
    key: tuple[str, str],
    image: torch.Tensor,
    tissue_mask: torch.Tensor,
    bundle,
    resize_mask_to_token_labels,
    feature_stage: str,
    tumor_label: int,
    min_tumor_tokens: int,
    descriptor_dtype: torch.dtype,
) -> torch.Tensor:
    if key not in cache:
        cache[key] = descriptor_from_tensor(
            image,
            tissue_mask=tissue_mask,
            bundle=bundle,
            resize_mask_to_token_labels=resize_mask_to_token_labels,
            feature_stage=feature_stage,
            tumor_label=tumor_label,
            min_tumor_tokens=min_tumor_tokens,
            descriptor_dtype=descriptor_dtype,
        )
    return cache[key]


def descriptor_from_pil(
    image: Image.Image,
    *,
    tissue_mask: torch.Tensor,
    bundle,
    resize_mask_to_token_labels,
    feature_stage: str,
    tumor_label: int,
    min_tumor_tokens: int,
    descriptor_dtype: torch.dtype,
) -> torch.Tensor:
    return descriptor_from_tensor(
        pil_to_tensor(image),
        tissue_mask=tissue_mask,
        bundle=bundle,
        resize_mask_to_token_labels=resize_mask_to_token_labels,
        feature_stage=feature_stage,
        tumor_label=tumor_label,
        min_tumor_tokens=min_tumor_tokens,
        descriptor_dtype=descriptor_dtype,
    )


def descriptor_from_tensor(
    image: torch.Tensor,
    *,
    tissue_mask: torch.Tensor,
    bundle,
    resize_mask_to_token_labels,
    feature_stage: str,
    tumor_label: int,
    min_tumor_tokens: int,
    descriptor_dtype: torch.dtype,
) -> torch.Tensor:
    ref_encoder = bundle.ref_encoder
    image_batch = image.unsqueeze(0).to(device=bundle.device, dtype=descriptor_dtype)
    mask_batch = tissue_mask.unsqueeze(0)
    with torch.no_grad():
        if feature_stage == "uni":
            tokens = ref_encoder.extract_uni_features(image_batch).float().cpu()
        elif feature_stage == "projected":
            tokens = ref_encoder.encode_projected_patch_tokens(image_batch).float().cpu()
        elif feature_stage == "encoder_hid_proj":
            projected = ref_encoder.encode_projected_patch_tokens(image_batch)
            gate = ref_encoder.reference_presence_gate(
                image_batch,
                device=projected.device,
                dtype=projected.dtype,
            )
            projected = projected * gate
            encoder_hid_proj = bundle.flux_pipeline.transformer.encoder_hid_proj
            tokens = encoder_hid_proj([projected])[0]
            tokens = (tokens * gate.to(device=tokens.device, dtype=tokens.dtype)).float().cpu()
        else:
            raise ValueError(f"unknown feature_stage: {feature_stage}")
    labels = resize_mask_to_token_labels(mask_batch, int(tokens.shape[1]))
    tumor = labels[0] == int(tumor_label)
    tumor_count = int(tumor.sum().item())
    if tumor_count < int(min_tumor_tokens):
        raise ValueError(f"insufficient tumor tokens: {tumor_count} < {min_tumor_tokens}")
    pooled = tokens[0, tumor].mean(dim=0)
    return F.normalize(pooled.float(), dim=0).cpu()


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left.float(), right.float()).item())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_directionality_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_combo = {}
    for alternate_mode, prompt_mode in sorted(
        {
            (str(row["alternate_mode"]), str(row["prompt_mode"]))
            for row in rows
        }
    ):
        combo_rows = [
            row for row in rows
            if str(row["alternate_mode"]) == alternate_mode
            and str(row["prompt_mode"]) == prompt_mode
        ]
        by_combo[f"{alternate_mode}/{prompt_mode}"] = summarize_rows(combo_rows)
    return {
        "overall": summarize_rows(rows),
        "by_combo": by_combo,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(row["cosine_margin"]) for row in rows if math.isfinite(float(row["cosine_margin"]))]
    wins = [1.0 if str(row["paired_win"]).lower() == "true" or row["paired_win"] is True else 0.0 for row in rows]
    return {
        "n": len(rows),
        "win_rate": finite_mean(wins),
        "mean_cosine_margin": finite_mean(margins),
        "mean_paired_ref_cosine": finite_mean([float(row["paired_ref_cosine"]) for row in rows]),
        "mean_alternate_ref_cosine": finite_mean([float(row["alternate_ref_cosine"]) for row in rows]),
        "mean_target_cosine": finite_mean([float(row["target_cosine"]) for row in rows]),
    }


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def make_directionality_panel(
    *,
    sample_id: str,
    target: Image.Image,
    paired_reference: Image.Image,
    alternates: dict[str, Image.Image],
    predictions: dict[float, Image.Image],
    scales: list[float],
    thumbnail_size: int,
) -> Image.Image:
    alt_items = sorted(alternates.items())[:2]
    columns = 3 + len(alt_items)
    rows = 1 + len(scales)
    label_height = 28
    canvas = Image.new("RGB", (columns * thumbnail_size, rows * (thumbnail_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    header = [("Target", target), ("Paired ref", paired_reference)]
    header.extend([(f"Alt {mode}", image) for mode, image in alt_items])
    header.append((safe_name(sample_id)[:24], target))
    for column, (label, image) in enumerate(header):
        paste_cell(canvas, draw, font, image, label, column, 0, thumbnail_size, label_height)
    for row_index, scale in enumerate(scales, start=1):
        cells = [(f"scale {scale:g}", target), ("paired gen", predictions[scale])]
        cells.extend([(f"Alt {mode}", image) for mode, image in alt_items])
        cells.append(("paired gen", predictions[scale]))
        for column, (label, image) in enumerate(cells):
            paste_cell(canvas, draw, font, image, label, column, row_index, thumbnail_size, label_height)
    return canvas


def paste_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    image: Image.Image,
    label: str,
    column: int,
    row: int,
    size: int,
    label_height: int,
) -> None:
    x = column * size
    y = row * (size + label_height)
    thumb = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    canvas.paste(thumb, (x, y))
    draw.text((x + 4, y + size + 7), label, fill="black", font=font)


def make_overview(paths: list[Path]) -> Image.Image:
    panels = [Image.open(path).convert("RGB") for path in paths]
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
