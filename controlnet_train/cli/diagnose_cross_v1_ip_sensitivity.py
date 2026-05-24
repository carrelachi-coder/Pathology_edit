"""Diagnose Cross V1 IP-Adapter sensitivity without retraining.

The script keeps the trained ControlNet/checkpoint fixed and runs a small grid:
- IP-Adapter scale values, typically 0, 0.5, 1, 2, 4
- reference image variants: normal, zero, random

By default only the reference image is replaced for zero/random variants; the
reference masks remain fixed so the measurement mostly reflects the IP image
branch instead of the ControlNet spatial branch.
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
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A0 diagnostic: scan Cross V1 IP-Adapter scale/reference sensitivity."
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V1 checkpoint dir.")
    parser.add_argument("--uni-checkpoint-path", required=True, help="UNI2-h pytorch_model.bin path.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--prompt-source", choices=["metadata", "dataset"], default="dataset")
    parser.add_argument("--prompt", default=None, help="Override every sample with one prompt.")
    parser.add_argument(
        "--scales",
        default="0,0.5,1,2,4",
        help="Comma-separated IP-Adapter scale values.",
    )
    parser.add_argument(
        "--reference-variants",
        default="normal,zero,random",
        help="Comma-separated variants from: normal, zero, random.",
    )
    parser.add_argument(
        "--replace-reference-masks",
        action="store_true",
        help=(
            "For zero/random variants, replace reference masks too. Leave off to isolate "
            "the IP image branch while keeping ControlNet reference masks fixed."
        ),
    )
    parser.add_argument("--thumbnail-size", type=int, default=160)
    parser.add_argument("--overview-max-samples", type=int, default=12)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def parse_scale_values(value: str) -> list[float]:
    scales: list[float] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        scale = float(part)
        if not math.isfinite(scale):
            raise ValueError(f"Scale must be finite, got {part!r}.")
        scales.append(scale)
    if not scales:
        raise ValueError("At least one scale value is required.")
    return scales


def parse_reference_variants(value: str) -> list[str]:
    allowed = {"normal", "zero", "random"}
    variants: list[str] = []
    for raw_part in value.split(","):
        variant = raw_part.strip().lower()
        if not variant:
            continue
        if variant not in allowed:
            raise ValueError(f"Unsupported reference variant {variant!r}; choose from {sorted(allowed)}.")
        if variant not in variants:
            variants.append(variant)
    if "normal" not in variants:
        variants.insert(0, "normal")
    return variants


def normalize_cross_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("pairs")
        if not isinstance(records, list):
            raise ValueError("cross metadata dict must contain a 'pairs' list")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported cross metadata payload type: {type(payload)!r}")


def select_eval_records(
    records: list[dict[str, Any]],
    *,
    num_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if num_samples is None or num_samples <= 0 or num_samples >= len(records):
        return list(records)
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return selected[:num_samples]


def read_cross_metadata(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    return normalize_cross_records(payload)


def main(argv=None) -> int:
    args = parse_args(argv)
    scales = parse_scale_values(args.scales)
    variants = parse_reference_variants(args.reference_variants)
    all_records = read_cross_metadata(args.metadata)
    records = select_eval_records(all_records, num_samples=args.num_samples, seed=args.seed)

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from controlnet_train.inference.pipeline_cross_v1 import load_cross_v1_bundle

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
    )

    weight_norms = collect_ip_weight_norms(bundle.flux_pipeline.transformer)
    (output_dir / "ip_weight_norms.json").write_text(
        json.dumps(weight_norms, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )

    rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    rng = random.Random(args.seed)

    for index, record in enumerate(records):
        sample_rows, panel_path = run_sample_grid(
            bundle=bundle,
            record=record,
            all_records=all_records,
            sample_index=index,
            output_root=samples_dir,
            scales=scales,
            variants=variants,
            rng=rng,
            prompt_override=args.prompt,
            prompt_source=args.prompt_source,
            replace_reference_masks=args.replace_reference_masks,
            thumbnail_size=args.thumbnail_size,
        )
        rows.extend(sample_rows)
        if panel_path is not None and len(panel_paths) < args.overview_max_samples:
            panel_paths.append(panel_path)
        print(
            f"[{index + 1}/{len(records)}] {sample_rows[0]['sample_id']} "
            f"wrote {len(sample_rows)} outputs"
        )

    write_rows(output_dir, rows)
    summary = aggregate_diagnostic_rows(rows)
    summary["ip_weight_norms"] = weight_norms["summary"]
    summary["scales"] = scales
    summary["reference_variants"] = variants
    summary["replace_reference_masks"] = bool(args.replace_reference_masks)
    summary["cross_v1_spatial_mode"] = getattr(bundle.control_spec, "spatial_mode", "reference_target")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths:
        make_overview(panel_paths).save(output_dir / "overview_grid.png")
    print(f"wrote A0 diagnostic outputs to {output_dir}")
    return 0


def run_sample_grid(
    *,
    bundle,
    record: dict[str, Any],
    all_records: list[dict[str, Any]],
    sample_index: int,
    output_root: Path,
    scales: list[float],
    variants: list[str],
    rng: random.Random,
    prompt_override: str | None,
    prompt_source: str,
    replace_reference_masks: bool,
    thumbnail_size: int,
) -> tuple[list[dict[str, Any]], Path | None]:
    sample_id = str(record.get("sample_id") or Path(record["target_image"]).stem)
    ref_id = str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)
    sample_dir = output_root / f"{sample_index:04d}_{safe_name(sample_id)}__ref_{safe_name(ref_id)}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from controlnet_train.data.common import (
        default_prompt_for_dataset,
        load_image_tensor,
        load_nuclei_mask,
        load_tissue_mask,
    )
    from controlnet_train.inference.pipeline_cross_v1 import run_cross_v1_bundle

    reference_image_path = Path(record["reference_image"])
    target_image_path = Path(record["target_image"])
    reference_image = load_image_tensor(reference_image_path)
    reference_tissue_mask = load_tissue_mask(record["reference_tissue_mask"])
    reference_nuclei_mask = load_nuclei_mask(record["reference_nuclei_mask"])
    target_tissue_mask = load_tissue_mask(record["target_tissue_mask"])
    target_nuclei_mask = load_nuclei_mask(record["target_nuclei_mask"])
    target_pil = Image.open(target_image_path).convert("RGB")
    reference_pil = Image.open(reference_image_path).convert("RGB")
    base_hw = tuple(int(v) for v in reference_image.shape[1:])

    reference_pil.save(sample_dir / "reference_original.png")
    target_pil.save(sample_dir / "target.png")

    random_record = choose_random_reference_record(all_records, record, rng)
    variant_inputs = build_reference_variants(
        variants=variants,
        base_hw=base_hw,
        reference_image=reference_image,
        reference_tissue_mask=reference_tissue_mask,
        reference_nuclei_mask=reference_nuclei_mask,
        random_record=random_record,
        replace_reference_masks=replace_reference_masks,
    )

    prompt = resolve_eval_prompt(
        record=record,
        prompt_override=prompt_override,
        prompt_source=prompt_source,
        default_prompt_for_dataset=default_prompt_for_dataset,
    )

    outputs: dict[tuple[str, float], Image.Image] = {}
    output_arrays: dict[tuple[str, float], np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for variant_name in variants:
            variant = variant_inputs[variant_name]
            for scale in scales:
                set_ip_adapter_scale(bundle.flux_pipeline.transformer, scale)
                prediction = run_cross_v1_bundle(
                    bundle,
                    reference_image=variant["image"],
                    reference_tissue_mask=variant["tissue_mask"],
                    reference_nuclei_mask=variant["nuclei_mask"],
                    target_tissue_mask=target_tissue_mask,
                    target_nuclei_mask=target_nuclei_mask,
                    prompt=prompt,
                )
                key = (variant_name, scale)
                outputs[key] = prediction
                output_arrays[key] = pil_to_chw_float(prediction)
                prediction_path = sample_dir / f"{variant_name}_scale_{format_scale(scale)}.png"
                prediction.save(prediction_path)

                target_array = pil_to_chw_float(target_pil)
                rows.append(
                    {
                        "sample_index": sample_index,
                        "sample_id": sample_id,
                        "reference_sample_id": ref_id,
                        "variant": variant_name,
                        "variant_reference_sample_id": variant.get("reference_sample_id", ref_id),
                        "scale": float(scale),
                        "path": str(prediction_path),
                        "l1_to_target": image_l1(output_arrays[key], target_array),
                        "mse_to_target": image_mse(output_arrays[key], target_array),
                    }
                )

    normal_by_scale = {
        scale: output_arrays[("normal", scale)]
        for scale in scales
        if ("normal", scale) in output_arrays
    }
    normal_scale0 = normal_by_scale.get(0.0)
    first_normal = normal_by_scale.get(scales[0])
    for row in rows:
        key = (row["variant"], float(row["scale"]))
        current = output_arrays[key]
        normal_same_scale = normal_by_scale.get(float(row["scale"]))
        if normal_same_scale is not None:
            row["l1_vs_normal_same_scale"] = image_l1(current, normal_same_scale)
            row["mse_vs_normal_same_scale"] = image_mse(current, normal_same_scale)
        if normal_scale0 is not None:
            row["l1_vs_normal_scale0"] = image_l1(current, normal_scale0)
            row["mse_vs_normal_scale0"] = image_mse(current, normal_scale0)
        elif first_normal is not None:
            row["l1_vs_first_normal"] = image_l1(current, first_normal)
            row["mse_vs_first_normal"] = image_mse(current, first_normal)

    (sample_dir / "diagnostics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    panel = make_sample_panel(
        reference=reference_pil,
        target=target_pil,
        outputs=outputs,
        variants=variants,
        scales=scales,
        thumbnail_size=thumbnail_size,
        title=f"{sample_id} | ref={ref_id}",
    )
    panel_path = sample_dir / "sensitivity_grid.png"
    panel.save(panel_path)
    return rows, panel_path


def build_reference_variants(
    *,
    variants: list[str],
    base_hw: tuple[int, int],
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    random_record: dict[str, Any],
    replace_reference_masks: bool,
) -> dict[str, dict[str, Any]]:
    import torch

    from controlnet_train.data.common import load_image_tensor, load_nuclei_mask, load_tissue_mask

    result: dict[str, dict[str, Any]] = {}
    for variant in variants:
        if variant == "normal":
            result[variant] = {
                "image": reference_image,
                "tissue_mask": reference_tissue_mask,
                "nuclei_mask": reference_nuclei_mask,
                "reference_sample_id": None,
            }
        elif variant == "zero":
            result[variant] = {
                "image": torch.zeros_like(reference_image),
                "tissue_mask": (
                    torch.zeros_like(reference_tissue_mask)
                    if replace_reference_masks
                    else reference_tissue_mask
                ),
                "nuclei_mask": (
                    torch.zeros_like(reference_nuclei_mask)
                    if replace_reference_masks
                    else reference_nuclei_mask
                ),
                "reference_sample_id": "zero",
            }
        elif variant == "random":
            random_image = resize_image_tensor(load_image_tensor(random_record["reference_image"]), base_hw)
            if replace_reference_masks:
                random_tissue = resize_mask_tensor(load_tissue_mask(random_record["reference_tissue_mask"]), base_hw)
                random_nuclei = resize_mask_tensor(load_nuclei_mask(random_record["reference_nuclei_mask"]), base_hw)
            else:
                random_tissue = reference_tissue_mask
                random_nuclei = reference_nuclei_mask
            result[variant] = {
                "image": random_image,
                "tissue_mask": random_tissue,
                "nuclei_mask": random_nuclei,
                "reference_sample_id": str(
                    random_record.get("reference_sample_id")
                    or Path(random_record["reference_image"]).stem
                ),
            }
        else:
            raise ValueError(f"Unsupported variant: {variant}")
    return result


def choose_random_reference_record(
    records: list[dict[str, Any]],
    current: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    if not records:
        return current
    current_ref = str(current.get("reference_image", ""))
    candidates = [record for record in records if str(record.get("reference_image", "")) != current_ref]
    if not candidates:
        candidates = records
    return rng.choice(candidates)


def resize_image_tensor(image: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    import torch.nn.functional as F

    if tuple(image.shape[1:]) == hw:
        return image
    resized = F.interpolate(
        image.unsqueeze(0),
        size=hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized.clamp(0.0, 1.0).contiguous()


def resize_mask_tensor(mask: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    if tuple(mask.shape) == hw:
        return mask
    resized = F.interpolate(
        mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        size=hw,
        mode="nearest",
    ).squeeze(0).squeeze(0)
    return resized.to(dtype=torch.long).contiguous()


def set_ip_adapter_scale(transformer, scale: float) -> None:
    import torch

    for block in getattr(transformer, "transformer_blocks", []):
        processor = getattr(getattr(block, "attn", None), "processor", None)
        if processor is None or not hasattr(processor, "scale"):
            continue
        current = processor.scale
        if isinstance(current, list):
            processor.scale = [float(scale) for _ in current] or [float(scale)]
        elif isinstance(current, tuple):
            processor.scale = tuple(float(scale) for _ in current) or (float(scale),)
        elif torch.is_tensor(current):
            current.fill_(float(scale))
        else:
            processor.scale = float(scale)


def collect_ip_weight_norms(transformer) -> dict[str, Any]:
    modules: dict[str, torch.nn.Module] = {}
    if hasattr(transformer, "encoder_hid_proj"):
        modules["encoder_hid_proj"] = transformer.encoder_hid_proj
    for index, block in enumerate(getattr(transformer, "transformer_blocks", [])):
        processor = getattr(getattr(block, "attn", None), "processor", None)
        for name in ("to_k_ip", "to_v_ip"):
            module = getattr(processor, name, None)
            if module is not None:
                modules[f"block_{index}_{name}"] = module

    by_module = {
        name: module_weight_stats(module)
        for name, module in modules.items()
    }
    trainable_modules = [stats for stats in by_module.values() if stats["num_parameters"] > 0]
    return {
        "summary": {
            "num_modules": len(by_module),
            "total_parameters": int(sum(stats["num_parameters"] for stats in by_module.values())),
            "total_l2_norm": float(
                math.sqrt(sum(stats["l2_norm"] ** 2 for stats in trainable_modules))
            ),
            "max_abs": float(max((stats["max_abs"] for stats in trainable_modules), default=0.0)),
        },
        "by_module": by_module,
    }


def module_weight_stats(module: torch.nn.Module) -> dict[str, float | int]:
    import torch

    num_parameters = 0
    squared_norm = 0.0
    max_abs = 0.0
    zero_count = 0
    for parameter in module.parameters():
        tensor = parameter.detach().float().cpu()
        num_parameters += tensor.numel()
        squared_norm += float(torch.sum(tensor * tensor).item())
        max_abs = max(max_abs, float(tensor.abs().max().item()) if tensor.numel() else 0.0)
        zero_count += int(torch.count_nonzero(tensor == 0).item())
    return {
        "num_parameters": int(num_parameters),
        "l2_norm": float(math.sqrt(squared_norm)),
        "max_abs": float(max_abs),
        "zero_fraction": float(zero_count / num_parameters) if num_parameters else math.nan,
    }


def resolve_eval_prompt(
    *,
    record: dict[str, Any],
    prompt_override: str | None,
    prompt_source: str,
    default_prompt_for_dataset,
) -> str:
    if prompt_override:
        return prompt_override
    if prompt_source == "metadata":
        prompt = record.get("prompt")
        if prompt:
            return str(prompt)
    if prompt_source == "dataset":
        dataset = record.get("dataset")
        if dataset:
            return default_prompt_for_dataset(str(dataset))
    return str(record.get("prompt") or "H&E stained cancer histopathology at 40x magnification")


def pil_to_chw_float(image: Image.Image) -> np.ndarray:
    return np.transpose(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0, (2, 0, 1))


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120]


def thumbnail(image: Image.Image, size: int) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - thumb.width) // 2
    y = (size - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def image_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).mean())


def image_mse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.square(left - right).mean())


def aggregate_diagnostic_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), float(row["scale"]))].append(row)

    by_variant_scale: dict[str, dict[str, float]] = {}
    metric_keys = [
        "l1_to_target",
        "mse_to_target",
        "l1_vs_normal_same_scale",
        "mse_vs_normal_same_scale",
        "l1_vs_normal_scale0",
        "mse_vs_normal_scale0",
    ]
    for (variant, scale), group_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        key = f"{variant}@{format_scale(scale)}"
        by_variant_scale[key] = {"num_samples": float(len(group_rows))}
        for metric_key in metric_keys:
            values = [
                float(row[metric_key])
                for row in group_rows
                if metric_key in row and math.isfinite(float(row[metric_key]))
            ]
            if values:
                by_variant_scale[key][f"{metric_key}_mean"] = float(np.mean(values))
                by_variant_scale[key][f"{metric_key}_std"] = float(np.std(values))

    return {
        "num_outputs": len(rows),
        "num_samples": len({row["sample_id"] for row in rows}),
        "by_variant_scale": by_variant_scale,
    }


def write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    jsonl_path = output_dir / "diagnostics.jsonl"
    with jsonl_path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")

    if not rows:
        return
    csv_path = output_dir / "diagnostics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_sample_panel(
    *,
    reference: Image.Image,
    target: Image.Image,
    outputs: dict[tuple[str, float], Image.Image],
    variants: list[str],
    scales: list[float],
    thumbnail_size: int,
    title: str,
) -> Image.Image:
    label_h = 26
    title_h = 30
    left_w = thumbnail_size
    cell_w = thumbnail_size
    columns = 2 + len(scales)
    rows = 1 + len(variants)
    width = columns * cell_w
    height = title_h + rows * (thumbnail_size + label_h)
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), title[:180], fill=(0, 0, 0))

    header_y = title_h
    header_cells = [("reference", reference), ("target", target)]
    for col, (label, image) in enumerate(header_cells):
        x = col * cell_w
        panel.paste(thumbnail(image.convert("RGB"), thumbnail_size), (x, header_y))
        draw.text((x + 6, header_y + thumbnail_size + 6), label, fill=(0, 0, 0))
    for scale_index, scale in enumerate(scales):
        x = (2 + scale_index) * cell_w
        draw.rectangle((x, header_y, x + cell_w - 1, header_y + thumbnail_size - 1), outline=(210, 210, 210))
        draw.text((x + 6, header_y + thumbnail_size // 2 - 8), f"scale {format_scale(scale)}", fill=(0, 0, 0))

    for row_index, variant in enumerate(variants):
        y = title_h + (row_index + 1) * (thumbnail_size + label_h)
        draw.text((8, y + thumbnail_size // 2 - 8), variant, fill=(0, 0, 0))
        for scale_index, scale in enumerate(scales):
            x = (2 + scale_index) * cell_w
            image = outputs[(variant, scale)]
            panel.paste(thumbnail(image.convert("RGB"), thumbnail_size), (x, y))
            draw.text((x + 6, y + thumbnail_size + 6), f"{variant} {format_scale(scale)}", fill=(0, 0, 0))
    return panel


def make_overview(panel_paths: list[Path]) -> Image.Image:
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    overview = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in panels:
        overview.paste(panel, (0, y))
        y += panel.height
    return overview


def format_scale(scale: float) -> str:
    text = f"{float(scale):g}"
    return text.replace("-", "m").replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
