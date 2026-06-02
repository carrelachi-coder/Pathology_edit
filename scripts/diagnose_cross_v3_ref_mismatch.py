"""Diagnose Cross V3 prompt/reference mismatch texture transfer.

This script fixes one target condition and swaps in multiple reference
conditions, then quantifies whether generated GLCM texture statistics move with
the reference image or stay anchored to the target/style bias.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross V3 fixed-target reference mismatch diagnostic.")
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--checkpoint", required=True, help="Cross V3 checkpoint dir, e.g. the 6k checkpoint.")
    parser.add_argument("--metadata", required=True, help="metadata_cross_{train,val}.json path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-sample-id", default=None, help="Fixed target sample_id. Defaults to a seeded pick.")
    parser.add_argument("--target-index", type=int, default=None, help="Fixed target record index after metadata load.")
    parser.add_argument(
        "--reference-sample-id",
        action="append",
        default=[],
        help="Reference sample_id to force into the fixed target. May be repeated.",
    )
    parser.add_argument("--num-refs", type=int, default=4, help="Number of automatic mismatch references.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument(
        "--color-match",
        choices=("none", "lab"),
        default="none",
        help="Keep 'none' for texture diagnostics; 'lab' is useful as a stain-only control.",
    )
    parser.add_argument("--thumbnail-size", type=int, default=192)
    parser.add_argument("--glcm-levels", type=int, default=32)
    parser.add_argument("--glcm-distances", default="1,2,4")
    parser.add_argument("--glcm-angles", default="0,45,90,135")
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)

    from controlnet_train.cli.eval_controlnet_flux_cross import (
        _match_image_color_to_reference,
        _pil_to_chw_float,
        _safe_name,
        compute_cross_metrics,
        read_cross_metadata,
    )
    from controlnet_train.data.common import load_image_tensor, load_nuclei_mask, load_tissue_mask
    from controlnet_train.inference.pipeline_cross_v3 import (
        CROSS_V3_PROMPT,
        CROSS_V3_REFERENCE_WITH_REF,
        load_cross_v3_bundle,
        run_cross_v3_bundle,
    )

    import torch

    records = read_cross_metadata(args.metadata)
    target_record = _select_target_record(
        records,
        target_sample_id=args.target_sample_id,
        target_index=args.target_index,
        seed=args.seed,
    )
    reference_records = _select_reference_records(
        records,
        target_record=target_record,
        reference_sample_ids=args.reference_sample_id,
        num_refs=args.num_refs,
        seed=args.seed,
        glcm_levels=args.glcm_levels,
        glcm_distances=_parse_int_list(args.glcm_distances),
        glcm_angles=_parse_angles(args.glcm_angles),
    )

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    bundle = load_cross_v3_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )

    target_image_path = Path(target_record["target_image"])
    target_tissue_mask_path = Path(target_record["target_tissue_mask"])
    target_nuclei_mask_path = Path(target_record["target_nuclei_mask"])
    target_image_tensor = load_image_tensor(target_image_path)
    target_tissue_mask = load_tissue_mask(target_tissue_mask_path)
    target_nuclei_mask = load_nuclei_mask(target_nuclei_mask_path)
    target_pil = Image.open(target_image_path).convert("RGB")
    target_array = _pil_to_chw_float(target_pil)

    target_image_stats = image_quant_stats(
        target_pil,
        levels=args.glcm_levels,
        distances=_parse_int_list(args.glcm_distances),
        angles=_parse_angles(args.glcm_angles),
    )
    target_nuclei_stats = nuclei_morphology_stats(np.asarray(Image.open(target_nuclei_mask_path)))

    rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for index, ref_record in enumerate(reference_records):
        ref_id = str(ref_record.get("reference_sample_id") or ref_record.get("sample_id") or Path(ref_record["reference_image"]).stem)
        sample_dir = samples_dir / f"{index:04d}_target_{_safe_name(str(target_record['sample_id']))}__ref_{_safe_name(ref_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        reference_image_path = Path(ref_record["reference_image"])
        reference_tissue_mask_path = Path(ref_record["reference_tissue_mask"])
        reference_nuclei_mask_path = Path(ref_record["reference_nuclei_mask"])
        reference_pil = Image.open(reference_image_path).convert("RGB")

        with torch.no_grad():
            prediction = run_cross_v3_bundle(
                bundle,
                reference_image=load_image_tensor(reference_image_path),
                reference_tissue_mask=load_tissue_mask(reference_tissue_mask_path),
                reference_nuclei_mask=load_nuclei_mask(reference_nuclei_mask_path),
                target_tissue_mask=target_tissue_mask,
                target_nuclei_mask=target_nuclei_mask,
                prompt=CROSS_V3_PROMPT,
                reference_condition_mode=CROSS_V3_REFERENCE_WITH_REF,
            ).convert("RGB")

        raw_prediction = prediction
        if args.color_match == "lab":
            prediction = _match_image_color_to_reference(
                source=raw_prediction,
                reference=reference_pil,
                method=args.color_match,
            )

        reference_pil.save(sample_dir / "reference.png")
        target_pil.save(sample_dir / "target.png")
        raw_prediction.save(sample_dir / "prediction_raw.png")
        prediction.save(sample_dir / "prediction.png")

        pred_array = _pil_to_chw_float(prediction)
        full_metrics = compute_cross_metrics(pred_array, target_array)
        reference_stats = image_quant_stats(
            reference_pil,
            levels=args.glcm_levels,
            distances=_parse_int_list(args.glcm_distances),
            angles=_parse_angles(args.glcm_angles),
        )
        prediction_stats = image_quant_stats(
            prediction,
            levels=args.glcm_levels,
            distances=_parse_int_list(args.glcm_distances),
            angles=_parse_angles(args.glcm_angles),
        )
        reference_nuclei_stats = nuclei_morphology_stats(np.asarray(Image.open(reference_nuclei_mask_path)))

        row = {
            "index": index,
            "target_sample_id": target_record.get("sample_id", ""),
            "reference_sample_id": ref_id,
            "dataset": target_record.get("dataset", ""),
            "reference_dataset": ref_record.get("dataset", ""),
            "color_match_applied": args.color_match != "none",
            **full_metrics,
            **_prefix_stats("target", target_image_stats),
            **_prefix_stats("reference", reference_stats),
            **_prefix_stats("prediction", prediction_stats),
            **_prefix_stats("target_nuclei", target_nuclei_stats),
            **_prefix_stats("reference_nuclei", reference_nuclei_stats),
        }
        row.update(_distance_stats(row, left="prediction", right="reference", prefix="pred_ref"))
        row.update(_distance_stats(row, left="prediction", right="target", prefix="pred_target"))
        rows.append(row)

        (sample_dir / "metrics.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf8",
        )
        panel_path = sample_dir / "panel.png"
        make_mismatch_panel(
            reference=reference_pil,
            prediction=prediction,
            target=target_pil,
            thumbnail_size=args.thumbnail_size,
            title=f"target={target_record.get('sample_id', '')} | ref={ref_id}",
        ).save(panel_path)
        panel_paths.append(panel_path)
        print(
            f"[{index + 1}/{len(reference_records)}] target={target_record.get('sample_id', '')} "
            f"ref={ref_id} full_l1={full_metrics['full_l1']:.4f} "
            f"pred_ref_glcm_l2={row['pred_ref_glcm_l2']:.4f} pred_target_glcm_l2={row['pred_target_glcm_l2']:.4f}"
        )

    _write_rows(output_dir, rows)
    summary = build_mismatch_summary(rows)
    summary["target_record"] = target_record
    summary["reference_count"] = len(reference_records)
    summary["glcm_config"] = {
        "levels": args.glcm_levels,
        "distances": _parse_int_list(args.glcm_distances),
        "angles_degrees": _parse_angles(args.glcm_angles),
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )
    if panel_paths:
        make_overview(panel_paths).save(output_dir / "overview_grid.png")
    print(f"wrote mismatch diagnostic outputs to {output_dir}")
    return 0


def _select_target_record(
    records: list[dict[str, Any]],
    *,
    target_sample_id: str | None,
    target_index: int | None,
    seed: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("metadata contains no records")
    if target_index is not None:
        if target_index < 0 or target_index >= len(records):
            raise IndexError(f"--target-index {target_index} is outside [0, {len(records) - 1}]")
        return records[target_index]
    if target_sample_id:
        for record in records:
            if str(record.get("sample_id", "")) == target_sample_id:
                return record
        raise ValueError(f"target sample_id not found: {target_sample_id}")
    return random.Random(seed).choice(records)


def _select_reference_records(
    records: list[dict[str, Any]],
    *,
    target_record: dict[str, Any],
    reference_sample_ids: list[str],
    num_refs: int,
    seed: int,
    glcm_levels: int,
    glcm_distances: list[int],
    glcm_angles: list[int],
) -> list[dict[str, Any]]:
    candidates = _unique_reference_candidates(records, target_record)
    if reference_sample_ids:
        by_id = {
            str(candidate.get("reference_sample_id") or candidate.get("sample_id") or Path(candidate["reference_image"]).stem): candidate
            for candidate in candidates
        }
        missing = [sample_id for sample_id in reference_sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"reference sample_id(s) not found: {missing}")
        return [by_id[sample_id] for sample_id in reference_sample_ids]

    target_stats = image_quant_stats(
        Image.open(target_record["target_image"]).convert("RGB"),
        levels=glcm_levels,
        distances=glcm_distances,
        angles=glcm_angles,
    )
    rng = random.Random(seed)
    scored: list[tuple[float, float, dict[str, Any]]] = []
    for candidate in candidates:
        stats = image_quant_stats(
            Image.open(candidate["reference_image"]).convert("RGB"),
            levels=glcm_levels,
            distances=glcm_distances,
            angles=glcm_angles,
        )
        texture_distance = _feature_l2(stats, target_stats, _GLCM_FEATURE_KEYS)
        color_distance = _feature_l2(stats, target_stats, _COLOR_FEATURE_KEYS)
        scored.append((texture_distance, color_distance + rng.random() * 1e-6, candidate))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [candidate for _, _, candidate in scored[: max(1, num_refs)]]


def _unique_reference_candidates(records: list[dict[str, Any]], target_record: dict[str, Any]) -> list[dict[str, Any]]:
    target_sample_id = str(target_record.get("sample_id", ""))
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        ref_id = str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)
        if ref_id == target_sample_id or ref_id in seen:
            continue
        seen.add(ref_id)
        output.append(
            {
                **record,
                "reference_sample_id": ref_id,
                "reference_image": record["reference_image"],
                "reference_tissue_mask": record["reference_tissue_mask"],
                "reference_nuclei_mask": record["reference_nuclei_mask"],
            }
        )
    if not output:
        raise ValueError("no reference candidates available after excluding the fixed target")
    return output


_GLCM_FEATURE_KEYS = (
    "gray_glcm_contrast",
    "gray_glcm_dissimilarity",
    "gray_glcm_homogeneity",
    "gray_glcm_energy",
    "gray_glcm_correlation",
    "hema_glcm_contrast",
    "hema_glcm_dissimilarity",
    "hema_glcm_homogeneity",
    "hema_glcm_energy",
    "hema_glcm_correlation",
)
_COLOR_FEATURE_KEYS = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
)


def image_quant_stats(
    image: Image.Image,
    *,
    levels: int,
    distances: list[int],
    angles: list[int],
) -> dict[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    hema = optical_density_hematoxylin(rgb)
    stats = {
        "rgb_mean_r": float(rgb[..., 0].mean()),
        "rgb_mean_g": float(rgb[..., 1].mean()),
        "rgb_mean_b": float(rgb[..., 2].mean()),
        "rgb_std_r": float(rgb[..., 0].std()),
        "rgb_std_g": float(rgb[..., 1].std()),
        "rgb_std_b": float(rgb[..., 2].std()),
    }
    stats.update({f"gray_{key}": value for key, value in glcm_stats(gray, levels=levels, distances=distances, angles=angles).items()})
    stats.update({f"hema_{key}": value for key, value in glcm_stats(hema, levels=levels, distances=distances, angles=angles).items()})
    return stats


def optical_density_hematoxylin(rgb: np.ndarray) -> np.ndarray:
    od = -np.log(np.clip(rgb, 1e-3, 1.0))
    hema = od @ np.array([0.65, 0.70, 0.29], dtype=np.float32)
    return _normalize_float_image(hema)


def glcm_stats(image: np.ndarray, *, levels: int, distances: list[int], angles: list[int]) -> dict[str, float]:
    quantized = quantize_image(image, levels)
    matrices = []
    for distance in distances:
        for angle in angles:
            dx, dy = _offset_from_angle(distance, angle)
            matrix = _glcm_matrix(quantized, levels=levels, dx=dx, dy=dy)
            if matrix.sum() > 0:
                matrices.append(matrix / matrix.sum())
    if not matrices:
        return {
            "glcm_contrast": math.nan,
            "glcm_dissimilarity": math.nan,
            "glcm_homogeneity": math.nan,
            "glcm_energy": math.nan,
            "glcm_correlation": math.nan,
        }
    props = [_glcm_props(matrix) for matrix in matrices]
    return {
        key: float(np.mean([prop[key] for prop in props]))
        for key in props[0]
    }


def quantize_image(image: np.ndarray, levels: int) -> np.ndarray:
    if levels < 2:
        raise ValueError("GLCM levels must be >= 2")
    normalized = _normalize_float_image(np.asarray(image, dtype=np.float32))
    return np.clip(np.floor(normalized * levels), 0, levels - 1).astype(np.int64)


def _normalize_float_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    min_value = float(arr.min())
    max_value = float(arr.max())
    if max_value <= min_value:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_value) / (max_value - min_value)


def _offset_from_angle(distance: int, angle_degrees: int) -> tuple[int, int]:
    if distance <= 0:
        raise ValueError("GLCM distances must be positive")
    normalized = angle_degrees % 180
    if normalized == 0:
        return distance, 0
    if normalized == 45:
        return distance, -distance
    if normalized == 90:
        return 0, -distance
    if normalized == 135:
        return -distance, -distance
    raise ValueError(f"Unsupported GLCM angle {angle_degrees}; use 0,45,90,135.")


def _glcm_matrix(image: np.ndarray, *, levels: int, dx: int, dy: int) -> np.ndarray:
    height, width = image.shape
    x0 = max(0, -dx)
    x1 = min(width, width - dx)
    y0 = max(0, -dy)
    y1 = min(height, height - dy)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((levels, levels), dtype=np.float64)
    source = image[y0:y1, x0:x1].reshape(-1)
    target = image[y0 + dy : y1 + dy, x0 + dx : x1 + dx].reshape(-1)
    matrix = np.bincount(source * levels + target, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
    return matrix + matrix.T


def _glcm_props(matrix: np.ndarray) -> dict[str, float]:
    levels = matrix.shape[0]
    i, j = np.indices((levels, levels), dtype=np.float64)
    diff = i - j
    contrast = float(np.sum(matrix * diff * diff))
    dissimilarity = float(np.sum(matrix * np.abs(diff)))
    homogeneity = float(np.sum(matrix / (1.0 + diff * diff)))
    energy = float(np.sqrt(np.sum(matrix * matrix)))
    mean_i = float(np.sum(i * matrix))
    mean_j = float(np.sum(j * matrix))
    std_i = math.sqrt(float(np.sum(((i - mean_i) ** 2) * matrix)))
    std_j = math.sqrt(float(np.sum(((j - mean_j) ** 2) * matrix)))
    if std_i <= 1e-12 or std_j <= 1e-12:
        correlation = 1.0
    else:
        correlation = float(np.sum((i - mean_i) * (j - mean_j) * matrix) / (std_i * std_j))
    return {
        "glcm_contrast": contrast,
        "glcm_dissimilarity": dissimilarity,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
    }


def nuclei_morphology_stats(mask: np.ndarray) -> dict[str, float]:
    from scipy import ndimage

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    binary = array > 0
    labeled, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=bool))
    areas: list[float] = []
    circularities: list[float] = []
    aspect_ratios: list[float] = []
    for label_id in range(1, count + 1):
        component = labeled == label_id
        area = float(component.sum())
        if area <= 0:
            continue
        slices = ndimage.find_objects(component.astype(np.uint8))[0]
        height = float(slices[0].stop - slices[0].start)
        width = float(slices[1].stop - slices[1].start)
        eroded = ndimage.binary_erosion(component, structure=np.ones((3, 3), dtype=bool), border_value=0)
        perimeter = float(np.count_nonzero(component & ~eroded))
        circularity = float(4.0 * math.pi * area / max(perimeter * perimeter, 1.0))
        areas.append(area)
        circularities.append(circularity)
        aspect_ratios.append(max(width, height) / max(min(width, height), 1.0))
    total_pixels = float(array.shape[0] * array.shape[1]) if array.size else 0.0
    return {
        "component_count": float(len(areas)),
        "density": float(np.count_nonzero(binary) / max(total_pixels, 1.0)),
        "area_mean": _finite_mean(areas),
        "area_std": _finite_std(areas),
        "circularity_mean": _finite_mean(circularities),
        "aspect_ratio_mean": _finite_mean(aspect_ratios),
    }


def build_mismatch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_refs": float(len(rows))}
    for key in ("full_l1", "pred_ref_glcm_l2", "pred_target_glcm_l2", "pred_ref_color_l2", "pred_target_color_l2"):
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    summary["reference_prediction_glcm_correlation"] = _paired_feature_correlation(
        rows,
        prefix_a="reference",
        prefix_b="prediction",
        keys=_GLCM_FEATURE_KEYS,
    )
    summary["target_prediction_glcm_correlation"] = _paired_feature_correlation(
        rows,
        prefix_a="target",
        prefix_b="prediction",
        keys=_GLCM_FEATURE_KEYS,
    )
    summary["texture_transfer_hint"] = _interpret_summary(summary)
    return summary


def _interpret_summary(summary: dict[str, Any]) -> str:
    ref_corr = float(summary.get("reference_prediction_glcm_correlation", math.nan))
    target_corr = float(summary.get("target_prediction_glcm_correlation", math.nan))
    pred_ref = float(summary.get("pred_ref_glcm_l2_mean", math.nan))
    pred_target = float(summary.get("pred_target_glcm_l2_mean", math.nan))
    if math.isfinite(ref_corr) and math.isfinite(target_corr) and ref_corr > target_corr + 0.15:
        return "prediction_glcm_moves_with_reference"
    if math.isfinite(pred_ref) and math.isfinite(pred_target) and pred_ref < pred_target * 0.85:
        return "prediction_glcm_closer_to_reference"
    return "no_clear_reference_texture_transfer"


def _paired_feature_correlation(rows: list[dict[str, Any]], *, prefix_a: str, prefix_b: str, keys: tuple[str, ...]) -> float:
    a_values: list[float] = []
    b_values: list[float] = []
    for row in rows:
        for key in keys:
            left = row.get(f"{prefix_a}_{key}")
            right = row.get(f"{prefix_b}_{key}")
            if left is None or right is None:
                continue
            left_f = float(left)
            right_f = float(right)
            if math.isfinite(left_f) and math.isfinite(right_f):
                a_values.append(left_f)
                b_values.append(right_f)
    if len(a_values) < 2:
        return math.nan
    return float(np.corrcoef(np.asarray(a_values), np.asarray(b_values))[0, 1])


def _distance_stats(row: dict[str, Any], *, left: str, right: str, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_glcm_l2": _row_feature_l2(row, left, right, _GLCM_FEATURE_KEYS),
        f"{prefix}_color_l2": _row_feature_l2(row, left, right, _COLOR_FEATURE_KEYS),
    }


def _row_feature_l2(row: dict[str, Any], left: str, right: str, keys: tuple[str, ...]) -> float:
    values = []
    for key in keys:
        a = row.get(f"{left}_{key}")
        b = row.get(f"{right}_{key}")
        if a is None or b is None:
            continue
        a_f = float(a)
        b_f = float(b)
        if math.isfinite(a_f) and math.isfinite(b_f):
            values.append((a_f - b_f) ** 2)
    if not values:
        return math.nan
    return float(math.sqrt(sum(values) / len(values)))


def _feature_l2(left: dict[str, float], right: dict[str, float], keys: tuple[str, ...]) -> float:
    values = []
    for key in keys:
        if key in left and key in right and math.isfinite(left[key]) and math.isfinite(right[key]):
            values.append((left[key] - right[key]) ** 2)
    if not values:
        return math.nan
    return float(math.sqrt(sum(values) / len(values)))


def _prefix_stats(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in stats.items()}


def make_mismatch_panel(*, reference: Image.Image, prediction: Image.Image, target: Image.Image, thumbnail_size: int, title: str) -> Image.Image:
    images = [
        ("reference", reference.convert("RGB")),
        ("prediction", prediction.convert("RGB")),
        ("target", target.convert("RGB")),
    ]
    thumbs = [(label, _thumbnail(image, thumbnail_size)) for label, image in images]
    label_h = 34
    title_h = 28
    panel = Image.new("RGB", (thumbnail_size * len(thumbs), thumbnail_size + label_h + title_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), title[:160], fill=(0, 0, 0))
    for idx, (label, image) in enumerate(thumbs):
        x = idx * thumbnail_size
        panel.paste(image, (x, title_h))
        draw.text((x + 6, title_h + thumbnail_size + 8), label, fill=(0, 0, 0))
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


def _thumbnail(image: Image.Image, size: int) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(thumb, ((size - thumb.width) // 2, (size - thumb.height) // 2))
    return canvas


def _write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.jsonl").open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (output_dir / "metrics.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_angles(value: str) -> list[int]:
    return _parse_int_list(value)


def _finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else math.nan


def _finite_std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.std(finite)) if finite else math.nan


if __name__ == "__main__":
    raise SystemExit(main())
