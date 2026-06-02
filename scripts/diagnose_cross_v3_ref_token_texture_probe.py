"""Probe whether Cross V3 reference tokens preserve z_ref texture.

This is a feature-space diagnostic only. It does not run the FLUX transformer,
ControlNet, or sampling. For a set of reference images, it compares texture
nearest-neighbor retrieval in:

    image GLCM texture stats
    raw VAE z_ref
    2x2 packed z_ref/reference grid
    MLP hidden activations
    final reference_tokens

If raw/packed features retrieve texture-similar references but final tokens do
not, the reference-context MLP is a likely texture bottleneck. If final tokens
also retrieve texture-similar references, the texture survives projection and
the later cross-attention/training objective is the more likely failure point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPRESENTATION_NAMES = (
    "z_ref_raw",
    "z_ref_packed",
    "mlp_hidden",
    "reference_tokens",
)


@dataclass
class RefTokenTextureProbeBundle:
    pretrained_model_name_or_path: str | Path
    checkpoint_path: Path
    device: str
    torch_dtype: torch.dtype
    vae: torch.nn.Module
    reference_context_encoder: torch.nn.Module
    reference_spec: object


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Feature-space probe for whether Cross V3 2x2 pack + MLP reference "
            "tokens preserve reference texture."
        )
    )
    parser.add_argument("--pretrained-model-name-or-path", required=True, help="FLUX model dir/path.")
    parser.add_argument("--checkpoint", required=True, help="Cross V3 checkpoint dir with phase5_conditioning.pt.")
    parser.add_argument("--metadata", default=None, help="Optional metadata_cross_{train,val}.json path.")
    parser.add_argument("--image", action="append", default=[], help="Reference image path. May be repeated.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference-sample-id",
        action="append",
        default=[],
        help="Specific reference_sample_id to include from metadata. May be repeated.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--thumbnail-size", type=int, default=160)
    parser.add_argument("--overview-max-samples", type=int, default=24)
    parser.add_argument("--glcm-levels", type=int, default=32)
    parser.add_argument("--glcm-distances", default="1,2,4")
    parser.add_argument("--glcm-angles", default="0,45,90,135")
    return parser


def parse_args(args=None) -> argparse.Namespace:
    return build_parser().parse_args(args)


def main(argv=None) -> int:
    args = parse_args(argv)

    from controlnet_train.cli.eval_controlnet_flux_cross import _safe_name, read_cross_metadata
    from controlnet_train.data.common import load_image_tensor
    from scripts.diagnose_cross_v3_ref_mismatch import (
        _COLOR_FEATURE_KEYS,
        _GLCM_FEATURE_KEYS,
        _parse_angles,
        _parse_int_list,
        image_quant_stats,
    )

    metadata_records = read_cross_metadata(args.metadata) if args.metadata else None
    records = build_reference_records(
        image_paths=args.image,
        metadata_records=metadata_records,
        reference_sample_ids=args.reference_sample_id,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    if len(records) < 2:
        raise ValueError("Need at least two unique references for texture retrieval probing.")

    dtype_by_name = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    bundle = load_ref_token_texture_probe_bundle(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        torch_dtype=dtype_by_name[args.torch_dtype],
    )

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    glcm_distances = _parse_int_list(args.glcm_distances)
    glcm_angles = _parse_angles(args.glcm_angles)

    sample_rows: list[dict[str, Any]] = []
    texture_stat_rows: list[dict[str, float]] = []
    pil_images: list[Image.Image] = []
    feature_vectors: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATION_NAMES}

    with torch.inference_mode():
        for index, record in enumerate(records):
            ref_id = reference_record_id(record)
            image_path = Path(record["reference_image"])
            reference_tensor = load_image_tensor(image_path)
            reference_pil = Image.open(image_path).convert("RGB")
            pil_images.append(reference_pil)

            image_stats = image_quant_stats(
                reference_pil,
                levels=args.glcm_levels,
                distances=glcm_distances,
                angles=glcm_angles,
            )
            texture_stat_rows.append(image_stats)

            representations, rep_stats = extract_reference_representations(
                bundle=bundle,
                reference_image=reference_tensor,
            )
            for name in REPRESENTATION_NAMES:
                feature_vectors[name].append(tensor_texture_signature(representations[name]))

            sample_row: dict[str, Any] = {
                "index": index,
                "reference_sample_id": ref_id,
                "reference_image": str(image_path),
                "dataset": record.get("dataset", ""),
            }
            sample_row.update({f"image_{key}": float(value) for key, value in image_stats.items()})
            sample_row.update(rep_stats)
            sample_rows.append(sample_row)

            sample_dir = samples_dir / f"{index:04d}_ref_{_safe_name(ref_id)}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            reference_pil.save(sample_dir / "reference.png")
            (sample_dir / "features.json").write_text(
                json.dumps(sample_row, indent=2, ensure_ascii=False, allow_nan=True),
                encoding="utf8",
            )
            print(
                f"[{index + 1}/{len(records)}] ref={ref_id} "
                f"z_ref_std={sample_row['z_ref_raw_std']:.4f} "
                f"packed_std={sample_row['z_ref_packed_std']:.4f} "
                f"token_std={sample_row['reference_tokens_std']:.4f}"
            )

    texture_matrix = stat_matrix(texture_stat_rows, _GLCM_FEATURE_KEYS)
    color_matrix = stat_matrix(texture_stat_rows, _COLOR_FEATURE_KEYS)
    texture_distance = pairwise_l2(zscore_columns(texture_matrix))
    color_distance = pairwise_l2(zscore_columns(color_matrix))
    feature_distances = {
        name: pairwise_l2(zscore_columns(np.stack(vectors, axis=0)))
        for name, vectors in feature_vectors.items()
    }

    pair_rows = build_pairwise_rows(
        records=records,
        texture_distance=texture_distance,
        color_distance=color_distance,
        feature_distances=feature_distances,
    )
    neighbor_rows = build_neighbor_rows(
        records=records,
        texture_distance=texture_distance,
        color_distance=color_distance,
        feature_distances=feature_distances,
        top_k=args.top_k,
    )

    write_rows(output_dir / "sample_features.csv", sample_rows)
    write_rows(output_dir / "pairwise_distances.csv", pair_rows)
    write_rows(output_dir / "nearest_neighbors.csv", neighbor_rows)
    (output_dir / "pairwise_distances.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, allow_nan=True) for row in pair_rows) + "\n",
        encoding="utf8",
    )

    summary = build_probe_summary(
        feature_distances=feature_distances,
        texture_distance=texture_distance,
        color_distance=color_distance,
        top_k=args.top_k,
    )
    summary["diagnostic"] = "cross_v3_ref_token_texture_probe"
    summary["interpretation"] = interpret_probe_summary(summary)
    summary["glcm_config"] = {
        "levels": args.glcm_levels,
        "distances": glcm_distances,
        "angles_degrees": glcm_angles,
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf8",
    )

    panel_paths = save_neighbor_panels(
        records=records,
        pil_images=pil_images,
        texture_distance=texture_distance,
        color_distance=color_distance,
        feature_distances=feature_distances,
        samples_dir=samples_dir,
        thumbnail_size=args.thumbnail_size,
        max_samples=args.overview_max_samples,
        safe_name=_safe_name,
    )
    if panel_paths:
        make_overview(panel_paths).save(output_dir / "overview_neighbors.png")

    print(f"interpretation={summary['interpretation']}")
    for name in REPRESENTATION_NAMES:
        print(
            f"{name}: texture_spearman={summary[f'{name}_pair_spearman_texture']:.4f} "
            f"color_spearman={summary[f'{name}_pair_spearman_color']:.4f} "
            f"top{int(summary['effective_top_k'])}_overlap={summary[f'{name}_topk_texture_overlap_mean']:.4f} "
            f"texture_nn_rank={summary[f'{name}_texture_nn_rank_in_feature_mean']:.2f}"
        )
    print(f"wrote Cross V3 ref token texture probe outputs to {output_dir}")
    return 0


def build_reference_records(
    *,
    image_paths: list[str],
    metadata_records: list[dict[str, Any]] | None,
    reference_sample_ids: list[str],
    num_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if image_paths:
        return [
            {"reference_sample_id": Path(path).stem, "reference_image": path, "dataset": ""}
            for path in image_paths
        ]
    if metadata_records is None:
        raise ValueError("Provide either --image or --metadata.")
    records = unique_reference_records(metadata_records)
    if reference_sample_ids:
        by_id = {reference_record_id(record): record for record in records}
        missing = [sample_id for sample_id in reference_sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"reference sample_id(s) not found: {missing}")
        return [by_id[sample_id] for sample_id in reference_sample_ids]
    if num_samples is None or num_samples <= 0 or num_samples >= len(records):
        return records
    selected = list(records)
    random.Random(seed).shuffle(selected)
    return selected[:num_samples]


def unique_reference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("metadata contains no records")
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for record in records:
        ref_id = reference_record_id(record)
        if ref_id in seen:
            continue
        seen.add(ref_id)
        output.append({**record, "reference_sample_id": ref_id})
    return output


def reference_record_id(record: dict[str, Any]) -> str:
    return str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)


def load_ref_token_texture_probe_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cuda",
    torch_dtype: torch.dtype | None = None,
) -> RefTokenTextureProbeBundle:
    from diffusers import AutoencoderKL

    from controlnet_train.inference.pipeline_cross_v3 import (
        _load_cross_v3_reference_spec,
        _resolve_device,
        _resolve_torch_dtype,
        _torch_load_weights,
    )
    from controlnet_train.modules.cross_v3_conditioning import CrossV3ReferenceContextEncoder

    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_torch_dtype(torch_dtype, resolved_device)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")
    conditioning_path = checkpoint / "phase5_conditioning.pt"
    if not conditioning_path.exists():
        raise FileNotFoundError(f"Missing phase5_conditioning.pt under checkpoint path: {checkpoint}")

    state = _torch_load_weights(conditioning_path)
    reference_state = state["reference_context_encoder"]
    reference_spec = _load_cross_v3_reference_spec(checkpoint)
    reference_context_encoder = CrossV3ReferenceContextEncoder(
        reference_latent_channels=reference_spec.reference_latent_channels,
        tissue_channels=reference_spec.tissue_channels,
        nuclei_channels=reference_spec.nuclei_channels,
        token_dim=reference_spec.token_dim,
        hidden_dim=reference_state["proj_in.weight"].shape[0],
        output_init_std=reference_spec.output_init_std,
        route_anchor_mode=reference_spec.route_anchor_mode,
        route_embedding_init_std=reference_spec.route_embedding_init_std,
    )
    reference_context_encoder.load_state_dict(reference_state)
    reference_context_encoder.to(device=resolved_device, dtype=resolved_dtype)
    reference_context_encoder.eval()

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=resolved_dtype,
    )
    vae.to(resolved_device)
    vae.eval()

    return RefTokenTextureProbeBundle(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        device=resolved_device,
        torch_dtype=resolved_dtype,
        vae=vae,
        reference_context_encoder=reference_context_encoder,
        reference_spec=reference_spec,
    )


@torch.inference_mode()
def extract_reference_representations(
    *,
    bundle: RefTokenTextureProbeBundle,
    reference_image: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    from controlnet_train.inference.pipeline_cross_v3 import _encode_images_to_latents
    from controlnet_train.modules.cross_v3_conditioning import pack_cross_v3_reference_grid

    z_ref = _encode_images_to_latents(
        bundle.vae,
        reference_image.unsqueeze(0),
        bundle.torch_dtype,
    )
    ref_tissue_feat = torch.zeros(
        z_ref.shape[0],
        int(bundle.reference_spec.tissue_channels),
        z_ref.shape[2],
        z_ref.shape[3],
        device=z_ref.device,
        dtype=bundle.torch_dtype,
    )
    ref_nuclei_feat = torch.zeros(
        z_ref.shape[0],
        int(bundle.reference_spec.nuclei_channels),
        z_ref.shape[2],
        z_ref.shape[3],
        device=z_ref.device,
        dtype=bundle.torch_dtype,
    )
    encoder = bundle.reference_context_encoder
    packed = pack_cross_v3_reference_grid(
        z_ref=z_ref,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
    )
    hidden = encoder.act(encoder.proj_in(encoder.norm(packed)))
    reference_tokens = encoder.proj_out(hidden)

    h2 = int(z_ref.shape[2] // 2)
    w2 = int(z_ref.shape[3] // 2)
    representations = {
        "z_ref_raw": z_ref[0].detach().float().cpu(),
        "z_ref_packed": sequence_to_chw(packed, height=h2, width=w2),
        "mlp_hidden": sequence_to_chw(hidden, height=h2, width=w2),
        "reference_tokens": sequence_to_chw(reference_tokens, height=h2, width=w2),
    }
    stats: dict[str, float] = {}
    for name, tensor in representations.items():
        value = tensor.detach().float()
        stats[f"{name}_channels"] = float(value.shape[0])
        stats[f"{name}_height"] = float(value.shape[1])
        stats[f"{name}_width"] = float(value.shape[2])
        stats[f"{name}_mean"] = float(value.mean().item())
        stats[f"{name}_std"] = float(value.std(unbiased=False).item())
        stats[f"{name}_l2_norm"] = float(torch.linalg.vector_norm(value).item())
    return representations, stats


def sequence_to_chw(sequence: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"sequence must have shape (B, N, C), got {tuple(sequence.shape)}")
    if sequence.shape[1] != height * width:
        raise ValueError(
            f"sequence length {sequence.shape[1]} does not match requested grid {height}x{width}"
        )
    return sequence[0].detach().float().reshape(height, width, sequence.shape[2]).permute(2, 0, 1).cpu().contiguous()


def tensor_texture_signature(grid: torch.Tensor) -> np.ndarray:
    """Build an order-aware but compact texture signature for a CxHxW tensor."""

    if grid.ndim != 3:
        raise ValueError(f"grid must have shape (C, H, W), got {tuple(grid.shape)}")
    import torch.nn.functional as F

    value = grid.detach().float()
    channels, height, width = value.shape
    flat = value.reshape(channels, -1)
    parts = [
        flat.mean(dim=1),
        flat.std(dim=1, unbiased=False),
    ]

    if width > 1:
        dx = value[:, :, 1:] - value[:, :, :-1]
        dx_flat = dx.reshape(channels, -1)
        parts.extend([dx_flat.abs().mean(dim=1), dx_flat.std(dim=1, unbiased=False)])
    else:
        parts.extend(
            [
                torch.zeros(channels, device=value.device, dtype=value.dtype),
                torch.zeros(channels, device=value.device, dtype=value.dtype),
            ]
        )

    if height > 1:
        dy = value[:, 1:, :] - value[:, :-1, :]
        dy_flat = dy.reshape(channels, -1)
        parts.extend([dy_flat.abs().mean(dim=1), dy_flat.std(dim=1, unbiased=False)])
    else:
        parts.extend(
            [
                torch.zeros(channels, device=value.device, dtype=value.dtype),
                torch.zeros(channels, device=value.device, dtype=value.dtype),
            ]
        )

    if height > 1 and width > 1:
        local_mean = F.avg_pool2d(
            value.unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )[0]
    else:
        local_mean = value.mean(dim=(1, 2), keepdim=True)
    high_pass = value - local_mean
    high_flat = high_pass.reshape(channels, -1)
    parts.extend([high_flat.abs().mean(dim=1), high_flat.std(dim=1, unbiased=False)])

    signature = torch.cat(parts, dim=0).detach().cpu().numpy().astype(np.float32)
    return np.nan_to_num(signature, nan=0.0, posinf=0.0, neginf=0.0)


def stat_matrix(rows: list[dict[str, float]], keys: tuple[str, ...]) -> np.ndarray:
    matrix = np.asarray(
        [[float(row.get(key, math.nan)) for key in keys] for row in rows],
        dtype=np.float32,
    )
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def zscore_columns(matrix: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pairwise_l2(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"matrix must have shape (N, D), got {values.shape}")
    squared = np.sum(values * values, axis=1, keepdims=True, dtype=np.float64)
    distances_sq = squared + squared.T - 2.0 * (values.astype(np.float64) @ values.astype(np.float64).T)
    distances = np.sqrt(np.maximum(distances_sq, 0.0))
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float64)


def build_probe_summary(
    *,
    feature_distances: dict[str, np.ndarray],
    texture_distance: np.ndarray,
    color_distance: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    n = int(texture_distance.shape[0])
    effective_top_k = max(1, min(int(top_k), n - 1))
    summary: dict[str, float] = {
        "num_samples": float(n),
        "effective_top_k": float(effective_top_k),
        "random_topk_overlap": float(effective_top_k / max(n - 1, 1)),
        "random_neighbor_rank_mean": float(n / 2.0),
    }
    for name in REPRESENTATION_NAMES:
        metrics = build_retrieval_metrics(
            feature_distances[name],
            texture_distance=texture_distance,
            color_distance=color_distance,
            top_k=effective_top_k,
        )
        for key, value in metrics.items():
            summary[f"{name}_{key}"] = value
    return summary


def build_retrieval_metrics(
    feature_distance: np.ndarray,
    *,
    texture_distance: np.ndarray,
    color_distance: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    n = int(feature_distance.shape[0])
    if n < 2:
        raise ValueError("Need at least two samples to build retrieval metrics.")
    effective_top_k = max(1, min(int(top_k), n - 1))

    texture_nn_ranks: list[float] = []
    feature_nn_texture_ranks: list[float] = []
    overlaps: list[float] = []
    hits: list[float] = []
    for anchor_index in range(n):
        texture_order = nearest_order(texture_distance, anchor_index)
        feature_order = nearest_order(feature_distance, anchor_index)
        texture_topk = set(int(idx) for idx in texture_order[:effective_top_k])
        feature_topk = [int(idx) for idx in feature_order[:effective_top_k]]
        texture_nn = int(texture_order[0])
        feature_nn = int(feature_order[0])
        texture_nn_ranks.append(float(rank_of(feature_order, texture_nn)))
        feature_nn_texture_ranks.append(float(rank_of(texture_order, feature_nn)))
        overlaps.append(float(len(texture_topk.intersection(feature_topk)) / effective_top_k))
        hits.append(1.0 if texture_nn in feature_topk else 0.0)

    feature_pairs = upper_triangle_values(feature_distance)
    texture_pairs = upper_triangle_values(texture_distance)
    color_pairs = upper_triangle_values(color_distance)
    return {
        "pair_spearman_texture": spearman_correlation(feature_pairs, texture_pairs),
        "pair_pearson_texture": pearson_correlation(feature_pairs, texture_pairs),
        "pair_spearman_color": spearman_correlation(feature_pairs, color_pairs),
        "pair_pearson_color": pearson_correlation(feature_pairs, color_pairs),
        "texture_nn_rank_in_feature_mean": float(np.mean(texture_nn_ranks)),
        "texture_nn_rank_in_feature_median": float(np.median(texture_nn_ranks)),
        "feature_nn_rank_in_texture_mean": float(np.mean(feature_nn_texture_ranks)),
        "feature_nn_rank_in_texture_median": float(np.median(feature_nn_texture_ranks)),
        "topk_texture_overlap_mean": float(np.mean(overlaps)),
        "topk_texture_overlap_std": float(np.std(overlaps)),
        "texture_nn_hit_at_k_mean": float(np.mean(hits)),
    }


def nearest_order(distance: np.ndarray, anchor_index: int) -> np.ndarray:
    row = np.asarray(distance[anchor_index], dtype=np.float64)
    order = np.argsort(row, kind="mergesort")
    return np.asarray([idx for idx in order if idx != anchor_index and math.isfinite(float(row[idx]))], dtype=np.int64)


def rank_of(order: np.ndarray, target_index: int) -> int:
    matches = np.where(order == target_index)[0]
    if matches.size == 0:
        raise ValueError(f"target index {target_index} is not present in nearest-neighbor order")
    return int(matches[0]) + 1


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    row, col = np.triu_indices(values.shape[0], k=1)
    return values[row, col]


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_values, right_values = finite_pair_values(left, right)
    if left_values.size < 2:
        return math.nan
    return pearson_correlation(rankdata_average(left_values), rankdata_average(right_values))


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_values, right_values = finite_pair_values(left, right)
    if left_values.size < 2:
        return math.nan
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom <= 1e-12:
        return math.nan
    return float(np.dot(left_centered, right_centered) / denom)


def finite_pair_values(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_values.shape != right_values.shape:
        raise ValueError(f"value shapes differ: {left_values.shape} vs {right_values.shape}")
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    return left_values[mask], right_values[mask]


def rankdata_average(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.shape[0], dtype=np.float64)
    start = 0
    while start < array.shape[0]:
        end = start + 1
        while end < array.shape[0] and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def interpret_probe_summary(summary: dict[str, float]) -> str:
    raw_good = representation_supports_texture(summary, "z_ref_raw")
    packed_good = representation_supports_texture(summary, "z_ref_packed")
    hidden_good = representation_supports_texture(summary, "mlp_hidden")
    token_good = representation_supports_texture(summary, "reference_tokens")

    token_texture = float(summary.get("reference_tokens_pair_spearman_texture", math.nan))
    token_color = float(summary.get("reference_tokens_pair_spearman_color", math.nan))
    token_tracks_color_more = (
        math.isfinite(token_texture)
        and math.isfinite(token_color)
        and token_color > token_texture + 0.15
    )

    if packed_good and not hidden_good:
        return "reference_mlp_input_projection_likely_loses_texture"
    if hidden_good and not token_good:
        return "reference_mlp_output_projection_likely_loses_texture"
    if packed_good and not token_good and token_tracks_color_more:
        return "reference_tokens_track_color_but_not_texture"
    if packed_good and not token_good:
        return "reference_mlp_likely_loses_texture"
    if token_good:
        return "reference_tokens_preserve_texture_cross_attention_or_training_is_suspect"
    if raw_good and not packed_good:
        return "packed_signature_or_2x2_layout_loses_texture"
    if not raw_good:
        return "feature_probe_does_not_recover_texture_from_z_ref"
    return "mixed_or_weak_reference_token_texture_preservation"


def representation_supports_texture(summary: dict[str, float], name: str) -> bool:
    corr = float(summary.get(f"{name}_pair_spearman_texture", math.nan))
    overlap = float(summary.get(f"{name}_topk_texture_overlap_mean", math.nan))
    rank = float(summary.get(f"{name}_texture_nn_rank_in_feature_mean", math.nan))
    baseline_overlap = float(summary.get("random_topk_overlap", math.nan))
    baseline_rank = float(summary.get("random_neighbor_rank_mean", math.nan))

    if math.isfinite(corr) and corr >= 0.25:
        return True
    if math.isfinite(overlap) and math.isfinite(baseline_overlap):
        if overlap >= min(1.0, max(baseline_overlap + 0.10, baseline_overlap * 1.6)):
            return True
    if math.isfinite(rank) and math.isfinite(baseline_rank) and rank <= baseline_rank * 0.65:
        return True
    return False


def build_pairwise_rows(
    *,
    records: list[dict[str, Any]],
    texture_distance: np.ndarray,
    color_distance: np.ndarray,
    feature_distances: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            row: dict[str, Any] = {
                "left_index": left_index,
                "right_index": right_index,
                "left_reference_sample_id": reference_record_id(records[left_index]),
                "right_reference_sample_id": reference_record_id(records[right_index]),
                "texture_distance": float(texture_distance[left_index, right_index]),
                "color_distance": float(color_distance[left_index, right_index]),
            }
            for name, distance in feature_distances.items():
                row[f"{name}_distance"] = float(distance[left_index, right_index])
            rows.append(row)
    return rows


def build_neighbor_rows(
    *,
    records: list[dict[str, Any]],
    texture_distance: np.ndarray,
    color_distance: np.ndarray,
    feature_distances: dict[str, np.ndarray],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    effective_top_k = max(1, min(int(top_k), len(records) - 1))
    all_distances = {
        "texture": texture_distance,
        "color": color_distance,
        **feature_distances,
    }
    for anchor_index, record in enumerate(records):
        row: dict[str, Any] = {
            "anchor_index": anchor_index,
            "anchor_reference_sample_id": reference_record_id(record),
            "top_k": effective_top_k,
        }
        for name, distance in all_distances.items():
            order = nearest_order(distance, anchor_index)
            top_indices = [int(idx) for idx in order[:effective_top_k]]
            row[f"{name}_top1_index"] = top_indices[0]
            row[f"{name}_top1_reference_sample_id"] = reference_record_id(records[top_indices[0]])
            row[f"{name}_top1_distance"] = float(distance[anchor_index, top_indices[0]])
            row[f"{name}_topk_reference_sample_ids"] = ",".join(reference_record_id(records[idx]) for idx in top_indices)
        rows.append(row)
    return rows


def save_neighbor_panels(
    *,
    records: list[dict[str, Any]],
    pil_images: list[Image.Image],
    texture_distance: np.ndarray,
    color_distance: np.ndarray,
    feature_distances: dict[str, np.ndarray],
    samples_dir: Path,
    thumbnail_size: int,
    max_samples: int,
    safe_name,
) -> list[Path]:
    panel_paths: list[Path] = []
    all_distances = {
        "texture_nn": texture_distance,
        "color_nn": color_distance,
        "z_ref_raw_nn": feature_distances["z_ref_raw"],
        "z_ref_packed_nn": feature_distances["z_ref_packed"],
        "mlp_hidden_nn": feature_distances["mlp_hidden"],
        "reference_tokens_nn": feature_distances["reference_tokens"],
    }
    for anchor_index, record in enumerate(records[: max(0, max_samples)]):
        ref_id = reference_record_id(record)
        neighbors = []
        for label, distance in all_distances.items():
            nn_index = int(nearest_order(distance, anchor_index)[0])
            neighbors.append((label, pil_images[nn_index], reference_record_id(records[nn_index])))
        panel = make_neighbor_panel(
            anchor=pil_images[anchor_index],
            anchor_id=ref_id,
            neighbors=neighbors,
            thumbnail_size=thumbnail_size,
        )
        sample_dir = samples_dir / f"{anchor_index:04d}_ref_{safe_name(ref_id)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        panel_path = sample_dir / "neighbor_panel.png"
        panel.save(panel_path)
        panel_paths.append(panel_path)
    return panel_paths


def make_neighbor_panel(
    *,
    anchor: Image.Image,
    anchor_id: str,
    neighbors: list[tuple[str, Image.Image, str]],
    thumbnail_size: int,
) -> Image.Image:
    images = [("anchor", anchor.convert("RGB"), anchor_id)] + [
        (label, image.convert("RGB"), ref_id) for label, image, ref_id in neighbors
    ]
    thumbs = [(label, thumbnail(image, thumbnail_size), ref_id) for label, image, ref_id in images]
    title_h = 28
    label_h = 48
    panel = Image.new("RGB", (thumbnail_size * len(thumbs), thumbnail_size + title_h + label_h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((6, 6), f"texture probe neighbors | anchor={anchor_id}"[:180], fill=(0, 0, 0))
    for idx, (label, image, ref_id) in enumerate(thumbs):
        x = idx * thumbnail_size
        panel.paste(image, (x, title_h))
        draw.text((x + 5, title_h + thumbnail_size + 5), label[:22], fill=(0, 0, 0))
        draw.text((x + 5, title_h + thumbnail_size + 22), ref_id[:26], fill=(80, 80, 80))
    return panel


def thumbnail(image: Image.Image, size: int) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(thumb, ((size - thumb.width) // 2, (size - thumb.height) // 2))
    return canvas


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


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
