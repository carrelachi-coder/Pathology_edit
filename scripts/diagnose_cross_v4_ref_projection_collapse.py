"""Compare Cross V4 z_ref distance against projected reference-context distance.

This diagnostic answers a narrow question:

    A/B reference images -> FLUX VAE z_ref -> Cross V4 reference encoder tokens

If z_ref(A, B) differs substantially but the projected context tokens are nearly
identical, then the reference context encoder is collapsing appearance
differences before FLUX ever sees them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_CHECKPOINT = "/data/wqx/flowedit/controlnet_cross_v4_mask_guided/checkpoint-4000-old-version"
DEFAULT_MODEL = "/data/huggingface/FLUX.1-dev"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether the Cross V4 reference context encoder collapses "
            "differences between two reference images."
        )
    )
    parser.add_argument("--pretrained-model-name-or-path", default=DEFAULT_MODEL, help="FLUX model dir/path.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Cross V4 checkpoint dir.")
    parser.add_argument(
        "--metadata",
        default=None,
        help="metadata_cross_{train,val}.json path. When set, A/B are selected from unique references.",
    )
    parser.add_argument("--image-a", default=None, help="Reference image A. Overrides metadata selection.")
    parser.add_argument("--image-b", default=None, help="Reference image B. Overrides metadata selection.")
    parser.add_argument(
        "--reference-sample-id-a",
        default=None,
        help="Select reference A by reference_sample_id from metadata.",
    )
    parser.add_argument(
        "--reference-sample-id-b",
        default=None,
        help="Select reference B by reference_sample_id from metadata.",
    )
    parser.add_argument(
        "--record-index-a",
        type=int,
        default=0,
        help="Unique-reference index for A when --reference-sample-id-a is not set.",
    )
    parser.add_argument(
        "--record-index-b",
        type=int,
        default=1,
        help="Unique-reference index for B when --reference-sample-id-b is not set.",
    )
    parser.add_argument(
        "--random-pair",
        action="store_true",
        help="Select a random unique-reference A/B pair from metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --random-pair.")
    parser.add_argument("--tissue-mask-a", default=None, help="Optional tissue mask for image A.")
    parser.add_argument("--tissue-mask-b", default=None, help="Optional tissue mask for image B.")
    parser.add_argument("--nuclei-mask-a", default=None, help="Optional nuclei mask for image A.")
    parser.add_argument("--nuclei-mask-b", default=None, help="Optional nuclei mask for image B.")
    parser.add_argument(
        "--use-mask-features",
        action="store_true",
        help=(
            "Use HTE/tissue downsampler and nuclei encoder features from the masks. "
            "Without this flag, mask features are zeroed so the probe isolates z_ref -> projection."
        ),
    )
    parser.add_argument("--output-json", default=None, help="Optional path for metrics JSON.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--collapse-token-rel-threshold",
        type=float,
        default=0.05,
        help="Context-token relative L2 at/below this value is flagged as collapsed.",
    )
    parser.add_argument(
        "--large-z-ref-rel-threshold",
        type=float,
        default=0.10,
        help="z_ref relative L2 at/above this value is treated as a large latent difference.",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from diffusers import AutoencoderKL

    from controlnet_train.data.common import load_image_tensor, load_nuclei_mask, load_tissue_mask
    from controlnet_train.cli.eval_controlnet_flux_cross import read_cross_metadata
    from controlnet_train.inference.pipeline_cross_v3 import (
        _resolve_device,
        _resolve_torch_dtype,
        _torch_load_weights,
    )
    from controlnet_train.inference.pipeline_cross_v4 import (
        _load_cross_v4_condition_modules,
        _load_cross_v4_reference_spec,
    )
    from controlnet_train.modules.cross_v4_conditioning import CrossV4ReferenceEncoding

    device = _resolve_device(args.device)
    dtype = _resolve_torch_dtype(resolve_dtype_arg(args.torch_dtype), device)
    checkpoint = Path(args.checkpoint)
    conditioning_path = checkpoint / "phase5_conditioning.pt"
    if not conditioning_path.exists():
        raise FileNotFoundError(f"Missing phase5_conditioning.pt under checkpoint path: {checkpoint}")

    state = _torch_load_weights(conditioning_path)
    reference_spec = _load_cross_v4_reference_spec(state)
    modules = _load_cross_v4_condition_modules(
        state=state,
        reference_spec=reference_spec,
        device=device,
        torch_dtype=dtype,
    )
    for module in modules.values():
        module.eval()
        module.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)

    selected_a, selected_b = resolve_reference_pair(args, read_cross_metadata=read_cross_metadata)
    image_a_path = selected_a["reference_image"]
    image_b_path = selected_b["reference_image"]
    image_a = load_image_tensor(image_a_path)
    image_b = load_image_tensor(image_b_path)
    if image_a.shape != image_b.shape:
        raise ValueError(
            "image A/B must have the same CHW shape for direct token comparison, "
            f"got {tuple(image_a.shape)} vs {tuple(image_b.shape)}."
        )

    if args.use_mask_features:
        require_mask_paths(selected_a, name="A")
        require_mask_paths(selected_b, name="B")
        tissue_a = load_tissue_mask(selected_a["reference_tissue_mask"])
        tissue_b = load_tissue_mask(selected_b["reference_tissue_mask"])
        nuclei_a = load_nuclei_mask(selected_a["reference_nuclei_mask"], remap=True)
        nuclei_b = load_nuclei_mask(selected_b["reference_nuclei_mask"], remap=True)
    else:
        height, width = int(image_a.shape[1]), int(image_a.shape[2])
        tissue_a = torch.zeros(height, width, dtype=torch.long)
        tissue_b = torch.zeros(height, width, dtype=torch.long)
        nuclei_a = torch.zeros(height, width, dtype=torch.long)
        nuclei_b = torch.zeros(height, width, dtype=torch.long)

    validate_mask_shape("tissue-mask-a", tissue_a, image_a)
    validate_mask_shape("tissue-mask-b", tissue_b, image_b)
    validate_mask_shape("nuclei-mask-a", nuclei_a, image_a)
    validate_mask_shape("nuclei-mask-b", nuclei_b, image_b)

    with torch.inference_mode():
        z_a, encoding_a = encode_reference(
            vae=vae,
            modules=modules,
            image=image_a,
            tissue_mask=tissue_a,
            nuclei_mask=nuclei_a,
            device=device,
            dtype=dtype,
            use_mask_features=bool(args.use_mask_features),
            reference_spec=reference_spec,
        )
        z_b, encoding_b = encode_reference(
            vae=vae,
            modules=modules,
            image=image_b,
            tissue_mask=tissue_b,
            nuclei_mask=nuclei_b,
            device=device,
            dtype=dtype,
            use_mask_features=bool(args.use_mask_features),
            reference_spec=reference_spec,
        )

    if not isinstance(encoding_a, CrossV4ReferenceEncoding) or not isinstance(
        encoding_b, CrossV4ReferenceEncoding
    ):
        raise TypeError("Expected CrossV4ReferenceEncoding from reference_context_encoder.")

    metrics = build_metrics(
        args=args,
        checkpoint=checkpoint,
        reference_spec=reference_spec,
        selected_a=selected_a,
        selected_b=selected_b,
        z_a=z_a,
        z_b=z_b,
        encoding_a=encoding_a,
        encoding_b=encoding_b,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf8",
        )

    print_summary(metrics)
    if args.output_json:
        print(f"wrote metrics JSON to {args.output_json}")
    return 0


def resolve_dtype_arg(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def resolve_reference_pair(args: argparse.Namespace, *, read_cross_metadata) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.image_a and args.image_b:
        return (
            explicit_reference_record(
                image=args.image_a,
                tissue_mask=args.tissue_mask_a,
                nuclei_mask=args.nuclei_mask_a,
                label="a",
            ),
            explicit_reference_record(
                image=args.image_b,
                tissue_mask=args.tissue_mask_b,
                nuclei_mask=args.nuclei_mask_b,
                label="b",
            ),
        )
    if args.image_a or args.image_b:
        raise ValueError("Provide both --image-a and --image-b, or use --metadata.")
    if not args.metadata:
        raise ValueError("Provide --metadata, or provide both --image-a and --image-b.")

    records = unique_reference_records(read_cross_metadata(args.metadata))
    if len(records) < 2:
        raise ValueError(f"Need at least two unique references in metadata: {args.metadata}")
    if args.random_pair:
        pair = random.Random(args.seed).sample(records, 2)
        return pair[0], pair[1]
    return (
        select_reference_record(
            records,
            sample_id=args.reference_sample_id_a,
            index=int(args.record_index_a),
            name="A",
        ),
        select_reference_record(
            records,
            sample_id=args.reference_sample_id_b,
            index=int(args.record_index_b),
            name="B",
        ),
    )


def explicit_reference_record(
    *,
    image: str,
    tissue_mask: str | None,
    nuclei_mask: str | None,
    label: str,
) -> dict[str, Any]:
    record = {
        "reference_sample_id": Path(image).stem,
        "reference_image": image,
        "source": "cli",
    }
    if tissue_mask is not None:
        record["reference_tissue_mask"] = tissue_mask
    if nuclei_mask is not None:
        record["reference_nuclei_mask"] = nuclei_mask
    missing = [
        key
        for key in ("reference_tissue_mask", "reference_nuclei_mask")
        if key not in record
    ]
    if missing:
        record["missing_mask_note"] = f"explicit image {label} has no masks: {missing}"
    return record


def reference_record_id(record: dict[str, Any]) -> str:
    return str(record.get("reference_sample_id") or Path(record["reference_image"]).stem)


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
        output.append({**record, "reference_sample_id": ref_id, "source": "metadata"})
    return output


def select_reference_record(
    records: list[dict[str, Any]],
    *,
    sample_id: str | None,
    index: int,
    name: str,
) -> dict[str, Any]:
    if sample_id:
        by_id = {reference_record_id(record): record for record in records}
        if sample_id not in by_id:
            raise ValueError(f"reference_sample_id for {name} not found: {sample_id}")
        return by_id[sample_id]
    if index < 0 or index >= len(records):
        raise ValueError(f"record-index-{name.lower()} out of range: {index}; metadata has {len(records)} references")
    return records[index]


def require_mask_paths(record: dict[str, Any], *, name: str) -> None:
    missing = [
        key
        for key in ("reference_tissue_mask", "reference_nuclei_mask")
        if not record.get(key)
    ]
    if missing:
        raise ValueError(
            f"--use-mask-features requires {name} to have metadata mask paths: {missing}. "
            "Use --metadata or pass --tissue-mask-a/b and --nuclei-mask-a/b with explicit images."
        )


def serialize_reference_record(record: dict[str, Any]) -> dict[str, str]:
    keys = (
        "source",
        "sample_id",
        "reference_sample_id",
        "reference_image",
        "reference_tissue_mask",
        "reference_nuclei_mask",
    )
    return {key: str(record.get(key, "")) for key in keys}


def validate_mask_shape(name: str, mask: torch.Tensor, image: torch.Tensor) -> None:
    expected = tuple(int(v) for v in image.shape[1:])
    if tuple(int(v) for v in mask.shape) != expected:
        raise ValueError(f"{name} must have HW shape {expected}, got {tuple(mask.shape)}.")


def encode_reference(
    *,
    vae,
    modules: dict[str, torch.nn.Module],
    image: torch.Tensor,
    tissue_mask: torch.Tensor,
    nuclei_mask: torch.Tensor,
    device: str,
    dtype: torch.dtype,
    use_mask_features: bool,
    reference_spec,
):
    from controlnet_train.inference.pipeline_cross_v4 import _encode_images_to_latents

    image_batch = image.unsqueeze(0).to(device=device)
    tissue_batch = tissue_mask.unsqueeze(0).to(device=device)
    nuclei_batch = nuclei_mask.unsqueeze(0).to(device=device)
    z_ref = _encode_images_to_latents(vae, image_batch, dtype)
    if use_mask_features:
        ref_tissue_feat = modules["tissue_downsampler"](modules["hte"](tissue_batch)).to(dtype=dtype)
        ref_nuclei_feat = modules["nuclei_encoder"](nuclei_batch).to(dtype=dtype)
    else:
        ref_tissue_feat = torch.zeros(
            z_ref.shape[0],
            int(reference_spec.tissue_channels),
            z_ref.shape[2],
            z_ref.shape[3],
            device=z_ref.device,
            dtype=dtype,
        )
        ref_nuclei_feat = torch.zeros(
            z_ref.shape[0],
            int(reference_spec.nuclei_channels),
            z_ref.shape[2],
            z_ref.shape[3],
            device=z_ref.device,
            dtype=dtype,
        )
    encoding = modules["reference_context_encoder"](
        z_ref=z_ref,
        ref_tissue_feat=ref_tissue_feat,
        ref_nuclei_feat=ref_nuclei_feat,
        ref_tissue_ids=tissue_batch,
        ref_nuclei_ids=nuclei_batch,
    )
    return z_ref, encoding


def build_metrics(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    reference_spec,
    selected_a: dict[str, Any],
    selected_b: dict[str, Any],
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    encoding_a,
    encoding_b,
) -> dict[str, Any]:
    local = pair_distance_stats(encoding_a.local_tokens, encoding_b.local_tokens)
    all_tokens = pair_distance_stats(encoding_a.tokens, encoding_b.tokens)
    route = pair_distance_stats(encoding_a.route_anchor_tokens, encoding_b.route_anchor_tokens)
    z = pair_distance_stats(z_a, z_b)
    local_over_z = safe_div(local["relative_l2"], z["relative_l2"])
    all_over_z = safe_div(all_tokens["relative_l2"], z["relative_l2"])
    collapsed = (
        z["relative_l2"] >= float(args.large_z_ref_rel_threshold)
        and local["relative_l2"] <= float(args.collapse_token_rel_threshold)
    )
    return {
        "diagnostic": "cross_v4_ref_projection_collapse",
        "checkpoint": str(checkpoint),
        "pretrained_model_name_or_path": str(args.pretrained_model_name_or_path),
        "metadata": str(args.metadata) if args.metadata else "",
        "reference_a": serialize_reference_record(selected_a),
        "reference_b": serialize_reference_record(selected_b),
        "image_a": str(selected_a["reference_image"]),
        "image_b": str(selected_b["reference_image"]),
        "use_mask_features": bool(args.use_mask_features),
        "thresholds": {
            "large_z_ref_relative_l2": float(args.large_z_ref_rel_threshold),
            "collapsed_context_relative_l2": float(args.collapse_token_rel_threshold),
        },
        "reference_spec": {
            "reference_latent_channels": int(reference_spec.reference_latent_channels),
            "tissue_channels": int(reference_spec.tissue_channels),
            "nuclei_channels": int(reference_spec.nuclei_channels),
            "token_dim": int(reference_spec.token_dim),
            "route_anchor_mode": str(reference_spec.normalized_route_anchor_mode),
            "route_class_count": int(reference_spec.route_class_count),
            "tissue_prior_tokens_per_class": int(reference_spec.tissue_prior_tokens_per_class),
            "cell_prior_tokens_per_class": int(reference_spec.cell_prior_tokens_per_class),
            "global_style_tokens": int(reference_spec.global_style_tokens),
        },
        "shapes": {
            "z_ref": list(z_a.shape),
            "local_tokens": list(encoding_a.local_tokens.shape),
            "route_anchor_tokens": list(encoding_a.route_anchor_tokens.shape),
            "tokens_route_plus_local": list(encoding_a.tokens.shape),
        },
        "z_ref": z,
        "local_context_tokens": local,
        "route_anchor_tokens": route,
        "tokens_route_plus_local": all_tokens,
        "ratios": {
            "local_context_relative_l2_over_z_ref_relative_l2": local_over_z,
            "tokens_route_plus_local_relative_l2_over_z_ref_relative_l2": all_over_z,
        },
        "verdict": (
            "collapse_likely"
            if collapsed
            else "no_collapse_by_threshold"
        ),
        "interpretation": interpret(z, local, local_over_z, collapsed),
    }


def pair_distance_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a32 = a.detach().float().cpu()
    b32 = b.detach().float().cpu()
    if a32.numel() == 0 and b32.numel() == 0:
        return {
            "shape": list(a.shape),
            "numel": 0,
            "l2": 0.0,
            "relative_l2": 0.0,
            "mse": 0.0,
            "mae": 0.0,
            "cosine_similarity": math.nan,
            "a_norm": 0.0,
            "b_norm": 0.0,
            "mean_norm": 0.0,
            "a_std": math.nan,
            "b_std": math.nan,
            "diff_std": math.nan,
            "max_abs": 0.0,
        }
    if tuple(a32.shape) != tuple(b32.shape):
        raise ValueError(f"tensor shapes must match, got {tuple(a32.shape)} vs {tuple(b32.shape)}.")
    diff = a32 - b32
    l2 = float(torch.linalg.vector_norm(diff).item())
    a_norm = float(torch.linalg.vector_norm(a32).item())
    b_norm = float(torch.linalg.vector_norm(b32).item())
    mean_norm = 0.5 * (a_norm + b_norm)
    relative_l2 = l2 / max(mean_norm, 1e-12)
    cosine = torch.nn.functional.cosine_similarity(a32.flatten(), b32.flatten(), dim=0)
    return {
        "shape": list(a32.shape),
        "numel": int(a32.numel()),
        "l2": l2,
        "relative_l2": float(relative_l2),
        "mse": float((diff * diff).mean().item()),
        "mae": float(diff.abs().mean().item()),
        "cosine_similarity": float(cosine.item()),
        "a_norm": a_norm,
        "b_norm": b_norm,
        "mean_norm": float(mean_norm),
        "a_std": safe_std(a32),
        "b_std": safe_std(b32),
        "diff_std": safe_std(diff),
        "max_abs": float(diff.abs().max().item()),
    }


def safe_std(value: torch.Tensor) -> float:
    if value.numel() < 2:
        return 0.0
    return float(value.std(unbiased=False).item())


def safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return math.inf if numerator > 0 else 0.0
    return float(numerator / denominator)


def interpret(z: dict[str, Any], local: dict[str, Any], local_over_z: float, collapsed: bool) -> str:
    if collapsed:
        return (
            "z_ref differs substantially, but projected local reference context tokens are nearly identical; "
            "this supports the hypothesis that the Cross V4 reference encoder/projection path has erased "
            "the A/B reference difference."
        )
    if z["relative_l2"] < 0.10:
        return "z_ref itself is not very different for A/B, so this pair does not isolate projection collapse."
    if local_over_z < 0.25:
        return (
            "Projected local reference context differences are much smaller than z_ref differences, "
            "but they did not cross the collapse threshold."
        )
    return "Projected local reference context tokens retain a non-trivial fraction of the z_ref difference."


def print_summary(metrics: dict[str, Any]) -> None:
    z = metrics["z_ref"]
    local = metrics["local_context_tokens"]
    all_tokens = metrics["tokens_route_plus_local"]
    ratios = metrics["ratios"]
    print("Cross V4 reference projection collapse diagnostic")
    print(f"checkpoint: {metrics['checkpoint']}")
    print(
        "references: "
        f"A={metrics['reference_a']['reference_sample_id']} "
        f"B={metrics['reference_b']['reference_sample_id']}"
    )
    print(f"use_mask_features: {metrics['use_mask_features']}")
    print(
        "z_ref: "
        f"rel_l2={z['relative_l2']:.6f} l2={z['l2']:.6f} "
        f"cos={z['cosine_similarity']:.6f} mse={z['mse']:.8f}"
    )
    print(
        "local context tokens: "
        f"rel_l2={local['relative_l2']:.6f} l2={local['l2']:.6f} "
        f"cos={local['cosine_similarity']:.6f} mse={local['mse']:.8f}"
    )
    print(
        "route+local tokens: "
        f"rel_l2={all_tokens['relative_l2']:.6f} l2={all_tokens['l2']:.6f} "
        f"cos={all_tokens['cosine_similarity']:.6f} mse={all_tokens['mse']:.8f}"
    )
    print(
        "compression: "
        f"local_rel_over_z_rel={ratios['local_context_relative_l2_over_z_ref_relative_l2']:.6f} "
        f"all_rel_over_z_rel={ratios['tokens_route_plus_local_relative_l2_over_z_ref_relative_l2']:.6f}"
    )
    print(f"verdict: {metrics['verdict']}")
    print(f"interpretation: {metrics['interpretation']}")


if __name__ == "__main__":
    raise SystemExit(main())
