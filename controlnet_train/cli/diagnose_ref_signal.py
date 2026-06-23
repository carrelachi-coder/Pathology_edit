"""Diagnose reference signal collapse in the Cross V1 IP-Adapter pipeline.

This traces the reference encoding chain:

    reference image -> UNI2-h -> proj_mlp -> Perceiver -> encoder_hid_proj

At each stage, the script compares normal / zero / random reference inputs
against the normal input with cosine similarity and distance metrics.
If a second reference image is provided, it also runs a pairwise comparison
so you can check whether the projected tokens collapse before the downstream
attention path ever sees them.
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
import torch.nn as nn
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class IPAdapterListProjection(nn.Module):
    """Small local wrapper matching the training-side encoder_hid_proj wrapper."""

    def __init__(self, proj: nn.Module):
        super().__init__()
        self.proj = proj

    def forward(self, image_embeds):
        target_dtype = next(self.proj.parameters()).dtype
        if isinstance(image_embeds, list):
            return [self.proj(embed).to(dtype=target_dtype) for embed in image_embeds]
        return self.proj(image_embeds).to(dtype=target_dtype)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Cross V1 reference signal collapse.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint dir with phase5_conditioning.pt.")
    parser.add_argument("--uni-checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help=(
            "Optional cross metadata JSON/JSONL path. If --reference-image is omitted, "
            "the script picks two distinct refs from metadata automatically."
        ),
    )
    parser.add_argument(
        "--reference-image",
        type=str,
        default=None,
        help="A real reference patch image. Omit when using --metadata auto-selection.",
    )
    parser.add_argument(
        "--reference-image-b",
        type=str,
        default=None,
        help=(
            "Optional second reference image. If omitted with --metadata, the script "
            "picks a second ref automatically."
        ),
    )
    parser.add_argument(
        "--reference-selection-mode",
        choices=("farthest_distance", "random"),
        default="farthest_distance",
        help="How to choose two refs from metadata when explicit image paths are omitted.",
    )
    parser.add_argument(
        "--reference-selection-seed",
        type=int,
        default=42,
        help="Seed used when --reference-selection-mode=random.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--disable-reference-perceiver-self-attn",
        action="store_true",
        help="Temporarily skip reference Perceiver self-attention during this diagnosis.",
    )
    parser.add_argument(
        "--reference-perceiver-cross-gate-init",
        type=float,
        default=None,
        help="Temporarily enable gated Cross-Attn mixing during this diagnosis.",
    )
    parser.add_argument(
        "--skip-reference-perceiver",
        action="store_true",
        help="Temporarily bypass the reference Perceiver during this diagnosis.",
    )
    parser.add_argument(
        "--pairwise-collapse-relative-l2-threshold",
        type=float,
        default=0.02,
        help="Relative-L2 threshold used to call a stage collapsed in pairwise mode.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args(argv)


def read_cross_metadata(path: str | Path) -> list[dict[str, Any]]:
    return read_metadata_records(path)


def read_metadata_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
        if not records:
            raise ValueError(f"metadata file is empty: {path}")
        return records
    return normalize_cross_records(payload)


def normalize_cross_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        records = payload.get("pairs")
        if not isinstance(records, list):
            raise ValueError("cross metadata dict must contain a 'pairs' list")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported cross metadata payload type: {type(payload)!r}")


def reference_record_id(record: dict[str, Any]) -> str:
    if record.get("reference_sample_id"):
        return str(record["reference_sample_id"])
    for key in ("sample_id", "reference_image", "image"):
        value = record.get(key)
        if value:
            return Path(str(value).replace("\\", "/")).stem
    raise KeyError("record is missing both reference_sample_id and reference_image")


def reference_record_label(record: dict[str, Any]) -> str:
    if record.get("reference_sample_id"):
        return str(record["reference_sample_id"])
    if record.get("sample_id"):
        return str(record["sample_id"])
    for key in ("reference_image", "image"):
        value = record.get(key)
        if value:
            return Path(str(value).replace("\\", "/")).stem
    return "reference"


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


def _reference_distance(record: dict[str, Any]) -> float:
    try:
        value = float(
            record.get("distance", record.get("pair_distance", record.get("manhattan_distance", math.nan)))
        )
    except (TypeError, ValueError):
        return -math.inf
    return value if math.isfinite(value) else -math.inf


def select_reference_records(
    records: list[dict[str, Any]],
    *,
    selection_mode: str,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = unique_reference_records(records)
    if len(candidates) < 2:
        raise ValueError("metadata must contain at least two unique reference records")

    if selection_mode == "random":
        return random.Random(seed).sample(candidates, 2)

    if selection_mode == "farthest_distance":
        finite_candidates = [record for record in candidates if math.isfinite(_reference_distance(record))]
        pool = finite_candidates if len(finite_candidates) >= 2 else candidates
        if len(finite_candidates) < 2:
            return random.Random(seed).sample(pool, 2)
        return sorted(
            pool,
            key=lambda record: (
                -_reference_distance(record),
                str(record.get("case_id") or ""),
                reference_record_id(record),
            ),
        )[:2]

    raise ValueError(f"unsupported selection_mode {selection_mode!r}")


def resolve_metadata_path(path_value: str | Path, *, metadata_path: Path | None = None) -> Path:
    candidate = Path(str(path_value).replace("\\", "/"))
    if candidate.is_absolute() or metadata_path is None:
        return candidate
    return metadata_path.parent / candidate


def reference_image_path(record: dict[str, Any], *, metadata_path: Path | None = None) -> Path:
    raw_path = record.get("reference_image") or record.get("image")
    if raw_path is None:
        raise KeyError("record is missing both reference_image and image")
    return resolve_metadata_path(raw_path, metadata_path=metadata_path)


def torch_load_weights(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_dtype(name: str, device: str) -> torch.dtype:
    if "cpu" in str(device).lower():
        return torch.float32
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def count_ref_perceiver_layers(state_dict: dict[str, torch.Tensor]) -> int:
    layer_indices = {
        int(key.split(".", 1)[0])
        for key in state_dict
        if key.split(".", 1)[0].isdigit()
    }
    return max(layer_indices) + 1 if layer_indices else 2


def load_ref_encoder_from_checkpoint(
    checkpoint_dir: str | Path,
    uni_checkpoint_path: str | Path,
    device: str,
    dtype: torch.dtype,
    disable_perceiver_self_attn: bool = False,
    perceiver_cross_gate_init: float | None = None,
    skip_perceiver: bool | None = None,
):
    checkpoint_dir = Path(checkpoint_dir)
    state = torch_load_weights(checkpoint_dir / "phase5_conditioning.pt")
    config = dict(state.get("ref_encoder_config") or {})

    config.setdefault("uni_embed_dim", int(state["ref_encoder_proj_mlp"]["0.weight"].shape[1]))
    config.setdefault("hidden_dim", int(state["ref_encoder_proj_mlp"]["0.weight"].shape[0]))
    config.setdefault("skip_perceiver", False)
    if skip_perceiver is not None:
        config["skip_perceiver"] = bool(skip_perceiver)
    config.setdefault(
        "num_tokens",
        int(state.get("ref_encoder_latent_queries", torch.empty(1, 16)).shape[1]),
    )
    config.setdefault(
        "num_perceiver_layers",
        count_ref_perceiver_layers(state.get("ref_encoder_perceiver_layers", {})),
    )
    config.setdefault("perceiver_heads", 8)
    config.setdefault("use_perceiver_self_attn", True)
    config.setdefault("perceiver_cross_gate_init", None)
    if disable_perceiver_self_attn:
        config["use_perceiver_self_attn"] = False
    if perceiver_cross_gate_init is not None:
        config["perceiver_cross_gate_init"] = float(perceiver_cross_gate_init)

    from controlnet_train.modules.reference_image_encoder import ReferenceImageEncoder

    ref_encoder = ReferenceImageEncoder(
        uni_checkpoint_path=uni_checkpoint_path,
        uni_embed_dim=int(config["uni_embed_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_tokens=int(config["num_tokens"]),
        num_perceiver_layers=int(config["num_perceiver_layers"]),
        perceiver_heads=int(config["perceiver_heads"]),
        use_perceiver_self_attn=bool(config["use_perceiver_self_attn"]),
        perceiver_cross_gate_init=(
            None
            if config["perceiver_cross_gate_init"] is None
            else float(config["perceiver_cross_gate_init"])
        ),
        skip_perceiver=bool(config["skip_perceiver"]),
    )
    ref_encoder.proj_mlp.load_state_dict(state["ref_encoder_proj_mlp"])
    if not ref_encoder.skip_perceiver:
        ref_encoder.load_perceiver_layers_state_dict(state["ref_encoder_perceiver_layers"])
        ref_encoder.latent_queries.data.copy_(
            state["ref_encoder_latent_queries"].to(ref_encoder.latent_queries.device)
        )
        ref_encoder.perceiver_norm.load_state_dict(state["ref_encoder_perceiver_norm"])
    ref_encoder.to(device=device)
    ref_encoder.proj_mlp.to(device=device, dtype=dtype)
    ref_encoder.perceiver_layers.to(device=device, dtype=dtype)
    ref_encoder.perceiver_norm.to(device=device, dtype=dtype)
    ref_encoder.latent_queries.data = ref_encoder.latent_queries.data.to(
        device=device,
        dtype=dtype,
    )
    ref_encoder.uni.to(device=device, dtype=torch.float32)
    ref_encoder.eval()
    return ref_encoder, config


def load_encoder_hid_proj_from_checkpoint(
    checkpoint_dir: str | Path,
    hidden_dim: int,
    device: str,
    dtype: torch.dtype,
) -> nn.Module | None:
    ip_path = Path(checkpoint_dir) / "phase5_ip_adapter.pt"
    if not ip_path.exists():
        return None

    from diffusers.models.embeddings import IPAdapterFullImageProjection

    state = torch_load_weights(ip_path)
    encoder_state = state.get("encoder_hid_proj")
    if encoder_state is None:
        return None

    raw_proj = IPAdapterFullImageProjection(
        image_embed_dim=hidden_dim,
        cross_attention_dim=hidden_dim,
    )
    proj = IPAdapterListProjection(raw_proj)
    proj.load_state_dict(encoder_state)
    proj.to(device=device, dtype=dtype)
    proj.eval()
    return proj


def load_image_as_tensor(image_path: str | Path, device: str, dtype: torch.dtype) -> torch.Tensor:
    from controlnet_train.data.common import load_image_tensor

    return load_image_tensor(image_path).unsqueeze(0).to(device=device, dtype=dtype)


def flatten_tensors(value) -> torch.Tensor:
    if torch.is_tensor(value):
        tensors = [value]
    elif isinstance(value, (list, tuple)):
        tensors = list(value)
    else:
        raise TypeError(f"Cannot flatten value of type {type(value)!r}.")
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([tensor.detach().float().reshape(-1).cpu() for tensor in tensors], dim=0)


def tensor_l2_norm(value) -> float:
    flat = flatten_tensors(value)
    if flat.numel() == 0:
        return math.nan
    return float(torch.linalg.vector_norm(flat).item())


def tensor_pair_metrics(left, right) -> dict[str, float]:
    left_flat = flatten_tensors(left)
    right_flat = flatten_tensors(right)
    if left_flat.numel() == 0 or right_flat.numel() == 0 or left_flat.numel() != right_flat.numel():
        return {
            "left_norm": math.nan,
            "right_norm": math.nan,
            "cosine": math.nan,
            "l1": math.nan,
            "rmse": math.nan,
            "l2": math.nan,
            "relative_l2": math.nan,
        }

    diff = left_flat - right_flat
    left_norm = float(torch.linalg.vector_norm(left_flat).item())
    right_norm = float(torch.linalg.vector_norm(right_flat).item())
    l2 = float(torch.linalg.vector_norm(diff).item())
    denom = max((left_norm + right_norm) * 0.5, 1e-12)
    return {
        "left_norm": left_norm,
        "right_norm": right_norm,
        "cosine": float(F.cosine_similarity(left_flat.unsqueeze(0), right_flat.unsqueeze(0)).item()),
        "l1": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "l2": l2,
        "relative_l2": float(l2 / denom),
    }


def compare_to_normal(normal, current) -> dict[str, float]:
    pair_metrics = tensor_pair_metrics(normal, current)
    return {key: pair_metrics[key] for key in ("cosine", "l1", "rmse", "l2")}


@torch.no_grad()
def encode_stages(
    ref_encoder,
    encoder_hid_proj: nn.Module | None,
    image: torch.Tensor,
) -> dict[str, Any]:
    stages: dict[str, Any] = {}

    uni = ref_encoder.extract_uni_features(image)
    stages["1_uni"] = uni

    proj_dtype = next(ref_encoder.proj_mlp.parameters()).dtype
    uni = uni.to(dtype=proj_dtype)
    projected = ref_encoder.proj_mlp(uni)
    stages["2_proj_mlp"] = projected

    if not ref_encoder.skip_perceiver:
        latents = ref_encoder.latent_queries.expand(projected.shape[0], -1, -1)
        for index, layer in enumerate(ref_encoder.perceiver_layers, start=1):
            latents = layer(latents, projected)
            stages[f"3_perceiver_layer_{index}"] = latents

        resampled = ref_encoder.perceiver_norm(latents)
        stages["4_perceiver_norm"] = resampled

    full = ref_encoder(image)
    stages["5_full_ref_encoder"] = full

    if encoder_hid_proj is not None:
        gate = ref_encoder.reference_presence_gate(
            image,
            device=full.device,
            dtype=full.dtype,
        )
        stages["6_encoder_hid_proj"] = [
            tensor * gate.to(device=tensor.device, dtype=tensor.dtype)
            for tensor in encoder_hid_proj([full])
        ]

    return stages


@torch.no_grad()
def diagnose(
    ref_encoder,
    encoder_hid_proj: nn.Module | None,
    normal_img: torch.Tensor,
) -> dict[str, Any]:
    variants = {
        "normal": normal_img,
        "zero": torch.zeros_like(normal_img),
        "random_noise": torch.rand_like(normal_img),
    }
    encoded = {
        name: encode_stages(ref_encoder, encoder_hid_proj, image)
        for name, image in variants.items()
    }

    results: dict[str, Any] = {}
    stage_names = list(encoded["normal"].keys())
    for stage_name in stage_names:
        normal_stage = encoded["normal"][stage_name]
        metrics: dict[str, Any] = {
            "normal_norm": tensor_l2_norm(normal_stage),
            "numel": int(flatten_tensors(normal_stage).numel()),
        }
        for variant_name in ("zero", "random_noise"):
            metrics[f"vs_{variant_name}"] = compare_to_normal(
                normal_stage,
                encoded[variant_name][stage_name],
            )
        results[stage_name] = metrics

    return results


@torch.no_grad()
def diagnose_pairwise(
    ref_encoder,
    encoder_hid_proj: nn.Module | None,
    reference_img_a: torch.Tensor,
    reference_img_b: torch.Tensor,
) -> dict[str, Any]:
    encoded_a = encode_stages(ref_encoder, encoder_hid_proj, reference_img_a)
    encoded_b = encode_stages(ref_encoder, encoder_hid_proj, reference_img_b)

    results: dict[str, Any] = {}
    for stage_name in encoded_a.keys():
        results[stage_name] = tensor_pair_metrics(encoded_a[stage_name], encoded_b[stage_name])
    return results


def print_table(results: dict[str, Any]) -> None:
    print(
        f"{'Stage':<28} {'norm':>10} {'zero_cos':>10} {'rand_cos':>10} "
        f"{'zero_l2':>10} {'rand_l2':>10}"
    )
    print("-" * 84)
    for stage_name, metrics in results.items():
        zero = metrics["vs_zero"]
        random_noise = metrics["vs_random_noise"]
        print(
            f"{stage_name:<28} "
            f"{metrics['normal_norm']:>10.2f} "
            f"{zero['cosine']:>10.6f} "
            f"{random_noise['cosine']:>10.6f} "
            f"{zero['l2']:>10.2f} "
            f"{random_noise['l2']:>10.2f}"
        )


def print_pairwise_table(results: dict[str, Any], *, label_a: str, label_b: str) -> None:
    print(f"Pairwise comparison: {label_a} vs {label_b}")
    print(
        f"{'Stage':<28} {'rel_l2':>10} {'l2':>10} {'cosine':>10} "
        f"{'a_norm':>10} {'b_norm':>10}"
    )
    print("-" * 84)
    for stage_name, metrics in results.items():
        print(
            f"{stage_name:<28} "
            f"{metrics['relative_l2']:>10.6f} "
            f"{metrics['l2']:>10.2f} "
            f"{metrics['cosine']:>10.6f} "
            f"{metrics['left_norm']:>10.2f} "
            f"{metrics['right_norm']:>10.2f}"
        )


def print_interpretation(results: dict[str, Any]) -> None:
    print("\n== Interpretation ==")
    for stage_name, metrics in results.items():
        cos_zero = metrics["vs_zero"]["cosine"]
        if cos_zero > 0.95:
            status = "COLLAPSED/VERY WEAK"
        elif cos_zero > 0.8:
            status = "WEAK"
        else:
            status = "SEPARATED"
        print(f"  {stage_name}: {status} vs zero, cosine={cos_zero:.4f}")

    stage_names = list(results.keys())
    for index in range(1, len(stage_names)):
        prev_name = stage_names[index - 1]
        curr_name = stage_names[index]
        prev = results[prev_name]["vs_zero"]["cosine"]
        curr = results[curr_name]["vs_zero"]["cosine"]
        if prev < 0.8 and curr > 0.9:
            print(
                f"\n  >>> Signal collapse likely happens between {prev_name} and {curr_name} "
                f"(cosine {prev:.4f} -> {curr:.4f})"
            )


def pairwise_verdict(
    results: dict[str, Any],
    *,
    collapse_relative_l2_threshold: float,
) -> tuple[str, str | None]:
    stage_order = (
        "2_proj_mlp",
        "3_perceiver_layer_1",
        "4_perceiver_norm",
        "5_full_ref_encoder",
        "6_encoder_hid_proj",
    )

    proj_metrics = results.get("2_proj_mlp")
    if proj_metrics and _stage_is_collapsed(proj_metrics, collapse_relative_l2_threshold):
        return "proj_collapsed", None

    for stage_name in stage_order[1:]:
        metrics = results.get(stage_name)
        if metrics and _stage_is_collapsed(metrics, collapse_relative_l2_threshold):
            return "downstream_collapsed", stage_name

    return "proj_informative", None


def _stage_is_collapsed(metrics: dict[str, Any], collapse_relative_l2_threshold: float) -> bool:
    relative_l2 = float(metrics.get("relative_l2", math.nan))
    cosine = float(metrics.get("cosine", math.nan))
    if not math.isfinite(relative_l2):
        return False
    if relative_l2 <= collapse_relative_l2_threshold:
        return True
    return math.isfinite(cosine) and cosine >= 0.999


def print_pairwise_interpretation(
    results: dict[str, Any],
    *,
    collapse_relative_l2_threshold: float,
) -> None:
    print("\n== Interpretation ==")
    for stage_name in (
        "2_proj_mlp",
        "3_perceiver_layer_1",
        "4_perceiver_norm",
        "5_full_ref_encoder",
        "6_encoder_hid_proj",
    ):
        metrics = results.get(stage_name)
        if not metrics:
            continue
        status = "COLLAPSED" if _stage_is_collapsed(metrics, collapse_relative_l2_threshold) else "SEPARATED"
        print(
            f"  {stage_name}: {status} "
            f"rel_l2={float(metrics['relative_l2']):.6f} cosine={float(metrics['cosine']):.6f}"
        )

    verdict, stage_name = pairwise_verdict(
        results,
        collapse_relative_l2_threshold=collapse_relative_l2_threshold,
    )
    if verdict == "proj_collapsed":
        print(
            "  >>> proj_mlp is effectively constant across the two refs. "
            "Check proj_mlp/soft_bias; this usually means the projection path has "
            "degenerated and likely needs lower bias + retraining."
        )
    elif verdict == "downstream_collapsed":
        downstream_label = stage_name or "downstream"
        print(
            f"  >>> proj_mlp separates the refs, but {downstream_label} collapses them. "
            "The projected token still carries signal; the loss is likely downstream "
            "(attention / soft_bias / routing)."
        )
    else:
        print(
            "  >>> proj_mlp remains separated across the two refs. The injected token "
            "has signal; if outputs still ignore the reference difference, focus on "
            "downstream attention / soft_bias."
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    dtype = resolve_dtype(args.torch_dtype, args.device)
    metadata_path = Path(args.metadata) if args.metadata else None

    print(f"Loading ref_encoder from: {args.checkpoint}")
    ref_encoder, config = load_ref_encoder_from_checkpoint(
        args.checkpoint,
        args.uni_checkpoint_path,
        args.device,
        dtype,
        disable_perceiver_self_attn=bool(args.disable_reference_perceiver_self_attn),
        perceiver_cross_gate_init=args.reference_perceiver_cross_gate_init,
        skip_perceiver=True if args.skip_reference_perceiver else None,
    )
    print(f"Detected ref_encoder config: {json.dumps(config, ensure_ascii=False)}")

    encoder_hid_proj = load_encoder_hid_proj_from_checkpoint(
        args.checkpoint,
        hidden_dim=int(config["hidden_dim"]),
        device=args.device,
        dtype=dtype,
    )
    print(f"encoder_hid_proj loaded: {encoder_hid_proj is not None}")

    if args.reference_image is None:
        if metadata_path is None:
            raise ValueError("--metadata is required when --reference-image is omitted.")
        metadata_records = read_metadata_records(metadata_path)
        selected_records = select_reference_records(
            metadata_records,
            selection_mode=args.reference_selection_mode,
            seed=args.reference_selection_seed,
        )
        reference_record_a, reference_record_b = selected_records
        reference_image_a = reference_image_path(reference_record_a, metadata_path=metadata_path)
        reference_image_b = reference_image_path(reference_record_b, metadata_path=metadata_path)
        label_a = reference_record_label(reference_record_a)
        label_b = reference_record_label(reference_record_b)
        print(
            "Auto-selected refs from metadata: "
            f"A={label_a} -> {reference_image_a} | B={label_b} -> {reference_image_b}"
        )
        normal_img = load_image_as_tensor(reference_image_a, args.device, dtype)
        print(f"Image A shape: {tuple(normal_img.shape)}")
        print(f"Loading second reference image: {reference_image_b}")
        reference_img_b = load_image_as_tensor(reference_image_b, args.device, dtype)
        print(f"Image B shape: {tuple(reference_img_b.shape)}")
        results = diagnose_pairwise(ref_encoder, encoder_hid_proj, normal_img, reference_img_b)
        print_pairwise_table(results, label_a=label_a, label_b=label_b)
        print_pairwise_interpretation(
            results,
            collapse_relative_l2_threshold=float(args.pairwise_collapse_relative_l2_threshold),
        )
    else:
        print(f"Loading reference image: {args.reference_image}")
        normal_img = load_image_as_tensor(args.reference_image, args.device, dtype)
        print(f"Image A shape: {tuple(normal_img.shape)}")
        results = diagnose(ref_encoder, encoder_hid_proj, normal_img)
        print_table(results)
        print_interpretation(results)

    if args.output_json:
        payload: dict[str, Any]
        if args.reference_image is None:
            payload = {
                "mode": "pairwise",
                "reference_image_a": str(reference_image_a),
                "reference_image_b": str(reference_image_b),
                "reference_sample_id_a": label_a,
                "reference_sample_id_b": label_b,
                "selection_mode": args.reference_selection_mode,
                "selection_seed": int(args.reference_selection_seed),
                "collapse_relative_l2_threshold": float(args.pairwise_collapse_relative_l2_threshold),
                "stage_results": results,
            }
        else:
            payload = results
        Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf8")
        print(f"\nResults saved to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
