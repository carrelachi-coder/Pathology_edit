"""Diagnose reference signal collapse in the Cross V1 IP-Adapter pipeline.

This traces the reference encoding chain:

    reference image -> UNI2-h -> proj_mlp -> Perceiver -> encoder_hid_proj

At each stage, the script compares normal / zero / random reference inputs
against the normal input with cosine similarity and distance metrics.
"""

from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--reference-image", type=str, required=True, help="A real reference patch image.")
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
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args(argv)


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
):
    checkpoint_dir = Path(checkpoint_dir)
    state = torch_load_weights(checkpoint_dir / "phase5_conditioning.pt")
    config = dict(state.get("ref_encoder_config") or {})

    config.setdefault("uni_embed_dim", int(state["ref_encoder_proj_mlp"]["0.weight"].shape[1]))
    config.setdefault("hidden_dim", int(state["ref_encoder_proj_mlp"]["0.weight"].shape[0]))
    config.setdefault("num_tokens", int(state["ref_encoder_latent_queries"].shape[1]))
    config.setdefault(
        "num_perceiver_layers",
        count_ref_perceiver_layers(state["ref_encoder_perceiver_layers"]),
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
    )
    ref_encoder.proj_mlp.load_state_dict(state["ref_encoder_proj_mlp"])
    ref_encoder.load_perceiver_layers_state_dict(state["ref_encoder_perceiver_layers"])
    ref_encoder.latent_queries.data.copy_(
        state["ref_encoder_latent_queries"].to(ref_encoder.latent_queries.device)
    )
    ref_encoder.perceiver_norm.load_state_dict(state["ref_encoder_perceiver_norm"])
    ref_encoder.to(device=device, dtype=dtype)
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


def compare_to_normal(normal, current) -> dict[str, float]:
    left = flatten_tensors(normal)
    right = flatten_tensors(current)
    if left.numel() == 0 or left.numel() != right.numel():
        return {"cosine": math.nan, "l1": math.nan, "rmse": math.nan, "l2": math.nan}
    diff = left - right
    return {
        "cosine": float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item()),
        "l1": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
        "l2": float(torch.linalg.vector_norm(diff).item()),
    }


@torch.no_grad()
def encode_stages(
    ref_encoder,
    encoder_hid_proj: nn.Module | None,
    image: torch.Tensor,
) -> dict[str, Any]:
    stages: dict[str, Any] = {}

    uni = ref_encoder.extract_uni_features(image)
    stages["1_uni"] = uni

    projected = ref_encoder.proj_mlp(uni)
    stages["2_proj_mlp"] = projected

    latents = ref_encoder.latent_queries.expand(projected.shape[0], -1, -1)
    for index, layer in enumerate(ref_encoder.perceiver_layers, start=1):
        latents = layer(latents, projected)
        stages[f"3_perceiver_layer_{index}"] = latents

    resampled = ref_encoder.perceiver_norm(latents)
    stages["4_perceiver_norm"] = resampled

    full = ref_encoder(image)
    stages["5_full_ref_encoder"] = full

    if encoder_hid_proj is not None:
        stages["6_encoder_hid_proj"] = encoder_hid_proj([full])

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


def main(argv=None) -> int:
    args = parse_args(argv)
    dtype = resolve_dtype(args.torch_dtype, args.device)

    print(f"Loading ref_encoder from: {args.checkpoint}")
    ref_encoder, config = load_ref_encoder_from_checkpoint(
        args.checkpoint,
        args.uni_checkpoint_path,
        args.device,
        dtype,
        disable_perceiver_self_attn=bool(args.disable_reference_perceiver_self_attn),
        perceiver_cross_gate_init=args.reference_perceiver_cross_gate_init,
    )
    print(f"Detected ref_encoder config: {json.dumps(config, ensure_ascii=False)}")

    encoder_hid_proj = load_encoder_hid_proj_from_checkpoint(
        args.checkpoint,
        hidden_dim=int(config["hidden_dim"]),
        device=args.device,
        dtype=dtype,
    )
    print(f"encoder_hid_proj loaded: {encoder_hid_proj is not None}")

    print(f"Loading reference image: {args.reference_image}")
    normal_img = load_image_as_tensor(args.reference_image, args.device, dtype)
    print(f"Image shape: {tuple(normal_img.shape)}")

    results = diagnose(ref_encoder, encoder_hid_proj, normal_img)
    print_table(results)
    print_interpretation(results)

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(results, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf8",
        )
        print(f"\nResults saved to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
