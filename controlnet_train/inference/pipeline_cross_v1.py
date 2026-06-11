from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from controlnet_train.data.common import (
    default_prompt_for_dataset,
    load_image_tensor,
    load_nuclei_mask,
    load_tissue_mask,
)
from controlnet_train.modules import (
    HierarchicalTissueEmbedding,
    NucleiConditionEncoder,
    TissueConditionDownsampler,
)
from controlnet_train.modules.cross_v1_conditioning import (
    CROSS_V1_SPATIAL_REFERENCE_TARGET,
    CROSS_V1_SPATIAL_REFERENCE_TARGET_DELTA,
    CrossV1ControlSpec,
    build_cross_v1_condition,
)
from controlnet_train.training.conditioning import patch_controlnet_x_embedder
from controlnet_train.inference.pipeline_cross_v2_1 import (
    _build_mask_change_map,
    _build_packed_change_gate,
    _pack_source_latents_for_sampling,
    _prepare_source_noised_latents,
    _sigma_for_timestep,
    _source_init_timesteps,
    _validate_nonnegative_float,
    _validate_source_latent_init_strength,
)
from controlnet_train.modules.cross_v2_1_conditioning import deterministic_latent_from_posterior
from controlnet_train.modules.reference_image_encoder import (
    build_region_ip_token_labels,
    normalize_region_ip_label_mode,
    normalize_region_ip_token_mode,
    resize_mask_to_token_labels,
)


@dataclass
class CrossV1InferenceBundle:
    pretrained_model_name_or_path: str | Path
    checkpoint_path: Path
    uni_checkpoint_path: str | Path
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    controlnet_conditioning_scale: float = 1.0
    ip_adapter_scale: float = 1.0
    flux_pipeline: object | None = None
    controlnet: object | None = None
    condition_modules: dict[str, nn.Module] = field(default_factory=dict)
    control_spec: CrossV1ControlSpec = field(default_factory=CrossV1ControlSpec)
    ip_adapter_modules: dict[str, nn.Module] = field(default_factory=dict)
    ref_encoder: ReferenceImageEncoder | None = None
    regional_ip_adapter: bool = False
    regional_ip_strict: bool = True
    regional_ip_token_mode: str = "spatial"
    regional_ip_label_mode: str = "tissue"


def load_cross_v1_bundle(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: str | Path,
    uni_checkpoint_path: str | Path,
    device: str = "cuda",
    torch_dtype: torch.dtype | None = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    controlnet_conditioning_scale: float = 1.0,
    ip_adapter_scale: float = 1.0,
) -> CrossV1InferenceBundle:
    device = _resolve_device(device)
    dtype = _resolve_torch_dtype(torch_dtype, device)
    checkpoint = _validate_checkpoint_dir(checkpoint_path)
    control_spec = _load_cross_v1_control_spec(checkpoint)
    ref_encoder_config = _load_ref_encoder_config(checkpoint)
    from controlnet_train.training.flux_phase5_cross_v1 import (
        _collect_ip_adapter_modules,
        install_flux_ip_adapter_attention,
        patch_flux_single_ip_forward,
    )

    # Load Flux ControlNet pipeline with the checkpoint's V1 spatial layout.
    pipe, controlnet = _load_flux_controlnet_pipeline(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        packed_channels=control_spec.packed_channels,
        device=device,
        torch_dtype=dtype,
    )

    # Load IP-Adapter weights from checkpoint
    ip_state = _torch_load_weights(checkpoint / "phase5_ip_adapter.pt")
    regional_ip_adapter = bool(ip_state.get("regional_ip_adapter", False))
    regional_ip_token_mode = normalize_region_ip_token_mode(
        ip_state.get(
            "regional_ip_token_mode",
            ref_encoder_config.get("regional_ip_token_mode", "spatial"),
        )
    )
    regional_ip_label_mode = normalize_region_ip_label_mode(
        ip_state.get(
            "regional_ip_label_mode",
            ref_encoder_config.get("regional_ip_label_mode", "tissue"),
        )
    )
    install_flux_ip_adapter_attention(
        pipe.transformer,
        num_tokens=int(ip_state.get("num_tokens", ref_encoder_config.get("num_output_tokens", ref_encoder_config["num_tokens"]))),
        num_single_layers=_resolve_saved_single_ip_layer_count(ip_state),
        regional=regional_ip_adapter,
    )
    patch_flux_single_ip_forward(pipe.transformer)
    pipe.transformer.encoder_hid_proj.load_state_dict(ip_state["encoder_hid_proj"])
    for i, block in enumerate(pipe.transformer.transformer_blocks):
        block.attn.processor.to_k_ip.load_state_dict(ip_state[f"block_{i}_to_k_ip"])
        block.attn.processor.to_v_ip.load_state_dict(ip_state[f"block_{i}_to_v_ip"])
        null_key = f"block_{i}_ip_null_tokens"
        if null_key in ip_state and hasattr(block.attn.processor, "ip_null_tokens"):
            block.attn.processor.ip_null_tokens.load_state_dict(ip_state[null_key])
    for i, block in enumerate(getattr(pipe.transformer, "single_transformer_blocks", [])):
        k_key = f"single_block_{i}_to_k_ip"
        v_key = f"single_block_{i}_to_v_ip"
        null_key = f"single_block_{i}_ip_null_tokens"
        if k_key in ip_state and v_key in ip_state:
            block.attn.processor.to_k_ip.load_state_dict(ip_state[k_key])
            block.attn.processor.to_v_ip.load_state_dict(ip_state[v_key])
            if null_key in ip_state and hasattr(block.attn.processor, "ip_null_tokens"):
                block.attn.processor.ip_null_tokens.load_state_dict(ip_state[null_key])
    _move_ip_adapter_modules(pipe.transformer, device=device, torch_dtype=dtype)
    set_ip_adapter_scale(pipe.transformer, ip_adapter_scale)

    ip_adapter_modules = _collect_ip_adapter_modules(pipe.transformer)

    # Load conditioning modules (hte, tissue_downsampler, nuclei_encoder, ref_encoder)
    modules = _load_condition_modules(
        checkpoint_path=checkpoint,
        uni_checkpoint_path=uni_checkpoint_path,
        device=device,
        torch_dtype=dtype,
        ref_encoder_config=ref_encoder_config,
    )

    return CrossV1InferenceBundle(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        checkpoint_path=checkpoint,
        uni_checkpoint_path=uni_checkpoint_path,
        device=device,
        torch_dtype=dtype,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        ip_adapter_scale=float(ip_adapter_scale),
        flux_pipeline=pipe,
        controlnet=controlnet,
        condition_modules=modules,
        control_spec=control_spec,
        ip_adapter_modules=ip_adapter_modules,
        ref_encoder=modules["ref_encoder"],
        regional_ip_adapter=regional_ip_adapter,
        regional_ip_token_mode=regional_ip_token_mode,
        regional_ip_label_mode=regional_ip_label_mode,
    )


def _resolve_saved_single_ip_layer_count(ip_state: dict[str, Any]) -> int:
    if "num_single_layers" in ip_state:
        return int(ip_state["num_single_layers"])
    indices = {
        int(key.split("_")[2])
        for key in ip_state
        if key.startswith("single_block_") and key.endswith(("_to_k_ip", "_to_v_ip"))
    }
    return len(indices)


def set_ip_adapter_scale(transformer: nn.Module, scale: float) -> None:
    """Set IP-Adapter scale on both double-stream and installed single-stream processors."""
    for blocks in (
        getattr(transformer, "transformer_blocks", []),
        getattr(transformer, "single_transformer_blocks", []),
    ):
        for block in blocks:
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


def _packed_flux_image_token_count(image: torch.Tensor, pipe) -> int:
    height, width = (int(v) for v in image.shape[-2:])
    vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8) or 8)
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    return (latent_height // 2) * (latent_width // 2)


def _tissue_fallback_region_labels(labels: torch.Tensor, *, label_mode: str) -> torch.Tensor:
    """Map exact IP labels back to tissue labels for strict fallback matching."""
    label_mode = normalize_region_ip_label_mode(label_mode)
    labels = labels.to(dtype=torch.long)
    if label_mode == "tissue":
        return labels
    fallback = torch.full_like(labels, -1)
    valid = labels >= 0
    fallback[valid] = labels[valid] // 256
    return fallback


@torch.inference_mode()
def run_cross_v1_bundle(
    bundle: CrossV1InferenceBundle,
    reference_image: torch.Tensor,
    reference_tissue_mask: torch.Tensor,
    reference_nuclei_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    target_nuclei_mask: torch.Tensor,
    prompt: str,
    source_latent_init_strength: float = 0.0,
    mask_chord_scale: float = 0.0,
    mask_chord_use_gate: bool = False,
    mask_chord_gate_dilate_radius: int = 0,
    mask_chord_gate_feather_radius: int = 0,
    mask_chord_gate_outside_scale: float = 0.0,
    seed: int = 42,
) -> Image.Image:
    source_latent_init_strength = _validate_source_latent_init_strength(source_latent_init_strength)
    mask_chord_scale = _validate_nonnegative_float(mask_chord_scale, "mask_chord_scale")
    # Encode reference image via UNI2-h + Perceiver resampler
    reference_batch = reference_image.unsqueeze(0).to(device=bundle.device, dtype=bundle.torch_dtype)
    if bundle.regional_ip_adapter:
        reference_tissue_batch = reference_tissue_mask.unsqueeze(0).to(device=bundle.device)
        reference_nuclei_batch = reference_nuclei_mask.unsqueeze(0).to(device=bundle.device)
        target_tissue_batch = target_tissue_mask.unsqueeze(0).to(device=bundle.device)
        target_nuclei_batch = target_nuclei_mask.unsqueeze(0).to(device=bundle.device)
        ref_features, region_token_labels = bundle.ref_encoder.encode_region_ip_tokens(
            reference_batch,
            reference_tissue_batch,
            nuclei_mask=reference_nuclei_batch,
            token_mode=bundle.regional_ip_token_mode,
            label_mode=bundle.regional_ip_label_mode,
        )
        query_token_count = _packed_flux_image_token_count(reference_image, bundle.flux_pipeline)
        query_region_labels = build_region_ip_token_labels(
            tissue_mask=target_tissue_batch,
            num_tokens=query_token_count,
            nuclei_mask=target_nuclei_batch,
            label_mode=bundle.regional_ip_label_mode,
        ).to(device=bundle.device)
        key_fallback_region_labels = _tissue_fallback_region_labels(
            region_token_labels,
            label_mode=bundle.regional_ip_label_mode,
        ).to(device=bundle.device)
        query_fallback_region_labels = resize_mask_to_token_labels(
            target_tissue_batch,
            query_token_count,
        ).to(device=bundle.device)
    else:
        ref_features = bundle.ref_encoder(reference_batch)
        region_token_labels = None
        query_region_labels = None
        key_fallback_region_labels = None
        query_fallback_region_labels = None
    ref_features = ref_features.to(device=bundle.device)
    ref_gate = bundle.ref_encoder.reference_presence_gate(
        reference_batch,
        device=bundle.device,
        dtype=next(bundle.flux_pipeline.transformer.encoder_hid_proj.parameters()).dtype,
    )
    ip_hidden_states = bundle.flux_pipeline.transformer.encoder_hid_proj([ref_features])
    ip_hidden_states = [
        hidden.to(device=bundle.device) * ref_gate.to(device=bundle.device, dtype=hidden.dtype)
        for hidden in ip_hidden_states
    ]

    # Build spatial control tensor (no reference_image_latent)
    target_tissue_feat = bundle.condition_modules["tissue_downsampler"](
        bundle.condition_modules["hte"](
            target_tissue_mask.unsqueeze(0).to(device=bundle.device)
        )
    ).to(dtype=bundle.torch_dtype)
    target_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
        target_nuclei_mask.unsqueeze(0).to(device=bundle.device)
    ).to(dtype=bundle.torch_dtype)
    reference_tissue_feat = None
    reference_nuclei_feat = None
    needs_reference_mask_features = (
        mask_chord_scale > 0.0
        or bundle.control_spec.spatial_mode in {
            CROSS_V1_SPATIAL_REFERENCE_TARGET,
            CROSS_V1_SPATIAL_REFERENCE_TARGET_DELTA,
        }
    )
    if needs_reference_mask_features:
        reference_tissue_feat = bundle.condition_modules["tissue_downsampler"](
            bundle.condition_modules["hte"](
                reference_tissue_mask.unsqueeze(0).to(device=bundle.device)
            )
        ).to(dtype=bundle.torch_dtype)
        reference_nuclei_feat = bundle.condition_modules["nuclei_encoder"](
            reference_nuclei_mask.unsqueeze(0).to(device=bundle.device)
        ).to(dtype=bundle.torch_dtype)

    control_tensor = build_cross_v1_condition(
        reference_tissue_feat=reference_tissue_feat,
        reference_nuclei_feat=reference_nuclei_feat,
        target_tissue_feat=target_tissue_feat,
        target_nuclei_feat=target_nuclei_feat,
        spatial_mode=bundle.control_spec.spatial_mode,
    )
    source_control_tensor = None
    if mask_chord_scale > 0.0:
        if reference_tissue_feat is None or reference_nuclei_feat is None:
            raise ValueError("Cross V1 mask chord guidance requires reference mask features.")
        source_control_tensor = build_cross_v1_condition(
            reference_tissue_feat=reference_tissue_feat,
            reference_nuclei_feat=reference_nuclei_feat,
            target_tissue_feat=reference_tissue_feat,
            target_nuclei_feat=reference_nuclei_feat,
            spatial_mode=bundle.control_spec.spatial_mode,
        )

    change_mask = None
    if mask_chord_use_gate:
        change_mask = _build_mask_change_map(
            reference_tissue_mask=reference_tissue_mask,
            reference_nuclei_mask=reference_nuclei_mask,
            target_tissue_mask=target_tissue_mask,
            target_nuclei_mask=target_nuclei_mask,
        )

    source_latents = None
    if source_latent_init_strength > 0.0:
        source_latents = _encode_images_to_latents(
            bundle.flux_pipeline.vae,
            reference_image.unsqueeze(0),
            bundle.torch_dtype,
        )

    output_size = tuple(int(v) for v in reference_image.shape[1:])
    joint_attention_kwargs = {"ip_hidden_states": ip_hidden_states}
    if bundle.regional_ip_adapter:
        joint_attention_kwargs.update(
            {
                "ip_adapter_masks": {
                    "key_region_labels": region_token_labels.to(device=bundle.device),
                    "query_region_labels": query_region_labels,
                    "key_fallback_region_labels": key_fallback_region_labels,
                    "query_fallback_region_labels": query_fallback_region_labels,
                    "strict": bool(bundle.regional_ip_strict),
                },
            }
        )

    return _sample_with_flux_controlnet(
        pipe=bundle.flux_pipeline,
        controlnet=bundle.controlnet,
        prompt=prompt,
        control_tensor=control_tensor,
        source_control_tensor=source_control_tensor,
        source_latents=source_latents,
        source_latent_init_strength=source_latent_init_strength,
        mask_chord_scale=mask_chord_scale,
        mask_chord_change_mask=change_mask,
        mask_chord_gate_dilate_radius=mask_chord_gate_dilate_radius,
        mask_chord_gate_feather_radius=mask_chord_gate_feather_radius,
        mask_chord_gate_outside_scale=mask_chord_gate_outside_scale,
        output_size=output_size,
        device=bundle.device,
        torch_dtype=bundle.torch_dtype,
        num_inference_steps=bundle.num_inference_steps,
        guidance_scale=bundle.guidance_scale,
        controlnet_conditioning_scale=bundle.controlnet_conditioning_scale,
        joint_attention_kwargs=joint_attention_kwargs,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Internal helpers (adapted from pipeline.py for independence)
# ---------------------------------------------------------------------------

def _resolve_torch_dtype(torch_dtype: torch.dtype | None, device: str) -> torch.dtype:
    if torch_dtype is not None:
        return torch_dtype
    return torch.bfloat16 if "cuda" in str(device).lower() else torch.float32


def _resolve_device(device: str | torch.device | None) -> str:
    value = str(device or "cuda").strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return value
    if value == "cuda":
        _validate_cuda_device(value, index=0)
        return value
    if value.startswith("cuda:"):
        try:
            index = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device {device!r}; expected cuda or cuda:<index>.") from exc
        _validate_cuda_device(value, index=index)
        return value
    raise ValueError(f"Unsupported device {device!r}; choose auto, cpu, cuda, or cuda:<index>.")


def _validate_cuda_device(device: str, *, index: int) -> None:
    if index < 0:
        raise ValueError(f"Invalid CUDA device {device!r}; CUDA index must be non-negative.")
    if not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device!r} was requested, but CUDA is not available.")
    visible_count = torch.cuda.device_count()
    if index >= visible_count:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_msg = f" CUDA_VISIBLE_DEVICES={visible!r}." if visible is not None else ""
        raise ValueError(
            f"CUDA device {device!r} is not visible to this process; "
            f"torch sees {visible_count} CUDA device(s).{visible_msg} "
            "Use 'cuda'/'cuda:0' for the first visible GPU, or adjust CUDA_VISIBLE_DEVICES."
        )


def _validate_checkpoint_dir(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint}")
    if not (checkpoint / "config.json").exists():
        raise FileNotFoundError(
            "Missing ControlNet config.json under checkpoint path: "
            f"{checkpoint}. This usually means the directory is an older "
            "accelerate resume-only checkpoint. Run diagnose/eval against the "
            "final output directory, or a newly saved checkpoint that includes "
            "eval-ready Cross V1 artifacts."
        )
    if not (checkpoint / "phase5_conditioning.pt").exists():
        raise FileNotFoundError(f"Missing phase5_conditioning.pt under checkpoint path: {checkpoint}")
    return checkpoint


def _move_ip_adapter_modules(transformer: nn.Module, *, device: str, torch_dtype: torch.dtype) -> None:
    train_dtype = torch.float32
    if hasattr(transformer, "encoder_hid_proj"):
        transformer.encoder_hid_proj.to(device=device, dtype=train_dtype)
    for blocks in (
        getattr(transformer, "transformer_blocks", []),
        getattr(transformer, "single_transformer_blocks", []),
    ):
        for block in blocks:
            processor = getattr(getattr(block, "attn", None), "processor", None)
            for name in ("to_k_ip", "to_v_ip", "ip_null_tokens"):
                module = getattr(processor, name, None)
                if module is not None:
                    module.to(device=device, dtype=train_dtype)


def _load_flux_controlnet_pipeline(
    *,
    pretrained_model_name_or_path: str | Path,
    checkpoint_path: Path,
    packed_channels: int,
    device: str,
    torch_dtype: torch.dtype,
) -> tuple:
    from diffusers import FluxControlNetModel, FluxControlNetPipeline

    controlnet_config = FluxControlNetModel.load_config(checkpoint_path)
    controlnet = FluxControlNetModel.from_config(controlnet_config)
    patch_controlnet_x_embedder(controlnet, packed_channels)
    controlnet.load_state_dict(_load_diffusers_model_state_dict(checkpoint_path), strict=True)
    controlnet.to(dtype=torch_dtype)

    pipe = FluxControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path, controlnet=controlnet, torch_dtype=torch_dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, controlnet


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


def _load_ref_encoder_config(checkpoint_path: Path) -> dict[str, Any]:
    state = _torch_load_weights(checkpoint_path / "phase5_conditioning.pt")
    config = dict(state.get("ref_encoder_config") or {})
    if "num_tokens" not in config:
        latent_queries = state.get("ref_encoder_latent_queries")
        config["num_tokens"] = int(latent_queries.shape[1]) if latent_queries is not None else 16
    if "num_perceiver_layers" not in config:
        config["num_perceiver_layers"] = _count_ref_perceiver_layers(
            state.get("ref_encoder_perceiver_layers", {})
        )
    config.setdefault("uni_embed_dim", 1536)
    config.setdefault("hidden_dim", 3072)
    config.setdefault("perceiver_heads", 8)
    config.setdefault("use_perceiver_self_attn", True)
    config.setdefault("skip_perceiver", False)
    config.setdefault("perceiver_cross_gate_init", None)
    config.setdefault("regional_ip_token_mode", "spatial")
    config.setdefault("regional_ip_label_mode", "tissue")
    return {
        "uni_embed_dim": int(config["uni_embed_dim"]),
        "hidden_dim": int(config["hidden_dim"]),
        "num_tokens": int(config["num_tokens"]),
        "num_output_tokens": int(config.get("num_output_tokens", config["num_tokens"])),
        "num_perceiver_layers": int(config["num_perceiver_layers"]),
        "perceiver_heads": int(config["perceiver_heads"]),
        "use_perceiver_self_attn": bool(config["use_perceiver_self_attn"]),
        "skip_perceiver": bool(config["skip_perceiver"]),
        "perceiver_cross_gate_init": (
            None
            if config["perceiver_cross_gate_init"] is None
            else float(config["perceiver_cross_gate_init"])
        ),
        "regional_ip_token_mode": normalize_region_ip_token_mode(config["regional_ip_token_mode"]),
        "regional_ip_label_mode": normalize_region_ip_label_mode(config["regional_ip_label_mode"]),
    }


def _load_cross_v1_control_spec(checkpoint_path: Path) -> CrossV1ControlSpec:
    state = _torch_load_weights(checkpoint_path / "phase5_conditioning.pt")
    saved_spec = state.get("cross_v1_control_spec") or {}
    return CrossV1ControlSpec(
        tissue_channels=int(saved_spec.get("tissue_channels", 64)),
        nuclei_channels=int(saved_spec.get("nuclei_channels", 16)),
        spatial_mode=str(
            saved_spec.get(
                "spatial_mode",
                state.get("cross_v1_spatial_mode", "reference_target"),
            )
        ),
    )


def _count_ref_perceiver_layers(state_dict: dict[str, torch.Tensor]) -> int:
    layer_indices = {
        int(key.split(".", 1)[0])
        for key in state_dict
        if key.split(".", 1)[0].isdigit()
    }
    if not layer_indices:
        return 2
    return max(layer_indices) + 1


def _load_condition_modules(
    *,
    checkpoint_path: Path,
    uni_checkpoint_path: str | Path,
    device: str,
    torch_dtype: torch.dtype,
    ref_encoder_config: dict[str, Any] | None = None,
) -> dict[str, nn.Module]:
    state = _torch_load_weights(checkpoint_path / "phase5_conditioning.pt")
    hte_state = state["hte"]
    tissue_state = state["tissue_downsampler"]
    nuclei_state = state["nuclei_encoder"]

    hte_dim = hte_state["parent_embeddings.weight"].shape[1]
    tissue_in = tissue_state["blocks.0.block.0.weight"].shape[1]
    tissue_hidden = tissue_state["blocks.0.block.0.weight"].shape[0]
    tissue_out = tissue_state[f"blocks.{_count_conv_blocks(tissue_state, 'blocks') - 1}.block.0.weight"].shape[0]
    nuclei_embed = nuclei_state["embedding.weight"].shape[1]
    nuclei_out = nuclei_state["downsampler.0.block.0.weight"].shape[0]
    nuclei_blocks = _count_conv_blocks(nuclei_state, "downsampler")

    modules: dict[str, nn.Module] = {
        "hte": HierarchicalTissueEmbedding(embedding_dim=hte_dim),
        "tissue_downsampler": TissueConditionDownsampler(
            in_channels=tissue_in,
            hidden_channels=tissue_hidden,
            out_channels=tissue_out,
            num_blocks=_count_conv_blocks(tissue_state, "blocks"),
        ),
        "nuclei_encoder": NucleiConditionEncoder(
            embedding_dim=nuclei_embed,
            out_channels=nuclei_out,
            num_blocks=nuclei_blocks,
        ),
    }

    for name, module in modules.items():
        module.load_state_dict(state[name])
        module.to(device=device, dtype=torch_dtype)
        module.eval()

    # Load ref_encoder
    from controlnet_train.modules.reference_image_encoder import ReferenceImageEncoder

    ref_config = ref_encoder_config or _load_ref_encoder_config(checkpoint_path)
    ref_encoder = ReferenceImageEncoder(
        uni_checkpoint_path=uni_checkpoint_path,
        uni_embed_dim=ref_config["uni_embed_dim"],
        hidden_dim=ref_config["hidden_dim"],
        num_tokens=ref_config["num_tokens"],
        num_perceiver_layers=ref_config["num_perceiver_layers"],
        perceiver_heads=ref_config["perceiver_heads"],
        use_perceiver_self_attn=ref_config.get("use_perceiver_self_attn", True),
        perceiver_cross_gate_init=ref_config.get("perceiver_cross_gate_init"),
        skip_perceiver=ref_config.get("skip_perceiver", False),
    )
    ref_encoder.proj_mlp.load_state_dict(state["ref_encoder_proj_mlp"])
    if not ref_encoder.skip_perceiver:
        perceiver_keys = (
            "ref_encoder_perceiver_layers",
            "ref_encoder_latent_queries",
            "ref_encoder_perceiver_norm",
        )
        if all(key in state for key in perceiver_keys):
            ref_encoder.load_perceiver_layers_state_dict(state["ref_encoder_perceiver_layers"])
            ref_encoder.latent_queries.data.copy_(
                state["ref_encoder_latent_queries"].to(ref_encoder.latent_queries.device)
            )
            ref_encoder.perceiver_norm.load_state_dict(state["ref_encoder_perceiver_norm"])
        else:
            warnings.warn(
                "phase5_conditioning.pt does not contain reference Perceiver weights; "
                "using the newly initialized Perceiver.",
                RuntimeWarning,
                stacklevel=2,
            )
    ref_encoder.to(device=device)
    ref_encoder.proj_mlp.to(device=device, dtype=torch.float32)
    ref_encoder.perceiver_layers.to(device=device, dtype=torch.float32)
    ref_encoder.perceiver_norm.to(device=device, dtype=torch.float32)
    ref_encoder.latent_queries.data = ref_encoder.latent_queries.data.to(
        device=device,
        dtype=torch.float32,
    )
    ref_encoder.uni.to(device=device, dtype=torch.float32)
    ref_encoder.eval()
    modules["ref_encoder"] = ref_encoder

    return modules


def _count_conv_blocks(state_dict: dict[str, torch.Tensor], prefix: str) -> int:
    return len(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith(prefix) and key.endswith("block.0.weight")
        }
    )


@torch.inference_mode()
def _sample_with_flux_controlnet(
    *,
    pipe,
    controlnet,
    prompt: str,
    control_tensor: torch.Tensor,
    source_control_tensor: torch.Tensor | None = None,
    source_latents: torch.Tensor | None = None,
    source_latent_init_strength: float = 0.0,
    mask_chord_scale: float = 0.0,
    mask_chord_change_mask: torch.Tensor | None = None,
    mask_chord_gate_dilate_radius: int = 0,
    mask_chord_gate_feather_radius: int = 0,
    mask_chord_gate_outside_scale: float = 0.0,
    output_size: tuple[int, int],
    device: str,
    torch_dtype: torch.dtype,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    joint_attention_kwargs: dict | None = None,
    seed: int = 42,
) -> Image.Image:
    from diffusers import FluxControlNetPipeline
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

    torch_device = torch.device(device)
    height, width = output_size
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=[prompt], prompt_2=[prompt], device=torch_device,
    )
    if text_ids.dim() == 3:
        text_ids = text_ids[0]

    control_image = FluxControlNetPipeline._pack_latents(
        control_tensor, 1, control_tensor.shape[1],
        control_tensor.shape[2], control_tensor.shape[3],
    )
    source_control_image = None
    if source_control_tensor is not None:
        source_control_image = FluxControlNetPipeline._pack_latents(
            source_control_tensor,
            1,
            source_control_tensor.shape[1],
            source_control_tensor.shape[2],
            source_control_tensor.shape[3],
        )
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        1, num_channels_latents, height, width,
        prompt_embeds.dtype, torch_device,
        generator=torch.Generator(device=torch_device).manual_seed(int(seed)),
        latents=None,
    )
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = _calculate_shift(
        image_seq_len=image_seq_len,
        base_seq_len=pipe.scheduler.config.get("base_image_seq_len", 256),
        max_seq_len=pipe.scheduler.config.get("max_image_seq_len", 4096),
        base_shift=pipe.scheduler.config.get("base_shift", 0.5),
        max_shift=pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, torch_device, sigmas=sigmas, mu=mu,
    )
    source_latent_init_strength = _validate_source_latent_init_strength(source_latent_init_strength)
    if source_latent_init_strength > 0.0:
        timesteps = _source_init_timesteps(timesteps, source_latent_init_strength)
        source_packed_latents = _pack_source_latents_for_sampling(
            source_latents=source_latents,
            expected_latents=latents,
            torch_dtype=prompt_embeds.dtype,
        )
        start_sigma = _sigma_for_timestep(
            pipe.scheduler,
            timesteps[:1].to(device=torch_device, dtype=torch.float32),
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        latents = _prepare_source_noised_latents(
            source_latents=source_packed_latents,
            noise_latents=latents,
            sigma=start_sigma,
        )

    mask_chord_scale = _validate_nonnegative_float(mask_chord_scale, "mask_chord_scale")
    mask_chord_gate = None
    if mask_chord_change_mask is not None:
        mask_chord_gate = _build_packed_change_gate(
            change_mask=mask_chord_change_mask,
            latent_height=control_tensor.shape[2],
            latent_width=control_tensor.shape[3],
            packed_channels=latents.shape[-1],
            device=torch_device,
            dtype=latents.dtype,
            dilate_radius=mask_chord_gate_dilate_radius,
            feather_radius=mask_chord_gate_feather_radius,
            outside_scale=mask_chord_gate_outside_scale,
        )
    controlnet_blocks_repeat = False if getattr(controlnet, "input_hint_block", None) is None else True

    for timestep in timesteps:
        if mask_chord_scale > 0.0:
            if source_control_image is None:
                raise ValueError("mask_chord_scale > 0 requires source_control_tensor.")
            source_noise_pred, noise_pred = _predict_flux_controlnet_velocity_pair(
                pipe=pipe,
                controlnet=controlnet,
                hidden_states=latents,
                source_controlnet_cond=source_control_image,
                target_controlnet_cond=control_image,
                conditioning_scale=controlnet_conditioning_scale,
                timestep=timestep,
                guidance_scale=guidance_scale,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            delta = noise_pred - source_noise_pred
            if mask_chord_gate is not None:
                delta = delta * mask_chord_gate
            noise_pred = source_noise_pred + mask_chord_scale * delta
        else:
            expanded_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
            controlnet_guidance = None
            if controlnet.config.guidance_embeds:
                controlnet_guidance = torch.tensor([guidance_scale], device=torch_device).expand(latents.shape[0])
            transformer_guidance = None
            if pipe.transformer.config.guidance_embeds:
                transformer_guidance = torch.tensor([guidance_scale], device=torch_device).expand(latents.shape[0])
            noise_pred = _predict_flux_controlnet_velocity(
                pipe=pipe,
                controlnet=controlnet,
                hidden_states=latents,
                controlnet_cond=control_image,
                conditioning_scale=controlnet_conditioning_scale,
                timestep=expanded_timestep / 1000,
                controlnet_guidance=controlnet_guidance,
                transformer_guidance=transformer_guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
                joint_attention_kwargs=joint_attention_kwargs,
            )
        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(dtype=torch_dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def _encode_images_to_latents(vae, images: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    device = next(vae.parameters()).device
    images = images.to(device=device, dtype=torch_dtype)
    images = images * 2.0 - 1.0
    posterior = vae.encode(images).latent_dist
    latents = deterministic_latent_from_posterior(posterior)
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def _predict_flux_controlnet_velocity_pair(
    *,
    pipe,
    controlnet,
    hidden_states: torch.Tensor,
    source_controlnet_cond: torch.Tensor,
    target_controlnet_cond: torch.Tensor,
    conditioning_scale: float,
    timestep: torch.Tensor,
    guidance_scale: float,
    pooled_projections: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    controlnet_blocks_repeat: bool,
    joint_attention_kwargs: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(hidden_states.shape[0])
    batched_hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
    batched_controlnet_cond = torch.cat([source_controlnet_cond, target_controlnet_cond], dim=0)
    batched_timestep = timestep.expand(batch_size * 2).to(dtype=hidden_states.dtype) / 1000
    batched_pooled = torch.cat([pooled_projections, pooled_projections], dim=0)
    batched_encoder_hidden = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
    device = hidden_states.device

    controlnet_guidance = None
    if controlnet.config.guidance_embeds:
        controlnet_guidance = torch.full(
            (batch_size * 2,),
            float(guidance_scale),
            device=device,
            dtype=hidden_states.dtype,
        )
    transformer_guidance = None
    if pipe.transformer.config.guidance_embeds:
        transformer_guidance = torch.full(
            (batch_size * 2,),
            float(guidance_scale),
            device=device,
            dtype=hidden_states.dtype,
        )

    batched_noise_pred = _predict_flux_controlnet_velocity(
        pipe=pipe,
        controlnet=controlnet,
        hidden_states=batched_hidden_states,
        controlnet_cond=batched_controlnet_cond,
        conditioning_scale=conditioning_scale,
        timestep=batched_timestep,
        controlnet_guidance=controlnet_guidance,
        transformer_guidance=transformer_guidance,
        pooled_projections=batched_pooled,
        encoder_hidden_states=batched_encoder_hidden,
        txt_ids=txt_ids,
        img_ids=img_ids,
        controlnet_blocks_repeat=controlnet_blocks_repeat,
        joint_attention_kwargs=_repeat_joint_attention_kwargs(joint_attention_kwargs, repeats=2),
    )
    source_noise_pred, target_noise_pred = batched_noise_pred.chunk(2, dim=0)
    return source_noise_pred, target_noise_pred


def _repeat_joint_attention_kwargs(kwargs: dict | None, *, repeats: int) -> dict | None:
    if kwargs is None:
        return None
    repeated: dict = {}
    for key, value in kwargs.items():
        if key == "ip_hidden_states" and isinstance(value, list):
            repeated[key] = [
                torch.cat([hidden_state] * repeats, dim=0)
                for hidden_state in value
            ]
        elif key == "ip_adapter_masks" and isinstance(value, dict):
            repeated[key] = {
                sub_key: (
                    torch.cat([sub_value] * repeats, dim=0)
                    if torch.is_tensor(sub_value) and sub_value.shape[:1] == (1,)
                    else sub_value
                )
                for sub_key, sub_value in value.items()
            }
        elif torch.is_tensor(value) and value.shape[:1] == (1,):
            repeated[key] = torch.cat([value] * repeats, dim=0)
        else:
            repeated[key] = value
    return repeated


def _predict_flux_controlnet_velocity(
    *,
    pipe,
    controlnet,
    hidden_states: torch.Tensor,
    controlnet_cond: torch.Tensor,
    conditioning_scale: float,
    timestep: torch.Tensor,
    controlnet_guidance: torch.Tensor | None,
    transformer_guidance: torch.Tensor | None,
    pooled_projections: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    controlnet_blocks_repeat: bool,
    joint_attention_kwargs: dict | None = None,
) -> torch.Tensor:
    controlnet_block_samples, controlnet_single_block_samples = controlnet(
        hidden_states=hidden_states,
        controlnet_cond=controlnet_cond,
        controlnet_mode=None,
        conditioning_scale=conditioning_scale,
        timestep=timestep,
        guidance=controlnet_guidance,
        pooled_projections=pooled_projections,
        encoder_hidden_states=encoder_hidden_states,
        txt_ids=txt_ids,
        img_ids=img_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )
    return pipe.transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        guidance=transformer_guidance,
        pooled_projections=pooled_projections,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_block_samples=controlnet_block_samples,
        controlnet_single_block_samples=controlnet_single_block_samples,
        txt_ids=txt_ids,
        img_ids=img_ids,
        joint_attention_kwargs=dict(joint_attention_kwargs) if joint_attention_kwargs is not None else None,
        return_dict=False,
        controlnet_blocks_repeat=controlnet_blocks_repeat,
    )[0]


def _calculate_shift(
    *,
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept

'''
python -m controlnet_train.cli.eval_controlnet_flux_cross_v1   --pretrained-model-name-or-path /data/huggingface/FLUX.1-dev   --checkpoint /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_cross_v1/checkpoint-40000   --uni-checkpoint-path /home/lyw/wqx-DL/flow-edit/FlowEdit-main/UNI-2h/pytorch_model.bin   --metadata /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_meta/metadata_cross_val.json   --output-dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_v1_eval_10cases   --num-samples 10   --device cuda   --torch-dtype bf16   --prompt-source dataset
'''
