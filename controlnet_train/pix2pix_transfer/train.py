"""Train supervised I0/reference -> target pix2pix texture transfer."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .adversarial import (
    RegionAwarePatchDiscriminator,
    conditional_mismatch_hinge_loss,
    discriminator_hinge_loss,
    discriminator_logit_stats,
    generator_hinge_loss,
    patch_mask_from_region,
    soft_boundary_patch_mask,
)
from .detail_corruption import (
    apply_local_detail_dropout,
    build_context_mismatch_condition,
    rotate_batch_d4,
    sample_nonzero_d4_codes,
)
from .dataset import (
    I0ReferenceTextureDataset,
    i0_cache_path,
    load_label_mask,
    one_hot_mask,
    remap_nuclei_mask,
    resolve_path,
    tissue_nuclei_region_labels,
)
from .dataset import load_rgb as load_rgb_neg1
from .losses import (
    Pix2PixTransferLoss,
    boundary_band_mask,
    masked_multiband_detail_loss,
    masked_orientation_consistency_loss,
    regional_rotation_invariant_style_loss,
)
from .ood_diagnose import (
    compute_identity_metrics,
    save_identity_metrics,
    save_ood_panel,
    save_ood_summary_grid,
)
from .orientation_supervision import (
    multiscale_target_orientation_loss,
    windowed_fine_texture_energy_floor_loss,
    windowed_i0_mean_orientation_loss,
)
from .inference_orientation import build_fine_texture_steering_weights
from .regional_cross_attention import Pix2PixCrossAttnUNet, model_parameter_count
from .reference_augmentation import (
    ramped_rotation_probability,
    rotate_reference_bundle,
    sample_continuous_rotation_angle,
)
from .rotation_monitor import (
    compute_rotation_monitor_metrics,
    rotation_monitor_reasons,
    save_rotation_monitor_panel,
)
from .trust_gate import build_reference_trust_map
from .training_modes import (
    DistributedWeightedSampler,
    build_cross_wsi_permutation,
    build_difficulty_sampling_weights,
)


DEFAULT_OOD_PROBE_ROOTS = (
    "/data/wqx/flowedit/ood_pix2pix_diagnosis_20260703",
    "/data/wqx/flowedit/ood_pix2pix_val_unseen_refs_20260703",
)


def parse_positive_int_tuple(value: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must be a comma-separated list of positive integers")
    return values


def parse_float_tuple(value: str, *, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise ValueError(f"{name} must be a comma-separated list of numbers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-root", default=None)
    parser.add_argument("--i0-field", default="i0_image")
    parser.add_argument(
        "--i0-cache-dir",
        default=None,
        help="Directory containing cached ControlNet I0 images named by metadata index.",
    )
    parser.add_argument("--val-metadata", default=None)
    parser.add_argument("--val-i0-cache-dir", default=None)
    parser.add_argument(
        "--lazy-generate-i0",
        action="store_true",
        help="Generate missing cached I0 images during training/eval and save them.",
    )
    parser.add_argument("--pretrained-model-name-or-path", default=None)
    parser.add_argument("--checkpoint", default=None, help="Cross V1 ControlNet checkpoint dir.")
    parser.add_argument("--uni-checkpoint-path", default=None)
    parser.add_argument("--controlnet-torch-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--controlnet-num-inference-steps", type=int, default=28)
    parser.add_argument("--controlnet-guidance-scale", type=float, default=3.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0)
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument("--source-latent-init-strength", type=float, default=0.0)
    parser.add_argument("--mask-chord-scale", type=float, default=0.0)
    parser.add_argument("--mask-chord-use-gate", action="store_true")
    parser.add_argument("--mask-chord-gate-dilate-radius", type=int, default=0)
    parser.add_argument("--mask-chord-gate-feather-radius", type=int, default=0)
    parser.add_argument("--mask-chord-gate-outside-scale", type=float, default=0.0)
    parser.add_argument("--i0-prompt-source", choices=("metadata", "dataset"), default="dataset")
    parser.add_argument("--i0-prompt", default=None)
    parser.add_argument("--i0-generation-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--trainable-scope",
        choices=(
            "all",
            "full_discriminative",
            "highres_decoder",
            "highres_cross4_qproj",
            "steered_cross4",
            "steered_cross4_cross8",
            "steered_full_pyramid",
        ),
        default="all",
    )
    parser.add_argument("--highres-lr", type=float, default=5.0e-6)
    parser.add_argument("--cross4-lr", type=float, default=1.0e-6)
    parser.add_argument("--midcross-lr", type=float, default=1.0e-6)
    parser.add_argument("--detail-dropout-prob", type=float, default=0.0)
    parser.add_argument("--detail-dropout-min-diameter", type=int, default=32)
    parser.add_argument("--detail-dropout-max-diameter", type=int, default=96)
    parser.add_argument("--detail-dropout-sigma-min", type=float, default=1.2)
    parser.add_argument("--detail-dropout-sigma-max", type=float, default=2.5)
    parser.add_argument("--detail-dropout-feather-radius", type=int, default=5)
    parser.add_argument("--lambda-detail-fine", type=float, default=0.0)
    parser.add_argument("--lambda-detail-mid", type=float, default=0.0)
    parser.add_argument("--lambda-baseline-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-anchor-teacher-consistency", type=float, default=0.0)
    parser.add_argument("--rotated-i0-dropout-prob", type=float, default=0.0)
    parser.add_argument("--rotated-i0-dropout-min-diameter", type=int, default=64)
    parser.add_argument("--rotated-i0-dropout-max-diameter", type=int, default=128)
    parser.add_argument("--rotated-i0-dropout-sigma-min", type=float, default=2.0)
    parser.add_argument("--rotated-i0-dropout-sigma-max", type=float, default=3.0)
    parser.add_argument("--reference-rotation-pair-prob", type=float, default=0.0)
    parser.add_argument("--lambda-ref-orientation-consistency", type=float, default=0.0)
    parser.add_argument("--lambda-ref-rotation-style", type=float, default=0.0)
    parser.add_argument("--main-ref-random-rotation-prob", type=float, default=0.0)
    parser.add_argument("--main-ref-random-rotation-min-degrees", type=float, default=15.0)
    parser.add_argument("--main-ref-random-rotation-max-degrees", type=float, default=180.0)
    parser.add_argument("--main-ref-random-rotation-ramp-steps", type=int, default=0)
    parser.add_argument("--rotated-ref-l1-scale", type=float, default=2.0)
    parser.add_argument("--rotated-ref-content-scale", type=float, default=2.0)
    parser.add_argument("--rotated-ref-gram-scale", type=float, default=0.5)
    parser.add_argument("--rotated-ref-contextual-scale", type=float, default=0.25)
    parser.add_argument("--lambda-target-orientation", type=float, default=0.0)
    parser.add_argument("--lambda-target-anisotropy", type=float, default=0.0)
    parser.add_argument("--target-orientation-min-coherence", type=float, default=0.20)
    parser.add_argument("--target-orientation-min-trust", type=float, default=0.50)
    parser.add_argument("--target-orientation-boundary-radius", type=int, default=1)
    parser.add_argument("--target-orientation-nuclei-radius", type=int, default=2)
    parser.add_argument("--lambda-i0-window-orientation", type=float, default=0.0)
    parser.add_argument("--lambda-i0-window-directionality", type=float, default=0.0)
    parser.add_argument("--lambda-i0-residual-orientation", type=float, default=0.0)
    parser.add_argument("--lambda-i0-texture-energy", type=float, default=0.0)
    parser.add_argument("--i0-texture-energy-floor-ratio", type=float, default=0.95)
    parser.add_argument("--i0-texture-energy-ceiling-ratio", type=float, default=0.0)
    parser.add_argument("--i0-orientation-window-sizes", default="32,64")
    parser.add_argument("--i0-orientation-window-strides", default="16,32")
    parser.add_argument("--i0-orientation-min-coherence", type=float, default=0.20)
    parser.add_argument("--i0-orientation-min-relative-energy", type=float, default=0.50)
    parser.add_argument("--i0-orientation-min-window-fraction", type=float, default=0.25)
    parser.add_argument("--i0-orientation-min-resultant", type=float, default=0.15)
    parser.add_argument("--i0-orientation-directionality-floor-ratio", type=float, default=0.50)
    parser.add_argument("--i0-orientation-min-trust", type=float, default=0.50)
    parser.add_argument("--i0-orientation-boundary-radius", type=int, default=3)
    parser.add_argument("--i0-orientation-nuclei-radius", type=int, default=5)
    parser.add_argument("--i0-orientation-ramp-steps", type=int, default=200)
    parser.add_argument("--cross4-texture-steering", action="store_true")
    parser.add_argument("--cross4-steering-angles", default="0,45,90,135")
    parser.add_argument("--cross4-steering-smoothing-sigma", type=float, default=8.0)
    parser.add_argument("--cross4-steering-min-coherence", type=float, default=0.20)
    parser.add_argument("--cross4-steering-min-relative-energy", type=float, default=0.50)
    parser.add_argument("--cross4-steering-min-resultant", type=float, default=0.15)
    parser.add_argument("--cross4-steering-minimum-strength", type=float, default=0.0)
    parser.add_argument("--cross4-steering-minimum-support", type=float, default=0.05)
    parser.add_argument("--cross4-steering-temperature", type=float, default=0.08)
    parser.add_argument(
        "--cross4-steering-reference-mode",
        choices=("global_mean", "local_histogram"),
        default="global_mean",
    )
    parser.add_argument("--cross4-steering-local-bins", type=int, default=36)
    parser.add_argument("--cross4-steering-local-kappa", type=float, default=8.0)
    parser.add_argument("--cross4-steering-scales", default="1/4")
    parser.add_argument("--cross4-steering-gain", type=float, default=1.0)
    parser.add_argument("--cross8-steering-gain", type=float, default=1.0)
    parser.add_argument("--cross16-steering-gain", type=float, default=1.0)
    parser.add_argument("--cross2-steering-gain", type=float, default=1.0)
    parser.add_argument("--cross1-steering-gain", type=float, default=1.0)
    parser.add_argument("--full-pyramid-texture-steering", action="store_true")
    parser.add_argument("--steering-highres-reference-size", type=int, default=8)
    parser.add_argument("--boundary-adv-floor", type=float, default=0.0)
    parser.add_argument("--condition-mismatch-d-weight", type=float, default=0.0)
    parser.add_argument("--context-ramp-steps", type=int, default=500)
    parser.add_argument("--max-continuation-steps", type=int, default=0)
    parser.add_argument("--rotation-monitor-every-steps", type=int, default=0)
    parser.add_argument("--rotation-monitor-stop-patience", type=int, default=2)
    parser.add_argument("--rotation-monitor-max-clean-drift", type=float, default=0.03)
    parser.add_argument("--rotation-monitor-max-ref-distance-ratio", type=float, default=1.10)
    parser.add_argument("--rotation-monitor-max-boundary-seam-ratio", type=float, default=1.10)
    parser.add_argument("--rotation-monitor-min-nuclei-band-ratio", type=float, default=0.85)
    parser.add_argument("--rotation-monitor-max-nuclei-band-ratio", type=float, default=1.15)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--upsample-mode",
        choices=("bilinear", "nearest"),
        default="bilinear",
        help=(
            "Decoder resize-conv upsampling mode. bilinear is smoother and usually "
            "reduces checkerboard artifacts; nearest is sharper but can look blockier."
        ),
    )
    parser.add_argument(
        "--cross-attn-scales",
        default="1/4,1/8,1/16",
        help=(
            "Comma-separated cross-attention scales. Default uses 1/4, 1/8 and 1/16 "
            "so fine reference texture reaches the high-resolution decoder."
        ),
    )
    parser.add_argument(
        "--region-label-mode",
        choices=("tissue", "nuclei", "tissue_nuclei"),
        default="tissue_nuclei",
    )
    parser.add_argument("--no-region-mask", action="store_true")
    parser.add_argument("--no-residual-output", action="store_true")
    parser.add_argument("--augment-flips", action="store_true")
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--lambda-perc", type=float, default=1.0)
    parser.add_argument("--lambda-gram", type=float, default=1.0)
    parser.add_argument("--lambda-contextual", type=float, default=1.0)
    parser.add_argument("--ref-trust-gate", action="store_true")
    parser.add_argument("--ref-fallback-scale", type=float, default=0.05)
    parser.add_argument("--ref-soft-context-scale", type=float, default=0.0)
    parser.add_argument("--ref-nuclei-context-scale", type=float, default=0.0)
    parser.add_argument("--ref-soft-context-radius", type=int, default=0)
    parser.add_argument("--ref-mismatch-prob", type=float, default=0.0)
    parser.add_argument("--cross-wsi-style-prob", type=float, default=None)
    parser.add_argument("--hard-pair-sampling", action="store_true")
    parser.add_argument("--hard-pair-full-mass", type=float, default=0.40)
    parser.add_argument("--hard-pair-hard-mass", type=float, default=0.30)
    parser.add_argument("--matched-tissue-trust-floor", type=float, default=0.0)
    parser.add_argument("--matched-nuclei-trust-floor", type=float, default=0.0)
    parser.add_argument("--wsi-identity-adapter", action="store_true")
    parser.add_argument("--identity-gamma-max", type=float, default=0.30)
    parser.add_argument("--identity-gamma-init", type=float, default=0.10)
    parser.add_argument("--identity-min-tissue-pixels", type=int, default=256)
    parser.add_argument("--identity-min-nuclei-pixels", type=int, default=64)
    parser.add_argument("--identity-warmup-steps", type=int, default=2000)
    parser.add_argument("--identity-lr", type=float, default=1.0e-4)
    parser.add_argument("--backbone-lr", type=float, default=2.0e-5)
    parser.add_argument("--boundary-feather-radius", type=int, default=0)
    parser.add_argument("--lambda-boundary-hf", type=float, default=0.0)
    parser.add_argument("--lambda-lowtrust-hf", type=float, default=0.0)
    parser.add_argument("--lambda-identity-od", type=float, default=0.10)
    parser.add_argument("--lambda-identity-feature", type=float, default=0.25)
    parser.add_argument("--lambda-identity-band", type=float, default=0.15)
    parser.add_argument("--lambda-identity-rank", type=float, default=0.10)
    parser.add_argument("--cross-lambda-identity-od", type=float, default=0.20)
    parser.add_argument("--cross-lambda-identity-feature", type=float, default=0.40)
    parser.add_argument("--cross-lambda-identity-band", type=float, default=0.20)
    parser.add_argument("--cross-lambda-identity-rank", type=float, default=0.15)
    parser.add_argument("--cross-lambda-gram", type=float, default=0.50)
    parser.add_argument("--cross-lambda-contextual", type=float, default=0.50)
    parser.add_argument("--lambda-structure-gray", type=float, default=0.50)
    parser.add_argument("--lambda-structure-edge", type=float, default=0.50)
    parser.add_argument("--identity-rank-margin", type=float, default=0.10)
    parser.add_argument("--ood-diagnose-every-epochs", type=int, default=0)
    parser.add_argument("--ood-diagnose-num-panels", type=int, default=5)
    parser.add_argument("--ood-diagnose-root", default=None)
    parser.add_argument(
        "--lambda-adv",
        type=float,
        default=0.0,
        help="Weight for region-aware PatchGAN generator loss. 0 disables GAN training.",
    )
    parser.add_argument(
        "--adv-warmup-steps",
        type=int,
        default=1000,
        help="Do not train/use PatchGAN before this global step.",
    )
    parser.add_argument(
        "--adv-mask-mode",
        choices=("non_background", "all"),
        default="non_background",
        help="Where to apply patch adversarial loss. non_background ignores white/background patches.",
    )
    parser.add_argument("--d-lr", type=float, default=None, help="Discriminator LR; defaults to --lr.")
    parser.add_argument("--d-weight-decay", type=float, default=0.0)
    parser.add_argument("--d-base-channels", type=int, default=64)
    parser.add_argument("--d-max-channels", type=int, default=512)
    parser.add_argument("--d-num-layers", type=int, default=3)
    parser.add_argument(
        "--no-d-spectral-norm",
        action="store_true",
        help="Disable spectral normalization in the PatchGAN discriminator.",
    )
    parser.add_argument(
        "--l1-blur-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma for low-frequency L1; 0 keeps full-resolution pixel L1.",
    )
    parser.add_argument("--content-layers", default="3,8,15,22")
    parser.add_argument("--gram-layers", default="3,8,15")
    parser.add_argument("--contextual-layers", default="8,15")
    parser.add_argument("--texture-min-pixels", type=int, default=8)
    parser.add_argument("--contextual-max-samples", type=int, default=256)
    parser.add_argument("--contextual-temperature", type=float, default=0.1)
    parser.add_argument("--loss-normalization-decay", type=float, default=0.99)
    parser.add_argument(
        "--loss-normalization-steps",
        type=int,
        default=200,
        help="Calibrate EMA loss scales for this many steps, then freeze them; <=0 never freezes.",
    )
    parser.add_argument(
        "--no-loss-normalization",
        action="store_true",
        help="Disable synchronized EMA normalization of L1/content/Gram/contextual losses.",
    )
    parser.add_argument("--vgg-weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--eval-num-samples", type=int, default=5)
    parser.add_argument("--eval-batch-size", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, int]:
    if not is_distributed():
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        return device, 0, 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank, rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def autocast_context(device: torch.device, mixed_precision: str):
    enabled = mixed_precision != "no" and device.type == "cuda"
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def controlnet_dtype_by_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[str(name)]


def _batch_item(value: Any, index: int) -> Any:
    if torch.is_tensor(value):
        return value[index].item() if value.ndim > 0 else value.item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _batch_string(batch: dict[str, Any], key: str, index: int) -> str:
    value = batch.get(key, "")
    item = _batch_item(value, index)
    return "" if item is None else str(item)


def _batch_int(batch: dict[str, Any], key: str, index: int) -> int:
    value = batch.get(key, index)
    item = _batch_item(value, index)
    try:
        return int(item)
    except (TypeError, ValueError):
        return int(index)


def _pil_to_tensor_neg1(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    image = image.convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _write_i0_into_batch(batch: dict[str, Any], index: int, i0: torch.Tensor) -> None:
    batch["i0"][index].copy_(i0)
    batch["target_cond"][index, :3].copy_(i0)
    missing = batch.get("i0_missing")
    if torch.is_tensor(missing):
        missing[index] = False


class LazyI0Generator:
    def __init__(self, args: argparse.Namespace, device: torch.device, *, rank: int) -> None:
        self.args = args
        self.device = device
        self.rank = int(rank)
        self.bundle = None
        self.generated = 0
        self.loaded_after_race = 0

    def _load_bundle(self):
        if self.bundle is not None:
            return self.bundle
        if not self.args.pretrained_model_name_or_path:
            raise ValueError("--lazy-generate-i0 requires --pretrained-model-name-or-path")
        if not self.args.checkpoint:
            raise ValueError("--lazy-generate-i0 requires --checkpoint")
        if not self.args.uni_checkpoint_path:
            raise ValueError("--lazy-generate-i0 requires --uni-checkpoint-path")
        from controlnet_train.inference.pipeline_cross_v1 import load_cross_v1_bundle

        self.bundle = load_cross_v1_bundle(
            pretrained_model_name_or_path=self.args.pretrained_model_name_or_path,
            checkpoint_path=self.args.checkpoint,
            uni_checkpoint_path=self.args.uni_checkpoint_path,
            device=str(self.device),
            torch_dtype=controlnet_dtype_by_name(self.args.controlnet_torch_dtype),
            num_inference_steps=self.args.controlnet_num_inference_steps,
            guidance_scale=self.args.controlnet_guidance_scale,
            controlnet_conditioning_scale=self.args.controlnet_conditioning_scale,
            ip_adapter_scale=self.args.ip_scale,
        )
        return self.bundle

    def _resolve_prompt(self, batch: dict[str, Any], index: int) -> str:
        if self.args.i0_prompt:
            return str(self.args.i0_prompt)
        if self.args.i0_prompt_source == "metadata":
            prompt = _batch_string(batch, "prompt", index)
            if prompt:
                return prompt
        if self.args.i0_prompt_source == "dataset":
            dataset_name = _batch_string(batch, "dataset", index)
            if dataset_name:
                from controlnet_train.data.common import default_prompt_for_dataset

                return default_prompt_for_dataset(dataset_name)
        prompt = _batch_string(batch, "prompt", index)
        return prompt or "H&E stained cancer histopathology at 40x magnification"

    def _missing_indices(self, batch: dict[str, Any]) -> list[int]:
        missing = batch.get("i0_missing")
        if missing is None:
            return []
        if torch.is_tensor(missing):
            return [int(i) for i in torch.nonzero(missing.bool(), as_tuple=False).view(-1).tolist()]
        return [i for i, value in enumerate(missing) if bool(value)]

    @torch.no_grad()
    def fill_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        missing_indices = self._missing_indices(batch)
        if not missing_indices:
            return batch

        from controlnet_train.inference.pipeline_cross_v1 import run_cross_v1_bundle

        bundle = self._load_bundle()
        height = int(batch["i0"].shape[-2])
        width = int(batch["i0"].shape[-1])
        for index in missing_indices:
            cache_path = Path(_batch_string(batch, "i0_cache_path", index))
            if cache_path.exists():
                i0 = load_rgb_neg1(cache_path, height)
                self.loaded_after_race += 1
                _write_i0_into_batch(batch, index, i0)
                continue

            reference_image = ((batch["reference_image"][index].float() + 1.0) * 0.5).clamp(0.0, 1.0)
            reference_tissue_mask = batch["reference_tissue_mask"][index, 0].long()
            reference_nuclei_mask = batch["reference_nuclei_mask"][index, 0].long()
            target_tissue_mask = batch["target_tissue_mask"][index, 0].long()
            target_nuclei_mask = batch["target_nuclei_mask"][index, 0].long()
            metadata_index = _batch_int(batch, "metadata_index", index)
            prompt = self._resolve_prompt(batch, index)
            image = run_cross_v1_bundle(
                bundle,
                reference_image=reference_image,
                reference_tissue_mask=reference_tissue_mask,
                reference_nuclei_mask=reference_nuclei_mask,
                target_tissue_mask=target_tissue_mask,
                target_nuclei_mask=target_nuclei_mask,
                prompt=prompt,
                source_latent_init_strength=self.args.source_latent_init_strength,
                mask_chord_scale=self.args.mask_chord_scale,
                mask_chord_use_gate=self.args.mask_chord_use_gate,
                mask_chord_gate_dilate_radius=self.args.mask_chord_gate_dilate_radius,
                mask_chord_gate_feather_radius=self.args.mask_chord_gate_feather_radius,
                mask_chord_gate_outside_scale=self.args.mask_chord_gate_outside_scale,
                seed=int(self.args.i0_generation_seed) + int(metadata_index),
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(cache_path)
            i0 = _pil_to_tensor_neg1(image, (width, height))
            self.generated += 1
            _write_i0_into_batch(batch, index, i0)
        if self.generated and self.generated % 10 == 0:
            print(f"[rank {self.rank}] lazy generated I0 count={self.generated}")
        return batch


def save_training_sample(
    *,
    output_dir: Path,
    step: int,
    batch: dict[str, Any],
    pred: torch.Tensor,
    max_items: int = 4,
) -> None:
    count = min(int(max_items), int(pred.shape[0]))
    grid = [
        batch["i0"][:count].detach().cpu(),
        batch["reference_image"][:count].detach().cpu(),
        pred[:count].detach().cpu(),
        batch["target_image"][:count].detach().cpu(),
    ]
    rows = ["I0", "Reference", "Prediction", "Target"]
    cell_w, cell_h = _tensor_to_pil(grid[0][0]).size
    label_h = 24
    row_label_w = 88
    canvas = Image.new(
        "RGB",
        (row_label_w + count * cell_w, label_h + len(grid) * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for col in range(count):
        draw.text((row_label_w + col * cell_w + 6, 5), f"sample {col}", fill=(0, 0, 0))
    for row, (label, tensor) in enumerate(zip(rows, grid)):
        draw.text((6, label_h + row * cell_h + 6), label, fill=(0, 0, 0))
        for col in range(count):
            canvas.paste(
                _tensor_to_pil(tensor[col]),
                (row_label_w + col * cell_w, label_h + row * cell_h),
            )
    target_path = output_dir / "samples" / f"step{step:08d}.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target_path)


def _flat_training_sample(
    *,
    batch: dict[str, Any],
    pred: torch.Tensor,
    max_items: int,
) -> torch.Tensor:
    count = min(int(max_items), int(pred.shape[0]))
    return torch.cat(
        [
            batch["i0"][:count].detach().cpu(),
            batch["reference_image"][:count].detach().cpu(),
            pred[:count].detach().cpu(),
            batch["target_image"][:count].detach().cpu(),
        ],
        dim=0,
    )


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        ((image.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _metadata_index_list(value: Any, count: int) -> list[str]:
    if torch.is_tensor(value):
        values = value.detach().cpu().view(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value] * count
    return [str(v) for v in values[:count]]


def _save_eval_panel(
    *,
    output_path: Path,
    i0: torch.Tensor,
    target: torch.Tensor,
    reference: torch.Tensor,
    pred: torch.Tensor,
    metadata_index: Any,
) -> None:
    count = int(pred.shape[0])
    if count <= 0:
        return
    columns = ["I0 ControlNet", "Target GT", "Reference GT", "Model output"]
    images = [i0, target, reference, pred]
    cell_w, cell_h = _tensor_to_pil(images[0][0]).size
    label_h = 26
    row_label_w = 92
    canvas = Image.new(
        "RGB",
        (row_label_w + len(columns) * cell_w, label_h + count * cell_h),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(columns):
        draw.text((row_label_w + col * cell_w + 6, 6), label, fill=(0, 0, 0))
    row_labels = _metadata_index_list(metadata_index, count)
    for row in range(count):
        draw.text((6, label_h + row * cell_h + 6), f"idx {row_labels[row]}", fill=(0, 0, 0))
        for col, tensor in enumerate(images):
            canvas.paste(
                _tensor_to_pil(tensor[row]),
                (row_label_w + col * cell_w, label_h + row * cell_h),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


@torch.no_grad()
def save_eval_panel(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    mixed_precision: str,
    i0_generator: LazyI0Generator | None = None,
) -> None:
    model.eval()
    for batch in loader:
        if i0_generator is not None:
            batch = i0_generator.fill_batch(batch)
        target_cond = batch["target_cond"].to(device, non_blocking=True)
        reference_cond = batch["reference_cond"].to(device, non_blocking=True)
        target_region = batch["target_region"].to(device, non_blocking=True)
        reference_region = batch["reference_region"].to(device, non_blocking=True)
        with autocast_context(device, mixed_precision):
            pred = model(
                target_cond,
                reference_cond,
                target_region=target_region,
                reference_region=reference_region,
                target_tissue_mask=batch["target_tissue_mask"].to(device, non_blocking=True),
                target_nuclei_mask=batch["target_nuclei_mask"].to(device, non_blocking=True),
                reference_tissue_mask=batch["reference_tissue_mask"].to(device, non_blocking=True),
                reference_nuclei_mask=batch["reference_nuclei_mask"].to(device, non_blocking=True),
            )
        _save_eval_panel(
            output_path=output_dir / "eval" / f"epoch{epoch + 1:04d}.png",
            i0=batch["i0"],
            target=batch["target_image"],
            reference=batch["reference_image"],
            pred=pred.detach().cpu(),
            metadata_index=batch.get("metadata_index", ""),
        )
        return


@torch.no_grad()
def run_rotation_monitor(
    *,
    current_model: torch.nn.Module,
    baseline_model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    continuation_step: int,
    mixed_precision: str,
    seed: int,
    ref_trust_gate: bool,
    ref_fallback_scale: float,
    ref_soft_context_scale: float,
    ref_nuclei_context_scale: float,
    ref_soft_context_radius: int,
    matched_tissue_trust_floor: float,
    matched_nuclei_trust_floor: float,
    max_clean_drift: float,
    max_ref_distance_ratio: float,
    max_boundary_seam_ratio: float,
    min_nuclei_band_ratio: float,
    max_nuclei_band_ratio: float,
) -> tuple[dict[str, float], list[str]]:
    """Compare epoch25 and current outputs on one fixed clean/rotated batch."""

    tensor_keys = (
        "target_cond",
        "reference_cond",
        "target_image",
        "target_region",
        "reference_region",
        "target_tissue_mask",
        "target_nuclei_mask",
        "reference_tissue_mask",
        "reference_nuclei_mask",
    )
    tensors = {
        key: batch[key].to(device, non_blocking=True)
        for key in tensor_keys
    }
    target_cond = tensors["target_cond"]
    reference_cond = tensors["reference_cond"]
    trust_map = None
    if ref_trust_gate:
        trust_map, _ = build_reference_trust_map(
            tensors["target_region"],
            tensors["reference_region"],
            fallback_scale=ref_fallback_scale,
            matched_tissue_floor=matched_tissue_trust_floor,
            matched_nuclei_floor=matched_nuclei_trust_floor,
        )
        trust_map = trust_map.to(device=device, dtype=target_cond.dtype)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    angles = torch.tensor(
        [
            sample_continuous_rotation_angle(
                probability=1.0,
                min_degrees=15.0,
                max_degrees=180.0,
                generator=generator,
            )
            for _ in range(reference_cond.shape[0])
        ],
        device=device,
        dtype=torch.float32,
    )
    rotated = rotate_reference_bundle(
        reference_cond,
        tensors["reference_region"],
        tensors["reference_tissue_mask"],
        tensors["reference_nuclei_mask"],
        angles_degrees=angles,
    )

    def forward(
        model: torch.nn.Module,
        ref_cond: torch.Tensor,
        ref_region: torch.Tensor,
        ref_tissue: torch.Tensor,
        ref_nuclei: torch.Tensor,
    ) -> torch.Tensor:
        return model(
            target_cond,
            ref_cond,
            target_region=tensors["target_region"],
            reference_region=ref_region,
            target_trust_map=trust_map,
            ref_fallback_scale=ref_fallback_scale if ref_trust_gate else 1.0,
            ref_soft_context_scale=ref_soft_context_scale,
            ref_nuclei_context_scale=ref_nuclei_context_scale,
            ref_soft_context_radius=ref_soft_context_radius,
            target_tissue_mask=tensors["target_tissue_mask"],
            target_nuclei_mask=tensors["target_nuclei_mask"],
            reference_tissue_mask=ref_tissue,
            reference_nuclei_mask=ref_nuclei,
        )

    was_training = current_model.training
    current_model.eval()
    baseline_model.eval()
    with autocast_context(device, mixed_precision):
        baseline_clean = forward(
            baseline_model,
            reference_cond,
            tensors["reference_region"],
            tensors["reference_tissue_mask"],
            tensors["reference_nuclei_mask"],
        )
        current_clean = forward(
            current_model,
            reference_cond,
            tensors["reference_region"],
            tensors["reference_tissue_mask"],
            tensors["reference_nuclei_mask"],
        )
        baseline_rotated = forward(
            baseline_model,
            rotated.reference_cond,
            rotated.reference_region,
            rotated.reference_tissue_mask,
            rotated.reference_nuclei_mask,
        )
        current_rotated = forward(
            current_model,
            rotated.reference_cond,
            rotated.reference_region,
            rotated.reference_tissue_mask,
            rotated.reference_nuclei_mask,
        )
    current_model.train(was_training)
    metrics = compute_rotation_monitor_metrics(
        target_i0=target_cond[:, :3].float(),
        target=tensors["target_image"].float(),
        reference=reference_cond[:, :3].float(),
        baseline_clean=baseline_clean.float(),
        current_clean=current_clean.float(),
        baseline_rotated=baseline_rotated.float(),
        current_rotated=current_rotated.float(),
        target_tissue_mask=tensors["target_tissue_mask"],
        target_nuclei_mask=tensors["target_nuclei_mask"],
        reference_tissue_mask=tensors["reference_tissue_mask"],
        reference_nuclei_mask=tensors["reference_nuclei_mask"],
        trust_map=(
            trust_map
            if trust_map is not None
            else target_cond.new_ones((target_cond.shape[0], 1, *target_cond.shape[-2:]))
        ),
    )
    reasons = rotation_monitor_reasons(
        metrics,
        max_clean_drift=max_clean_drift,
        max_ref_distance_ratio=max_ref_distance_ratio,
        max_boundary_seam_ratio=max_boundary_seam_ratio,
        min_nuclei_band_ratio=min_nuclei_band_ratio,
        max_nuclei_band_ratio=max_nuclei_band_ratio,
    )
    monitor_dir = output_dir / "rotation_monitor" / f"step{continuation_step:06d}"
    save_rotation_monitor_panel(
        output_path=monitor_dir / "comparison.png",
        target_i0=target_cond[:, :3],
        target=tensors["target_image"],
        reference=reference_cond[:, :3],
        rotated_reference=rotated.reference_cond[:, :3],
        baseline_clean=baseline_clean,
        current_clean=current_clean,
        baseline_rotated=baseline_rotated,
        current_rotated=current_rotated,
        angles_degrees=angles,
    )
    (monitor_dir / "metrics.json").write_text(
        json.dumps(
            {
                "continuation_step": int(continuation_step),
                "angles_degrees": [float(value) for value in angles.cpu().tolist()],
                "metrics": metrics,
                "safety_reasons": reasons,
            },
            indent=2,
        ),
        encoding="utf8",
    )
    return metrics, reasons


def _load_mask_pair(
    tissue_path: str | Path,
    nuclei_path: str | Path,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tissue = load_label_mask(tissue_path, image_size)
    nuclei = remap_nuclei_mask(load_label_mask(nuclei_path, image_size))
    return tissue, nuclei


def _make_condition(image: torch.Tensor, tissue: torch.Tensor, nuclei: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            image,
            one_hot_mask(tissue, 16),
            one_hot_mask(nuclei, 6),
        ],
        dim=0,
    )


def _load_existing_ood_probe(
    root: Path,
    *,
    image_size: int,
    region_label_mode: str,
) -> dict[str, Any] | None:
    target_i0_path = root / "target_i0_cache.png"
    target_path = root / "target_real.png"
    target_tissue_path = root / "target_tissue_mask.png"
    target_nuclei_path = root / "target_nuclei_mask.png"
    if not all(path.exists() for path in (target_i0_path, target_path, target_tissue_path, target_nuclei_path)):
        return None
    ref_dirs = sorted(path for path in root.glob("ref_*") if path.is_dir())
    if not ref_dirs:
        return None
    target_i0 = load_rgb_neg1(target_i0_path, image_size)
    target = load_rgb_neg1(target_path, image_size)
    target_tissue, target_nuclei = _load_mask_pair(target_tissue_path, target_nuclei_path, image_size)
    target_cond = _make_condition(target_i0, target_tissue, target_nuclei)
    target_region = tissue_nuclei_region_labels(
        target_tissue,
        target_nuclei,
        label_mode=region_label_mode,
    )
    references = []
    reference_conds = []
    reference_regions = []
    reference_tissues = []
    reference_nuclei = []
    for ref_dir in ref_dirs[:5]:
        ref_path = ref_dir / "reference.png"
        ref_tissue_path = ref_dir / "reference_tissue_mask.png"
        ref_nuclei_path = ref_dir / "reference_nuclei_mask.png"
        if not all(path.exists() for path in (ref_path, ref_tissue_path, ref_nuclei_path)):
            continue
        reference = load_rgb_neg1(ref_path, image_size)
        ref_tissue, ref_nuclei = _load_mask_pair(ref_tissue_path, ref_nuclei_path, image_size)
        references.append(reference)
        reference_conds.append(_make_condition(reference, ref_tissue, ref_nuclei))
        reference_regions.append(tissue_nuclei_region_labels(ref_tissue, ref_nuclei, label_mode=region_label_mode))
        reference_tissues.append(ref_tissue)
        reference_nuclei.append(ref_nuclei)
    if not references:
        return None
    return {
        "title": root.name,
        "target_i0": target_i0,
        "target": target,
        "target_cond": torch.stack([target_cond] * len(references), dim=0),
        "target_region": torch.stack([target_region] * len(references), dim=0),
        "target_tissue_mask": torch.stack([target_tissue] * len(references), dim=0),
        "target_nuclei_mask": torch.stack([target_nuclei] * len(references), dim=0),
        "references": references,
        "reference_cond": torch.stack(reference_conds, dim=0),
        "reference_region": torch.stack(reference_regions, dim=0),
        "reference_tissue_mask": torch.stack(reference_tissues, dim=0),
        "reference_nuclei_mask": torch.stack(reference_nuclei, dim=0),
    }


def _record_path(dataset: I0ReferenceTextureDataset, record: dict[str, Any], field: str) -> Path:
    return resolve_path(record[field], metadata_root=dataset.metadata_root)


def _record_mask_stats(
    dataset: I0ReferenceTextureDataset,
    record: dict[str, Any],
    image_size: int,
) -> tuple[torch.Tensor, float] | None:
    try:
        tissue = load_label_mask(_record_path(dataset, record, "target_tissue_mask"), image_size)
        nuclei = remap_nuclei_mask(load_label_mask(_record_path(dataset, record, "target_nuclei_mask"), image_size))
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None
    labels = tissue.squeeze(0).long().clamp(min=0, max=15).flatten()
    hist = torch.bincount(labels, minlength=16).float()
    hist = hist / hist.sum().clamp_min(1.0)
    nuclei_fraction = float((nuclei != 0).float().mean().item())
    return hist, nuclei_fraction


def _make_record_ood_probe(
    dataset: I0ReferenceTextureDataset,
    *,
    target_index: int,
    reference_indices: list[int],
    image_size: int,
    region_label_mode: str,
    i0_cache_dir: str | Path | None,
) -> dict[str, Any] | None:
    target_record = dataset.records[target_index]
    if i0_cache_dir is None:
        return None
    target_i0_path = i0_cache_path(i0_cache_dir, target_record, target_index)
    if not target_i0_path.exists():
        return None
    try:
        target_i0 = load_rgb_neg1(target_i0_path, image_size)
        target = load_rgb_neg1(_record_path(dataset, target_record, "target_image"), image_size)
        target_tissue, target_nuclei = _load_mask_pair(
            _record_path(dataset, target_record, "target_tissue_mask"),
            _record_path(dataset, target_record, "target_nuclei_mask"),
            image_size,
        )
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None
    target_cond = _make_condition(target_i0, target_tissue, target_nuclei)
    target_region = tissue_nuclei_region_labels(target_tissue, target_nuclei, label_mode=region_label_mode)
    references = []
    reference_conds = []
    reference_regions = []
    reference_tissues = []
    reference_nuclei = []
    for ref_index in reference_indices[:5]:
        ref_record = dataset.records[ref_index]
        try:
            reference = load_rgb_neg1(_record_path(dataset, ref_record, "reference_image"), image_size)
            ref_tissue, ref_nuclei = _load_mask_pair(
                _record_path(dataset, ref_record, "reference_tissue_mask"),
                _record_path(dataset, ref_record, "reference_nuclei_mask"),
                image_size,
            )
        except (FileNotFoundError, KeyError, OSError, ValueError):
            continue
        references.append(reference)
        reference_conds.append(_make_condition(reference, ref_tissue, ref_nuclei))
        reference_regions.append(tissue_nuclei_region_labels(ref_tissue, ref_nuclei, label_mode=region_label_mode))
        reference_tissues.append(ref_tissue)
        reference_nuclei.append(ref_nuclei)
    if not references:
        return None
    return {
        "title": f"val_target_{target_index}",
        "target_i0": target_i0,
        "target": target,
        "target_cond": torch.stack([target_cond] * len(references), dim=0),
        "target_region": torch.stack([target_region] * len(references), dim=0),
        "target_tissue_mask": torch.stack([target_tissue] * len(references), dim=0),
        "target_nuclei_mask": torch.stack([target_nuclei] * len(references), dim=0),
        "references": references,
        "reference_cond": torch.stack(reference_conds, dim=0),
        "reference_region": torch.stack(reference_regions, dim=0),
        "reference_tissue_mask": torch.stack(reference_tissues, dim=0),
        "reference_nuclei_mask": torch.stack(reference_nuclei, dim=0),
    }


def prepare_ood_probes(
    *,
    val_dataset: I0ReferenceTextureDataset | None,
    image_size: int,
    region_label_mode: str,
    i0_cache_dir: str | Path | None,
    num_panels: int,
    seed: int,
) -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    for raw_root in DEFAULT_OOD_PROBE_ROOTS:
        if len(probes) >= int(num_panels):
            break
        probe = _load_existing_ood_probe(Path(raw_root), image_size=image_size, region_label_mode=region_label_mode)
        if probe is not None:
            probes.append(probe)
    if val_dataset is None or len(probes) >= int(num_panels):
        return probes[: int(num_panels)]

    rng = random.Random(int(seed))
    candidate_indices = list(range(len(val_dataset.records)))
    rng.shuffle(candidate_indices)
    candidate_indices = candidate_indices[: min(96, len(candidate_indices))]
    stats: dict[int, tuple[torch.Tensor, float]] = {}
    for idx in candidate_indices:
        stat = _record_mask_stats(val_dataset, val_dataset.records[idx], image_size)
        if stat is not None:
            stats[idx] = stat
    for target_index in list(stats.keys()):
        if len(probes) >= int(num_panels):
            break
        target_hist, target_nuclei = stats[target_index]
        scored = []
        for ref_index, (ref_hist, ref_nuclei) in stats.items():
            if ref_index == target_index:
                continue
            score = float(torch.abs(target_hist - ref_hist).sum().item()) + abs(target_nuclei - ref_nuclei)
            scored.append((score, ref_index))
        scored.sort(reverse=True)
        reference_indices = [idx for _, idx in scored[:5]]
        probe = _make_record_ood_probe(
            val_dataset,
            target_index=target_index,
            reference_indices=reference_indices,
            image_size=image_size,
            region_label_mode=region_label_mode,
            i0_cache_dir=i0_cache_dir,
        )
        if probe is not None:
            probes.append(probe)
    return probes[: int(num_panels)]


@torch.no_grad()
def save_ood_diagnose(
    *,
    model: torch.nn.Module,
    probes: list[dict[str, Any]],
    device: torch.device,
    output_dir: Path,
    epoch: int,
    mixed_precision: str,
    ref_trust_gate: bool,
    ref_fallback_scale: float,
    ref_soft_context_scale: float,
    ref_nuclei_context_scale: float,
    ref_soft_context_radius: int,
    matched_tissue_trust_floor: float = 0.0,
    matched_nuclei_trust_floor: float = 0.0,
) -> Path | None:
    if not probes:
        return None
    epoch_dir = output_dir / "ood_diagnose" / f"epoch{epoch + 1:04d}"
    panel_paths = []
    identity_metrics = []
    model.eval()
    for probe_index, probe in enumerate(probes):
        target_cond = probe["target_cond"].to(device)
        reference_cond = probe["reference_cond"].to(device)
        target_region = probe["target_region"].to(device)
        reference_region = probe["reference_region"].to(device)
        target_tissue_mask = probe["target_tissue_mask"].to(device)
        target_nuclei_mask = probe["target_nuclei_mask"].to(device)
        reference_tissue_mask = probe["reference_tissue_mask"].to(device)
        reference_nuclei_mask = probe["reference_nuclei_mask"].to(device)
        trust_map = None
        if ref_trust_gate:
            trust_map, _ = build_reference_trust_map(
                target_region,
                reference_region,
                fallback_scale=ref_fallback_scale,
                matched_tissue_floor=matched_tissue_trust_floor,
                matched_nuclei_floor=matched_nuclei_trust_floor,
            )
            trust_map = trust_map.to(device)
        with autocast_context(device, mixed_precision):
            pred = model(
                target_cond,
                reference_cond,
                target_region=target_region,
                reference_region=reference_region,
                target_trust_map=trust_map,
                ref_fallback_scale=ref_fallback_scale if ref_trust_gate else 1.0,
                ref_soft_context_scale=ref_soft_context_scale,
                ref_nuclei_context_scale=ref_nuclei_context_scale,
                ref_soft_context_radius=ref_soft_context_radius,
                target_tissue_mask=target_tissue_mask,
                target_nuclei_mask=target_nuclei_mask,
                reference_tissue_mask=reference_tissue_mask,
                reference_nuclei_mask=reference_nuclei_mask,
            )
        panel_path = save_ood_panel(
            output_path=epoch_dir / f"probe_{probe_index:02d}_panel.png",
            target_i0=probe["target_i0"],
            target=probe["target"],
            references=probe["references"],
            outputs=[item.cpu() for item in pred],
            title=str(probe.get("title", f"probe {probe_index}")),
        )
        panel_paths.append(panel_path)
        identity_metrics.append(
            compute_identity_metrics(
                target_i0=probe["target_i0"],
                target=probe["target"],
                references=probe["references"],
                outputs=[item.detach().float().cpu() for item in pred],
                target_tissue_mask=probe["target_tissue_mask"][0],
                target_nuclei_mask=probe["target_nuclei_mask"][0],
                reference_tissue_masks=[item for item in probe["reference_tissue_mask"]],
                reference_nuclei_masks=[item for item in probe["reference_nuclei_mask"]],
            )
        )
    save_identity_metrics(identity_metrics, epoch_dir / "identity_metrics.json")
    return save_ood_summary_grid(panel_paths, epoch_dir / "summary_grid.png")


def select_eval_indices(length: int, count: int, seed: int) -> list[int]:
    if count <= 0 or count >= length:
        return list(range(length))
    indices = list(range(length))
    random.Random(seed).shuffle(indices)
    return indices[:count]


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def set_requires_grad(model: torch.nn.Module | None, value: bool) -> None:
    if model is None:
        return
    for parameter in model.parameters():
        parameter.requires_grad_(value)


def configure_trainable_scope(
    model: torch.nn.Module,
    scope: str,
) -> list[str]:
    """Freeze the epoch25 baseline except for an explicitly selected scope."""

    normalized = str(scope).strip().lower()
    if normalized in {"all", "full", "full_discriminative"}:
        set_requires_grad(model, True)
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if normalized == "highres_decoder":
        prefixes = ("up1.", "up0.", "out.")
    elif normalized == "highres_cross4_qproj":
        prefixes = ("up1.", "up0.", "out.", "cross_4.to_q.", "cross_4.proj.")
    elif normalized == "steered_cross4":
        prefixes = (
            "up1.",
            "up0.",
            "out.",
            "cross_4.",
            "ref_in.",
            "ref_down1.",
            "ref_down2.",
        )
    elif normalized == "steered_cross4_cross8":
        prefixes = (
            "up1.",
            "up0.",
            "out.",
            "cross_4.",
            "cross_8.",
            "ref_in.",
            "ref_down1.",
            "ref_down2.",
            "ref_down3.",
        )
    elif normalized == "steered_full_pyramid":
        prefixes = (
            "up1.",
            "up0.",
            "out.",
            "steering_cross_1.",
            "steering_cross_2.",
            "cross_4.",
            "cross_8.",
            "cross_16.",
            "ref_in.",
            "ref_down1.",
            "ref_down2.",
            "ref_down3.",
            "ref_down4.",
        )
    else:
        raise ValueError(
            "trainable scope must be 'all', 'highres_decoder', or "
            "'highres_cross4_qproj', 'steered_cross4', "
            "'steered_cross4_cross8', or 'steered_full_pyramid'"
        )
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(prefixes)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise RuntimeError(f"{normalized} scope selected no parameters")
    return trainable


def is_detail_continuation_scope(scope: str) -> bool:
    return str(scope).strip().lower() in {
        "full_discriminative",
        "highres_decoder",
        "highres_cross4_qproj",
        "steered_cross4",
        "steered_cross4_cross8",
        "steered_full_pyramid",
    }


def build_continuation_optimizer_groups(
    model: torch.nn.Module,
    *,
    scope: str,
    highres_lr: float,
    cross4_lr: float,
) -> list[dict[str, Any]]:
    normalized = str(scope).strip().lower()
    if not is_detail_continuation_scope(normalized):
        raise ValueError(f"{scope!r} is not a detail continuation scope")

    highres_parameters = []
    cross4_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(
            ("up1.", "up0.", "out.", "steering_cross_1.", "steering_cross_2.")
        ):
            highres_parameters.append(parameter)
        elif name.startswith(
            (
                "cross_4.",
                "cross_8.",
                "cross_16.",
                "ref_in.",
                "ref_down1.",
                "ref_down2.",
                "ref_down3.",
                "ref_down4.",
            )
        ):
            cross4_parameters.append(parameter)
        else:
            raise RuntimeError(f"unexpected trainable continuation parameter: {name}")
    if not highres_parameters:
        raise RuntimeError("detail continuation selected no high-resolution parameters")

    groups: list[dict[str, Any]] = [
        {"params": highres_parameters, "lr": float(highres_lr), "group_name": "highres"}
    ]
    if normalized in {
        "highres_cross4_qproj",
        "steered_cross4",
        "steered_cross4_cross8",
        "steered_full_pyramid",
    }:
        if not cross4_parameters:
            raise RuntimeError(f"{normalized} selected no cross4/reference parameters")
        groups.append(
            {
                "params": cross4_parameters,
                "lr": float(cross4_lr),
                "group_name": (
                    "cross4_steering"
                    if normalized
                    in {"steered_cross4", "steered_cross4_cross8", "steered_full_pyramid"}
                    else "cross4_qproj"
                ),
            }
        )
    elif cross4_parameters:
        raise RuntimeError("highres_decoder unexpectedly selected cross_4 parameters")
    return groups


def build_full_discriminative_optimizer_groups(
    model: torch.nn.Module,
    *,
    backbone_lr: float,
    identity_lr: float,
    midcross_lr: float,
    cross4_lr: float,
    highres_lr: float,
) -> list[dict[str, Any]]:
    """Group a full continuation by behavioral risk and spatial resolution."""

    grouped: dict[str, list[torch.nn.Parameter]] = {
        "backbone": [],
        "identity": [],
        "cross_midcoarse": [],
        "cross4": [],
        "highres": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise RuntimeError(f"full_discriminative found frozen parameter: {name}")
        if name.startswith(
            ("up1.", "up0.", "out.", "steering_cross_1.", "steering_cross_2.")
        ):
            group_name = "highres"
        elif name.startswith("cross_4."):
            group_name = "cross4"
        elif name.startswith(("cross_8.", "cross_16.")):
            group_name = "cross_midcoarse"
        elif name.startswith("identity_adapter."):
            group_name = "identity"
        else:
            group_name = "backbone"
        grouped[group_name].append(parameter)

    learning_rates = {
        "backbone": float(backbone_lr),
        "identity": float(identity_lr),
        "cross_midcoarse": float(midcross_lr),
        "cross4": float(cross4_lr),
        "highres": float(highres_lr),
    }
    missing = [name for name, parameters in grouped.items() if not parameters]
    if missing:
        raise RuntimeError(f"full_discriminative selected empty optimizer groups: {missing}")
    return [
        {
            "params": grouped[name],
            "lr": learning_rates[name],
            "group_name": name,
        }
        for name in ("backbone", "identity", "cross_midcoarse", "cross4", "highres")
    ]


def main() -> int:
    args = parse_args()
    i0_orientation_window_sizes = parse_positive_int_tuple(
        args.i0_orientation_window_sizes,
        name="--i0-orientation-window-sizes",
    )
    i0_orientation_window_strides = parse_positive_int_tuple(
        args.i0_orientation_window_strides,
        name="--i0-orientation-window-strides",
    )
    if len(i0_orientation_window_sizes) != len(i0_orientation_window_strides):
        raise ValueError(
            "--i0-orientation-window-sizes and --i0-orientation-window-strides "
            "must contain the same number of values"
        )
    if any(value > int(args.image_size) for value in i0_orientation_window_sizes):
        raise ValueError("I0 orientation window sizes cannot exceed --image-size")
    cross4_steering_angles = parse_float_tuple(
        args.cross4_steering_angles,
        name="--cross4-steering-angles",
    )
    if abs(cross4_steering_angles[0]) > 1.0e-6:
        raise ValueError("--cross4-steering-angles must begin with 0")
    if any(value < 0.0 or value >= 180.0 for value in cross4_steering_angles):
        raise ValueError("--cross4-steering-angles values must be in [0, 180)")
    if len(set(cross4_steering_angles)) != len(cross4_steering_angles):
        raise ValueError("--cross4-steering-angles values must be unique")
    cross4_steering_scales = tuple(
        value.strip()
        for value in str(args.cross4_steering_scales).split(",")
        if value.strip()
    )
    valid_steering_scales = {"1/1", "1/2", "1/4", "1/8", "1/16"}
    if not cross4_steering_scales or any(
        value not in valid_steering_scales for value in cross4_steering_scales
    ):
        raise ValueError(
            "--cross4-steering-scales must contain only 1/1, 1/2, 1/4, 1/8, 1/16"
        )
    if set(cross4_steering_scales).intersection({"1/1", "1/2", "1/16"}) and not bool(
        args.full_pyramid_texture_steering
    ):
        raise ValueError(
            "1/1, 1/2, and 1/16 steering require --full-pyramid-texture-steering"
        )
    model_cross_scales = {
        value.strip() for value in str(args.cross_attn_scales).split(",") if value.strip()
    }
    model_required_scales = set(cross4_steering_scales).intersection(
        {"1/4", "1/8", "1/16"}
    )
    if args.cross4_texture_steering and not model_required_scales.issubset(model_cross_scales):
        raise ValueError("texture steering scales must be enabled in --cross-attn-scales")
    if not 0.0 <= float(args.cross4_steering_minimum_strength) <= 1.0:
        raise ValueError("--cross4-steering-minimum-strength must be in [0, 1]")
    if not 0.0 <= float(args.cross4_steering_minimum_support) <= 1.0:
        raise ValueError("--cross4-steering-minimum-support must be in [0, 1]")
    if int(args.cross4_steering_local_bins) < 8:
        raise ValueError("--cross4-steering-local-bins must be at least 8")
    if float(args.cross4_steering_local_kappa) <= 0.0:
        raise ValueError("--cross4-steering-local-kappa must be positive")
    steering_gains = (
        args.cross1_steering_gain,
        args.cross2_steering_gain,
        args.cross4_steering_gain,
        args.cross8_steering_gain,
        args.cross16_steering_gain,
    )
    if any(float(value) <= 0.0 for value in steering_gains):
        raise ValueError("texture steering gains must be positive")
    if int(args.steering_highres_reference_size) <= 0:
        raise ValueError("--steering-highres-reference-size must be positive")
    device, local_rank, rank, world_size = setup_distributed(args)
    seed_everything(args.seed, rank)
    if args.lazy_generate_i0:
        if not args.i0_cache_dir:
            raise ValueError("--lazy-generate-i0 requires --i0-cache-dir")
        if args.augment_flips:
            raise ValueError(
                "--lazy-generate-i0 cannot be combined with --augment-flips. "
                "First train/fill the cache without flips, then resume with augmentation."
            )

    output_dir = Path(args.output_dir)
    if rank == 0:
        (output_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        (output_dir / "samples").mkdir(parents=True, exist_ok=True)
        (output_dir / "eval").mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2),
            encoding="utf8",
        )

    dataset = I0ReferenceTextureDataset(
        args.metadata,
        image_size=args.image_size,
        i0_field=args.i0_field,
        i0_cache_dir=args.i0_cache_dir,
        allow_missing_i0=args.lazy_generate_i0,
        metadata_root=args.metadata_root,
        max_samples=args.max_samples,
        region_label_mode=args.region_label_mode,
        augment_flips=args.augment_flips,
        split="train",
    )
    sampler = None
    if args.hard_pair_sampling:
        sampling_weights = build_difficulty_sampling_weights(
            dataset.records,
            full_mass=args.hard_pair_full_mass,
            hard_mass=args.hard_pair_hard_mass,
        )
        sampler = (
            DistributedWeightedSampler(
                sampling_weights,
                num_replicas=world_size,
                rank=rank,
                seed=args.seed,
            )
            if world_size > 1
            else WeightedRandomSampler(
                sampling_weights,
                num_samples=len(dataset),
                replacement=True,
            )
        )
    elif world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    eval_loader = None
    val_dataset = None
    rotation_monitor_batch = None
    if rank == 0 and args.val_metadata:
        val_dataset = I0ReferenceTextureDataset(
            args.val_metadata,
            image_size=args.image_size,
            i0_field=args.i0_field,
            i0_cache_dir=args.val_i0_cache_dir or args.i0_cache_dir,
            allow_missing_i0=args.lazy_generate_i0,
            metadata_root=args.metadata_root,
            max_samples=None,
            region_label_mode=args.region_label_mode,
            augment_flips=False,
            split="val",
        )
        eval_indices = select_eval_indices(
            len(val_dataset),
            int(args.eval_num_samples),
            int(args.eval_seed),
        )
        eval_subset = Subset(val_dataset, eval_indices)
        eval_loader = DataLoader(
            eval_subset,
            batch_size=max(1, int(args.eval_batch_size)),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        print(f"fixed eval metadata indices: {eval_indices}")
        if int(args.rotation_monitor_every_steps) > 0:
            if world_size > 1:
                raise ValueError("rotation step monitor currently requires single-GPU training")
            rotation_monitor_batch = next(iter(eval_loader))

    ood_probes: list[dict[str, Any]] = []
    if rank == 0 and args.ood_diagnose_every_epochs > 0:
        try:
            ood_probes = prepare_ood_probes(
                val_dataset=val_dataset,
                image_size=int(args.image_size),
                region_label_mode=args.region_label_mode,
                i0_cache_dir=args.val_i0_cache_dir or args.i0_cache_dir,
                num_panels=int(args.ood_diagnose_num_panels),
                seed=int(args.eval_seed),
            )
            print(f"prepared {len(ood_probes)} OOD diagnose probes")
        except Exception as exc:
            print(f"[ood-diagnose] failed to prepare probes: {exc}")

    in_ch = 3 + 16 + 6
    region_condition_channels = in_ch - 3
    model = Pix2PixCrossAttnUNet(
        in_ch=in_ch,
        out_ch=3,
        base=args.base_channels,
        num_heads=args.num_heads,
        use_region_mask=not args.no_region_mask,
        residual_output=not args.no_residual_output,
        cross_attn_scales=args.cross_attn_scales,
        upsample_mode=args.upsample_mode,
        use_wsi_identity=bool(args.wsi_identity_adapter),
        identity_gamma_max=args.identity_gamma_max,
        identity_gamma_init=args.identity_gamma_init,
        identity_min_tissue_pixels=args.identity_min_tissue_pixels,
        identity_min_nuclei_pixels=args.identity_min_nuclei_pixels,
        full_pyramid_texture_steering=bool(args.full_pyramid_texture_steering),
        steering_highres_reference_size=int(args.steering_highres_reference_size),
    ).to(device)
    trainable_names = configure_trainable_scope(model, args.trainable_scope)
    detail_continuation_scope = is_detail_continuation_scope(args.trainable_scope)
    if rank == 0:
        print(f"model trainable params: {model_parameter_count(model):,}")
        print(f"decoder upsample mode: {args.upsample_mode} + conv")
        print(
            f"trainable scope: {args.trainable_scope} "
            f"({len(trainable_names)} tensors)"
        )

    discriminator = None
    d_optimizer = None
    if args.lambda_adv > 0.0:
        discriminator = RegionAwarePatchDiscriminator(
            image_channels=3,
            condition_channels=region_condition_channels,
            base_channels=args.d_base_channels,
            max_channels=args.d_max_channels,
            num_layers=args.d_num_layers,
            spectral_norm=not args.no_d_spectral_norm,
        ).to(device)
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.d_lr if args.d_lr is not None else args.lr,
            betas=(0.5, 0.999),
            weight_decay=args.d_weight_decay,
        )
        if rank == 0:
            print(
                "region-aware PatchGAN enabled: "
                f"lambda_adv={args.lambda_adv:g} warmup={args.adv_warmup_steps} "
                f"mask={args.adv_mask_mode} params={model_parameter_count(discriminator):,}"
            )

    criterion = Pix2PixTransferLoss(
        lambda_l1=args.lambda_l1,
        lambda_perc=args.lambda_perc,
        lambda_gram=args.lambda_gram,
        lambda_contextual=args.lambda_contextual,
        vgg_weights=args.vgg_weights,
        content_layers=args.content_layers,
        gram_layers=args.gram_layers,
        contextual_layers=args.contextual_layers,
        texture_min_pixels=args.texture_min_pixels,
        contextual_max_samples=args.contextual_max_samples,
        contextual_temperature=args.contextual_temperature,
        normalize_losses=not args.no_loss_normalization,
        normalization_decay=args.loss_normalization_decay,
        normalization_steps=args.loss_normalization_steps,
        l1_blur_sigma=args.l1_blur_sigma,
        boundary_feather_radius=args.boundary_feather_radius,
        lambda_boundary_hf=args.lambda_boundary_hf,
        lambda_lowtrust_hf=args.lambda_lowtrust_hf,
        lambda_identity_od=args.lambda_identity_od if args.wsi_identity_adapter else 0.0,
        lambda_identity_feature=args.lambda_identity_feature if args.wsi_identity_adapter else 0.0,
        lambda_identity_band=args.lambda_identity_band if args.wsi_identity_adapter else 0.0,
        lambda_identity_rank=args.lambda_identity_rank if args.wsi_identity_adapter else 0.0,
        cross_lambda_identity_od=args.cross_lambda_identity_od if args.wsi_identity_adapter else 0.0,
        cross_lambda_identity_feature=(
            args.cross_lambda_identity_feature if args.wsi_identity_adapter else 0.0
        ),
        cross_lambda_identity_band=args.cross_lambda_identity_band if args.wsi_identity_adapter else 0.0,
        cross_lambda_identity_rank=args.cross_lambda_identity_rank if args.wsi_identity_adapter else 0.0,
        cross_lambda_gram=args.cross_lambda_gram,
        cross_lambda_contextual=args.cross_lambda_contextual,
        lambda_structure_gray=args.lambda_structure_gray,
        lambda_structure_edge=args.lambda_structure_edge,
        identity_rank_margin=args.identity_rank_margin,
        identity_min_tissue_pixels=args.identity_min_tissue_pixels,
        identity_min_nuclei_pixels=args.identity_min_nuclei_pixels,
    ).to(device)
    if str(args.trainable_scope).strip().lower() == "full_discriminative":
        optimizer = torch.optim.AdamW(
            build_full_discriminative_optimizer_groups(
                model,
                backbone_lr=args.backbone_lr,
                identity_lr=args.identity_lr,
                midcross_lr=args.midcross_lr,
                cross4_lr=args.cross4_lr,
                highres_lr=args.highres_lr,
            ),
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay,
        )
    elif detail_continuation_scope:
        optimizer = torch.optim.AdamW(
            build_continuation_optimizer_groups(
                model,
                scope=args.trainable_scope,
                highres_lr=args.highres_lr,
                cross4_lr=args.cross4_lr,
            ),
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay,
        )
    elif args.wsi_identity_adapter:
        identity_parameters = []
        backbone_parameters = []
        for name, parameter in model.named_parameters():
            (identity_parameters if name.startswith("identity_adapter.") else backbone_parameters).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": args.backbone_lr, "group_name": "backbone"},
                {"params": identity_parameters, "lr": args.identity_lr, "group_name": "identity"},
            ],
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.5, 0.999),
            weight_decay=args.weight_decay,
        )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.mixed_precision == "fp16" and device.type == "cuda"
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        if str(args.trainable_scope).strip().lower() == "steered_full_pyramid":
            incompatible = model.load_state_dict(ckpt["model"], strict=False)
            allowed_missing_prefixes = ("steering_cross_1.", "steering_cross_2.")
            invalid_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(allowed_missing_prefixes)
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "full-pyramid resume only permits new high-resolution steering modules; "
                    f"missing={invalid_missing} unexpected={incompatible.unexpected_keys}"
                )
            if rank == 0:
                print(
                    "steered_full_pyramid loaded legacy generator; "
                    f"initialized {len(incompatible.missing_keys)} high-resolution parameters"
                )
        elif detail_continuation_scope:
            model.load_state_dict(ckpt["model"], strict=True)
            if rank == 0:
                print(f"{args.trainable_scope} continuation strictly loaded epoch25 generator")
        elif args.wsi_identity_adapter:
            incompatible = model.load_state_dict(ckpt["model"], strict=False)
            invalid_missing = [
                key for key in incompatible.missing_keys if not key.startswith("identity_adapter.")
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Identity resume only permits missing identity_adapter.* keys; "
                    f"missing={invalid_missing} unexpected={incompatible.unexpected_keys}"
                )
            if rank == 0:
                print(
                    "identity resume loaded old generator; "
                    f"initialized {len(incompatible.missing_keys)} identity parameters"
                )
        else:
            model.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
        if "loss_normalizer" in ckpt:
            criterion.normalizer.load_state_dict(ckpt["loss_normalizer"])
        if "identity_loss_normalizer" in ckpt:
            criterion.identity_normalizer.load_state_dict(ckpt["identity_loss_normalizer"])
        if discriminator is not None:
            if "discriminator" in ckpt:
                discriminator.load_state_dict(ckpt["discriminator"], strict=True)
            elif rank == 0:
                print("resume checkpoint has no discriminator state; PatchGAN starts fresh")
            if d_optimizer is not None and "d_optimizer" in ckpt and not args.wsi_identity_adapter:
                d_optimizer.load_state_dict(ckpt["d_optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        if rank == 0:
            print(f"resumed from {args.resume} at epoch={start_epoch} step={global_step}")

    identity_resume_step = int(global_step)
    continuation_start_step = int(global_step)
    baseline_teacher = None
    if detail_continuation_scope and (
        args.lambda_baseline_consistency > 0.0
        or args.lambda_anchor_teacher_consistency > 0.0
        or args.lambda_i0_texture_energy > 0.0
        or args.rotation_monitor_every_steps > 0
    ):
        baseline_teacher = copy.deepcopy(model).eval()
        set_requires_grad(baseline_teacher, False)
        if rank == 0:
            print("created frozen epoch25 baseline teacher")

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
        if discriminator is not None:
            discriminator = DistributedDataParallel(discriminator, device_ids=[local_rank])

    i0_generator = (
        LazyI0Generator(args, device, rank=rank)
        if args.lazy_generate_i0
        else None
    )
    detail_generator = torch.Generator(device="cpu")
    detail_generator.manual_seed(int(args.seed) + 100003 * int(rank))
    reference_rotation_generator = torch.Generator(device="cpu")
    reference_rotation_generator.manual_seed(int(args.seed) + 200003 * int(rank))
    frozen_gamma_values = {
        name: float(parameter.detach().item())
        for name, parameter in unwrap_model(model).named_parameters()
        if name.endswith("gamma")
    }
    pilot_stop = False
    unsafe_rotation_monitor_count = 0

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            if (
                int(args.max_continuation_steps) > 0
                and global_step - continuation_start_step >= int(args.max_continuation_steps)
            ):
                pilot_stop = True
                break
            if i0_generator is not None:
                batch = i0_generator.fill_batch(batch)
            target_cond = batch["target_cond"].to(device, non_blocking=True)
            reference_cond = batch["reference_cond"].to(device, non_blocking=True)
            target = batch["target_image"].to(device, non_blocking=True)
            target_region = batch["target_region"].to(device, non_blocking=True)
            reference_region = batch["reference_region"].to(device, non_blocking=True)
            target_tissue_mask = batch["target_tissue_mask"].to(device, non_blocking=True)
            target_nuclei_mask = batch["target_nuclei_mask"].to(device, non_blocking=True)
            reference_tissue_mask = batch["reference_tissue_mask"].to(device, non_blocking=True)
            reference_nuclei_mask = batch["reference_nuclei_mask"].to(device, non_blocking=True)
            original_reference = batch["reference_image"].to(device, non_blocking=True)
            case_ids = [str(value) for value in batch.get("case_id", [])]
            permutation = build_cross_wsi_permutation(case_ids) if len(case_ids) > 1 else None
            cross_probability = (
                float(args.cross_wsi_style_prob)
                if args.cross_wsi_style_prob is not None
                else float(args.ref_mismatch_prob)
            )
            cross_requested = (
                cross_probability > 0.0
                and reference_cond.shape[0] > 1
                and random.random() < cross_probability
            )
            cross_wsi_active = bool(cross_requested and permutation is not None)
            cross_wsi_fallback = bool(cross_requested and permutation is None)
            negative_reference = None
            negative_reference_tissue_mask = None
            negative_reference_nuclei_mask = None
            if permutation is not None:
                permutation_index = torch.tensor(permutation, device=device, dtype=torch.long)
                if cross_wsi_active:
                    negative_reference = original_reference
                    negative_reference_tissue_mask = reference_tissue_mask
                    negative_reference_nuclei_mask = reference_nuclei_mask
                    reference_cond = reference_cond.index_select(0, permutation_index)
                    reference_region = reference_region.index_select(0, permutation_index)
                    reference_tissue_mask = reference_tissue_mask.index_select(0, permutation_index)
                    reference_nuclei_mask = reference_nuclei_mask.index_select(0, permutation_index)
                else:
                    negative_reference = original_reference.index_select(0, permutation_index)
                    negative_reference_tissue_mask = reference_tissue_mask.index_select(0, permutation_index)
                    negative_reference_nuclei_mask = reference_nuclei_mask.index_select(0, permutation_index)

            continuation_step = max(0, global_step - continuation_start_step)
            reference_rotation_probability = ramped_rotation_probability(
                args.main_ref_random_rotation_prob,
                step=continuation_step,
                ramp_steps=args.main_ref_random_rotation_ramp_steps,
            )
            reference_rotation_angle = sample_continuous_rotation_angle(
                probability=reference_rotation_probability,
                min_degrees=args.main_ref_random_rotation_min_degrees,
                max_degrees=args.main_ref_random_rotation_max_degrees,
                generator=reference_rotation_generator,
            )
            reference_rotation_active = reference_rotation_angle != 0.0
            if reference_rotation_active:
                rotated_reference = rotate_reference_bundle(
                    reference_cond,
                    reference_region,
                    reference_tissue_mask,
                    reference_nuclei_mask,
                    angles_degrees=reference_rotation_angle,
                )
                reference_cond = rotated_reference.reference_cond
                reference_region = rotated_reference.reference_region
                reference_tissue_mask = rotated_reference.reference_tissue_mask
                reference_nuclei_mask = rotated_reference.reference_nuclei_mask
            active_reference = reference_cond[:, :3]
            region_condition = target_cond[:, 3:].detach()
            trust_map = None
            trust_logs = {
                "mean_trust": 1.0,
                "low_trust_fraction": 0.0,
                "unmatched_regions": 0.0,
            }
            if args.ref_trust_gate:
                trust_map, trust_logs = build_reference_trust_map(
                    target_region,
                    reference_region,
                    fallback_scale=args.ref_fallback_scale,
                    matched_tissue_floor=args.matched_tissue_trust_floor,
                    matched_nuclei_floor=args.matched_nuclei_trust_floor,
                )
                trust_map = trust_map.to(device=device, dtype=target_cond.dtype)

            clean_target_cond = target_cond
            corruption_mask = target_cond.new_zeros(
                (target_cond.shape[0], 1, target_cond.shape[-2], target_cond.shape[-1])
            )
            corruption_active = False
            rotated_i0_corruption_active = False
            corruption_sigma = target_cond.new_zeros(target_cond.shape[0])
            if reference_rotation_active and float(args.rotated_i0_dropout_prob) > 0.0:
                eligible = target_tissue_mask.ne(0) & target_nuclei_mask.eq(0)
                if trust_map is not None:
                    eligible = eligible & trust_map.ge(
                        float(args.target_orientation_min_trust)
                    )
                detail_dropout = apply_local_detail_dropout(
                    target_cond[:, :3],
                    target_region,
                    probability=float(args.rotated_i0_dropout_prob),
                    min_diameter=int(args.rotated_i0_dropout_min_diameter),
                    max_diameter=int(args.rotated_i0_dropout_max_diameter),
                    sigma_min=float(args.rotated_i0_dropout_sigma_min),
                    sigma_max=float(args.rotated_i0_dropout_sigma_max),
                    feather_radius=int(args.detail_dropout_feather_radius),
                    eligible_mask=eligible,
                    generator=detail_generator,
                )
                corruption_mask = detail_dropout.mask
                corruption_sigma = detail_dropout.sigma
                corruption_active = bool(detail_dropout.active.any().item())
                rotated_i0_corruption_active = corruption_active
                if corruption_active:
                    target_cond = target_cond.clone()
                    target_cond[:, :3] = detail_dropout.image
            elif not cross_wsi_active and float(args.detail_dropout_prob) > 0.0:
                detail_dropout = apply_local_detail_dropout(
                    target_cond[:, :3],
                    target_region,
                    probability=float(args.detail_dropout_prob),
                    min_diameter=int(args.detail_dropout_min_diameter),
                    max_diameter=int(args.detail_dropout_max_diameter),
                    sigma_min=float(args.detail_dropout_sigma_min),
                    sigma_max=float(args.detail_dropout_sigma_max),
                    feather_radius=int(args.detail_dropout_feather_radius),
                    generator=detail_generator,
                )
                corruption_mask = detail_dropout.mask
                corruption_sigma = detail_dropout.sigma
                corruption_active = bool(detail_dropout.active.any().item())
                if corruption_active:
                    target_cond = target_cond.clone()
                    target_cond[:, :3] = detail_dropout.image

            identity_warmup_active = bool(
                args.wsi_identity_adapter
                and not detail_continuation_scope
                and global_step - identity_resume_step < int(args.identity_warmup_steps)
            )
            if args.wsi_identity_adapter and not detail_continuation_scope:
                for group in optimizer.param_groups:
                    if group.get("group_name") == "backbone":
                        group["lr"] = 0.0 if identity_warmup_active else float(args.backbone_lr)
                    elif group.get("group_name") == "identity":
                        group["lr"] = float(args.identity_lr)
            cross4_rotation_weights = None
            steering_confidence = target_cond.new_zeros(())
            steering_raw_confidence = target_cond.new_zeros(())
            steering_active_fraction = target_cond.new_zeros(())
            steering_fallback_fraction = target_cond.new_zeros(())
            steering_mean_angle = target_cond.new_zeros(())
            steering_candidate_fractions = target_cond.new_zeros(
                len(cross4_steering_angles)
            )
            if args.cross4_texture_steering:
                steering_result = build_fine_texture_steering_weights(
                    clean_target_cond[:, :3],
                    reference_cond[:, :3],
                    target_tissue_mask=target_tissue_mask,
                    target_nuclei_mask=target_nuclei_mask,
                    reference_tissue_mask=reference_tissue_mask,
                    reference_nuclei_mask=reference_nuclei_mask,
                    candidate_angles_degrees=cross4_steering_angles,
                    smoothing_sigma=float(args.cross4_steering_smoothing_sigma),
                    min_coherence=float(args.cross4_steering_min_coherence),
                    min_relative_energy=float(args.cross4_steering_min_relative_energy),
                    min_resultant=float(args.cross4_steering_min_resultant),
                    minimum_strength=float(args.cross4_steering_minimum_strength),
                    minimum_support=float(args.cross4_steering_minimum_support),
                    temperature=float(args.cross4_steering_temperature),
                    reference_direction_mode=str(args.cross4_steering_reference_mode),
                    local_histogram_bins=int(args.cross4_steering_local_bins),
                    local_histogram_concentration=float(
                        args.cross4_steering_local_kappa
                    ),
                    boundary_exclusion_radius=int(args.i0_orientation_boundary_radius),
                    nuclei_exclusion_radius=int(args.i0_orientation_nuclei_radius),
                )
                cross4_rotation_weights = steering_result.weights
                steering_confidence = steering_result.mean_confidence
                steering_raw_confidence = steering_result.raw_mean_confidence
                steering_active_fraction = steering_result.active_fraction
                steering_fallback_fraction = steering_result.fallback_fraction
                steering_mean_angle = steering_result.mean_selected_angle_degrees
                steering_candidate_fractions = steering_result.candidate_fractions
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.mixed_precision):
                baseline_prediction = None
                detail_teacher_active = bool(
                    baseline_teacher is not None
                    and corruption_active
                    and not reference_rotation_active
                    and float(args.lambda_baseline_consistency) > 0.0
                )
                anchor_teacher_active = bool(
                    baseline_teacher is not None
                    and not corruption_active
                    and not reference_rotation_active
                    and float(args.lambda_anchor_teacher_consistency) > 0.0
                )
                energy_teacher_active = bool(
                    baseline_teacher is not None
                    and float(args.lambda_i0_texture_energy) > 0.0
                )
                if detail_teacher_active or anchor_teacher_active or energy_teacher_active:
                    teacher_target_cond = (
                        clean_target_cond
                        if detail_teacher_active or energy_teacher_active
                        else target_cond
                    )
                    with torch.inference_mode():
                        baseline_prediction = baseline_teacher(
                            teacher_target_cond,
                            reference_cond,
                            target_region=target_region,
                            reference_region=reference_region,
                            target_trust_map=trust_map,
                            ref_fallback_scale=args.ref_fallback_scale if args.ref_trust_gate else 1.0,
                            ref_soft_context_scale=args.ref_soft_context_scale,
                            ref_nuclei_context_scale=args.ref_nuclei_context_scale,
                            ref_soft_context_radius=args.ref_soft_context_radius,
                            target_tissue_mask=target_tissue_mask,
                            target_nuclei_mask=target_nuclei_mask,
                            reference_tissue_mask=reference_tissue_mask,
                            reference_nuclei_mask=reference_nuclei_mask,
                        )
                pred = model(
                    target_cond,
                    reference_cond,
                    target_region=target_region,
                    reference_region=reference_region,
                    target_trust_map=trust_map,
                    ref_fallback_scale=args.ref_fallback_scale if args.ref_trust_gate else 1.0,
                    ref_soft_context_scale=args.ref_soft_context_scale,
                    ref_nuclei_context_scale=args.ref_nuclei_context_scale,
                    ref_soft_context_radius=args.ref_soft_context_radius,
                    target_tissue_mask=target_tissue_mask,
                    target_nuclei_mask=target_nuclei_mask,
                    reference_tissue_mask=reference_tissue_mask,
                    reference_nuclei_mask=reference_nuclei_mask,
                    cross4_rotation_weights=cross4_rotation_weights,
                    cross4_rotation_angles=cross4_steering_angles,
                    texture_steering_scales=cross4_steering_scales,
                    cross4_steering_gain=float(args.cross4_steering_gain),
                    cross8_steering_gain=float(args.cross8_steering_gain),
                    cross16_steering_gain=float(args.cross16_steering_gain),
                    cross2_steering_gain=float(args.cross2_steering_gain),
                    cross1_steering_gain=float(args.cross1_steering_gain),
                )
                loss, logs = criterion(
                    pred,
                    target,
                    reference=active_reference,
                    target_region=target_region,
                    reference_region=reference_region,
                    boundary_region=target_tissue_mask,
                    target_tissue_mask=target_tissue_mask,
                    target_nuclei_mask=target_nuclei_mask,
                    reference_tissue_mask=reference_tissue_mask,
                    reference_nuclei_mask=reference_nuclei_mask,
                    negative_reference=negative_reference,
                    negative_reference_tissue_mask=negative_reference_tissue_mask,
                    negative_reference_nuclei_mask=negative_reference_nuclei_mask,
                    i0=clean_target_cond[:, :3],
                    trust_map=trust_map,
                    corruption_mask=corruption_mask,
                    reference_texture_scale=1.0,
                    l1_scale=(
                        args.rotated_ref_l1_scale
                        if reference_rotation_active and not cross_wsi_active
                        else 1.0
                    ),
                    content_scale=(
                        args.rotated_ref_content_scale
                        if reference_rotation_active and not cross_wsi_active
                        else 1.0
                    ),
                    gram_scale=(
                        args.rotated_ref_gram_scale if reference_rotation_active else 1.0
                    ),
                    contextual_scale=(
                        args.rotated_ref_contextual_scale if reference_rotation_active else 1.0
                    ),
                    training_mode="cross_wsi" if cross_wsi_active else "same_wsi",
                )

                anchor_teacher_consistency = pred.new_zeros(())
                if anchor_teacher_active and baseline_prediction is not None:
                    anchor_teacher_consistency = F.l1_loss(
                        pred,
                        baseline_prediction.detach(),
                    )
                    loss = (
                        loss
                        + float(args.lambda_anchor_teacher_consistency)
                        * anchor_teacher_consistency
                    )

                target_orientation = pred.new_zeros(())
                target_anisotropy = pred.new_zeros(())
                target_orientation_valid = pred.new_zeros(())
                target_orientation_coherence = pred.new_zeros(())
                if reference_rotation_active and (
                    float(args.lambda_target_orientation) > 0.0
                    or float(args.lambda_target_anisotropy) > 0.0
                ):
                    orientation_result = multiscale_target_orientation_loss(
                        pred,
                        target,
                        target_tissue_mask=target_tissue_mask,
                        target_nuclei_mask=target_nuclei_mask,
                        trust_map=trust_map,
                        min_coherence=float(args.target_orientation_min_coherence),
                        min_trust=float(args.target_orientation_min_trust),
                        boundary_exclusion_radius=int(
                            args.target_orientation_boundary_radius
                        ),
                        nuclei_exclusion_radius=int(args.target_orientation_nuclei_radius),
                    )
                    target_orientation = orientation_result.orientation
                    target_anisotropy = orientation_result.anisotropy
                    target_orientation_valid = orientation_result.valid_fraction
                    target_orientation_coherence = orientation_result.mean_coherence
                    loss = (
                        loss
                        + float(args.lambda_target_orientation) * target_orientation
                        + float(args.lambda_target_anisotropy) * target_anisotropy
                    )

                i0_window_orientation = pred.new_zeros(())
                i0_window_directionality = pred.new_zeros(())
                i0_window_valid = pred.new_zeros(())
                i0_window_angle_degrees = pred.new_zeros(())
                i0_window_resultant = pred.new_zeros(())
                pred_window_resultant = pred.new_zeros(())
                i0_residual_orientation = pred.new_zeros(())
                i0_residual_angle_degrees = pred.new_zeros(())
                i0_texture_energy = pred.new_zeros(())
                i0_texture_energy_ratio = pred.new_zeros(())
                i0_texture_under_floor = pred.new_zeros(())
                i0_texture_over_ceiling = pred.new_zeros(())
                i0_orientation_ramp = 1.0
                if int(args.i0_orientation_ramp_steps) > 0:
                    i0_orientation_ramp = min(
                        1.0,
                        float(continuation_step + 1)
                        / float(args.i0_orientation_ramp_steps),
                    )
                if (
                    float(args.lambda_i0_window_orientation) > 0.0
                    or float(args.lambda_i0_window_directionality) > 0.0
                ):
                    i0_orientation_result = windowed_i0_mean_orientation_loss(
                        pred,
                        clean_target_cond[:, :3],
                        target_tissue_mask=target_tissue_mask,
                        target_nuclei_mask=target_nuclei_mask,
                        trust_map=trust_map,
                        window_sizes=i0_orientation_window_sizes,
                        window_strides=i0_orientation_window_strides,
                        min_coherence=float(args.i0_orientation_min_coherence),
                        min_relative_energy=float(
                            args.i0_orientation_min_relative_energy
                        ),
                        min_valid_fraction=float(
                            args.i0_orientation_min_window_fraction
                        ),
                        min_resultant=float(args.i0_orientation_min_resultant),
                        directionality_floor_ratio=float(
                            args.i0_orientation_directionality_floor_ratio
                        ),
                        min_trust=float(args.i0_orientation_min_trust),
                        boundary_exclusion_radius=int(
                            args.i0_orientation_boundary_radius
                        ),
                        nuclei_exclusion_radius=int(args.i0_orientation_nuclei_radius),
                    )
                    i0_window_orientation = i0_orientation_result.direction
                    i0_window_directionality = i0_orientation_result.directionality
                    i0_window_valid = i0_orientation_result.valid_window_fraction
                    i0_window_angle_degrees = i0_orientation_result.mean_angle_degrees
                    i0_window_resultant = i0_orientation_result.mean_i0_resultant
                    pred_window_resultant = (
                        i0_orientation_result.mean_prediction_resultant
                    )
                    loss = loss + i0_orientation_ramp * (
                        float(args.lambda_i0_window_orientation)
                        * i0_window_orientation
                        + float(args.lambda_i0_window_directionality)
                        * i0_window_directionality
                    )
                if float(args.lambda_i0_residual_orientation) > 0.0:
                    residual_orientation_result = windowed_i0_mean_orientation_loss(
                        pred - clean_target_cond[:, :3].detach(),
                        clean_target_cond[:, :3],
                        target_tissue_mask=target_tissue_mask,
                        target_nuclei_mask=target_nuclei_mask,
                        trust_map=trust_map,
                        window_sizes=i0_orientation_window_sizes,
                        window_strides=i0_orientation_window_strides,
                        min_coherence=float(args.i0_orientation_min_coherence),
                        min_relative_energy=float(
                            args.i0_orientation_min_relative_energy
                        ),
                        min_valid_fraction=float(
                            args.i0_orientation_min_window_fraction
                        ),
                        min_resultant=float(args.i0_orientation_min_resultant),
                        directionality_floor_ratio=0.0,
                        min_trust=float(args.i0_orientation_min_trust),
                        boundary_exclusion_radius=int(
                            args.i0_orientation_boundary_radius
                        ),
                        nuclei_exclusion_radius=int(args.i0_orientation_nuclei_radius),
                    )
                    i0_residual_orientation = residual_orientation_result.direction
                    i0_residual_angle_degrees = (
                        residual_orientation_result.mean_angle_degrees
                    )
                    loss = loss + (
                        i0_orientation_ramp
                        * float(args.lambda_i0_residual_orientation)
                        * i0_residual_orientation
                    )
                if float(args.lambda_i0_texture_energy) > 0.0:
                    if baseline_prediction is None:
                        raise RuntimeError(
                            "I0 texture energy supervision requires the frozen epoch25 teacher"
                        )
                    energy_result = windowed_fine_texture_energy_floor_loss(
                        pred,
                        baseline_prediction,
                        target_tissue_mask=target_tissue_mask,
                        target_nuclei_mask=target_nuclei_mask,
                        trust_map=trust_map,
                        window_sizes=i0_orientation_window_sizes,
                        window_strides=i0_orientation_window_strides,
                        energy_floor_ratio=float(args.i0_texture_energy_floor_ratio),
                        energy_ceiling_ratio=float(args.i0_texture_energy_ceiling_ratio),
                        min_baseline_relative_energy=float(
                            args.i0_orientation_min_relative_energy
                        ),
                        min_valid_fraction=float(
                            args.i0_orientation_min_window_fraction
                        ),
                        min_trust=float(args.i0_orientation_min_trust),
                        boundary_exclusion_radius=int(
                            args.i0_orientation_boundary_radius
                        ),
                        nuclei_exclusion_radius=int(args.i0_orientation_nuclei_radius),
                    )
                    i0_texture_energy = energy_result.loss
                    i0_texture_energy_ratio = energy_result.mean_energy_ratio
                    i0_texture_under_floor = energy_result.under_floor_fraction
                    i0_texture_over_ceiling = energy_result.over_ceiling_fraction
                    loss = loss + (
                        i0_orientation_ramp
                        * float(args.lambda_i0_texture_energy)
                        * i0_texture_energy
                    )

                detail_fine = pred.new_zeros(())
                detail_mid = pred.new_zeros(())
                baseline_consistency = pred.new_zeros(())
                ref_orientation_consistency = pred.new_zeros(())
                ref_target_alignment = pred.new_zeros(())
                ref_rotation_style = pred.new_zeros(())
                ref_rotation_paired = False
                if corruption_active and not cross_wsi_active:
                    detail_fine, detail_mid = masked_multiband_detail_loss(
                        pred,
                        target,
                        corruption_mask,
                        target_region,
                    )
                    loss = (
                        loss
                        + float(args.lambda_detail_fine) * detail_fine
                        + float(args.lambda_detail_mid) * detail_mid
                    )
                    if baseline_prediction is not None and float(args.lambda_baseline_consistency) > 0.0:
                        expanded = F.max_pool2d(
                            corruption_mask.float(),
                            kernel_size=11,
                            stride=1,
                            padding=5,
                        ).clamp(0.0, 1.0)
                        outside = 1.0 - expanded
                        denominator = outside.sum() * pred.shape[1]
                        baseline_consistency = (
                            (pred - baseline_prediction.detach()).abs() * outside
                        ).sum() / denominator.clamp_min(1.0)
                        loss = loss + float(args.lambda_baseline_consistency) * baseline_consistency

                    ref_rotation_paired = (
                        float(args.reference_rotation_pair_prob) > 0.0
                        and random.random() < float(args.reference_rotation_pair_prob)
                    )
                    if ref_rotation_paired:
                        d4_codes = sample_nonzero_d4_codes(
                            reference_cond.shape[0],
                            generator=detail_generator,
                            device=reference_cond.device,
                        )
                        rotated_reference_cond = rotate_batch_d4(reference_cond, d4_codes)
                        rotated_reference_region = rotate_batch_d4(reference_region, d4_codes)
                        rotated_reference_tissue = rotate_batch_d4(reference_tissue_mask, d4_codes)
                        rotated_reference_nuclei = rotate_batch_d4(reference_nuclei_mask, d4_codes)
                        pred_rotated_reference = model(
                            target_cond,
                            rotated_reference_cond,
                            target_region=target_region,
                            reference_region=rotated_reference_region,
                            target_trust_map=trust_map,
                            ref_fallback_scale=args.ref_fallback_scale if args.ref_trust_gate else 1.0,
                            ref_soft_context_scale=args.ref_soft_context_scale,
                            ref_nuclei_context_scale=args.ref_nuclei_context_scale,
                            ref_soft_context_radius=args.ref_soft_context_radius,
                            target_tissue_mask=target_tissue_mask,
                            target_nuclei_mask=target_nuclei_mask,
                            reference_tissue_mask=rotated_reference_tissue,
                            reference_nuclei_mask=rotated_reference_nuclei,
                        )
                        rotated_fine, rotated_mid = masked_multiband_detail_loss(
                            pred_rotated_reference,
                            target,
                            corruption_mask,
                            target_region,
                        )
                        ref_orientation_consistency, ref_target_alignment = (
                            masked_orientation_consistency_loss(
                                pred,
                                pred_rotated_reference,
                                target,
                                corruption_mask,
                            )
                        )
                        ref_rotation_style = regional_rotation_invariant_style_loss(
                            pred_rotated_reference,
                            rotated_reference_cond[:, :3],
                            target_region,
                            rotated_reference_region,
                            min_pixels=int(args.texture_min_pixels),
                        )
                        loss = (
                            loss
                            + 0.5 * float(args.lambda_detail_fine) * rotated_fine
                            + 0.5 * float(args.lambda_detail_mid) * rotated_mid
                            + float(args.lambda_ref_orientation_consistency)
                            * (ref_orientation_consistency + ref_target_alignment)
                            + float(args.lambda_ref_rotation_style) * ref_rotation_style
                        )

            logs["sup_total"] = float(loss.detach().float().item())
            logs["detail_fine"] = float(detail_fine.detach().float().item())
            logs["detail_mid"] = float(detail_mid.detach().float().item())
            logs["baseline_consistency"] = float(baseline_consistency.detach().float().item())
            logs["anchor_teacher_consistency"] = float(
                anchor_teacher_consistency.detach().float().item()
            )
            logs["ref_orientation_consistency"] = float(
                ref_orientation_consistency.detach().float().item()
            )
            logs["ref_target_alignment"] = float(ref_target_alignment.detach().float().item())
            logs["ref_rotation_style"] = float(ref_rotation_style.detach().float().item())
            logs["ref_rotation_paired"] = float(ref_rotation_paired)
            logs["main_ref_rotation_active"] = float(reference_rotation_active)
            logs["main_ref_rotation_angle"] = float(reference_rotation_angle)
            logs["main_ref_rotation_probability"] = float(reference_rotation_probability)
            logs["target_orientation"] = float(target_orientation.detach().float().item())
            logs["target_anisotropy"] = float(target_anisotropy.detach().float().item())
            logs["target_orientation_valid"] = float(
                target_orientation_valid.detach().float().item()
            )
            logs["target_orientation_coherence"] = float(
                target_orientation_coherence.detach().float().item()
            )
            logs["i0_window_orientation"] = float(
                i0_window_orientation.detach().float().item()
            )
            logs["i0_window_directionality"] = float(
                i0_window_directionality.detach().float().item()
            )
            logs["i0_window_valid"] = float(i0_window_valid.detach().float().item())
            logs["i0_window_angle_degrees"] = float(
                i0_window_angle_degrees.detach().float().item()
            )
            logs["i0_window_resultant"] = float(
                i0_window_resultant.detach().float().item()
            )
            logs["pred_window_resultant"] = float(
                pred_window_resultant.detach().float().item()
            )
            logs["i0_residual_orientation"] = float(
                i0_residual_orientation.detach().float().item()
            )
            logs["i0_residual_angle_degrees"] = float(
                i0_residual_angle_degrees.detach().float().item()
            )
            logs["i0_texture_energy"] = float(i0_texture_energy.detach().float().item())
            logs["i0_texture_energy_ratio"] = float(
                i0_texture_energy_ratio.detach().float().item()
            )
            logs["i0_texture_under_floor"] = float(
                i0_texture_under_floor.detach().float().item()
            )
            logs["i0_texture_over_ceiling"] = float(
                i0_texture_over_ceiling.detach().float().item()
            )
            logs["steering_confidence"] = float(steering_confidence.detach().float().item())
            logs["steering_raw_confidence"] = float(
                steering_raw_confidence.detach().float().item()
            )
            logs["steering_active_fraction"] = float(
                steering_active_fraction.detach().float().item()
            )
            logs["steering_fallback_fraction"] = float(
                steering_fallback_fraction.detach().float().item()
            )
            logs["steering_mean_angle"] = float(steering_mean_angle.detach().float().item())
            logs["i0_orientation_ramp"] = float(i0_orientation_ramp)
            logs["corruption_fraction"] = float(corruption_mask.gt(0.05).float().mean().item())
            logs["rotated_i0_corruption_active"] = float(rotated_i0_corruption_active)
            active_sigmas = corruption_sigma[corruption_sigma.gt(0)]
            logs["corruption_sigma"] = float(active_sigmas.mean().item()) if active_sigmas.numel() else 0.0
            logs["ref_mismatch_active"] = float(cross_wsi_active)
            logs["cross_wsi_fallback"] = float(cross_wsi_fallback)
            logs["identity_warmup_active"] = float(identity_warmup_active)
            logs["mean_trust"] = float(trust_logs["mean_trust"])
            logs["low_trust_fraction"] = float(trust_logs["low_trust_fraction"])
            logs["unmatched_regions"] = float(trust_logs["unmatched_regions"])
            logs["adv_g"] = 0.0
            logs["adv_d"] = 0.0
            logs["d_real"] = 0.0
            logs["d_fake"] = 0.0
            logs["adv_active"] = 0.0
            logs["condition_mismatch_d"] = 0.0
            ramp_steps = max(1, int(args.context_ramp_steps))
            context_ramp = min(1.0, float(continuation_step + 1) / float(ramp_steps))
            boundary_adv_floor = float(args.boundary_adv_floor) * context_ramp
            condition_mismatch_weight = float(args.condition_mismatch_d_weight) * context_ramp
            logs["boundary_adv_floor"] = boundary_adv_floor
            logs["condition_mismatch_weight"] = condition_mismatch_weight
            adv_active = (
                discriminator is not None
                and d_optimizer is not None
                and args.lambda_adv > 0.0
                and global_step >= int(args.adv_warmup_steps)
            )
            if adv_active:
                d_optimizer.zero_grad(set_to_none=True)
                set_requires_grad(discriminator, True)
                with autocast_context(device, args.mixed_precision):
                    real_logits = discriminator(target.detach(), region_condition)
                    fake_logits = discriminator(pred.detach(), region_condition)
                    adv_mask = patch_mask_from_region(
                        target_region,
                        real_logits,
                        mode=args.adv_mask_mode,
                    )
                    if int(args.boundary_feather_radius) > 0:
                        boundary = boundary_band_mask(
                            target_tissue_mask,
                            radius=args.boundary_feather_radius,
                        )
                        adv_mask = soft_boundary_patch_mask(
                            adv_mask,
                            boundary,
                            real_logits,
                            floor=boundary_adv_floor,
                            corruption_mask=corruption_mask if corruption_active else None,
                        )
                    d_loss = discriminator_hinge_loss(
                        real_logits,
                        fake_logits,
                        mask=adv_mask,
                    )
                    condition_mismatch_d = pred.new_zeros(())
                    if (
                        corruption_active
                        and not cross_wsi_active
                        and condition_mismatch_weight > 0.0
                    ):
                        wrong_condition, mismatch_core, _ = build_context_mismatch_condition(
                            region_condition,
                            corruption_mask,
                            generator=detail_generator,
                            ring_radius=32,
                        )
                        wrong_condition_logits = discriminator(target.detach(), wrong_condition)
                        mismatch_patch = F.interpolate(
                            mismatch_core.to(
                                device=wrong_condition_logits.device,
                                dtype=wrong_condition_logits.dtype,
                            ),
                            size=wrong_condition_logits.shape[-2:],
                            mode="area",
                        )
                        condition_mismatch_d = conditional_mismatch_hinge_loss(
                            wrong_condition_logits,
                            mask=mismatch_patch,
                        )
                        d_loss = d_loss + condition_mismatch_weight * condition_mismatch_d
                    d_real, d_fake = discriminator_logit_stats(
                        real_logits,
                        fake_logits,
                        mask=adv_mask,
                    )
                if scaler.is_enabled():
                    scaler.scale(d_loss).backward()
                    scaler.step(d_optimizer)
                else:
                    d_loss.backward()
                    d_optimizer.step()

                set_requires_grad(discriminator, False)
                with autocast_context(device, args.mixed_precision):
                    fake_logits_for_g = discriminator(pred, region_condition)
                    adv_g_loss = generator_hinge_loss(fake_logits_for_g, mask=adv_mask)
                    loss = loss + float(args.lambda_adv) * adv_g_loss

                logs["adv_g"] = float(adv_g_loss.detach().float().item())
                logs["adv_d"] = float(d_loss.detach().float().item())
                logs["d_real"] = float(d_real.detach().float().item())
                logs["d_fake"] = float(d_fake.detach().float().item())
                logs["adv_active"] = 1.0
                logs["condition_mismatch_d"] = float(
                    condition_mismatch_d.detach().float().item()
                )
            logs["total"] = float(loss.detach().float().item())

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            if adv_active:
                set_requires_grad(discriminator, True)


            if rank == 0 and global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"epoch={epoch:04d} step={global_step:08d} "
                    f"loss={logs['total']:.5f} l1={logs['l1']:.5f} "
                    f"perc={logs['perc']:.5f} elapsed={elapsed/60:.1f}m"
                )
                # gamma 打印:放在 rank==0 和 log 同一个块里
                net = unwrap_model(model)
                gammas = [
                    f"{name}={p.item():.5f}"
                    for name, p in net.named_parameters()
                    if name.endswith("gamma")
                ]
                if gammas:
                    print(f"  [gamma] {' | '.join(gammas)}")
                print(
                    "  [texture-loss] "
                    f"gram={logs['gram']:.5f} contextual={logs['contextual']:.5f} "
                    f"norm_l1={logs['norm_l1']:.3f} norm_content={logs['norm_perc']:.3f} "
                    f"norm_gram={logs['norm_gram']:.3f} "
                    f"norm_contextual={logs['norm_contextual']:.3f}"
                )
                print(
                    "  [wsi-identity] "
                    f"mode={'cross_wsi' if logs['ref_mismatch_active'] else 'same_wsi'} "
                    f"od={logs['identity_od']:.5f} feature={logs['identity_feature']:.5f} "
                    f"band={logs['identity_band']:.5f} rank={logs['identity_rank']:.5f} "
                    f"gray={logs['structure_gray']:.5f} edge={logs['structure_edge']:.5f} "
                    f"warmup={int(logs['identity_warmup_active'])} "
                    f"swap_fallback={int(logs['cross_wsi_fallback'])}"
                )
                print(
                    "  [trust-boundary] "
                    f"mean_trust={logs['mean_trust']:.3f} "
                    f"low_trust={logs['low_trust_fraction']:.3f} "
                    f"unmatched={logs['unmatched_regions']:.1f} "
                    f"boundary_hf={logs['boundary_hf']:.5f} "
                    f"lowtrust_hf={logs['lowtrust_hf']:.5f} "
                    f"cross_wsi={int(logs['ref_mismatch_active'])}"
                )
                print(
                    "  [detail-context] "
                    f"corrupt={logs['corruption_fraction']:.3f} "
                    f"sigma={logs['corruption_sigma']:.3f} "
                    f"fine={logs['detail_fine']:.5f} mid={logs['detail_mid']:.5f} "
                    f"base_cons={logs['baseline_consistency']:.5f} "
                    f"anchor_cons={logs['anchor_teacher_consistency']:.5f} "
                    f"rot_i0={int(logs['rotated_i0_corruption_active'])} "
                    f"rot_pair={int(logs['ref_rotation_paired'])} "
                    f"orient={logs['ref_orientation_consistency']:.5f} "
                    f"align={logs['ref_target_alignment']:.5f} "
                    f"ref_style={logs['ref_rotation_style']:.5f}"
                )
                print(
                    "  [main-ref-rotation] "
                    f"active={int(logs['main_ref_rotation_active'])} "
                    f"angle={logs['main_ref_rotation_angle']:.2f} "
                    f"prob={logs['main_ref_rotation_probability']:.3f} "
                    f"l1x={logs['l1_scale']:.2f} contentx={logs['content_scale']:.2f} "
                    f"gramx={logs['gram_scale']:.2f} "
                    f"contextualx={logs['contextual_scale']:.2f}"
                )
                print(
                    "  [target-orientation] "
                    f"loss={logs['target_orientation']:.5f} "
                    f"anis={logs['target_anisotropy']:.5f} "
                    f"valid={logs['target_orientation_valid']:.3f} "
                    f"coherence={logs['target_orientation_coherence']:.3f} "
                    f"lambdas={args.lambda_target_orientation:g}/"
                    f"{args.lambda_target_anisotropy:g}"
                )
                print(
                    "  [i0-window-orientation] "
                    f"loss={logs['i0_window_orientation']:.5f} "
                    f"dir_floor={logs['i0_window_directionality']:.5f} "
                    f"angle={logs['i0_window_angle_degrees']:.2f}deg "
                    f"valid={logs['i0_window_valid']:.3f} "
                    f"resultant={logs['i0_window_resultant']:.3f}->"
                    f"{logs['pred_window_resultant']:.3f} "
                    f"windows={args.i0_orientation_window_sizes} "
                    f"ramp={logs['i0_orientation_ramp']:.3f} "
                    f"lambdas={args.lambda_i0_window_orientation:g}/"
                    f"{args.lambda_i0_window_directionality:g}"
                )
                print(
                    "  [i0-steered-texture] "
                    f"residual={logs['i0_residual_orientation']:.5f} "
                    f"residual_angle={logs['i0_residual_angle_degrees']:.2f}deg "
                    f"energy={logs['i0_texture_energy']:.5f} "
                    f"energy_ratio={logs['i0_texture_energy_ratio']:.3f} "
                    f"under_floor={logs['i0_texture_under_floor']:.3f} "
                    f"over_ceiling={logs['i0_texture_over_ceiling']:.3f} "
                    f"steer_conf={logs['steering_confidence']:.3f} "
                    f"raw_conf={logs['steering_raw_confidence']:.3f} "
                    f"steer_active={logs['steering_active_fraction']:.3f} "
                    f"fallback={logs['steering_fallback_fraction']:.3f} "
                    f"mean_angle={logs['steering_mean_angle']:.1f} "
                    f"mode={args.cross4_steering_reference_mode} "
                    "fractions="
                    + ",".join(
                        f"{angle:g}:{float(fraction):.2f}"
                        for angle, fraction in zip(
                            cross4_steering_angles,
                            steering_candidate_fractions.detach().float().cpu().tolist(),
                            strict=True,
                        )
                    )
                    + " "
                    + f"lambdas={args.lambda_i0_residual_orientation:g}/"
                    f"{args.lambda_i0_texture_energy:g} "
                    f"band={args.i0_texture_energy_floor_ratio:g}.."
                    f"{args.i0_texture_energy_ceiling_ratio:g} "
                    f"scales={args.cross4_steering_scales} "
                    f"gains={args.cross1_steering_gain:g}/"
                    f"{args.cross2_steering_gain:g}/"
                    f"{args.cross4_steering_gain:g}/"
                    f"{args.cross8_steering_gain:g}/"
                    f"{args.cross16_steering_gain:g}"
                )
                if discriminator is not None:
                    print(
                        "  [patchgan] "
                        f"active={int(logs['adv_active'])} "
                        f"lambda_adv={args.lambda_adv:g} "
                        f"g_adv={logs['adv_g']:.5f} d_loss={logs['adv_d']:.5f} "
                        f"d_real={logs['d_real']:.5f} d_fake={logs['d_fake']:.5f} "
                        f"boundary_floor={logs['boundary_adv_floor']:.3f} "
                        f"mismatch_w={logs['condition_mismatch_weight']:.3f} "
                        f"mismatch_d={logs['condition_mismatch_d']:.5f}"
                    )
                current_gammas = {
                    name: float(parameter.detach().item())
                    for name, parameter in net.named_parameters()
                    if name.endswith("gamma")
                }
                gamma_delta = max(
                    (
                        abs(value - frozen_gamma_values.get(name, value))
                        for name, value in current_gammas.items()
                    ),
                    default=0.0,
                )
                print(
                    "  [freeze] "
                    f"scope={args.trainable_scope} lrs="
                    + ",".join(
                        f"{group.get('group_name', 'group')}:{group['lr']:.2e}"
                        for group in optimizer.param_groups
                    )
                    + " "
                    f"max_gamma_delta={gamma_delta:.8f}"
                )

            if rank == 0 and global_step % args.sample_every == 0:
                cpu_batch = {
                    "i0": target_cond[:, :3].detach().cpu(),
                    "reference_image": active_reference.detach().cpu(),
                    "target_image": batch["target_image"],
                }
                save_training_sample(
                    output_dir=output_dir,
                    step=global_step,
                    batch=cpu_batch,
                    pred=pred,
                )
            continuation_after_step = global_step - continuation_start_step + 1
            if (
                rank == 0
                and rotation_monitor_batch is not None
                and baseline_teacher is not None
                and int(args.rotation_monitor_every_steps) > 0
                and continuation_after_step % int(args.rotation_monitor_every_steps) == 0
            ):
                try:
                    monitor_metrics, monitor_reasons = run_rotation_monitor(
                        current_model=unwrap_model(model),
                        baseline_model=baseline_teacher,
                        batch=rotation_monitor_batch,
                        device=device,
                        output_dir=output_dir,
                        continuation_step=continuation_after_step,
                        mixed_precision=args.mixed_precision,
                        seed=args.eval_seed,
                        ref_trust_gate=args.ref_trust_gate,
                        ref_fallback_scale=args.ref_fallback_scale,
                        ref_soft_context_scale=args.ref_soft_context_scale,
                        ref_nuclei_context_scale=args.ref_nuclei_context_scale,
                        ref_soft_context_radius=args.ref_soft_context_radius,
                        matched_tissue_trust_floor=args.matched_tissue_trust_floor,
                        matched_nuclei_trust_floor=args.matched_nuclei_trust_floor,
                        max_clean_drift=args.rotation_monitor_max_clean_drift,
                        max_ref_distance_ratio=args.rotation_monitor_max_ref_distance_ratio,
                        max_boundary_seam_ratio=args.rotation_monitor_max_boundary_seam_ratio,
                        min_nuclei_band_ratio=args.rotation_monitor_min_nuclei_band_ratio,
                        max_nuclei_band_ratio=args.rotation_monitor_max_nuclei_band_ratio,
                    )
                    unsafe_rotation_monitor_count = (
                        unsafe_rotation_monitor_count + 1 if monitor_reasons else 0
                    )
                    print(
                        "  [rotation-monitor] "
                        f"step={continuation_after_step} "
                        f"clean_drift={monitor_metrics['clean_drift_mae']:.5f} "
                        f"ref_ratio={monitor_metrics['current_clean_ref_distance_ratio']:.3f} "
                        f"seam_ratio={monitor_metrics['current_boundary_seam_ratio']:.3f} "
                        f"nuclei_ratio={monitor_metrics['current_nuclei_band_ratio_vs_baseline']:.3f} "
                        f"orient={monitor_metrics['baseline_rotated_orientation']:.5f}->"
                        f"{monitor_metrics['current_rotated_orientation']:.5f} "
                        f"unsafe={monitor_reasons or 'none'} "
                        f"count={unsafe_rotation_monitor_count}"
                    )
                    if unsafe_rotation_monitor_count >= max(
                        1,
                        int(args.rotation_monitor_stop_patience),
                    ):
                        print("[rotation-monitor] safety gate requested early pilot stop")
                        pilot_stop = True
                except Exception as exc:
                    print(f"[rotation-monitor] failed without stopping training: {exc}")
            global_step += 1
            if pilot_stop:
                break

        if world_size > 1:
            dist.barrier()

        if pilot_stop:
            if rank == 0:
                continuation_steps = global_step - continuation_start_step
                checkpoint = {
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss_normalizer": criterion.normalizer.state_dict(),
                    "identity_loss_normalizer": criterion.identity_normalizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "continuation_steps": continuation_steps,
                    "args": vars(args),
                }
                if discriminator is not None:
                    checkpoint["discriminator"] = unwrap_model(discriminator).state_dict()
                if d_optimizer is not None:
                    checkpoint["d_optimizer"] = d_optimizer.state_dict()
                pilot_path = output_dir / "ckpt" / f"pilot_step{continuation_steps:06d}.pt"
                torch.save(checkpoint, pilot_path)
                print(f"saved pilot checkpoint {pilot_path}")
            if world_size > 1:
                dist.barrier()
            break

        if (
            rank == 0
            and eval_loader is not None
            and args.eval_every_epochs > 0
            and (epoch + 1) % args.eval_every_epochs == 0
        ):
            save_eval_panel(
                model=unwrap_model(model),
                loader=eval_loader,
                device=device,
                output_dir=output_dir,
                epoch=epoch,
                mixed_precision=args.mixed_precision,
                i0_generator=i0_generator,
            )
            print(f"saved eval panel epoch {epoch + 1}")

        if rank == 0 and (epoch + 1) % args.save_every == 0:
            checkpoint = {
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_normalizer": criterion.normalizer.state_dict(),
                "identity_loss_normalizer": criterion.identity_normalizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "args": vars(args),
            }
            if discriminator is not None:
                checkpoint["discriminator"] = unwrap_model(discriminator).state_dict()
            if d_optimizer is not None:
                checkpoint["d_optimizer"] = d_optimizer.state_dict()
            torch.save(checkpoint, output_dir / "ckpt" / f"epoch{epoch + 1:04d}.pt")
            print(f"saved checkpoint epoch {epoch + 1}")

        if (
            rank == 0
            and args.ood_diagnose_every_epochs > 0
            and (epoch + 1) % int(args.ood_diagnose_every_epochs) == 0
        ):
            try:
                ood_root = Path(args.ood_diagnose_root) if args.ood_diagnose_root else output_dir
                summary_path = save_ood_diagnose(
                    model=unwrap_model(model),
                    probes=ood_probes,
                    device=device,
                    output_dir=ood_root,
                    epoch=epoch,
                    mixed_precision=args.mixed_precision,
                    ref_trust_gate=bool(args.ref_trust_gate),
                    ref_fallback_scale=float(args.ref_fallback_scale),
                    ref_soft_context_scale=float(args.ref_soft_context_scale),
                    ref_nuclei_context_scale=float(args.ref_nuclei_context_scale),
                    ref_soft_context_radius=int(args.ref_soft_context_radius),
                    matched_tissue_trust_floor=float(args.matched_tissue_trust_floor),
                    matched_nuclei_trust_floor=float(args.matched_nuclei_trust_floor),
                )
                if summary_path is not None:
                    print(f"[ood-diagnose] saved {summary_path}")
                else:
                    print("[ood-diagnose] skipped: no probes available")
            except Exception as exc:
                print(f"[ood-diagnose] failed for epoch {epoch + 1}: {exc}")

        if world_size > 1:
            dist.barrier()

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
