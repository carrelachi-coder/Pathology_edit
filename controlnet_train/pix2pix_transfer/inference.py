"""Checkpoint-faithful inference for the production pix2pix postprocessor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from .dataset import (
    NUM_CELL_CLASSES,
    NUM_FINE,
    load_label_mask,
    load_rgb,
    one_hot_mask,
    remap_nuclei_mask,
    tissue_nuclei_region_labels,
)
from .regional_cross_attention import Pix2PixCrossAttnUNet
from .inference_orientation import build_fine_texture_steering_weights
from .trust_gate import build_highres_nuclei_reference_trust_map


@dataclass(frozen=True)
class Pix2PixPostprocessConfig:
    """Architecture and trust settings stored with a production checkpoint."""

    base_channels: int = 64
    num_heads: int = 4
    cross_attn_scales: str = "1/4,1/8,1/16"
    upsample_mode: str = "bilinear"
    region_label_mode: str = "tissue_nuclei"
    use_region_mask: bool = True
    residual_output: bool = True
    use_wsi_identity: bool = False
    identity_gamma_max: float = 0.30
    identity_gamma_init: float = 0.10
    identity_min_tissue_pixels: int = 256
    identity_min_nuclei_pixels: int = 64
    cross4_texture_steering: bool = False
    cross4_steering_angles: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    cross4_steering_smoothing_sigma: float = 8.0
    cross4_steering_min_coherence: float = 0.20
    cross4_steering_min_relative_energy: float = 0.50
    cross4_steering_min_resultant: float = 0.15
    cross4_steering_minimum_strength: float = 0.0
    cross4_steering_minimum_support: float = 0.05
    cross4_steering_temperature: float = 0.08
    cross4_steering_reference_mode: str = "global_mean"
    cross4_steering_local_bins: int = 36
    cross4_steering_local_kappa: float = 8.0
    cross4_steering_scales: tuple[str, ...] = ("1/4",)
    cross4_steering_gain: float = 1.0
    cross8_steering_gain: float = 1.0
    cross16_steering_gain: float = 1.0
    cross2_steering_gain: float = 1.0
    cross1_steering_gain: float = 1.0
    full_pyramid_texture_steering: bool = False
    steering_highres_reference_size: int = 8
    highres_nuclei_trust_enabled: bool = True
    highres_nuclei_unmatched_scale: float = 0.20
    highres_nuclei_matched_floor: float = 0.60
    highres_nuclei_sufficient_tokens: int = 4
    highres_nuclei_min_reference_pixels: int = 64

    @classmethod
    def from_checkpoint_args(cls, args: Mapping[str, Any] | None) -> "Pix2PixPostprocessConfig":
        values = dict(args or {})
        return cls(
            base_channels=int(values.get("base_channels", 64)),
            num_heads=int(values.get("num_heads", 4)),
            cross_attn_scales=str(values.get("cross_attn_scales", "1/4,1/8,1/16")),
            upsample_mode=str(values.get("upsample_mode", "bilinear")),
            region_label_mode=str(values.get("region_label_mode", "tissue_nuclei")),
            use_region_mask=not bool(values.get("no_region_mask", False)),
            residual_output=not bool(values.get("no_residual_output", False)),
            use_wsi_identity=bool(values.get("wsi_identity_adapter", False)),
            identity_gamma_max=float(values.get("identity_gamma_max", 0.30)),
            identity_gamma_init=float(values.get("identity_gamma_init", 0.10)),
            identity_min_tissue_pixels=int(values.get("identity_min_tissue_pixels", 256)),
            identity_min_nuclei_pixels=int(values.get("identity_min_nuclei_pixels", 64)),
            cross4_texture_steering=bool(values.get("cross4_texture_steering", False)),
            cross4_steering_angles=_float_tuple(
                values.get("cross4_steering_angles", "0,45,90,135")
            ),
            cross4_steering_smoothing_sigma=float(
                values.get("cross4_steering_smoothing_sigma", 8.0)
            ),
            cross4_steering_min_coherence=float(
                values.get("cross4_steering_min_coherence", 0.20)
            ),
            cross4_steering_min_relative_energy=float(
                values.get("cross4_steering_min_relative_energy", 0.50)
            ),
            cross4_steering_min_resultant=float(
                values.get("cross4_steering_min_resultant", 0.15)
            ),
            cross4_steering_minimum_strength=float(
                values.get("cross4_steering_minimum_strength", 0.0)
            ),
            cross4_steering_minimum_support=float(
                values.get("cross4_steering_minimum_support", 0.05)
            ),
            cross4_steering_temperature=float(
                values.get("cross4_steering_temperature", 0.08)
            ),
            cross4_steering_reference_mode=str(
                values.get("cross4_steering_reference_mode", "global_mean")
            ),
            cross4_steering_local_bins=int(
                values.get("cross4_steering_local_bins", 36)
            ),
            cross4_steering_local_kappa=float(
                values.get("cross4_steering_local_kappa", 8.0)
            ),
            cross4_steering_scales=_string_tuple(
                values.get("cross4_steering_scales", "1/4")
            ),
            cross4_steering_gain=float(values.get("cross4_steering_gain", 1.0)),
            cross8_steering_gain=float(values.get("cross8_steering_gain", 1.0)),
            cross16_steering_gain=float(values.get("cross16_steering_gain", 1.0)),
            cross2_steering_gain=float(values.get("cross2_steering_gain", 1.0)),
            cross1_steering_gain=float(values.get("cross1_steering_gain", 1.0)),
            full_pyramid_texture_steering=bool(
                values.get("full_pyramid_texture_steering", False)
            ),
            steering_highres_reference_size=int(
                values.get("steering_highres_reference_size", 8)
            ),
            highres_nuclei_trust_enabled=bool(
                values.get("highres_nuclei_trust_enabled", True)
            ),
            highres_nuclei_unmatched_scale=float(
                values.get("highres_nuclei_unmatched_scale", 0.20)
            ),
            highres_nuclei_matched_floor=float(
                values.get("highres_nuclei_matched_floor", 0.60)
            ),
            highres_nuclei_sufficient_tokens=int(
                values.get("highres_nuclei_sufficient_tokens", 4)
            ),
            highres_nuclei_min_reference_pixels=int(
                values.get("highres_nuclei_min_reference_pixels", 64)
            ),
        )


@dataclass(frozen=True)
class LoadedPix2PixPostprocessor:
    model: Pix2PixCrossAttnUNet
    config: Pix2PixPostprocessConfig
    checkpoint_path: Path
    epoch: int | None
    global_step: int | None


@dataclass(frozen=True)
class CrossLowStainProtectionConfig:
    """Inference-only guard that preserves low-stain Cross structures."""

    policy: str = "cross_rgb_od_low_stain_v1"
    minimum_luma: float = 0.70
    maximum_chroma: float = 0.28
    maximum_otsu_od: float = 0.35
    nuclei_exclusion_radius_at_512: int = 5
    closing_iterations_at_512: int = 4
    opening_iterations_at_512: int = 2
    minimum_component_area_at_512: int = 500
    erosion_iterations_at_512: int = 2
    feather_sigma_at_512: float = 4.0


def load_pix2pix_postprocessor(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    torch_dtype: torch.dtype,
) -> LoadedPix2PixPostprocessor:
    """Load the exact architecture recorded by an epoch checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pix2pix checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f"Pix2pix checkpoint has no 'model' state: {checkpoint_path}")
    config = Pix2PixPostprocessConfig.from_checkpoint_args(checkpoint.get("args"))
    model = Pix2PixCrossAttnUNet(
        in_ch=3 + NUM_FINE + (NUM_CELL_CLASSES + 1),
        out_ch=3,
        base=config.base_channels,
        num_heads=config.num_heads,
        use_region_mask=config.use_region_mask,
        residual_output=config.residual_output,
        cross_attn_scales=config.cross_attn_scales,
        upsample_mode=config.upsample_mode,
        use_wsi_identity=config.use_wsi_identity,
        identity_gamma_max=config.identity_gamma_max,
        identity_gamma_init=config.identity_gamma_init,
        identity_min_tissue_pixels=config.identity_min_tissue_pixels,
        identity_min_nuclei_pixels=config.identity_min_nuclei_pixels,
        full_pyramid_texture_steering=config.full_pyramid_texture_steering,
        steering_highres_reference_size=config.steering_highres_reference_size,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device=device, dtype=torch_dtype).eval()
    return LoadedPix2PixPostprocessor(
        model=model,
        config=config,
        checkpoint_path=checkpoint_path,
        epoch=_optional_int(checkpoint.get("epoch")),
        global_step=_optional_int(checkpoint.get("global_step")),
    )


@torch.inference_mode()
def run_pix2pix_postprocess(
    *,
    bundle: LoadedPix2PixPostprocessor,
    i0_image: Image.Image,
    reference_image_path: str | Path,
    target_tissue_mask_path: str | Path,
    target_nuclei_mask_path: str | Path,
    reference_tissue_mask_path: str | Path,
    reference_nuclei_mask_path: str | Path,
    image_size: int,
    device: str | torch.device,
    torch_dtype: torch.dtype,
    enable_highres_nuclei_trust: bool | None = None,
    low_stain_protection_region_path: str | Path | None = None,
    low_stain_protection_mask_output_path: str | Path | None = None,
    unprotected_output_path: str | Path | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply the epoch checkpoint with its trained trust and identity settings."""

    config = bundle.config
    i0 = _pil_to_neg1_tensor(i0_image, image_size).unsqueeze(0).to(
        device=device, dtype=torch_dtype
    )
    reference = load_rgb(reference_image_path, image_size).unsqueeze(0).to(
        device=device, dtype=torch_dtype
    )
    target_tissue = load_label_mask(target_tissue_mask_path, image_size)
    reference_tissue = load_label_mask(reference_tissue_mask_path, image_size)
    target_nuclei = remap_nuclei_mask(load_label_mask(target_nuclei_mask_path, image_size))
    reference_nuclei = remap_nuclei_mask(
        load_label_mask(reference_nuclei_mask_path, image_size)
    )

    target_cond = _condition(i0[0].cpu(), target_tissue, target_nuclei).unsqueeze(0).to(
        device=device, dtype=torch_dtype
    )
    reference_cond = _condition(
        reference[0].cpu(), reference_tissue, reference_nuclei
    ).unsqueeze(0).to(device=device, dtype=torch_dtype)
    target_region = tissue_nuclei_region_labels(
        target_tissue, target_nuclei, label_mode=config.region_label_mode
    ).unsqueeze(0).to(device=device)
    reference_region = tissue_nuclei_region_labels(
        reference_tissue, reference_nuclei, label_mode=config.region_label_mode
    ).unsqueeze(0).to(device=device)

    steering_weights = None
    steering_info: dict[str, Any] = {"enabled": False}
    if config.cross4_texture_steering:
        steering_result = build_fine_texture_steering_weights(
            i0,
            reference,
            target_tissue_mask=target_tissue.unsqueeze(0).to(device=device),
            target_nuclei_mask=target_nuclei.unsqueeze(0).to(device=device),
            reference_tissue_mask=reference_tissue.unsqueeze(0).to(device=device),
            reference_nuclei_mask=reference_nuclei.unsqueeze(0).to(device=device),
            candidate_angles_degrees=config.cross4_steering_angles,
            smoothing_sigma=config.cross4_steering_smoothing_sigma,
            min_coherence=config.cross4_steering_min_coherence,
            min_relative_energy=config.cross4_steering_min_relative_energy,
            min_resultant=config.cross4_steering_min_resultant,
            minimum_strength=config.cross4_steering_minimum_strength,
            minimum_support=config.cross4_steering_minimum_support,
            temperature=config.cross4_steering_temperature,
            reference_direction_mode=config.cross4_steering_reference_mode,
            local_histogram_bins=config.cross4_steering_local_bins,
            local_histogram_concentration=config.cross4_steering_local_kappa,
        )
        steering_weights = steering_result.weights
        steering_info = {
            "enabled": True,
            "angles": list(config.cross4_steering_angles),
            "reference_direction_mode": config.cross4_steering_reference_mode,
            "mean_confidence": float(steering_result.mean_confidence.item()),
            "raw_mean_confidence": float(steering_result.raw_mean_confidence.item()),
            "active_fraction": float(steering_result.active_fraction.item()),
            "fallback_fraction": float(steering_result.fallback_fraction.item()),
            "scales": list(config.cross4_steering_scales),
            "gains": {
                "1/4": config.cross4_steering_gain,
                "1/8": config.cross8_steering_gain,
                "1/16": config.cross16_steering_gain,
                "1/2": config.cross2_steering_gain,
                "1/1": config.cross1_steering_gain,
            },
            "candidate_fractions": [
                float(value) for value in steering_result.candidate_fractions.tolist()
            ],
            "mean_selected_angle_degrees": float(
                steering_result.mean_selected_angle_degrees.item()
            ),
        }

    highres_nuclei_trust_map = None
    nuclei_trust_info: dict[str, Any] = {"enabled": False}
    highres_scales_enabled = bool(
        config.full_pyramid_texture_steering
        and steering_weights is not None
        and {"1/1", "1/2"}.intersection(config.cross4_steering_scales)
    )
    trust_enabled = (
        config.highres_nuclei_trust_enabled
        if enable_highres_nuclei_trust is None
        else bool(enable_highres_nuclei_trust)
    )
    if trust_enabled and highres_scales_enabled:
        highres_nuclei_trust_map, nuclei_trust_stats = (
            build_highres_nuclei_reference_trust_map(
                target_nuclei.unsqueeze(0).to(device=device),
                reference_nuclei.unsqueeze(0).to(device=device),
                reference_weights=steering_weights,
                candidate_angles_degrees=config.cross4_steering_angles,
                reference_pool_size=config.steering_highres_reference_size,
                unmatched_scale=config.highres_nuclei_unmatched_scale,
                matched_floor=config.highres_nuclei_matched_floor,
                sufficient_reference_tokens=config.highres_nuclei_sufficient_tokens,
                min_reference_pixels=config.highres_nuclei_min_reference_pixels,
            )
        )
        nuclei_trust_info = {"enabled": True, **nuclei_trust_stats}

    pred = bundle.model(
        target_cond,
        reference_cond,
        target_region=target_region,
        reference_region=reference_region,
        target_trust_map=highres_nuclei_trust_map,
        highres_nuclei_trust_map=highres_nuclei_trust_map,
        target_tissue_mask=target_tissue.unsqueeze(0).to(device=device),
        target_nuclei_mask=target_nuclei.unsqueeze(0).to(device=device),
        reference_tissue_mask=reference_tissue.unsqueeze(0).to(device=device),
        reference_nuclei_mask=reference_nuclei.unsqueeze(0).to(device=device),
        cross4_rotation_weights=steering_weights,
        cross4_rotation_angles=config.cross4_steering_angles,
        texture_steering_scales=config.cross4_steering_scales,
        cross4_steering_gain=config.cross4_steering_gain,
        cross8_steering_gain=config.cross8_steering_gain,
        cross16_steering_gain=config.cross16_steering_gain,
        cross2_steering_gain=config.cross2_steering_gain,
        cross1_steering_gain=config.cross1_steering_gain,
    )[0]
    unprotected = _tensor_to_pil(pred)
    if unprotected_output_path is not None:
        unprotected_path = Path(unprotected_output_path)
        unprotected_path.parent.mkdir(parents=True, exist_ok=True)
        unprotected.save(unprotected_path)

    protection_region = None
    if low_stain_protection_region_path is not None:
        protection_region = load_label_mask(
            low_stain_protection_region_path,
            image_size,
        ).ne(0)
    protected, protection_mask, protection_info = apply_cross_low_stain_protection(
        cross_image=i0_image,
        pix2pix_image=unprotected,
        target_nuclei_mask=target_nuclei,
        protection_region=protection_region,
    )
    if low_stain_protection_mask_output_path is not None:
        protection_mask_path = Path(low_stain_protection_mask_output_path)
        protection_mask_path.parent.mkdir(parents=True, exist_ok=True)
        protection_mask.save(protection_mask_path)
        protection_info["mask"] = str(protection_mask_path)
    if unprotected_output_path is not None:
        protection_info["unprotected_output"] = str(unprotected_output_path)

    info = {
        "checkpoint": str(bundle.checkpoint_path),
        "epoch": bundle.epoch,
        "global_step": bundle.global_step,
        "use_wsi_identity": config.use_wsi_identity,
        "trust_gate": (
            "nuclei_reference_support_v2"
            if highres_nuclei_trust_map is not None
            else "removed_from_production_inference"
        ),
        "nuclei_reference_trust": nuclei_trust_info,
        "texture_steering": steering_info,
        "cross_rgb_od_low_stain_protection": protection_info,
    }
    return protected, info


def apply_cross_low_stain_protection(
    *,
    cross_image: Image.Image,
    pix2pix_image: Image.Image,
    target_nuclei_mask: torch.Tensor | np.ndarray,
    protection_region: torch.Tensor | np.ndarray | None = None,
    config: CrossLowStainProtectionConfig | None = None,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Blend low-stain, cell-free Cross components back over Pix2Pix output."""

    from scipy import ndimage

    config = config or CrossLowStainProtectionConfig()
    pix2pix = np.asarray(pix2pix_image.convert("RGB"), dtype=np.float32)
    height, width = pix2pix.shape[:2]
    cross = np.asarray(
        cross_image.convert("RGB").resize(
            (width, height),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )
    nuclei = _as_numpy_mask(target_nuclei_mask, size=(width, height)) != 0
    if protection_region is None:
        region = np.ones((height, width), dtype=bool)
        region_scope = "full_image"
    else:
        region = _as_numpy_mask(protection_region, size=(width, height)) != 0
        region_scope = "generation_change_region"

    scale = float(min(height, width)) / 512.0
    nuclei_radius = max(
        1,
        int(round(config.nuclei_exclusion_radius_at_512 * scale)),
    )
    closing_iterations = max(
        1,
        int(round(config.closing_iterations_at_512 * scale)),
    )
    opening_iterations = max(
        1,
        int(round(config.opening_iterations_at_512 * scale)),
    )
    erosion_iterations = max(
        1,
        int(round(config.erosion_iterations_at_512 * scale)),
    )
    minimum_component_area = max(
        1,
        int(round(config.minimum_component_area_at_512 * scale * scale)),
    )
    feather_sigma = max(0.5, config.feather_sigma_at_512 * scale)

    rgb = cross / 255.0
    luma = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    optical_density = -np.log((cross + 1.0) / 256.0).mean(axis=2)
    region_values = optical_density[region]
    raw_otsu_threshold = _otsu_threshold(region_values)
    od_threshold = min(raw_otsu_threshold, float(config.maximum_otsu_od))

    nuclei_exclusion = ndimage.binary_dilation(
        nuclei,
        iterations=nuclei_radius,
    )
    candidate = (
        region
        & ~nuclei_exclusion
        & (luma >= float(config.minimum_luma))
        & (chroma <= float(config.maximum_chroma))
        & (optical_density <= od_threshold)
    )
    candidate = ndimage.binary_closing(
        candidate,
        iterations=closing_iterations,
    )
    candidate = ndimage.binary_opening(
        candidate,
        iterations=opening_iterations,
    )
    labels, component_count = ndimage.label(candidate)
    counts = np.bincount(labels.ravel())
    keep = np.zeros(component_count + 1, dtype=bool)
    if component_count:
        keep[1:] = counts[1:] >= minimum_component_area
    core = keep[labels] & region
    core = ndimage.binary_erosion(
        core,
        iterations=erosion_iterations,
    )
    alpha = ndimage.gaussian_filter(
        core.astype(np.float32),
        sigma=feather_sigma,
    ).clip(0.0, 1.0)
    alpha *= region.astype(np.float32)
    blended = (
        alpha[..., None] * cross
        + (1.0 - alpha[..., None]) * pix2pix
    )
    protected_pixels = int(np.count_nonzero(core))
    region_pixels = int(np.count_nonzero(region))
    info = {
        "policy": config.policy,
        "enabled": True,
        "applied": protected_pixels > 0,
        "region_scope": region_scope,
        "raw_otsu_od_threshold": float(raw_otsu_threshold),
        "effective_od_threshold": float(od_threshold),
        "maximum_otsu_od": float(config.maximum_otsu_od),
        "minimum_luma": float(config.minimum_luma),
        "maximum_chroma": float(config.maximum_chroma),
        "nuclei_exclusion_radius_px": nuclei_radius,
        "minimum_component_area_px": minimum_component_area,
        "feather_sigma_px": float(feather_sigma),
        "candidate_components": int(component_count),
        "protected_pixels": protected_pixels,
        "protected_fraction_image": float(protected_pixels / core.size),
        "protected_fraction_region": float(
            protected_pixels / region_pixels if region_pixels else 0.0
        ),
        "organ_specific_constraints": False,
    }
    mask = Image.fromarray(
        np.round(alpha * 255.0).astype(np.uint8),
        mode="L",
    )
    output = Image.fromarray(
        np.round(blended).clip(0, 255).astype(np.uint8),
        mode="RGB",
    )
    return output, mask, info


def _as_numpy_mask(
    mask: torch.Tensor | np.ndarray,
    *,
    size: tuple[int, int],
) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"mask must resolve to HxW, got {array.shape}")
    if tuple(array.shape[::-1]) != tuple(size):
        array = np.asarray(
            Image.fromarray(array.astype(np.uint8)).resize(
                size,
                Image.Resampling.NEAREST,
            )
        )
    return array


def _otsu_threshold(values: np.ndarray, *, bins: int = 256) -> float:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1.0e-8:
        return maximum
    histogram, edges = np.histogram(
        values,
        bins=bins,
        range=(minimum, maximum),
    )
    centers = (edges[:-1] + edges[1:]) * 0.5
    lower_weight = np.cumsum(histogram)
    upper_weight = int(histogram.sum()) - lower_weight
    weighted = np.cumsum(histogram * centers)
    total_weighted = float(weighted[-1])
    lower_mean = weighted / np.maximum(lower_weight, 1)
    upper_mean = (total_weighted - weighted) / np.maximum(upper_weight, 1)
    between = lower_weight * upper_weight * np.square(lower_mean - upper_mean)
    return float(centers[int(np.argmax(between))])


def _condition(image: torch.Tensor, tissue: torch.Tensor, nuclei: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            image,
            one_hot_mask(tissue, NUM_FINE),
            one_hot_mask(nuclei, NUM_CELL_CLASSES + 1),
        ],
        dim=0,
    )


def _pil_to_neg1_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    import numpy as np

    image = image.convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        ((image.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _float_tuple(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    else:
        values = tuple(float(item) for item in value)
    if not values:
        raise ValueError("cross4 steering angles cannot be empty")
    return values


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        values = tuple(str(item).strip() for item in value if str(item).strip())
    if not values:
        raise ValueError("texture steering scales cannot be empty")
    return values
