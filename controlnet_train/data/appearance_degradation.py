"""Appearance degradation for self-supervised stain/texture restoration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .hed_stain_augment import HEDStainAugment


@dataclass(frozen=True)
class TextureDegradationConfig:
    blur_prob: float = 0.7
    blur_sigma_min: float = 0.4
    blur_sigma_max: float = 1.4
    downsample_prob: float = 0.7
    downsample_scale_min: float = 0.35
    downsample_scale_max: float = 0.75
    noise_prob: float = 0.35
    noise_std_min: float = 0.005
    noise_std_max: float = 0.03


class TextureDegradationAugment:
    """Corrupt texture while preserving coarse structure and color layout."""

    def __init__(self, config: TextureDegradationConfig | None = None) -> None:
        self.config = config or TextureDegradationConfig()
        _validate_probability("blur_prob", self.config.blur_prob)
        _validate_probability("downsample_prob", self.config.downsample_prob)
        _validate_probability("noise_prob", self.config.noise_prob)
        _validate_range("blur_sigma", self.config.blur_sigma_min, self.config.blur_sigma_max, minimum=0.0)
        _validate_range("downsample_scale", self.config.downsample_scale_min, self.config.downsample_scale_max, minimum=0.0)
        _validate_range("noise_std", self.config.noise_std_min, self.config.noise_std_max, minimum=0.0)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"expected CHW RGB tensor with 3 channels, got {tuple(image.shape)}")

        dtype = image.dtype
        out = image.to(dtype=torch.float32).clamp(0.0, 1.0)
        if _sample_bool(self.config.blur_prob, device=out.device):
            sigma = _sample_uniform(
                self.config.blur_sigma_min,
                self.config.blur_sigma_max,
                device=out.device,
            )
            out = _gaussian_blur(out, sigma=sigma)
        if _sample_bool(self.config.downsample_prob, device=out.device):
            scale = _sample_uniform(
                self.config.downsample_scale_min,
                self.config.downsample_scale_max,
                device=out.device,
            )
            out = _downsample_upsample(out, scale=scale)
        if _sample_bool(self.config.noise_prob, device=out.device):
            std = _sample_uniform(
                self.config.noise_std_min,
                self.config.noise_std_max,
                device=out.device,
            )
            out = out + torch.randn_like(out) * float(std)
        return out.clamp(0.0, 1.0).to(dtype=dtype)


class AppearanceDegradationAugment:
    """Build the clean image used for noising: wrong stain plus degraded texture."""

    def __init__(
        self,
        *,
        mode: str,
        hed_augment: HEDStainAugment | None = None,
        texture_augment: TextureDegradationAugment | None = None,
    ) -> None:
        mode = str(mode or "none").strip().lower().replace("-", "_")
        aliases = {
            "none": "none",
            "hed": "hed",
            "stain": "hed",
            "texture": "texture",
            "hed_texture": "hed_texture",
            "stain_texture": "hed_texture",
        }
        if mode not in aliases:
            raise ValueError(
                f"Unsupported noising degradation {mode!r}; choose none, hed, texture, or hed_texture."
            )
        self.mode = aliases[mode]
        self.hed_augment = hed_augment
        self.texture_augment = texture_augment or TextureDegradationAugment()
        if self.mode in {"hed", "hed_texture"} and self.hed_augment is None:
            raise ValueError("HED degradation requires hed_augment.")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        out = image
        if self.mode in {"hed", "hed_texture"}:
            params = self.hed_augment.sample(device=image.device)
            out = self.hed_augment(out, params)
        if self.mode in {"texture", "hed_texture"}:
            out = self.texture_augment(out)
        return out


def _sample_bool(probability: float, *, device: torch.device) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return bool((torch.rand((), device=device) < float(probability)).item())


def _sample_uniform(low: float, high: float, *, device: torch.device) -> float:
    if high <= low:
        return float(low)
    return float((torch.rand((), device=device) * (high - low) + low).item())


def _gaussian_blur(image: torch.Tensor, *, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return image
    height, width = int(image.shape[1]), int(image.shape[2])
    if height < 2 or width < 2:
        return image
    radius = int(max(1, round(float(sigma) * 3.0)))
    radius = min(radius, height - 1, width - 1)
    coords = torch.arange(-radius, radius + 1, device=image.device, dtype=torch.float32)
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * float(sigma) * float(sigma)))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    weight = kernel_2d[None, None].repeat(image.shape[0], 1, 1, 1)
    padded = F.pad(image.unsqueeze(0), (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, weight, groups=image.shape[0]).squeeze(0)


def _downsample_upsample(image: torch.Tensor, *, scale: float) -> torch.Tensor:
    height, width = int(image.shape[1]), int(image.shape[2])
    if height < 2 or width < 2:
        return image
    scale = max(0.01, min(1.0, float(scale)))
    down_h = max(1, int(round(height * scale)))
    down_w = max(1, int(round(width * scale)))
    if down_h == height and down_w == width:
        return image
    batch = image.unsqueeze(0)
    down = F.interpolate(batch, size=(down_h, down_w), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(height, width), mode="bilinear", align_corners=False)
    return up.squeeze(0)


def _validate_probability(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be within [0, 1].")


def _validate_range(name: str, low: float, high: float, *, minimum: float) -> None:
    if float(low) < minimum or float(high) < minimum or float(high) < float(low):
        raise ValueError(f"{name} bounds must satisfy {minimum} <= low <= high.")
