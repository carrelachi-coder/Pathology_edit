"""Region-aware PatchGAN utilities for pix2pix texture transfer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if enabled else module


class RegionAwarePatchDiscriminator(nn.Module):
    """Patch discriminator conditioned on target-region masks.

    The discriminator sees ``[image, target_region_condition]`` rather than the
    image alone, so it can learn class-specific local texture statistics without
    needing spatial correspondence to a reference image.
    """

    def __init__(
        self,
        *,
        image_channels: int = 3,
        condition_channels: int = 22,
        base_channels: int = 64,
        max_channels: int = 512,
        num_layers: int = 3,
        spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        in_channels = int(image_channels) + int(condition_channels)
        base = int(base_channels)
        max_ch = int(max_channels)

        layers: list[nn.Module] = [
            _maybe_spectral_norm(nn.Conv2d(in_channels, base, 4, stride=2, padding=1), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base
        for layer_index in range(1, int(num_layers)):
            out_channels = min(base * (2**layer_index), max_ch)
            layers.extend(
                [
                    _maybe_spectral_norm(
                        nn.Conv2d(channels, out_channels, 4, stride=2, padding=1),
                        spectral_norm,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            channels = out_channels

        out_channels = min(channels * 2, max_ch)
        layers.extend(
            [
                _maybe_spectral_norm(
                    nn.Conv2d(channels, out_channels, 4, stride=1, padding=1),
                    spectral_norm,
                ),
                nn.LeakyReLU(0.2, inplace=True),
                _maybe_spectral_norm(nn.Conv2d(out_channels, 1, 4, stride=1, padding=1), spectral_norm),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor, region_condition: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}")
        if region_condition.ndim != 4:
            raise ValueError(
                "region_condition must have shape [B,C,H,W], "
                f"got {tuple(region_condition.shape)}"
            )
        if image.shape[0] != region_condition.shape[0]:
            raise ValueError("image and region_condition batch sizes must match")
        if image.shape[-2:] != region_condition.shape[-2:]:
            region_condition = F.interpolate(
                region_condition.float(),
                size=image.shape[-2:],
                mode="nearest",
            )
        return self.net(torch.cat([image, region_condition.to(dtype=image.dtype)], dim=1))


def patch_mask_from_region(
    target_region: torch.Tensor,
    logits: torch.Tensor,
    *,
    mode: str = "non_background",
) -> torch.Tensor | None:
    """Create a patch-level adversarial mask from integer target-region labels."""

    mode = str(mode).strip().lower()
    if mode == "all":
        return None
    if mode not in {"non_background", "foreground", "tissue"}:
        raise ValueError("adv mask mode must be all or non_background")
    if target_region.ndim == 3:
        target_region = target_region.unsqueeze(1)
    if target_region.ndim != 4 or target_region.shape[1] != 1:
        raise ValueError(
            f"target_region must have shape [B,1,H,W] or [B,H,W], got {tuple(target_region.shape)}"
        )
    foreground = (target_region != 0).float()
    return F.interpolate(foreground, size=logits.shape[-2:], mode="nearest")


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    values = values.float()
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    if mask.shape != values.shape:
        mask = mask.expand_as(values)
    denominator = mask.sum()
    if float(denominator.detach().item()) <= 0.0:
        return values.mean()
    return (values * mask).sum() / denominator.clamp_min(1.0)


def discriminator_hinge_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    real_loss = masked_mean(F.relu(1.0 - real_logits), mask)
    fake_loss = masked_mean(F.relu(1.0 + fake_logits), mask)
    return 0.5 * (real_loss + fake_loss)


def generator_hinge_loss(
    fake_logits: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return -masked_mean(fake_logits, mask)


def discriminator_logit_stats(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return masked_mean(real_logits, mask), masked_mean(fake_logits, mask)
