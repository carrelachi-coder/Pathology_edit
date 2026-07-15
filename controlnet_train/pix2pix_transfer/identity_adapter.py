"""Family-wise WSI appearance modulation for pix2pix reference transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must have shape [B,1,H,W] or [B,H,W], got {tuple(mask.shape)}")
    return mask


def masked_channel_stats(
    feature: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_pixels: int,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return channel mean/log-std and a per-sample support gate."""

    binary = _as_mask(mask).to(device=feature.device).ne(0)
    support = binary.flatten(1).sum(dim=1).ge(max(1, int(min_pixels)))
    weight = F.interpolate(binary.float(), size=feature.shape[-2:], mode="area")
    denominator = weight.sum(dim=(2, 3)).clamp_min(eps)
    mean = (feature.float() * weight).sum(dim=(2, 3)) / denominator
    centered = feature.float() - mean[:, :, None, None]
    variance = (centered.square() * weight).sum(dim=(2, 3)) / denominator
    log_std = 0.5 * torch.log(variance.clamp_min(eps))
    gate = support.to(dtype=feature.dtype)[:, None]
    return mean.to(feature.dtype), log_std.to(feature.dtype), gate


class FamilyFeatureFiLM(nn.Module):
    """Map reference feature moments to a bounded affine residual."""

    def __init__(self, channels: int, *, gamma_max: float, gamma_init: float) -> None:
        super().__init__()
        self.channels = int(channels)
        self.gamma_max = float(gamma_max)
        hidden = max(16, self.channels // 2)
        self.input = nn.Linear(self.channels * 2, hidden)
        self.output = nn.Linear(hidden, self.channels * 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.identity_gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def effective_gamma(self) -> float:
        return float(self.identity_gamma.detach().clamp(-self.gamma_max, self.gamma_max).item())

    def forward(
        self,
        target_feature: torch.Tensor,
        reference_feature: torch.Tensor,
        *,
        target_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        min_pixels: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std, support = masked_channel_stats(
            reference_feature,
            reference_mask,
            min_pixels=min_pixels,
        )
        affine = self.output(F.silu(self.input(torch.cat([mean, log_std], dim=1))))
        delta_scale, delta_bias = affine.chunk(2, dim=1)
        delta_scale = torch.tanh(delta_scale)[:, :, None, None]
        delta_bias = torch.tanh(delta_bias)[:, :, None, None]
        target_weight = F.interpolate(
            _as_mask(target_mask).to(device=target_feature.device).float(),
            size=target_feature.shape[-2:],
            mode="area",
        ).clamp(0.0, 1.0).to(dtype=target_feature.dtype)
        gamma = self.identity_gamma.clamp(-self.gamma_max, self.gamma_max).to(target_feature.dtype)
        update = gamma * support[:, :, None, None] * target_weight * (
            target_feature * delta_scale + delta_bias
        )
        return target_feature + update, support


class FamilyWSIIdentityAdapter(nn.Module):
    """Inject non-spatial reference appearance while preserving target layout."""

    def __init__(
        self,
        *,
        channels_by_scale: Mapping[str, int],
        tissue_scales: Sequence[str] = ("1/4", "1/8", "1/16"),
        nuclei_scales: Sequence[str] = ("1/4",),
        gamma_max: float = 0.30,
        gamma_init: float = 0.10,
        min_tissue_pixels: int = 256,
        min_nuclei_pixels: int = 64,
    ) -> None:
        super().__init__()
        self.min_tissue_pixels = int(min_tissue_pixels)
        self.min_nuclei_pixels = int(min_nuclei_pixels)
        self.tissue_adapters = nn.ModuleDict(
            {
                scale: FamilyFeatureFiLM(
                    int(channels_by_scale[scale]),
                    gamma_max=gamma_max,
                    gamma_init=gamma_init,
                )
                for scale in tissue_scales
            }
        )
        self.nuclei_adapters = nn.ModuleDict(
            {
                scale: FamilyFeatureFiLM(
                    int(channels_by_scale[scale]),
                    gamma_max=gamma_max,
                    gamma_init=gamma_init,
                )
                for scale in nuclei_scales
            }
        )

    @staticmethod
    def _family_masks(
        tissue_mask: torch.Tensor,
        nuclei_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tissue = _as_mask(tissue_mask).ne(0)
        nuclei = _as_mask(nuclei_mask).ne(0)
        return tissue & ~nuclei, nuclei

    def forward_scale(
        self,
        scale: str,
        target_feature: torch.Tensor,
        reference_feature: torch.Tensor,
        *,
        target_tissue_mask: torch.Tensor,
        target_nuclei_mask: torch.Tensor,
        reference_tissue_mask: torch.Tensor,
        reference_nuclei_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target_tissue, target_nuclei = self._family_masks(target_tissue_mask, target_nuclei_mask)
        reference_tissue, reference_nuclei = self._family_masks(
            reference_tissue_mask,
            reference_nuclei_mask,
        )
        output = target_feature
        logs = {"tissue_support": 0.0, "nuclei_support": 0.0}
        if scale in self.tissue_adapters:
            output, support = self.tissue_adapters[scale](
                output,
                reference_feature,
                target_mask=target_tissue,
                reference_mask=reference_tissue,
                min_pixels=self.min_tissue_pixels,
            )
            logs["tissue_support"] = float(support.detach().float().mean().item())
        if scale in self.nuclei_adapters:
            output, support = self.nuclei_adapters[scale](
                output,
                reference_feature,
                target_mask=target_nuclei,
                reference_mask=reference_nuclei,
                min_pixels=self.min_nuclei_pixels,
            )
            logs["nuclei_support"] = float(support.detach().float().mean().item())
        return output, logs
