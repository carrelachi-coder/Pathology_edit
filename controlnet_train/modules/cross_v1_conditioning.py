"""Cross V1 conditioning spec and builder — spatial conditioning only, no reference image latent.

Reference appearance is injected via IP-Adapter cross-attention, not VAE latent concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from controlnet_train.training.conditioning import packed_control_channels


@dataclass(frozen=True)
class CrossV1ControlSpec:
    """Cross V1: spatial conditioning only (no reference image latent).
    Reference appearance is injected via IP-Adapter cross-attention."""
    tissue_channels: int = 64
    nuclei_channels: int = 16

    @property
    def raw_channels(self) -> int:
        # ref_tissue + ref_nuclei + tgt_tissue + tgt_nuclei
        return self.tissue_channels + self.nuclei_channels + self.tissue_channels + self.nuclei_channels

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)


def build_cross_v1_condition(
    *,
    reference_tissue_feat: torch.Tensor,
    reference_nuclei_feat: torch.Tensor,
    target_tissue_feat: torch.Tensor,
    target_nuclei_feat: torch.Tensor,
) -> torch.Tensor:
    """Concatenate V1 cross-ControlNet conditions (no reference image latent)."""

    features = {
        "reference_tissue_feat": reference_tissue_feat,
        "reference_nuclei_feat": reference_nuclei_feat,
        "target_tissue_feat": target_tissue_feat,
        "target_nuclei_feat": target_nuclei_feat,
    }
    for name, value in features.items():
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(value.shape)}.")

    reference = reference_tissue_feat
    for name, value in features.items():
        if reference.shape[0] != value.shape[0] or reference.shape[2:] != value.shape[2:]:
            raise ValueError(
                f"{name} must match reference_tissue_feat on batch/spatial dims, "
                f"got {tuple(value.shape)} vs {tuple(reference.shape)}."
            )

    return torch.cat(
        [
            reference_tissue_feat,
            reference_nuclei_feat,
            target_tissue_feat,
            target_nuclei_feat,
        ],
        dim=1,
    )