"""Cross V2.1 conditioning: reference latent plus reference/target masks.

V2.1 deliberately removes IP-Adapter reference attention. Reference appearance
is carried by the first ControlNet condition block, ``z_ref``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from controlnet_train.training.conditioning import packed_control_channels


@dataclass(frozen=True)
class CrossV21ControlSpec:
    """Fixed Cross V2.1 ControlNet condition layout.

    Raw condition order:
    [z_ref, ref_tissue_feat, ref_nuclei_feat, tar_tissue_feat, tar_nuclei_feat]
    """

    reference_latent_channels: int = 16
    tissue_channels: int = 64
    nuclei_channels: int = 16

    @property
    def raw_channels(self) -> int:
        mask_channels = self.tissue_channels + self.nuclei_channels
        return self.reference_latent_channels + mask_channels * 2

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)

    @property
    def packed_reference_latent_channels(self) -> int:
        return packed_control_channels(self.reference_latent_channels)

    @property
    def packed_mask_channels(self) -> int:
        return packed_control_channels(self.tissue_channels + self.nuclei_channels)

    @property
    def packed_reference_mask_start(self) -> int:
        return self.packed_reference_latent_channels

    @property
    def packed_target_mask_start(self) -> int:
        return self.packed_reference_latent_channels + self.packed_mask_channels


def build_cross_v2_1_condition(
    *,
    z_ref: torch.Tensor,
    ref_tissue_feat: torch.Tensor,
    ref_nuclei_feat: torch.Tensor,
    tar_tissue_feat: torch.Tensor,
    tar_nuclei_feat: torch.Tensor,
) -> torch.Tensor:
    """Concatenate Cross V2.1 ControlNet conditions in the planned order."""

    features = {
        "z_ref": z_ref,
        "ref_tissue_feat": ref_tissue_feat,
        "ref_nuclei_feat": ref_nuclei_feat,
        "tar_tissue_feat": tar_tissue_feat,
        "tar_nuclei_feat": tar_nuclei_feat,
    }
    for name, value in features.items():
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(value.shape)}.")

    for name, value in features.items():
        if z_ref.shape[0] != value.shape[0] or z_ref.shape[2:] != value.shape[2:]:
            raise ValueError(
                f"{name} must match z_ref on batch/spatial dims, "
                f"got {tuple(value.shape)} vs {tuple(z_ref.shape)}."
            )

    return torch.cat(
        [
            z_ref,
            ref_tissue_feat,
            ref_nuclei_feat,
            tar_tissue_feat,
            tar_nuclei_feat,
        ],
        dim=1,
    )


def deterministic_latent_from_posterior(posterior) -> torch.Tensor:
    """Return a stable latent from a VAE posterior for ControlNet conditioning."""

    return posterior.mode() if hasattr(posterior, "mode") else posterior.mean
