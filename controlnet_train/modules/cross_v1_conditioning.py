"""Cross V1 conditioning spec and builder — spatial conditioning only, no reference image latent.

Reference appearance is injected via IP-Adapter cross-attention, not VAE latent concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from controlnet_train.training.conditioning import packed_control_channels


CROSS_V1_SPATIAL_REFERENCE_TARGET = "reference_target"
CROSS_V1_SPATIAL_TARGET_ONLY = "target_only"
CROSS_V1_SPATIAL_MODES = (
    CROSS_V1_SPATIAL_REFERENCE_TARGET,
    CROSS_V1_SPATIAL_TARGET_ONLY,
)


def normalize_cross_v1_spatial_mode(spatial_mode: str | None) -> str:
    mode = (spatial_mode or CROSS_V1_SPATIAL_REFERENCE_TARGET).strip().lower().replace("-", "_")
    if mode not in CROSS_V1_SPATIAL_MODES:
        raise ValueError(
            f"Unsupported Cross V1 spatial mode {spatial_mode!r}; "
            f"choose from {', '.join(CROSS_V1_SPATIAL_MODES)}."
        )
    return mode


@dataclass(frozen=True)
class CrossV1ControlSpec:
    """Cross V1: spatial conditioning only (no reference image latent).
    Reference appearance is injected via IP-Adapter cross-attention."""
    tissue_channels: int = 64
    nuclei_channels: int = 16
    spatial_mode: str = CROSS_V1_SPATIAL_REFERENCE_TARGET

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "spatial_mode",
            normalize_cross_v1_spatial_mode(self.spatial_mode),
        )

    @property
    def raw_channels(self) -> int:
        target_channels = self.tissue_channels + self.nuclei_channels
        if self.spatial_mode == CROSS_V1_SPATIAL_TARGET_ONLY:
            return target_channels
        # ref_tissue + ref_nuclei + tgt_tissue + tgt_nuclei
        return target_channels * 2

    @property
    def packed_channels(self) -> int:
        return packed_control_channels(self.raw_channels)

    @property
    def packed_target_start(self) -> int:
        if self.spatial_mode == CROSS_V1_SPATIAL_TARGET_ONLY:
            return 0
        return packed_control_channels(self.tissue_channels + self.nuclei_channels)

    @property
    def packed_target_channels(self) -> int:
        return packed_control_channels(self.tissue_channels + self.nuclei_channels)


def build_cross_v1_condition(
    *,
    reference_tissue_feat: torch.Tensor | None = None,
    reference_nuclei_feat: torch.Tensor | None = None,
    target_tissue_feat: torch.Tensor,
    target_nuclei_feat: torch.Tensor,
    spatial_mode: str = CROSS_V1_SPATIAL_REFERENCE_TARGET,
) -> torch.Tensor:
    """Concatenate V1 cross-ControlNet conditions (no reference image latent)."""
    spatial_mode = normalize_cross_v1_spatial_mode(spatial_mode)

    features = {
        "target_tissue_feat": target_tissue_feat,
        "target_nuclei_feat": target_nuclei_feat,
    }
    if spatial_mode == CROSS_V1_SPATIAL_REFERENCE_TARGET:
        if reference_tissue_feat is None or reference_nuclei_feat is None:
            raise ValueError("reference_tissue_feat and reference_nuclei_feat are required in reference_target mode.")
        features["reference_tissue_feat"] = reference_tissue_feat
        features["reference_nuclei_feat"] = reference_nuclei_feat

    for name, value in features.items():
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(value.shape)}.")

    reference = target_tissue_feat
    for name, value in features.items():
        if reference.shape[0] != value.shape[0] or reference.shape[2:] != value.shape[2:]:
            raise ValueError(
                f"{name} must match target_tissue_feat on batch/spatial dims, "
                f"got {tuple(value.shape)} vs {tuple(reference.shape)}."
            )

    if spatial_mode == CROSS_V1_SPATIAL_TARGET_ONLY:
        return torch.cat(
            [
                target_tissue_feat,
                target_nuclei_feat,
            ],
            dim=1,
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
