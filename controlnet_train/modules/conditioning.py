"""Helpers for assembling Phase 5 conditioning tensors."""

from __future__ import annotations

import torch


def _validate_spatial_feature(name: str, value: torch.Tensor) -> None:
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape (B, C, H, W), got {tuple(value.shape)}.")


def _validate_same_shape_prefix(reference_name: str, reference: torch.Tensor, other_name: str, other: torch.Tensor) -> None:
    if reference.shape[0] != other.shape[0] or reference.shape[2:] != other.shape[2:]:
        raise ValueError(
            f"{other_name} must match {reference_name} on batch/spatial dims, "
            f"got {tuple(other.shape)} vs {tuple(reference.shape)}."
        )


def build_inpaint_condition(
    *,
    source_image_latent: torch.Tensor,
    target_tissue_feat: torch.Tensor,
    target_nuclei_feat: torch.Tensor,
    change_mask_feat: torch.Tensor,
) -> torch.Tensor:
    """Concatenate inpaint-ControlNet conditions in plan order."""

    features = {
        "source_image_latent": source_image_latent,
        "target_tissue_feat": target_tissue_feat,
        "target_nuclei_feat": target_nuclei_feat,
        "change_mask_feat": change_mask_feat,
    }
    for name, value in features.items():
        _validate_spatial_feature(name, value)

    reference = source_image_latent
    for name, value in features.items():
        _validate_same_shape_prefix("source_image_latent", reference, name, value)

    return torch.cat(
        [
            source_image_latent,
            target_tissue_feat,
            target_nuclei_feat,
            change_mask_feat,
        ],
        dim=1,
    )
