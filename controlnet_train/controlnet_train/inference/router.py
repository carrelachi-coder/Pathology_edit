"""Routing helpers for the unified Phase 5.4 edit pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EditRoutingConfig:
    t_inpaint: float = 0.12
    t_cross: float = 0.30


@dataclass(frozen=True)
class EditRoutingDecision:
    change_region_mask: torch.Tensor
    change_ratio: float
    selected_mode: str
    changed_tissue_ids_from: list[int]
    changed_tissue_ids_to: list[int]


def compute_change_region_mask(
    reference_tissue_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
) -> torch.Tensor:
    if reference_tissue_mask.ndim != 2 or target_tissue_mask.ndim != 2:
        raise ValueError(
            "reference_tissue_mask and target_tissue_mask must both have shape (H, W)."
        )
    if reference_tissue_mask.shape != target_tissue_mask.shape:
        raise ValueError(
            "reference_tissue_mask and target_tissue_mask must have the same shape."
        )
    return (reference_tissue_mask != target_tissue_mask).to(dtype=torch.float32)


def route_edit_request(
    reference_tissue_mask: torch.Tensor,
    target_tissue_mask: torch.Tensor,
    config: EditRoutingConfig | None = None,
) -> EditRoutingDecision:
    config = config or EditRoutingConfig()
    if config.t_inpaint < 0 or config.t_cross < 0:
        raise ValueError("Routing thresholds must be non-negative.")
    if config.t_inpaint > config.t_cross:
        raise ValueError("t_inpaint must be less than or equal to t_cross.")

    change_region_mask = compute_change_region_mask(reference_tissue_mask, target_tissue_mask)
    change_ratio = float(change_region_mask.mean().item())

    if change_ratio <= config.t_inpaint:
        selected_mode = "inpaint"
    elif change_ratio >= config.t_cross:
        selected_mode = "cross"
    else:
        selected_mode = "inpaint"

    changed = change_region_mask.bool()
    changed_tissue_ids_from = (
        sorted(int(v) for v in torch.unique(reference_tissue_mask[changed]).tolist())
        if changed.any()
        else []
    )
    changed_tissue_ids_to = (
        sorted(int(v) for v in torch.unique(target_tissue_mask[changed]).tolist())
        if changed.any()
        else []
    )

    return EditRoutingDecision(
        change_region_mask=change_region_mask,
        change_ratio=change_ratio,
        selected_mode=selected_mode,
        changed_tissue_ids_from=changed_tissue_ids_from,
        changed_tissue_ids_to=changed_tissue_ids_to,
    )
