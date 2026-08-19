"""Routing helpers for the unified Phase 5.4 edit pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


@dataclass(frozen=True)
class AgenticRoutingConfig:
    """Policy for local-preserving inpaint versus structural cross generation."""

    t_inpaint: float = 0.12
    t_cross: float = 0.30
    max_local_components: int = 4
    max_local_bbox_fraction: float = 0.45
    max_local_transitions: int = 2
    distributed_component_threshold: int = 8
    distributed_bbox_fraction: float = 0.60
    enable_gray_zone_dual_run: bool = True
    cell_only_decrease_cross_first: bool = True
    cell_only_increase_inpaint_first: bool = True
    generic_immune_decrease_cross_first: bool = True


@dataclass(frozen=True)
class AgenticRouteFeatures:
    change_ratio_image: float
    change_ratio_tissue: float
    component_count: int
    largest_component_fraction: float
    bbox_fraction: float
    transition_count: int
    changed_tissue_ids_from: tuple[int, ...]
    changed_tissue_ids_to: tuple[int, ...]


@dataclass(frozen=True)
class AgenticRoutingDecision:
    primary_mode: str
    candidate_modes: tuple[str, ...]
    confidence: float
    reason: str
    features: AgenticRouteFeatures


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


def compute_agentic_route_features(
    reference_tissue_mask: torch.Tensor | np.ndarray,
    target_tissue_mask: torch.Tensor | np.ndarray,
    *,
    background_ids: tuple[int, ...] = (0,),
    change_region: torch.Tensor | np.ndarray | None = None,
) -> AgenticRouteFeatures:
    """Measure edit extent, topology, and semantic transition complexity."""

    reference = _to_numpy_mask(reference_tissue_mask)
    target = _to_numpy_mask(target_tissue_mask)
    if reference.shape != target.shape or reference.ndim != 2:
        raise ValueError("reference and target tissue masks must be same-shape 2D arrays.")
    if change_region is None:
        change = reference != target
    else:
        change = _to_numpy_mask(change_region).astype(bool)
        if change.shape != reference.shape:
            raise ValueError(
                "change_region must match reference and target tissue masks."
            )
    changed_pixels = int(np.count_nonzero(change))
    image_pixels = int(change.size)
    tissue_union = ~np.isin(reference, background_ids) | ~np.isin(target, background_ids)
    tissue_pixels = int(np.count_nonzero(tissue_union))

    component_count = 0
    largest_component = 0
    bbox_fraction = 0.0
    if changed_pixels:
        from scipy import ndimage

        components, component_count = ndimage.label(change)
        sizes = np.bincount(components.ravel())[1:]
        largest_component = int(sizes.max()) if sizes.size else 0
        ys, xs = np.where(change)
        bbox_pixels = int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        bbox_fraction = bbox_pixels / image_pixels

    semantic_change = change & (reference != target)
    transitions = set(
        zip(
            reference[semantic_change].astype(int).tolist(),
            target[semantic_change].astype(int).tolist(),
        )
    )
    return AgenticRouteFeatures(
        change_ratio_image=changed_pixels / image_pixels if image_pixels else 0.0,
        change_ratio_tissue=changed_pixels / tissue_pixels if tissue_pixels else 0.0,
        component_count=int(component_count),
        largest_component_fraction=(
            largest_component / changed_pixels if changed_pixels else 0.0
        ),
        bbox_fraction=float(bbox_fraction),
        transition_count=len(transitions),
        changed_tissue_ids_from=tuple(
            sorted(int(v) for v in np.unique(reference[semantic_change]))
        ),
        changed_tissue_ids_to=tuple(
            sorted(int(v) for v in np.unique(target[semantic_change]))
        ),
    )


def route_agentic_edit_request(
    reference_tissue_mask: torch.Tensor | np.ndarray,
    target_tissue_mask: torch.Tensor | np.ndarray,
    *,
    config: AgenticRoutingConfig | None = None,
    change_region: torch.Tensor | np.ndarray | None = None,
    change_scope: str = "tissue",
    cell_only_direction: str | None = None,
    edit_primitive_id: str | None = None,
) -> AgenticRoutingDecision:
    """Return an explainable primary route and bounded fallback candidates."""

    config = config or AgenticRoutingConfig()
    if not 0.0 <= config.t_inpaint <= config.t_cross <= 1.0:
        raise ValueError("Agentic thresholds must satisfy 0 <= t_inpaint <= t_cross <= 1.")
    if change_scope not in {"tissue", "nuclei"}:
        raise ValueError("change_scope must be 'tissue' or 'nuclei'.")
    if cell_only_direction not in {None, "increase", "decrease"}:
        raise ValueError(
            "cell_only_direction must be None, 'increase', or 'decrease'."
        )
    if edit_primitive_id is not None and not str(edit_primitive_id).strip():
        raise ValueError("edit_primitive_id must be a non-empty string when supplied.")
    features = compute_agentic_route_features(
        reference_tissue_mask,
        target_tissue_mask,
        change_region=change_region,
    )
    if features.change_ratio_image == 0.0:
        return AgenticRoutingDecision(
            primary_mode="noop",
            candidate_modes=("noop",),
            confidence=1.0,
            reason=(
                "reference and target tissue masks are identical"
                if change_scope == "tissue"
                else "reference and target nuclei masks are identical"
            ),
            features=features,
        )

    if change_scope == "nuclei":
        decrease_cross_first = (
            cell_only_direction == "decrease"
            and config.cell_only_decrease_cross_first
        )
        increase_inpaint_first = (
            cell_only_direction == "increase"
            and config.cell_only_increase_inpaint_first
        )
        if decrease_cross_first:
            candidates = (
                ("cross", "inpaint")
                if config.enable_gray_zone_dual_run
                else ("cross",)
            )
            return AgenticRoutingDecision(
                primary_mode="cross",
                candidate_modes=candidates,
                confidence=0.90,
                reason=(
                    "cell-only decrease; start with Cross-v1 to preserve "
                    "nucleus-scale structure without local inpaint blur, with "
                    "inpaint retained as the bounded preservation fallback"
                ),
                features=features,
            )
        candidates = (
            ("inpaint", "cross")
            if config.enable_gray_zone_dual_run
            else ("inpaint",)
        )
        return AgenticRoutingDecision(
            primary_mode="inpaint",
            candidate_modes=candidates,
            confidence=0.90 if increase_inpaint_first else 0.70,
            reason=(
                "cell-only increase; start with preservation-oriented inpaint, "
                "with Cross-v1 retained as the bounded structural fallback"
                if increase_inpaint_first
                else "cell-only direction is unavailable; start with the "
                "preservation-oriented inpaint route and retain Cross-v1 as "
                "the bounded fallback"
            ),
            features=features,
        )

    generic_immune_decrease = (
        edit_primitive_id == "generic-immune-infiltrate-decrease-v1"
        and config.generic_immune_decrease_cross_first
    )
    if generic_immune_decrease:
        candidates = (
            ("cross", "inpaint")
            if config.enable_gray_zone_dual_run
            else ("cross",)
        )
        return AgenticRoutingDecision(
            primary_mode="cross",
            candidate_modes=candidates,
            confidence=0.90,
            reason=(
                "generic immune-infiltrate decrease starts with Cross-v1 to "
                "avoid localized inpaint blur and false cell-like texture; "
                "inpaint remains the bounded fallback"
            ),
            features=features,
        )

    distributed = (
        features.component_count >= config.distributed_component_threshold
        and features.bbox_fraction >= config.distributed_bbox_fraction
    )
    structurally_large = (
        features.change_ratio_tissue >= config.t_cross
        or features.transition_count > config.max_local_transitions
        or distributed
    )
    compact_local = (
        features.change_ratio_tissue <= config.t_inpaint
        and features.component_count <= config.max_local_components
        and features.bbox_fraction <= config.max_local_bbox_fraction
        and features.transition_count <= config.max_local_transitions
    )
    if structurally_large:
        reasons = []
        if features.change_ratio_tissue >= config.t_cross:
            reasons.append(f"tissue-normalized change {features.change_ratio_tissue:.1%}")
        if features.transition_count > config.max_local_transitions:
            reasons.append(f"{features.transition_count} semantic transitions")
        if distributed:
            reasons.append("distributed multi-component edit")
        return AgenticRoutingDecision(
            primary_mode="cross",
            candidate_modes=("cross", "inpaint"),
            confidence=0.90,
            reason=(
                ("cell-only nuclei edit; " if change_scope == "nuclei" else "")
                + "; ".join(reasons)
            ),
            features=features,
        )
    if compact_local:
        return AgenticRoutingDecision(
            primary_mode="inpaint",
            candidate_modes=("inpaint", "cross"),
            confidence=0.90,
            reason=(
                ("cell-only nuclei edit; " if change_scope == "nuclei" else "")
                + f"compact local edit: tissue-normalized change {features.change_ratio_tissue:.1%}, "
                f"components={features.component_count}, bbox={features.bbox_fraction:.1%}"
            ),
            features=features,
        )

    candidates = (
        ("inpaint", "cross")
        if config.enable_gray_zone_dual_run
        else ("inpaint",)
    )
    return AgenticRoutingDecision(
        primary_mode="inpaint",
        candidate_modes=candidates,
        confidence=0.55,
        reason=(
            ("cell-only nuclei edit; " if change_scope == "nuclei" else "")
            + "gray-zone edit; start with preservation-oriented inpaint and allow "
            "cross-v1 plus pix2pix-v2 fallback"
        ),
        features=features,
    )


def _to_numpy_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask)
