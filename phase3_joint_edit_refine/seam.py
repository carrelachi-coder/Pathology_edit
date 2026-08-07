"""Anchor-conditioned cellular continuity zones for joint tissue edits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from phase3_mask_edit.core.labels import MaskProfileSchema

from .models import JointContractError
from .nuclei import iter_instances
from .scene import JointSceneAnalysis
from .skills.schema import SeamContract


@dataclass(frozen=True)
class AdaptiveSeam:
    """One candidate-local seam compiled from Planner anchors and cell scale."""

    mode: str
    anchor_mask: np.ndarray
    continuity_region: np.ndarray
    cell_diameter_px: float
    local_nnd_px: float
    width_px: int
    maximum_empty_run_px: int
    minimum_anchor_coverage_fraction: float
    density_ratio_range: tuple[float, float]
    reference_area_quantiles: tuple[float, float]
    requires_new_target_cells: bool

    def to_metadata(self) -> dict:
        return {
            "mode": self.mode,
            "anchor_pixels": int(np.count_nonzero(self.anchor_mask)),
            "continuity_region_pixels": int(
                np.count_nonzero(self.continuity_region)
            ),
            "cell_diameter_px": float(self.cell_diameter_px),
            "local_nnd_px": float(self.local_nnd_px),
            "width_px": int(self.width_px),
            "maximum_empty_run_px": int(self.maximum_empty_run_px),
            "minimum_anchor_coverage_fraction": float(
                self.minimum_anchor_coverage_fraction
            ),
            "density_ratio_range": list(self.density_ratio_range),
            "reference_area_quantiles": list(self.reference_area_quantiles),
            "requires_new_target_cells": bool(self.requires_new_target_cells),
            "definition": "planner_anchor_x_actual_tissue_change_x_local_cell_scale",
        }


def target_cell_class_for_tissue(
    target_label: str,
    schema: MaskProfileSchema,
) -> int:
    """Resolve the executable observation class for a tissue target.

    Keeping this resolver next to seam compilation avoids letting Planner
    ordering choose a class when a mechanism exposes several legal classes.
    ``schema`` remains part of the API because the mapping is an executable
    tissue/cell contract, even though the current CellViT-5 mapping is
    canonical-label based.
    """

    del schema
    if target_label == "Tumor":
        return 1
    if target_label in {"Stroma", "Other tissue"}:
        return 3
    if target_label == "Immune infiltrate":
        return 2
    if target_label == "Necrosis":
        # CellViT class-4 dead nuclei are a valid but very sparse observation
        # in the public population libraries.  Class-2 inflammatory nuclei
        # provide the stable capacity/reference scaffold for necrosis; the
        # executable primitive may still allow both 2 and 4 at sampling time.
        return 2
    if target_label == "Normal epithelium":
        return 5
    raise JointContractError(
        f"no executable cell-class contract for {target_label!r}"
    )


def compile_adaptive_seam(
    *,
    scene: JointSceneAnalysis,
    tissue_change: np.ndarray,
    interface_ids: tuple[str, ...],
    anchor_ids: tuple[str, ...],
    target_class: int,
    contract: SeamContract,
) -> AdaptiveSeam:
    """Compile a seam after Planner anchoring and tissue candidate drawing.

    ``halo_distance_px`` is deliberately absent.  A biological mechanism halo,
    a complete nucleus footprint and a cellular continuity seam are independent
    quantities and must never share one radius.
    """

    change = np.asarray(tissue_change, dtype=bool)
    anchor = np.zeros_like(change)
    for anchor_id in anchor_ids:
        current = scene.tissue.anchor_masks.get(anchor_id)
        if current is not None:
            anchor |= np.asarray(current, dtype=bool)
    # A legacy/research plan may expose only an interface ID.  Keep the fallback
    # explicit in metadata through the selected IDs; production Planner
    # validation still requires real anchors.
    if not np.any(anchor):
        for interface_id in interface_ids:
            current = scene.tissue.interface_masks.get(interface_id)
            if current is not None:
                anchor |= np.asarray(current, dtype=bool)

    diameter, local_nnd = _local_cell_scale(
        scene,
        interface_ids=interface_ids,
        target_class=target_class,
    )
    local_spacing = max(diameter, local_nnd)
    low, high = contract.width_cell_diameters
    width = round(
        np.clip(
            local_spacing,
            max(1.0, low * diameter),
            max(1.0, high * diameter),
        )
    )
    maximum_empty_run = max(
        1,
        round(contract.maximum_empty_run_cell_diameters * local_spacing),
    )
    if contract.mode == "not_applicable" or not np.any(change) or not np.any(anchor):
        active_anchor = np.zeros_like(change)
        region = np.zeros_like(change)
    else:
        active_anchor = anchor & ndimage.binary_dilation(change, iterations=2)
        if not np.any(active_anchor):
            active_anchor = anchor
        region = (
            change
            & ndimage.binary_dilation(active_anchor, iterations=max(1, width))
        )
    return AdaptiveSeam(
        mode=contract.mode,
        anchor_mask=active_anchor,
        continuity_region=region,
        cell_diameter_px=float(diameter),
        local_nnd_px=float(local_nnd),
        width_px=width,
        maximum_empty_run_px=maximum_empty_run,
        minimum_anchor_coverage_fraction=(
            contract.minimum_anchor_coverage_fraction
        ),
        density_ratio_range=contract.density_ratio_range,
        reference_area_quantiles=contract.reference_area_quantiles,
        requires_new_target_cells=contract.requires_new_target_cells,
    )


def anchor_coverage_fraction(
    anchor_mask: np.ndarray,
    center_mask: np.ndarray,
    *,
    maximum_empty_run_px: int,
) -> float:
    """Fraction of the edited anchor within the permitted center-gap radius."""

    anchor = np.asarray(anchor_mask, dtype=bool)
    centers = np.asarray(center_mask, dtype=bool)
    if not np.any(anchor):
        return 1.0
    if not np.any(centers):
        return 0.0
    distance = ndimage.distance_transform_edt(~centers)
    return float(
        np.mean(distance[anchor] <= max(1, int(maximum_empty_run_px)))
    )


def class_center_mask(mask: np.ndarray, *, class_id: int) -> np.ndarray:
    """Rasterize one center per complete semantic/native-compatible component."""

    values = np.asarray(mask)
    result = np.zeros_like(values, dtype=bool)
    for _, current_class, component in iter_instances(values):
        if current_class != class_id or not np.any(component):
            continue
        row, col = ndimage.center_of_mass(component)
        row, col = round(row), round(col)
        if 0 <= row < result.shape[0] and 0 <= col < result.shape[1]:
            result[row, col] = True
    return result


def _local_cell_scale(
    scene: JointSceneAnalysis,
    *,
    interface_ids: tuple[str, ...],
    target_class: int,
) -> tuple[float, float]:
    selected = [
        item
        for item in scene.cells.instances
        if item.class_id == target_class
        and not item.touches_border
        and "merged_suspect" not in item.quality_flags
        and (
            not interface_ids
            or item.nearest_interface_id in set(interface_ids)
        )
    ]
    if len(selected) < 3:
        selected = [
            item
            for item in scene.cells.instances
            if item.class_id == target_class
            and not item.touches_border
            and "merged_suspect" not in item.quality_flags
        ]
    diameters = [
        2.0 * np.sqrt(max(1.0, float(item.area_px)) / np.pi)
        for item in selected
    ]
    diameter = float(
        np.median(diameters)
        if diameters
        else (scene.population.nominal_nucleus_diameter_px or 8.0)
    )
    centers = np.asarray(
        [(item.centroid_xy[1], item.centroid_xy[0]) for item in selected],
        dtype=float,
    )
    if len(centers) >= 2:
        distances, _ = cKDTree(centers).query(centers, k=2)
        nnd = float(np.median(distances[:, 1]))
    else:
        nnd = float(scene.cells.mean_nearest_neighbor_px or 1.25 * diameter)
    return max(2.0, diameter), max(2.0, nnd)
