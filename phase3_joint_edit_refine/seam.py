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


@dataclass(frozen=True)
class ContinuityCenterQuota:
    """Finite-raster compilation of a seam density envelope."""

    minimum_count: int
    maximum_count: int | None
    target_count: int
    expected_count: float
    outer_count: int
    outer_pixels: int
    inner_pixels: int
    outer_density: float


def compile_executable_continuity_count(
    quota: ContinuityCenterQuota,
    *,
    anchor_pixels: int,
    maximum_empty_run_px: int,
    minimum_anchor_coverage_fraction: float,
) -> int:
    """Resolve density and finite-anchor continuity into one integer quota.

    A density-only target can be too small to cover a long edited interface,
    while blindly taking the skill's density upper bound can overpopulate a
    short seam.  One center can cover at most roughly one diameter of a
    one-pixel interface under the empty-run gate.  The resulting geometric
    lower bound is intersected with the observed density interval.  This is a
    conservative finite-raster compiler, not an organ or case heuristic.
    """

    density_target = (
        int(np.ceil((quota.target_count + quota.maximum_count) / 2.0))
        if quota.maximum_count is not None
        else int(quota.target_count)
    )
    minimum = compile_minimum_continuity_count(
        quota,
        anchor_pixels=anchor_pixels,
        maximum_empty_run_px=maximum_empty_run_px,
        minimum_anchor_coverage_fraction=minimum_anchor_coverage_fraction,
    )
    target = max(minimum, density_target)
    if quota.maximum_count is not None:
        target = min(target, int(quota.maximum_count))
    return max(0, int(target))


def compile_minimum_continuity_count(
    quota: ContinuityCenterQuota,
    *,
    anchor_pixels: int,
    maximum_empty_run_px: int,
    minimum_anchor_coverage_fraction: float,
) -> int:
    """Return the hard lower edge of a finite-raster seam contract.

    The executable target above is a preferred point inside the observed
    density interval.  Exact footprint packing may prove that preferred point
    unreachable even though a smaller count still satisfies both the density
    lower bound and the anchor-coverage geometry.  Keeping this hard minimum
    explicit lets the packing certifier make that distinction without
    weakening either constraint.
    """

    pixels_per_center = max(1, 2 * int(maximum_empty_run_px) + 1)
    geometric_minimum = int(
        np.ceil(
            max(0, int(anchor_pixels))
            * float(np.clip(minimum_anchor_coverage_fraction, 0.0, 1.0))
            / float(pixels_per_center)
        )
    )
    minimum = max(int(quota.minimum_count), geometric_minimum)
    if quota.maximum_count is not None:
        minimum = min(minimum, int(quota.maximum_count))
    return max(0, int(minimum))


def compile_continuity_center_quota(
    *,
    nuclei_mask: np.ndarray,
    target_tissue_mask: np.ndarray,
    tissue_change: np.ndarray,
    continuity_region: np.ndarray,
    continuity_anchor_mask: np.ndarray,
    continuity_width_px: int,
    density_ratio_range: tuple[float, float],
    requires_new_target_cells: bool,
    target_class: int,
    target_fine_ids: tuple[int, ...],
    target_center_mask: np.ndarray | None = None,
) -> ContinuityCenterQuota:
    """Compile the gate's density interval into an executable center quota.

    The unchanged target-tissue band is observable before cell generation, so
    Planner/tool execution and the later gate can use the exact same integer
    interval.  ``target_count`` is the deterministic point inside that interval
    closest to the observed local expectation.
    """

    change = np.asarray(tissue_change, dtype=bool)
    inner = np.asarray(continuity_region, dtype=bool)
    anchor = np.asarray(continuity_anchor_mask, dtype=bool)
    outer = (
        ~change
        & _binary_dilation_taxicab(
            anchor, max(1, int(continuity_width_px))
        )
        & np.isin(np.asarray(target_tissue_mask), target_fine_ids)
    )
    centers = (
        class_center_mask(nuclei_mask, class_id=target_class)
        if target_center_mask is None
        else np.asarray(target_center_mask, dtype=bool)
    )
    if centers.shape != inner.shape:
        raise JointContractError(
            "continuity center ledger and seam geometry must align"
        )
    outer_count = int(np.count_nonzero(centers & outer))
    inner_pixels = int(np.count_nonzero(inner))
    outer_pixels = int(np.count_nonzero(outer))
    outer_density = outer_count / max(1, outer_pixels)
    expected = outer_density * inner_pixels
    lower, upper = density_ratio_range
    minimum = int(np.ceil(lower * expected - 1e-12))
    if requires_new_target_cells:
        minimum = max(1, minimum)
    maximum = None
    if outer_pixels > 0 and outer_count > 0:
        maximum = max(
            minimum,
            int(np.ceil(upper * expected - 1e-12)),
        )
    target = max(minimum, round(expected))
    if maximum is not None:
        target = min(target, maximum)
    return ContinuityCenterQuota(
        minimum_count=minimum,
        maximum_count=maximum,
        target_count=target,
        expected_count=float(expected),
        outer_count=outer_count,
        outer_pixels=outer_pixels,
        inner_pixels=inner_pixels,
        outer_density=float(outer_density),
    )


def target_cell_class_for_tissue(
    target_label: str,
    schema: MaskProfileSchema | None,
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
        active_anchor = anchor & _binary_dilation_taxicab(change, 2)
        if not np.any(active_anchor):
            active_anchor = anchor
        region = (
            change
            & _binary_dilation_taxicab(active_anchor, max(1, width))
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


def _binary_dilation_taxicab(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Fast exact replacement for SciPy's repeated default dilation."""

    region = np.asarray(mask, dtype=bool)
    count = max(0, int(iterations))
    if count == 0 or not np.any(region):
        return region.copy()
    distance = ndimage.distance_transform_cdt(
        ~region,
        metric="taxicab",
    )
    return distance <= count


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
