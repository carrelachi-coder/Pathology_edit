"""Conservative H&E/mask/nucleus observation of gland luminal spaces.

This module deliberately produces protection maps rather than tissue labels.
It accepts border-truncated spaces, uses nucleus density as its primary signal,
and treats H&E brightness/stain as supporting evidence.  Ambiguous low-cell
spaces are protected from editing just like confirmed lumina.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .nuclei import iter_instances


STRUCTURE_8 = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class LumenRegion:
    region_id: str
    classification: str
    area_px: int
    nucleus_count: int
    nuclei_per_cell_area: float
    relative_nucleus_density: float
    tumor_contact_fraction: float
    tumor_contact_px: int
    boundary_nucleus_count: int
    boundary_nucleus_support_fraction: float
    median_luminance: float
    white_fraction: float
    median_optical_density: float
    median_raw_luminance: float
    median_raw_optical_density: float
    touches_patch_edge: bool
    patch_edge_contact_fraction: float
    architecture_component_count: int
    region_to_architecture_area_ratio: float
    source_component_id: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "classification": self.classification,
            "area_px": self.area_px,
            "nucleus_count": self.nucleus_count,
            "nuclei_per_cell_area": self.nuclei_per_cell_area,
            "relative_nucleus_density": self.relative_nucleus_density,
            "tumor_contact_fraction": self.tumor_contact_fraction,
            "tumor_contact_px": self.tumor_contact_px,
            "boundary_nucleus_count": self.boundary_nucleus_count,
            "boundary_nucleus_support_fraction": self.boundary_nucleus_support_fraction,
            "median_luminance": self.median_luminance,
            "white_fraction": self.white_fraction,
            "median_optical_density": self.median_optical_density,
            "median_raw_luminance": self.median_raw_luminance,
            "median_raw_optical_density": self.median_raw_optical_density,
            "touches_patch_edge": self.touches_patch_edge,
            "patch_edge_contact_fraction": self.patch_edge_contact_fraction,
            "architecture_component_count": self.architecture_component_count,
            "region_to_architecture_area_ratio": self.region_to_architecture_area_ratio,
            "source_component_id": self.source_component_id,
        }


@dataclass(frozen=True)
class LumenObservation:
    confirmed_lumen: np.ndarray
    uncertain_low_cell_space: np.ndarray
    external_stroma: np.ndarray
    low_cell_seed: np.ndarray
    local_nucleus_count: np.ndarray
    luminance: np.ndarray
    optical_density: np.ndarray
    raw_luminance: np.ndarray
    raw_optical_density: np.ndarray
    nominal_cell_diameter_px: float
    lumen_encoding: str
    regions: tuple[LumenRegion, ...]

    @property
    def protected_space(self) -> np.ndarray:
        return self.confirmed_lumen | self.uncertain_low_cell_space

    def to_metadata(self) -> dict[str, Any]:
        return {
            "observer_id": "annotation-aware-three-layer-lumen-observer-v4",
            "nominal_cell_diameter_px": self.nominal_cell_diameter_px,
            "lumen_encoding": self.lumen_encoding,
            "confirmed_lumen_pixels": int(np.count_nonzero(self.confirmed_lumen)),
            "uncertain_low_cell_space_pixels": int(
                np.count_nonzero(self.uncertain_low_cell_space)
            ),
            "external_stroma_pixels": int(np.count_nonzero(self.external_stroma)),
            "regions": [item.to_metadata() for item in self.regions],
        }


def observe_luminal_spaces(
    image_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    nuclei_mask: np.ndarray,
    *,
    stroma_fine_ids: tuple[int, ...] = (2,),
    architecture_fine_ids: tuple[int, ...],
    lumen_encoding: str = "stroma",
) -> LumenObservation:
    """Observe confirmed and uncertain luminal protection regions.

    ``stroma`` encoding is used when the annotation stores lumen as Stroma
    (PANDA). ``within_architecture`` is used when lumen shares the gland/tumor
    label (GLaS); there the first layer derives a low-cell, lightly stained
    candidate inside the semantic gland component.  Both encodings then apply
    the same nucleus-density, H&E, boundary, and edge-truncation audit.
    """

    image = np.asarray(image_rgb, dtype=np.uint8)
    tissue = np.asarray(tissue_mask)
    nuclei = np.asarray(nuclei_mask)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3")
    if image.shape[:2] != tissue.shape or tissue.shape != nuclei.shape:
        raise ValueError("image, tissue and nuclei must be aligned")

    if lumen_encoding not in {"stroma", "within_architecture"}:
        raise ValueError("lumen_encoding must be stroma or within_architecture")
    operational_stroma = np.isin(tissue, stroma_fine_ids)
    semantic_architecture = np.isin(tissue, architecture_fine_ids)
    centers, nominal_diameter = _nucleus_centers_and_diameter(nuclei)
    cell_area = max(np.pi * (0.5 * nominal_diameter) ** 2, 1.0)
    window_px = _odd(max(9, int(round(4.0 * nominal_diameter))))
    local_count = ndimage.uniform_filter(
        centers.astype(np.float32), size=window_px, mode="constant"
    ) * float(window_px * window_px)

    (
        luminance,
        optical_density,
        raw_luminance,
        raw_optical_density,
        white_support,
    ) = _appearance_fields(image, tissue)
    reference_domain = (
        operational_stroma
        if np.any(operational_stroma)
        else semantic_architecture
    )
    stroma_density = float(np.count_nonzero(centers & reference_domain)) / max(
        float(np.count_nonzero(reference_domain)) / cell_area, 1.0
    )
    low_count_limit = max(1.35, min(2.25, 0.42 * stroma_density * 16.0))
    low_cell = local_count <= low_count_limit
    if lumen_encoding == "within_architecture":
        nucleus_band = ndimage.binary_dilation(
            centers,
            structure=_disk(max(1, int(round(0.70 * nominal_diameter)))),
        )
        epithelial_support = semantic_architecture & (
            nucleus_band
            | (raw_luminance < 0.52)
            | (raw_optical_density > 0.68)
        )
        architecture = ndimage.binary_closing(
            epithelial_support,
            structure=_disk(max(1, int(round(0.20 * nominal_diameter)))),
        ) & semantic_architecture
        bright_space = (raw_luminance >= 0.72) | (raw_optical_density <= 0.38)
        colored_sparse_space = (
            (raw_luminance >= 0.52)
            & (raw_optical_density <= 0.65)
            & (local_count <= 0.35)
        )
        candidate_domain = (
            semantic_architecture
            & ~architecture
            & (bright_space | colored_sparse_space)
        )
        architecture_units = semantic_architecture
    else:
        architecture = semantic_architecture
        candidate_domain = operational_stroma
        architecture_units = semantic_architecture
    architecture_distance = ndimage.distance_transform_edt(~architecture)
    near_architecture = architecture_distance <= max(6.0, 4.0 * nominal_diameter)

    low_cell_seed = candidate_domain & low_cell & near_architecture & (
        white_support | (local_count <= 0.35)
    )
    seed_radius = max(1, int(round(0.35 * nominal_diameter)))
    low_cell_seed = ndimage.binary_closing(
        low_cell_seed,
        structure=_disk(seed_radius),
    ) & candidate_domain
    low_cell_seed = _remove_small(low_cell_seed, max(6, int(0.18 * cell_area)))

    proposals = _proposal_regions(
        stroma=candidate_domain,
        architecture=architecture,
        low_cell_seed=low_cell_seed,
        local_count=local_count,
        optical_density=optical_density,
        nominal_diameter=nominal_diameter,
    )

    confirmed = np.zeros(tissue.shape, dtype=bool)
    uncertain = np.zeros(tissue.shape, dtype=bool)
    external = operational_stroma.copy()
    region_records: list[LumenRegion] = []
    border = np.zeros(tissue.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    arch_dilated = ndimage.binary_dilation(architecture, structure=STRUCTURE_8)
    architecture_labels, _ = ndimage.label(
        architecture_units, structure=STRUCTURE_8
    )
    architecture_centers = centers & architecture_units
    if np.any(architecture_centers):
        nucleus_distance, nearest_nucleus_indices = ndimage.distance_transform_edt(
            ~architecture_centers,
            return_indices=True,
        )
        center_ids = np.zeros(tissue.shape, dtype=np.int32)
        center_ids[architecture_centers] = np.arange(
            1,
            int(np.count_nonzero(architecture_centers)) + 1,
            dtype=np.int32,
        )
        nearest_nucleus_id = center_ids[
            nearest_nucleus_indices[0], nearest_nucleus_indices[1]
        ]
    else:
        nucleus_distance = np.full(tissue.shape, np.inf, dtype=np.float32)
        nearest_nucleus_id = np.zeros(tissue.shape, dtype=np.int32)
    patch_edge_size = max(int(np.count_nonzero(border)), 1)

    for index, (source_component_id, region) in enumerate(proposals, start=1):
        area = int(np.count_nonzero(region))
        if area == 0:
            continue
        region_boundary = region & ~ndimage.binary_erosion(region, structure=STRUCTURE_8)
        internal_boundary = region_boundary & ~border
        denominator = max(int(np.count_nonzero(internal_boundary)), 1)
        tumor_contact = internal_boundary & arch_dilated
        neighboring_architecture = ndimage.binary_dilation(
            region_boundary, structure=STRUCTURE_8
        ) & architecture
        tumor_contact_px = int(np.count_nonzero(neighboring_architecture))
        contact_fraction = tumor_contact_px / denominator
        boundary_nucleus_support = internal_boundary & (
            nucleus_distance <= 2.75 * nominal_diameter
        )
        boundary_nucleus_support_fraction = float(
            np.count_nonzero(boundary_nucleus_support)
        ) / denominator
        boundary_nucleus_ids = np.unique(
            nearest_nucleus_id[
                internal_boundary
                & (nucleus_distance <= 3.25 * nominal_diameter)
            ]
        )
        boundary_nucleus_count = int(
            np.count_nonzero(boundary_nucleus_ids > 0)
        )
        nucleus_count = int(np.count_nonzero(centers & region))
        density = nucleus_count / max(area / cell_area, 1.0)
        relative_density = density / max(stroma_density, 0.08)
        median_luminance = float(np.median(luminance[region]))
        white_fraction = float(np.mean(white_support[region]))
        median_od = float(np.median(optical_density[region]))
        median_raw_luminance = float(np.median(raw_luminance[region]))
        median_raw_od = float(np.median(raw_optical_density[region]))
        touches_edge = bool(np.any(region & border))
        edge_contact_fraction = float(np.count_nonzero(region & border)) / patch_edge_size
        adjacent_architecture_ids = np.unique(
            architecture_labels[neighboring_architecture]
        )
        adjacent_architecture_ids = adjacent_architecture_ids[
            adjacent_architecture_ids > 0
        ]
        architecture_component_count = int(len(adjacent_architecture_ids))
        if architecture_component_count:
            adjacent_architecture_area = max(
                int(np.count_nonzero(architecture_labels == int(component_id)))
                for component_id in adjacent_architecture_ids
            )
        else:
            adjacent_architecture_area = 0
        region_to_architecture_ratio = area / max(adjacent_architecture_area, 1)
        large_sparse = bool(
            area >= 3.0 * cell_area
            and nucleus_count <= max(2, int(np.ceil(area / (28.0 * cell_area))))
        )
        low_density = bool(
            density <= 0.045
            or relative_density <= 0.25
            or (
                large_sparse
                and density <= 0.065
                and relative_density <= 0.42
            )
        )
        gland_supported = bool(
            tumor_contact_px >= max(4, int(round(0.75 * nominal_diameter)))
            and contact_fraction >= (0.13 if touches_edge else 0.18)
            and boundary_nucleus_count >= (2 if area >= 2.0 * cell_area else 1)
            and boundary_nucleus_support_fraction >= 0.055
        )
        appearance_supported = bool(
            median_raw_luminance >= 0.68
            or median_raw_od <= 0.42
            or (
                white_fraction >= 0.30
                and median_raw_luminance >= 0.50
            )
        )
        not_dark_or_debris = bool(
            median_raw_luminance >= 0.35
            and median_raw_od <= 1.10
        )
        local_gland_geometry = bool(
            architecture_component_count == 1
            and region_to_architecture_ratio <= (5.0 if touches_edge else 3.5)
            and (not touches_edge or edge_contact_fraction <= 0.24)
        )
        strongly_supported = bool(
            low_density
            and gland_supported
            and local_gland_geometry
            and not_dark_or_debris
            and (
                appearance_supported
                or (
                    large_sparse
                    and contact_fraction >= 0.30
                    and median_raw_luminance >= 0.45
                )
            )
        )
        ambiguous_supported = bool(
            low_density
            and not_dark_or_debris
            and tumor_contact_px >= max(3, int(round(0.45 * nominal_diameter)))
            and boundary_nucleus_count >= 1
            and boundary_nucleus_support_fraction >= 0.025
            and (near_architecture & region).any()
            and architecture_component_count <= 2
            and region_to_architecture_ratio <= (8.0 if touches_edge else 5.0)
            and (not touches_edge or edge_contact_fraction <= 0.38)
        )
        if strongly_supported:
            classification = "open_edge_lumen" if touches_edge else "confirmed_lumen"
            confirmed |= region
            external &= ~region
        elif ambiguous_supported:
            classification = "uncertain_low_cell_space"
            uncertain |= region
            external &= ~region
        else:
            classification = "external_stroma"
        region_records.append(
            LumenRegion(
                region_id=f"space:{index:04d}",
                classification=classification,
                area_px=area,
                nucleus_count=nucleus_count,
                nuclei_per_cell_area=float(density),
                relative_nucleus_density=float(relative_density),
                tumor_contact_fraction=float(contact_fraction),
                tumor_contact_px=tumor_contact_px,
                boundary_nucleus_count=boundary_nucleus_count,
                boundary_nucleus_support_fraction=boundary_nucleus_support_fraction,
                median_luminance=median_luminance,
                white_fraction=white_fraction,
                median_optical_density=median_od,
                median_raw_luminance=median_raw_luminance,
                median_raw_optical_density=median_raw_od,
                touches_patch_edge=touches_edge,
                patch_edge_contact_fraction=edge_contact_fraction,
                architecture_component_count=architecture_component_count,
                region_to_architecture_area_ratio=float(region_to_architecture_ratio),
                source_component_id=int(source_component_id),
            )
        )

    return LumenObservation(
        confirmed_lumen=confirmed,
        uncertain_low_cell_space=uncertain,
        external_stroma=external,
        low_cell_seed=low_cell_seed,
        local_nucleus_count=local_count,
        luminance=luminance,
        optical_density=optical_density,
        raw_luminance=raw_luminance,
        raw_optical_density=raw_optical_density,
        nominal_cell_diameter_px=float(nominal_diameter),
        lumen_encoding=lumen_encoding,
        regions=tuple(region_records),
    )


def _nucleus_centers_and_diameter(nuclei: np.ndarray) -> tuple[np.ndarray, float]:
    centers = np.zeros(np.asarray(nuclei).shape, dtype=bool)
    areas: list[int] = []
    for _instance_id, _class_id, component in iter_instances(nuclei):
        rows, cols = np.nonzero(component)
        if not len(rows):
            continue
        cy, cx = ndimage.center_of_mass(component)
        y = int(np.clip(round(float(cy)), 0, centers.shape[0] - 1))
        x = int(np.clip(round(float(cx)), 0, centers.shape[1] - 1))
        centers[y, x] = True
        areas.append(int(len(rows)))
    if areas:
        reference_area = float(np.median(np.asarray(areas, dtype=float)))
        diameter = float(np.sqrt(4.0 * reference_area / np.pi))
    else:
        diameter = 9.0
    return centers, float(np.clip(diameter, 5.0, 24.0))


def _appearance_fields(
    image: np.ndarray, tissue: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = image.astype(np.float32) / 255.0
    luminance_raw = (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    )
    od_channels = -np.log(np.clip((image.astype(np.float32) + 1.0) / 256.0, 1e-4, 1.0))
    od_raw = np.mean(od_channels, axis=2)
    valid = tissue != 0
    if np.any(valid):
        low_l, high_l = np.quantile(luminance_raw[valid], (0.05, 0.95))
        low_od, high_od = np.quantile(od_raw[valid], (0.05, 0.95))
    else:
        low_l, high_l, low_od, high_od = 0.0, 1.0, 0.0, 1.0
    luminance = np.clip((luminance_raw - low_l) / max(high_l - low_l, 1e-6), 0.0, 1.0)
    optical_density = np.clip((od_raw - low_od) / max(high_od - low_od, 1e-6), 0.0, 1.0)
    white_support = (luminance >= 0.62) & (optical_density <= 0.42)
    return luminance, optical_density, luminance_raw, od_raw, white_support


def _proposal_regions(
    *,
    stroma: np.ndarray,
    architecture: np.ndarray,
    low_cell_seed: np.ndarray,
    local_count: np.ndarray,
    optical_density: np.ndarray,
    nominal_diameter: float,
) -> tuple[tuple[int, np.ndarray], ...]:
    border = np.zeros(stroma.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    # Removing the one-pixel patch border closes a gland wall that was cut by
    # the crop.  Its lumen and the exterior then become separate proposals,
    # without assuming that every lumen must be a fully enclosed mask hole.
    labeled, count = ndimage.label(stroma & ~border, structure=STRUCTURE_8)
    proposals: list[tuple[int, np.ndarray]] = []
    minimum_area = max(12, int(round(0.65 * np.pi * (0.5 * nominal_diameter) ** 2)))
    architecture_near = ndimage.binary_dilation(
        architecture,
        structure=_disk(max(2, int(round(1.25 * nominal_diameter)))),
    )
    architecture_distance = ndimage.distance_transform_edt(~architecture)
    for component_id in range(1, int(count) + 1):
        component = labeled == component_id
        component |= border & stroma & ndimage.binary_dilation(
            component, structure=STRUCTURE_8
        )
        if not np.any(component & architecture_near):
            continue
        area = int(np.count_nonzero(component))
        if area < minimum_area:
            continue
        component_edge_fraction = float(np.count_nonzero(component & border)) / max(
            int(np.count_nonzero(border)), 1
        )
        component_seed = component & low_cell_seed
        if not np.any(component_seed):
            # Keep compact tumor-adjacent components so colored, sparse lumina
            # without bright pixels still receive the density/contact audit.
            if area <= int(80.0 * nominal_diameter * nominal_diameter):
                proposals.append((component_id, component))
            continue
        mixed_large = bool(
            component_edge_fraction > 0.20
            or area > int(80.0 * nominal_diameter * nominal_diameter)
        )
        if not mixed_large:
            proposals.append((component_id, component))
            continue
        seed_labels, seed_count = ndimage.label(component_seed, structure=STRUCTURE_8)
        permissive = component & (
            (local_count <= 1.25)
            & (optical_density <= 0.78)
            & (architecture_distance <= 6.0 * nominal_diameter)
        )
        for seed_id in range(1, int(seed_count) + 1):
            seed = seed_labels == seed_id
            if int(np.count_nonzero(seed)) < minimum_area:
                continue
            grown = ndimage.binary_propagation(
                seed,
                structure=STRUCTURE_8,
                mask=permissive,
            )
            grown &= component
            if int(np.count_nonzero(grown)) >= minimum_area:
                proposals.append((component_id, grown))
    # Large-component seeds can grow into the same low-density basin. Dedup by
    # exact raster bytes while preserving deterministic component order.
    unique: list[tuple[int, np.ndarray]] = []
    seen: set[bytes] = set()
    for component_id, region in proposals:
        key = np.packbits(region.astype(np.uint8), axis=None).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append((component_id, region))
    return tuple(unique)


def _remove_small(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    labeled, count = ndimage.label(mask, structure=STRUCTURE_8)
    output = np.zeros(np.asarray(mask).shape, dtype=bool)
    for index in range(1, int(count) + 1):
        component = labeled == index
        if int(np.count_nonzero(component)) >= minimum_area:
            output |= component
    return output


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    rows, cols = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return rows * rows + cols * cols <= radius * radius


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 else value + 1
