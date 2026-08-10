"""Execution-compatible complete-nucleus packing certificates.

This module answers a deliberately narrower question than ProbNet: can the
compiled center region physically hold the required number of complete,
source-matched nucleus footprints while respecting V, retained instances and
the one-pixel separation contract?  ProbNet is invoked only after this
deterministic certificate passes and remains responsible for ranking the legal
landing positions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import ndimage

from .cell_layouts import ReferenceNucleusShape

PACKING_CERTIFIER_VERSION = "complete-footprint-packing-v16"
MAX_PACKING_REFERENCE_SHAPES_PER_CLASS = 3
MINIMUM_LOCAL_MEDIAN_AREA_RATIO = 0.60
MAXIMUM_LOCAL_MEDIAN_AREA_RATIO = 1.67


@dataclass(frozen=True)
class PackingPlacement:
    row: int
    col: int
    class_id: int
    reference_instance_id: str
    area_px: int
    required_seam: bool


@dataclass(frozen=True)
class PackingCertificate:
    version: str
    nominal_requested_count: int
    requested_count: int
    minimum_safe_count: int
    finite_count_fallback_used: bool
    placed_count: int
    passed: bool
    required_seam_count: int
    placed_seam_count: int
    nominal_required_seam_count: int
    minimum_safe_seam_count: int
    seam_count_fallback_used: bool
    capacity_optimized_shape_fallback_used: bool
    center_region_pixels: int
    valid_footprint_pixels: int
    retained_occupied_pixels: int
    class_request_weights: dict[int, float]
    class_requested_counts: dict[int, int]
    class_reference_median_area_px: dict[int, float]
    class_placed_counts: dict[int, int]
    placements: tuple[PackingPlacement, ...]
    footprint_union: np.ndarray
    failure_reasons: tuple[str, ...]

    def to_metadata(self) -> dict:
        return {
            "version": self.version,
            "nominal_requested_count": self.nominal_requested_count,
            "requested_count": self.requested_count,
            "minimum_safe_count": self.minimum_safe_count,
            "finite_count_fallback_used": self.finite_count_fallback_used,
            "placed_count": self.placed_count,
            "passed": self.passed,
            "required_seam_count": self.required_seam_count,
            "placed_seam_count": self.placed_seam_count,
            "nominal_required_seam_count": self.nominal_required_seam_count,
            "minimum_safe_seam_count": self.minimum_safe_seam_count,
            "seam_count_fallback_used": self.seam_count_fallback_used,
            "capacity_optimized_shape_fallback_used": (
                self.capacity_optimized_shape_fallback_used
            ),
            "center_region_pixels": self.center_region_pixels,
            "valid_footprint_pixels": self.valid_footprint_pixels,
            "retained_occupied_pixels": self.retained_occupied_pixels,
            "class_request_weights": {
                str(key): value
                for key, value in sorted(self.class_request_weights.items())
            },
            "class_requested_counts": {
                str(key): value
                for key, value in sorted(self.class_requested_counts.items())
            },
            "class_reference_median_area_px": {
                str(key): value
                for key, value in sorted(
                    self.class_reference_median_area_px.items()
                )
            },
            "local_median_area_ratio_interval": [
                MINIMUM_LOCAL_MEDIAN_AREA_RATIO,
                MAXIMUM_LOCAL_MEDIAN_AREA_RATIO,
            ],
            "class_placed_counts": {
                str(key): value
                for key, value in sorted(self.class_placed_counts.items())
            },
            "placements": [asdict(item) for item in self.placements],
            "footprint_union_pixels": int(
                np.count_nonzero(self.footprint_union)
            ),
            "failure_reasons": list(self.failure_reasons),
        }


def certify_complete_footprint_packing(
    *,
    source_nuclei: np.ndarray,
    erased_footprint: np.ndarray,
    center_region: np.ndarray,
    valid_footprint_region: np.ndarray,
    references_by_class: dict[int, tuple[ReferenceNucleusShape, ...]],
    requested_count: int,
    class_request_weights: dict[int, float] | None = None,
    continuity_region: np.ndarray | None = None,
    required_seam_count: int = 0,
    minimum_seam_count: int | None = None,
    required_seam_class: int | None = None,
    allow_finite_count_fallback: bool = True,
    allow_shape_capacity_fallback: bool = True,
) -> PackingCertificate:
    """Greedily certify real footprint placements under the execution rules.

    The returned count is an achieved, auditable lower bound using actual
    complete source shapes, not an area/median-area estimate.  It is therefore
    safe for fail-closed routing: a failed certificate triggers another tissue
    plan; a passing certificate supplies concrete witness placements but does
    not force ProbNet to reuse their coordinates.
    """

    source = np.asarray(source_nuclei)
    erased = np.asarray(erased_footprint, dtype=bool)
    centers = np.asarray(center_region, dtype=bool)
    valid = np.asarray(valid_footprint_region, dtype=bool)
    if not (source.shape == erased.shape == centers.shape == valid.shape):
        raise ValueError("packing inputs must share one shape")
    # A component touching the raster edge is observationally censored even
    # when its stored footprint happens to fit inside the array.  The mature
    # sampler and the final instance gate use the same one-pixel exclusion.
    valid = valid.copy()
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    seam_required = max(0, int(required_seam_count))
    seam_minimum = (
        seam_required
        if minimum_seam_count is None
        else min(seam_required, max(0, int(minimum_seam_count)))
    )
    # The executor realizes the typed seam stratum first and then the remaining
    # population quota over the full legal center domain. If continuity itself
    # needs more centers than the density-derived population count, the
    # realized total is the seam count; the certificate exposes that same
    # effective integer.
    requested = max(0, int(requested_count), seam_required)
    seam = (
        np.zeros_like(centers, dtype=bool)
        if continuity_region is None
        else np.asarray(continuity_region, dtype=bool) & centers
    )
    complete_references = {
        int(class_id): tuple(items)
        for class_id, items in references_by_class.items()
        if items
    }
    class_reference_median_area_px = {
        class_id: float(np.median([item.area_px for item in items]))
        for class_id, items in complete_references.items()
    }
    normalized_references = {
        class_id: _central_complete_references(
            items,
            reference_median_area_px=class_reference_median_area_px[class_id],
        )
        for class_id, items in complete_references.items()
    }
    normalized_references = {
        class_id: items
        for class_id, items in normalized_references.items()
        if items
    }
    weights = _normalize_weights(
        class_request_weights,
        available_classes=tuple(sorted(normalized_references)),
    )
    class_requested_counts = _allocate_class_counts(
        weights,
        requested,
        minimum_by_class=(
            {int(required_seam_class): seam_required}
            if required_seam_class in normalized_references
            else {}
        ),
    )
    target = source.copy()
    target[erased] = 0
    occupied = target > 0
    footprint_union = np.zeros_like(centers, dtype=bool)
    placements: list[PackingPlacement] = []
    class_counts = {class_id: 0 for class_id in normalized_references}
    reference_offsets = {class_id: 0 for class_id in normalized_references}

    if seam_required:
        seam_class = (
            int(required_seam_class)
            if required_seam_class in normalized_references
            else None
        )
        if seam_class is not None:
            _pack_into_zone(
                requested=seam_required,
                zone=seam,
                valid=valid,
                occupied=occupied,
                footprint_union=footprint_union,
                references_by_class={
                    seam_class: normalized_references[seam_class]
                },
                class_quotas={seam_class: seam_required},
                class_center_regions=None,
                class_counts=class_counts,
                reference_offsets=reference_offsets,
                placements=placements,
                required_seam=True,
            )

    placed_seam = sum(item.required_seam for item in placements)
    remaining_quotas = {
        class_id: max(
            0,
            requested_count - class_counts.get(class_id, 0),
        )
        for class_id, requested_count in class_requested_counts.items()
    }
    remaining = sum(remaining_quotas.values())
    if remaining:
        _pack_into_zone(
            requested=remaining,
            # The seam quota is typed: it reserves exactly the required
            # target-class population, but it is not a biological exclusion
            # zone for other compatible populations.  For example,
            # inflammatory cells may remain interspersed at a newly exposed
            # melanoma--stroma boundary.  Reusing the full P here therefore
            # certifies the capacity the type-aware executor can consume.
            zone=centers,
            valid=valid,
            occupied=occupied,
            footprint_union=footprint_union,
            references_by_class=normalized_references,
            class_quotas=remaining_quotas,
            class_center_regions=(
                {seam_class: centers & ~seam}
                if seam_required and seam_class is not None
                else None
            ),
            class_counts=class_counts,
            reference_offsets=reference_offsets,
            placements=placements,
            required_seam=False,
        )

    reasons = []
    if requested and not normalized_references:
        reasons.append("no_complete_reference_shape_for_packing")
    if seam_required and placed_seam < seam_required:
        reasons.append("exact_seam_packing_capacity_shortfall")
    if len(placements) < requested:
        reasons.append("exact_complete_footprint_packing_capacity_shortfall")
    certificate = PackingCertificate(
        version=PACKING_CERTIFIER_VERSION,
        nominal_requested_count=requested,
        requested_count=requested,
        minimum_safe_count=requested,
        finite_count_fallback_used=False,
        placed_count=len(placements),
        passed=not reasons,
        required_seam_count=seam_required,
        placed_seam_count=int(placed_seam),
        nominal_required_seam_count=seam_required,
        minimum_safe_seam_count=seam_minimum,
        seam_count_fallback_used=False,
        capacity_optimized_shape_fallback_used=False,
        center_region_pixels=int(np.count_nonzero(centers)),
        valid_footprint_pixels=int(np.count_nonzero(valid)),
        retained_occupied_pixels=int(np.count_nonzero(source > 0) - np.count_nonzero((source > 0) & erased)),
        class_request_weights=weights,
        class_requested_counts=class_requested_counts,
        class_reference_median_area_px=class_reference_median_area_px,
        class_placed_counts={
            key: int(value) for key, value in class_counts.items() if value
        },
        placements=tuple(placements),
        footprint_union=footprint_union,
        failure_reasons=tuple(reasons),
    )
    if (
        set(reasons) == {"exact_seam_packing_capacity_shortfall"}
        and seam_minimum <= placed_seam < seam_required
    ):
        safe = certify_complete_footprint_packing(
            source_nuclei=source_nuclei,
            erased_footprint=erased_footprint,
            center_region=center_region,
            valid_footprint_region=valid_footprint_region,
            references_by_class=references_by_class,
            requested_count=requested_count,
            class_request_weights=class_request_weights,
            continuity_region=continuity_region,
            required_seam_count=placed_seam,
            minimum_seam_count=placed_seam,
            required_seam_class=required_seam_class,
            allow_finite_count_fallback=allow_finite_count_fallback,
            allow_shape_capacity_fallback=allow_shape_capacity_fallback,
        )
        if safe.passed:
            return replace(
                safe,
                nominal_required_seam_count=seam_required,
                minimum_safe_seam_count=seam_minimum,
                seam_count_fallback_used=True,
            )
    if (
        allow_shape_capacity_fallback
        and set(reasons)
        == {"exact_complete_footprint_packing_capacity_shortfall"}
    ):
        capacity_references = {
            class_id: (
                min(
                    items,
                    key=lambda item: (item.area_px, item.instance_id),
                ),
            )
            for class_id, items in normalized_references.items()
        }
        if any(
            len(normalized_references[class_id]) > 1
            for class_id in capacity_references
        ):
            safe = certify_complete_footprint_packing(
                source_nuclei=source_nuclei,
                erased_footprint=erased_footprint,
                center_region=center_region,
                valid_footprint_region=valid_footprint_region,
                references_by_class=capacity_references,
                requested_count=requested_count,
                class_request_weights=class_request_weights,
                continuity_region=continuity_region,
                required_seam_count=required_seam_count,
                minimum_seam_count=minimum_seam_count,
                required_seam_class=required_seam_class,
                allow_finite_count_fallback=allow_finite_count_fallback,
                allow_shape_capacity_fallback=False,
            )
            if safe.passed:
                return replace(
                    safe,
                    class_reference_median_area_px=(
                        class_reference_median_area_px
                    ),
                    capacity_optimized_shape_fallback_used=True,
                )
    minimum_safe_count = max(
        seam_required,
        1 if requested else 0,
        int(np.ceil(0.80 * requested)),
        int(np.ceil(requested - np.sqrt(requested))),
    )
    if (
        allow_finite_count_fallback
        and set(reasons)
        == {"exact_complete_footprint_packing_capacity_shortfall"}
        and minimum_safe_count <= certificate.placed_count < requested
    ):
        safe = certify_complete_footprint_packing(
            source_nuclei=source_nuclei,
            erased_footprint=erased_footprint,
            center_region=center_region,
            valid_footprint_region=valid_footprint_region,
            references_by_class=references_by_class,
            requested_count=certificate.placed_count,
            class_request_weights=class_request_weights,
            continuity_region=continuity_region,
            required_seam_count=required_seam_count,
            minimum_seam_count=minimum_seam_count,
            required_seam_class=required_seam_class,
            allow_finite_count_fallback=False,
            allow_shape_capacity_fallback=allow_shape_capacity_fallback,
        )
        if safe.passed:
            return replace(
                safe,
                nominal_requested_count=requested,
                minimum_safe_count=minimum_safe_count,
                finite_count_fallback_used=True,
            )
    return replace(certificate, minimum_safe_count=minimum_safe_count)


def _pack_into_zone(
    *,
    requested: int,
    zone: np.ndarray,
    valid: np.ndarray,
    occupied: np.ndarray,
    footprint_union: np.ndarray,
    references_by_class: dict[int, tuple[ReferenceNucleusShape, ...]],
    class_quotas: dict[int, int],
    class_center_regions: dict[int, np.ndarray] | None,
    class_counts: dict[int, int],
    reference_offsets: dict[int, int],
    placements: list[PackingPlacement],
    required_seam: bool,
) -> None:
    remaining_by_class = {
        int(class_id): max(0, int(count))
        for class_id, count in class_quotas.items()
        if int(count) > 0 and int(class_id) in references_by_class
    }
    if (
        requested <= 0
        or not np.any(zone)
        or not references_by_class
        or not remaining_by_class
    ):
        return
    # Fast path: one vectorized static screen followed by deterministic local
    # collision checks. Successful candidates normally finish here.
    static_fit_maps = _initial_fit_center_maps(
        valid=valid,
        occupied=occupied,
        references_by_class=references_by_class,
    )
    static_fit_by_class = {
        class_id: np.logical_or.reduce(fit_maps)
        for class_id, fit_maps in static_fit_maps.items()
        if fit_maps
    }
    active_static_fit = np.zeros_like(zone, dtype=bool)
    for class_id in remaining_by_class:
        class_fit = static_fit_by_class.get(class_id)
        if class_fit is None:
            continue
        class_region = (class_center_regions or {}).get(class_id)
        active_static_fit |= (
            class_fit
            if class_region is None
            else class_fit & class_region
        )
    coords = _distributed_center_order(zone & active_static_fit)
    placed_here = 0
    consecutive_failures = 0
    for row, col in coords:
        if placed_here >= requested:
            break
        accepted = _place_at_center(
            row=int(row),
            col=int(col),
            zone=zone,
            valid=valid,
            occupied=occupied,
            footprint_union=footprint_union,
            references_by_class=references_by_class,
            reference_fit_maps=static_fit_maps,
            static_fit_by_class=static_fit_by_class,
            class_center_regions=class_center_regions,
            class_counts=class_counts,
            remaining_by_class=remaining_by_class,
            reference_offsets=reference_offsets,
            placements=placements,
            required_seam=required_seam,
        )
        if accepted:
            placed_here += 1
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 512:
                break

    remaining = requested - placed_here
    if remaining > 0 and any(remaining_by_class.values()):
        _pack_dynamic_tail(
            requested=remaining,
            zone=zone,
            valid=valid,
            occupied=occupied,
            footprint_union=footprint_union,
            references_by_class=references_by_class,
            class_center_regions=class_center_regions,
            class_counts=class_counts,
            remaining_by_class=remaining_by_class,
            reference_offsets=reference_offsets,
            placements=placements,
            required_seam=required_seam,
        )


def _pack_dynamic_tail(
    *,
    requested: int,
    zone: np.ndarray,
    valid: np.ndarray,
    occupied: np.ndarray,
    footprint_union: np.ndarray,
    references_by_class: dict[int, tuple[ReferenceNucleusShape, ...]],
    class_center_regions: dict[int, np.ndarray] | None,
    class_counts: dict[int, int],
    remaining_by_class: dict[int, int],
    reference_offsets: dict[int, int],
    placements: list[PackingPlacement],
    required_seam: bool,
) -> None:
    """Finish or disprove a shortfall with current-occupancy fit maps."""

    placed = 0
    while placed < requested and any(remaining_by_class.values()):
        fit_maps = _initial_fit_center_maps(
            valid=valid,
            occupied=occupied,
            references_by_class=references_by_class,
        )
        fit_by_class = {
            class_id: np.logical_or.reduce(items)
            for class_id, items in fit_maps.items()
            if items
        }
        active = np.zeros_like(zone, dtype=bool)
        for class_id, remaining in remaining_by_class.items():
            if remaining <= 0 or class_id not in fit_by_class:
                continue
            class_region = (class_center_regions or {}).get(class_id)
            active |= (
                fit_by_class[class_id]
                if class_region is None
                else fit_by_class[class_id] & class_region
            )
        coords = _distributed_center_order(zone & active)
        accepted = False
        for row, col in coords:
            if _place_at_center(
                row=int(row),
                col=int(col),
                zone=zone,
                valid=valid,
                occupied=occupied,
                footprint_union=footprint_union,
                references_by_class=references_by_class,
                reference_fit_maps=fit_maps,
                static_fit_by_class=fit_by_class,
                class_center_regions=class_center_regions,
                class_counts=class_counts,
                remaining_by_class=remaining_by_class,
                reference_offsets=reference_offsets,
                placements=placements,
                required_seam=required_seam,
            ):
                placed += 1
                accepted = True
                break
        if not accepted:
            break


def _place_at_center(
    *,
    row: int,
    col: int,
    zone: np.ndarray,
    valid: np.ndarray,
    occupied: np.ndarray,
    footprint_union: np.ndarray,
    references_by_class: dict[int, tuple[ReferenceNucleusShape, ...]],
    reference_fit_maps: dict[int, tuple[np.ndarray, ...]],
    static_fit_by_class: dict[int, np.ndarray],
    class_center_regions: dict[int, np.ndarray] | None,
    class_counts: dict[int, int],
    remaining_by_class: dict[int, int],
    reference_offsets: dict[int, int],
    placements: list[PackingPlacement],
    required_seam: bool,
) -> bool:
    alternatives = tuple(
        sorted(
            remaining_by_class,
            key=lambda item: (-remaining_by_class[item], item),
        )
    )
    for class_id in alternatives:
        if remaining_by_class.get(class_id, 0) <= 0:
            continue
        class_region = (class_center_regions or {}).get(class_id)
        if class_region is not None and not bool(class_region[row, col]):
            continue
        if not bool(static_fit_by_class[class_id][row, col]):
            continue
        references = references_by_class[class_id]
        fit_maps = reference_fit_maps[class_id]
        start = reference_offsets[class_id] % len(references)
        for offset in range(len(references)):
            reference_index = (start + offset) % len(references)
            if not bool(fit_maps[reference_index][row, col]):
                continue
            reference = references[reference_index]
            window = _placement_window(
                reference.mask,
                center_y=row,
                center_x=col,
                canvas_shape=zone.shape,
            )
            if window is None:
                continue
            y0, y1, x0, x1 = window
            shape = np.asarray(reference.mask, dtype=bool)
            gy0, gy1 = max(0, y0 - 1), min(zone.shape[0], y1 + 1)
            gx0, gx1 = max(0, x0 - 1), min(zone.shape[1], x1 + 1)
            local = np.zeros((gy1 - gy0, gx1 - gx0), dtype=bool)
            local[y0 - gy0 : y1 - gy0, x0 - gx0 : x1 - gx0] = shape
            guard = ndimage.binary_dilation(
                local,
                structure=np.ones((3, 3), dtype=bool),
                iterations=1,
            )
            if np.any(guard & occupied[gy0:gy1, gx0:gx1]):
                continue
            occupied[y0:y1, x0:x1] |= shape
            footprint_union[y0:y1, x0:x1] |= shape
            reference_offsets[class_id] = start + offset + 1
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            remaining_by_class[class_id] -= 1
            placements.append(
                PackingPlacement(
                    row=row,
                    col=col,
                    class_id=class_id,
                    reference_instance_id=reference.instance_id,
                    area_px=int(np.count_nonzero(shape)),
                    required_seam=required_seam,
                )
            )
            return True
    return False


def _initial_fit_center_maps(
    *,
    valid: np.ndarray,
    occupied: np.ndarray,
    references_by_class: dict[int, tuple[ReferenceNucleusShape, ...]],
) -> dict[int, tuple[np.ndarray, ...]]:
    """Exact static-fit maps for every class and complete footprint.

    SciPy anchors the structuring element at ``shape // 2``, matching
    ``_placement_window``.  Eroding the free/valid canvas by each actual
    footprint therefore returns legal centers for that exact footprint.
    Keeping the maps separate prevents a capacity-short class from scanning
    centers that are legal only for a different, already-satisfied class.
    Dynamic collisions between newly accepted witnesses remain checked in the
    local placement loop.
    """

    free = np.asarray(valid, dtype=bool) & ~ndimage.binary_dilation(
        np.asarray(occupied, dtype=bool),
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    result = {}
    for class_id, references in references_by_class.items():
        fit_maps = []
        for reference in references:
            structure = np.asarray(reference.mask, dtype=bool)
            if not np.any(structure):
                fit_maps.append(np.zeros_like(free, dtype=bool))
            else:
                fit_maps.append(_binary_erosion_for_footprint(free, structure))
        result[int(class_id)] = tuple(fit_maps)
    return result


def _binary_erosion_for_footprint(
    free: np.ndarray,
    structure: np.ndarray,
) -> np.ndarray:
    """Return SciPy-identical footprint centers with an optional fast backend.

    Exact packing repeatedly erodes a 512x512 legal canvas by real nucleus
    footprints.  OpenCV implements the same centered binary erosion much more
    efficiently on the server; the local research environment does not always
    ship OpenCV, so SciPy remains a value-identical fallback.  The explicit
    anchor is essential for even-sized nucleus bounding boxes.
    """

    canvas = np.asarray(free, dtype=bool)
    kernel = np.asarray(structure, dtype=bool)
    try:
        import cv2
    except ImportError:
        return ndimage.binary_erosion(
            canvas,
            structure=kernel,
            border_value=0,
        )
    return (
        cv2.erode(
            canvas.astype(np.uint8),
            kernel.astype(np.uint8),
            anchor=(kernel.shape[1] // 2, kernel.shape[0] // 2),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )


def _distributed_center_order(zone: np.ndarray) -> np.ndarray:
    coords = np.argwhere(zone)
    if not len(coords):
        return coords
    distance = ndimage.distance_transform_edt(zone)
    # Interleave four raster phases after sorting by distance to the zone edge.
    # This avoids consuming one corner first while remaining deterministic.
    phase = (coords[:, 0] % 2) * 2 + (coords[:, 1] % 2)
    order = np.lexsort((coords[:, 1], coords[:, 0], phase, -distance[zone]))
    return coords[order]


def _central_complete_references(
    references: Iterable[ReferenceNucleusShape],
    *,
    reference_median_area_px: float,
) -> tuple[ReferenceNucleusShape, ...]:
    values = tuple(references)
    if not values:
        return ()
    lower = MINIMUM_LOCAL_MEDIAN_AREA_RATIO * float(
        reference_median_area_px
    )
    upper = MAXIMUM_LOCAL_MEDIAN_AREA_RATIO * float(
        reference_median_area_px
    )
    values = tuple(
        item for item in values if lower <= float(item.area_px) <= upper
    )
    if not values:
        return ()
    ordered = sorted(values, key=lambda item: (item.area_px, item.instance_id))
    if len(ordered) <= 4:
        return tuple(ordered)
    low = int(np.floor(0.20 * (len(ordered) - 1)))
    high = int(np.ceil(0.80 * (len(ordered) - 1))) + 1
    central = ordered[low:high]
    if len(central) <= MAX_PACKING_REFERENCE_SHAPES_PER_CLASS:
        return tuple(central)
    # Capacity certification needs a representative, auditable set of real
    # complete footprints, not every source instance.  Trying hundreds of
    # near-duplicate masks at every center made the pre-ProbNet compiler
    # super-linear on dense patches.  Evenly sampled central quantiles retain
    # size/shape diversity while giving a fixed execution bound.
    indices = np.linspace(
        0,
        len(central) - 1,
        num=MAX_PACKING_REFERENCE_SHAPES_PER_CLASS,
        dtype=int,
    )
    return tuple(central[int(index)] for index in np.unique(indices))


def _normalize_weights(
    weights: dict[int, float] | None,
    *,
    available_classes: tuple[int, ...],
) -> dict[int, float]:
    values = {
        int(key): max(0.0, float(value))
        for key, value in (weights or {}).items()
        if int(key) in available_classes and float(value) > 0
    }
    if not values:
        values = {item: 1.0 for item in available_classes}
    total = sum(values.values())
    return {key: value / total for key, value in sorted(values.items())}


def _allocate_class_counts(
    weights: dict[int, float],
    requested_count: int,
    *,
    minimum_by_class: dict[int, int] | None = None,
) -> dict[int, int]:
    """Compile continuous population weights into one exact class ledger.

    Largest-remainder rounding preserves the requested total.  A typed seam
    quota is then satisfied by transferring quota from classes above their
    own minima; it never silently increases total cellularity.
    """

    requested = max(0, int(requested_count))
    if requested <= 0 or not weights:
        return {}
    raw = {key: float(value) * requested for key, value in weights.items()}
    counts = {key: int(np.floor(value)) for key, value in raw.items()}
    for key in sorted(raw, key=lambda item: (-(raw[item] - counts[item]), item))[
        : requested - sum(counts.values())
    ]:
        counts[key] += 1
    minima = {
        int(key): max(0, int(value))
        for key, value in (minimum_by_class or {}).items()
        if int(key) in counts
    }
    for key, minimum in sorted(minima.items()):
        deficit = max(0, minimum - counts[key])
        while deficit:
            donors = [
                donor
                for donor in counts
                if donor != key
                and counts[donor] > minima.get(donor, 0)
            ]
            if not donors:
                break
            donor = max(
                donors,
                key=lambda item: (counts[item] - minima.get(item, 0), -item),
            )
            counts[donor] -= 1
            counts[key] += 1
            deficit -= 1
    return {key: value for key, value in sorted(counts.items()) if value > 0}


def _weighted_class_order(weights: dict[int, float], length: int) -> tuple[int, ...]:
    if not weights:
        return ()
    counts = {
        key: max(1, round(value * max(length, len(weights))))
        for key, value in weights.items()
    }
    result = []
    remaining = dict(counts)
    while any(remaining.values()):
        for key in sorted(remaining, key=lambda item: (-weights[item], item)):
            if remaining[key] > 0:
                result.append(key)
                remaining[key] -= 1
    return tuple(result)


def _placement_window(
    shape: np.ndarray,
    *,
    center_y: int,
    center_x: int,
    canvas_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    height, width = np.asarray(shape).shape
    y0 = int(center_y) - height // 2
    x0 = int(center_x) - width // 2
    y1, x1 = y0 + height, x0 + width
    if y0 < 0 or x0 < 0 or y1 > canvas_shape[0] or x1 > canvas_shape[1]:
        return None
    return y0, y1, x0, x1
