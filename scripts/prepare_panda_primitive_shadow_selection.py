#!/usr/bin/env python3
"""Build a mask-only PANDA candidate pool from the frozen cross-meta eval.

The complete ``metadata_cross_val.json`` is the source authority.  The selector
deduplicates its PANDA target patches and reads only the already-prepared tissue
and nuclei masks.  H&E paths are bound as later execution inputs but H&E pixels
are never read here.  Primitive-specific hard requirements are evaluated before
native-instance materialization and live compiler replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.cell_programs import (
    _depletion_band_edges,
    _enforce_density_field_gradient_quotas,
)
from phase3_joint_edit_refine.nuclei import load_nuclei_mask

SCHEMA_VERSION = "panda-cross-meta-eval-primitive-candidate-pool-v1"
PRODUCER_ID = "cross-meta-eval-mask-only-panda-primitive-selector-v5"
TUMOR_FINE_IDS = (8, 9, 10)
STRUCTURE_8 = np.ones((3, 3), dtype=bool)
SAMPLE_OFFSET = re.compile(r"_py(?P<y>\d+)_px(?P<x>\d+)$")
PROSTATE_IMMUNE_DEPLETION_MINIMUM_COUNT = 6


@dataclass(frozen=True)
class EvaluationSpec:
    mechanism_id: str
    primitive_id: str
    instruction: str

    @property
    def evaluation_id(self) -> str:
        return (
            "panda-gleason-v1::"
            f"{self.mechanism_id}::{self.primitive_id}"
        )


EVALUATIONS = (
    EvaluationSpec(
        "prostate-local-population-modulation",
        "cell-type-abundance-increase-v1",
        "Increase connective tissue cells in the selected region.",
    ),
    EvaluationSpec(
        "prostate-local-population-modulation",
        "cell-type-abundance-decrease-v1",
        "Decrease connective tissue cells in the selected region.",
    ),
    EvaluationSpec(
        "prostate-local-population-modulation",
        "cellularity-increase-v1",
        "Increase cellularity in the selected region.",
    ),
    EvaluationSpec(
        "prostate-local-population-modulation",
        "cellularity-decrease-v1",
        "Decrease cellularity in the selected region.",
    ),
    EvaluationSpec(
        "prostate-local-population-modulation",
        "neoplastic-cell-abundance-increase-v1",
        "Increase neoplastic cell abundance in the selected tumor region.",
    ),
    EvaluationSpec(
        "prostate-local-population-modulation",
        "neoplastic-cell-abundance-decrease-v1",
        "Decrease neoplastic cell abundance in the selected tumor region.",
    ),
    EvaluationSpec(
        "prostate-local-tumor-clearance",
        "local-invasive-clearance-v1",
        "Clear tumor in this explicitly selected local ROI.",
    ),
    EvaluationSpec(
        "prostate-operational-tumor-retreat",
        "stroma-increase-v1",
        "After treatment, increase operational stroma as annotated tumor retreats.",
    ),
    EvaluationSpec(
        "prostate-operational-tumor-retreat",
        "invasive-tumor-footprint-decrease-v1",
        "After treatment, decrease the annotated invasive tumor footprint.",
    ),
    EvaluationSpec(
        "prostate-operational-tumor-retreat",
        "residual-tumor-fragmentation-v1",
        "After treatment, fragment the annotated residual tumor footprint.",
    ),
    EvaluationSpec(
        "prostate-pattern-4-growth",
        "cohesive-boundary-expansion-v1",
        "Expand an annotation-defined Pattern-4 boundary into adjacent stroma.",
    ),
    EvaluationSpec(
        "prostate-pattern-5-growth",
        "cohesive-boundary-expansion-v1",
        "Expand an annotation-defined Pattern-5 boundary into adjacent stroma.",
    ),
    EvaluationSpec(
        "prostate-pattern-5-infiltrative-front",
        "infiltrative-nest-cord-extension-v1",
        "Extend one annotation-defined Pattern-5 boundary as a synthetic narrow cord.",
    ),
    EvaluationSpec(
        "prostate-pattern-5-peripheral-scatter",
        "peritumoral-neoplastic-scatter-increase-v1",
        "Add synthetic sparse class-1 nuclei outside an annotated Pattern-5 boundary.",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_evaluation_count_overrides(value: str | None) -> dict[int, int]:
    overrides: dict[int, int] = {}
    for token in (value or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            index_text, count_text = token.split(":", 1)
            index, count = int(index_text), int(count_text)
        except ValueError as exc:
            raise ValueError(
                "evaluation count overrides must use INDEX:COUNT entries"
            ) from exc
        if index < 0 or count < 5:
            raise ValueError(
                "evaluation count overrides require nonnegative indices and "
                "at least five candidates"
            )
        if index in overrides:
            raise ValueError(f"duplicate evaluation count override: {index}")
        overrides[index] = count
    return overrides


def _slide_id(stem: str) -> str:
    return stem.split("_y", 1)[0]


def _mask(path: Path) -> np.ndarray:
    value = np.asarray(Image.open(path))
    if value.ndim != 2:
        raise ValueError(f"expected a 2-D mask: {path}")
    return value


def _contact_pixels(mask: np.ndarray, left: int, right: int) -> int:
    left_region = mask == left
    right_region = mask == right
    if not np.any(left_region) or not np.any(right_region):
        return 0
    contact = left_region & ndimage.binary_dilation(
        right_region, structure=STRUCTURE_8
    )
    return int(np.count_nonzero(contact))


def _component_metrics(region: np.ndarray) -> tuple[int, int, int]:
    labeled, count = ndimage.label(region, structure=STRUCTURE_8)
    if not count:
        return 0, 0, 0
    areas = np.bincount(labeled.ravel())[1:]
    return int(count), int(areas.max(initial=0)), int(np.count_nonzero(areas >= 256))


def _tissue_metrics(path: Path) -> dict[str, Any]:
    tissue = _mask(path)
    if tissue.shape != (512, 512):
        raise ValueError(f"PANDA shadow requires 512x512 patches: {path} {tissue.shape}")
    pixels = Counter(int(value) for value in tissue.ravel())
    p4_components = _component_metrics(tissue == 9)
    p5_components = _component_metrics(tissue == 10)
    p4_contact = _contact_pixels(tissue, 9, 2)
    p5_contact = _contact_pixels(tissue, 10, 2)
    fine10_distance = ndimage.distance_transform_edt(tissue != 10)
    p5_outer_annulus = (tissue == 2) & (fine10_distance >= 4) & (fine10_distance <= 48)
    stroma_components, _ = ndimage.label(
        tissue == 2, structure=STRUCTURE_8
    )
    adjacent_stroma_ids = np.unique(
        stroma_components[
            (tissue == 2)
            & ndimage.binary_dilation(tissue == 10, structure=STRUCTURE_8)
        ]
    )
    adjacent_stroma_ids = adjacent_stroma_ids[adjacent_stroma_ids > 0]
    stroma_component_areas = np.bincount(stroma_components.ravel())
    largest_p5_adjacent_stroma = max(
        (int(stroma_component_areas[value]) for value in adjacent_stroma_ids),
        default=0,
    )
    tumor = np.isin(tissue, TUMOR_FINE_IDS)
    tumor_components = _component_metrics(tumor)
    height, width = tissue.shape
    y0, y1 = height // 8, height - height // 8
    x0, x1 = width // 8, width - width // 8
    central = np.zeros(tissue.shape, dtype=bool)
    central[y0:y1, x0:x1] = True
    editable_tumor = np.isin(tissue, (9, 10))
    central_editable = central & editable_tumor
    central_stroma = central & (tissue == 2)
    central_contact = central_editable & ndimage.binary_dilation(
        central_stroma, structure=STRUCTURE_8
    )
    return {
        "filename": path.name,
        "slide_id": _slide_id(path.stem),
        "shape": [512, 512],
        "fine_pixel_counts": {
            str(value): int(pixels.get(value, 0)) for value in (0, 2, 5, 8, 9, 10)
        },
        "p4_stroma_contact_pixels": p4_contact,
        "p5_stroma_contact_pixels": p5_contact,
        "p5_outer_annulus_pixels": int(np.count_nonzero(p5_outer_annulus)),
        "p5_adjacent_stroma_component_count": int(len(adjacent_stroma_ids)),
        "p5_largest_adjacent_stroma_component_pixels": (
            largest_p5_adjacent_stroma
        ),
        "p4_component_count": p4_components[0],
        "p4_largest_component_pixels": p4_components[1],
        "p5_component_count": p5_components[0],
        "p5_largest_component_pixels": p5_components[1],
        "tumor_component_count": tumor_components[0],
        "tumor_largest_component_pixels": tumor_components[1],
        "tumor_pixels": int(np.count_nonzero(tumor)),
        "stroma_pixels": int(pixels.get(2, 0)),
        "fixed_central_roi": {
            "pixel_bounds_xyxy": [x0, y0, x1, y1],
            "editable_pattern_9_or_10_pixels": int(
                np.count_nonzero(central_editable)
            ),
            "stroma_pixels": int(np.count_nonzero(central_stroma)),
            "tumor_stroma_contact_pixels": int(
                np.count_nonzero(central_contact)
            ),
        },
    }


def _coarse_group_scores(metrics: dict[str, Any]) -> dict[str, float]:
    fine = metrics["fine_pixel_counts"]
    p4_area = int(fine["9"])
    p5_area = int(fine["10"])
    stroma = int(metrics["stroma_pixels"])
    tumor = int(metrics["tumor_pixels"])
    p4_edge = int(metrics["p4_stroma_contact_pixels"])
    p5_edge = int(metrics["p5_stroma_contact_pixels"])
    annulus = int(metrics["p5_outer_annulus_pixels"])
    central = metrics["fixed_central_roi"]
    return {
        "local": float(tumor + 0.35 * stroma),
        "p4": float(min(p4_area, stroma) + 24 * p4_edge),
        "p5": float(min(p5_area, stroma) + 24 * p5_edge),
        "p5_cord": float(
            2.0 * annulus
            + min(
                int(metrics["p5_largest_adjacent_stroma_component_pixels"]),
                80000,
            )
            + 4 * p5_edge
        ),
        "p5_scatter": float(annulus + 12 * p5_edge + 0.25 * p5_area),
        "retreat": float(min(tumor, stroma) + 20 * max(p4_edge, p5_edge)),
        "fragmentation": float(
            200000
            - abs(int(metrics["tumor_largest_component_pixels"]) - 70000)
            + 16 * max(p4_edge, p5_edge)
        ),
        "clearance": float(
            min(
                int(central["editable_pattern_9_or_10_pixels"]),
                int(central["stroma_pixels"]),
            )
            + 30 * int(central["tumor_stroma_contact_pixels"])
        ),
    }


def _coarse_eligible(group: str, metrics: dict[str, Any]) -> bool:
    fine = metrics["fine_pixel_counts"]
    stroma = int(metrics["stroma_pixels"])
    if group == "local":
        # Local population editing is certified against one exact host
        # component and one exact interface.  A patch-wide 30k/30k
        # Tumor/Stroma requirement excludes small but fully executable local
        # fields and has no primitive-level biological meaning.
        return metrics["tumor_pixels"] > 0 and stroma > 0
    if group == "clearance":
        central = metrics["fixed_central_roi"]
        return (
            int(central["editable_pattern_9_or_10_pixels"]) >= 24000
            and int(central["stroma_pixels"]) >= 24000
            and int(central["tumor_stroma_contact_pixels"]) >= 256
        )
    if group == "p4":
        return (
            int(fine["9"]) >= 50000
            and stroma >= 70000
            and metrics["p4_stroma_contact_pixels"] >= 512
        )
    if group in {"p5", "p5_cord"}:
        return (
            int(fine["10"]) >= 40000
            and stroma >= 70000
            and metrics["p5_stroma_contact_pixels"] >= 512
        )
    if group == "p5_scatter":
        return (
            int(fine["10"]) >= 25000
            and metrics["p5_stroma_contact_pixels"] >= 256
            and metrics["p5_outer_annulus_pixels"] >= 12000
        )
    if group == "retreat":
        return (
            int(fine["10"]) >= 50000
            and stroma >= 50000
            and metrics["p5_stroma_contact_pixels"] >= 512
        )
    if group == "fragmentation":
        # PANDA owns a 3% selected-component visibility floor.  Even the
        # largest possible 512x512 source component can therefore satisfy it
        # inside the frozen 3--5% patch budget.  The former 109226-pixel cap
        # was inherited from the generic 12% floor and wrongly discarded
        # large Pattern-5 fields that contain suitable narrow tumor bridges.
        largest = int(metrics["tumor_largest_component_pixels"])
        return (
            int(fine["10"]) >= 50000
            and stroma >= 50000
            and metrics["p5_stroma_contact_pixels"] >= 512
            and largest >= 20000
        )
    raise ValueError(group)


def _keep_slide_best(
    best_by_slide: dict[
        str, dict[str, list[tuple[float, str, dict[str, Any]]]]
    ],
    *,
    group: str,
    score: float,
    metrics: dict[str, Any],
    maximum_per_slide: int,
) -> None:
    item = (float(score), str(metrics["filename"]), metrics)
    slide_id = str(metrics["slide_id"])
    current = best_by_slide[group].setdefault(slide_id, [])
    current.append(item)
    current.sort(key=lambda value: (-value[0], value[1]))
    del current[maximum_per_slide:]


def _instance_metrics(
    *, tissue_path: Path, nuclei_path: Path
) -> dict[str, Any]:
    """Return conservative same-class component counts for pool screening.

    This stage deliberately avoids watershed pseudo-instances.  Exact identity
    is supplied later by frozen CellViT JSON and checked by the live compiler.
    A touching same-class group therefore counts as one conservative component
    here, which is sufficient for shortlist ranking without pretending that a
    semantic raster is native instance authority.
    """

    tissue = _mask(tissue_path)
    nuclei = load_nuclei_mask(nuclei_path)
    by_class = Counter()
    tumor_by_class = Counter()
    stroma_by_class = Counter()
    complete_by_class = Counter()
    complete_tumor_by_class = Counter()
    complete_stroma_by_class = Counter()
    complete_records: list[tuple[int, int, int, int, int]] = []
    class2_components = np.zeros(tissue.shape, dtype=np.int32)
    total = 0
    complete = 0
    for class_id in range(1, 6):
        labeled, count = ndimage.label(
            nuclei == class_id, structure=STRUCTURE_8
        )
        if not count:
            continue
        centers = ndimage.center_of_mass(
            np.ones(nuclei.shape, dtype=np.uint8),
            labels=labeled,
            index=range(1, count + 1),
        )
        objects = ndimage.find_objects(labeled)
        component_areas = np.bincount(labeled.ravel())
        if class_id == 2:
            class2_components = labeled
        for component_id, (center, bounds) in enumerate(
            zip(centers, objects, strict=True), start=1
        ):
            if bounds is None:
                continue
            total += 1
            by_class[class_id] += 1
            row_slice, col_slice = bounds
            touches_border = bool(
                row_slice.start == 0
                or row_slice.stop == nuclei.shape[0]
                or col_slice.start == 0
                or col_slice.stop == nuclei.shape[1]
            )
            row = int(np.clip(round(float(center[0])), 0, tissue.shape[0] - 1))
            col = int(np.clip(round(float(center[1])), 0, tissue.shape[1] - 1))
            if not touches_border:
                complete += 1
                complete_by_class[class_id] += 1
                complete_records.append(
                    (
                        class_id,
                        row,
                        col,
                        int(component_areas[component_id]),
                        component_id,
                    )
                )
            fine_id = int(tissue[row, col])
            if fine_id in TUMOR_FINE_IDS:
                tumor_by_class[class_id] += 1
                if not touches_border:
                    complete_tumor_by_class[class_id] += 1
            if fine_id == 2:
                stroma_by_class[class_id] += 1
                if not touches_border:
                    complete_stroma_by_class[class_id] += 1
    empty_clearance = ndimage.distance_transform_edt(nuclei == 0)

    def packing_centers(region: np.ndarray) -> int:
        # Conservative nucleus-center proxy: keep four pixels inside the host
        # label and six pixels away from every existing semantic nucleus.
        interior = ndimage.binary_erosion(
            np.asarray(region, dtype=bool),
            structure=STRUCTURE_8,
            iterations=4,
            border_value=0,
        )
        return int(np.count_nonzero(interior & (empty_clearance >= 6.0)))

    tumor_region = np.isin(tissue, TUMOR_FINE_IDS)
    stroma_region = tissue == 2
    fine10_distance = ndimage.distance_transform_edt(tissue != 10)
    p5_outer_annulus = (
        stroma_region & (fine10_distance >= 4) & (fine10_distance <= 48)
    )

    # A depletion program is not authorized by a patch-wide class count.  It
    # needs one Tumor/Normal-epithelium component touching Stroma and a local
    # interface-inward core, transition, and unchanged outer reference.  This
    # semantic-mask proxy deliberately underclaims authority: frozen native
    # instances and the production compiler remain the exact decision maker.
    immune_instances = [
        (row, col, area, component_id)
        for class_id, row, col, area, component_id in complete_records
        if class_id == 2
    ]
    immune_diameter_px = (
        max(
            3.0,
            2.0
            * math.sqrt(
                float(np.median([item[2] for item in immune_instances]))
                / math.pi
            ),
        )
        if immune_instances
        else 3.0
    )
    immune_fields = []
    for host_kind, host_region, neighbor_region in (
        ("tumor", tumor_region, stroma_region),
        ("normal_epithelium", tissue == 5, stroma_region),
        # Prostate immune abundance is commonly represented in the explicit
        # stromal population adjacent to annotated Tumor.  The mechanism
        # contract permits that Tumor/Stroma anchor; the edit remains inside
        # Stroma and preserves every tissue/fine label.
        ("stroma", stroma_region, tumor_region),
    ):
        host_components, host_count = ndimage.label(
            host_region, structure=STRUCTURE_8
        )
        neighbor_components, _ = ndimage.label(
            neighbor_region, structure=STRUCTURE_8
        )
        for component_index in range(1, host_count + 1):
            component = host_components == component_index
            touching_neighbor_ids = np.unique(
                neighbor_components[
                    ndimage.binary_dilation(
                        component, structure=STRUCTURE_8
                    )
                    & neighbor_region
                ]
            )
            for neighbor_component_index in touching_neighbor_ids[
                touching_neighbor_ids > 0
            ]:
                neighbor_component = (
                    neighbor_components == neighbor_component_index
                )
                anchor = component & ndimage.binary_dilation(
                    neighbor_component, structure=STRUCTURE_8
                )
                if not np.any(anchor):
                    continue
                distance = ndimage.distance_transform_edt(~anchor)
                maximum_observed_distance = float(
                    np.max(distance[component], initial=0.0)
                )
                minimum_effect_span = int(
                    np.floor(6.0 * immune_diameter_px)
                )
                patch_support_limit = max(
                    48, int(np.floor(0.40 * min(tissue.shape)))
                )
                maximum_extent = min(
                    patch_support_limit,
                    max(
                        minimum_effect_span,
                        int(
                            np.clip(
                                round(6.0 * immune_diameter_px), 48, 96
                            )
                        ),
                    ),
                )
                try:
                    core_end, transition_end, outer_end = (
                        _depletion_band_edges(
                            diameter_px=immune_diameter_px,
                            core_width_cell_diameters=1.25,
                            transition_width_cell_diameters=3.0,
                            outer_width_cell_diameters=1.5,
                            maximum_extent_px=maximum_extent,
                            maximum_observed_distance_px=(
                                maximum_observed_distance
                            ),
                        )
                    )
                except Exception:
                    continue
                transition_width = max(1.0, transition_end - core_end)
                radial_counts = [0, 0, 0, 0, 0]
                outer_count = 0
                for row, col, _area, instance_id in immune_instances:
                    if not component[row, col]:
                        continue
                    footprint = class2_components == instance_id
                    if np.any(footprint & ~component):
                        continue
                    radial = float(distance[row, col])
                    if radial <= core_end:
                        radial_counts[0] += 1
                    elif radial <= transition_end:
                        fraction = (radial - core_end) / transition_width
                        subband = min(3, max(0, int(fraction * 4.0)))
                        radial_counts[1 + subband] += 1
                    elif radial <= outer_end:
                        outer_count += 1
                maximum_removals = [
                    max(
                        0,
                        radial_counts[0]
                        - math.ceil(0.4 * radial_counts[0]),
                    ),
                    *[
                        max(0, count - math.ceil(0.5 * count))
                        for count in radial_counts[1:]
                    ],
                ]
                target_fractions = [0.55, 0.42, 0.313333, 0.206667, 0.10]
                quotas = [
                    min(
                        capacity,
                        int(np.floor(count * target + 0.5)),
                    )
                    for count, capacity, target in zip(
                        radial_counts,
                        maximum_removals,
                        target_fractions,
                        strict=True,
                    )
                ]
                if maximum_removals[0] > 0:
                    quotas[0] = max(1, quotas[0])
                if sum(quotas[1:]) < 1:
                    for index in range(1, len(quotas)):
                        if maximum_removals[index] > 0:
                            quotas[index] = 1
                            break
                maximum_count = min(32, sum(maximum_removals))
                resolved_quotas = None
                if maximum_count >= PROSTATE_IMMUNE_DEPLETION_MINIMUM_COUNT:
                    try:
                        candidate_quotas = (
                            _enforce_density_field_gradient_quotas(
                                quotas=quotas,
                                source_counts=radial_counts,
                                maximum_removals=maximum_removals,
                                target_fractions=target_fractions,
                                minimum_count=(
                                    PROSTATE_IMMUNE_DEPLETION_MINIMUM_COUNT
                                ),
                                maximum_count=maximum_count,
                                minimum_core=1,
                                minimum_transition=1,
                            )
                        )
                    except Exception:
                        pass
                    else:
                        mismatched = False
                        for count, quota, target in zip(
                            radial_counts,
                            candidate_quotas,
                            target_fractions,
                            strict=True,
                        ):
                            if count <= 0:
                                continue
                            tolerance = max(0.12, 1.0 / count)
                            if abs(quota / count - target) > tolerance:
                                mismatched = True
                                break
                        if not mismatched:
                            resolved_quotas = candidate_quotas
                field = component & (distance <= outer_end)
                field_area = float(np.count_nonzero(field)) / max(
                    1.0, immune_diameter_px**2
                )
                gradient_feasible = bool(
                    resolved_quotas is not None
                    and sum(resolved_quotas)
                    >= PROSTATE_IMMUNE_DEPLETION_MINIMUM_COUNT
                    and outer_count >= 3
                    and field_area >= 57.0
                )
                immune_fields.append(
                    {
                        "host_kind": host_kind,
                        "component_index": component_index,
                        "neighbor_component_index": int(
                            neighbor_component_index
                        ),
                        "estimated_immune_diameter_px": round(
                            immune_diameter_px, 6
                        ),
                        "core_complete_class2": radial_counts[0],
                        "transition_complete_class2": sum(
                            radial_counts[1:]
                        ),
                        "transition_radial_complete_class2": list(
                            radial_counts[1:]
                        ),
                        "outer_reference_complete_class2": outer_count,
                        "residual_safe_removal_capacity": sum(
                            maximum_removals
                        ),
                        "density_gradient_resolved_removals": (
                            sum(resolved_quotas)
                            if resolved_quotas is not None
                            else 0
                        ),
                        "density_gradient_feasible": gradient_feasible,
                        "field_area_cell_diameter_squares": round(
                            field_area, 6
                        ),
                        "three_band_complete_class2": (
                            sum(radial_counts) + outer_count
                        ),
                        "interface_anchor_pixels": int(
                            np.count_nonzero(anchor)
                        ),
                    }
                )
    best_immune_field = max(
        immune_fields,
        key=lambda item: (
            bool(item["density_gradient_feasible"]),
            int(item["density_gradient_resolved_removals"]),
            min(
                int(item["residual_safe_removal_capacity"]),
                PROSTATE_IMMUNE_DEPLETION_MINIMUM_COUNT,
            ),
            min(int(item["outer_reference_complete_class2"]), 3),
            min(float(item["field_area_cell_diameter_squares"]), 57.0),
            int(item["three_band_complete_class2"]),
            int(item["interface_anchor_pixels"]),
            str(item["host_kind"]),
            -int(item["component_index"]),
        ),
        default={
            "host_kind": None,
            "component_index": 0,
            "neighbor_component_index": 0,
            "estimated_immune_diameter_px": round(immune_diameter_px, 6),
            "core_complete_class2": 0,
            "transition_complete_class2": 0,
            "transition_radial_complete_class2": [0, 0, 0, 0],
            "outer_reference_complete_class2": 0,
            "residual_safe_removal_capacity": 0,
            "density_gradient_resolved_removals": 0,
            "density_gradient_feasible": False,
            "field_area_cell_diameter_squares": 0.0,
            "three_band_complete_class2": 0,
            "interface_anchor_pixels": 0,
        },
    )

    # Scatter must fit four complete, mutually separated reference footprints
    # on one fine-10/Stroma component pair.  Summing annuli across unrelated
    # Pattern-5 islands grossly overestimates that capacity, so retain the
    # strongest single pair and use conservative 18-pixel footprint/occupancy
    # clearance plus the frozen 72-pixel sparse-focus separation proxy.
    p5_components, p5_count = ndimage.label(
        tissue == 10, structure=STRUCTURE_8
    )
    stroma_components, _ = ndimage.label(
        stroma_region, structure=STRUCTURE_8
    )
    stroma_interior_distance = ndimage.distance_transform_edt(stroma_region)

    def separated_site_count(region: np.ndarray) -> int:
        rows, cols = np.nonzero(region)
        if not len(rows):
            return 0
        clearance = np.minimum(
            stroma_interior_distance[rows, cols], empty_clearance[rows, cols]
        )
        order = np.lexsort((cols, rows, -clearance))
        chosen: list[tuple[int, int]] = []
        minimum_squared = 72 * 72
        for index in order:
            row, col = int(rows[index]), int(cols[index])
            if all(
                (row - prior_row) ** 2 + (col - prior_col) ** 2
                >= minimum_squared
                for prior_row, prior_col in chosen
            ):
                chosen.append((row, col))
                if len(chosen) == 16:
                    break
        return len(chosen)

    scatter_pairs = []
    for component_index in range(1, p5_count + 1):
        component = p5_components == component_index
        distance = ndimage.distance_transform_edt(~component)
        annulus = stroma_region & (distance >= 4.0) & (distance <= 48.0)
        adjacent_stroma_ids = np.unique(stroma_components[annulus])
        for stroma_index in adjacent_stroma_ids[adjacent_stroma_ids > 0]:
            local_annulus = annulus & (stroma_components == stroma_index)
            conservative_centers = (
                local_annulus
                & (stroma_interior_distance >= 18.0)
                & (empty_clearance >= 18.0)
            )
            scatter_pairs.append(
                {
                    "p5_component_index": component_index,
                    "stroma_component_index": int(stroma_index),
                    "outer_annulus_pixels": int(
                        np.count_nonzero(local_annulus)
                    ),
                    "conservative_center_pixels": int(
                        np.count_nonzero(conservative_centers)
                    ),
                    "separated_complete_footprint_proxy_capacity": (
                        separated_site_count(conservative_centers)
                    ),
                }
            )
    best_scatter_pair = max(
        scatter_pairs,
        key=lambda item: (
            int(item["separated_complete_footprint_proxy_capacity"]),
            int(item["conservative_center_pixels"]),
            int(item["outer_annulus_pixels"]),
            -int(item["p5_component_index"]),
            -int(item["stroma_component_index"]),
        ),
        default={
            "p5_component_index": 0,
            "stroma_component_index": 0,
            "outer_annulus_pixels": 0,
            "conservative_center_pixels": 0,
            "separated_complete_footprint_proxy_capacity": 0,
        },
    )
    return {
        "semantic_instance_count": total,
        "semantic_complete_instance_count": complete,
        "semantic_class_counts": {
            str(value): int(by_class[value]) for value in range(1, 6)
        },
        "semantic_complete_class_counts": {
            str(value): int(complete_by_class[value]) for value in range(1, 6)
        },
        "semantic_tumor_class_counts": {
            str(value): int(tumor_by_class[value]) for value in range(1, 6)
        },
        "semantic_stroma_class_counts": {
            str(value): int(stroma_by_class[value]) for value in range(1, 6)
        },
        "semantic_complete_tumor_class_counts": {
            str(value): int(complete_tumor_by_class[value])
            for value in range(1, 6)
        },
        "semantic_complete_stroma_class_counts": {
            str(value): int(complete_stroma_by_class[value])
            for value in range(1, 6)
        },
        "semantic_packing_center_pixels": {
            "local_host": packing_centers(tumor_region | stroma_region),
            "tumor": packing_centers(tumor_region),
            "stroma": packing_centers(stroma_region),
            "p5_outer_stroma_annulus": packing_centers(p5_outer_annulus),
        },
        "semantic_immune_depletion_interface_proxy": best_immune_field,
        "semantic_p5_scatter_interface_proxy": best_scatter_pair,
    }


def _evaluation_group(spec: EvaluationSpec) -> str:
    if spec.mechanism_id == "prostate-local-population-modulation":
        return "local"
    if spec.mechanism_id == "prostate-local-tumor-clearance":
        return "clearance"
    if spec.mechanism_id == "prostate-pattern-4-growth":
        return "p4"
    if spec.mechanism_id == "prostate-pattern-5-growth":
        return "p5"
    if spec.mechanism_id == "prostate-pattern-5-infiltrative-front":
        return "p5_cord"
    if spec.mechanism_id == "prostate-pattern-5-peripheral-scatter":
        return "p5_scatter"
    if spec.primitive_id == "residual-tumor-fragmentation-v1":
        return "fragmentation"
    return "retreat"


def _evaluation_eligible(spec: EvaluationSpec, item: dict[str, Any]) -> bool:
    tumor_classes = item["semantic_complete_tumor_class_counts"]
    stroma_classes = item["semantic_complete_stroma_class_counts"]
    complete_classes = item["semantic_complete_class_counts"]
    primitive = spec.primitive_id
    if primitive == "cell-type-abundance-increase-v1":
        return int(stroma_classes["3"]) >= 8
    if primitive == "cell-type-abundance-decrease-v1":
        return int(stroma_classes["3"]) >= 6
    if primitive == "cellularity-increase-v1":
        return int(item["semantic_complete_instance_count"]) >= 16
    if primitive == "cellularity-decrease-v1":
        return int(item["semantic_complete_instance_count"]) >= 36
    if primitive == "neoplastic-cell-abundance-increase-v1":
        return int(tumor_classes["1"]) >= 8
    if primitive == "neoplastic-cell-abundance-decrease-v1":
        return int(tumor_classes["1"]) >= 12
    if spec.mechanism_id == "prostate-pattern-5-peripheral-scatter":
        proxy = item["semantic_p5_scatter_interface_proxy"]
        return (
            int(complete_classes["1"]) >= 4
            and int(stroma_classes["3"]) >= 4
            and int(proxy["separated_complete_footprint_proxy_capacity"])
            >= 3
        )
    if spec.mechanism_id in {
        "prostate-pattern-4-growth",
        "prostate-pattern-5-growth",
        "prostate-pattern-5-infiltrative-front",
    }:
        return int(tumor_classes["1"]) >= 4 and int(stroma_classes["3"]) >= 4
    if spec.mechanism_id in {
        "prostate-operational-tumor-retreat",
        "prostate-local-tumor-clearance",
    }:
        return int(tumor_classes["1"]) >= 8 and int(stroma_classes["3"]) >= 4
    return True


def _evaluation_score(spec: EvaluationSpec, item: dict[str, Any]) -> float:
    group = _evaluation_group(spec)
    base = _coarse_group_scores(item)[group]
    primitive = spec.primitive_id
    tumor_classes = item["semantic_complete_tumor_class_counts"]
    stroma_classes = item["semantic_complete_stroma_class_counts"]
    complete_classes = item["semantic_complete_class_counts"]
    packing = item["semantic_packing_center_pixels"]
    if primitive.startswith("cell-type-abundance-"):
        free_space = (
            int(packing["local_host"])
            if primitive.endswith("increase-v1")
            else 0
        )
        return (
            base
            + 700 * int(stroma_classes["3"])
            + 2.0 * free_space
        )
    if primitive.startswith("cellularity-"):
        free_space = (
            int(packing["local_host"])
            if primitive.endswith("increase-v1")
            else 0
        )
        return (
            base
            + 250 * int(item["semantic_complete_instance_count"])
            + 2.0 * free_space
        )
    if primitive.startswith("neoplastic-cell-abundance-"):
        free_space = (
            int(packing["tumor"])
            if primitive.endswith("increase-v1")
            else 0
        )
        return base + 700 * int(tumor_classes["1"]) + 2.0 * free_space
    if spec.mechanism_id == "prostate-pattern-5-peripheral-scatter":
        scatter = item["semantic_p5_scatter_interface_proxy"]
        return (
            base
            + 700 * int(complete_classes["1"])
            + 20000
            * min(
                int(scatter["separated_complete_footprint_proxy_capacity"]),
                8,
            )
            + 4.0 * int(scatter["conservative_center_pixels"])
            + 0.25 * int(packing["p5_outer_stroma_annulus"])
        )
    if spec.mechanism_id in {
        "prostate-operational-tumor-retreat",
        "prostate-local-tumor-clearance",
    }:
        return base + 1200 * int(tumor_classes["1"])
    return base


def _cross_meta_targets(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return one immutable record per PANDA target patch in cross-meta val."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs") if isinstance(payload, dict) else None
    if not isinstance(pairs, list):
        raise TypeError("cross-meta eval must contain one top-level pairs list")
    panda = [
        item
        for item in pairs
        if isinstance(item, dict) and str(item.get("dataset", "")).upper() == "PANDA"
    ]
    if not panda:
        raise ValueError("cross-meta eval has no PANDA pairs")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in panda:
        sample_id = str(item.get("sample_id") or "")
        if not sample_id:
            raise ValueError("PANDA cross-meta pair lacks sample_id")
        grouped.setdefault(sample_id, []).append(item)
    records = []
    target_fields = (
        "case_id",
        "target_image",
        "target_tissue_mask",
        "target_nuclei_mask",
    )
    for sample_id, rows in sorted(grouped.items()):
        for field in target_fields:
            values = {str(item.get(field) or "") for item in rows}
            if len(values) != 1 or not next(iter(values)):
                raise ValueError(
                    f"cross-meta target {sample_id} has inconsistent {field}"
                )
        first = rows[0]
        image = Path(str(first["target_image"]))
        tissue = Path(str(first["target_tissue_mask"]))
        nuclei = Path(str(first["target_nuclei_mask"]))
        for asset in (image, tissue, nuclei):
            if not asset.is_file():
                raise FileNotFoundError(asset)
        if len({image.name, tissue.name, nuclei.name}) != 1:
            raise ValueError(f"cross-meta target basenames drifted: {sample_id}")
        records.append(
            {
                "filename": tissue.name,
                "sample_id": sample_id,
                "cross_meta_case_id": str(first["case_id"]),
                "slide_id": _slide_id(sample_id),
                "source_image": str(image.resolve()),
                "source_tissue_mask": str(tissue.resolve()),
                "source_nuclei_mask": str(nuclei.resolve()),
                "cross_meta_pair_count": len(rows),
                "cross_meta_reference_sample_ids": sorted(
                    {str(item.get("reference_sample_id")) for item in rows}
                ),
                "cross_meta_pair_difficulties": sorted(
                    {str(item.get("pair_difficulty")) for item in rows}
                ),
                "cross_meta_distances": sorted(
                    {int(item.get("distance")) for item in rows}
                ),
                "cross_meta_tissue_coverage_min": min(
                    float(item.get("tissue_coverage_ratio", 0.0)) for item in rows
                ),
                "cross_meta_area_coverage_min": min(
                    float(item.get("area_coverage_ratio", 0.0)) for item in rows
                ),
            }
        )
    return records, {
        "all_pair_count": len(pairs),
        "panda_pair_count": len(panda),
        "panda_unique_target_count": len(records),
    }


def _diverse_top(
    rows: Iterable[dict[str, Any]], *, maximum: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda item: (-float(item["selection_score"]), str(item["filename"])),
    )
    raw_by_slide: dict[str, list[dict[str, Any]]] = {}
    slide_order: list[str] = []
    for row in ranked:
        slide = str(row["slide_id"])
        if slide not in raw_by_slide:
            raw_by_slide[slide] = []
            slide_order.append(slide)
        raw_by_slide[slide].append(row)
    by_slide = {
        slide: _maximum_nonoverlap_subset(items)
        for slide, items in raw_by_slide.items()
    }
    # Round-robin preserves one-patch-per-slide whenever at least ``maximum``
    # slides are eligible.  Pattern-5 has only three source slides in the
    # frozen cross-meta val, so later rounds produce non-overlapping 2+2+1
    # source patches rather than substituting a wrong fine-label pattern.
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < maximum:
        progressed = False
        for slide in slide_order:
            current = by_slide[slide]
            if round_index < len(current):
                selected.append(current[round_index])
                progressed = True
                if len(selected) == maximum:
                    break
        if not progressed:
            break
        round_index += 1
    return selected


def _maximum_nonoverlap_subset(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Maximize independent patch count before score on one source slide.

    Targeted primitive screens normally leave only a small candidate set per
    slide.  Exhaustive search there avoids the classic greedy failure where a
    high-scoring central patch blocks two mutually non-overlapping flanks.  A
    deterministic greedy fallback keeps broad legacy screens bounded.
    """

    ordered = sorted(
        rows,
        key=lambda item: (
            -float(item["selection_score"]),
            str(item["filename"]),
        ),
    )
    if len(ordered) > 18:
        selected: list[dict[str, Any]] = []
        for row in ordered:
            if not any(_patches_overlap(row, prior) for prior in selected):
                selected.append(row)
        return selected
    best: tuple[int, float, tuple[str, ...], tuple[int, ...]] | None = None
    for bits in range(1 << len(ordered)):
        indices = tuple(
            index for index in range(len(ordered)) if bits & (1 << index)
        )
        if any(
            _patches_overlap(ordered[left], ordered[right])
            for offset, left in enumerate(indices)
            for right in indices[offset + 1 :]
        ):
            continue
        filenames = tuple(sorted(str(ordered[index]["filename"]) for index in indices))
        key = (
            len(indices),
            sum(float(ordered[index]["selection_score"]) for index in indices),
            tuple(reversed(filenames)),
            indices,
        )
        if best is None or key[:2] > best[:2] or (
            key[:2] == best[:2] and key[2] < best[2]
        ):
            best = key
    assert best is not None
    selected = [ordered[index] for index in best[3]]
    selected.sort(
        key=lambda item: (
            -float(item["selection_score"]),
            str(item["filename"]),
        )
    )
    return selected


def _patch_origin(item: dict[str, Any]) -> tuple[int, int]:
    sample_id = str(item.get("sample_id") or Path(str(item["filename"])).stem)
    match = SAMPLE_OFFSET.search(sample_id)
    if match is None:
        raise ValueError(f"PANDA sample lacks py/px patch offsets: {sample_id}")
    return int(match.group("y")), int(match.group("x"))


def _patches_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left["slide_id"]) != str(right["slide_id"]):
        return False
    left_y, left_x = _patch_origin(left)
    right_y, right_x = _patch_origin(right)
    return abs(left_y - right_y) < 512 and abs(left_x - right_x) < 512


def _patch_overlap_fraction(
    left: dict[str, Any], right: dict[str, Any]
) -> float:
    if str(left["slide_id"]) != str(right["slide_id"]):
        return 0.0
    left_y, left_x = _patch_origin(left)
    right_y, right_x = _patch_origin(right)
    overlap_y = max(0, 512 - abs(left_y - right_y))
    overlap_x = max(0, 512 - abs(left_x - right_x))
    return float(overlap_y * overlap_x) / float(512 * 512)


def _fill_distinct_overlap_minimized(
    selected: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    *,
    maximum: int,
) -> list[dict[str, Any]]:
    """Fill a targeted candidate stage without weakening case-level gates.

    Cases execute independently, so patch overlap is a diversity preference,
    not a biological or execution constraint.  Keep the maximum independent
    set first, then minimize overlap before considering score for alternatives.
    """

    result = list(selected)
    selected_filenames = {str(item["filename"]) for item in result}
    remaining = [
        item
        for item in eligible
        if str(item["filename"]) not in selected_filenames
    ]
    while len(result) < maximum and remaining:
        remaining.sort(
            key=lambda item: (
                max(
                    (_patch_overlap_fraction(item, prior) for prior in result),
                    default=0.0,
                ),
                sum(_patch_overlap_fraction(item, prior) for prior in result),
                -float(item["selection_score"]),
                str(item["filename"]),
            )
        )
        chosen = remaining.pop(0)
        result.append(chosen)
    return result


def _minimum_feasible_max_cases_per_slide(
    candidates: list[dict[str, Any]], *, final_case_count: int
) -> int:
    counts = Counter(str(item["slide_id"]) for item in candidates)
    if sum(counts.values()) < final_case_count:
        raise ValueError("candidate pool cannot supply the final case count")
    for maximum in range(1, final_case_count + 1):
        if sum(min(count, maximum) for count in counts.values()) >= final_case_count:
            return maximum
    raise AssertionError("finite candidate counts must yield a feasible cap")


def build_candidate_pool(
    *, cross_meta_eval: Path, coarse_keep_per_slide: int, per_evaluation: int,
    per_evaluation_overrides: dict[int, int] | None = None,
    base_pool: dict[str, Any] | None = None,
    target_evaluation_indices: set[int] | None = None,
    screening_workers: int = 1,
    rejected_native_filenames: set[str] | None = None,
    native_status_cache_binding: dict[str, str] | None = None,
) -> dict[str, Any]:
    targets, source_counts = _cross_meta_targets(cross_meta_eval)
    active_indices = (
        set(range(len(EVALUATIONS)))
        if target_evaluation_indices is None
        else set(target_evaluation_indices)
    )
    if not active_indices or not active_indices.issubset(
        set(range(len(EVALUATIONS)))
    ):
        raise ValueError("target evaluation indices are empty or out of range")
    if active_indices != set(range(len(EVALUATIONS))) and base_pool is None:
        raise ValueError("targeted selection requires one validated base pool")
    groups = tuple(
        sorted(
            {
                _evaluation_group(EVALUATIONS[index])
                for index in active_indices
            }
        )
    )
    best_by_slide: dict[
        str, dict[str, list[tuple[float, str, dict[str, Any]]]]
    ] = {
        group: {} for group in groups
    }
    aggregate = Counter()
    for path_index, target in enumerate(targets, start=1):
        metrics = _tissue_metrics(Path(target["source_tissue_mask"]))
        metrics.update(target)
        scores = _coarse_group_scores(metrics)
        for group in groups:
            if _coarse_eligible(group, metrics):
                aggregate[f"coarse_eligible::{group}"] += 1
                _keep_slide_best(
                    best_by_slide,
                    group=group,
                    score=scores[group],
                    metrics=metrics,
                    maximum_per_slide=coarse_keep_per_slide,
                )
        if path_index % 2000 == 0 or path_index == len(targets):
            print(
                json.dumps(
                    {
                        "stage": "mask_only_tissue_scan",
                        "completed": path_index,
                        "total": len(targets),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    heaps = {
        group: sorted(
            (
                item
                for slide_items in by_slide.values()
                for item in slide_items
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for group, by_slide in best_by_slide.items()
    }
    shortlist_by_filename: dict[str, dict[str, Any]] = {}
    for heap in heaps.values():
        for _, filename, metrics in heap:
            shortlist_by_filename[filename] = dict(metrics)
    shortlist_items = sorted(shortlist_by_filename.items())
    if screening_workers <= 0:
        raise ValueError("screening workers must be positive")

    def screen_one(filename: str, metrics: dict[str, Any]):
        nuclei_path = Path(str(metrics["source_nuclei_mask"]))
        image_path = Path(str(metrics["source_image"]))
        tissue_path = Path(str(metrics["source_tissue_mask"]))
        if (
            not nuclei_path.is_file()
            or not image_path.is_file()
            or not tissue_path.is_file()
        ):
            raise FileNotFoundError(f"PANDA cross-meta source asset missing: {filename}")
        return filename, _instance_metrics(
            tissue_path=tissue_path,
            nuclei_path=nuclei_path,
        )

    if screening_workers == 1:
        screened = (
            screen_one(filename, metrics)
            for filename, metrics in shortlist_items
        )
        iterator = enumerate(screened, start=1)
        for shortlist_index, (filename, instance_metrics) in iterator:
            shortlist_by_filename[filename].update(instance_metrics)
            if (
                shortlist_index % 100 == 0
                or shortlist_index == len(shortlist_items)
            ):
                print(
                    json.dumps(
                        {
                            "stage": "semantic_component_screening",
                            "completed": shortlist_index,
                            "total": len(shortlist_items),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=screening_workers) as executor:
            futures = {
                executor.submit(screen_one, filename, metrics): filename
                for filename, metrics in shortlist_items
            }
            for shortlist_index, future in enumerate(
                as_completed(futures), start=1
            ):
                filename, instance_metrics = future.result()
                shortlist_by_filename[filename].update(instance_metrics)
                if (
                    shortlist_index % 100 == 0
                    or shortlist_index == len(shortlist_items)
                ):
                    print(
                        json.dumps(
                            {
                                "stage": "semantic_component_screening",
                                "completed": shortlist_index,
                                "total": len(shortlist_items),
                                "workers": screening_workers,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    evaluations: list[dict[str, Any] | None] = (
        copy.deepcopy(base_pool["evaluations"])
        if base_pool is not None
        else [None] * len(EVALUATIONS)
    )
    count_overrides = per_evaluation_overrides or {}
    native_rejections = rejected_native_filenames or set()
    for evaluation_index, spec in enumerate(EVALUATIONS):
        if evaluation_index not in active_indices:
            continue
        required_candidate_count = count_overrides.get(
            evaluation_index, per_evaluation
        )
        group = _evaluation_group(spec)
        group_names = {item[1] for item in heaps[group]}
        eligible = []
        for filename in sorted(group_names):
            if filename in native_rejections:
                continue
            item = dict(shortlist_by_filename[filename])
            if not _evaluation_eligible(spec, item):
                continue
            item["selection_score"] = round(_evaluation_score(spec, item), 6)
            eligible.append(item)
        strict_nonoverlap_candidates = _diverse_top(
            eligible, maximum=required_candidate_count
        )
        minimum_candidate_count = 5
        candidates = strict_nonoverlap_candidates
        overlap_exception = None
        if (
            len(candidates) < required_candidate_count
            and len(eligible) >= minimum_candidate_count
        ):
            candidates = _fill_distinct_overlap_minimized(
                candidates,
                eligible,
                maximum=required_candidate_count,
            )
            maximum_overlap = max(
                (
                    _patch_overlap_fraction(left, right)
                    for index, left in enumerate(candidates)
                    for right in candidates[index + 1 :]
                ),
                default=0.0,
            )
            overlap_exception = {
                "policy_id": (
                    "targeted-candidate-overlap-fallback-v1"
                    if target_evaluation_indices is not None
                    else "full-pool-candidate-overlap-fallback-v1"
                ),
                "reason": "strict_nonoverlap_maximum_below_requested_count",
                "strict_nonoverlap_candidate_count": len(
                    strict_nonoverlap_candidates
                ),
                "overlap_filled_candidate_count": (
                    len(candidates) - len(strict_nonoverlap_candidates)
                ),
                "maximum_pair_overlap_fraction": round(maximum_overlap, 6),
                "distinct_patch_identity_required": True,
                "physiology_and_execution_gates_unchanged": True,
            }
        if len(candidates) < minimum_candidate_count:
            raise RuntimeError(
                f"{spec.evaluation_id} yielded only {len(candidates)} "
                "distinct cross-meta target patches"
            )
        candidate_slides = {str(item["slide_id"]) for item in candidates}
        minimum_final_distinct_slides = min(5, len(candidate_slides))
        maximum_cases_per_source_slide = (
            _minimum_feasible_max_cases_per_slide(
                candidates,
                final_case_count=5,
            )
        )
        evaluations[evaluation_index] = {
                "evaluation_id": spec.evaluation_id,
                "mechanism_id": spec.mechanism_id,
                "primitive_id": spec.primitive_id,
                "instruction": spec.instruction,
                "coarse_group": group,
                "candidate_count": len(candidates),
                "eligible_candidate_count": len(eligible),
                "candidate_source_slide_count": len(candidate_slides),
                "candidate_overlap_exception": overlap_exception,
                "final_diversity_policy": {
                    "final_case_count": 5,
                    "minimum_distinct_source_slides": minimum_final_distinct_slides,
                    "maximum_cases_per_source_slide": (
                        maximum_cases_per_source_slide
                    ),
                    "same_slide_patch_overlap_forbidden": (
                        overlap_exception is None
                    ),
                    "same_slide_patch_overlap_minimized": True,
                    "selection_order": (
                        "maximum_nonoverlap_then_minimum_overlap_then_score"
                        if overlap_exception is not None
                        else "score_ranked_slide_round_robin"
                    ),
                },
                "candidates": candidates,
            }
    if any(item is None for item in evaluations):
        raise RuntimeError("targeted candidate pool composition is incomplete")
    bound_evaluations = [item for item in evaluations if item is not None]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "producer_id": PRODUCER_ID,
        "production_status": "shadow_only",
        "planner_observation_policy": "mask_only_no_h_e",
        "semantic_instance_screening_policy": (
            "conservative_components_plus_single_interface_capacity_proxies_"
            "not_native_or_execution_authority"
        ),
        "cross_meta_eval": str(cross_meta_eval.resolve()),
        "cross_meta_eval_sha256": _sha256(cross_meta_eval),
        "selection_scope": "PANDA_unique_target_patches_only",
        "cross_meta_reference_assets_used_for_screening": False,
        "cross_meta_counts": source_counts,
        "dataset_file_count": len(targets),
        "coarse_keep_per_slide_per_group": coarse_keep_per_slide,
        "candidate_count_per_evaluation": min(
            int(item["candidate_count"]) for item in bound_evaluations
        ),
        "candidate_count_overrides": {
            str(index): count
            for index, count in sorted(count_overrides.items())
        },
        "aggregate_counts": dict(sorted(aggregate.items())),
        "base_candidate_pool_sha256": (
            base_pool.get("candidate_pool_sha256")
            if base_pool is not None
            else None
        ),
        "targeted_replacement_evaluation_indices": sorted(active_indices),
        "native_status_cache_binding": native_status_cache_binding,
        "evaluation_count": len(bound_evaluations),
        "evaluations": bound_evaluations,
        "freeze_status": "candidate_pool_only_pending_live_compiler_and_full_replay",
    }
    payload["candidate_pool_sha256"] = _canonical_sha256(payload)
    return payload


def validate_candidate_pool(payload: dict[str, Any]) -> None:
    declared = payload.get("candidate_pool_sha256")
    canonical = dict(payload)
    canonical.pop("candidate_pool_sha256", None)
    if declared != _canonical_sha256(canonical):
        raise ValueError("candidate pool digest mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("candidate pool schema mismatch")
    if payload.get("planner_observation_policy") != "mask_only_no_h_e":
        raise ValueError("PANDA selector must remain mask-only")
    native_binding = payload.get("native_status_cache_binding")
    if native_binding is not None:
        if not isinstance(native_binding, dict):
            raise ValueError("native status cache binding must be an object")
        native_path = Path(str(native_binding.get("path") or ""))
        if (
            native_binding.get("use")
            != "exclude_frozen_cellvit_rejections_only"
            or not native_path.is_file()
            or _sha256(native_path) != native_binding.get("sha256")
        ):
            raise ValueError("native status cache binding is missing or drifted")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != len(EVALUATIONS):
        raise ValueError("candidate pool evaluation count mismatch")
    expected = {item.evaluation_id for item in EVALUATIONS}
    observed = {str(item.get("evaluation_id")) for item in evaluations}
    if observed != expected:
        raise ValueError("candidate pool evaluation identities drifted")
    for evaluation in evaluations:
        candidates = evaluation.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 5:
            raise ValueError("every PANDA evaluation needs at least five candidates")
        filenames = [str(item.get("filename")) for item in candidates]
        if len(filenames) != len(set(filenames)):
            raise ValueError("candidate pool repeats a target patch")
        policy = evaluation.get("final_diversity_policy") or {}
        overlap_forbidden = bool(
            policy.get("same_slide_patch_overlap_forbidden", True)
        )
        for index, left in enumerate(candidates):
            for right in candidates[index + 1 :]:
                if overlap_forbidden and _patches_overlap(left, right):
                    raise ValueError(
                        "candidate pool contains overlapping same-slide patches"
                    )
        overlap_exception = evaluation.get("candidate_overlap_exception")
        if not overlap_forbidden:
            if (
                not isinstance(overlap_exception, dict)
                or overlap_exception.get("policy_id") not in {
                    "targeted-candidate-overlap-fallback-v1",
                    "full-pool-candidate-overlap-fallback-v1",
                }
                or not overlap_exception.get(
                    "physiology_and_execution_gates_unchanged"
                )
            ):
                raise ValueError("candidate overlap exception audit is invalid")
        if int(policy.get("final_case_count", 0)) != 5:
            raise ValueError("candidate pool final diversity policy drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-meta-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coarse-keep-per-slide", type=int, default=12)
    parser.add_argument("--per-evaluation", type=int, default=12)
    parser.add_argument(
        "--per-evaluation-overrides",
        help="Comma-separated INDEX:COUNT overrides for targeted expansion.",
    )
    parser.add_argument(
        "--base-pool",
        type=Path,
        help="Validated full pool retained outside targeted evaluation indices.",
    )
    parser.add_argument(
        "--target-evaluation-indices",
        help="Comma-separated evaluation indices recomputed over cross-meta masks.",
    )
    parser.add_argument("--screening-workers", type=int, default=1)
    parser.add_argument(
        "--native-status-cache",
        type=Path,
        help="Optional frozen CellViT status cache used only to exclude rejected patches.",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        validate_candidate_pool(payload)
        return 0
    base_pool = None
    if args.base_pool is not None:
        base_pool = json.loads(args.base_pool.read_text(encoding="utf-8"))
        validate_candidate_pool(base_pool)
        if (
            base_pool.get("cross_meta_eval_sha256")
            != _sha256(args.cross_meta_eval.resolve())
        ):
            raise ValueError("base pool is detached from requested cross-meta eval")
    target_indices = None
    if args.target_evaluation_indices:
        target_indices = {
            int(value.strip())
            for value in args.target_evaluation_indices.split(",")
            if value.strip()
        }
    rejected_native_filenames: set[str] = set()
    native_status_cache_binding = None
    if args.native_status_cache is not None:
        native_cache = json.loads(
            args.native_status_cache.read_text(encoding="utf-8")
        )
        if not isinstance(native_cache, dict):
            raise ValueError("native status cache must be an object")
        rejected_native_filenames = {
            str(filename)
            for filename, record in native_cache.items()
            if isinstance(record, dict) and record.get("status") == "rejected"
        }
        native_status_cache_binding = {
            "path": str(args.native_status_cache.resolve()),
            "sha256": _sha256(args.native_status_cache),
            "use": "exclude_frozen_cellvit_rejections_only",
        }
    payload = build_candidate_pool(
        cross_meta_eval=args.cross_meta_eval.resolve(),
        coarse_keep_per_slide=int(args.coarse_keep_per_slide),
        per_evaluation=int(args.per_evaluation),
        per_evaluation_overrides=_parse_evaluation_count_overrides(
            args.per_evaluation_overrides
        ),
        base_pool=base_pool,
        target_evaluation_indices=target_indices,
        screening_workers=int(args.screening_workers),
        rejected_native_filenames=rejected_native_filenames,
        native_status_cache_binding=native_status_cache_binding,
    )
    validate_candidate_pool(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_pool_sha256": payload["candidate_pool_sha256"],
                "evaluation_count": payload["evaluation_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
