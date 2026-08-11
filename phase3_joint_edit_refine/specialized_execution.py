"""Deterministic adapters for closed fine-architecture and native-void edits.

These executors make the mask-side contracts testable without claiming that a
frozen H&E generator can render the corresponding pathology.  Execution scope
remains closed until paired generator capability evidence is attached.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import CandidateMask

from .models import JointContractError
from .nuclei import iter_instances

# Initialize the legacy public registry before importing its fine-transition
# adapter.  The legacy package exports the registry eagerly; importing the
# specialized module first would otherwise enter its historical import cycle.
# isort: off
importlib.import_module("phase3_mask_edit.generic")

from phase3_mask_edit.specialized.catalog import specialized_primitives_for
from phase3_mask_edit.specialized.fine_transition import (
    apply_fine_label_transition,
)
# isort: on

SPECIALIZED_EXECUTOR_VERSION = "joint-specialized-executor-v1"

ARCHITECTURE_PROGRESSIONS = frozenset(
    {"gleason_upgrade_3to4", "gleason_upgrade_4to5"}
)


@dataclass(frozen=True)
class StructuralVoidExecutionContract:
    placement_center_region: np.ndarray
    valid_footprint_region: np.ndarray
    protected_structure_region: np.ndarray
    generation_support: np.ndarray
    source_tissue_digest_binding: str
    target_delta_count: int
    estimated_capacity: int
    nominal_nucleus_diameter_px: float
    minimum_primary_separation_px: float
    maximum_primary_separation_px: float
    tool_trace: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in asdict(self).items()
                if not isinstance(value, np.ndarray)
            },
            "placement_center_pixels": int(
                np.count_nonzero(self.placement_center_region)
            ),
            "valid_footprint_pixels": int(
                np.count_nonzero(self.valid_footprint_region)
            ),
            "protected_structure_pixels": int(
                np.count_nonzero(self.protected_structure_region)
            ),
            "generation_support_pixels": int(
                np.count_nonzero(self.generation_support)
            ),
        }


def compile_structural_void_execution(
    *,
    source_tissue: np.ndarray,
    source_nuclei: np.ndarray,
    source_tissue_sha256: str,
    primary_tumor_region: np.ndarray,
    receiving_void_region: np.ndarray,
    protected_structure_region: np.ndarray,
    target_delta_count: int,
    maximum_primary_separation_px: float,
) -> StructuralVoidExecutionContract:
    """Compile legal STAS-like placement support without inventing airspaces."""

    tissue = np.asarray(source_tissue)
    nuclei = np.asarray(source_nuclei)
    tumor = np.asarray(primary_tumor_region, dtype=bool)
    receiving = np.asarray(receiving_void_region, dtype=bool)
    protected = np.asarray(protected_structure_region, dtype=bool)
    if not (
        tissue.ndim == 2
        and tissue.shape == nuclei.shape == tumor.shape == receiving.shape == protected.shape
    ):
        raise JointContractError("structural-void inputs must be aligned 2-D rasters")
    if not source_tissue_sha256:
        raise JointContractError("structural-void execution requires source digest binding")
    if target_delta_count <= 0:
        raise JointContractError("structural-void execution requires a positive cell quota")
    if not np.any(tumor) or not np.any(receiving):
        raise JointContractError("primary tumor and receiving void must both be observed")
    if np.any(receiving & protected):
        raise JointContractError(
            "receiving void overlaps a protected native structure"
        )
    complete_areas = [
        int(np.count_nonzero(component))
        for _instance_id, class_id, component in iter_instances(nuclei)
        if class_id == 1 and not _touches_border(component)
    ]
    if not complete_areas:
        raise JointContractError(
            "structural-void execution has no complete same-patch neoplastic reference"
        )
    median_area = float(np.median(complete_areas))
    diameter = max(3.0, 2.0 * np.sqrt(median_area / np.pi))
    minimum_separation = diameter
    if maximum_primary_separation_px < minimum_separation:
        raise JointContractError(
            "structural-void maximum separation is smaller than one local nucleus diameter"
        )
    distance_from_primary = ndimage.distance_transform_edt(~tumor)
    valid_footprint = (
        receiving
        & ~protected
        & (distance_from_primary >= minimum_separation)
        & (distance_from_primary <= maximum_primary_separation_px)
    )
    footprint_radius = max(1, int(np.ceil(diameter / 2.0)))
    placement = ndimage.binary_erosion(
        valid_footprint,
        structure=_disk(footprint_radius),
        border_value=0,
    )
    if not np.any(placement):
        raise JointContractError(
            "no complete neoplastic footprint fits the verified structural void"
        )
    separation_radius = max(1, int(np.ceil(diameter)))
    estimated_capacity = max(
        0,
        _greedy_center_capacity(placement, separation_radius=separation_radius),
    )
    if estimated_capacity < target_delta_count:
        raise JointContractError(
            "verified structural void lacks complete-instance packing capacity: "
            f"{estimated_capacity}<{target_delta_count}"
        )
    support = ndimage.binary_dilation(
        valid_footprint,
        iterations=max(1, int(np.ceil(1.5 * diameter))),
    )
    support &= ~protected
    support |= valid_footprint
    return StructuralVoidExecutionContract(
        placement_center_region=placement,
        valid_footprint_region=valid_footprint,
        protected_structure_region=protected,
        generation_support=support,
        source_tissue_digest_binding=source_tissue_sha256,
        target_delta_count=target_delta_count,
        estimated_capacity=estimated_capacity,
        nominal_nucleus_diameter_px=diameter,
        minimum_primary_separation_px=minimum_separation,
        maximum_primary_separation_px=float(maximum_primary_separation_px),
        tool_trace={
            "executor_version": SPECIALIZED_EXECUTOR_VERSION,
            "primitive_id": "structural-void-spread-v1",
            "tissue_policy": "pixel_immutable",
            "receiving_policy": "producer_bound_native_void_only",
            "reference_policy": "complete_same_patch_neoplastic_instances",
            "placement_policy": "separated_from_primary_and_protected_structure",
        },
    )


def execute_architecture_progression(
    source_tissue: np.ndarray,
    *,
    schema: MaskProfileSchema,
    transition_id: str,
    target_tissue_pixels: int,
    gland_lumen_map: np.ndarray,
) -> CandidateMask:
    """Adapt the mature fine-transition tool to one explicit Gleason upgrade."""

    if schema.reference_profile.upper() != "PANDA":
        raise JointContractError(
            "architecture progression v1 is executable only in PANDA fine labels"
        )
    if transition_id not in ARCHITECTURE_PROGRESSIONS:
        raise JointContractError(
            "architecture progression requires an explicit supported upgrade"
        )
    configurations = {
        str(item["name"]): item
        for item in specialized_primitives_for("PANDA")
    }
    config = configurations[transition_id]
    operation = config["mask_operation"]
    source_ids = tuple(int(value) for value in operation["source_fine_ids"])
    source_pixels = int(np.count_nonzero(np.isin(source_tissue, source_ids)))
    if source_pixels <= 0:
        raise JointContractError(
            "architecture progression source fine pattern is absent"
        )
    if target_tissue_pixels <= 0:
        raise JointContractError(
            "architecture progression requires a positive tissue budget"
        )
    target_fraction = min(1.0, target_tissue_pixels / source_pixels)
    context = MaskEditContext.from_mask(source_tissue, schema)
    result = apply_fine_label_transition(
        source_tissue,
        schema,
        context,
        config,
        EditIntent(
            primitive=transition_id,
            reference_profile="PANDA",
            target_change_fraction=target_fraction,
        ),
    )
    lumen = np.asarray(gland_lumen_map, dtype=bool)
    if lumen.shape != result.change_region.shape:
        raise JointContractError("gland/lumen map is not aligned")
    if np.any(result.change_region & lumen):
        raise JointContractError(
            "architecture transition intersects a protected gland/lumen space"
        )
    target_id = int(operation["target_fine_id"])
    if not np.all(result.target_mask[result.change_region] == target_id):
        raise JointContractError("architecture tool failed its target fine-ID contract")
    return CandidateMask(
        candidate_id=f"architecture:{transition_id}",
        interface_id=f"architecture-unit:{transition_id}",
        tool_name="legacy_fine_transition_adapter",
        target_mask=result.target_mask,
        change_region=result.change_region,
        tool_trace={
            **result.ops_log,
            "executor_version": SPECIALIZED_EXECUTOR_VERSION,
            "joint_primitive_id": "architecture-progression-v1",
            "transition_id": transition_id,
            "protected_lumen_overlap_pixels": 0,
            "requested_whole_patch_pixels": int(target_tissue_pixels),
        },
    )


def validate_architecture_postcondition(
    *,
    source_tissue: np.ndarray,
    candidate: CandidateMask,
    transition_id: str,
    gland_lumen_map: np.ndarray,
) -> dict[str, Any]:
    configs = {
        str(item["name"]): item
        for item in specialized_primitives_for("PANDA")
    }
    if transition_id not in configs:
        raise JointContractError("unknown architecture transition")
    operation = configs[transition_id]["mask_operation"]
    source_ids = tuple(int(value) for value in operation["source_fine_ids"])
    target_id = int(operation["target_fine_id"])
    changed = np.asarray(candidate.change_region, dtype=bool)
    source = np.asarray(source_tissue)
    target = np.asarray(candidate.target_mask)
    lumen = np.asarray(gland_lumen_map, dtype=bool)
    checks = {
        "change_region_exact": bool(np.array_equal(changed, source != target)),
        "source_fine_ids_only": bool(np.all(np.isin(source[changed], source_ids))),
        "target_fine_id_only": bool(np.all(target[changed] == target_id)),
        "unrequested_pixels_preserved": bool(np.array_equal(source[~changed], target[~changed])),
        "gland_lumen_preserved": bool(not np.any(changed & lumen)),
        "whole_component_policy": bool(
            candidate.tool_trace.get("selection_policy")
            == "whole_source_components"
        ),
    }
    return {
        "passed": all(checks.values()) and bool(np.any(changed)),
        "transition_id": transition_id,
        "checks": checks,
        "changed_pixels": int(np.count_nonzero(changed)),
    }


def _touches_border(component: np.ndarray) -> bool:
    return bool(
        np.any(component[0])
        or np.any(component[-1])
        or np.any(component[:, 0])
        or np.any(component[:, -1])
    )


def _disk(radius: int) -> np.ndarray:
    rows, cols = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (rows**2 + cols**2) <= radius**2


def _greedy_center_capacity(
    placement: np.ndarray,
    *,
    separation_radius: int,
) -> int:
    available = np.asarray(placement, dtype=bool).copy()
    count = 0
    exclusion = _disk(separation_radius)
    radius = separation_radius
    while np.any(available):
        row, col = np.argwhere(available)[0]
        count += 1
        row0, row1 = max(0, row - radius), min(available.shape[0], row + radius + 1)
        col0, col1 = max(0, col - radius), min(available.shape[1], col + radius + 1)
        kernel = exclusion[
            row0 - (row - radius) : exclusion.shape[0] - ((row + radius + 1) - row1),
            col0 - (col - radius) : exclusion.shape[1] - ((col + radius + 1) - col1),
        ]
        available[row0:row1, col0:col1][kernel] = False
    return count
