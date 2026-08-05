"""Pre-compile nuclei and halo capacity before tissue candidate drawing.

This module is deliberately deterministic.  It turns the joint skill contract
into executable exclusions and per-interface capacity estimates so that a
tissue front is not approved first and discovered to be unrealizable by the
cell tools later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil, hypot
from typing import Any

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import CandidateMask
from phase3_mask_edit_refine.scene import SceneAnalysis
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle

from .budget import JointBudgetAllocation
from .cell_layouts import (
    ReferenceNucleusShape,
    build_reference_shape_library,
    target_cell_class,
)
from .models import JointCaseContext, JointContractError, JointEditPlan
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle


PREFLIGHT_VERSION = "joint-nuclei-preflight-v1"


@dataclass(frozen=True)
class InterfaceNucleiCapacity:
    interface_id: str
    source_component_id: str
    target_component_id: str
    contact_pixels: int
    gate_bounded_depth_px: int
    editable_tissue_capacity_pixels: int
    removable_instance_ids: tuple[str, ...]
    protected_instance_overlap_ids: tuple[str, ...]
    legal_halo_pixels: int
    reference_fit_center_pixels: int
    estimated_add_capacity: int
    feasible: bool
    reasons: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointNucleiPreflight:
    version: str
    target_cell_class: int
    target_tissue_label: str
    eligible_reference_ids: tuple[str, ...]
    rejected_reference_ids: dict[str, str]
    removable_instance_ids: tuple[str, ...]
    protected_instance_ids: tuple[str, ...]
    tissue_exclusion_instance_ids: tuple[str, ...]
    protected_instance_reasons: dict[str, str]
    maximum_halo_px: int
    required_auxiliary_missing: tuple[str, ...]
    required_provenance_missing: tuple[str, ...]
    interfaces: tuple[InterfaceNucleiCapacity, ...]
    protected_tissue_change_mask: np.ndarray

    @property
    def feasible_interface_ids(self) -> tuple[str, ...]:
        return tuple(item.interface_id for item in self.interfaces if item.feasible)

    def interface(self, interface_id: str) -> InterfaceNucleiCapacity | None:
        return next(
            (item for item in self.interfaces if item.interface_id == interface_id),
            None,
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target_cell_class": self.target_cell_class,
            "target_tissue_label": self.target_tissue_label,
            "eligible_reference_ids": list(self.eligible_reference_ids),
            "rejected_reference_ids": dict(self.rejected_reference_ids),
            "removable_instance_ids": list(self.removable_instance_ids),
            "protected_instance_ids": list(self.protected_instance_ids),
            "tissue_exclusion_instance_ids": list(
                self.tissue_exclusion_instance_ids
            ),
            "protected_instance_reasons": dict(self.protected_instance_reasons),
            "maximum_halo_px": self.maximum_halo_px,
            "required_auxiliary_missing": list(self.required_auxiliary_missing),
            "required_provenance_missing": list(self.required_provenance_missing),
            "protected_tissue_change_pixels": int(
                np.count_nonzero(self.protected_tissue_change_mask)
            ),
            "feasible_interface_ids": list(self.feasible_interface_ids),
            "interfaces": [item.to_metadata() for item in self.interfaces],
        }


@dataclass(frozen=True)
class CandidateCellFeasibility:
    candidate_id: str
    passed: bool
    removable_instance_ids: tuple[str, ...]
    protected_overlap_ids: tuple[str, ...]
    nonlocal_extension_ids: tuple[str, ...]
    legal_core_pixels: int
    reference_fit_center_pixels: int
    estimated_add_capacity: int
    reasons: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def build_joint_nuclei_preflight(
    *,
    case: JointCaseContext,
    source_tissue: np.ndarray,
    schema: MaskProfileSchema,
    scene: JointSceneAnalysis,
    tissue_bundle: ActiveKnowledgeBundle,
    joint_bundle: JointSkillBundle,
    allocation: JointBudgetAllocation,
) -> JointNucleiPreflight:
    """Compile cell executability before selecting/drawing tissue fronts."""

    del allocation  # retained in the public contract for future calibrated quotas
    target_label = tissue_bundle.edit_contract.target_label
    target_class = target_cell_class(target_label, schema)
    if target_class not in joint_bundle.mechanism.cell_program.allowed_cell_classes:
        raise JointContractError(
            f"target cell class {target_class} is unavailable to the mechanism"
        )
    references, rejected = build_reference_shape_library(
        scene,
        class_id=target_class,
    )
    maximum_halo = int(joint_bundle.mechanism.cell_program.halo_distance_px[1])
    removable: list[str] = []
    protected: list[str] = []
    tissue_exclusions: list[str] = []
    protected_reasons: dict[str, str] = {}
    protected_mask = np.zeros_like(source_tissue, dtype=bool)
    for item in scene.cells.instances:
        component = np.asarray(scene.instance_masks[item.instance_id], dtype=bool)
        x0, y0, x1, y1 = item.bbox_xyxy
        span = int(ceil(hypot(max(0, x1 - x0), max(0, y1 - y0))))
        reason = None
        if item.touches_border:
            reason = "patch_boundary_censored_instance"
        elif ndimage.label(
            component, structure=np.ones((3, 3), dtype=bool)
        )[1] != 1:
            reason = "disconnected_instance"
        elif span > maximum_halo:
            # If any pixel of this instance intersects T, a diameter larger
            # than the authorized halo cannot guarantee that REMOVE_WHOLE stays
            # local to T.  Such observations remain pixel-exactly protected.
            reason = "whole_instance_extent_exceeds_authorized_halo"
        if reason is None:
            removable.append(item.instance_id)
        else:
            protected.append(item.instance_id)
            protected_reasons[item.instance_id] = reason
            # RETAIN is a valid cell action. A censored/non-local source
            # observation therefore does not automatically carve a hole in
            # every tissue edit. Exclusion is required only when retaining its
            # cell class would contradict the target tissue semantics.
            if _requires_tissue_exclusion(item.class_id, target_label):
                tissue_exclusions.append(item.instance_id)
                protected_mask |= component

    required_auxiliary = set(
        joint_bundle.mechanism.representability.required_auxiliary_structures
    )
    missing_auxiliary = tuple(
        sorted(required_auxiliary - set(scene.auxiliary_structure_masks))
    )
    missing_provenance = tuple(
        field
        for field in joint_bundle.annotation_profile.required_provenance_fields
        if not _known_provenance(case.provenance.get(field))
    )
    prohibited_tissue = np.isin(
        source_tissue,
        joint_bundle.annotation_profile.prohibit_cell_placement_fine_ids,
    )
    source_contract = joint_bundle.mechanism.tissue_program.primitive_label_contracts.get(
        case.primitive_id
    )
    if source_contract is None:
        raise JointContractError("joint mechanism has no primitive label contract")
    allowed_sources = set(tissue_bundle.edit_contract.source_label_options).intersection(
        source_contract["source_labels"]
    )
    removable_set = set(removable)
    reference = _median_reference(references)
    interface_reports: list[InterfaceNucleiCapacity] = []
    for interface in scene.tissue.graph.interfaces:
        if (
            interface.source_label not in allowed_sources
            or interface.target_label != target_label
        ):
            continue
        source_component = scene.tissue.component_masks[interface.source_component_id]
        interface_mask = scene.tissue.interface_masks[interface.interface_id]
        # Bind the generator to a depth that can satisfy the gate even before
        # a candidate-specific contact span is known.  A 0.80 margin absorbs
        # tapered ends and pixels that do not become part of the final front.
        depth_cap = max(1, min(128, int(np.floor(interface.contact_pixels * 0.80))))
        distance = ndimage.distance_transform_edt(~interface_mask)
        envelope = (
            source_component
            & (distance <= depth_cap)
            & ~prohibited_tissue
            & ~protected_mask
        )
        overlapping_removable = tuple(
            sorted(
                instance_id
                for instance_id in removable_set
                if np.any(scene.instance_masks[instance_id] & envelope)
            )
        )
        overlapping_protected = tuple(
            sorted(
                instance_id
                for instance_id in protected
                if np.any(scene.instance_masks[instance_id] & source_component)
            )
        )
        free = _free_after_removing_instances(
            envelope,
            source_nuclei=scene.source_nuclei,
            scene=scene,
            removable_ids=overlapping_removable,
        )
        fit_centers = _reference_fit_centers(free, reference)
        average_area = (
            float(np.median([item.area_px for item in references]))
            if references
            else 0.0
        )
        add_capacity = (
            min(
                int(np.count_nonzero(fit_centers)),
                int(np.count_nonzero(free) / max(1.0, average_area * 2.0)),
            )
            if reference is not None
            else 0
        )
        halo = np.zeros_like(envelope)
        if joint_bundle.mechanism.coupling.cell_only_target_fraction > 0:
            halo = (
                ndimage.binary_dilation(interface_mask, iterations=maximum_halo)
                & ~prohibited_tissue
                & ~protected_mask
                & ~envelope
            )
        reasons: list[str] = []
        if not references:
            reasons.append("no_complete_same_class_reference_shape")
        if missing_auxiliary:
            reasons.append("required_auxiliary_missing")
        if missing_provenance:
            reasons.append("required_profile_provenance_missing")
        if not np.any(envelope):
            reasons.append("no_cell_safe_tissue_capacity")
        if "add" in joint_bundle.mechanism.cell_program.actions and add_capacity <= 0:
            reasons.append("no_complete_shape_placement_capacity")
        if (
            joint_bundle.mechanism.coupling.cell_only_target_fraction > 0
            and not np.any(halo)
        ):
            reasons.append("no_legal_cell_only_halo")
        interface_reports.append(
            InterfaceNucleiCapacity(
                interface_id=interface.interface_id,
                source_component_id=interface.source_component_id,
                target_component_id=interface.target_component_id,
                contact_pixels=int(interface.contact_pixels),
                gate_bounded_depth_px=depth_cap,
                editable_tissue_capacity_pixels=int(np.count_nonzero(envelope)),
                removable_instance_ids=overlapping_removable,
                protected_instance_overlap_ids=overlapping_protected,
                legal_halo_pixels=int(np.count_nonzero(halo)),
                reference_fit_center_pixels=int(np.count_nonzero(fit_centers)),
                estimated_add_capacity=max(0, int(add_capacity)),
                feasible=not reasons,
                reasons=tuple(reasons),
            )
        )
    return JointNucleiPreflight(
        version=PREFLIGHT_VERSION,
        target_cell_class=target_class,
        target_tissue_label=target_label,
        eligible_reference_ids=tuple(item.instance_id for item in references),
        rejected_reference_ids=dict(rejected),
        removable_instance_ids=tuple(sorted(removable)),
        protected_instance_ids=tuple(sorted(protected)),
        tissue_exclusion_instance_ids=tuple(sorted(tissue_exclusions)),
        protected_instance_reasons=protected_reasons,
        maximum_halo_px=maximum_halo,
        required_auxiliary_missing=missing_auxiliary,
        required_provenance_missing=missing_provenance,
        interfaces=tuple(interface_reports),
        protected_tissue_change_mask=protected_mask,
    )


def augment_tissue_scene_with_nuclei_preflight(
    scene: SceneAnalysis,
    preflight: JointNucleiPreflight,
) -> SceneAnalysis:
    """Make non-local/protected nucleus footprints unavailable to tissue tools."""

    prohibited = dict(scene.prohibited_region_masks)
    prohibited["joint:nuclei:protected_instances"] = np.asarray(
        preflight.protected_tissue_change_mask,
        dtype=bool,
    )
    return replace(scene, prohibited_region_masks=prohibited)


def bind_joint_plan_to_nuclei_preflight(
    plan: JointEditPlan,
    preflight: JointNucleiPreflight,
) -> JointEditPlan:
    """Bind deterministic whole-instance protection into any Planner output."""

    protected = tuple(
        sorted(
            set(plan.cell_plan.protected_instance_ids).union(
                preflight.protected_instance_ids
            )
        )
    )
    return replace(
        plan,
        cell_plan=replace(plan.cell_plan, protected_instance_ids=protected),
    )


def assess_candidate_cell_feasibility(
    candidate: CandidateMask,
    *,
    source_tissue: np.ndarray,
    scene: JointSceneAnalysis,
    preflight: JointNucleiPreflight,
    joint_bundle: JointSkillBundle,
) -> CandidateCellFeasibility:
    """Exact candidate-local closure/containment check before cell drawing."""

    core = np.asarray(candidate.change_region, dtype=bool)
    protected_set = set(preflight.protected_instance_ids)
    exclusion_set = set(preflight.tissue_exclusion_instance_ids)
    removable_set = set(preflight.removable_instance_ids)
    protected_overlap = tuple(
        sorted(
            instance_id
            for instance_id in exclusion_set
            if np.any(scene.instance_masks[instance_id] & core)
        )
    )
    intersecting = tuple(
        sorted(
            instance_id
            for instance_id, component in scene.instance_masks.items()
            if np.any(component & core)
        )
    )
    removals = tuple(
        instance_id for instance_id in intersecting if instance_id in removable_set
    )
    allowed_closure = ndimage.binary_dilation(
        core,
        iterations=max(1, preflight.maximum_halo_px),
    )
    nonlocal_extensions = tuple(
        sorted(
            instance_id
            for instance_id in removals
            if np.any(scene.instance_masks[instance_id] & ~allowed_closure)
        )
    )
    prohibited_ids = joint_bundle.annotation_profile.prohibit_cell_placement_fine_ids
    legal_core = core & ~np.isin(candidate.target_mask, prohibited_ids)
    references = tuple(
        _reference_from_scene(scene, instance_id, preflight.target_cell_class)
        for instance_id in preflight.eligible_reference_ids
    )
    references = tuple(item for item in references if item is not None)
    reference = _median_reference(references)
    free = _free_after_removing_instances(
        legal_core,
        source_nuclei=scene.source_nuclei,
        scene=scene,
        removable_ids=removals,
    )
    fit_centers = _reference_fit_centers(free, reference)
    average_area = (
        float(np.median([item.area_px for item in references]))
        if references
        else 0.0
    )
    add_capacity = (
        min(
            int(np.count_nonzero(fit_centers)),
            int(np.count_nonzero(free) / max(1.0, average_area * 2.0)),
        )
        if reference is not None
        else 0
    )
    reasons: list[str] = []
    if protected_overlap:
        reasons.append("tissue_change_intersects_protected_instance")
    if set(intersecting) - removable_set - protected_set:
        reasons.append("tissue_change_intersects_unclassified_instance")
    if nonlocal_extensions:
        reasons.append("whole_instance_removal_exceeds_authorized_halo")
    if not np.any(legal_core):
        reasons.append("candidate_has_no_legal_cell_core")
    capacity_adaptive_tissue_fallback = bool(
        candidate.tool_trace.get("area_fallback_used")
    ) and joint_bundle.mechanism.coupling.cell_only_target_fraction == 0
    if (
        "add" in joint_bundle.mechanism.cell_program.actions
        and add_capacity <= 0
        and not capacity_adaptive_tissue_fallback
    ):
        reasons.append("candidate_cannot_fit_one_complete_reference_shape")
    return CandidateCellFeasibility(
        candidate_id=candidate.candidate_id,
        passed=not reasons,
        removable_instance_ids=removals,
        protected_overlap_ids=protected_overlap,
        nonlocal_extension_ids=nonlocal_extensions,
        legal_core_pixels=int(np.count_nonzero(legal_core)),
        reference_fit_center_pixels=int(np.count_nonzero(fit_centers)),
        estimated_add_capacity=max(0, int(add_capacity)),
        reasons=tuple(reasons),
    )


def _known_provenance(value: Any) -> bool:
    if not value:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith("unknown") or "not_recorded" in lowered:
            return False
    return True


def _requires_tissue_exclusion(class_id: int, target_label: str) -> bool:
    """Whether a protected source cell cannot be retained under the target."""

    if target_label == "Tumor":
        # Tumor semantic regions legitimately contain immune, stromal and
        # epithelial observations in addition to neoplastic nuclei.
        return False
    if target_label in {"Stroma", "Other tissue", "Immune infiltrate"}:
        return int(class_id) == 1
    if target_label == "Normal epithelium":
        return int(class_id) == 1
    if target_label == "Necrosis":
        return int(class_id) not in {2, 4}
    return True


def _median_reference(
    references: tuple[ReferenceNucleusShape, ...],
) -> ReferenceNucleusShape | None:
    if not references:
        return None
    return sorted(references, key=lambda item: (item.area_px, item.instance_id))[
        len(references) // 2
    ]


def _free_after_removing_instances(
    zone: np.ndarray,
    *,
    source_nuclei: np.ndarray,
    scene: JointSceneAnalysis,
    removable_ids: tuple[str, ...],
) -> np.ndarray:
    occupied = np.asarray(source_nuclei) > 0
    occupied = occupied.copy()
    for instance_id in removable_ids:
        occupied[scene.instance_masks[instance_id]] = False
    guard = ndimage.binary_dilation(occupied, iterations=1)
    return np.asarray(zone, dtype=bool) & ~guard


def _reference_fit_centers(
    free: np.ndarray,
    reference: ReferenceNucleusShape | None,
) -> np.ndarray:
    if reference is None or not np.any(free):
        return np.zeros_like(free, dtype=bool)
    return ndimage.binary_erosion(
        np.asarray(free, dtype=bool),
        structure=np.asarray(reference.mask, dtype=bool),
        border_value=0,
    )


def _reference_from_scene(
    scene: JointSceneAnalysis,
    instance_id: str,
    class_id: int,
) -> ReferenceNucleusShape | None:
    metadata = next(
        (item for item in scene.cells.instances if item.instance_id == instance_id),
        None,
    )
    component = scene.instance_masks.get(instance_id)
    if metadata is None or component is None or metadata.class_id != class_id:
        return None
    x0, y0, x1, y1 = metadata.bbox_xyxy
    cropped = np.asarray(component, dtype=bool)[y0:y1, x0:x1]
    if not np.any(cropped):
        return None
    return ReferenceNucleusShape(
        instance_id=instance_id,
        class_id=class_id,
        mask=cropped,
        source=metadata.source,
        area_px=int(np.count_nonzero(cropped)),
    )
