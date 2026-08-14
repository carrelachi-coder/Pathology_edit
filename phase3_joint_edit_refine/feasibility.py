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
    independent_focus_minimum_center_separation_px,
)
from .instance_authority import build_scene_instance_authority
from .models import JointCaseContext, JointContractError, JointEditPlan
from .packing import certify_complete_footprint_packing
from .scene import JointSceneAnalysis
from .seam import (
    anchor_coverage_fraction,
    class_center_mask,
    compile_adaptive_seam,
    compile_continuity_center_quota,
    compile_executable_continuity_count,
    compile_minimum_continuity_count,
    target_cell_class_for_tissue,
)
from .skills.repository import JointSkillBundle

PREFLIGHT_VERSION = "joint-nuclei-preflight-v14"
SHAPE_CAPACITY_CLEARANCE_FACTOR = 1.25


@dataclass(frozen=True)
class InterfaceNucleiCapacity:
    interface_id: str
    source_component_id: str
    target_component_id: str
    boundary_topology: str
    external_tumor_stroma_boundary: bool
    internal_enclosed_interface: bool
    protected_fine_ids_within_band: tuple[int, ...]
    contact_pixels: int
    gate_bounded_depth_px: int
    source_component_capacity_pixels: int
    editable_tissue_capacity_pixels: int
    removable_instance_ids: tuple[str, ...]
    required_removal_cell_classes: tuple[int, ...]
    estimated_removal_capacity: int
    protected_instance_overlap_ids: tuple[str, ...]
    legal_halo_pixels: int
    reference_fit_center_pixels: int
    reference_area_p95: float
    target_density_per_pixel: float
    required_add_count: int
    required_seam_count: int
    estimated_add_capacity: int
    estimated_seam_capacity: int
    continuity_assessment_stage: str
    cell_feasible_anchor_segment_ids: tuple[str, ...]
    anchor_continuity_reports: tuple[dict[str, Any], ...]
    capacity_margin_count: int
    exact_packing_certificate: dict[str, Any]
    feasible: bool
    reasons: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointNucleiPreflight:
    version: str
    source_instance_authority_sha256: str
    source_instance_authority_count: int
    source_instance_observation_quality: str
    target_cell_class: int
    target_cell_classes: tuple[int, ...]
    target_tissue_label: str
    target_density_per_pixel: float
    target_density_by_class: dict[int, float]
    reference_area_p95: float
    reference_area_capacity_quantile: float
    eligible_reference_ids: tuple[str, ...]
    rejected_reference_ids: dict[str, str]
    removable_instance_ids: tuple[str, ...]
    required_removal_cell_classes: tuple[int, ...]
    protected_instance_ids: tuple[str, ...]
    tissue_exclusion_instance_ids: tuple[str, ...]
    protected_instance_reasons: dict[str, str]
    maximum_halo_px: int
    whole_instance_closure_px: int
    continuity_assessment_stage: str
    required_auxiliary_missing: tuple[str, ...]
    required_provenance_missing: tuple[str, ...]
    meaningful_tissue_floor_pixels: int
    aggregate_feasible_tissue_capacity_pixels: int
    feasible_tissue_capacity_by_source_component: dict[str, int]
    meaningful_tissue_capacity_passed: bool
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
            "source_instance_authority_sha256": (
                self.source_instance_authority_sha256
            ),
            "source_instance_authority_count": (
                self.source_instance_authority_count
            ),
            "source_instance_observation_quality": (
                self.source_instance_observation_quality
            ),
            "target_cell_class": self.target_cell_class,
            "target_cell_classes": list(self.target_cell_classes),
            "target_tissue_label": self.target_tissue_label,
            "target_density_per_pixel": self.target_density_per_pixel,
            "target_density_by_class": {
                str(key): value
                for key, value in sorted(self.target_density_by_class.items())
            },
            "reference_area_p95": self.reference_area_p95,
            "reference_area_capacity_quantile": (
                self.reference_area_capacity_quantile
            ),
            "eligible_reference_ids": list(self.eligible_reference_ids),
            "rejected_reference_ids": dict(self.rejected_reference_ids),
            "removable_instance_ids": list(self.removable_instance_ids),
            "required_removal_cell_classes": list(
                self.required_removal_cell_classes
            ),
            "protected_instance_ids": list(self.protected_instance_ids),
            "tissue_exclusion_instance_ids": list(
                self.tissue_exclusion_instance_ids
            ),
            "protected_instance_reasons": dict(self.protected_instance_reasons),
            "maximum_halo_px": self.maximum_halo_px,
            "whole_instance_closure_px": self.whole_instance_closure_px,
            "continuity_assessment_stage": self.continuity_assessment_stage,
            "required_auxiliary_missing": list(self.required_auxiliary_missing),
            "required_provenance_missing": list(self.required_provenance_missing),
            "meaningful_tissue_floor_pixels": self.meaningful_tissue_floor_pixels,
            "aggregate_feasible_tissue_capacity_pixels": (
                self.aggregate_feasible_tissue_capacity_pixels
            ),
            "feasible_tissue_capacity_by_source_component": dict(
                sorted(
                    self.feasible_tissue_capacity_by_source_component.items()
                )
            ),
            "meaningful_tissue_capacity_passed": (
                self.meaningful_tissue_capacity_passed
            ),
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
    required_removal_cell_classes: tuple[int, ...]
    estimated_removal_count: int
    protected_overlap_ids: tuple[str, ...]
    nonlocal_extension_ids: tuple[str, ...]
    legal_core_pixels: int
    reference_fit_center_pixels: int
    required_add_count: int
    required_seam_count: int
    estimated_add_capacity: int
    estimated_seam_capacity: int
    continuity_mode: str
    continuity_width_px: int
    continuity_maximum_empty_run_px: int
    continuity_anchor_pixels: int
    continuity_region_pixels: int
    potential_anchor_coverage_fraction: float
    minimum_anchor_coverage_fraction: float
    meaningful_tissue_floor_pixels: int
    tissue_change_pixels: int
    exact_packing_certificate: dict[str, Any]
    complete_instance_spill_pixels: int
    target_footprint_spill_pixels: int
    predicted_joint_pixels: int
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

    target_label = tissue_bundle.edit_contract.target_label
    target_class = target_cell_class_for_tissue(target_label, schema)
    if target_class not in joint_bundle.mechanism.cell_program.allowed_cell_classes:
        raise JointContractError(
            f"target cell class {target_class} is unavailable to the mechanism"
        )
    target_compatible_classes = set(
        joint_bundle.cell_observation_profile.tissue_compatible_classes.get(
            target_label, ()
        )
    ).intersection(joint_bundle.mechanism.cell_program.allowed_cell_classes)
    target_compatible_classes.add(target_class)
    references_by_class: dict[int, tuple[ReferenceNucleusShape, ...]] = {}
    rejected: dict[str, str] = {}
    eligible_ids: set[str] = set()
    for class_id in sorted(target_compatible_classes):
        current, current_rejected = build_reference_shape_library(
            scene,
            class_id=class_id,
        )
        references_by_class[class_id] = tuple(current)
        eligible_ids.update(item.instance_id for item in current)
        rejected.update(current_rejected)
    for instance_id in eligible_ids:
        rejected.pop(instance_id, None)
    references = tuple(
        item
        for class_id in sorted(references_by_class)
        for item in references_by_class[class_id]
    )
    required_clearance_classes = set(
        joint_bundle.primitive.required_source_clearance_classes
    )
    maximum_halo = int(joint_bundle.mechanism.cell_program.halo_distance_px[1])
    representability = joint_bundle.mechanism.representability
    required_auxiliary = set(
        representability.required_auxiliary_structures
    )
    protected_auxiliary = set(
        representability.protected_auxiliary_structures
    )
    receiving_auxiliary = set(
        representability.receiving_auxiliary_structures
    )
    missing_auxiliary = tuple(
        sorted(required_auxiliary - set(scene.auxiliary_structure_masks))
    )
    protected_structure = np.zeros_like(source_tissue, dtype=bool)
    for structure_id in sorted(protected_auxiliary - set(missing_auxiliary)):
        protected_structure |= np.asarray(
            scene.auxiliary_structure_masks[structure_id], dtype=bool
        )
    receiving_structure = np.ones_like(source_tissue, dtype=bool)
    if receiving_auxiliary:
        receiving_structure = np.zeros_like(source_tissue, dtype=bool)
        for structure_id in sorted(
            receiving_auxiliary - set(missing_auxiliary)
        ):
            receiving_structure |= np.asarray(
                scene.auxiliary_structure_masks[structure_id], dtype=bool
            )
    source_generation_prohibited = np.isin(
        source_tissue,
        joint_bundle.annotation_profile.prohibit_generation_support_fine_ids,
    ) | protected_structure
    removable: list[str] = []
    protected: list[str] = []
    tissue_exclusions: list[str] = []
    protected_reasons: dict[str, str] = {}
    protected_mask = np.zeros_like(source_tissue, dtype=bool)
    for item in scene.cells.instances:
        component = np.asarray(scene.instance_masks[item.instance_id], dtype=bool)
        reason = None
        if item.touches_border:
            reason = "patch_boundary_censored_instance"
        elif ndimage.label(
            component, structure=np.ones((3, 3), dtype=bool)
        )[1] != 1:
            reason = "disconnected_instance"
        elif "merged_suspect" in item.quality_flags:
            reason = "merged_suspect_instance"
        elif "irregular_or_fragmented_shape" in item.quality_flags:
            reason = "irregular_or_fragmented_instance"
        elif np.any(component & source_generation_prohibited):
            reason = "instance_spans_prohibited_generation_region"
        if reason is None:
            removable.append(item.instance_id)
        else:
            protected.append(item.instance_id)
            protected_reasons[item.instance_id] = reason
            # A censored/irregular source shape is never eligible for shape
            # migration and remains pixel-frozen. It need not, however, veto a
            # tissue transition when its observed class is already compatible
            # with the target tissue and the primitive does not require that
            # class to be cleared. This distinction prevents border-shape
            # filtering from deleting a large, otherwise executable T region.
            if (
                item.class_id in required_clearance_classes
                or item.class_id not in target_compatible_classes
            ):
                tissue_exclusions.append(item.instance_id)
                protected_mask |= component

    if protected:
        protected_set = set(protected)
        references = tuple(
            item
            for item in references
            if item.instance_id not in protected_set
        )
        references_by_class = {
            class_id: tuple(
                item
                for item in items
                if item.instance_id not in protected_set
            )
            for class_id, items in references_by_class.items()
        }
        for instance_id in protected:
            rejected.setdefault(
                instance_id,
                protected_reasons[instance_id],
            )

    missing_provenance = tuple(
        field
        for field in joint_bundle.annotation_profile.required_provenance_fields
        if not _known_provenance(case.provenance.get(field))
    )
    prohibited_tissue = np.isin(
        source_tissue,
        joint_bundle.annotation_profile.prohibit_cell_placement_fine_ids,
    ) | protected_structure
    source_contract = joint_bundle.mechanism.tissue_program.primitive_label_contracts.get(
        case.primitive_id
    )
    if source_contract is None:
        raise JointContractError("joint mechanism has no primitive label contract")
    allowed_sources = set(tissue_bundle.edit_contract.source_label_options).intersection(
        source_contract["source_labels"]
    )
    render_owned_clearance = (
        case.primitive_id
        in joint_bundle.mechanism.cell_program.render_owned_clearance_primitives
    )
    seam_contract = joint_bundle.mechanism.cell_program.seam_for(
        case.primitive_id
    )
    requires_add = not render_owned_clearance
    required_removal_classes = tuple(sorted(required_clearance_classes))
    removable_set = set(removable)
    capacity_quantile = seam_contract.reference_area_quantiles[1]
    reference_area_p95 = _reference_area_p95(references)
    reference_area_capacity = _reference_area_at_quantile(
        references,
        capacity_quantile,
    )
    whole_instance_closure = _whole_instance_closure_px(
        scene,
        removable,
    )
    source_authority = build_scene_instance_authority(
        scene, scene.source_nuclei
    )
    editable_source_ids = (
        joint_bundle.annotation_profile.mechanism_editable_source_fine_ids.get(
            f"{joint_bundle.mechanism.mechanism_id}::{case.primitive_id}",
            joint_bundle.annotation_profile.mechanism_editable_source_fine_ids.get(
                joint_bundle.mechanism.mechanism_id, ()
            ),
        )
    )
    target_density, target_density_by_class = (
        _target_interface_population_density(
            scene,
            source_tissue=source_tissue,
            target_classes=tuple(sorted(target_compatible_classes)),
            target_label=target_label,
            schema=schema,
            reference_area_p95=reference_area_p95,
        )
        if requires_add
        else (0.0, {})
    )
    instance_class = {
        item.instance_id: item.class_id for item in scene.cells.instances
    }
    interface_reports: list[InterfaceNucleiCapacity] = []
    feasible_envelopes: list[tuple[str, np.ndarray, int]] = []
    topology_fallback = joint_bundle.mechanism.tissue_program.topology_fallback_for(
        case.primitive_id
    )
    maximum_source_fraction = float(
        max(
            joint_bundle.primitive.maximum_source_component_changed_fraction,
            (
                topology_fallback.maximum_source_component_changed_fraction
                if topology_fallback is not None
                else 0.0
            ),
        )
    )
    effective_allow_source_resolution = bool(
        joint_bundle.primitive.allow_source_component_resolution
        or (
            topology_fallback is not None
            and topology_fallback.allow_source_component_resolution
        )
    )
    minimum_source_remaining = int(
        min(
            joint_bundle.primitive.minimum_source_component_remaining_px,
            (
                topology_fallback.minimum_source_component_remaining_px
                if topology_fallback is not None
                else joint_bundle.primitive.minimum_source_component_remaining_px
            ),
        )
    )
    retained_target_centers = class_center_mask(
        scene.source_nuclei,
        class_id=target_class,
    )
    for interface in scene.tissue.graph.interfaces:
        if (
            interface.source_label not in allowed_sources
            or interface.target_label != target_label
        ):
            continue
        source_component_metadata = next(
            (
                item
                for item in scene.tissue.graph.components
                if item.component_id == interface.source_component_id
            ),
            None,
        )
        if (
            editable_source_ids
            and source_component_metadata is not None
            and not set(source_component_metadata.fine_ids).issubset(
                editable_source_ids
            )
        ):
            continue
        source_component = scene.tissue.component_masks[interface.source_component_id]
        interface_mask = scene.tissue.interface_masks[interface.interface_id]
        boundary_topology = classify_tumor_stroma_boundary(
            scene=scene,
            interface=interface,
            allowed_host_labels=tuple(
                sorted(
                    (
                        set(source_contract["source_labels"])
                        | set(source_contract["target_labels"])
                    )
                    - {"Tumor"}
                )
            ),
        )
        # Capacity must use the same geometric upper bound as the downstream
        # depth/span gate.  The former 0.80*contact heuristic was only a shape
        # preference, yet it was treated as a hard capacity ceiling; short
        # segments of a broad, multi-front edit were therefore declared full
        # long before the executable topology solver ran.  Natural taper is
        # still compiled by the tool program, while the skill-owned ratio is
        # the audited hard depth/span maximum shared with the mask gate.
        front_contract = joint_bundle.mechanism.tissue_program.front
        if effective_allow_source_resolution:
            # Resolution may consume any safe part of a source compartment,
            # including its final pixels. Its executable depth is therefore
            # the observed component depth, not a generic front-band cap.
            distance = ndimage.distance_transform_edt(~interface_mask)
            observed_depth = distance[source_component & ~prohibited_tissue]
            depth_cap = max(
                1,
                int(np.ceil(float(observed_depth.max(initial=0.0)))),
            )
        elif (
            joint_bundle.primitive.tissue_geometry_mode
            == "component_boundary_turnover"
        ):
            # Closed intratumoral compartments are edited from their complete
            # boundary. A single raster segment's contact length is not the
            # biological span and must not cap radial/component turnover.
            depth_cap = int(front_contract.maximum_band_px)
        else:
            depth_cap = max(
                1,
                min(
                    front_contract.maximum_band_px,
                    int(
                        np.floor(
                            interface.contact_pixels
                            * front_contract.maximum_depth_span_ratio
                        )
                    ),
                ),
            )
        distance = ndimage.distance_transform_edt(~interface_mask)
        raw_envelope = source_component & ~prohibited_tissue & ~protected_mask
        raw_envelope &= receiving_structure
        if editable_source_ids:
            raw_envelope &= np.isin(source_tissue, editable_source_ids)
        if not effective_allow_source_resolution:
            raw_envelope &= distance <= depth_cap
        authority_key = (
            f"{joint_bundle.mechanism.mechanism_id}::{case.primitive_id}"
        )
        transition_authorized_ids = set(
            joint_bundle.annotation_profile.mechanism_editable_source_fine_ids.get(
                authority_key,
                joint_bundle.annotation_profile.mechanism_editable_source_fine_ids.get(
                    joint_bundle.mechanism.mechanism_id, ()
                ),
            )
        )
        protected_fine_ids_in_band = tuple(
            sorted(
                int(value)
                for value in np.unique(source_tissue[raw_envelope])
                if int(value)
                in set(joint_bundle.annotation_profile.protected_fine_ids)
                and int(value) not in transition_authorized_ids
            )
        )
        source_component_pixels = int(np.count_nonzero(source_component))
        if effective_allow_source_resolution:
            source_component_capacity = source_component_pixels
        else:
            source_component_capacity = min(
                int(
                    np.floor(
                        source_component_pixels * maximum_source_fraction
                    )
                ),
                max(0, source_component_pixels - minimum_source_remaining),
            )
        anchor_continuity_reports: list[dict[str, Any]] = []
        feasible_anchor_ids: list[str] = []
        feasible_anchor_envelope = np.zeros_like(raw_envelope, dtype=bool)
        for anchor_id in interface.anchor_segment_ids:
            anchor_mask = scene.tissue.anchor_masks.get(anchor_id)
            if anchor_mask is None:
                continue
            anchor_distance = ndimage.distance_transform_edt(~anchor_mask)
            anchor_envelope = raw_envelope & (anchor_distance <= depth_cap)
            anchor_overlapping_removable = tuple(
                sorted(
                    instance_id
                    for instance_id in removable_set
                    if np.any(scene.instance_masks[instance_id] & anchor_envelope)
                )
            )
            anchor_free = _free_after_removing_instances(
                anchor_envelope,
                source_nuclei=scene.source_nuclei,
                scene=scene,
                removable_ids=anchor_overlapping_removable,
            )
            anchor_fit_centers = _reference_fit_centers_union(
                anchor_free,
                _representative_fit_references(references),
            )
            adaptive_seam = compile_adaptive_seam(
                scene=scene,
                tissue_change=anchor_envelope,
                interface_ids=(interface.interface_id,),
                anchor_ids=(anchor_id,),
                target_class=target_class,
                contract=seam_contract,
            )
            seam_fit = anchor_fit_centers & adaptive_seam.continuity_region
            potential_centers = seam_fit | (
                retained_target_centers & ~anchor_envelope
            )
            coverage = anchor_coverage_fraction(
                adaptive_seam.anchor_mask,
                potential_centers,
                maximum_empty_run_px=adaptive_seam.maximum_empty_run_px,
            )
            anchor_feasible = bool(
                np.any(anchor_envelope)
                and (
                    not adaptive_seam.requires_new_target_cells
                    or (
                        np.any(seam_fit)
                        and coverage
                        >= adaptive_seam.minimum_anchor_coverage_fraction
                    )
                )
            )
            anchor_continuity_reports.append(
                {
                    "anchor_segment_id": anchor_id,
                    "editable_tissue_capacity_pixels": int(
                        np.count_nonzero(anchor_envelope)
                    ),
                    "reference_fit_center_pixels": int(
                        np.count_nonzero(anchor_fit_centers)
                    ),
                    "seam_fit_center_pixels": int(np.count_nonzero(seam_fit)),
                    "potential_anchor_coverage_fraction": float(coverage),
                    "minimum_anchor_coverage_fraction": float(
                        adaptive_seam.minimum_anchor_coverage_fraction
                    ),
                    "feasible": anchor_feasible,
                }
            )
            if anchor_feasible:
                feasible_anchor_ids.append(anchor_id)
                feasible_anchor_envelope |= anchor_envelope
        envelope = (
            feasible_anchor_envelope
            if seam_contract.requires_new_target_cells
            else raw_envelope
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
        removal_targets = tuple(
            instance_id
            for instance_id in overlapping_removable
            if instance_class.get(instance_id) in set(required_removal_classes)
        )
        free = _free_after_removing_instances(
            envelope,
            source_nuclei=scene.source_nuclei,
            scene=scene,
            removable_ids=overlapping_removable,
        )
        fit_centers = _reference_fit_centers_union(
            free,
            _representative_fit_references(references),
        )
        halo = np.zeros_like(envelope)
        if joint_bundle.mechanism.coupling.cell_only_target_fraction > 0:
            halo = (
                ndimage.binary_dilation(interface_mask, iterations=maximum_halo)
                & ~prohibited_tissue
                & ~protected_mask
                & ~envelope
            )
        executable_envelope_pixels = min(
            int(np.count_nonzero(envelope)),
            int(source_component_capacity),
        )
        interface_target_pixels = min(
            int(allocation.tissue_target_pixels),
            executable_envelope_pixels,
        )
        required_count = (
            max(1, int(np.ceil(interface_target_pixels * target_density)))
            if requires_add
            else 0
        )
        reserved_halo_count = int(
            np.ceil(
                allocation.reserved_layout_halo_pixels
                / max(1.0, reference_area_capacity)
            )
        )
        if joint_bundle.mechanism.coupling.cell_only_target_fraction > 0:
            required_count += reserved_halo_count
        erased = np.zeros_like(envelope, dtype=bool)
        for instance_id in overlapping_removable:
            erased |= np.asarray(scene.instance_masks[instance_id], dtype=bool)
        target_ids = tuple(schema.resolve_fine_ids(target_label))
        hypothetical_valid = np.isin(source_tissue, target_ids) | envelope
        packing = certify_complete_footprint_packing(
            source_nuclei=scene.source_nuclei,
            erased_footprint=erased,
            center_region=envelope,
            valid_footprint_region=hypothetical_valid,
            references_by_class=references_by_class,
            requested_count=required_count,
            class_request_weights=target_density_by_class,
        )
        add_capacity = packing.placed_count
        # Planner has not selected an anchor and no concrete tissue candidate
        # exists yet.  Any exact seam quota here would be invented geometry.
        # Continuity is therefore deferred to candidate-local compilation.
        seam_capacity = 0
        required_seam = 0
        reasons: list[str] = []
        if requires_add and not references:
            reasons.append("no_complete_same_class_reference_shape")
        if missing_auxiliary:
            reasons.append("required_auxiliary_missing")
        if missing_provenance:
            reasons.append("required_profile_provenance_missing")
        if not np.any(envelope):
            reasons.append("no_cell_safe_tissue_capacity")
        if (
            "external_boundary_binding"
            in joint_bundle.mechanism.planner_policy.hard_constraint_checker_ids
            and not boundary_topology["external_tumor_stroma_boundary"]
        ):
            reasons.append("interface_is_not_external_tumor_stroma_boundary")
        if protected_fine_ids_in_band:
            reasons.append("protected_fine_id_in_candidate_band")
        if executable_envelope_pixels <= 0:
            reasons.append("no_topology_safe_source_component_capacity")
        if (
            seam_contract.requires_new_target_cells
            and not feasible_anchor_ids
        ):
            reasons.append("no_cell_feasible_anchor_segment")
        if requires_add and not packing.passed:
            reasons.append("insufficient_complete_shape_placement_capacity")
        if (
            joint_bundle.primitive.minimum_source_clearance_instances > 0
            and len(removal_targets)
            < joint_bundle.primitive.minimum_source_clearance_instances
        ):
            reasons.append("no_complete_viable_instance_for_render_clearance")
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
                boundary_topology=str(boundary_topology["classification"]),
                external_tumor_stroma_boundary=bool(
                    boundary_topology["external_tumor_stroma_boundary"]
                ),
                internal_enclosed_interface=bool(
                    boundary_topology["internal_enclosed_interface"]
                ),
                protected_fine_ids_within_band=protected_fine_ids_in_band,
                contact_pixels=int(interface.contact_pixels),
                gate_bounded_depth_px=depth_cap,
                source_component_capacity_pixels=int(
                    source_component_capacity
                ),
                editable_tissue_capacity_pixels=executable_envelope_pixels,
                removable_instance_ids=overlapping_removable,
                required_removal_cell_classes=required_removal_classes,
                estimated_removal_capacity=len(removal_targets),
                protected_instance_overlap_ids=overlapping_protected,
                legal_halo_pixels=int(np.count_nonzero(halo)),
                reference_fit_center_pixels=int(np.count_nonzero(fit_centers)),
                reference_area_p95=float(reference_area_p95),
                target_density_per_pixel=float(target_density),
                required_add_count=int(packing.requested_count),
                required_seam_count=int(required_seam),
                estimated_add_capacity=max(0, int(add_capacity)),
                estimated_seam_capacity=max(0, int(seam_capacity)),
                continuity_assessment_stage=(
                    "anchor_preflight_then_candidate_exact"
                ),
                cell_feasible_anchor_segment_ids=tuple(feasible_anchor_ids),
                anchor_continuity_reports=tuple(anchor_continuity_reports),
                capacity_margin_count=int(
                    add_capacity - packing.requested_count
                ),
                exact_packing_certificate=packing.to_metadata(),
                feasible=not reasons,
                reasons=tuple(reasons),
            )
        )
        if not reasons:
            feasible_envelopes.append(
                (
                    interface.source_component_id,
                    np.asarray(envelope, dtype=bool),
                    int(source_component_capacity),
                )
            )
    aggregate_by_component: dict[str, np.ndarray] = {}
    capacity_by_component: dict[str, int] = {}
    for component_id, envelope, component_capacity in feasible_envelopes:
        aggregate_by_component.setdefault(
            component_id, np.zeros_like(source_tissue, dtype=bool)
        )
        aggregate_by_component[component_id] |= envelope
        capacity_by_component[component_id] = component_capacity
    aggregate_capacity = sum(
        min(
            int(np.count_nonzero(envelope)),
            capacity_by_component[component_id],
        )
        for component_id, envelope in aggregate_by_component.items()
    )
    # The manifest owns both floors. Capacity-adaptive burden tasks compile
    # against their explicit minimum-effective floor (3% in the fixed breast
    # tumor-burden cases); the ordinary 4% contribution floor remains binding
    # only for strict tasks. Downstream gates still require proof that a
    # below-standard result is the maximum topology-safe fallback.
    meaningful_floor = int(allocation.tissue_execution_floor_pixels)
    return JointNucleiPreflight(
        version=PREFLIGHT_VERSION,
        source_instance_authority_sha256=str(
            source_authority["authority_sha256"]
        ),
        source_instance_authority_count=len(source_authority["instances"]),
        source_instance_observation_quality=str(
            source_authority["observation_quality"]
        ),
        target_cell_class=target_class,
        target_cell_classes=tuple(sorted(target_compatible_classes)),
        target_tissue_label=target_label,
        target_density_per_pixel=float(target_density),
        target_density_by_class={
            int(key): float(value)
            for key, value in target_density_by_class.items()
        },
        reference_area_p95=float(reference_area_p95),
        reference_area_capacity_quantile=float(capacity_quantile),
        eligible_reference_ids=tuple(item.instance_id for item in references),
        rejected_reference_ids=dict(rejected),
        removable_instance_ids=tuple(sorted(removable)),
        required_removal_cell_classes=required_removal_classes,
        protected_instance_ids=tuple(sorted(protected)),
        tissue_exclusion_instance_ids=tuple(sorted(tissue_exclusions)),
        protected_instance_reasons=protected_reasons,
        maximum_halo_px=maximum_halo,
        whole_instance_closure_px=whole_instance_closure,
        continuity_assessment_stage=(
            "anchor_preflight_then_candidate_exact"
        ),
        required_auxiliary_missing=missing_auxiliary,
        required_provenance_missing=missing_provenance,
        meaningful_tissue_floor_pixels=meaningful_floor,
        aggregate_feasible_tissue_capacity_pixels=aggregate_capacity,
        feasible_tissue_capacity_by_source_component={
            component_id: min(
                int(np.count_nonzero(envelope)),
                capacity_by_component[component_id],
            )
            for component_id, envelope in aggregate_by_component.items()
        },
        meaningful_tissue_capacity_passed=(
            aggregate_capacity >= meaningful_floor
        ),
        interfaces=tuple(interface_reports),
        protected_tissue_change_mask=protected_mask,
    )


def classify_tumor_stroma_boundary(
    *, scene, interface, allowed_host_labels: tuple[str, ...] = ("Stroma",)
) -> dict[str, Any]:
    """Classify an external Tumor/authorized-host boundary from masks only.

    The historical name remains for artifact compatibility.  Profiles such as
    ORCA encode the legal receiving compartment as ``Other tissue`` rather
    than ``Stroma``; the active mechanism label contract supplies that closed
    host set and no histologic subtype is inferred.
    """

    labels = {interface.source_label, interface.target_label}
    host_labels = labels - {"Tumor"}
    if (
        "Tumor" not in labels
        or len(host_labels) != 1
        or not host_labels.issubset(set(allowed_host_labels))
    ):
        return {
            "classification": "not_tumor_stroma",
            "external_tumor_stroma_boundary": False,
            "internal_enclosed_interface": False,
        }
    tumor_component_id = (
        interface.source_component_id
        if interface.source_label == "Tumor"
        else interface.target_component_id
    )
    host_label = next(iter(host_labels))
    stroma_component_id = (
        interface.source_component_id
        if interface.source_label == host_label
        else interface.target_component_id
    )
    tumor = np.asarray(scene.tissue.component_masks[tumor_component_id], dtype=bool)
    stroma = np.asarray(scene.tissue.component_masks[stroma_component_id], dtype=bool)
    exterior = _border_connected_complement(tumor)
    contact_host = stroma & ndimage.binary_dilation(tumor, iterations=1)
    external = bool(np.any(contact_host & exterior))
    enclosed = bool(np.any(contact_host) and not external)
    return {
        "classification": (
            "external_tumor_stroma_boundary"
            if external
            else "internal_enclosed_interface"
        ),
        "external_tumor_stroma_boundary": external,
        "internal_enclosed_interface": enclosed,
    }


def _border_connected_complement(component: np.ndarray) -> np.ndarray:
    complement = ~np.asarray(component, dtype=bool)
    seeds = np.zeros_like(complement, dtype=bool)
    seeds[0, :] = complement[0, :]
    seeds[-1, :] = complement[-1, :]
    seeds[:, 0] |= complement[:, 0]
    seeds[:, -1] |= complement[:, -1]
    return ndimage.binary_propagation(
        seeds,
        mask=complement,
        structure=np.ones((3, 3), dtype=bool),
    )


def augment_tissue_scene_with_nuclei_preflight(
    scene: SceneAnalysis,
    preflight: JointNucleiPreflight,
    *,
    auxiliary_structure_masks: dict[str, np.ndarray] | None = None,
    required_auxiliary_structure_ids: tuple[str, ...] = (),
    receiving_auxiliary_structure_ids: tuple[str, ...] = (),
) -> SceneAnalysis:
    """Make cell and native-structure exclusions unavailable to tissue tools.

    Auxiliary protection is compiled before candidate drawing.  A required
    gland/lumen/airspace map therefore constrains the same legal raster used
    by the topology solver instead of merely vetoing an otherwise finished
    joint candidate.
    """

    prohibited = dict(scene.prohibited_region_masks)
    prohibited["joint:nuclei:protected_instances"] = np.asarray(
        preflight.protected_tissue_change_mask,
        dtype=bool,
    )
    available = auxiliary_structure_masks or {}
    receiving_ids = set(receiving_auxiliary_structure_ids)
    required_ids = set(required_auxiliary_structure_ids) | receiving_ids
    for structure_id in sorted(required_ids):
        if structure_id not in available:
            raise JointContractError(
                f"required auxiliary structure {structure_id!r} is unavailable"
            )
        structure = np.asarray(available[structure_id], dtype=bool)
        # A protected auxiliary is itself forbidden. A receiving auxiliary is
        # the opposite contract: the edit must remain *inside* it, so its
        # complement is forbidden. Local invasive clearance uses the latter
        # for the explicit user ROI. Treating receiving maps as a late-only
        # joint gate allowed the topology solver to draw a few tissue pixels
        # outside the ROI and then reject an otherwise executable candidate.
        prohibited[f"joint:auxiliary:{structure_id}"] = (
            ~structure if structure_id in receiving_ids else structure
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
    case: JointCaseContext,
    source_tissue: np.ndarray,
    scene: JointSceneAnalysis,
    preflight: JointNucleiPreflight,
    joint_bundle: JointSkillBundle,
    joint_plan: JointEditPlan,
    allocation: JointBudgetAllocation,
) -> CandidateCellFeasibility:
    """Certify candidate area, closure, seam and real-shape packing pre-ProbNet."""

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
        item for item in intersecting if item in removable_set
    )
    metadata = {item.instance_id: item for item in scene.cells.instances}
    removal_targets = tuple(
        item
        for item in removals
        if metadata[item].class_id in set(preflight.required_removal_cell_classes)
    )
    allowed_closure = ndimage.binary_dilation(
        core,
        iterations=max(1, preflight.whole_instance_closure_px),
    )
    nonlocal_extensions = tuple(
        sorted(
            item
            for item in removals
            if np.any(scene.instance_masks[item] & ~allowed_closure)
        )
    )
    prohibited_ids = joint_bundle.annotation_profile.prohibit_cell_placement_fine_ids
    legal_core = core & ~np.isin(candidate.target_mask, prohibited_ids)
    target_fine_ids = tuple(
        int(value)
        for value in np.unique(candidate.target_mask[legal_core])
    )
    target_fine_set = set(target_fine_ids)
    local_references_by_class: dict[int, list[ReferenceNucleusShape]] = {}
    fallback_references_by_class: dict[int, list[ReferenceNucleusShape]] = {}
    for instance_id in preflight.eligible_reference_ids:
        nucleus = metadata.get(instance_id)
        if nucleus is None:
            continue
        item = _reference_from_scene(scene, instance_id, nucleus.class_id)
        if item is not None:
            fallback_references_by_class.setdefault(
                nucleus.class_id, []
            ).append(item)
            if nucleus.tissue_fine_id in target_fine_set:
                local_references_by_class.setdefault(
                    nucleus.class_id, []
                ).append(item)
    reference_groups = {
        class_id: tuple(
            local_references_by_class.get(class_id) or fallback_items
        )
        for class_id, fallback_items in fallback_references_by_class.items()
    }
    references = tuple(item for items in reference_groups.values() for item in items)
    fit_references = _representative_fit_references(references)
    free = _free_after_removing_instances(
        legal_core,
        source_nuclei=scene.source_nuclei,
        scene=scene,
        removable_ids=removals,
    )
    fit_centers = _reference_fit_centers_union(free, fit_references)
    required_count = int(
        np.ceil(np.count_nonzero(legal_core) * preflight.target_density_per_pixel)
    )
    if (
        "add" in joint_bundle.mechanism.cell_program.actions
        and np.any(legal_core)
    ):
        # A nonempty target-population program cannot be certified by a
        # vacuous zero-placement witness.  Sparse finite fields still require
        # one complete, source-matched target nucleus; otherwise the tissue
        # candidate is rejected before ProbNet rather than crashing while the
        # immutable packing ledger is bound.
        required_count = max(1, required_count)
    adaptive_seam = compile_adaptive_seam(
        scene=scene,
        tissue_change=core,
        interface_ids=joint_plan.cell_plan.interface_ids,
        anchor_ids=joint_plan.cell_plan.anchor_ids,
        target_class=preflight.target_cell_class,
        contract=joint_bundle.mechanism.cell_program.seam_for(
            case.primitive_id
        ),
    )
    erased = np.zeros_like(core, dtype=bool)
    for instance_id in removals:
        erased |= np.asarray(scene.instance_masks[instance_id], dtype=bool)
    retained_nuclei = np.asarray(scene.source_nuclei).copy()
    retained_nuclei[erased] = 0
    seam_fit = fit_centers & adaptive_seam.continuity_region
    retained_target_centers = class_center_mask(
        retained_nuclei,
        class_id=preflight.target_cell_class,
    )
    potential_continuity_centers = seam_fit | retained_target_centers
    potential_coverage = anchor_coverage_fraction(
        adaptive_seam.anchor_mask,
        potential_continuity_centers,
        maximum_empty_run_px=adaptive_seam.maximum_empty_run_px,
    )
    seam_capacity = int(np.count_nonzero(seam_fit))
    required_seam = 0
    minimum_seam = 0
    if adaptive_seam.requires_new_target_cells and target_fine_ids:
        quota = compile_continuity_center_quota(
            nuclei_mask=retained_nuclei,
            target_tissue_mask=candidate.target_mask,
            tissue_change=core,
            continuity_region=adaptive_seam.continuity_region,
            continuity_anchor_mask=adaptive_seam.anchor_mask,
            continuity_width_px=adaptive_seam.width_px,
            density_ratio_range=adaptive_seam.density_ratio_range,
            requires_new_target_cells=True,
            target_class=preflight.target_cell_class,
            target_fine_ids=target_fine_ids,
        )
        required_seam = compile_executable_continuity_count(
            quota,
            anchor_pixels=int(np.count_nonzero(adaptive_seam.anchor_mask)),
            maximum_empty_run_px=adaptive_seam.maximum_empty_run_px,
            minimum_anchor_coverage_fraction=(
                adaptive_seam.minimum_anchor_coverage_fraction
            ),
        )
        minimum_seam = compile_minimum_continuity_count(
            quota,
            anchor_pixels=int(np.count_nonzero(adaptive_seam.anchor_mask)),
            maximum_empty_run_px=adaptive_seam.maximum_empty_run_px,
            minimum_anchor_coverage_fraction=(
                adaptive_seam.minimum_anchor_coverage_fraction
            ),
        )
    effective_required_count = max(required_count, required_seam)
    packing = certify_complete_footprint_packing(
        source_nuclei=scene.source_nuclei,
        erased_footprint=erased,
        center_region=legal_core,
        valid_footprint_region=(
            np.isin(candidate.target_mask, target_fine_ids)
            if target_fine_ids
            else np.zeros_like(core, dtype=bool)
        ),
        references_by_class=reference_groups,
        requested_count=effective_required_count,
        class_request_weights=preflight.target_density_by_class,
        continuity_region=adaptive_seam.continuity_region,
        continuity_anchor_mask=adaptive_seam.anchor_mask,
        preexisting_continuity_centers=retained_target_centers,
        continuity_maximum_empty_run_px=(
            adaptive_seam.maximum_empty_run_px
        ),
        required_seam_count=required_seam,
        minimum_seam_count=minimum_seam,
        required_seam_class=preflight.target_cell_class,
    )
    meaningful_floor = int(allocation.tissue_execution_floor_pixels)
    reasons: list[str] = []
    if np.count_nonzero(core) < meaningful_floor:
        reasons.append("candidate_below_meaningful_tissue_floor")
    if protected_overlap:
        reasons.append("tissue_change_intersects_protected_instance")
    if set(intersecting) - removable_set - protected_set:
        reasons.append("tissue_change_intersects_unclassified_instance")
    if nonlocal_extensions:
        reasons.append("whole_instance_removal_exceeds_authorized_halo")
    if (
        joint_bundle.primitive.minimum_source_clearance_instances > 0
        and len(removal_targets)
        < joint_bundle.primitive.minimum_source_clearance_instances
    ):
        reasons.append("candidate_has_no_complete_viable_instance_to_clear")
    if not np.any(legal_core):
        reasons.append("candidate_has_no_legal_cell_core")
    if "add" in joint_bundle.mechanism.cell_program.actions and not packing.passed:
        reasons.extend(packing.failure_reasons)
    if adaptive_seam.requires_new_target_cells:
        if not np.any(adaptive_seam.anchor_mask):
            reasons.append("candidate_has_no_active_planner_anchor")
        elif not np.any(adaptive_seam.continuity_region):
            reasons.append("candidate_has_no_anchor_conditioned_continuity_region")
        elif seam_capacity <= 0:
            reasons.append("candidate_cannot_fit_cell_in_continuity_region")
        elif potential_coverage < adaptive_seam.minimum_anchor_coverage_fraction:
            reasons.append("candidate_anchor_continuity_coverage_below_contract")
    return CandidateCellFeasibility(
        candidate_id=candidate.candidate_id,
        passed=not reasons,
        removable_instance_ids=removals,
        required_removal_cell_classes=preflight.required_removal_cell_classes,
        estimated_removal_count=len(removal_targets),
        protected_overlap_ids=protected_overlap,
        nonlocal_extension_ids=nonlocal_extensions,
        legal_core_pixels=int(np.count_nonzero(legal_core)),
        reference_fit_center_pixels=int(np.count_nonzero(fit_centers)),
        required_add_count=int(packing.requested_count),
        required_seam_count=int(packing.required_seam_count),
        estimated_add_capacity=max(0, int(packing.placed_count)),
        estimated_seam_capacity=max(0, int(packing.placed_seam_count)),
        continuity_mode=adaptive_seam.mode,
        continuity_width_px=adaptive_seam.width_px,
        continuity_maximum_empty_run_px=adaptive_seam.maximum_empty_run_px,
        continuity_anchor_pixels=int(np.count_nonzero(adaptive_seam.anchor_mask)),
        continuity_region_pixels=int(np.count_nonzero(adaptive_seam.continuity_region)),
        potential_anchor_coverage_fraction=float(potential_coverage),
        minimum_anchor_coverage_fraction=float(
            adaptive_seam.minimum_anchor_coverage_fraction
        ),
        meaningful_tissue_floor_pixels=meaningful_floor,
        tissue_change_pixels=int(np.count_nonzero(core)),
        exact_packing_certificate=packing.to_metadata(),
        complete_instance_spill_pixels=int(
            np.count_nonzero(erased & ~core)
        ),
        target_footprint_spill_pixels=int(
            np.count_nonzero(packing.footprint_union & ~core & ~erased)
        ),
        predicted_joint_pixels=int(
            np.count_nonzero(core | erased | packing.footprint_union)
        ),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def certify_compiled_cell_program_feasibility(
    report: CandidateCellFeasibility,
    *,
    candidate: CandidateMask,
    contract,
    scene: JointSceneAnalysis,
    preflight: JointNucleiPreflight,
    authoritative_references_by_class: dict[
        int, tuple[ReferenceNucleusShape, ...]
    ]
    | None = None,
    minimum_acceptable_add_count: int | None = None,
) -> CandidateCellFeasibility:
    """Re-certify packing on the compiler's exact P/V/E/C masks.

    The first candidate check runs before the immutable executable contract is
    available.  This second check is still pre-ProbNet, but now uses exactly
    the center, footprint, erasure and continuity rasters consumed by the
    mature executor.  A tissue candidate is never exposed as certified if the
    two stages disagree.
    """

    if authoritative_references_by_class is not None:
        references_by_class = {
            int(class_id): tuple(items)
            for class_id, items in authoritative_references_by_class.items()
            if items
            and int(class_id) in set(contract.allowed_new_cell_classes)
        }
    else:
        metadata = {item.instance_id: item for item in scene.cells.instances}
        target_fine_ids = set(contract.target_host_fine_ids)
        local_references_by_class: dict[
            int, list[ReferenceNucleusShape]
        ] = {}
        fallback_references_by_class: dict[
            int, list[ReferenceNucleusShape]
        ] = {}
        for instance_id in preflight.eligible_reference_ids:
            item = metadata.get(instance_id)
            if (
                item is None
                or item.class_id
                not in set(contract.allowed_new_cell_classes)
            ):
                continue
            reference = _reference_from_scene(
                scene, instance_id, item.class_id
            )
            if reference is not None:
                fallback_references_by_class.setdefault(
                    item.class_id, []
                ).append(reference)
                if item.tissue_fine_id in target_fine_ids:
                    local_references_by_class.setdefault(
                        item.class_id, []
                    ).append(reference)
        references_by_class = {
            class_id: tuple(
                local_references_by_class.get(class_id) or fallback_items
            )
            for class_id, fallback_items in fallback_references_by_class.items()
        }
    program = contract.cell_program
    required_seam_count = 0
    minimum_seam_count = 0
    target_class = preflight.target_cell_class
    if (
        program.continuity_requires_new_target_cells
        and np.any(program.continuity_region)
    ):
        retained_for_quota = np.asarray(scene.source_nuclei).copy()
        retained_for_quota[
            np.asarray(program.erasure_region, dtype=bool)
        ] = 0
        quota = compile_continuity_center_quota(
            nuclei_mask=retained_for_quota,
            target_tissue_mask=np.asarray(candidate.target_mask),
            tissue_change=np.asarray(candidate.change_region, dtype=bool),
            continuity_region=program.continuity_region,
            continuity_anchor_mask=program.continuity_anchor_mask,
            continuity_width_px=program.continuity_width_px,
            density_ratio_range=program.continuity_density_ratio_range,
            requires_new_target_cells=True,
            target_class=target_class,
            target_fine_ids=contract.target_host_fine_ids,
        )
        required_seam_count = compile_executable_continuity_count(
            quota,
            anchor_pixels=int(
                np.count_nonzero(program.continuity_anchor_mask)
            ),
            maximum_empty_run_px=program.continuity_maximum_empty_run_px,
            minimum_anchor_coverage_fraction=(
                program.continuity_minimum_anchor_coverage_fraction
            ),
        )
        minimum_seam_count = compile_minimum_continuity_count(
            quota,
            anchor_pixels=int(
                np.count_nonzero(program.continuity_anchor_mask)
            ),
            maximum_empty_run_px=program.continuity_maximum_empty_run_px,
            minimum_anchor_coverage_fraction=(
                program.continuity_minimum_anchor_coverage_fraction
            ),
        )
    prior_certificate = report.exact_packing_certificate or {}
    prior_fallback_used = bool(
        prior_certificate.get("finite_count_fallback_used", False)
    )
    retained_centers = class_center_mask(
        np.where(
            np.asarray(program.erasure_region, dtype=bool),
            0,
            np.asarray(scene.source_nuclei),
        ),
        class_id=target_class,
    )
    certificate = certify_complete_footprint_packing(
        source_nuclei=scene.source_nuclei,
        erased_footprint=program.erasure_region,
        center_region=program.placement_center_region,
        valid_footprint_region=program.valid_footprint_region,
        references_by_class=references_by_class,
        requested_count=report.required_add_count,
        class_request_weights=preflight.target_density_by_class,
        continuity_region=program.continuity_region,
        continuity_anchor_mask=program.continuity_anchor_mask,
        preexisting_continuity_centers=retained_centers,
        continuity_maximum_empty_run_px=(
            program.continuity_maximum_empty_run_px
        ),
        required_seam_count=required_seam_count,
        minimum_seam_count=minimum_seam_count,
        required_seam_class=target_class,
        minimum_acceptable_count=minimum_acceptable_add_count,
        minimum_center_separation_px=(
            independent_focus_minimum_center_separation_px(
                contract.primitive_id,
                program.nominal_nucleus_diameter_px,
            )
        ),
        # Exactly one bounded fallback is allowed across the two feasibility
        # stages. A broad candidate core may fit the nominal count while the
        # final E/P/V/C compiler exposes the true lower capacity; in that case
        # the final stage owns the fallback. If candidate assessment already
        # used it, the exact recheck cannot reduce the count again.
        allow_finite_count_fallback=not prior_fallback_used,
    )
    if prior_fallback_used:
        certificate = replace(
            certificate,
            nominal_requested_count=int(
                prior_certificate.get(
                    "nominal_requested_count", certificate.requested_count
                )
            ),
            minimum_safe_count=int(
                prior_certificate.get(
                    "minimum_safe_count", certificate.requested_count
                )
            ),
            finite_count_fallback_used=True,
        )
    reasons = [
        item
        for item in report.reasons
        if item
        not in {
            "exact_complete_footprint_packing_capacity_shortfall",
            "exact_seam_packing_capacity_shortfall",
            "no_complete_reference_shape_for_packing",
        }
    ]
    if not certificate.passed:
        reasons.extend(certificate.failure_reasons)
    if (
        program.continuity_requires_new_target_cells
        and certificate.requested_count <= 0
    ):
        reasons.append("required_new_target_population_has_zero_quota")
    if certificate.requested_count > 0 and (
        certificate.placed_count != certificate.requested_count
        or len(certificate.placements) != certificate.requested_count
    ):
        reasons.append("packing_certificate_witness_ledger_incomplete")
    witness_centers = retained_centers.copy()
    for placement in certificate.placements:
        if placement.class_id == target_class:
            witness_centers[placement.row, placement.col] = True
    exact_coverage = anchor_coverage_fraction(
        program.continuity_anchor_mask,
        witness_centers,
        maximum_empty_run_px=program.continuity_maximum_empty_run_px,
    )
    if (
        program.continuity_requires_new_target_cells
        and exact_coverage
        < program.continuity_minimum_anchor_coverage_fraction
    ):
        reasons.append("exact_continuity_coverage_below_contract")
    return replace(
        report,
        passed=not reasons,
        required_add_count=int(certificate.requested_count),
        required_seam_count=int(certificate.required_seam_count),
        estimated_add_capacity=certificate.placed_count,
        estimated_seam_capacity=certificate.placed_seam_count,
        reference_fit_center_pixels=int(
            certificate.center_region_pixels
        ),
        exact_packing_certificate={
            **certificate.to_metadata(),
            "assessment_stage": "compiled_E_P_V_C_pre_probnet",
            "exact_anchor_coverage_fraction": float(exact_coverage),
            "minimum_anchor_coverage_fraction": float(
                program.continuity_minimum_anchor_coverage_fraction
            ),
        },
        complete_instance_spill_pixels=int(
            np.count_nonzero(
                np.asarray(program.erasure_region, dtype=bool)
                & ~np.asarray(candidate.change_region, dtype=bool)
            )
        ),
        target_footprint_spill_pixels=int(
            np.count_nonzero(
                certificate.footprint_union
                & ~np.asarray(candidate.change_region, dtype=bool)
                & ~np.asarray(program.erasure_region, dtype=bool)
            )
        ),
        predicted_joint_pixels=int(
            np.count_nonzero(
                np.asarray(candidate.change_region, dtype=bool)
                | np.asarray(program.erasure_region, dtype=bool)
                | certificate.footprint_union
            )
        ),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _known_provenance(value: Any) -> bool:
    if not value:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        if lowered.startswith("unknown") or "not_recorded" in lowered:
            return False
    return True


def _reference_at_quantile(
    references: tuple[ReferenceNucleusShape, ...],
    quantile: float,
) -> ReferenceNucleusShape | None:
    if not references:
        return None
    ordered = sorted(references, key=lambda item: (item.area_px, item.instance_id))
    index = min(
        len(ordered) - 1,
        max(0, int(np.ceil(float(quantile) * len(ordered))) - 1),
    )
    return ordered[index]


def _reference_area_at_quantile(
    references: tuple[ReferenceNucleusShape, ...],
    quantile: float,
) -> float:
    if not references:
        return 0.0
    return float(np.quantile([item.area_px for item in references], quantile))


def _reference_area_p95(
    references: tuple[ReferenceNucleusShape, ...],
) -> float:
    if not references:
        return 0.0
    return float(np.quantile([item.area_px for item in references], 0.95))


def _target_interface_population_density(
    scene: JointSceneAnalysis,
    *,
    source_tissue: np.ndarray,
    target_classes: tuple[int, ...],
    target_label: str,
    schema: MaskProfileSchema,
    reference_area_p95: float,
) -> tuple[float, dict[int, float]]:
    """Measure abundance from the same instance authority used by every joint tool.

    Native instance JSON is authoritative when mounted; otherwise the scene's
    deterministic watershed fallback is authoritative. Density, erasure,
    packing and gates must never silently re-segment the semantic raster with a
    different ruler.
    """

    target_ids = set(schema.resolve_fine_ids(target_label))
    tissue = np.asarray(source_tissue)
    if tissue.shape != scene.source_nuclei.shape:
        raise JointContractError(
            "source tissue and nuclei must align for abundance estimation"
        )
    class_counts: dict[int, int] = {}
    for item in scene.cells.instances:
        row = int(np.clip(round(item.centroid_xy[1]), 0, tissue.shape[0] - 1))
        col = int(np.clip(round(item.centroid_xy[0]), 0, tissue.shape[1] - 1))
        if int(tissue[row, col]) not in target_ids:
            continue
        class_counts[item.class_id] = class_counts.get(item.class_id, 0) + 1
    area = int(np.count_nonzero(np.isin(tissue, tuple(target_ids))))
    eligible_count = int(sum(class_counts.values()))
    if area > 0 and eligible_count:
        by_class = {
            class_id: float(count / area)
            for class_id, count in class_counts.items()
            if class_id in set(target_classes)
        }
        if not by_class:
            by_class = {int(target_classes[0]): float(eligible_count / area)}
        return float(eligible_count / area), by_class
    # No observable local target population means no empirical density prior.
    # In particular, p95=0 must never become one nucleus per pixel.
    if not eligible_count or reference_area_p95 <= 0:
        return 0.0, {}
    fallback = float(1.0 / max(1.0, reference_area_p95 * 3.0))
    return fallback, {target_classes[0]: fallback}


def _whole_instance_closure_px(
    scene: JointSceneAnalysis,
    removable_ids: list[str],
) -> int:
    removable = set(removable_ids)
    diagonals = []
    for item in scene.cells.instances:
        if item.instance_id not in removable:
            continue
        x0, y0, x1, y1 = item.bbox_xyxy
        diagonals.append(hypot(max(0, x1 - x0), max(0, y1 - y0)))
    if not diagonals:
        return max(1, round(scene.population.nominal_nucleus_diameter_px or 8.0))
    # Every instance left in ``removable`` has already passed the border,
    # connectivity, merged-suspect and fragment checks.  Atomic removal must
    # cover the largest such complete instance, not the 95th percentile;
    # otherwise the largest legitimate nucleus is declared removable during
    # preflight and then paradoxically rejected as "non-local" downstream.
    return max(1, ceil(float(max(diagonals))))


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


def _representative_fit_references(
    references: tuple[ReferenceNucleusShape, ...],
) -> tuple[ReferenceNucleusShape, ...]:
    """Return complete local Q25/Q50/Q75 shapes for capacity prediction.

    The mature sampler chooses among patch-local complete shapes. A preflight
    based on one fixed P75 footprint is falsely pessimistic at legitimate
    seams, while using the smallest observation would repeat the old size
    mismatch. The three central quantiles mirror the mature local shape pool
    without admitting fragments or large outliers.
    """

    selected = []
    seen: set[str] = set()
    for quantile in (0.25, 0.50, 0.75):
        item = _reference_at_quantile(references, quantile)
        if item is not None and item.instance_id not in seen:
            selected.append(item)
            seen.add(item.instance_id)
    return tuple(selected)


def _reference_fit_centers_union(
    free: np.ndarray,
    references: tuple[ReferenceNucleusShape, ...],
) -> np.ndarray:
    result = np.zeros_like(free, dtype=bool)
    for reference in references:
        result |= _reference_fit_centers(free, reference)
    return result


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
