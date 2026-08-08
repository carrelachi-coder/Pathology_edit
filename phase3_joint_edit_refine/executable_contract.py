"""One immutable execution contract for tissue, cell tools, gates and handoff.

The Planner is allowed to choose semantic IDs.  This compiler owns every
pixel-valued region and freezes the result before either a cell executor or a
joint gate can consume it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any

import numpy as np
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import CandidateMask, GateReport

from .budget import JointBudgetAllocation
from .cell_programs import CellToolProgramCompiler, CompiledCellToolProgram
from .models import JointCaseContext, JointContractError, JointEditPlan
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle

EXECUTABLE_CONTRACT_VERSION = "joint-executable-contract-v4"


@dataclass(frozen=True)
class ExecutableJointContract:
    """Candidate-level authority shared by all deterministic executors.

    E/P/V/S are stored in ``cell_program``.  Metadata carries hashes for the
    masks, source assets, Planner plan, tissue result, skills and gates so a
    downstream component cannot silently reinterpret or replace any part.
    """

    schema_version: str
    contract_id: str
    case_id: str
    primitive_id: str
    mechanism_id: str
    tissue_candidate_id: str
    source_labels: tuple[str, ...]
    target_label: str | None
    selected_interface_ids: tuple[str, ...]
    selected_anchor_ids: tuple[str, ...]
    selected_structural_unit_ids: tuple[str, ...]
    affected_structural_unit_ids: tuple[str, ...]
    structural_hierarchy_digest: str
    erase_instance_ids: tuple[str, ...]
    protected_instance_ids: tuple[str, ...]
    allowed_new_cell_classes: tuple[int, ...]
    forbidden_new_cell_classes: tuple[int, ...]
    target_host_fine_ids: tuple[int, ...]
    population_dataset_name: str
    required_checker_ids: tuple[str, ...]
    active_rule_ids: tuple[str, ...]
    source_asset_digests: dict[str, str]
    skill_versions: dict[str, str]
    budget_allocation: dict[str, Any] | None
    plan_digest: str
    source_tissue_digest: str
    source_nuclei_digest: str
    target_tissue_digest: str
    tissue_change_digest: str
    tissue_gate_report_digest: str
    cell_program: CompiledCellToolProgram
    packing_certificate: dict[str, Any] | None

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTABLE_CONTRACT_VERSION:
            raise JointContractError("unsupported executable contract version")
        if not self.contract_id:
            raise JointContractError("executable contract ID is required")
        if not self.required_checker_ids or not self.active_rule_ids:
            raise JointContractError(
                "executable contract must bind checkers and active skill rules"
            )
        if not self.allowed_new_cell_classes:
            raise JointContractError(
                "executable contract has no legal target cell class"
            )
        if set(self.allowed_new_cell_classes).intersection(
            self.forbidden_new_cell_classes
        ):
            raise JointContractError(
                "allowed and forbidden new cell classes overlap"
            )
        if tuple(self.cell_program.target_classes) != tuple(
            self.allowed_new_cell_classes
        ):
            raise JointContractError(
                "cell program classes differ from executable contract classes"
            )
        if set(self.affected_structural_unit_ids) - set(
            self.selected_structural_unit_ids
        ):
            raise JointContractError(
                "affected structural units are not authorized by the Planner"
            )
        if not self.structural_hierarchy_digest:
            raise JointContractError("structural hierarchy digest is required")
        if self.packing_certificate is not None:
            certificate = self.packing_certificate
            if certificate.get("passed") is not True:
                raise JointContractError(
                    "executable packing certificate is not passing"
                )
            requested = int(certificate.get("requested_count", 0))
            placements = certificate.get("placements") or []
            if requested <= 0 or len(placements) != requested:
                raise JointContractError(
                    "executable packing certificate has an incomplete witness ledger"
                )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "case_id": self.case_id,
            "primitive_id": self.primitive_id,
            "mechanism_id": self.mechanism_id,
            "tissue_candidate_id": self.tissue_candidate_id,
            "tissue_label_contract": {
                "source_labels": list(self.source_labels),
                "target_label": self.target_label,
            },
            "selected_interface_ids": list(self.selected_interface_ids),
            "selected_anchor_ids": list(self.selected_anchor_ids),
            "structural_hierarchy": {
                "selected_unit_ids": list(self.selected_structural_unit_ids),
                "affected_unit_ids": list(self.affected_structural_unit_ids),
                "digest": self.structural_hierarchy_digest,
            },
            "cell_instance_contract": {
                "erase_instance_ids": list(self.erase_instance_ids),
                "protected_instance_ids": list(self.protected_instance_ids),
                "allowed_new_cell_classes": list(self.allowed_new_cell_classes),
                "forbidden_new_cell_classes": list(
                    self.forbidden_new_cell_classes
                ),
                "target_host_fine_ids": list(self.target_host_fine_ids),
            },
            "population_dataset_name": self.population_dataset_name,
            "required_checker_ids": list(self.required_checker_ids),
            "active_rule_ids": list(self.active_rule_ids),
            "source_asset_digests": dict(sorted(self.source_asset_digests.items())),
            "skill_versions": dict(sorted(self.skill_versions.items())),
            "budget_allocation": self.budget_allocation,
            "digests": {
                "plan": self.plan_digest,
                "source_tissue": self.source_tissue_digest,
                "source_nuclei": self.source_nuclei_digest,
                "target_tissue": self.target_tissue_digest,
                "tissue_change": self.tissue_change_digest,
                "tissue_gate_report": self.tissue_gate_report_digest,
            },
            "cell_program": self.cell_program.to_metadata(),
            "packing_certificate": self.packing_certificate,
        }

    def bind_packing_certificate(
        self, certificate: dict[str, Any]
    ) -> ExecutableJointContract:
        """Return a new immutable contract that owns the exact packing witness."""

        payload = dict(certificate)
        # A preliminary contract ID may appear in diagnostic metadata.  It
        # cannot be part of the final certificate without creating a circular
        # hash dependency.
        payload.pop("contract_id", None)
        draft = replace(
            self,
            contract_id="pending",
            packing_certificate=payload,
        )
        metadata = draft.to_metadata()
        metadata["contract_id"] = ""
        bound = replace(draft, contract_id=_canonical_digest(metadata))
        bound.validate_identity()
        return bound

    def validate_identity(self) -> None:
        payload = self.to_metadata()
        payload["contract_id"] = ""
        expected = _canonical_digest(payload)
        if expected != self.contract_id:
            raise JointContractError(
                "executable contract metadata or E/P/V/S digest was mutated"
            )

    def tissue_gate_report_matches(self, report: GateReport) -> bool:
        return (
            report.candidate_id == self.tissue_candidate_id
            and report.passed
            and _canonical_digest(report.to_metadata())
            == self.tissue_gate_report_digest
        )

    def validate_handoff_binding(self, candidate) -> None:
        """Reject a selected pair that is not the result bound by this contract."""

        self.validate_identity()
        errors = []
        if candidate.tissue_candidate_id != self.tissue_candidate_id:
            errors.append("tissue_candidate_id_mismatch")
        if candidate.mechanism_id != self.mechanism_id:
            errors.append("mechanism_id_mismatch")
        if candidate.tool_trace.get("executable_contract_id") != self.contract_id:
            errors.append("candidate_trace_contract_id_mismatch")
        if _array_digest(candidate.target_tissue_mask) != self.target_tissue_digest:
            errors.append("target_tissue_digest_mismatch")
        if _array_digest(candidate.tissue_change) != self.tissue_change_digest:
            errors.append("tissue_change_digest_mismatch")
        if not np.array_equal(
            np.asarray(candidate.generation_support, dtype=bool),
            np.asarray(self.cell_program.support_context_region, dtype=bool),
        ):
            errors.append("generation_support_not_contract_S")
        if errors:
            raise JointContractError(
                "generation handoff detached from executable contract: "
                + "; ".join(errors)
            )

    def validate_candidate(
        self,
        *,
        source_tissue: np.ndarray,
        source_nuclei: np.ndarray,
        target_tissue: np.ndarray,
        target_nuclei: np.ndarray,
        tissue_change: np.ndarray,
        cell_change: np.ndarray,
        scene: JointSceneAnalysis,
        new_cell_center_ledger: tuple[tuple[int, int, int], ...] | None = None,
    ) -> tuple[str, ...]:
        """Return deterministic result violations against this exact contract."""

        errors: list[str] = []
        try:
            self.validate_identity()
        except JointContractError as exc:
            errors.append(str(exc))
        if _array_digest(source_tissue) != self.source_tissue_digest:
            errors.append("source_tissue_digest_mismatch")
        if _array_digest(source_nuclei) != self.source_nuclei_digest:
            errors.append("source_nuclei_digest_mismatch")
        if _array_digest(target_tissue) != self.target_tissue_digest:
            errors.append("target_tissue_digest_mismatch")
        if _array_digest(np.asarray(tissue_change, dtype=bool)) != self.tissue_change_digest:
            errors.append("tissue_change_digest_mismatch")
        if _canonical_digest(scene.structural_hierarchy) != self.structural_hierarchy_digest:
            errors.append("structural_hierarchy_digest_mismatch")
        known_units = set(scene.structural_unit_masks)
        if set(self.selected_structural_unit_ids) - known_units:
            errors.append("selected_structural_unit_missing_from_scene")
        change_neighborhood = ndimage.binary_dilation(
            np.asarray(tissue_change, dtype=bool), iterations=1
        )
        observed_affected_units = tuple(
            sorted(
                unit_id
                for unit_id, mask in scene.structural_unit_masks.items()
                if np.any(np.asarray(mask, dtype=bool) & change_neighborhood)
            )
        )
        if observed_affected_units != self.affected_structural_unit_ids:
            errors.append("affected_structural_unit_binding_mismatch")
        program = self.cell_program
        changed = np.asarray(cell_change, dtype=bool)
        population = np.asarray(program.population_target_region, dtype=bool)
        if not np.any(population):
            errors.append("population_target_region_is_empty")
        if np.any(population & ~program.valid_footprint_region):
            errors.append("population_target_region_outside_target_host")
        if np.any(population & ~program.support_context_region):
            errors.append("population_target_region_outside_contract_support")
        if np.any(changed & ~program.support_context_region):
            errors.append("cell_change_outside_contract_support")
        if np.any(program.placement_center_region & ~program.valid_footprint_region):
            errors.append("placement_region_outside_valid_footprint_region")
        if np.any(program.continuity_region & ~program.placement_center_region):
            errors.append("continuity_region_outside_placement_region")
        if program.depletion_profile_id is not None:
            core = np.asarray(program.depletion_core_region, dtype=bool)
            transition = np.asarray(
                program.depletion_transition_region, dtype=bool
            )
            outer = np.asarray(
                program.depletion_outer_reference_region, dtype=bool
            )
            if not all(np.any(item) for item in (core, transition, outer)):
                errors.append("depletion_three_band_contract_is_empty")
            if np.any(core & transition) or np.any(core & outer) or np.any(
                transition & outer
            ):
                errors.append("depletion_bands_overlap")
            if not np.array_equal(
                program.placement_center_region, core | transition
            ):
                errors.append("depletion_placement_region_not_core_transition")
            if not np.array_equal(
                program.population_target_region, core | transition | outer
            ):
                errors.append("depletion_population_region_not_three_bands")
            if not np.any(program.depletion_anchor_mask):
                errors.append("depletion_anchor_is_empty")
        if (
            program.continuity_requires_new_target_cells
            and not np.any(program.continuity_anchor_mask)
        ):
            errors.append("required_continuity_anchor_is_empty")
        if (
            program.continuity_requires_new_target_cells
            and not np.any(program.continuity_region)
        ):
            errors.append("required_continuity_region_is_empty")
        target = np.asarray(target_nuclei)
        source = np.asarray(source_nuclei)
        for instance_id in self.protected_instance_ids:
            component = scene.instance_masks.get(instance_id)
            if component is None:
                errors.append(f"protected_instance_missing:{instance_id}")
            elif np.any(target[component] != source[component]):
                errors.append(f"protected_instance_changed:{instance_id}")
        allowed = set(self.allowed_new_cell_classes)
        if new_cell_center_ledger is not None:
            for row, col, class_id in new_cell_center_ledger:
                if class_id not in allowed:
                    errors.append(
                        f"new_cell_center_class_forbidden:{row}:{col}:{class_id}"
                    )
                elif not (
                    0 <= row < target.shape[0]
                    and 0 <= col < target.shape[1]
                    and (
                        program.placement_center_region[row, col]
                        or program.mechanism_region[row, col]
                    )
                ):
                    errors.append(
                        f"new_cell_accepted_center_outside_contract_zone:{row}:{col}"
                    )
        for instance_id, class_id, component in _target_instances(target):
            is_new = bool(np.any(component & changed))
            if not is_new:
                continue
            if class_id not in allowed:
                errors.append(f"new_cell_class_forbidden:{instance_id}:{class_id}")
            if np.any(component & ~program.valid_footprint_region):
                errors.append(f"new_cell_outside_valid_footprint:{instance_id}")
            if new_cell_center_ledger is None:
                row, col = ndimage.center_of_mass(component)
                row, col = round(row), round(col)
                center_ok = (
                    0 <= row < target.shape[0]
                    and 0 <= col < target.shape[1]
                    and (
                        program.placement_center_region[row, col]
                        or program.mechanism_region[row, col]
                    )
                )
                if not center_ok:
                    errors.append(
                        f"new_cell_center_outside_contract_zone:{instance_id}"
                    )
        return tuple(errors)


class ExecutableJointContractCompiler:
    """Compile and freeze one executable contract per tissue candidate."""

    def __init__(self, cell_program_compiler: CellToolProgramCompiler | None = None):
        self.cell_program_compiler = (
            cell_program_compiler or CellToolProgramCompiler()
        )

    def compile(
        self,
        *,
        case: JointCaseContext,
        source_tissue: np.ndarray,
        source_nuclei: np.ndarray,
        schema: MaskProfileSchema,
        scene: JointSceneAnalysis,
        plan: JointEditPlan,
        bundle: JointSkillBundle,
        tissue_candidate: CandidateMask,
        tissue_gate_report: GateReport,
        allocation: JointBudgetAllocation | None,
        required_checker_ids: tuple[str, ...],
    ) -> ExecutableJointContract:
        if not required_checker_ids:
            raise JointContractError(
                "executable contract requires the complete fail-closed checker program"
            )
        if not tissue_gate_report.passed:
            raise JointContractError(
                "cannot compile an executable contract from a failed tissue candidate"
            )
        if tissue_gate_report.candidate_id != tissue_candidate.candidate_id:
            raise JointContractError("tissue gate report candidate binding mismatch")
        source_tissue = np.asarray(source_tissue)
        source_nuclei = np.asarray(source_nuclei)
        target_tissue = np.asarray(tissue_candidate.target_mask)
        tissue_change = np.asarray(tissue_candidate.change_region, dtype=bool)
        if any(
            item.shape != source_tissue.shape
            for item in (source_nuclei, target_tissue, tissue_change)
        ):
            raise JointContractError(
                "executable contract arrays must share one shape"
            )
        expected_change = source_tissue != target_tissue
        if not np.array_equal(tissue_change, expected_change):
            raise JointContractError(
                "tissue candidate change mask is not source/target exact"
            )
        known_units = set(scene.structural_unit_masks)
        selected_units = tuple(plan.structural_unit_ids)
        unknown_units = set(selected_units) - known_units
        if unknown_units:
            raise JointContractError(
                "Planner selected unknown structural units: "
                + ", ".join(sorted(unknown_units))
            )
        change_neighborhood = ndimage.binary_dilation(
            tissue_change, iterations=1
        )
        affected_units = tuple(
            sorted(
                unit_id
                for unit_id, mask in scene.structural_unit_masks.items()
                if np.any(np.asarray(mask, dtype=bool) & change_neighborhood)
            )
        )
        unauthorized_units = set(affected_units) - set(selected_units)
        if unauthorized_units:
            raise JointContractError(
                "candidate touches structural units omitted by the Planner: "
                + ", ".join(sorted(unauthorized_units))
            )
        base_program = self.cell_program_compiler.compile(
            case=case,
            schema=schema,
            scene=scene,
            plan=plan,
            bundle=bundle,
            tissue_candidate=tissue_candidate,
        )
        protected = set(plan.cell_plan.protected_instance_ids)
        erase_ids = tuple(
            sorted(
                instance_id
                for instance_id, component in scene.instance_masks.items()
                if np.any(
                    np.asarray(component, dtype=bool)
                    & base_program.erasure_region
                )
                and instance_id not in protected
            )
        )
        protected_overlap = tuple(
            sorted(
                instance_id
                for instance_id in protected
                if instance_id in scene.instance_masks
                and np.any(
                    np.asarray(scene.instance_masks[instance_id], dtype=bool)
                    & base_program.erasure_region
                )
            )
        )
        if protected_overlap:
            raise JointContractError(
                "tissue candidate intersects protected source nuclei: "
                + ", ".join(protected_overlap)
            )
        erasure = np.zeros_like(tissue_change)
        for instance_id in erase_ids:
            erasure |= np.asarray(scene.instance_masks[instance_id], dtype=bool)
        closure_seed = (
            tissue_change
            if bundle.primitive.scope == "tissue_and_cell"
            else np.asarray(base_program.placement_center_region, dtype=bool)
        )
        whole_instance_closure_px = _exact_instance_closure_px(
            scene,
            erase_ids,
            fallback=base_program.whole_instance_closure_px,
        )
        closure = ndimage.binary_dilation(
            closure_seed,
            iterations=whole_instance_closure_px,
        )
        if np.any(erasure & ~closure):
            raise JointContractError(
                "whole-instance erasure exceeds its candidate-local atomic closure"
            )

        target_label = (
            plan.tissue_plan.target_label if plan.tissue_plan is not None else None
        )
        allowed_classes = _resolve_allowed_new_classes(
            requested=plan.cell_plan.allowed_cell_classes,
            target_label=target_label,
            allow_neoplastic_in_non_tumor=(
                plan.coupling_plan.allow_neoplastic_in_non_tumor_tissue
            ),
            primitive_target_classes=bundle.primitive.target_cell_classes,
            tissue_compatible_classes=(
                bundle.cell_observation_profile.tissue_compatible_classes
            ),
        )
        observation_classes = set(bundle.cell_observation_profile.class_ids)
        forbidden_classes = tuple(sorted(observation_classes - set(allowed_classes)))
        host_ids = _resolve_target_host_fine_ids(
            target_label=target_label,
            primitive_host_labels=bundle.primitive.host_tissue_labels,
            schema=schema,
        )
        protected_structure = np.zeros_like(target_tissue, dtype=bool)
        for structure_id in sorted(
            bundle.mechanism.representability.required_auxiliary_structures
        ):
            if structure_id not in scene.auxiliary_structure_masks:
                raise JointContractError(
                    f"required auxiliary structure {structure_id!r} is unavailable"
                )
            protected_structure |= np.asarray(
                scene.auxiliary_structure_masks[structure_id], dtype=bool
            )
        profile_valid = ~np.isin(
            target_tissue,
            bundle.annotation_profile.prohibit_cell_placement_fine_ids,
        ) & ~protected_structure
        valid = profile_valid & np.isin(target_tissue, host_ids)
        # P is a center-domain contract, not a pre-eroded approximation of V.
        # The exact packing certificate below tests every concrete reference
        # footprint against V. Eroding P here by a nominal radius and then
        # testing the real footprint again double-counted containment and made
        # valid narrow interface bands fail only after tissue generation.
        center_constraint = valid
        placement = (
            np.asarray(base_program.placement_center_region, dtype=bool)
            & center_constraint
        )
        mechanism_region = (
            np.asarray(base_program.mechanism_region, dtype=bool)
            & center_constraint
        )
        continuity_region = (
            np.asarray(base_program.continuity_region, dtype=bool) & placement
        )
        if "add" in plan.cell_plan.actions and not np.any(
            placement | mechanism_region
        ):
            raise JointContractError(
                "no complete nucleus footprint fits the compiled placement zone"
            )
        generation_allowed = ~np.isin(
            target_tissue,
            bundle.annotation_profile.prohibit_generation_support_fine_ids,
        ) & ~protected_structure
        # P contains legal *centers*, whereas S must contain the complete
        # footprint ultimately pasted at those centers.  The mature library
        # permits calibrated, mildly elongated shapes; 1.25 diameters left a
        # few legitimate edge pixels outside S in otherwise valid layouts.
        # A 1.5-diameter footprint margin is still bounded local context and is
        # not charged to J, but it closes the center-versus-footprint contract.
        support_radius = max(
            1, int(np.ceil(1.5 * base_program.nominal_nucleus_diameter_px))
        )
        population = np.asarray(
            base_program.population_target_region, dtype=bool
        ) & valid
        if not np.any(population):
            raise JointContractError(
                "compiled target population region is empty"
            )
        support_seed = (
            erasure
            | tissue_change
            | population
            | placement
            | mechanism_region
        )
        if np.any(support_seed & ~generation_allowed):
            raise JointContractError(
                "required T/E/P/mechanism support intersects a profile-prohibited region"
            )
        support = ndimage.binary_dilation(
            support_seed,
            iterations=support_radius,
        ) & generation_allowed
        support |= support_seed
        program = replace(
            base_program,
            population_target_region=population,
            erasure_region=erasure,
            placement_center_region=placement,
            valid_footprint_region=valid,
            support_context_region=support,
            mechanism_region=mechanism_region,
            continuity_region=continuity_region,
            whole_instance_closure_px=whole_instance_closure_px,
            target_classes=allowed_classes,
            policies={
                **base_program.policies,
                "T_pop": (
                    "target-population-abundance-denominator-distinct-from-P"
                ),
                "E": "exact-union-of-complete-source-instance-footprints",
                "P": "contract-legal-centers-exact-footprint-certified-against-V",
                "V": "profile-and-target-host-fine-id-containment",
                "S": "bounded-context-containing-T-E-P-and-mechanism-zone",
                "S_footprint_margin": (
                    "placement-centers-dilated-by-1.5-local-nucleus-diameters"
                ),
                "class_host": "contract-filtered-target-tissue-compatibility",
                "whole_instance_closure": (
                    "candidate-local-complete-instance-bbox-diagonal"
                ),
            },
        )
        interfaces = tuple(plan.cell_plan.interface_ids)
        anchors = tuple(plan.cell_plan.anchor_ids)
        source_labels = (
            tuple(plan.tissue_plan.source_labels)
            if plan.tissue_plan is not None
            else tuple(bundle.primitive.host_tissue_labels)
        )
        source_assets = _provenance_digests(case.provenance)
        source_assets["provenance_sha256"] = _canonical_digest(case.provenance)
        draft = ExecutableJointContract(
            schema_version=EXECUTABLE_CONTRACT_VERSION,
            contract_id="pending",
            case_id=case.case_id,
            primitive_id=case.primitive_id,
            mechanism_id=plan.selected_mechanism_id,
            tissue_candidate_id=tissue_candidate.candidate_id,
            source_labels=source_labels,
            target_label=target_label,
            selected_interface_ids=interfaces,
            selected_anchor_ids=anchors,
            selected_structural_unit_ids=selected_units,
            affected_structural_unit_ids=affected_units,
            structural_hierarchy_digest=_canonical_digest(
                scene.structural_hierarchy
            ),
            erase_instance_ids=erase_ids,
            protected_instance_ids=tuple(sorted(protected)),
            allowed_new_cell_classes=allowed_classes,
            forbidden_new_cell_classes=forbidden_classes,
            target_host_fine_ids=host_ids,
            population_dataset_name=(
                bundle.cell_population_profile.probnet_dataset_name
            ),
            required_checker_ids=tuple(required_checker_ids),
            active_rule_ids=tuple(bundle.active_rule_ids),
            source_asset_digests=source_assets,
            skill_versions={
                "primitive": bundle.primitive.version,
                "mechanism": bundle.mechanism.version,
                "annotation_profile": bundle.annotation_profile.version,
                "cell_observation_profile": (
                    bundle.cell_observation_profile.version
                ),
                "cell_population_profile": bundle.cell_population_profile.version,
            },
            budget_allocation=(
                allocation.to_metadata() if allocation is not None else None
            ),
            plan_digest=_canonical_digest(plan.to_metadata()),
            source_tissue_digest=_array_digest(source_tissue),
            source_nuclei_digest=_array_digest(source_nuclei),
            target_tissue_digest=_array_digest(target_tissue),
            tissue_change_digest=_array_digest(tissue_change),
            tissue_gate_report_digest=_canonical_digest(
                tissue_gate_report.to_metadata()
            ),
            cell_program=program,
            packing_certificate=None,
        )
        payload = draft.to_metadata()
        payload["contract_id"] = ""
        contract = replace(draft, contract_id=_canonical_digest(payload))
        contract.validate_identity()
        return contract


def _resolve_allowed_new_classes(
    *,
    requested: tuple[int, ...],
    target_label: str | None,
    allow_neoplastic_in_non_tumor: bool,
    primitive_target_classes: tuple[int, ...],
    tissue_compatible_classes: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    allowed = {int(value) for value in requested}
    if primitive_target_classes:
        allowed &= {int(value) for value in primitive_target_classes}
    if target_label is not None:
        compatible = set(tissue_compatible_classes.get(target_label, ()))
        if allow_neoplastic_in_non_tumor:
            compatible.add(1)
        allowed &= compatible
    if not allowed:
        raise JointContractError(
            "Planner, primitive and target-tissue cell classes have empty intersection"
        )
    return tuple(sorted(allowed))


def _resolve_target_host_fine_ids(
    *,
    target_label: str | None,
    primitive_host_labels: tuple[str, ...],
    schema: MaskProfileSchema,
) -> tuple[int, ...]:
    labels = (target_label,) if target_label is not None else primitive_host_labels
    fine_ids: set[int] = set()
    for label in labels:
        if label in schema.readable_labels:
            fine_ids.update(schema.resolve_fine_ids(label))
    if not fine_ids:
        raise JointContractError("executable contract has no target host fine IDs")
    return tuple(sorted(fine_ids))


def _target_instances(mask: np.ndarray):
    from .nuclei import iter_instances

    return tuple(iter_instances(mask))


def _exact_instance_closure_px(
    scene: JointSceneAnalysis,
    instance_ids: tuple[str, ...],
    *,
    fallback: int,
) -> int:
    """Radius needed to close whole instances intersecting the edit core.

    This is intentionally independent of the biological mechanism halo. The
    bbox diagonal is a conservative bound from any intersection pixel to the
    farthest pixel of the same connected instance.
    """

    metadata = {item.instance_id: item for item in scene.cells.instances}
    diagonals = []
    for instance_id in instance_ids:
        item = metadata.get(instance_id)
        if item is None:
            continue
        x0, y0, x1, y1 = item.bbox_xyxy
        diagonals.append(
            float(np.hypot(max(0, x1 - x0), max(0, y1 - y0)))
        )
    return max(
        1,
        int(np.ceil(max(diagonals))) if diagonals else int(fallback),
    )


def _provenance_digests(provenance: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in provenance.items():
        if key.endswith("_sha256") and isinstance(value, str):
            result[key] = value
        elif key.endswith("_sha256") and isinstance(value, dict):
            for child_key, child_value in value.items():
                if isinstance(child_value, str):
                    result[f"{key}.{child_key}"] = child_value
    return result


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "sha256": _array_digest(value),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if hasattr(value, "to_metadata"):
        return value.to_metadata()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


__all__ = [
    "EXECUTABLE_CONTRACT_VERSION",
    "ExecutableJointContract",
    "ExecutableJointContractCompiler",
]
