"""Typed runtime schema for atomic tissue--cell mechanism skills."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from phase3_joint_edit_refine.models import (
    CELL_ACTIONS,
    LAYOUT_PROGRAMS,
    JointContractError,
)

SUPPORT_STATUSES = frozenset(
    {"supported", "conditionally_supported", "render_only", "unsupported"}
)
REVIEW_STATUSES = frozenset({"draft", "empirically_validated", "internally_reviewed"})
PRIMITIVE_SCOPES = frozenset({"tissue_and_cell", "cell_only"})
PRIMITIVE_BUDGET_MODES = frozenset({"joint_area_with_tissue_floor", "count_extent"})
TISSUE_GEOMETRY_MODES = frozenset(
    {"interface_front", "component_boundary_turnover"}
)
SEAM_MODES = frozenset(
    {"adaptive_population_continuity", "turnover_transition", "not_applicable"}
)


@dataclass(frozen=True)
class RecognitionContract:
    required_observations: tuple[str, ...]
    contraindications: tuple[str, ...]
    minimum_confidence: float


@dataclass(frozen=True)
class RepresentabilityContract:
    status: str
    required_cell_classes: tuple[int, ...]
    required_auxiliary_structures: tuple[str, ...]
    allow_semantic_instance_fallback: bool
    failure_action: str


@dataclass(frozen=True)
class TissueProgramContract:
    mode: str
    primitive_label_contracts: dict[str, dict[str, tuple[str, ...]]]
    allowed_tools: tuple[str, ...]
    required_checker_ids: tuple[str, ...]
    prohibited_structures: tuple[str, ...]
    front: TissueFrontContract


@dataclass(frozen=True)
class TissueFrontContract:
    """Mechanism-owned executable shape bounds for tissue displacement."""

    profile_mode: str
    edge_depth_ratio: float
    taper_fraction: float
    lobe_count: int
    noise_depth_ratio: float
    maximum_band_px: int
    maximum_depth_span_ratio: float


@dataclass(frozen=True)
class CellProgramContract:
    actions: tuple[str, ...]
    allowed_cell_classes: tuple[int, ...]
    layout_programs: tuple[str, ...]
    core_policy: str
    halo_policy: str
    halo_distance_px: tuple[int, int]
    cluster_size_range: tuple[int, int]
    seam: SeamContract
    cellularity_depletion: CellularityDepletionContract | None
    render_owned_clearance_primitives: tuple[str, ...]
    required_checker_ids: tuple[str, ...]


@dataclass(frozen=True)
class SeamContract:
    """Skill-owned cellular continuity contract at an edited tissue seam.

    The seam is compiled only after Planner-selected anchors and a concrete
    tissue candidate exist.  Distances are expressed in local cell scales so
    the same executor works across organs and resolutions.
    """

    mode: str
    width_cell_diameters: tuple[float, float]
    reference_area_quantiles: tuple[float, float]
    maximum_empty_run_cell_diameters: float
    density_ratio_range: tuple[float, float]
    minimum_anchor_coverage_fraction: float
    requires_new_target_cells: bool


@dataclass(frozen=True)
class CellularityDepletionContract:
    """Executable, skill-owned bounds for localized cellularity reduction."""

    program_id: str
    resolution_mode: str
    allowed_anchor_types: tuple[str, ...]
    allowed_neighbor_labels: tuple[str, ...]
    core_width_cell_diameters: float
    transition_width_cell_diameters: float
    outer_reference_width_cell_diameters: float
    core_removal_weight: float
    transition_removal_weight: float
    core_target_removal_fraction: float
    transition_start_removal_fraction: float
    transition_end_removal_fraction: float
    transition_subband_count: int
    minimum_core_residual_fraction: float
    minimum_transition_residual_fraction: float
    minimum_core_removals: int
    minimum_transition_removals: int
    maximum_new_gap_cell_diameters: float
    minimum_outer_reference_instances: int
    minimum_field_area_cell_diameter_squares: float


@dataclass(frozen=True)
class CouplingContract:
    compatibility_rule_ids: tuple[str, ...]
    allow_neoplastic_in_non_tumor_tissue: bool
    joint_area_mode: str
    tissue_floor_applies: bool
    cell_only_target_fraction: float
    cell_footprint_spill_reserve_fraction: float
    render_support_policy_id: str


@dataclass(frozen=True)
class RenderContract:
    required_findings: tuple[str, ...]
    veto_findings: tuple[str, ...]
    mask_guarantees: tuple[str, ...]
    render_only_claims: tuple[str, ...]
    required_findings_by_primitive: dict[str, tuple[str, ...]]
    veto_findings_by_primitive: dict[str, tuple[str, ...]]

    def required_for(self, primitive_id: str) -> tuple[str, ...]:
        return self.required_findings_by_primitive.get(
            primitive_id, self.required_findings
        )

    def vetoes_for(self, primitive_id: str) -> tuple[str, ...]:
        return self.veto_findings_by_primitive.get(
            primitive_id, self.veto_findings
        )


@dataclass(frozen=True)
class JointMechanismSkill:
    mechanism_id: str
    pathology_domain_id: str
    supported_primitives: tuple[str, ...]
    version: str
    review_status: str
    summary: str
    recognition: RecognitionContract
    representability: RepresentabilityContract
    tissue_program: TissueProgramContract
    cell_program: CellProgramContract
    coupling: CouplingContract
    joint_gate_ids: tuple[str, ...]
    render: RenderContract
    evidence_citations: tuple[str, ...]
    counterexamples: tuple[str, ...]
    source_path: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, source_path: str) -> JointMechanismSkill:
        mechanism_id = _string(payload, "mechanism_id")
        review_status = _string(payload, "review_status")
        if review_status not in REVIEW_STATUSES:
            raise JointContractError(f"unknown joint skill review status: {review_status}")
        recognition = _mapping(payload, "recognition_contract")
        representability = _mapping(payload, "representability_contract")
        tissue = _mapping(payload, "tissue_program")
        front = tissue.get("front_contract", {})
        if not isinstance(front, Mapping):
            raise JointContractError(
                f"{mechanism_id}.tissue_program.front_contract must be a mapping"
            )
        cell = _mapping(payload, "cell_program")
        coupling = _mapping(payload, "coupling_contract")
        render = _mapping(payload, "render_contract")
        status = _string(representability, "status")
        if status not in SUPPORT_STATUSES:
            raise JointContractError(f"unknown representability status: {status}")
        actions = _strings(cell, "actions")
        if not actions or set(actions) - CELL_ACTIONS:
            raise JointContractError(f"{mechanism_id} contains invalid cell actions")
        layouts = _strings(cell, "layout_programs")
        if not layouts or set(layouts) - LAYOUT_PROGRAMS:
            raise JointContractError(f"{mechanism_id} contains invalid layout programs")
        required_classes = _ints(representability, "required_cell_classes")
        allowed_classes = _ints(cell, "allowed_cell_classes")
        if set(required_classes) - set(allowed_classes):
            raise JointContractError(
                f"{mechanism_id} required cell classes are not allowed by its cell program"
            )
        halo = _pair(cell, "halo_distance_px")
        cluster = _pair(cell, "cluster_size_range")
        if halo[0] < 0 or halo[1] < halo[0] or cluster[0] < 1 or cluster[1] < cluster[0]:
            raise JointContractError(f"{mechanism_id} contains invalid cell ranges")
        confidence = float(recognition.get("minimum_confidence", 0.7))
        if not 0.0 <= confidence <= 1.0:
            raise JointContractError("minimum recognition confidence must be in [0,1]")
        cell_only_fraction = float(coupling.get("cell_only_target_fraction", 0.0))
        if not 0.0 <= cell_only_fraction <= 1.0:
            raise JointContractError("cell_only_target_fraction must be in [0,1]")
        footprint_spill_fraction = float(
            coupling.get("cell_footprint_spill_reserve_fraction", 0.0)
        )
        if not 0.0 <= footprint_spill_fraction <= 1.0:
            raise JointContractError(
                "cell_footprint_spill_reserve_fraction must be in [0,1]"
            )
        if cell_only_fraction + footprint_spill_fraction > 1.0:
            raise JointContractError(
                "cell-only and footprint-spill reserves cannot exceed the patch"
            )
        seam = _seam_contract(cell.get("seam_contract"))
        depletion = _cellularity_depletion_contract(
            cell.get("cellularity_depletion_contract"),
            required="cellularity-decrease-v1"
            in _strings(payload, "supported_primitives"),
            mechanism_id=mechanism_id,
        )
        return cls(
            mechanism_id=mechanism_id,
            pathology_domain_id=_string(payload, "pathology_domain_id"),
            supported_primitives=_strings(payload, "supported_primitives"),
            version=_string(payload, "version"),
            review_status=review_status,
            summary=_string(payload, "summary"),
            recognition=RecognitionContract(
                required_observations=_strings(recognition, "required_observations"),
                contraindications=_strings(
                    recognition, "contraindications", allow_empty=True
                ),
                minimum_confidence=confidence,
            ),
            representability=RepresentabilityContract(
                status=status,
                required_cell_classes=required_classes,
                required_auxiliary_structures=_strings(
                    representability, "required_auxiliary_structures", allow_empty=True
                ),
                allow_semantic_instance_fallback=bool(
                    representability.get("allow_semantic_instance_fallback", False)
                ),
                failure_action=_string(representability, "failure_action"),
            ),
            tissue_program=TissueProgramContract(
                mode=_string(tissue, "mode"),
                primitive_label_contracts=_primitive_label_contracts(
                    tissue, "primitive_label_contracts"
                ),
                allowed_tools=_strings(tissue, "allowed_tools"),
                required_checker_ids=_strings(tissue, "required_checker_ids"),
                prohibited_structures=_strings(
                    tissue, "prohibited_structures", allow_empty=True
                ),
                front=_tissue_front_contract(front, mechanism_id=mechanism_id),
            ),
            cell_program=CellProgramContract(
                actions=actions,
                allowed_cell_classes=allowed_classes,
                layout_programs=layouts,
                core_policy=_string(cell, "core_policy"),
                halo_policy=_string(cell, "halo_policy"),
                halo_distance_px=halo,
                cluster_size_range=cluster,
                seam=seam,
                cellularity_depletion=depletion,
                render_owned_clearance_primitives=_strings(
                    cell,
                    "render_owned_clearance_primitives",
                    allow_empty=True,
                ),
                required_checker_ids=_strings(cell, "required_checker_ids"),
            ),
            coupling=CouplingContract(
                compatibility_rule_ids=_strings(coupling, "compatibility_rule_ids"),
                allow_neoplastic_in_non_tumor_tissue=bool(
                    coupling.get("allow_neoplastic_in_non_tumor_tissue", False)
                ),
                joint_area_mode=_string(coupling, "joint_area_mode"),
                tissue_floor_applies=bool(coupling.get("tissue_floor_applies", True)),
                cell_only_target_fraction=cell_only_fraction,
                cell_footprint_spill_reserve_fraction=footprint_spill_fraction,
                render_support_policy_id=_string(coupling, "render_support_policy_id"),
            ),
            joint_gate_ids=_strings(payload, "joint_gate_ids"),
            render=RenderContract(
                required_findings=_strings(render, "required_findings"),
                veto_findings=_strings(render, "veto_findings"),
                mask_guarantees=_strings(render, "mask_guarantees", allow_empty=True),
                render_only_claims=_strings(render, "render_only_claims", allow_empty=True),
                required_findings_by_primitive=_string_sequence_mapping(
                    render.get("required_findings_by_primitive", {}),
                    key="render_contract.required_findings_by_primitive",
                ),
                veto_findings_by_primitive=_string_sequence_mapping(
                    render.get("veto_findings_by_primitive", {}),
                    key="render_contract.veto_findings_by_primitive",
                ),
            ),
            evidence_citations=_strings(payload, "evidence_citations"),
            counterexamples=_strings(payload, "counterexamples"),
            source_path=source_path,
        )


@dataclass(frozen=True)
class JointPrimitiveSkill:
    """Intent-level edit semantics independent of cancer realization."""

    primitive_id: str
    version: str
    review_status: str
    scope: str
    summary: str
    tissue_action: str
    budget_mode: str
    allowed_baseline_modes: tuple[str, ...]
    allowed_quota_roles: tuple[str, ...]
    host_tissue_labels: tuple[str, ...]
    target_cell_classes: tuple[int, ...]
    tissue_geometry_mode: str
    allow_source_component_resolution: bool
    allow_target_hole_resolution: bool
    maximum_source_component_changed_fraction: float
    minimum_source_component_remaining_px: int
    required_source_clearance_classes: tuple[int, ...]
    minimum_source_clearance_instances: int
    required_checker_ids: tuple[str, ...]
    source_path: str

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, source_path: str
    ) -> JointPrimitiveSkill:
        status = _string(payload, "review_status")
        if status not in REVIEW_STATUSES:
            raise JointContractError(f"unknown primitive review status: {status}")
        scope = _string(payload, "scope")
        if scope not in PRIMITIVE_SCOPES:
            raise JointContractError(f"unknown joint primitive scope: {scope}")
        tissue_action = _string(payload, "tissue_action")
        if tissue_action not in {"required", "forbidden"}:
            raise JointContractError(
                f"unknown joint primitive tissue action: {tissue_action}"
            )
        if (scope == "cell_only") != (tissue_action == "forbidden"):
            raise JointContractError(
                "cell_only primitives must forbid tissue changes and tissue primitives must require them"
            )
        budget_mode = _string(payload, "budget_mode")
        if budget_mode not in PRIMITIVE_BUDGET_MODES:
            raise JointContractError(f"unknown primitive budget mode: {budget_mode}")
        from phase3_joint_edit_refine.models import (
            CELL_BASELINE_MODES,
            CELL_QUOTA_ROLES,
        )

        baseline_modes = _strings(payload, "allowed_baseline_modes")
        quota_roles = _strings(payload, "allowed_quota_roles")
        if set(baseline_modes) - CELL_BASELINE_MODES:
            raise JointContractError("joint primitive contains unknown cell baseline mode")
        if set(quota_roles) - CELL_QUOTA_ROLES:
            raise JointContractError("joint primitive contains unknown quota role")
        topology = payload.get("tissue_topology_contract", {})
        if not isinstance(topology, Mapping):
            raise JointContractError(
                "joint primitive tissue_topology_contract must be a mapping"
            )
        geometry_mode = str(
            topology.get("geometry_mode", "interface_front")
        )
        if geometry_mode not in TISSUE_GEOMETRY_MODES:
            raise JointContractError(
                f"unknown primitive tissue geometry mode: {geometry_mode}"
            )
        maximum_changed = float(
            topology.get("maximum_source_component_changed_fraction", 0.55)
        )
        minimum_remaining = int(
            topology.get("minimum_source_component_remaining_px", 64)
        )
        minimum_clearance = int(
            payload.get("minimum_source_clearance_instances", 0)
        )
        if not 0.0 < maximum_changed <= 1.0:
            raise JointContractError(
                "maximum source-component changed fraction must lie in (0,1]"
            )
        if minimum_remaining < 0 or minimum_clearance < 0:
            raise JointContractError(
                "primitive source retention/clearance counts must be non-negative"
            )
        return cls(
            primitive_id=_string(payload, "primitive_id"),
            version=_string(payload, "version"),
            review_status=status,
            scope=scope,
            summary=_string(payload, "summary"),
            tissue_action=tissue_action,
            budget_mode=budget_mode,
            allowed_baseline_modes=baseline_modes,
            allowed_quota_roles=quota_roles,
            host_tissue_labels=_strings(
                payload, "host_tissue_labels", allow_empty=True
            ),
            target_cell_classes=_ints(payload, "target_cell_classes"),
            tissue_geometry_mode=geometry_mode,
            allow_source_component_resolution=bool(
                topology.get("allow_source_component_resolution", False)
            ),
            allow_target_hole_resolution=bool(
                topology.get("allow_target_hole_resolution", False)
            ),
            maximum_source_component_changed_fraction=maximum_changed,
            minimum_source_component_remaining_px=minimum_remaining,
            required_source_clearance_classes=_ints(
                payload, "required_source_clearance_classes"
            ),
            minimum_source_clearance_instances=minimum_clearance,
            required_checker_ids=_strings(payload, "required_checker_ids"),
            source_path=source_path,
        )


@dataclass(frozen=True)
class JointProfileContract:
    annotation_profile_id: str
    version: str
    review_status: str
    prohibited_fine_ids: tuple[int, ...]
    prohibit_cell_placement_fine_ids: tuple[int, ...]
    prohibit_generation_support_fine_ids: tuple[int, ...]
    required_provenance_fields: tuple[str, ...]
    unavailable_mechanisms: tuple[str, ...]
    conditional_mechanisms: tuple[str, ...]
    required_checker_ids: tuple[str, ...]
    mechanism_required_fine_ids: dict[str, tuple[int, ...]]
    source_path: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, source_path: str) -> JointProfileContract:
        status = _string(payload, "review_status")
        if status not in REVIEW_STATUSES:
            raise JointContractError(f"unknown profile review status: {status}")
        return cls(
            annotation_profile_id=_string(payload, "annotation_profile_id"),
            version=_string(payload, "version"),
            review_status=status,
            prohibited_fine_ids=_ints(payload, "prohibited_fine_ids"),
            prohibit_cell_placement_fine_ids=_ints(
                payload, "prohibit_cell_placement_fine_ids"
            ),
            prohibit_generation_support_fine_ids=_ints(
                payload, "prohibit_generation_support_fine_ids"
            ),
            required_provenance_fields=_strings(payload, "required_provenance_fields"),
            unavailable_mechanisms=_strings(
                payload, "unavailable_mechanisms", allow_empty=True
            ),
            conditional_mechanisms=_strings(
                payload, "conditional_mechanisms", allow_empty=True
            ),
            required_checker_ids=_strings(payload, "required_checker_ids"),
            mechanism_required_fine_ids={
                str(key): tuple(int(value) for value in values)
                for key, values in payload.get("mechanism_required_fine_ids", {}).items()
            },
            source_path=source_path,
        )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise JointContractError(f"{key} is required and must be a mapping")
    return value


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JointContractError(f"{key} is required and must be a non-empty string")
    return value.strip()


def _strings(payload: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    value = payload.get(key, ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise JointContractError(f"{key} must be a sequence")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if len(result) != len(value) or (not result and not allow_empty):
        raise JointContractError(f"{key} contains empty values")
    return result


def _string_sequence_mapping(value: Any, *, key: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise JointContractError(f"{key} must be a mapping")
    result = {}
    for current_key, items in value.items():
        if not isinstance(current_key, str) or not current_key.strip():
            raise JointContractError(f"{key} contains an empty key")
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise JointContractError(f"{key}.{current_key} must be a sequence")
        normalized = tuple(str(item).strip() for item in items if str(item).strip())
        if not normalized or len(normalized) != len(items):
            raise JointContractError(f"{key}.{current_key} contains empty values")
        result[current_key.strip()] = normalized
    return result


def _ints(payload: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = payload.get(key, ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise JointContractError(f"{key} must be a sequence")
    return tuple(int(item) for item in value)


def _pair(payload: Mapping[str, Any], key: str) -> tuple[int, int]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise JointContractError(f"{key} must contain two integers")
    return int(value[0]), int(value[1])


def _float_pair(value: Any, *, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise JointContractError(f"{name} must contain two numbers")
    result = float(value[0]), float(value[1])
    if not all(isfinite(item) for item in result):
        raise JointContractError(f"{name} must contain finite numbers")
    return result


def _seam_contract(value: Any) -> SeamContract:
    # Existing draft mechanisms remain loadable, but the default itself is a
    # complete, auditable contract rather than the old fixed 25% quota.
    payload = value if isinstance(value, Mapping) else {}
    mode = str(payload.get("mode", "adaptive_population_continuity"))
    if mode not in SEAM_MODES:
        raise JointContractError(f"unknown seam mode: {mode}")
    width = _float_pair(
        payload.get("width_cell_diameters", (1.0, 1.5)),
        name="seam width_cell_diameters",
    )
    quantiles = _float_pair(
        payload.get("reference_area_quantiles", (0.25, 0.75)),
        name="seam reference_area_quantiles",
    )
    density = _float_pair(
        payload.get("density_ratio_range", (0.25, 4.0)),
        name="seam density_ratio_range",
    )
    maximum_empty_run = float(
        payload.get("maximum_empty_run_cell_diameters", 2.0)
    )
    coverage = float(payload.get("minimum_anchor_coverage_fraction", 0.5))
    if not 0.25 <= width[0] <= width[1] <= 4.0:
        raise JointContractError("seam width_cell_diameters is outside [0.25,4]")
    if not 0.0 <= quantiles[0] <= quantiles[1] <= 1.0:
        raise JointContractError("seam reference quantiles must lie in [0,1]")
    if not 0.0 < density[0] <= density[1]:
        raise JointContractError("seam density ratio range is invalid")
    if not 0.5 <= maximum_empty_run <= 6.0:
        raise JointContractError(
            "maximum_empty_run_cell_diameters is outside [0.5,6]"
        )
    if not 0.0 <= coverage <= 1.0:
        raise JointContractError(
            "minimum_anchor_coverage_fraction must lie in [0,1]"
        )
    return SeamContract(
        mode=mode,
        width_cell_diameters=width,
        reference_area_quantiles=quantiles,
        maximum_empty_run_cell_diameters=maximum_empty_run,
        density_ratio_range=density,
        minimum_anchor_coverage_fraction=coverage,
        requires_new_target_cells=bool(
            payload.get("requires_new_target_cells", mode == "adaptive_population_continuity")
        ),
    )


def _cellularity_depletion_contract(
    value: Any, *, required: bool, mechanism_id: str
) -> CellularityDepletionContract | None:
    if value is None:
        if required:
            raise JointContractError(
                f"{mechanism_id} supports cellularity decrease but has no "
                "cellularity_depletion_contract"
            )
        return None
    if not isinstance(value, Mapping):
        raise JointContractError(
            f"{mechanism_id}.cellularity_depletion_contract must be a mapping"
        )
    anchors = _strings(value, "allowed_anchor_types")
    if set(anchors) - {"interface"}:
        raise JointContractError(
            f"{mechanism_id} contains an unsupported depletion anchor type"
        )
    neighbors = _strings(value, "allowed_neighbor_labels")
    numeric = {
        "core_width_cell_diameters": float(
            value.get("core_width_cell_diameters", 1.25)
        ),
        "transition_width_cell_diameters": float(
            value.get("transition_width_cell_diameters", 1.75)
        ),
        "outer_reference_width_cell_diameters": float(
            value.get("outer_reference_width_cell_diameters", 1.50)
        ),
        "core_removal_weight": float(value.get("core_removal_weight", 1.0)),
        "transition_removal_weight": float(
            value.get("transition_removal_weight", 0.45)
        ),
        "core_target_removal_fraction": float(
            value.get("core_target_removal_fraction", 0.55)
        ),
        "transition_start_removal_fraction": float(
            value.get("transition_start_removal_fraction", 0.42)
        ),
        "transition_end_removal_fraction": float(
            value.get("transition_end_removal_fraction", 0.10)
        ),
        "minimum_core_residual_fraction": float(
            value.get("minimum_core_residual_fraction", 0.25)
        ),
        "minimum_transition_residual_fraction": float(
            value.get("minimum_transition_residual_fraction", 0.50)
        ),
        "maximum_new_gap_cell_diameters": float(
            value.get("maximum_new_gap_cell_diameters", 3.0)
        ),
        "minimum_field_area_cell_diameter_squares": float(
            value.get("minimum_field_area_cell_diameter_squares", 60.0)
        ),
    }
    if not all(isfinite(item) and item > 0 for item in numeric.values()):
        raise JointContractError(
            f"{mechanism_id} depletion numeric bounds must be finite and positive"
        )
    if not (
        numeric["core_removal_weight"]
        > numeric["transition_removal_weight"]
    ):
        raise JointContractError(
            f"{mechanism_id} core removal weight must exceed transition weight"
        )
    if not (
        0.0
        < numeric["transition_end_removal_fraction"]
        < numeric["transition_start_removal_fraction"]
        < numeric["core_target_removal_fraction"]
        < 1.0
    ):
        raise JointContractError(
            f"{mechanism_id} depletion target fractions must decrease outward"
        )
    for key in (
        "minimum_core_residual_fraction",
        "minimum_transition_residual_fraction",
    ):
        if not 0.0 < numeric[key] < 1.0:
            raise JointContractError(f"{mechanism_id}.{key} must lie in (0,1)")
    minimum_core = int(value.get("minimum_core_removals", 1))
    minimum_transition = int(value.get("minimum_transition_removals", 1))
    transition_subbands = int(value.get("transition_subband_count", 4))
    minimum_outer = int(value.get("minimum_outer_reference_instances", 3))
    if minimum_core < 1 or minimum_transition < 1:
        raise JointContractError(
            f"{mechanism_id} depletion bands must each remove at least one nucleus"
        )
    if not 2 <= transition_subbands <= 8:
        raise JointContractError(
            f"{mechanism_id} transition_subband_count must lie in [2,8]"
        )
    if minimum_outer < 1:
        raise JointContractError(
            f"{mechanism_id} depletion needs at least one outer reference instance"
        )
    resolution_mode = str(value.get("resolution_mode", "density_field"))
    if resolution_mode != "density_field":
        raise JointContractError(
            f"{mechanism_id} has unsupported depletion resolution mode"
        )
    if numeric["core_target_removal_fraction"] > (
        1.0 - numeric["minimum_core_residual_fraction"] + 1e-9
    ):
        raise JointContractError(
            f"{mechanism_id} core density target violates its residual floor"
        )
    if numeric["transition_start_removal_fraction"] > (
        1.0 - numeric["minimum_transition_residual_fraction"] + 1e-9
    ):
        raise JointContractError(
            f"{mechanism_id} transition density target violates its residual floor"
        )
    return CellularityDepletionContract(
        program_id=_string(value, "program_id"),
        resolution_mode=resolution_mode,
        allowed_anchor_types=anchors,
        allowed_neighbor_labels=neighbors,
        minimum_core_removals=minimum_core,
        minimum_transition_removals=minimum_transition,
        transition_subband_count=transition_subbands,
        minimum_outer_reference_instances=minimum_outer,
        **numeric,
    )


def _tissue_front_contract(
    payload: Mapping[str, Any], *, mechanism_id: str
) -> TissueFrontContract:
    """Parse executable front geometry, with a conservative legacy default."""

    mode = str(payload.get("profile_mode", "multi_lobe"))
    if mode not in {"tapered_lobe", "uniform_front", "multi_lobe"}:
        raise JointContractError(
            f"{mechanism_id} has unsupported tissue front profile: {mode}"
        )
    edge = float(payload.get("edge_depth_ratio", 0.10))
    taper = float(payload.get("taper_fraction", 0.42))
    lobe_count = int(payload.get("lobe_count", 3))
    noise = float(payload.get("noise_depth_ratio", 0.30))
    maximum_band = int(payload.get("maximum_band_px", 128))
    depth_span = float(payload.get("maximum_depth_span_ratio", 1.25))
    if not 0.0 <= edge <= 1.0:
        raise JointContractError("tissue front edge_depth_ratio must lie in [0,1]")
    if not 0.0 <= taper <= 0.5:
        raise JointContractError("tissue front taper_fraction must lie in [0,0.5]")
    if not 1 <= lobe_count <= 3:
        raise JointContractError("tissue front lobe_count must lie in [1,3]")
    if not 0.0 <= noise <= 1.0:
        raise JointContractError("tissue front noise_depth_ratio must lie in [0,1]")
    if not 1 <= maximum_band <= 256:
        raise JointContractError("tissue front maximum_band_px must lie in [1,256]")
    if not 0.25 <= depth_span <= 2.0:
        raise JointContractError(
            "tissue front maximum_depth_span_ratio must lie in [0.25,2.0]"
        )
    if mode == "uniform_front":
        edge = 1.0
        taper = 0.0
        lobe_count = 1
    return TissueFrontContract(
        profile_mode=mode,
        edge_depth_ratio=edge,
        taper_fraction=taper,
        lobe_count=lobe_count,
        noise_depth_ratio=noise,
        maximum_band_px=maximum_band,
        maximum_depth_span_ratio=depth_span,
    )


def _primitive_label_contracts(payload: Mapping[str, Any], key: str) -> dict[str, dict[str, tuple[str, ...]]]:
    raw = _mapping(payload, key)
    result = {}
    for primitive_id, value in raw.items():
        if not isinstance(primitive_id, str) or not primitive_id or not isinstance(value, Mapping):
            raise JointContractError("primitive label contracts are malformed")
        result[primitive_id] = {
            "source_labels": _strings(value, "source_labels"),
            "target_labels": _strings(value, "target_labels"),
        }
    return result
