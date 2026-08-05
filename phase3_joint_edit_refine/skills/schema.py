"""Typed runtime schema for atomic tissue--cell mechanism skills."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from phase3_joint_edit_refine.models import CELL_ACTIONS, LAYOUT_PROGRAMS, JointContractError

SUPPORT_STATUSES = frozenset(
    {"supported", "conditionally_supported", "render_only", "unsupported"}
)
REVIEW_STATUSES = frozenset({"draft", "empirically_validated", "internally_reviewed"})
PRIMITIVE_SCOPES = frozenset({"tissue_and_cell", "cell_only"})
PRIMITIVE_BUDGET_MODES = frozenset({"joint_area_with_tissue_floor", "count_extent"})


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


@dataclass(frozen=True)
class CellProgramContract:
    actions: tuple[str, ...]
    allowed_cell_classes: tuple[int, ...]
    layout_programs: tuple[str, ...]
    core_policy: str
    halo_policy: str
    halo_distance_px: tuple[int, int]
    cluster_size_range: tuple[int, int]
    required_checker_ids: tuple[str, ...]


@dataclass(frozen=True)
class CouplingContract:
    compatibility_rule_ids: tuple[str, ...]
    allow_neoplastic_in_non_tumor_tissue: bool
    joint_area_mode: str
    tissue_floor_applies: bool
    cell_only_target_fraction: float
    render_support_policy_id: str


@dataclass(frozen=True)
class RenderContract:
    required_findings: tuple[str, ...]
    veto_findings: tuple[str, ...]
    mask_guarantees: tuple[str, ...]
    render_only_claims: tuple[str, ...]


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
            ),
            cell_program=CellProgramContract(
                actions=actions,
                allowed_cell_classes=allowed_classes,
                layout_programs=layouts,
                core_policy=_string(cell, "core_policy"),
                halo_policy=_string(cell, "halo_policy"),
                halo_distance_px=halo,
                cluster_size_range=cluster,
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
                render_support_policy_id=_string(coupling, "render_support_policy_id"),
            ),
            joint_gate_ids=_strings(payload, "joint_gate_ids"),
            render=RenderContract(
                required_findings=_strings(render, "required_findings"),
                veto_findings=_strings(render, "veto_findings"),
                mask_guarantees=_strings(render, "mask_guarantees", allow_empty=True),
                render_only_claims=_strings(render, "render_only_claims", allow_empty=True),
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
    required_checker_ids: tuple[str, ...]
    source_path: str

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, source_path: str
    ) -> "JointPrimitiveSkill":
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
