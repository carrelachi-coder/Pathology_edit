"""Strict runtime representation of evidence-backed skill packages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from phase3_mask_edit_refine.models import RefineContractError

SKILL_KINDS = frozenset({"pathology_domain", "annotation_profile", "edit_primitive"})
REVIEW_STATUSES = frozenset({"draft", "empirically_validated", "internally_reviewed"})
SEVERITIES = frozenset({"hard", "soft", "advisory"})
MASK_ENFORCEMENT_LEVELS = frozenset(
    {"deterministic", "conditional", "planner_veto"}
)
MASK_OBSERVABILITY_KINDS = frozenset(
    {
        "provenance",
        "semantic_mask",
        "native_annotation",
        "scene_graph",
        "auxiliary_structure_map",
        "source_he",
    }
)
MASK_ENFORCEMENT_STAGES = frozenset(
    {"input_validation", "planner", "candidate_generation", "mask_gate"}
)
EXECUTION_RULE_ROLES = frozenset(
    {
        "deterministic_mask_invariant",
        "provenance_precondition",
        "semantic_capability_precondition",
        "profile_auxiliary_selection_preference",
        "certified_candidate_selection_preference",
    }
)
EXECUTION_OBSERVATION_SOURCES = frozenset(
    {
        "instruction_semantic_intent",
        "tissue_mask",
        "nuclei_mask",
        "scene_graph",
        "profile_owned_auxiliary_map",
        "case_provenance",
        "candidate_certificate",
        "deterministic_metric",
    }
)


@dataclass(frozen=True)
class ObservationAuthority:
    """One positive, typed binding for an execution-time observation."""

    source: str
    binding: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ObservationAuthority:
        if not isinstance(payload, Mapping):
            raise RefineContractError("observation_authority entries must be mappings")
        source = _required_string(payload, "source")
        if source not in EXECUTION_OBSERVATION_SOURCES:
            raise RefineContractError(
                f"unknown execution observation authority source: {source}"
            )
        return cls(
            source=source,
            binding=_required_string(payload, "binding"),
        )


@dataclass(frozen=True)
class KnowledgeRule:
    rule_id: str
    scope: str
    applies_when: dict[str, Any]
    claim: str
    required_observation: str
    severity: str
    deterministic_check_id: str | None
    critic_requirement: str | None
    exceptions: tuple[str, ...]
    counterexamples: tuple[str, ...]
    evidence_citations: tuple[str, ...]
    dataset_statistics: dict[str, Any]
    version: str
    review_status: str
    known_limitations: tuple[str, ...]
    expected_morphology: tuple[str, ...]
    forbidden_morphology: tuple[str, ...]
    execution_role: str | None
    observation_authority: tuple[ObservationAuthority, ...]
    selection_preference: str | None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> KnowledgeRule:
        severity = _required_string(payload, "severity")
        if severity not in SEVERITIES:
            raise RefineContractError(f"unknown rule severity: {severity}")
        status = _required_string(payload, "review_status")
        if status not in REVIEW_STATUSES:
            raise RefineContractError(f"unknown rule review status: {status}")
        deterministic = _optional_string(payload.get("deterministic_check_id"))
        critic = _optional_string(payload.get("critic_requirement"))
        execution_role = _optional_string(payload.get("execution_role"))
        if execution_role is not None and execution_role not in EXECUTION_RULE_ROLES:
            raise RefineContractError(
                f"unknown execution rule role: {execution_role}"
            )
        raw_authority = payload.get("observation_authority", ())
        if not isinstance(raw_authority, (list, tuple)):
            raise RefineContractError("observation_authority must be a list")
        observation_authority = tuple(
            ObservationAuthority.from_mapping(item) for item in raw_authority
        )
        authority_pairs = [
            (item.source, item.binding) for item in observation_authority
        ]
        if len(authority_pairs) != len(set(authority_pairs)):
            raise RefineContractError("observation_authority contains duplicates")
        selection_preference = _optional_string(
            payload.get("selection_preference")
        )
        if selection_preference is not None and execution_role not in {
            "profile_auxiliary_selection_preference",
            "certified_candidate_selection_preference",
        }:
            raise RefineContractError(
                "selection_preference requires an execution selection-preference role"
            )
        if severity == "hard" and not (deterministic or critic):
            raise RefineContractError(
                f"hard rule {_required_string(payload, 'rule_id')} has no checker or critic veto"
            )
        applies = payload.get("applies_when", {})
        statistics = payload.get("dataset_statistics", {})
        if not isinstance(applies, Mapping) or not isinstance(statistics, Mapping):
            raise RefineContractError("applies_when and dataset_statistics must be mappings")
        return cls(
            rule_id=_required_string(payload, "rule_id"),
            scope=_required_string(payload, "scope"),
            applies_when=dict(applies),
            claim=_required_string(payload, "claim"),
            required_observation=_required_string(payload, "required_observation"),
            severity=severity,
            deterministic_check_id=deterministic,
            critic_requirement=critic,
            exceptions=_strings(payload.get("exceptions", ()), "exceptions"),
            counterexamples=_strings(payload.get("counterexamples", ()), "counterexamples"),
            evidence_citations=_strings(
                payload.get("evidence_citations", ()), "evidence_citations"
            ),
            dataset_statistics=dict(statistics),
            version=_required_string(payload, "version"),
            review_status=status,
            known_limitations=_strings(
                payload.get("known_limitations", ()), "known_limitations"
            ),
            expected_morphology=_strings(
                payload.get("expected_morphology", ()), "expected_morphology"
            ),
            forbidden_morphology=_strings(
                payload.get("forbidden_morphology", ()), "forbidden_morphology"
            ),
            execution_role=execution_role,
            observation_authority=observation_authority,
            selection_preference=selection_preference,
        )

    def to_execution_metadata(self) -> dict[str, Any]:
        """Return only typed authority, never free-form pathology prose."""

        payload: dict[str, Any] = {
            "rule_id": self.rule_id,
            "scope": self.scope,
            "applies_when": dict(self.applies_when),
            "severity": self.severity,
            "deterministic_check_id": self.deterministic_check_id,
            "execution_role": self.execution_role,
            "observation_authority": [
                asdict(item) for item in self.observation_authority
            ],
        }
        if self.selection_preference is not None:
            payload["selection_preference"] = self.selection_preference
        return payload


@dataclass(frozen=True)
class MaskConstraint:
    """A constraint whose mask-stage observability and enforcement are explicit."""

    constraint_id: str
    applies_when: dict[str, Any]
    mask_statement: str
    observability: tuple[str, ...]
    enforcement: str
    enforcement_stages: tuple[str, ...]
    checker_ids: tuple[str, ...]
    critic_requirement: str | None
    required_inputs: tuple[str, ...]
    failure_action: str
    generation_handoff: tuple[str, ...]
    known_limitations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> MaskConstraint:
        enforcement = _required_string(payload, "enforcement")
        if enforcement not in MASK_ENFORCEMENT_LEVELS:
            raise RefineContractError(
                f"unknown mask enforcement level: {enforcement}"
            )
        observability = _strings(payload.get("observability"), "observability")
        unknown_observability = sorted(
            set(observability) - MASK_OBSERVABILITY_KINDS
        )
        if not observability or unknown_observability:
            raise RefineContractError(
                "mask constraint observability is empty or unknown: "
                + ", ".join(unknown_observability)
            )
        stages = _strings(payload.get("enforcement_stages"), "enforcement_stages")
        unknown_stages = sorted(set(stages) - MASK_ENFORCEMENT_STAGES)
        if not stages or unknown_stages:
            raise RefineContractError(
                "mask constraint enforcement stages are empty or unknown: "
                + ", ".join(unknown_stages)
            )
        checker_ids = _strings(payload.get("checker_ids", ()), "checker_ids")
        critic = _optional_string(payload.get("critic_requirement"))
        if enforcement == "deterministic" and not checker_ids:
            raise RefineContractError(
                f"deterministic mask constraint "
                f"{_required_string(payload, 'constraint_id')} has no checker"
            )
        if enforcement in {"conditional", "planner_veto"} and not (
            checker_ids or critic
        ):
            raise RefineContractError(
                f"conditional mask constraint "
                f"{_required_string(payload, 'constraint_id')} has no checker or critic veto"
            )
        applies = payload.get("applies_when", {})
        if not isinstance(applies, Mapping):
            raise RefineContractError("mask constraint applies_when must be a mapping")
        failure_action = _required_string(payload, "failure_action")
        if failure_action not in {"reject_candidate", "abstain_case"}:
            raise RefineContractError(
                f"unknown mask constraint failure_action: {failure_action}"
            )
        return cls(
            constraint_id=_required_string(payload, "constraint_id"),
            applies_when=dict(applies),
            mask_statement=_required_string(payload, "mask_statement"),
            observability=observability,
            enforcement=enforcement,
            enforcement_stages=stages,
            checker_ids=checker_ids,
            critic_requirement=critic,
            required_inputs=_strings(
                payload.get("required_inputs", ()), "required_inputs"
            ),
            failure_action=failure_action,
            generation_handoff=_strings(
                payload.get("generation_handoff", ()), "generation_handoff"
            ),
            known_limitations=_strings(
                payload.get("known_limitations", ()), "known_limitations"
            ),
        )


@dataclass(frozen=True)
class SkillPackage:
    skill_id: str
    skill_kind: str
    version: str
    review_status: str
    summary: str
    capabilities: dict[str, Any]
    rules: tuple[KnowledgeRule, ...]
    mask_constraints: tuple[MaskConstraint, ...]
    source_path: str

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, source_path: str
    ) -> SkillPackage:
        kind = _required_string(payload, "skill_kind")
        if kind not in SKILL_KINDS:
            raise RefineContractError(f"unknown skill kind: {kind}")
        status = _required_string(payload, "review_status")
        if status not in REVIEW_STATUSES:
            raise RefineContractError(f"unknown skill review status: {status}")
        capabilities = payload.get("capabilities", {})
        raw_rules = payload.get("rules", [])
        raw_constraints = payload.get("mask_constraints", [])
        if (
            not isinstance(capabilities, Mapping)
            or not isinstance(raw_rules, list)
            or not isinstance(raw_constraints, list)
        ):
            raise RefineContractError("skill capabilities must be a mapping and rules a list")
        rules = tuple(KnowledgeRule.from_mapping(rule) for rule in raw_rules)
        constraints = tuple(
            MaskConstraint.from_mapping(item) for item in raw_constraints
        )
        rule_ids = [rule.rule_id for rule in rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise RefineContractError("skill contains duplicate rule_id values")
        constraint_ids = [item.constraint_id for item in constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise RefineContractError("skill contains duplicate mask constraint IDs")
        if set(rule_ids) & set(constraint_ids):
            raise RefineContractError("rule and mask constraint IDs must be distinct")
        return cls(
            skill_id=_required_string(payload, "skill_id"),
            skill_kind=kind,
            version=_required_string(payload, "version"),
            review_status=status,
            summary=_required_string(payload, "summary"),
            capabilities=dict(capabilities),
            rules=rules,
            mask_constraints=constraints,
            source_path=source_path,
        )


@dataclass(frozen=True)
class ResolvedEditContract:
    primitive_id: str
    source_label_options: tuple[str, ...]
    target_label: str
    allowed_tools: tuple[str, ...]
    required_check_ids: tuple[str, ...]


@dataclass(frozen=True)
class ActiveKnowledgeBundle:
    pathology_domain: SkillPackage
    annotation_profile: SkillPackage
    edit_primitive: SkillPackage
    edit_contract: ResolvedEditContract
    active_rules: tuple[KnowledgeRule, ...]
    active_mask_constraints: tuple[MaskConstraint, ...]
    warnings: tuple[str, ...]

    @property
    def review_statuses(self) -> tuple[str, ...]:
        return (
            self.pathology_domain.review_status,
            self.annotation_profile.review_status,
            self.edit_primitive.review_status,
        )

    def to_metadata(self) -> dict[str, Any]:
        non_breast_execution = bool(
            self.pathology_domain.skill_id != "breast-invasive-carcinoma-v1"
            or self.annotation_profile.skill_id != "bcss-semantic-v1"
        )
        if non_breast_execution:
            constraint_metadata = [
                {
                    "constraint_id": item.constraint_id,
                    "applies_when": dict(item.applies_when),
                    "observability": list(item.observability),
                    "enforcement": item.enforcement,
                    "enforcement_stages": list(item.enforcement_stages),
                    "checker_ids": list(item.checker_ids),
                    "required_inputs": list(item.required_inputs),
                    "failure_action": item.failure_action,
                }
                for item in self.active_mask_constraints
            ]
        else:
            constraint_metadata = [
                asdict(item) for item in self.active_mask_constraints
            ]
        return {
            "pathology_domain_id": self.pathology_domain.skill_id,
            "annotation_profile_id": self.annotation_profile.skill_id,
            "edit_primitive_id": self.edit_primitive.skill_id,
            "review_statuses": list(self.review_statuses),
            "edit_contract": asdict(self.edit_contract),
            # This metadata is consumed by execution Planners.  Pathology prose,
            # expected morphology, and reader-only facts belong to a separate
            # post-generation reader surface and are deliberately absent here.
            "active_rules": [
                rule.to_execution_metadata() for rule in self.active_rules
            ],
            "active_mask_constraints": constraint_metadata,
            "warnings": list(self.warnings),
            "source_paths": {
                "pathology_domain": self.pathology_domain.source_path,
                "annotation_profile": self.annotation_profile.source_path,
                "edit_primitive": self.edit_primitive.source_path,
            },
        }


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RefineContractError(f"{key} is required and must be a non-empty string")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RefineContractError("optional string fields must be non-empty when provided")
    return value.strip()


def _strings(value: Any, key: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple)):
        values = tuple(value)
    else:
        raise RefineContractError(f"{key} must be a string or list")
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise RefineContractError(f"{key} must contain non-empty strings")
    return tuple(item.strip() for item in values)
