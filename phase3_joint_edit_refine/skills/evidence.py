"""Fail-closed evidence governance for joint pathology skills.

The executable contract deliberately separates four authorities:

* pathology facts describe biological/histologic mechanisms;
* dataset facts describe what an annotation protocol actually labels;
* engineering proxies are measurable approximations implemented by tools/gates;
* model representability records what the frozen condition/generator stack can
  demonstrably realize.

None of these categories can authorize a claim owned by another category.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phase3_joint_edit_refine.models import JointContractError

EVIDENCE_CATEGORIES = frozenset(
    {
        "pathology_fact",
        "dataset_fact",
        "engineering_proxy",
        "model_representability",
    }
)

_MECHANISM_FIELDS = {
    "mechanism_id": "metadata",
    "pathology_domain_id": "metadata",
    "supported_primitives": "metadata",
    "version": "metadata",
    "review_status": "metadata",
    "summary": "pathology_fact",
    "recognition_contract": "pathology_fact",
    "planner_policy": "engineering_proxy",
    "gland_authority_contract": "engineering_proxy",
    "representability_contract": "model_representability",
    "tissue_program": "engineering_proxy",
    "cell_program": "engineering_proxy",
    "coupling_contract": "engineering_proxy",
    "joint_gate_ids": "engineering_proxy",
    "render_contract": "model_representability",
    # Retained for backward-compatible human display. The v2 source registry,
    # not this legacy list, is the evidence authority.
    "evidence_citations": "metadata",
    "counterexamples": "pathology_fact",
}

_PRIMITIVE_FIELDS = {
    "primitive_id": "metadata",
    "version": "metadata",
    "review_status": "metadata",
    "scope": "engineering_proxy",
    "summary": "engineering_proxy",
    "tissue_action": "engineering_proxy",
    "budget_mode": "engineering_proxy",
    "allowed_baseline_modes": "engineering_proxy",
    "allowed_quota_roles": "engineering_proxy",
    "host_tissue_labels": "engineering_proxy",
    "target_cell_classes": "engineering_proxy",
    "tissue_topology_contract": "engineering_proxy",
    "required_source_clearance_classes": "engineering_proxy",
    "minimum_source_clearance_instances": "engineering_proxy",
    "cell_effect_contract": "engineering_proxy",
    "required_checker_ids": "engineering_proxy",
}

_PROFILE_FIELDS = {
    "annotation_profile_id": "metadata",
    "version": "metadata",
    "review_status": "metadata",
    "prohibited_fine_ids": "dataset_fact",
    "prohibit_cell_placement_fine_ids": "dataset_fact",
    "prohibit_generation_support_fine_ids": "dataset_fact",
    "required_provenance_fields": "engineering_proxy",
    "gland_instance_authority_policy": "engineering_proxy",
    "unavailable_mechanisms": "dataset_fact",
    "conditional_mechanisms": "dataset_fact",
    "required_checker_ids": "engineering_proxy",
    "mechanism_required_fine_ids": "dataset_fact",
    "supports_explicit_stroma": "dataset_fact",
    "protected_fine_ids": "engineering_proxy",
    "operational_stroma_fine_ids": "engineering_proxy",
    "operational_stroma_policy": "engineering_proxy",
    "fibrosis_claim_authorized": "dataset_fact",
    "mechanism_editable_source_fine_ids": "engineering_proxy",
    "mechanism_editable_target_fine_ids": "engineering_proxy",
    "visual_veto_requirements": "engineering_proxy",
}


@dataclass(frozen=True)
class SkillEvidenceStatus:
    kind: str
    skill_id: str
    field_categories: dict[str, str]
    source_ids: tuple[str, ...]
    category_status: dict[str, str]
    production_allowed: bool
    gaps: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "skill_id": self.skill_id,
            "field_categories": dict(self.field_categories),
            "source_ids": list(self.source_ids),
            "category_status": dict(self.category_status),
            "production_allowed": self.production_allowed,
            "gaps": list(self.gaps),
        }


class EvidenceGovernance:
    """Validate category ownership and evidence-source bindings at startup."""

    def __init__(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise JointContractError(
                f"could not load joint evidence policy {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise JointContractError("joint evidence policy root must be an object")
        if payload.get("schema_version") != "joint-evidence-governance-v2":
            raise JointContractError("unsupported joint evidence governance schema")
        if set(payload.get("categories", ())) != EVIDENCE_CATEGORIES:
            raise JointContractError(
                "joint evidence policy must declare exactly the four authority categories"
            )
        sources = payload.get("sources")
        if not isinstance(sources, dict) or not sources:
            raise JointContractError("joint evidence source registry is empty")
        self.path = path
        self.sources = self._validate_sources(sources)
        self.mechanism_pathology_sources = self._source_binding_map(
            payload, "mechanism_pathology_sources", "pathology_fact"
        )
        self.profile_dataset_sources = self._source_binding_map(
            payload, "profile_dataset_sources", "dataset_fact"
        )
        self.engineering_source_ids = self._source_ids(
            payload.get("engineering_source_ids"),
            category="engineering_proxy",
            field="engineering_source_ids",
        )
        self.model_source_ids = self._source_ids(
            payload.get("model_representability_source_ids"),
            category="model_representability",
            field="model_representability_source_ids",
        )

    @staticmethod
    def _validate_sources(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for source_id, value in raw.items():
            if not isinstance(source_id, str) or not source_id or not isinstance(value, dict):
                raise JointContractError("invalid evidence source entry")
            category = value.get("category")
            if category not in EVIDENCE_CATEGORIES:
                raise JointContractError(
                    f"evidence source {source_id} has an invalid category"
                )
            for key in ("title", "uri", "locator", "verification_status"):
                if not isinstance(value.get(key), str) or not value[key].strip():
                    raise JointContractError(
                        f"evidence source {source_id} is missing {key}"
                    )
            if value["verification_status"] not in {
                "verified",
                "pending_internal_validation",
            }:
                raise JointContractError(
                    f"evidence source {source_id} has an invalid verification status"
                )
            result[source_id] = dict(value)
        return result

    def _source_ids(
        self, value: Any, *, category: str, field: str
    ) -> tuple[str, ...]:
        if not isinstance(value, list) or not value:
            raise JointContractError(f"{field} must be a non-empty list")
        result = tuple(str(item) for item in value)
        for source_id in result:
            source = self.sources.get(source_id)
            if source is None:
                raise JointContractError(f"unknown evidence source: {source_id}")
            if source["category"] != category:
                raise JointContractError(
                    f"evidence source {source_id} cannot authorize {category}"
                )
        return result

    def _source_binding_map(
        self, payload: Mapping[str, Any], field: str, category: str
    ) -> dict[str, tuple[str, ...]]:
        raw = payload.get(field)
        if not isinstance(raw, dict) or not raw:
            raise JointContractError(f"{field} must be a non-empty object")
        return {
            str(key): self._source_ids(value, category=category, field=f"{field}.{key}")
            for key, value in raw.items()
        }

    def audit_mechanism(self, payload: Mapping[str, Any]) -> SkillEvidenceStatus:
        mechanism_id = str(payload.get("mechanism_id") or "")
        pathology_sources = self.mechanism_pathology_sources.get(mechanism_id)
        if not pathology_sources:
            raise JointContractError(
                f"mechanism {mechanism_id} has no mechanism-specific pathology evidence binding"
            )
        return self._audit(
            kind="joint-mechanism",
            skill_id=mechanism_id,
            payload=payload,
            classification=_MECHANISM_FIELDS,
            sources={
                "pathology_fact": pathology_sources,
                "engineering_proxy": self.engineering_source_ids,
                "model_representability": self.model_source_ids,
            },
        )

    def validate_catalog_coverage(
        self,
        *,
        mechanism_ids: set[str],
        annotation_profile_ids: set[str],
    ) -> None:
        mechanism_gap = sorted(
            mechanism_ids ^ set(self.mechanism_pathology_sources)
        )
        profile_gap = sorted(
            annotation_profile_ids ^ set(self.profile_dataset_sources)
        )
        if mechanism_gap or profile_gap:
            raise JointContractError(
                "evidence governance must bind every catalog skill exactly once; "
                f"mechanism mismatch={mechanism_gap}, profile mismatch={profile_gap}"
            )

    def audit_primitive(self, payload: Mapping[str, Any]) -> SkillEvidenceStatus:
        primitive_id = str(payload.get("primitive_id") or "")
        return self._audit(
            kind="edit-primitive",
            skill_id=primitive_id,
            payload=payload,
            classification=_PRIMITIVE_FIELDS,
            sources={"engineering_proxy": self.engineering_source_ids},
        )

    def audit_profile(self, payload: Mapping[str, Any]) -> SkillEvidenceStatus:
        profile_id = str(payload.get("annotation_profile_id") or "")
        dataset_sources = self.profile_dataset_sources.get(profile_id)
        if not dataset_sources:
            raise JointContractError(
                f"annotation profile {profile_id} has no dataset evidence binding"
            )
        return self._audit(
            kind="annotation-profile",
            skill_id=profile_id,
            payload=payload,
            classification=_PROFILE_FIELDS,
            sources={
                "dataset_fact": dataset_sources,
                "engineering_proxy": self.engineering_source_ids,
            },
        )

    def _audit(
        self,
        *,
        kind: str,
        skill_id: str,
        payload: Mapping[str, Any],
        classification: Mapping[str, str],
        sources: Mapping[str, tuple[str, ...]],
    ) -> SkillEvidenceStatus:
        if not skill_id:
            raise JointContractError(f"{kind} evidence audit has no skill ID")
        unknown_fields = sorted(set(payload) - set(classification))
        if unknown_fields:
            raise JointContractError(
                f"{kind} {skill_id} has unclassified rule fields: "
                + ", ".join(unknown_fields)
            )
        missing_fields = sorted(set(classification) - set(payload))
        # Optional primitive fields are still classified but may be absent.
        optional = {
            "planner_policy",
            "gland_authority_contract",
            "gland_instance_authority_policy",
            "tissue_topology_contract",
            "required_source_clearance_classes",
            "minimum_source_clearance_instances",
            "cell_effect_contract",
            "mechanism_required_fine_ids",
            "protected_fine_ids",
            "operational_stroma_fine_ids",
            "operational_stroma_policy",
            "fibrosis_claim_authorized",
            "mechanism_editable_source_fine_ids",
            "mechanism_editable_target_fine_ids",
            "visual_veto_requirements",
        }
        missing_fields = [item for item in missing_fields if item not in optional]
        if missing_fields:
            raise JointContractError(
                f"{kind} {skill_id} has missing classified fields: "
                + ", ".join(missing_fields)
            )
        used_categories = {
            classification[field]
            for field in payload
            if classification[field] != "metadata"
        }
        gaps: list[str] = []
        category_status: dict[str, str] = {}
        source_ids: list[str] = []
        for category in sorted(used_categories):
            bound = sources.get(category, ())
            if not bound:
                raise JointContractError(
                    f"{kind} {skill_id} has no evidence source for {category}"
                )
            source_ids.extend(bound)
            pending = [
                source_id
                for source_id in bound
                if self.sources[source_id]["verification_status"]
                != "verified"
            ]
            if pending:
                category_status[category] = "pending_internal_validation"
                gaps.append(
                    f"{category} pending: " + ", ".join(sorted(pending))
                )
            else:
                category_status[category] = "verified_source_binding"
        # Evidence binding does not substitute for the skill's internal review.
        if payload.get("review_status") != "internally_reviewed":
            gaps.append("skill has not completed internal pathology/engineering review")
        production_allowed = not gaps
        return SkillEvidenceStatus(
            kind=kind,
            skill_id=skill_id,
            field_categories={
                field: classification[field] for field in payload
            },
            source_ids=tuple(dict.fromkeys(source_ids)),
            category_status=category_status,
            production_allowed=production_allowed,
            gaps=tuple(gaps),
        )

    def validate_local_resource(
        self,
        payload: Mapping[str, Any],
        *,
        expected: SkillEvidenceStatus,
    ) -> None:
        """Require every non-metadata contract field to be bound exactly once."""

        if payload.get("schema_version") != "joint-skill-evidence-v2":
            raise JointContractError(
                f"{expected.kind} {expected.skill_id} has legacy/invalid evidence schema"
            )
        if payload.get("skill_kind") != expected.kind or payload.get(
            "skill_id"
        ) != expected.skill_id:
            raise JointContractError(
                f"local evidence identity mismatch for {expected.skill_id}"
            )
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise JointContractError(
                f"local evidence records are empty for {expected.skill_id}"
            )
        expected_fields = {
            field: category
            for field, category in expected.field_categories.items()
            if category != "metadata"
        }
        claimed: dict[str, str] = {}
        evidence_ids: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise JointContractError(
                    f"invalid local evidence record for {expected.skill_id}"
                )
            evidence_id = record.get("evidence_id")
            category = record.get("authority_category")
            scopes = record.get("claim_scope")
            source_ids = record.get("source_ids")
            if not isinstance(evidence_id, str) or not evidence_id:
                raise JointContractError(
                    f"local evidence record has no ID for {expected.skill_id}"
                )
            if evidence_id in evidence_ids:
                raise JointContractError(
                    f"duplicate evidence ID {evidence_id}"
                )
            evidence_ids.add(evidence_id)
            if category not in EVIDENCE_CATEGORIES:
                raise JointContractError(
                    f"local evidence record {evidence_id} has invalid authority"
                )
            if not isinstance(scopes, list) or not scopes:
                raise JointContractError(
                    f"local evidence record {evidence_id} has no claim scope"
                )
            normalized_sources = self._source_ids(
                source_ids,
                category=category,
                field=f"{expected.skill_id}.{evidence_id}.source_ids",
            )
            if not normalized_sources:
                raise JointContractError(
                    f"local evidence record {evidence_id} has no sources"
                )
            for field in scopes:
                if not isinstance(field, str) or field not in expected_fields:
                    raise JointContractError(
                        f"local evidence {evidence_id} claims unknown field {field}"
                    )
                if expected_fields[field] != category:
                    raise JointContractError(
                        f"local evidence {evidence_id} uses {category} for "
                        f"{field}, which is owned by {expected_fields[field]}"
                    )
                if field in claimed:
                    raise JointContractError(
                        f"contract field {field} has multiple evidence authorities"
                    )
                claimed[field] = evidence_id
        missing = sorted(set(expected_fields) - set(claimed))
        if missing:
            raise JointContractError(
                f"{expected.skill_id} has unbound contract fields: "
                + ", ".join(missing)
            )
