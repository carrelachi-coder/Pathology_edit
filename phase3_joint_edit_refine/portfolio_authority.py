"""Canonical runtime authority bindings for pre-LLM candidate portfolios."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phase3_mask_edit_refine.models import CaseContext
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle

from .models import JointCaseContext, JointContractError
from .skills.repository import JointSkillBundle

PORTFOLIO_AUTHORITY_SCHEMA_VERSION = "joint-portfolio-authority-v3"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "array_dtype": str(array.dtype),
            "array_shape": list(array.shape),
            "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if is_dataclass(value):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical_metadata_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _canonical_value(value),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_source_asset_digests(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete declared source-digest set with one tissue alias.

    ``CaseContext`` adds ``source_mask_sha256`` as a compatibility alias for
    ``JointCaseContext.source_tissue_mask_sha256``. The alias is normalized,
    but every other digest-bearing key is retained. Missing keys therefore
    change the exact authority binding instead of passing a subset check.
    """

    result = {
        str(key): _canonical_value(value)
        for key, value in provenance.items()
        if str(key).endswith(("sha256", "digest"))
    }
    legacy = result.pop("source_mask_sha256", None)
    canonical = result.get("source_tissue_mask_sha256")
    if legacy is not None:
        if canonical is not None and canonical != legacy:
            raise JointContractError(
                "source tissue digest aliases disagree in runtime provenance"
            )
        result["source_tissue_mask_sha256"] = legacy
    return dict(sorted(result.items()))


def _joint_skill_identity(bundle: JointSkillBundle) -> dict[str, Any]:
    return {
        "full_bundle_sha256": canonical_metadata_sha256(bundle),
        "primitive_id": bundle.primitive.primitive_id,
        "primitive_version": bundle.primitive.version,
        "mechanism_id": bundle.mechanism.mechanism_id,
        "mechanism_version": bundle.mechanism.version,
        "annotation_profile_id": bundle.annotation_profile.annotation_profile_id,
        "annotation_profile_version": bundle.annotation_profile.version,
        "cell_observation_profile_id": bundle.cell_observation_profile.profile_id,
        "cell_observation_profile_version": bundle.cell_observation_profile.version,
        "cell_population_profile_id": bundle.cell_population_profile.profile_id,
        "cell_population_profile_version": bundle.cell_population_profile.version,
        "active_rule_ids": list(bundle.active_rule_ids),
        "required_checker_ids": list(bundle.required_checker_ids),
        "planner_policy_sha256": canonical_metadata_sha256(
            bundle.mechanism.planner_policy
        ),
        "tissue_tool_policy_sha256": canonical_metadata_sha256(
            bundle.mechanism.tissue_program
        ),
        "cell_tool_policy_sha256": canonical_metadata_sha256(
            bundle.mechanism.cell_program
        ),
    }


def _tissue_skill_identity(bundle: ActiveKnowledgeBundle) -> dict[str, Any]:
    return {
        "full_bundle_sha256": canonical_metadata_sha256(bundle),
        "pathology_domain_id": bundle.pathology_domain.skill_id,
        "pathology_domain_version": bundle.pathology_domain.version,
        "annotation_profile_id": bundle.annotation_profile.skill_id,
        "annotation_profile_version": bundle.annotation_profile.version,
        "edit_primitive_id": bundle.edit_primitive.skill_id,
        "edit_primitive_version": bundle.edit_primitive.version,
        "active_rule_ids": [item.rule_id for item in bundle.active_rules],
        "required_checker_ids": list(bundle.edit_contract.required_check_ids),
        "allowed_tools": list(bundle.edit_contract.allowed_tools),
    }


def build_tissue_portfolio_authority_binding(
    *,
    joint_case: JointCaseContext,
    tissue_case: CaseContext,
    source_tissue: np.ndarray,
    joint_bundle: JointSkillBundle,
    tissue_bundle: ActiveKnowledgeBundle,
    allocation: Any,
    nuclei_preflight: Any,
) -> dict[str, Any]:
    if tissue_case.case_id != joint_case.case_id:
        raise JointContractError("joint and tissue authority cases disagree")
    return {
        "schema_version": PORTFOLIO_AUTHORITY_SCHEMA_VERSION,
        "portfolio_kind": "tissue",
        "case_id": joint_case.case_id,
        "primitive_id": joint_case.primitive_id,
        "pathology_domain_id": joint_case.pathology_domain_id,
        "annotation_profile_id": joint_case.annotation_profile_id,
        "cell_observation_profile_id": joint_case.cell_observation_profile_id,
        "cell_population_profile_id": joint_case.cell_population_profile_id,
        "source_asset_digests": canonical_source_asset_digests(
            joint_case.provenance
        ),
        "source_tissue_array_sha256": array_sha256(source_tissue),
        "tissue_case": _canonical_value(tissue_case.to_metadata()),
        "joint_area_budget": _canonical_value(joint_case.joint_area_budget),
        "allocation": _canonical_value(allocation),
        "nuclei_preflight": _canonical_value(nuclei_preflight),
        "nuclei_preflight_sha256": canonical_metadata_sha256(nuclei_preflight),
        "joint_skill": _joint_skill_identity(joint_bundle),
        "tissue_skill": _tissue_skill_identity(tissue_bundle),
    }


def build_cell_portfolio_authority_binding(
    *,
    case: JointCaseContext,
    source_tissue: np.ndarray,
    source_nuclei: np.ndarray,
    joint_bundle: JointSkillBundle,
) -> dict[str, Any]:
    return {
        "schema_version": PORTFOLIO_AUTHORITY_SCHEMA_VERSION,
        "portfolio_kind": "cell",
        "case_id": case.case_id,
        "primitive_id": case.primitive_id,
        "pathology_domain_id": case.pathology_domain_id,
        "annotation_profile_id": case.annotation_profile_id,
        "cell_observation_profile_id": case.cell_observation_profile_id,
        "cell_population_profile_id": case.cell_population_profile_id,
        "source_asset_digests": canonical_source_asset_digests(case.provenance),
        "source_tissue_array_sha256": array_sha256(source_tissue),
        "source_nuclei_array_sha256": array_sha256(source_nuclei),
        "cell_count_extent_budget": _canonical_value(
            case.cell_count_extent_budget
        ),
        "joint_skill": _joint_skill_identity(joint_bundle),
    }
