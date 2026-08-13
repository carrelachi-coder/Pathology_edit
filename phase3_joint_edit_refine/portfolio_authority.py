"""Canonical runtime authority bindings for pre-LLM candidate portfolios."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phase3_mask_edit_refine.evidence import load_id_mask, sha256_file
from phase3_mask_edit_refine.models import CaseContext
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle

from .models import JointCaseContext, JointContractError
from .nuclei import load_nuclei_mask
from .skills.repository import JointSkillBundle

PORTFOLIO_AUTHORITY_SCHEMA_VERSION = "joint-portfolio-authority-v4"


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


def _local_source_path(uri: str, *, label: str) -> Path:
    if "://" in uri and not uri.startswith("file://"):
        raise JointContractError(
            f"direct Planner authority requires a local {label} asset"
        )
    path = Path(uri.removeprefix("file://"))
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise JointContractError(f"{label} source asset is unavailable: {path}") from exc
    if not resolved.is_file():
        raise JointContractError(f"{label} source asset is not a regular file: {path}")
    return resolved


def _verified_file(
    uri: str,
    *,
    label: str,
    declared_sha256: Any,
) -> tuple[Path, str]:
    if not isinstance(declared_sha256, str) or not declared_sha256:
        raise JointContractError(f"{label} source digest is missing")
    path = _local_source_path(uri, label=label)
    live = sha256_file(path)
    if live != declared_sha256:
        raise JointContractError(
            f"{label} live source authority is detached from provenance"
        )
    return path, live


def verify_live_source_assets(
    case: JointCaseContext,
    *,
    required_auxiliary_structure_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Re-read every live source that can affect scene or preflight authority.

    A portfolio may retain declared provenance after an on-disk source has
    changed.  Direct Planner entry points therefore cannot trust the case or a
    precomputed witness alone: this verifier hashes the current files, compares
    them with the immutable provenance contract, and derives the exact raster
    identities used by the portfolio binding.
    """

    case.validate_local_inputs()
    provenance = case.provenance
    image_path, image_file_sha = _verified_file(
        case.source_image_uri,
        label="source image",
        declared_sha256=provenance.get("source_image_sha256"),
    )
    tissue_path, tissue_file_sha = _verified_file(
        case.source_tissue_mask_uri,
        label="source tissue mask",
        declared_sha256=provenance.get("source_tissue_mask_sha256"),
    )
    nuclei_path, nuclei_file_sha = _verified_file(
        case.source_nuclei_mask_uri,
        label="source nuclei mask",
        declared_sha256=provenance.get("source_nuclei_mask_sha256"),
    )
    source_tissue = np.ascontiguousarray(load_id_mask(tissue_path))
    source_nuclei = np.ascontiguousarray(load_nuclei_mask(nuclei_path))
    if source_tissue.shape != source_nuclei.shape:
        raise JointContractError("live tissue and nuclei source rasters are not aligned")

    instance_uri = case.source_nuclei_instances_uri
    declared_instance_sha = provenance.get("source_nuclei_instances_sha256")
    if bool(instance_uri) != bool(declared_instance_sha):
        raise JointContractError(
            "nucleus-instance URI and provenance digest authority disagree"
        )
    instance_asset: dict[str, Any] | None = None
    if instance_uri:
        instance_path, instance_file_sha = _verified_file(
            instance_uri,
            label="source nuclei instances",
            declared_sha256=declared_instance_sha,
        )
        instance_asset = {
            "canonical_path": str(instance_path),
            "file_sha256": instance_file_sha,
        }

    auxiliary_uris = dict(case.auxiliary_structure_uris)
    auxiliary_digests = provenance.get("auxiliary_structure_sha256", {})
    auxiliary_provenance = provenance.get("auxiliary_structure_provenance", {})
    if not isinstance(auxiliary_digests, Mapping) or not isinstance(
        auxiliary_provenance, Mapping
    ):
        raise JointContractError(
            "auxiliary live authority requires digest and producer provenance maps"
        )
    if set(auxiliary_uris) != set(auxiliary_digests) or set(auxiliary_uris) != set(
        auxiliary_provenance
    ):
        raise JointContractError(
            "auxiliary URI, digest and producer provenance IDs differ"
        )
    missing_required = set(required_auxiliary_structure_ids) - set(auxiliary_uris)
    if missing_required:
        raise JointContractError(
            "required auxiliary source authority is missing: "
            + ", ".join(sorted(missing_required))
        )
    auxiliary_assets: dict[str, Any] = {}
    for structure_id, uri in sorted(auxiliary_uris.items()):
        record = auxiliary_provenance.get(structure_id)
        if not isinstance(record, Mapping):
            raise JointContractError(
                f"auxiliary structure {structure_id!r} lacks producer provenance"
            )
        declared = auxiliary_digests.get(structure_id)
        if (
            record.get("output_sha256") != declared
            or record.get("source_tissue_mask_sha256") != tissue_file_sha
            or not record.get("producer_id")
            or not record.get("producer_version")
        ):
            raise JointContractError(
                f"auxiliary structure {structure_id!r} provenance is detached"
            )
        path, live_sha = _verified_file(
            uri,
            label=f"auxiliary structure {structure_id}",
            declared_sha256=declared,
        )
        auxiliary_assets[structure_id] = {
            "canonical_path": str(path),
            "file_sha256": live_sha,
            "producer_id": str(record["producer_id"]),
            "producer_version": str(record["producer_version"]),
            "producer_record_sha256": canonical_metadata_sha256(record),
        }

    return {
        "schema_version": "joint-live-source-authority-v1",
        "source_image": {
            "canonical_path": str(image_path),
            "file_sha256": image_file_sha,
        },
        "source_tissue": {
            "canonical_path": str(tissue_path),
            "file_sha256": tissue_file_sha,
            "array_sha256": array_sha256(source_tissue),
        },
        "source_nuclei": {
            "canonical_path": str(nuclei_path),
            "file_sha256": nuclei_file_sha,
            "array_sha256": array_sha256(source_nuclei),
        },
        "source_nuclei_instances": instance_asset,
        "auxiliary_structures": auxiliary_assets,
        "declared_source_asset_digests": canonical_source_asset_digests(
            provenance
        ),
    }


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
    live_sources = verify_live_source_assets(
        joint_case,
        required_auxiliary_structure_ids=(
            joint_bundle.mechanism.representability.required_auxiliary_structures
        ),
    )
    if array_sha256(source_tissue) != live_sources["source_tissue"]["array_sha256"]:
        raise JointContractError(
            "caller tissue raster is detached from current live source authority"
        )
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
        "source_nuclei_array_sha256": live_sources["source_nuclei"][
            "array_sha256"
        ],
        "live_source_assets": live_sources,
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
    live_sources = verify_live_source_assets(
        case,
        required_auxiliary_structure_ids=(
            joint_bundle.mechanism.representability.required_auxiliary_structures
        ),
    )
    if array_sha256(source_tissue) != live_sources["source_tissue"]["array_sha256"]:
        raise JointContractError(
            "caller tissue raster is detached from current live source authority"
        )
    if array_sha256(source_nuclei) != live_sources["source_nuclei"]["array_sha256"]:
        raise JointContractError(
            "caller nuclei raster is detached from current live source authority"
        )
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
        "live_source_assets": live_sources,
        "cell_count_extent_budget": _canonical_value(
            case.cell_count_extent_budget
        ),
        "joint_skill": _joint_skill_identity(joint_bundle),
    }
