"""Fail-closed P1 authority materialization and pre-visualization preflight.

This module never edits a target mask, invokes a Planner, renders a review
board, or calls an external API.  It records every frozen GLaS/PANDA binding in
selection order and stops at the first failed authority stage.  A later
candidate compiler may run only after all source, profile, auxiliary, and
runtime authority is live and digest-bound.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from phase3_mask_edit_refine.evidence import load_id_mask, sha256_file
from phase3_mask_edit_refine.gates import GateRegistry
from phase3_mask_edit_refine.skills import SkillRepository as MaskSkillRepository
from phase3_mask_edit_refine.skills import (
    bind_active_bundle_to_case,
    validate_active_bundle_authority,
)

from .auxiliary import materialize_profile_auxiliaries
from .budget import JointFeasibilitySolver
from .candidate_feasibility import CandidateFeasibilityCompiler
from .feasibility import build_joint_nuclei_preflight
from .g2_execution_qualification import _with_scene_calibrated_cell_budget
from .gates import JointGateRegistry
from .models import JointAreaBudget, JointCaseContext
from .nuclei import load_nuclei_mask
from .portfolio_authority import array_sha256, canonical_metadata_sha256
from .scene import build_joint_scene_analysis
from .semantic_parser import RuleBasedSemanticParser, bind_semantic_intent
from .skills.execution_aliases import tissue_tool_primitive_id
from .skills.repository import JointSkillRepository
from .workflow import (
    JointPathologyEditWorkflow,
    JointWorkflowConfig,
    _as_tissue_case,
    _tissue_portfolio_authority_binding,
)

AUTHORITY_MANIFEST_SCHEMA = "p1-glas-panda-authority-manifest-v1"
AUTHORITY_RECORD_SCHEMA = "p1-glas-panda-authority-record-v1"
AUXILIARY_MANIFEST_SCHEMA = "p1-profile-auxiliary-materialization-v1"
PREFLIGHT_RECORD_SCHEMA = "p1-deterministic-candidate-preflight-v1"
STATUS_TABLE_SCHEMA = "p1-preflight-status-table-v1"
AUTHORITY_ERRATUM_SCHEMA = "p1-glas-panda-authority-erratum-v1"
RUNTIME_INPUT_SCHEMA = "p1-glas-panda-runtime-authority-v1"

PROFILE_OWNED_AUXILIARY_STRUCTURES = {
    "glas-gland-v1": ("gland_or_lumen_support",),
    "panda-gleason-v1": (
        "native_pattern_and_lumen_map",
        "native_pattern_map",
        "gland_lumen_map",
    ),
}
EXTERNAL_ONLY_AUXILIARY_STRUCTURES = frozenset(
    {"native_gland_instance_map", "local_clearance_roi"}
)
GLAS_PATCH_GRADE_VALUES = frozenset(
    {
        "benign",
        "malignant",
        "well_differentiated",
        "moderately_differentiated",
        "poorly_differentiated",
    }
)
RUNTIME_DIGEST_FIELDS = (
    "mature_probnet_checkpoint_sha256",
    "frozen_spatial_ranker_sha256",
    "instance_library_sha256",
    "generator_checkpoint_sha256",
)
RUNTIME_ASSET_IDS = frozenset(
    {
        "mature_probnet_checkpoint",
        "frozen_probnet_spatial_ranker_checkpoint",
        "glas_nucleus_instance_library",
        "panda_nucleus_instance_library",
        "later_he_generator_checkpoint",
    }
)
AUTHORITY_ERRATUM_FILENAME = "p1_glas_panda_authority_erratum_v1.json"
RUNTIME_AUTHORITY_FILENAME = "p1_glas_panda_runtime_authority_v1.json"
RUNTIME_CODE_PATHS = (
    "phase3_joint_edit_refine/mature_probnet_adapter.py",
    "phase3_joint_edit_refine/probnet_adapter.py",
    "phase3_joint_edit_refine/candidate_feasibility.py",
    "phase3_joint_edit_refine/cell_programs.py",
    "phase3_joint_edit_refine/gates.py",
    "phase3_joint_edit_refine/portfolio_authority.py",
    "phase3_joint_edit_refine/workflow.py",
    "phase3_mask_edit_refine/skills/catalog_manifest_v1.json",
    "inpaint_cells/generate.py",
    "inpaint_cells/nuclei_library/library.py",
)
OUTPUT_FILENAMES = {
    "summary": "p1_glas_panda_authority_manifest_v1.json",
    "authority": "p1_glas_panda_authority_records_v1.jsonl",
    "auxiliary": "p1_glas_panda_auxiliary_materialization_v1.json",
    "preflight": "p1_glas_panda_candidate_preflight_v1.jsonl",
    "status_table": "p1_glas_panda_preflight_status_v1.tsv",
}


def _canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    separators = (",", ":") if indent is None else None
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=separators,
            indent=indent,
        )
        + ("\n" if indent is not None else "")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sealed_record(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("record_sha256", None)
    return {**payload, "record_sha256": canonical_metadata_sha256(payload)}


def _resolved_path(uri: Any) -> str | None:
    if not isinstance(uri, str) or not uri.strip() or "://" in uri:
        return None
    return str(Path(uri).expanduser().resolve(strict=False))


def _portable_input_path(path: Path, *, root: Path) -> str:
    resolved = path.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    if resolved.is_relative_to(root_resolved):
        return str(resolved.relative_to(root_resolved))
    return "authority-input://" + path.name


def _inspect_raster_asset(
    *,
    uri: Any,
    declared_sha256: Any,
    role: str,
    decoder: str,
) -> dict[str, Any]:
    declared_path = str(uri) if isinstance(uri, str) and uri else None
    canonical_path = _resolved_path(uri)
    base = {
        "role": role,
        "declared_path": declared_path,
        "canonical_path": canonical_path,
        "declared_file_sha256": (
            str(declared_sha256) if _is_sha256(declared_sha256) else None
        ),
        "observed_file_sha256": None,
        "decoded_array_sha256": None,
        "decoded_dtype": None,
        "decoded_shape": None,
        "available": False,
        "digest_matches_declaration": False,
        "authority_verified": False,
        "failure_codes": [],
    }
    failures: list[str] = []
    if canonical_path is None:
        failures.append("source_path_missing_or_nonlocal")
        return {**base, "failure_codes": failures}
    path = Path(canonical_path)
    if not path.is_file():
        failures.append("source_asset_unavailable")
        if not _is_sha256(declared_sha256):
            failures.append("source_digest_not_frozen")
        return {**base, "failure_codes": failures}
    observed = sha256_file(path)
    base["available"] = True
    base["observed_file_sha256"] = observed
    if not _is_sha256(declared_sha256):
        failures.append("source_digest_not_frozen")
    elif observed != declared_sha256:
        failures.append("source_digest_mismatch")
    else:
        base["digest_matches_declaration"] = True
    try:
        if decoder == "tissue":
            array = np.ascontiguousarray(load_id_mask(path))
        elif decoder == "nuclei":
            array = np.ascontiguousarray(load_nuclei_mask(path))
        elif decoder == "image":
            with Image.open(path) as image:
                array = np.ascontiguousarray(np.asarray(image))
        else:
            raise ValueError(f"unsupported authority decoder: {decoder}")
        base["decoded_array_sha256"] = array_sha256(array)
        base["decoded_dtype"] = str(array.dtype)
        base["decoded_shape"] = list(array.shape)
    except Exception as exc:  # noqa: BLE001 - one auditable failure per asset
        failures.append(f"source_decode_failed:{type(exc).__name__}")
    base["failure_codes"] = failures
    base["authority_verified"] = not failures
    return base


def _inspect_instance_asset(*, uri: Any, declared_sha256: Any) -> dict[str, Any]:
    canonical_path = _resolved_path(uri)
    record = {
        "role": "execution_optional_native_nuclei_instance_authority",
        "declared_path": str(uri) if isinstance(uri, str) and uri else None,
        "canonical_path": canonical_path,
        "declared_file_sha256": (
            str(declared_sha256) if _is_sha256(declared_sha256) else None
        ),
        "observed_file_sha256": None,
        "decoded_record_sha256": None,
        "available": False,
        "authority_verified": False,
        "required_for_all_primitives": False,
        "failure_codes": [],
    }
    if uri is None and declared_sha256 is None:
        return {**record, "failure_codes": ["native_instance_asset_not_declared"]}
    failures: list[str] = []
    if canonical_path is None or not Path(canonical_path).is_file():
        failures.append("native_instance_asset_unavailable")
    elif not _is_sha256(declared_sha256):
        failures.append("native_instance_digest_not_frozen")
    else:
        path = Path(canonical_path)
        observed = sha256_file(path)
        record["available"] = True
        record["observed_file_sha256"] = observed
        if observed != declared_sha256:
            failures.append("native_instance_digest_mismatch")
        else:
            try:
                decoded = json.loads(path.read_text(encoding="utf-8"))
                record["decoded_record_sha256"] = canonical_metadata_sha256(decoded)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"native_instance_decode_failed:{type(exc).__name__}")
    record["failure_codes"] = failures
    record["authority_verified"] = not failures
    return record


def _inspect_json_authority_asset(
    *, uri: Any, declared_sha256: Any, role: str
) -> dict[str, Any]:
    canonical_path = _resolved_path(uri)
    record = {
        "role": role,
        "declared_path": str(uri) if isinstance(uri, str) and uri else None,
        "canonical_path": canonical_path,
        "declared_file_sha256": (
            str(declared_sha256) if _is_sha256(declared_sha256) else None
        ),
        "observed_file_sha256": None,
        "decoded_record_sha256": None,
        "available": False,
        "authority_verified": False,
        "required_for_all_primitives": False,
        "failure_codes": [],
    }
    if uri is None and declared_sha256 is None:
        return {**record, "failure_codes": ["profile_metadata_asset_not_declared"]}
    failures: list[str] = []
    if canonical_path is None or not Path(canonical_path).is_file():
        failures.append("profile_metadata_asset_unavailable")
    elif not _is_sha256(declared_sha256):
        failures.append("profile_metadata_digest_not_frozen")
    else:
        path = Path(canonical_path)
        observed = sha256_file(path)
        record["available"] = True
        record["observed_file_sha256"] = observed
        if observed != declared_sha256:
            failures.append("profile_metadata_digest_mismatch")
        else:
            try:
                decoded = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(decoded, Mapping):
                    raise TypeError("profile metadata must be an object")
                record["decoded_record_sha256"] = canonical_metadata_sha256(
                    decoded
                )
            except (OSError, TypeError, ValueError) as exc:
                failures.append(
                    f"profile_metadata_decode_failed:{type(exc).__name__}"
                )
    record["failure_codes"] = failures
    record["authority_verified"] = not failures
    return record


def _load_authority_inputs(
    *,
    erratum_path: Path,
    runtime_path: Path,
    selection_sha256: str,
    source_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    erratum = json.loads(erratum_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if erratum.get("schema_version") != AUTHORITY_ERRATUM_SCHEMA:
        raise ValueError("unsupported P1 authority erratum schema")
    if runtime.get("schema_version") != RUNTIME_INPUT_SCHEMA:
        raise ValueError("unsupported P1 runtime authority schema")
    for payload, name in ((erratum, "authority erratum"), (runtime, "runtime")):
        if payload.get("selection_manifest_sha256") != selection_sha256:
            raise ValueError(f"P1 {name} input is detached from frozen selection")
    if erratum.get("source_manifest_sha256") != source_sha256:
        raise ValueError("P1 authority erratum is detached from source case pool")
    source_entries = erratum.get("source_case_authority")
    external_entries = erratum.get("external_auxiliary_authority")
    runtime_assets = runtime.get("assets")
    if not isinstance(source_entries, list) or not isinstance(external_entries, list):
        raise TypeError("P1 authority erratum requires typed source and auxiliary lists")
    if not isinstance(runtime_assets, list):
        raise TypeError("P1 runtime authority requires an asset list")
    source_ids = [item.get("case_id") for item in source_entries if isinstance(item, Mapping)]
    if len(source_ids) != len(source_entries) or len(set(source_ids)) != len(source_ids):
        raise ValueError("P1 source authority erratum case IDs are malformed or duplicated")
    external_keys = [
        (item.get("binding_id"), item.get("structure_id"))
        for item in external_entries
        if isinstance(item, Mapping)
    ]
    if len(external_keys) != len(external_entries) or len(set(external_keys)) != len(
        external_keys
    ):
        raise ValueError("P1 external auxiliary authority bindings are malformed or duplicated")
    runtime_ids = [item.get("asset_id") for item in runtime_assets if isinstance(item, Mapping)]
    if len(runtime_ids) != len(runtime_assets) or len(set(runtime_ids)) != len(runtime_ids):
        raise ValueError("P1 runtime asset IDs are malformed or duplicated")
    return erratum, runtime


def _effective_source_row(
    row: Mapping[str, Any],
    *,
    erratum_entry: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective = dict(row)
    if erratum_entry is None:
        return effective, {
            "status": "missing",
            "entry_sha256": None,
            "source_case_record_sha256": row.get("case_record_sha256"),
            "profile_provenance": {},
            "source_asset_authority": {},
        }
    if erratum_entry.get("source_case_record_sha256") != row.get(
        "case_record_sha256"
    ):
        raise ValueError("P1 authority erratum is detached from source case record")
    supplements = erratum_entry.get("source_asset_authority") or {}
    if not isinstance(supplements, Mapping):
        raise TypeError("P1 source asset authority supplement must be an object")
    allowed = {
        "source_image": "source_image_sha256",
        "source_tissue_mask": "source_tissue_mask_sha256",
        "source_nuclei_mask": "source_nuclei_mask_sha256",
        "source_nuclei_instances": "source_nuclei_instances_sha256",
        "source_gland_instance_mask": "source_gland_instance_mask_sha256",
        "source_profile_metadata": "source_profile_metadata_sha256",
    }
    if set(supplements) - set(allowed):
        raise ValueError("P1 source authority supplement names an unknown asset role")
    normalized_supplements: dict[str, dict[str, Any]] = {}
    for role, raw in sorted(supplements.items()):
        if not isinstance(raw, Mapping):
            raise TypeError("P1 source authority asset record must be an object")
        path = raw.get("path")
        digest = raw.get("sha256")
        if not isinstance(path, str) or not path.strip() or not _is_sha256(digest):
            raise ValueError("P1 source authority supplement requires path and SHA256")
        digest_field = allowed[role]
        frozen_digest = row.get(digest_field)
        if _is_sha256(frozen_digest) and frozen_digest != digest:
            raise ValueError("P1 source authority supplement contradicts frozen digest")
        effective[role] = path
        effective[digest_field] = digest
        normalized_supplements[role] = {"path": path, "sha256": digest}
    profile_provenance = erratum_entry.get("profile_provenance") or {}
    if not isinstance(profile_provenance, Mapping):
        raise TypeError("P1 profile provenance erratum must be an object")
    binding = {
        "status": "bound",
        "source_case_record_sha256": row.get("case_record_sha256"),
        "profile_provenance": dict(profile_provenance),
        "source_asset_authority": normalized_supplements,
    }
    binding["entry_sha256"] = canonical_metadata_sha256(binding)
    return effective, binding


def _validate_frozen_source_case_record(row: Mapping[str, Any]) -> None:
    declared = row.get("case_record_sha256")
    canonical = dict(row)
    canonical.pop("case_record_sha256", None)
    if declared != canonical_metadata_sha256(canonical):
        raise ValueError(
            "P1 frozen source case record digest mismatch: "
            + str(row.get("case_id"))
        )


def _runtime_asset_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    inventory = [
        {
            "path": str(item.relative_to(path)),
            "sha256": sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    if not inventory:
        raise ValueError("runtime asset directory is empty")
    return canonical_metadata_sha256(inventory)


def _runtime_authority(
    *,
    root: Path,
    selection_runtime: Mapping[str, Any],
    runtime_input: Mapping[str, Any],
    runtime_input_sha256: str,
    code_commit: str,
) -> dict[str, Any]:
    configuration = runtime_input.get("preflight_configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError("P1 runtime preflight configuration must be an object")
    if (
        configuration.get("cell_budget_policy_id")
        != "scene-calibrated-local-population-budget-v1"
    ):
        raise ValueError("P1 runtime names an unsupported cell budget policy")
    JointAreaBudget.from_value(configuration.get("joint_area_budget"))
    maximum_candidates = configuration.get("maximum_tissue_candidates")
    if (
        not isinstance(maximum_candidates, int)
        or isinstance(maximum_candidates, bool)
        or not 1 <= maximum_candidates <= 16
    ):
        raise ValueError("P1 maximum tissue candidates must be in [1, 16]")
    raw_assets = tuple(runtime_input.get("assets", ()))
    asset_ids = {str(item["asset_id"]) for item in raw_assets}
    if asset_ids != RUNTIME_ASSET_IDS:
        raise ValueError("P1 runtime authority must bind the exact asset catalog")
    code_assets = []
    for relative in RUNTIME_CODE_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"P1 runtime code asset is missing: {relative}")
        code_assets.append(
            {
                "path": relative,
                "file_sha256": sha256_file(path),
            }
        )
    external_assets = []
    for raw in raw_assets:
        asset_id = str(raw["asset_id"])
        declared_path = raw.get("path")
        declared_digest = raw.get("sha256")
        canonical_path = _resolved_path(declared_path)
        observed = None
        failure_codes = []
        if canonical_path is None:
            failure_codes.append("runtime_asset_path_unbound")
        elif not _is_sha256(declared_digest):
            failure_codes.append("runtime_asset_digest_unbound")
        else:
            try:
                observed = _runtime_asset_sha256(Path(canonical_path))
            except (FileNotFoundError, ValueError):
                failure_codes.append("runtime_asset_unavailable")
            else:
                if observed != declared_digest:
                    failure_codes.append("runtime_asset_digest_mismatch")
        if raw.get("required_for_preflight") is not True:
            failure_codes.append("runtime_asset_not_required_by_input")
        if asset_id == "later_he_generator_checkpoint":
            if raw.get("reader_side_only") is not True:
                failure_codes.append("generator_not_reader_side_only")
            if raw.get("used_during_this_stage") is not False:
                failure_codes.append("generator_marked_used_during_preflight")
        external_assets.append(
            {
                "asset_id": asset_id,
                "asset_kind": raw.get("asset_kind"),
                "declared_path": declared_path,
                "canonical_path": canonical_path,
                "declared_sha256": (
                    str(declared_digest) if _is_sha256(declared_digest) else None
                ),
                "observed_sha256": observed,
                "required_for_preflight": bool(
                    raw.get("required_for_preflight", True)
                ),
                "verified": not failure_codes,
                "failure_codes": failure_codes,
                "reader_side_only": bool(raw.get("reader_side_only", False)),
                "used_during_this_stage": bool(
                    raw.get("used_during_this_stage", False)
                ),
            }
        )
    assets_by_id = {item["asset_id"]: item for item in external_assets}
    selection_bindings = {
        "mature_probnet_checkpoint_sha256": "mature_probnet_checkpoint",
        "frozen_spatial_ranker_sha256": "frozen_probnet_spatial_ranker_checkpoint",
        "generator_checkpoint_sha256": "later_he_generator_checkpoint",
    }
    selection_mismatches = []
    for field, asset_id in selection_bindings.items():
        frozen = selection_runtime.get(field)
        declared = (assets_by_id.get(asset_id) or {}).get("declared_sha256")
        if _is_sha256(frozen) and declared != frozen:
            selection_mismatches.append(field)
    library_digests = [
        (assets_by_id.get(asset_id) or {}).get("declared_sha256")
        for asset_id in (
            "glas_nucleus_instance_library",
            "panda_nucleus_instance_library",
        )
    ]
    library_set_digest = (
        canonical_metadata_sha256(library_digests)
        if all(_is_sha256(item) for item in library_digests)
        else None
    )
    frozen_library_digest = selection_runtime.get("instance_library_sha256")
    if _is_sha256(frozen_library_digest) and frozen_library_digest != library_set_digest:
        selection_mismatches.append("instance_library_sha256")
    effective_runtime_digests = {
        "mature_probnet_checkpoint_sha256": (
            (assets_by_id.get("mature_probnet_checkpoint") or {}).get(
                "declared_sha256"
            )
        ),
        "frozen_spatial_ranker_sha256": (
            (
                assets_by_id.get(
                    "frozen_probnet_spatial_ranker_checkpoint"
                )
                or {}
            ).get("declared_sha256")
        ),
        "instance_library_sha256": library_set_digest,
        "generator_checkpoint_sha256": (
            (assets_by_id.get("later_he_generator_checkpoint") or {}).get(
                "declared_sha256"
            )
        ),
    }
    effective_missing_fields = [
        field
        for field, digest in effective_runtime_digests.items()
        if not _is_sha256(digest)
    ]
    missing_selection_fields = [
        field for field in RUNTIME_DIGEST_FIELDS if not _is_sha256(selection_runtime.get(field))
    ]
    payload = {
        "schema_version": "p1-runtime-authority-materialization-v1",
        "authority_materializer_code_commit": code_commit,
        "runtime_input_manifest_sha256": runtime_input_sha256,
        "preflight_configuration": dict(configuration),
        "selection_runtime_authority": dict(selection_runtime),
        "selection_runtime_digest_fields_missing": missing_selection_fields,
        "effective_runtime_digest_fields": effective_runtime_digests,
        "effective_runtime_digest_fields_missing": effective_missing_fields,
        "selection_runtime_binding_mismatches": sorted(selection_mismatches),
        "instance_library_set_sha256": library_set_digest,
        "code_assets": code_assets,
        "runtime_code_set_sha256": canonical_metadata_sha256(code_assets),
        "external_assets": external_assets,
        "unverified_external_asset_ids": [
            item["asset_id"]
            for item in external_assets
            if item["required_for_preflight"] and not item["verified"]
        ],
        "all_required_runtime_assets_verified": bool(
            asset_ids == RUNTIME_ASSET_IDS
            and not effective_missing_fields
            and not selection_mismatches
            and all(
                item["verified"]
                for item in external_assets
                if item["required_for_preflight"]
            )
        ),
    }
    return {**payload, "runtime_authority_sha256": canonical_metadata_sha256(payload)}


def _source_authority(row: Mapping[str, Any]) -> dict[str, Any]:
    image = _inspect_raster_asset(
        uri=row.get("source_image"),
        declared_sha256=row.get("source_image_sha256"),
        role="reader_side_only_never_execution_planner_input",
        decoder="image",
    )
    tissue = _inspect_raster_asset(
        uri=row.get("source_tissue_mask"),
        declared_sha256=row.get("source_tissue_mask_sha256"),
        role="execution_tissue_mask_authority",
        decoder="tissue",
    )
    nuclei = _inspect_raster_asset(
        uri=row.get("source_nuclei_mask"),
        declared_sha256=row.get("source_nuclei_mask_sha256"),
        role="execution_nuclei_mask_authority",
        decoder="nuclei",
    )
    instance_uri = row.get("source_nuclei_instances") or row.get(
        "source_nuclei_instances_uri"
    )
    instances = _inspect_instance_asset(
        uri=instance_uri,
        declared_sha256=row.get("source_nuclei_instances_sha256"),
    )
    gland_instances = _inspect_raster_asset(
        uri=row.get("source_gland_instance_mask"),
        declared_sha256=row.get("source_gland_instance_mask_sha256"),
        role="execution_optional_native_gland_instance_authority",
        decoder="tissue",
    )
    profile_metadata = _inspect_json_authority_asset(
        uri=row.get("source_profile_metadata"),
        declared_sha256=row.get("source_profile_metadata_sha256"),
        role="execution_profile_dataset_metadata_authority",
    )
    required = (image, tissue, nuclei)
    failure_codes = sorted(
        {
            f"{item['role']}:{code}"
            for item in required
            for code in item["failure_codes"]
        }
    )
    shape_aligned = bool(
        tissue["authority_verified"]
        and nuclei["authority_verified"]
        and tissue["decoded_shape"] == nuclei["decoded_shape"]
    )
    if tissue["authority_verified"] and nuclei["authority_verified"] and not shape_aligned:
        failure_codes.append("execution_masks:source_mask_shape_mismatch")
    payload = {
        "source_image": image,
        "source_tissue_mask": tissue,
        "source_nuclei_mask": nuclei,
        "source_nuclei_instances": instances,
        "source_gland_instance_mask": gland_instances,
        "source_profile_metadata": profile_metadata,
        "tissue_nuclei_shape_aligned": shape_aligned,
        "required_source_authority_verified": not failure_codes,
        "failure_codes": sorted(set(failure_codes)),
    }
    return {**payload, "source_authority_sha256": canonical_metadata_sha256(payload)}


def _profile_required_provenance(
    *, repository: MaskSkillRepository,
    profile_id: str,
    erratum_binding: Mapping[str, Any],
    source_row: Mapping[str, Any],
    source_authority: Mapping[str, Any],
) -> dict[str, Any]:
    package = repository.get(profile_id, expected_kind="annotation_profile")
    required = tuple(
        str(value)
        for value in package.capabilities.get("required_provenance_fields", ())
    )
    supplied = erratum_binding.get("profile_provenance")
    supplied_mapping = dict(supplied) if isinstance(supplied, Mapping) else {}
    bound = {
        field: supplied_mapping[field]
        for field in required
        if supplied_mapping.get(field) not in (None, "")
    }
    invalid: dict[str, str] = {}
    metadata: Mapping[str, Any] = {}
    metadata_authority = source_authority.get("source_profile_metadata") or {}
    if metadata_authority.get("authority_verified"):
        decoded = json.loads(
            Path(str(metadata_authority["canonical_path"])).read_text(
                encoding="utf-8"
            )
        )
        if isinstance(decoded, Mapping):
            metadata = decoded
    if "preprocessing_revision" in bound and bound.get(
        "preprocessing_revision"
    ) != metadata.get("preprocessing_revision"):
        invalid["preprocessing_revision"] = "dataset_metadata_binding_mismatch"
    if profile_id == "glas-gland-v1":
        gland_authority = source_authority.get(
            "source_gland_instance_mask"
        ) or {}
        if "original_instance_mask_digest" in bound and (
            not gland_authority.get("authority_verified")
            or bound.get("original_instance_mask_digest")
            != gland_authority.get("observed_file_sha256")
            or bound.get("original_instance_mask_digest")
            != metadata.get("original_instance_mask_digest")
        ):
            invalid["original_instance_mask_digest"] = (
                "must_bind_live_native_gland_instance_annotation"
            )
        if "patch_grade" in bound and (
            bound.get("patch_grade") not in GLAS_PATCH_GRADE_VALUES
            or bound.get("patch_grade") != metadata.get("patch_grade")
        ):
            invalid["patch_grade"] = "must_bind_frozen_field_grade_metadata"
    elif profile_id == "panda-gleason-v1":
        if "provider" in bound and (
            bound.get("provider") != "PANDA"
            or bound.get("provider") != metadata.get("provider")
        ):
            invalid["provider"] = "unsupported_provider_contract"
        if "original_label_map_digest" in bound and (
            bound.get("original_label_map_digest")
            != source_authority.get("source_tissue_mask", {}).get(
                "observed_file_sha256"
            )
            or bound.get("original_label_map_digest")
            != metadata.get("original_label_map_digest")
        ):
            invalid["original_label_map_digest"] = (
                "must_bind_live_source_tissue_label_map"
            )
    missing = sorted(set(required) - set(bound))
    return {
        "required_fields": list(required),
        "bound_fields": bound,
        "missing_fields": missing,
        "invalid_fields": invalid,
        "authority_verified": not missing and not invalid,
        "provenance_source": "digest_bound_authority_erratum",
        "source_profile_metadata_authority_sha256": canonical_metadata_sha256(
            metadata_authority
        ),
        "authority_erratum_entry_sha256": erratum_binding.get("entry_sha256"),
        "source_case_record_sha256": source_row.get("case_record_sha256"),
        "inferred_from_he_or_untyped_metadata": False,
    }


def _auxiliary_inventory(
    *,
    selection: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, Any]],
    source_authority_by_case: Mapping[str, Mapping[str, Any]],
    profile_provenance_by_case: Mapping[tuple[str, str], Mapping[str, Any]],
    root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    profile_by_case: dict[str, str] = {}
    for evaluation in selection["evaluations"]:
        for row in evaluation["selected_cases"]:
            profile_by_case[str(row["case_id"])] = str(
                evaluation["annotation_profile_id"]
            )
    evaluation_by_profile = {
        str(item["annotation_profile_id"]): item
        for item in selection["evaluations"]
    }
    population_by_profile = {
        "glas-gland-v1": "colorectal-cellvit-source-first-v1",
        "panda-gleason-v1": "prostate-cellvit-source-first-v1",
    }
    entries = []
    actual_paths_by_case: dict[str, dict[str, str]] = {}
    for case_id in sorted(profile_by_case):
        profile_id = profile_by_case[case_id]
        source = source_authority_by_case[case_id]
        provenance = profile_provenance_by_case[(profile_id, case_id)]
        expected_ids = PROFILE_OWNED_AUXILIARY_STRUCTURES.get(profile_id, ())
        output_case_dir = output_dir / case_id
        produced_by_id = {}
        materialization_error = None
        if (
            source["required_source_authority_verified"]
            and provenance["authority_verified"]
        ):
            evaluation = evaluation_by_profile[profile_id]
            row = source_rows[case_id]
            case_provenance = {
                "source_image_sha256": row["source_image_sha256"],
                "source_tissue_mask_sha256": row["source_tissue_mask_sha256"],
                "source_nuclei_mask_sha256": row["source_nuclei_mask_sha256"],
                **provenance["bound_fields"],
            }
            try:
                context = JointCaseContext(
                    case_id=case_id,
                    instruction=str(evaluation["instruction"]),
                    source_image_uri=str(row["source_image"]),
                    source_tissue_mask_uri=str(row["source_tissue_mask"]),
                    source_nuclei_mask_uri=str(row["source_nuclei_mask"]),
                    pathology_domain_id=str(evaluation["pathology_domain_id"]),
                    annotation_profile_id=profile_id,
                    cell_observation_profile_id="cellvit-five-class-v1",
                    cell_population_profile_id=population_by_profile[profile_id],
                    primitive_id=str(evaluation["primitive_id"]),
                    joint_area_budget=None,
                    seed=int(row.get("organic_seed", 0)),
                    provenance=case_provenance,
                )
                _, produced = materialize_profile_auxiliaries(
                    context,
                    source_tissue=load_id_mask(row["source_tissue_mask"]),
                    output_dir=output_case_dir,
                )
                produced_by_id = {item.structure_id: item for item in produced}
            except Exception as exc:  # noqa: BLE001 - preserve typed failure
                materialization_error = f"{type(exc).__name__}: {exc}"
        for structure_id in expected_ids:
            output = output_case_dir / f"{structure_id}.png"
            produced = produced_by_id.get(structure_id)
            if produced is not None and output.is_file():
                decoded = np.ascontiguousarray(load_id_mask(output))
                display_path = (
                    str(output.relative_to(root))
                    if output.is_relative_to(root)
                    else f"artifact://profile_auxiliary/{case_id}/{output.name}"
                )
                status = "materialized"
                output_file_sha256 = sha256_file(output)
                output_array_sha256 = array_sha256(decoded)
                producer_id = produced.provenance.get("producer_id")
                producer_version = produced.provenance.get("producer_version")
                producer_provenance = produced.provenance
                actual_paths_by_case.setdefault(case_id, {})[structure_id] = str(
                    output
                )
            else:
                display_path = (
                    f"artifact://profile_auxiliary/{case_id}/{structure_id}.png"
                )
                output_file_sha256 = None
                output_array_sha256 = None
                producer_id = "joint-semantic-topology-auxiliary-v1"
                producer_version = "joint-semantic-topology-auxiliary-v1"
                producer_provenance = None
                if not source["source_tissue_mask"]["authority_verified"]:
                    status = "blocked_unverified_source_tissue"
                elif not provenance["authority_verified"]:
                    status = "blocked_missing_profile_provenance"
                elif materialization_error:
                    status = "materialization_failed"
                else:
                    status = "materialization_output_missing"
            entries.append(
                {
                    "case_id": case_id,
                    "annotation_profile_id": profile_id,
                    "structure_id": structure_id,
                    "classification": "profile_owned",
                    "producer_id": producer_id,
                    "producer_version": producer_version,
                    "source_tissue_mask_sha256": source_rows[case_id].get(
                        "source_tissue_mask_sha256"
                    ),
                    "source_tissue_array_sha256": source["source_tissue_mask"].get(
                        "decoded_array_sha256"
                    ),
                    "output_path": display_path,
                    "output_file_sha256": output_file_sha256,
                    "output_array_sha256": output_array_sha256,
                    "producer_provenance": producer_provenance,
                    "materialization_error": materialization_error,
                    "status": status,
                    "he_or_llm_used": False,
                }
            )
    payload = {
        "schema_version": AUXILIARY_MANIFEST_SCHEMA,
        "producer_source_path": "phase3_joint_edit_refine/auxiliary.py",
        "producer_source_sha256": sha256_file(
            root / "phase3_joint_edit_refine" / "auxiliary.py"
        ),
        "entry_count": len(entries),
        "entries": entries,
        "external_only_structure_ids": sorted(
            EXTERNAL_ONLY_AUXILIARY_STRUCTURES
        ),
        "profile_owned_output_count": sum(
            item["status"] == "materialized" for item in entries
        ),
        "visualization_run": False,
        "api_used": False,
    }
    return (
        {**payload, "manifest_content_sha256": canonical_metadata_sha256(payload)},
        actual_paths_by_case,
    )


def _external_auxiliary_authority(
    *,
    binding_id: str,
    required_structure_ids: tuple[str, ...],
    erratum_entries: tuple[Mapping[str, Any], ...],
    source_tissue_authority: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    by_structure = {
        str(item["structure_id"]): item
        for item in erratum_entries
        if item.get("binding_id") == binding_id
    }
    records = []
    paths = {}
    missing = []
    for structure_id in sorted(required_structure_ids):
        raw = by_structure.get(structure_id)
        failure_codes = []
        declared_path = raw.get("path") if raw else None
        declared_file_sha = raw.get("file_sha256") if raw else None
        declared_array_sha = raw.get("decoded_array_sha256") if raw else None
        provenance = raw.get("provenance") if raw else None
        canonical_path = _resolved_path(declared_path)
        observed_file_sha = None
        observed_array_sha = None
        decoded_shape = None
        distinct_positive_ids = []
        if raw is None:
            failure_codes.append("typed_external_authority_missing")
        elif not isinstance(provenance, Mapping):
            failure_codes.append("external_auxiliary_provenance_missing")
        else:
            if not _is_sha256(declared_file_sha) or not _is_sha256(
                declared_array_sha
            ):
                failure_codes.append("external_auxiliary_digest_unbound")
            if canonical_path is None or not Path(canonical_path).is_file():
                failure_codes.append("external_auxiliary_asset_unavailable")
            else:
                path = Path(canonical_path)
                observed_file_sha = sha256_file(path)
                if observed_file_sha != declared_file_sha:
                    failure_codes.append("external_auxiliary_file_digest_mismatch")
                try:
                    decoded = np.ascontiguousarray(load_id_mask(path))
                    observed_array_sha = array_sha256(decoded)
                    decoded_shape = list(decoded.shape)
                    distinct_positive_ids = sorted(
                        int(value)
                        for value in np.unique(decoded)
                        if int(value) > 0
                    )
                except Exception as exc:  # noqa: BLE001
                    failure_codes.append(
                        f"external_auxiliary_decode_failed:{type(exc).__name__}"
                    )
                else:
                    if observed_array_sha != declared_array_sha:
                        failure_codes.append(
                            "external_auxiliary_array_digest_mismatch"
                        )
                    if decoded_shape != source_tissue_authority.get(
                        "decoded_shape"
                    ):
                        failure_codes.append("external_auxiliary_shape_mismatch")
                    if not distinct_positive_ids:
                        failure_codes.append("external_auxiliary_is_empty")
            source_digest = source_tissue_authority.get("observed_file_sha256")
            if provenance.get("source_tissue_mask_sha256") != source_digest:
                failure_codes.append("external_auxiliary_source_binding_mismatch")
            if provenance.get("output_sha256") != declared_file_sha:
                failure_codes.append("external_auxiliary_output_binding_mismatch")
            if not provenance.get("producer_id") or not provenance.get(
                "producer_version"
            ):
                failure_codes.append("external_auxiliary_producer_missing")
            expected_type = (
                "digest_bound_user_local_roi"
                if structure_id == "local_clearance_roi"
                else "native_gland_instance_annotation"
            )
            if provenance.get("authority_type") != expected_type:
                failure_codes.append(
                    "external_auxiliary_authority_type_mismatch"
                )
            if provenance.get("decoded_array_sha256") != declared_array_sha:
                failure_codes.append("external_auxiliary_array_binding_mismatch")
        verified = not failure_codes
        if verified:
            paths[structure_id] = str(canonical_path)
        else:
            missing.append(structure_id)
        payload = {
            "binding_id": binding_id,
            "structure_id": structure_id,
            "classification": "external_only",
            "declared_path": declared_path,
            "canonical_path": canonical_path,
            "declared_file_sha256": (
                str(declared_file_sha) if _is_sha256(declared_file_sha) else None
            ),
            "observed_file_sha256": observed_file_sha,
            "declared_array_sha256": (
                str(declared_array_sha)
                if _is_sha256(declared_array_sha)
                else None
            ),
            "observed_array_sha256": observed_array_sha,
            "decoded_shape": decoded_shape,
            "distinct_positive_ids": distinct_positive_ids,
            "provenance": (
                dict(provenance) if isinstance(provenance, Mapping) else None
            ),
            "authority_verified": verified,
            "failure_codes": sorted(set(failure_codes)),
            "he_or_llm_used": False,
        }
        records.append(
            {**payload, "record_sha256": canonical_metadata_sha256(payload)}
        )
    return records, paths, missing


def _build_live_case(
    *,
    evaluation: Mapping[str, Any],
    selected_case: Mapping[str, Any],
    source_row: Mapping[str, Any],
    source_authority: Mapping[str, Any],
    profile_provenance: Mapping[str, Any],
    profile_auxiliary_records: list[Mapping[str, Any]],
    profile_auxiliary_paths: Mapping[str, str],
    external_auxiliary_records: list[Mapping[str, Any]],
    external_auxiliary_paths: Mapping[str, str],
    runtime_configuration: Mapping[str, Any],
    joint_repository: JointSkillRepository,
) -> JointCaseContext:
    if profile_provenance.get("authority_verified") is not True:
        raise ValueError("live case requires verified profile provenance authority")
    profile_id = str(evaluation["annotation_profile_id"])
    population_by_profile = {
        "glas-gland-v1": "colorectal-cellvit-source-first-v1",
        "panda-gleason-v1": "prostate-cellvit-source-first-v1",
    }
    primitive_id = str(evaluation["primitive_id"])
    primitive = joint_repository.primitives[primitive_id]
    joint_budget = (
        None
        if primitive.scope == "cell_only"
        else JointAreaBudget.from_value(runtime_configuration["joint_area_budget"])
    )
    auxiliary_paths = {
        **dict(profile_auxiliary_paths),
        **dict(external_auxiliary_paths),
    }
    auxiliary_records = {
        str(item["structure_id"]): dict(item["producer_provenance"])
        for item in profile_auxiliary_records
        if item.get("status") == "materialized"
        and isinstance(item.get("producer_provenance"), Mapping)
    }
    auxiliary_records.update(
        {
            str(item["structure_id"]): dict(item["provenance"])
            for item in external_auxiliary_records
            if item.get("authority_verified")
            and isinstance(item.get("provenance"), Mapping)
        }
    )
    auxiliary_digests = {
        str(item["structure_id"]): str(item["output_file_sha256"])
        for item in profile_auxiliary_records
        if item.get("status") == "materialized"
    }
    auxiliary_digests.update(
        {
            str(item["structure_id"]): str(item["observed_file_sha256"])
            for item in external_auxiliary_records
            if item.get("authority_verified")
        }
    )
    provenance = {
        "source_image_sha256": source_authority["source_image"][
            "observed_file_sha256"
        ],
        "source_tissue_mask_sha256": source_authority["source_tissue_mask"][
            "observed_file_sha256"
        ],
        "source_nuclei_mask_sha256": source_authority["source_nuclei_mask"][
            "observed_file_sha256"
        ],
        **dict(profile_provenance["bound_fields"]),
        "joint_mechanism_id": str(evaluation["mechanism_id"]),
        "joint_primitive_id": primitive_id,
        "available_auxiliary_structures": sorted(auxiliary_paths),
        "auxiliary_structure_sha256": auxiliary_digests,
        "auxiliary_structure_provenance": auxiliary_records,
        "require_mature_probnet_regeneration": True,
    }
    instance_uri = source_row.get("source_nuclei_instances")
    if source_authority["source_nuclei_instances"].get("authority_verified"):
        provenance["source_nuclei_instances_sha256"] = source_authority[
            "source_nuclei_instances"
        ]["observed_file_sha256"]
    else:
        instance_uri = None
    tissue = load_id_mask(source_row["source_tissue_mask"])
    if profile_id == "glas-gland-v1":
        provenance["gland_fine_label_signature"] = sorted(
            int(value)
            for value in np.unique(tissue)
            if int(value) in {5, 11, 12, 13}
        )
    raw = {
        "case_id": str(selected_case["case_id"]),
        "instruction": str(evaluation["instruction"]),
        "source_image_uri": str(source_row["source_image"]),
        "source_tissue_mask_uri": str(source_row["source_tissue_mask"]),
        "source_nuclei_mask_uri": str(source_row["source_nuclei_mask"]),
        "source_nuclei_instances_uri": instance_uri,
        "auxiliary_structure_uris": auxiliary_paths,
        "pathology_domain_id": str(evaluation["pathology_domain_id"]),
        "annotation_profile_id": profile_id,
        "cell_observation_profile_id": "cellvit-five-class-v1",
        "cell_population_profile_id": population_by_profile[profile_id],
        "primitive_id": primitive_id,
        "joint_area_budget": (
            joint_budget.__dict__ if joint_budget is not None else None
        ),
        "cell_count_extent_budget": None,
        "seed": int(selected_case["seed"]),
        "provenance": provenance,
    }
    case, _ = bind_semantic_intent(raw, RuleBasedSemanticParser())
    case.validate_local_inputs()
    return case


def _validate_live_compile_authority(
    *,
    case: JointCaseContext,
    source_authority: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    if not source_authority.get("required_source_authority_verified"):
        raise ValueError("live compiler requires verified source authority")
    unsigned_runtime = dict(runtime)
    declared_runtime_sha = unsigned_runtime.pop("runtime_authority_sha256", None)
    if (
        declared_runtime_sha != canonical_metadata_sha256(unsigned_runtime)
        or not runtime.get("all_required_runtime_assets_verified")
    ):
        raise ValueError("live compiler requires sealed verified runtime authority")
    for asset in runtime.get("external_assets", ()):
        path = Path(str(asset.get("canonical_path"))).resolve(strict=False)
        if (
            asset.get("verified") is not True
            or not path.exists()
            or _runtime_asset_sha256(path) != asset.get("declared_sha256")
            or asset.get("observed_sha256") != asset.get("declared_sha256")
        ):
            raise ValueError("live compiler runtime asset replay mismatch")
    tissue = _inspect_raster_asset(
        uri=case.source_tissue_mask_uri,
        declared_sha256=case.provenance.get("source_tissue_mask_sha256"),
        role="execution_tissue_mask_authority",
        decoder="tissue",
    )
    nuclei = _inspect_raster_asset(
        uri=case.source_nuclei_mask_uri,
        declared_sha256=case.provenance.get("source_nuclei_mask_sha256"),
        role="execution_nuclei_mask_authority",
        decoder="nuclei",
    )
    for role, fresh in (("source_tissue_mask", tissue), ("source_nuclei_mask", nuclei)):
        supplied = source_authority.get(role) or {}
        if (
            not fresh["authority_verified"]
            or fresh["observed_file_sha256"]
            != supplied.get("observed_file_sha256")
            or fresh["decoded_array_sha256"]
            != supplied.get("decoded_array_sha256")
        ):
            raise ValueError("live compiler source authority replay mismatch")
    auxiliary_digests = case.provenance.get("auxiliary_structure_sha256") or {}
    auxiliary_provenance = case.provenance.get(
        "auxiliary_structure_provenance"
    ) or {}
    source_tissue_sha = tissue["observed_file_sha256"]
    for structure_id, uri in sorted(case.auxiliary_structure_uris.items()):
        path = Path(str(uri)).expanduser().resolve(strict=False)
        provenance = auxiliary_provenance.get(structure_id)
        declared_sha = auxiliary_digests.get(structure_id)
        if (
            not path.is_file()
            or not _is_sha256(declared_sha)
            or sha256_file(path) != declared_sha
            or not isinstance(provenance, Mapping)
            or provenance.get("output_sha256") != declared_sha
            or provenance.get("source_tissue_mask_sha256") != source_tissue_sha
            or not provenance.get("producer_id")
            or not provenance.get("producer_version")
        ):
            raise ValueError(
                "live compiler auxiliary authority replay mismatch: "
                + structure_id
            )


def _compile_live_preflight(
    *,
    case: JointCaseContext,
    mechanism_id: str,
    source_authority: Mapping[str, Any],
    runtime: Mapping[str, Any],
    mask_repository: MaskSkillRepository,
    joint_repository: JointSkillRepository,
    skill_audit_sink: dict[str, Any] | None = None,
) -> tuple[JointCaseContext, dict[str, Any], dict[str, Any]]:
    _validate_live_compile_authority(
        case=case,
        source_authority=source_authority,
        runtime=runtime,
    )
    source_tissue = load_id_mask(case.source_tissue_mask_uri)
    source_nuclei = load_nuclei_mask(case.source_nuclei_mask_uri)
    schema = mask_repository.annotation_schema(case.annotation_profile_id)
    scene = build_joint_scene_analysis(
        source_tissue,
        source_nuclei,
        schema=schema,
        pixel_size_um=case.pixel_size_um,
        nuclei_instances_path=case.source_nuclei_instances_uri,
        auxiliary_structure_paths=case.auxiliary_structure_uris,
        auxiliary_structure_provenance=case.provenance.get(
            "auxiliary_structure_provenance", {}
        ),
    )
    case = _with_scene_calibrated_cell_budget(
        case=case,
        scene=scene,
        joint_skills=joint_repository,
    )
    joint_bundle = joint_repository.compose(
        case=case,
        mechanism_id=mechanism_id,
        available_checker_ids=set(JointGateRegistry().available_checker_ids),
        production=False,
    )
    scene_metadata = scene.to_metadata()
    joint_metadata = joint_bundle.to_metadata()
    skill_record = {
        "status": "composed_and_live_bound",
        "joint_bundle": joint_metadata,
        "joint_bundle_sha256": canonical_metadata_sha256(joint_metadata),
        "mask_bundle": None,
        "mask_bundle_sha256": None,
        "scene_graph_sha256": canonical_metadata_sha256(scene_metadata),
        "source_tissue_array_sha256": source_authority["source_tissue_mask"][
            "decoded_array_sha256"
        ],
        "source_nuclei_array_sha256": source_authority["source_nuclei_mask"][
            "decoded_array_sha256"
        ],
        "runtime_authority_sha256": runtime["runtime_authority_sha256"],
    }
    if joint_bundle.primitive.scope == "cell_only":
        if skill_audit_sink is not None:
            skill_audit_sink.update(skill_record)
        workflow = JointPathologyEditWorkflow(
            tissue_planner=object(),
            joint_planner=object(),
            critic=object(),
            mask_skills=mask_repository,
            joint_skills=joint_repository,
            config=JointWorkflowConfig(production=False),
        )
        compiled = workflow._compile_cell_only_candidate_portfolio(
            case=case,
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            schema=schema,
            scene=scene,
            bundle=joint_bundle,
        ).certificates
        portfolio_metadata = compiled.to_metadata()
        survivors = portfolio_metadata["surviving_candidates"]
        vetoed = portfolio_metadata["vetoed_candidates"]
        kind = "cell"
    else:
        allocation = JointFeasibilitySolver().allocate(
            shape=source_tissue.shape,
            budget=case.joint_area_budget,
            bundle=joint_bundle,
        )
        tissue_bundle = mask_repository.compose(
            pathology_domain_id=case.pathology_domain_id,
            annotation_profile_id=case.annotation_profile_id,
            primitive_id=tissue_tool_primitive_id(case.primitive_id),
            production=False,
            available_checker_ids=set(GateRegistry().available_checker_ids),
            case_provenance=case.provenance,
        )
        tissue_bundle = bind_active_bundle_to_case(
            tissue_bundle,
            case=case,
            scene=scene.tissue,
            semantic_primitive_id=case.primitive_id,
        )
        validate_active_bundle_authority(
            tissue_bundle,
            case_provenance=case.provenance,
            require_live_binding=True,
            case=case,
            scene=scene.tissue,
        )
        tissue_metadata = tissue_bundle.to_metadata()
        skill_record["mask_bundle"] = tissue_metadata
        skill_record["mask_bundle_sha256"] = canonical_metadata_sha256(
            tissue_metadata
        )
        if skill_audit_sink is not None:
            skill_audit_sink.update(skill_record)
        nuclei_preflight = build_joint_nuclei_preflight(
            case=case,
            source_tissue=source_tissue,
            schema=schema,
            scene=scene,
            tissue_bundle=tissue_bundle,
            joint_bundle=joint_bundle,
            allocation=allocation,
        )
        if nuclei_preflight.required_auxiliary_missing:
            raise ValueError("required auxiliary missing after live bundle compose")
        if nuclei_preflight.required_provenance_missing:
            raise ValueError("required profile provenance missing after live bundle compose")
        if not nuclei_preflight.feasible_interface_ids:
            raise ValueError("no nuclei-safe executable tissue interface")
        tissue_case = _as_tissue_case(
            case,
            allocation=allocation,
            shape=source_tissue.shape,
        )
        binding = _tissue_portfolio_authority_binding(
            case=case,
            tissue_case=tissue_case,
            source_tissue=source_tissue,
            bundle=joint_bundle,
            tissue_bundle=tissue_bundle,
            allocation=allocation,
            nuclei_preflight=nuclei_preflight,
        )
        compiled = CandidateFeasibilityCompiler(
            gates=GateRegistry()
        ).compile_tissue_portfolio(
            tissue_case=tissue_case,
            source_tissue=source_tissue,
            schema=schema,
            scene=scene,
            tissue_bundle=tissue_bundle,
            joint_bundle=joint_bundle,
            nuclei_preflight=nuclei_preflight,
            authority_binding=binding,
            maximum_candidates=int(
                runtime["preflight_configuration"].get(
                    "maximum_tissue_candidates", 4
                )
            ),
        )
        portfolio_metadata = compiled.to_metadata()
        survivors = portfolio_metadata["surviving_candidates"]
        vetoed = portfolio_metadata["vetoed_candidates"]
        kind = "tissue"
    skill_record["bundle_sha256"] = canonical_metadata_sha256(skill_record)
    if skill_audit_sink is not None:
        skill_audit_sink.clear()
        skill_audit_sink.update(skill_record)
    portfolio = {
        "status": "compiled",
        "portfolio_kind": kind,
        "portfolio_sha256": canonical_metadata_sha256(portfolio_metadata),
        "survivor_count": len(survivors),
        "veto_count": len(vetoed),
        "surviving_certificate_ids": [
            str(item["candidate_id"]) for item in survivors
        ],
        "veto_certificate_ids": [str(item["candidate_id"]) for item in vetoed],
        "certificates": portfolio_metadata,
        "pixels_persisted": False,
        "external_planner_called": False,
    }
    return case, skill_record, portfolio


def build_artifacts(
    *,
    root: Path,
    selection_path: Path,
    source_manifest_path: Path,
    code_commit: str,
    authority_erratum_path: Path | None = None,
    runtime_authority_path: Path | None = None,
    auxiliary_output_dir: Path | None = None,
) -> dict[str, bytes]:
    """Build deterministic authority/portfolio ledgers without execution."""

    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("authority materializer code commit must be one full Git SHA")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "p1-glas-panda-meta-eval-selection-v1":
        raise ValueError("unsupported P1 selection schema")
    if source.get("schema_version") != "p1-glas-panda-source-case-pool-v1":
        raise ValueError("unsupported P1 source-pool schema")
    selection_sha = sha256_file(selection_path)
    source_sha = sha256_file(source_manifest_path)
    if selection.get("source_manifest_sha256") != source_sha:
        raise ValueError("P1 selection is detached from its source case pool")
    evaluations = selection.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 24:
        raise ValueError("P1 authority materialization requires 24 evaluations")
    resources = root / "phase3_joint_edit_refine" / "resources"
    authority_erratum_path = authority_erratum_path or resources / (
        AUTHORITY_ERRATUM_FILENAME
    )
    runtime_authority_path = runtime_authority_path or resources / (
        RUNTIME_AUTHORITY_FILENAME
    )
    auxiliary_output_dir = auxiliary_output_dir or resources / (
        "p1_glas_panda_profile_auxiliary_v1"
    )
    authority_erratum, runtime_input = _load_authority_inputs(
        erratum_path=authority_erratum_path,
        runtime_path=runtime_authority_path,
        selection_sha256=selection_sha,
        source_sha256=source_sha,
    )
    authority_erratum_sha = sha256_file(authority_erratum_path)
    runtime_input_sha = sha256_file(runtime_authority_path)
    raw_cases = tuple(source.get("cases", ()))
    for item in raw_cases:
        if not isinstance(item, Mapping):
            raise TypeError("P1 source case record must be an object")
        _validate_frozen_source_case_record(item)
    raw_source_rows = {str(item["case_id"]): item for item in raw_cases}
    if len(raw_source_rows) != 10:
        raise ValueError(
            "P1 authority materialization requires ten unique source cases"
        )
    erratum_by_case = {
        str(item["case_id"]): item
        for item in authority_erratum["source_case_authority"]
    }
    if set(erratum_by_case) - set(raw_source_rows):
        raise ValueError("P1 authority erratum names a non-frozen case")
    source_rows: dict[str, dict[str, Any]] = {}
    source_erratum_by_case: dict[str, dict[str, Any]] = {}
    for case_id, row in sorted(raw_source_rows.items()):
        effective, binding = _effective_source_row(
            row,
            erratum_entry=erratum_by_case.get(case_id),
        )
        source_rows[case_id] = effective
        source_erratum_by_case[case_id] = binding

    repository = MaskSkillRepository()
    joint_repository = JointSkillRepository()
    source_authority_by_case = {
        case_id: _source_authority(row)
        for case_id, row in sorted(source_rows.items())
    }
    profile_by_case = {
        (str(evaluation["annotation_profile_id"]), str(item["case_id"]))
        for evaluation in evaluations
        for item in evaluation["selected_cases"]
    }
    profile_provenance_by_case = {
        (profile_id, case_id): _profile_required_provenance(
            repository=repository,
            profile_id=profile_id,
            erratum_binding=source_erratum_by_case[case_id],
            source_row=source_rows[case_id],
            source_authority=source_authority_by_case[case_id],
        )
        for profile_id, case_id in sorted(profile_by_case)
    }
    runtime = _runtime_authority(
        root=root,
        selection_runtime=selection.get("runtime_authority") or {},
        runtime_input=runtime_input,
        runtime_input_sha256=runtime_input_sha,
        code_commit=code_commit,
    )
    auxiliary, profile_auxiliary_paths_by_case = _auxiliary_inventory(
        selection=selection,
        source_rows=source_rows,
        source_authority_by_case=source_authority_by_case,
        profile_provenance_by_case=profile_provenance_by_case,
        root=root,
        output_dir=auxiliary_output_dir,
    )
    auxiliary_entries = {
        (item["case_id"], item["structure_id"]): item
        for item in auxiliary["entries"]
    }

    authority_records: list[dict[str, Any]] = []
    preflight_records: list[dict[str, Any]] = []
    expected_case_ids_by_profile: dict[str, tuple[str, ...]] = {}
    semantic_parser = RuleBasedSemanticParser()
    for evaluation_index, evaluation in enumerate(evaluations):
        cases = evaluation.get("selected_cases")
        if not isinstance(cases, list) or len(cases) != 5:
            raise ValueError("every P1 evaluation must retain exactly five cases")
        profile_id = str(evaluation["annotation_profile_id"])
        case_ids = tuple(str(item["case_id"]) for item in cases)
        previous = expected_case_ids_by_profile.setdefault(profile_id, case_ids)
        if previous != case_ids:
            raise ValueError("P1 frozen case identities changed across evaluations")
        semantic_intent = semantic_parser.parse(
            str(evaluation["instruction"])
        ).to_metadata()
        if str(evaluation["primitive_id"]) not in {
            str(item["primitive_id"])
            for item in semantic_intent["primitive_hypotheses"]
        }:
            raise ValueError("P1 evaluation instruction is detached from its primitive")
        for case_index, selected_case in enumerate(cases):
            case_id = str(selected_case["case_id"])
            if selected_case.get("fixed_case_no_replacement") is not True:
                raise ValueError("P1 binding lost fixed-case-no-replacement authority")
            if selected_case.get("execution_allowed") is not False:
                raise ValueError("authority materialization cannot unlock execution")
            source_row = source_rows.get(case_id)
            if source_row is None:
                raise ValueError(f"P1 binding names an unknown source case: {case_id}")
            if selected_case.get("source_case_record_sha256") != source_row.get(
                "case_record_sha256"
            ):
                raise ValueError("P1 binding is detached from source case record")
            profile_provenance = profile_provenance_by_case[(profile_id, case_id)]
            source_authority = source_authority_by_case[case_id]
            required_aux = tuple(
                sorted(
                set(evaluation.get("required_auxiliary_structures") or ())
                )
            )
            profile_aux = [
                auxiliary_entries[(case_id, structure_id)]
                for structure_id in required_aux
                if (case_id, structure_id) in auxiliary_entries
            ]
            binding_id = f"{evaluation['evaluation_id']}::{case_id}"
            external_required = tuple(
                sorted(set(required_aux) & EXTERNAL_ONLY_AUXILIARY_STRUCTURES)
            )
            (
                external_records,
                external_paths,
                external_missing,
            ) = _external_auxiliary_authority(
                binding_id=binding_id,
                required_structure_ids=external_required,
                erratum_entries=tuple(
                    authority_erratum["external_auxiliary_authority"]
                ),
                source_tissue_authority=source_authority["source_tissue_mask"],
            )
            terminal_details = list(source_authority["failure_codes"])
            if not terminal_details and profile_provenance["missing_fields"]:
                terminal_details = [
                    "required_profile_provenance_missing:"
                    + field
                    for field in profile_provenance["missing_fields"]
                ]
            if not terminal_details and profile_provenance["invalid_fields"]:
                terminal_details = [
                    "profile_provenance_invalid:"
                    + field
                    + ":"
                    + reason
                    for field, reason in sorted(
                        profile_provenance["invalid_fields"].items()
                    )
                ]
            if not terminal_details:
                blocked_profile_aux = [
                    item["structure_id"]
                    for item in profile_aux
                    if item["status"] != "materialized"
                ]
                terminal_details = [
                    "profile_owned_auxiliary_not_materialized:" + item
                    for item in blocked_profile_aux
                ]
            if not terminal_details:
                terminal_details = [
                    "external_auxiliary_missing:" + item
                    for item in external_missing
                ]
            if not terminal_details:
                terminal_details = [
                    "runtime_authority_missing:" + item
                    for item in runtime["unverified_external_asset_ids"]
                ]
                terminal_details.extend(
                    "runtime_selection_binding_mismatch:" + item
                    for item in runtime["selection_runtime_binding_mismatches"]
                )
            official_skill_bundle = {
                "status": "not_composed_due_prior_authority_failure",
                "bundle_sha256": None,
            }
            candidate_portfolio = {
                "status": "not_compiled",
                "portfolio_sha256": None,
                "survivor_count": 0,
                "veto_count": 0,
                "surviving_certificate_ids": [],
                "veto_certificate_ids": [],
                "pixels_persisted": False,
                "external_planner_called": False,
            }
            compiled_case = None
            compile_error = None
            partial_skill_bundle: dict[str, Any] = {}
            if not terminal_details:
                try:
                    live_case = _build_live_case(
                        evaluation=evaluation,
                        selected_case=selected_case,
                        source_row=source_row,
                        source_authority=source_authority,
                        profile_provenance=profile_provenance,
                        profile_auxiliary_records=profile_aux,
                        profile_auxiliary_paths=(
                            {
                                structure_id: path
                                for structure_id, path in (
                                    profile_auxiliary_paths_by_case.get(
                                        case_id, {}
                                    )
                                ).items()
                                if structure_id in required_aux
                            }
                        ),
                        external_auxiliary_records=external_records,
                        external_auxiliary_paths=external_paths,
                        runtime_configuration=runtime[
                            "preflight_configuration"
                        ],
                        joint_repository=joint_repository,
                    )
                    (
                        compiled_case,
                        official_skill_bundle,
                        candidate_portfolio,
                    ) = _compile_live_preflight(
                        case=live_case,
                        mechanism_id=str(evaluation["mechanism_id"]),
                        source_authority=source_authority,
                        runtime=runtime,
                        mask_repository=repository,
                        joint_repository=joint_repository,
                        skill_audit_sink=partial_skill_bundle,
                    )
                except Exception as exc:  # noqa: BLE001 - one terminal row
                    compile_error = f"{type(exc).__name__}: {exc}"
                    terminal_details = [compile_error]
                    if partial_skill_bundle:
                        partial_skill_bundle["bundle_sha256"] = (
                            canonical_metadata_sha256(partial_skill_bundle)
                        )
                        official_skill_bundle = partial_skill_bundle
            if source_authority["failure_codes"]:
                terminal_reason = "frozen_source_authority_failed"
                failed_stage = "01_frozen_source_digest_verification"
            elif not profile_provenance["authority_verified"]:
                terminal_reason = "required_profile_provenance_missing"
                failed_stage = "01_frozen_source_digest_verification"
            elif any(item["status"] != "materialized" for item in profile_aux):
                terminal_reason = "profile_owned_auxiliary_not_materialized"
                failed_stage = "02_profile_owned_auxiliary_materialization"
            elif external_missing:
                terminal_reason = "external_auxiliary_authority_missing"
                failed_stage = "03_external_and_runtime_authority"
            elif not runtime["all_required_runtime_assets_verified"]:
                terminal_reason = "runtime_authority_incomplete"
                failed_stage = "03_external_and_runtime_authority"
            elif compile_error is not None and not partial_skill_bundle:
                terminal_reason = "official_skill_bundle_compose_failed"
                failed_stage = "04_official_live_skill_bundle"
            elif (
                compile_error is not None
                or candidate_portfolio["survivor_count"] < 1
            ):
                terminal_reason = "candidate_portfolio_no_survivor"
                failed_stage = "05_deterministic_candidate_compiler"
            else:
                terminal_reason = "eligible_compiler_survivor_available"
                failed_stage = None
            eligible = bool(
                failed_stage is None
                and candidate_portfolio["status"] == "compiled"
                and candidate_portfolio["survivor_count"] > 0
            )
            terminal_status = (
                "eligible"
                if eligible
                else (
                    "abstain"
                    if failed_stage == "05_deterministic_candidate_compiler"
                    else "reject"
                )
            )
            authority = _sealed_record(
                {
                    "schema_version": AUTHORITY_RECORD_SCHEMA,
                    "binding_id": binding_id,
                    "evaluation_index": evaluation_index,
                    "case_index": case_index,
                    "authority_materializer_code_commit": code_commit,
                    "selection_manifest_sha256": selection_sha,
                    "source_manifest_sha256": source_sha,
                    "authority_erratum_manifest_sha256": authority_erratum_sha,
                    "runtime_input_manifest_sha256": runtime_input_sha,
                    "source_case_record_sha256": source_row[
                        "case_record_sha256"
                    ],
                    "source_authority_erratum": source_erratum_by_case[case_id],
                    "fixed_case_no_replacement": True,
                    "pathology_domain_id": evaluation["pathology_domain_id"],
                    "annotation_profile_id": profile_id,
                    "primitive_id": evaluation["primitive_id"],
                    "mechanism_id": evaluation["mechanism_id"],
                    "scenario": semantic_intent["scenario"],
                    "treatment_context": semantic_intent["treatment_context"],
                    "instruction": evaluation["instruction"],
                    "semantic_intent": (
                        compiled_case.semantic_intent
                        if compiled_case is not None
                        else semantic_intent
                    ),
                    "semantic_intent_sha256": canonical_metadata_sha256(
                        compiled_case.semantic_intent
                        if compiled_case is not None
                        else semantic_intent
                    ),
                    "case_id": case_id,
                    "seed": int(selected_case["seed"]),
                    "source_authority": source_authority,
                    "required_profile_provenance": profile_provenance,
                    "required_auxiliary_structure_ids": required_aux,
                    "profile_owned_auxiliary_records": profile_aux,
                    "external_auxiliary_authority_records": external_records,
                    "external_only_auxiliary_missing": external_missing,
                    "runtime_authority_sha256": runtime[
                        "runtime_authority_sha256"
                    ],
                    "official_skill_bundle": official_skill_bundle,
                    "terminal_status": terminal_status,
                    "terminal_reason_code": terminal_reason,
                    "terminal_reason_details": terminal_details,
                    "failed_stage": failed_stage,
                    "eligible_for_later_visualization": eligible,
                    "planner_called": False,
                    "executor_called": False,
                    "target_mask_created": False,
                    "visualization_run": False,
                    "api_used": False,
                }
            )
            authority_records.append(authority)
            preflight_records.append(
                _sealed_record(
                    {
                        "schema_version": PREFLIGHT_RECORD_SCHEMA,
                        "binding_id": binding_id,
                        "authority_record_sha256": authority["record_sha256"],
                        "ordered_stages": [
                            {
                                "stage_id": "01_frozen_source_digest_verification",
                                "status": (
                                    "failed"
                                    if failed_stage
                                    == "01_frozen_source_digest_verification"
                                    else "passed"
                                ),
                            },
                            {
                                "stage_id": "02_profile_owned_auxiliary_materialization",
                                "status": (
                                    "failed"
                                    if failed_stage
                                    == "02_profile_owned_auxiliary_materialization"
                                    else (
                                        "not_run_due_prior_failure"
                                        if failed_stage
                                        == "01_frozen_source_digest_verification"
                                        else "passed"
                                    )
                                ),
                            },
                            {
                                "stage_id": "03_external_and_runtime_authority",
                                "status": (
                                    "failed"
                                    if failed_stage
                                    == "03_external_and_runtime_authority"
                                    else (
                                        "not_run_due_prior_failure"
                                        if failed_stage
                                        in {
                                            "01_frozen_source_digest_verification",
                                            "02_profile_owned_auxiliary_materialization",
                                        }
                                        else "passed"
                                    )
                                ),
                            },
                            {
                                "stage_id": "04_official_live_skill_bundle",
                                "status": (
                                    "not_run_due_prior_failure"
                                    if failed_stage
                                    in {
                                        "01_frozen_source_digest_verification",
                                        "02_profile_owned_auxiliary_materialization",
                                        "03_external_and_runtime_authority",
                                    }
                                    else (
                                        "failed"
                                        if failed_stage
                                        == "04_official_live_skill_bundle"
                                        else "passed"
                                    )
                                ),
                            },
                            {
                                "stage_id": "05_deterministic_candidate_compiler",
                                "status": (
                                    "not_run_due_prior_failure"
                                    if failed_stage
                                    in {
                                        "01_frozen_source_digest_verification",
                                        "02_profile_owned_auxiliary_materialization",
                                        "03_external_and_runtime_authority",
                                        "04_official_live_skill_bundle",
                                    }
                                    else (
                                        "failed"
                                        if failed_stage
                                        == "05_deterministic_candidate_compiler"
                                        else "passed"
                                    )
                                ),
                            },
                        ],
                        "candidate_portfolio": candidate_portfolio,
                        "terminal_status": terminal_status,
                        "terminal_reason_code": terminal_reason,
                        "terminal_reason_details": terminal_details,
                        "eligible_for_later_visualization": eligible,
                        "planner_called": False,
                        "executor_called": False,
                        "visualization_run": False,
                        "api_used": False,
                    }
                )
            )

    if len(authority_records) != 120 or len(preflight_records) != 120:
        raise ValueError("P1 authority ledgers must contain exactly 120 bindings")
    authority_bytes = b"".join(
        _canonical_json_bytes(item) + b"\n" for item in authority_records
    )
    preflight_bytes = b"".join(
        _canonical_json_bytes(item) + b"\n" for item in preflight_records
    )
    auxiliary_bytes = _canonical_json_bytes(auxiliary, indent=2)
    status_header = (
        "schema_version\tevaluation_id\tcase_id\tannotation_profile_id\t"
        "mechanism_id\tprimitive_id\tstatus\teligible_for_later_visualization\t"
        "terminal_reason_code\tterminal_reason_details\n"
    )
    status_rows = []
    for item in authority_records:
        evaluation_id = item["binding_id"].removesuffix(
            "::" + item["case_id"]
        )
        status_rows.append(
            "\t".join(
                (
                    STATUS_TABLE_SCHEMA,
                    evaluation_id,
                    item["case_id"],
                    item["annotation_profile_id"],
                    item["mechanism_id"],
                    item["primitive_id"],
                    item["terminal_status"],
                    (
                        "true"
                        if item["eligible_for_later_visualization"]
                        else "false"
                    ),
                    item["terminal_reason_code"],
                    "|".join(item["terminal_reason_details"]),
                )
            )
            + "\n"
        )
    status_bytes = (status_header + "".join(status_rows)).encode("utf-8")

    declared_missing_field_count = sum(
        len(item.get("missing_source_asset_digests") or ())
        for evaluation in evaluations
        for item in evaluation["selected_cases"]
    )
    declared_missing_binding_count = sum(
        bool(item.get("missing_source_asset_digests"))
        for evaluation in evaluations
        for item in evaluation["selected_cases"]
    )
    required_source_digest_fields = (
        "source_image_sha256",
        "source_tissue_mask_sha256",
        "source_nuclei_mask_sha256",
    )
    effective_missing_field_count = sum(
        not _is_sha256(source_rows[str(item["case_id"])].get(field))
        for evaluation in evaluations
        for item in evaluation["selected_cases"]
        for field in required_source_digest_fields
    )
    effective_missing_binding_count = sum(
        any(
            not _is_sha256(source_rows[str(item["case_id"])].get(field))
            for field in required_source_digest_fields
        )
        for evaluation in evaluations
        for item in evaluation["selected_cases"]
    )
    external_missing_before = sum(
        len(
            set(evaluation.get("required_auxiliary_structures") or ())
            & EXTERNAL_ONLY_AUXILIARY_STRUCTURES
            - set(item.get("available_auxiliary_structures") or ())
        )
        for evaluation in evaluations
        for item in evaluation["selected_cases"]
    )
    external_missing_after = sum(
        len(item["external_only_auxiliary_missing"])
        for item in authority_records
    )
    roi_missing_before = sum(
        "local_clearance_roi"
        in set(evaluation.get("required_auxiliary_structures") or ())
        and "local_clearance_roi"
        not in set(item.get("available_auxiliary_structures") or ())
        for evaluation in evaluations
        for item in evaluation["selected_cases"]
    )
    roi_missing_after = sum(
        "local_clearance_roi" in item["external_only_auxiliary_missing"]
        for item in authority_records
    )
    reason_counts = dict(
        sorted(Counter(item["terminal_reason_code"] for item in authority_records).items())
    )
    status_counts = dict(
        Counter(item["terminal_status"] for item in authority_records)
    )
    for status in ("eligible", "reject", "abstain"):
        status_counts.setdefault(status, 0)
    summary_payload = {
        "schema_version": AUTHORITY_MANIFEST_SCHEMA,
        "production_status": "shadow_only",
        "execution_status": "blocked_pending_authority_and_surviving_certificates",
        "authority_materializer_code_commit": code_commit,
        "selection_manifest": _portable_input_path(selection_path, root=root),
        "selection_manifest_sha256": selection_sha,
        "source_manifest": _portable_input_path(
            source_manifest_path, root=root
        ),
        "source_manifest_sha256": source_sha,
        "authority_erratum_manifest": _portable_input_path(
            authority_erratum_path, root=root
        ),
        "authority_erratum_manifest_sha256": authority_erratum_sha,
        "runtime_input_manifest": _portable_input_path(
            runtime_authority_path, root=root
        ),
        "runtime_input_manifest_sha256": runtime_input_sha,
        "frozen_binding_count": 120,
        "evaluation_count": 24,
        "fixed_cases_per_evaluation": 5,
        "frozen_case_replacement_allowed": False,
        "authority_records": OUTPUT_FILENAMES["authority"],
        "authority_records_sha256": _sha256_bytes(authority_bytes),
        "auxiliary_materialization_manifest": OUTPUT_FILENAMES["auxiliary"],
        "auxiliary_materialization_manifest_sha256": _sha256_bytes(
            auxiliary_bytes
        ),
        "candidate_preflight_records": OUTPUT_FILENAMES["preflight"],
        "candidate_preflight_records_sha256": _sha256_bytes(preflight_bytes),
        "full_case_status_table": OUTPUT_FILENAMES["status_table"],
        "full_case_status_table_sha256": _sha256_bytes(status_bytes),
        "runtime_authority": runtime,
        "before_after_counts": {
            "bindings_with_missing_source_digest": {
                "before": declared_missing_binding_count,
                "after": effective_missing_binding_count,
            },
            "source_digest_fields_missing": {
                "before": declared_missing_field_count,
                "after": effective_missing_field_count,
            },
            "binding_external_auxiliary_missing": {
                "before": external_missing_before,
                "after": external_missing_after,
            },
            "binding_local_clearance_roi_missing": {
                "before": roi_missing_before,
                "after": roi_missing_after,
            },
            "selection_runtime_digest_fields_missing": {
                "before": len(
                    runtime["selection_runtime_digest_fields_missing"]
                ),
                "after": len(
                    runtime["effective_runtime_digest_fields_missing"]
                ),
            },
            "profile_owned_auxiliary_outputs_materialized": {
                "before": 0,
                "after": auxiliary["profile_owned_output_count"],
            },
            "bindings_missing_profile_provenance": {
                "before": 120,
                "after": sum(
                    not item["required_profile_provenance"]["authority_verified"]
                    for item in authority_records
                ),
            },
        },
        "status_counts": status_counts,
        "terminal_reason_counts": reason_counts,
        "all_required_authority_bound": all(
            item["terminal_status"] != "reject" for item in authority_records
        ),
        "all_candidate_portfolios_compiled": all(
            item["candidate_portfolio"]["status"] == "compiled"
            for item in preflight_records
        ),
        "eligible_for_later_visualization_count": status_counts["eligible"],
        "planner_called": False,
        "executor_called": False,
        "target_mask_created": False,
        "source_asset_mutated": False,
        "frozen_cases_changed": False,
        "visualization_run": False,
        "api_used": False,
        "generated_he_run": False,
    }
    summary = {
        **summary_payload,
        "manifest_content_sha256": canonical_metadata_sha256(summary_payload),
    }
    artifacts = {
        OUTPUT_FILENAMES["summary"]: _canonical_json_bytes(summary, indent=2),
        OUTPUT_FILENAMES["authority"]: authority_bytes,
        OUTPUT_FILENAMES["auxiliary"]: auxiliary_bytes,
        OUTPUT_FILENAMES["preflight"]: preflight_bytes,
        OUTPUT_FILENAMES["status_table"]: status_bytes,
    }
    validate_artifacts(artifacts)
    return artifacts


def validate_artifacts(artifacts: Mapping[str, bytes]) -> None:
    expected_names = set(OUTPUT_FILENAMES.values())
    if set(artifacts) != expected_names:
        raise ValueError("P1 authority artifact set is incomplete")
    summary = json.loads(artifacts[OUTPUT_FILENAMES["summary"]])
    if summary.get("schema_version") != AUTHORITY_MANIFEST_SCHEMA:
        raise ValueError("unsupported P1 authority manifest schema")
    declared_summary_digest = summary.get("manifest_content_sha256")
    unsigned_summary = dict(summary)
    unsigned_summary.pop("manifest_content_sha256", None)
    if declared_summary_digest != canonical_metadata_sha256(unsigned_summary):
        raise ValueError("P1 authority manifest content digest mismatch")
    if any(
        summary.get(field) is not False
        for field in (
            "frozen_case_replacement_allowed",
            "planner_called",
            "executor_called",
            "target_mask_created",
            "source_asset_mutated",
            "frozen_cases_changed",
            "visualization_run",
            "api_used",
            "generated_he_run",
        )
    ):
        raise ValueError("P1 authority manifest crosses a preflight safety boundary")
    bindings = [
        json.loads(line)
        for line in artifacts[OUTPUT_FILENAMES["authority"]].splitlines()
        if line.strip()
    ]
    preflight = [
        json.loads(line)
        for line in artifacts[OUTPUT_FILENAMES["preflight"]].splitlines()
        if line.strip()
    ]
    if len(bindings) != 120 or len(preflight) != 120:
        raise ValueError("P1 authority/preflight ledgers must each contain 120 rows")
    if len({item["binding_id"] for item in bindings}) != 120:
        raise ValueError("P1 authority binding IDs are not unique")
    if [item["binding_id"] for item in bindings] != [
        item["binding_id"] for item in preflight
    ]:
        raise ValueError("P1 authority and preflight ledger order differs")
    for item in (*bindings, *preflight):
        expected = _sealed_record(item)["record_sha256"]
        if item.get("record_sha256") != expected:
            raise ValueError("P1 authority record digest mismatch")
        if (
            item.get("planner_called") is not False
            or item.get("executor_called") is not False
            or item.get("visualization_run") is not False
            or item.get("api_used") is not False
        ):
            raise ValueError("P1 preflight record illegally enables execution")
    if any(
        item.get("fixed_case_no_replacement") is not True
        or item.get("terminal_status") not in {"eligible", "reject", "abstain"}
        for item in bindings
    ):
        raise ValueError("P1 authority ledger changed or unlocked a frozen case")
    if any(
        item.get("semantic_intent_sha256")
        != canonical_metadata_sha256(item.get("semantic_intent"))
        for item in bindings
    ):
        raise ValueError("P1 semantic intent binding digest mismatch")
    runtime = summary.get("runtime_authority") or {}
    for authority, preflight_item in zip(bindings, preflight):
        eligible = bool(authority["eligible_for_later_visualization"])
        if eligible != bool(preflight_item["eligible_for_later_visualization"]):
            raise ValueError("P1 authority and preflight eligibility differ")
        portfolio = preflight_item.get("candidate_portfolio") or {}
        if portfolio.get("external_planner_called") is not False:
            raise ValueError("P1 compiler portfolio crossed Planner authority")
        if eligible:
            if (
                authority.get("terminal_status") != "eligible"
                or authority.get("failed_stage") is not None
                or authority.get("official_skill_bundle", {}).get("status")
                != "composed_and_live_bound"
                or portfolio.get("status") != "compiled"
                or int(portfolio.get("survivor_count", 0)) < 1
                or not _is_sha256(portfolio.get("portfolio_sha256"))
                or portfolio.get("portfolio_sha256")
                != canonical_metadata_sha256(portfolio.get("certificates"))
                or not runtime.get("all_required_runtime_assets_verified")
                or not authority.get("source_authority", {}).get(
                    "required_source_authority_verified"
                )
                or not authority.get("required_profile_provenance", {}).get(
                    "authority_verified"
                )
                or authority.get("external_only_auxiliary_missing")
                or any(
                    item.get("status") != "materialized"
                    for item in authority.get(
                        "profile_owned_auxiliary_records", ()
                    )
                )
                or any(
                    item.get("status") != "passed"
                    for item in preflight_item.get("ordered_stages", ())
                )
            ):
                raise ValueError("P1 eligible record lacks complete live authority")
        elif portfolio.get("status") == "not_compiled" and (
            portfolio.get("survivor_count") != 0
            or portfolio.get("portfolio_sha256") is not None
        ):
            raise ValueError("P1 uncompiled portfolio has survivor authority")
        bundle_text = json.dumps(
            authority.get("official_skill_bundle"),
            ensure_ascii=False,
            sort_keys=True,
        ).casefold()
        if any(
            token in bundle_text
            for token in (
                "source_image_uri",
                "raw h&e",
                "h&e crop",
                "h&e overlay",
            )
        ):
            raise ValueError("P1 execution skill bundle contains reader-only raster authority")
    for field, filename in (
        ("authority_records_sha256", OUTPUT_FILENAMES["authority"]),
        (
            "auxiliary_materialization_manifest_sha256",
            OUTPUT_FILENAMES["auxiliary"],
        ),
        (
            "candidate_preflight_records_sha256",
            OUTPUT_FILENAMES["preflight"],
        ),
        ("full_case_status_table_sha256", OUTPUT_FILENAMES["status_table"]),
    ):
        if summary.get(field) != _sha256_bytes(artifacts[filename]):
            raise ValueError(f"P1 authority artifact digest mismatch: {filename}")
    auxiliary = json.loads(artifacts[OUTPUT_FILENAMES["auxiliary"]])
    auxiliary_digest = auxiliary.get("manifest_content_sha256")
    unsigned_auxiliary = dict(auxiliary)
    unsigned_auxiliary.pop("manifest_content_sha256", None)
    if auxiliary_digest != canonical_metadata_sha256(unsigned_auxiliary):
        raise ValueError("P1 auxiliary manifest content digest mismatch")
    for item in auxiliary.get("entries", ()):
        if item["structure_id"] in EXTERNAL_ONLY_AUXILIARY_STRUCTURES:
            raise ValueError("external-only auxiliary was self-materialized")
        if item["status"] == "materialized" and not (
            _is_sha256(item.get("output_file_sha256"))
            and _is_sha256(item.get("output_array_sha256"))
        ):
            raise ValueError("profile-owned auxiliary lacks output authority")
    runtime_digest = runtime.get("runtime_authority_sha256")
    unsigned_runtime = dict(runtime)
    unsigned_runtime.pop("runtime_authority_sha256", None)
    if runtime_digest != canonical_metadata_sha256(unsigned_runtime):
        raise ValueError("P1 runtime authority content digest mismatch")
    status_lines = artifacts[OUTPUT_FILENAMES["status_table"]].splitlines()
    if len(status_lines) != 121 or not status_lines[0].startswith(
        b"schema_version\tevaluation_id\tcase_id\t"
    ):
        raise ValueError("P1 full status table does not contain all 120 bindings")


def validate_committed_artifacts(*, root: Path, resources: Path) -> None:
    artifacts = {
        filename: (resources / filename).read_bytes()
        for filename in OUTPUT_FILENAMES.values()
    }
    validate_artifacts(artifacts)
    summary = json.loads(artifacts[OUTPUT_FILENAMES["summary"]])
    regenerated = build_artifacts(
        root=root,
        selection_path=root
        / "phase3_joint_edit_refine"
        / "resources"
        / "p1_glas_panda_meta_eval_selection_v1.json",
        source_manifest_path=root
        / "phase3_joint_edit_refine"
        / "resources"
        / "p1_glas_panda_source_case_pool_v1.json",
        code_commit=str(summary["authority_materializer_code_commit"]),
    )
    if artifacts != regenerated:
        raise ValueError("P1 authority materialization artifacts drifted")
