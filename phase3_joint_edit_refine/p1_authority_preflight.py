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
from phase3_mask_edit_refine.skills import SkillRepository as MaskSkillRepository

from .nuclei import load_nuclei_mask
from .portfolio_authority import array_sha256, canonical_metadata_sha256
from .semantic_parser import RuleBasedSemanticParser

AUTHORITY_MANIFEST_SCHEMA = "p1-glas-panda-authority-manifest-v1"
AUTHORITY_RECORD_SCHEMA = "p1-glas-panda-authority-record-v1"
AUXILIARY_MANIFEST_SCHEMA = "p1-profile-auxiliary-materialization-v1"
PREFLIGHT_RECORD_SCHEMA = "p1-deterministic-candidate-preflight-v1"
STATUS_TABLE_SCHEMA = "p1-preflight-status-table-v1"

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
RUNTIME_DIGEST_FIELDS = (
    "mature_probnet_checkpoint_sha256",
    "frozen_spatial_ranker_sha256",
    "instance_library_sha256",
    "generator_checkpoint_sha256",
)
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


def _runtime_authority(
    *,
    root: Path,
    selection_runtime: Mapping[str, Any],
    code_commit: str,
) -> dict[str, Any]:
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
    external_assets = [
        {
            "asset_id": "mature_probnet_checkpoint",
            "canonical_path": None,
            "declared_sha256": selection_runtime.get(
                "mature_probnet_checkpoint_sha256"
            ),
            "observed_sha256": None,
            "verified": False,
            "failure_code": "runtime_asset_path_unbound",
        },
        {
            "asset_id": "frozen_probnet_spatial_ranker_checkpoint",
            "canonical_path": None,
            "declared_sha256": selection_runtime.get(
                "frozen_spatial_ranker_sha256"
            ),
            "observed_sha256": None,
            "verified": False,
            "failure_code": "runtime_asset_digest_and_path_unbound",
        },
        {
            "asset_id": "glas_nucleus_instance_library",
            "canonical_path": None,
            "declared_sha256": None,
            "observed_sha256": None,
            "verified": False,
            "failure_code": "runtime_asset_digest_and_path_unbound",
        },
        {
            "asset_id": "panda_nucleus_instance_library",
            "canonical_path": None,
            "declared_sha256": None,
            "observed_sha256": None,
            "verified": False,
            "failure_code": "runtime_asset_digest_and_path_unbound",
        },
        {
            "asset_id": "later_he_generator_checkpoint",
            "canonical_path": None,
            "declared_sha256": selection_runtime.get(
                "generator_checkpoint_sha256"
            ),
            "observed_sha256": None,
            "verified": False,
            "failure_code": "runtime_asset_digest_and_path_unbound",
            "reader_side_only": True,
            "used_during_this_stage": False,
        },
    ]
    missing_selection_fields = [
        field for field in RUNTIME_DIGEST_FIELDS if not _is_sha256(selection_runtime.get(field))
    ]
    payload = {
        "schema_version": "p1-runtime-authority-materialization-v1",
        "authority_materializer_code_commit": code_commit,
        "selection_runtime_authority": dict(selection_runtime),
        "selection_runtime_digest_fields_missing": missing_selection_fields,
        "code_assets": code_assets,
        "runtime_code_set_sha256": canonical_metadata_sha256(code_assets),
        "external_assets": external_assets,
        "unverified_external_asset_ids": [
            item["asset_id"] for item in external_assets if not item["verified"]
        ],
        "all_required_runtime_assets_verified": False,
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
    required = (image, tissue, nuclei)
    failure_codes = sorted(
        {
            code
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
        failure_codes.append("source_mask_shape_mismatch")
    payload = {
        "source_image": image,
        "source_tissue_mask": tissue,
        "source_nuclei_mask": nuclei,
        "source_nuclei_instances": instances,
        "tissue_nuclei_shape_aligned": shape_aligned,
        "required_source_authority_verified": not failure_codes,
        "failure_codes": sorted(set(failure_codes)),
    }
    return {**payload, "source_authority_sha256": canonical_metadata_sha256(payload)}


def _profile_required_provenance(
    *, repository: MaskSkillRepository,
    profile_id: str,
    source_row: Mapping[str, Any],
) -> dict[str, Any]:
    package = repository.get(profile_id, expected_kind="annotation_profile")
    required = tuple(
        str(value)
        for value in package.capabilities.get("required_provenance_fields", ())
    )
    supplied = source_row.get("profile_provenance")
    supplied_mapping = dict(supplied) if isinstance(supplied, Mapping) else {}
    bound = {
        field: supplied_mapping[field]
        for field in required
        if supplied_mapping.get(field) not in (None, "")
    }
    return {
        "required_fields": list(required),
        "bound_fields": bound,
        "missing_fields": sorted(set(required) - set(bound)),
        "provenance_source": "source_case_pool.profile_provenance_only",
        "inferred_from_he_or_untyped_metadata": False,
    }


def _auxiliary_inventory(
    *,
    selection: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, Any]],
    source_authority_by_case: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> dict[str, Any]:
    profile_by_case: dict[str, str] = {}
    for evaluation in selection["evaluations"]:
        for row in evaluation["selected_cases"]:
            profile_by_case[str(row["case_id"])] = str(
                evaluation["annotation_profile_id"]
            )
    entries = []
    for case_id in sorted(profile_by_case):
        profile_id = profile_by_case[case_id]
        source = source_authority_by_case[case_id]
        for structure_id in PROFILE_OWNED_AUXILIARY_STRUCTURES.get(profile_id, ()):
            output = (
                root
                / "phase3_joint_edit_refine"
                / "resources"
                / "p1_glas_panda_profile_auxiliary_v1"
                / case_id
                / f"{structure_id}.png"
            )
            entries.append(
                {
                    "case_id": case_id,
                    "annotation_profile_id": profile_id,
                    "structure_id": structure_id,
                    "classification": "profile_owned",
                    "producer_id": "joint-semantic-topology-auxiliary-v1",
                    "producer_version": "joint-semantic-topology-auxiliary-v1",
                    "source_tissue_mask_sha256": source_rows[case_id].get(
                        "source_tissue_mask_sha256"
                    ),
                    "source_tissue_array_sha256": source["source_tissue_mask"].get(
                        "decoded_array_sha256"
                    ),
                    "output_path": str(output),
                    "output_file_sha256": None,
                    "output_array_sha256": None,
                    "status": (
                        "blocked_unverified_source_tissue"
                        if not source["source_tissue_mask"]["authority_verified"]
                        else "pending_deterministic_materialization"
                    ),
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
    return {**payload, "manifest_content_sha256": canonical_metadata_sha256(payload)}


def _external_auxiliary_missing(
    *, evaluation: Mapping[str, Any],
    selected_case: Mapping[str, Any],
) -> list[str]:
    required = set(evaluation.get("required_auxiliary_structures") or ())
    available = set(selected_case.get("available_auxiliary_structures") or ())
    return sorted((required & EXTERNAL_ONLY_AUXILIARY_STRUCTURES) - available)


def build_artifacts(
    *,
    root: Path,
    selection_path: Path,
    source_manifest_path: Path,
    code_commit: str,
) -> dict[str, bytes]:
    """Build deterministic ledgers without invoking a Planner or executor."""

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
    source_rows = {
        str(item["case_id"]): item for item in source.get("cases", ())
    }
    if len(source_rows) != 10:
        raise ValueError("P1 authority materialization requires ten source cases")

    repository = MaskSkillRepository()
    source_authority_by_case = {
        case_id: _source_authority(row)
        for case_id, row in sorted(source_rows.items())
    }
    profile_provenance_by_case: dict[tuple[str, str], dict[str, Any]] = {}
    runtime = _runtime_authority(
        root=root,
        selection_runtime=selection.get("runtime_authority") or {},
        code_commit=code_commit,
    )
    auxiliary = _auxiliary_inventory(
        selection=selection,
        source_rows=source_rows,
        source_authority_by_case=source_authority_by_case,
        root=root,
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
            key = (profile_id, case_id)
            profile_provenance = profile_provenance_by_case.setdefault(
                key,
                _profile_required_provenance(
                    repository=repository,
                    profile_id=profile_id,
                    source_row=source_row,
                ),
            )
            source_authority = source_authority_by_case[case_id]
            required_aux = sorted(
                set(evaluation.get("required_auxiliary_structures") or ())
            )
            profile_aux = [
                auxiliary_entries[(case_id, structure_id)]
                for structure_id in required_aux
                if (case_id, structure_id) in auxiliary_entries
            ]
            external_missing = _external_auxiliary_missing(
                evaluation=evaluation,
                selected_case=selected_case,
            )
            binding_id = f"{evaluation['evaluation_id']}::{case_id}"
            terminal_details = list(source_authority["failure_codes"])
            if not terminal_details and profile_provenance["missing_fields"]:
                terminal_details = [
                    "required_profile_provenance_missing:"
                    + field
                    for field in profile_provenance["missing_fields"]
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
            if not terminal_details:
                # This branch deliberately remains fail closed.  Reaching it
                # requires a new reviewed compiler run that persists immutable
                # survivor/veto certificates; simply toggling a field is not
                # an authority materialization.
                terminal_details = ["candidate_portfolio_not_compiled"]
            if source_authority["failure_codes"]:
                terminal_reason = "frozen_source_authority_failed"
                failed_stage = "01_frozen_source_digest_verification"
            elif profile_provenance["missing_fields"]:
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
            else:
                terminal_reason = "candidate_portfolio_not_compiled"
                failed_stage = "05_deterministic_candidate_compiler"
            authority = _sealed_record(
                {
                    "schema_version": AUTHORITY_RECORD_SCHEMA,
                    "binding_id": binding_id,
                    "evaluation_index": evaluation_index,
                    "case_index": case_index,
                    "authority_materializer_code_commit": code_commit,
                    "selection_manifest_sha256": selection_sha,
                    "source_manifest_sha256": source_sha,
                    "source_case_record_sha256": source_row[
                        "case_record_sha256"
                    ],
                    "fixed_case_no_replacement": True,
                    "pathology_domain_id": evaluation["pathology_domain_id"],
                    "annotation_profile_id": profile_id,
                    "primitive_id": evaluation["primitive_id"],
                    "mechanism_id": evaluation["mechanism_id"],
                    "scenario": semantic_intent["scenario"],
                    "treatment_context": semantic_intent["treatment_context"],
                    "instruction": evaluation["instruction"],
                    "semantic_intent": semantic_intent,
                    "semantic_intent_sha256": canonical_metadata_sha256(
                        semantic_intent
                    ),
                    "case_id": case_id,
                    "seed": int(selected_case["seed"]),
                    "source_authority": source_authority,
                    "required_profile_provenance": profile_provenance,
                    "required_auxiliary_structure_ids": required_aux,
                    "profile_owned_auxiliary_records": profile_aux,
                    "external_only_auxiliary_missing": external_missing,
                    "runtime_authority_sha256": runtime[
                        "runtime_authority_sha256"
                    ],
                    "official_skill_bundle": {
                        "status": "not_composed_due_prior_authority_failure",
                        "bundle_sha256": None,
                    },
                    "terminal_status": "reject",
                    "terminal_reason_code": terminal_reason,
                    "terminal_reason_details": terminal_details,
                    "failed_stage": failed_stage,
                    "eligible_for_later_visualization": False,
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
                                "status": "not_run_due_prior_failure",
                            },
                            {
                                "stage_id": "05_deterministic_candidate_compiler",
                                "status": "not_run_due_prior_failure",
                            },
                        ],
                        "candidate_portfolio": {
                            "status": "not_compiled",
                            "portfolio_sha256": None,
                            "survivor_count": 0,
                            "veto_count": 0,
                            "surviving_certificate_ids": [],
                            "veto_certificate_ids": [],
                            "pixels_persisted": False,
                        },
                        "terminal_status": "reject",
                        "terminal_reason_code": terminal_reason,
                        "terminal_reason_details": terminal_details,
                        "eligible_for_later_visualization": False,
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
                    "false",
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
    external_missing_before = sum(
        len(
            set(evaluation.get("required_auxiliary_structures") or ())
            & EXTERNAL_ONLY_AUXILIARY_STRUCTURES
            - set(item.get("available_auxiliary_structures") or ())
        )
        for evaluation in evaluations
        for item in evaluation["selected_cases"]
    )
    roi_missing_before = sum(
        "local_clearance_roi"
        in _external_auxiliary_missing(
            evaluation=evaluation,
            selected_case=item,
        )
        for evaluation in evaluations
        for item in evaluation["selected_cases"]
    )
    reason_counts = dict(
        sorted(Counter(item["terminal_reason_code"] for item in authority_records).items())
    )
    summary_payload = {
        "schema_version": AUTHORITY_MANIFEST_SCHEMA,
        "production_status": "shadow_only",
        "execution_status": "blocked_pending_authority_and_surviving_certificates",
        "authority_materializer_code_commit": code_commit,
        "selection_manifest": str(selection_path.relative_to(root)),
        "selection_manifest_sha256": selection_sha,
        "source_manifest": str(source_manifest_path.relative_to(root)),
        "source_manifest_sha256": source_sha,
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
                "after": declared_missing_binding_count,
            },
            "source_digest_fields_missing": {
                "before": declared_missing_field_count,
                "after": declared_missing_field_count,
            },
            "binding_external_auxiliary_missing": {
                "before": external_missing_before,
                "after": external_missing_before,
            },
            "binding_local_clearance_roi_missing": {
                "before": roi_missing_before,
                "after": roi_missing_before,
            },
            "selection_runtime_digest_fields_missing": {
                "before": len(
                    runtime["selection_runtime_digest_fields_missing"]
                ),
                "after": len(runtime["selection_runtime_digest_fields_missing"]),
            },
            "profile_owned_auxiliary_outputs_materialized": {
                "before": 0,
                "after": auxiliary["profile_owned_output_count"],
            },
        },
        "status_counts": {"eligible": 0, "reject": 120, "abstain": 0},
        "terminal_reason_counts": reason_counts,
        "all_required_authority_bound": False,
        "all_candidate_portfolios_compiled": False,
        "eligible_for_later_visualization_count": 0,
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
            item.get("eligible_for_later_visualization") is not False
            or item.get("planner_called") is not False
            or item.get("executor_called") is not False
            or item.get("visualization_run") is not False
            or item.get("api_used") is not False
        ):
            raise ValueError("P1 preflight record illegally enables execution")
    if any(
        item.get("fixed_case_no_replacement") is not True
        or item.get("terminal_status") not in {"reject", "abstain"}
        for item in bindings
    ):
        raise ValueError("P1 authority ledger changed or unlocked a frozen case")
    if any(
        item.get("semantic_intent_sha256")
        != canonical_metadata_sha256(item.get("semantic_intent"))
        for item in bindings
    ):
        raise ValueError("P1 semantic intent binding digest mismatch")
    if any(
        item.get("candidate_portfolio", {}).get("status") != "not_compiled"
        or item.get("candidate_portfolio", {}).get("survivor_count") != 0
        or item.get("candidate_portfolio", {}).get("portfolio_sha256") is not None
        for item in preflight
    ):
        raise ValueError("P1 candidate portfolio lacks reviewed survivor authority")
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
    runtime = summary.get("runtime_authority") or {}
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
