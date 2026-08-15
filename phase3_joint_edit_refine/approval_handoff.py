"""Promote one explicitly user-approved audited candidate to Generate.

The mask workflow deliberately stops at ``review_required`` when its research
critic abstains.  This module does not rerun or reinterpret that workflow.  It
hash-validates the persisted candidate, its passing gate report, immutable
executable contract, and exact E/P/V/S masks before emitting the standard
``joint-generation-handoff-v3`` consumed by the Online product.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from phase3_mask_edit.core.mask_io import load_id_mask

from .ledger import analyze_joint_change
from .models import JointCaseContext, JointContractError
from .scene import build_joint_scene_analysis
from .skills.repository import JointSkillRepository

APPROVAL_SCHEMA_VERSION = "joint-candidate-user-approval-v1"
HANDOFF_SCHEMA_VERSION = "joint-generation-handoff-v3"
RESULT_BINDING_SCHEMA_VERSION = "joint-result-binding-v2"

_CONTRACT_MASKS = {
    "contract_T_population": (
        "T_pop_population_target.png",
        "population_target_region_sha256",
    ),
    "contract_E_erasure": ("E_erasure.png", "erasure_region_sha256"),
    "contract_P_placement_centers": (
        "P_placement_centers.png",
        "placement_center_region_sha256",
    ),
    "contract_V_valid_footprints": (
        "V_valid_footprints.png",
        "valid_footprint_region_sha256",
    ),
    "contract_S_support_context": (
        "S_support_context.png",
        "support_context_region_sha256",
    ),
    "contract_M_mechanism_region": (
        "M_mechanism_region.png",
        "mechanism_region_sha256",
    ),
    "contract_C_continuity_region": (
        "C_continuity_region.png",
        "continuity_region_sha256",
    ),
    "contract_A_selected_anchor": (
        "A_selected_anchor.png",
        "continuity_anchor_mask_sha256",
    ),
}


def promote_audited_joint_candidate(
    audit_case_dir: str | Path,
    *,
    candidate_id: str,
    approval: Mapping[str, Any],
    output_dir: str | Path | None = None,
    repository: JointSkillRepository | None = None,
) -> dict[str, str]:
    """Emit a hash-bound handoff without changing the approved mask condition."""

    audit_dir = Path(audit_case_dir).resolve()
    if not audit_dir.is_dir():
        raise JointContractError(f"audited case directory does not exist: {audit_dir}")
    case_payload = _read_object(audit_dir / "case_context.json")
    case = JointCaseContext.from_mapping(case_payload)
    case.validate_local_inputs()
    normalized_approval = _validate_approval(
        approval,
        case_id=case.case_id,
        candidate_id=candidate_id,
    )

    candidate_records = _read_list(audit_dir / "candidates.json")
    matching = [
        item
        for item in candidate_records
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id
    ]
    if len(matching) != 1:
        raise JointContractError(
            f"approved candidate must occur exactly once in candidates.json: {candidate_id}"
        )
    candidate = matching[0]
    gate_reports = _read_list(audit_dir / "joint_gate_reports.json")
    gate_matches = [
        item
        for item in gate_reports
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id
    ]
    if len(gate_matches) != 1 or gate_matches[0].get("passed") is not True:
        raise JointContractError(
            "only one deterministically passing candidate may be approved"
        )
    failed_hard = [
        str(item.get("check_id") or "unknown")
        for item in gate_matches[0].get("checks", [])
        if isinstance(item, dict)
        and item.get("severity") == "hard"
        and item.get("passed") is not True
    ]
    if failed_hard:
        raise JointContractError(
            "approved candidate has failed hard gates: " + ", ".join(failed_hard)
        )

    plan = _read_object(audit_dir / "joint_edit_plan.json")
    mechanism_id = str(candidate.get("mechanism_id") or "")
    if (
        plan.get("case_id") != case.case_id
        or plan.get("selected_mechanism_id") != mechanism_id
    ):
        raise JointContractError("candidate is detached from its audited joint plan")
    trace = candidate.get("tool_trace")
    if not isinstance(trace, dict):
        raise JointContractError("candidate has no persisted tool trace")
    contract_id = str(trace.get("executable_contract_id") or "")
    contract_dir = audit_dir / "executable_contracts" / contract_id
    contract_path = contract_dir / "contract.json"
    contract = _read_object(contract_path)
    _validate_contract_identity(contract, expected_contract_id=contract_id)
    if (
        contract.get("case_id") != case.case_id
        or contract.get("primitive_id") != case.primitive_id
        or contract.get("mechanism_id") != mechanism_id
        or contract.get("tissue_candidate_id") != candidate.get("tissue_candidate_id")
    ):
        raise JointContractError("approved candidate identity differs from its contract")

    repository = repository or JointSkillRepository()
    try:
        mechanism = repository.mechanisms[mechanism_id]
        primitive = repository.primitives[case.primitive_id]
    except KeyError as exc:
        raise JointContractError(
            f"approved candidate references an unavailable current skill: {exc.args[0]}"
        ) from exc
    contract_versions = contract.get("skill_versions") or {}
    if trace.get("skill_version") != contract_versions.get("mechanism"):
        raise JointContractError(
            "candidate tool trace differs from its audited mechanism version"
        )

    source_digest_fields = {
        "source_image_sha256": Path(case.source_image_uri),
        "source_tissue_mask_sha256": Path(case.source_tissue_mask_uri),
        "source_nuclei_mask_sha256": Path(case.source_nuclei_mask_uri),
    }
    for digest_key, path in source_digest_fields.items():
        if _file_sha256(path) != case.provenance.get(digest_key):
            raise JointContractError(
                f"source asset differs from audited provenance: {digest_key}"
            )

    source_tissue = load_id_mask(case.source_tissue_mask_uri)
    source_nuclei = load_id_mask(case.source_nuclei_mask_uri)
    target_tissue_path = _candidate_path(
        audit_dir, candidate, "target_tissue_mask"
    )
    target_nuclei_path = _candidate_path(
        audit_dir, candidate, "target_nuclei_mask"
    )
    joint_change_path = _candidate_path(
        audit_dir, candidate, "joint_change_mask"
    )
    target_tissue = load_id_mask(target_tissue_path)
    target_nuclei = load_id_mask(target_nuclei_path)
    persisted_joint = _load_binary(joint_change_path)

    contract_program = contract.get("cell_program")
    if not isinstance(contract_program, dict):
        raise JointContractError("persisted executable contract has no cell program")
    contract_masks: dict[str, Path] = {}
    for output_name, (source_name, digest_key) in _CONTRACT_MASKS.items():
        path = contract_dir / source_name
        mask = _load_binary(path)
        if _mask_digest(mask) != contract_program.get(digest_key):
            raise JointContractError(
                f"persisted executable contract mask has digest drift: {source_name}"
            )
        contract_masks[output_name] = path
    support = _load_binary(contract_masks["contract_S_support_context"])

    schema = repository.annotation_schema(case.annotation_profile_id)
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
    analysis = analyze_joint_change(
        source_tissue=source_tissue,
        target_tissue=target_tissue,
        source_nuclei=source_nuclei,
        target_nuclei=target_nuclei,
        generation_halo_px=0,
        generation_support_contract=support,
        source_instance_masks=scene.instance_masks,
        source_instance_classes={
            item.instance_id: item.class_id for item in scene.cells.instances
        },
        erased_source_instance_ids=tuple(
            (contract.get("cell_instance_contract") or {}).get(
                "erase_instance_ids", ()
            )
        ),
    )
    if not np.array_equal(analysis.joint_change, persisted_joint):
        raise JointContractError(
            "persisted joint change differs from reconstructed result"
        )
    reconstructed_ledger = _json_value(analysis.ledger.to_metadata())
    if reconstructed_ledger != candidate.get("ledger"):
        raise JointContractError("persisted candidate ledger differs from reconstructed result")
    if analysis.ledger.generation_support_pixels <= analysis.ledger.joint_pixels:
        raise JointContractError(
            "Generate requires contract S to strictly exceed semantic joint change J"
        )
    if not _persisted_id_mask_matches_array_digest(
        target_tissue,
        (contract.get("digests") or {}).get("target_tissue"),
    ):
        raise JointContractError("approved target tissue differs from its contract")
    if _array_digest(analysis.tissue_change) != (
        contract.get("digests") or {}
    ).get("tissue_change"):
        raise JointContractError("approved tissue change differs from its contract")

    root = Path(output_dir).resolve() if output_dir else audit_dir
    handoff_dir = root / "generation_handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    digests: dict[str, str] = {}

    _copy_artifact(target_tissue_path, handoff_dir / "target_tissue_mask.png")
    _copy_artifact(target_nuclei_path, handoff_dir / "target_nuclei_mask.png")
    _save_binary(handoff_dir / "tissue_change.png", analysis.tissue_change)
    _save_binary(handoff_dir / "cell_change.png", analysis.cell_change)
    _copy_artifact(joint_change_path, handoff_dir / "joint_change.png")
    _copy_artifact(
        contract_masks["contract_S_support_context"],
        handoff_dir / "generation_support.png",
    )
    for output_name, source_path in contract_masks.items():
        _copy_artifact(source_path, handoff_dir / f"{output_name}.png")
    _copy_artifact(contract_path, handoff_dir / "executable_contract.json")

    artifact_names = (
        "target_tissue_mask",
        "target_nuclei_mask",
        "tissue_change",
        "cell_change",
        "joint_change",
        "generation_support",
        *_CONTRACT_MASKS,
        "executable_contract",
    )
    for name in artifact_names:
        suffix = ".json" if name == "executable_contract" else ".png"
        path = (handoff_dir / f"{name}{suffix}").resolve()
        paths[name] = str(path)
        digests[name + "_sha256"] = _file_sha256(path)

    ledger = reconstructed_ledger
    manifest = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "case_id": case.case_id,
        "candidate_id": candidate_id,
        "executable_contract_id": contract_id,
        "primitive_id": case.primitive_id,
        "primitive_scope": primitive.scope,
        "mechanism_id": mechanism_id,
        "active_rule_ids": list(contract.get("active_rule_ids") or ()),
        "render_expectations": list(
            mechanism.render.required_for(case.primitive_id)
        ),
        "render_vetoes": list(mechanism.render.vetoes_for(case.primitive_id)),
        "ledger": ledger,
        "generation_context": {
            "policy": "contract_S_strict_superset_of_semantic_J",
            "semantic_joint_pixels": analysis.ledger.joint_pixels,
            "generation_support_pixels": analysis.ledger.generation_support_pixels,
            "adjacent_context_pixels": (
                analysis.ledger.generation_support_pixels
                - analysis.ledger.joint_pixels
            ),
        },
        "execution_contract": {
            "executable_contract": contract,
            "budget_mode": primitive.budget_mode,
            "joint_area_budget": case_payload.get("joint_area_budget"),
            "cell_count_extent_budget": case_payload.get(
                "cell_count_extent_budget"
            ),
            "cell_plan": plan.get("cell_plan"),
            "coupling_plan": plan.get("coupling_plan"),
        },
        "source_assets": {
            "image": case.source_image_uri,
            "tissue": case.source_tissue_mask_uri,
            "nuclei": case.source_nuclei_mask_uri,
        },
        "paths": paths,
        "digests": digests,
        "provenance": {
            **trace,
            "user_approval": normalized_approval,
            "promotion_policy": "persisted_audit_exact_candidate_v1",
            "audited_skill_versions": contract_versions,
            "current_render_skill_versions": {
                "mechanism": mechanism.version,
                "primitive": primitive.version,
            },
            "source_audit_case_dir": str(audit_dir),
            "joint_gate_report_sha256": _file_sha256(
                audit_dir / "joint_gate_reports.json"
            ),
            "joint_edit_plan_sha256": _file_sha256(
                audit_dir / "joint_edit_plan.json"
            ),
        },
    }
    binding = {
        "schema_version": RESULT_BINDING_SCHEMA_VERSION,
        "contract_id": contract_id,
        "candidate_id": candidate_id,
        "target_tissue_sha256": digests["target_tissue_mask_sha256"],
        "target_nuclei_sha256": digests["target_nuclei_mask_sha256"],
        "tissue_change_sha256": digests["tissue_change_sha256"],
        "cell_change_sha256": digests["cell_change_sha256"],
        "joint_change_sha256": digests["joint_change_sha256"],
        "generation_support_sha256": digests["generation_support_sha256"],
        "contract_T_population_sha256": digests[
            "contract_T_population_sha256"
        ],
    }
    binding["binding_id"] = _canonical_digest(binding)
    manifest["result_binding"] = binding
    manifest_path = handoff_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["manifest"] = str(manifest_path.resolve())
    return paths


def _validate_approval(
    approval: Mapping[str, Any], *, case_id: str, candidate_id: str
) -> dict[str, Any]:
    if not isinstance(approval, Mapping):
        raise JointContractError("user approval must be a JSON object")
    normalized = dict(approval)
    required = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "decision": "approved",
        "case_id": case_id,
        "candidate_id": candidate_id,
        "approval_scope": "mask_condition_for_online_generation",
    }
    for key, expected in required.items():
        if normalized.get(key) != expected:
            raise JointContractError(
                f"user approval {key} must be exactly {expected!r}"
            )
    if str(normalized.get("approved_by") or "").strip() != "user":
        raise JointContractError("candidate approval must be an explicit user decision")
    evidence_sha256 = str(normalized.get("evidence_sha256") or "")
    if len(evidence_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in evidence_sha256
    ):
        raise JointContractError("user approval requires a lowercase evidence_sha256")
    return normalized


def _validate_contract_identity(
    contract: Mapping[str, Any], *, expected_contract_id: str
) -> None:
    if contract.get("contract_id") != expected_contract_id:
        raise JointContractError("persisted executable contract ID mismatch")
    payload = dict(contract)
    payload["contract_id"] = ""
    if _canonical_digest(payload) != expected_contract_id:
        raise JointContractError("persisted executable contract metadata has digest drift")


def _candidate_path(
    audit_dir: Path, candidate: Mapping[str, Any], key: str
) -> Path:
    value = candidate.get(key)
    if not isinstance(value, str) or not value:
        raise JointContractError(f"candidate artifact path is missing: {key}")
    path = Path(value).resolve()
    try:
        path.relative_to(audit_dir)
    except ValueError as exc:
        raise JointContractError(
            f"candidate artifact escapes its audited case directory: {key}"
        ) from exc
    if not path.is_file():
        raise JointContractError(f"candidate artifact does not exist: {path}")
    return path


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JointContractError(f"expected a JSON object: {path}")
    return payload


def _read_list(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise JointContractError(f"expected a JSON list: {path}")
    return payload


def _json_value(value: Any) -> Any:
    """Normalize tuples exactly as the persisted JSON audit writer does."""

    return json.loads(json.dumps(value))


def _load_binary(path: Path) -> np.ndarray:
    if not path.is_file():
        raise JointContractError(f"required audited mask does not exist: {path}")
    return np.asarray(Image.open(path).convert("L")) > 0


def _save_binary(path: Path, value: np.ndarray) -> None:
    Image.fromarray(np.asarray(value, dtype=np.uint8) * 255).save(path)


def _copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mask_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=bool))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _persisted_id_mask_matches_array_digest(
    value: np.ndarray, expected: Any
) -> bool:
    """Match the in-memory dtype lost when an audited ID mask was saved as PNG."""

    if not isinstance(expected, str):
        return False
    return any(
        _array_digest(np.asarray(value).astype(dtype, copy=False)) == expected
        for dtype in (np.uint8, np.uint16, np.int16, np.int32, np.int64)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
