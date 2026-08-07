"""Frozen-generator handoff for one approved joint candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .executable_contract import ExecutableJointContract
from .models import JointCandidate, JointCaseContext, JointEditPlan
from .nuclei import to_raw_nuclei_mask
from .skills.repository import JointSkillBundle


def write_generation_handoff(
    output_dir: str | Path,
    *,
    case: JointCaseContext,
    plan: JointEditPlan,
    bundle: JointSkillBundle,
    candidate: JointCandidate,
    executable_contract: ExecutableJointContract,
) -> dict[str, str]:
    executable_contract.validate_handoff_binding(candidate)
    root = Path(output_dir) / "generation_handoff"
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "target_tissue_mask": candidate.target_tissue_mask,
        "target_nuclei_mask": to_raw_nuclei_mask(candidate.target_nuclei_mask),
        "tissue_change": candidate.tissue_change.astype(np.uint8) * 255,
        "cell_change": candidate.cell_change.astype(np.uint8) * 255,
        "joint_change": candidate.joint_change.astype(np.uint8) * 255,
        "generation_support": candidate.generation_support.astype(np.uint8) * 255,
        "contract_E_erasure": (
            executable_contract.cell_program.erasure_region.astype(np.uint8) * 255
        ),
        "contract_P_placement_centers": (
            executable_contract.cell_program.placement_center_region.astype(np.uint8)
            * 255
        ),
        "contract_V_valid_footprints": (
            executable_contract.cell_program.valid_footprint_region.astype(np.uint8)
            * 255
        ),
        "contract_S_support_context": (
            executable_contract.cell_program.support_context_region.astype(np.uint8)
            * 255
        ),
        "contract_M_mechanism_region": (
            executable_contract.cell_program.mechanism_region.astype(np.uint8) * 255
        ),
        "contract_C_continuity_region": (
            executable_contract.cell_program.continuity_region.astype(np.uint8)
            * 255
        ),
        "contract_A_selected_anchor": (
            executable_contract.cell_program.continuity_anchor_mask.astype(np.uint8)
            * 255
        ),
    }
    paths = {}
    digests = {}
    for name, array in arrays.items():
        path = root / f"{name}.png"
        _save(path, array)
        paths[name] = str(path)
        digests[name + "_sha256"] = _sha256(path)
    manifest = {
        "schema_version": "joint-generation-handoff-v2",
        "case_id": case.case_id,
        "candidate_id": candidate.candidate_id,
        "executable_contract_id": executable_contract.contract_id,
        "primitive_id": case.primitive_id,
        "primitive_scope": bundle.primitive.scope,
        "mechanism_id": plan.selected_mechanism_id,
        "active_rule_ids": list(bundle.active_rule_ids),
        "render_expectations": list(
            bundle.mechanism.render.required_for(case.primitive_id)
        ),
        "render_vetoes": list(
            bundle.mechanism.render.vetoes_for(case.primitive_id)
        ),
        "ledger": candidate.ledger.to_metadata(),
        "execution_contract": {
            "executable_contract": executable_contract.to_metadata(),
            "budget_mode": bundle.primitive.budget_mode,
            "joint_area_budget": (
                case.joint_area_budget.__dict__
                if case.joint_area_budget is not None
                else None
            ),
            "cell_count_extent_budget": (
                case.cell_count_extent_budget.__dict__
                if case.cell_count_extent_budget is not None
                else None
            ),
            "cell_plan": plan.cell_plan.__dict__,
            "coupling_plan": plan.coupling_plan.__dict__,
        },
        "source_assets": {
            "image": case.source_image_uri,
            "tissue": case.source_tissue_mask_uri,
            "nuclei": case.source_nuclei_mask_uri,
        },
        "paths": paths,
        "digests": digests,
        "provenance": candidate.tool_trace,
    }
    contract_path = root / "executable_contract.json"
    contract_path.write_text(
        json.dumps(
            executable_contract.to_metadata(), indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
    paths["executable_contract"] = str(contract_path)
    digests["executable_contract_sha256"] = _sha256(contract_path)
    manifest["result_binding"] = {
        "schema_version": "joint-result-binding-v1",
        "contract_id": executable_contract.contract_id,
        "candidate_id": candidate.candidate_id,
        "target_tissue_sha256": digests["target_tissue_mask_sha256"],
        "target_nuclei_sha256": digests["target_nuclei_mask_sha256"],
        "tissue_change_sha256": digests["tissue_change_sha256"],
        "cell_change_sha256": digests["cell_change_sha256"],
        "joint_change_sha256": digests["joint_change_sha256"],
        "generation_support_sha256": digests["generation_support_sha256"],
    }
    manifest["result_binding"]["binding_id"] = _canonical_digest(
        manifest["result_binding"]
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["manifest"] = str(manifest_path)
    return paths


def _save(path: Path, array: np.ndarray) -> None:
    values = np.asarray(array)
    maximum = int(values.max(initial=0))
    Image.fromarray(values.astype(np.uint16 if maximum > 255 else np.uint8)).save(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
