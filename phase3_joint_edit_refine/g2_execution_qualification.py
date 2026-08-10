"""Read-only executable qualification for H&E-reviewed G2-v2 pairs.

No target tissue or nucleus mask is generated.  The qualifier constructs the
source scene, materializes only source-derived auxiliary maps, composes the
reviewed mechanism, and runs the same topology/nucleus capacity compilers that
the joint executor will later consume.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from phase3_mask_edit_refine.evidence import load_id_mask
from phase3_mask_edit_refine.gates import GateRegistry
from phase3_mask_edit_refine.models import CandidateMask
from phase3_mask_edit_refine.skills import SkillRepository as MaskSkillRepository

from .auxiliary import materialize_profile_auxiliaries
from .budget import JointFeasibilitySolver
from .cell_layouts import build_reference_shape_library
from .cell_programs import CellToolProgramCompiler
from .feasibility import build_joint_nuclei_preflight
from .g2_v2_shadow import _materialize_joint_context
from .models import JointCaseContext, JointContractError
from .nuclei import load_nuclei_mask
from .packing import certify_complete_footprint_packing
from .planner import HeuristicJointPlanner
from .scene import build_joint_scene_analysis
from .semantic_parser import PreboundSemanticParser, bind_semantic_intent
from .skills.repository import JointSkillRepository
from .workflow import _derive_infiltration_budget, _derive_local_population_budget

EXECUTION_QUALIFICATION_SCHEMA = "g2-v2-read-only-execution-qualification-v1"
SUPPORTED_G2_MANIFEST_SCHEMA = "g2-v2-image-instruction-mechanism-manifest-v2"


def qualify_g2_v2_execution(
    frozen_manifest_path: str | Path,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    source = Path(frozen_manifest_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SUPPORTED_G2_MANIFEST_SCHEMA:
        raise ValueError("unsupported G2-v2 manifest schema")
    rows = payload.get("cases")
    if not isinstance(rows, list) or len(rows) != int(
        payload.get("case_count", -1)
    ):
        raise ValueError("G2-v2 manifest case count is inconsistent")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_digest = _sha256(source)
    mask_skills = MaskSkillRepository()
    joint_skills = JointSkillRepository()
    budget_solver = JointFeasibilitySolver()
    records = []
    for row in rows:
        records.append(
            _qualify_case(
                row,
                manifest_digest=manifest_digest,
                auxiliary_root=root / "source_auxiliary",
                mask_skills=mask_skills,
                joint_skills=joint_skills,
                budget_solver=budget_solver,
            )
        )
    ledger = root / "execution_qualification.jsonl"
    ledger.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": EXECUTION_QUALIFICATION_SCHEMA,
        "source_manifest": str(source),
        "source_manifest_sha256": manifest_digest,
        "case_count": len(records),
        "status_counts": dict(
            sorted(Counter(item["status"] for item in records).items())
        ),
        "failure_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for item in records
                    if item["status"] == "execution_requalification_required"
                    for reason in item["failure_reasons"]
                ).items()
            )
        ),
        "ledger": str(ledger),
        "ledger_sha256": _sha256(ledger),
        "target_mask_created": False,
        "source_asset_mutated": False,
        "llm_api_used": False,
    }
    summary_path = root / "execution_qualification_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**summary, "summary": str(summary_path)}


def _qualify_case(
    row: dict[str, Any],
    *,
    manifest_digest: str,
    auxiliary_root: Path,
    mask_skills: MaskSkillRepository,
    joint_skills: JointSkillRepository,
    budget_solver: JointFeasibilitySolver,
) -> dict[str, Any]:
    case_id = str(row["case_id"])
    base = {
        "schema_version": EXECUTION_QUALIFICATION_SCHEMA,
        "case_id": case_id,
        "source_index": int(row["source_index"]),
        "organ": row["organ"],
        "primitive_id": row.get("primitive_id"),
        "mechanism_id": row.get("mechanism_id"),
        "source_manifest_sha256": manifest_digest,
        "target_mask_created": False,
        "source_asset_mutated": False,
        "llm_api_used": False,
    }
    if not row.get("execution_allowed"):
        return {
            **base,
            "status": "upstream_abstain",
            "failure_reasons": [str(row.get("decision_reason_code"))],
            "metrics": {},
        }
    try:
        raw = _materialize_joint_context(
            row, manifest_sha256=manifest_digest
        )
        case, semantic = bind_semantic_intent(
            raw,
            PreboundSemanticParser(raw["prebound_semantic_intent"]),
        )
        if semantic.primitive_id != row["primitive_id"]:
            raise JointContractError(
                "Codex semantic primitive differs from reviewed G2 primitive"
            )
        case.validate_local_inputs()
        source_tissue = load_id_mask(case.source_tissue_mask_uri)
        source_nuclei = load_nuclei_mask(case.source_nuclei_mask_uri)
        case, produced = materialize_profile_auxiliaries(
            case,
            source_tissue=source_tissue,
            output_dir=auxiliary_root / case_id,
        )
        schema = mask_skills.annotation_schema(case.annotation_profile_id)
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
        bundle = joint_skills.compose(
            case=case,
            mechanism_id=str(row["mechanism_id"]),
            available_checker_ids=_all_joint_checker_ids(),
            production=False,
        )
        if bundle.primitive.scope == "tissue_and_cell":
            metrics, failures = _qualify_tissue_case(
                case=case,
                source_tissue=source_tissue,
                schema=schema,
                scene=scene,
                bundle=bundle,
                mask_skills=mask_skills,
                budget_solver=budget_solver,
            )
        else:
            metrics, failures = _qualify_cell_only_case(
                case=case,
                source_tissue=source_tissue,
                source_nuclei=source_nuclei,
                schema=schema,
                scene=scene,
                bundle=bundle,
            )
        metrics["source_auxiliary_producer_ids"] = [
            str(item.provenance.get("producer_id")) for item in produced
        ]
        return {
            **base,
            "status": (
                "executable_preflight_passed"
                if not failures
                else "execution_requalification_required"
            ),
            "failure_reasons": sorted(set(failures)),
            "metrics": metrics,
        }
    except Exception as exc:  # fail closed, preserving one record per pair
        return {
            **base,
            "status": "execution_requalification_required",
            "failure_reasons": [
                f"{type(exc).__name__}: {exc}"
            ],
            "metrics": {},
        }


def _qualify_tissue_case(
    *,
    case,
    source_tissue,
    schema,
    scene,
    bundle,
    mask_skills,
    budget_solver,
) -> tuple[dict[str, Any], list[str]]:
    allocation = budget_solver.allocate(
        shape=source_tissue.shape,
        budget=case.joint_area_budget,
        bundle=bundle,
    )
    tissue_bundle = mask_skills.compose(
        pathology_domain_id=case.pathology_domain_id,
        annotation_profile_id=case.annotation_profile_id,
        primitive_id=case.primitive_id,
        production=False,
        available_checker_ids=set(GateRegistry().available_checker_ids),
    )
    preflight = build_joint_nuclei_preflight(
        case=case,
        source_tissue=source_tissue,
        schema=schema,
        scene=scene,
        tissue_bundle=tissue_bundle,
        joint_bundle=bundle,
        allocation=allocation,
    )
    failures = []
    if preflight.required_auxiliary_missing:
        failures.append("required_auxiliary_missing")
    if preflight.required_provenance_missing:
        failures.append("required_profile_provenance_missing")
    if not preflight.feasible_interface_ids:
        failures.append("no_nuclei_safe_executable_interface")
    if not preflight.meaningful_tissue_capacity_passed:
        failures.append("meaningful_tissue_area_not_executable")
    return (
        {
            "budget_allocation": allocation.to_metadata(),
            "meaningful_tissue_floor_pixels": (
                preflight.meaningful_tissue_floor_pixels
            ),
            "aggregate_executable_tissue_capacity_pixels": (
                preflight.aggregate_feasible_tissue_capacity_pixels
            ),
            "feasible_interface_ids": list(preflight.feasible_interface_ids),
            "interface_capacity": [
                item.to_metadata() for item in preflight.interfaces
            ],
        },
        failures,
    )


def _qualify_cell_only_case(
    *,
    case: JointCaseContext,
    source_tissue,
    source_nuclei,
    schema,
    scene,
    bundle,
) -> tuple[dict[str, Any], list[str]]:
    if case.primitive_id == "neoplastic-cell-infiltration-increase-v1":
        budget, budget_metadata = _derive_infiltration_budget(scene)
    else:
        budget, budget_metadata = _derive_local_population_budget(
            scene,
            primitive_id=case.primitive_id,
            semantic_intent=case.semantic_intent,
            host_tissue_labels=bundle.primitive.host_tissue_labels,
        )
    provenance = dict(case.provenance)
    selected_zone = budget_metadata.get("selected_population_zone_id")
    if selected_zone:
        provenance["joint_population_zone_id"] = selected_zone
    case = replace(
        case,
        cell_count_extent_budget=budget,
        provenance=provenance,
    )
    plan, _usage = HeuristicJointPlanner().create_plan(
        case=case,
        scene=scene,
        bundle=bundle,
        tissue_plan=None,
        image_paths=(),
    )
    preserved = CandidateMask(
        candidate_id="read-only-preserved-tissue",
        interface_id=(
            plan.cell_plan.interface_ids[0]
            if plan.cell_plan.interface_ids
            else plan.cell_plan.core_zone
        ),
        tool_name="read_only_preserve_tissue",
        target_mask=np.asarray(source_tissue),
        change_region=np.zeros_like(source_tissue, dtype=bool),
        tool_trace={"read_only": True},
    )
    program = CellToolProgramCompiler().compile(
        case=case,
        schema=schema,
        scene=scene,
        plan=plan,
        bundle=bundle,
        tissue_candidate=preserved,
    )
    failures: list[str] = []
    packing_metadata: dict[str, Any] | None = None
    if "add" in bundle.mechanism.cell_program.actions:
        metadata = {item.instance_id: item for item in scene.cells.instances}
        references_by_class = {}
        component_id = (
            plan.cell_plan.core_zone.removeprefix("pop:component:")
            if plan.cell_plan.core_zone.startswith("pop:component:")
            else None
        )
        for class_id in plan.cell_plan.allowed_cell_classes:
            references, _rejected = build_reference_shape_library(
                scene, class_id=class_id
            )
            if component_id is not None:
                references = tuple(
                    item
                    for item in references
                    if metadata[item.instance_id].tissue_component_id
                    == component_id
                )
            if references:
                references_by_class[class_id] = references
        requested = int(program.target_delta_count or 0)
        packing = certify_complete_footprint_packing(
            source_nuclei=source_nuclei,
            erased_footprint=program.erasure_region,
            center_region=program.placement_center_region,
            valid_footprint_region=program.valid_footprint_region,
            references_by_class=references_by_class,
            requested_count=requested,
            allow_finite_count_fallback=False,
        )
        packing_metadata = packing.to_metadata()
        if not packing.passed:
            failures.extend(packing.failure_reasons)
        if packing.placed_count < budget.min_delta_count:
            failures.append("meaningful_cell_count_not_executable")
    elif int(program.target_delta_count or 0) < budget.min_delta_count:
        failures.append("meaningful_cell_count_not_executable")
    return (
        {
            "cell_budget": budget.__dict__,
            "cell_budget_derivation": budget_metadata,
            "cell_program": program.to_metadata(),
            "exact_packing_certificate": packing_metadata,
        },
        failures,
    )


def _all_joint_checker_ids() -> set[str]:
    from .gates import JointGateRegistry

    return set(JointGateRegistry().available_checker_ids)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
