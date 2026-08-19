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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from phase3_mask_edit_refine.evidence import load_id_mask
from phase3_mask_edit_refine.gates import GateRegistry
from phase3_mask_edit_refine.skills import (
    SkillRepository as MaskSkillRepository,
)
from phase3_mask_edit_refine.skills import (
    bind_active_bundle_to_case,
    validate_active_bundle_authority,
)

from .auxiliary import materialize_profile_auxiliaries
from .budget import JointFeasibilitySolver
from .candidate_feasibility import CandidateFeasibilityCompiler
from .feasibility import build_joint_nuclei_preflight
from .g2_v2_shadow import _materialize_joint_context
from .models import JointCaseContext, JointContractError
from .nuclei import load_nuclei_mask
from .planner import HeuristicJointPlanner
from .scene import build_joint_scene_analysis
from .semantic_parser import PreboundSemanticParser, bind_semantic_intent
from .skills.execution_aliases import tissue_tool_primitive_id
from .skills.repository import JointSkillRepository
from .workflow import (
    INFILTRATION_BUDGET_PRIMITIVES,
    JointPathologyEditWorkflow,
    _as_tissue_case,
    _derive_infiltration_budget,
    _derive_local_population_budget,
    _tissue_portfolio_authority_binding,
)

EXECUTION_QUALIFICATION_SCHEMA = "g2-v2-read-only-execution-qualification-v1"
SUPPORTED_G2_MANIFEST_SCHEMA = "g2-v2-image-instruction-mechanism-manifest-v2"


def qualify_g2_v2_execution(
    frozen_manifest_path: str | Path,
    *,
    output_dir: str | Path,
    workers: int = 1,
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
    if workers <= 0:
        raise ValueError("execution qualification workers must be positive")
    tasks = [
        (
            row,
            manifest_digest,
            str(root / "source_auxiliary"),
        )
        for row in rows
    ]
    if workers == 1:
        _initialize_worker()
        records = [_qualify_case_worker(task) for task in tasks]
    else:
        # executor.map preserves source order.  Every worker owns immutable
        # skill repositories and a budget solver; cases never share outputs.
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
        ) as executor:
            records = list(executor.map(_qualify_case_worker, tasks, chunksize=1))
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
        "workers": workers,
    }
    summary_path = root / "execution_qualification_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**summary, "summary": str(summary_path)}


_WORKER_MASK_SKILLS: MaskSkillRepository | None = None
_WORKER_JOINT_SKILLS: JointSkillRepository | None = None
_WORKER_BUDGET_SOLVER: JointFeasibilitySolver | None = None
_WORKER_PORTFOLIO_WORKFLOW: JointPathologyEditWorkflow | None = None


def _initialize_worker() -> None:
    global _WORKER_MASK_SKILLS
    global _WORKER_JOINT_SKILLS
    global _WORKER_BUDGET_SOLVER
    global _WORKER_PORTFOLIO_WORKFLOW
    _WORKER_MASK_SKILLS = MaskSkillRepository()
    _WORKER_JOINT_SKILLS = JointSkillRepository()
    _WORKER_BUDGET_SOLVER = JointFeasibilitySolver()
    # Qualification must use the exact production pre-LLM portfolio compiler.
    # The three runtime agents are not consulted by that private compilation
    # stage; deterministic placeholders are therefore sufficient here.
    _WORKER_PORTFOLIO_WORKFLOW = JointPathologyEditWorkflow(
        tissue_planner=None,
        joint_planner=HeuristicJointPlanner(),
        critic=None,
        mask_skills=_WORKER_MASK_SKILLS,
        joint_skills=_WORKER_JOINT_SKILLS,
    )


def _qualify_case_worker(
    task: tuple[dict[str, Any], str, str],
) -> dict[str, Any]:
    row, manifest_digest, auxiliary_root = task
    if (
        _WORKER_MASK_SKILLS is None
        or _WORKER_JOINT_SKILLS is None
        or _WORKER_BUDGET_SOLVER is None
        or _WORKER_PORTFOLIO_WORKFLOW is None
    ):
        raise RuntimeError("execution qualification worker was not initialized")
    return _qualify_case(
        row,
        manifest_digest=manifest_digest,
        auxiliary_root=Path(auxiliary_root),
        mask_skills=_WORKER_MASK_SKILLS,
        joint_skills=_WORKER_JOINT_SKILLS,
        budget_solver=_WORKER_BUDGET_SOLVER,
        portfolio_workflow=_WORKER_PORTFOLIO_WORKFLOW,
    )


def _qualify_case(
    row: dict[str, Any],
    *,
    manifest_digest: str,
    auxiliary_root: Path,
    mask_skills: MaskSkillRepository,
    joint_skills: JointSkillRepository,
    budget_solver: JointFeasibilitySolver,
    portfolio_workflow: JointPathologyEditWorkflow,
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
    selection_reason = joint_skills.execution_selection_reason(
        primitive_id=str(row.get("primitive_id") or ""),
        mechanism_id=str(row.get("mechanism_id") or ""),
    )
    if selection_reason is not None:
        return {
            **base,
            "status": "execution_requalification_required",
            "failure_reasons": [
                "execution_scope_rejected: " + selection_reason
            ],
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
        frozen_selection = raw["prebound_semantic_intent"].get(
            "selected_primitive_id"
        )
        semantic_binding = (
            str(frozen_selection)
            if frozen_selection is not None
            else semantic.primitive_id
        )
        if semantic_binding != row["primitive_id"]:
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
        case = _with_scene_calibrated_cell_budget(
            case=case,
            scene=scene,
            joint_skills=joint_skills,
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
                portfolio_workflow=portfolio_workflow,
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
    except Exception as exc:  # noqa: BLE001 - preserve one fail-closed record per pair
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
    tool_primitive_id = tissue_tool_primitive_id(case.primitive_id)
    tissue_bundle = mask_skills.compose(
        pathology_domain_id=case.pathology_domain_id,
        annotation_profile_id=case.annotation_profile_id,
        primitive_id=tool_primitive_id,
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
    witness = None
    if not failures:
        tissue_case = _as_tissue_case(
            case,
            allocation=allocation,
            shape=source_tissue.shape,
        )
        try:
            witness = CandidateFeasibilityCompiler(
                gates=GateRegistry()
            ).compile_tissue_witness(
                tissue_case=tissue_case,
                source_tissue=source_tissue,
                schema=schema,
                scene=scene,
                tissue_bundle=tissue_bundle,
                joint_bundle=bundle,
                nuclei_preflight=preflight,
                authority_binding=_tissue_portfolio_authority_binding(
                    case=case,
                    tissue_case=tissue_case,
                    source_tissue=source_tissue,
                    bundle=bundle,
                    tissue_bundle=tissue_bundle,
                    allocation=allocation,
                    nuclei_preflight=preflight,
                ),
            ).to_metadata()
        except Exception as exc:  # noqa: BLE001 - qualification is fail closed
            failures.append("whole_mask_topology_witness_not_executable")
            witness = {
                "schema_version": "joint-candidate-feasibility-compiler-v2",
                "errors": [f"{type(exc).__name__}: {exc}"],
                "persisted_target_mask": False,
            }
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
            "whole_mask_topology_witness": witness,
            "tissue_executor_binding": {
                "joint_primitive_id": case.primitive_id,
                "tool_primitive_id": tool_primitive_id,
            },
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
    portfolio_workflow: JointPathologyEditWorkflow,
) -> tuple[dict[str, Any], list[str]]:
    budget = case.cell_count_extent_budget
    if budget is None:
        raise JointContractError(
            "source scene did not compile the required cell count/extent budget"
        )
    budget_metadata = dict(
        case.semantic_intent.get("derived_budget_policies", {}).get(
            case.primitive_id, {}
        )
    )
    portfolio = portfolio_workflow._compile_cell_only_candidate_portfolio(
        case=case,
        source_tissue=source_tissue,
        source_nuclei=source_nuclei,
        schema=schema,
        scene=scene,
        bundle=bundle,
    )
    survivor_preflights = [
        item.preflight.to_metadata() for item in portfolio.choices
    ]
    return (
        {
            "cell_budget": budget.__dict__,
            "cell_budget_derivation": budget_metadata,
            "production_pre_llm_portfolio": portfolio.certificates.to_metadata(),
            "production_pre_llm_survivor_count": len(portfolio.choices),
            "production_pre_llm_veto_count": len(
                portfolio.certificates.vetoed
            ),
            "survivor_exact_capacity_reports": survivor_preflights,
        },
        [],
    )


def _certified_center_span(center_points: np.ndarray) -> float:
    """Lower-bound an executable span using two actual legal centers."""

    points = np.asarray(center_points, dtype=float)
    if len(points) < 2:
        return 0.0
    seed = points[0]
    first = points[int(np.argmax(np.sum((points - seed) ** 2, axis=1)))]
    distances = np.sum((points - first) ** 2, axis=1)
    return float(np.sqrt(np.max(distances)))


def _with_scene_calibrated_cell_budget(
    *,
    case: JointCaseContext,
    scene,
    joint_skills: JointSkillRepository,
) -> JointCaseContext:
    """Compile a cell-only budget before skill composition checks it."""

    primitive = joint_skills.primitives.get(case.primitive_id)
    if primitive is None or primitive.scope != "cell_only":
        return case
    if case.cell_count_extent_budget is not None:
        return case
    if case.primitive_id in INFILTRATION_BUDGET_PRIMITIVES:
        budget, budget_metadata = _derive_infiltration_budget(
            scene,
            minimum_effect_delta_count=(
                primitive.minimum_effect_delta_count_for(
                    case.pathology_domain_id
                )
            ),
            minimum_effect_span_cell_diameters=(
                primitive.minimum_effect_span_cell_diameters
            ),
            minimum_effect_foci=primitive.minimum_effect_foci,
        )
    else:
        budget, budget_metadata = _derive_local_population_budget(
            scene,
            primitive_id=case.primitive_id,
            semantic_intent=case.semantic_intent,
            host_tissue_labels=primitive.host_tissue_labels,
            minimum_effect_delta_count=(
                primitive.minimum_effect_delta_count_for(
                    case.pathology_domain_id
                )
            ),
            minimum_effect_span_cell_diameters=(
                primitive.minimum_effect_span_cell_diameters
            ),
            minimum_effect_foci=primitive.minimum_effect_foci,
        )
    semantic = dict(case.semantic_intent)
    derived = dict(semantic.get("derived_budget_policies", {}))
    derived[case.primitive_id] = budget_metadata
    semantic["derived_budget_policies"] = derived
    provenance = dict(case.provenance)
    selected_zone = budget_metadata.get("selected_population_zone_id")
    if selected_zone:
        provenance["joint_population_zone_id"] = selected_zone
    return replace(
        case,
        cell_count_extent_budget=budget,
        semantic_intent=semantic,
        provenance=provenance,
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
