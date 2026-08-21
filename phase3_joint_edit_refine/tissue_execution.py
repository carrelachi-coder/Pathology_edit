"""Gate-aware tissue execution for the joint pipeline.

The public result contains only tissue candidates that already hold a complete
tissue-gate certificate and an exact candidate-local nuclei feasibility
certificate.  Rejected exploratory draws remain available only in the audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.gates import GateContext, GateRegistry
from phase3_mask_edit_refine.models import (
    CandidateMask,
    CaseContext,
    EditPlan,
    GateReport,
)
from phase3_mask_edit_refine.scene import SceneAnalysis
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle

from .budget import JointBudgetAllocation
from .executable_contract import (
    ExecutableJointContract,
    ExecutableJointContractCompiler,
)
from .feasibility import (
    CandidateCellFeasibility,
    JointNucleiPreflight,
    assess_candidate_cell_feasibility,
    certify_compiled_cell_program_feasibility,
)
from .instance_authority import build_scene_instance_authority
from .invasive_architecture import generate_joint_tissue_candidates
from .models import JointCaseContext, JointContractError, JointEditPlan
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle
from .tissue_tools import (
    compile_tissue_tool_program,
    validate_tissue_plan_tool_binding,
)

TISSUE_EXECUTION_VERSION = "joint-gate-aware-tissue-executor-v3"


@dataclass(frozen=True)
class TissueExecutionBatch:
    certified_candidates: tuple[CandidateMask, ...]
    all_candidates: tuple[CandidateMask, ...]
    tissue_gate_reports: tuple[GateReport, ...]
    cell_feasibility_reports: tuple[CandidateCellFeasibility, ...]
    executable_contracts: tuple[ExecutableJointContract, ...]
    executable_contract_errors: dict[str, str]

    def to_metadata(self) -> dict:
        return {
            "version": TISSUE_EXECUTION_VERSION,
            "generated_candidate_count": len(self.all_candidates),
            "tissue_gate_pass_count": sum(
                item.passed for item in self.tissue_gate_reports
            ),
            "joint_preflight_pass_count": sum(
                item.passed for item in self.cell_feasibility_reports
            ),
            "certified_candidate_ids": [
                item.candidate_id for item in self.certified_candidates
            ],
            "executable_contract_ids": {
                item.tissue_candidate_id: item.contract_id
                for item in self.executable_contracts
            },
            "executable_contract_errors": dict(
                sorted(self.executable_contract_errors.items())
            ),
            "cell_feasibility_reports": [
                item.to_metadata() for item in self.cell_feasibility_reports
            ],
        }


def execute_gate_aware_tissue_candidates(
    source_tissue: np.ndarray,
    *,
    source_nuclei: np.ndarray,
    case: JointCaseContext,
    schema: MaskProfileSchema,
    tissue_scene: SceneAnalysis,
    joint_scene: JointSceneAnalysis,
    tissue_case: CaseContext,
    tissue_plan: EditPlan,
    joint_plan: JointEditPlan,
    tissue_bundle: ActiveKnowledgeBundle,
    joint_bundle: JointSkillBundle,
    nuclei_preflight: JointNucleiPreflight,
    allocation: JointBudgetAllocation,
    executable_contract_compiler: ExecutableJointContractCompiler,
    joint_required_checker_ids: tuple[str, ...],
    gates: GateRegistry,
    seed: int,
    compiled_replay_parts: tuple[Any, ...] | None = None,
    compiled_replay_audit: dict[str, Any] | None = None,
    precompiled_candidate: CandidateMask | None = None,
) -> TissueExecutionBatch:
    compiled_tissue_tools = compile_tissue_tool_program(
        primitive_id=tissue_plan.primitive_id,
        mechanism_id=joint_bundle.mechanism.mechanism_id,
        mechanism_allowed_families=(
            joint_bundle.mechanism.tissue_program.allowed_tools
        ),
        primitive_allowed_executors=tissue_bundle.edit_contract.allowed_tools,
    )
    validate_tissue_plan_tool_binding(
        tissue_plan, compiled=compiled_tissue_tools
    )
    source_authority = build_scene_instance_authority(
        joint_scene, source_nuclei
    )
    if (
        source_authority["authority_sha256"]
        != nuclei_preflight.source_instance_authority_sha256
        or len(source_authority["instances"])
        != nuclei_preflight.source_instance_authority_count
    ):
        raise JointContractError(
            "nuclei preflight density authority differs from the execution scene"
        )
    baseline_candidates = (
        (precompiled_candidate,)
        if precompiled_candidate is not None
        else generate_joint_tissue_candidates(
            source_tissue,
            schema=schema,
            tissue_scene=tissue_scene,
            joint_scene=joint_scene,
            plan=tissue_plan,
            bundle=tissue_bundle,
            seed=seed,
            candidate_limit=1,
            compiled_replay_parts=compiled_replay_parts,
            compiled_replay_audit=compiled_replay_audit,
        )
    )
    _bind_and_validate_tissue_candidate_traces(
        baseline_candidates, compiled_tissue_tools
    )
    baseline = _certify_tissue_candidate_set(
        baseline_candidates,
        source_tissue=source_tissue,
        source_nuclei=source_nuclei,
        case=case,
        schema=schema,
        tissue_scene=tissue_scene,
        joint_scene=joint_scene,
        tissue_case=tissue_case,
        tissue_plan=tissue_plan,
        joint_plan=joint_plan,
        tissue_bundle=tissue_bundle,
        joint_bundle=joint_bundle,
        nuclei_preflight=nuclei_preflight,
        allocation=allocation,
        executable_contract_compiler=executable_contract_compiler,
        joint_required_checker_ids=joint_required_checker_ids,
        gates=gates,
    )
    if baseline.certified_candidates:
        baseline.certified_candidates[0].tool_trace[
            "candidate_portfolio_policy"
        ] = "compiler_witness_early_accept"
        return baseline
    if precompiled_candidate is not None:
        # A Planner-selected compiler witness is immutable. If its current
        # joint cell/executable-contract checks fail, return that evidence so
        # the workflow can revoke it; never replace it with an unrelated draw.
        return baseline
    all_candidates = generate_joint_tissue_candidates(
        source_tissue,
        schema=schema,
        tissue_scene=tissue_scene,
        joint_scene=joint_scene,
        plan=tissue_plan,
        bundle=tissue_bundle,
        seed=seed,
        compiled_replay_parts=compiled_replay_parts,
        compiled_replay_audit=compiled_replay_audit,
    )
    _bind_and_validate_tissue_candidate_traces(
        all_candidates, compiled_tissue_tools
    )
    return _certify_tissue_candidate_set(
        all_candidates,
        source_tissue=source_tissue,
        source_nuclei=source_nuclei,
        case=case,
        schema=schema,
        tissue_scene=tissue_scene,
        joint_scene=joint_scene,
        tissue_case=tissue_case,
        tissue_plan=tissue_plan,
        joint_plan=joint_plan,
        tissue_bundle=tissue_bundle,
        joint_bundle=joint_bundle,
        nuclei_preflight=nuclei_preflight,
        allocation=allocation,
        executable_contract_compiler=executable_contract_compiler,
        joint_required_checker_ids=joint_required_checker_ids,
        gates=gates,
    )


def _bind_and_validate_tissue_candidate_traces(candidates, compiled) -> None:
    metadata = compiled.to_metadata()
    for candidate in candidates:
        if candidate.tool_name not in compiled.allowed_concrete_executors:
            raise JointContractError(
                "tissue executor emitted a tool outside the compiled mechanism program"
            )
        existing_program = candidate.tool_trace.get(
            "joint_tissue_tool_program"
        )
        if existing_program is not None and existing_program != metadata:
            raise JointContractError(
                "tissue executor trace is detached from the compiled mechanism program"
            )
        existing_executor = candidate.tool_trace.get(
            "concrete_tissue_executor"
        )
        if (
            existing_executor is not None
            and existing_executor != candidate.tool_name
        ):
            raise JointContractError(
                "tissue executor trace names a different concrete executor"
            )
        candidate.tool_trace["joint_tissue_tool_program"] = metadata
        candidate.tool_trace["concrete_tissue_executor"] = candidate.tool_name


def _certify_tissue_candidate_set(
    all_candidates,
    *,
    source_tissue,
    source_nuclei,
    case,
    schema,
    tissue_scene,
    joint_scene,
    tissue_case,
    tissue_plan,
    joint_plan,
    tissue_bundle,
    joint_bundle,
    nuclei_preflight,
    allocation,
    executable_contract_compiler,
    joint_required_checker_ids,
    gates,
) -> TissueExecutionBatch:
    reports = tuple(
        gates.run(
            GateContext(
                case=tissue_case,
                source_mask=source_tissue,
                schema=schema,
                scene=tissue_scene,
                bundle=tissue_bundle,
                plan=tissue_plan,
                candidate=candidate,
            )
        )
        for candidate in all_candidates
    )
    by_id = {item.candidate_id: item for item in reports}
    cell_reports = tuple(
        assess_candidate_cell_feasibility(
            candidate,
            case=case,
            source_tissue=source_tissue,
            scene=joint_scene,
            preflight=nuclei_preflight,
            joint_bundle=joint_bundle,
            joint_plan=joint_plan,
            allocation=allocation,
        )
        for candidate in all_candidates
        if by_id[candidate.candidate_id].passed
    )
    cell_by_id = {item.candidate_id: item for item in cell_reports}
    certified = []
    contracts: list[ExecutableJointContract] = []
    contract_errors: dict[str, str] = {}
    for candidate in all_candidates:
        tissue_report = by_id[candidate.candidate_id]
        cell_report = cell_by_id.get(candidate.candidate_id)
        if not tissue_report.passed or cell_report is None or not cell_report.passed:
            continue
        try:
            contract = executable_contract_compiler.compile(
                case=case,
                source_tissue=source_tissue,
                source_nuclei=source_nuclei,
                schema=schema,
                scene=joint_scene,
                plan=joint_plan,
                bundle=joint_bundle,
                tissue_candidate=candidate,
                tissue_gate_report=tissue_report,
                allocation=allocation,
                required_checker_ids=joint_required_checker_ids,
            )
        except JointContractError as exc:
            contract_errors[candidate.candidate_id] = str(exc)
            continue
        cell_report = certify_compiled_cell_program_feasibility(
            cell_report,
            candidate=candidate,
            contract=contract,
            scene=joint_scene,
            preflight=nuclei_preflight,
        )
        cell_by_id[candidate.candidate_id] = cell_report
        if not cell_report.passed:
            continue
        contract = contract.bind_packing_certificate(
            cell_report.exact_packing_certificate
        )
        candidate.tool_trace["tissue_execution_contract_version"] = (
            TISSUE_EXECUTION_VERSION
        )
        candidate.tool_trace["tissue_gate_certified"] = True
        candidate.tool_trace["tissue_gate_check_ids"] = [
            item.check_id for item in tissue_report.checks if item.severity == "hard"
        ]
        candidate.tool_trace["nuclei_preflight_version"] = nuclei_preflight.version
        candidate.tool_trace["candidate_cell_feasibility"] = cell_report.to_metadata()
        candidate.tool_trace["executable_contract_id"] = contract.contract_id
        candidate.tool_trace["executable_contract_version"] = (
            contract.schema_version
        )
        contracts.append(contract)
        certified.append(candidate)
    certified.sort(
        key=lambda item: (
            -cell_by_id[item.candidate_id].estimated_add_capacity,
            item.candidate_id,
        )
    )
    return TissueExecutionBatch(
        certified_candidates=tuple(certified),
        all_candidates=tuple(all_candidates),
        tissue_gate_reports=reports,
        cell_feasibility_reports=tuple(
            cell_by_id[item.candidate_id]
            for item in cell_reports
        ),
        executable_contracts=tuple(contracts),
        executable_contract_errors=contract_errors,
    )
