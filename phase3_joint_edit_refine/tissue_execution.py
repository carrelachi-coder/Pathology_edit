"""Gate-aware tissue execution for the joint pipeline.

The public result contains only tissue candidates that already hold a complete
tissue-gate certificate and an exact candidate-local nuclei feasibility
certificate.  Rejected exploratory draws remain available only in the audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.candidates import generate_candidates
from phase3_mask_edit_refine.gates import GateContext, GateRegistry
from phase3_mask_edit_refine.models import (
    CandidateMask,
    CaseContext,
    EditPlan,
    GateReport,
)
from phase3_mask_edit_refine.scene import SceneAnalysis
from phase3_mask_edit_refine.skills import ActiveKnowledgeBundle

from .feasibility import (
    CandidateCellFeasibility,
    JointNucleiPreflight,
    assess_candidate_cell_feasibility,
)
from .scene import JointSceneAnalysis
from .skills.repository import JointSkillBundle


TISSUE_EXECUTION_VERSION = "joint-gate-aware-tissue-executor-v1"


@dataclass(frozen=True)
class TissueExecutionBatch:
    certified_candidates: tuple[CandidateMask, ...]
    all_candidates: tuple[CandidateMask, ...]
    tissue_gate_reports: tuple[GateReport, ...]
    cell_feasibility_reports: tuple[CandidateCellFeasibility, ...]

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
            "cell_feasibility_reports": [
                item.to_metadata() for item in self.cell_feasibility_reports
            ],
        }


def execute_gate_aware_tissue_candidates(
    source_tissue: np.ndarray,
    *,
    schema: MaskProfileSchema,
    tissue_scene: SceneAnalysis,
    joint_scene: JointSceneAnalysis,
    tissue_case: CaseContext,
    tissue_plan: EditPlan,
    tissue_bundle: ActiveKnowledgeBundle,
    joint_bundle: JointSkillBundle,
    nuclei_preflight: JointNucleiPreflight,
    gates: GateRegistry,
    seed: int,
) -> TissueExecutionBatch:
    all_candidates = generate_candidates(
        source_tissue,
        schema=schema,
        scene=tissue_scene,
        plan=tissue_plan,
        bundle=tissue_bundle,
        seed=seed,
    )
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
            source_tissue=source_tissue,
            scene=joint_scene,
            preflight=nuclei_preflight,
            joint_bundle=joint_bundle,
        )
        for candidate in all_candidates
        if by_id[candidate.candidate_id].passed
    )
    cell_by_id = {item.candidate_id: item for item in cell_reports}
    certified = []
    for candidate in all_candidates:
        tissue_report = by_id[candidate.candidate_id]
        cell_report = cell_by_id.get(candidate.candidate_id)
        if not tissue_report.passed or cell_report is None or not cell_report.passed:
            continue
        candidate.tool_trace["tissue_execution_contract_version"] = (
            TISSUE_EXECUTION_VERSION
        )
        candidate.tool_trace["tissue_gate_certified"] = True
        candidate.tool_trace["tissue_gate_check_ids"] = [
            item.check_id for item in tissue_report.checks if item.severity == "hard"
        ]
        candidate.tool_trace["nuclei_preflight_version"] = nuclei_preflight.version
        candidate.tool_trace["candidate_cell_feasibility"] = cell_report.to_metadata()
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
        cell_feasibility_reports=cell_reports,
    )
