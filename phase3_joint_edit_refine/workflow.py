"""Independent joint Architecture-B orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from phase3_mask_edit_refine.agents import Planner, validate_edit_plan
from phase3_mask_edit_refine.evidence import load_id_mask, sha256_file
from phase3_mask_edit_refine.execution import compile_edit_plan
from phase3_mask_edit_refine.gates import GateRegistry
from phase3_mask_edit_refine.models import (
    AreaBudget,
    CandidateMask,
    CaseContext,
    GateReport,
    RefineContractError,
)
from phase3_mask_edit_refine.skills import SkillRepository as MaskSkillRepository
from phase3_mask_edit_refine.visualization import save_planner_panels

from .audit import JointAuditWriter
from .auxiliary import materialize_profile_auxiliaries
from .budget import JointFeasibilitySolver
from .cell_layouts import (
    SpatialRanker,
    build_reference_shape_library,
    generate_cell_layouts,
)
from .cell_programs import CellToolProgramCompiler
from .critic import JointCritic
from .executable_contract import (
    ExecutableJointContract,
    ExecutableJointContractCompiler,
)
from .feasibility import (
    augment_tissue_scene_with_nuclei_preflight,
    bind_joint_plan_to_nuclei_preflight,
    build_joint_nuclei_preflight,
)
from .gates import JointGateContext, JointGateRegistry
from .handoff import write_generation_handoff
from .instance_authority import authority_trace, build_scene_instance_authority
from .ledger import build_joint_candidate
from .mature_probnet_adapter import MatureProbNetCellExecutor
from .models import (
    CellCountExtentBudget,
    JointCaseContext,
    JointCondition,
    JointContractError,
    JointWorkflowResult,
)
from .nuclei import load_nuclei_mask
from .packing import certify_complete_footprint_packing
from .planner import JointInterpretationOption, JointPlanner
from .scene import build_joint_scene_analysis
from .skills.repository import JointSkillBundle, JointSkillRepository
from .tissue_execution import execute_gate_aware_tissue_candidates


@dataclass(frozen=True)
class JointWorkflowConfig:
    production: bool = False
    maximum_tissue_candidates: int = 4
    cell_layouts_per_tissue: int = 3
    critic_confidence_threshold: float = 0.70
    require_mature_probnet_in_production: bool = True
    # One initial solve plus two feedback-directed alternatives.  Each pass
    # executes a fixed 12-candidate contract; a case that still lacks a safe
    # tissue--cell pair abstains instead of creating an unbounded search tail.
    maximum_tissue_planning_attempts: int = 3
    # Fully executed complete nuclei reveal the exact J spill only after the
    # tissue loop. Reserve separate bounded attempts so earlier interface/tool
    # retries cannot consume the only opportunity to correct that observation.
    maximum_joint_area_feedback_attempts: int = 2


@dataclass(frozen=True)
class _PreparedInterpretation:
    option: JointInterpretationOption
    case: JointCaseContext
    bundle: JointSkillBundle
    allocation: Any | None
    tissue_bundle: Any | None
    nuclei_preflight: Any | None


class JointPathologyEditWorkflow:
    """Select or abstain on one atomic ``(tissue, nuclei)`` condition."""

    def __init__(
        self,
        *,
        tissue_planner: Planner,
        joint_planner: JointPlanner,
        critic: JointCritic,
        mask_skills: MaskSkillRepository | None = None,
        joint_skills: JointSkillRepository | None = None,
        tissue_gates: GateRegistry | None = None,
        joint_gates: JointGateRegistry | None = None,
        ranker: SpatialRanker | None = None,
        cell_executor: MatureProbNetCellExecutor | None = None,
        config: JointWorkflowConfig | None = None,
    ) -> None:
        self.tissue_planner = tissue_planner
        self.joint_planner = joint_planner
        self.critic = critic
        self.mask_skills = mask_skills or MaskSkillRepository()
        self.joint_skills = joint_skills or JointSkillRepository()
        self.tissue_gates = tissue_gates or GateRegistry()
        self.joint_gates = joint_gates or JointGateRegistry()
        self.joint_skills.validate_checker_registry(
            set(self.joint_gates.available_checker_ids)
        )
        self.ranker = ranker
        self.cell_executor = cell_executor
        self.config = config or JointWorkflowConfig()
        self.budget_solver = JointFeasibilitySolver()
        self.executable_contract_compiler = ExecutableJointContractCompiler(
            CellToolProgramCompiler()
        )

    def run(
        self, case: JointCaseContext, *, output_root: str | Path
    ) -> JointWorkflowResult:
        audit = JointAuditWriter(output_root, case_id=case.case_id)
        plan = None
        source_tissue = None
        source_nuclei = None
        scene = None
        nuclei_preflight = None
        critic_result = None
        joint_reports = ()
        reasons: list[str] = []
        usage: dict = {
            "mechanism_selection": {},
            "tissue_planner": {},
            "joint_planner": {},
            "critic": {},
        }
        candidates = ()
        try:
            if self.config.production and not case.semantic_intent:
                raise JointContractError(
                    "production joint execution requires a bound Semantic Parser intent"
                )
            case.validate_local_inputs()
            _validate_digests(case)
            source_tissue = load_id_mask(case.source_tissue_mask_uri)
            source_nuclei = load_nuclei_mask(case.source_nuclei_mask_uri)
            _validate_dimensions(
                case.source_image_uri, source_tissue.shape, source_nuclei.shape
            )
            audit.write_json("input_case_context.json", case.to_metadata())
            case, produced_auxiliaries = materialize_profile_auxiliaries(
                case,
                source_tissue=source_tissue,
                output_dir=audit.case_dir / "auxiliary_structures",
            )
            if produced_auxiliaries:
                audit.write_json(
                    "auxiliary_producer_report.json",
                    [item.to_metadata() for item in produced_auxiliaries],
                )
            case.validate_local_inputs()
            _validate_digests(case)
            schema = self.mask_skills.annotation_schema(case.annotation_profile_id)
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
            audit.write_json("case_context.json", case.to_metadata())
            audit.write_json("joint_scene_graph.json", scene.to_metadata())
            audit.write_json(
                "source_instance_authority.json",
                build_scene_instance_authority(scene, source_nuclei),
            )
            tissue_panels = save_planner_panels(
                image_path=case.source_image_uri,
                mask=source_tissue,
                scene=scene.tissue,
                output_dir=audit.case_dir / "planner_panels",
            )
            joint_overlay = audit.write_source_joint_overlay(
                source_image_path=case.source_image_uri,
                source_tissue=source_tissue,
                source_nuclei=source_nuclei,
            )
            planner_images = (*tissue_panels, joint_overlay)
            prepared, interpretation_rejections = self._prepare_interpretations(
                case=case,
                source_tissue=source_tissue,
                schema=schema,
                scene=scene,
            )
            resolution_base = {
                "instruction": case.instruction,
                "semantic_request": case.semantic_intent,
                "candidate_interpretations": [
                    item.option.to_metadata() for item in prepared.values()
                ],
                "rejected_interpretations": interpretation_rejections,
            }
            if not prepared:
                audit.write_json(
                    "semantic_resolution.json",
                    {
                        **resolution_base,
                        "status": "no_executable_interpretation",
                        "selected_option_id": None,
                        "selection": None,
                    },
                )
                raise JointContractError(
                    "no natural-language interpretation survives skill and "
                    "deterministic feasibility checks: "
                    + "; ".join(
                        f"{key}={value}"
                        for key, value in sorted(
                            interpretation_rejections.items()
                        )
                    )
                )
            try:
                primitive_id, mechanism_id, selection_usage = (
                    self.joint_planner.select_interpretation(
                        case=case,
                        scene=scene,
                        options=tuple(
                            item.option for item in prepared.values()
                        ),
                        image_paths=planner_images,
                    )
                )
            except JointContractError as exc:
                audit.write_json(
                    "semantic_resolution.json",
                    {
                        **resolution_base,
                        "status": "visual_resolution_abstained",
                        "selected_option_id": None,
                        "selection": None,
                        "abstain_reason": str(exc),
                    },
                )
                raise
            option_id = f"{primitive_id}::{mechanism_id}"
            selected = prepared.get(option_id)
            if selected is None:
                raise JointContractError(
                    "joint Planner returned an interpretation that was not prepared"
                )
            selection_record = dict(selection_usage.get("selection") or {})
            if selected.case.semantic_intent:
                semantic_intent = dict(selected.case.semantic_intent)
                semantic_intent.update(
                    {
                        "selected_primitive_id": primitive_id,
                        "selected_mechanism_id": mechanism_id,
                        "selection_explanation": selection_record.get(
                            "interpretation_explanation"
                        ),
                        "selection_semantic_fit": selected.option.semantic_fit,
                    }
                )
                case = replace(
                    selected.case, semantic_intent=semantic_intent
                )
            else:
                # Backward-compatible offline fixtures are allowed to carry an
                # explicit primitive/mechanism without claiming that a
                # Semantic Parser ran.
                case = selected.case
            case.validate_local_inputs()
            audit.write_json("case_context.json", case.to_metadata())
            audit.write_json(
                "semantic_resolution.json",
                {
                    **resolution_base,
                    "status": "selected",
                    "selected_option_id": option_id,
                    "selection": selection_record,
                    "selected_cell_budget": (
                        case.cell_count_extent_budget.__dict__
                        if case.cell_count_extent_budget is not None
                        else None
                    ),
                },
            )
            usage["mechanism_selection"] = selection_usage
            bundle = selected.bundle
            if bundle.primitive.scope == "cell_only":
                return self._run_cell_only(
                    audit=audit,
                    case=case,
                    source_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    schema=schema,
                    scene=scene,
                    bundle=bundle,
                    mechanism_id=mechanism_id,
                    planner_images=planner_images,
                    usage=usage,
                )
            allocation = selected.allocation
            tissue_bundle = selected.tissue_bundle
            nuclei_preflight = selected.nuclei_preflight
            if (
                allocation is None
                or tissue_bundle is None
                or nuclei_preflight is None
            ):
                raise JointContractError(
                    "selected tissue interpretation lacks its prepared preflight"
                )
            audit.write_json(
                "joint_nuclei_preflight.json",
                nuclei_preflight.to_metadata(),
            )
            tissue_scene = augment_tissue_scene_with_nuclei_preflight(
                scene.tissue,
                nuclei_preflight,
                auxiliary_structure_masks=scene.auxiliary_structure_masks,
                required_auxiliary_structure_ids=(
                    bundle.mechanism.representability.required_auxiliary_structures
                ),
            )
            audit.write_inputs(
                case=case,
                scene_metadata=scene.to_metadata(),
                skill_metadata={
                    **bundle.to_metadata(),
                    "budget_allocation": allocation.to_metadata(),
                    "tissue_skill_bundle": tissue_bundle.to_metadata(),
                },
            )
            budget_revisions = []
            tissue_pass_usage = []
            joint_pass_usage = []
            execution_feedback: dict = {}
            budget_rebalance_count = 0
            joint_area_feedback_count = 0
            maximum_planning_attempts = (
                self.config.maximum_tissue_planning_attempts
                + self.config.maximum_joint_area_feedback_attempts
            )
            passing_joint = []
            contract_by_joint_candidate = {}
            review_board = None
            tissue_reports = ()
            area_fallback_state = None

            def activate_rebalance_exhausted_area_fallback() -> bool:
                """Restore a proven safe pair after exact-area replan fails."""

                nonlocal allocation
                nonlocal candidates
                nonlocal contract_by_joint_candidate
                nonlocal joint_reports
                nonlocal nuclei_preflight
                nonlocal passing_joint
                nonlocal plan
                nonlocal review_board
                nonlocal tissue_reports
                if (
                    area_fallback_state is None
                    or joint_area_feedback_count <= 0
                ):
                    return False
                candidates = area_fallback_state["candidates"]
                contract_by_joint_candidate = area_fallback_state["contracts"]
                plan = area_fallback_state["plan"]
                review_board = area_fallback_state["review_board"]
                tissue_reports = area_fallback_state["tissue_reports"]
                allocation = area_fallback_state["allocation"]
                nuclei_preflight = area_fallback_state["nuclei_preflight"]
                reports_by_id = area_fallback_state["reports_by_id"]
                certified_min = int(area_fallback_state["certified_min"])
                for candidate in candidates:
                    candidate.tool_trace["batch_min_safe_joint_pixels"] = (
                        certified_min
                    )
                    candidate.tool_trace[
                        "batch_min_safe_joint_certified"
                    ] = True
                    candidate.tool_trace[
                        "joint_area_rebalance_exhausted"
                    ] = True
                joint_reports = tuple(
                    self.joint_gates.run(
                        JointGateContext(
                            case=case,
                            source_tissue=source_tissue,
                            source_nuclei=source_nuclei,
                            schema=schema,
                            scene=scene,
                            bundle=bundle,
                            plan=plan,
                            candidate=candidate,
                            tissue_gate_report=reports_by_id[
                                candidate.tissue_candidate_id
                            ],
                            executable_contract=contract_by_joint_candidate[
                                candidate.candidate_id
                            ],
                        )
                    )
                    for candidate in candidates
                )
                passing_joint = [
                    candidate
                    for candidate in candidates
                    if next(
                        item.passed
                        for item in joint_reports
                        if item.candidate_id == candidate.candidate_id
                    )
                ]
                return bool(passing_joint)

            # A provisional tissue boundary may intersect a complete semantic or
            # native nucleus whose footprint extends beyond T.  Since v1 forbids
            # partial nucleus edits, that extension necessarily belongs to C and
            # therefore J. Re-broker to a fixed point using the observed
            # whole-instance closure cost instead of silently overshooting the
            # joint target. Four bounded attempts leave room for both the
            # provisional T-union-E solve and one feedback pass from fully
            # executed target-nucleus footprints before final execution.
            for planning_pass in range(maximum_planning_attempts):
                tissue_case = _as_tissue_case(
                    case,
                    allocation=allocation,
                    shape=source_tissue.shape,
                )
                tissue_usage = {}
                compiler_usage = {}
                joint_usage = {}
                try:
                    if hasattr(self.tissue_planner, "create_joint_tissue_plan"):
                        raw_tissue_plan, tissue_usage = (
                            self.tissue_planner.create_joint_tissue_plan(
                                case=tissue_case,
                                scene=tissue_scene,
                                bundle=tissue_bundle,
                                joint_bundle=bundle,
                                image_paths=planner_images,
                                nuclei_preflight=nuclei_preflight,
                                execution_feedback=execution_feedback,
                            )
                        )
                    else:
                        raw_tissue_plan, tissue_usage = self.tissue_planner.create_plan(
                            case=tissue_case,
                            scene=tissue_scene,
                            bundle=tissue_bundle,
                            image_paths=planner_images,
                        )
                    validate_edit_plan(
                        raw_tissue_plan,
                        case=tissue_case,
                        scene=tissue_scene,
                        bundle=tissue_bundle,
                    )
                    tissue_plan, compiler_usage = compile_edit_plan(
                        raw_tissue_plan,
                        source_mask=source_tissue,
                        schema=schema,
                        scene=tissue_scene,
                    )
                    validate_edit_plan(
                        tissue_plan,
                        case=tissue_case,
                        scene=tissue_scene,
                        bundle=tissue_bundle,
                    )
                    plan, joint_usage = self.joint_planner.create_plan(
                        case=case,
                        scene=scene,
                        bundle=bundle,
                        tissue_plan=tissue_plan,
                        image_paths=planner_images,
                    )
                    plan = bind_joint_plan_to_nuclei_preflight(
                        plan,
                        nuclei_preflight,
                    )
                except (RefineContractError, JointContractError) as exc:
                    execution_feedback = {
                        "retry_index": planning_pass + 1,
                        "stage": "planning_or_compilation",
                        "errors": [f"{type(exc).__name__}: {exc}"],
                        "failed_interface_ids": [],
                    }
                    audit.write_json(
                        f"execution_feedback_pass_{planning_pass + 1}.json",
                        execution_feedback,
                    )
                    tissue_pass_usage.append(
                        {
                            "pass": planning_pass + 1,
                            **tissue_usage,
                            "compiler": compiler_usage,
                            "budget_allocation": allocation.to_metadata(),
                            "execution_feedback": execution_feedback,
                        }
                    )
                    if (
                        planning_pass + 1 >= maximum_planning_attempts
                    ):
                        raise JointContractError(
                            "tissue planning/compilation exhausted feedback retries: "
                            + "; ".join(execution_feedback["errors"])
                        ) from exc
                    continue
                tissue_pass_usage.append(
                    {
                        "pass": planning_pass + 1,
                        **tissue_usage,
                        "compiler": compiler_usage,
                        "budget_allocation": allocation.to_metadata(),
                    }
                )
                joint_pass_usage.append(
                    {
                        "pass": planning_pass + 1,
                        **joint_usage,
                        "budget_allocation": allocation.to_metadata(),
                    }
                )
                execution_batch = execute_gate_aware_tissue_candidates(
                    source_tissue,
                    source_nuclei=source_nuclei,
                    case=case,
                    schema=schema,
                    tissue_scene=tissue_scene,
                    joint_scene=scene,
                    tissue_case=tissue_case,
                    tissue_plan=tissue_plan,
                    joint_plan=plan,
                    tissue_bundle=tissue_bundle,
                    joint_bundle=bundle,
                    nuclei_preflight=nuclei_preflight,
                    allocation=allocation,
                    executable_contract_compiler=(self.executable_contract_compiler),
                    joint_required_checker_ids=(
                        self.joint_gates.required_checker_ids_for(bundle)
                    ),
                    gates=self.tissue_gates,
                    seed=case.seed + planning_pass * 1_000_003,
                )
                tissue_reports = execution_batch.tissue_gate_reports
                audit.write_json(
                    f"tissue_gate_reports_pass_{planning_pass + 1}.json",
                    [item.to_metadata() for item in tissue_reports],
                )
                audit.write_json(
                    f"tissue_execution_contract_pass_{planning_pass + 1}.json",
                    execution_batch.to_metadata(),
                )
                audit.write_json(
                    f"tissue_edit_plan_pass_{planning_pass + 1}.json",
                    tissue_plan.to_metadata(),
                )
                audit.write_json(
                    f"joint_edit_plan_pass_{planning_pass + 1}.json",
                    plan.to_metadata(),
                )
                audit.write_tissue_execution_review(
                    pass_index=planning_pass + 1,
                    source_image_path=case.source_image_uri,
                    source_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    tissue_scene=tissue_scene,
                    tissue_plan=tissue_plan,
                    execution_batch=execution_batch,
                )
                reports_by_id = {item.candidate_id: item for item in tissue_reports}
                passing_tissue = list(execution_batch.certified_candidates)[
                    : self.config.maximum_tissue_candidates
                ]
                if not passing_tissue:
                    execution_feedback = _summarize_tissue_execution_failure(
                        execution_batch,
                        retry_index=planning_pass + 1,
                    )
                    audit.write_json(
                        f"execution_feedback_pass_{planning_pass + 1}.json",
                        execution_feedback,
                    )
                    if activate_rebalance_exhausted_area_fallback():
                        break
                    if (
                        planning_pass + 1 >= maximum_planning_attempts
                    ):
                        raise JointContractError(
                            "no tissue candidate passed after feedback-directed "
                            "replan/retool attempts"
                        )
                    continue
                contracts_by_tissue_id = {
                    item.tissue_candidate_id: item
                    for item in execution_batch.executable_contracts
                }
                desired_min, desired_max = (
                    case.joint_area_budget.desired_interval_pixels(source_tissue.shape)
                )
                _, hard_max = case.joint_area_budget.hard_interval_pixels(
                    source_tissue.shape
                )
                predicted = [
                    int(
                        np.count_nonzero(
                            np.asarray(item.change_region, dtype=bool)
                            | contracts_by_tissue_id[
                                item.candidate_id
                            ].cell_program.erasure_region
                        )
                    )
                    for item in passing_tissue
                ]
                predicted_above = _provisional_union_requires_rebalance(
                    predicted,
                    hard_max_pixels=hard_max,
                )
                # ``predicted`` contains T plus source-instance erasure E, but
                # it intentionally cannot contain the target footprints that
                # the mature cell executor has not generated yet.  Therefore
                # it is a sound pre-execution lower bound only when it already
                # exceeds the declared hard maximum.  Treating a provisional
                # underfill as final used to reduce T before ADD was realized,
                # which could turn a 19/21 exact packing witness into 17/21 on
                # the next pass.  Underfill feedback belongs exclusively to
                # the executed T ∪ C ledger below.
                if (
                    planning_pass + 1 < maximum_planning_attempts
                    and budget_rebalance_count
                    < maximum_planning_attempts - 1
                    and predicted_above
                ):
                    closure_values = [
                        int(
                            np.count_nonzero(
                                contracts_by_tissue_id[
                                    item.candidate_id
                                ].cell_program.erasure_region
                                & ~np.asarray(item.change_region, dtype=bool)
                            )
                        )
                        for item in passing_tissue
                    ]
                    # Candidate selection needs one executable pair, not a
                    # budget that makes every provisional sibling safe. Bind
                    # the next fixed point to the smallest observed complete-
                    # instance spill: that candidate returns to the joint
                    # target while preserving the largest tissue/P domain for
                    # nuclei packing. Siblings with larger spill remain free
                    # to fail the later joint-area gate.
                    closure = _candidate_preserving_closure_pixels(
                        closure_values
                    )
                    revised = self.budget_solver.reserve_complete_instances(
                        allocation,
                        reserve_pixels=closure,
                    )
                    if revised.tissue_target_pixels != allocation.tissue_target_pixels:
                        direction = "whole_instance_union_closure"
                        budget_revisions.append(
                            {
                                "reason": direction,
                                "desired_joint_interval": [desired_min, desired_max],
                                "provisional_joint_pixels": predicted,
                                "before": allocation.to_metadata(),
                                "after": revised.to_metadata(),
                            }
                        )
                        allocation = revised
                        budget_rebalance_count += 1
                        nuclei_preflight = build_joint_nuclei_preflight(
                            case=case,
                            source_tissue=source_tissue,
                            schema=schema,
                            scene=scene,
                            tissue_bundle=tissue_bundle,
                            joint_bundle=bundle,
                            allocation=allocation,
                        )
                        if not nuclei_preflight.feasible_interface_ids:
                            raise JointContractError(
                                "revised joint budget has no feasible tissue--cell interface"
                            )
                        if not nuclei_preflight.meaningful_tissue_capacity_passed:
                            raise JointContractError(
                                "revised joint budget cannot reach the meaningful tissue floor"
                            )
                        tissue_scene = augment_tissue_scene_with_nuclei_preflight(
                            scene.tissue,
                            nuclei_preflight,
                            auxiliary_structure_masks=(
                                scene.auxiliary_structure_masks
                            ),
                            required_auxiliary_structure_ids=(
                                bundle.mechanism.representability.required_auxiliary_structures
                            ),
                        )
                        audit.write_json(
                            f"joint_nuclei_preflight_pass_{planning_pass + 2}.json",
                            nuclei_preflight.to_metadata(),
                        )
                        execution_feedback = {
                            "retry_index": planning_pass + 1,
                            "stage": "budget_rebalance",
                            "errors": [direction],
                            "failed_interface_ids": [],
                        }
                        continue
                (
                    candidates,
                    joint_reports,
                    contract_by_joint_candidate,
                    review_board,
                    cell_execution_failures,
                ) = self._build_and_gate_joint_candidates(
                    audit=audit,
                    case=case,
                    source_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    schema=schema,
                    scene=scene,
                    bundle=bundle,
                    mechanism_id=mechanism_id,
                    plan=plan,
                    allocation=allocation,
                    passing_tissue=passing_tissue,
                    execution_batch=execution_batch,
                    reports_by_id=reports_by_id,
                )
                audit.write_json(
                    f"joint_gate_reports_pass_{planning_pass + 1}.json",
                    [item.to_metadata() for item in joint_reports],
                )
                passing_joint = [
                    candidate
                    for candidate in candidates
                    if next(
                        item.passed
                        for item in joint_reports
                        if item.candidate_id == candidate.candidate_id
                    )
                ]
                if passing_joint:
                    break
                if cell_execution_failures and not candidates:
                    execution_feedback = {
                        "retry_index": planning_pass + 1,
                        "stage": "cell_execution",
                        "errors": [
                            item["error"] for item in cell_execution_failures
                        ],
                        "failed_interface_ids": list(
                            plan.cell_plan.interface_ids
                        ),
                        "failed_tissue_candidate_ids": [
                            item["tissue_candidate_id"]
                            for item in cell_execution_failures
                        ],
                        "required_action": (
                            "replan a different or broader cell-feasible tissue "
                            "domain after all candidate-local executor retries failed"
                        ),
                    }
                    audit.write_json(
                        f"execution_feedback_pass_{planning_pass + 1}.json",
                        execution_feedback,
                    )
                    if planning_pass + 1 < maximum_planning_attempts:
                        continue
                    raise JointContractError(
                        "all candidate-local cell executions failed after "
                        "bounded replanning"
                    )

                _, hard_max = (
                    case.joint_area_budget.hard_interval_pixels(
                        source_tissue.shape
                    )
                )
                provisional_min = _minimum_safe_above_target_joint_pixels(
                    candidates,
                    joint_reports,
                    desired_max_pixels=desired_max,
                    hard_max_pixels=hard_max,
                    tissue_floor_pixels=(
                        case.joint_area_budget.tissue_floor_pixels(
                            source_tissue.shape
                        )
                    ),
                    require_tissue_floor=False,
                )
                if provisional_min is not None:
                    area_fallback_state = {
                        "allocation": allocation,
                        "candidates": candidates,
                        "certified_min": provisional_min,
                        "contracts": dict(contract_by_joint_candidate),
                        "nuclei_preflight": nuclei_preflight,
                        "plan": plan,
                        "reports_by_id": dict(reports_by_id),
                        "review_board": review_board,
                        "tissue_reports": tuple(tissue_reports),
                    }

                area_feedback_ids = _joint_area_feedback_candidate_ids(
                    joint_reports
                )
                area_feedback_candidates = [
                    item
                    for item in candidates
                    if item.candidate_id in area_feedback_ids
                ]
                actual_joint_pixels = [
                    item.ledger.joint_pixels
                    for item in area_feedback_candidates
                ]
                actual_above = bool(
                    actual_joint_pixels
                    and min(actual_joint_pixels) > desired_max
                )
                actual_below = bool(
                    actual_joint_pixels
                    and max(actual_joint_pixels) < desired_min
                )
                can_retry_joint = bool(
                    planning_pass + 1 < maximum_planning_attempts
                    and joint_area_feedback_count
                    < self.config.maximum_joint_area_feedback_attempts
                )
                if (
                    area_feedback_candidates
                    and can_retry_joint
                    and (actual_above or actual_below)
                ):
                    observed_spill = []
                    for candidate in area_feedback_candidates:
                        tissue_change = np.asarray(
                            candidate.tissue_change, dtype=bool
                        )
                        source_cells = np.asarray(source_nuclei) > 0
                        target_cells = (
                            np.asarray(candidate.target_nuclei_mask) > 0
                        )
                        removed_spill = int(
                            np.count_nonzero(
                                source_cells & ~target_cells & ~tissue_change
                            )
                        )
                        added_spill = int(
                            np.count_nonzero(
                                ~source_cells & target_cells & ~tissue_change
                            )
                        )
                        observed_spill.append(
                            (candidate, removed_spill, added_spill)
                        )
                    selected_spill = (
                        max(
                            observed_spill,
                            key=lambda item: item[1] + item[2],
                        )
                        if actual_above
                        else min(
                            observed_spill,
                            key=lambda item: item[1] + item[2],
                        )
                    )
                    revised = self.budget_solver.reserve_observed_cell_spill(
                        allocation,
                        complete_instance_pixels=selected_spill[1],
                        footprint_spill_pixels=selected_spill[2],
                    )
                    if (
                        revised.tissue_target_pixels
                        != allocation.tissue_target_pixels
                    ):
                        direction = (
                            "executed_joint_union_overfill"
                            if actual_above
                            else "executed_joint_union_underfill"
                        )
                        budget_revisions.append(
                            {
                                "reason": direction,
                                "desired_joint_interval": [
                                    desired_min,
                                    desired_max,
                                ],
                                "executed_joint_pixels": actual_joint_pixels,
                                "selected_feedback_candidate_id": (
                                    selected_spill[0].candidate_id
                                ),
                                "observed_complete_instance_spill_pixels": (
                                    selected_spill[1]
                                ),
                                "observed_target_footprint_spill_pixels": (
                                    selected_spill[2]
                                ),
                                "before": allocation.to_metadata(),
                                "after": revised.to_metadata(),
                            }
                        )
                        allocation = revised
                        budget_rebalance_count += 1
                        joint_area_feedback_count += 1
                        nuclei_preflight = build_joint_nuclei_preflight(
                            case=case,
                            source_tissue=source_tissue,
                            schema=schema,
                            scene=scene,
                            tissue_bundle=tissue_bundle,
                            joint_bundle=bundle,
                            allocation=allocation,
                        )
                        if not nuclei_preflight.feasible_interface_ids:
                            raise JointContractError(
                                "executed joint-area feedback left no feasible "
                                "tissue--cell interface"
                            )
                        if not nuclei_preflight.meaningful_tissue_capacity_passed:
                            raise JointContractError(
                                "executed joint-area feedback cannot reach the "
                                "meaningful tissue floor"
                            )
                        tissue_scene = augment_tissue_scene_with_nuclei_preflight(
                            scene.tissue,
                            nuclei_preflight,
                            auxiliary_structure_masks=(
                                scene.auxiliary_structure_masks
                            ),
                            required_auxiliary_structure_ids=(
                                bundle.mechanism.representability.required_auxiliary_structures
                            ),
                        )
                        audit.write_json(
                            f"joint_nuclei_preflight_pass_{planning_pass + 2}.json",
                            nuclei_preflight.to_metadata(),
                        )
                        execution_feedback = {
                            "retry_index": planning_pass + 1,
                            "stage": "joint_area_gate",
                            "errors": [direction],
                            "failed_interface_ids": [],
                        }
                        audit.write_json(
                            f"execution_feedback_pass_{planning_pass + 1}.json",
                            execution_feedback,
                        )
                        continue
                if activate_rebalance_exhausted_area_fallback():
                    break
                raise JointContractError(
                    "no paired tissue--cell candidate passed all joint gates"
                )
            if not passing_joint or review_board is None:
                raise JointContractError(
                    "joint planning attempts ended before an executable paired "
                    "candidate passed every hard gate"
                )

            # A later exact-area feedback pass may restore an earlier, proven-safe
            # batch after marking its minimum whole-instance closure as certified.
            # Persist the final in-memory state atomically here so the canonical
            # audit files and review panels can never describe the pre-fallback
            # reports while the critic receives the post-fallback reports.
            audit.write_candidates(candidates)
            audit.write_json(
                "joint_gate_reports.json",
                [item.to_metadata() for item in joint_reports],
            )
            audit.write_joint_execution_review(
                source_image_path=case.source_image_uri,
                source_tissue=source_tissue,
                source_nuclei=source_nuclei,
                candidates=candidates,
                gate_reports=joint_reports,
                plan=plan,
                scene=scene,
                executable_contracts=contract_by_joint_candidate,
            )
            review_board = audit.write_review_board(
                source_image_path=case.source_image_uri,
                source_tissue=source_tissue,
                source_nuclei=source_nuclei,
                candidates=candidates,
            )
            usage["tissue_planner"] = {
                "passes": tissue_pass_usage,
                "budget_revisions": budget_revisions,
            }
            usage["joint_planner"] = {"passes": joint_pass_usage}
            audit.write_inputs(
                case=case,
                scene_metadata=scene.to_metadata(),
                skill_metadata={
                    **bundle.to_metadata(),
                    "budget_allocation": allocation.to_metadata(),
                    "budget_revisions": budget_revisions,
                    "tissue_skill_bundle": tissue_bundle.to_metadata(),
                },
            )
            audit.write_json(
                "tissue_gate_reports.json",
                [item.to_metadata() for item in tissue_reports],
            )
            audit.write_json("joint_edit_plan.json", plan.to_metadata())
            critic_result = self.critic.review(
                case=case,
                bundle=bundle,
                candidates=passing_joint,
                gate_reports=joint_reports,
                image_paths=(case.source_image_uri, review_board),
            )
            usage["critic"] = critic_result.usage
            audit.write_json("joint_critic.json", critic_result.to_metadata())
            if critic_result.abstain or not critic_result.rankings:
                reasons.append("independent_multimodal_critic_approval_required")
                return self._finish(
                    audit=audit,
                    case=case,
                    plan=plan,
                    reports=joint_reports,
                    critic=critic_result,
                    status="review_required",
                    reasons=reasons,
                    selected=None,
                    condition=None,
                    usage=usage,
                )
            ranking = critic_result.rankings[0]
            if (
                ranking.confidence < self.config.critic_confidence_threshold
                or ranking.veto_reasons
            ):
                reasons.append("joint_critic_low_confidence_or_veto")
                return self._finish(
                    audit=audit,
                    case=case,
                    plan=plan,
                    reports=joint_reports,
                    critic=critic_result,
                    status="abstained",
                    reasons=reasons,
                    selected=None,
                    condition=None,
                    usage=usage,
                )
            selected = next(
                item
                for item in passing_joint
                if item.candidate_id == ranking.candidate_id
            )
            selected_contract = contract_by_joint_candidate[selected.candidate_id]
            condition = JointCondition(
                case_id=case.case_id,
                candidate_id=selected.candidate_id,
                executable_contract_id=selected_contract.contract_id,
                target_tissue_mask=selected.target_tissue_mask,
                target_nuclei_mask=selected.target_nuclei_mask,
                tissue_change=selected.tissue_change,
                cell_change=selected.cell_change,
                joint_change=selected.joint_change,
                generation_support=selected.generation_support,
                pathology_mechanism=mechanism_id,
                active_skill_rules=bundle.active_rule_ids,
                ledger=selected.ledger,
            )
            handoff_paths = write_generation_handoff(
                audit.case_dir,
                case=case,
                plan=plan,
                bundle=bundle,
                candidate=selected,
                executable_contract=selected_contract,
            )
            audit.paths.update(
                {"handoff_" + key: value for key, value in handoff_paths.items()}
            )
            return self._finish(
                audit=audit,
                case=case,
                plan=plan,
                reports=joint_reports,
                critic=critic_result,
                status=("selected" if self.config.production else "selected_research"),
                reasons=(),
                selected=selected.candidate_id,
                condition=condition,
                usage=usage,
            )
        except Exception as exc:  # noqa: BLE001 - orchestration must fail closed
            reasons.append(f"{type(exc).__name__}: {exc}")
            if candidates:
                audit.write_candidates(candidates)
            if (
                source_tissue is not None
                and source_nuclei is not None
                and scene is not None
            ):
                audit.write_abstain_review(
                    source_image_path=case.source_image_uri,
                    source_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    scene=scene,
                    reason=reasons[-1],
                    plan=plan,
                    nuclei_preflight=nuclei_preflight,
                )
            return self._finish(
                audit=audit,
                case=case,
                plan=plan,
                reports=joint_reports,
                critic=critic_result,
                status="abstained",
                reasons=reasons,
                selected=None,
                condition=None,
                usage=usage,
            )

    def _prepare_interpretations(
        self,
        *,
        case: JointCaseContext,
        source_tissue: np.ndarray,
        schema,
        scene,
    ) -> tuple[dict[str, _PreparedInterpretation], dict[str, str]]:
        """Compile every semantic hypothesis before visual disambiguation.

        The visual Planner only sees primitive--mechanism pairs that already
        satisfy the four knowledge axes and deterministic tissue/nucleus
        preflight. This allows a contextual interpretation to remain available
        when the direct interpretation is physically impossible, without
        letting the LLM invent a primitive or numeric budget.
        """

        raw_hypotheses = case.semantic_intent.get(
            "primitive_hypotheses", ()
        )
        if not raw_hypotheses:
            raw_hypotheses = (
                {
                    "primitive_id": case.primitive_id,
                    "semantic_fit": "explicit",
                    "priority": 0,
                    "rationale": "legacy research case supplies one explicit primitive",
                },
            )
        prepared: dict[str, _PreparedInterpretation] = {}
        rejected: dict[str, str] = {}
        for raw in sorted(
            raw_hypotheses,
            key=lambda item: (
                int(item.get("priority", 0)) if isinstance(item, Mapping) else 0,
                str(item.get("primitive_id", "")) if isinstance(item, Mapping) else "",
            ),
        ):
            if not isinstance(raw, Mapping):
                rejected["malformed-hypothesis"] = (
                    "semantic primitive hypothesis is not an object"
                )
                continue
            primitive_id = str(raw.get("primitive_id") or "")
            semantic_fit = str(raw.get("semantic_fit") or "")
            rationale = str(raw.get("rationale") or "")
            priority = int(raw.get("priority", 0))
            candidate_case = replace(case, primitive_id=primitive_id)
            primitive_contract = self.joint_skills.primitives.get(primitive_id)
            if (
                primitive_contract is not None
                and primitive_contract.scope == "cell_only"
                and candidate_case.cell_count_extent_budget is None
            ):
                if primitive_id == "neoplastic-cell-infiltration-increase-v1":
                    budget, budget_metadata = _derive_infiltration_budget(scene)
                else:
                    budget, budget_metadata = _derive_local_population_budget(
                        scene,
                        primitive_id=primitive_id,
                        semantic_intent=candidate_case.semantic_intent,
                        host_tissue_labels=primitive_contract.host_tissue_labels,
                    )
                semantic_metadata = dict(candidate_case.semantic_intent)
                semantic_metadata.setdefault(
                    "derived_budget_policies", {}
                )[primitive_id] = budget_metadata
                candidate_provenance = dict(candidate_case.provenance)
                selected_zone_id = budget_metadata.get(
                    "selected_population_zone_id"
                )
                if selected_zone_id:
                    candidate_provenance["joint_population_zone_id"] = (
                        selected_zone_id
                    )
                candidate_case = replace(
                    candidate_case,
                    cell_count_extent_budget=budget,
                    semantic_intent=semantic_metadata,
                    provenance=candidate_provenance,
                )
            try:
                candidate_case.validate_local_inputs()
                mechanisms, mechanism_rejections = (
                    self.joint_skills.eligible_mechanisms_for_case(
                        case=candidate_case,
                        available_checker_ids=set(
                            self.joint_gates.available_checker_ids
                        ),
                        production=self.config.production,
                    )
                )
            except (JointContractError, RefineContractError, ValueError) as exc:
                rejected[primitive_id or "missing-primitive"] = str(exc)
                continue
            if not mechanisms:
                rejected[primitive_id] = "; ".join(
                    f"{key}={value}"
                    for key, value in sorted(mechanism_rejections.items())
                ) or "no joint mechanism supports this primitive"
                continue
            for mechanism in mechanisms:
                option_id = f"{primitive_id}::{mechanism.mechanism_id}"
                try:
                    bundle = self.joint_skills.compose(
                        case=candidate_case,
                        mechanism_id=mechanism.mechanism_id,
                        available_checker_ids=set(
                            self.joint_gates.available_checker_ids
                        ),
                        production=self.config.production,
                    )
                    allocation = None
                    tissue_bundle = None
                    nuclei_preflight = None
                    feasibility: dict[str, Any] = {
                        "four_axis_skill_intersection": "passed",
                        "deterministic_preflight": "passed",
                    }
                    if bundle.primitive.scope == "tissue_and_cell":
                        if candidate_case.joint_area_budget is None:
                            raise JointContractError(
                                "tissue interpretation has no system-owned joint area budget"
                            )
                        allocation = self.budget_solver.allocate(
                            shape=source_tissue.shape,
                            budget=candidate_case.joint_area_budget,
                            bundle=bundle,
                        )
                        tissue_bundle = self.mask_skills.compose(
                            pathology_domain_id=(
                                candidate_case.pathology_domain_id
                            ),
                            annotation_profile_id=(
                                candidate_case.annotation_profile_id
                            ),
                            primitive_id=primitive_id,
                            production=self.config.production,
                            available_checker_ids=(
                                self.tissue_gates.available_checker_ids
                            ),
                        )
                        nuclei_preflight = build_joint_nuclei_preflight(
                            case=candidate_case,
                            source_tissue=source_tissue,
                            schema=schema,
                            scene=scene,
                            tissue_bundle=tissue_bundle,
                            joint_bundle=bundle,
                            allocation=allocation,
                        )
                        failures = []
                        if nuclei_preflight.required_auxiliary_missing:
                            failures.append(
                                "missing auxiliary="
                                + ",".join(
                                    nuclei_preflight.required_auxiliary_missing
                                )
                            )
                        if nuclei_preflight.required_provenance_missing:
                            failures.append(
                                "missing provenance="
                                + ",".join(
                                    nuclei_preflight.required_provenance_missing
                                )
                            )
                        if not nuclei_preflight.meaningful_tissue_capacity_passed:
                            failures.append(
                                "meaningful tissue capacity "
                                f"{nuclei_preflight.aggregate_feasible_tissue_capacity_pixels}"
                                f"<{nuclei_preflight.meaningful_tissue_floor_pixels}"
                            )
                        if not nuclei_preflight.feasible_interface_ids:
                            failures.append("no nuclei-safe executable interface")
                        if failures:
                            raise JointContractError("; ".join(failures))
                        feasibility.update(
                            {
                                "feasible_interface_count": len(
                                    nuclei_preflight.feasible_interface_ids
                                ),
                                "aggregate_tissue_capacity_pixels": (
                                    nuclei_preflight.aggregate_feasible_tissue_capacity_pixels
                                ),
                                "meaningful_tissue_floor_pixels": (
                                    nuclei_preflight.meaningful_tissue_floor_pixels
                                ),
                            }
                        )
                    elif "add" in bundle.mechanism.cell_program.actions:
                        allowed = set(
                            bundle.mechanism.cell_program.allowed_cell_classes
                        )
                        complete_references = sum(
                            item.completeness_status == "complete"
                            and item.class_id in allowed
                            for item in scene.cells.instances
                        )
                        if complete_references <= 0:
                            raise JointContractError(
                                "cell interpretation has no complete same-patch reference nucleus"
                            )
                        if (
                            primitive_id
                            == "neoplastic-cell-infiltration-increase-v1"
                        ):
                            label_contract = (
                                bundle.mechanism.tissue_program.primitive_label_contracts[
                                    primitive_id
                                ]
                            )
                            receiving_labels = set(
                                label_contract["source_labels"]
                            )
                            interface_pairs = {
                                tuple(
                                    sorted(
                                        (
                                            interface.source_component_id,
                                            interface.target_component_id,
                                        )
                                    )
                                )
                                for interface in scene.tissue.graph.interfaces
                                if "Tumor"
                                in {
                                    interface.source_label,
                                    interface.target_label,
                                }
                                and (
                                    {
                                        interface.source_label,
                                        interface.target_label,
                                    }
                                    - {"Tumor"}
                                ).intersection(receiving_labels)
                            }
                            if not interface_pairs:
                                raise JointContractError(
                                    "contextual infiltration has no tumor-to-receiving-tissue interface"
                                )
                            feasibility[
                                "candidate_infiltration_interface_count"
                            ] = len(interface_pairs)
                        feasibility["complete_reference_instances"] = int(
                            complete_references
                        )
                        feasibility["cell_budget"] = (
                            candidate_case.cell_count_extent_budget.__dict__
                            if candidate_case.cell_count_extent_budget
                            is not None
                            else None
                        )
                    option = JointInterpretationOption(
                        primitive_id=primitive_id,
                        semantic_fit=semantic_fit,
                        semantic_priority=priority,
                        semantic_rationale=rationale,
                        mechanism=mechanism,
                        feasibility=feasibility,
                    )
                    prepared[option_id] = _PreparedInterpretation(
                        option=option,
                        case=candidate_case,
                        bundle=bundle,
                        allocation=allocation,
                        tissue_bundle=tissue_bundle,
                        nuclei_preflight=nuclei_preflight,
                    )
                except (JointContractError, RefineContractError, ValueError) as exc:
                    rejected[option_id] = str(exc)
        return prepared, rejected

    def _build_and_gate_joint_candidates(
        self,
        *,
        audit,
        case,
        source_tissue,
        source_nuclei,
        schema,
        scene,
        bundle,
        mechanism_id,
        plan,
        allocation,
        passing_tissue,
        execution_batch,
        reports_by_id,
    ):
        """Execute cell tools and gate atomic tissue--cell candidate pairs."""

        contracts_by_tissue_id = {
            item.tissue_candidate_id: item
            for item in execution_batch.executable_contracts
            if item.tissue_candidate_id
            in {candidate.candidate_id for candidate in passing_tissue}
        }
        for contract in contracts_by_tissue_id.values():
            audit.write_executable_contract(contract)

        joint_candidates = []
        prohibited = set(
            bundle.annotation_profile.prohibit_generation_support_fine_ids
        )
        generation_allowed = ~np.isin(source_tissue, tuple(prohibited))
        contract_by_joint_candidate: dict[str, ExecutableJointContract] = {}
        cell_execution_failures: list[dict[str, str]] = []
        source_authority = build_scene_instance_authority(scene, source_nuclei)
        for tissue_candidate in passing_tissue:
            executable_contract = contracts_by_tissue_id[
                tissue_candidate.candidate_id
            ]
            mature_supported = bool(
                self.cell_executor is not None
                and self.cell_executor.supports(executable_contract)
            )
            requires_mature = bool(
                case.provenance.get(
                    "require_mature_probnet_regeneration", False
                )
                and plan.cell_plan.baseline_mode
                == "regenerate_target_population"
            )
            if requires_mature and not mature_supported:
                raise JointContractError(
                    "case contract requires the mature ProbNet regeneration "
                    "pipeline; a checkpoint ranker plus research layout is insufficient"
                )
            if mature_supported:
                try:
                    layouts = self.cell_executor.execute(
                        contract=executable_contract,
                        source_tissue=source_tissue,
                        target_tissue=tissue_candidate.target_mask,
                        source_nuclei=source_nuclei,
                        scene=scene,
                        output_dir=(
                            audit.case_dir
                            / "mature_probnet"
                            / tissue_candidate.candidate_id
                        ),
                        prohibited_tissue_ids=tuple(
                            sorted(
                                bundle.annotation_profile.prohibit_cell_placement_fine_ids
                            )
                        ),
                        seed=case.seed,
                        variants=self.config.cell_layouts_per_tissue,
                    )
                except JointContractError as exc:
                    cell_execution_failures.append(
                        {
                            "tissue_candidate_id": tissue_candidate.candidate_id,
                            "executable_contract_id": executable_contract.contract_id,
                            "executor": type(self.cell_executor).__name__,
                            "error": str(exc),
                        }
                    )
                    continue
            else:
                if (
                    self.config.production
                    and self.config.require_mature_probnet_in_production
                    and plan.cell_plan.baseline_mode
                    == "regenerate_target_population"
                ):
                    raise JointContractError(
                        "production target-population regeneration requires "
                        "the mature ProbNet executor"
                    )
                if self.config.production and (
                    self.ranker is None
                    or getattr(self.ranker, "name", "")
                    == "deterministic_distance_ranker"
                ):
                    raise JointContractError(
                        "production structured cell execution requires the frozen ProbNet ranker"
                    )
                layouts = generate_cell_layouts(
                    source_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    tissue_candidate=tissue_candidate,
                    schema=schema,
                    scene=scene,
                    plan=plan,
                    bundle=bundle,
                    allocation=allocation,
                    executable_contract=executable_contract,
                    seed=case.seed,
                    ranker=self.ranker,
                    variants=self.config.cell_layouts_per_tissue,
                )
            if not layouts:
                cell_execution_failures.append(
                    {
                        "tissue_candidate_id": tissue_candidate.candidate_id,
                        "executable_contract_id": executable_contract.contract_id,
                        "executor": (
                            type(self.cell_executor).__name__
                            if mature_supported
                            else "deterministic_cell_layouts"
                        ),
                        "error": "executor returned no candidate layouts",
                    }
                )
                continue
            if layouts and "reference_shape_review" not in audit.paths:
                audit.write_reference_shape_review(
                    source_image_path=case.source_image_uri,
                    instance_masks=scene.instance_masks,
                    eligible_ids=layouts[0].trace.get(
                        "reference_shape_ids", ()
                    ),
                    rejected=layouts[0].trace.get(
                        "reference_shape_rejections", {}
                    ),
                )
            for layout in layouts:
                candidate_id = (
                    f"joint-{tissue_candidate.candidate_id}-"
                    f"{layout.cell_candidate_id}"
                )
                contract_by_joint_candidate[candidate_id] = executable_contract
                trace = {
                    **layout.trace,
                    "mechanism_id": mechanism_id,
                    "skill_version": bundle.mechanism.version,
                    "tissue_tool_trace": tissue_candidate.tool_trace,
                    "budget_allocation": allocation.to_metadata(),
                    "source_instance_authority": authority_trace(
                        source_authority
                    ),
                }
                joint_candidates.append(
                    build_joint_candidate(
                        candidate_id=candidate_id,
                        tissue_candidate_id=tissue_candidate.candidate_id,
                        cell_candidate_id=layout.cell_candidate_id,
                        mechanism_id=mechanism_id,
                        source_tissue=source_tissue,
                        target_tissue=tissue_candidate.target_mask,
                        source_nuclei=source_nuclei,
                        target_nuclei=layout.target_nuclei_mask,
                        generation_halo_px=plan.coupling_plan.maximum_halo_px,
                        generation_allowed_region=generation_allowed,
                        generation_support_contract=(
                            executable_contract.cell_program.support_context_region
                        ),
                        source_instance_masks=scene.instance_masks,
                        source_instance_classes={
                            item.instance_id: item.class_id
                            for item in scene.cells.instances
                        },
                        erased_source_instance_ids=(
                            executable_contract.erase_instance_ids
                        ),
                        tool_trace=trace,
                    )
                )
        audit.write_json(
            "cell_execution_failures.json", cell_execution_failures
        )
        candidates = tuple(joint_candidates[:12])
        batch_max_joint = max(
            (item.ledger.joint_pixels for item in candidates), default=0
        )
        for candidate in candidates:
            candidate.tool_trace["batch_max_observed_joint_pixels"] = (
                batch_max_joint
            )
            candidate.tool_trace["batch_max_safe_joint_pixels"] = -1
            candidate.tool_trace["batch_max_safe_joint_certified"] = False
            candidate.tool_trace["batch_min_safe_joint_pixels"] = -1
            candidate.tool_trace["batch_min_safe_joint_certified"] = False
        joint_reports = tuple(
            self.joint_gates.run(
                JointGateContext(
                    case=case,
                    source_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    schema=schema,
                    scene=scene,
                    bundle=bundle,
                    plan=plan,
                    candidate=candidate,
                    tissue_gate_report=reports_by_id[
                        candidate.tissue_candidate_id
                    ],
                    executable_contract=contract_by_joint_candidate[
                        candidate.candidate_id
                    ],
                )
            )
            for candidate in candidates
        )
        if not any(report.passed for report in joint_reports):
            hard_min, hard_max = case.joint_area_budget.hard_interval_pixels(
                source_tissue.shape
            )
            desired_min, desired_max = case.joint_area_budget.desired_interval_pixels(
                source_tissue.shape
            )
            certified_max = _maximum_safe_below_target_joint_pixels(
                candidates,
                joint_reports,
                hard_min_pixels=hard_min,
                desired_min_pixels=desired_min,
            )
            if certified_max is not None:
                for candidate in candidates:
                    candidate.tool_trace["batch_max_safe_joint_pixels"] = (
                        certified_max
                    )
                    candidate.tool_trace["batch_max_safe_joint_certified"] = (
                        True
                    )
                joint_reports = tuple(
                    self.joint_gates.run(
                        JointGateContext(
                            case=case,
                            source_tissue=source_tissue,
                            source_nuclei=source_nuclei,
                            schema=schema,
                            scene=scene,
                            bundle=bundle,
                            plan=plan,
                            candidate=candidate,
                            tissue_gate_report=reports_by_id[
                                candidate.tissue_candidate_id
                            ],
                            executable_contract=contract_by_joint_candidate[
                                candidate.candidate_id
                            ],
                        )
                    )
                    for candidate in candidates
                )
            else:
                certified_min = _minimum_safe_above_target_joint_pixels(
                    candidates,
                    joint_reports,
                    desired_max_pixels=desired_max,
                    hard_max_pixels=hard_max,
                    tissue_floor_pixels=case.joint_area_budget.tissue_floor_pixels(
                        source_tissue.shape
                    ),
                )
                if certified_min is not None:
                    for candidate in candidates:
                        candidate.tool_trace["batch_min_safe_joint_pixels"] = (
                            certified_min
                        )
                        candidate.tool_trace[
                            "batch_min_safe_joint_certified"
                        ] = True
                    joint_reports = tuple(
                        self.joint_gates.run(
                            JointGateContext(
                                case=case,
                                source_tissue=source_tissue,
                                source_nuclei=source_nuclei,
                                schema=schema,
                                scene=scene,
                                bundle=bundle,
                                plan=plan,
                                candidate=candidate,
                                tissue_gate_report=reports_by_id[
                                    candidate.tissue_candidate_id
                                ],
                                executable_contract=contract_by_joint_candidate[
                                    candidate.candidate_id
                                ],
                            )
                        )
                        for candidate in candidates
                    )
        audit.write_candidates(candidates)
        audit.write_json(
            "joint_gate_reports.json",
            [item.to_metadata() for item in joint_reports],
        )
        audit.write_joint_execution_review(
            source_image_path=case.source_image_uri,
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            candidates=candidates,
            gate_reports=joint_reports,
            plan=plan,
            scene=scene,
            executable_contracts=contract_by_joint_candidate,
        )
        review_board = audit.write_review_board(
            source_image_path=case.source_image_uri,
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            candidates=candidates,
        )
        return (
            candidates,
            joint_reports,
            contract_by_joint_candidate,
            review_board,
            tuple(cell_execution_failures),
        )

    def _run_cell_only(
        self,
        *,
        audit,
        case,
        source_tissue,
        source_nuclei,
        schema,
        scene,
        bundle,
        mechanism_id,
        planner_images,
        usage,
    ):
        """Execute a count/extent primitive without entering the tissue solver."""

        plan, joint_usage = self.joint_planner.create_plan(
            case=case,
            scene=scene,
            bundle=bundle,
            tissue_plan=None,
            image_paths=planner_images,
        )
        usage["joint_planner"] = joint_usage
        audit.write_inputs(
            case=case,
            scene_metadata=scene.to_metadata(),
            skill_metadata={
                **bundle.to_metadata(),
                "budget_mode": "count_extent",
                "cell_count_extent_budget": (
                    case.cell_count_extent_budget.__dict__
                    if case.cell_count_extent_budget is not None
                    else None
                ),
                "tissue_skill_bundle": None,
            },
        )
        audit.write_json("joint_edit_plan.json", plan.to_metadata())
        interface_id = (
            plan.cell_plan.interface_ids[0]
            if plan.cell_plan.interface_ids
            else plan.cell_plan.core_zone
        )
        preserved_tissue = CandidateMask(
            candidate_id="tissue-preserved",
            interface_id=interface_id,
            tool_name="preserve_tissue",
            target_mask=np.asarray(source_tissue).copy(),
            change_region=np.zeros_like(source_tissue, dtype=bool),
            tool_trace={
                "tool_name": "preserve_tissue",
                "desired_target_pixels": 0,
                "resolved_target_pixels": 0,
                "unrequested_label_violations": 0,
            },
        )
        tissue_report = GateReport(
            candidate_id=preserved_tissue.candidate_id,
            passed=True,
            checks=(),
        )
        audit.write_json("tissue_gate_reports.json", [tissue_report.to_metadata()])
        executable_contract = self.executable_contract_compiler.compile(
            case=case,
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            schema=schema,
            scene=scene,
            plan=plan,
            bundle=bundle,
            tissue_candidate=preserved_tissue,
            tissue_gate_report=tissue_report,
            allocation=None,
            required_checker_ids=(self.joint_gates.required_checker_ids_for(bundle)),
        )
        audit.write_executable_contract(executable_contract)
        # The mature pipeline owns target-population regeneration. Structured
        # additions are deterministic templates whose anchors may be ranked by
        # the frozen ProbNet but whose count and geometry remain skill-owned.
        if self.config.production and (
            self.ranker is None
            or getattr(self.ranker, "name", "") == "deterministic_distance_ranker"
        ):
            raise JointContractError(
                "production cell-only execution requires the frozen ProbNet ranker"
            )
        layouts = generate_cell_layouts(
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            tissue_candidate=preserved_tissue,
            schema=schema,
            scene=scene,
            plan=plan,
            bundle=bundle,
            allocation=None,
            executable_contract=executable_contract,
            seed=case.seed,
            ranker=self.ranker,
            variants=self.config.cell_layouts_per_tissue,
        )
        if not layouts:
            raise JointContractError(
                "cell-only tool program could not realize the count/extent contract"
            )
        if "reference_shape_review" not in audit.paths:
            audit.write_reference_shape_review(
                source_image_path=case.source_image_uri,
                instance_masks=scene.instance_masks,
                eligible_ids=layouts[0].trace.get("reference_shape_ids", ()),
                rejected=layouts[0].trace.get("reference_shape_rejections", {}),
            )
        prohibited = set(bundle.annotation_profile.prohibit_generation_support_fine_ids)
        generation_allowed = ~np.isin(source_tissue, tuple(prohibited))
        candidates = []
        source_authority = build_scene_instance_authority(scene, source_nuclei)
        for layout in layouts:
            trace = {
                **layout.trace,
                "mechanism_id": mechanism_id,
                "skill_version": bundle.mechanism.version,
                "tissue_tool_trace": preserved_tissue.tool_trace,
                "budget_mode": "count_extent",
                "source_instance_authority": authority_trace(
                    source_authority
                ),
            }
            candidates.append(
                build_joint_candidate(
                    candidate_id=f"joint-cell-only-{layout.cell_candidate_id}",
                    tissue_candidate_id=preserved_tissue.candidate_id,
                    cell_candidate_id=layout.cell_candidate_id,
                    mechanism_id=mechanism_id,
                    source_tissue=source_tissue,
                    target_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    target_nuclei=layout.target_nuclei_mask,
                    generation_halo_px=plan.coupling_plan.maximum_halo_px,
                    generation_allowed_region=generation_allowed,
                    generation_support_contract=(
                        executable_contract.cell_program.support_context_region
                    ),
                    source_instance_masks=scene.instance_masks,
                    source_instance_classes={
                        item.instance_id: item.class_id
                        for item in scene.cells.instances
                    },
                    erased_source_instance_ids=(
                        executable_contract.erase_instance_ids
                    ),
                    tool_trace=trace,
                )
            )
        candidates = tuple(candidates)
        for candidate in candidates:
            candidate.tool_trace["batch_max_safe_joint_pixels"] = max(
                item.ledger.joint_pixels for item in candidates
            )
            candidate.tool_trace["batch_max_safe_joint_certified"] = True
        reports = tuple(
            self.joint_gates.run(
                JointGateContext(
                    case=case,
                    source_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    schema=schema,
                    scene=scene,
                    bundle=bundle,
                    plan=plan,
                    candidate=candidate,
                    tissue_gate_report=tissue_report,
                    executable_contract=executable_contract,
                )
            )
            for candidate in candidates
        )
        audit.write_candidates(candidates)
        audit.write_json(
            "joint_gate_reports.json", [item.to_metadata() for item in reports]
        )
        audit.write_joint_execution_review(
            source_image_path=case.source_image_uri,
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            candidates=candidates,
            gate_reports=reports,
            plan=plan,
            scene=scene,
            executable_contracts=executable_contract,
        )
        review_board = audit.write_review_board(
            source_image_path=case.source_image_uri,
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            candidates=candidates,
        )
        passing = [
            candidate
            for candidate in candidates
            if next(
                report.passed
                for report in reports
                if report.candidate_id == candidate.candidate_id
            )
        ]
        if not passing:
            raise JointContractError(
                "no cell-only candidate passed its joint condition gates"
            )
        critic = self.critic.review(
            case=case,
            bundle=bundle,
            candidates=passing,
            gate_reports=reports,
            image_paths=(case.source_image_uri, review_board),
        )
        usage["critic"] = critic.usage
        audit.write_json("joint_critic.json", critic.to_metadata())
        if critic.abstain or not critic.rankings:
            return self._finish(
                audit=audit,
                case=case,
                plan=plan,
                reports=reports,
                critic=critic,
                status="review_required",
                reasons=("independent_multimodal_critic_approval_required",),
                selected=None,
                condition=None,
                usage=usage,
            )
        ranking = critic.rankings[0]
        if (
            ranking.confidence < self.config.critic_confidence_threshold
            or ranking.veto_reasons
        ):
            return self._finish(
                audit=audit,
                case=case,
                plan=plan,
                reports=reports,
                critic=critic,
                status="abstained",
                reasons=("joint_critic_low_confidence_or_veto",),
                selected=None,
                condition=None,
                usage=usage,
            )
        selected = next(
            item for item in passing if item.candidate_id == ranking.candidate_id
        )
        condition = JointCondition(
            case_id=case.case_id,
            candidate_id=selected.candidate_id,
            executable_contract_id=executable_contract.contract_id,
            target_tissue_mask=selected.target_tissue_mask,
            target_nuclei_mask=selected.target_nuclei_mask,
            tissue_change=selected.tissue_change,
            cell_change=selected.cell_change,
            joint_change=selected.joint_change,
            generation_support=selected.generation_support,
            pathology_mechanism=mechanism_id,
            active_skill_rules=bundle.active_rule_ids,
            ledger=selected.ledger,
        )
        handoff_paths = write_generation_handoff(
            audit.case_dir,
            case=case,
            plan=plan,
            bundle=bundle,
            candidate=selected,
            executable_contract=executable_contract,
        )
        audit.paths.update(
            {"handoff_" + key: value for key, value in handoff_paths.items()}
        )
        return self._finish(
            audit=audit,
            case=case,
            plan=plan,
            reports=reports,
            critic=critic,
            status=("selected" if self.config.production else "selected_research"),
            reasons=(),
            selected=selected.candidate_id,
            condition=condition,
            usage=usage,
        )

    def _finish(
        self,
        *,
        audit,
        case,
        plan,
        reports,
        critic,
        status,
        reasons,
        selected,
        condition,
        usage,
    ):
        summary = {
            "status": status,
            "selected_candidate_id": selected,
            "abstain_reasons": list(reasons),
            "usage": usage,
        }
        audit.write_json("result.json", summary)
        return JointWorkflowResult(
            status=status,
            case_context=case,
            joint_plan=plan,
            gate_reports=tuple(reports),
            critic_result=critic,
            selected_candidate_id=selected,
            condition=condition,
            abstain_reasons=tuple(reasons),
            artifact_paths=dict(audit.paths),
            usage=usage,
        )


def _maximum_safe_below_target_joint_pixels(
    candidates,
    reports,
    *,
    hard_min_pixels: int,
    desired_min_pixels: int,
) -> int | None:
    """Certify the largest under-target candidate that passed every other gate.

    The raw largest union is not necessarily safe: it may fail morphology,
    topology, seam, or label constraints. The fallback maximum is therefore
    computed only after all non-area hard gates have run.
    """

    candidates_by_id = {item.candidate_id: item for item in candidates}
    safe_pixels = []
    for report in reports:
        candidate = candidates_by_id.get(report.candidate_id)
        if candidate is None:
            continue
        hard_failures = {
            check.check_id
            for check in report.checks
            if check.severity == "hard" and not check.passed
        }
        actual = int(candidate.ledger.joint_pixels)
        if (
            hard_failures == {"joint_area"}
            and int(hard_min_pixels) <= actual < int(desired_min_pixels)
        ):
            safe_pixels.append(actual)
    return max(safe_pixels) if safe_pixels else None


def _joint_area_feedback_candidate_ids(reports) -> set[str]:
    """Return candidates that need only a deterministic joint-area re-broker.

    A bad shape variant from the same tissue candidate must not suppress
    feedback for a sibling variant that already passed every non-area gate.
    Only otherwise-safe candidates contribute spill observations.
    """

    eligible = set()
    for report in reports:
        hard_failures = {
            check.check_id
            for check in report.checks
            if check.severity == "hard" and not check.passed
        }
        if hard_failures == {"joint_area"}:
            eligible.add(report.candidate_id)
    return eligible


def _candidate_preserving_closure_pixels(values) -> int:
    """Choose the fixed point that preserves one feasible tissue sibling."""

    normalized = [max(0, int(value)) for value in values]
    return min(normalized) if normalized else 0


def _provisional_union_requires_rebalance(
    predicted_pixels,
    *,
    hard_max_pixels: int,
) -> bool:
    """Allow pre-cell feedback only for an already-certain hard overfill.

    The provisional ledger is ``T ∪ E`` and omits ADD footprints that do
    not exist until cell execution.  It may therefore prove that the declared
    hard maximum is unreachable, but it cannot prove underfill and must not
    optimize the narrower desired tolerance before the paired condition
    exists.
    """

    values = [max(0, int(value)) for value in predicted_pixels]
    return bool(values and min(values) > int(hard_max_pixels))


def _minimum_safe_above_target_joint_pixels(
    candidates,
    reports,
    *,
    desired_max_pixels: int,
    hard_max_pixels: int,
    tissue_floor_pixels: int,
    require_tissue_floor: bool = True,
) -> int | None:
    """Certify the closest safe over-target union forced by whole instances.

    This fallback is deliberately narrower than the below-target capacity
    fallback.  It is available only after tissue change has reached its hard
    floor, every other hard gate has passed, and complete-nucleus closure
    leaves the paired condition slightly above the desired interval but still
    inside the declared hard range.  We then select the smallest safe union in
    the executed batch; arbitrary overshoot is never accepted.
    """

    candidates_by_id = {item.candidate_id: item for item in candidates}
    safe_pixels = []
    for report in reports:
        candidate = candidates_by_id.get(report.candidate_id)
        if candidate is None:
            continue
        hard_failures = {
            check.check_id
            for check in report.checks
            if check.severity == "hard" and not check.passed
        }
        actual = int(candidate.ledger.joint_pixels)
        tissue = int(candidate.ledger.tissue_pixels)
        if (
            hard_failures == {"joint_area"}
            and (
                not require_tissue_floor
                or tissue == int(tissue_floor_pixels)
            )
            and int(desired_max_pixels) < actual <= int(hard_max_pixels)
        ):
            safe_pixels.append(actual)
    return min(safe_pixels) if safe_pixels else None


def _complete_instance_extension_pixels(
    tissue_change: np.ndarray,
    *,
    scene,
    protected_ids: set[str],
) -> int:
    """Pixels that whole-instance removal must add to C outside provisional T."""

    change = np.asarray(tissue_change, dtype=bool)
    closure = np.zeros_like(change)
    for instance_id, component in scene.instance_masks.items():
        if instance_id in protected_ids:
            continue
        current = np.asarray(component, dtype=bool)
        if np.any(current & change):
            closure |= current & ~change
    return int(np.count_nonzero(closure))


def _summarize_tissue_execution_failure(
    execution_batch,
    *,
    retry_index: int,
) -> dict:
    """Turn hard failures into structured feedback for Planner and tools."""

    hard_failures: dict[str, int] = {}
    tissue_passed_ids = set()
    for report in execution_batch.tissue_gate_reports:
        if report.passed:
            tissue_passed_ids.add(report.candidate_id)
        for check in report.checks:
            if check.severity == "hard" and not check.passed:
                hard_failures[check.check_id] = (
                    hard_failures.get(check.check_id, 0) + 1
                )
    cell_failures: dict[str, int] = {}
    failed_candidate_ids = set()
    for report in execution_batch.cell_feasibility_reports:
        if report.passed:
            continue
        failed_candidate_ids.add(report.candidate_id)
        for reason in report.reasons:
            cell_failures[reason] = cell_failures.get(reason, 0) + 1
    candidate_by_id = {
        item.candidate_id: item for item in execution_batch.all_candidates
    }
    failed_interface_ids = set()
    for candidate_id in failed_candidate_ids or set(candidate_by_id):
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            continue
        failed_interface_ids.update(
            candidate.tool_trace.get("interface_ids", (candidate.interface_id,))
        )
    return {
        "retry_index": int(retry_index),
        "stage": (
            "cell_feasibility" if tissue_passed_ids else "tissue_gate"
        ),
        "hard_tissue_failure_counts": dict(sorted(hard_failures.items())),
        "cell_feasibility_failure_counts": dict(sorted(cell_failures.items())),
        "failed_interface_ids": sorted(failed_interface_ids),
        "required_action": (
            "select broader or alternative cell-feasible interfaces and anchors; "
            "redistribute area across shallower fronts; use a new deterministic tool seed"
        ),
    }


def _as_tissue_case(case: JointCaseContext, *, allocation, shape) -> CaseContext:
    total = int(np.prod(shape))
    target = allocation.tissue_target_pixels / max(1, total)
    # Keep the public burden floor in the tissue tool's own contract.  If an
    # older/adaptive manifest exposes a lower compiler floor, generating many
    # tiny candidates only to reject them in the joint preflight is both slow
    # and semantically wrong: the tissue tool must emit only candidates it is
    # itself authorized to call meaningful.
    floor_pixels = max(
        int(allocation.tissue_execution_floor_pixels),
        int(case.joint_area_budget.tissue_floor_pixels(shape)),
    )
    if int(allocation.tissue_target_pixels) < floor_pixels:
        raise JointContractError(
            "joint budget allocation placed the tissue target below the "
            f"binding meaningful floor: target={allocation.tissue_target_pixels}, "
            f"floor={floor_pixels}"
        )
    floor = floor_pixels / max(1, total)
    budget = AreaBudget(
        target_fraction=target,
        min_fraction=floor,
        max_fraction=target,
        basis="whole_mask",
        relative_tolerance=0.0,
        fallback_policy=("max_feasible_below_target" if target > floor else "exact"),
    )
    provenance = dict(case.provenance)
    provenance["source_mask_sha256"] = provenance["source_tissue_mask_sha256"]
    return CaseContext(
        case_id=case.case_id,
        # The tissue Planner receives the parser-owned normalized intent. The
        # doctor's original wording remains preserved in JointCaseContext and
        # the audit bundle, but this downstream visual stage cannot reinterpret
        # it into a different primitive.
        instruction=case.compiled_normalized_intent(),
        source_image_uri=case.source_image_uri,
        source_mask_uri=case.source_tissue_mask_uri,
        pathology_domain_id=case.pathology_domain_id,
        annotation_profile_id=case.annotation_profile_id,
        primitive_id=case.primitive_id,
        area_budget=budget,
        seed=case.seed,
        provenance=provenance,
        pixel_size_um=case.pixel_size_um,
    )


def _derive_infiltration_budget(
    scene,
) -> tuple[CellCountExtentBudget, dict[str, Any]]:
    """Derive a scale-aware cell budget for a contextual tumor interpretation.

    This policy is deterministic and auditable. It uses source complete-nucleus
    scale plus the amount of visible tumor/non-tumor interface; neither the
    Semantic Parser nor the visual Planner is allowed to invent a cell count.
    """

    complete_areas = [
        item.area_px
        for item in scene.cells.instances
        if item.completeness_status == "complete" and item.class_id == 1
    ]
    if not complete_areas:
        complete_areas = [
            item.area_px
            for item in scene.cells.instances
            if item.completeness_status == "complete"
        ]
    median_area = float(np.median(complete_areas)) if complete_areas else 64.0
    diameter_px = max(3.0, 2.0 * np.sqrt(median_area / np.pi))
    pair_contacts: dict[tuple[str, str], int] = {}
    for interface in scene.tissue.graph.interfaces:
        labels = {interface.source_label, interface.target_label}
        if "Tumor" not in labels or len(labels) < 2:
            continue
        pair = tuple(
            sorted(
                (
                    interface.source_component_id,
                    interface.target_component_id,
                )
            )
        )
        pair_contacts[pair] = max(
            pair_contacts.get(pair, 0), int(interface.contact_pixels)
        )
    interface_pixels = sum(pair_contacts.values())
    estimated_slots = int(
        np.ceil(interface_pixels / max(1.0, 6.0 * diameter_px))
    )
    target = int(np.clip(estimated_slots, 4, 24))
    minimum = max(1, int(np.floor(target * 0.5)))
    maximum = max(target, min(32, int(np.ceil(target * 1.5))))
    maximum_extent = int(np.clip(round(5.0 * diameter_px), 24, 64))
    budget = CellCountExtentBudget(
        target_delta_count=target,
        min_delta_count=minimum,
        max_delta_count=maximum,
        maximum_extent_px=maximum_extent,
        interface_min_px=2,
        interface_max_px=maximum_extent,
    )
    return budget, {
        "policy_id": "contextual-infiltration-budget-v1",
        "authority": "deterministic_source_scene",
        "complete_reference_count": len(complete_areas),
        "median_complete_nucleus_area_px": median_area,
        "estimated_nucleus_diameter_px": diameter_px,
        "unique_tumor_interface_pixels": interface_pixels,
        "budget": budget.__dict__,
    }


def _derive_local_population_budget(
    scene,
    *,
    primitive_id: str,
    semantic_intent: Mapping[str, Any],
    host_tissue_labels: tuple[str, ...] = (),
) -> tuple[CellCountExtentBudget, dict[str, Any]]:
    """Compile a source-calibrated cell-only budget before candidates exist.

    This is deliberately based on the shared scene instance authority and its
    component population zones.  It does not use a fixed global count, and it
    never lets the Planner invent one.  The downstream feasibility compiler
    may still reject a case when exact removal or packing capacity is lower.
    """

    resolved = semantic_intent.get("resolved_cell_class_ids", ())
    if not isinstance(resolved, (list, tuple)):
        raise JointContractError(
            "cell-only budget requires observation-profile class resolution"
        )
    class_ids = tuple(sorted({int(value) for value in resolved}))
    abundance = primitive_id.startswith("cell-type-abundance-")
    if abundance and len(class_ids) != 1:
        raise JointContractError(
            "cell abundance budget requires exactly one observable class"
        )

    component_labels = {
        item.component_id: item.label
        for item in scene.tissue.graph.components
    }
    host_labels = set(host_tissue_labels)
    component_zones = [
        zone
        for zone in scene.population.zones
        if zone.zone_kind == "component"
        and zone.area_px > 0
        and (
            not host_labels
            or component_labels.get(zone.tissue_component_id) in host_labels
        )
    ]
    if not component_zones:
        raise JointContractError(
            "cell-only budget has no legal host population zone"
        )
    if class_ids:
        zone_counts = [
            sum(int(zone.class_counts.get(class_id, 0)) for class_id in class_ids)
            for zone in component_zones
        ]
    else:
        zone_counts = [int(zone.nucleus_count) for zone in component_zones]
    complete_areas = [
        int(item.area_px)
        for item in scene.cells.instances
        if item.completeness_status == "complete"
        and not item.quality_flags
        and (not class_ids or item.class_id in class_ids)
    ]
    if not complete_areas:
        raise JointContractError(
            "cell-only budget has no complete source instance authority"
        )
    median_area = float(np.median(complete_areas))
    diameter_px = max(3.0, 2.0 * np.sqrt(median_area / np.pi))
    increase = primitive_id.endswith("increase-v1")
    complete_by_zone = {
        zone.zone_id: sum(
            1
            for item in scene.cells.instances
            if item.tissue_component_id == zone.tissue_component_id
            and item.completeness_status == "complete"
            and not item.quality_flags
            and (not class_ids or item.class_id in class_ids)
        )
        for zone in component_zones
    }
    coarse_capacity_by_zone = {}
    for zone in component_zones:
        zone_mask = scene.population_zone_masks[zone.zone_id]
        free_pixels = int(
            np.count_nonzero(
                np.asarray(zone_mask, dtype=bool)
                & (np.asarray(scene.source_nuclei) == 0)
            )
        )
        coarse_capacity_by_zone[zone.zone_id] = int(
            np.floor(free_pixels / max(1.0, median_area * 1.25))
        )
    if increase:
        # Exact complete-footprint packing, not free-pixel area, is the
        # execution authority.  Screen only the eight most promising host
        # components to keep preflight bounded, then certify real source-shape
        # placements against their retained nuclei and full containment.
        screened = sorted(
            component_zones,
            key=lambda zone: (
                -coarse_capacity_by_zone[zone.zone_id],
                -zone_counts[component_zones.index(zone)],
                -zone.area_px,
                zone.zone_id,
            ),
        )[:8]
        instance_metadata = {
            item.instance_id: item for item in scene.cells.instances
        }
        shape_library_by_class = {}
        shape_class_ids = class_ids or tuple(
            sorted(
                {
                    item.class_id
                    for item in scene.cells.instances
                    if item.completeness_status == "complete"
                    and not item.quality_flags
                }
            )
        )
        for class_id in shape_class_ids:
            shapes, _rejected = build_reference_shape_library(
                scene, class_id=class_id
            )
            shape_library_by_class[class_id] = tuple(shapes)
        packing_by_zone = {}
        for zone in screened:
            zone_label = component_labels.get(zone.tissue_component_id)
            references_by_class = {
                class_id: tuple(
                    shape
                    for shape in shapes
                    if component_labels.get(
                        instance_metadata[shape.instance_id].tissue_component_id
                    )
                    == zone_label
                )
                for class_id, shapes in shape_library_by_class.items()
            }
            references_by_class = {
                class_id: shapes
                for class_id, shapes in references_by_class.items()
                if shapes
            }
            request = min(
                40,
                max(2, coarse_capacity_by_zone[zone.zone_id]),
            )
            class_weights = {
                int(class_id): float(zone.class_counts.get(class_id, 0))
                for class_id in references_by_class
            }
            if not any(class_weights.values()):
                class_weights = {
                    int(class_id): 1.0 for class_id in references_by_class
                }
            zone_mask = np.asarray(
                scene.population_zone_masks[zone.zone_id], dtype=bool
            )
            packing_by_zone[zone.zone_id] = certify_complete_footprint_packing(
                source_nuclei=scene.source_nuclei,
                erased_footprint=np.zeros_like(zone_mask, dtype=bool),
                center_region=zone_mask,
                valid_footprint_region=zone_mask,
                references_by_class=references_by_class,
                requested_count=request,
                class_request_weights=class_weights,
                allow_finite_count_fallback=False,
            )
        selected_zone = max(
            screened,
            key=lambda zone: (
                packing_by_zone[zone.zone_id].placed_count,
                zone_counts[component_zones.index(zone)],
                zone.area_px,
                zone.zone_id,
            ),
        )
        local_authority_count = zone_counts[component_zones.index(selected_zone)]
        executable_capacity = packing_by_zone[selected_zone.zone_id].placed_count
        packing_certificates = {
            zone_id: certificate.to_metadata()
            for zone_id, certificate in sorted(packing_by_zone.items())
        }
    else:
        selected_zone = max(
            component_zones,
            key=lambda zone: (
                complete_by_zone[zone.zone_id],
                zone_counts[component_zones.index(zone)],
                zone.area_px,
                zone.zone_id,
            ),
        )
        local_authority_count = complete_by_zone[selected_zone.zone_id]
        executable_capacity = local_authority_count
        packing_certificates = {}
    if increase:
        if executable_capacity < 2:
            raise JointContractError(
                "cell increase has fewer than two conservative complete-shape slots"
            )
        source_scaled = max(2, round(max(local_authority_count, 10) * 0.20))
        target = int(min(32, executable_capacity, source_scaled))
    else:
        if local_authority_count < 6:
            raise JointContractError(
                "cell decrease has fewer than six observable local source instances"
            )
        target = int(
            np.clip(
                round(local_authority_count * 0.20),
                2,
                min(32, max(2, int(np.floor(local_authority_count * 0.35)))),
            )
        )
    minimum = max(1, int(np.floor(target * 0.60)))
    maximum = max(target, min(40, int(np.ceil(target * 1.35))))
    maximum_extent = int(np.clip(round(6.0 * diameter_px), 32, 48))
    budget = CellCountExtentBudget(
        target_delta_count=target,
        min_delta_count=minimum,
        max_delta_count=maximum,
        maximum_extent_px=maximum_extent,
        interface_min_px=0,
        interface_max_px=maximum_extent,
    )
    return budget, {
        "policy_id": "scene-calibrated-local-population-budget-v1",
        "authority": "shared_scene_instance_authority",
        "primitive_id": primitive_id,
        "resolved_cell_class_ids": list(class_ids),
        "selected_population_zone_id": selected_zone.zone_id,
        "selected_tissue_component_id": selected_zone.tissue_component_id,
        "selected_tissue_label": component_labels.get(
            selected_zone.tissue_component_id
        ),
        "selected_zone_source_count": local_authority_count,
        "selected_zone_executable_capacity": executable_capacity,
        "zone_complete_source_counts": dict(sorted(complete_by_zone.items())),
        "zone_coarse_add_capacities": dict(
            sorted(coarse_capacity_by_zone.items())
        ),
        "zone_exact_packing_certificates": packing_certificates,
        "complete_reference_count": len(complete_areas),
        "median_complete_nucleus_area_px": median_area,
        "estimated_nucleus_diameter_px": diameter_px,
        "target_fraction_of_local_source": 0.20,
        "maximum_decrease_fraction_of_local_source": 0.35,
        "budget": budget.__dict__,
    }


def _validate_digests(case: JointCaseContext) -> None:
    expected = {
        case.source_image_uri: case.provenance["source_image_sha256"],
        case.source_tissue_mask_uri: case.provenance["source_tissue_mask_sha256"],
        case.source_nuclei_mask_uri: case.provenance["source_nuclei_mask_sha256"],
    }
    auxiliary_digests = case.provenance.get("auxiliary_structure_sha256", {})
    expected.update(
        {
            path: auxiliary_digests[structure_id]
            for structure_id, path in case.auxiliary_structure_uris.items()
        }
    )
    if case.source_nuclei_instances_uri:
        instance_digest = case.provenance.get("source_nuclei_instances_sha256")
        if not instance_digest:
            raise JointContractError(
                "native nucleus instance provenance digest is missing"
            )
        expected[case.source_nuclei_instances_uri] = instance_digest
    mismatches = [
        path for path, digest in expected.items() if sha256_file(path) != digest
    ]
    if mismatches:
        raise JointContractError("source digest mismatch: " + ", ".join(mismatches))


def _validate_dimensions(image_path: str, tissue_shape, nuclei_shape) -> None:
    if tissue_shape != nuclei_shape:
        raise JointContractError("source tissue and nuclei masks are not aligned")
    with Image.open(image_path) as image:
        if image.size != (tissue_shape[1], tissue_shape[0]):
            raise JointContractError("source H&E and masks are not aligned")
