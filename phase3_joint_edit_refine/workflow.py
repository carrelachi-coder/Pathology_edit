"""Independent joint Architecture-B orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
)
from phase3_mask_edit_refine.skills import SkillRepository as MaskSkillRepository
from phase3_mask_edit_refine.visualization import save_planner_panels

from .audit import JointAuditWriter
from .budget import JointFeasibilitySolver
from .cell_layouts import SpatialRanker, generate_cell_layouts
from .cell_programs import CellToolProgramCompiler
from .critic import JointCritic
from .feasibility import (
    augment_tissue_scene_with_nuclei_preflight,
    bind_joint_plan_to_nuclei_preflight,
    build_joint_nuclei_preflight,
)
from .gates import JointGateContext, JointGateRegistry
from .handoff import write_generation_handoff
from .ledger import build_joint_candidate
from .models import (
    JointCaseContext,
    JointCondition,
    JointContractError,
    JointWorkflowResult,
)
from .mature_probnet_adapter import MatureProbNetCellExecutor
from .nuclei import load_nuclei_mask
from .planner import JointPlanner
from .scene import build_joint_scene_analysis
from .skills.repository import JointSkillRepository
from .tissue_execution import execute_gate_aware_tissue_candidates


@dataclass(frozen=True)
class JointWorkflowConfig:
    production: bool = False
    maximum_tissue_candidates: int = 4
    cell_layouts_per_tissue: int = 3
    critic_confidence_threshold: float = 0.70
    require_mature_probnet_in_production: bool = True


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
        self.cell_program_compiler = CellToolProgramCompiler()

    def run(self, case: JointCaseContext, *, output_root: str | Path) -> JointWorkflowResult:
        audit = JointAuditWriter(output_root, case_id=case.case_id)
        plan = None
        critic_result = None
        joint_reports = ()
        reasons: list[str] = []
        usage: dict = {"mechanism_selection": {}, "tissue_planner": {}, "joint_planner": {}, "critic": {}}
        candidates = ()
        try:
            case.validate_local_inputs()
            _validate_digests(case)
            source_tissue = load_id_mask(case.source_tissue_mask_uri)
            source_nuclei = load_nuclei_mask(case.source_nuclei_mask_uri)
            _validate_dimensions(case.source_image_uri, source_tissue.shape, source_nuclei.shape)
            schema = self.mask_skills.annotation_schema(case.annotation_profile_id)
            scene = build_joint_scene_analysis(
                source_tissue,
                source_nuclei,
                schema=schema,
                pixel_size_um=case.pixel_size_um,
                nuclei_instances_path=case.source_nuclei_instances_uri,
                auxiliary_structure_paths=case.auxiliary_structure_uris,
            )
            audit.write_json("case_context.json", case.to_metadata())
            audit.write_json("joint_scene_graph.json", scene.to_metadata())
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
            mechanisms, mechanism_rejections = self.joint_skills.eligible_mechanisms_for_case(
                case=case,
                available_checker_ids=set(self.joint_gates.available_checker_ids),
                production=self.config.production,
            )
            if not mechanisms:
                raise JointContractError(
                    "no joint mechanism survives the four-axis capability intersection: "
                    + "; ".join(
                        f"{key}={value}"
                        for key, value in sorted(mechanism_rejections.items())
                    )
                )
            mechanism_id, selection_usage = self.joint_planner.select_mechanism(
                case=case,
                scene=scene,
                mechanisms=mechanisms,
                image_paths=planner_images,
            )
            usage["mechanism_selection"] = selection_usage
            bundle = self.joint_skills.compose(
                case=case,
                mechanism_id=mechanism_id,
                available_checker_ids=set(self.joint_gates.available_checker_ids),
                production=self.config.production,
            )
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
            allocation = self.budget_solver.allocate(
                shape=source_tissue.shape,
                budget=case.joint_area_budget,
                bundle=bundle,
            )
            tissue_bundle = self.mask_skills.compose(
                pathology_domain_id=case.pathology_domain_id,
                annotation_profile_id=case.annotation_profile_id,
                primitive_id=case.primitive_id,
                production=self.config.production,
                available_checker_ids=self.tissue_gates.available_checker_ids,
            )
            nuclei_preflight = build_joint_nuclei_preflight(
                case=case,
                source_tissue=source_tissue,
                schema=schema,
                scene=scene,
                tissue_bundle=tissue_bundle,
                joint_bundle=bundle,
                allocation=allocation,
            )
            audit.write_json(
                "joint_nuclei_preflight.json",
                nuclei_preflight.to_metadata(),
            )
            if nuclei_preflight.required_auxiliary_missing:
                raise JointContractError(
                    "joint preflight lacks required auxiliary maps: "
                    + ", ".join(nuclei_preflight.required_auxiliary_missing)
                )
            if nuclei_preflight.required_provenance_missing:
                raise JointContractError(
                    "joint preflight lacks required annotation provenance: "
                    + ", ".join(nuclei_preflight.required_provenance_missing)
                )
            if not nuclei_preflight.feasible_interface_ids:
                raise JointContractError(
                    "joint preflight found no interface with tissue, complete-instance, "
                    "reference-shape and halo capacity"
                )
            tissue_scene = augment_tissue_scene_with_nuclei_preflight(
                scene.tissue,
                nuclei_preflight,
            )
            audit.write_inputs(case=case, scene_metadata=scene.to_metadata(), skill_metadata={**bundle.to_metadata(), "budget_allocation": allocation.to_metadata(), "tissue_skill_bundle": tissue_bundle.to_metadata()})
            budget_revisions = []
            tissue_pass_usage = []
            # A provisional tissue boundary may intersect a complete semantic or
            # native nucleus whose footprint extends beyond T.  Since v1 forbids
            # partial nucleus edits, that extension necessarily belongs to C and
            # therefore J.  Re-broker once using the observed whole-instance
            # closure cost instead of silently overshooting the joint target.
            for budget_pass in range(2):
                tissue_case = _as_tissue_case(
                    case,
                    allocation=allocation,
                    shape=source_tissue.shape,
                )
                if hasattr(self.tissue_planner, "create_joint_tissue_plan"):
                    raw_tissue_plan, tissue_usage = self.tissue_planner.create_joint_tissue_plan(
                        case=tissue_case,
                        scene=tissue_scene,
                        bundle=tissue_bundle,
                        joint_bundle=bundle,
                        image_paths=planner_images,
                        nuclei_preflight=nuclei_preflight,
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
                tissue_pass_usage.append(
                    {
                        "pass": budget_pass + 1,
                        **tissue_usage,
                        "compiler": compiler_usage,
                        "budget_allocation": allocation.to_metadata(),
                    }
                )
                execution_batch = execute_gate_aware_tissue_candidates(
                    source_tissue,
                    schema=schema,
                    tissue_scene=tissue_scene,
                    joint_scene=scene,
                    tissue_case=tissue_case,
                    tissue_plan=tissue_plan,
                    tissue_bundle=tissue_bundle,
                    joint_bundle=bundle,
                    nuclei_preflight=nuclei_preflight,
                    gates=self.tissue_gates,
                    seed=case.seed,
                )
                tissue_candidates = execution_batch.all_candidates
                tissue_reports = execution_batch.tissue_gate_reports
                audit.write_json(
                    f"tissue_gate_reports_pass_{budget_pass + 1}.json",
                    [item.to_metadata() for item in tissue_reports],
                )
                audit.write_json(
                    f"tissue_execution_contract_pass_{budget_pass + 1}.json",
                    execution_batch.to_metadata(),
                )
                reports_by_id = {
                    item.candidate_id: item for item in tissue_reports
                }
                passing_tissue = list(execution_batch.certified_candidates)[
                    : self.config.maximum_tissue_candidates
                ]
                if not passing_tissue:
                    raise JointContractError(
                        "no tissue candidate passed both the inherited hard gates "
                        "and candidate-local nuclei feasibility"
                    )
                desired_min, desired_max = case.joint_area_budget.desired_interval_pixels(
                    source_tissue.shape
                )
                predicted = [
                    int(item.change_region.sum())
                    + _complete_instance_extension_pixels(
                        item.change_region,
                        scene=scene,
                        protected_ids={
                            cell.instance_id
                            for cell in scene.cells.instances
                            if cell.instance_id
                            in set(nuclei_preflight.protected_instance_ids)
                        },
                    )
                    for item in passing_tissue
                ]
                if (
                    budget_pass == 0
                    and predicted
                    and min(predicted) > desired_max
                ):
                    closure = int(
                        np.median(
                            [
                                _complete_instance_extension_pixels(
                                    item.change_region,
                                    scene=scene,
                                    protected_ids={
                                        cell.instance_id
                                        for cell in scene.cells.instances
                                        if cell.instance_id
                                        in set(nuclei_preflight.protected_instance_ids)
                                    },
                                )
                                for item in passing_tissue
                            ]
                        )
                    )
                    revised = self.budget_solver.reserve_complete_instances(
                        allocation,
                        reserve_pixels=closure,
                    )
                    if revised.tissue_target_pixels < allocation.tissue_target_pixels:
                        budget_revisions.append(
                            {
                                "reason": "whole_instance_union_closure",
                                "desired_joint_interval": [desired_min, desired_max],
                                "provisional_joint_pixels": predicted,
                                "before": allocation.to_metadata(),
                                "after": revised.to_metadata(),
                            }
                        )
                        allocation = revised
                        continue
                break
            usage["tissue_planner"] = {
                "passes": tissue_pass_usage,
                "budget_revisions": budget_revisions,
            }
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
            usage["joint_planner"] = joint_usage
            audit.write_json("joint_edit_plan.json", plan.to_metadata())

            joint_candidates = []
            prohibited = set(bundle.annotation_profile.prohibit_generation_support_fine_ids)
            generation_allowed = ~np.isin(source_tissue, tuple(prohibited))
            for tissue_candidate in passing_tissue:
                compiled_cell_program = self.cell_program_compiler.compile(
                    case=case,
                    schema=schema,
                    scene=scene,
                    plan=plan,
                    bundle=bundle,
                    tissue_candidate=tissue_candidate,
                )
                if (
                    self.cell_executor is not None
                    and self.cell_executor.supports(compiled_cell_program)
                ):
                    layouts = self.cell_executor.execute(
                        program=compiled_cell_program,
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
                    layouts = generate_cell_layouts(
                        source_tissue=source_tissue,
                        source_nuclei=source_nuclei,
                        tissue_candidate=tissue_candidate,
                        schema=schema,
                        scene=scene,
                        plan=plan,
                        bundle=bundle,
                        allocation=allocation,
                        compiled_program=compiled_cell_program,
                        seed=case.seed,
                        ranker=self.ranker,
                        variants=self.config.cell_layouts_per_tissue,
                    )
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
                    candidate_id = f"joint-{tissue_candidate.candidate_id}-{layout.cell_candidate_id}"
                    trace = {
                        **layout.trace,
                        "mechanism_id": mechanism_id,
                        "skill_version": bundle.mechanism.version,
                        "tissue_tool_trace": tissue_candidate.tool_trace,
                        "budget_allocation": allocation.to_metadata(),
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
                            source_instance_masks=scene.instance_masks,
                            source_instance_classes={item.instance_id: item.class_id for item in scene.cells.instances},
                            tool_trace=trace,
                        )
                    )
            candidates = tuple(joint_candidates[:12])
            batch_max_joint = max((item.ledger.joint_pixels for item in candidates), default=0)
            for candidate in candidates:
                candidate.tool_trace["batch_max_safe_joint_pixels"] = batch_max_joint
            tissue_report_for_joint = reports_by_id
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
                        tissue_gate_report=tissue_report_for_joint[candidate.tissue_candidate_id],
                    )
                )
                for candidate in candidates
            )
            audit.write_candidates(candidates)
            audit.write_json("joint_gate_reports.json", [item.to_metadata() for item in joint_reports])
            review_board = audit.write_review_board(
                source_image_path=case.source_image_uri,
                source_tissue=source_tissue,
                source_nuclei=source_nuclei,
                candidates=candidates,
            )
            passing_joint = [candidate for candidate in candidates if next(item.passed for item in joint_reports if item.candidate_id == candidate.candidate_id)]
            if not passing_joint:
                raise JointContractError("no paired tissue--cell candidate passed all joint gates")
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
                    audit=audit, case=case, plan=plan, reports=joint_reports,
                    critic=critic_result, status="review_required", reasons=reasons,
                    selected=None, condition=None, usage=usage,
                )
            ranking = critic_result.rankings[0]
            if ranking.confidence < self.config.critic_confidence_threshold or ranking.veto_reasons:
                reasons.append("joint_critic_low_confidence_or_veto")
                return self._finish(
                    audit=audit, case=case, plan=plan, reports=joint_reports,
                    critic=critic_result, status="abstained", reasons=reasons,
                    selected=None, condition=None, usage=usage,
                )
            selected = next(item for item in passing_joint if item.candidate_id == ranking.candidate_id)
            condition = JointCondition(
                case_id=case.case_id,
                candidate_id=selected.candidate_id,
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
            )
            audit.paths.update({"handoff_" + key: value for key, value in handoff_paths.items()})
            return self._finish(
                audit=audit, case=case, plan=plan, reports=joint_reports,
                critic=critic_result,
                status=("selected" if self.config.production else "selected_research"),
                reasons=(), selected=selected.candidate_id, condition=condition, usage=usage,
            )
        except Exception as exc:  # noqa: BLE001 - orchestration must fail closed
            reasons.append(f"{type(exc).__name__}: {exc}")
            if candidates:
                audit.write_candidates(candidates)
            return self._finish(
                audit=audit, case=case, plan=plan, reports=joint_reports,
                critic=critic_result, status="abstained", reasons=reasons,
                selected=None, condition=None, usage=usage,
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
        interface_id = plan.cell_plan.interface_ids[0]
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
        program = self.cell_program_compiler.compile(
            case=case,
            schema=schema,
            scene=scene,
            plan=plan,
            bundle=bundle,
            tissue_candidate=preserved_tissue,
        )
        # The mature pipeline owns target-population regeneration. Structured
        # additions are deterministic templates whose anchors may be ranked by
        # the frozen ProbNet but whose count and geometry remain skill-owned.
        layouts = generate_cell_layouts(
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            tissue_candidate=preserved_tissue,
            schema=schema,
            scene=scene,
            plan=plan,
            bundle=bundle,
            allocation=None,
            compiled_program=program,
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
        prohibited = set(
            bundle.annotation_profile.prohibit_generation_support_fine_ids
        )
        generation_allowed = ~np.isin(source_tissue, tuple(prohibited))
        candidates = []
        for layout in layouts:
            trace = {
                **layout.trace,
                "mechanism_id": mechanism_id,
                "skill_version": bundle.mechanism.version,
                "tissue_tool_trace": preserved_tissue.tool_trace,
                "budget_mode": "count_extent",
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
                    source_instance_masks=scene.instance_masks,
                    source_instance_classes={
                        item.instance_id: item.class_id
                        for item in scene.cells.instances
                    },
                    tool_trace=trace,
                )
            )
        candidates = tuple(candidates)
        for candidate in candidates:
            candidate.tool_trace["batch_max_safe_joint_pixels"] = max(
                item.ledger.joint_pixels for item in candidates
            )
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
                )
            )
            for candidate in candidates
        )
        audit.write_candidates(candidates)
        audit.write_json(
            "joint_gate_reports.json", [item.to_metadata() for item in reports]
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
        if ranking.confidence < self.config.critic_confidence_threshold or ranking.veto_reasons:
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

    def _finish(self, *, audit, case, plan, reports, critic, status, reasons, selected, condition, usage):
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


def _as_tissue_case(case: JointCaseContext, *, allocation, shape) -> CaseContext:
    total = int(np.prod(shape))
    target = allocation.tissue_target_pixels / max(1, total)
    floor = allocation.tissue_execution_floor_pixels / max(1, total)
    budget = AreaBudget(
        target_fraction=target,
        min_fraction=(floor if target > floor else target),
        max_fraction=target,
        basis="whole_mask",
        relative_tolerance=0.0,
        fallback_policy=("max_feasible_below_target" if target > floor else "exact"),
    )
    provenance = dict(case.provenance)
    provenance["source_mask_sha256"] = provenance["source_tissue_mask_sha256"]
    return CaseContext(
        case_id=case.case_id,
        instruction=case.instruction,
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
    mismatches = [path for path, digest in expected.items() if sha256_file(path) != digest]
    if mismatches:
        raise JointContractError("source digest mismatch: " + ", ".join(mismatches))


def _validate_dimensions(image_path: str, tissue_shape, nuclei_shape) -> None:
    if tissue_shape != nuclei_shape:
        raise JointContractError("source tissue and nuclei masks are not aligned")
    with Image.open(image_path) as image:
        if image.size != (tissue_shape[1], tissue_shape[0]):
            raise JointContractError("source H&E and masks are not aligned")
