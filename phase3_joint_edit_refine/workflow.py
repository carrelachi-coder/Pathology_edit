"""Independent joint Architecture-B orchestration."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from phase3_mask_edit_refine.agents import Planner, validate_edit_plan
from phase3_mask_edit_refine.evidence import load_id_mask, sha256_file
from phase3_mask_edit_refine.gates import GateRegistry
from phase3_mask_edit_refine.models import (
    AreaBudget,
    CandidateMask,
    CaseContext,
    GateReport,
    RefineContractError,
)
from phase3_mask_edit_refine.skills import (
    SkillRepository as MaskSkillRepository,
)
from phase3_mask_edit_refine.skills import (
    bind_active_bundle_to_case,
    validate_active_bundle_authority,
)

from .audit import JointAuditWriter
from .authority import (
    count_authoritative_complete_references,
    validate_mechanism_nucleus_authority,
)
from .auxiliary import materialize_profile_auxiliaries
from .budget import JointFeasibilitySolver
from .candidate_feasibility import CandidateFeasibilityCompiler
from .cell_layouts import (
    SpatialRanker,
    _prioritize_local_references,
    build_reference_shape_library,
    generate_cell_layouts,
)
from .cell_programs import CellToolProgramCompiler
from .clarification import (
    PlannerClarificationRequired,
    build_primitive_clarification_request,
    build_scenario_clarification_request,
    requires_budding_claim_downgrade,
    resolve_clarification_decision,
)
from .critic import JointCritic
from .executable_contract import (
    ExecutableJointContract,
    ExecutableJointContractCompiler,
)
from .feasibility import (
    augment_tissue_scene_with_nuclei_preflight,
    bind_joint_plan_to_nuclei_preflight,
    build_joint_nuclei_preflight,
    certify_compiled_cell_program_feasibility,
)
from .gates import (
    JointGateContext,
    JointGateRegistry,
    audit_cell_effect_foci,
)
from .handoff import write_generation_handoff
from .instance_authority import authority_trace, build_scene_instance_authority
from .invasive_architecture import (
    SPECIALIZED_ARCHITECTURE_PRIMITIVES,
    compile_joint_tissue_plan_with_witness,
)
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
from .planner import (
    CellPlanCandidateVeto,
    CellPlanSelectionHandle,
    CertifiedCellPlanPortfolio,
    CompilerOwnedDepletionAnchor,
    HeuristicJointPlanner,
    JointInterpretationOption,
    JointPlanner,
    _issue_cell_plan_portfolio,
)
from .planner_inputs import MaskPlannerArtifactRegistry
from .portfolio_authority import (
    build_cell_portfolio_authority_binding,
    build_tissue_portfolio_authority_binding,
    canonical_metadata_sha256,
)
from .reference_shapes import load_reference_shape_authority
from .scene import build_joint_scene_analysis
from .skills.execution_aliases import tissue_tool_primitive_id
from .skills.repository import JointSkillBundle, JointSkillRepository
from .spatial_contracts import (
    BREAST_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS,
    GLAS_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS,
    SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS,
    SMALL_CLUSTER_TARGET_FOCUS_COUNT,
    small_cluster_maximum_hotspot_span_px,
)
from .tissue_execution import execute_gate_aware_tissue_candidates
from .tissue_tools import (
    bind_tissue_plan_tool_program,
    compile_tissue_tool_program,
)


INFILTRATION_BUDGET_PRIMITIVES = frozenset(
    {
        "neoplastic-microinfiltration-increase-v1",
        "peritumoral-neoplastic-scatter-increase-v1",
        "peritumoral-small-cluster-increase-v1",
    }
)

PANDA_CELL_CAPACITY_FALLBACK_PRIMITIVES = frozenset(
    {
        "local-invasive-clearance-v1",
        "stroma-increase-v1",
        "invasive-tumor-footprint-decrease-v1",
        "cohesive-boundary-expansion-v1",
    }
)

PANDA_CELL_EFFECT_EXTENT_OVERRIDES = {
    "cell-type-abundance-increase-v1": (2.5, 1),
    "cell-type-abundance-decrease-v1": (1.5, 0),
    # Cellularity is one bounded mixed-population field.  It must span a
    # visible local region, but it is not three independent abundance foci.
    "cellularity-increase-v1": (3.0, 1),
    "cellularity-decrease-v1": (4.0, 0),
    "neoplastic-cell-abundance-increase-v1": (2.0, 1),
    "neoplastic-cell-abundance-decrease-v1": (1.5, 0),
    "peritumoral-neoplastic-scatter-increase-v1": (4.0, 3),
}


@dataclass(frozen=True)
class JointWorkflowConfig:
    production: bool = False
    maximum_tissue_candidates: int = 4
    cell_layouts_per_tissue: int = 3
    critic_confidence_threshold: float = 0.70
    require_mature_probnet_in_production: bool = True
    require_mature_probnet_for_target_population_regeneration: bool = False
    require_probnet_ranker_for_cell_addition: bool = False
    require_evaluation_input_bindings: bool = False
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
    tissue_feasibility_portfolio: Any | None


@dataclass(frozen=True)
class _CertifiedCellExecutionChoice:
    certificate: Any
    executable_contract: ExecutableJointContract
    preflight: Any


@dataclass(frozen=True)
class _CertifiedCellExecutionPortfolio:
    choices: tuple[_CertifiedCellExecutionChoice, ...]
    certificates: CertifiedCellPlanPortfolio


def _localized_focus_capacity_metrics(
    *,
    center_rows,
    center_cols,
    nominal_nucleus_diameter_px: float,
    minimum_effect_span_px: int,
    required_focus_count: int = SMALL_CLUSTER_TARGET_FOCUS_COUNT,
    minimum_anchor_separation_diameters: float = (
        SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS
    ),
    strict_breast_small_cluster: bool = False,
) -> tuple[int, float]:
    """Score whether packing witnesses contain a localized focus window."""

    centers = np.column_stack([center_rows, center_cols]).astype(float)
    if not len(centers):
        return 0, 0.0
    nominal = max(1.0, float(nominal_nucleus_diameter_px))
    maximum_span = small_cluster_maximum_hotspot_span_px(
        nominal,
        minimum_effect_span_px,
        compact_breast=strict_breast_small_cluster,
    )
    minimum_between = (
        max(0.0, float(minimum_anchor_separation_diameters)) * nominal
    )
    pairwise = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=2
    )
    capacity = 1
    for size in range(2, len(centers) + 1):
        if any(
            (
                float(np.max(pairwise[np.ix_(subset, subset)]))
                <= maximum_span + 1e-6
                and all(
                    pairwise[left, right] > minimum_between
                    for left, right in combinations(subset, 2)
                )
            )
            for subset in combinations(range(len(centers)), size)
        ):
            capacity = size
        else:
            break

    margins = []
    required = min(max(1, int(required_focus_count)), len(centers))
    for subset in combinations(range(len(centers)), required):
        span = float(np.max(pairwise[np.ix_(subset, subset)]))
        separated = all(
            pairwise[left, right] > minimum_between
            for left, right in combinations(subset, 2)
        )
        if (
            separated
            and float(minimum_effect_span_px) <= span <= maximum_span
        ):
            margins.append(
                min(
                    span - float(minimum_effect_span_px),
                    maximum_span - span,
                )
            )
    return capacity, (max(margins) if margins else 0.0)


def _select_cell_execution_choice(
    *,
    portfolio: _CertifiedCellExecutionPortfolio,
    plan: Any,
    planner_usage: Mapping[str, Any],
) -> _CertifiedCellExecutionChoice:
    """Resolve the exact certificate handle, never a value-equal plan."""

    handle = CellPlanSelectionHandle.from_metadata(
        planner_usage.get("selection_handle")
    )
    legacy_bindings = {
        "selected_candidate_id": handle.candidate_id,
        "selected_tool_program_id": handle.selected_tool_program_id,
        "compiler_certificate_sha256": handle.compiler_certificate_sha256,
        "executable_contract_id": handle.executable_contract_id,
    }
    if any(
        key in planner_usage and planner_usage.get(key) != expected
        for key, expected in legacy_bindings.items()
    ):
        raise JointContractError(
            "cell Planner usage is detached from the typed selection handle"
        )
    selected = next(
        (
            item
            for item in portfolio.choices
            if item.certificate.candidate_id == handle.candidate_id
        ),
        None,
    )
    if selected is None:
        raise JointContractError(
            "cell Planner selected an unknown or vetoed certificate ID"
        )
    handle.validate_candidate(selected.certificate, plan=plan)
    if (
        handle.selected_tool_program_id
        != selected.executable_contract.execution_program_id
        or handle.executable_contract_id
        != selected.executable_contract.contract_id
    ):
        raise JointContractError(
            "cell Planner selection is detached from the executable contract"
        )
    return selected


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
        self.mask_skills.require_official_execution_catalog()
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
            if self.config.require_evaluation_input_bindings:
                mechanism_id = case.provenance.get("joint_mechanism_id")
                if not isinstance(mechanism_id, str) or mechanism_id in {
                    "",
                    "__abstain__",
                }:
                    raise JointContractError(
                        "evaluation input requires an explicit joint_mechanism_id binding"
                    )
                if case.provenance.get("joint_primitive_id") != case.primitive_id:
                    raise JointContractError(
                        "evaluation input requires a matching joint_primitive_id binding"
                    )
                mechanism = self.joint_skills.mechanisms.get(mechanism_id)
                if mechanism is None or case.primitive_id not in mechanism.supported_primitives:
                    raise JointContractError(
                        "evaluation input mechanism binding does not support the requested primitive"
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
                source_image=np.asarray(
                    Image.open(case.source_image_uri).convert("RGB"),
                    dtype=np.uint8,
                ),
                source_nuclei=source_nuclei,
                output_dir=audit.case_dir / "auxiliary_structures",
            )
            if produced_auxiliaries:
                audit.write_json(
                    "auxiliary_producer_report.json",
                    [item.to_metadata() for item in produced_auxiliaries],
                )
            case.validate_local_inputs()
            _validate_digests(case)
            if self.config.require_evaluation_input_bindings:
                mechanism_id = case.provenance["joint_mechanism_id"]
                mechanism = self.joint_skills.mechanisms[mechanism_id]
                missing_auxiliary = sorted(
                    set(mechanism.representability.required_auxiliary_structures)
                    - set(case.auxiliary_structure_uris)
                )
                if missing_auxiliary:
                    raise JointContractError(
                        "evaluation input lacks required auxiliary authority: "
                        + ", ".join(missing_auxiliary)
                    )
            schema = self.mask_skills.annotation_schema(case.annotation_profile_id)
            reference_shape_authority = None
            if (
                case.cell_count_extent_budget is not None
                and self.cell_executor is not None
                and hasattr(self.cell_executor, "config")
            ):
                resolved_classes = tuple(
                    sorted(
                        {
                            int(value)
                            for value in case.semantic_intent.get(
                                "resolved_cell_class_ids", ()
                            )
                        }
                    )
                )
                if not resolved_classes and (
                    "neoplastic" in case.primitive_id
                    or "peritumoral" in case.primitive_id
                ):
                    resolved_classes = (1,)
                if not resolved_classes:
                    resolved_classes = (1, 2, 3, 4, 5)
                reference_shape_authority = (
                    load_reference_shape_authority(
                        self.cell_executor.config.instance_library,
                        dataset_name=(
                            self.cell_executor.config.dataset_name
                        ),
                        class_ids=resolved_classes,
                    )
                )
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
                reference_shape_authority=reference_shape_authority,
            )
            audit.write_json("case_context.json", case.to_metadata())
            audit.write_json("joint_scene_graph.json", scene.to_metadata())
            audit.write_json(
                "source_instance_authority.json",
                build_scene_instance_authority(scene, source_nuclei),
            )
            planner_artifacts = MaskPlannerArtifactRegistry.issue(
                case=case,
                pipeline_owned_root=audit.case_dir,
                source_tissue=source_tissue,
                source_nuclei=source_nuclei,
                schema=schema,
                pixel_size_um=case.pixel_size_um,
            )
            audit.write_json(
                "mask_planner_artifact_registry.json",
                planner_artifacts.to_metadata(),
            )
            planner_images = planner_artifacts.source_image_paths
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
            requested_mechanism = case.provenance.get("joint_mechanism_id")
            requested_primitive = case.provenance.get("joint_primitive_id")
            if isinstance(requested_mechanism, str) and isinstance(
                requested_primitive, str
            ):
                requested_pair = (
                    f"{requested_primitive}::{requested_mechanism}"
                )
                prepared_pairs = {
                    (
                        item.option.primitive_id,
                        item.option.mechanism.mechanism_id,
                    )
                    for item in prepared.values()
                }
                if (
                    (requested_primitive, requested_mechanism)
                    not in prepared_pairs
                ):
                    matching_rejections = [
                        reason
                        for option_id, reason in interpretation_rejections.items()
                        if option_id == requested_pair
                        or option_id.endswith("::" + requested_pair)
                    ]
                    reason = (
                        "; ".join(matching_rejections)
                        if matching_rejections
                        else "the bound pair is absent from the certified interpretation portfolio"
                    )
                    raise JointContractError(
                        "explicit provenance-bound primitive/mechanism pair "
                        "did not survive deterministic preflight: "
                        f"{requested_pair}={reason}"
                    )
            if case.semantic_intent.get("scenario") == "post_treatment_change":
                scenario_order = (
                    "treatment_response",
                    "post_treatment_progression",
                    "residual_disease",
                )
                prepared_scenarios = {
                    str(item.case.semantic_intent.get("scenario") or "")
                    for item in prepared.values()
                }
                available_scenarios = tuple(
                    scenario
                    for scenario in scenario_order
                    if scenario in prepared_scenarios
                )
                if not available_scenarios:
                    raise JointContractError(
                        "no treatment direction has an executable primitive after preflight"
                    )
                request = build_scenario_clarification_request(
                    case_id=case.case_id,
                    instruction=case.instruction,
                    input_digests={
                        key: str(value)
                        for key, value in case.provenance.items()
                        if key.startswith("source_")
                        and key.endswith("_sha256")
                        and value
                    },
                    knowledge_context={
                        "pathology_domain_id": case.pathology_domain_id,
                        "annotation_profile_id": case.annotation_profile_id,
                        "cell_observation_profile_id": case.cell_observation_profile_id,
                        "cell_population_profile_id": case.cell_population_profile_id,
                    },
                    why_required=(
                        "The post-treatment direction is unspecified. These options "
                        "survived the active profile, mechanism, auxiliary-authority, "
                        "and capacity preflight."
                    ),
                    available_scenarios=available_scenarios,
                ).to_metadata()
                audit.write_json("clarification_request.json", request)
                return self._finish(
                    audit=audit,
                    case=case,
                    plan=None,
                    reports=(),
                    critic=None,
                    status="clarification_required",
                    reasons=(),
                    selected=None,
                    condition=None,
                    usage=usage,
                    clarification_request=request,
                )
            clarification_resolution = resolve_clarification_decision(
                case=case,
                prepared_options=tuple(
                    item.option for item in prepared.values()
                ),
            )
            if (
                clarification_resolution is None
                and requires_budding_claim_downgrade(case.instruction)
                and any(
                    item.option.primitive_id
                    == "peritumoral-small-cluster-increase-v1"
                    for item in prepared.values()
                )
            ):
                request = build_primitive_clarification_request(
                    case=case,
                    prepared_options=tuple(
                        item.option for item in prepared.values()
                    ),
                    why_required=(
                        "the requested diagnostic budding term exceeds mask authority; "
                        "execution requires explicit acceptance of a non-diagnostic "
                        "synthetic one-to-four-cell small-cluster representation"
                    ),
                    primitive_ids=(
                        "peritumoral-small-cluster-increase-v1",
                    ),
                ).to_metadata()
                audit.write_json("clarification_request.json", request)
                return self._finish(
                    audit=audit,
                    case=case,
                    plan=None,
                    reports=(),
                    critic=None,
                    status="clarification_required",
                    reasons=(),
                    selected=None,
                    condition=None,
                    usage=usage,
                    clarification_request=request,
                )
            clarification_usage = None
            if clarification_resolution is not None:
                selected_primitive, clarification_usage = clarification_resolution
                prepared = {
                    option_id: item
                    for option_id, item in prepared.items()
                    if item.option.primitive_id == selected_primitive
                }
                if not prepared:
                    raise JointContractError(
                        "clarification-selected primitive has no executable interpretation"
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
                        artifact_registry=planner_artifacts,
                    )
                )
            except PlannerClarificationRequired as exc:
                request = build_primitive_clarification_request(
                    case=case,
                    prepared_options=tuple(
                        item.option for item in prepared.values()
                    ),
                    why_required=exc.reason,
                    primitive_ids=exc.primitive_ids,
                )
                request_metadata = request.to_metadata()
                audit.write_json(
                    "clarification_request.json", request_metadata
                )
                audit.write_json(
                    "semantic_resolution.json",
                    {
                        **resolution_base,
                        "status": "clarification_required",
                        "selected_option_id": None,
                        "selection": None,
                        "clarification_request": request_metadata,
                    },
                )
                usage["mechanism_selection"] = {
                    "provider": self.joint_planner.name,
                    "status": "clarification_required",
                }
                return self._finish(
                    audit=audit,
                    case=case,
                    plan=None,
                    reports=(),
                    critic=None,
                    status="clarification_required",
                    reasons=(),
                    selected=None,
                    condition=None,
                    usage=usage,
                    clarification_request=request_metadata,
                )
            except JointContractError as exc:
                audit.write_json(
                    "semantic_resolution.json",
                    {
                        **resolution_base,
                        "status": "mask_graph_resolution_abstained",
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
            if clarification_usage is not None:
                selection_usage = {
                    **selection_usage,
                    "user_clarification": clarification_usage,
                }
                selection_record["user_clarification"] = clarification_usage
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
                    planner_artifacts=planner_artifacts,
                    usage=usage,
                )
            allocation = selected.allocation
            tissue_bundle = selected.tissue_bundle
            nuclei_preflight = selected.nuclei_preflight
            tissue_feasibility_portfolio = selected.tissue_feasibility_portfolio
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
            if tissue_feasibility_portfolio:
                audit.write_json(
                    "candidate_feasibility_portfolio.json",
                    tissue_feasibility_portfolio.to_metadata(),
                )
            tissue_scene = augment_tissue_scene_with_nuclei_preflight(
                scene.tissue,
                nuclei_preflight,
                auxiliary_structure_masks=scene.auxiliary_structure_masks,
                required_auxiliary_structure_ids=(
                    bundle.mechanism.representability.protected_auxiliary_structures
                ),
                receiving_auxiliary_structure_ids=(
                    bundle.mechanism.representability.receiving_auxiliary_structures
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
            last_deterministic_feedback_signature: str | None = None
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
            revoked_tissue_candidate_ids: set[str] = set()
            panda_cell_capacity_floor_fallback_used = False

            def reject_repeated_deterministic_feedback(feedback: Mapping[str, Any]) -> None:
                """Stop when another pass would execute the same failed contract.

                Budget fixed-point passes are allowed to repeat their stage name
                because their numeric allocation changes. Planning, tissue-gate,
                cell-feasibility and cell-execution failures are stalled when the
                normalized evidence is identical on consecutive passes.
                """

                nonlocal last_deterministic_feedback_signature
                signature = _deterministic_feedback_signature(feedback)
                if signature is None:
                    last_deterministic_feedback_signature = None
                    return
                if signature == last_deterministic_feedback_signature:
                    raise JointContractError(
                        "deterministic_replan_stalled: the same executable "
                        "failure repeated without a new interface, anchor, tool "
                        "program or capacity witness"
                    )
                last_deterministic_feedback_signature = signature

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

            def activate_panda_cell_capacity_floor_fallback(
                feedback: Mapping[str, Any],
            ) -> bool:
                """Use the declared PANDA hard floor after proven cell failure."""

                nonlocal allocation
                nonlocal budget_rebalance_count
                nonlocal execution_feedback
                nonlocal last_deterministic_feedback_signature
                nonlocal nuclei_preflight
                nonlocal panda_cell_capacity_floor_fallback_used
                nonlocal tissue_scene
                if panda_cell_capacity_floor_fallback_used:
                    return False
                if (
                    case.annotation_profile_id != "panda-gleason-v1"
                    or case.primitive_id
                    not in PANDA_CELL_CAPACITY_FALLBACK_PRIMITIVES
                    or case.joint_area_budget.fallback_policy
                    != "max_feasible_below_target"
                ):
                    return False
                stage = str(feedback.get("stage") or "")
                feasibility_failures = set(
                    (feedback.get("cell_feasibility_failure_counts") or {}).keys()
                )
                execution_errors = " ".join(
                    str(item) for item in feedback.get("errors", ())
                )
                proven_capacity_failure = bool(
                    stage == "cell_feasibility"
                    and feasibility_failures.intersection(
                        {
                            "candidate_cannot_fit_cell_in_continuity_region",
                            "exact_complete_footprint_packing_capacity_shortfall",
                            "exact_seam_packing_capacity_shortfall",
                        }
                    )
                ) or bool(
                    stage == "cell_execution"
                    and any(
                        marker in execution_errors
                        for marker in (
                            "PROBNET_UNDERFOLLOW",
                            "exact-count placement failed",
                            "target-tissue count quota",
                        )
                    )
                )
                if not proven_capacity_failure:
                    return False
                revised = self.budget_solver.fallback_tissue_target_to_execution_floor(
                    allocation
                )
                if revised.tissue_target_pixels == allocation.tissue_target_pixels:
                    return False
                budget_revisions.append(
                    {
                        "reason": "panda_cell_capacity_hard_floor_fallback",
                        "failure_stage": stage,
                        "before": allocation.to_metadata(),
                        "after": revised.to_metadata(),
                    }
                )
                allocation = revised
                budget_rebalance_count += 1
                panda_cell_capacity_floor_fallback_used = True
                revoked_tissue_candidate_ids.clear()
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
                        "PANDA cell-capacity floor has no feasible tissue--cell interface"
                    )
                if not nuclei_preflight.meaningful_tissue_capacity_passed:
                    raise JointContractError(
                        "PANDA cell-capacity floor is below the meaningful tissue floor"
                    )
                tissue_scene = augment_tissue_scene_with_nuclei_preflight(
                    scene.tissue,
                    nuclei_preflight,
                    auxiliary_structure_masks=scene.auxiliary_structure_masks,
                    required_auxiliary_structure_ids=(
                        bundle.mechanism.representability.protected_auxiliary_structures
                    ),
                    receiving_auxiliary_structure_ids=(
                        bundle.mechanism.representability.receiving_auxiliary_structures
                    ),
                )
                audit.write_json(
                    "joint_nuclei_preflight_cell_capacity_floor.json",
                    nuclei_preflight.to_metadata(),
                )
                execution_feedback = {
                    "retry_index": int(feedback.get("retry_index") or 0),
                    "stage": "budget_rebalance",
                    "errors": ["panda_cell_capacity_hard_floor_fallback"],
                    "failed_interface_ids": [],
                }
                last_deterministic_feedback_signature = None
                return True

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
                selected_tissue_witness = None
                try:
                    current_portfolio_binding = (
                        _tissue_portfolio_authority_binding(
                            case=case,
                            tissue_case=tissue_case,
                            source_tissue=source_tissue,
                            bundle=bundle,
                            tissue_bundle=tissue_bundle,
                            allocation=allocation,
                            nuclei_preflight=nuclei_preflight,
                        )
                    )
                    current_portfolio_binding_sha = _canonical_metadata_sha256(
                        current_portfolio_binding
                    )
                    portfolio_is_current = bool(
                        tissue_feasibility_portfolio
                        and getattr(
                            tissue_feasibility_portfolio,
                            "authority_binding_sha256",
                            None,
                        )
                        == current_portfolio_binding_sha
                        and not revoked_tissue_candidate_ids
                        and planning_pass == 0
                    )
                    if not portfolio_is_current:
                        tissue_feasibility_portfolio = (
                            CandidateFeasibilityCompiler(
                                maximum_attempts=(
                                    self.config.maximum_tissue_planning_attempts
                                ),
                                gates=self.tissue_gates,
                            ).compile_tissue_portfolio(
                                tissue_case=tissue_case,
                                source_tissue=source_tissue,
                                schema=schema,
                                scene=scene,
                                tissue_bundle=tissue_bundle,
                                joint_bundle=bundle,
                                nuclei_preflight=nuclei_preflight,
                                authority_binding=current_portfolio_binding,
                                maximum_candidates=(
                                    self.config.maximum_tissue_candidates
                                ),
                                revoked_candidate_ids=tuple(
                                    sorted(revoked_tissue_candidate_ids)
                                ),
                            )
                        )
                    tissue_feasibility_portfolio.validate_authority(
                        expected_binding_sha256=current_portfolio_binding_sha
                    )
                    audit.write_json(
                        f"candidate_feasibility_portfolio_pass_{planning_pass + 1}.json",
                        tissue_feasibility_portfolio.to_metadata(),
                    )
                    if hasattr(self.tissue_planner, "create_joint_tissue_plan"):
                        raw_tissue_plan, tissue_usage = (
                            self.tissue_planner.create_joint_tissue_plan(
                                case=tissue_case,
                                scene=tissue_scene,
                                bundle=tissue_bundle,
                                joint_bundle=bundle,
                                image_paths=planner_images,
                                nuclei_preflight=nuclei_preflight,
                                joint_case=case,
                                allocation=allocation,
                                execution_feedback=execution_feedback,
                                artifact_registry=planner_artifacts,
                                candidate_portfolio=(
                                    tissue_feasibility_portfolio
                                ),
                            )
                        )
                    else:
                        raw_tissue_plan, tissue_usage = self.tissue_planner.create_plan(
                            case=tissue_case,
                            scene=tissue_scene,
                            bundle=tissue_bundle,
                            image_paths=planner_images,
                        )
                    selection_certificate = (
                        raw_tissue_plan.tool_program.parameter_ranges.get(
                            "planner_selection_certificate"
                        )
                    )
                    if isinstance(selection_certificate, Mapping):
                        selected_candidate_id = selection_certificate.get(
                            "selected_candidate_id"
                        )
                        selected_tissue_witness = next(
                            (
                                item
                                for item in tissue_feasibility_portfolio.survivors
                                if item.candidate_id == selected_candidate_id
                            ),
                            None,
                        )
                        if selected_tissue_witness is None:
                            raise JointContractError(
                                "selected tissue candidate is absent from the current compiler portfolio"
                            )
                    compiled_tool_program = compile_tissue_tool_program(
                        primitive_id=raw_tissue_plan.primitive_id,
                        mechanism_id=bundle.mechanism.mechanism_id,
                        mechanism_allowed_families=(
                            bundle.mechanism.tissue_program.allowed_tools
                        ),
                        primitive_allowed_executors=(
                            tissue_bundle.edit_contract.allowed_tools
                        ),
                    )
                    raw_tissue_plan = bind_tissue_plan_tool_program(
                        raw_tissue_plan,
                        compiled=compiled_tool_program,
                    )
                    validate_edit_plan(
                        raw_tissue_plan,
                        case=tissue_case,
                        scene=tissue_scene,
                        bundle=tissue_bundle,
                    )
                    if selected_tissue_witness is not None:
                        # The feasibility compiler already produced and hard-
                        # gated this exact raster. Preserve the Planner's
                        # selection certificate on the bound plan and reuse
                        # that immutable execution artifact instead of solving
                        # the same topology problem a second time.
                        tissue_plan = raw_tissue_plan
                        compiler_usage = {
                            "compiler": "candidate_feasibility_witness_reuse",
                            "resolved_pixels": (
                                selected_tissue_witness.realized_tissue_pixels
                            ),
                        }
                        compiled_replay_parts = ()
                        compiled_replay_audit = dict(
                            selected_tissue_witness.replay_audit
                        )
                    else:
                        (
                            tissue_plan,
                            compiler_usage,
                            compiled_replay_parts,
                            compiled_replay_audit,
                        ) = compile_joint_tissue_plan_with_witness(
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
                        artifact_registry=planner_artifacts,
                    )
                    plan = bind_joint_plan_to_nuclei_preflight(
                        plan,
                        nuclei_preflight,
                    )
                except (RefineContractError, JointContractError) as exc:
                    if "unannotated pathology claim" in str(exc):
                        raise
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
                    reject_repeated_deterministic_feedback(execution_feedback)
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
                    seed=tissue_case.seed,
                    compiled_replay_parts=compiled_replay_parts,
                    compiled_replay_audit=compiled_replay_audit,
                    precompiled_candidate=(
                        selected_tissue_witness.execution_candidate
                        if selected_tissue_witness is not None
                        else None
                    ),
                )
                tissue_reports = execution_batch.tissue_gate_reports
                if selected_tissue_witness is not None:
                    try:
                        selected_tissue_witness.validate_reexecution(
                            candidates=execution_batch.all_candidates,
                            gate_reports=tissue_reports,
                        )
                    except JointContractError as exc:
                        revoked_tissue_candidate_ids.add(
                            selected_tissue_witness.candidate_id
                        )
                        execution_feedback = {
                            "retry_index": planning_pass + 1,
                            "stage": "tissue_gate",
                            "errors": [f"JointContractError: {exc}"],
                            "failed_interface_ids": [
                                item.interface_id
                                for item in tissue_plan.candidate_interfaces
                            ],
                            "failed_tissue_candidate_ids": [
                                selected_tissue_witness.candidate_id
                            ],
                            "required_action": (
                                "revoke the detached candidate and recompile the "
                                "portfolio against current source, budget and preflight"
                            ),
                        }
                        audit.write_json(
                            f"execution_feedback_pass_{planning_pass + 1}.json",
                            execution_feedback,
                        )
                        if planning_pass + 1 >= maximum_planning_attempts:
                            raise
                        continue
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
                    if selected_tissue_witness is not None:
                        revoked_tissue_candidate_ids.add(
                            selected_tissue_witness.candidate_id
                        )
                    execution_feedback = _summarize_tissue_execution_failure(
                        execution_batch,
                        retry_index=planning_pass + 1,
                    )
                    audit.write_json(
                        f"execution_feedback_pass_{planning_pass + 1}.json",
                        execution_feedback,
                    )
                    if activate_panda_cell_capacity_floor_fallback(
                        execution_feedback
                    ):
                        continue
                    reject_repeated_deterministic_feedback(execution_feedback)
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
                desired_min, desired_max = (
                    case.joint_area_budget.desired_interval_pixels(source_tissue.shape)
                )
                cell_feasibility_by_tissue_id = {
                    item.candidate_id: item
                    for item in execution_batch.cell_feasibility_reports
                }
                predicted = [
                    int(
                        cell_feasibility_by_tissue_id[
                            item.candidate_id
                        ].predicted_joint_pixels
                    )
                    for item in passing_tissue
                ]
                predicted_above = _provisional_union_requires_rebalance(
                    predicted,
                    maximum_pixels=desired_max,
                )
                _, joint_hard_max = case.joint_area_budget.hard_interval_pixels(
                    source_tissue.shape
                )
                retain_visible_regression_closure = (
                    _retain_visible_regression_whole_instance_closure(
                        annotation_profile_id=case.annotation_profile_id,
                        primitive_id=case.primitive_id,
                        fallback_policy=case.joint_area_budget.fallback_policy,
                        predicted_pixels=predicted,
                        desired_max_pixels=desired_max,
                        hard_max_pixels=joint_hard_max,
                    )
                )
                # Candidate-local exact packing supplies a complete source-
                # instance erasure E and a concrete target-footprint witness
                # F.  ``predicted`` is therefore the executable T ∪ E ∪ F,
                # not an area heuristic.  It can safely rebalance an overfill
                # before invoking ProbNet; executed masks remain authoritative
                # for any later underfill correction.
                # Cord/nest packing has a discrete minimum group size. Let one
                # hard-range-safe paired execution establish the real union so
                # the existing whole-instance fallback can retain that proof;
                # shrinking tissue pre-execution can make the required group
                # geometrically impossible and discard the only safe pair.
                if (
                    planning_pass + 1 < maximum_planning_attempts
                    and budget_rebalance_count
                    < maximum_planning_attempts - 1
                    and predicted_above
                    and not retain_visible_regression_closure
                    and case.primitive_id
                    not in SPECIALIZED_ARCHITECTURE_PRIMITIVES
                ):
                    feedback_reports = [
                        cell_feasibility_by_tissue_id[item.candidate_id]
                        for item in passing_tissue
                    ]
                    # Candidate selection needs one executable pair, not a
                    # budget that makes every provisional sibling safe. Bind
                    # the next fixed point to the smallest observed complete-
                    # instance spill: that candidate returns to the joint
                    # target while preserving the largest tissue/P domain for
                    # nuclei packing. Siblings with larger spill remain free
                    # to fail the later joint-area gate.
                    selected_feedback = min(
                        feedback_reports,
                        key=lambda item: (
                            item.predicted_joint_pixels,
                            item.candidate_id,
                        ),
                    )
                    revised = self.budget_solver.reserve_observed_cell_spill(
                        allocation,
                        complete_instance_pixels=(
                            selected_feedback.complete_instance_spill_pixels
                        ),
                        footprint_spill_pixels=(
                            selected_feedback.target_footprint_spill_pixels
                        ),
                    )
                    if revised.tissue_target_pixels != allocation.tissue_target_pixels:
                        direction = "exact_pre_probnet_joint_union_overfill"
                        budget_revisions.append(
                            {
                                "reason": direction,
                                "desired_joint_interval": [desired_min, desired_max],
                                "provisional_joint_pixels": predicted,
                                "selected_feedback_candidate_id": (
                                    selected_feedback.candidate_id
                                ),
                                "exact_complete_instance_spill_pixels": (
                                    selected_feedback.complete_instance_spill_pixels
                                ),
                                "exact_target_footprint_spill_pixels": (
                                    selected_feedback.target_footprint_spill_pixels
                                ),
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
                                bundle.mechanism.representability.protected_auxiliary_structures
                            ),
                            receiving_auxiliary_structure_ids=(
                                bundle.mechanism.representability.receiving_auxiliary_structures
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
                    if activate_panda_cell_capacity_floor_fallback(
                        execution_feedback
                    ):
                        continue
                    if selected_tissue_witness is not None:
                        revoked_tissue_candidate_ids.add(
                            selected_tissue_witness.candidate_id
                        )
                    reject_repeated_deterministic_feedback(execution_feedback)
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
                    if _retain_visible_regression_whole_instance_closure(
                        annotation_profile_id=case.annotation_profile_id,
                        primitive_id=case.primitive_id,
                        fallback_policy=(
                            case.joint_area_budget.fallback_policy
                        ),
                        predicted_pixels=(provisional_min,),
                        desired_max_pixels=desired_max,
                        hard_max_pixels=hard_max,
                    ):
                        # This executed mature batch has already passed every
                        # hard gate except the preferred joint interval, and
                        # its smallest complete-instance closure remains
                        # inside the declared hard range. Certify that batch
                        # directly for the two visible-regression primitives;
                        # shrinking and recompiling the tissue edit would
                        # weaken the intended footprint or destroy a valid
                        # fragmentation seam merely to chase a soft target.
                        joint_area_feedback_count += 1
                        if activate_rebalance_exhausted_area_fallback():
                            break

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
                                bundle.mechanism.representability.protected_auxiliary_structures
                            ),
                            receiving_auxiliary_structure_ids=(
                                bundle.mechanism.representability.receiving_auxiliary_structures
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
            mask_review_board = planner_artifacts.write_candidate_board(
                candidates=passing_joint,
            )
            audit.write_json(
                "mask_planner_artifact_registry.json",
                planner_artifacts.to_metadata(),
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
                image_paths=(mask_review_board,),
                artifact_registry=planner_artifacts,
            )
            usage["critic"] = critic_result.usage
            audit.write_json("joint_critic.json", critic_result.to_metadata())
            if critic_result.abstain or not critic_result.rankings:
                reasons.append("independent_mask_condition_critic_approval_required")
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
        """Compile every semantic hypothesis before mask-graph disambiguation.

        The mask-graph Planner only sees primitive--mechanism pairs that already
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
        directionless_scenario = (
            case.semantic_intent.get("scenario") == "post_treatment_change"
        )
        raw_hypothesis_items = tuple(
            item for item in raw_hypotheses if isinstance(item, Mapping)
        )
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
            candidate_scenario = str(raw.get("scenario") or "")
            if directionless_scenario:
                scenario_directions = {
                    "treatment_response": "improve",
                    "post_treatment_progression": "worsen",
                    "residual_disease": "persist",
                }
                if candidate_scenario not in scenario_directions:
                    rejected[
                        f"unbound-scenario::{primitive_id or 'missing-primitive'}"
                    ] = (
                        "directionless post-treatment hypothesis lacks a bound "
                        "executable scenario"
                    )
                    continue
                bound_hypotheses = [
                    dict(item)
                    for item in raw_hypothesis_items
                    if item.get("scenario") == candidate_scenario
                ]
                bound_intent = {
                    **case.semantic_intent,
                    "scenario": candidate_scenario,
                    "clinical_direction": scenario_directions[candidate_scenario],
                    "direction": scenario_directions[candidate_scenario],
                    "treatment_context": "post_treatment",
                    "primitive_hypotheses": bound_hypotheses,
                }
                candidate_case = replace(
                    candidate_case,
                    semantic_intent=bound_intent,
                )
            primitive_contract = self.joint_skills.primitives.get(primitive_id)
            if (
                primitive_contract is not None
                and primitive_contract.scope == "cell_only"
                and candidate_case.cell_count_extent_budget is None
            ):
                minimum_effect_span_cell_diameters = (
                    primitive_contract.minimum_effect_span_cell_diameters
                )
                minimum_effect_foci = primitive_contract.minimum_effect_foci
                if candidate_case.annotation_profile_id == "panda-gleason-v1":
                    if primitive_id in PANDA_CELL_EFFECT_EXTENT_OVERRIDES:
                        (
                            minimum_effect_span_cell_diameters,
                            minimum_effect_foci,
                        ) = PANDA_CELL_EFFECT_EXTENT_OVERRIDES[primitive_id]
                if primitive_id in INFILTRATION_BUDGET_PRIMITIVES:
                    try:
                        budget, budget_metadata = _derive_infiltration_budget(
                            scene,
                            minimum_effect_delta_count=(
                                primitive_contract.minimum_effect_delta_count_for(
                                    candidate_case.pathology_domain_id
                                )
                            ),
                            minimum_effect_span_cell_diameters=(
                                minimum_effect_span_cell_diameters
                            ),
                            minimum_effect_foci=minimum_effect_foci,
                        )
                    except (JointContractError, RefineContractError, ValueError) as exc:
                        rejected[primitive_id] = str(exc)
                        continue
                else:
                    try:
                        budget, budget_metadata = _derive_local_population_budget(
                            scene,
                            primitive_id=primitive_id,
                            semantic_intent=candidate_case.semantic_intent,
                            annotation_profile_id=(
                                candidate_case.annotation_profile_id
                            ),
                            host_tissue_labels=primitive_contract.host_tissue_labels,
                            minimum_effect_delta_count=(
                                primitive_contract.minimum_effect_delta_count_for(
                                    candidate_case.pathology_domain_id
                                )
                            ),
                            minimum_effect_span_cell_diameters=(
                                minimum_effect_span_cell_diameters
                            ),
                            minimum_effect_foci=minimum_effect_foci,
                        )
                    except (JointContractError, RefineContractError, ValueError) as exc:
                        rejected[primitive_id] = str(exc)
                        continue
                budget, budget_metadata = _apply_panda_profile_cell_budget(
                    candidate_case,
                    primitive_id=primitive_id,
                    budget=budget,
                    metadata=budget_metadata,
                )
                budget, budget_metadata = _apply_glas_visible_cell_budget(
                    candidate_case,
                    primitive_id=primitive_id,
                    budget=budget,
                    metadata=budget_metadata,
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
                manifest_hint = case.semantic_intent.get(
                    "manifest_primitive_hint"
                )
                if (
                    mechanism.planner_policy.requires_explicit_primitive_intent
                    and semantic_fit == "contextual"
                    and manifest_hint != primitive_id
                ):
                    rejected[primitive_id] = (
                        "no joint mechanism supports this primitive without "
                        "an explicit primitive request"
                    )
                    continue
                option_id = "::".join(
                    item
                    for item in (
                        candidate_scenario if directionless_scenario else "",
                        primitive_id,
                        mechanism.mechanism_id,
                    )
                    if item
                )
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
                    tissue_feasibility_portfolio = ()
                    feasibility: dict[str, Any] = {
                        "four_axis_skill_intersection": "passed",
                        "deterministic_preflight": "passed",
                        "annotation_operational_stroma_policy": (
                            bundle.annotation_profile.operational_stroma_policy
                        ),
                        "annotation_visual_veto_requirements": list(
                            bundle.annotation_profile.visual_veto_requirements
                        ),
                    }
                    nucleus_authority = validate_mechanism_nucleus_authority(
                        scene.cells.instances,
                        allow_semantic_instance_fallback=(
                            bundle.mechanism.representability.allow_semantic_instance_fallback
                        ),
                        required_cell_classes=(
                            bundle.mechanism.representability.required_cell_classes
                            or bundle.primitive.target_cell_classes
                        ),
                        actions=bundle.mechanism.cell_program.actions,
                    )
                    feasibility["nucleus_instance_authority"] = nucleus_authority
                    if not nucleus_authority["passed"]:
                        raise JointContractError(
                            "nucleus instance authority preflight failed: "
                            + ", ".join(nucleus_authority["reasons"])
                        )
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
                        tool_primitive_id = tissue_tool_primitive_id(
                            primitive_id
                        )
                        tissue_bundle = self.mask_skills.compose(
                            pathology_domain_id=(
                                candidate_case.pathology_domain_id
                            ),
                            annotation_profile_id=(
                                candidate_case.annotation_profile_id
                            ),
                            primitive_id=tool_primitive_id,
                            production=self.config.production,
                            available_checker_ids=(
                                self.tissue_gates.available_checker_ids
                            ),
                            case_provenance=candidate_case.provenance,
                        )
                        tissue_bundle = bind_active_bundle_to_case(
                            tissue_bundle,
                            case=candidate_case,
                            scene=scene.tissue,
                            semantic_primitive_id=primitive_id,
                        )
                        validate_active_bundle_authority(
                            tissue_bundle,
                            case_provenance=candidate_case.provenance,
                            require_live_binding=True,
                            case=candidate_case,
                            scene=scene.tissue,
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
                        tissue_case = _as_tissue_case(
                            candidate_case,
                            allocation=allocation,
                            shape=source_tissue.shape,
                        )
                        tissue_feasibility_portfolio = (
                            CandidateFeasibilityCompiler(
                                maximum_attempts=(
                                    self.config.maximum_tissue_planning_attempts
                                ),
                                gates=self.tissue_gates,
                            ).compile_tissue_portfolio(
                                tissue_case=tissue_case,
                                source_tissue=source_tissue,
                                schema=schema,
                                scene=scene,
                                tissue_bundle=tissue_bundle,
                                joint_bundle=bundle,
                                nuclei_preflight=nuclei_preflight,
                                authority_binding=(
                                    _tissue_portfolio_authority_binding(
                                        case=candidate_case,
                                        tissue_case=tissue_case,
                                        source_tissue=source_tissue,
                                        bundle=bundle,
                                        tissue_bundle=tissue_bundle,
                                        allocation=allocation,
                                        nuclei_preflight=nuclei_preflight,
                                    )
                                ),
                            )
                        )
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
                                "whole_mask_topology_portfolio": (
                                    tissue_feasibility_portfolio.to_metadata()
                                ),
                            }
                        )
                    elif "add" in bundle.mechanism.cell_program.actions:
                        allowed = set(
                            bundle.mechanism.cell_program.allowed_cell_classes
                        )
                        library_reference_counts = (
                            {
                                int(class_id): len(items)
                                for class_id, items in (
                                    scene.reference_shape_authority.shapes_by_class.items()
                                )
                            }
                            if scene.reference_shape_authority is not None
                            else {}
                        )
                        reference_authority = (
                            count_authoritative_complete_references(
                                scene.cells.instances,
                                allowed_cell_classes=allowed,
                                allow_semantic_instance_fallback=(
                                    bundle.mechanism.representability.allow_semantic_instance_fallback
                                ),
                                library_reference_counts=library_reference_counts,
                            )
                        )
                        complete_references = int(
                            reference_authority["total_reference_count"]
                        )
                        feasibility["reference_shape_authority"] = (
                            reference_authority
                        )
                        if complete_references <= 0:
                            raise JointContractError(
                                "cell interpretation has no authoritative complete reference nucleus"
                            )
                        if (
                            primitive_id
                            in {
                                "neoplastic-microinfiltration-increase-v1",
                                "peritumoral-neoplastic-scatter-increase-v1",
                                "peritumoral-small-cluster-increase-v1",
                            }
                        ):
                            label_contract = (
                                bundle.mechanism.tissue_program.primitive_label_contracts[
                                    primitive_id
                                ]
                            )
                            receiving_labels = set(
                                label_contract["source_labels"]
                            )
                            compatible_interfaces = [
                                interface
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
                            ]
                            if (
                                "external_boundary_binding"
                                in bundle.mechanism.planner_policy.hard_constraint_checker_ids
                            ):
                                from .feasibility import classify_tumor_stroma_boundary

                                compatible_interfaces = [
                                    interface
                                    for interface in compatible_interfaces
                                    if classify_tumor_stroma_boundary(
                                        scene=scene,
                                        interface=interface,
                                        allowed_host_labels=tuple(
                                            sorted(receiving_labels)
                                        ),
                                    )["external_tumor_stroma_boundary"]
                                ]
                            interface_pairs = {
                                tuple(
                                    sorted(
                                        (
                                            interface.source_component_id,
                                            interface.target_component_id,
                                        )
                                    )
                                )
                                for interface in compatible_interfaces
                            }
                            if not interface_pairs:
                                raise JointContractError(
                                    "contextual infiltration has no tumor-to-receiving-tissue interface"
                                )
                            feasibility[
                                "candidate_infiltration_interface_count"
                            ] = len(interface_pairs)
                            feasibility["external_boundary_required"] = (
                                "external_boundary_binding"
                                in bundle.mechanism.planner_policy.hard_constraint_checker_ids
                            )
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
                        tissue_feasibility_portfolio=(
                            tissue_feasibility_portfolio
                        ),
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
                (
                    case.provenance.get(
                        "require_mature_probnet_regeneration", False
                    )
                    or self.config.require_mature_probnet_for_target_population_regeneration
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
                    reference_shape_authority=layouts[0].trace.get(
                        "reference_shape_authority"
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
        audit.write_mask_review_board(
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
        planner_artifacts,
        usage,
    ):
        """Execute a count/extent primitive without entering the tissue solver."""

        cell_portfolio = self._compile_cell_only_candidate_portfolio(
            case=case,
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            schema=schema,
            scene=scene,
            bundle=bundle,
        )
        plan, joint_usage = self.joint_planner.create_plan(
            case=case,
            scene=scene,
            bundle=bundle,
            tissue_plan=None,
            image_paths=planner_images,
            artifact_registry=planner_artifacts,
            candidate_portfolio=cell_portfolio.certificates,
        )
        usage["joint_planner"] = joint_usage
        selected_cell_choice = _select_cell_execution_choice(
            portfolio=cell_portfolio,
            plan=plan,
            planner_usage=joint_usage,
        )
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
            candidate_id=selected_cell_choice.executable_contract.tissue_candidate_id,
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
        executable_contract = selected_cell_choice.executable_contract
        cell_only_preflight = selected_cell_choice.preflight
        executable_contract.validate_identity()
        plan_digest = hashlib.sha256(
            json.dumps(
                plan.to_metadata(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if executable_contract.plan_digest != plan_digest:
            raise JointContractError(
                "selected cell candidate plan digest is detached from its contract"
            )
        if executable_contract.plan_digest != self.executable_contract_compiler.compile(
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
            required_checker_ids=(
                self.joint_gates.required_checker_ids_for(bundle)
            ),
        ).plan_digest:
            raise JointContractError(
                "selected cell candidate no longer matches the executable compiler"
            )
        audit.write_json(
            "cell_only_executable_capacity_preflight.json",
            cell_only_preflight.to_metadata(),
        )
        audit.write_executable_contract(executable_contract)
        # The mature pipeline owns target-population regeneration. Structured
        # additions are deterministic templates whose anchors may be ranked by
        # the frozen ProbNet but whose count and geometry remain skill-owned.
        cell_addition = "add" in bundle.mechanism.cell_program.actions
        ranker_required = bool(
            self.config.production
            or (
                self.config.require_probnet_ranker_for_cell_addition
                and cell_addition
            )
        )
        if ranker_required and (
            self.ranker is None
            or getattr(self.ranker, "name", "") == "deterministic_distance_ranker"
        ):
            raise JointContractError(
                "cell-only addition requires the frozen ProbNet spatial ranker; "
                "deterministic distance fallback is forbidden by the active evaluation contract"
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
                reference_shape_authority=layouts[0].trace.get(
                    "reference_shape_authority"
                ),
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
        audit.write_review_board(
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
        mask_review_board = planner_artifacts.write_candidate_board(
            candidates=passing,
        )
        audit.write_json(
            "mask_planner_artifact_registry.json",
            planner_artifacts.to_metadata(),
        )
        critic = self.critic.review(
            case=case,
            bundle=bundle,
            candidates=passing,
            gate_reports=reports,
            image_paths=(mask_review_board,),
            artifact_registry=planner_artifacts,
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
                reasons=("independent_mask_condition_critic_approval_required",),
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

    @staticmethod
    def _certify_cell_only_executable_capacity(
        *,
        case,
        source_nuclei,
        scene,
        bundle,
        plan,
        tissue_candidate,
        executable_contract,
    ):
        """Run exact P/V/E packing before any cell-only layout execution."""

        from .feasibility import CandidateCellFeasibility

        program = executable_contract.cell_program
        budget = case.cell_count_extent_budget
        requested = int(budget.target_delta_count if budget is not None else 0)
        is_addition = "add" in plan.cell_plan.actions
        initial = CandidateCellFeasibility(
            candidate_id=tissue_candidate.candidate_id,
            passed=True,
            removable_instance_ids=tuple(
                executable_contract.erase_instance_ids
            ),
            required_removal_cell_classes=tuple(
                sorted(plan_class for plan_class in program.target_classes)
            ),
            estimated_removal_count=len(
                executable_contract.erase_instance_ids
            ),
            protected_overlap_ids=(),
            nonlocal_extension_ids=(),
            legal_core_pixels=int(
                np.count_nonzero(program.placement_center_region)
            ),
            reference_fit_center_pixels=int(
                np.count_nonzero(program.placement_center_region)
            ),
            required_add_count=(
                requested if is_addition else 0
            ),
            required_seam_count=0,
            estimated_add_capacity=requested,
            estimated_seam_capacity=0,
            continuity_mode=program.continuity_mode,
            continuity_width_px=program.continuity_width_px,
            continuity_maximum_empty_run_px=(
                program.continuity_maximum_empty_run_px
            ),
            continuity_anchor_pixels=int(
                np.count_nonzero(program.continuity_anchor_mask)
            ),
            continuity_region_pixels=int(
                np.count_nonzero(program.continuity_region)
            ),
            potential_anchor_coverage_fraction=1.0,
            minimum_anchor_coverage_fraction=(
                program.continuity_minimum_anchor_coverage_fraction
            ),
            meaningful_tissue_floor_pixels=0,
            tissue_change_pixels=0,
            exact_packing_certificate={},
            complete_instance_spill_pixels=0,
            target_footprint_spill_pixels=0,
            predicted_joint_pixels=int(
                np.count_nonzero(program.erasure_region)
            ),
            reasons=(),
        )
        # Reuse the same exact packing verifier as the tissue-changing path.
        # Packing and execution must allocate the same observed local class
        # composition. Equal synthetic weights certified one frozen case at
        # 7/7/6 while execution correctly requested 17/2/1, producing a false
        # capacity pass.
        local_class_counts = {
            int(class_id): 0 for class_id in program.target_classes
        }
        placement_region = np.asarray(
            program.placement_center_region, dtype=bool
        )
        for item in scene.cells.instances:
            if item.class_id not in local_class_counts:
                continue
            x, y = item.centroid_xy
            row, col = round(y), round(x)
            if (
                0 <= row < placement_region.shape[0]
                and 0 <= col < placement_region.shape[1]
                and placement_region[row, col]
            ):
                local_class_counts[item.class_id] += 1
        if sum(local_class_counts.values()) <= 0:
            local_class_counts = {
                int(class_id): 1 for class_id in program.target_classes
            }
        pseudo_preflight = type(
            "CellOnlyPackingAuthority",
            (),
            {
                "eligible_reference_ids": tuple(
                    item.instance_id
                    for item in scene.cells.instances
                    if item.completeness_status == "complete"
                    and not item.touches_border
                    and not item.quality_flags
                ),
                "target_cell_class": int(program.target_classes[0]),
                "target_density_by_class": {
                    class_id: float(count)
                    for class_id, count in local_class_counts.items()
                },
            },
        )()
        references_by_class = {}
        for class_id in program.target_classes:
            references, _rejected = build_reference_shape_library(
                scene,
                class_id=int(class_id),
                allow_calibrated_fallback=True,
            )
            references = _prioritize_local_references(
                references,
                scene=scene,
                interface_ids=plan.cell_plan.interface_ids,
                core_zone=plan.cell_plan.core_zone,
            )
            if references:
                references_by_class[int(class_id)] = tuple(references)
        minimum_acceptable_count = (
            max(
                int(budget.min_delta_count),
                int(
                    bundle.primitive.minimum_effect_delta_count_for(
                        case.pathology_domain_id
                    )
                ),
            )
            if budget is not None
            and is_addition
            else None
        )
        certified = certify_compiled_cell_program_feasibility(
            initial,
            candidate=tissue_candidate,
            contract=executable_contract,
            scene=scene,
            preflight=pseudo_preflight,
            authoritative_references_by_class=references_by_class,
            minimum_acceptable_add_count=minimum_acceptable_count,
            allow_final_capacity_refinement=(
                case.annotation_profile_id
                in {"glas-gland-v1", "panda-gleason-v1"}
            ),
            relax_group_preflight=(
                case.annotation_profile_id
                in {"glas-gland-v1", "panda-gleason-v1"}
            ),
        )
        if not certified.passed:
            raise JointContractError(
                "cell-only exact packing preflight failed before execution: "
                + ", ".join(certified.reasons)
            )
        # NOTE: Premature witness spatial audit (effect_span / effect_foci)
        # was here but was removed.  The packing certificate's distance-based
        # witness placements do not reflect the score-based ordering used by
        # actual execution, so auditing them here produced false negatives.
        # Spatial validity is properly verified by post-execution gates such
        # as joint_area and mechanism_postcondition.
        return certified

    def _compile_cell_only_candidate_portfolio(
        self,
        *,
        case,
        source_tissue,
        source_nuclei,
        schema,
        scene,
        bundle,
    ) -> _CertifiedCellExecutionPortfolio:
        """Certify zone/interface/annulus choices before the cell LLM stage."""

        variants: list[dict[str, Any]] = []
        local_population = case.primitive_id in {
            "cell-type-abundance-decrease-v1",
            "cell-type-abundance-increase-v1",
            "cellularity-decrease-v1",
            "cellularity-increase-v1",
            "neoplastic-cell-abundance-decrease-v1",
            "neoplastic-cell-abundance-increase-v1",
            "generic-inflammatory-cell-abundance-decrease-v1",
            "generic-inflammatory-cell-abundance-increase-v1",
        }
        if local_population:
            component_labels = {
                item.component_id: item.label
                for item in scene.tissue.graph.components
            }
            raw_classes = case.provenance.get("target_cell_class_ids", ())
            if isinstance(raw_classes, int):
                raw_classes = (raw_classes,)
            requested_class_ids = (
                tuple(sorted({int(value) for value in raw_classes}))
                if isinstance(raw_classes, (list, tuple))
                else ()
            )
            abundance = case.primitive_id.startswith(
                (
                    "cell-type-abundance-",
                    "neoplastic-cell-abundance-",
                    "generic-inflammatory-cell-abundance-",
                )
            )
            zones = [
                item
                for item in scene.population.zones
                if item.zone_kind == "component"
                and component_labels.get(item.tissue_component_id)
                in set(bundle.primitive.host_tissue_labels)
                and item.area_px > 0
            ]
            if (
                abundance
                and requested_class_ids
                and case.annotation_profile_id
                in {"glas-gland-v1", "panda-gleason-v1"}
            ):
                required_label = (
                    "Tumor" if set(requested_class_ids) == {1} else "Stroma"
                )
                zones = [
                    item
                    for item in zones
                    if component_labels.get(item.tissue_component_id)
                    == required_label
                ]
            zones.sort(
                key=lambda item: (
                    -sum(
                        int(item.class_counts.get(class_id, 0))
                        for class_id in requested_class_ids
                    )
                    if abundance and requested_class_ids
                    else -item.nucleus_count,
                    -item.nucleus_count,
                    -item.area_px,
                    item.zone_id,
                )
            )
            layout_program = bundle.mechanism.cell_program.layout_for(
                case.primitive_id
            )
            requires_depletion_anchor = (
                layout_program == "localized_density_gradient"
            )
            depletion = bundle.mechanism.cell_program.cellularity_depletion
            anchor_records = {
                item.anchor_segment_id: item
                for item in scene.tissue.graph.anchor_segments
            }
            candidate_zones = zones if requires_depletion_anchor else zones[:4]
            for zone in candidate_zones:
                provenance = {"joint_population_zone_id": zone.zone_id}
                if requires_depletion_anchor:
                    touching = [
                        interface
                        for interface in scene.tissue.graph.interfaces
                        if zone.tissue_component_id
                        in {
                            interface.source_component_id,
                            interface.target_component_id,
                        }
                    ]
                    allowed_neighbors = set(
                        depletion.allowed_neighbor_labels
                        if depletion is not None
                        else ()
                    )
                    eligible_touching = []
                    for interface in touching:
                        if interface.source_component_id == zone.tissue_component_id:
                            neighbor = interface.target_label
                        else:
                            neighbor = interface.source_label
                        if allowed_neighbors and neighbor not in allowed_neighbors:
                            continue
                        eligible_touching.append(interface)
                    eligible_touching.sort(
                        key=lambda item: (-item.contact_pixels, item.interface_id)
                    )
                    if (
                        case.annotation_profile_id
                        in {"glas-gland-v1", "panda-gleason-v1"}
                        and len(eligible_touching) > 1
                    ):
                        all_anchor_ids = tuple(
                            dict.fromkeys(
                                anchor_id
                                for interface in eligible_touching
                                for anchor_id in interface.anchor_segment_ids
                                if anchor_id in anchor_records
                            )
                        )
                        if all_anchor_ids:
                            variants.append(
                                {
                                    **provenance,
                                    "cellularity_depletion_anchor": {
                                        "type": "interface",
                                        "interface_ids": [
                                            item.interface_id
                                            for item in eligible_touching
                                        ],
                                        "anchor_ids": list(all_anchor_ids),
                                        "observation": (
                                            "deterministic selected-component "
                                            "boundary union"
                                        ),
                                        "confidence": 1.0,
                                    },
                                }
                            )
                    # The scene graph may split one biological component-pair
                    # interface into several deterministic interface records.
                    # A density field is allowed to bind that one component
                    # pair broadly, but must not combine unrelated neighbours.
                    # Certifying this union prevents graph segmentation from
                    # artificially shrinking the three-band field below its
                    # area/count contract.
                    interfaces_by_neighbor: dict[
                        str, list[Any]
                    ] = defaultdict(list)
                    for interface in eligible_touching:
                        neighbor_component_id = (
                            interface.target_component_id
                            if interface.source_component_id
                            == zone.tissue_component_id
                            else interface.source_component_id
                        )
                        interfaces_by_neighbor[
                            str(neighbor_component_id)
                        ].append(interface)
                    for grouped_interfaces in interfaces_by_neighbor.values():
                        if len(grouped_interfaces) < 2:
                            continue
                        grouped_interfaces.sort(
                            key=lambda item: (
                                -item.contact_pixels,
                                item.interface_id,
                            )
                        )
                        grouped_interface_ids = tuple(
                            item.interface_id for item in grouped_interfaces
                        )
                        grouped_anchor_ids = tuple(
                            item.anchor_segment_id
                            for item in sorted(
                                (
                                    anchor_records[anchor_id]
                                    for interface in grouped_interfaces
                                    for anchor_id in interface.anchor_segment_ids
                                    if anchor_id in anchor_records
                                ),
                                key=lambda item: (
                                    -item.contact_pixels,
                                    item.display_index,
                                    item.anchor_segment_id,
                                ),
                            )
                        )
                        if grouped_anchor_ids:
                            variants.append(
                                {
                                    **provenance,
                                    "cellularity_depletion_anchor": {
                                        "type": "interface",
                                        "interface_ids": list(
                                            grouped_interface_ids
                                        ),
                                        "anchor_ids": list(grouped_anchor_ids),
                                        "observation": (
                                            "deterministic same-component-pair "
                                            "interface union"
                                        ),
                                        "confidence": 1.0,
                                    },
                                }
                            )
                    for interface in eligible_touching:
                        if not interface.anchor_segment_ids:
                            continue
                        ordered_anchor_ids = tuple(
                            item.anchor_segment_id
                            for item in sorted(
                                (
                                    anchor_records[anchor_id]
                                    for anchor_id in interface.anchor_segment_ids
                                    if anchor_id in anchor_records
                                ),
                                key=lambda item: (
                                    -item.contact_pixels,
                                    item.display_index,
                                    item.anchor_segment_id,
                                ),
                            )
                        )
                        if not ordered_anchor_ids:
                            continue
                        # Certify both a broad interface-owned density field
                        # and the strongest local sub-arcs.  The portfolio
                        # compiler, not the LLM, decides which groups have
                        # enough complete class-1 instances after residual
                        # floors and minimum-effect checks.
                        anchor_groups = [ordered_anchor_ids]
                        anchor_groups.extend(
                            (anchor_id,) for anchor_id in ordered_anchor_ids
                        )
                        for anchor_ids in dict.fromkeys(anchor_groups):
                            variants.append(
                                {
                                    **provenance,
                                    "cellularity_depletion_anchor": {
                                        "type": "interface",
                                        "interface_ids": [
                                            interface.interface_id
                                        ],
                                        "anchor_ids": list(anchor_ids),
                                        "observation": (
                                            "deterministic component-interface "
                                            "adjacency"
                                        ),
                                        "confidence": 1.0,
                                    },
                                }
                            )
                else:
                    variants.append(provenance)
        else:
            host = set(bundle.primitive.host_tissue_labels)
            compatible = [
                item
                for item in scene.tissue.graph.interfaces
                if (
                    item.source_label in host and item.target_label == "Tumor"
                )
                or (
                    item.target_label in host and item.source_label == "Tumor"
                )
            ]
            if (
                "external_boundary_binding"
                in bundle.mechanism.planner_policy.hard_constraint_checker_ids
            ):
                from .feasibility import classify_tumor_stroma_boundary

                compatible = [
                    item
                    for item in compatible
                    if classify_tumor_stroma_boundary(
                        scene=scene,
                        interface=item,
                        allowed_host_labels=tuple(sorted(host)),
                    )["external_tumor_stroma_boundary"]
                ]
            compatible.sort(key=lambda item: (-item.contact_pixels, item.interface_id))
            prostate_scatter = (
                bundle.mechanism.mechanism_id
                == "prostate-pattern-5-peripheral-scatter"
            )
            exhaustive_peritumoral = bool(
                case.annotation_profile_id == "glas-gland-v1"
                and case.primitive_id
                in {
                    "peritumoral-neoplastic-scatter-increase-v1",
                    "peritumoral-small-cluster-increase-v1",
                }
            )
            interfaces_to_consider = (
                compatible[:8]
                if exhaustive_peritumoral
                else compatible
                if prostate_scatter
                else compatible[:4]
            )
            if exhaustive_peritumoral or prostate_scatter:
                interfaces_by_component_pair: dict[
                    tuple[str, str], list[Any]
                ] = defaultdict(list)
                for interface in interfaces_to_consider:
                    pair = tuple(
                        sorted(
                            (
                                str(interface.source_component_id),
                                str(interface.target_component_id),
                            )
                        )
                    )
                    interfaces_by_component_pair[pair].append(interface)
                for grouped in interfaces_by_component_pair.values():
                    if len(grouped) < 2:
                        continue
                    grouped.sort(
                        key=lambda item: (-item.contact_pixels, item.interface_id)
                    )
                    grouped_anchors = tuple(
                        dict.fromkeys(
                            anchor_id
                            for interface in grouped
                            for anchor_id in interface.anchor_segment_ids
                        )
                    )
                    if grouped_anchors:
                        variants.append(
                            {
                                "joint_interface_ids": [
                                    item.interface_id for item in grouped
                                ],
                                "joint_anchor_ids": list(grouped_anchors),
                            }
                        )
            if prostate_scatter:
                # PANDA may split one Pattern-5 outer boundary across several
                # adjacent Stroma components.  Scatter belongs to the selected
                # tumor component's complete external boundary, so certify
                # that union before its local sub-arcs.
                interfaces_by_tumor: dict[str, list[Any]] = defaultdict(list)
                for interface in interfaces_to_consider:
                    tumor_component_id = (
                        interface.source_component_id
                        if interface.source_label == "Tumor"
                        else interface.target_component_id
                    )
                    interfaces_by_tumor[str(tumor_component_id)].append(
                        interface
                    )
                for grouped in interfaces_by_tumor.values():
                    if len(grouped) < 2:
                        continue
                    grouped.sort(
                        key=lambda item: (-item.contact_pixels, item.interface_id)
                    )
                    grouped_anchors = tuple(
                        dict.fromkeys(
                            anchor_id
                            for interface in grouped
                            for anchor_id in interface.anchor_segment_ids
                            if anchor_id in scene.tissue.anchor_masks
                        )
                    )
                    if grouped_anchors:
                        variants.append(
                            {
                                "joint_interface_ids": [
                                    item.interface_id for item in grouped
                                ],
                                "joint_anchor_ids": list(grouped_anchors),
                            }
                        )
            for interface in interfaces_to_consider:
                anchors = interface.anchor_segment_ids or ()
                if prostate_scatter:
                    anchors = tuple(
                        anchor_id
                        for anchor_id in anchors
                        if anchor_id in scene.tissue.anchor_masks
                        and np.any(
                            ndimage.binary_dilation(
                                np.asarray(
                                    scene.tissue.anchor_masks[anchor_id],
                                    dtype=bool,
                                ),
                                structure=np.ones((3, 3), dtype=bool),
                            )
                            & (np.asarray(source_tissue) == 10)
                        )
                        and np.any(
                            ndimage.binary_dilation(
                                np.asarray(
                                    scene.tissue.anchor_masks[anchor_id],
                                    dtype=bool,
                                ),
                                structure=np.ones((3, 3), dtype=bool),
                            )
                            & (np.asarray(source_tissue) == 2)
                        )
                    )
                anchor_groups = (
                    [
                        anchors,
                        *((anchor_id,) for anchor_id in anchors[:2]),
                    ]
                    if (prostate_scatter or exhaustive_peritumoral) and anchors
                    else [(anchor_id,) for anchor_id in anchors[:2]]
                )
                for anchor_ids in dict.fromkeys(anchor_groups):
                    variants.append(
                        {
                            "joint_interface_ids": [interface.interface_id],
                            "joint_anchor_ids": list(anchor_ids),
                        }
                    )
        if not variants:
            raise JointContractError(
                "cell-only feasibility compiler found no legal zone/interface choice"
            )

        planner = HeuristicJointPlanner()
        choice_payloads: list[tuple[dict[str, Any], Any, Any]] = []
        seen_sha: set[str] = set()
        vetoes: list[CellPlanCandidateVeto] = []
        for index, provenance_update in enumerate(variants):
            compiler_anchor_payload = provenance_update.get(
                "cellularity_depletion_anchor"
            )
            case_provenance = dict(case.provenance)
            if case.pathology_domain_id != "breast-invasive-carcinoma-v1":
                case_provenance.pop("cellularity_depletion_anchor", None)
            case_provenance.update(
                {
                    key: value
                    for key, value in provenance_update.items()
                    if (
                        key != "cellularity_depletion_anchor"
                        or case.pathology_domain_id
                        == "breast-invasive-carcinoma-v1"
                    )
                }
            )
            candidate_case = replace(
                case,
                provenance=case_provenance,
            )
            compiler_anchor = None
            if isinstance(compiler_anchor_payload, Mapping):
                compiler_anchor = CompilerOwnedDepletionAnchor.issue(
                    case=candidate_case,
                    zone_id=str(candidate_case.provenance["joint_population_zone_id"]),
                    interface_ids=compiler_anchor_payload.get(
                        "interface_ids", ()
                    ),
                    anchor_ids=compiler_anchor_payload.get("anchor_ids", ()),
                )
            try:
                plan, _usage = planner.create_plan(
                    case=candidate_case,
                    scene=scene,
                    bundle=bundle,
                    tissue_plan=None,
                    image_paths=(),
                    compiler_owned_depletion_anchor=compiler_anchor,
                )
                interface_id = (
                    plan.cell_plan.interface_ids[0]
                    if plan.cell_plan.interface_ids
                    else plan.cell_plan.core_zone
                )
                preserved = CandidateMask(
                    candidate_id=f"tissue-preserved-portfolio-{index}",
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
                report = GateReport(
                    candidate_id=preserved.candidate_id,
                    passed=True,
                    checks=(),
                )
                contract = self.executable_contract_compiler.compile(
                    case=candidate_case,
                    source_tissue=source_tissue,
                    source_nuclei=source_nuclei,
                    schema=schema,
                    scene=scene,
                    plan=plan,
                    bundle=bundle,
                    tissue_candidate=preserved,
                    tissue_gate_report=report,
                    allocation=None,
                    required_checker_ids=(
                        self.joint_gates.required_checker_ids_for(bundle)
                    ),
                )
                preflight = self._certify_cell_only_executable_capacity(
                    case=candidate_case,
                    source_nuclei=source_nuclei,
                    scene=scene,
                    bundle=bundle,
                    plan=plan,
                    tissue_candidate=preserved,
                    executable_contract=contract,
                )
                packing = preflight.exact_packing_certificate
                if int(packing.get("requested_count", 0)) > 0:
                    contract = contract.bind_packing_certificate(packing)
                placements = packing.get("placements", ())
                center_rows = [int(item["row"]) for item in placements]
                center_cols = [int(item["col"]) for item in placements]
                median_distance = 0.0
                if center_rows:
                    tumor = np.isin(
                        source_tissue,
                        tuple(schema.resolve_fine_ids("Tumor")),
                    )
                    distance = ndimage.distance_transform_edt(~tumor)
                    median_distance = float(
                        np.median(distance[center_rows, center_cols])
                    )
                nominal_requested_count = int(
                    packing.get(
                        "nominal_requested_count",
                        preflight.required_add_count,
                    )
                )
                packing_margin = float(
                    preflight.estimated_add_capacity
                    - nominal_requested_count
                )
                executable_packing_margin = float(
                    preflight.estimated_add_capacity
                    - preflight.required_add_count
                )
                placement_mask = np.zeros_like(source_tissue, dtype=bool)
                if center_rows:
                    placement_mask[center_rows, center_cols] = True
                focus_radius = max(
                    1,
                    int(
                        np.ceil(
                            contract.cell_program.nominal_nucleus_diameter_px
                            * 1.25
                        )
                    ),
                )
                focus_labels, focus_count = ndimage.label(
                    ndimage.binary_dilation(
                        placement_mask,
                        iterations=focus_radius,
                    ),
                    structure=np.ones((3, 3), dtype=bool),
                )
                del focus_labels
                valid_region = np.asarray(
                    contract.cell_program.valid_footprint_region,
                    dtype=bool,
                )
                protected_distance = ndimage.distance_transform_edt(valid_region)
                minimum_protected_distance = (
                    float(np.min(protected_distance[placement_mask]))
                    if np.any(placement_mask)
                    else 0.0
                )
                tumor_population = np.asarray(source_nuclei) == 1
                bridge_zone = ndimage.binary_dilation(
                    tumor_population,
                    iterations=max(
                        1,
                        int(
                            np.ceil(
                                contract.cell_program.nominal_nucleus_diameter_px
                            )
                        ),
                    ),
                )
                bridge_risk = int(
                    np.count_nonzero(placement_mask & bridge_zone)
                )
                structural_risk = int(
                    np.count_nonzero(
                        placement_mask
                        & ~np.asarray(
                            contract.cell_program.placement_center_region,
                            dtype=bool,
                        )
                    )
                )
                if (
                    case.primitive_id
                    == "peritumoral-small-cluster-increase-v1"
                ):
                    (
                        localized_multifocus_capacity,
                        localized_front_span_window_margin,
                    ) = _localized_focus_capacity_metrics(
                        center_rows=center_rows,
                        center_cols=center_cols,
                        nominal_nucleus_diameter_px=(
                            contract.cell_program.nominal_nucleus_diameter_px
                        ),
                        minimum_effect_span_px=(
                            contract.cell_program.minimum_effect_span_px
                        ),
                        required_focus_count=max(
                            SMALL_CLUSTER_TARGET_FOCUS_COUNT,
                            int(contract.cell_program.minimum_effect_foci),
                        ),
                        minimum_anchor_separation_diameters=(
                            BREAST_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS
                            if bundle.mechanism.mechanism_id
                            == "breast-peritumoral-small-cluster"
                            else GLAS_SMALL_CLUSTER_MINIMUM_ANCHOR_SEPARATION_DIAMETERS
                            if case.annotation_profile_id == "glas-gland-v1"
                            else SMALL_CLUSTER_BETWEEN_FOCUS_SEPARATION_DIAMETERS
                        ),
                        strict_breast_small_cluster=(
                            bundle.mechanism.mechanism_id
                            == "breast-peritumoral-small-cluster"
                        ),
                    )
                else:
                    localized_multifocus_capacity = 0
                    localized_front_span_window_margin = 0.0
                metrics = {
                    "certificate_capacity_margin": packing_margin,
                    "executable_capacity_margin": (
                        executable_packing_margin
                    ),
                    "packing_seam_capacity_margin": float(
                        preflight.estimated_seam_capacity
                        - preflight.required_seam_count
                    ),
                    "separated_focus_capacity": float(focus_count),
                    "median_tumor_distance_px": median_distance,
                    "complete_shape_packing_margin": packing_margin,
                    "bridge_risk_count": float(bridge_risk),
                    "structural_risk_count": float(structural_risk),
                    "protected_distance_px": minimum_protected_distance,
                    "localized_multifocus_capacity": float(
                        localized_multifocus_capacity
                    ),
                    "localized_front_span_window_margin": float(
                        localized_front_span_window_margin
                    ),
                }
                candidate_payload = {
                    "plan": plan,
                    "deterministic_candidate_metrics": metrics,
                    "allowed_tool_program_ids": (
                        contract.execution_program_id,
                    ),
                    "executable_contract_id": contract.contract_id,
                }
                provisional_sha = hashlib.sha256(
                    json.dumps(
                        {
                            "plan": plan.to_metadata(),
                            "metrics": metrics,
                            "program": contract.execution_program_id,
                            "contract": contract.contract_id,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                if provisional_sha in seen_sha:
                    continue
                seen_sha.add(provisional_sha)
                choice_payloads.append((candidate_payload, contract, preflight))
            except (JointContractError, RefineContractError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                depletion_anchor = provenance_update.get(
                    "cellularity_depletion_anchor"
                )
                if not isinstance(depletion_anchor, Mapping):
                    depletion_anchor = {}
                interface_ids = tuple(
                    str(value)
                    for value in provenance_update.get(
                        "joint_interface_ids",
                        depletion_anchor.get("interface_ids", ()),
                    )
                )
                anchor_ids = tuple(
                    str(value)
                    for value in provenance_update.get(
                        "joint_anchor_ids",
                        depletion_anchor.get("anchor_ids", ()),
                    )
                )
                zone_id = provenance_update.get("joint_population_zone_id")
                veto_payload = {
                    "variant_index": index,
                    "interface_ids": list(interface_ids),
                    "anchor_ids": list(anchor_ids),
                    "zone_id": zone_id,
                    "veto_reasons": [error],
                }
                veto_sha = hashlib.sha256(
                    json.dumps(
                        veto_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                vetoes.append(
                    CellPlanCandidateVeto(
                        candidate_id="cell-veto:" + veto_sha[:20],
                        bound_interface_ids=interface_ids,
                        bound_anchor_ids=anchor_ids,
                        bound_zone_id=(
                            str(zone_id) if zone_id is not None else None
                        ),
                        veto_reasons=(error,),
                        compiler_certificate_sha256=veto_sha,
                    )
                )
        if not choice_payloads:
            raise JointContractError(
                "cell-only pre-LLM portfolio has no exact-capacity survivor: "
                + "; ".join(
                    (
                        f"[{item.candidate_id} zone={item.bound_zone_id} "
                        f"interfaces={','.join(item.bound_interface_ids) or '-'} "
                        f"anchors={','.join(item.bound_anchor_ids) or '-'}] "
                        f"{reason}"
                    )
                    for item in vetoes
                    for reason in item.veto_reasons
                )
            )
        authority_binding = _cell_portfolio_authority_binding(
            case=case,
            source_tissue=source_tissue,
            source_nuclei=source_nuclei,
            bundle=bundle,
        )
        certificates = _issue_cell_plan_portfolio(
            candidates=tuple(item[0] for item in choice_payloads),
            vetoed=tuple(vetoes),
            authority_binding=authority_binding,
        )
        choices = tuple(
            _CertifiedCellExecutionChoice(
                certificate=certificate,
                executable_contract=payload[1],
                preflight=payload[2],
            )
            for certificate, payload in zip(
                certificates.survivors, choice_payloads
            )
        )
        return _CertifiedCellExecutionPortfolio(
            choices=choices,
            certificates=certificates,
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
        clarification_request=None,
    ):
        summary = {
            "status": status,
            "selected_candidate_id": selected,
            "abstain_reasons": list(reasons),
            "usage": usage,
            "clarification_request": clarification_request,
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
            clarification_request=clarification_request,
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
    maximum_pixels: int,
) -> bool:
    """Rebalance when every exact paired witness exceeds a target ceiling.

    Candidate feasibility has already compiled complete source erasure ``E``
    and a concrete target-footprint packing witness ``F``.  The ledger is the
    executable ``T ∪ E ∪ F``, so it may safely optimize the desired
    ceiling before ProbNet ranking.  Replanning remains candidate-preserving:
    one witness at or below the ceiling is enough to continue execution.
    """

    values = [max(0, int(value)) for value in predicted_pixels]
    return bool(values and min(values) > int(maximum_pixels))


def _retain_visible_regression_whole_instance_closure(
    *,
    annotation_profile_id: str = "",
    primitive_id: str,
    fallback_policy: str,
    predicted_pixels,
    desired_max_pixels: int,
    hard_max_pixels: int,
) -> bool:
    """Execute a valid visible-regression closure before shrinking its mask.

    Whole-instance cell removal can move an otherwise valid footprint or
    fragmentation candidate slightly above the desired joint interval.
    Shrinking the tissue edit first weakens a visible external retreat or can
    destroy the only fragmentation seam that supports replacement-cell
    packing.  When the exact predicted closure remains inside the declared
    hard maximum, execute it and let the existing post-execution minimum-safe
    fallback certify the raster instead of pre-emptively shrinking it.
    """

    values = [max(0, int(value)) for value in predicted_pixels]
    return bool(
        (
            primitive_id
            in {
                "invasive-tumor-footprint-decrease-v1",
                "residual-tumor-fragmentation-v1",
            }
            or (
                annotation_profile_id == "panda-gleason-v1"
                and primitive_id
                in {
                    "tumor-burden-increase-v1",
                    "cohesive-boundary-expansion-v1",
                }
            )
        )
        and fallback_policy == "max_feasible_below_target"
        and values
        and int(desired_max_pixels) < min(values) <= int(hard_max_pixels)
    )


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


def _deterministic_feedback_signature(
    feedback: Mapping[str, Any],
) -> str | None:
    """Canonicalize evidence that should change before another costly retry."""

    stage = str(feedback.get("stage") or "")
    if stage not in {
        "planning_or_compilation",
        "tissue_gate",
        "cell_feasibility",
        "cell_execution",
    }:
        return None
    payload = {
        "stage": stage,
        "errors": sorted(str(item) for item in feedback.get("errors", ())),
        "hard_tissue_failure_counts": dict(
            sorted(
                (str(key), int(value))
                for key, value in dict(
                    feedback.get("hard_tissue_failure_counts") or {}
                ).items()
            )
        ),
        "cell_feasibility_failure_counts": dict(
            sorted(
                (str(key), int(value))
                for key, value in dict(
                    feedback.get("cell_feasibility_failure_counts") or {}
                ).items()
            )
        ),
        "failed_interface_ids": sorted(
            str(item) for item in feedback.get("failed_interface_ids", ())
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _canonical_metadata_sha256(value: Any) -> str:
    return canonical_metadata_sha256(value)


def _tissue_portfolio_authority_binding(
    *,
    case,
    tissue_case,
    source_tissue,
    bundle,
    tissue_bundle,
    allocation,
    nuclei_preflight,
) -> dict[str, Any]:
    """Bind a compiler portfolio to exact case, skills, budget and preflight."""

    return build_tissue_portfolio_authority_binding(
        joint_case=case,
        tissue_case=tissue_case,
        source_tissue=source_tissue,
        joint_bundle=bundle,
        tissue_bundle=tissue_bundle,
        allocation=allocation,
        nuclei_preflight=nuclei_preflight,
    )


def _cell_portfolio_authority_binding(
    *,
    case,
    source_tissue,
    source_nuclei,
    bundle,
) -> dict[str, Any]:
    return build_cell_portfolio_authority_binding(
        case=case,
        source_tissue=source_tissue,
        source_nuclei=source_nuclei,
        joint_bundle=bundle,
    )


def _as_tissue_case(case: JointCaseContext, *, allocation, shape) -> CaseContext:
    total = int(np.prod(shape))
    target = allocation.tissue_target_pixels / max(1, total)
    # Capacity-adaptive tasks intentionally bind the tissue compiler to the
    # manifest's minimum-effective floor. The joint gates separately require
    # proof that any result below the standard tissue floor is the maximum
    # topology-safe fallback, so this does not make an arbitrary small edit a
    # success.
    floor_pixels = int(allocation.tissue_execution_floor_pixels)
    if int(allocation.tissue_target_pixels) < floor_pixels:
        raise JointContractError(
            "joint budget allocation placed the tissue target below the "
            f"binding meaningful floor: target={allocation.tissue_target_pixels}, "
            f"floor={floor_pixels}"
        )
    floor = floor_pixels / max(1, total)
    maximum = target
    if (
        case.annotation_profile_id == "panda-gleason-v1"
        and case.primitive_id == "residual-tumor-fragmentation-v1"
    ):
        # PANDA fragmentation owns a component-relative 12% breakup floor.
        # Permit the tissue compiler to rise above the nominal 3.5% target,
        # but never beyond the frozen joint task maximum (5%).
        maximum = allocation.joint_hard_max_pixels / max(1, total)
    budget = AreaBudget(
        target_fraction=target,
        min_fraction=floor,
        max_fraction=maximum,
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


def _apply_glas_visible_cell_budget(
    case: JointCaseContext,
    *,
    primitive_id: str,
    budget: CellCountExtentBudget,
    metadata: dict[str, Any],
) -> tuple[CellCountExtentBudget, dict[str, Any]]:
    """Apply the reviewed post-7c6 GLaS saliency budgets.

    Explicit manifest budgets remain authoritative because this helper is used
    only after the workflow derives a missing budget.  Sparse cases that cannot
    reach the new minimum fail deterministic preflight rather than presenting a
    tiny edit as success.
    """

    if case.annotation_profile_id != "glas-gland-v1":
        return budget, metadata

    local_population = {
        "cell-type-abundance-increase-v1",
        "cell-type-abundance-decrease-v1",
        "cellularity-increase-v1",
        "cellularity-decrease-v1",
        "neoplastic-cell-abundance-increase-v1",
        "neoplastic-cell-abundance-decrease-v1",
    }
    if primitive_id in local_population:
        visible = CellCountExtentBudget(
            target_delta_count=20,
            min_delta_count=12,
            max_delta_count=28,
            maximum_extent_px=max(384, budget.maximum_extent_px),
            interface_min_px=0,
            interface_max_px=max(48, budget.interface_max_px),
            minimum_effect_span_px=0,
            minimum_effect_foci=0,
        )
        if primitive_id == "cellularity-decrease-v1":
            visible = CellCountExtentBudget(
                target_delta_count=16,
                min_delta_count=12,
                max_delta_count=24,
                maximum_extent_px=max(384, budget.maximum_extent_px),
                interface_min_px=0,
                interface_max_px=max(48, budget.interface_max_px),
                minimum_effect_span_px=0,
                minimum_effect_foci=0,
            )
    elif primitive_id == "peritumoral-neoplastic-scatter-increase-v1":
        visible = CellCountExtentBudget(
            target_delta_count=10,
            min_delta_count=4,
            max_delta_count=14,
            maximum_extent_px=max(144, budget.maximum_extent_px),
            interface_min_px=max(4, budget.interface_min_px),
            interface_max_px=max(48, budget.interface_max_px),
            minimum_effect_span_px=max(32, budget.minimum_effect_span_px),
            minimum_effect_foci=max(4, budget.minimum_effect_foci),
        )
    elif primitive_id == "peritumoral-small-cluster-increase-v1":
        visible = CellCountExtentBudget(
            target_delta_count=8,
            min_delta_count=6,
            max_delta_count=12,
            maximum_extent_px=max(160, budget.maximum_extent_px),
            interface_min_px=max(4, budget.interface_min_px),
            interface_max_px=max(48, budget.interface_max_px),
            minimum_effect_span_px=max(32, budget.minimum_effect_span_px),
            minimum_effect_foci=max(2, budget.minimum_effect_foci),
        )
    else:
        return budget, metadata

    return visible, {
        **metadata,
        "pre_scale_budget": budget.__dict__,
        "policy_id": "glas-visible-cell-effect-budget-v3-feasible",
        "authority": "system_owned_profile_specific_budget",
        "budget": visible.__dict__,
    }


def _apply_panda_profile_cell_budget(
    case: JointCaseContext,
    *,
    primitive_id: str,
    budget: CellCountExtentBudget,
    metadata: dict[str, Any],
) -> tuple[CellCountExtentBudget, dict[str, Any]]:
    """Keep PANDA cell edits meaningful without over-scaling small patches."""

    if (
        case.annotation_profile_id != "panda-gleason-v1"
        or primitive_id not in PANDA_CELL_EFFECT_EXTENT_OVERRIDES
    ):
        return budget, metadata
    meaningful_count = int(metadata.get("skill_minimum_effect_delta_count", 0))
    if meaningful_count <= 0:
        return budget, metadata
    calibrated = replace(
        budget,
        target_delta_count=meaningful_count,
        min_delta_count=meaningful_count,
        max_delta_count=max(
            meaningful_count,
            min(
                budget.max_delta_count,
                int(np.ceil(meaningful_count * 1.5)),
            ),
        ),
    )
    return calibrated, {
        **metadata,
        "pre_profile_budget": budget.__dict__,
        "policy_id": "panda-profile-cell-effect-budget-v1",
        "authority": "system_owned_profile_specific_budget",
        "budget": calibrated.__dict__,
    }


def _derive_infiltration_budget(
    scene,
    *,
    minimum_effect_delta_count: int = 0,
    minimum_effect_span_cell_diameters: float = 0.0,
    minimum_effect_foci: int = 0,
) -> tuple[CellCountExtentBudget, dict[str, Any]]:
    """Derive a scale-aware cell budget for a contextual tumor interpretation.

    This policy is deterministic and auditable. It uses source complete-nucleus
    scale plus the amount of visible tumor/non-tumor interface; neither the
    Semantic Parser nor the mask-graph Planner is allowed to invent a cell count.
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
    target = max(
        int(minimum_effect_delta_count),
        int(np.clip(estimated_slots, 4, 24)),
    )
    minimum = max(
        1,
        int(np.floor(target * 0.5)),
        int(minimum_effect_delta_count),
    )
    maximum = max(target, min(32, int(np.ceil(target * 1.5))))
    minimum_effect_span = int(
        np.floor(float(minimum_effect_span_cell_diameters) * diameter_px)
    )
    patch_support_limit = max(
        48,
        int(np.floor(0.40 * min(np.asarray(scene.source_nuclei).shape))),
    )
    if minimum_effect_span > patch_support_limit:
        raise JointContractError(
            "source-calibrated infiltration span exceeds the patch-relative "
            "bounded-support limit"
        )
    maximum_extent = min(
        patch_support_limit,
        max(
            minimum_effect_span,
            int(np.clip(round(5.0 * diameter_px), 24, 96)),
        ),
    )
    budget = CellCountExtentBudget(
        target_delta_count=target,
        min_delta_count=minimum,
        max_delta_count=maximum,
        maximum_extent_px=maximum_extent,
        interface_min_px=2,
        interface_max_px=maximum_extent,
        minimum_effect_span_px=minimum_effect_span,
        minimum_effect_foci=int(minimum_effect_foci),
    )
    return budget, {
        "policy_id": "contextual-infiltration-budget-v1",
        "authority": "deterministic_source_scene",
        "complete_reference_count": len(complete_areas),
        "median_complete_nucleus_area_px": median_area,
        "estimated_nucleus_diameter_px": diameter_px,
        "unique_tumor_interface_pixels": interface_pixels,
        "skill_minimum_effect_delta_count": int(
            minimum_effect_delta_count
        ),
        "skill_minimum_effect_span_cell_diameters": float(
            minimum_effect_span_cell_diameters
        ),
        "skill_minimum_effect_foci": int(minimum_effect_foci),
        "budget": budget.__dict__,
    }


def _derive_local_population_budget(
    scene,
    *,
    primitive_id: str,
    semantic_intent: Mapping[str, Any],
    annotation_profile_id: str = "",
    host_tissue_labels: tuple[str, ...] = (),
    minimum_effect_delta_count: int = 0,
    minimum_effect_span_cell_diameters: float = 0.0,
    minimum_effect_foci: int = 0,
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
    abundance = primitive_id.startswith(
        ("cell-type-abundance-", "generic-inflammatory-cell-abundance-")
    )
    if abundance and len(class_ids) != 1:
        raise JointContractError(
            "cell abundance budget requires exactly one observable class"
        )

    component_labels = {
        item.component_id: item.label
        for item in scene.tissue.graph.components
    }
    protected_lumen = None
    external_cellular_stroma = None
    if annotation_profile_id == "panda-gleason-v1":
        protected_lumen = scene.auxiliary_structure_masks.get(
            "gland_lumen_map"
        )
        external_cellular_stroma = scene.auxiliary_structure_masks.get(
            "external_cellular_stroma_map"
        )

    def legal_zone_mask(zone) -> np.ndarray:
        mask = np.asarray(
            scene.population_zone_masks[zone.zone_id], dtype=bool
        ).copy()
        if protected_lumen is not None:
            mask &= ~np.asarray(protected_lumen, dtype=bool)
        if (
            primitive_id.endswith("increase-v1")
            and component_labels.get(zone.tissue_component_id) == "Stroma"
            and external_cellular_stroma is not None
        ):
            mask &= np.asarray(external_cellular_stroma, dtype=bool)
        return mask

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
            and not np.any(
                np.asarray(scene.instance_masks[item.instance_id], dtype=bool)
                & ~legal_zone_mask(zone)
            )
        )
        for zone in component_zones
    }
    coarse_capacity_by_zone = {}
    for zone in component_zones:
        zone_mask = legal_zone_mask(zone)
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
            zone_mask = legal_zone_mask(zone)
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
    required_capacity = max(2, int(minimum_effect_delta_count))
    if executable_capacity < required_capacity:
        direction = "increase exact packing" if increase else "decrease complete-instance"
        raise JointContractError(
            f"cell {direction} capacity is below the skill-owned meaningful "
            f"minimum: {executable_capacity} < {required_capacity}"
        )
    if increase:
        source_scaled = max(2, round(max(local_authority_count, 10) * 0.20))
        target = int(
            min(
                32,
                executable_capacity,
                max(source_scaled, int(minimum_effect_delta_count)),
            )
        )
    else:
        source_scaled = max(2, round(local_authority_count * 0.20))
        target = int(
            min(
                32,
                executable_capacity,
                max(source_scaled, int(minimum_effect_delta_count)),
            )
        )
    minimum = max(
        1,
        int(np.floor(target * 0.60)),
        int(minimum_effect_delta_count),
    )
    maximum = max(target, min(40, int(np.ceil(target * 1.35))))
    minimum_effect_span = int(
        np.floor(float(minimum_effect_span_cell_diameters) * diameter_px)
    )
    patch_support_limit = max(
        48,
        int(np.floor(0.40 * min(np.asarray(scene.source_nuclei).shape))),
    )
    span_capped_to_patch_support = False
    if (
        minimum_effect_span > patch_support_limit
        and annotation_profile_id == "panda-gleason-v1"
    ):
        # PANDA patches can contain unusually large source nuclei.  The
        # primitive's cell-diameter span is a meaningful-effect target, but
        # it cannot exceed the profile's explicit bounded-support envelope.
        # Clamp only this profile to that already-authoritative envelope; the
        # post-execution span gate still verifies the full bounded effect.
        minimum_effect_span = patch_support_limit
        span_capped_to_patch_support = True
    if minimum_effect_span > patch_support_limit:
        raise JointContractError(
            "source-calibrated meaningful cell effect span exceeds the "
            "patch-relative bounded-support limit"
        )
    footprint_span_margin = (
        int(np.ceil(2.0 * diameter_px))
        if annotation_profile_id == "panda-gleason-v1"
        and increase
        and minimum_effect_span > 0
        else 0
    )
    maximum_extent = max(
        minimum_effect_span + footprint_span_margin,
        int(np.clip(round(6.0 * diameter_px), 48, 96)),
    )
    maximum_extent = min(patch_support_limit, maximum_extent)
    budget = CellCountExtentBudget(
        target_delta_count=target,
        min_delta_count=minimum,
        max_delta_count=maximum,
        maximum_extent_px=maximum_extent,
        interface_min_px=0,
        interface_max_px=maximum_extent,
        minimum_effect_span_px=minimum_effect_span,
        minimum_effect_foci=int(minimum_effect_foci),
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
        "skill_minimum_effect_delta_count": int(
            minimum_effect_delta_count
        ),
        "skill_minimum_effect_span_cell_diameters": float(
            minimum_effect_span_cell_diameters
        ),
        "skill_minimum_effect_foci": int(minimum_effect_foci),
        "patch_relative_support_limit_px": patch_support_limit,
        "span_capped_to_patch_support": span_capped_to_patch_support,
        "complete_footprint_span_margin_px": footprint_span_margin,
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
