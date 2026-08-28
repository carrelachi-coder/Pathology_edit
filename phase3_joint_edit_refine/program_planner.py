"""Resolve primitive-free semantic intents into an auditable edit program."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

from .clarification import PlannerClarificationRequired
from .models import JointCaseContext, JointContractError
from .planner import HeuristicJointPlanner, JointInterpretationOption
from .semantic_request import SemanticIntentClause, SemanticRequest
from .skills.repository import JointSkillRepository


EDIT_PROGRAM_SCHEMA_VERSION = "joint-edit-program-v1"


@dataclass(frozen=True)
class PrimitiveCandidate:
    primitive_id: str
    semantic_priority: int
    semantic_fit: str
    rationale: str
    compatible_mechanism_ids: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EditProgramStep:
    step_id: str
    intent_id: str
    order_index: int
    source_text: str
    depends_on: tuple[str, ...]
    candidates: tuple[PrimitiveCandidate, ...]
    selected_primitive_id: str | None
    selected_mechanism_id: str | None = None
    status: str = "planned"
    selection_rationale: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            "planned",
            "requires_mask_resolution",
            "clarification_required",
            "review_required",
            "selected",
            "validated",
            "failed",
            "not_run",
        }:
            raise JointContractError("edit-program step has an unsupported status")
        candidate_ids = {item.primitive_id for item in self.candidates}
        if self.selected_primitive_id is not None and (
            self.selected_primitive_id not in candidate_ids
        ):
            raise JointContractError(
                "selected primitive is absent from the step candidate set"
            )

    def to_metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "candidates": [item.to_metadata() for item in self.candidates],
        }


@dataclass(frozen=True)
class EditProgram:
    request_sha256: str
    instruction: str
    steps: tuple[EditProgramStep, ...]
    status: str
    global_constraints: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    schema_version: str = EDIT_PROGRAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EDIT_PROGRAM_SCHEMA_VERSION:
            raise JointContractError("unsupported edit-program schema")
        if self.status not in {
            "ready",
            "requires_mask_resolution",
            "clarification_required",
            "review_required",
            "running",
            "validated",
            "partially_validated",
            "failed",
        }:
            raise JointContractError("edit program has an unsupported status")
        step_ids = tuple(item.step_id for item in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise JointContractError("edit program contains duplicate step IDs")

    @property
    def program_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_metadata(include_digest=False),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def to_metadata(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "request_sha256": self.request_sha256,
            "instruction": self.instruction,
            "status": self.status,
            "steps": [item.to_metadata() for item in self.steps],
            "global_constraints": list(self.global_constraints),
            "conflicts": list(self.conflicts),
        }
        if include_digest:
            result["program_sha256"] = self.program_sha256
        return result


class PrimitiveResolver:
    """Closed semantic lookup followed by static four-axis capability filtering."""

    def __init__(self, repository: JointSkillRepository | None = None) -> None:
        self.repository = repository or JointSkillRepository()
        self._capability_rows = self.repository.capability_matrix()

    def resolve(
        self,
        intent: SemanticIntentClause,
        *,
        case_template: JointCaseContext,
        production: bool = False,
    ) -> tuple[PrimitiveCandidate, ...]:
        if intent.polarity == "negated":
            return ()
        raw = self._semantic_candidates(intent)
        normalized = self._normalize_profile_aliases(
            raw,
            annotation_profile_id=case_template.annotation_profile_id,
        )
        result: list[PrimitiveCandidate] = []
        for candidate in normalized:
            if candidate.primitive_id not in self.repository.executable_primitive_ids:
                continue
            mechanism_ids = self._statically_compatible_mechanisms(
                primitive_id=candidate.primitive_id,
                case_template=case_template,
                production=production,
            )
            if not mechanism_ids:
                continue
            result.append(
                replace(
                    candidate,
                    compatible_mechanism_ids=mechanism_ids,
                )
            )
        return tuple(result)

    def _statically_compatible_mechanisms(
        self,
        *,
        primitive_id: str,
        case_template: JointCaseContext,
        production: bool,
    ) -> tuple[str, ...]:
        """Filter catalog capabilities without pretending to preflight a mask.

        Budgets, auxiliary materialization, instance authority and geometric
        capacity are current-state questions and remain owned by the existing
        workflow preflight.  The program Planner performs only the static four-
        axis intersection here, so it must not reject a cell edit merely because
        the workflow has not derived its count budget yet.
        """

        annotation = self.repository.annotation_profiles.get(
            case_template.annotation_profile_id
        )
        observation = self.repository.cell_observation_profiles.get(
            case_template.cell_observation_profile_id
        )
        population = self.repository.cell_population_profiles.get(
            case_template.cell_population_profile_id
        )
        if annotation is None or observation is None or population is None:
            return ()
        if population.pathology_domain_id != case_template.pathology_domain_id:
            return ()
        result: list[str] = []
        for row in self._capability_rows:
            if (
                row["pathology_domain_id"] != case_template.pathology_domain_id
                or row["annotation_profile_id"]
                != case_template.annotation_profile_id
                or primitive_id not in row["supported_primitives"]
                or row["status"] in {"unsupported", "render_only"}
            ):
                continue
            mechanism_id = str(row["mechanism_id"])
            if self.repository.execution_selection_reason(
                primitive_id=primitive_id,
                mechanism_id=mechanism_id,
            ) is not None:
                continue
            mechanism = self.repository.mechanisms[mechanism_id]
            if set(mechanism.representability.required_cell_classes) - set(
                observation.class_ids
            ):
                continue
            if set(mechanism.cell_program.allowed_cell_classes) - set(
                population.allowed_cell_classes
            ):
                continue
            if production:
                primitive = self.repository.primitives[primitive_id]
                evidence = (
                    self.repository.skill_evidence_status[
                        f"edit-primitive:{primitive_id}"
                    ],
                    self.repository.skill_evidence_status[
                        f"joint-mechanism:{mechanism_id}"
                    ],
                    self.repository.skill_evidence_status[
                        f"annotation-profile:{case_template.annotation_profile_id}"
                    ],
                )
                resource = self.repository.mechanism_resource_status.get(
                    mechanism_id, {}
                )
                if (
                    primitive.review_status != "internally_reviewed"
                    or mechanism.review_status != "internally_reviewed"
                    or annotation.review_status != "internally_reviewed"
                    or not all(item.production_allowed for item in evidence)
                    or resource.get("statistics_status") != "calibrated"
                    or resource.get("production_allowed") is not True
                ):
                    continue
            result.append(mechanism_id)
        return tuple(sorted(set(result)))

    def _semantic_candidates(
        self, intent: SemanticIntentClause
    ) -> tuple[PrimitiveCandidate, ...]:
        target = intent.target
        operation = intent.operation
        morphology = intent.morphology
        explicit = intent.intent_type == "direct_edit"
        fit = "explicit" if explicit else "contextual"

        def candidates(*items: tuple[str, int, str]) -> tuple[PrimitiveCandidate, ...]:
            return tuple(
                PrimitiveCandidate(
                    primitive_id=primitive_id,
                    semantic_priority=priority,
                    semantic_fit=fit,
                    rationale=rationale,
                )
                for primitive_id, priority, rationale in items
            )

        if target == "tumor_extent" and operation == "increase":
            if morphology == "cohesive" or intent.spatial_scope == "boundary":
                return candidates(
                    (
                        "cohesive-boundary-expansion-v1",
                        0,
                        "the intent explicitly requests cohesive boundary expansion",
                    ),
                )
            return candidates(
                (
                    "tumor-burden-increase-v1",
                    0,
                    "the intent requests a larger tumor extent without a morphology",
                ),
                (
                    "cohesive-boundary-expansion-v1",
                    1,
                    "cohesive expansion is the bounded fallback for a larger extent",
                ),
            )
        if target == "tumor_extent" and operation == "decrease":
            return candidates(
                (
                    "invasive-tumor-footprint-decrease-v1",
                    0,
                    "the intent requests coherent reduction of invasive tumor extent",
                ),
            )
        if target == "tumor_extent" and operation == "clear":
            return candidates(
                (
                    "local-invasive-clearance-v1",
                    0,
                    "the intent explicitly requests local tumor clearance",
                ),
            )
        if target == "tumor_topology" and operation == "fragment":
            return candidates(
                (
                    "residual-tumor-fragmentation-v1",
                    0,
                    "the intent requests disconnected residual tumor foci",
                ),
            )
        if target == "stroma" and operation == "increase":
            return candidates(
                (
                    "stroma-increase-v1",
                    0,
                    "the intent explicitly requests stromal replacement",
                ),
            )
        if target == "necrosis" and operation == "appear":
            return candidates(
                (
                    "necrosis-appearance-v1",
                    0,
                    "the intent requests new intratumoral necrosis",
                ),
            )
        if target == "necrosis" and operation == "repopulate":
            return candidates(
                (
                    "necrosis-resolution-v1",
                    0,
                    "the intent requests viable tumor repopulation of necrosis",
                ),
            )
        if target == "neoplastic_cell_population":
            primitive = {
                "increase": "neoplastic-cell-abundance-increase-v1",
                "decrease": "neoplastic-cell-abundance-decrease-v1",
            }.get(operation)
            return (
                candidates(
                    (primitive, 0, "the intent changes only neoplastic-cell abundance"),
                )
                if primitive
                else ()
            )
        if target == "selected_cell_population":
            primitive = {
                "increase": "cell-type-abundance-increase-v1",
                "decrease": "cell-type-abundance-decrease-v1",
            }.get(operation)
            return (
                candidates(
                    (primitive, 0, "the intent changes one explicitly named cell class"),
                )
                if primitive
                else ()
            )
        if target == "overall_cellularity":
            primitive = {
                "increase": "cellularity-increase-v1",
                "decrease": "cellularity-decrease-v1",
            }.get(operation)
            return (
                candidates(
                    (primitive, 0, "the intent changes overall local cellularity"),
                )
                if primitive
                else ()
            )
        if target == "immune_compartment":
            primitive = {
                "increase": "generic-immune-infiltrate-increase-v1",
                "decrease": "generic-immune-infiltrate-decrease-v1",
            }.get(operation)
            return (
                candidates(
                    (primitive, 0, "the intent changes the immune-infiltrate compartment"),
                )
                if primitive
                else ()
            )
        if target == "invasion_pattern" and operation == "increase":
            by_morphology = {
                "cord": "invasive-cord-formation-v1",
                "nest": "peritumoral-tumor-nest-formation-v1",
                "single_file": "neoplastic-microinfiltration-increase-v1",
                "single_cell": "peritumoral-neoplastic-scatter-increase-v1",
                "small_cluster": "peritumoral-small-cluster-increase-v1",
            }
            if morphology == "nest_cord":
                return candidates(
                    (
                        "infiltrative-nest-cord-extension-v1",
                        0,
                        "the intent explicitly requests a mixed nest-cord extension",
                    ),
                    (
                        "invasive-cord-formation-v1",
                        1,
                        "a pure invasive cord is the bounded contextual fallback",
                    ),
                    (
                        "peritumoral-tumor-nest-formation-v1",
                        1,
                        "a discrete tumor nest is the bounded contextual fallback",
                    ),
                )
            if morphology == "invasive_front":
                return candidates(
                    (
                        "invasive-front-expansion-v1",
                        0,
                        "the intent explicitly requests expansion of an invasive front",
                    ),
                    (
                        "infiltrative-nest-cord-extension-v1",
                        0,
                        "an infiltrative nest-cord front is an organ-specific realization",
                    ),
                    (
                        "cohesive-boundary-expansion-v1",
                        1,
                        "cohesive boundary expansion is a less infiltrative fallback",
                    ),
                )
            primitive = by_morphology.get(morphology)
            if primitive:
                return candidates(
                    (primitive, 0, f"the intent explicitly requests {morphology} invasion"),
                )
            return candidates(
                (
                    "invasive-cord-formation-v1",
                    0,
                    "cord formation is one possible unresolved invasion morphology",
                ),
                (
                    "peritumoral-tumor-nest-formation-v1",
                    0,
                    "tumor nests are one possible unresolved invasion morphology",
                ),
                (
                    "peritumoral-neoplastic-scatter-increase-v1",
                    0,
                    "single-cell scatter is one possible unresolved invasion morphology",
                ),
                (
                    "peritumoral-small-cluster-increase-v1",
                    0,
                    "small clusters are one possible unresolved invasion morphology",
                ),
                (
                    "neoplastic-microinfiltration-increase-v1",
                    0,
                    "single-file microinfiltration is one possible unresolved morphology",
                ),
                (
                    "infiltrative-nest-cord-extension-v1",
                    0,
                    "mixed nest-cord extension is one possible unresolved morphology",
                ),
                (
                    "invasive-front-expansion-v1",
                    0,
                    "cohesive invasive-front expansion is one possible unresolved morphology",
                ),
            )
        if target == "tumor_state" and operation == "worsen":
            return candidates(
                (
                    "tumor-burden-increase-v1",
                    0,
                    "larger tumor extent is one progression endpoint",
                ),
                (
                    "cohesive-boundary-expansion-v1",
                    0,
                    "cohesive boundary expansion is one progression endpoint",
                ),
                (
                    "peritumoral-neoplastic-scatter-increase-v1",
                    0,
                    "peritumoral scatter is one progression endpoint",
                ),
                (
                    "neoplastic-cell-abundance-increase-v1",
                    0,
                    "increased neoplastic abundance is one progression endpoint",
                ),
            )
        if target == "tumor_state" and operation == "improve":
            return candidates(
                (
                    "invasive-tumor-footprint-decrease-v1",
                    0,
                    "smaller invasive footprint is one response endpoint",
                ),
                (
                    "neoplastic-cell-abundance-decrease-v1",
                    0,
                    "lower viable neoplastic abundance is one response endpoint",
                ),
                (
                    "necrosis-appearance-v1",
                    0,
                    "new necrosis is one possible treatment-response endpoint",
                ),
            )
        return ()

    @staticmethod
    def _normalize_profile_aliases(
        candidates: tuple[PrimitiveCandidate, ...], *, annotation_profile_id: str
    ) -> tuple[PrimitiveCandidate, ...]:
        result: list[PrimitiveCandidate] = []
        seen: set[str] = set()
        for item in candidates:
            primitive_id = item.primitive_id
            rationale = item.rationale
            if (
                annotation_profile_id in {"panda-gleason-v1", "bcss-semantic-v1"}
                and primitive_id == "tumor-burden-increase-v1"
            ):
                primitive_id = "cohesive-boundary-expansion-v1"
                rationale += "; this profile retires the duplicate burden alias"
            if primitive_id in seen:
                continue
            seen.add(primitive_id)
            result.append(
                replace(
                    item,
                    primitive_id=primitive_id,
                    semantic_priority=len(result) if item.semantic_priority > 0 else 0,
                    rationale=rationale,
                )
            )
        return tuple(result)


class SemanticProgramPlanner:
    """Build one step per user intent; primitives remain mask-resolvable if needed."""

    def __init__(self, resolver: PrimitiveResolver | None = None) -> None:
        self.resolver = resolver or PrimitiveResolver()

    def plan(
        self,
        request: SemanticRequest,
        *,
        case_template: JointCaseContext,
        production: bool = False,
    ) -> EditProgram:
        conflicts = _semantic_conflicts(request)
        ordered = request.ordered_intents()
        step_id_by_intent = {
            item.intent_id: f"step-{index:03d}"
            for index, item in enumerate(ordered, start=1)
        }
        explicit_dependencies = {
            item.intent_id: tuple(
                step_id_by_intent[relation.before_intent_id]
                for relation in request.relations
                if relation.relation_type == "explicit_sequence"
                and relation.after_intent_id == item.intent_id
            )
            for item in ordered
        }
        steps: list[EditProgramStep] = []
        for index, intent in enumerate(ordered, start=1):
            candidates = self.resolver.resolve(
                intent,
                case_template=case_template,
                production=production,
            )
            selected = candidates[0].primitive_id if len(candidates) == 1 else None
            status = (
                "clarification_required"
                if not candidates
                else "planned"
                if selected is not None
                else "requires_mask_resolution"
            )
            steps.append(
                EditProgramStep(
                    step_id=f"step-{index:03d}",
                    intent_id=intent.intent_id,
                    order_index=index,
                    source_text=intent.source_text,
                    depends_on=explicit_dependencies[intent.intent_id],
                    candidates=candidates,
                    selected_primitive_id=selected,
                    status=status,
                    selection_rationale=(
                        candidates[0].rationale if selected is not None else None
                    ),
                )
            )
        if conflicts or any(not item.candidates for item in steps):
            status = "clarification_required"
        elif any(item.selected_primitive_id is None for item in steps):
            status = "requires_mask_resolution"
        else:
            status = "ready"
        return EditProgram(
            request_sha256=request.request_sha256,
            instruction=request.instruction,
            steps=tuple(steps),
            status=status,
            global_constraints=request.global_constraints,
            conflicts=conflicts,
        )


@dataclass(frozen=True)
class DeterministicProgramJointPlanner:
    """Select one preflight-surviving interpretation, never raw geometry."""

    base: HeuristicJointPlanner = field(default_factory=HeuristicJointPlanner)
    name: str = "deterministic_program_joint_planner_v1"
    supports_pathology_vision: bool = False

    def select_interpretation(
        self,
        *,
        case: JointCaseContext,
        scene,
        options: Sequence[JointInterpretationOption],
        image_paths,
        artifact_registry=None,
    ):
        del scene, image_paths, artifact_registry
        if not options:
            raise JointContractError("program Planner received no executable option")
        minimum_priority = min(item.semantic_priority for item in options)
        semantic_survivors = tuple(
            item for item in options if item.semantic_priority == minimum_priority
        )
        primitive_ids = tuple(
            dict.fromkeys(item.primitive_id for item in semantic_survivors)
        )
        if len(primitive_ids) > 1:
            raise PlannerClarificationRequired(
                "multiple materially different primitives remain executable for one user intent",
                primitive_ids=primitive_ids[:3],
            )
        selected = min(
            semantic_survivors,
            key=lambda item: (
                float(
                    item.feasibility.get(
                        "structural_risk_count", 0.0
                    )
                ),
                -float(
                    item.feasibility.get(
                        "protected_distance_px", 0.0
                    )
                ),
                -float(
                    item.feasibility.get(
                        "certificate_capacity_margin", 0.0
                    )
                ),
                item.mechanism.mechanism_id,
            ),
        )
        return selected.primitive_id, selected.mechanism.mechanism_id, {
            "provider": self.name,
            "selection_mode": "deterministic_preflight_survivor",
            "supports_pathology_vision": False,
            "selection": {
                "option_id": selected.option_id,
                "primitive_id": selected.primitive_id,
                "mechanism_id": selected.mechanism.mechanism_id,
                "semantic_fit": selected.semantic_fit,
                "interpretation_explanation": (
                    "one semantic primitive survived at the best intent priority; "
                    "the mechanism was ranked only by compiler-owned safety metrics"
                ),
            },
        }

    def create_plan(self, **kwargs):
        return self.base.create_plan(**kwargs)


def legacy_semantic_intent_for_step(
    *,
    intent: SemanticIntentClause,
    candidates: Sequence[PrimitiveCandidate],
) -> dict[str, Any]:
    """Adapt a v4 intent to the existing single-step compiler boundary.

    This adapter is Planner-owned.  It exists only because the current
    ``JointCaseContext`` still requires a primitive hypothesis ledger before
    mask preflight.  The v4 Parser never sees or emits this representation.
    """

    if not candidates:
        raise JointContractError("cannot bind a step without primitive candidates")
    scenario = {
        "disease_progression": "disease_progression",
        "disease_regression": "disease_regression",
        "post_treatment": (
            "post_treatment_progression"
            if intent.operation in {"increase", "worsen"}
            else "treatment_response"
            if intent.operation in {"decrease", "improve", "clear"}
            else "residual_disease"
        ),
        "residual_disease": "residual_disease",
        "local_recurrence": "local_recurrence",
    }.get(intent.clinical_context, "direct_edit")
    direction = {
        "increase": "increase",
        "decrease": "decrease",
        "fragment": "decrease",
        "clear": "decrease",
        "appear": "increase",
        "repopulate": "increase",
        "worsen": "worsen",
        "improve": "improve",
    }.get(intent.operation, "unspecified")
    cell_class = {
        "inflammatory": "immune",
        "neoplastic": "neoplastic",
        "connective": "connective",
    }.get(intent.cell_class or "", intent.cell_class)
    hypotheses = [
        {
            "primitive_id": item.primitive_id,
            "semantic_fit": item.semantic_fit,
            "priority": item.semantic_priority,
            "rationale": item.rationale,
            "scenario": scenario,
        }
        for item in candidates
    ]
    return {
        "schema_version": "joint-semantic-intent-v3",
        "instruction": intent.source_text,
        "instruction_mode": (
            "direct_edit"
            if intent.intent_type == "direct_edit"
            else "clinical_scenario"
        ),
        "scenario": scenario,
        "clinical_direction": direction,
        "treatment_context": (
            "post_treatment"
            if intent.clinical_context == "post_treatment"
            else "none"
        ),
        "scenario_target": (
            "cells"
            if "cell" in intent.target or intent.target == "overall_cellularity"
            else "stroma"
            if intent.target == "stroma"
            else "tumor"
        ),
        "explicit_edit_scope": intent.target,
        "primitive_id": candidates[0].primitive_id,
        "subject": intent.target,
        "direction": direction,
        "explicit_cell_class": cell_class,
        "explicit_location": (
            None if intent.spatial_scope == "unspecified" else intent.spatial_scope
        ),
        "user_constraints": list(intent.constraints),
        "uncertainties": list(intent.uncertainties),
        "parser": "semantic_program_planner_v1_adapter",
        "primitive_hypotheses": hypotheses,
        "parser_metadata": {
            "source_schema": "joint-semantic-request-v4",
            "intent_id": intent.intent_id,
            "primitive_resolution_owner": "planner",
        },
    }


def bind_program_step_selection(
    step: EditProgramStep,
    *,
    primitive_id: str,
    mechanism_id: str,
    validated: bool,
) -> EditProgramStep:
    return replace(
        step,
        selected_primitive_id=primitive_id,
        selected_mechanism_id=mechanism_id,
        status="validated" if validated else "selected",
        selection_rationale=(
            "selected after semantic matching and current-state deterministic preflight"
        ),
    )


def _semantic_conflicts(request: SemanticRequest) -> tuple[str, ...]:
    explicit_pairs = {
        (item.before_intent_id, item.after_intent_id)
        for item in request.relations
        if item.relation_type == "explicit_sequence"
    }
    conflicts: list[str] = []
    intents = request.intents
    for left_index, left in enumerate(intents):
        for right in intents[left_index + 1 :]:
            if left.target != right.target:
                continue
            inverse = {left.operation, right.operation} in (
                {"increase", "decrease"},
                {"worsen", "improve"},
                {"appear", "repopulate"},
            )
            explicitly_ordered = (
                (left.intent_id, right.intent_id) in explicit_pairs
                or (right.intent_id, left.intent_id) in explicit_pairs
            )
            if inverse and not explicitly_ordered:
                conflicts.append(
                    f"{left.intent_id} and {right.intent_id} request opposing "
                    f"changes to {left.target} without an explicit order"
                )
    return tuple(conflicts)
