"""Contracts for the v4 Parser -> Planner -> sequential edit program."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from phase3_joint_edit_refine.clarification import PlannerClarificationRequired
from phase3_joint_edit_refine.models import (
    ChangeLedger,
    JointAreaBudget,
    JointCaseContext,
    JointCondition,
    JointContractError,
    JointGateReport,
    JointWorkflowResult,
)
from phase3_joint_edit_refine.planner import JointInterpretationOption
from phase3_joint_edit_refine.program_planner import (
    DeterministicProgramJointPlanner,
    SemanticProgramPlanner,
)
from phase3_joint_edit_refine.program_workflow import (
    DeterministicMaskProgramEvaluator,
    SequentialEditProgramWorkflow,
)
from phase3_joint_edit_refine.semantic_request import (
    OpenAISemanticRequestParser,
    RuleBasedSemanticRequestParser,
    SEMANTIC_REQUEST_SCHEMA_VERSION,
    semantic_request_from_metadata,
)
from phase3_joint_edit_refine.skills.repository import JointSkillRepository


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_stub(
    *,
    domain: str = "breast-invasive-carcinoma-v1",
    annotation: str = "bcss-semantic-v1",
    population: str = "breast-cellvit-source-first-v1",
) -> JointCaseContext:
    return JointCaseContext(
        case_id="program-stub",
        instruction="placeholder",
        source_image_uri="image.png",
        source_tissue_mask_uri="tissue.npy",
        source_nuclei_mask_uri="nuclei.png",
        pathology_domain_id=domain,
        annotation_profile_id=annotation,
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id=population,
        primitive_id="cohesive-boundary-expansion-v1",
        joint_area_budget=JointAreaBudget(
            target_fraction=0.04,
            min_fraction=0.01,
            max_fraction=0.08,
            tissue_min_fraction=0.01,
        ),
        seed=7,
        provenance={
            "source_image_sha256": "image",
            "source_tissue_mask_sha256": "tissue",
            "source_nuclei_mask_sha256": "nuclei",
        },
    )


class _RecordingSemanticClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload, {"provider": "fixture"}


def _semantic_payload():
    return {
        "schema_version": SEMANTIC_REQUEST_SCHEMA_VERSION,
        "intents": [
            {
                "intent_id": "intent-001",
                "intent_type": "direct_edit",
                "target": "tumor_extent",
                "operation": "increase",
                "polarity": "affirmed",
                "clinical_context": "none",
                "spatial_scope": "boundary",
                "morphology": "cohesive",
                "cell_class": None,
                "strength": "moderate",
                "source_text": "让肿瘤边界连续扩张",
                "constraints": [],
                "uncertainties": [],
            }
        ],
        "relations": [],
        "global_constraints": [],
        "uncertainties": [],
    }


def test_api_parser_is_primitive_free_and_receives_no_images():
    client = _RecordingSemanticClient(_semantic_payload())
    request = OpenAISemanticRequestParser(client).parse("让肿瘤边界连续扩张")

    call = client.calls[0]
    assert call["image_paths"] == ()
    assert "Never select, name, rank, or suggest an edit primitive" in call[
        "system_prompt"
    ]
    assert "cohesive-boundary-expansion-v1" not in call["system_prompt"]
    assert "primitive_id" not in request.to_metadata()["intents"][0]


def test_semantic_request_rejects_a_parser_selected_primitive():
    payload = _semantic_payload()
    payload["instruction"] = "让肿瘤边界连续扩张"
    payload["parser"] = "fixture"
    payload["parser_metadata"] = {}
    payload["intents"][0]["primitive_id"] = "cohesive-boundary-expansion-v1"

    with pytest.raises(JointContractError, match="primitive or mechanism decision"):
        semantic_request_from_metadata(payload)


def test_rule_parser_preserves_three_explicitly_ordered_intents():
    request = RuleBasedSemanticRequestParser().parse(
        "先让肿瘤边界连续扩张，然后减少肿瘤细胞，最后增加坏死。"
    )

    assert [(item.target, item.operation) for item in request.intents] == [
        ("tumor_extent", "increase"),
        ("neoplastic_cell_population", "decrease"),
        ("necrosis", "appear"),
    ]
    assert [item.relation_type for item in request.relations] == [
        "explicit_sequence",
        "explicit_sequence",
    ]
    assert [item.intent_id for item in request.ordered_intents()] == [
        "intent-001",
        "intent-002",
        "intent-003",
    ]


def test_planner_resolves_one_primitive_per_intent_and_uses_step_dependencies():
    request = RuleBasedSemanticRequestParser().parse(
        "先让肿瘤边界连续扩张，然后在治疗后减少肿瘤细胞。"
    )
    program = SemanticProgramPlanner().plan(
        request,
        case_template=_case_stub(),
    )

    assert program.status == "ready"
    assert [item.selected_primitive_id for item in program.steps] == [
        "cohesive-boundary-expansion-v1",
        "neoplastic-cell-abundance-decrease-v1",
    ]
    assert program.steps[1].depends_on == ("step-001",)


def test_planner_keeps_unresolved_invasion_morphologies_for_mask_preflight():
    request = RuleBasedSemanticRequestParser().parse("增加浸润。")
    program = SemanticProgramPlanner().plan(
        request,
        case_template=_case_stub(),
    )

    assert program.status == "requires_mask_resolution"
    assert len(program.steps[0].candidates) >= 2
    assert program.steps[0].selected_primitive_id is None


def test_planner_flags_unordered_opposing_intents():
    request = RuleBasedSemanticRequestParser().parse(
        "增加肿瘤面积，并且减少肿瘤面积。"
    )
    program = SemanticProgramPlanner().plan(
        request,
        case_template=_case_stub(),
    )

    assert program.status == "clarification_required"
    assert program.conflicts


def test_negated_edit_is_preserved_and_never_resolved_to_a_primitive():
    request = RuleBasedSemanticRequestParser().parse("不要增加肿瘤面积。")
    program = SemanticProgramPlanner().plan(
        request,
        case_template=_case_stub(),
    )

    assert request.intents[0].polarity == "negated"
    assert program.status == "clarification_required"
    assert program.steps[0].candidates == ()


def test_glas_closed_tissue_growth_is_not_exposed_as_executable():
    request = RuleBasedSemanticRequestParser().parse("增加肿瘤面积。")
    program = SemanticProgramPlanner().plan(
        request,
        case_template=_case_stub(
            domain="colorectal-adenocarcinoma-v1",
            annotation="glas-gland-v1",
            population="colorectal-cellvit-source-first-v1",
        ),
    )

    assert program.status == "clarification_required"
    assert program.steps[0].candidates == ()


def test_deterministic_joint_planner_selects_one_semantic_survivor_only():
    repository = JointSkillRepository()
    mechanism = repository.mechanisms["breast-annotation-anchored-boundary-growth"]
    option = JointInterpretationOption(
        primitive_id="cohesive-boundary-expansion-v1",
        semantic_fit="explicit",
        semantic_priority=0,
        semantic_rationale="explicit boundary request",
        mechanism=mechanism,
        feasibility={"certificate_capacity_margin": 20},
    )

    primitive_id, mechanism_id, usage = (
        DeterministicProgramJointPlanner().select_interpretation(
            case=_case_stub(),
            scene=None,
            options=(option,),
            image_paths=(),
        )
    )

    assert primitive_id == option.primitive_id
    assert mechanism_id == mechanism.mechanism_id
    assert usage["provider"] == "deterministic_program_joint_planner_v1"


def test_deterministic_joint_planner_clarifies_distinct_surviving_primitives():
    repository = JointSkillRepository()
    options = (
        JointInterpretationOption(
            primitive_id="invasive-cord-formation-v1",
            semantic_fit="contextual",
            semantic_priority=0,
            semantic_rationale="one invasion morphology",
            mechanism=repository.mechanisms["breast-cell-seeded-invasive-cord"],
            feasibility={},
        ),
        JointInterpretationOption(
            primitive_id="peritumoral-tumor-nest-formation-v1",
            semantic_fit="contextual",
            semantic_priority=0,
            semantic_rationale="another invasion morphology",
            mechanism=repository.mechanisms["breast-peritumoral-tumor-nest"],
            feasibility={},
        ),
    )

    with pytest.raises(PlannerClarificationRequired):
        DeterministicProgramJointPlanner().select_interpretation(
            case=_case_stub(),
            scene=None,
            options=options,
            image_paths=(),
        )


class _FakeStepWorkflow:
    def __init__(self):
        self.cases = []

    def run(self, case, *, output_root):
        del output_root
        self.cases.append(case)
        tissue_path = Path(case.source_tissue_mask_uri)
        if tissue_path.suffix == ".npy":
            tissue = np.load(tissue_path, allow_pickle=False)
        else:
            tissue = np.asarray(Image.open(tissue_path))
        nuclei = np.asarray(Image.open(case.source_nuclei_mask_uri))
        target_tissue = tissue.copy()
        location = (0, len(self.cases) - 1)
        target_tissue[location] = 1 if target_tissue[location] != 1 else 2
        tissue_change = target_tissue != tissue
        cell_change = np.zeros_like(tissue_change)
        ledger = ChangeLedger(
            tissue_pixels=int(np.count_nonzero(tissue_change)),
            removed_nucleus_pixels=0,
            added_nucleus_pixels=0,
            cell_pixels=0,
            cell_only_pixels=0,
            joint_pixels=int(np.count_nonzero(tissue_change)),
            generation_support_pixels=int(np.count_nonzero(tissue_change)),
            total_pixels=tissue.size,
            removed_instance_ids=(),
            added_instance_ids=(),
            retained_instance_ids=(),
        )
        condition = JointCondition(
            case_id=case.case_id,
            candidate_id="candidate-001",
            executable_contract_id="fake-contract",
            target_tissue_mask=target_tissue,
            target_nuclei_mask=nuclei,
            tissue_change=tissue_change,
            cell_change=cell_change,
            joint_change=tissue_change,
            generation_support=tissue_change,
            pathology_mechanism=f"fake::{case.primitive_id}",
            active_skill_rules=("fake-hard-gate",),
            ledger=ledger,
        )
        return JointWorkflowResult(
            status="selected_research",
            case_context=case,
            joint_plan=None,
            gate_reports=(JointGateReport("candidate-001", True, ()),),
            critic_result=None,
            selected_candidate_id="candidate-001",
            condition=condition,
            abstain_reasons=(),
            artifact_paths={},
        )


def test_sequential_program_commits_then_replans_from_the_latest_masks(tmp_path):
    tissue = np.full((8, 8), 2, dtype=np.uint8)
    nuclei = np.zeros((8, 8), dtype=np.uint8)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    tissue_path = tmp_path / "tissue.npy"
    nuclei_path = tmp_path / "nuclei.png"
    image_path = tmp_path / "image.png"
    np.save(tissue_path, tissue, allow_pickle=False)
    Image.fromarray(nuclei).save(nuclei_path)
    Image.fromarray(image).save(image_path)
    raw = {
        "case_id": "sequential-case",
        "instruction": (
            "先让肿瘤边界连续扩张，然后在治疗后减少肿瘤细胞。"
        ),
        "source_image_uri": str(image_path),
        "source_tissue_mask_uri": str(tissue_path),
        "source_nuclei_mask_uri": str(nuclei_path),
        "pathology_domain_id": "breast-invasive-carcinoma-v1",
        "annotation_profile_id": "bcss-semantic-v1",
        "cell_observation_profile_id": "cellvit-five-class-v1",
        "cell_population_profile_id": "breast-cellvit-source-first-v1",
        "joint_area_budget": {
            "target_fraction": 0.04,
            "min_fraction": 0.01,
            "max_fraction": 0.08,
            "tissue_min_fraction": 0.01,
        },
        "seed": 9,
        "provenance": {
            "source_image_sha256": _sha(image_path),
            "source_tissue_mask_sha256": _sha(tissue_path),
            "source_nuclei_mask_sha256": _sha(nuclei_path),
        },
    }
    fake = _FakeStepWorkflow()
    evaluator = DeterministicMaskProgramEvaluator()
    result = SequentialEditProgramWorkflow(
        step_workflow=fake,
        evaluator=evaluator,
    ).run(
        raw,
        semantic_parser=RuleBasedSemanticRequestParser(),
        output_root=tmp_path / "program",
    )

    assert result.status == "validated"
    assert result.evaluation["passed"] is True
    assert result.evaluation["deterministic"] is True
    assert result.evaluation["visual_pathology_approval"] is False
    assert len(fake.cases) == 2
    assert fake.cases[1].source_tissue_mask_uri.endswith("step-001.tissue.png")
    assert fake.cases[1].provenance["target_cell_class_ids"] == [1]
    assert (
        result.step_audits[1].input_tissue_sha256
        == result.step_audits[0].output_tissue_sha256
    )


def test_evaluator_uses_only_hard_gate_pass_and_budget_error():
    evaluator = DeterministicMaskProgramEvaluator()
    candidate = SimpleNamespace(
        candidate_id="candidate-001",
        ledger=SimpleNamespace(joint_fraction=0.04),
        tool_trace={},
    )
    case = SimpleNamespace(
        joint_area_budget=SimpleNamespace(target_fraction=0.04),
        cell_count_extent_budget=None,
    )
    bundle = SimpleNamespace(
        primitive=SimpleNamespace(budget_mode="joint_area"),
        active_rule_ids=("hard-rule",),
    )
    result = evaluator.review(
        case=case,
        bundle=bundle,
        candidates=(candidate,),
        gate_reports=(JointGateReport("candidate-001", True, ()),),
        image_paths=(),
    )

    assert result.abstain is False
    assert result.rankings[0].candidate_id == "candidate-001"
    assert result.usage["deterministic"] is True
    assert evaluator.supports_pathology_vision is False
