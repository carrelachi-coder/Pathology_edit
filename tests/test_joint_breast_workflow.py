"""End-to-end executable contracts for Breast + BCSS treatment edits."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from phase3_joint_edit_refine.critic import DeterministicJointResearchCritic
from phase3_joint_edit_refine.models import (
    CellCountExtentBudget,
    JointAreaBudget,
    JointCaseContext,
    JointContractError,
    JointCriticRanking,
    JointCriticResult,
)
from phase3_joint_edit_refine.planner import HeuristicJointPlanner
from phase3_joint_edit_refine.semantic_parser import (
    RuleBasedSemanticParser,
    bind_semantic_intent,
)
from phase3_joint_edit_refine.skills.repository import JointSkillRepository
from phase3_joint_edit_refine.tissue_planner import MultiInterfaceResearchTissuePlanner
from phase3_joint_edit_refine.workflow import (
    JointPathologyEditWorkflow,
    JointWorkflowConfig,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _ApprovingCritic(DeterministicJointResearchCritic):
    supports_pathology_vision = True

    def review(
        self,
        *,
        case,
        bundle,
        candidates,
        gate_reports,
        image_paths,
        artifact_registry=None,
    ):
        del case, gate_reports, image_paths, artifact_registry
        candidate = candidates[0]
        return JointCriticResult(
            rankings=(
                JointCriticRanking(
                    candidate_id=candidate.candidate_id,
                    score=0.97,
                    confidence=0.97,
                    supporting_rule_ids=bundle.active_rule_ids,
                ),
            ),
            abstain=False,
            summary="Synthetic fixture visual approval.",
            usage={"provider": "synthetic-fixture"},
        )


def _write_breast_case(
    root: Path,
    *,
    instruction: str,
    mechanism_id: str | None,
    with_roi: bool = False,
) -> JointCaseContext:
    """Create one BCSS tumor with a legal retreat front and editable neck."""

    size = 192
    rows, cols = np.ogrid[:size, :size]
    left = (rows - 96) ** 2 + (cols - 58) ** 2 <= 35**2
    right = (rows - 96) ** 2 + (cols - 134) ** 2 <= 35**2
    neck = (rows >= 88) & (rows <= 104) & (cols >= 58) & (cols <= 134)
    tumor = left | right | neck
    tissue = np.full((size, size), 2, dtype=np.uint8)
    tissue[tumor] = 1
    nuclei = np.zeros_like(tissue)
    for row in range(7, size - 7, 8):
        for col in range(7, size - 7, 8):
            class_id = 1 if tumor[row, col] else 3
            nuclei[row - 1 : row + 2, col - 1 : col + 2] = class_id

    image = np.full((size, size, 3), (221, 184, 204), dtype=np.uint8)
    image[tumor] = (170, 91, 132)
    tissue_path = root / "tissue.npy"
    nuclei_path = root / "nuclei.png"
    image_path = root / "image.png"
    np.save(tissue_path, tissue, allow_pickle=False)
    Image.fromarray(nuclei).save(nuclei_path)
    Image.fromarray(image).save(image_path)

    provenance: dict[str, object] = {
        "source_image_sha256": _sha(image_path),
        "source_tissue_mask_sha256": _sha(tissue_path),
        "source_nuclei_mask_sha256": _sha(nuclei_path),
        "original_label_map_digest": _sha(tissue_path),
        "original_instance_mask_digest": _sha(nuclei_path),
        "preprocessing_revision": "synthetic-bcss-treatment-v1",
        "available_auxiliary_structures": [],
    }
    if mechanism_id:
        provenance["joint_mechanism_id"] = mechanism_id
    if mechanism_id == "breast-post-treatment-residual-neoplastic-depletion":
        # Keep this test focused on workflow realization rather than on the
        # automatic source-calibrated count broker, which has independent
        # coverage. The fixture owns a reachable gradient quota, while the
        # deterministic cell portfolio must now discover and certify its own
        # mask-graph interface/anchor instead of receiving one in provenance.
        cell_budget = {
            "target_delta_count": 12,
            "min_delta_count": 12,
            "max_delta_count": 14,
            "maximum_extent_px": 64,
            "interface_min_px": 0,
            "interface_max_px": 64,
            "minimum_effect_span_px": 30,
            "minimum_effect_foci": 0,
        }
    else:
        cell_budget = None
    auxiliaries: dict[str, str] = {}
    if with_roi:
        roi = (
            (rows - 96) ** 2 + (cols - 58) ** 2 <= 70**2
        ).astype(np.uint8)
        roi_path = root / "local-clearance-roi.png"
        Image.fromarray(roi).save(roi_path)
        roi_digest = _sha(roi_path)
        auxiliaries["local_clearance_roi"] = str(roi_path)
        provenance["available_auxiliary_structures"] = ["local_clearance_roi"]
        provenance["auxiliary_structure_sha256"] = {
            "local_clearance_roi": roi_digest
        }
        provenance["auxiliary_structure_provenance"] = {
            "local_clearance_roi": {
                "producer_id": "synthetic-user-roi",
                "producer_version": "synthetic-user-roi-v1",
                "source_tissue_mask_sha256": _sha(tissue_path),
                "output_sha256": roi_digest,
            }
        }

    raw = {
        "case_id": "breast-" + hashlib.sha256(instruction.encode()).hexdigest()[:8],
        "instruction": instruction,
        "source_image_uri": str(image_path),
        "source_tissue_mask_uri": str(tissue_path),
        "source_nuclei_mask_uri": str(nuclei_path),
        "pathology_domain_id": "breast-invasive-carcinoma-v1",
        "annotation_profile_id": "bcss-semantic-v1",
        "cell_observation_profile_id": "cellvit-five-class-v1",
        "cell_population_profile_id": "breast-cellvit-source-first-v1",
        "joint_area_budget": {
            "target_fraction": 0.04,
            "min_fraction": 0.018,
            "max_fraction": 0.075,
            "tissue_min_fraction": 0.015,
        },
        "seed": 27,
        "provenance": provenance,
        "auxiliary_structure_uris": auxiliaries,
        "pixel_size_um": 0.5,
        "cell_count_extent_budget": cell_budget,
    }
    case, _ = bind_semantic_intent(raw, RuleBasedSemanticParser())
    if mechanism_id:
        case = replace(
            case,
            provenance={
                **case.provenance,
                "joint_primitive_id": case.primitive_id,
            },
        )
    return case


def _run(case: JointCaseContext, root: Path):
    return JointPathologyEditWorkflow(
        tissue_planner=MultiInterfaceResearchTissuePlanner(),
        joint_planner=HeuristicJointPlanner(),
        critic=_ApprovingCritic(),
    ).run(case, output_root=root / "output")


def test_directionless_treatment_workflow_offers_three_preflighted_scenarios(
    tmp_path,
):
    case = _write_breast_case(
        tmp_path,
        instruction="Simulate a post-treatment change.",
        mechanism_id=None,
    )

    result = _run(case, tmp_path)

    assert result.status == "clarification_required", result.abstain_reasons
    options = result.clarification_request["options"]
    assert [item["scenario"] for item in options] == [
        "treatment_response",
        "post_treatment_progression",
        "residual_disease",
    ]
    assert all(item["clinician_label"].isascii() for item in options)


def test_breast_regression_executes_tissue_and_whole_instance_turnover(tmp_path):
    case = _write_breast_case(
        tmp_path,
        instruction="Simulate local tumor shrinkage after treatment.",
        mechanism_id="breast-post-treatment-invasive-regression",
    )
    result = _run(case, tmp_path)

    assert result.status == "selected_research", result.abstain_reasons
    source = np.load(case.source_tissue_mask_uri)
    changed = result.condition.tissue_change
    assert np.any(changed)
    assert set(np.unique(source[changed])) == {1}
    assert set(np.unique(result.condition.target_tissue_mask[changed])) == {2}
    assert np.count_nonzero(result.condition.target_tissue_mask == 1) > 0


def test_breast_residual_neoplastic_depletion_is_cell_only(tmp_path):
    case = _write_breast_case(
        tmp_path,
        instruction="Reduce residual tumor cells after treatment.",
        mechanism_id="breast-post-treatment-residual-neoplastic-depletion",
    )
    result = _run(case, tmp_path)

    assert result.status == "selected_research", result.abstain_reasons
    assert "cellularity_depletion_anchor" not in case.provenance
    source_tissue = np.load(case.source_tissue_mask_uri)
    source_nuclei = np.asarray(Image.open(case.source_nuclei_mask_uri))
    assert np.array_equal(result.condition.target_tissue_mask, source_tissue)
    assert np.count_nonzero(result.condition.target_nuclei_mask == 1) < np.count_nonzero(
        source_nuclei == 1
    )
    assert np.array_equal(
        result.condition.target_nuclei_mask[source_nuclei == 3],
        source_nuclei[source_nuclei == 3],
    )


def test_breast_residual_fragmentation_splits_the_source_component(tmp_path):
    case = _write_breast_case(
        tmp_path,
        instruction="Fragment residual tumor after treatment.",
        mechanism_id="breast-residual-disease-fragmentation",
    )
    source = np.load(case.source_tissue_mask_uri)
    assert ndimage.label(source == 1, structure=np.ones((3, 3)))[1] == 1
    # A narrow stromal corridor needs far fewer pixels than a broad burden
    # regression. This budget still produces a measurable local split while
    # leaving both residual foci well above their skill-owned area floor.
    case = replace(
        case,
        joint_area_budget=JointAreaBudget(
            target_fraction=0.035,
            min_fraction=0.03,
            max_fraction=0.05,
            tissue_min_fraction=0.03,
        ),
    )
    result = _run(case, tmp_path)

    assert result.status == "selected_research", result.abstain_reasons
    target_labels, raw_target_count = ndimage.label(
        result.condition.target_tissue_mask == 1,
        structure=np.ones((3, 3)),
    )
    target_count = sum(
        np.count_nonzero(target_labels == index) >= 96
        for index in range(1, raw_target_count + 1)
    )
    assert target_count >= 2


def test_breast_local_clearance_is_bound_to_explicit_roi(tmp_path):
    case = _write_breast_case(
        tmp_path,
        instruction="Clear invasive tumor in this local ROI.",
        mechanism_id="breast-local-invasive-clearance",
        with_roi=True,
    )
    result = _run(case, tmp_path)

    assert result.status == "selected_research", result.abstain_reasons
    roi = np.asarray(
        Image.open(case.auxiliary_structure_uris["local_clearance_roi"]),
        dtype=bool,
    )
    assert np.any(result.condition.tissue_change)
    assert not np.any(result.condition.tissue_change & ~roi)


def test_breast_cellularity_requires_runtime_shadow_review_authority():
    repository = JointSkillRepository()
    case = JointCaseContext(
        case_id="breast-cellularity-review",
        instruction="Increase local cellularity.",
        source_image_uri="image.png",
        source_tissue_mask_uri="tissue.npy",
        source_nuclei_mask_uri="nuclei.png",
        pathology_domain_id="breast-invasive-carcinoma-v1",
        annotation_profile_id="bcss-semantic-v1",
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id="breast-cellvit-source-first-v1",
        primitive_id="cellularity-increase-v1",
        joint_area_budget=None,
        cell_count_extent_budget=CellCountExtentBudget(
            target_delta_count=12,
            min_delta_count=8,
            max_delta_count=16,
            maximum_extent_px=48,
            minimum_effect_span_px=24,
            minimum_effect_foci=3,
        ),
        seed=1,
        provenance={
            "source_image_sha256": "image",
            "source_tissue_mask_sha256": "tissue",
            "source_nuclei_mask_sha256": "nuclei",
        },
    )
    with pytest.raises(
        JointContractError, match="pending mechanism-level pathology review"
    ):
        repository.compose(
            case=case,
            mechanism_id="breast-local-population-modulation",
            available_checker_ids=(),
            production=False,
        )


def test_meta_eval_config_rejects_probnet_fallback_for_target_regeneration(tmp_path):
    case = _write_breast_case(
        tmp_path,
        instruction="Simulate local tumor shrinkage after treatment.",
        mechanism_id="breast-post-treatment-invasive-regression",
    )
    workflow = JointPathologyEditWorkflow(
        tissue_planner=MultiInterfaceResearchTissuePlanner(),
        joint_planner=HeuristicJointPlanner(),
        critic=_ApprovingCritic(),
        config=JointWorkflowConfig(
            require_mature_probnet_for_target_population_regeneration=True
        ),
    )

    result = workflow.run(case, output_root=tmp_path / "strict-output")
    assert result.status == "abstained"
    assert any(
        "requires the mature ProbNet" in reason
        for reason in result.abstain_reasons
    )


def test_meta_eval_config_rejects_distance_ranker_for_cell_only_addition(tmp_path):
    case = _write_breast_case(
        tmp_path,
        instruction="Increase tumor cells.",
        mechanism_id="breast-local-population-modulation",
    )
    case = replace(
        case,
        cell_count_extent_budget=CellCountExtentBudget(
            target_delta_count=16,
            min_delta_count=12,
            max_delta_count=20,
            maximum_extent_px=64,
            minimum_effect_span_px=24,
            minimum_effect_foci=3,
        ),
    )
    workflow = JointPathologyEditWorkflow(
        tissue_planner=MultiInterfaceResearchTissuePlanner(),
        joint_planner=HeuristicJointPlanner(),
        critic=_ApprovingCritic(),
        config=JointWorkflowConfig(
            require_probnet_ranker_for_cell_addition=True
        ),
    )

    result = workflow.run(case, output_root=tmp_path / "strict-cell-output")
    assert result.status == "abstained"
    assert any(
        "frozen ProbNet spatial ranker" in reason
        for reason in result.abstain_reasons
    )
