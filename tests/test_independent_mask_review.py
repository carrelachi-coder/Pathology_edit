"""Bind real mask rendering to a model decision, without user impersonation."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from phase3_joint_edit_refine.audit import build_mask_review_board
from phase3_joint_edit_refine.independent_review import (
    REVIEW_SCHEMA, review_evidence, validate_independent_review,
)
from phase3_joint_edit_refine.models import JointContractError
from phase3_mask_edit.benchmark.experiment_identity import sha256_file
from tests.test_joint_edit_refine import _write_synthetic_case, _as_breast_growth_case


def png_case(root):
    case=_as_breast_growth_case(_write_synthetic_case(root))
    path=root/'source_tissue.png'
    Image.fromarray(np.load(case.source_tissue_mask_uri)).save(path)
    return replace(case,source_tissue_mask_uri=str(path),provenance={**case.provenance,
        'source_tissue_mask_sha256':sha256_file(path),'original_label_map_digest':sha256_file(path)})


@pytest.fixture
def review(tmp_path):
    case=png_case(tmp_path)
    root=tmp_path/'audit';root.mkdir()
    (root/'case_context.json').write_text(json.dumps(case.to_metadata()))
    from phase3_mask_edit.core.mask_io import load_id_mask
    tissue=load_id_mask(case.source_tissue_mask_uri).copy()
    from phase3_joint_edit_refine.nuclei import load_nuclei_mask
    nuclei=load_nuclei_mask(case.source_nuclei_mask_uri)
    joint=np.zeros_like(tissue,dtype=bool)
    paths={}
    for field,array in [('target_tissue_mask',tissue),('target_nuclei_mask',nuclei),('joint_change_mask',joint.astype(np.uint8)*255)]:
        path=root/(field+'.png');Image.fromarray(np.asarray(array,dtype=np.uint8)).save(path);paths[field]=str(path)
    candidate={'candidate_id':'candidate-1',**paths}
    (root/'candidates.json').write_text(json.dumps([candidate]))
    (root/'joint_gate_reports.json').write_text(json.dumps([{'candidate_id':'candidate-1','passed':True,'checks':[{'check_id':'fixture','severity':'hard','passed':True}]}]))
    (root/'joint_edit_plan.json').write_text('{}')
    (root/'mask_planner_artifact_registry.json').write_text('{}')
    board=build_mask_review_board(source_tissue=tissue,source_nuclei=nuclei,candidates=(
        SimpleNamespace(candidate_id='candidate-1',target_tissue_mask=tissue,target_nuclei_mask=nuclei,joint_change=joint),))
    Image.fromarray(board).save(root/'joint_condition_mask_review.png')
    value={'schema_version':REVIEW_SCHEMA,'case_id':case.case_id,
        'candidate_id':'candidate-1','reviewer':'codex_current_chat',
        'decision':'approved','scope':'research_mask_condition_only',
        'source_he_provided_to_review':False,'hard_gate_waiver':False,
        'confidence':.8,'reviewed_utc':'2026-11-02T00:00:00+00:00',
        'evidence':review_evidence(root,candidate_id='candidate-1')}
    for field in ('tissue_geometry','nucleus_layout','coupling','preserved_conditions','limitations','verdict'):
        value[field]='Synthetic validation fixture; not an actual visual approval.'
    path=tmp_path/'review.json';path.write_text(json.dumps(value))
    return root,path,value


def test_exact_model_decision_has_its_own_identity(review):
    root,path,value=review
    result=validate_independent_review(path,audit_dir=root,case_id=value['case_id'],candidate_id='candidate-1')
    assert result['reviewer']=='codex_current_chat'
    assert result['review_sha256']==sha256_file(path)
    assert 'approved_by' not in result


@pytest.mark.parametrize('field,value',[('decision','abstained'),('confidence',.35),('source_he_provided_to_review',True),('hard_gate_waiver',True),('candidate_id','other')])
def test_no_approval_for_abstention_low_confidence_or_wrong_scope(review,field,value):
    root,path,payload=review;payload[field]=value;path.write_text(json.dumps(payload))
    with pytest.raises(JointContractError):
        validate_independent_review(path,audit_dir=root,case_id=payload['case_id'],candidate_id='candidate-1')


def test_model_cannot_waive_new_hard_gate_failure(review):
    root,path,value=review
    reports=json.loads((root/'joint_gate_reports.json').read_text())
    reports[0]['checks'][0]['passed']=False
    (root/'joint_gate_reports.json').write_text(json.dumps(reports))
    with pytest.raises(JointContractError,match='gate-failing'):
        validate_independent_review(path,audit_dir=root,case_id=value['case_id'],candidate_id='candidate-1')


def test_rehashed_counterfeit_board_cannot_be_review_evidence(review):
    root,path,value=review
    board=root/'joint_condition_mask_review.png';Image.new('RGB',Image.open(board).size,'red').save(board)
    value['evidence']['artifact_sha256'][board.name]=sha256_file(board)
    path.write_text(json.dumps(value))
    with pytest.raises(JointContractError,match='does not reconstruct'):
        validate_independent_review(path,audit_dir=root,case_id=value['case_id'],candidate_id='candidate-1')


def test_changed_target_invalidates_old_review(review):
    root,path,value=review
    target=root/'target_tissue_mask.png';array=np.asarray(Image.open(target)).copy();array[0,0]=1
    Image.fromarray(array).save(target)
    with pytest.raises(JointContractError):
        validate_independent_review(path,audit_dir=root,case_id=value['case_id'],candidate_id='candidate-1')


def test_review_promotion_reuses_exact_contract_and_emits_model_provenance(tmp_path):
    from tests.test_joint_edit_refine import (_PassingTissueGateFixture,
        _PassingJointGateFixture, _ApprovingJointCritic)
    from phase3_mask_edit_refine.agents import HeuristicInterfacePlanner
    from phase3_joint_edit_refine.planner import HeuristicJointPlanner
    from phase3_joint_edit_refine.workflow import JointPathologyEditWorkflow
    from phase3_joint_edit_refine.approval_handoff import promote_independently_reviewed_joint_candidate
    from phase3_joint_edit_refine.generator_adapter import build_frozen_generator_inputs
    case=png_case(tmp_path)
    # Explicit CPU gate/critic fixtures exercise persistence and reconstruction;
    # this is not a new deterministic-gate or actual visual acceptance claim.
    result=JointPathologyEditWorkflow(tissue_planner=HeuristicInterfacePlanner(),
        joint_planner=HeuristicJointPlanner(),critic=_ApprovingJointCritic(),
        tissue_gates=_PassingTissueGateFixture(),joint_gates=_PassingJointGateFixture()).run(case,output_root=tmp_path/'workflow')
    assert result.status=='selected_research'
    root=tmp_path/'workflow'/case.case_id
    value={'schema_version':REVIEW_SCHEMA,'case_id':case.case_id,
        'candidate_id':result.selected_candidate_id,'reviewer':'codex_current_chat',
        'decision':'approved','scope':'research_mask_condition_only',
        'source_he_provided_to_review':False,'hard_gate_waiver':False,
        'confidence':.8,'reviewed_utc':'2026-11-02T00:00:00+00:00',
        'evidence':review_evidence(root,candidate_id=result.selected_candidate_id)}
    for field in ('tissue_geometry','nucleus_layout','coupling','preserved_conditions','limitations','verdict'):
        value[field]='Explicit synthetic reconstruction fixture; no biological validation.'
    path=tmp_path/'review.json';path.write_text(json.dumps(value))
    paths=promote_independently_reviewed_joint_candidate(root,
        candidate_id=result.selected_candidate_id,review_path=path,output_dir=tmp_path/'promoted')
    _,_,handoff=build_frozen_generator_inputs(paths['manifest'],output_dir=tmp_path/'consume')
    assert 'user_approval' not in handoff['provenance']
    assert handoff['provenance']['independent_critic_review']['review_sha256']==sha256_file(path)
    assert handoff['result_binding']['candidate_id']==result.selected_candidate_id
    with pytest.raises(JointContractError,match='preserve previous handoff'):
        promote_independently_reviewed_joint_candidate(root,
            candidate_id=result.selected_candidate_id,review_path=path,output_dir=tmp_path/'promoted')
