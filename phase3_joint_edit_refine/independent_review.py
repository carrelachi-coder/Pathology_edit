"""Exact persisted mask evidence for an independent Codex research review.

This module checks a decision already made after visual inspection. It cannot
create an approving decision, waive gates, or grant physician/user authority.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .approval_handoff import _candidate_path, _file_sha256, _read_list, _read_object
from .audit import build_mask_review_board
from .models import JointCaseContext, JointContractError
from .nuclei import load_nuclei_mask
from phase3_mask_edit.core.mask_io import load_id_mask

REVIEW_SCHEMA = 'joint-mask-condition-independent-review-v1'


def review_evidence(audit_dir, *, candidate_id):
    """Reconstruct the mask-only board and bind all displayed passing masks."""
    root = Path(audit_dir).resolve()
    case = JointCaseContext.from_mapping(_read_object(root/'case_context.json'))
    reports = _read_list(root/'joint_gate_reports.json')
    passing = {r['candidate_id'] for r in reports if r.get('passed') is True
        and not any(c.get('severity') == 'hard' and c.get('passed') is not True
                    for c in r.get('checks', []))}
    if candidate_id not in passing:
        raise JointContractError('independent review cannot approve a gate-failing candidate')
    records = [r for r in _read_list(root/'candidates.json') if r['candidate_id'] in passing]
    if {r['candidate_id'] for r in records} != passing or len(records) != len(passing):
        raise JointContractError('review portfolio does not match passing candidates')
    bound = {name:_file_sha256(root/name) for name in (
        'case_context.json','candidates.json','joint_gate_reports.json',
        'joint_edit_plan.json','joint_condition_mask_review.png',
        'mask_planner_artifact_registry.json')}
    candidates = []
    for record in records:
        paths = {field:_candidate_path(root,record,field) for field in
            ('target_tissue_mask','target_nuclei_mask','joint_change_mask')}
        for path in paths.values():
            bound[str(path.relative_to(root))] = _file_sha256(path)
        candidates.append(SimpleNamespace(candidate_id=record['candidate_id'],
            target_tissue_mask=load_id_mask(paths['target_tissue_mask']),
            target_nuclei_mask=load_nuclei_mask(paths['target_nuclei_mask']),
            joint_change=np.asarray(Image.open(paths['joint_change_mask'])) > 0))
    source_paths = {'source_image_sha256':case.source_image_uri,
        'source_tissue_mask_sha256':case.source_tissue_mask_uri,
        'source_nuclei_mask_sha256':case.source_nuclei_mask_uri}
    source_hashes = {key:_file_sha256(Path(path)) for key,path in source_paths.items()}
    if any(source_hashes[key] != case.provenance.get(key) for key in source_hashes):
        raise JointContractError('independent review source content drift')
    expected = build_mask_review_board(
        source_tissue=load_id_mask(case.source_tissue_mask_uri),
        source_nuclei=load_nuclei_mask(case.source_nuclei_mask_uri),
        candidates=tuple(candidates))
    actual = np.asarray(Image.open(root/'joint_condition_mask_review.png').convert('RGB'))
    if not np.array_equal(expected,actual):
        raise JointContractError('review board does not reconstruct from the exact passing masks')
    return {'case_id':case.case_id,'candidate_id':candidate_id,
        'source_hashes':source_hashes,'artifact_sha256':bound}


def validate_independent_review(path, *, audit_dir, case_id, candidate_id):
    review = _read_object(Path(path))
    required = {'schema_version':REVIEW_SCHEMA,'case_id':case_id,
        'candidate_id':candidate_id,'reviewer':'codex_current_chat',
        'decision':'approved','scope':'research_mask_condition_only',
        'source_he_provided_to_review':False,'hard_gate_waiver':False}
    for field,value in required.items():
        if type(review.get(field)) is not type(value) or review.get(field) != value:
            raise JointContractError(f'independent review {field} is invalid')
    confidence = review.get('confidence')
    if (type(confidence) not in (int,float) or not math.isfinite(confidence)
            or not .7 <= confidence <= 1.):
        raise JointContractError('independent review lacks sufficient confidence')
    for field in ('tissue_geometry','nucleus_layout','coupling',
                  'preserved_conditions','limitations','verdict'):
        if not isinstance(review.get(field),str) or not review[field].strip():
            raise JointContractError(f'independent review requires {field} observations')
    if review.get('evidence') != review_evidence(audit_dir,candidate_id=candidate_id):
        raise JointContractError('independent review evidence content drift')
    if not review.get('reviewed_utc'):
        raise JointContractError('independent review timestamp is missing')
    return {**review,'review_path':str(Path(path).resolve()),
        'review_sha256':_file_sha256(Path(path))}
