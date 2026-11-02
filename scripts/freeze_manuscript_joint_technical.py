#!/usr/bin/env python3
"""Freeze four actually reviewed targets for a separate nonformal GPU check."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from controlnet_train.inference.model_paths import DEFAULT_CELLVIT_ROOT
from controlnet_train.pix2pix_transfer.inference import CrossLowStainProtectionConfig
from phase3_joint_edit_refine.generator_adapter import build_frozen_generator_inputs
from phase3_joint_edit_refine.independent_review import validate_independent_review
from phase3_mask_edit.benchmark.experiment_identity import asset_identity,sha256_file
from phase3_mask_edit.benchmark.manuscript_contract import coarse_tissue
from phase3_mask_edit.benchmark.manuscript_measurements import CELL_POINT_POLICY
from scripts.run_manuscript_global_cohort import validate_cohort


def distribution_versions():
    names={d.metadata.get('Name') for d in importlib.metadata.distributions()}
    return {name:importlib.metadata.version(name) for name in sorted(names-{None,''})}


def segmentator_environment(python):
    script="""import sys,json,importlib.metadata as m
import torch,numpy,mmseg,mmengine
names={d.metadata.get('Name') for d in m.distributions()}
print(json.dumps({'python_path':sys.executable,'python':sys.version,
'torch':torch.__version__,'numpy':numpy.__version__,'mmseg':mmseg.__version__,
'mmengine':mmengine.__version__,'packages':{n:m.version(n) for n in sorted(names-{None,''})}}))
"""
    result=subprocess.run([str(python),'-c',script],capture_output=True,text=True,check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def frozen(path):return {'status':'frozen',**asset_identity(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('protocol-template','source-census','mapping-audit','overlap-audit','exclusion-audit','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--reference-id',required=True)
    for state in ('u1-moderate','u1-significant','u2-moderate','u2-significant'):
        parser.add_argument('--'+state,type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
        raise ValueError('technical freeze requires a clean committed implementation')
    protocol=deepcopy(json.loads(args.protocol_template.read_text()))
    census=json.loads(args.source_census.read_text())
    source=next(s for s in census['candidate_sources'] if s['reference_id']==args.reference_id)
    exclusion=json.loads(args.exclusion_audit.read_text())
    if (source['reference_id'] not in exclusion['excluded_reference_ids']
            or source['patient_id'] not in exclusion['excluded_patient_ids']):
        raise ValueError('technical source and patient must already be excluded')
    source_paths={key:value['path'] for key,value in source['inputs'].items()}
    for key,asset in source['inputs'].items():
        if sha256_file(asset['path'])!=asset['sha256']:raise ValueError('source census content drift')
    mappings=json.loads(args.mapping_audit.read_text())['mapping']
    overlaps=json.loads(args.overlap_audit.read_text())['cases']
    if mappings[args.reference_id]!={'patient_id':source['patient_id'],'wsi_id':source['wsi_id']}:
        raise ValueError('source mapping differs from census')
    targets={};handoffs={};joints=[]
    for state in ('u1_moderate','u1_significant','u2_moderate','u2_significant'):
        path=getattr(args,state);_,_,h=build_frozen_generator_inputs(path,output_dir=path.parent/'validation_only')
        code,strength=state.split('_');op=protocol['operations'][code]
        if (h['primitive_id']!=op['primitive_id'] or h['mechanism_id']!=op['mechanism_id']
                or h['case_id']!=args.reference_id+'__'+code+'__'+strength):
            raise ValueError('technical target operation, strength, or source differs')
        expected={**protocol['joint_budget_definitions'][strength],
            'basis':'whole_patch','relative_tolerance':.02,'fallback_policy':'max_feasible_below_target',
            'capacity_floor_policy':'strict','minimum_effective_fraction':0.}
        if h['execution_contract'].get('joint_area_budget')!=expected:
            raise ValueError('technical target differs from the authorized fixed budget')
        review=h['provenance']['independent_critic_review']
        actual=validate_independent_review(review['review_path'],
            audit_dir=h['provenance']['source_audit_case_dir'],case_id=h['case_id'],candidate_id=h['candidate_id'])
        if actual!=review:raise ValueError('reviewed handoff no longer matches its model decision')
        for field,name in (('reference_image','image'),('reference_tissue_mask','tissue'),('reference_nuclei_mask','nuclei')):
            if sha256_file(h['source_assets'][name])!=source['inputs'][field]['sha256']:
                raise ValueError('targets do not share the exact frozen source')
        joints.append(np.asarray(Image.open(h['paths']['joint_change']))>0);handoffs[state]=h
        targets[state]={'handoff':frozen(path),
            'execution_budget':{key:h['execution_contract'].get(key) for key in ('joint_area_budget','cell_count_extent_budget')},
            'target_sha256':[sha256_file(h['paths'][key]) for key in ('target_tissue_mask','target_nuclei_mask')]}
    source_coarse=coarse_tissue(source_paths['reference_tissue_mask'])
    if any(j.shape!=source_coarse.shape for j in joints):raise ValueError('joint frame differs from frozen source')
    args.output.mkdir(parents=True,exist_ok=False)
    regions={'u1_roi':source_coarse>0,'u2_roi':np.isin(source_coarse,(2,4)),
        'protected_roi':(source_coarse>0)&~binary_dilation(np.logical_or.reduce(joints),iterations=8),
        'postprocess_eligibility':np.ones(source_coarse.shape,bool),'sham_generation_support':np.zeros(source_coarse.shape,bool)}
    files={}
    for name,mask in regions.items():
        path=args.output/(name+'.png');Image.fromarray(mask.astype(np.uint8)*255).save(path);files[name]=path
    common={**source_paths,'reference_id':args.reference_id,'patient_id':source['patient_id'],
        'wsi_id':source['wsi_id'],'profile':'BCSS','prompt':protocol['sampling']['prompt'],
        'generation_seed':protocol['sampling']['seed'],'postprocess_eligibility_region':str(files['postprocess_eligibility'])}
    rows=[{**common,'state':'sham','sample_id':args.reference_id+'__sham',
        'target_tissue_mask':source_paths['reference_tissue_mask'],'target_nuclei_mask':source_paths['reference_nuclei_mask'],
        'generation_change_region':str(files['sham_generation_support'])}]
    for state,h in handoffs.items():
        rows.append({**common,'state':state,'sample_id':args.reference_id+'__'+state,
            'target_tissue_mask':h['paths']['target_tissue_mask'],'target_nuclei_mask':h['paths']['target_nuclei_mask'],
            'generation_change_region':h['paths']['generation_support'],'joint_handoff_manifest':str(getattr(args,state))})
    manifest=args.output/'generation_manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assets={'approved_joint_targets':{args.reference_id:targets},
        'protected_roi_files_and_hashes':{args.reference_id:{name:frozen(files[name]) for name in ('u1_roi','u2_roi','protected_roi')}},
        'patient_wsi_mapping_audit':{args.reference_id:mappings[args.reference_id]},
        'training_overlap_audits':{args.reference_id:overlaps[args.reference_id]},'pilot_exclusion_audit':exclusion}
    protocol.update(scope='joint_gpu_technical_acceptance',status='frozen',formal_launch_permitted=False,
        source_count=1,unique_image_count_if_shared_sham=6,generated_image_count=5,blockers=[],
        implementation_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        cohort_selection={'technical_source_list_frozen':True,'selection':'excluded source-mask-qualified technical source; no H&E or efficacy selection'},
        technical_reference_ids=[args.reference_id],technical_patient_ids=[source['patient_id']],
        generation_manifest_sha256=sha256_file(manifest),acceptance_status='pending_real_generation_and_measurements')
    protocol['postprocess']['configuration']=asdict(CrossLowStainProtectionConfig())
    protocol['measurement']={'nominal_mpp':.25,'cell_point_policy':CELL_POINT_POLICY,
        'scale_basis':'frozen nominal patch scale; no recovered full-slide physical calibration'}
    for name,definition in protocol['model_assets'].items():
        if asset_identity(definition['path'])['sha256']!=definition['sha256']:
            raise ValueError('frozen model content drift: '+name)
    protocol['cellvit_implementation_asset']=frozen(Path(DEFAULT_CELLVIT_ROOT)/'cellvit')
    release=repo/protocol['evaluator_release'];protocol['evaluator_release_asset']=frozen(release)
    old_parent=protocol['runtime_environments']['generation_and_cellvit']
    if Path(sys.executable).resolve()!=Path(old_parent['python_path']).resolve() or sys.version!=old_parent['python']:
        raise ValueError('technical freeze is running in a different parent interpreter')
    for name,version in old_parent['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('parent environment differs from template')
    old_parent['packages']={**distribution_versions(),**old_parent['packages']}
    expected_seg=protocol['runtime_environments']['segmentator'];actual_seg=segmentator_environment(expected_seg['python_path'])
    if any(actual_seg[key]!=value for key,value in expected_seg.items()):
        raise ValueError('segmentation environment differs from template')
    protocol['runtime_environments']['segmentator']=actual_seg
    protocol['formal_assets']={'generation_manifest':frozen(manifest)}
    for name,data in assets.items():
        path=args.output/(name+'.json');path.write_text(json.dumps(data,indent=2));protocol['formal_assets'][name]=frozen(path)
    validate_cohort(rows,source_count=1,handoff_validator=build_frozen_generator_inputs,
        protocol=protocol,assets=assets,technical_validation=True)
    path=args.output/'technical_protocol.json';path.write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'protocol':str(path),'manifest':str(manifest),'formal_launch_permitted':False}))
    return 0


if __name__=='__main__':raise SystemExit(main())
