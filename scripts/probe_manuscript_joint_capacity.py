#!/usr/bin/env python3
"""Source-only necessary capacity screen, with the unchanged 19/32 budgets."""
import argparse
import json
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from phase3_joint_edit_refine.models import JointCaseContext
from phase3_joint_edit_refine.auxiliary import materialize_profile_auxiliaries
from phase3_joint_edit_refine.scene import build_joint_scene_analysis
from phase3_joint_edit_refine.g2_execution_qualification import (MaskSkillRepository,JointSkillRepository,JointFeasibilitySolver,_all_joint_checker_ids,build_joint_nuclei_preflight,tissue_tool_primitive_id,bind_active_bundle_to_case,validate_active_bundle_authority,GateRegistry)
from phase3_mask_edit_refine.evidence import load_id_mask
from phase3_joint_edit_refine.nuclei import load_nuclei_mask


def probe(raw, output, *, topology=False):
    case=JointCaseContext.from_mapping(raw);case.validate_local_inputs()
    tissue=load_id_mask(case.source_tissue_mask_uri);nuclei=load_nuclei_mask(case.source_nuclei_mask_uri)
    case,_=materialize_profile_auxiliaries(case,source_tissue=tissue,source_nuclei=nuclei,source_image=np.asarray(Image.open(case.source_image_uri).convert('RGB')),output_dir=output)
    skills=MaskSkillRepository();joint=JointSkillRepository();schema=skills.annotation_schema(case.annotation_profile_id)
    scene=build_joint_scene_analysis(tissue,nuclei,schema=schema,pixel_size_um=case.pixel_size_um,nuclei_instances_path=case.source_nuclei_instances_uri,auxiliary_structure_paths=case.auxiliary_structure_uris,auxiliary_structure_provenance=case.provenance.get('auxiliary_structure_provenance',{}))
    bundle=joint.compose(case=case,mechanism_id=case.provenance['joint_mechanism_id'],available_checker_ids=_all_joint_checker_ids(),production=False)
    allocation=JointFeasibilitySolver().allocate(shape=tissue.shape,budget=case.joint_area_budget,bundle=bundle)
    tissue_bundle=skills.compose(pathology_domain_id=case.pathology_domain_id,annotation_profile_id=case.annotation_profile_id,primitive_id=tissue_tool_primitive_id(case.primitive_id),production=False,available_checker_ids=set(GateRegistry().available_checker_ids),case_provenance=case.provenance)
    tissue_bundle=bind_active_bundle_to_case(tissue_bundle,case=case,scene=scene.tissue,semantic_primitive_id=case.primitive_id)
    validate_active_bundle_authority(tissue_bundle,case_provenance=case.provenance,require_live_binding=True,case=case,scene=scene.tissue)
    evidence=build_joint_nuclei_preflight(case=case,source_tissue=tissue,schema=schema,scene=scene,tissue_bundle=tissue_bundle,joint_bundle=bundle,allocation=allocation)
    passed=bool(evidence.meaningful_tissue_capacity_passed and evidence.feasible_interface_ids and not evidence.required_auxiliary_missing and not evidence.required_provenance_missing)
    if topology and passed:
        from phase3_joint_edit_refine.g2_execution_qualification import _qualify_tissue_case
        metrics, failures = _qualify_tissue_case(case=case, source_tissue=tissue, schema=schema, scene=scene, bundle=bundle, mask_skills=skills, budget_solver=JointFeasibilitySolver())
        return {'passed_necessary_capacity':passed,'passed_topology_witness':not failures,'failure_reasons':failures,'metrics':metrics}
    return {'passed_necessary_capacity':passed,'floor_pixels':evidence.meaningful_tissue_floor_pixels,'capacity_pixels':evidence.aggregate_feasible_tissue_capacity_pixels,'interfaces':list(evidence.feasible_interface_ids)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-freeze',type=Path,required=True);parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit',type=int,default=5);parser.add_argument('--start-index',type=int,default=0);parser.add_argument('--topology-witness',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    freeze=json.loads(args.source_freeze.read_text());templates=json.loads(args.cases.read_text());records=[]
    sources=freeze.get('candidate_sources') or [freeze['technical_source'],*freeze['formal_sources']]
    for source in sources[args.start_index:args.start_index+args.limit]:
        record={'reference_id':source['reference_id'],'patient_id':source['patient_id'],'source_only':True,'target_mask_created':False,'checks':[]}
        for template in (templates[1],templates[3]):
            raw=json.loads(json.dumps(template));raw['case_id']=source['reference_id']+'__'+template['primitive_id']+'__capacity'
            for key,field in [('source_image_uri','reference_image'),('source_tissue_mask_uri','reference_tissue_mask'),('source_nuclei_mask_uri','reference_nuclei_mask')]:raw[key]=source['inputs'][field]['path']
            for key,field in [('source_image_sha256','reference_image'),('source_tissue_mask_sha256','reference_tissue_mask'),('source_nuclei_mask_sha256','reference_nuclei_mask')]:raw['provenance'][key]=source['inputs'][field]['sha256']
            raw['provenance']['original_label_map_digest']=source['inputs']['reference_tissue_mask']['sha256'];raw['provenance']['original_instance_mask_digest']=source['inputs']['reference_nuclei_mask']['sha256']
            try:check=probe(raw,args.output/raw['case_id'],topology=args.topology_witness)
            except Exception as exc:check={'passed_necessary_capacity':False,'error':f'{type(exc).__name__}: {exc}'}
            record['checks'].append({'primitive_id':template['primitive_id'],**check})
        record['both_significant_topology_pass']=all(check.get('passed_topology_witness',False) for check in record['checks'])
        record['both_significant_capacity_pass']=all(check['passed_necessary_capacity'] for check in record['checks'])
        records.append(record);(args.output/'capacity_screen.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
    return 0

if __name__=='__main__':main()
