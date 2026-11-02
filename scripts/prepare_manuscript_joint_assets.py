#!/usr/bin/env python3
"""Freeze source-only selection and user-chosen joint budgets before H&E generation."""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from phase3_mask_edit.benchmark.experiment_identity import asset_identity, sha256_file


def patient_from_image(path):
    match=re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}',Path(path).name)
    if not match: raise ValueError('source image has no verifiable TCGA patient prefix')
    return match.group()


def prepare(rows, output, formal_count=30):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    rows=sorted(rows,key=lambda row:row['sample_id'])
    selected=[]; excluded=[]; seen=set()
    for row in rows:
        patient=patient_from_image(row['reference_image'])
        if row.get('patient_id') != patient:
            raise ValueError('manifest patient differs from image prefix')
        if patient in seen: continue
        image=np.asarray(Image.open(row['reference_image']).convert('RGB'))
        tissue=np.asarray(Image.open(row['reference_tissue_mask']))
        nuclei=np.asarray(Image.open(row['reference_nuclei_mask']))
        if tissue.ndim!=2 or image.shape[:2]!=tissue.shape or tissue.shape!=nuclei.shape:
            raise ValueError('source assets are not pixel aligned')
        reason=None
        if not np.isin(tissue,range(16)).all(): reason='unsupported_frozen_generator_tissue_ids'
        elif not (np.any(tissue==1) and np.any(tissue==2) and np.any(tissue==4)): reason='requires_source_tumor_stroma_and_immune_compartments'
        record={'reference_id':row['reference_id'],'patient_id':patient,'wsi_id':row['wsi_id'],
            'wsi_id_scope':'dataset WSI/patient split group, not a recovered full TCGA slide barcode',
            'inputs':{k:asset_identity(row[k]) for k in ('reference_image','reference_tissue_mask','reference_nuclei_mask')},
            'parent_counts':{str(i):int(np.count_nonzero(tissue==i)) for i in range(8)}}
        if reason:
            excluded.append({**record,'reason':reason});continue
        seen.add(patient)
        selected.append(record)
        if len(selected)==formal_count+1: break
    if len(selected)<formal_count+1: raise ValueError(f'need {formal_count+1} eligible independent patients; found {len(selected)}')
    technical=selected[0]; formal=selected[1:]
    rules={'scope':'source_and_mask_target_freeze_only','formal_generation_launched':False,
        'selection':'lexicographic sample_id, first eligible independent patient for technical chain, next 30 unique patients for prospective formal source list; eligibility uses source masks only',
        'strength_authorization':'user explicitly selected historical 19% / 32% whole-patch joint area budgets in this chat',
        'strengths':{'moderate':{'target_fraction':.19,'min_fraction':.14,'max_fraction':.24,'tissue_min_fraction':.14},
                     'significant':{'target_fraction':.32,'min_fraction':.27,'max_fraction':.37,'tissue_min_fraction':.27}},
        'budget_policy':{'basis':'whole_patch','relative_tolerance':.02,'fallback_policy':'max_feasible_below_target','capacity_floor_policy':'strict','minimum_effective_fraction':0.},
        'target_dose_intervals':{'moderate':[0.,1.],'significant':[0.,1.]},
        'dose_interpretation':'strengths controlled by disjoint whole-patch joint budgets; ROI fractions reported as measured, never called clinical grades',
        'minimum_target_gap':.05,'target_direction':1,'missing_and_abstention':'retain all failures; never lower dose or reuse a duplicate target to fill a strength',
        'technical_source':technical,'formal_sources':formal,'source_exclusions':excluded,
        'instance_authority':'source semantic CellViT mask and shared watershed; native JSON only when independently available and hash verified',
        'source_count':formal_count,'technical_exclusion_reference_ids':[technical['reference_id']],
        'technical_exclusion_patient_ids':[technical['patient_id']]}
    (output/'source_and_strength_freeze.json').write_text(json.dumps(rules,indent=2,ensure_ascii=False))
    cases=[]
    for code,primitive,mechanism,subject in [('u1','cohesive-boundary-expansion-v1','breast-annotation-anchored-boundary-growth','annotated tumor boundary into adjacent stroma'),('u2','generic-immune-infiltrate-increase-v1','breast-generic-immune-compartment-turnover','existing annotated immune compartment into adjacent stroma')]:
        for strength,budget in rules['strengths'].items():
            source=technical['inputs']; provenance={
                'source_image_sha256':source['reference_image']['sha256'],'source_tissue_mask_sha256':source['reference_tissue_mask']['sha256'],
                'source_nuclei_mask_sha256':source['reference_nuclei_mask']['sha256'],'original_label_map_digest':source['reference_tissue_mask']['sha256'],
                'original_instance_mask_digest':source['reference_nuclei_mask']['sha256'],'preprocessing_revision':'unchanged frozen BCSS paired300 source export',
                'joint_mechanism_id':mechanism,'joint_primitive_id':primitive,'available_auxiliary_structures':[],
                'instance_authority_source':'cellvit_semantic_mask_shared_watershed_v1','require_mature_probnet_regeneration':True,
                'interpretation_scope':'user-defined annotation-conditioned operation; no histologic subtype or clinical progression inferred'}
            cases.append({'case_id':technical['reference_id']+'__'+code+'__'+strength,'instruction':'Increase '+subject+' under the frozen '+strength+' joint-area budget; preserve other conditions.',
                'source_image_uri':source['reference_image']['path'],'source_tissue_mask_uri':source['reference_tissue_mask']['path'],'source_nuclei_mask_uri':source['reference_nuclei_mask']['path'],
                'pathology_domain_id':'breast-invasive-carcinoma-v1','annotation_profile_id':'bcss-semantic-v1','cell_observation_profile_id':'cellvit-five-class-v1','cell_population_profile_id':'breast-cellvit-source-first-v1',
                'primitive_id':primitive,'joint_area_budget':{**budget,**rules['budget_policy']},'pixel_size_um':.25,'seed':42,'provenance':provenance})
    (output/'technical_joint_cases.json').write_text(json.dumps(cases,indent=2))
    return rules


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=[json.loads(line) for line in args.source_manifest.read_text().splitlines() if line.strip()]
    rules=prepare(rows,args.output)
    (args.output/'source_manifest_identity.json').write_text(json.dumps(asset_identity(args.source_manifest),indent=2))
    print(json.dumps({'formal_sources':len(rules['formal_sources']),'technical_source':rules['technical_source']['reference_id'],'source_exclusions':len(rules['source_exclusions'])}))

if __name__=='__main__':main()
