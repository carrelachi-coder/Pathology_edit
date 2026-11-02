#!/usr/bin/env python3
"""Append mask-only candidates from the same BCSS export; keep the old census."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from phase3_mask_edit.benchmark.experiment_identity import asset_identity
from scripts.prepare_manuscript_joint_assets import patient_from_image


def freeze(manifest_path, initial_path, exclusion_path, source_root, output, *, per_patient=12):
    if type(per_patient) is not int or per_patient < 1:
        raise ValueError('per-patient source cap must be positive')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source_root=Path(source_root).resolve()
    manifest=json.loads(Path(manifest_path).read_text())
    initial=json.loads(Path(initial_path).read_text())
    excluded=set(json.loads(Path(exclusion_path).read_text())['excluded_patient_ids'])
    old_names={Path(s['inputs']['reference_image']['path']).name for s in initial['candidate_sources']}
    lookup={};row_count=0
    for split in ('train','val','test'):
        for row in manifest[split]:
            if row.get('dataset_id')!='bcss':continue
            row_count+=1;name=row['image']
            if (Path(name).name != name or row['mask'] != name or row['nuclei'] != name
                    or Path(row['dataset_root']).resolve()!=source_root
                    or (row['images_dir'],row['masks_dir'],row['nuclei_dir'])!=('images','tissue_masks','nuclei_masks')):
                raise ValueError('supplemental source must belong to the same canonical BCSS export')
            patient=patient_from_image(name)
            if patient!=row['group_id']:raise ValueError('dataset group differs from patient prefix')
            if name in lookup and lookup[name]['patient_id']!=patient:raise ValueError('ambiguous source group')
            lookup[name]={'patient_id':patient,'name':name}
    selected=[];patient_counts=Counter();reasons=Counter()
    ledger_path=output/'source_screen_ledger.jsonl'
    with ledger_path.open('x') as ledger:
        for source in sorted(lookup.values(),key=lambda x:(x['patient_id'],x['name'])):
            patient=source['patient_id'];name=source['name'];reason=None
            if patient in excluded:reason='excluded_technical_patient'
            elif name in old_names:reason='already_in_initial_census'
            elif patient_counts[patient]>=per_patient:reason='unexamined_after_per_patient_cap'
            paths={field:source_root/folder/name for field,folder in (
                ('reference_image','images'),('reference_tissue_mask','tissue_masks'),('reference_nuclei_mask','nuclei_masks'))}
            if reason is None:
                try:
                    tissue=np.asarray(Image.open(paths['reference_tissue_mask']))
                    nuclei=np.asarray(Image.open(paths['reference_nuclei_mask']))
                    with Image.open(paths['reference_image']) as image:size=image.size
                    if tissue.shape!=(512,512) or nuclei.shape!=tissue.shape or size!=(512,512):reason='unsupported_or_unaligned_frame'
                    elif not np.isin(tissue,range(16)).all():reason='unsupported_generator_tissue_ids'
                    elif not all(np.any(tissue==label) for label in (1,2,4)):reason='missing_source_tumor_stroma_or_immune'
                    elif np.count_nonzero(tissue==2)<int(np.ceil(.27*tissue.size)):reason='stroma_below_27_percent_tissue_floor'
                    elif not np.isin(nuclei,[0,1,2,3,4,5,101,102,103,104,105]).all():reason='unsupported_nuclear_ids'
                except (OSError,ValueError) as exc:reason='source_read_error:'+type(exc).__name__
            if reason is None:
                record={'reference_id':'BCSS_joint_supplement_'+hashlib.sha256(name.encode()).hexdigest()[:12],
                    'patient_id':patient,'wsi_id':patient,
                    'wsi_id_scope':'retained dataset patient/WSI group alias; full slide barcode unavailable',
                    'supplemental_sequence_index':len(selected),
                    'inputs':{field:asset_identity(path) for field,path in paths.items()}}
                selected.append(record);patient_counts[patient]+=1;reason='candidate_for_expensive_qualification'
            reasons[reason]+=1
            ledger.write(json.dumps({**source,'status':reason})+'\n')
    payload={'candidate_sources':selected,'candidate_count':len(selected),
        'patient_count':len(patient_counts),'raw_dataset_manifest_rows':row_count,
        'raw_unique_images':len(lookup),'raw_patient_groups':len({s['patient_id'] for s in lookup.values()}),
        'dataset_manifest':asset_identity(manifest_path),'append_to_initial_census':asset_identity(initial_path),
        'technical_patient_exclusion':asset_identity(exclusion_path),'screen_ledger':asset_identity(ledger_path),
        'screen_counts':dict(reasons),'per_patient_cap':per_patient,
        'selection_rule':f'preserve the initial census first; append same-BCSS sources in patient/filename order, first {per_patient} passing cheap source-mask eligibility per patient; final selection still requires both Significant topology witnesses, four target gates and independent mask review; first fully qualified source per patient; no generated RGB or embedding selection',
        'cheap_eligibility':'aligned 512-square known tissue/nucleus IDs; existing fine Tumor1/Stroma2/Immune4 and at least 27% source Stroma; technical patients excluded',
        'authority':'user explicitly authorized supplementing the same BCSS dataset while retaining 30 patients and 19/32 budgets',
        'joint_budgets':[.19,.32],'budget_reduction_allowed':False,
        'formal_source_count_required':30,'formal_generation_launched':False,
        'scope':'supplemental candidate census only, not accepted targets or a frozen formal cohort'}
    (output/'supplemental_candidate_pool.json').write_text(json.dumps(payload,indent=2))
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('dataset-manifest','initial-census','exclusion-audit','source-root','output'):
        p.add_argument('--'+flag,type=Path,required=True)
    args=p.parse_args();d=freeze(args.dataset_manifest,args.initial_census,args.exclusion_audit,args.source_root,args.output)
    print(json.dumps({key:d[key] for key in ('candidate_count','patient_count','raw_unique_images','screen_counts')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
