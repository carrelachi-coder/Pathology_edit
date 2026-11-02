#!/usr/bin/env python3
"""Freeze the entire retained 300-source pool before mask-only qualification."""
import argparse,json,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase3_mask_edit.benchmark.experiment_identity import asset_identity
from scripts.prepare_manuscript_joint_assets import patient_from_image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('candidate census is immutable; choose new output')
    rows=sorted([json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()],key=lambda row:row['sample_id'])
    records=[];seen=set()
    for row in rows:
        if row['reference_id'] in seen:raise ValueError('duplicate candidate source')
        seen.add(row['reference_id']);patient=patient_from_image(row['reference_image'])
        if patient!=row['patient_id']:raise ValueError('patient identity drift')
        records.append({'reference_id':row['reference_id'],'patient_id':patient,'wsi_id':row['wsi_id'],
            'wsi_id_scope':'retained dataset patient/WSI split-group alias; full slide barcode unavailable',
            'inputs':{k:asset_identity(row[k]) for k in ('reference_image','reference_tissue_mask','reference_nuclei_mask')}})
    payload={'source_manifest':asset_identity(args.manifest),'candidate_sources':records,'candidate_count':len(records),'patient_count':len({r['patient_id'] for r in records}),
        'formal_source_count_required':30,'formal_generation_launched':False,
        'selection_rule':'source-only necessary capacity, then bounded whole-mask topology witness, then complete four-target gates and independent mask-condition review; first qualifying source per patient in sample_id order; exclude every technical patient; never select by generated RGB or embedding',
        'scope':'candidate census, not an accepted formal cohort; transient topology witnesses are not frozen targets',
        'joint_budgets':[.19,.32],'budget_reduction_allowed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2));print(json.dumps({'candidates':len(records),'patients':payload['patient_count']}))
if __name__=='__main__':main()
