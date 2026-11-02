#!/usr/bin/env python3
"""Bind retained source identities to the dataset's own split-group records."""
import argparse
import json
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase3_mask_edit.benchmark.experiment_identity import asset_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-freeze',type=Path,required=True);parser.add_argument('--dataset-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    freeze=json.loads(args.source_freeze.read_text());manifest=json.loads(args.dataset_manifest.read_text());lookup={}
    for split in ('train','val','test'):
        for row in manifest.get(split,[]):
            if row.get('dataset_id')=='bcss':lookup.setdefault(Path(row['image'].replace('\\','/')).name,set()).add(row['group_id'])
    mappings={};details={}
    sources=freeze.get('candidate_sources') or [freeze['technical_source'],*freeze['formal_sources']]
    for source in sources:
        name=Path(source['inputs']['reference_image']['path']).name
        groups=lookup.get(name,set())
        if groups!={source['patient_id']} or source['wsi_id']!=source['patient_id']:
            raise ValueError('source group is absent, ambiguous, or inconsistent with the legacy cohort patient identity')
        mappings[source['reference_id']]={'patient_id':source['patient_id'],'wsi_id':source['wsi_id']}
        details[source['reference_id']]={'dataset_group_id':source['wsi_id'],'full_slide_barcode':'not supplied by retained patch export',
            'independent_unit':'patient_id','grouping_limit':'wsi_id is the retained dataset patient/WSI group alias; no full slide barcode is asserted'}
    if args.output.exists():raise ValueError('preserve prior mapping audit')
    args.output.write_text(json.dumps({'mapping':mappings,'scope':details,'source_freeze':asset_identity(args.source_freeze),'dataset_manifest':asset_identity(args.dataset_manifest)},indent=2))
    print(json.dumps({'mapped_sources':len(mappings),'independent_unit':'patient_id','full_slide_barcode_recovered':False}))

if __name__=='__main__':main()
