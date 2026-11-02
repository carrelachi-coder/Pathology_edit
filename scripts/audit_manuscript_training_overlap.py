#!/usr/bin/env python3
"""Record exact filename and patient-group overlap; do not infer independence."""
import argparse
import json
import re
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phase3_mask_edit.benchmark.experiment_identity import asset_identity


def index_strings(value):
    names=set();patients=set()
    def visit(item):
        if isinstance(item,dict):
            for child in item.values():visit(child)
        elif isinstance(item,list):
            for child in item:visit(child)
        elif isinstance(item,str):
            normal=item.replace('\\','/')
            if normal.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):names.add(normal.rsplit('/',1)[-1])
            patients.update(re.findall(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}',item))
    visit(value);return names,patients


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-freeze',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    freeze=json.loads(args.source_freeze.read_text());sources=freeze.get('candidate_sources') or [freeze['technical_source'],*freeze['formal_sources']]
    base=Path('/home/lyw/wqx-DL/flow-edit/FlowEdit-main')
    specs={
        'cross_train':(base/'phase5_runs/cross_meta/metadata_cross_train.json',None),
        'cross_val':(base/'phase5_runs/cross_meta/metadata_cross_val.json',None),
        'segmentator_coarse_train':(base/'segmentator_runs/stage4_multidataset_manifest.json','train'),
        'segmentator_coarse_val':(base/'segmentator_runs/stage4_multidataset_manifest.json','val'),
        'segmentator_fine_train':(Path('/data1/zhao/wqx/segmentator_manifests/grouped_seed42.json'),'train'),
        'segmentator_fine_val':(Path('/data1/zhao/wqx/segmentator_manifests/grouped_seed42.json'),'val'),
        'segmentator_fine_test':(Path('/data1/zhao/wqx/segmentator_manifests/grouped_seed42.json'),'test'),
    }
    loaded={};indices={};evidence={}
    for key,(path,split) in specs.items():
        if str(path) not in loaded:loaded[str(path)]=json.loads(path.read_text())
        payload=loaded[str(path)];records=payload[split] if split else payload
        indices[key]=index_strings(records);evidence[key]={'asset':asset_identity(path),'split':split,'filename_count':len(indices[key][0]),'patient_group_count':len(indices[key][1])}
        print('indexed '+key,flush=True)
    config_path=Path('/data/wqx/flowedit/pix2pix_texture_transfer_lazy_ver4_wsi_identity_i0_local_full_pyramid_v3_ft/config.json')
    pix_config=json.loads(config_path.read_text())
    pix_evidence={'asset':asset_identity(config_path),'metadata':pix_config.get('metadata'),'val_metadata':pix_config.get('val_metadata'),
        'lineage_limit':'config refers to a historical training directory; checkpoint tensor hashes are frozen separately, but archived config alone does not prove complete checkpoint-training lineage'}
    records={}
    for source in sources:
        name=Path(source['inputs']['reference_image']['path']).name;patient=source['patient_id']
        comparisons={key:{'exact_filename_referenced':name in names,'patient_group_referenced':patient in patients} for key,(names,patients) in indices.items()}
        records[source['reference_id']]={'status':'completed','patient_id':patient,'source_image_sha256':source['inputs']['reference_image']['sha256'],
            'generator':{'cross_manifest_comparisons':{k:v for k,v in comparisons.items() if k.startswith('cross_')},'pix2pix_training_config':pix_evidence,
                'base_model_patient_training_list':'unavailable; overlap unknown','interpretation':'observed references/potential training exposure; absence is not proof of independence'},
            'segmentator':{k:v for k,v in comparisons.items() if k.startswith('segmentator_')},
            'cellvit':{'status':'unknown_patient_overlap','reason':'no checkpoint-linked per-patient training manifest supplied; pretrained model may overlap TCGA-derived cohorts'},
            'publication_scope':'coupled computational readouts; not independent clinical validation'}
    payload={'source_freeze':asset_identity(args.source_freeze),'manifest_evidence':evidence,'cases':records,'formal_generation_launched':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise ValueError('preserve prior audit; use a new output path')
    args.output.write_text(json.dumps(payload,indent=2));print(json.dumps({'audited_sources':len(records),'output':str(args.output)}))

if __name__=='__main__':main()
