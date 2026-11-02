#!/usr/bin/env python3
"""Resume measurement of immutable pilot outputs after an environment failure."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from phase3_mask_edit.benchmark.experiment_identity import asset_identity, sha256_file
from phase3_mask_edit.benchmark.conditional_fidelity import detections_from_cellvit_json
from controlnet_train.inference.model_paths import DEFAULT_CELLVIT_MODEL, DEFAULT_CELLVIT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--segmentator-python', default='/home/lyw/anaconda3/envs/pathology-segmentator-mmseg/bin/python')
    parser.add_argument('--segmentator-release', type=Path, default=Path('benchmark_configs/releases/segmentator_fine_legacy_anchor.json'))
    parser.add_argument('--device', default='cuda:6')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    version_path = args.generation_output / 'experiment_version.json'
    version = json.loads(version_path.read_text())
    checks_path = args.generation_output / 'generation_checks.json'
    checks = json.loads(checks_path.read_text())
    if {r['state'] for r in checks} != {'sham', 'edited'} or version['endpoint_stage'] != 'postprocessed_precomposite':
        raise ValueError('pilot has no complete paired precomposite generation')
    for record in checks:
        if asset_identity(record['endpoint']['path']) != record['endpoint']:
            raise ValueError('generated endpoint content drift')
        if sha256_file(args.generation_output / 'measurement_images' / (record['state'] + '.png')) != record['endpoint']['sha256']:
            raise ValueError('measurement image differs from generated endpoint')
    for asset in version['inputs'].values():
        if asset_identity(asset['path']) != asset:
            raise ValueError('frozen source/target content drift')
    first, second = (record['postprocess'] for record in checks)
    if first['configuration'] != second['configuration'] or first['eligibility_region_sha256'] != second['eligibility_region_sha256']:
        raise ValueError('postprocessing configuration or eligibility differs')
    release = json.loads(args.segmentator_release.read_text())
    if sha256_file(release['checkpoint']) != version['models']['segmentator']['sha256'] or sha256_file(DEFAULT_CELLVIT_MODEL) != version['models']['cellvit']['sha256']:
        raise ValueError('measurement evaluator content drift')
    images = args.generation_output / 'measurement_images'
    if sha256_file(images/'original.png') != version['inputs']['reference_image']['sha256']:
        raise ValueError('original measurement image content drift')
    identity = {'generation_version': asset_identity(version_path), 'generation_checks': asset_identity(checks_path),
        'measurement_code_commit': (repo/'SOURCE_COMMIT').read_text().strip() if (repo/'SOURCE_COMMIT').exists() else subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'segmentator_python': args.segmentator_python, 'parent_python': sys.executable,
        'package_versions': {name: importlib.metadata.version(name) for name in ('torch','numpy','scipy','Pillow')},
        'segmentator_release': asset_identity(args.segmentator_release),
        'measurement_script_sha256': sha256_file(__file__), 'formal_launch_permitted': False}
    (args.output/'measurement_version.json').write_text(json.dumps(identity,indent=2))
    source = np.asarray(Image.open(version['inputs']['reference_tissue_mask']['path']))
    roi = (source > 0) & (source != 255)
    if roi.ndim != 2 or not roi.any():
        raise ValueError('invalid source tissue ROI')
    measurements = []
    for image in sorted(images.glob('*.png')):
        out = args.output/'segmentator'/image.stem
        subprocess.run([args.segmentator_python,str(repo/'scripts/predict_segmentator_mask.py'), '--release',str(args.segmentator_release),
            '--input',str(image),'--output-dir',str(out),'--profile','BCSS','--device',args.device],check=True)
        mask_path = out/'coarse_mask.png'; mask=np.asarray(Image.open(mask_path))
        if mask.shape != roi.shape:
            raise ValueError('prediction frame differs from frozen source')
        measurements.append({'state':image.stem,'status':'completed','mask':asset_identity(mask_path),
            'fixed_roi_pixels':int(roi.sum()),'tumor_fraction':float(np.mean(mask[roi]==1)),
            'inflammatory_compartment_fraction':float(np.mean(mask[roi]==4))})
        (args.output/'partial_measurements.json').write_text(json.dumps(measurements,indent=2))
    subprocess.run([sys.executable,str(repo/'batch_cellvit_ui_equiv.py'),'--images-dir',str(images), '--output-dir',str(args.output/'cell_masks'),
        '--json-output-dir',str(args.output/'cell_json'),'--raw-outdir',str(args.output/'cell_raw'), '--model',DEFAULT_CELLVIT_MODEL,
        '--cellvit-root',DEFAULT_CELLVIT_ROOT,'--gpu',args.device.split(':')[-1],'--batch-size','3'],check=True)
    for record in measurements:
        cells=args.output/'cell_json'/(record['state']+'.json')
        detection=detections_from_cellvit_json(cells,mpp=.25,image_size=(roi.shape[1],roi.shape[0]))
        record.update(cell_measurement_status='valid_zero' if not len(detection.xy) else 'valid_nonzero',cell_count=len(detection.xy),cell_json=asset_identity(cells))
    summary={'status':'complete','scope':'nonformal_multienvironment_wiring_only','formal_launch_permitted':False,
        'real_generated':2,'requested_measurements':3,'completed_measurements':len(measurements),
        'postprocess_eligibility_match':True,'generation_code_commit':version['code_commit'],
        'measurement_code_commit':identity['measurement_code_commit'],'measurements':measurements,
        'limitations':['historical targets; approved formal joint cohort not exercised','technical pilot readouts are not efficacy results','different stage commits are recorded explicitly']}
    (args.output/'chain_pilot_report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
