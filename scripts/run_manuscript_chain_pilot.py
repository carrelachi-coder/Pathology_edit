#!/usr/bin/env python3
"""Nonformal source/sham/edit wiring check, never an endpoint-selection experiment."""
from __future__ import annotations
import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from phase3_mask_edit.benchmark.experiment_identity import asset_identity, sha256_file
from phase3_mask_edit.benchmark.conditional_fidelity import detections_from_cellvit_json
from scripts.run_phase3_inpaint_pipeline import _run_generation_stage, _release_generation_model_caches
from controlnet_train.inference.model_paths import DEFAULT_CROSS_V1_CHECKPOINT, DEFAULT_PIX2PIX_CHECKPOINT, DEFAULT_CELLVIT_MODEL, DEFAULT_CELLVIT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--segmentator-release', type=Path, default=Path('benchmark_configs/releases/segmentator_fine_legacy_anchor.json'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    row = sorted(rows, key=lambda item: item['sample_id'])[0]
    repo = Path(__file__).resolve().parents[1]
    release = json.loads(args.segmentator_release.read_text())
    identity = {
        'scope': 'nonformal_wiring_only_not_formal_joint_cohort',
        'selection': 'first sample_id lexicographically; fixed before generation; exclude its source from formal cohort',
        'manifest': asset_identity(args.manifest), 'sample_id': row['sample_id'],
        'inputs': {key: asset_identity(row[key]) for key in ('reference_image', 'reference_tissue_mask', 'reference_nuclei_mask', 'target_tissue_mask', 'target_nuclei_mask')},
        'models': {key: asset_identity(path) for key, path in (
            ('cross', DEFAULT_CROSS_V1_CHECKPOINT), ('pix2pix', DEFAULT_PIX2PIX_CHECKPOINT),
            ('segmentator', release['checkpoint']), ('cellvit', DEFAULT_CELLVIT_MODEL))},
        'segmentator_release': asset_identity(args.segmentator_release),
        'code_commit': (repo / 'SOURCE_COMMIT').read_text().strip() if (repo / 'SOURCE_COMMIT').is_file() else subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'code_file_sha256': {str(p.relative_to(repo)): sha256_file(p) for folder in ('scripts', 'controlnet_train', 'phase3_mask_edit', 'phase3_joint_edit_refine') for p in sorted((repo / folder).rglob('*.py'))},
        'environment': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda},
        'endpoint_stage': 'postprocessed_precomposite',
        'postprocess_eligibility': 'full_frame_same_for_both_arms; nuclei exclusions and adaptive protection masks logged per arm',
        'formal_launch_permitted': False,
    }
    (args.output / 'experiment_version.json').write_text(json.dumps(identity, indent=2))
    reference = np.asarray(Image.open(row['reference_image']).convert('RGB'))
    images = args.output / 'measurement_images'; images.mkdir()
    shutil.copyfile(row['reference_image'], images / 'original.png')
    records = []
    for state in ('sham', 'edited'):
        directory = args.output / state; directory.mkdir()
        params = SimpleNamespace(profile='BCSS', generation_mode='cross-v1', cross_backend='cross-v1', route_threshold=.35,
            pretrained_model_name_or_path='/data/huggingface/FLUX.1-dev', cross_v1_checkpoint=DEFAULT_CROSS_V1_CHECKPOINT,
            pix2pix_checkpoint=DEFAULT_PIX2PIX_CHECKPOINT, reference_image=Path(row['reference_image']),
            reference_tissue_mask=Path(row['reference_tissue_mask']), reference_nuclei_mask=Path(row['reference_nuclei_mask']),
            prompt='H&E stained breast cancer histopathology at 40x magnification', prompt_source='explicit',
            device=args.device, torch_dtype='bf16', num_inference_steps=28, guidance_scale=3.5,
            controlnet_conditioning_scale=1., seed=42, endpoint_stage='postprocessed_precomposite')
        endpoint, info = _run_generation_stage(args=params, output_dir=directory, reference_image=reference,
            change_region=np.ones(reference.shape[:2], dtype=bool),
            target_tissue_path=Path(row['reference_tissue_mask'] if state == 'sham' else row['target_tissue_mask']),
            target_nuclei_path=Path(row['reference_nuclei_mask'] if state == 'sham' else row['target_nuclei_mask']))
        if not info.get('real_generation_completed') or info['endpoint_stage'] != identity['endpoint_stage']:
            raise RuntimeError('invalid real-generation endpoint')
        shutil.copyfile(endpoint, images / f'{state}.png')
        protection = info['cross_v1']['pix2pix_v2']['cross_rgb_od_low_stain_protection']
        records.append({'state': state, 'endpoint': asset_identity(endpoint), 'postprocess': protection})
        (args.output / 'generation_checks.json').write_text(json.dumps(records, indent=2))
    if records[0]['postprocess']['configuration'] != records[1]['postprocess']['configuration'] or records[0]['postprocess']['eligibility_region_sha256'] != records[1]['postprocess']['eligibility_region_sha256']:
        raise RuntimeError('postprocessing eligibility/configuration mismatch')
    _release_generation_model_caches(); gc.collect(); torch.cuda.empty_cache()
    measurements = []
    source_roi = np.asarray(Image.open(row['reference_tissue_mask'])) > 0
    if source_roi.ndim != 2 or not source_roi.any():
        raise ValueError('invalid fixed source tissue ROI')
    for image in sorted(images.glob('*.png')):
        out = args.output / 'measurements' / image.stem
        subprocess.run([sys.executable, str(repo / 'scripts/predict_segmentator_mask.py'), '--release', str(args.segmentator_release),
            '--input', str(image), '--output-dir', str(out), '--profile', 'BCSS', '--device', args.device], check=True)
        mask_path = out / 'coarse_mask.png'
        mask = np.asarray(Image.open(mask_path))
        if mask.shape != source_roi.shape:
            raise ValueError('measurement frame mismatch')
        measurements.append({'state': image.stem, 'status': 'completed', 'mask': asset_identity(mask_path),
            'fixed_roi_pixels': int(source_roi.sum()), 'tumor_fraction': float(np.mean(mask[source_roi] == 1)),
            'inflammatory_compartment_fraction': float(np.mean(mask[source_roi] == 4))})
    subprocess.run([sys.executable, str(repo / 'batch_cellvit_ui_equiv.py'), '--images-dir', str(images), '--output-dir', str(args.output / 'cell_masks'),
        '--json-output-dir', str(args.output / 'cell_json'), '--raw-outdir', str(args.output / 'cell_raw'),
        '--model', DEFAULT_CELLVIT_MODEL, '--cellvit-root', DEFAULT_CELLVIT_ROOT, '--gpu', args.device.split(':')[-1], '--batch-size', '3'], check=True)
    for record in measurements:
        cells = args.output / 'cell_json' / (record['state'] + '.json')
        detections = detections_from_cellvit_json(cells, mpp=.25, image_size=(reference.shape[1], reference.shape[0]))
        record['cell_measurement_status'] = 'valid_zero' if not len(detections.xy) else 'valid_nonzero'
        record['cell_count'] = len(detections.xy)
        record['cell_json'] = asset_identity(cells)
    summary = {'status': 'complete', 'scope': identity['scope'], 'formal_launch_permitted': False,
        'real_generated': 2, 'requested_measurements': 3, 'completed_measurements': len(measurements),
        'postprocess_eligibility_match': True, 'measurements': measurements,
        'limitations': ['historical mask targets; formal joint handoff not exercised', 'pilot estimates cannot select formal endpoints or cases', 'no formal efficacy or inferential result']}
    (args.output / 'chain_pilot_report.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
