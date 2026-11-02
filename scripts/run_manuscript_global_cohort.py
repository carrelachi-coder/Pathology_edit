#!/usr/bin/env python3
"""Fixed global reconstruction entry; a frozen protocol is mandatory."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from phase3_mask_edit.benchmark.experiment_identity import sha256_file

EXPECTED_STATES = {'sham', 'u1_moderate', 'u1_significant', 'u2_moderate', 'u2_significant'}


def validate_cohort(rows, *, source_count, handoff_validator, protocol, assets):
    by_source = {}
    for row in rows:
        source = row['reference_id']
        state = row['state']
        if state not in EXPECTED_STATES or state in by_source.setdefault(source, {}):
            raise ValueError('unknown or duplicated source state')
        if not row.get('patient_id') or not row.get('wsi_id') or not row.get('prompt'):
            raise ValueError('source identity and fixed source description prompt are mandatory')
        by_source[source][state] = row
    if len(by_source) != source_count:
        raise ValueError('source count differs from frozen protocol')
    patients = set()
    for states in by_source.values():
        if set(states) != EXPECTED_STATES:
            raise ValueError('incomplete paired five-output source')
        sham = states['sham']
        if sham['patient_id'] in patients:
            raise ValueError('minimum cohort requires one source per independent patient')
        patients.add(sham['patient_id'])
        audit = assets['training_overlap_audits'].get(sham['reference_id'], {})
        if audit.get('status') != 'completed' or not all(key in audit for key in ('generator', 'segmentator', 'cellvit')):
            raise ValueError('training overlap audit is incomplete')
        mapping = assets['patient_wsi_mapping_audit'][sham['reference_id']]
        if mapping != {'patient_id': sham['patient_id'], 'wsi_id': sham['wsi_id']}:
            raise ValueError('source mapping differs from frozen audit')
        if sham['reference_id'] in assets['pilot_exclusion_audit']['excluded_reference_ids']:
            raise ValueError('pilot source cannot enter formal cohort')
        from phase3_mask_edit.benchmark.manuscript_contract import verified_asset, validate_target_pair
        verified_asset(assets['protected_roi_files_and_hashes'][sham['reference_id']]['protected_roi'])
        import re
        match = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', Path(sham['reference_image']).name)
        if not match or match.group() != sham['patient_id']:
            raise ValueError('TCGA patient identity does not match the source filename')
        shared = ('reference_image', 'reference_tissue_mask', 'reference_nuclei_mask')
        source_hashes = {key: sha256_file(sham[key]) for key in shared}
        for state, row in states.items():
            for key in shared:
                if sha256_file(row[key]) != source_hashes[key]:
                    raise ValueError('source asset differs between paired states')
            for key in ('prompt', 'generation_seed', 'patient_id', 'wsi_id', 'profile'):
                if row.get(key) != sham.get(key):
                    raise ValueError(f'paired generation condition differs: {key}')
            from scripts.run_embedding_utility_generation import _generation_change_region
            support = np.asarray(Image.open(row['postprocess_eligibility_region']).convert('L'))
            with Image.open(row['reference_image']) as image:
                expected_shape = (image.height, image.width)
            if support.shape != expected_shape or not np.all(support > 0):
                raise ValueError('global cohort requires full-frame shared postprocess eligibility')
            if state == 'sham':
                if sha256_file(row['target_tissue_mask']) != source_hashes['reference_tissue_mask'] or sha256_file(row['target_nuclei_mask']) != source_hashes['reference_nuclei_mask']:
                    raise ValueError('sham target differs from source conditions')
            else:
                inputs, _, handoff = handoff_validator(row['joint_handoff_manifest'], output_dir=Path(row['joint_handoff_manifest']).parent / 'validation-only')
                operation = protocol['operations'][state.split('_')[0]]
                expected_primitive = operation['primitive_id']
                if handoff['mechanism_id'] != operation['mechanism_id']:
                    raise ValueError('mechanism differs from frozen operation')
                if handoff['primitive_id'] != expected_primitive:
                    raise ValueError('operation label differs from approved joint primitive')
                if sha256_file(_generation_change_region(row)) != sha256_file(inputs.generation_change_region):
                    raise ValueError('generation support differs from approved joint handoff')
                for key in (*shared, 'target_tissue_mask', 'target_nuclei_mask'):
                    if sha256_file(getattr(inputs, key)) != sha256_file(row[key]):
                        raise ValueError(f'formal row differs from approved joint handoff: {key}')
        from phase3_mask_edit.core.mask_io import load_id_mask
        from scipy.ndimage import binary_dilation
        source_tissue = load_id_mask(sham['reference_tissue_mask'])
        source_nuclei = load_id_mask(sham['reference_nuclei_mask'])
        changed = np.zeros_like(source_tissue, dtype=bool)
        for row in states.values():
            changed |= (load_id_mask(row['target_tissue_mask']) != source_tissue) | (load_id_mask(row['target_nuclei_mask']) != source_nuclei)
        expected = (source_tissue > 0) & (source_tissue != 255) & ~binary_dilation(changed, iterations=8)
        protected_path = verified_asset(assets['protected_roi_files_and_hashes'][sham['reference_id']]['protected_roi'])
        actual = np.asarray(Image.open(protected_path)) > 0
        if not np.array_equal(actual, expected):
            raise ValueError('protected ROI differs from frozen source/target construction')
        for operation in protocol['operations'].values():
            validate_target_pair(states, operation, assets['approved_joint_targets'][sham['reference_id']], assets['protected_roi_files_and_hashes'][sham['reference_id']])
    return by_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--cohort-manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get('status') != 'frozen' or protocol.get('blockers') or not protocol.get('formal_launch_permitted'):
        raise ValueError('formal cohort remains blocked; protocol has not passed its prerequisites')
    from phase3_mask_edit.benchmark.manuscript_contract import preflight
    assets = preflight(protocol, args.cohort_manifest, Path(__file__).resolve().parents[1])
    if sha256_file(args.cohort_manifest) != protocol['generation_manifest_sha256']:
        raise ValueError('cohort manifest content drift')
    if protocol['endpoint_stage'] != 'postprocessed_precomposite' or protocol['cluster_unit'] != 'patient_id':
        raise ValueError('unsupported frozen endpoint or independent unit')
    from phase3_joint_edit_refine.generator_adapter import build_frozen_generator_inputs
    rows = [json.loads(line) for line in args.cohort_manifest.read_text().splitlines() if line.strip()]
    if any(row.get('prompt') != protocol['sampling']['prompt'] for row in rows):
        raise ValueError('prompt differs from frozen sampling')
    if any(row.get('generation_seed') != protocol['sampling']['seed'] for row in rows):
        raise ValueError('manifest seed differs from frozen sampling')
    validate_cohort(rows, source_count=protocol['source_count'], handoff_validator=build_frozen_generator_inputs, protocol=protocol, assets=assets)
    from scripts.run_embedding_utility_generation import main as generate
    sampling = protocol['sampling']
    command = ['--cohort-manifest', str(args.cohort_manifest), '--output-root', str(args.output_root),
        '--backend', 'cross-v1', '--endpoint-stage', 'postprocessed_precomposite', '--expected-count', str(len(rows)),
        '--device', args.device, '--dtype', sampling['dtype'], '--num-inference-steps', str(sampling['steps']),
        '--guidance-scale', str(sampling['guidance']), '--controlnet-conditioning-scale', str(sampling['conditioning_scale'])]
    for flag, key in (('--pretrained-model', 'base_model'), ('--cross-v1-checkpoint', 'cross'), ('--pix2pix-checkpoint', 'pix2pix')):
        from phase3_mask_edit.benchmark.experiment_identity import asset_identity
        asset = protocol['model_assets'][key]
        if asset_identity(asset['path'])['sha256'] != asset['sha256']:
            raise ValueError(f'frozen model content drift: {key}')
        command += [flag, asset['path']]
    if args.resume:
        command += ['--resume']
    return generate(command)


if __name__ == '__main__':
    raise SystemExit(main())
