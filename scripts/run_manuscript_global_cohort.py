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


def validate_cohort(rows, *, source_count, handoff_validator, protocol, assets, technical_validation=False):
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
        exclusion = assets['pilot_exclusion_audit']
        for field in ('excluded_reference_ids', 'excluded_patient_ids'):
            if (not isinstance(exclusion.get(field), list) or not exclusion[field]
                    or any(not isinstance(value, str) or not value for value in exclusion[field])):
                raise ValueError('technical exclusion audit requires source and patient identities')
        excluded_source = sham['reference_id'] in exclusion['excluded_reference_ids']
        excluded_patient = sham['patient_id'] in exclusion['excluded_patient_ids']
        if technical_validation:
            if (not excluded_source or not excluded_patient
                    or sham['reference_id'] not in protocol.get('technical_reference_ids', [])
                    or sham['patient_id'] not in protocol.get('technical_patient_ids', [])):
                raise ValueError('technical source and patient must be reserved and excluded from formal analysis')
        elif excluded_source or excluded_patient:
            raise ValueError('pilot source or patient cannot enter formal cohort')
        from phase3_mask_edit.benchmark.manuscript_contract import verified_asset, validate_target_pair
        verified_asset(assets['protected_roi_files_and_hashes'][sham['reference_id']]['protected_roi'])
        import re
        match = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', Path(sham['reference_image']).name)
        if not match or match.group() != sham['patient_id']:
            raise ValueError('TCGA patient identity does not match the source filename')
        shared = ('reference_image', 'reference_tissue_mask', 'reference_nuclei_mask')
        source_hashes = {key: sha256_file(sham[key]) for key in shared}
        approved_joint_changes = []
        # Frozen HTE generation accepts unified IDs 0..15, not ignore=255.
        # Do not silently repaint an approved condition into background.
        from phase3_mask_edit.core.mask_io import load_id_mask
        from dataset_config.unified_labels import FINE_TO_PARENT
        for row in states.values():
            for key in ('reference_tissue_mask', 'target_tissue_mask'):
                if not np.isin(load_id_mask(row[key]), list(FINE_TO_PARENT)).all():
                    raise ValueError('formal generation tissue conditions require known IDs 0..15; ignore must be resolved before freezing')
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
                # Consume the same hash-bound semantic J already checked by
                # the handoff validator, including complete native footprints.
                joint_path = Path(handoff['paths']['joint_change'])
                if sha256_file(joint_path) != handoff['digests']['joint_change_sha256']:
                    raise ValueError('approved joint change content drift')
                joint = np.asarray(Image.open(joint_path)) > 0
                if joint.shape != expected_shape or joint.ndim != 2:
                    raise ValueError('approved joint change frame mismatch')
                approved_joint_changes.append(joint)
                for key in (*shared, 'target_tissue_mask', 'target_nuclei_mask'):
                    if sha256_file(getattr(inputs, key)) != sha256_file(row[key]):
                        raise ValueError(f'formal row differs from approved joint handoff: {key}')
        from phase3_mask_edit.benchmark.manuscript_contract import coarse_tissue
        from scipy.ndimage import binary_dilation
        source_parents = coarse_tissue(sham['reference_tissue_mask'])
        changed = np.logical_or.reduce(approved_joint_changes)
        expected = (source_parents > 0) & ~binary_dilation(changed, iterations=8)
        protected_path = verified_asset(assets['protected_roi_files_and_hashes'][sham['reference_id']]['protected_roi'])
        actual = np.asarray(Image.open(protected_path)) > 0
        if not np.array_equal(actual, expected):
            raise ValueError('protected ROI differs from frozen source/target construction')
        for operation in protocol['operations'].values():
            validate_target_pair(states, operation, assets['approved_joint_targets'][sham['reference_id']], assets['protected_roi_files_and_hashes'][sham['reference_id']])
    if technical_validation and (
            set(protocol.get('technical_reference_ids', [])) != set(by_source)
            or set(protocol.get('technical_patient_ids', [])) != patients):
        raise ValueError('technical reservation differs from the complete frozen source list')
    return by_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--cohort-manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--technical-validation', action='store_true',
        help='Use a separately frozen one/two-source technical protocol; formal launch stays disabled.')
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    from phase3_mask_edit.benchmark.manuscript_contract import preflight, validate_launch_scope
    validate_launch_scope(protocol, technical_validation=args.technical_validation)
    if args.technical_validation:
        assets = preflight(protocol, args.cohort_manifest, Path(__file__).resolve().parents[1], technical_validation=True)
    else:
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
    cohort = validate_cohort(rows, source_count=protocol['source_count'], handoff_validator=build_frozen_generator_inputs, protocol=protocol, assets=assets, technical_validation=args.technical_validation)
    from phase3_mask_edit.benchmark.manuscript_contract import roi_evidence
    evidence = {source: {name: roi_evidence(asset) for name, asset in
                assets['protected_roi_files_and_hashes'][source].items()}
                for source in cohort}
    args.output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_root / 'cohort_roi_validation.json'
    payload = json.dumps(evidence, sort_keys=True, indent=2)
    if evidence_path.exists() and evidence_path.read_text() != payload:
        raise ValueError('existing ROI validation evidence differs; use a new output root')
    evidence_path.write_text(payload)
    scope_path = args.output_root / 'cohort_scope_validation.json'
    scope_payload = json.dumps({
        'scope':'joint_gpu_technical_acceptance' if args.technical_validation else 'formal_manuscript_cohort',
        'formal_launch_permitted':protocol['formal_launch_permitted'],
        'protocol_sha256':sha256_file(args.protocol),
        'generation_manifest_sha256':sha256_file(args.cohort_manifest),
        'source_count':len(cohort),
        'expected_generated_states':sorted(EXPECTED_STATES),
    },sort_keys=True,indent=2)
    if scope_path.exists() and scope_path.read_text() != scope_payload:
        raise ValueError('existing execution scope differs; use a new output root')
    scope_path.write_text(scope_payload)
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
