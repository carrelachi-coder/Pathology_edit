"""Pre-generation checks for frozen research assets and target doses."""
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from .experiment_identity import sha256_file

ASSETS = ('generation_manifest', 'approved_joint_targets', 'protected_roi_files_and_hashes', 'patient_wsi_mapping_audit', 'training_overlap_audits', 'pilot_exclusion_audit')


def verified_asset(asset):
    if not isinstance(asset, dict) or asset.get('status') != 'frozen' or not asset.get('path') or not asset.get('sha256'):
        raise ValueError('required frozen asset is unresolved')
    path = Path(asset['path'])
    if not path.is_file() or sha256_file(path) != asset['sha256']:
        raise ValueError('frozen asset missing or content drift')
    return path


def validate_launch_scope(protocol, *, technical_validation=False):
    if protocol.get('status') != 'frozen' or protocol.get('blockers'):
        raise ValueError('execution remains blocked; protocol has not passed its prerequisites')
    if technical_validation:
        count = protocol.get('source_count')
        if (protocol.get('scope') != 'joint_gpu_technical_acceptance'
                or protocol.get('formal_launch_permitted') is not False
                or type(count) is not int or not 1 <= count <= 2):
            raise ValueError('technical validation requires a separate nonformal protocol with one or two excluded sources')
    elif (protocol.get('formal_launch_permitted') is not True
            or protocol.get('scope') == 'joint_gpu_technical_acceptance'):
        raise ValueError('formal cohort remains blocked; protocol has not passed its prerequisites')


def preflight(protocol, manifest, repo, *, technical_validation=False):
    field = 'technical_source_list_frozen' if technical_validation else 'formal_source_list_frozen'
    if not protocol.get('cohort_selection', {}).get(field):
        raise ValueError('technical source list is not frozen' if technical_validation else 'formal source list is not frozen')
    if technical_validation:
        validate_launch_scope(protocol, technical_validation=True)
    assets = {key: verified_asset(protocol.get('formal_assets', {}).get(key)) for key in ASSETS}
    if sha256_file(manifest) != sha256_file(assets['generation_manifest']):
        raise ValueError('formal manifest differs from frozen asset')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != protocol.get('implementation_commit'):
        raise ValueError('runtime code commit differs from frozen protocol')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=normal'], text=True).strip():
        raise ValueError('runtime checkout is dirty or contains untracked execution dependencies')
    import sys
    expected = protocol['runtime_environments']['generation_and_cellvit']['python_path']
    if Path(sys.executable).resolve() != Path(expected).resolve():
        raise ValueError('runtime interpreter differs from frozen environment')
    import importlib.metadata
    packages = protocol['runtime_environments']['generation_and_cellvit']['packages']
    required = {'torch', 'numpy', 'scipy', 'Pillow', 'diffusers', 'transformers', 'safetensors', 'torchvision'}
    if not required <= packages.keys() or any(importlib.metadata.version(name) != version for name, version in packages.items()):
        raise ValueError('execution dependencies differ from frozen environment')
    return {key: json.loads(path.read_text()) for key, path in assets.items() if key != 'generation_manifest'}


def coarse_tissue(path):
    """Map known fine labels; 255 stays ignore and is never tissue."""
    from phase3_mask_edit.core.mask_io import load_id_mask
    from dataset_config.unified_labels import FINE_TO_PARENT
    labels = load_id_mask(path)
    parents = np.full(labels.shape, -1, dtype=np.int16)
    for label in np.unique(labels):
        if int(label) == 255:
            continue
        if int(label) not in FINE_TO_PARENT:
            raise ValueError('unknown tissue label')
        parents[labels == label] = FINE_TO_PARENT[int(label)]
    return parents


def validate_source_roi(source_path, operation, roi_asset):
    parents = coarse_tissue(source_path)
    if operation['code'] == 'u1' and operation['target_parent'] == 1:
        expected = parents > 0
    elif operation['code'] == 'u2' and operation['target_parent'] == 4:
        expected = np.isin(parents, (2, 4))
    else:
        raise ValueError('unsupported measurement ROI definition')
    roi = np.asarray(Image.open(verified_asset(roi_asset))) > 0
    if roi.ndim != 2 or not np.array_equal(roi, expected) or not roi.any():
        raise ValueError('measurement ROI differs from protocol source definition or is empty')
    return roi


def roi_evidence(roi_asset):
    import hashlib
    path = verified_asset(roi_asset)
    roi = np.asarray(Image.open(path)) > 0
    if roi.ndim != 2:
        raise ValueError('ROI must be a two-dimensional mask')
    return {'file_sha256': sha256_file(path), 'shape': list(roi.shape),
            'denominator_pixels': int(roi.sum()),
            'binary_mask_sha256': hashlib.sha256(roi.astype(np.uint8).tobytes()).hexdigest()}


def target_dose(row, operation, roi_asset):
    roi = validate_source_roi(row['reference_tissue_mask'], operation, roi_asset)
    parents = coarse_tissue(row['target_tissue_mask'])
    if parents.shape != roi.shape or np.any(parents[roi] < 0):
        raise ValueError('target has ignore or invalid frame inside fixed source ROI')
    return float(np.mean(parents[roi] == operation['target_parent']))


def validate_target_pair(states, operation, frozen_targets, rois):
    doses = []
    signatures = []
    for strength in ('moderate', 'significant'):
        state = operation['code'] + '_' + strength
        row = states[state]; frozen = frozen_targets[state]
        handoff_path = verified_asset(frozen['handoff'])
        if sha256_file(row['joint_handoff_manifest']) != sha256_file(handoff_path):
            raise ValueError('strength is bound to a different approved handoff')
        handoff = json.loads(handoff_path.read_text())
        budget = {key: handoff['execution_contract'].get(key) for key in ('joint_area_budget', 'cell_count_extent_budget')}
        if budget != frozen['execution_budget'] or not any(budget.values()):
            raise ValueError('strength budget differs from frozen definition')
        signature = tuple(sha256_file(row[key]) for key in ('target_tissue_mask', 'target_nuclei_mask'))
        if list(signature) != frozen['target_sha256']:
            raise ValueError('strength target content drift')
        signatures.append(signature)
        dose = target_dose(row, operation, rois[operation['code'] + '_roi'])
        interval = operation['strengths'][strength]['dose_interval']
        if not isinstance(interval, list) or len(interval) != 2 or not isinstance(operation.get('minimum_target_gap'), (int, float)):
            raise ValueError('strength dose definitions remain unresolved')
        lo, hi = interval
        if not lo <= dose <= hi:
            raise ValueError('target dose is outside frozen strength interval')
        doses.append(dose)
    if signatures[0] == signatures[1]:
        raise ValueError('duplicate or unordered target strengths')
    source_row = dict(states['sham'], target_tissue_mask=states['sham']['reference_tissue_mask'])
    baseline = target_dose(source_row, operation, rois[operation['code'] + '_roi'])
    if operation['direction'] * (doses[0] - baseline) <= 0:
        raise ValueError('moderate target does not follow frozen direction from source')
    if signatures[0] == signatures[1] or not operation['direction'] * (doses[1] - doses[0]) >= operation['minimum_target_gap'] > 0:
        raise ValueError('duplicate or unordered target strengths')
