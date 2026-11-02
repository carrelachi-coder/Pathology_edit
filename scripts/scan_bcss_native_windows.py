#!/usr/bin/env python3
"""Read-only native BCSS lattice audit and deterministic source-only recovery census.

The pilot verifies geometry only. A full scan requires a separately frozen policy.
No ProbNet, target compilation or H&E generation is called here.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image

SIZE = 512
STRIDE = 256
REMAP = [0, 1, 2, 4, 3, 2, 7, 0, 7, 7, 4, 4, 7, 5, 6, 0, 7, 7, 6, 15, 14, 7]
NUCLEI_IDS = [0, 1, 2, 3, 4, 5, 101, 102, 103, 104, 105]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(array):
    """Decoded uint8 content identity, independent of PNG compression/metadata."""
    array = np.ascontiguousarray(array, dtype=np.uint8)
    header = json.dumps({'shape': list(array.shape), 'dtype': 'uint8'}, sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + array.tobytes()).hexdigest()


def windows(width, height):
    return [(y, x) for y in range(0, height-SIZE+1, STRIDE)
            for x in range(0, width-SIZE+1, STRIDE)]


def neighbors(roi, y, x, width, height):
    return [{'window_id': f'{roi}:y{yy}:x{xx}',
             'overlap_pixels': (SIZE-abs(y-yy))*(SIZE-abs(x-xx))}
            for yy in (y-STRIDE, y, y+STRIDE)
            for xx in (x-STRIDE, x, x+STRIDE)
            if (yy, xx) != (y, x) and yy >= 0 and xx >= 0
            and yy+SIZE <= height and xx+SIZE <= width]


def cheap(tissue, nuclei):
    reasons = []
    if tissue.shape != (SIZE, SIZE) or nuclei.shape != tissue.shape:
        return ['unaligned_frame']
    if not np.isin(tissue, range(16)).all(): reasons.append('unknown_tissue_id')
    if not np.isin(nuclei, NUCLEI_IDS).all(): reasons.append('unknown_nucleus_id')
    for label, name in [(1, 'tumor'), (2, 'stroma'), (4, 'immune')]:
        if not np.any(tissue == label): reasons.append('missing_'+name)
    if int(np.count_nonzero(tissue == 2)) < 70779:
        reasons.append('stroma_below_27_percent')
    return reasons


def read_array(path, mode=None, expected=None):
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError('source content drift: '+str(path))
    with Image.open(io.BytesIO(data)) as image:
        result = np.array(image.convert(mode) if mode else image)
    if result.dtype != np.uint8:
        raise ValueError('unexpected source dtype')
    return result, actual


def triplet(rgb, tissue, nuclei):
    return tuple(digest(a) for a in (rgb, tissue, nuclei))


def roi_parts(name):
    match = re.fullmatch(r'(TCGA-..-....)-DX\d+_xmin(\d+)_ymin(\d+)_MPP-0\.2500\.png', name)
    if match is None: raise ValueError('unsupported ROI identity: '+name)
    return match[1], int(match[2]), int(match[3])


def index_old(root, names, output):
    by_coord, by_content = {}, {}
    with (output/'old_export_index.jsonl').open('x') as handle:
        for i, name in enumerate(sorted(names)):
            match = re.fullmatch(r'(TCGA-..-....)_x(\d+)_y(\d+)_py(\d+)_px(\d+)\.png', name)
            if match is None: raise ValueError('unsupported exported identity')
            arrays, identities, errors = [], {}, {}
            for folder, mode in [('images', 'RGB'), ('tissue_masks', None), ('nuclei_masks', None)]:
                path = root/'BCSS_PATCHES'/folder/name
                try:
                    a, file_hash = read_array(path, mode)
                    arrays.append(a)
                except (OSError, ValueError) as exc:
                    arrays.append(None); file_hash = sha(path)
                    errors[folder] = type(exc).__name__+': '+str(exc)
                identities[folder] = file_hash
            hashes = triplet(*arrays) if not errors else None
            key = (match[1], int(match[2]), int(match[3]), int(match[4]), int(match[5]))
            if key in by_coord: raise ValueError('ambiguous old coordinates')
            by_coord[key] = {'name': name, 'hashes': hashes, 'errors': errors,
                'readable_hashes': [digest(a) if a is not None else None for a in arrays]}
            if hashes is not None: by_content.setdefault(hashes, []).append(name)
            handle.write(json.dumps({'name': name, 'coordinate_key': key,
                'file_sha256': identities, 'decoded_sha256': hashes,
                'decode_errors': errors})+'\n')
            if (i+1) % 2000 == 0: print(json.dumps({'old_indexed': i+1}), flush=True)
    return by_coord, by_content


def conservative_native_aliases(root, records, by_coord, by_content, lookup, output):
    """Never repair old files: bind intended native crops to exclude duplicates.

    Every old coordinate remains excluded. Invalid old PNGs remain invalid;
    native crop identities add conservative exclusions, not verified old pixels.
    """
    aliases = []
    for record in records:
        patient, ox, oy = roi_parts(record['raw_name'])
        bad = [(key, value) for key, value in by_coord.items()
               if key[:3] == (patient,ox,oy) and value['errors']]
        if not bad: continue
        arrays = [read_array(root/folder/record['raw_name'], mode,
            record['inputs'][key]['sha256'])[0] for key,folder,mode in
            [('rgb','rgbs','RGB'),('tissue','masks',None),('nuclei','nuclei_masks',None)]]
        rgb, raw, nuclei = arrays; tissue = lookup[raw]
        for key, value in bad:
            y,x = key[-2:]; crop = (rgb[y:y+SIZE,x:x+SIZE], tissue[y:y+SIZE,x:x+SIZE], nuclei[y:y+SIZE,x:x+SIZE])
            if any(a.shape[:2] != (SIZE,SIZE) for a in crop):
                raise ValueError('invalid old coordinate cannot be conservatively bound')
            hashes = triplet(*crop)
            if any(saved is not None and saved != actual for saved,actual in zip(value['readable_hashes'],hashes)):
                raise ValueError('readable old component differs from native coordinate')
            value['hashes'] = hashes
            by_content.setdefault(hashes, []).append(value['name'])
            aliases.append({'old_name': value['name'], 'decode_errors': value['errors'],
                'native_coordinate_content_sha256': hashes, 'old_pixel_identity_verified': False,
                'action': 'exclude old coordinate and native-content duplicates; preserve invalid old file'})
    (output/'invalid_old_native_aliases.json').write_text(json.dumps(aliases,indent=2))
    return aliases


def scan(args):
    root = args.root.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit = json.loads(args.source_audit.read_text())
    records = sorted(audit, key=lambda r: (r['raw_name'][:12], r['raw_name']))
    mode = 'geometry_pilot' if args.pilot else 'frozen_source_only_scan'
    policy = None
    if not args.pilot:
        policy = json.loads(args.policy.read_text())
        if (policy['status'] != 'frozen' or policy['patch_size'] != SIZE
                or policy['stride'] != STRIDE or policy['remap_22_to_unified'] != REMAP
                or policy['source_audit_sha256'] != sha(args.source_audit)
                or policy['scan_script_sha256'] != sha(__file__)
                or policy['qualification']['nuclear_ids'] != NUCLEI_IDS
                or policy['qualification']['required_tissue_ids'] != [1, 2, 4]
                or policy['qualification']['stroma_minimum_pixels'] != 70779
                or policy['padding_allowed'] is not False
                or policy['coordinate_origin'] != 'native_roi_top_left'
                or policy['invalid_old_export_policy'] != 'preserve_error_exclude_coordinate_and_native_content_alias'
                or policy['formal_launch_permitted'] is not False):
            raise ValueError('unsupported or drifted frozen policy')
        if sha(args.exclusion) != policy['technical_exclusion_sha256']:
            raise ValueError('technical exclusion drift')
        if sha(args.pilot_receipt) != policy['geometry_pilot_sha256']:
            raise ValueError('pilot receipt drift')
        if not json.loads(args.pilot_receipt.read_text())['geometry_verified']:
            raise ValueError('geometry pilot not accepted')
    excluded = set(json.loads(args.exclusion.read_text())['excluded_patient_ids'])
    script = root/'prepare_bcss_dataset.py'
    lookup = np.zeros(256, np.uint8)
    for node in ast.parse(script.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == 'REMAP_TABLE':
                lookup[ast.literal_eval(target.slice)] = ast.literal_eval(node.value)
    if lookup[:22].tolist() != REMAP: raise ValueError('historical label remap changed')
    expected_names = {r['raw_name'] for r in records}
    for folder in ('rgbs', 'masks', 'nuclei_masks'):
        if {f.name for f in (root/folder).glob('*.png')} != expected_names:
            raise ValueError('raw source list differs from audited 151 ROI set')
    if args.pilot: records = records[:2]
    names = {f.name for f in (root/'BCSS_PATCHES/images').glob('*.png')}
    for folder in ('tissue_masks', 'nuclei_masks'):
        if {f.name for f in (root/'BCSS_PATCHES'/folder).glob('*.png')} != names:
            raise ValueError('old export triplets differ')
    if policy and (len(names) != policy['old_export_count'] or
            hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest() != policy['old_export_filename_set_sha256']):
        raise ValueError('old export source set drift')
    if args.pilot: names = {n for n in names if n[:12] in {r['raw_name'][:12] for r in records}}
    identity = {'scope': mode, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'source_audit_sha256': sha(args.source_audit), 'script_sha256': sha(__file__),
        'exclusion_sha256': sha(args.exclusion), 'historical_prepare_script_sha256': sha(script),
        'policy_sha256': sha(args.policy) if policy else None,
        'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'status': 'started', 'formal_launch_permitted': False}
    def save_identity(): (output/'execution_identity.json').write_text(json.dumps(identity, indent=2))
    save_identity()
    try:
        by_coord, by_content = index_old(root, names, output)
        aliases = conservative_native_aliases(root, records, by_coord, by_content, lookup, output)
        invalid_names = {r['old_name'] for r in aliases}
        reproduced, window_counts, new_seen = set(), [], {}
        total_reasons, primary, candidate_sources = Counter(), Counter(), []
        with (output/'windows.jsonl').open('x') as manifest:
            for index, record in enumerate(records):
                name = record['raw_name']; roi = Path(name).stem
                patient, origin_x, origin_y = roi_parts(name)
                arrays = [read_array(root/folder/name, mode, record['inputs'][key]['sha256'])[0]
                    for key, folder, mode in [('rgb', 'rgbs', 'RGB'), ('tissue', 'masks', None), ('nuclei', 'nuclei_masks', None)]]
                rgb, raw, nuclei = arrays; height, width = rgb.shape[:2]
                if raw.shape != (height, width) or nuclei.shape != raw.shape or not np.isin(raw, range(22)).all():
                    raise ValueError('native alignment or original-label failure')
                tissue = lookup[raw]; stats = Counter(); multi = Counter()
                for y, x in windows(width, height):
                    crop = (rgb[y:y+SIZE, x:x+SIZE], tissue[y:y+SIZE, x:x+SIZE], nuclei[y:y+SIZE, x:x+SIZE])
                    hashes = triplet(*crop); key = (patient, origin_x, origin_y, y, x)
                    old = by_coord.get(key)
                    if old and hashes != old['hashes']: raise ValueError('old crop fails pixel reconstruction: '+old['name'])
                    if old: reproduced.add(old['name'])
                    row = {'window_id': f'{roi}:y{y}:x{x}', 'roi_id': roi,
                        'patient_id': patient, 'roi_origin_base_pixels': [origin_x, origin_y],
                        'crop_xy': [x, y], 'size_wh': [SIZE, SIZE], 'decoded_sha256': hashes,
                        'same_coordinate_old_export': old['name'] if old else None,
                        'old_decode_errors': old['errors'] if old else {},
                        'content_equal_readable_old_exports': [n for n in by_content.get(hashes, []) if n not in invalid_names],
                        'conservative_native_aliases_of_invalid_old_exports': [n for n in by_content.get(hashes, []) if n in invalid_names],
                        'overlaps': neighbors(roi, y, x, width, height)}
                    stats['windows'] += 1
                    if args.pilot:
                        row['status'] = 'geometry_only_no_qualification'
                    else:
                        reasons = cheap(crop[1], crop[2]); multi.update(reasons)
                        row.update(cheap_failure_reasons=reasons,
                            stroma_pixels=int(np.count_nonzero(crop[1] == 2)),
                            historical_background_fraction=float(np.mean(crop[1] == 0)))
                        if patient in excluded: status = 'excluded_technical_patient'
                        elif row['content_equal_readable_old_exports']: status = 'duplicate_old_content'
                        elif row['conservative_native_aliases_of_invalid_old_exports']: status = 'excluded_invalid_old_native_alias'
                        elif old: status = 'duplicate_old_coordinate'
                        elif hashes in new_seen: status = 'duplicate_new_content'
                        elif reasons: status = 'cheap_failed'
                        else: status = 'new_cheap_pass_not_capacity_qualified'
                        row['status'] = status
                        row['same_content_first_new_window'] = new_seen.get(hashes)
                        new_seen.setdefault(hashes, row['window_id'])
                        stats[status] += 1; primary[status] += 1
                        if not reasons: stats['cheap_pass_including_duplicates_and_technical'] += 1
                        if status == 'new_cheap_pass_not_capacity_qualified':
                            reference = 'BCSS_native_recovery_'+hashlib.sha256(row['window_id'].encode()).hexdigest()[:16]
                            inputs = {}
                            for field, folder, a in zip(('reference_image','reference_tissue_mask','reference_nuclei_mask'), ('images','tissue_masks','nuclei_masks'), crop):
                                path = output/'new_sources'/folder/(reference+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                                Image.fromarray(a).save(path)
                                inputs[field] = {'path': str(path), 'sha256': sha(path)}
                            candidate_sources.append({'reference_id': reference, 'patient_id': patient,
                                'wsi_id': patient, 'wsi_id_scope': 'retained alias; full slide metadata is a separate asset',
                                'inputs': inputs, 'source_window_id': row['window_id'], 'source_crop_xy': [x,y],
                                'source_roi_id': roi, 'decoded_sha256': hashes})
                            row['reference_id'] = reference
                    manifest.write(json.dumps(row)+'\n')
                window_counts.append({'patient_id': patient, 'roi_id': roi, 'frame_wh': [width,height],
                    'counts': dict(stats), 'independent_cheap_failure_counts': dict(multi)})
                total_reasons.update(multi)
                print(json.dumps({'roi_completed': index+1, 'roi_requested': len(records),
                    'windows_this_roi': stats['windows'], 'new_cheap_sources': len(candidate_sources)}), flush=True)
                (output/'patient_progress.json').write_text(json.dumps(window_counts,indent=2))
        if reproduced != names: raise ValueError('not all old exported coordinates were reproduced')
        summary = {'scope': mode, 'roi_count': len(records), 'patient_count': len(window_counts),
            'patch_size': SIZE, 'stride': STRIDE, 'window_count': sum(r['counts']['windows'] for r in window_counts),
            'old_export_indexed': len(names), 'old_export_coordinates_accounted': len(reproduced),
            'old_export_pixel_reconstructed': len(reproduced)-len(aliases),
            'invalid_old_exports_conservatively_excluded': len(aliases),
            'geometry_verified': True, 'independent_cheap_failure_counts': dict(total_reasons),
            'status_counts': dict(primary), 'new_cheap_sources': len(candidate_sources),
            'new_cheap_patient_groups': len({r['patient_id'] for r in candidate_sources}),
            'capacity_checks_run': 0, 'topology_checks_run': 0, 'four_target_compilations_run': 0,
            'new_qualified_patients': 0, 'formal_launch_permitted': False,
            'window_manifest_sha256': sha(output/'windows.jsonl'),
            'old_export_index_sha256': sha(output/'old_export_index.jsonl'),
            'patient_summary_sha256': sha(output/'patient_progress.json')}
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        if policy:
            (output/'candidate_sources.json').write_text(json.dumps({'candidate_sources': candidate_sources,
                'scope': 'source-only cheap pass; not capacity/topology/target-qualified',
                'policy_sha256': sha(args.policy), 'summary_sha256': sha(output/'summary.json')},indent=2))
        identity.update(status='complete', summary_sha256=sha(output/'summary.json'));save_identity()
        return summary
    except Exception as exc:
        identity.update(status='failed', error=type(exc).__name__+': '+str(exc));save_identity();raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('root','source-audit','exclusion','output'): p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--pilot',action='store_true');p.add_argument('--policy',type=Path);p.add_argument('--pilot-receipt',type=Path)
    args = p.parse_args()
    if not args.pilot and (args.policy is None or args.pilot_receipt is None): p.error('full scan needs frozen policy and pilot receipt')
    print(json.dumps(scan(args)), flush=True)
