#!/usr/bin/env python3
"""One-shot, manifest-bound large B/C synthetic trial; never calls pathology."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import struct
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def distribution_inventory():
    rows = Counter()
    for dist in importlib.metadata.distributions():
        record = Path(dist._path) / 'RECORD'
        rows[(dist.metadata.get('Name'), dist.version, sha(record) if record.is_file() else None)] += 1
    return rows


def identity():
    return {
        'reference_id': 'synthetic-source', 'patient_id': 'synthetic-patient',
        'operations': [
            {'case_id': 'synthetic-U1', 'primitive_id': 'synthetic-primitive1'},
            {'case_id': 'synthetic-U2', 'primitive_id': 'synthetic-primitive2'},
        ],
        **{k: '1' * 64 for k in ('run_manifest_sha256', 'input_identity_sha256',
                                'case_mapping_sha256', 'contract_sha256', 'startup_identity_sha256')},
    }


@dataclass
class Case:
    case_id: str
    primitive_id: str


def receipt(path):
    row = {
        'reference_id': 'synthetic-source', 'patient_id': 'synthetic-patient',
        'source_only': True, 'target_mask_created': False,
        'source_content_verified_against_census': True,
        'checks': [{
            'primitive_id': op['primitive_id'], 'passed_necessary_capacity': True,
            'passed_topology_witness': False,
            'failure_reasons': ['synthetic-negative'],
            'metrics': {'whole_mask_topology_witness': {'errors': ['synthetic-negative']}},
        } for op in identity()['operations']],
        'both_significant_capacity_pass': True, 'both_significant_topology_pass': False,
    }
    path.open('x').write(json.dumps([row], indent=2) + '\n')
    return path


def allocated(root, extra_files=()):
    seen = set()
    total = 0
    for path in (*root.rglob('*'), *extra_files):
        if not path.is_file():
            continue
        stat = path.stat()
        key = stat.st_dev, stat.st_ino
        if key not in seen:
            seen.add(key)
            total += stat.st_blocks * 512
    return {'unique_file_allocated_bytes': total, 'unique_file_count': len(seen)}


def independent_arrays(root, count):
    import numpy as np
    rng = np.random.default_rng(20261103)
    found = 0
    chain = hashlib.sha256()
    with (root / 'codec/arrays.index.jsonl').open('rb') as index:
        for raw in index:
            entry = json.loads(raw)
            descriptor = entry['header']['descriptor']
            if descriptor['shape'] != [512, 512] or descriptor['dtype'] != np.dtype('float64').str:
                continue
            expected = rng.random((512, 512), dtype=np.float64).tobytes()
            with (root / 'codec' / entry['file']).open('rb') as blob:
                if blob.read(8) != bytes.fromhex('4754524143453200'):
                    raise ValueError('array magic')
                size = struct.unpack('>I', blob.read(4))[0]
                if size > 16384 or json.loads(blob.read(size)) != entry['header']:
                    raise ValueError('array header')
                if blob.read(len(expected) + 1) != expected:
                    raise ValueError('array content')
            chain.update(hashlib.sha256(expected).digest())
            found += 1
    if found != count:
        raise ValueError('array occurrence count')
    return {'array_occurrences': found, 'raw_payload_bytes': found * 512 * 512 * 8,
            'every_original_payload_byte_equal': True, 'ordered_reference_hash': chain.hexdigest()}


def preflight(manifest, approval):
    raw = Path(manifest).read_bytes()
    m = json.loads(raw)
    a = json.loads(Path(approval).read_bytes())
    if not (a.get('approved') is True and a.get('kind') == m.get('kind') == 'observer-budget-large-synthetic-v1'
            and a.get('manifest_sha256') == hashlib.sha256(raw).hexdigest()):
        raise ValueError('manifest-bound authorization required')
    return m


def run(manifest, approval):
    started = time.monotonic()
    m = preflight(manifest, approval)
    root = Path(m['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'authorization_consumed.json').open('x').write(json.dumps({'manifest_sha256': sha(manifest), 'consumed': True}) + '\n')
    result = {'status': 'HOLD', 'automatic_retries': 0, 'S07_calls': 0, 'HE_calls': 0,
              'witness_calls': 0, 'outer_acceptance_published': False, 'arms': []}
    arm = None
    try:
        deadline = started + m['setup_seconds']
        def check_setup():
            if time.monotonic() >= deadline:
                raise TimeoutError('setup deadline')
        here = Path(__file__).resolve().parents[1]
        for name, expected in m['code_files_sha256'].items():
            check_setup()
            if sha(here / name) != expected:
                raise ValueError('code identity mismatch: ' + name)
        runtime = m['runtime']
        if sha(sys.executable) != runtime['executable_sha256'] or sys.version != runtime['python'] or platform.platform() != runtime['platform']:
            raise ValueError('runtime identity mismatch')
        frozen = Counter((x['name'], x['version'], x['record_sha256']) for x in runtime['packages'])
        if distribution_inventory() != frozen:
            raise ValueError('distribution identity mismatch')
        if any(os.environ.get(k) != v for k, v in runtime['environment'].items()):
            raise ValueError('environment identity mismatch')
        if shutil.disk_usage(root).free < m['free_bytes_required']:
            raise ValueError('insufficient free bytes')
        pending, final = root / 'hardlink.pending', root / 'hardlink.final'
        pending.open('xb').write(b'original')
        os.link(pending, final)
        try:
            os.link(pending, final)
        except FileExistsError:
            pass
        else:
            raise ValueError('hardlink no-replace failed')
        check_setup()
        result['setup_seconds'] = time.monotonic() - started
        import numpy as np
        from diagnostic_storage.observer_adapter_v2 import ObserverAdapter
        from diagnostic_storage.validator_process_gate_v3_budget import run_validation
        from diagnostic_storage import guard_blocks_v2 as codec
        global_before = hashlib.sha256(repr((random.getstate(), np.random.get_state())).encode()).hexdigest()
        if any(n.startswith(('phase3_', 'inpaint_cells')) for n in sys.modules) or 'torch' in sys.modules:
            raise ValueError('business or torch import')
        for arm, mode in [('B', 'hash_only'), ('C', 'full_capture')]:
            arm_started = time.monotonic()
            core = ObserverAdapter(root / arm, mode=mode, identity=identity())
            if not core.valid:
                raise RuntimeError('adapter initialization')
            rng = np.random.default_rng(20261103)
            snapshots = 0
            interval = (m['events_per_arm'] - 4) // (m['arrays_per_arm'] + 1)
            guard_ordinal = 0
            if interval <= 0:
                raise ValueError('invalid fixed workload')
            for i in range(m['events_per_arm']):
                if i % 4096 == 0 and time.monotonic() - arm_started >= m['seconds_per_arm']:
                    raise TimeoutError('arm generation deadline')
                if i in (0, m['events_per_arm'] // 2):
                    op = identity()['operations'][0 if i == 0 else 1]
                    core.sink.record('qualification.begin', {'case': Case(**op)})
                elif i in (m['events_per_arm'] // 2 - 1, m['events_per_arm'] - 1):
                    core.sink.record('qualification.complete', {
                        'failures': ['synthetic-negative'],
                        'witness': {'errors': ['synthetic-negative']},
                    })
                else:
                    guard_ordinal += 1
                    values = {'row': (i // 512) % 512, 'col': i % 512, 'accepted': i % 3 == 0}
                    if guard_ordinal % interval == 0 and snapshots < m['arrays_per_arm']:
                        values['snapshot'] = rng.random((512, 512), dtype=np.float64)
                        snapshots += 1
                    core.sink.record('guard.stage', values)
                if core.sink.failed or not core.valid:
                    raise RuntimeError('adapter recording failure')
            if snapshots != m['arrays_per_arm']:
                raise ValueError('snapshot count mismatch')
            core.sink.close()
            rp = receipt(root / (arm + '_receipt.json'))
            core.finalize(normal_invocation_end=True, business_receipt_path=rp)
            if not core.valid or core.status()['phase'] != 'SEALED_PENDING_VALIDATION':
                raise RuntimeError('adapter sealing failure')
            encode_and_seal_seconds = time.monotonic() - arm_started
            capture_request = json.dumps({'kind': 'capture', 'identity': identity(),
                                          'capture': str(core.root), 'receipt': str(rp),
                                          'mode': mode}, sort_keys=True).encode()
            gate = run_validation(capture_request, python_executable=sys.executable)
            if gate['validation_ok'] is not True:
                raise RuntimeError('capture validation HOLD')
            storage = allocated(core.root, (rp,))
            if storage['unique_file_allocated_bytes'] > m['max_bytes_per_arm']:
                raise ValueError('arm storage over budget')
            result['arms'].append({'arm': arm, 'mode': mode, 'events': core.events,
                                  'arrays': snapshots, 'encode_and_seal_seconds': encode_and_seal_seconds,
                                  'arm_total_seconds': time.monotonic() - arm_started,
                                  'storage': storage, 'validation': gate})
        comparison_request = json.dumps({'kind': 'compare', 'identity': identity(),
                                         'left': str(root / 'B'), 'right': str(root / 'C'),
                                         'left_receipt': str(root / 'B_receipt.json'),
                                         'right_receipt': str(root / 'C_receipt.json')}, sort_keys=True).encode()
        comparison = run_validation(comparison_request, python_executable=sys.executable)
        if comparison['validation_ok'] is not True:
            raise RuntimeError('B/C comparison HOLD')
        result['comparison'] = comparison
        result['independent_arrays'] = independent_arrays(root / 'C', m['arrays_per_arm'])
        global_after = hashlib.sha256(repr((random.getstate(), np.random.get_state())).encode()).hexdigest()
        if global_after != global_before:
            raise ValueError('global default RNG changed')
        if any(n.startswith(('phase3_', 'inpaint_cells')) for n in sys.modules) or 'torch' in sys.modules:
            raise ValueError('business or torch imported')
        if any((root / arm / 'adapter_complete.json').exists() for arm in ('B', 'C')):
            raise ValueError('outer acceptance marker created')
        result.update(status='synthetic_BC_strict_match', global_rng_unchanged=True,
                      logical_events_each=m['events_per_arm'], arrays_each=m['arrays_per_arm'])
    except Exception as exc:
        result.update(failure_type=type(exc).__name__, failure_text=str(exc)[:512], stopped_arm=arm)
    result['elapsed_seconds'] = time.monotonic() - started
    (root / 'trial_result.json').open('x').write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                      'arms_complete': len(result['arms']), 'failure_type': result.get('failure_type')}))
    return 0 if result['status'] == 'synthetic_BC_strict_match' else 78


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.manifest, args.approval))
