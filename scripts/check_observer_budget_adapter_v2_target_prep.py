"""One-shot isolated target preparation smoke; no witness or model calls."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import threading
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def inventory():
    rows = Counter()
    for dist in importlib.metadata.distributions():
        record = Path(dist._path) / 'RECORD'
        rows[(dist.metadata.get('Name'), dist.version, sha(record) if record.is_file() else None)] += 1
    return rows


def native_threads():
    tasks = Path('/proc/self/task')
    return len(list(tasks.iterdir())) if tasks.is_dir() else threading.active_count()


def run(manifest, approval):
    started = time.monotonic()
    raw = Path(manifest).read_bytes()
    m = json.loads(raw)
    a = json.loads(Path(approval).read_bytes())
    if not (a.get('approved') is True and a.get('kind') == m.get('kind') == 'observer-budget-adapter-v2-target-preparation'
            and a.get('manifest_sha256') == hashlib.sha256(raw).hexdigest()):
        raise ValueError('manifest-bound target-preparation authorization required')
    root = Path(m['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'authorization_consumed.json').write_text(json.dumps({'manifest_sha256': sha(manifest), 'consumed': True}) + '\n')
    result = {'status': 'HOLD', 'S07_calls': 0, 'HE_calls': 0, 'witness_calls': 0,
              'automatic_retries': 0, 'outer_acceptance_published': False}
    try:
        deadline = started + m['setup_seconds']
        def check():
            if time.monotonic() >= deadline:
                raise TimeoutError('setup deadline')
        here = Path(__file__).resolve().parents[1]
        for name, expected in m['code_files_sha256'].items():
            check()
            if sha(here / name) != expected:
                raise ValueError('code identity mismatch: ' + name)
        runtime = m['runtime']
        if sha(sys.executable) != runtime['executable_sha256'] or sys.version != runtime['python'] or platform.platform() != runtime['platform']:
            raise ValueError('runtime identity mismatch')
        frozen = Counter((x['name'], x['version'], x['record_sha256']) for x in runtime['packages'])
        if inventory() != frozen:
            raise ValueError('distribution inventory mismatch')
        if any(os.environ.get(k) != v for k, v in runtime['environment'].items()):
            raise ValueError('environment mismatch')
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
            raise ValueError('hardlink no-replace failure')
        if final.read_bytes() != b'original':
            raise ValueError('hardlink content mismatch')
        check()
        result['setup_seconds'] = time.monotonic() - started
        import numpy as np
        py_rng = hashlib.sha256(repr(random.getstate()).encode()).hexdigest()
        np_rng = hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()
        hash_fingerprint = hash('observer-budget-adapter-v2-target-preparation')
        threads_before = native_threads()
        if 'torch' in sys.modules:
            raise ValueError('torch imported before smoke')
        from diagnostic_storage import guard_blocks_v2 as codec
        from diagnostic_storage.guard_budget_candidate_v1 import CandidateBudget
        from diagnostic_storage import observer_adapter_contract_v2 as contract
        from diagnostic_storage.observer_adapter_v2 import ObserverAdapter
        from diagnostic_storage import observer_adapter_validation_v2_budget as validator
        smoke = root / 'synthetic_init'
        identity = {'reference_id': 'synthetic-source', 'patient_id': 'synthetic-patient',
                    'operations': [{'case_id': 'U1', 'primitive_id': 'p1'}, {'case_id': 'U2', 'primitive_id': 'p2'}],
                    **{k: '1' * 64 for k in ('run_manifest_sha256', 'input_identity_sha256',
                                           'case_mapping_sha256', 'contract_sha256', 'startup_identity_sha256')}}
        core = ObserverAdapter(smoke, mode='hash_only', identity=identity)
        if not core.valid or type(core.writer.budget) is not CandidateBudget:
            raise ValueError('new adapter initialization or budget selection failed')
        bound = contract.code_identity()
        if bound['guard_blocks_v2.py'] != contract.CODEC_PIN or bound['guard_budget_candidate_v1.py'] != contract.BUDGET_PIN:
            raise ValueError('budget or codec pin mismatch')
        if core.manifest['code_sha256'] != bound or core.manifest['format'] != contract.FORMAT:
            raise ValueError('capture code identity mismatch')
        if validator.FORMAT != contract.FORMAT or codec.Budget is CandidateBudget:
            raise ValueError('new validator or legacy codec mutation')
        core.sink.close()
        if (smoke / 'adapter_complete.json').exists() or (smoke / 'codec/observer_complete.json').exists():
            raise ValueError('unexpected completion marker')
        if hashlib.sha256(repr(random.getstate()).encode()).hexdigest() != py_rng or hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest() != np_rng:
            raise ValueError('default RNG changed')
        if hash('observer-budget-adapter-v2-target-preparation') != hash_fingerprint:
            raise ValueError('hash fingerprint changed')
        threads_after = native_threads()
        if threads_after != threads_before or 'torch' in sys.modules or any(k.startswith('phase3_joint_edit_refine') for k in sys.modules):
            raise ValueError('unexpected threads, torch or business import')
        result.update(status='target_preparation_smoke_passed', threads_before=threads_before,
                      threads_after=threads_after, code_identity=bound, format=contract.FORMAT,
                      rng_unchanged=True, torch_imported=False, event_count=0)
    except Exception as exc:
        result['failure_type'] = type(exc).__name__
        result['failure_text'] = str(exc)[:512]
    result['elapsed_seconds'] = time.monotonic() - started
    (root / 'preparation_result.json').open('x').write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['status'] == 'target_preparation_smoke_passed' else 78


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.manifest, args.approval))
