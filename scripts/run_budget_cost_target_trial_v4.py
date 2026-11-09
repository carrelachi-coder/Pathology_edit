"""Manifest-bound isolated synthetic profile trial using raw RECORD identity.

This version preserves the v3 execution contract and uses a synthetic
producer that locates code identity files from its installed snapshot. A new
manifest and authorization are required; prior HOLD records remain immutable.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_budget_cost_target_trial_v1 import sha, check_approval, check_report, compare_reports


def record_sha256(dist):
    """Hash the exact bytes captured in frozen distribution manifests."""
    record = Path(dist._path) / 'RECORD'
    return sha(record) if record.is_file() else None


def distribution_inventory():
    return Counter((dist.metadata.get('Name'), dist.version, record_sha256(dist))
                   for dist in importlib.metadata.distributions())


def frozen_distribution_inventory(rows):
    return Counter((row['name'], row['version'], row['record_sha256']) for row in rows)


def run(manifest, approval):
    start = time.monotonic()
    m = check_approval(manifest, approval)
    root = Path(m['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    (root / 'authorization_consumed.json').write_text(json.dumps({'manifest_sha256': sha(manifest), 'consumed': True}))
    result = {'status': 'HOLD', 'automatic_retries': 0, 'S07_calls': 0, 'HE_calls': 0,
              'outer_acceptance_published': False, 'profiles': []}
    previous_alarm = signal.getsignal(signal.SIGALRM)

    def setup_timeout(signum, frame):
        raise TimeoutError('setup hard deadline')

    signal.signal(signal.SIGALRM, setup_timeout)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, 60 - (time.monotonic() - start)))
    try:
        deadline = start + 60

        def check():
            if time.monotonic() >= deadline:
                raise TimeoutError('setup')

        here = Path(__file__).resolve().parents[1]
        for p, h in m['code_files_sha256'].items():
            check()
            assert sha(here / p) == h
        r = m['runtime']
        check()
        assert sha(sys.executable) == r['executable_sha256'] and sys.version == r['python'] and platform.platform() == r['platform']
        check()
        assert distribution_inventory() == frozen_distribution_inventory(r['packages'])
        check()
        for k, v in r['environment'].items():
            check()
            assert os.environ.get(k) == v
        assert shutil.disk_usage(root).free >= 12 * 1024**3
        pending = root / 'hardlink.pending'
        final = root / 'hardlink.final'
        pending.open('xb').write(b'original')
        os.link(pending, final)
        try:
            os.link(pending, final)
        except FileExistsError:
            pass
        else:
            raise ValueError('no-replace failure')
        assert final.read_bytes() == b'original'
        check()
        result['setup_seconds'] = time.monotonic() - start
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        from diagnostic_storage.validator_process_gate_v2 import _run_process
        reports = []
        for i, variant in enumerate(m['variants']):
            if variant['label'] != ['original', 'candidate'][i]:
                raise ValueError('variant order')
            expected = ['/usr/bin/timeout', '--signal=INT', '--kill-after=5s', '60s', r['executable'],
                        str(here / 'scripts/profile_adapter_synthetic_cost_v2.py'), '--events', '100000',
                        '--output', str(root / variant['label'])] + (['--candidate-budget'] if i else [])
            if variant['argv'] != expected:
                raise ValueError('argv scope mismatch')
            if time.monotonic() - start >= 180:
                raise TimeoutError('whole trial')
            transport = _run_process(expected, b'{}', 60, kind='compare')
            public = {k: v for k, v in transport.items() if k != 'stdout'}
            result['profiles'].append({'label': variant['label'], 'transport': public})
            if not transport['transport_ok']:
                raise RuntimeError('producer HOLD')
            report = root / variant['label'] / 'report.json'
            if report.stat().st_size > 1024**2:
                raise ValueError('report limit')
            reports.append(json.loads(report.read_bytes()))
            (root / (variant['label'] + '_transport.json')).open('x').write(json.dumps(public, indent=2))
            check_report(reports[-1], bool(i))
        compare_reports(reports)
        if time.monotonic() - start >= 180:
            raise TimeoutError('whole trial')
        result.update(status='synthetic_profiles_complete_stream_hash_equal',
                      seconds=[r['elapsed_seconds'] for r in reports],
                      interpretation='Profiled observations only. No real-source timing, complete capture or eligibility conclusion.')
    except Exception as exc:
        result['failure_type'] = type(exc).__name__
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous_alarm)
    result['elapsed_seconds'] = time.monotonic() - start
    (root / 'trial_result.json').open('x').write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['status'] != 'HOLD' else 78


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--approval', type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(run(a.manifest, a.approval))
