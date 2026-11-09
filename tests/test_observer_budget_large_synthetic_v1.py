import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from scripts import run_observer_budget_large_synthetic_v1 as trial


def manifest_for(tmp_path):
    root = Path(__file__).resolve().parents[1]
    files = ['diagnostic_storage/__init__.py', 'diagnostic_storage/guard_blocks_v2.py',
             'diagnostic_storage/guard_budget_candidate_v1.py',
             'diagnostic_storage/observer_adapter_contract_v2.py',
             'diagnostic_storage/observer_adapter_v2.py',
             'diagnostic_storage/observer_adapter_validation_v2_budget.py',
             'diagnostic_storage/validation_budget_policy_v3_budget.py',
             'diagnostic_storage/observer_adapter_validation_v3_budget.py',
             'diagnostic_storage/validator_process_gate_v3_budget.py',
             'scripts/validator_readonly_worker_v3_budget.py',
             'scripts/run_observer_budget_large_synthetic_v1.py']
    rows = []
    for (name, version, record), count in trial.distribution_inventory().items():
        rows.extend([{'name': name, 'version': version, 'record_sha256': record}] * count)
    manifest = {'kind': 'observer-budget-large-synthetic-v1',
                'output_root': str(tmp_path / 'result'), 'setup_seconds': 60,
                'free_bytes_required': 0, 'events_per_arm': 1000,
                'arrays_per_arm': 4, 'seconds_per_arm': 30,
                'max_bytes_per_arm': 4 * 1024**3,
                'code_files_sha256': {f: trial.sha(root / f) for f in files},
                'runtime': {'executable_sha256': trial.sha(sys.executable),
                            'python': sys.version, 'platform': platform.platform(),
                            'packages': rows, 'environment': {}}}
    mp = tmp_path / 'manifest.json'; mp.write_text(json.dumps(manifest))
    ap = tmp_path / 'approval.json'; ap.write_text(json.dumps({
        'kind': manifest['kind'], 'approved': True,
        'manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest()}))
    return root, mp, ap


def test_small_BC_synthetic_strict_match_no_outer_acceptance(tmp_path):
    root, mp, ap = manifest_for(tmp_path)
    completed = subprocess.run([sys.executable, str(root / 'scripts/run_observer_budget_large_synthetic_v1.py'),
                                '--manifest', str(mp), '--approval', str(ap)],
                               cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((tmp_path / 'result/trial_result.json').read_text())
    assert result['status'] == 'synthetic_BC_strict_match'
    assert result['logical_events_each'] == 1000 and result['arrays_each'] == 4
    assert result['independent_arrays']['every_original_payload_byte_equal']
    assert result['comparison']['validation_ok']
    assert result['S07_calls'] == result['HE_calls'] == result['witness_calls'] == 0
    assert not result['outer_acceptance_published']
    assert not (tmp_path / 'result/B/adapter_complete.json').exists()
    assert not (tmp_path / 'result/C/adapter_complete.json').exists()


def test_unbound_approval_stops_before_output_creation(tmp_path):
    root, mp, ap = manifest_for(tmp_path)
    approval = json.loads(ap.read_text())
    approval['manifest_sha256'] = '0' * 64
    ap.write_text(json.dumps(approval))
    completed = subprocess.run([sys.executable, str(root / 'scripts/run_observer_budget_large_synthetic_v1.py'),
                                '--manifest', str(mp), '--approval', str(ap)],
                               cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode != 0
    assert not (tmp_path / 'result').exists()
