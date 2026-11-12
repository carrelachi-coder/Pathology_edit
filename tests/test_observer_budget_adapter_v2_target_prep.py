import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import pytest

from scripts import check_observer_budget_adapter_v2_target_prep as prep


def test_local_synthetic_preparation_keeps_experiment_closed(tmp_path):
    root = Path(__file__).resolve().parents[1]
    files = ['diagnostic_storage/__init__.py', 'diagnostic_storage/guard_blocks_v2.py',
             'diagnostic_storage/guard_budget_candidate_v1.py',
             'diagnostic_storage/observer_adapter_contract_v2.py',
             'diagnostic_storage/observer_adapter_v2.py',
             'diagnostic_storage/observer_adapter_validation_v2_budget.py',
             'scripts/check_observer_budget_adapter_v2_target_prep.py']
    rows = []
    for (name, version, record), count in prep.inventory().items():
        rows.extend([{'name': name, 'version': version, 'record_sha256': record}] * count)
    manifest = {'kind': 'observer-budget-adapter-v2-target-preparation',
                'output_root': str(tmp_path / 'result'), 'setup_seconds': 60,
                'free_bytes_required': 0,
                'code_files_sha256': {f: prep.sha(root / f) for f in files},
                'runtime': {'executable_sha256': prep.sha(sys.executable),
                            'python': sys.version, 'platform': platform.platform(),
                            'packages': rows, 'environment': {}}}
    mp = tmp_path / 'manifest.json'
    mp.write_text(json.dumps(manifest))
    ap = tmp_path / 'approval.json'
    ap.write_text(json.dumps({'kind': manifest['kind'], 'approved': True,
                              'manifest_sha256': hashlib.sha256(mp.read_bytes()).hexdigest()}))
    completed = subprocess.run(
        [sys.executable, str(root / 'scripts/check_observer_budget_adapter_v2_target_prep.py'),
         '--manifest', str(mp), '--approval', str(ap)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / 'result/preparation_result.json').read_text())
    assert report['status'] == 'target_preparation_smoke_passed'
    assert report['event_count'] == report['S07_calls'] == report['HE_calls'] == 0
    assert report['outer_acceptance_published'] is False
    assert not (tmp_path / 'result/synthetic_init/adapter_complete.json').exists()


def test_missing_approval_refuses_before_output_creation(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'kind': 'observer-budget-adapter-v2-target-preparation',
                                    'output_root': str(tmp_path / 'result')}))
    approval = tmp_path / 'approval.json'
    approval.write_text(json.dumps({'kind': 'observer-budget-adapter-v2-target-preparation',
                                    'approved': False, 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match='manifest-bound'):
        prep.run(manifest, approval)
    assert not (tmp_path / 'result').exists()
