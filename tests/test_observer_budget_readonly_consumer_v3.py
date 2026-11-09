"""Read-only 30/60 consumer of the pinned Adapter v2 budget capture."""
import hashlib
import json
from unittest import mock

import pytest

from diagnostic_storage import validation_budget_policy_v3_budget as policy
from diagnostic_storage import observer_adapter_validation_v3_budget as validator
from diagnostic_storage import validator_process_gate_v3_budget as gate
from tests.test_observer_codec_adapter_v2_budget import complete, identity
from tests.test_observer_codec_adapter_v1 import complete as old_complete
from tests.test_validator_process_gate_v1 import command


def request_capture(core, receipt):
    return json.dumps({'kind': 'capture', 'identity': identity(), 'capture': str(core.root),
                       'receipt': str(receipt), 'mode': 'hash_only'}).encode()


def request_compare(b, br, c, cr):
    return json.dumps({'kind': 'compare', 'identity': identity(), 'left': str(b.root),
                       'right': str(c.root), 'left_receipt': str(br), 'right_receipt': str(cr)}).encode()


def test_policy_pins_new_producer_and_separate_budgets():
    assert policy.budget_for('capture') == 30
    assert policy.budget_for('compare') == 60
    assert policy.verify_producer_files() == dict(policy.PRODUCER_PINS)
    assert 'guard_budget_candidate_v1.py' in policy.PRODUCER_PINS
    with pytest.raises(TypeError):
        policy.BUDGETS['compare'] = 61
    for kind in ('unknown', None, 60):
        with pytest.raises(ValueError):
            policy.budget_for(kind)


@pytest.mark.parametrize('kind,seconds', [('capture', 31), ('compare', 61),
                                          ('compare', float('inf')), ('capture', True)])
def test_deadline_cannot_exceed_trusted_policy(kind, seconds):
    with pytest.raises(ValueError):
        validator._deadline(kind, seconds)
    with pytest.raises(ValueError):
        gate._run_process(command('pass'), b'{}', seconds, kind=kind)


@pytest.mark.parametrize('kind,expected', [('capture', 30), ('compare', 60)])
def test_real_worker_validates_new_capture_without_publication(tmp_path, kind, expected):
    b, br = complete(tmp_path, 'B')
    if kind == 'capture':
        raw = request_capture(b, br)
    else:
        c, cr = complete(tmp_path, 'C', 'full_capture')
        raw = request_compare(b, br, c, cr)
    result = gate.run_validation(raw)
    assert result['validation_ok'], result
    assert result['transport']['hard_timeout_seconds'] == expected
    child = result['worker_receipt']
    assert child['validation_budget_seconds'] == child['verdict']['validation_budget_seconds'] == expected
    assert child['verdict']['producer_component_sha256'] == dict(policy.PRODUCER_PINS)
    assert child['verdict']['consumer_component_sha256'] == policy.consumer_identity()
    assert child['verdict']['compatibility_profile'] == policy.COMPATIBILITY_ID
    assert not result['execution_permitted'] and not result['capture_marker_published']
    assert not (b.root / 'adapter_complete.json').exists()


def test_old_capture_cannot_be_accepted_as_new(tmp_path):
    b, br = old_complete(tmp_path, 'old')
    result = gate.run_validation(request_capture(b, br))
    assert result['status'] == 'HOLD' and not result['validation_ok']


def test_new_producer_drift_rejected_before_worker_spawn(tmp_path, monkeypatch):
    b, br = complete(tmp_path)
    monkeypatch.setattr(policy, 'file_sha', lambda _: '0' * 64)
    monkeypatch.setattr(gate.subprocess, 'Popen', mock.Mock(side_effect=AssertionError('must not spawn')))
    with pytest.raises(ValueError, match='producer component drift'):
        gate.run_validation(request_capture(b, br))


def test_forged_worker_producer_identity_holds(tmp_path, monkeypatch):
    b, br = complete(tmp_path)
    raw = request_capture(b, br)
    good = gate.run_validation(raw)
    assert good['validation_ok']
    payload = good['worker_receipt']
    payload['verdict']['producer_component_sha256'] = {}
    monkeypatch.setattr(gate, '_run_process', lambda *a, **kw: {
        'transport_ok': True, 'reason': None, 'stdout': json.dumps(payload).encode(),
        'exit_code': 0, 'child_reaped': True,
    })
    assert gate.run_validation(raw)['status'] == 'HOLD'


def test_full_capture_array_corruption_is_rejected(tmp_path):
    b, br = complete(tmp_path, 'B')
    c, cr = complete(tmp_path, 'C', 'full_capture')
    raw = request_compare(b, br, c, cr)
    assert gate.run_validation(raw)['validation_ok']
    array = next((c.root / 'codec').glob('array_*.bin'))
    with array.open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'X')
    assert gate.run_validation(raw)['status'] == 'HOLD'


def test_short_process_timeout_reaps_worker():
    result = gate._run_process(command('import time;time.sleep(2)'), b'{}', 0.05, kind='compare')
    assert result['timed_out'] and result['child_reaped'] and not result['transport_ok']
