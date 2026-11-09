import hashlib
from pathlib import Path
from types import SimpleNamespace

from scripts.run_budget_cost_target_trial_v2 import record_sha256


def test_record_sha256_uses_raw_bytes(tmp_path):
    record = tmp_path / 'RECORD'
    raw = b'a,b,c\r\nx,y,z\r\n'
    record.write_bytes(raw)
    dist = SimpleNamespace(_path=tmp_path)
    assert record_sha256(dist) == hashlib.sha256(raw).hexdigest()
    assert record_sha256(dist) != hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()


def test_record_sha256_missing_is_unknown(tmp_path):
    assert record_sha256(SimpleNamespace(_path=tmp_path)) is None
