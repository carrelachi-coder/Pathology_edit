import json
from pathlib import Path
import subprocess
import sys


def test_profile_producer_resolves_code_from_snapshot_when_cwd_is_elsewhere(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/profile_adapter_synthetic_cost_v2.py'
    out = tmp_path / 'profile'
    result = subprocess.run(
        [sys.executable, str(script), '--events', '5', '--output', str(out)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((out / 'report.json').read_text())
    assert report['logical_events_written'] == 5
    assert report['capture_complete'] is False
    assert any(k.endswith('/diagnostic_storage/guard_blocks_v2.py') for k in report['code_sha256'])
