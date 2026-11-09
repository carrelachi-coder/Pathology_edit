import hashlib
import json
import pytest
from diagnostic_storage.setup_result_verifier_v3 import verify_setup_result, STAGES
from scripts.preflight_setup_telemetry_v3 import SetupTelemetry


def sealed(tmp_path):
    t = SetupTelemetry(tmp_path / "setup_telemetry.jsonl")
    for stage, count in zip(STAGES, [5, 11, 1, 220, 3, 809, 3]):
        with t.stage(stage):
            for _ in range(count):
                t.completed()
    s = t.finish(True)
    s["status"] = "preparation_only_pass"
    (tmp_path / "setup_summary.json").write_text(json.dumps(s))
    return bind(tmp_path)


def bind(root):
    row = {
        "request_sha256": "1" * 64,
        "summary_sha256": hashlib.sha256(
            (root / "setup_summary.json").read_bytes()
        ).hexdigest(),
        "journal_sha256": hashlib.sha256(
            (root / "setup_telemetry.jsonl").read_bytes()
        ).hexdigest(),
    }
    (root / "setup_binding.json").write_text(json.dumps(row))
    return row


def test_full_evidence_and_request_binding(tmp_path):
    b = sealed(tmp_path)
    assert (
        verify_setup_result(tmp_path, request_sha256="1" * 64, binding=b)["status"]
        == "setup_evidence_verified"
    )
    with pytest.raises(ValueError):
        verify_setup_result(tmp_path, request_sha256="2" * 64, binding=b)


@pytest.mark.parametrize("change", ["truncate", "reorder", "fault", "count", "summary"])
def test_rebound_corrupt_evidence_rejected(tmp_path, change):
    sealed(tmp_path)
    p = tmp_path / "setup_telemetry.jsonl"
    rows = p.read_text().splitlines()
    s = json.loads((tmp_path / "setup_summary.json").read_text())
    if change == "truncate":
        rows.pop()
    if change == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    if change == "fault":
        s["fault_types"] = ["OSError"]
    if change == "count":
        s["stages"][5]["items_completed"] = 808
    if change == "summary":
        s["stages"][3]["bytes_processed"] = 99
    p.write_text("\n".join(rows) + "\n")
    (tmp_path / "setup_summary.json").write_text(json.dumps(s))
    b = bind(tmp_path)
    with pytest.raises((ValueError, IndexError)):
        verify_setup_result(tmp_path, request_sha256="1" * 64, binding=b)
