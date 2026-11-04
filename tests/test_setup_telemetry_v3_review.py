from pathlib import Path
import shutil
import pytest
from scripts.review_setup_telemetry_v3 import audit


def snapshot(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in [
        "scripts/preflight_setup_telemetry_v3.py",
        "scripts/run_readonly_budget_v2_trial.py",
        "diagnostic_storage/validation_budget_policy_v2.py",
    ]:
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(root / name, target)
    return tmp_path


def test_review_structure_checks_pass_without_importing_or_running_setup(tmp_path):
    result = audit(snapshot(tmp_path))
    assert result["status"] == "static_structure_checks_pass"
    assert result["server_calls"] == 0
    assert result["preflight_identity_AST_after_declared_telemetry_removal_equal"]


@pytest.mark.parametrize(
    "before,after",
    [
        ("!= 809", "!= 808"),
        ("stream.read(1048576)", "stream.read(1024)"),
        (
            'pins["validator_process_gate_v1.py"] = policy.LEGACY_GATE_PIN',
            'pins["validator_process_gate_v1.py"] = "0" * 64',
        ),
        ("            check()\n", "            pass\n"),
        ("return hash_file(path, telemetry)", 'return "ignored"'),
    ],
)
def test_review_rejects_identity_hash_legacy_deadline_or_delegate_changes(
    tmp_path, before, after
):
    root = snapshot(tmp_path)
    target = root / "scripts/preflight_setup_telemetry_v3.py"
    text = target.read_text()
    assert before in text
    target.write_text(text.replace(before, after, 1))
    with pytest.raises(ValueError):
        audit(root)
