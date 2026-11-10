import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.smoke_topology_track_b_wiring_imports_v1 import _identity_gate

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_topology_track_b_wiring_imports_v1.py"
LOCKED = (
    "scripts/limited_topology_startup.py",
    "scripts/smoke_topology_track_b_wiring_imports_v1.py",
    "phase3_joint_edit_refine/candidate_feasibility.py",
    "phase3_joint_edit_refine/tissue_planner.py",
    "phase3_joint_edit_refine/topology_track_b.py",
    "phase3_joint_edit_refine/workflow.py",
    "phase3_mask_edit_refine/execution.py",
    "phase3_mask_edit_refine/gates.py",
    "phase3_mask_edit_refine/research_anchor_subdivision_v1.py",
    "phase3_mask_edit_refine/topology_semantics.py",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest():
    return {
        "kind": "topology-track-b-target-import-manifest-v1",
        "repository_root": str(ROOT),
        "locked_files": {item: _sha(ROOT / item) for item in LOCKED},
        "modules": [
            "phase3_joint_edit_refine.topology_track_b",
            "phase3_joint_edit_refine.tissue_planner",
            "phase3_joint_edit_refine.workflow",
        ],
        "require_linux_single_thread_parent": False,
        "import_timeout_seconds": 90,
        "execution_authorized": False,
        "real_patient_calls": 0,
        "witness_calls": 0,
        "four_target_calls": 0,
        "h_and_e_calls": 0,
    }


def test_import_only_smoke_passes_without_business_calls(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    child = json.loads(
        (output / "child_import_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "pass"
    assert child["status"] == "pass"
    assert child["default_rng_and_hash_unchanged"] is True
    assert child["blocked_calls"] == []
    assert child["witness_calls"] == 0


def test_identity_gate_rejects_changed_locked_file(tmp_path):
    manifest_value = _manifest()
    manifest_value["locked_files"][LOCKED[0]] = "0" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked code identity mismatch"):
        _identity_gate(manifest_value, manifest)


def test_identity_gate_rejects_any_execution_scope(tmp_path):
    manifest_value = _manifest()
    manifest_value["witness_calls"] = 1
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden call"):
        _identity_gate(manifest_value, manifest)
