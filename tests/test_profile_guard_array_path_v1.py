import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from scripts import profile_guard_array_path_v1 as trial


def _manifest(tmp_path):
    root = Path(__file__).resolve().parents[1]
    files = [
        "diagnostic_storage/__init__.py",
        "diagnostic_storage/guard_blocks_v2.py",
        "scripts/run_observer_budget_large_synthetic_v1.py",
        "scripts/profile_guard_array_path_v1.py",
    ]
    packages = [
        {"name": name, "version": version, "record_sha256": record}
        for (name, version, record), count in trial.distribution_inventory().items()
        for _ in range(count)
    ]
    m = {
        "kind": "observer-array-microprofile-v1",
        "output_root": str(tmp_path / "result"),
        "setup_seconds": 60,
        "seconds_per_arm": 30,
        "events_per_arm": 1000,
        "arrays_per_arm": 4,
        "free_bytes_required": 0,
        "code_files_sha256": {f: trial.sha(root / f) for f in files},
        "runtime": {
            "executable_sha256": trial.sha(sys.executable),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
            "environment": {},
        },
    }
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(m))
    ap = tmp_path / "approval.json"
    ap.write_text(json.dumps({
        "kind": m["kind"], "approved": True,
        "manifest_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
    }))
    return root, mp, ap


def test_microprofile_strict_same_stream_and_no_business(tmp_path):
    root, mp, ap = _manifest(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/profile_guard_array_path_v1.py"),
         "--manifest", str(mp), "--approval", str(ap)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "result/trial_result.json").read_text())
    assert report["status"] == "synthetic_codec_microprofile_complete"
    assert report["strict_match"] and report["global_rng_unchanged"]
    assert [x["events"] for x in report["arms"]] == [1000, 1000]
    assert [x["arrays"] for x in report["arms"]] == [4, 4]
    assert report["S07_calls"] == report["witness_calls"] == report["HE_calls"] == 0
    assert not report["outer_acceptance_published"]
    assert any("fsync" in x["function"] for x in report["arms"][1]["profile_rows"])


def test_unbound_approval_stops_before_output(tmp_path):
    root, mp, ap = _manifest(tmp_path)
    a = json.loads(ap.read_text())
    a["manifest_sha256"] = "0" * 64
    ap.write_text(json.dumps(a))
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/profile_guard_array_path_v1.py"),
         "--manifest", str(mp), "--approval", str(ap)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode != 0
    assert not (tmp_path / "result").exists()


def test_changed_code_identity_holds_without_arms(tmp_path):
    root, mp, ap = _manifest(tmp_path)
    m = json.loads(mp.read_text())
    m["code_files_sha256"]["diagnostic_storage/guard_blocks_v2.py"] = "0" * 64
    mp.write_text(json.dumps(m))
    ap.write_text(json.dumps({
        "kind": m["kind"], "approved": True,
        "manifest_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
    }))
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/profile_guard_array_path_v1.py"),
         "--manifest", str(mp), "--approval", str(ap)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 78
    report = json.loads((tmp_path / "result/trial_result.json").read_text())
    assert report["status"] == "HOLD" and report["arms"] == []
