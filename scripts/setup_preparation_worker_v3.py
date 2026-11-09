"""Bounded preparation worker; all evidence verification precedes process exit."""
import hashlib
import json
import os
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    request_raw = sys.stdin.buffer.read(32769)
    if len(request_raw) > 32768:
        raise ValueError("request limit")
    request = json.loads(request_raw)
    if set(request) != {"manifest_path", "manifest_sha256"}:
        raise ValueError("request schema")
    with Path(request["manifest_path"]).open("rb") as stream:
        raw = stream.read(1048577)
    if (
        len(raw) > 1048576
        or hashlib.sha256(raw).hexdigest() != request["manifest_sha256"]
    ):
        raise ValueError("manifest binding")
    manifest = json.loads(raw)
    if (
        manifest.get("kind") != "setup-v3-target-trial"
        or manifest.get("setup_watchdog_seconds") != 60
    ):
        raise ValueError("setup-only policy")
    if manifest.get("permissions") != {
        "setup": True,
        "validation": False,
        "S07": False,
        "generation": False,
        "publication": False,
    }:
        raise ValueError("setup-only scope")
    from scripts.preflight_setup_telemetry_v3 import capture_preflight
    from diagnostic_storage.setup_result_verifier_v3 import (
        verify_setup_result,
        read_control,
    )
    from diagnostic_storage.setup_process_gate_v3 import source_identity

    report = capture_preflight(manifest, manifest["output_root"])
    if report["status"] != "preparation_only_pass":
        return 78
    root = Path(manifest["output_root"])
    request_sha = hashlib.sha256(request_raw).hexdigest()
    binding = {
        "request_sha256": request_sha,
        "summary_sha256": hashlib.sha256(
            read_control(root / "setup_summary.json", 16384)
        ).hexdigest(),
        "journal_sha256": hashlib.sha256(
            read_control(root / "setup_telemetry.jsonl", 1048576)
        ).hexdigest(),
    }
    with (root / "setup_binding.json").open("x") as stream:
        json.dump(binding, stream)
        stream.flush()
        os.fsync(stream.fileno())
    verdict = verify_setup_result(root, request_sha256=request_sha, binding=binding)
    print(
        json.dumps(
            {
                **verdict,
                "manifest_sha256": request["manifest_sha256"],
                "source_identity": source_identity(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(type(exc).__name__, file=sys.stderr)
        raise SystemExit(78)
