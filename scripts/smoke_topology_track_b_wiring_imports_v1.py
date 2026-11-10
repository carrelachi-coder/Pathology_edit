#!/usr/bin/env python3
"""Import-only Track B wiring smoke; never executes planning or qualification."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

FORBIDDEN_CALLS = frozenset(
    {
        "build_joint_nuclei_preflight",
        "compile_tissue_portfolio",
        "compile_tissue_portfolio_with_track_b",
        "compile_tissue_witness",
        "create_joint_tissue_plan",
        "main",
        "run",
        "_compile_one",
        "_qualify_tissue_case",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_startup(root: Path):
    path = root / "scripts" / "limited_topology_startup.py"
    spec = importlib.util.spec_from_file_location("_track_b_startup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load controlled startup helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_gate(manifest: dict, manifest_path: Path) -> Path:
    if manifest.get("kind") != "topology-track-b-target-import-manifest-v1":
        raise RuntimeError("unsupported target import manifest")
    if manifest.get("execution_authorized") is not False:
        raise RuntimeError("import smoke requires execution_authorized=false")
    if any(
        int(manifest.get(key, -1)) != 0
        for key in (
            "real_patient_calls",
            "witness_calls",
            "four_target_calls",
            "h_and_e_calls",
        )
    ):
        raise RuntimeError("import smoke manifest permits a forbidden call")
    for name, expected in sorted(manifest.get("required_environment", {}).items()):
        if os.environ.get(name) != str(expected):
            raise RuntimeError(f"required environment mismatch: {name}")
    declared_manifest_sha = manifest.get("manifest_sha256")
    observed_manifest_sha = _sha256(manifest_path)
    # A self-hash is intentionally omitted from the canonical manifest bytes.
    # If supplied by an outer launch record, it must bind the exact file.
    if declared_manifest_sha is not None and declared_manifest_sha != observed_manifest_sha:
        raise RuntimeError("manifest file SHA mismatch")
    root = Path(manifest["repository_root"]).resolve(strict=True)
    for relative, expected in sorted(manifest["locked_files"].items()):
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"locked code identity mismatch: {relative}")
    return root


def _child_import(
    *,
    root: Path,
    modules: tuple[str, ...],
    common_state,
    report_path: Path,
    require_linux_threads: bool,
) -> int:
    startup = _load_startup(root)
    startup.restore_controlled(common_state)
    startup.parent_gate(require_linux_threads=require_linux_threads)
    before = startup.capture()
    blocked = []
    entries = []

    def profile(frame, event, _argument):
        if event != "call":
            return
        filename = frame.f_code.co_filename
        function = frame.f_code.co_name
        module = frame.f_globals.get("__name__", "")
        if function in FORBIDDEN_CALLS and (
            module.startswith(("phase3_", "inpaint_cells"))
            or filename.startswith(str(root) + os.sep)
        ):
            blocked.append(
                {"function": function, "module": module, "file": filename}
            )
            raise RuntimeError("forbidden business call during import-only smoke")
        if function == "<module>" and filename.startswith(str(root) + os.sep):
            entries.append(
                {
                    "index": len(entries),
                    "module": module,
                    "file": str(Path(filename).relative_to(root)),
                    "entry_state_sha256": startup.digest(startup.capture()),
                }
            )

    error = None
    sys.path.insert(0, str(root))
    sys.setprofile(profile)
    try:
        for module in modules:
            importlib.import_module(module)
    except BaseException as exc:  # noqa: BLE001 - report fail-closed import evidence
        error = {"type": type(exc).__name__, "text": str(exc)}
    finally:
        sys.setprofile(None)
    after = startup.capture()
    torch = sys.modules.get("torch")
    report = {
        "status": "pass" if error is None and not blocked else "HOLD",
        "modules": list(modules),
        "startup_digest": startup.digest(before),
        "post_import_digest": startup.digest(after),
        "default_rng_and_hash_unchanged": startup.digest(before)
        == startup.digest(after),
        "native_threads_after_import": startup.native_threads(),
        "cuda_initialized_after_import": bool(
            torch is not None and torch.cuda.is_initialized()
        ),
        "ordered_source_module_entries": entries,
        "blocked_calls": blocked,
        "error": error,
        "real_patient_calls": 0,
        "witness_calls": 0,
        "four_target_calls": 0,
        "h_and_e_calls": 0,
    }
    _write_new(report_path, report)
    return 0 if report["status"] == "pass" else 78


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = _identity_gate(manifest, manifest_path)
    if args.output.exists():
        raise RuntimeError("output path already exists")
    args.output.mkdir(parents=True, exist_ok=False)

    import torch  # Preloaded once under the reviewed controlled-startup policy.

    startup = _load_startup(root)
    require_linux_threads = bool(manifest["require_linux_single_thread_parent"])
    startup.parent_gate(require_linux_threads=require_linux_threads)
    common_state = startup.capture()
    child_report = args.output / "child_import_report.json"
    child = os.fork()
    if child == 0:
        try:
            exit_code = _child_import(
                root=root,
                modules=tuple(manifest["modules"]),
                common_state=common_state,
                report_path=child_report,
                require_linux_threads=require_linux_threads,
            )
            os._exit(exit_code)
        except BaseException:
            os._exit(79)
    deadline = time.monotonic() + float(manifest["import_timeout_seconds"])
    status = None
    while time.monotonic() < deadline:
        waited, status = os.waitpid(child, os.WNOHANG)
        if waited == child:
            break
        time.sleep(0.02)
    timed_out = status is None
    if timed_out:
        os.kill(child, signal.SIGKILL)
        _, status = os.waitpid(child, 0)
    exit_code = os.waitstatus_to_exitcode(status)
    child_payload = (
        json.loads(child_report.read_text(encoding="utf-8"))
        if child_report.exists()
        else None
    )
    report = {
        "kind": "topology-track-b-target-import-report-v1",
        "status": (
            "pass"
            if exit_code == 0
            and not timed_out
            and child_payload is not None
            and child_payload.get("status") == "pass"
            else "HOLD"
        ),
        "manifest_file_sha256": _sha256(manifest_path),
        "timed_out": timed_out,
        "child_exit_code": exit_code,
        "child_report_sha256": (
            _sha256(child_report) if child_report.exists() else None
        ),
        "execution_authorized": False,
        "real_patient_calls": 0,
        "witness_calls": 0,
        "four_target_calls": 0,
        "h_and_e_calls": 0,
    }
    _write_new(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 78


if __name__ == "__main__":
    raise SystemExit(main())
