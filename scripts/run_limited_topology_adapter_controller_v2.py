#!/usr/bin/env python3
"""Fail-closed sequential fork controller; default-denied by manifest + approval.

Versioned Adapter v2 controller candidate; no inherited S07 authorization. Controlled postfork restoration requires a separately
approved RNG decision. An accepted code review never satisfies launch approval.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time


def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            d.update(block)
    return d.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def bounded_fork(function, *, timeout):
    """One attempt, own process group, bounded reap; never retries."""
    started = time.monotonic()
    pid = os.fork()
    if pid == 0:
        try:
            os.setsid()
            function()
            os._exit(0)
        except BaseException:
            os._exit(78)
    try:
        while True:
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                return {
                    "pid": pid,
                    "exit_code": os.waitstatus_to_exitcode(status),
                    "timed_out": False,
                    "elapsed_seconds": time.monotonic() - started,
                }
            if time.monotonic() - started >= timeout:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # If setsid had not yet completed, kill the child itself as well.
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                os.waitpid(pid, 0)
                return {
                    "pid": pid,
                    "exit_code": None,
                    "timed_out": True,
                    "elapsed_seconds": time.monotonic() - started,
                }
            time.sleep(0.01)
    except BaseException:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        raise


def sequence(execute, assess, *, emit, timeout=1200):
    """Policy independent of fixtures: reserve calls BEFORE each sole launch."""
    calls = 0
    launched = []
    for arm in ("A", "B", "C"):
        if calls + 2 > 6:
            raise RuntimeError("witness budget exhausted")
        calls += 2
        launched.append(arm)
        emit({"event": "launch_reserved", "arm": arm, "reserved_witness_calls": calls})
        try:
            receipt = execute(arm, timeout)
            emit({"event": "arm_process_receipt", "arm": arm, "receipt": receipt})
            if receipt["timed_out"] or receipt["exit_code"] != 0:
                raise RuntimeError("arm timeout/nonzero exit")
            if assess(arm) is not True:
                raise RuntimeError("mismatch or invalid evidence")
        except Exception as exc:
            emit(
                {
                    "event": "mismatch_hold",
                    "arm": arm,
                    "reason": type(exc).__name__ + ": " + str(exc),
                }
            )
            return {
                "status": "mismatch_hold",
                "launched": launched,
                "reserved_witness_calls": calls,
            }
        emit({"event": "arm_accepted", "arm": arm})
    return {
        "status": "diagnostic_comparison_complete",
        "launched": launched,
        "reserved_witness_calls": calls,
        "qualification_updates_permitted": False,
    }


def approval_gate(plan, manifest_sha, approval):
    if plan.get("preparation_only") is not False:
        raise PermissionError("preparation-only manifest cannot execute")
    if (
        plan.get("controller_kind") != "limited-topology-adapter-v2-budget"
        or approval.get("kind") != "limited-topology-adapter-v2-budget"
    ):
        raise PermissionError("new Adapter controller authorization required")
    required = ("execution_permitted", "rng_decision_approved", "preflight_complete")
    if not all(plan.get(k) is True for k in required):
        raise PermissionError("manifest execution/RNG/preflight gate closed")
    if (
        approval.get("approved") is not True
        or approval.get("manifest_sha256") != manifest_sha
    ):
        raise PermissionError("separate manifest-bound execution authorization absent")
    if approval.get("rng_decision_sha256") != plan["rng_decision"]["sha256"]:
        raise PermissionError("RNG decision not covered by authorization")
    if plan["limits"] != {
        "source_processes": 3,
        "witness_calls": 6,
        "seconds_per_source_pair": 1200,
        "retries": 0,
    }:
        raise ValueError("frozen finite budget changed")
    if plan["rng_policy"] != "common_parent_explicit_postfork_restore_v1":
        raise ValueError("unreviewed startup policy")


def scope_gate(plan):
    expected = [
        "--source-freeze",
        plan["source_subset"]["path"],
        "--cases",
        plan["case_templates"]["path"],
        "--start-index",
        "7",
        "--limit",
        "1",
        "--topology-witness",
    ]
    for arm in ("A", "B", "C"):
        spec = plan["arms"][arm]
        if (
            spec["probe"] != "scripts/probe_manuscript_joint_capacity.py"
            or spec["probe_args"] != expected
        ):
            raise ValueError("source index/limit/probe scope changed")
        mode = {"A": None, "B": "hash_only", "C": "full_capture"}[arm]
        if spec["mode"] != mode:
            raise ValueError("observer arm mode changed")
    sources = json.loads(Path(plan["source_subset"]["path"]).read_text())[
        "candidate_sources"
    ]
    if sources[7]["reference_id"] != plan["reference_id"]:
        raise ValueError("frozen S07 selection changed")
    source = sources[7]
    ident = plan["adapter_identity"]
    templates = json.loads(Path(plan["case_templates"]["path"]).read_text())
    operations = [
        {
            "case_id": source["reference_id"]
            + "__"
            + templates[i]["primitive_id"]
            + "__capacity",
            "primitive_id": templates[i]["primitive_id"],
        }
        for i in (1, 3)
    ]
    if (
        ident["reference_id"] != source["reference_id"]
        or ident["patient_id"] != source["patient_id"]
        or ident["operations"] != operations
    ):
        raise ValueError("adapter identity does not match frozen probe cases")


def distributions():
    rows = []
    for d in metadata.distributions():
        record = Path(d._path) / "RECORD"
        rows.append(
            {
                "name": d.metadata["Name"],
                "version": d.version,
                "record_sha256": sha(record) if record.is_file() else None,
            }
        )
    return sorted(rows, key=lambda x: (x["name"] or "").lower())


def identity_gate(plan, *, require_fresh_output=True):
    if sha(plan["rng_decision"]["path"]) != plan["rng_decision"]["sha256"]:
        raise ValueError("RNG decision changed")
    for item in plan["pinned_files"]:
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("pinned file changed")
    for snapshot in plan["snapshots"]:
        for relative, expected in snapshot["files_sha256"].items():
            if sha(Path(snapshot["path"]) / relative) != expected:
                raise ValueError("snapshot drift")
    runtime = plan["runtime"]
    if (
        sys.executable != runtime["executable"]
        or sha(sys.executable) != runtime["executable_sha256"]
    ):
        raise ValueError("interpreter identity drift")
    if distributions() != runtime["packages"]:
        raise ValueError("distribution identity drift")
    if any(os.environ.get(k) != v for k, v in runtime["environment"].items()):
        raise ValueError("effective environment drift")
    reservation = plan["reservation"]
    stat = Path(reservation["path"]).stat()
    if (
        stat.st_size != reservation["bytes"]
        or stat.st_blocks * 512 < reservation["bytes"]
    ):
        raise ValueError("finite physical reservation missing/sparse")
    out = Path(plan["output_root"])
    if require_fresh_output and out.exists():
        raise FileExistsError("refuse existing execution output")
    if Path(reservation["path"]).stat().st_dev != out.parent.stat().st_dev:
        raise ValueError("reservation and outputs not on same device")
    if (
        os.statvfs(out.parent).f_bavail * os.statvfs(out.parent).f_frsize
        < reservation["bytes"]
    ):
        raise ValueError("insufficient live free output headroom")


def business_receipt(path, reference):
    rows = json.loads(Path(path).read_text())
    if (
        len(rows) != 1
        or rows[0]["reference_id"] != reference
        or len(rows[0]["checks"]) != 2
    ):
        raise ValueError("not exactly one source/two operations")
    return rows[0]


def assess_real(plan, arm):
    output = Path(plan["output_root"])
    reference = plan["reference_id"]
    current = business_receipt(
        output / arm / "business/capacity_screen.json", reference
    )
    if arm == "A":
        rows = json.loads(Path(plan["historical_receipt"]).read_text())
        matched = [r["result"] for r in rows if r["reference_id"] == reference]
        if len(matched) != 1 or current != matched[0]:
            return False
    else:
        a = business_receipt(output / "A/business/capacity_screen.json", reference)
        if current != a:
            return False
        request = {"identity": plan["adapter_identity"]}
        if arm == "B":
            request.update(
                kind="capture",
                capture=str(output / "B/observer"),
                receipt=str(output / "B/business/capacity_screen.json"),
                mode="hash_only",
            )
        else:
            request.update(
                kind="compare",
                left=str(output / "B/observer"),
                right=str(output / "C/observer"),
                left_receipt=str(output / "B/business/capacity_screen.json"),
                right_receipt=str(output / "C/business/capacity_screen.json"),
            )
        # Keep the common parent import state identical for B and C. The
        # process gate and its policy are imported only in this disposable
        # helper, which itself starts a bounded read-only validation worker.
        def validate_in_child():
            sys.path.insert(0, plan["adapter_snapshot"])
            from diagnostic_storage.validator_process_gate_v3_budget import (
                run_validation,
            )

            result = run_validation(
                json.dumps(request, sort_keys=True).encode(),
                python_executable=plan["runtime"]["executable"],
            )
            write_new(output / arm / "validation_gate_receipt.json", result)
            if result["validation_ok"] is not True:
                raise RuntimeError("read-only validation HOLD")

        # The gate's own OS deadline remains 30/60 seconds. This extra five
        # seconds only bounds helper termination and receipt handoff.
        helper = bounded_fork(
            validate_in_child, timeout=35 if arm == "B" else 65
        )
        if helper["timed_out"] or helper["exit_code"] != 0:
            return False
    return True


def run(plan, manifest_sha, approval):
    approval_gate(
        plan, manifest_sha, approval
    )  # Before output creation/import/restoration.
    identity_gate(plan)
    scope_gate(plan)
    # Explicit approved preload order, recorded as a new experimental condition.
    import numpy

    if plan["preload_torch_cpu"]:
        import torch
    # Avoid caching a generic `scripts` package from the controller snapshot in
    # the parent; business imports must resolve against each pinned arm snapshot.
    spec = importlib.util.spec_from_file_location(
        "_s07_startup_support", Path(__file__).with_name("limited_topology_startup.py")
    )
    startup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(startup)
    capture, digest, parent_gate, restore_controlled = (
        startup.capture,
        startup.digest,
        startup.parent_gate,
        startup.restore_controlled,
    )
    parent_gate()
    common = capture()
    out = Path(plan["output_root"])
    out.mkdir(exist_ok=False)
    # Prove non-replacement hardlinks at the ACTUAL output location, before A.
    pending = out / "filesystem_probe.pending"
    pending.write_bytes(b"probe\n")
    os.link(pending, out / "filesystem_probe.final")
    try:
        os.link(pending, out / "filesystem_probe.final")
    except FileExistsError:
        pass
    else:
        raise RuntimeError("filesystem allowed overwrite")
    write_new(out / "private_startup_state.json", common)
    write_new(
        out / "startup_identity.json",
        {
            "digest": digest(common),
            "state_sha256": sha(out / "private_startup_state.json"),
            "controlled_condition": True,
            "historical_state_recovered": False,
        },
    )
    with (out / "controller_events.jsonl").open("x", buffering=1) as ledger:

        def emit(event):
            ledger.write(json.dumps(event, sort_keys=True) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())

        def execute(arm, timeout):
            identity_gate(plan, require_fresh_output=False)
            parent_gate()
            identity_before = Path(plan["reservation"]["path"]).stat()
            if identity_before.st_blocks * 512 < plan["reservation"]["bytes"]:
                raise ValueError("reservation lost")
            if (
                os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
                < plan["reservation"]["bytes"]
            ):
                raise ValueError("free headroom lost")
            spec = plan["arms"][arm]

            def child():
                restore_controlled(common)
                parent_gate()
                directory = out / arm
                directory.mkdir(exist_ok=False)
                write_new(
                    directory / "startup_identity.json",
                    {
                        "digest": digest(capture()),
                        "hash_fingerprint": capture()["hash_fingerprint"],
                    },
                )
                # Route output before importing or running business modules.
                fd = os.open(
                    directory / "stdout_stderr.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                os.dup2(fd, 1)
                os.dup2(fd, 2)
                os.close(fd)
                os.chdir(spec["snapshot"])
                sys.path.insert(0, spec["snapshot"])
                script = str(Path(spec["snapshot"]) / spec["probe"])
                sys.argv = [
                    script,
                    *spec["probe_args"],
                    "--output",
                    str(directory / "business"),
                ]

                def invoke():
                    try:
                        runpy.run_path(script, run_name="__main__")
                    except SystemExit as exc:
                        if exc.code not in (None, 0):
                            raise

                if arm == "A":
                    invoke()
                else:
                    sys.path.insert(0, plan["adapter_snapshot"])
                    from diagnostic_storage.observer_adapter_v2 import ObserverAdapter
                    from diagnostic_storage.observer_invocation_bridge_v1 import (
                        invoke_with_adapter,
                    )

                    adapter = ObserverAdapter(
                        directory / "observer",
                        mode=spec["mode"],
                        identity=plan["adapter_identity"],
                    )
                    invoke_with_adapter(
                        invoke,
                        adapter,
                        business_receipt_path=directory
                        / "business/capacity_screen.json",
                    )
                    state = adapter.status()
                    if state["phase"] != "SEALED_PENDING_VALIDATION":
                        raise RuntimeError("adapter invalid")

            return bounded_fork(child, timeout=timeout)

        result = sequence(execute, lambda arm: assess_real(plan, arm), emit=emit)
    write_new(out / "controller_result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--authorization", type=Path, required=True)
    a = p.parse_args()
    try:
        result = run(
            json.loads(a.manifest.read_text()),
            sha(a.manifest),
            json.loads(a.authorization.read_text()),
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "preflight_or_authorization_hold",
                    "reason": type(exc).__name__ + ": " + str(exc),
                }
            )
        )
        return 78
    print(json.dumps(result))
    return 0 if result["status"] == "diagnostic_comparison_complete" else 78


if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
