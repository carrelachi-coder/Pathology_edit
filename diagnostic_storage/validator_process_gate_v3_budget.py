"""Candidate v2 task-specific process gate; not production/S07 authorization.

Parent imports stdlib only. Worker cannot publish capture acceptance. All helper
failures/late exits hold; no retries, timeout extension, model/source invocation,
or automatic connection to the existing Controller.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

MAX_REQUEST = 32768
MAX_STDOUT = 32768
MAX_STDERR = 8192
from .validation_budget_policy_v3_budget import (
    COMPATIBILITY_ID,
    budget_for,
    consumer_identity,
    policy_identity,
    verify_producer_files,
)


def request_value(raw):
    if type(raw) is not bytes or len(raw) > MAX_REQUEST:
        raise ValueError("bounded request bytes required")

    def invalid(_):
        raise ValueError("nonfinite request")

    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError("duplicate request key")
            d[k] = v
        return d

    value = json.loads(raw, parse_constant=invalid, object_pairs_hook=unique)
    if type(value) is not dict or type(value.get("identity")) is not dict:
        raise ValueError("identity required")
    if value.get("kind") == "capture":
        keys = {"kind", "identity", "capture", "receipt", "mode"}
        if value.get("mode") not in ("hash_only", "full_capture"):
            raise ValueError("mode")
        paths = ("capture", "receipt")
    elif value.get("kind") == "compare":
        keys = {"kind", "identity", "left", "right", "left_receipt", "right_receipt"}
        paths = ("left", "right", "left_receipt", "right_receipt")
    else:
        raise ValueError("read-only validation kind required")
    if set(value) != keys or any(
        type(value[k]) is not str or not 0 < len(value[k]) <= 4096 for k in paths
    ):
        raise ValueError("request fields")
    return value


def _kill_reap(process):
    # Kill the entire worker group, including descendants holding pipe FDs.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        return False
    return True


def _run_process(argv, raw, seconds, *, kind):
    """Private transport for fixed worker; injectable synthetic commands in tests."""
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 < seconds <= budget_for(kind)
    ):
        raise ValueError("hard timeout exceeds trusted task policy")
    started = time.monotonic()
    deadline = started + seconds
    process = None
    selector = selectors.DefaultSelector()
    streams = []
    output = bytearray()
    errors = bytearray()
    sent = 0
    reason = None
    reaped = False
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        for stream, event in (
            (process.stdin, selectors.EVENT_WRITE),
            (process.stdout, selectors.EVENT_READ),
            (process.stderr, selectors.EVENT_READ),
        ):
            streams.append(stream)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, event)
        while selector.get_map():
            # Check time BEFORE accepting completion; exit0 after deadline holds.
            now = time.monotonic()
            if now >= deadline:
                reason = "timeout"
                break
            for key, _ in selector.select(min(0.01, deadline - now)):
                stream = key.fileobj
                if stream is process.stdin:
                    try:
                        n = os.write(stream.fileno(), raw[sent : sent + 4096])
                    except BrokenPipeError:
                        reason = "request_pipe_closed"
                        break
                    sent += n
                    if sent == len(raw):
                        selector.unregister(stream)
                        stream.close()
                else:
                    block = os.read(stream.fileno(), 4096)
                    if not block:
                        selector.unregister(stream)
                        stream.close()
                        continue
                    buffer = output if stream is process.stdout else errors
                    limit = MAX_STDOUT if stream is process.stdout else MAX_STDERR
                    if len(buffer) + len(block) > limit:
                        reason = "worker_output_limit"
                        break
                    buffer.extend(block)
            if reason:
                break
        if reason is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "timeout"
            else:
                try:
                    process.wait(timeout=remaining)
                    reaped = True
                except subprocess.TimeoutExpired:
                    reason = "timeout"
        # Elapsed includes spawn, input transfer, output and wait/exit overhead.
        if reason is None and time.monotonic() >= deadline:
            reason = "timeout"
        if reason is None and process.returncode != 0:
            reason = "worker_nonzero_or_signal"
        if reason is not None:
            reaped = _kill_reap(process)
        return {
            "transport_ok": reason is None,
            "reason": reason,
            "exit_code": process.returncode,
            "timed_out": reason == "timeout",
            "child_reaped": reaped,
            "stdout": bytes(output),
            "stderr_sha256": hashlib.sha256(errors).hexdigest(),
            "stderr_bytes": len(errors),
            "elapsed_seconds": time.monotonic() - started,
            "hard_timeout_seconds": seconds,
        }
    except BaseException:
        if process is not None:
            _kill_reap(process)
        raise
    finally:
        selector.close()
        for stream in streams:
            if not stream.closed:
                stream.close()


def run_validation(raw, *, python_executable=sys.executable, timeout_seconds=None):
    """Return validation_ok only after timely exit0 AND a bound valid verdict.

    No accepted marker is written, even on success. New source/arm execution and
    final publication remain outside this preparation-only API.
    """
    request = request_value(raw)
    verify_producer_files()
    seconds = (
        budget_for(request["kind"]) if timeout_seconds is None else timeout_seconds
    )
    worker = (
        Path(__file__).resolve().parents[1] / "scripts/validator_readonly_worker_v3_budget.py"
    )
    started = time.monotonic()
    try:
        transport = _run_process(
            [str(python_executable), str(worker)], raw, seconds, kind=request["kind"]
        )
    except Exception as exc:
        transport = {
            "transport_ok": False,
            "reason": "process_gate_error",
            "error_type": type(exc).__name__,
            "stdout": b"",
            "child_reaped": None,
            "timed_out": False,
            "elapsed_seconds": time.monotonic() - started,
            "hard_timeout_seconds": seconds,
        }
    wire = transport.pop("stdout")
    reason = transport["reason"]
    payload = None
    if transport["transport_ok"]:
        try:
            payload = json.loads(wire)
            expected = hashlib.sha256(raw).hexdigest()
            if (
                payload.get("request_sha256") != expected
                or payload.get("worker_readonly") is not True
                or payload.get("kind") != request["kind"]
                or payload.get("business_modules_loaded") is not False
                or payload.get("torch_loaded") is not False
                or payload.get("validation_budget_seconds")
                != budget_for(request["kind"])
                or any(payload.get(k) != v for k, v in policy_identity().items())
            ):
                raise ValueError("worker receipt binding")
            verdict = payload["verdict"]
            if (
                verdict.get("validation_budget_seconds") != budget_for(request["kind"])
                or any(verdict.get(k) != v for k, v in policy_identity().items())
                or verdict.get("consumer_component_sha256") != consumer_identity()
                or verdict.get("producer_component_sha256") != verify_producer_files()
                or verdict.get("compatibility_profile") != COMPATIBILITY_ID
            ):
                raise ValueError("validator did not use bound v3 policy/producer/code")
            ok = (
                (
                    verdict.get("candidate_validated") is True
                    and verdict.get("interpretation_permitted") is False
                )
                if request["kind"] == "capture"
                else verdict.get("status") == "semantic_match"
            )
            if not ok or verdict.get("qualification_updates_permitted") is not False:
                raise ValueError("incomplete worker verdict")
        except Exception:
            reason = "worker_receipt_invalid"
        if time.monotonic() - started >= seconds:
            reason = "timeout"
    return {
        "status": "validated_readonly" if reason is None else "HOLD",
        "validation_ok": reason is None,
        "reason": reason,
        "transport": transport,
        "worker_receipt": payload if reason is None else None,
        "interpretation_permitted": False,
        "capture_marker_published": False,
        "execution_permitted": False,
        "qualification_updates_permitted": False,
        "validation_budget_seconds": budget_for(request["kind"]),
        **policy_identity(),
        "attempts": 1,
        "automatic_retries": 0,
    }


def validate_in_order(jobs, **kwargs):
    """Validation steps only. Never invokes a source/arm; stop on first HOLD."""
    rows = []
    for label, request in jobs:
        try:
            row = run_validation(request, **kwargs)
        except Exception as exc:
            row = {
                "validation_ok": False,
                "status": "HOLD",
                "reason": "request_or_gate_error",
                "error_type": type(exc).__name__,
                "execution_permitted": False,
            }
        rows.append({"step": label, "result": row})
        if not row["validation_ok"]:
            return {
                "status": "HOLD",
                "steps": rows,
                "remaining_steps_started": False,
                "execution_permitted": False,
            }
    return {
        "status": "readonly_validation_sequence_complete",
        "steps": rows,
        "execution_permitted": False,
    }
