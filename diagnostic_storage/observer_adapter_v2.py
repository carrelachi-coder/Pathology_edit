"""Independent opt-in adapter. NOT installed into production Observer/controller.

Use core.sink with the existing public observation_session only in a reviewed
harness. Sink.close quiesces; explicit finalize binds an existing receipt later.
Only the separate validator may publish outer acceptance. No runner/model calls.
"""
from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from pathlib import Path
import queue
import sys
import json

import numpy as np

from . import guard_blocks_v2 as codec
from .guard_budget_candidate_v1 import attach_for_independent_trial
from .observer_adapter_contract_v2 import (
    AdapterError,
    FORMAT,
    TOTAL,
    OUTER,
    bounded_json,
    business_receipt,
    code_identity,
    identity_copy,
    publish_control,
)

_CHECKPOINTS = frozenset(
    {
        "qualification.begin",
        "compiler.attempt",
        "planner.ready",
        "execution.begin",
        "probe.complete",
        "grow.begin",
        "grow.end",
        "compiler.exception",
        "qualification.exception",
    }
)


def _json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


class FailIsolatedSink:
    """Lowest stable boundary; backend and backend fault-handler may both fail."""

    __slots__ = ("core", "failed", "closed")

    def __init__(self, core):
        self.core, self.failed, self.closed = core, False, False

    def _notify(self, stage, exc):
        self.failed = True  # Before any potentially fallible notification.
        try:
            self.core.invalidate(stage, exc)
        except Exception:
            pass

    def record(self, event, values):
        if self.failed:
            return None
        try:
            if self.closed:
                raise AdapterError("event after session close")
            self.core.record(event, values)
        except Exception as exc:
            self._notify("record", exc)
        return None

    def invalidate(self, stage, exc):
        self._notify(stage, exc)
        return None

    def close(self):
        if self.closed:
            return None
        self.closed = True
        try:
            self.core.quiesce()
        except Exception as exc:
            self._notify("quiesce", exc)
        return None


class ObserverAdapter:
    def __init__(self, directory, *, mode, identity, total_limit_bytes=TOTAL):
        self.valid = True
        self.faults = []
        self.phase = "NEW"
        self.events = 0
        self._metadata_used = 0
        self.max_event_metadata_bytes = 8 * 1024**2
        self._queue = queue.Queue(maxsize=1)
        self._finalized = False
        self._owns_directory = False
        self.writer = None
        self.sink = FailIsolatedSink(self)
        self.root = None
        self.mode = mode
        try:
            self.root = Path(directory)
            self.identity = identity_copy(identity)
            if (
                type(total_limit_bytes) is not int
                or not 2 * OUTER <= total_limit_bytes <= TOTAL
            ):
                raise AdapterError("invalid total budget")
            if mode not in ("hash_only", "full_capture"):
                raise AdapterError("invalid mode")
            self.manifest = {
                "format": FORMAT,
                "identity": self.identity,
                "mode": mode,
                "code_sha256": code_identity(),
                "limits": {
                    "total_bytes": total_limit_bytes,
                    "outer_bytes": OUTER,
                    "codec_bytes": total_limit_bytes - OUTER,
                },
            }
            self.root.mkdir(parents=True, exist_ok=False)
            self._owns_directory = True
            publish_control(self.root, "adapter_manifest.json", self.manifest)
            self.writer = codec.Writer(
                self.root / "codec",
                mode=mode,
                expected_operations=[
                    op["case_id"] for op in self.identity["operations"]
                ],
                max_storage_bytes=total_limit_bytes - OUTER,
            )
            attach_for_independent_trial(self.writer)
            self.phase = "PREPARED"
        except Exception as exc:
            self._invalidate_safely("initialize", exc)

    def _invalidate_safely(self, stage, exc):
        self.valid = False
        try:
            self.invalidate(stage, exc)
        except Exception:
            pass

    def invalidate(self, stage, exc):
        self.valid = False
        if not self.faults:
            self.faults.append(
                {
                    "stage": stage,
                    "type": type(exc).__name__,
                    "next_sequence": self.events,
                }
            )

    def record(self, event, values):
        if not self.valid:
            return
        if self.phase not in ("PREPARED", "ACTIVE"):
            raise AdapterError("record outside active lifecycle")
        self.phase = "ACTIVE"
        self._metadata_used = 0
        self._reserve_metadata(event)
        record = {
            "sequence": self.events,
            "event": event,
            "values": self._encode(values),
        }
        if event in _CHECKPOINTS:
            record["rng"] = self._encode(self._rng())
        self._queue.put_nowait(record)
        pending = self._queue.get_nowait()
        self.writer.write_event((_json(pending) + "\n").encode("utf-8"))
        self.events += 1

    def quiesce(self):
        if self.phase not in ("SEALED_PENDING_VALIDATION", "INVALID"):
            self.phase = "QUIESCED"

    def status(self):
        return {
            "format": FORMAT,
            "phase": self.phase,
            "core_fault_free": self.valid,
            "sink_fault_free": not self.sink.failed,
            "sink_closed": self.sink.closed,
            "events": self.events,
            "faults": [dict(f) for f in self.faults],
            "finalized": self._finalized,
            "interpretation_permitted": False,
            "qualification_updates_permitted": False,
        }

    def finalize(self, *, normal_invocation_end, business_receipt_path):
        # Defensive repeated calls are idempotent: never retry a failed finish.
        if self._finalized:
            return self.status()
        self._finalized = True
        binding = None
        normal = False
        try:
            if (
                not self.sink.closed
                or self.phase != "QUIESCED"
                or not self.valid
                or self.sink.failed
                or normal_invocation_end is not True
            ):
                raise AdapterError("session incomplete or recorder invalid")
            _, binding = business_receipt(business_receipt_path, self.identity)
            normal = True
        except Exception as exc:
            self._invalidate_safely("receipt_or_session", exc)
        try:
            if self.writer is not None:
                envelope = {
                    "normal_end": normal,
                    "operation_ids": [
                        op["case_id"] for op in self.identity["operations"]
                    ],
                    "payload": {
                        "receipt": binding,
                        "identity_sha256": codec.digest(bounded_json(self.identity)),
                    },
                }
                if len(bounded_json(envelope)) > 8192:
                    raise AdapterError("compact receipt limit")
                summary = self.writer.finish(normal_end=normal, receipt=envelope)
                if normal and not (
                    summary["block_integrity_complete"]
                    and summary["business_process_complete"]
                ):
                    raise AdapterError("operation boundaries incomplete")
        except Exception as exc:
            self._invalidate_safely("finalize", exc)
        self.phase = (
            "SEALED_PENDING_VALIDATION"
            if self.valid and not self.sink.failed
            else "INVALID"
        )
        try:
            if self._owns_directory:
                status = self.status()
                status["identity_sha256"] = codec.digest(bounded_json(self.identity))
                status["receipt"] = binding
                publish_control(self.root, "adapter_status.json", status)
        except Exception as exc:
            self._invalidate_safely("terminal_status", exc)
            self.phase = "INVALID"
        return self.status()

    def _reserve_metadata(self, value):
        # Conservative payload accounting, not a claim to cap Python/process RSS.
        charge = 512
        if type(value) is str:
            charge += 12 * len(value)  # ensure_ascii can use two Unicode escapes.
        elif type(value) is int:
            charge += value.bit_length() // 3 + 2
        self._metadata_used += charge
        if self._metadata_used > self.max_event_metadata_bytes:
            raise MemoryError("diagnostic event metadata limit")

    def _encode(self, value):
        self._reserve_metadata(value)
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if math.isfinite(value):
                return value
            return {
                "nonfinite": "nan"
                if math.isnan(value)
                else ("+inf" if value > 0 else "-inf")
            }
        if isinstance(value, np.generic):
            return self._encode(value.item())
        if type(value) is np.ndarray:
            return self.writer.add_array(value)
        if type(value) is np.random.Generator:
            return {
                "local_numpy_generator_state": self._encode(value.bit_generator.state)
            }
        if type(value) in (list, tuple, set, frozenset):
            return {
                "sequence_type": type(value).__name__,
                "items": [self._encode(v) for v in value],
            }
        if type(value) is dict:
            # Encode insertion order and key types instead of sorting business data.
            return {
                "mapping": [
                    [self._encode(k), self._encode(v)] for k, v in value.items()
                ]
            }
        if isinstance(value, BaseException):
            return {
                "exception_type": type(value).__module__
                + "."
                + type(value).__qualname__,
                "args": self._encode(value.args),
                "attributes": self._encode(vars(value)),
            }
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "dataclass": type(value).__module__ + "." + type(value).__qualname__,
                "fields": [
                    [f.name, self._encode(object.__getattribute__(value, f.name))]
                    for f in fields(value)
                ],
            }
        # Never exhaust arbitrary iterables or invoke custom serialization code.
        raise TypeError("unsupported passive evidence type")

    def _rng(self):
        random_module = sys.modules.get("random")
        python_state = (
            random_module.getstate()
            if random_module is not None
            else {"unavailable": "random not already loaded"}
        )
        states = {
            "python": python_state,
            "numpy_legacy": np.random.get_state(),
            "torch_cpu": {"unavailable": "torch not already loaded"},
            "torch_cuda": {
                "unavailable": "not queried; observer never initializes CUDA"
            },
            "unpassed_local_generators": {
                "unavailable": "no generator discovery or initialization"
            },
        }
        torch = sys.modules.get("torch")
        if torch is not None:
            generator = vars(torch).get("default_generator")
            if generator is not None:
                states["torch_cpu"] = generator.get_state().numpy()
        return states
