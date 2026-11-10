"""Standalone Writer experiment; not wired into the production Observer.

The caller supplies the already constructed event object. This writer creates
the same JSONL bytes as ObserverAdapter._json and validates the same sequence,
array references, operation boundaries and storage budget without parsing its
own freshly serialized bytes a second time. The existing Reader remains the
authority for final stream validation.
"""
from __future__ import annotations

import hashlib
import json

from .guard_blocks_v2 import CodecError, MAX_HEADER, Writer, refs


class PreparsedWriterCandidate(Writer):
    def write_record(self, event):
        try:
            self._guard()
            if (
                not isinstance(event, dict)
                or type(event.get("sequence")) is not int
                or event["sequence"] != self.events
                or not isinstance(event.get("event"), str)
            ):
                raise CodecError("event sequence/schema mismatch")
            if self.events >= 2**64 - 1:
                raise CodecError("sequence overflow")
            # This is the exact byte encoding used by ObserverAdapter._json.
            line = (
                json.dumps(
                    event,
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            if len(line) > self.max_record:
                raise CodecError("invalid/oversized event bytes")
            if list(refs(event)) != self.pending_refs:
                raise CodecError("array occurrence/event mismatch")
            self.ops.event(event)
            if self.buffer and len(self.buffer) + len(line) > self.target:
                self._flush()
            maximum = len(self.buffer) + len(line)
            bound = maximum + maximum // 64 + MAX_HEADER + 8192
            self.budget.claim_many(
                [
                    (self.root / f"block_{self.blocks:012d}.bin.pending", bound),
                    (self.block_index, self.bi.tell() + 4096),
                ]
            )
            self.buffer.extend(line)
            self.buffer_events += 1
            self.events += 1
            self.pending_refs = []
            self.raw_hash.update(line)
            self.raw_chain = hashlib.sha256(self.raw_chain.encode() + line).hexdigest()
        except Exception as exc:
            self._fault("event", exc)
            raise
