"""Independent byte-lossless block codec v2, NOT connected to Observer.

Logical events are UTF-8 JSONL bytes, preserved exactly. Physical files/indices,
array occurrences and business completion have separate identities. Exceptions
invalidate the writer and propagate to this standalone caller; a future Observer
adapter must isolate them from business control flow (not implemented here).
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import zlib

VERSION = 2
MAGIC = b"GTRACE2\0"
TARGET = 4 * 1024**2
MAX_RECORD = 8 * 1024**2
MAX_HEADER = 16 * 1024
MAX_ARRAY = 64 * 1024**2
CONTROL_RESERVE = 64 * 1024
ZERO = "0" * 64


class CodecError(ValueError):
    pass


class BudgetError(CodecError):
    pass


def dump(value):
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def unique_pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise CodecError("duplicate JSON object key")
        result[k] = v
    return result


def parse(raw):
    def invalid(value):
        raise CodecError("unencoded nonfinite value: " + value)

    return json.loads(
        raw.decode("utf-8"), object_pairs_hook=unique_pairs, parse_constant=invalid
    )


def integer(value):
    return type(value) is int and value >= 0


def refs(value):
    if isinstance(value, dict):
        if list(value) == ["array"]:
            ref = value["array"]
            if not isinstance(ref, dict) or set(ref) != {
                "dtype",
                "shape",
                "content_sha256",
            }:
                raise CodecError("bad array token")
            yield ref
        else:
            for nested in value.values():
                yield from refs(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from refs(nested)


class Operations:
    def __init__(self, expected):
        self.expected = list(expected)
        if (
            not self.expected
            or len(self.expected) > 8
            or any(type(x) is not str or len(x) > 512 for x in self.expected)
            or len(set(self.expected)) != len(self.expected)
        ):
            raise CodecError("invalid expected operations")
        self.started = []
        self.completed = []
        self.active = None

    def event(self, event):
        name = event["event"]
        if name not in ("qualification.begin", "qualification.complete"):
            return
        operation = event.get("operation_id")
        if operation is None and name == "qualification.begin":
            mapping = dict(event.get("values", {}).get("mapping", []))
            case = mapping.get("case", {})
            operation = dict(case.get("fields", [])).get("case_id")
        if name == "qualification.begin":
            if (
                self.active is not None
                or len(self.started) >= len(self.expected)
                or operation != self.expected[len(self.started)]
            ):
                raise CodecError("operation start/order mismatch")
            self.active = operation
            self.started.append(operation)
        else:
            if self.active is None or operation not in (None, self.active):
                raise CodecError("operation completion mismatch")
            self.completed.append(self.active)
            self.active = None

    def complete(self):
        return self.active is None and self.started == self.completed == self.expected


class Budget:
    """Conservative allocated-file budget; hardlink aliases count once.

    Reserve block-rounded bytes plus one page per file before writes. Check real
    allocated blocks as well. Filesystem metadata, process RSS and other writers
    are not isolated by this accounting (not a quota).
    """

    def __init__(self, limit):
        self.limit = limit
        self.charges = {}
        self.sizes = {}
        self.total = 0

    def reservation(self, path, size):
        if Path(path).name.endswith(".index.pending"):
            # Append-only indices can retain filesystem preallocation well above
            # their logical length. Reserve growth headroom BEFORE each append,
            # and never forget previously observed allocation when they grow.
            size = 2 * size + 65536
            return max(
                ((size + 4095) // 4096 + 1) * 4096, self.charges.get(str(path), 0)
            )
        return ((size + 4095) // 4096 + 1) * 4096

    def check(self, path, size, control=False):
        reserve = self.reservation(path, size)
        allowed = self.limit if control else self.limit - CONTROL_RESERVE
        if self.total - self.charges.get(str(path), 0) + reserve > allowed:
            raise BudgetError("unified storage budget")

    def claim(self, path, size, control=False):
        self.check(path, size, control)
        key = str(path)
        charge = self.reservation(path, size)
        self.total += charge - self.charges.get(key, 0)
        self.charges[key] = charge
        self.sizes[key] = size

    def actual(self, path):
        st = Path(path).stat()
        key = str(path)
        if st.st_blocks * 512 > self.charges[key]:
            self.total += st.st_blocks * 512 - self.charges[key]
            self.charges[key] = st.st_blocks * 512
            if self.total > self.limit:
                raise BudgetError("filesystem allocation exceeded budget")

    def seal(self, path):
        # Retain cumulative allocation, not one Python object per artifact.
        self.charges.pop(str(path), None)
        self.sizes.pop(str(path), None)

    def claim_many(self, requests):
        total = self.total
        for path, size in requests:
            total += self.reservation(path, size) - self.charges.get(str(path), 0)
        if total > self.limit - CONTROL_RESERVE:
            raise BudgetError("combined storage reservation")
        for path, size in requests:
            self.claim(path, size)


class Writer:
    def __init__(
        self,
        directory,
        *,
        mode="hash_only",
        expected_operations=("U1", "U2"),
        max_storage_bytes=4 * 1024**3,
        target_block_bytes=TARGET,
        max_record_bytes=MAX_RECORD,
        max_array_bytes=MAX_ARRAY,
        max_array_occurrences=100000,
    ):
        if (
            mode not in ("hash_only", "full_capture")
            or not 0 < target_block_bytes <= max_record_bytes <= MAX_RECORD
        ):
            raise CodecError("bad codec configuration")
        if not 4 * CONTROL_RESERVE <= max_storage_bytes <= 4 * 1024**3:
            raise CodecError("budget too small for controls")
        if (
            type(max_array_bytes) is not int
            or not 0 < max_array_bytes <= MAX_ARRAY
            or type(max_array_occurrences) is not int
            or not 0 < max_array_occurrences <= 100000
        ):
            raise CodecError("invalid array resource limits")
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.mode = mode
        self.ops = Operations(expected_operations)
        self.budget = Budget(max_storage_bytes)
        self.target = target_block_bytes
        self.max_record = max_record_bytes
        self.max_array = max_array_bytes
        self.max_arrays = max_array_occurrences
        self.buffer = bytearray()
        self.buffer_events = 0
        self.events = 0
        self.blocks = 0
        self.arrays = 0
        self.pending_refs = []
        self.raw_hash = hashlib.sha256()
        self.raw_chain = ZERO
        self.block_chain = ZERO
        self.array_chain = ZERO
        self.faults = []
        self.closed = False
        self.block_index = self.root / "blocks.index.pending"
        self.array_index = self.root / "arrays.index.pending"
        self.budget.claim(self.block_index, 0)
        self.bi = self.block_index.open("xb")
        try:
            self.budget.claim(self.array_index, 0)
            self.ai = self.array_index.open("xb")
        except Exception:
            self.bi.close()
            raise

    def _fault(self, stage, exc):
        if not self.faults:
            self.faults.append({"stage": stage, "type": type(exc).__name__})

    def _guard(self):
        if self.closed or self.faults:
            raise CodecError("writer closed/invalid")

    def _append(self, stream, path, entry):
        raw = dump(entry) + b"\n"
        size = stream.tell() + len(raw)
        self.budget.claim(path, size)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
        self.budget.actual(path)

    def _publish(self, name, header, payload):
        encoded_header = dump(header)
        if len(encoded_header) > MAX_HEADER:
            raise CodecError("header too large")
        prefix = MAGIC + struct.pack(">I", len(encoded_header)) + encoded_header
        pending = self.root / (name + ".pending")
        final = self.root / name
        self.budget.claim(pending, len(prefix) + len(payload))
        with pending.open("xb") as stream:
            stream.write(prefix)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.budget.actual(pending)
        os.link(pending, final)
        self.budget.seal(pending)
        whole = hashlib.sha256(prefix)
        whole.update(payload)
        return {
            "file": name,
            "header_sha256": digest(encoded_header),
            "file_sha256": whole.hexdigest(),
        }

    def _flush(self):
        if not self.buffer:
            return
        raw = bytes(self.buffer)
        encoded = zlib.compress(raw, 1)
        header = {
            "format_version": VERSION,
            "kind": "events",
            "codec": "zlib",
            "level": 1,
            "index": self.blocks,
            "first_sequence": self.events - self.buffer_events,
            "last_sequence": self.events - 1,
            "logical_events": self.buffer_events,
            "decoded_bytes": len(raw),
            "encoded_bytes": len(encoded),
            "decoded_sha256": digest(raw),
            "encoded_sha256": digest(encoded),
            "previous_header_sha256": self.block_chain,
        }
        entry = self._publish(f"block_{self.blocks:012d}.bin", header, encoded)
        entry.update(
            {
                "index": self.blocks,
                "first_sequence": header["first_sequence"],
                "last_sequence": header["last_sequence"],
                "logical_events": self.buffer_events,
                "previous_header_sha256": self.block_chain,
            }
        )
        self._append(self.bi, self.block_index, entry)
        self.block_chain = entry["header_sha256"]
        self.blocks += 1
        self.buffer = bytearray()
        self.buffer_events = 0

    def _copy_array(self, array):
        import numpy as np

        return np.array(array, copy=True, order="C", subok=False)

    def add_array(self, array):
        try:
            self._guard()
            if array.dtype.hasobject or array.dtype.fields or array.dtype.subdtype:
                raise CodecError("unsupported object/structured array dtype")
            if (
                array.nbytes > self.max_array
                or self.arrays >= self.max_arrays
                or len(self.pending_refs) >= 4096
            ):
                raise BudgetError("array/metadata buffer ceiling")
            # Check blob AND inventory space before detaching the snapshot.
            upcoming = self.root / f"array_{self.arrays:012d}.bin.pending"
            reservations = [(self.array_index, self.ai.tell() + 4096)]
            if self.mode == "full_capture":
                reservations.append((upcoming, array.nbytes + MAX_HEADER + 12 + 4096))
            self.budget.claim_many(reservations)
            snapshot = self._copy_array(array)
            data = (
                memoryview(snapshot).cast("B") if snapshot.nbytes else memoryview(b"")
            )
            content = hashlib.sha256()
            for offset in range(0, len(data), 1048576):
                content.update(data[offset : offset + 1048576])
            descriptor = {
                "dtype": snapshot.dtype.str,
                "shape": list(snapshot.shape),
                "content_sha256": content.hexdigest(),
            }
            header = {
                "format_version": VERSION,
                "kind": "array",
                "codec": "raw",
                "occurrence": self.arrays,
                "event_sequence": self.events,
                "ordinal": len(self.pending_refs),
                "descriptor": descriptor,
                "decoded_bytes": len(data),
                "previous_header_sha256": self.array_chain,
            }
            if self.mode == "full_capture":
                entry = self._publish(f"array_{self.arrays:012d}.bin", header, data)
            else:
                entry = {"file": None, "header_sha256": digest(dump(header))}
            entry.update({"header": header, "previous_header_sha256": self.array_chain})
            self._append(self.ai, self.array_index, entry)
            self.array_chain = entry["header_sha256"]
            self.arrays += 1
            self.pending_refs.append(descriptor)
            return {"array": descriptor}
        except Exception as exc:
            self._fault("array", exc)
            raise

    def write_event(self, line):
        try:
            self._guard()
            if (
                type(line) is not bytes
                or len(line) > self.max_record
                or not line.endswith(b"\n")
            ):
                raise CodecError("invalid/oversized event bytes")
            event = parse(line)
            if (
                not isinstance(event, dict)
                or type(event.get("sequence")) is not int
                or event["sequence"] != self.events
                or not isinstance(event.get("event"), str)
            ):
                raise CodecError("event sequence/schema mismatch")
            if self.events >= 2**64 - 1:
                raise CodecError("sequence overflow")
            if list(refs(event)) != self.pending_refs:
                raise CodecError("array occurrence/event mismatch")
            self.ops.event(event)
            if self.buffer and len(self.buffer) + len(line) > self.target:
                self._flush()
            # Reserve conservative next encoded-block room; flush can release the
            # difference between decoded and compressed storage. No count quota.
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

    def _control(self, name, value):
        raw = dump(value) + b"\n"
        pending = self.root / (name + ".pending")
        self.budget.claim(pending, len(raw), control=True)
        with pending.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.budget.actual(pending)
        os.link(pending, self.root / name)
        self.budget.seal(pending)
        return digest(raw)

    def finish(self, *, normal_end, receipt):
        if self.closed:
            raise CodecError("finish called twice")
        try:
            if not self.faults:
                self._flush()
            self.bi.close()
            self.ai.close()
            os.link(self.block_index, self.root / "blocks.index.jsonl")
            os.link(self.array_index, self.root / "arrays.index.jsonl")
            self.budget.seal(self.block_index)
            self.budget.seal(self.array_index)
            receipt_hash = self._control("business_receipt.json", receipt)
            physical = not self.faults and not self.buffer and not self.pending_refs
            business = (
                physical
                and normal_end is True
                and self.ops.complete()
                and receipt.get("normal_end") is True
                and receipt.get("operation_ids") == self.ops.expected
            )
            summary = {
                "format_version": VERSION,
                "mode": self.mode,
                "closed": True,
                "block_integrity_complete": physical,
                "business_process_complete": business,
                "faults": list(self.faults),
                "events": self.events,
                "blocks": self.blocks,
                "arrays": self.arrays,
                "expected_operations": self.ops.expected,
                "started_operations": self.ops.started,
                "completed_operations": self.ops.completed,
                "stream_sha256": self.raw_hash.hexdigest(),
                "logical_line_chain": self.raw_chain,
                "last_block_header_sha256": self.block_chain,
                "last_array_header_sha256": self.array_chain,
                "block_index_sha256": file_hash(self.root / "blocks.index.jsonl"),
                "array_index_sha256": file_hash(self.root / "arrays.index.jsonl"),
                "receipt_sha256": receipt_hash,
                "limits": {
                    "storage_bytes": self.budget.limit,
                    "max_record_bytes": self.max_record,
                    "max_array_bytes": self.max_array,
                    "max_array_occurrences": self.max_arrays,
                    "target_block_bytes": self.target,
                },
                "storage_reserved_before_final_controls": self.budget.total,
                "qualification_updates_permitted": False,
            }
            summary_hash = self._control("summary.json", summary)
            if physical and business:
                self._control(
                    "observer_complete.json", {"summary_sha256": summary_hash}
                )
            self.closed = True
            return summary
        except Exception as exc:
            self._fault("finish", exc)
            self.closed = True
            for stream in (self.bi, self.ai):
                try:
                    stream.close()
                except Exception:
                    pass
            raise


def _read_small(path, limit=MAX_HEADER):
    if path.is_symlink():
        raise CodecError("symlink not permitted")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise CodecError("oversized metadata")
    return raw


def _alias(root, name):
    path = root / name
    pending = root / (
        name[:-6] + ".pending" if name.endswith(".jsonl") else name + ".pending"
    )
    if path.is_symlink() or pending.is_symlink():
        raise CodecError("symlink alias")
    a = path.stat()
    b = pending.stat()
    if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
        raise CodecError("pending/final alias mismatch")


def _entries(root, name):
    _alias(root, name)
    with (root / name).open("rb") as stream:
        while True:
            line = stream.readline(MAX_HEADER + 1)
            if not line:
                break
            if len(line) > MAX_HEADER or not line.endswith(b"\n"):
                raise CodecError("partial/oversized index entry")
            yield parse(line)


def _file_header(root, name, expected):
    if name != expected:
        raise CodecError("unexpected indexed filename")
    _alias(root, name)
    path = root / name
    with path.open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise CodecError("bad block magic")
        length = stream.read(4)
        if len(length) != 4:
            raise CodecError("partial block header")
        length = struct.unpack(">I", length)[0]
        if length > MAX_HEADER:
            raise CodecError("header length limit")
        raw = stream.read(length)
        if len(raw) != length:
            raise CodecError("partial header")
    header = parse(raw)
    if header.get("format_version") != VERSION:
        raise CodecError("unsupported format")
    return header, digest(raw), len(MAGIC) + 4 + length


class Reader:
    def __init__(self, directory):
        self.root = Path(directory)
        self.last_validation = None

    def _summary(self, require_business_complete):
        _alias(self.root, "summary.json")
        raw = _read_small(self.root / "summary.json")
        summary = parse(raw)
        if (
            summary.get("format_version") != VERSION
            or summary.get("closed") is not True
        ):
            raise CodecError("bad summary")
        marker = self.root / "observer_complete.json"
        self.has_complete = marker.exists()
        if self.has_complete:
            _alias(self.root, "observer_complete.json")
            if parse(_read_small(marker)).get("summary_sha256") != digest(raw):
                raise CodecError("completion binding mismatch")
        if summary.get("block_integrity_complete") is not True or summary.get("faults"):
            raise CodecError("physical stream invalid")
        if require_business_complete and (
            not self.has_complete
            or summary.get("business_process_complete") is not True
        ):
            raise CodecError("business process incomplete")
        for k in ("events", "blocks", "arrays"):
            if not integer(summary.get(k)):
                raise CodecError("bad summary count")
        if summary["mode"] not in ("hash_only", "full_capture"):
            raise CodecError("bad mode")
        for name, key in [
            ("blocks.index.jsonl", "block_index_sha256"),
            ("arrays.index.jsonl", "array_index_sha256"),
            ("business_receipt.json", "receipt_sha256"),
        ]:
            _alias(self.root, name)
            if file_hash(self.root / name) != summary[key]:
                raise CodecError("index/receipt digest mismatch")
        if (
            not integer(summary["limits"].get("storage_bytes"))
            or not 4 * CONTROL_RESERVE
            <= summary["limits"]["storage_bytes"]
            <= 4 * 1024**3
            or not 0 < summary["limits"].get("max_record_bytes", 0) <= MAX_RECORD
        ):
            raise CodecError("invalid stored resource limits")
        self._inventory(summary)
        return summary

    def _inventory(self, s):
        controls = {
            "blocks.index.pending",
            "blocks.index.jsonl",
            "arrays.index.pending",
            "arrays.index.jsonl",
            "summary.json",
            "summary.json.pending",
            "business_receipt.json",
            "business_receipt.json.pending",
        }
        if self.has_complete:
            controls |= {"observer_complete.json", "observer_complete.json.pending"}
        seen = 0
        allocated = 0
        for entry in os.scandir(self.root):
            if not entry.name.endswith(".pending"):
                allocated += entry.stat(follow_symlinks=False).st_blocks * 512
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise CodecError("foreign/symlink file")
            if entry.name in controls:
                seen += 1
                continue
            match = re.fullmatch(r"(block|array)_(\d{12})\.bin(\.pending)?", entry.name)
            if not match:
                raise CodecError("unindexed/orphan artifact")
            kind, index = match.group(1), int(match.group(2))
            cap = (
                s["blocks"]
                if kind == "block"
                else s["arrays"]
                if s["mode"] == "full_capture"
                else 0
            )
            if index >= cap:
                raise CodecError("orphan artifact index")
            seen += 1
        expected = (
            len(controls)
            + 2 * s["blocks"]
            + (2 * s["arrays"] if s["mode"] == "full_capture" else 0)
        )
        if seen != expected:
            raise CodecError("missing/duplicate artifact inventory")
        if allocated > s["limits"]["storage_bytes"]:
            raise CodecError("physical allocation over budget")

    def _array(self, entry, occurrence, event_sequence, ordinal, token, previous, mode):
        header = entry["header"]
        if (
            header.get("format_version") != VERSION
            or header.get("kind") != "array"
            or header.get("codec") != "raw"
        ):
            raise CodecError("array schema")
        if (
            header["occurrence"] != occurrence
            or header["event_sequence"] != event_sequence
            or header["ordinal"] != ordinal
            or header["descriptor"] != token
        ):
            raise CodecError("array occurrence identity mismatch")
        if (
            header["previous_header_sha256"] != previous
            or entry["previous_header_sha256"] != previous
            or digest(dump(header)) != entry["header_sha256"]
        ):
            raise CodecError("array chain/header mismatch")
        import numpy as np

        dtype = np.dtype(token["dtype"])
        shape = token["shape"]
        if (
            dtype.hasobject
            or dtype.fields
            or dtype.subdtype
            or any(not integer(x) for x in shape)
        ):
            raise CodecError("invalid array dtype/shape")
        size = math.prod(shape) * dtype.itemsize
        if size != header["decoded_bytes"] or size > MAX_ARRAY:
            raise CodecError("array byte bound")
        if mode == "hash_only":
            if entry["file"] is not None:
                raise CodecError("hash-only raster file")
        else:
            actual, hh, offset = _file_header(
                self.root, entry["file"], f"array_{occurrence:012d}.bin"
            )
            if actual != header or hh != entry["header_sha256"]:
                raise CodecError("array file/header corruption")
            if (self.root / entry["file"]).stat().st_size != offset + size:
                raise CodecError("array file length")
            if file_hash(self.root / entry["file"]) != entry["file_sha256"]:
                raise CodecError("array file digest")
            content = hashlib.sha256()
            with (self.root / entry["file"]).open("rb") as stream:
                stream.seek(offset)
                for block in iter(lambda: stream.read(1048576), b""):
                    content.update(block)
            if content.hexdigest() != token["content_sha256"]:
                raise CodecError("array content mismatch")
        return entry["header_sha256"]

    def iter_events(self, *, require_business_complete=True):
        """Bounded block/record buffers; caller must exhaust iterator to validate."""
        self.last_validation = None
        s = self._summary(require_business_complete)
        ops = Operations(s["expected_operations"])
        sequence = 0
        blocks = 0
        arrays = 0
        previous = ZERO
        array_previous = ZERO
        raw_sha = hashlib.sha256()
        chain = ZERO
        array_entries = iter(_entries(self.root, "arrays.index.jsonl"))
        for entry in _entries(self.root, "blocks.index.jsonl"):
            if (
                entry["index"] != blocks
                or entry["first_sequence"] != sequence
                or entry["previous_header_sha256"] != previous
            ):
                raise CodecError("block index order/chain")
            header, hh, offset = _file_header(
                self.root, entry["file"], f"block_{blocks:012d}.bin"
            )
            if hh != entry["header_sha256"]:
                raise CodecError("block header digest")
            if (
                header["kind"] != "events"
                or header["codec"] != "zlib"
                or header["level"] != 1
                or header["index"] != blocks
                or header["first_sequence"] != sequence
                or header["previous_header_sha256"] != previous
            ):
                raise CodecError("block header identity")
            n = header["decoded_bytes"]
            compressed = header["encoded_bytes"]
            if (
                not integer(n)
                or not 0 < n <= MAX_RECORD
                or not integer(compressed)
                or compressed > n + n // 64 + 1024
            ):
                raise CodecError("bounded decoder lengths")
            if (self.root / entry["file"]).stat().st_size != offset + compressed:
                raise CodecError("block file length")
            if file_hash(self.root / entry["file"]) != entry["file_sha256"]:
                raise CodecError("block file digest")
            with (self.root / entry["file"]).open("rb") as stream:
                stream.seek(offset)
                payload = stream.read(compressed)
            if digest(payload) != header["encoded_sha256"]:
                raise CodecError("encoded digest")
            decoder = zlib.decompressobj()
            decoded = decoder.decompress(payload, n + 1)
            if (
                len(decoded) != n
                or not decoder.eof
                or decoder.unused_data
                or decoder.unconsumed_tail
                or digest(decoded) != header["decoded_sha256"]
            ):
                raise CodecError("bounded decompression/digest")
            count = 0
            buffer = io.BytesIO(decoded)
            while True:
                line = buffer.readline(s["limits"]["max_record_bytes"] + 1)
                if not line:
                    break
                if len(line) > s["limits"]["max_record_bytes"] or not line.endswith(
                    b"\n"
                ):
                    raise CodecError("event boundary/length")
                event = parse(line)
                if (
                    type(event.get("sequence")) is not int
                    or event["sequence"] != sequence
                    or not isinstance(event.get("event"), str)
                ):
                    raise CodecError("logical sequence/schema")
                ops.event(event)
                for ordinal, token in enumerate(refs(event)):
                    try:
                        array_entry = next(array_entries)
                    except StopIteration:
                        raise CodecError("missing array occurrence") from None
                    array_previous = self._array(
                        array_entry,
                        arrays,
                        sequence,
                        ordinal,
                        token,
                        array_previous,
                        s["mode"],
                    )
                    arrays += 1
                raw_sha.update(line)
                chain = hashlib.sha256(chain.encode() + line).hexdigest()
                sequence += 1
                count += 1
                yield line
            if (
                count != header["logical_events"]
                or count != entry["logical_events"]
                or header["last_sequence"] != sequence - 1
                or entry["last_sequence"] != sequence - 1
            ):
                raise CodecError("block event counts")
            previous = hh
            blocks += 1
        try:
            next(array_entries)
        except StopIteration:
            pass
        else:
            raise CodecError("orphan array occurrence")
        if (
            (sequence, blocks, arrays) != (s["events"], s["blocks"], s["arrays"])
            or raw_sha.hexdigest() != s["stream_sha256"]
            or chain != s["logical_line_chain"]
            or previous != s["last_block_header_sha256"]
            or array_previous != s["last_array_header_sha256"]
        ):
            raise CodecError("final stream/count/chain mismatch")
        if (
            ops.started != s["started_operations"]
            or ops.completed != s["completed_operations"]
        ):
            raise CodecError("operation summary mismatch")
        receipt = parse(_read_small(self.root / "business_receipt.json", MAX_RECORD))
        complete = bool(
            self.has_complete
            and ops.complete()
            and receipt.get("normal_end") is True
            and receipt.get("operation_ids") == ops.expected
            and s["business_process_complete"] is True
        )
        if require_business_complete and not complete:
            raise CodecError("missing business boundaries/receipt")
        self.last_validation = {
            "block_integrity_complete": True,
            "business_process_complete": complete,
            "logical_events": sequence,
            "physical_blocks": blocks,
            "array_occurrences": arrays,
            "stream_sha256": raw_sha.hexdigest(),
        }

    def validate(self, *, require_business_complete=True):
        for _ in self.iter_events(require_business_complete=require_business_complete):
            pass
        return self.last_validation


def compare(left, right):
    """Strict byte-level logical comparison, streaming; no field exclusions."""
    a = Reader(left)
    b = Reader(right)
    ia = iter(a.iter_events())
    ib = iter(b.iter_events())
    count = 0
    sentinel = object()
    while True:
        x = next(ia, sentinel)
        y = next(ib, sentinel)
        if x is sentinel or y is sentinel:
            if x is not y:
                raise CodecError("event counts differ")
            break
        if x != y:
            raise CodecError("logical event differs at " + str(count))
        count += 1
    if not a.last_validation or not b.last_validation:
        raise CodecError("comparison not exhausted")
    return {
        "logical_events": count,
        "strict_match": True,
        "excluded_fields": [],
        "left": a.last_validation,
        "right": b.last_validation,
    }
