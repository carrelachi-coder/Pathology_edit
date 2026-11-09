"""Bounded independent setup evidence checks for the future preparation gate."""
import hashlib
import json
from pathlib import Path

STAGES = (
    "legacy_code",
    "runtime_code",
    "interpreter_environment",
    "distributions",
    "native_binaries",
    "input_inventory",
    "storage",
)


def read_control(path, limit):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("control limit")
    return raw


def verify_setup_result(directory, *, request_sha256, binding):
    root = Path(directory)
    summary_raw = read_control(root / "setup_summary.json", 16384)
    journal = read_control(root / "setup_telemetry.jsonl", 1048576)
    receipt = json.loads(read_control(root / "setup_binding.json", 16384))
    expected = {
        "request_sha256": request_sha256,
        "summary_sha256": hashlib.sha256(summary_raw).hexdigest(),
        "journal_sha256": hashlib.sha256(journal).hexdigest(),
    }
    if receipt != expected or len(request_sha256) != 64:
        raise ValueError("request or evidence binding mismatch")
    if binding != expected:
        raise ValueError("worker binding mismatch")
    summary = json.loads(summary_raw)
    if any(
        summary.get(k) is not True
        for k in ("telemetry_valid", "identity_check_passed", "preparation_accepted")
    ):
        raise ValueError("preparation incomplete")
    if (
        summary.get("status") != "preparation_only_pass"
        or summary.get("fault_types") != []
    ):
        raise ValueError("invalid preparation summary")
    rows = []
    for line in journal.splitlines():
        if len(line) + 1 > 2048 or len(rows) >= 2048:
            raise ValueError("journal limit")
        row = json.loads(line)
        if row.get("sequence") != len(rows):
            raise ValueError("journal sequence")
        rows.append(row)
    if (
        not journal.endswith(b"\n")
        or summary.get("records") != len(rows)
        or summary.get("telemetry_bytes") != len(journal)
    ):
        raise ValueError("journal completeness")
    summaries = summary.get("stages", [])
    if len(summaries) != len(STAGES):
        raise ValueError("stage count")
    cursor = 0
    for stage, final in zip(STAGES, summaries):
        start = rows[cursor]
        cursor += 1
        if start.get("event") != "stage_started" or start.get("stage") != stage:
            raise ValueError("stage order")
        completed = 0
        while cursor < len(rows) and rows[cursor].get("event") == "item_completed":
            row = rows[cursor]
            completed += 1
            if (
                row.get("stage") != stage
                or row.get("items_completed") != completed
                or row.get("last_completed_item_index") != completed - 1
            ):
                raise ValueError("item progress")
            cursor += 1
        end = rows[cursor]
        cursor += 1
        if (
            end.get("event") != "stage_completed"
            or end.get("stage") != stage
            or end.get("stage_completed") is not True
        ):
            raise ValueError("stage incomplete")
        for field in (
            "items_completed",
            "bytes_processed",
            "last_completed_item_index",
        ):
            if final.get(field) != end.get(field):
                raise ValueError("summary/journal mismatch")
        if (
            final.get("stage") != stage
            or final.get("stage_completed") is not True
            or end.get("items_completed") != completed
        ):
            raise ValueError("stage summary")
    if cursor != len(rows) or summary.get("last_completed_stage") != "storage":
        raise ValueError("extra or missing stage events")
    if summaries[5]["items_completed"] != 809 or summaries[6]["items_completed"] != 3:
        raise ValueError("required identity checks missing")
    return {
        "status": "setup_evidence_verified",
        **expected,
        "execution_permitted": False,
    }
