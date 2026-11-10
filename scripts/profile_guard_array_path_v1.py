#!/usr/bin/env python3
"""One-shot synthetic Codec B/C microprofile; no pathology imports or calls."""
import argparse
from collections import Counter
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import random
import shutil
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_observer_budget_large_synthetic_v1 import (
    distribution_inventory, sha,
)


def preflight(manifest, approval):
    raw = Path(manifest).read_bytes()
    m = json.loads(raw)
    a = json.loads(Path(approval).read_bytes())
    if not (a.get("approved") is True
            and a.get("kind") == m.get("kind") == "observer-array-microprofile-v1"
            and a.get("manifest_sha256") == hashlib.sha256(raw).hexdigest()):
        raise ValueError("manifest-bound authorization required")
    return m


def _selected_profile(path):
    selected = {"add_array", "_copy_array", "_publish", "_append",
                "write_event", "_flush", "fsync"}
    stats = pstats.Stats(str(path))
    rows = []
    for (file, line, function), (_, calls, self_time, cumulative, _) in stats.stats.items():
        if function in selected or "fsync" in function:
            rows.append({
                "function": function, "file": Path(file).name, "line": line,
                "calls": calls, "self_seconds": self_time,
                "cumulative_seconds": cumulative,
            })
    return sorted(rows, key=lambda row: (row["function"], row["file"], row["line"]))


def _event(sequence, kind, fields):
    return (json.dumps({"sequence": sequence, "event": kind, **fields},
                       sort_keys=True, separators=(",", ":")) + "\n").encode()


def _arm(root, mode, manifest, codec):
    import numpy as np

    started = time.monotonic()
    writer = codec.Writer(root / mode, mode=mode)
    rng = np.random.default_rng(20261110)
    count = manifest["events_per_arm"]
    arrays = manifest["arrays_per_arm"]
    interval = (count - 4) // (arrays + 1)
    if interval <= 0:
        raise ValueError("invalid workload")
    profiler = cProfile.Profile()
    generated = 0
    guard_ordinal = 0
    profiler.enable()
    try:
        for i in range(count):
            if i % 4096 == 0 and time.monotonic() - started >= manifest["seconds_per_arm"]:
                raise TimeoutError("microprofile arm deadline")
            if i in (0, count // 2):
                kind = "qualification.begin"
                fields = {"operation_id": "U1" if i == 0 else "U2"}
            elif i in (count // 2 - 1, count - 1):
                kind = "qualification.complete"
                fields = {"operation_id": "U1" if i == count // 2 - 1 else "U2"}
            else:
                kind = "guard.stage"
                guard_ordinal += 1
                fields = {"accepted": i % 3 == 0, "row": (i // 512) % 512,
                          "col": i % 512}
                if guard_ordinal % interval == 0 and generated < arrays:
                    fields["snapshot"] = writer.add_array(
                        rng.random((512, 512), dtype=np.float64)
                    )
                    generated += 1
            writer.write_event(_event(i, kind, fields))
    finally:
        profiler.disable()
        profiler.dump_stats(str(root / f"{mode}.pstats"))
    if generated != arrays:
        raise ValueError("array count mismatch")
    loop_seconds = time.monotonic() - started
    if loop_seconds >= manifest["seconds_per_arm"]:
        raise TimeoutError("microprofile arm deadline")
    summary = writer.finish(
        normal_end=True,
        receipt={"normal_end": True, "operation_ids": ["U1", "U2"],
                 "payload": {"synthetic_only": True}},
    )
    if not (summary["block_integrity_complete"]
            and summary["business_process_complete"]):
        raise ValueError("incomplete synthetic stream")
    total_seconds = time.monotonic() - started
    if total_seconds >= manifest["seconds_per_arm"]:
        raise TimeoutError("microprofile arm encode/seal deadline")
    return {
        "mode": mode, "events": summary["events"], "arrays": summary["arrays"],
        "stream_sha256": summary["stream_sha256"],
        "loop_seconds": loop_seconds,
        "total_seconds": total_seconds,
        "profile_sha256": sha(root / f"{mode}.pstats"),
        "profile_rows": _selected_profile(root / f"{mode}.pstats"),
    }


def run(manifest, approval):
    started = time.monotonic()
    m = preflight(manifest, approval)
    if m.get("controls"):
        controls = Path(m["controls"]).resolve()
        if (Path(manifest).resolve() != controls / "manifest.json"
                or Path(approval).resolve() != controls / "approval.json"):
            raise ValueError("control path identity mismatch")
    root = Path(m["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "authorization_consumed.json").open("x").write(
        json.dumps({"manifest_sha256": sha(manifest), "consumed": True}) + "\n"
    )
    result = {"status": "HOLD", "automatic_retries": 0, "S07_calls": 0,
              "witness_calls": 0, "HE_calls": 0, "outer_acceptance_published": False,
              "arms": []}
    arm = None
    try:
        here = Path(__file__).resolve().parents[1]
        if m.get("snapshot") and here != Path(m["snapshot"]).resolve():
            raise ValueError("snapshot path identity mismatch")
        deadline = started + m["setup_seconds"]
        for name, expected in m["code_files_sha256"].items():
            if time.monotonic() >= deadline or sha(here / name) != expected:
                raise ValueError("setup timeout or code identity mismatch: " + name)
        runtime = m["runtime"]
        if (sha(sys.executable) != runtime["executable_sha256"]
                or sys.version != runtime["python"]
                or platform.platform() != runtime["platform"]):
            raise ValueError("runtime identity mismatch")
        frozen = Counter((x["name"], x["version"], x["record_sha256"])
                         for x in runtime["packages"])
        if distribution_inventory() != frozen:
            raise ValueError("distribution identity mismatch")
        if any(os.environ.get(k) != v for k, v in runtime["environment"].items()):
            raise ValueError("environment identity mismatch")
        if shutil.disk_usage(root).free < m["free_bytes_required"]:
            raise ValueError("insufficient free bytes")
        pending, final = root / "hardlink.pending", root / "hardlink.final"
        pending.open("xb").write(b"original")
        os.link(pending, final)
        try:
            os.link(pending, final)
        except FileExistsError:
            pass
        else:
            raise ValueError("hardlink no-replace failed")
        import numpy as np
        from diagnostic_storage import guard_blocks_v2 as codec
        if time.monotonic() >= deadline:
            raise TimeoutError("setup deadline")
        result["setup_seconds"] = time.monotonic() - started
        rng_before = hashlib.sha256(
            repr((random.getstate(), np.random.get_state())).encode()
        ).hexdigest()
        if any(n.startswith(("phase3_", "inpaint_cells")) for n in sys.modules) or "torch" in sys.modules:
            raise ValueError("business or torch imported")
        for arm in ("hash_only", "full_capture"):
            result["arms"].append(_arm(root, arm, m, codec))
        if result["arms"][0]["stream_sha256"] != result["arms"][1]["stream_sha256"]:
            raise ValueError("logical stream hash mismatch")
        comparison = codec.compare(root / "hash_only", root / "full_capture")
        if comparison["strict_match"] is not True:
            raise ValueError("strict Codec B/C mismatch")
        result["strict_match"] = True
        result["comparison"] = comparison
        rng_after = hashlib.sha256(
            repr((random.getstate(), np.random.get_state())).encode()
        ).hexdigest()
        if rng_after != rng_before:
            raise ValueError("global RNG changed")
        if any(n.startswith(("phase3_", "inpaint_cells")) for n in sys.modules) or "torch" in sys.modules:
            raise ValueError("business or torch imported")
        result["global_rng_unchanged"] = True
        result["status"] = "synthetic_codec_microprofile_complete"
    except Exception as exc:
        result.update(failure_type=type(exc).__name__,
                      failure_text=str(exc)[:512], stopped_arm=arm)
    result["elapsed_seconds"] = time.monotonic() - started
    (root / "trial_result.json").open("x").write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "elapsed_seconds": result["elapsed_seconds"],
                      "arms_complete": len(result["arms"]),
                      "failure_type": result.get("failure_type")}))
    return 0 if result["status"] == "synthetic_codec_microprofile_complete" else 78


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.manifest, args.approval))
