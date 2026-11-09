#!/usr/bin/env python3
"""Import-only acceptance probe. Never executes main/probe/witness functions.

Uses the reviewed controlled startup proposal solely for this smoke test; it is
not S07 authorization or an A/B/C diagnostic invocation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

FORBIDDEN = {
    "probe",
    "_qualify_tissue_case",
    "compile_tissue_witness",
    "compile_tissue_portfolio",
    "_compile_one",
    "_resolve_topology_safe_area",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    spec = importlib.util.spec_from_file_location(
        "_review_controller", plan["controller"]["path"]
    )
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    controller.identity_gate(plan)
    controller.scope_gate(plan)  # Read-only, no run().
    import numpy
    import torch

    path = Path(plan["controller"]["path"]).with_name("limited_topology_startup.py")
    spec = importlib.util.spec_from_file_location("_review_startup", path)
    startup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(startup)
    startup.parent_gate()
    common = startup.capture()
    sys.path.insert(0, plan["adapter_snapshot"])
    import diagnostic_storage.validator_process_gate_v2

    startup.parent_gate()
    if startup.digest(startup.capture()) != startup.digest(common):
        raise RuntimeError("parent control imports changed RNG/hash")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for arm in ("A", "B", "C"):
        destination = args.output / (arm + "_import_smoke.json")

        def child():
            startup.restore_controlled(common)
            startup.parent_gate()
            before = startup.capture()
            events = []
            blocked = []
            error = None
            snapshot = Path(plan["arms"][arm]["snapshot"])
            os.chdir(snapshot)
            sys.path.insert(0, str(snapshot))
            probe = snapshot / plan["arms"][arm]["probe"]
            sys.argv = [
                str(probe),
                *plan["arms"][arm]["probe_args"],
                "--output",
                str(Path(plan["output_root"]) / arm / "business"),
            ]

            def profile(frame, event, arg):
                if event != "call":
                    return
                filename = frame.f_code.co_filename
                name = frame.f_code.co_name
                module = frame.f_globals.get("__name__", "")
                if (
                    name in FORBIDDEN
                    and (
                        module.startswith(("phase3_", "inpaint_cells"))
                        or filename == str(probe)
                    )
                ) or (name == "main" and filename == str(probe)):
                    blocked.append({"name": name, "module": module, "file": filename})
                    raise RuntimeError(
                        "forbidden business execution during import-only smoke"
                    )
                if name == "<module>" and (
                    filename.startswith(str(snapshot) + "/")
                    or filename.startswith(plan["adapter_snapshot"] + "/")
                ):
                    base = (
                        snapshot
                        if filename.startswith(str(snapshot) + "/")
                        else Path(plan["adapter_snapshot"])
                    )
                    events.append(
                        {
                            "index": len(events),
                            "module": module,
                            "file": str(Path(filename).relative_to(base)),
                            "entry_state_sha256": startup.digest(startup.capture()),
                        }
                    )

            sys.setprofile(profile)
            try:
                if arm != "A":
                    sys.path.insert(0, plan["adapter_snapshot"])
                    from diagnostic_storage.observer_adapter_v1 import ObserverAdapter
                    from diagnostic_storage.observer_invocation_bridge_v1 import (
                        invoke_with_adapter,
                    )
                # __main__ guard is FALSE; no CLI main or source asset is invoked.
                imported = runpy.run_path(str(probe), run_name="_s07_import_only_smoke")
                if not callable(imported.get("probe")) or not callable(
                    imported.get("main")
                ):
                    raise RuntimeError("expected definitions absent")
            except BaseException as exc:
                error = {"type": type(exc).__name__, "text": str(exc)}
            finally:
                sys.setprofile(None)
            after = startup.capture()
            report = {
                "arm_label_for_import_path_only": arm,
                "startup_digest": startup.digest(before),
                "post_import_digest": startup.digest(after),
                "initial_matches_parent": startup.digest(before)
                == startup.digest(common),
                "default_rng_unchanged_by_import": startup.digest(before)
                == startup.digest(after),
                "hash_fingerprint_before": before["hash_fingerprint"],
                "hash_fingerprint_after": after["hash_fingerprint"],
                "per_rng_before": {k: startup.digest(v) for k, v in before.items()},
                "per_rng_after": {k: startup.digest(v) for k, v in after.items()},
                "native_threads_after_import": startup.native_threads(),
                "CUDA_initialized_after": torch.cuda.is_initialized(),
                "ordered_source_module_entries": events,
                "blocked_calls": blocked,
                "error": error,
                "witness_calls": 0,
            }
            controller.write_new(destination, report)
            if error or blocked:
                raise RuntimeError("import smoke failed")

        receipt = controller.bounded_fork(child, timeout=90)
        row = (
            json.loads(destination.read_text())
            if destination.exists()
            else {"error": "no child report"}
        )
        row["process_receipt"] = receipt
        rows.append(row)
        if receipt["timed_out"] or receipt["exit_code"] != 0:
            break

    def sequence(row, omit_observer=False):
        return [
            e["file"]
            for e in row.get("ordered_source_module_entries", [])
            if not omit_observer
            or e["file"] != "phase3_mask_edit_refine/passive_observer.py"
        ]

    complete = len(rows) == 3
    report = {
        "scope": "Import-only smoke after read-only identity checks; no main/probe/witness invocation",
        "manifest_sha256": controller.sha(args.manifest),
        "execution_permitted": False,
        "witness_calls": 0,
        "parent_native_threads": startup.native_threads(),
        "parent_business_modules": [
            n for n in sys.modules if n.startswith(("phase3_", "inpaint_cells"))
        ],
        "completed_import_children": len(rows),
        "all_initial_states_equal": complete
        and len({r.get("startup_digest") for r in rows}) == 1,
        "all_hash_fingerprints_equal_before_business": complete
        and len({r.get("hash_fingerprint_before") for r in rows}) == 1,
        "all_default_states_unchanged_by_import": complete
        and all(r.get("default_rng_unchanged_by_import") for r in rows),
        "BC_import_order_equal": complete and sequence(rows[1]) == sequence(rows[2]),
        "AB_import_order_equal": complete and sequence(rows[0]) == sequence(rows[1]),
        "comparison_note": "Relative source names used solely to compare import chronology; no production receipt/transcript normalization or exclusions added.",
        "children": rows,
    }
    report["parent_control_import_state_unchanged"] = True
    report["status"] = (
        "pass"
        if (
            complete
            and report["all_initial_states_equal"]
            and report["all_hash_fingerprints_equal_before_business"]
            and report["all_default_states_unchanged_by_import"]
            and report["BC_import_order_equal"]
            and all(
                r.get("native_threads_after_import") == 1
                and not r.get("CUDA_initialized_after")
                and not r.get("error")
                and not r.get("blocked_calls")
                for r in rows
            )
        )
        else "HOLD"
    )
    controller.write_new(args.output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "children"}, indent=2))
    return 0 if report["status"] == "pass" else 78


if __name__ == "__main__":
    raise SystemExit(main())
