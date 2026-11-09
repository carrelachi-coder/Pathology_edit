"""Import-only controlled-startup audit for the Adapter controller candidate."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(output, torch_cpu=False):
    startup = load(
        "_adapter_startup_audit", ROOT / "scripts/limited_topology_startup.py"
    )
    controller = load(
        "_adapter_control_audit",
        ROOT / "scripts/run_limited_topology_adapter_controller_v1.py",
    )
    if torch_cpu:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("CPU-only environment required")
        import torch
    require_linux = sys.platform.startswith("linux")
    startup.parent_gate(require_linux_threads=require_linux)
    before = startup.capture()
    # Same stdlib-only control imports performed by assessment between A and B.
    import diagnostic_storage.validator_process_gate_v2

    startup.parent_gate(require_linux_threads=require_linux)
    after = startup.capture()
    if startup.digest(before) != startup.digest(after):
        raise RuntimeError("parent control imports changed RNG/hash")
    output = Path(output)
    output.mkdir(exist_ok=False)
    rows = []
    for arm in ["A", "B", "C"]:
        path = output / (arm + ".json")

        def child():
            startup.restore_controlled(before)
            initial = startup.capture()
            if arm != "A":
                import diagnostic_storage.observer_adapter_v1
                import diagnostic_storage.observer_invocation_bridge_v1
            final = startup.capture()
            row = {
                "arm": arm,
                "before_digest": startup.digest(initial),
                "after_digest": startup.digest(final),
                "unchanged": startup.digest(initial) == startup.digest(final),
                "native_threads": startup.native_threads(),
                "torch_loaded": "torch" in sys.modules,
                "CUDA_initialized": sys.modules["torch"].cuda.is_initialized()
                if "torch" in sys.modules
                else False,
                "module_order": [
                    n
                    for n in sys.modules
                    if n.startswith(("phase3_", "diagnostic_storage"))
                ],
            }
            with path.open("x") as f:
                json.dump(row, f, indent=2)

        process = controller.bounded_fork(child, timeout=20)
        if process["timed_out"] or process["exit_code"] != 0:
            raise RuntimeError("startup child HOLD")
        row = json.loads(path.read_text())
        row["process"] = process
        rows.append(row)
    report = {
        "status": "pass"
        if all(r["unchanged"] and not r["CUDA_initialized"] for r in rows)
        else "HOLD",
        "scope": "Parent assessment imports plus child Adapter/bridge imports only; excludes business probe import and actual witness execution",
        "parent_control_import_rng_hash_unchanged": True,
        "platform": sys.platform,
        "linux_thread_gate_verified": require_linux,
        "parent_native_threads": startup.native_threads(),
        "torch_preloaded_cpu": torch_cpu,
        "children": rows,
        "BC_module_order_equal": rows[1]["module_order"] == rows[2]["module_order"],
        "witness_calls": 0,
        "execution_permitted": False,
    }
    if not report["BC_module_order_equal"]:
        report["status"] = "HOLD"
    with (output / "report.json").open("x") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--torch-cpu", action="store_true")
    args = p.parse_args()
    result = run(args.output, args.torch_cpu)
    print(json.dumps({"status": result["status"], "witness_calls": 0}))
    raise SystemExit(0 if result["status"] == "pass" else 78)
