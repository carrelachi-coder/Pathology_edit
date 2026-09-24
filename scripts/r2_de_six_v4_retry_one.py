"""Rerun one frozen six-dataset case without overwriting its first attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data1/lyw/pathology_edit_eval/r2_de_terra_six_v4_20260924")


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--code-root", type=Path, default=ROOT / "code")
    parser.add_argument("--queue-root", type=Path, default=ROOT / "terra_queue")
    args = parser.parse_args()
    code = args.code_root.resolve()
    if args.attempt < 2:
        parser.error("retry attempt must be at least 2")
    protocol = json.loads((ROOT / "protocol.json").read_text())
    frozen_bytes = (ROOT / "frozen_cohort.json").read_bytes()
    if hashlib.sha256(frozen_bytes).hexdigest() != protocol["cohort_sha256"]:
        raise RuntimeError("frozen cohort digest changed")
    rows = json.loads(frozen_bytes)
    matches = [row for row in rows if row["record"]["case_id"] == args.case_id]
    if len(matches) != 1:
        raise RuntimeError("case is not uniquely present in frozen cohort")
    row = matches[0]
    manifest = ROOT / "inputs" / f"{args.case_id}.json"
    if json.loads(manifest.read_text()) != [row["record"]]:
        raise RuntimeError("first-attempt manifest differs from frozen record")
    first_result = ROOT / "results" / f"{args.case_id}.json"
    if not first_result.exists():
        raise RuntimeError("first attempt must be retained before retry")
    attempt_dir = ROOT / "retry_attempts" / args.case_id / f"attempt-{args.attempt}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    start = time.time()
    source_hashes = {
        str(path.relative_to(code)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            code / "phase3_joint_edit_refine" / "agents.py",
            code / "phase3_joint_edit_refine" / "tissue_planner.py",
            code / "phase3_joint_edit_refine" / "planner.py",
            code / "phase3_joint_edit_refine" / "mature_probnet_adapter.py",
            code / "inpaint_cells" / "generate.py",
        )
    }
    metadata = {
        "case_id": args.case_id,
        "attempt": args.attempt,
        "dataset": row["dataset"],
        "code_commit": args.code_commit,
        "code_root": str(code),
        "runtime_source_hashes": source_hashes,
        "frozen_cohort_sha256": protocol["cohort_sha256"],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "first_result_sha256": hashlib.sha256(first_result.read_bytes()).hexdigest(),
        "first_result_path": str(first_result),
        "started_at_unix": start,
        "gpu": args.gpu,
        "queue_root": str(args.queue_root),
        "status": "running",
    }
    dump(attempt_dir / "retry_metadata.json", metadata)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        PYTHONPATH=str(code) + ":/home/lyw/wqx-DL/flow-edit/FlowEdit-main",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    dataset = "GlaS" if row["dataset"] == "GLAS" else row["dataset"]
    output_root = attempt_dir / "run"
    command = [
        sys.executable, "-u", "-c",
        "from phase3_joint_edit_refine.program_cli import main; raise SystemExit(main())",
        "--manifest", str(manifest), "--output-root", str(output_root),
        "--semantic-parser", "prebound", "--agent-mode", "cli-queue",
        "--planner-queue-root", str(args.queue_root),
        "--model", "gpt-5.6-terra", "--reasoning-effort", "medium",
        "--cell-executor", "mature",
        "--probnet-checkpoint",
        "/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/best_epoch29_c29607f1b609accb.pt",
        "--nuclei-instance-library",
        "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/nuclei_library/" + dataset,
        "--probnet-dataset", dataset, "--device", "cuda", "--meta-eval",
    ]
    with (attempt_dir / "stdout.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=code, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=protocol["per_request_timeout_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return_code = None
    program_result = output_root / args.case_id / "program_result.json"
    metadata.update(
        status="timeout" if return_code is None else "terminal",
        return_code=return_code,
        elapsed_seconds=round(time.time() - start, 2),
        program_result_path=str(program_result) if program_result.exists() else None,
    )
    if program_result.exists():
        result = json.loads(program_result.read_text())
        metadata["program_status"] = result["status"]
        metadata["program_evaluation_passed"] = result["evaluation"].get("passed")
        if result["status"] == "validated" and result["evaluation"].get("passed"):
            handoffs = list((output_root / args.case_id).rglob("generation_handoff/manifest.json"))
            if len(handoffs) != 1:
                raise RuntimeError("validated retry lacks a unique generation handoff")
            sys.path[:0] = [str(code), "/home/lyw/wqx-DL/flow-edit/FlowEdit-main"]
            from measure import measure

            independent = measure(row, handoffs[0])
            dump(attempt_dir / "independent_E.json", independent)
            metadata["independent_E_passed"] = independent[
                "independent_raster_contract_checks_passed"
            ]
    dump(attempt_dir / "retry_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    return 0 if metadata.get("independent_E_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
