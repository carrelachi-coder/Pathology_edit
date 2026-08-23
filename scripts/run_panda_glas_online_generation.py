#!/usr/bin/env python3
"""Run the final PANDA/GLaS Online generation cohort on one worker per GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE5_PYTHON = Path(
    "/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python"
)
SEGMENTATOR_PYTHON = Path(
    "/home/lyw/anaconda3/envs/pathology-segmentator-mmseg/bin/python3.10"
)
MODEL_ENVIRONMENT = {
    "PATHOLOGY_INPAINT_CHECKPOINT": (
        "/home/lyw/wqx-DL/flow-edit/hf_generation_release/"
        "pathology-inpaint-controlnet"
    ),
    "PATHOLOGY_CROSS_V1_CHECKPOINT": (
        "/home/lyw/wqx-DL/flow-edit/hf_generation_release/"
        "pathology-cross-v1-pix2pix/cross_v1"
    ),
    "PATHOLOGY_PIX2PIX_CHECKPOINT": (
        "/home/lyw/wqx-DL/flow-edit/hf_generation_release/"
        "pathology-cross-v1-pix2pix/pix2pix/"
        "pix2pix_epoch26_step214895.pt"
    ),
    "PATHOLOGY_SEGMENTATOR_CHECKPOINT": (
        "/data1/zhao/wqx/segmentator_fine/legacy_anchor_fine_seed42/"
        "best_composite.pt"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", default="4,5,6")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-both-routes",
        action="store_true",
        help=(
            "Generate the normal first route and deliberately continue to its "
            "approved fallback so Inpaint and Cross can both be visually reviewed."
        ),
    )
    parser.add_argument(
        "--prompt-overrides",
        type=Path,
        help="Optional JSON object mapping case IDs to review-specific render prompts.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def completed_record(row: dict[str, Any], case_output: Path) -> dict[str, Any] | None:
    summary_path = case_output / "pipeline_summary.json"
    generated_path = case_output / "generated_image.png"
    workflow_path = case_output / "agentic_workflow.json"
    if not (summary_path.is_file() and generated_path.is_file() and workflow_path.is_file()):
        return None
    summary = read_json(summary_path)
    workflow = read_json(workflow_path)
    return {
        **row,
        "agentic_generation_dir": str(case_output),
        "return_code": 0 if workflow.get("status") in {"validated_first_pass", "recovered", "noop"} else 2,
        "workflow_status": workflow.get("status"),
        "selected_mode": (workflow.get("selected_attempt") or {}).get("requested_mode"),
        "quality_score": ((workflow.get("selected_attempt") or {}).get("verification") or {}).get("quality_score"),
        "generation_context_policy": (summary.get("change_regions") or {}).get("generation_context_policy"),
        "resumed": True,
    }


def run_case(
    row: dict[str, Any],
    *,
    gpu: int,
    output_root: Path,
    max_attempts: int,
    seed: int,
    run_both_routes: bool,
    prompt_override: str | None,
    force: bool,
) -> dict[str, Any]:
    case_output = output_root / "cases" / row["case_id"]
    if not force:
        completed = completed_record(row, case_output)
        if completed is not None:
            return completed
    case_output.mkdir(parents=True, exist_ok=True)
    command = [
        str(PHASE5_PYTHON),
        str(REPO_ROOT / "scripts" / "run_agentic_edit_workflow.py"),
        "--profile", str(row["profile"]),
        "--reference-image", str(row["source_image"]),
        "--reference-tissue-mask", str(row["source_tissue_mask"]),
        "--reference-nuclei-mask", str(row["source_nuclei_mask"]),
        "--target-tissue-mask", str(row["target_tissue_mask"]),
        "--target-nuclei-mask", str(row["target_nuclei_mask"]),
        "--joint-generation-handoff", str(row["joint_generation_handoff"]),
        "--semantic-change-region", str(row["semantic_change_region"]),
        "--generation-change-region", str(row["generation_change_region"]),
        "--primitive-id", str(row["primitive_id"]),
        "--output", str(case_output),
        "--device", f"cuda:{gpu}",
        "--segmentator-python", str(SEGMENTATOR_PYTHON),
        "--segmentator-device", f"cuda:{gpu}",
        "--cellvit-gpu", str(gpu),
        "--max-attempts", str(max_attempts),
        "--seed", str(seed),
    ]
    if run_both_routes:
        command.extend(["--inject-verifier-failure-attempt", "1"])
    if prompt_override:
        command.extend(["--prompt", prompt_override])
    environment = dict(os.environ)
    environment.update(MODEL_ENVIRONMENT)
    started = time.time()
    with (case_output / "runner.stdout.log").open("w", encoding="utf-8") as stdout, (
        case_output / "runner.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    completed = completed_record(row, case_output)
    if completed is None:
        return {
            **row,
            "agentic_generation_dir": str(case_output),
            "return_code": int(process.returncode),
            "workflow_status": "runtime_failed",
            "gpu": gpu,
            "wall_time_seconds": round(time.time() - started, 3),
            "resumed": False,
        }
    return {
        **completed,
        "return_code": int(process.returncode),
        "gpu": gpu,
        "wall_time_seconds": round(time.time() - started, 3),
        "resumed": False,
    }


def main() -> int:
    args = parse_args()
    payload = read_json(args.manifest)
    prompt_overrides = (
        read_json(args.prompt_overrides) if args.prompt_overrides is not None else {}
    )
    if not isinstance(prompt_overrides, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in prompt_overrides.items()
    ):
        raise ValueError("--prompt-overrides must contain a JSON object of strings")
    rows = list(payload.get("records") or [])
    selected_ids = set(args.case_id)
    if selected_ids:
        rows = [row for row in rows if row["case_id"] in selected_ids]
        missing = selected_ids - {row["case_id"] for row in rows}
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "generation_progress.json"
    result_path = output / "generation_results.json"
    lock = threading.Lock()
    results: list[dict[str, Any]] = []

    def worker(worker_rows: list[dict[str, Any]], gpu: int) -> list[dict[str, Any]]:
        worker_results = []
        for row in worker_rows:
            result = run_case(
                row,
                gpu=gpu,
                output_root=output,
                max_attempts=args.max_attempts,
                seed=args.seed,
                run_both_routes=args.run_both_routes,
                prompt_override=prompt_overrides.get(row["case_id"]),
                force=args.force,
            )
            worker_results.append(result)
            with lock:
                results.append(result)
                write_json(
                    progress_path,
                    {
                        "completed": len(results),
                        "total": len(rows),
                        "records": sorted(results, key=lambda item: item["case_id"]),
                    },
                )
        return worker_results

    shards = [rows[index::len(gpus)] for index in range(len(gpus))]
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(worker, shard, gpu)
            for shard, gpu in zip(shards, gpus)
            if shard
        ]
        for future in as_completed(futures):
            future.result()

    ordered = sorted(
        results,
        key=lambda item: (
            0 if item["dataset"] == "PANDA" else 1,
            int(item["primitive_ordinal"]),
            str(item["case_id"]),
        ),
    )
    manifest = {
        "schema_version": "panda-glas-online-generation-results-v1",
        "source_manifest": str(args.manifest.resolve()),
        "case_count": len(ordered),
        "successful_runtime_count": sum(
            item["workflow_status"] != "runtime_failed" for item in ordered
        ),
        "validated_count": sum(
            item["workflow_status"] in {"validated_first_pass", "recovered", "noop"}
            for item in ordered
        ),
        "needs_review_count": sum(
            item["workflow_status"] == "needs_review" for item in ordered
        ),
        "evaluator_uncertain_count": sum(
            item["workflow_status"] == "evaluator_uncertain" for item in ordered
        ),
        "records": ordered,
    }
    write_json(result_path, manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}))
    return 0 if manifest["successful_runtime_count"] == len(ordered) else 1


if __name__ == "__main__":
    raise SystemExit(main())
