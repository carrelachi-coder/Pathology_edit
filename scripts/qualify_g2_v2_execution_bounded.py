#!/usr/bin/env python3
"""Run G2-v2 source-only compilation with per-case timeout and resume.

Each case runs in its own process and is written atomically as soon as it
finishes.  One pathological geometry search therefore cannot block or erase
the other qualification records.  The worker never creates target masks and
never invokes ProbNet, an H&E model, an LLM, or an external API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

SUPPORTED_SCHEMA = "g2-v2-image-instruction-mechanism-manifest-v2"
RECORD_SCHEMA = "g2-v2-read-only-execution-qualification-v1"
CONTROLLER_SCHEMA = "g2-v2-bounded-execution-qualification-v1"
THREAD_LIMIT_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_path(root: Path, index: int, case_id: str) -> Path:
    safe = "".join(value if value.isalnum() or value in "-_" else "_" for value in case_id)
    return root / "case_records" / f"{index:04d}_{safe}.json"


def _failure_record(
    *, row: dict[str, Any], manifest_sha256: str, reason: str,
    timeout_seconds: int, wall_time_seconds: float,
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA,
        "case_id": str(row["case_id"]),
        "source_index": int(row["source_index"]),
        "organ": row["organ"],
        "primitive_id": row.get("primitive_id"),
        "mechanism_id": row.get("mechanism_id"),
        "source_manifest_sha256": manifest_sha256,
        "target_mask_created": False,
        "source_asset_mutated": False,
        "llm_api_used": False,
        "status": "execution_requalification_required",
        "failure_reasons": [reason],
        "metrics": {},
        "bounded_compilation": {
            "status": "failed_closed",
            "timeout_seconds": timeout_seconds,
            "wall_time_seconds": round(wall_time_seconds, 3),
            "thread_limit_environment": dict(THREAD_LIMIT_ENV),
        },
    }


def _worker_main(args: argparse.Namespace) -> int:
    from phase3_joint_edit_refine.g2_execution_qualification import (
        _initialize_worker,
        _qualify_case_worker,
    )

    manifest = Path(args.frozen_manifest).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("cases")
    if not isinstance(rows, list) or args.worker_index not in range(len(rows)):
        raise ValueError("bounded qualification worker index is invalid")
    observed_sha256 = _sha256(manifest)
    if observed_sha256 != args.manifest_sha256:
        raise ValueError("bounded qualification manifest drifted before worker start")
    _initialize_worker()
    record = _qualify_case_worker(
        (
            rows[args.worker_index],
            observed_sha256,
            str(Path(args.output_dir).resolve() / "source_auxiliary"),
        )
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


def _compile_one(
    *, index: int, row: dict[str, Any], manifest: Path,
    manifest_sha256: str, output_dir: Path, timeout_seconds: int,
    resume: bool,
) -> dict[str, Any]:
    case_id = str(row["case_id"])
    path = _record_path(output_dir, index, case_id)
    if resume and path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("case_id") == case_id
            and cached.get("source_manifest_sha256") == manifest_sha256
            and cached.get("schema_version") == RECORD_SCHEMA
        ):
            return cached
    log_root = output_dir / "case_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{index:04d}_{case_id}.stdout.log"
    stderr_path = log_root / f"{index:04d}_{case_id}.stderr.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--frozen-manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--worker-index",
        str(index),
        "--manifest-sha256",
        manifest_sha256,
    ]
    environment = dict(os.environ)
    environment.update(THREAD_LIMIT_ENV)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        duration = time.monotonic() - started
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode:
            record = _failure_record(
                row=row,
                manifest_sha256=manifest_sha256,
                reason=f"bounded_candidate_compiler_exit_{completed.returncode}",
                timeout_seconds=timeout_seconds,
                wall_time_seconds=duration,
            )
        else:
            try:
                record = json.loads(completed.stdout)
            except (TypeError, ValueError):
                record = _failure_record(
                    row=row,
                    manifest_sha256=manifest_sha256,
                    reason="bounded_candidate_compiler_output_invalid",
                    timeout_seconds=timeout_seconds,
                    wall_time_seconds=duration,
                )
            else:
                record["bounded_compilation"] = {
                    "status": "completed",
                    "timeout_seconds": timeout_seconds,
                    "wall_time_seconds": round(duration, 3),
                    "thread_limit_environment": dict(THREAD_LIMIT_ENV),
                }
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        record = _failure_record(
            row=row,
            manifest_sha256=manifest_sha256,
            reason=f"bounded_candidate_compilation_timeout_{timeout_seconds}s",
            timeout_seconds=timeout_seconds,
            wall_time_seconds=duration,
        )
    record["bounded_compilation"]["stdout"] = str(stdout_path)
    record["bounded_compilation"]["stderr"] = str(stderr_path)
    _write_json(path, record)
    return record


def _controller_main(args: argparse.Namespace) -> int:
    manifest = Path(args.frozen_manifest).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("cases")
    if (
        payload.get("schema_version") != SUPPORTED_SCHEMA
        or not isinstance(rows, list)
        or len(rows) != int(payload.get("case_count", -1))
    ):
        raise ValueError("unsupported or inconsistent G2-v2 manifest")
    if args.workers <= 0 or args.timeout_seconds <= 0:
        raise ValueError("workers and timeout seconds must be positive")
    selected_evaluations = None
    if args.evaluation_indices:
        selected_evaluations = {
            int(value.strip())
            for value in args.evaluation_indices.split(",")
            if value.strip()
        }
        if not selected_evaluations:
            raise ValueError("evaluation indices must not be empty")
    indexed_rows = [
        (index, row)
        for index, row in enumerate(rows)
        if selected_evaluations is None
        or int(row.get("evaluation_index", -1)) in selected_evaluations
    ]
    if not indexed_rows:
        raise ValueError("evaluation filter selected no manifest cases")
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = _sha256(manifest)
    records_by_index: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _compile_one,
                index=index,
                row=row,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                output_dir=root,
                timeout_seconds=args.timeout_seconds,
                resume=args.resume,
            ): index
            for index, row in indexed_rows
        }
        completed_count = 0
        for future in as_completed(futures):
            index = futures[future]
            record = future.result()
            records_by_index[index] = record
            completed_count += 1
            print(
                json.dumps(
                    {
                        "stage": "bounded_candidate_compilation",
                        "completed": completed_count,
                        "total": len(indexed_rows),
                        "case_id": record["case_id"],
                        "status": record["status"],
                        "wall_time_seconds": record["bounded_compilation"][
                            "wall_time_seconds"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    records = [records_by_index[index] for index, _row in indexed_rows]
    ledger = root / "execution_qualification.jsonl"
    ledger.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": CONTROLLER_SCHEMA,
        "record_schema_version": RECORD_SCHEMA,
        "source_manifest": str(manifest),
        "source_manifest_sha256": manifest_sha256,
        "case_count": len(records),
        "source_case_indices": [index for index, _row in indexed_rows],
        "evaluation_indices": (
            sorted(selected_evaluations)
            if selected_evaluations is not None
            else None
        ),
        "status_counts": dict(
            sorted(Counter(str(item["status"]) for item in records).items())
        ),
        "failure_reason_counts": dict(
            sorted(
                Counter(
                    str(reason)
                    for item in records
                    for reason in item.get("failure_reasons", [])
                ).items()
            )
        ),
        "ledger": str(ledger),
        "ledger_sha256": _sha256(ledger),
        "case_record_dir": str(root / "case_records"),
        "workers": args.workers,
        "per_case_timeout_seconds": args.timeout_seconds,
        "thread_limit_environment": dict(THREAD_LIMIT_ENV),
        "target_mask_created": False,
        "source_asset_mutated": False,
        "probnet_called": False,
        "llm_api_used": False,
    }
    summary_path = root / "execution_qualification_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps({**summary, "summary": str(summary_path)}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--evaluation-indices",
        help="optional comma-separated evaluation indices for bounded rescreening",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_index is not None:
        if not args.manifest_sha256:
            raise ValueError("bounded worker requires manifest sha256")
        return _worker_main(args)
    return _controller_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
