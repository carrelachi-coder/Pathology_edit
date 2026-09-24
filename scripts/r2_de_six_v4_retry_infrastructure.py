"""Freeze and run numbered same-case retries for initial infrastructure failures.

The frozen request cohort and first-attempt result files are never modified.
Scientific abstentions are intentionally excluded from this queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from r2_de_six_v4_failure_audit import audit


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, output: Path) -> None:
    protocol = json.loads((root / "protocol.json").read_text())
    if digest(root / "frozen_cohort.json") != protocol["cohort_sha256"]:
        raise RuntimeError("frozen cohort hash mismatch")
    report = audit(root)
    if report["completed"] != 222:
        raise RuntimeError("all 222 first attempts must be terminal before freezing retries")
    cases = []
    for record in report["records"]:
        if record["failure_category"] not in {"planner_transport", "wall_clock_timeout"}:
            continue
        case_id = record["case_id"]
        first_result = root / "results" / f"{case_id}.json"
        cases.append({
            "case_id": case_id,
            "initial_failure_category": record["failure_category"],
            "initial_result_sha256": digest(first_result),
        })
    manifest = {
        "schema_version": "r2-de-six-v4-infrastructure-retry-manifest-v1",
        "frozen_cohort_sha256": protocol["cohort_sha256"],
        "initial_case_count": 222,
        "attempt": 2,
        "cases": cases,
    }
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise RuntimeError("retry manifest already exists with different content")
    else:
        output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "retry_cases": len(cases)}))


def run(root: Path, manifest_path: Path, code_root: Path, code_commit: str,
        worker_index: int, worker_count: int, gpu: int) -> None:
    manifest = json.loads(manifest_path.read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if digest(root / "frozen_cohort.json") != manifest["frozen_cohort_sha256"]:
        raise RuntimeError("frozen cohort hash mismatch")
    if manifest["frozen_cohort_sha256"] != protocol["cohort_sha256"]:
        raise RuntimeError("retry manifest belongs to a different cohort")
    if not 0 <= worker_index < worker_count:
        raise ValueError("invalid worker index")
    retry_one = root / "harness" / "retry_one.py"
    for item in manifest["cases"][worker_index::worker_count]:
        case_id = item["case_id"]
        first_result = root / "results" / f"{case_id}.json"
        if digest(first_result) != item["initial_result_sha256"]:
            raise RuntimeError(f"first result changed: {case_id}")
        attempt_dir = root / "retry_attempts" / case_id / "attempt-2"
        if attempt_dir.exists():
            print(json.dumps({"case_id": case_id, "action": "existing_attempt_preserved"}), flush=True)
            continue
        command = [
            sys.executable, "-u", str(retry_one),
            "--case-id", case_id, "--attempt", "2", "--gpu", str(gpu),
            "--code-commit", code_commit, "--code-root", str(code_root),
        ]
        print(json.dumps({"case_id": case_id, "action": "start", "gpu": gpu}), flush=True)
        result = subprocess.run(command, check=False)
        print(json.dumps({"case_id": case_id, "action": "terminal", "return_code": result.returncode}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--code-commit")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, args.manifest)
    else:
        if any(value is None for value in (args.code_root, args.code_commit, args.worker_index, args.worker_count, args.gpu)):
            parser.error("run requires code root, commit, worker index/count, and GPU")
        run(args.root, args.manifest, args.code_root, args.code_commit,
            args.worker_index, args.worker_count, args.gpu)


if __name__ == "__main__":
    main()
