"""Audit and replay A4 interface-error cases as numbered attempt 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


TYPE_ERROR = (
    "_select_gradient_removal_instances() got an unexpected keyword "
    "argument 'composition_include_outer_reference'"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, manifest_path: Path) -> None:
    protocol = json.loads((root / "protocol.json").read_text())
    if sha(root / "frozen_cohort.json") != protocol["cohort_sha256"]:
        raise RuntimeError("frozen cohort hash mismatch")
    cases = []
    for metadata_path in sorted((root / "retry_attempts").glob("*/attempt-2/retry_metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        program_path = metadata.get("program_result_path")
        if metadata.get("dataset") != "ORCA" or not program_path:
            continue
        program = json.loads(Path(program_path).read_text())
        reasons = "\n".join(
            str(reason)
            for step in program.get("steps", [])
            for reason in step.get("reasons", [])
        )
        if TYPE_ERROR not in reasons:
            continue
        if metadata["status"] != "terminal" or metadata["program_status"] != "failed":
            raise RuntimeError(f"unexpected A4 predecessor status: {metadata_path}")
        cases.append({
            "case_id": metadata["case_id"],
            "attempt_2_metadata_sha256": sha(metadata_path),
            "attempt_2_program_sha256": sha(Path(program_path)),
        })
    if len(cases) != 8:
        raise RuntimeError(f"expected eight A4 interface errors, found {len(cases)}")
    manifest = {
        "schema_version": "r2-de-six-v4-A4-attempt3-manifest-v1",
        "cohort_sha256": protocol["cohort_sha256"],
        "code_commit": "1b4f248",
        "cases": cases,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("existing A4 manifest differs")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "cases": len(cases)}))


def run(root: Path, manifest_path: Path, code_root: Path,
        worker_index: int, worker_count: int, gpu: int) -> None:
    manifest = json.loads(manifest_path.read_text())
    if sha(root / "frozen_cohort.json") != manifest["cohort_sha256"]:
        raise RuntimeError("frozen cohort hash mismatch")
    if not 0 <= worker_index < worker_count:
        raise ValueError("invalid worker index")
    for item in manifest["cases"][worker_index::worker_count]:
        case_id = item["case_id"]
        attempt_2 = root / "retry_attempts" / case_id / "attempt-2"
        metadata_path = attempt_2 / "retry_metadata.json"
        if sha(metadata_path) != item["attempt_2_metadata_sha256"]:
            raise RuntimeError(f"attempt 2 metadata changed: {case_id}")
        metadata = json.loads(metadata_path.read_text())
        if sha(Path(metadata["program_result_path"])) != item["attempt_2_program_sha256"]:
            raise RuntimeError(f"attempt 2 program changed: {case_id}")
        if (root / "retry_attempts" / case_id / "attempt-3").exists():
            print(json.dumps({"case_id": case_id, "action": "existing_attempt_preserved"}), flush=True)
            continue
        command = [
            sys.executable, "-u", str(root / "harness" / "retry_one.py"),
            "--case-id", case_id, "--attempt", "3", "--gpu", str(gpu),
            "--code-commit", manifest["code_commit"],
            "--code-root", str(code_root),
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
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, args.manifest)
    else:
        if any(value is None for value in (args.code_root, args.worker_index, args.worker_count, args.gpu)):
            parser.error("run requires code root, worker index/count, and GPU")
        run(args.root, args.manifest, args.code_root,
            args.worker_index, args.worker_count, args.gpu)


if __name__ == "__main__":
    main()
