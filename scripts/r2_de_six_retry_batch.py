"""Run an explicit, frozen list of same-case retries without overwriting attempts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def dump(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    protocol = json.loads((args.root / "protocol.json").read_text())
    if manifest["cohort_sha256"] != protocol["cohort_sha256"]:
        raise RuntimeError("retry manifest does not match frozen cohort")
    cases = manifest["case_ids"]
    if len(cases) != len(set(cases)) or not cases:
        raise RuntimeError("retry list must be nonempty and unique")
    for case_id in cases:
        if not (args.root / "results" / f"{case_id}.json").exists():
            raise RuntimeError(f"first attempt missing: {case_id}")
    progress_path = args.manifest.with_name(args.manifest.stem + "_progress.json")
    progress = {"manifest": str(args.manifest), "code_commit": args.code_commit,
                "queue_root": str(args.queue_root), "gpu": args.gpu,
                "items": []}
    for case_id in cases:
        parent = args.root / "retry_attempts" / case_id
        previous = sorted(parent.glob("attempt-*"), key=lambda p: int(p.name.split("-")[-1]))
        passed = [p for p in previous if (p / "retry_metadata.json").exists() and
                  json.loads((p / "retry_metadata.json").read_text()).get("independent_E_passed") is True]
        if passed:
            item = {"case_id": case_id, "status": "already_validated",
                    "accepted_attempt": passed[-1].name}
            progress["items"].append(item)
            if not args.dry_run:
                dump(progress_path, progress)
            continue
        attempt = max((int(p.name.split("-")[-1]) for p in previous), default=1) + 1
        item = {"case_id": case_id, "attempt": attempt, "status": "planned"}
        progress["items"].append(item)
        if args.dry_run:
            print(json.dumps(item), flush=True)
            continue
        dump(progress_path, progress)
        item["status"] = "running"
        item["started_at_unix"] = time.time()
        dump(progress_path, progress)
        command = [sys.executable, "-u", str(args.root / "harness" / "retry_one.py"),
                   "--case-id", case_id, "--attempt", str(attempt),
                   "--gpu", str(args.gpu), "--code-commit", args.code_commit,
                   "--queue-root", str(args.queue_root)]
        completed = subprocess.run(command, check=False)
        item["return_code"] = completed.returncode
        item["status"] = "terminal"
        item["finished_at_unix"] = time.time()
        retry_metadata = parent / f"attempt-{attempt}" / "retry_metadata.json"
        if retry_metadata.exists():
            record = json.loads(retry_metadata.read_text())
            item["program_status"] = record.get("program_status")
            item["independent_E_passed"] = record.get("independent_E_passed")
        dump(progress_path, progress)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
