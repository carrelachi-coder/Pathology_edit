#!/usr/bin/env python3
"""Run a frozen G2-v2 shadow one case per bounded subprocess."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.skills.repository import JointSkillRepository

SHADOW_RUN_SCHEMA = "g2-v2-bounded-joint-shadow-run-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--probnet-checkpoint", required=True)
    parser.add_argument(
        "--nuclei-library-root",
        required=True,
        help="Directory containing BCSS/GlaS/PANDA/IGNITE/PUMA/ORCA libraries",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")

    manifest = Path(args.manifest)
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("shadow manifest must contain a non-empty JSON list")
    if args.case_id:
        requested = set(args.case_id)
        records = [item for item in records if item.get("case_id") in requested]
        missing = requested.difference(item.get("case_id") for item in records)
        if missing:
            raise ValueError(f"case IDs absent from shadow manifest: {sorted(missing)}")

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "bounded_execution_ledger.json"
    existing = (
        json.loads(ledger_path.read_text(encoding="utf-8"))
        if args.resume and ledger_path.is_file()
        else []
    )
    if not isinstance(existing, list):
        raise ValueError("existing bounded execution ledger is malformed")
    by_id = {str(item["case_id"]): item for item in existing}
    repository = JointSkillRepository()
    checkpoint = Path(args.probnet_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    library_root = Path(args.nuclei_library_root)

    for record in records:
        case_id = str(record["case_id"])
        if args.resume and case_id in by_id:
            continue
        population = repository.cell_population_profiles[
            str(record["cell_population_profile_id"])
        ]
        dataset = population.probnet_dataset_name
        library = library_root / dataset
        if not (library / "statistics.json").is_file():
            raise FileNotFoundError(
                f"mature nucleus library is unavailable for {dataset}: {library}"
            )
        case_root = root / "cases" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "phase3_joint_edit_refine.cli",
            "--manifest",
            str(manifest),
            "--output-root",
            str(case_root),
            "--case-id",
            case_id,
            "--agent-mode",
            "offline",
            "--semantic-parser",
            "prebound",
            "--cell-executor",
            "mature",
            "--probnet-checkpoint",
            str(checkpoint),
            "--nuclei-instance-library",
            str(library),
            "--probnet-dataset",
            dataset,
            "--device",
            args.device,
        ]
        started = time.monotonic()
        timed_out = False
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=args.timeout_seconds)
            return_code = int(process.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            return_code = 124
        duration = time.monotonic() - started
        (case_root / "bounded_stdout.log").write_text(stdout, encoding="utf-8")
        (case_root / "bounded_stderr.log").write_text(stderr, encoding="utf-8")
        summary_path = case_root / "joint_run_summary.json"
        workflow_status = None
        abstain_reasons = []
        selected_candidate_id = None
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary, list) and len(summary) == 1:
                workflow_status = summary[0].get("status")
                abstain_reasons = list(summary[0].get("abstain_reasons") or ())
                selected_candidate_id = summary[0].get("selected_candidate_id")
        if timed_out:
            bounded_status = "execution_timeout"
        elif workflow_status is not None:
            bounded_status = str(workflow_status)
        else:
            bounded_status = "execution_error"
        item = {
            "schema_version": SHADOW_RUN_SCHEMA,
            "case_id": case_id,
            "organ": record.get("pathology_domain_id"),
            "primitive_id": record.get("primitive_id"),
            "mechanism_id": (record.get("provenance") or {}).get("joint_mechanism_id"),
            "probnet_dataset": dataset,
            "bounded_status": bounded_status,
            "workflow_status": workflow_status,
            "selected_candidate_id": selected_candidate_id,
            "abstain_reasons": abstain_reasons,
            "return_code": return_code,
            "timed_out": timed_out,
            "timeout_seconds": args.timeout_seconds,
            "wall_time_seconds": round(duration, 3),
            "case_output_root": str(case_root),
            "joint_run_summary": str(summary_path) if summary_path.is_file() else None,
            "llm_api_used": False,
        }
        by_id[case_id] = item
        _write_ledger(ledger_path, [by_id[key] for key in sorted(by_id)])
        print(json.dumps(item, ensure_ascii=False, sort_keys=True), flush=True)

    ledger = [by_id[key] for key in sorted(by_id)]
    summary = {
        "schema_version": SHADOW_RUN_SCHEMA,
        "manifest": str(manifest),
        "case_count": len(ledger),
        "status_counts": dict(
            sorted(Counter(item["bounded_status"] for item in ledger).items())
        ),
        "timeout_seconds": args.timeout_seconds,
        "ledger": str(ledger_path),
        "llm_api_used": False,
    }
    summary_path = root / "bounded_execution_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


def _write_ledger(path: Path, payload: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
if __name__ == "__main__":
    raise SystemExit(main())
