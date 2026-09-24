"""Freeze A2 radial-depletion gate cases for numbered same-case retries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2_de_six_v4_failure_audit import audit
from r2_de_six_v4_retry_infrastructure import digest, run


def prepare(root: Path, output: Path) -> None:
    protocol = json.loads((root / "protocol.json").read_text())
    if digest(root / "frozen_cohort.json") != protocol["cohort_sha256"]:
        raise RuntimeError("frozen cohort hash mismatch")
    records = audit(root)["records"]
    cases = []
    for record in records:
        if (
            record["dataset"] not in {"GLAS", "ORCA"}
            or record["failure_category"] != "cell_condition_gate_rejection"
            or not record["gate_failures"].get("local_population_density")
        ):
            continue
        case_id = record["case_id"]
        cases.append({
            "case_id": case_id,
            "initial_failure_category": record["failure_category"],
            "initial_failed_gate": "local_population_density",
            "initial_result_sha256": digest(root / "results" / f"{case_id}.json"),
        })
    if len(cases) != 8:
        raise RuntimeError(f"expected eight prespecified A2 gate cases; found {len(cases)}")
    manifest = {
        "schema_version": "r2-de-six-v4-A2-attempt2-manifest-v1",
        "frozen_cohort_sha256": protocol["cohort_sha256"],
        "initial_case_count": 222,
        "attempt": 2,
        "cases": cases,
    }
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise RuntimeError("existing A2 manifest differs")
    else:
        output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "cases": len(cases)}))


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
        if any(value is None for value in (
            args.code_root, args.code_commit, args.worker_index,
            args.worker_count, args.gpu,
        )):
            parser.error("run requires code root, commit, worker index/count, and GPU")
        run(args.root, args.manifest, args.code_root, args.code_commit,
            args.worker_index, args.worker_count, args.gpu)


if __name__ == "__main__":
    main()
