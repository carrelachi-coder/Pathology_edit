#!/usr/bin/env python3
"""Build one immutable patient-partition source freeze in census order."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_SPLIT_SHA256 = (
    "1f44bfd0598629f4ea48f8c8af73956c9c45ced58af5249b86c7b00520b65af3"
)
SUBSETS = (
    ("capacity_qualified_subset.json", "7bc6d10ea1c9ccf6570ea21a4dda5dc2c612ac23ddb57451c372207369e0bfc9"),
    ("supplemental_capacity_qualified_subset_v1.json", "7429fed6f03e80c9fbae7c5f0cb363e3c913a6627d2062b27729b3685577727b"),
    ("supplemental_capacity_v2_wave00_subset.json", "3cc20c33d22420a8fa502f3d863b280ef93d2dd567d4fe562ad4b4526ef57867"),
    ("supplemental_capacity_v2_wave01_subset.json", "2a9135f23823b40cffbf2f217ee632f3f590b00f531de094cf04221973f63595"),
    ("supplemental_capacity_v2_wave02_subset.json", "4b4acdc2ecd9252ad96378e638643368ed97874627dad65a4dd52ebcd0ae7305"),
    ("native_crop_recovery_20261103_v1/qualification_v1/capacity_passing_sources.json", "5837eca5a8305aecf77ffe0bf87a091d33a21c87b83c284688ff8a2be66bd0d1"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=("development", "reserved_evaluation"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.split) != EXPECTED_SPLIT_SHA256:
        raise ValueError("patient split identity differs from the frozen proposal")
    split = json.loads(args.split.read_text())
    patient_key = (
        "development_patient_aliases"
        if args.partition == "development"
        else "reserved_evaluation_patient_aliases"
    )
    patients = tuple(str(value) for value in split[patient_key])
    if len(patients) != len(set(patients)):
        raise ValueError("patient partition contains duplicates")
    allowed = set(patients)
    rows = []
    seen_references: set[str] = set()
    rank_by_patient: Counter[str] = Counter()
    inputs = []
    for subset_index, (relative, expected_sha) in enumerate(SUBSETS):
        path = args.asset_root / relative
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"source subset identity differs: {relative}")
        payload = json.loads(path.read_text())
        candidates = payload["candidate_sources"]
        inputs.append(
            {
                "path": str(path),
                "sha256": actual_sha,
                "candidate_source_count": len(candidates),
            }
        )
        for subset_row_index, source in enumerate(candidates):
            patient = str(source["patient_id"])
            if patient not in allowed:
                continue
            reference = str(source["reference_id"])
            if reference in seen_references:
                raise ValueError(f"duplicate source reference: {reference}")
            seen_references.add(reference)
            patient_rank = rank_by_patient[patient]
            rank_by_patient[patient] += 1
            rows.append(
                {
                    **source,
                    "method_partition": args.partition,
                    "partition_patient_order": patients.index(patient),
                    "patient_source_rank": patient_rank,
                    "source_subset_index": subset_index,
                    "source_subset_row_index": subset_row_index,
                }
            )
    missing = [patient for patient in patients if rank_by_patient[patient] == 0]
    if missing:
        raise ValueError(f"partition patients without sources: {missing}")
    rows.sort(
        key=lambda row: (
            row["partition_patient_order"],
            row["patient_source_rank"],
        )
    )
    expected_counts = {
        "development": (16, 109),
        "reserved_evaluation": (68, 418),
    }
    if (len(patients), len(rows)) != expected_counts[args.partition]:
        raise ValueError("partition patient/source count differs from frozen audit")
    output = {
        "schema_version": "topology-method-partition-source-freeze-v1",
        "partition": args.partition,
        "split": {"path": str(args.split), "sha256": EXPECTED_SPLIT_SHA256},
        "source_subsets": inputs,
        "patient_count": len(patients),
        "source_count": len(rows),
        "patient_aliases": list(patients),
        "ordering_rule": "frozen patient order, then frozen subset order, then source row order",
        "candidate_sources": rows,
        "scope": "source-only topology method evaluation; no target approval or H&E generation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
