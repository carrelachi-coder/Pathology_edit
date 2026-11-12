#!/usr/bin/env python3
"""Freeze technical/development exclusions for the reserved formal R3 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build(candidate: dict, technical: dict, split: dict) -> dict:
    if candidate.get("schema_version") != "reserved-four-target-candidate-freeze-v1":
        raise ValueError("unsupported reserved candidate manifest")
    selected = candidate.get("selected_candidates")
    if not isinstance(selected, list) or not selected:
        raise ValueError("reserved candidate manifest is empty")
    development = split.get("development_patient_aliases")
    reserved = split.get("reserved_evaluation_patient_aliases")
    if not isinstance(development, list) or not isinstance(reserved, list):
        raise ValueError("patient split lacks development/reserved assignments")
    if set(development) & set(reserved):
        raise ValueError("development and reserved patient assignments overlap")
    technical_patients = set(technical.get("excluded_patient_ids", []))
    technical_references = set(technical.get("excluded_reference_ids", []))
    excluded_patients = technical_patients | set(development)
    selected_patients = {str(row.get("patient_id", "")) for row in selected}
    selected_references = {str(row.get("reference_id", "")) for row in selected}
    if "" in selected_patients or "" in selected_references:
        raise ValueError("reserved candidate identity is incomplete")
    if not selected_patients <= set(reserved):
        raise ValueError("formal candidate lies outside the frozen reserved partition")
    patient_overlap = sorted(selected_patients & excluded_patients)
    reference_overlap = sorted(selected_references & technical_references)
    if patient_overlap or reference_overlap:
        raise ValueError("formal candidates overlap a technical/development exclusion")
    return {
        "schema_version": "formal-r3-exclusion-audit-v1",
        "scope": "exclude all technical and topology-method-development patients before formal R3",
        "excluded_patient_ids": sorted(excluded_patients),
        "excluded_reference_ids": sorted(technical_references),
        "technical_patient_count": len(technical_patients),
        "development_patient_count": len(development),
        "reserved_candidate_patient_count": len(selected_patients),
        "reserved_candidate_reference_count": len(selected_references),
        "selected_patient_overlap": patient_overlap,
        "selected_reference_overlap": reference_overlap,
        "formal_candidates_all_in_reserved_partition": True,
        "formal_generation_launched": False,
        "qualification_updated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--technical-exclusion", type=Path, required=True)
    parser.add_argument("--patient-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve prior audit; use a new output path")
    payload = build(
        json.loads(args.candidate_manifest.read_text()),
        json.loads(args.technical_exclusion.read_text()),
        json.loads(args.patient_split.read_text()),
    )
    payload["inputs"] = {
        "candidate_manifest": identity(args.candidate_manifest),
        "technical_exclusion": identity(args.technical_exclusion),
        "patient_split": identity(args.patient_split),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "excluded_patients": len(payload["excluded_patient_ids"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
