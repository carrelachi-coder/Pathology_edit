#!/usr/bin/env python3
"""Verify every frozen source asset in the reserved formal R3 candidate manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


INPUTS = ("reference_image", "reference_tissue_mask", "reference_nuclei_mask")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(manifest: dict) -> dict:
    if manifest.get("schema_version") != "reserved-four-target-candidate-freeze-v1":
        raise ValueError("unsupported reserved candidate manifest")
    rows = manifest.get("selected_candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("reserved candidate manifest is empty")
    patients = set()
    references = set()
    verified = []
    for row in rows:
        patient = str(row.get("patient_id", ""))
        reference = str(row.get("reference_id", ""))
        if not patient or not reference or patient in patients or reference in references:
            raise ValueError("candidate patient/reference identity is empty or duplicated")
        patients.add(patient)
        references.add(reference)
        inputs = row.get("verified_source_inputs", {})
        row_assets = {}
        for name in INPUTS:
            record = inputs.get(name, {})
            path = Path(str(record.get("path", "")))
            if not path.is_file():
                raise ValueError(f"frozen source asset is missing: {path}")
            actual = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if actual["bytes"] != record.get("bytes") or actual["sha256"] != record.get("sha256"):
                raise ValueError(f"frozen source asset content drift: {path}")
            row_assets[name] = actual
        match = re.search(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", Path(row_assets["reference_image"]["path"]).name)
        if not match or match.group() != patient:
            raise ValueError("candidate patient identity differs from the source filename")
        verified.append({"patient_id": patient, "reference_id": reference, "assets": row_assets})
    if manifest.get("unique_reserved_topology_pass_patients") != len(rows):
        raise ValueError("candidate count differs from frozen topology-pass count")
    if manifest.get("candidate_case_count") != len(rows) * 4:
        raise ValueError("four-target case count differs from candidate count")
    return {
        "schema_version": "formal-r3-candidate-asset-preflight-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "candidate_patients": len(patients),
        "candidate_references": len(references),
        "verified_assets": len(verified) * len(INPUTS),
        "candidate_cases": len(rows) * 4,
        "records": verified,
        "formal_generation_launched": False,
        "qualification_updated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve prior preflight; use a new output path")
    result = verify(json.loads(args.candidate_manifest.read_text()))
    result["candidate_manifest"] = {
        "path": str(args.candidate_manifest.resolve()),
        "bytes": args.candidate_manifest.stat().st_size,
        "sha256": sha256_file(args.candidate_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("candidate_patients", "candidate_cases", "verified_assets")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
