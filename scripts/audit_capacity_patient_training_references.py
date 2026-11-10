#!/usr/bin/env python3
"""Read-only patient-level references for the frozen BCSS capacity universe.

This checks retained manifest references, not checkpoint training lineage or
clinical independence. JSON is emitted to stdout; no input asset is changed.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


PATIENT = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def index_strings(value):
    names, patients = set(), set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str):
            normal = item.replace("\\", "/")
            if normal.lower().endswith(IMAGE_SUFFIXES):
                names.add(normal.rsplit("/", 1)[-1])
            patients.update(PATIENT.findall(item))
    return names, patients


def build(base, training_paths):
    subset_specs = [
        ("capacity_qualified_subset.json", "7bc6d10ea1c9ccf6570ea21a4dda5dc2c612ac23ddb57451c372207369e0bfc9"),
        ("supplemental_capacity_qualified_subset_v1.json", "7429fed6f03e80c9fbae7c5f0cb363e3c913a6627d2062b27729b3685577727b"),
        ("supplemental_capacity_v2_wave00_subset.json", "3cc20c33d22420a8fa502f3d863b280ef93d2dd567d4fe562ad4b4526ef57867"),
        ("supplemental_capacity_v2_wave01_subset.json", "2a9135f23823b40cffbf2f217ee632f3f590b00f531de094cf04221973f63595"),
        ("supplemental_capacity_v2_wave02_subset.json", "4b4acdc2ecd9252ad96378e638643368ed97874627dad65a4dd52ebcd0ae7305"),
        ("native_crop_recovery_20261103_v1/qualification_v1/capacity_passing_sources.json", "5837eca5a8305aecf77ffe0bf87a091d33a21c87b83c284688ff8a2be66bd0d1"),
    ]
    rows = []
    subset_evidence = []
    for name, expected in subset_specs:
        path = base / name
        actual = sha(path)
        if actual != expected:
            raise ValueError("source subset SHA mismatch: " + name)
        subset = json.loads(path.read_bytes())["candidate_sources"]
        subset_evidence.append({"path": name, "sha256": actual, "sources": len(subset)})
        rows.extend(subset)
    if len(rows) != 538 or len({row["reference_id"] for row in rows}) != 538:
        raise ValueError("source universe count or uniqueness mismatch")

    exclusion = base / "technical_patient_exclusion_audit_v2.json"
    exclusion_sha = sha(exclusion)
    if exclusion_sha != "e99577970b5e4c0a09fce4768b2c6e1368b2d0454cc29fa90662848e87f30126":
        raise ValueError("technical exclusion SHA mismatch")
    excluded = set(json.loads(exclusion.read_bytes())["excluded_patient_ids"])
    grouped = defaultdict(list)
    for row in rows:
        if row["patient_id"] not in excluded:
            grouped[row["patient_id"]].append(row)
    if len(grouped) != 84 or sum(map(len, grouped.values())) != 527:
        raise ValueError("patient/source denominator mismatch")
    if any(row["wsi_id"] != patient for patient, members in grouped.items() for row in members):
        raise ValueError("unexpected WSI group alias")

    cross_train = Path(training_paths["cross_train"])
    cross_val = Path(training_paths["cross_val"])
    coarse = Path(training_paths["segmentator_coarse"])
    fine = Path(training_paths["segmentator_fine"])
    inputs = {
        "cross_train": (cross_train, None),
        "cross_val": (cross_val, None),
        "segmentator_coarse_train": (coarse, "train"),
        "segmentator_coarse_val": (coarse, "val"),
        "segmentator_fine_train": (fine, "train"),
        "segmentator_fine_val": (fine, "val"),
        "segmentator_fine_test": (fine, "test"),
    }
    indices, manifest_evidence = {}, {}
    grouped_inputs = defaultdict(list)
    for key, (path, split) in inputs.items():
        grouped_inputs[path].append((key, split))
    for path, keys in grouped_inputs.items():
        file_sha = sha(path)
        loaded = json.loads(path.read_bytes())
        for key, split in keys:
            payload = loaded[split] if split else loaded
            names, patients = index_strings(payload)
            indices[key] = names, patients
            manifest_evidence[key] = {
                "path": str(path), "sha256": file_sha, "split": split,
                "filename_count": len(names), "patient_group_count": len(patients),
            }
            print("indexed " + key, file=sys.stderr, flush=True)
        del loaded

    patient_records = {}
    for patient in sorted(grouped):
        members = grouped[patient]
        filenames = [Path(row["inputs"]["reference_image"]["path"]).name for row in members]
        references = {}
        for key, (names, patients) in indices.items():
            hits = sorted(set(filenames) & names)
            references[key] = {
                "patient_group_referenced": patient in patients,
                "exact_candidate_filename_references": len(hits),
                "reference_status": "confirmed_reference" if patient in patients or hits else "checked_no_reference_found",
            }
        patient_records[patient] = {
            "candidate_source_count": len(members),
            "candidate_reference_ids": sorted(row["reference_id"] for row in members),
            "candidate_source_image_sha256": {row["reference_id"]: row["inputs"]["reference_image"]["sha256"] for row in members},
            "manifest_references": references,
            "flux_patient_training_overlap": "unknown",
            "cellvit_patient_training_overlap": "unknown",
            "pix2pix_final_checkpoint_training_lineage": "unknown",
        }
    summary = {key: {
        "patients_with_group_reference": sum(r["manifest_references"][key]["patient_group_referenced"] for r in patient_records.values()),
        "patients_with_exact_candidate_filename_reference": sum(r["manifest_references"][key]["exact_candidate_filename_references"] > 0 for r in patient_records.values()),
    } for key in indices}
    pix_config = Path(training_paths["pix2pix_config"])
    result = {
        "kind": "capacity-patient-training-reference-audit-v1",
        "status": "read_only_manifest_reference_audit_no_independence_claim",
        "source_subsets": subset_evidence,
        "technical_exclusion_v2_sha256": exclusion_sha,
        "total_source_refs": len(rows),
        "nontechnical_source_refs": sum(map(len, grouped.values())),
        "nontechnical_patient_groups": len(grouped),
        "manifest_evidence": manifest_evidence,
        "pix2pix_config": {"path": str(pix_config), "sha256": sha(pix_config), "lineage_status": "checkpoint_training_lineage_unverified"},
        "patient_summary": summary,
        "patients": patient_records,
        "s07_patient_id": "TCGA-E2-A1LL",
        "s07_in_old_technical_exclusion": "TCGA-E2-A1LL" in excluded,
        "interpretation": "Manifest references indicate possible exposure; an absent reference is not proof of independence or complete checkpoint lineage. These are capacity-positive patients, not topology or four-target approvals. Patient/WSI aliases are not verified complete slide barcodes.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--cross-train", type=Path, required=True)
    parser.add_argument("--cross-val", type=Path, required=True)
    parser.add_argument("--segmentator-coarse", type=Path, required=True)
    parser.add_argument("--segmentator-fine", type=Path, required=True)
    parser.add_argument("--pix2pix-config", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "cross_train": args.cross_train, "cross_val": args.cross_val,
        "segmentator_coarse": args.segmentator_coarse,
        "segmentator_fine": args.segmentator_fine,
        "pix2pix_config": args.pix2pix_config,
    }
    print(json.dumps(build(args.base, paths), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
