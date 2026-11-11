#!/usr/bin/env python3
"""Freeze four-target candidate cases from reserved topology receipts.

This is an accounting and case-construction step only.  A topology pass is a
necessary condition and is never promoted to a target approval by this tool.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


PASS_STATUS = "first_source_passed"
EXPECTED_TARGETS = {
    ("cohesive-boundary-expansion-v1", 0.19): "u1_moderate",
    ("cohesive-boundary-expansion-v1", 0.32): "u1_significant",
    ("generic-immune-infiltrate-increase-v1", 0.19): "u2_moderate",
    ("generic-immune-infiltrate-increase-v1", 0.32): "u2_significant",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_templates(path: Path) -> dict[str, dict[str, Any]]:
    raw = read_json(path)
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("case template file must contain exactly four targets")
    templates: dict[str, dict[str, Any]] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("case templates must be objects")
        budget = row.get("joint_area_budget")
        if not isinstance(budget, dict):
            raise ValueError("case template lacks joint_area_budget")
        key = (str(row.get("primitive_id")), float(budget.get("target_fraction")))
        label = EXPECTED_TARGETS.get(key)
        if label is None or label in templates:
            raise ValueError(f"unexpected or duplicate target template: {key}")
        expected_floor = 0.14 if key[1] == 0.19 else 0.27
        if float(budget.get("tissue_min_fraction")) != expected_floor:
            raise ValueError(f"target {label} changed the frozen tissue floor")
        if budget.get("capacity_floor_policy") != "strict":
            raise ValueError(f"target {label} does not use the strict floor policy")
        templates[label] = row
    if set(templates) != set(EXPECTED_TARGETS.values()):
        raise ValueError("the four frozen target identities are incomplete")
    return templates


def collect_receipts(roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    passes: list[dict[str, Any]] = []
    for root_order, root in enumerate(roots):
        if not root.is_dir():
            raise ValueError(f"qualification root does not exist: {root}")
        receipt_paths = sorted(root.glob("patient_*/patient_receipt.json"))
        if not receipt_paths:
            raise ValueError(f"qualification root has no patient receipts: {root}")
        for receipt_path in receipt_paths:
            receipt = read_json(receipt_path)
            row = {
                "root_order": root_order,
                "root": str(root.resolve()),
                "receipt": file_identity(receipt_path),
                "patient_id": str(receipt.get("patient_id")),
                "status": str(receipt.get("status")),
                "selected_reference_id": receipt.get("selected_reference_id"),
                "attempted_sources": int(receipt.get("attempted_sources", 0)),
            }
            history.append(row)
            if row["status"] == PASS_STATUS:
                if not row["selected_reference_id"]:
                    raise ValueError(f"passing receipt lacks selected source: {receipt_path}")
                passes.append(row)
    return history, passes


def build_candidates(
    *,
    freeze: dict[str, Any],
    templates: dict[str, dict[str, Any]],
    pass_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if freeze.get("partition") != "reserved_evaluation":
        raise ValueError("only the reserved_evaluation partition can be frozen")
    sources = freeze.get("candidate_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source freeze has no candidate_sources")
    by_reference: dict[str, dict[str, Any]] = {}
    patient_order = {
        str(patient): index for index, patient in enumerate(freeze.get("patient_aliases", []))
    }
    for source in sources:
        reference = str(source.get("reference_id"))
        if not reference or reference in by_reference:
            raise ValueError(f"duplicate or empty reference_id: {reference}")
        patient = str(source.get("patient_id"))
        if patient not in patient_order:
            raise ValueError(f"source patient absent from patient_aliases: {patient}")
        if source.get("method_partition") != "reserved_evaluation":
            raise ValueError(f"source is not reserved evaluation: {reference}")
        by_reference[reference] = source

    selected_by_patient: dict[str, dict[str, Any]] = {}
    for row in pass_rows:
        patient = row["patient_id"]
        reference = str(row["selected_reference_id"])
        source = by_reference.get(reference)
        if source is None or str(source.get("patient_id")) != patient:
            raise ValueError(f"receipt/source identity mismatch: {patient} / {reference}")
        previous = selected_by_patient.get(patient)
        if previous is not None and previous["selected_reference_id"] != reference:
            raise ValueError(f"patient has conflicting passing sources: {patient}")
        selected_by_patient[patient] = row

    selected_rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for patient in sorted(selected_by_patient, key=patient_order.__getitem__):
        pass_row = selected_by_patient[patient]
        reference = str(pass_row["selected_reference_id"])
        source = by_reference[reference]
        inputs = source.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError(f"source inputs are missing: {reference}")
        input_keys = {
            "reference_image": "source_image_uri",
            "reference_tissue_mask": "source_tissue_mask_uri",
            "reference_nuclei_mask": "source_nuclei_mask_uri",
        }
        normalized_inputs: dict[str, dict[str, Any]] = {}
        for source_key in input_keys:
            asset = inputs.get(source_key)
            if not isinstance(asset, dict) or not asset.get("path") or not asset.get("sha256"):
                raise ValueError(f"source input is incomplete: {reference} / {source_key}")
            normalized_inputs[source_key] = {
                "path": str(asset["path"]),
                "sha256": str(asset["sha256"]),
            }
        case_start = len(cases)
        for label in ("u1_moderate", "u1_significant", "u2_moderate", "u2_significant"):
            case = deepcopy(templates[label])
            case["case_id"] = f"{reference}__{label}"
            for source_key, case_key in input_keys.items():
                case[case_key] = normalized_inputs[source_key]["path"]
            provenance = dict(case["provenance"])
            provenance.update(
                source_image_sha256=normalized_inputs["reference_image"]["sha256"],
                source_tissue_mask_sha256=normalized_inputs["reference_tissue_mask"]["sha256"],
                source_nuclei_mask_sha256=normalized_inputs["reference_nuclei_mask"]["sha256"],
                original_label_map_digest=normalized_inputs["reference_tissue_mask"]["sha256"],
                original_instance_mask_digest=normalized_inputs["reference_nuclei_mask"]["sha256"],
                reserved_patient_id=patient,
                reserved_wsi_id=str(source.get("wsi_id")),
                reserved_reference_id=reference,
                reserved_patient_source_rank=int(source.get("patient_source_rank")),
                topology_receipt_sha256=pass_row["receipt"]["sha256"],
                topology_method_partition="reserved_evaluation",
            )
            case["provenance"] = provenance
            cases.append(case)
        selected_rows.append(
            {
                "patient_order": patient_order[patient],
                "patient_id": patient,
                "wsi_id": str(source.get("wsi_id")),
                "reference_id": reference,
                "patient_source_rank": int(source.get("patient_source_rank")),
                "qualification_receipt": pass_row["receipt"],
                "case_indices": list(range(case_start, case_start + 4)),
                "status": "topology_pass_candidate_only",
            }
        )
    return selected_rows, cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--case-templates", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; candidate freezes are append-only")
    freeze = read_json(args.source_freeze)
    templates = load_templates(args.case_templates)
    history, passes = collect_receipts(args.qualification_root)
    selected, cases = build_candidates(
        freeze=freeze,
        templates=templates,
        pass_rows=passes,
    )
    args.output.mkdir(parents=True)
    cases_path = args.output / "four_target_candidate_cases.json"
    cases_path.write_text(json.dumps(cases, indent=2) + "\n")
    manifest = {
        "schema_version": "reserved-four-target-candidate-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "reserved topology-pass candidates only; no four-target approval, H&E generation, or formal cohort admission",
        "source_freeze": file_identity(args.source_freeze),
        "case_templates": file_identity(args.case_templates),
        "qualification_roots": [str(path.resolve()) for path in args.qualification_root],
        "receipt_history": history,
        "unique_reserved_topology_pass_patients": len(selected),
        "candidate_case_count": len(cases),
        "selection_rule": "unique reserved patient in frozen patient order; topology runner's first passing source in frozen patient-local order",
        "targets_per_patient": [
            "u1_moderate",
            "u1_significant",
            "u2_moderate",
            "u2_significant",
        ],
        "nominal_joint_fractions": [0.19, 0.32],
        "tissue_hard_floors": [0.14, 0.27],
        "selected_candidates": selected,
        "cases": file_identity(cases_path),
        "formal_qualification_updated": False,
    }
    (args.output / "candidate_freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"patients": len(selected), "cases": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
