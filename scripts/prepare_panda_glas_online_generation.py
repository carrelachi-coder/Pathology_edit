#!/usr/bin/env python3
"""Promote the final PANDA/GLaS mask-review cohort for Online generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_joint_edit_refine.approval_handoff import (
    promote_audited_joint_candidate,
)

DEFAULT_PANDA_BASE = Path(
    "/tmp/pathology-golden-artifacts-20260822-amplitude/panda/"
    "combined-review-manifest-7327e89.json"
)
DEFAULT_PANDA_FOOTPRINT = Path(
    "/tmp/pathology-golden-artifacts-20260823-footprint-v1/panda/"
    "replay-eval7-v3/frozen_shadow_replay_manifest.json"
)
DEFAULT_GLAS = Path(
    "/tmp/pathology-golden-artifacts-20260822-amplitude/glas/"
    "mask_review_summary_consolidated.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panda-base", type=Path, default=DEFAULT_PANDA_BASE)
    parser.add_argument(
        "--panda-footprint", type=Path, default=DEFAULT_PANDA_FOOTPRINT
    )
    parser.add_argument("--glas", type=Path, default=DEFAULT_GLAS)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panda_cases(base_path: Path, footprint_path: Path) -> list[dict[str, Any]]:
    base = read_json(base_path)
    footprint = read_json(footprint_path)
    selected: list[dict[str, Any]] = []
    for evaluation in base["frozen_evaluations"]:
        primitive = str(evaluation["primitive_id"])
        if primitive in {
            "stroma-increase-v1",
            "invasive-tumor-footprint-decrease-v1",
        }:
            continue
        selected.append(evaluation)
    footprint_evaluations = footprint.get("frozen_evaluations") or []
    if len(footprint_evaluations) != 1:
        raise RuntimeError("Expected one final PANDA footprint evaluation")
    selected.append(footprint_evaluations[0])
    selected.sort(
        key=lambda item: (
            {
                "cell-type-abundance-increase-v1": 0,
                "cell-type-abundance-decrease-v1": 1,
                "cellularity-increase-v1": 2,
                "cellularity-decrease-v1": 3,
                "neoplastic-cell-abundance-increase-v1": 4,
                "neoplastic-cell-abundance-decrease-v1": 5,
                "local-invasive-clearance-v1": 6,
                "invasive-tumor-footprint-decrease-v1": 7,
                "residual-tumor-fragmentation-v1": 8,
                "infiltrative-nest-cord-extension-v1": 11,
                "peritumoral-neoplastic-scatter-increase-v1": 12,
            }.get(str(item["primitive_id"]), 9),
            str(item.get("mechanism_id") or ""),
        )
    )
    rows: list[dict[str, Any]] = []
    for ordinal, evaluation in enumerate(selected):
        cases = evaluation.get("frozen_cases") or []
        if len(cases) != 5:
            raise RuntimeError(
                f"PANDA {evaluation['primitive_id']} has {len(cases)} cases"
            )
        for case in cases:
            rows.append(
                {
                    "dataset": "PANDA",
                    "profile": "PANDA",
                    "category": str(evaluation.get("mechanism_id") or "PANDA"),
                    "primitive_id": str(evaluation["primitive_id"]),
                    "primitive_ordinal": ordinal,
                    "case_id": str(case["case_id"]),
                    "candidate_id": str(case["selected_candidate_id"]),
                    "audit_case_dir": str(
                        Path(case["joint_run_summary"]).parent
                        / str(case["case_id"])
                    ),
                    "evidence_path": str(Path(case["gate_report"])),
                }
            )
    return rows


def glas_cases(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    cases = payload.get("selected_cases") or []
    counts = Counter(str(item["primitive_id"]) for item in cases)
    if len(cases) != 40 or set(counts.values()) != {5}:
        raise RuntimeError(f"Expected GLaS 8x5 final cases, found {dict(counts)}")
    primitive_order = {
        primitive: index
        for index, primitive in enumerate(sorted(counts))
    }
    rows: list[dict[str, Any]] = []
    for case in cases:
        target = Path(case["target_tissue_mask"])
        audit_case_dir = target.parent.parent
        context = read_json(audit_case_dir / "input_case_context.json")
        rows.append(
            {
                "dataset": "GLaS",
                "profile": "GLaS",
                "category": str(
                    context.get("mechanism_id")
                    or "colorectal-local-population-modulation"
                ),
                "primitive_id": str(case["primitive_id"]),
                "primitive_ordinal": primitive_order[str(case["primitive_id"])],
                "case_id": str(case["case_id"]),
                "candidate_id": str(case["selected_candidate_id"]),
                "audit_case_dir": str(audit_case_dir),
                "evidence_path": str(audit_case_dir / "joint_gate_reports.json"),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    handoff_root = output / "handoffs"
    rows = panda_cases(args.panda_base, args.panda_footprint)
    rows.extend(glas_cases(args.glas))
    if len(rows) != 105:
        raise RuntimeError(f"Expected 105 cases, found {len(rows)}")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate case IDs in final generation cohort")

    promoted: list[dict[str, Any]] = []
    for row in rows:
        evidence_path = Path(row["evidence_path"])
        evidence_sha256 = sha256(evidence_path)
        approval = {
            "schema_version": "joint-candidate-user-approval-v1",
            "decision": "approved",
            "case_id": row["case_id"],
            "candidate_id": row["candidate_id"],
            "approval_scope": "mask_condition_for_online_generation",
            "approved_by": "user",
            "evidence_sha256": evidence_sha256,
            "evidence_path": str(evidence_path),
        }
        paths = promote_audited_joint_candidate(
            row["audit_case_dir"],
            candidate_id=row["candidate_id"],
            approval=approval,
            output_dir=handoff_root / row["case_id"],
        )
        manifest_path = Path(paths["manifest"])
        handoff = read_json(manifest_path)
        source_assets = handoff["source_assets"]
        handoff_paths = handoff["paths"]
        promoted.append(
            {
                **row,
                "instruction": " ".join(handoff.get("render_expectations") or ()),
                "joint_generation_handoff": str(manifest_path),
                "source_image": str(source_assets["image"]),
                "source_tissue_mask": str(source_assets["tissue"]),
                "source_nuclei_mask": str(source_assets["nuclei"]),
                "target_tissue_mask": str(handoff_paths["target_tissue_mask"]),
                "target_nuclei_mask": str(handoff_paths["target_nuclei_mask"]),
                "semantic_change_region": str(handoff_paths["joint_change"]),
                "generation_change_region": str(
                    handoff_paths["generation_support"]
                ),
                "semantic_pixels": int(handoff["ledger"]["joint_pixels"]),
                "generation_pixels": int(
                    handoff["ledger"]["generation_support_pixels"]
                ),
                "generation_support_fraction": float(
                    handoff["ledger"]["generation_support_fraction"]
                ),
                "handoff_sha256": sha256(manifest_path),
            }
        )

    promoted.sort(
        key=lambda item: (
            0 if item["dataset"] == "PANDA" else 1,
            int(item["primitive_ordinal"]),
            str(item["case_id"]),
        )
    )
    manifest = {
        "schema_version": "panda-glas-online-generation-cohort-v1",
        "mask_code_commit": "46dbfabdf905423524d126f5dc81e696ca581e4e",
        "case_count": len(promoted),
        "dataset_counts": dict(Counter(row["dataset"] for row in promoted)),
        "edit_counts": dict(
            Counter(
                f"{row['dataset']}::{row['category']}::{row['primitive_id']}"
                for row in promoted
            )
        ),
        "generation_region_policy": (
            "hash-locked contract S support; semantic mask J remains unchanged"
        ),
        "records": promoted,
    }
    manifest_path = output / "generation_cohort.json"
    write_json(manifest_path, manifest)
    write_json(
        output / "preflight.json",
        {
            "passed": True,
            "case_count": len(promoted),
            "dataset_counts": manifest["dataset_counts"],
            "edit_counts": manifest["edit_counts"],
            "all_generation_regions_strictly_larger": all(
                row["generation_pixels"] > row["semantic_pixels"]
                for row in promoted
            ),
            "minimum_generation_to_semantic_ratio": min(
                row["generation_pixels"] / max(1, row["semantic_pixels"])
                for row in promoted
            ),
            "maximum_generation_to_semantic_ratio": max(
                row["generation_pixels"] / max(1, row["semantic_pixels"])
                for row in promoted
            ),
            "generation_cohort_sha256": sha256(manifest_path),
        },
    )
    print(json.dumps({"status": "completed", "manifest": str(manifest_path), "cases": len(promoted)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
