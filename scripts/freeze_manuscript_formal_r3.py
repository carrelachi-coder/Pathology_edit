#!/usr/bin/env python3
"""Freeze the actual fully reviewed reserved patients as the formal R3 cohort."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from controlnet_train.inference.model_paths import DEFAULT_CELLVIT_ROOT
from controlnet_train.pix2pix_transfer.inference import CrossLowStainProtectionConfig
from phase3_joint_edit_refine.generator_adapter import build_frozen_generator_inputs
from phase3_joint_edit_refine.independent_review import validate_independent_review
from phase3_mask_edit.benchmark.experiment_identity import asset_identity, sha256_file
from phase3_mask_edit.benchmark.manuscript_contract import coarse_tissue
from phase3_mask_edit.benchmark.manuscript_measurements import CELL_POINT_POLICY
from scripts.freeze_manuscript_joint_technical import distribution_versions, segmentator_environment
from scripts.run_manuscript_global_cohort import validate_cohort


TARGETS = ("u1_moderate", "u1_significant", "u2_moderate", "u2_significant")


def frozen(path: str | Path) -> dict:
    return {"status": "frozen", **asset_identity(path)}


def verify_identity(record: dict) -> Path:
    path = Path(str(record.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError(f"frozen input content drift: {path}")
    return path


def realized_target_record(ledger: dict) -> dict:
    required = ("joint_fraction", "tissue_fraction", "joint_pixels", "tissue_pixels")
    if any(ledger.get(key) is None for key in required):
        raise ValueError("approved handoff ledger lacks realized joint or tissue area")
    return {key: ledger[key] for key in required}


def load_selected(candidate_manifest: dict, materialization: dict) -> list[tuple[dict, dict]]:
    if candidate_manifest.get("schema_version") != "reserved-four-target-candidate-freeze-v1":
        raise ValueError("unsupported reserved candidate manifest")
    if materialization.get("schema_version") != "reserved-four-target-handoff-materialization-v1":
        raise ValueError("unsupported handoff materialization manifest")
    candidates = {row["reference_id"]: row for row in candidate_manifest.get("selected_candidates", [])}
    selected = []
    seen_patients = set()
    for patient in materialization.get("patients", []):
        reference = str(patient.get("reference_id", ""))
        candidate = candidates.get(reference)
        if candidate is None:
            raise ValueError("approved handoff patient is outside the reserved candidate freeze")
        if (
            patient.get("patient_id") != candidate.get("patient_id")
            or patient.get("patient_order") != candidate.get("patient_order")
        ):
            raise ValueError("approved handoff patient identity differs from the candidate freeze")
        if patient["patient_id"] in seen_patients:
            raise ValueError("formal cohort contains a duplicate patient")
        seen_patients.add(patient["patient_id"])
        cases = patient.get("cases", [])
        if [row.get("target") for row in cases] != list(TARGETS):
            raise ValueError("approved patient does not contain the frozen four-target order")
        selected.append((candidate, patient))
    if not selected:
        raise ValueError("no patient has four independently approved targets")
    return sorted(selected, key=lambda pair: int(pair[0]["patient_order"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "protocol-template",
        "candidate-manifest",
        "handoff-materialization",
        "overlap-audit",
        "exclusion-audit",
        "output",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("formal freeze requires a clean committed implementation")
    protocol = deepcopy(json.loads(args.protocol_template.read_text()))
    candidate_manifest = json.loads(args.candidate_manifest.read_text())
    materialization = json.loads(args.handoff_materialization.read_text())
    selected = load_selected(candidate_manifest, materialization)
    overlaps = json.loads(args.overlap_audit.read_text())["cases"]
    exclusion = json.loads(args.exclusion_audit.read_text())
    args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    approved_targets = {}
    roi_assets = {}
    mappings = {}
    selected_overlaps = {}
    realized = {}
    for candidate, patient in selected:
        reference = candidate["reference_id"]
        patient_id = candidate["patient_id"]
        if reference in exclusion.get("excluded_reference_ids", []) or patient_id in exclusion.get("excluded_patient_ids", []):
            raise ValueError("development or technical source/patient cannot enter formal R3")
        source_inputs = candidate.get("verified_source_inputs", {})
        source_paths = {name: str(verify_identity(source_inputs[name])) for name in (
            "reference_image", "reference_tissue_mask", "reference_nuclei_mask"
        )}
        overlap = overlaps.get(reference)
        if (
            not isinstance(overlap, dict)
            or overlap.get("status") != "completed"
            or not all(name in overlap for name in ("generator", "segmentator", "cellvit"))
        ):
            raise ValueError("formal source training-overlap audit is incomplete")
        selected_overlaps[reference] = overlap
        mappings[reference] = {"patient_id": patient_id, "wsi_id": candidate["wsi_id"]}
        source_coarse = coarse_tissue(source_paths["reference_tissue_mask"])
        source_dir = args.output / f"source_{int(candidate['patient_order']):03d}"
        source_dir.mkdir()
        handoffs = {}
        targets = {}
        joints = []
        realized[reference] = {}
        for state, case in zip(TARGETS, patient["cases"]):
            handoff_path = verify_identity(case["handoff_manifest"])
            inputs, _, handoff = build_frozen_generator_inputs(
                handoff_path, output_dir=source_dir / (state + "_validation_only")
            )
            operation, strength = state.split("_")
            expected_case_id = reference + "__" + state
            definition = protocol["operations"][operation]
            if (
                handoff.get("case_id") != expected_case_id
                or handoff.get("candidate_id") != case.get("candidate_id")
                or handoff.get("primitive_id") != definition["primitive_id"]
                or handoff.get("mechanism_id") != definition["mechanism_id"]
            ):
                raise ValueError("approved handoff operation, strength, source, or candidate differs")
            expected_budget = {
                **protocol["joint_budget_definitions"][strength],
                "basis": "whole_patch",
                "relative_tolerance": 0.02,
                "fallback_policy": "max_feasible_below_target",
                "capacity_floor_policy": "strict",
                "minimum_effective_fraction": 0.0,
            }
            if handoff["execution_contract"].get("joint_area_budget") != expected_budget:
                raise ValueError("approved handoff differs from the frozen nominal budget")
            review = handoff["provenance"].get("independent_critic_review") or {}
            actual_review = validate_independent_review(
                review.get("review_path"),
                audit_dir=handoff["provenance"]["source_audit_case_dir"],
                case_id=handoff["case_id"],
                candidate_id=handoff["candidate_id"],
            )
            if actual_review != review:
                raise ValueError("independent review content differs from the handoff")
            for input_field, handoff_field in (
                ("reference_image", "image"),
                ("reference_tissue_mask", "tissue"),
                ("reference_nuclei_mask", "nuclei"),
            ):
                if sha256_file(handoff["source_assets"][handoff_field]) != source_inputs[input_field]["sha256"]:
                    raise ValueError("approved targets do not share the frozen source")
                if sha256_file(getattr(inputs, input_field)) != source_inputs[input_field]["sha256"]:
                    raise ValueError("generator inputs differ from the frozen source")
            joint = np.asarray(Image.open(handoff["paths"]["joint_change"])) > 0
            if joint.shape != source_coarse.shape:
                raise ValueError("approved joint-change frame differs from the source")
            joints.append(joint)
            handoffs[state] = handoff
            targets[state] = {
                "handoff": frozen(handoff_path),
                "execution_budget": {
                    key: handoff["execution_contract"].get(key)
                    for key in ("joint_area_budget", "cell_count_extent_budget")
                },
                "target_sha256": [
                    sha256_file(handoff["paths"][key])
                    for key in ("target_tissue_mask", "target_nuclei_mask")
                ],
            }
            ledger = handoff["ledger"]
            realized[reference][state] = realized_target_record(ledger)
        regions = {
            "u1_roi": source_coarse > 0,
            "u2_roi": np.isin(source_coarse, (2, 4)),
            "protected_roi": (source_coarse > 0)
            & ~binary_dilation(np.logical_or.reduce(joints), iterations=8),
            "postprocess_eligibility": np.ones(source_coarse.shape, bool),
            "sham_generation_support": np.zeros(source_coarse.shape, bool),
        }
        files = {}
        for name, mask in regions.items():
            path = source_dir / (name + ".png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(path)
            files[name] = path
        common = {
            **source_paths,
            "reference_id": reference,
            "patient_id": patient_id,
            "wsi_id": candidate["wsi_id"],
            "profile": "BCSS",
            "prompt": protocol["sampling"]["prompt"],
            "generation_seed": protocol["sampling"]["seed"],
            "postprocess_eligibility_region": str(files["postprocess_eligibility"]),
        }
        rows.append(
            {
                **common,
                "state": "sham",
                "sample_id": reference + "__sham",
                "target_tissue_mask": source_paths["reference_tissue_mask"],
                "target_nuclei_mask": source_paths["reference_nuclei_mask"],
                "generation_change_region": str(files["sham_generation_support"]),
            }
        )
        for state in TARGETS:
            handoff = handoffs[state]
            rows.append(
                {
                    **common,
                    "state": state,
                    "sample_id": reference + "__" + state,
                    "target_tissue_mask": handoff["paths"]["target_tissue_mask"],
                    "target_nuclei_mask": handoff["paths"]["target_nuclei_mask"],
                    "generation_change_region": handoff["paths"]["generation_support"],
                    "joint_handoff_manifest": str(Path(patient["cases"][TARGETS.index(state)]["handoff_manifest"]["path"]).resolve()),
                }
            )
        approved_targets[reference] = targets
        roi_assets[reference] = {
            name: frozen(files[name]) for name in ("u1_roi", "u2_roi", "protected_roi")
        }

    manifest = args.output / "generation_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assets = {
        "approved_joint_targets": approved_targets,
        "protected_roi_files_and_hashes": roi_assets,
        "patient_wsi_mapping_audit": mappings,
        "training_overlap_audits": selected_overlaps,
        "pilot_exclusion_audit": exclusion,
    }
    source_count = len(selected)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    protocol.update(
        scope="formal_r3",
        status="frozen",
        formal_launch_permitted=True,
        source_count=source_count,
        unique_image_count_if_shared_sham=source_count * 6,
        generated_image_count=source_count * 5,
        blockers=[],
        implementation_commit=commit,
        cohort_selection={
            "formal_source_list_frozen": True,
            "selection": "reserved evaluation partition only; first source in frozen topology order with all four targets independently approved; no H&E or downstream-outcome selection",
            "target_patient_count": 30,
            "actual_patient_count": source_count,
        },
        generation_manifest_sha256=sha256_file(manifest),
        acceptance_status="pending_real_generation_and_measurements",
    )
    protocol.pop("technical_reference_ids", None)
    protocol.pop("technical_patient_ids", None)
    protocol["postprocess"]["configuration"] = asdict(CrossLowStainProtectionConfig())
    protocol["measurement"] = {
        "nominal_mpp": 0.25,
        "cell_point_policy": CELL_POINT_POLICY,
        "scale_basis": "frozen nominal patch scale; no recovered full-slide physical calibration",
    }
    for name, definition in protocol["model_assets"].items():
        if asset_identity(definition["path"])["sha256"] != definition["sha256"]:
            raise ValueError("frozen model content drift: " + name)
    protocol["cellvit_implementation_asset"] = frozen(Path(DEFAULT_CELLVIT_ROOT) / "cellvit")
    protocol["evaluator_release_asset"] = frozen(repo / protocol["evaluator_release"])
    parent = protocol["runtime_environments"]["generation_and_cellvit"]
    if Path(sys.executable).resolve() != Path(parent["python_path"]).resolve() or sys.version != parent["python"]:
        raise ValueError("formal freeze is running in a different parent interpreter")
    for name, version in parent["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("parent environment differs from the template")
    parent["packages"] = {**distribution_versions(), **parent["packages"]}
    expected_seg = protocol["runtime_environments"]["segmentator"]
    actual_seg = segmentator_environment(expected_seg["python_path"])
    if any(actual_seg[key] != value for key, value in expected_seg.items()):
        raise ValueError("segmentation environment differs from the template")
    protocol["runtime_environments"]["segmentator"] = actual_seg
    protocol["formal_assets"] = {"generation_manifest": frozen(manifest)}
    realized_path = args.output / "realized_target_fractions.json"
    realized_path.write_text(json.dumps(realized, indent=2) + "\n")
    protocol["formal_assets"]["realized_target_fractions"] = frozen(realized_path)
    for name, data in assets.items():
        path = args.output / (name + ".json")
        path.write_text(json.dumps(data, indent=2) + "\n")
        protocol["formal_assets"][name] = frozen(path)
    validate_cohort(
        rows,
        source_count=source_count,
        handoff_validator=build_frozen_generator_inputs,
        protocol=protocol,
        assets=assets,
        technical_validation=False,
    )
    path = args.output / "formal_r3_protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                "protocol": str(path),
                "manifest": str(manifest),
                "source_count": source_count,
                "formal_launch_permitted": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
