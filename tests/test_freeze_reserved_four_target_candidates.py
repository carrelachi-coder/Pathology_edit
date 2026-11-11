import hashlib
import json
from pathlib import Path

import pytest

from scripts.freeze_reserved_four_target_candidates import (
    build_candidates,
    collect_receipts,
    load_templates,
    verify_selected_source_assets,
)


def templates_payload():
    rows = []
    for primitive, operation in (
        ("cohesive-boundary-expansion-v1", "u1"),
        ("generic-immune-infiltrate-increase-v1", "u2"),
    ):
        for target, strength, floor in ((0.19, "moderate", 0.14), (0.32, "significant", 0.27)):
            rows.append(
                {
                    "case_id": f"old__{operation}__{strength}",
                    "instruction": "frozen",
                    "source_image_uri": "/old/rgb.png",
                    "source_tissue_mask_uri": "/old/tissue.png",
                    "source_nuclei_mask_uri": "/old/nuclei.png",
                    "pathology_domain_id": "breast-invasive-carcinoma-v1",
                    "annotation_profile_id": "bcss-semantic-v1",
                    "cell_observation_profile_id": "cellvit-five-class-v1",
                    "cell_population_profile_id": "breast-cellvit-source-first-v1",
                    "primitive_id": primitive,
                    "joint_area_budget": {
                        "target_fraction": target,
                        "min_fraction": floor,
                        "max_fraction": 0.24 if target == 0.19 else 0.37,
                        "tissue_min_fraction": floor,
                        "basis": "whole_patch",
                        "relative_tolerance": 0.02,
                        "fallback_policy": "max_feasible_below_target",
                        "capacity_floor_policy": "strict",
                        "minimum_effective_fraction": 0.0,
                    },
                    "pixel_size_um": 0.25,
                    "seed": 42,
                    "provenance": {
                        "source_image_sha256": "old-rgb",
                        "source_tissue_mask_sha256": "old-tissue",
                        "source_nuclei_mask_sha256": "old-nuclei",
                    },
                }
            )
    return rows


def freeze_payload(partition="reserved_evaluation"):
    return {
        "partition": partition,
        "patient_aliases": ["P0", "P1"],
        "candidate_sources": [
            {
                "reference_id": "R0",
                "patient_id": "P0",
                "wsi_id": "P0",
                "method_partition": partition,
                "patient_source_rank": 0,
                "inputs": {
                    "reference_image": {"path": "/rgb0.png", "sha256": "rgb0"},
                    "reference_tissue_mask": {"path": "/t0.png", "sha256": "t0"},
                    "reference_nuclei_mask": {"path": "/n0.png", "sha256": "n0"},
                },
            },
            {
                "reference_id": "R1",
                "patient_id": "P1",
                "wsi_id": "P1",
                "method_partition": partition,
                "patient_source_rank": 0,
                "inputs": {
                    "reference_image": {"path": "/rgb1.png", "sha256": "rgb1"},
                    "reference_tissue_mask": {"path": "/t1.png", "sha256": "t1"},
                    "reference_nuclei_mask": {"path": "/n1.png", "sha256": "n1"},
                },
            },
        ],
    }


def receipt_row(patient, reference, sha="receipt"):
    return {
        "patient_id": patient,
        "status": "first_source_passed",
        "selected_reference_id": reference,
        "receipt": {"sha256": sha},
    }


def test_builds_four_cases_per_reserved_patient_in_frozen_order(tmp_path: Path):
    template_path = tmp_path / "templates.json"
    template_path.write_text(json.dumps(templates_payload()))
    templates = load_templates(template_path)
    selected, cases = build_candidates(
        freeze=freeze_payload(),
        templates=templates,
        pass_rows=[receipt_row("P1", "R1", "s1"), receipt_row("P0", "R0", "s0")],
    )
    assert [row["patient_id"] for row in selected] == ["P0", "P1"]
    assert len(cases) == 8
    assert [case["case_id"] for case in cases[:4]] == [
        "R0__u1_moderate",
        "R0__u1_significant",
        "R0__u2_moderate",
        "R0__u2_significant",
    ]
    assert cases[0]["source_image_uri"] == "/rgb0.png"
    assert cases[0]["provenance"]["topology_receipt_sha256"] == "s0"
    assert cases[0]["provenance"]["topology_method_partition"] == "reserved_evaluation"


def test_rejects_development_partition(tmp_path: Path):
    template_path = tmp_path / "templates.json"
    template_path.write_text(json.dumps(templates_payload()))
    with pytest.raises(ValueError, match="reserved_evaluation"):
        build_candidates(
            freeze=freeze_payload("development"),
            templates=load_templates(template_path),
            pass_rows=[receipt_row("P0", "R0")],
        )


def test_rejects_conflicting_passes_for_same_patient(tmp_path: Path):
    template_path = tmp_path / "templates.json"
    template_path.write_text(json.dumps(templates_payload()))
    freeze = freeze_payload()
    freeze["candidate_sources"].append(
        {
            **freeze["candidate_sources"][0],
            "reference_id": "R0b",
            "patient_source_rank": 1,
        }
    )
    with pytest.raises(ValueError, match="conflicting passing sources"):
        build_candidates(
            freeze=freeze,
            templates=load_templates(template_path),
            pass_rows=[receipt_row("P0", "R0"), receipt_row("P0", "R0b")],
        )


def test_collect_receipts_keeps_history_but_only_returns_passes(tmp_path: Path):
    root = tmp_path / "q"
    for index, status in enumerate(("indeterminate_execution", "first_source_passed")):
        patient = root / f"patient_{index:03d}_P{index}"
        patient.mkdir(parents=True)
        (patient / "patient_receipt.json").write_text(
            json.dumps(
                {
                    "patient_id": f"P{index}",
                    "status": status,
                    "selected_reference_id": "R1" if index else None,
                    "attempted_sources": 1,
                }
            )
        )
    history, passes = collect_receipts([root])
    assert len(history) == 2
    assert [row["patient_id"] for row in passes] == ["P1"]


def test_selected_source_assets_are_content_verified(tmp_path: Path):
    template_path = tmp_path / "templates.json"
    template_path.write_text(json.dumps(templates_payload()))
    freeze = freeze_payload()
    for label, asset in freeze["candidate_sources"][0]["inputs"].items():
        path = tmp_path / f"{label}.bin"
        path.write_bytes(label.encode())
        asset["path"] = str(path)
        asset["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    selected, _cases = build_candidates(
        freeze=freeze,
        templates=load_templates(template_path),
        pass_rows=[receipt_row("P0", "R0")],
    )
    verify_selected_source_assets(freeze=freeze, selected_rows=selected)
    assert set(selected[0]["verified_source_inputs"]) == {
        "reference_image",
        "reference_tissue_mask",
        "reference_nuclei_mask",
    }
    freeze["candidate_sources"][0]["inputs"]["reference_image"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_selected_source_assets(freeze=freeze, selected_rows=selected)
