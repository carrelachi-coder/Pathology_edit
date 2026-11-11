import pytest

from scripts.freeze_manuscript_formal_r3 import (
    load_selected,
    realized_target_record,
    validate_overlap_record,
)


def manifests():
    candidate = {
        "schema_version": "reserved-four-target-candidate-freeze-v1",
        "selected_candidates": [
            {
                "patient_order": 2,
                "patient_id": "P0",
                "wsi_id": "P0",
                "reference_id": "R0",
            }
        ],
    }
    materialized = {
        "schema_version": "reserved-four-target-handoff-materialization-v1",
        "patients": [
            {
                "patient_order": 2,
                "patient_id": "P0",
                "reference_id": "R0",
                "cases": [
                    {"target": target}
                    for target in ("u1_moderate", "u1_significant", "u2_moderate", "u2_significant")
                ],
            }
        ],
    }
    return candidate, materialized


def test_only_materialized_reserved_patients_are_selected():
    candidate, materialized = manifests()
    selected = load_selected(candidate, materialized)
    assert len(selected) == 1
    assert selected[0][0]["reference_id"] == "R0"


def test_patient_identity_drift_is_rejected():
    candidate, materialized = manifests()
    materialized["patients"][0]["patient_id"] = "different"
    with pytest.raises(ValueError, match="identity differs"):
        load_selected(candidate, materialized)


def test_incomplete_or_reordered_targets_are_rejected():
    candidate, materialized = manifests()
    materialized["patients"][0]["cases"].pop()
    with pytest.raises(ValueError, match="four-target order"):
        load_selected(candidate, materialized)


def test_empty_approval_set_cannot_freeze_formal_cohort():
    candidate, materialized = manifests()
    materialized["patients"] = []
    with pytest.raises(ValueError, match="no patient"):
        load_selected(candidate, materialized)


def test_realized_target_uses_actual_ledger_tissue_pixels_field():
    ledger = {
        "joint_fraction": 0.19,
        "tissue_fraction": 0.17,
        "joint_pixels": 100,
        "tissue_pixels": 90,
        "tissue_change_pixels": 999,
    }
    assert realized_target_record(ledger) == {
        "joint_fraction": 0.19,
        "tissue_fraction": 0.17,
        "joint_pixels": 100,
        "tissue_pixels": 90,
    }


def test_incomplete_realized_ledger_is_rejected_instead_of_writing_null():
    with pytest.raises(ValueError, match="lacks realized"):
        realized_target_record(
            {"joint_fraction": 0.19, "tissue_fraction": 0.17, "joint_pixels": 100}
        )


def overlap(patient="P0", source_sha="a" * 64):
    return {
        "status": "completed",
        "patient_id": patient,
        "source_image_sha256": source_sha,
        "generator": {},
        "segmentator": {},
        "cellvit": {},
    }


def test_overlap_record_is_bound_to_patient_and_source_image():
    record = overlap()
    assert validate_overlap_record(record, patient_id="P0", source_sha256="a" * 64) is record
    with pytest.raises(ValueError, match="different patient or source"):
        validate_overlap_record(record, patient_id="P1", source_sha256="a" * 64)
    with pytest.raises(ValueError, match="different patient or source"):
        validate_overlap_record(record, patient_id="P0", source_sha256="b" * 64)


def test_incomplete_overlap_record_is_rejected():
    record = overlap()
    record.pop("cellvit")
    with pytest.raises(ValueError, match="incomplete"):
        validate_overlap_record(record, patient_id="P0", source_sha256="a" * 64)
