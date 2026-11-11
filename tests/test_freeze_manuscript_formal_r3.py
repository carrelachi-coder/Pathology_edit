import pytest

from scripts.freeze_manuscript_formal_r3 import load_selected


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
