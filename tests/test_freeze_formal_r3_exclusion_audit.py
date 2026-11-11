import pytest

from scripts.freeze_formal_r3_exclusion_audit import build


def inputs():
    candidate = {
        "schema_version": "reserved-four-target-candidate-freeze-v1",
        "selected_candidates": [{"patient_id": "R", "reference_id": "REF"}],
    }
    technical = {"excluded_patient_ids": ["T"], "excluded_reference_ids": ["TREF"]}
    split = {
        "development_patient_aliases": ["D"],
        "reserved_evaluation_patient_aliases": ["R"],
    }
    return candidate, technical, split


def test_combines_technical_and_development_exclusions():
    candidate, technical, split = inputs()
    result = build(candidate, technical, split)
    assert result["excluded_patient_ids"] == ["D", "T"]
    assert result["formal_candidates_all_in_reserved_partition"] is True


def test_candidate_outside_reserved_partition_is_rejected():
    candidate, technical, split = inputs()
    candidate["selected_candidates"][0]["patient_id"] = "X"
    with pytest.raises(ValueError, match="outside the frozen reserved"):
        build(candidate, technical, split)


def test_candidate_overlap_with_exclusion_is_rejected():
    candidate, technical, split = inputs()
    split["development_patient_aliases"] = ["R"]
    split["reserved_evaluation_patient_aliases"] = ["R"]
    with pytest.raises(ValueError, match="assignments overlap"):
        build(candidate, technical, split)
