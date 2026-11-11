import pytest

from scripts.audit_manuscript_training_overlap import candidate_sources


def image(path="/tmp/source.png"):
    return {"path": path, "bytes": 1, "sha256": "0" * 64}


def test_reserved_candidate_manifest_is_normalized_without_identity_changes():
    row = {
        "reference_id": "R0",
        "patient_id": "P0",
        "verified_source_inputs": {"reference_image": image()},
    }
    normalized = candidate_sources({"selected_candidates": [row]})
    assert normalized[0]["reference_id"] == "R0"
    assert normalized[0]["inputs"] is row["verified_source_inputs"]


def test_legacy_candidate_sources_remain_supported():
    row = {
        "reference_id": "R0",
        "patient_id": "P0",
        "inputs": {"reference_image": image()},
    }
    assert candidate_sources({"candidate_sources": [row]})[0]["inputs"] is row["inputs"]


def test_incomplete_source_identity_is_rejected():
    with pytest.raises(ValueError, match="frozen reference image"):
        candidate_sources({"selected_candidates": [{"reference_id": "R0"}]})
