from scripts.run_topology_method_patient_qualification import group_patient_sources


def test_group_patient_sources_preserves_frozen_patient_and_source_order():
    freeze = {
        "patient_aliases": ["p2", "p1", "p3"],
        "candidate_sources": [
            {"patient_id": "p2", "patient_source_rank": 0, "reference_id": "a"},
            {"patient_id": "p2", "patient_source_rank": 1, "reference_id": "b"},
            {"patient_id": "p1", "patient_source_rank": 0, "reference_id": "c"},
            {"patient_id": "p3", "patient_source_rank": 0, "reference_id": "d"},
        ],
    }
    grouped = group_patient_sources(freeze, excluded_patients={"p1"})
    assert [patient for patient, _rows in grouped] == ["p2", "p3"]
    assert [row[1]["reference_id"] for row in grouped[0][1]] == ["a", "b"]


def test_group_patient_sources_rejects_noncontiguous_patient_rank():
    freeze = {
        "patient_aliases": ["p1"],
        "candidate_sources": [
            {"patient_id": "p1", "patient_source_rank": 1, "reference_id": "a"}
        ],
    }
    try:
        group_patient_sources(freeze, excluded_patients=set())
    except ValueError as exc:
        assert "not contiguous" in str(exc)
    else:
        raise AssertionError("noncontiguous ranks must be rejected")
