import pytest

from scripts.merge_reserved_review_decisions import merge


def queue():
    return {
        "patients": [
            {
                "cases": [
                    {
                        "case_id": "case-a",
                        "candidates": [
                            {"candidate_id": "cand-1"},
                            {"candidate_id": "cand-2"},
                        ],
                    }
                ]
            }
        ]
    }


def document(rows):
    return {
        "schema_version": "reserved-four-target-mask-review-decisions-v1",
        "decisions": rows,
        "h_and_e_selection_used": False,
    }


def test_merges_disjoint_case_decisions():
    rows = merge(
        queue(),
        [
            document([{"case_id": "case-a", "candidate_id": "cand-1", "decision": "approved"}]),
            document([{"case_id": "case-a", "candidate_id": "cand-2", "decision": "rejected"}]),
        ],
        require_complete=True,
    )
    assert [row["candidate_id"] for row in rows] == ["cand-1", "cand-2"]


def test_rejects_conflicting_duplicate():
    with pytest.raises(ValueError, match="conflicting duplicate"):
        merge(
            queue(),
            [
                document([{"case_id": "case-a", "candidate_id": "cand-1", "decision": "approved"}]),
                document([{"case_id": "case-a", "candidate_id": "cand-1", "decision": "rejected"}]),
            ],
            require_complete=False,
        )


def test_rejects_candidate_outside_target_queue():
    with pytest.raises(ValueError, match="outside target queue"):
        merge(
            queue(),
            [document([{"case_id": "case-x", "candidate_id": "cand-1", "decision": "approved"}])],
            require_complete=False,
        )


def test_complete_mode_rejects_missing_candidate():
    with pytest.raises(ValueError, match="without decisions"):
        merge(
            queue(),
            [document([{"case_id": "case-a", "candidate_id": "cand-1", "decision": "approved"}])],
            require_complete=True,
        )


def test_rejects_h_and_e_based_source():
    source = document([])
    source["h_and_e_selection_used"] = True
    with pytest.raises(ValueError, match="prohibited"):
        merge(queue(), [source], require_complete=False)
