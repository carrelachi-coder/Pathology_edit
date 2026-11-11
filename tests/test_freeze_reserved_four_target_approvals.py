import hashlib
import json
from pathlib import Path

import pytest

from scripts.freeze_reserved_four_target_approvals import freeze_approvals


TARGETS = ("u1_moderate", "u1_significant", "u2_moderate", "u2_significant")


def artifact(path: Path, text: str) -> dict:
    path.write_text(text)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def inputs(tmp_path: Path):
    cases = []
    decisions = []
    for index, target in enumerate(TARGETS):
        case_id = f"R0__{target}"
        cases.append(
            {
                "case_index": index,
                "case_id": case_id,
                "target": target,
                "status": "review_required",
                "audit_case_dir": str(tmp_path),
                "review_board": artifact(tmp_path / f"board{index}.png", "board"),
                "candidates": [
                    {
                        "candidate_id": f"candidate:{index}",
                        "evidence": artifact(tmp_path / f"evidence{index}.json", "{}"),
                    }
                ],
            }
        )
        decisions.append(
            {
                "case_id": case_id,
                "candidate_id": f"candidate:{index}",
                "decision": "approved",
                "reviewer": "codex_current_chat",
                "reason": "mask-only contract accepted",
                "confidence": 0.9,
                "reviewed_utc": "2026-11-12T00:00:00Z",
                "tissue_geometry": "target geometry follows the approved interface",
                "nucleus_layout": "nuclei remain inside compatible tissue",
                "coupling": "tissue and nuclei changes are coupled",
                "preserved_conditions": "unrequested regions remain unchanged",
                "limitations": "mask-only review without generated H&E",
                "verdict": "acceptable research condition",
            }
        )
    queue = {
        "schema_version": "reserved-four-target-mask-review-queue-v1",
        "patients": [
            {
                "patient_order": 0,
                "patient_id": "P0",
                "reference_id": "R0",
                "status": "ready_for_independent_mask_review",
                "cases": cases,
            }
        ],
    }
    decision_file = {
        "schema_version": "reserved-four-target-mask-review-decisions-v1",
        "decisions": decisions,
    }
    return queue, decision_file


def test_all_four_explicit_approvals_freeze_patient(tmp_path: Path):
    queue, decisions = inputs(tmp_path)
    rows = freeze_approvals(queue, decisions)
    assert rows[0]["status"] == "four_targets_approved"
    assert {row["status"] for row in rows[0]["cases"]} == {"approved"}


def test_one_rejection_keeps_patient_out(tmp_path: Path):
    queue, decisions = inputs(tmp_path)
    decisions["decisions"][2]["decision"] = "rejected"
    rows = freeze_approvals(queue, decisions)
    assert rows[0]["status"] == "not_approved"
    assert rows[0]["cases"][2]["status"] == "not_approved"


def test_missing_candidate_decision_is_rejected(tmp_path: Path):
    queue, decisions = inputs(tmp_path)
    decisions["decisions"].pop()
    with pytest.raises(ValueError, match="missing explicit"):
        freeze_approvals(queue, decisions)


def test_approval_without_detailed_mask_observations_is_rejected(tmp_path: Path):
    queue, decisions = inputs(tmp_path)
    decisions["decisions"][0].pop("tissue_geometry")
    with pytest.raises(ValueError, match="tissue_geometry"):
        freeze_approvals(queue, decisions)


def test_evidence_drift_is_rejected(tmp_path: Path):
    queue, decisions = inputs(tmp_path)
    Path(queue["patients"][0]["cases"][0]["candidates"][0]["evidence"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        freeze_approvals(queue, decisions)


def test_unknown_decision_is_rejected(tmp_path: Path):
    queue, decisions = inputs(tmp_path)
    decisions["decisions"].append(
        {
            "case_id": "outside",
            "candidate_id": "outside",
            "decision": "rejected",
            "reviewer": "codex_current_chat",
            "reason": "outside",
        }
    )
    with pytest.raises(ValueError, match="outside the review queue"):
        freeze_approvals(queue, decisions)
