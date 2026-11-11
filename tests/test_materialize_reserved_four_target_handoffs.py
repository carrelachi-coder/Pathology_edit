import hashlib
import json
from pathlib import Path

import pytest

from scripts.materialize_reserved_four_target_handoffs import materialize


TARGETS = ("u1_moderate", "u1_significant", "u2_moderate", "u2_significant")


def identity(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def approval_freeze(tmp_path: Path) -> dict:
    cases = []
    for index, target in enumerate(TARGETS):
        case_id = f"R0__{target}"
        candidate_id = f"candidate:{index}"
        audit = tmp_path / f"audit{index}"
        audit.mkdir()
        evidence = tmp_path / f"evidence{index}.json"
        evidence.write_text(json.dumps({"case_id": case_id, "candidate_id": candidate_id}))
        cases.append(
            {
                "case_index": index,
                "case_id": case_id,
                "target": target,
                "status": "approved",
                "audit_case_dir": str(audit),
                "approved_candidate": {
                    "candidate_id": candidate_id,
                    "evidence": identity(evidence),
                    "reviewer": "codex_current_chat",
                    "reason": "accepted",
                    "confidence": 0.9,
                    "reviewed_utc": "2026-11-12T00:00:00Z",
                    "tissue_geometry": "geometry acceptable",
                    "nucleus_layout": "layout acceptable",
                    "coupling": "coupling acceptable",
                    "preserved_conditions": "preservation acceptable",
                    "limitations": "mask-only review",
                    "verdict": "approve",
                },
            }
        )
    return {
        "schema_version": "reserved-four-target-approval-freeze-v1",
        "patients": [
            {
                "patient_order": 0,
                "patient_id": "P0",
                "reference_id": "R0",
                "status": "four_targets_approved",
                "cases": cases,
            }
        ],
    }


def test_materialize_writes_exact_reviews_and_four_handoffs(tmp_path: Path):
    freeze = approval_freeze(tmp_path)
    calls = []

    def promoter(audit_dir, *, candidate_id, review_path, output_dir):
        review = json.loads(Path(review_path).read_text())
        assert review["candidate_id"] == candidate_id
        assert review["source_he_provided_to_review"] is False
        assert review["hard_gate_waiver"] is False
        handoff = Path(output_dir) / "generation_handoff" / "manifest.json"
        handoff.parent.mkdir()
        handoff.write_text(json.dumps({"candidate_id": candidate_id}))
        calls.append((Path(audit_dir), candidate_id))
        return {"manifest": str(handoff)}

    rows = materialize(freeze, output=tmp_path / "output", promoter=promoter)
    assert len(calls) == 4
    assert len(rows) == 1 and len(rows[0]["cases"]) == 4


def test_materialize_rejects_evidence_identity_drift(tmp_path: Path):
    freeze = approval_freeze(tmp_path)
    evidence = Path(freeze["patients"][0]["cases"][0]["approved_candidate"]["evidence"]["path"])
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="identity changed"):
        materialize(freeze, output=tmp_path / "output", promoter=lambda *args, **kwargs: {})


def test_materialize_skips_nonapproved_patient_without_promoting(tmp_path: Path):
    freeze = approval_freeze(tmp_path)
    freeze["patients"][0]["status"] = "not_approved"
    rows = materialize(
        freeze,
        output=tmp_path / "output",
        promoter=lambda *args, **kwargs: pytest.fail("promoter must not run"),
    )
    assert rows == []
