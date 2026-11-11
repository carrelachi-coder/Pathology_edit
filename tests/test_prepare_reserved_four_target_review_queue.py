from pathlib import Path
import subprocess
import sys

from PIL import Image

from scripts.prepare_reserved_four_target_review_queue import prepare_queue


def test_direct_cli_help_imports_repository_modules():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repository / "scripts" / "prepare_reserved_four_target_review_queue.py"), "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_only_complete_four_target_patient_is_ready(tmp_path: Path):
    cases = [{"case_id": f"R0__{label}"} for label in (
        "u1_moderate", "u1_significant", "u2_moderate", "u2_significant"
    )]
    candidate_manifest = {
        "selected_candidates": [{
            "patient_order": 0,
            "patient_id": "P0",
            "reference_id": "R0",
            "case_indices": [0, 1, 2, 3],
        }]
    }
    receipts = {}
    for index, case in enumerate(cases):
        audit = tmp_path / f"audit{index}"
        audit.mkdir()
        Image.new("RGB", (2, 2), "white").save(audit / "joint_condition_mask_review.png")
        receipts[index] = {
            "case_index": index,
            "case_id": case["case_id"],
            "terminal_status": "review_required",
            "audit_case_dir": str(audit),
            "passing_candidate_ids": [f"cand:{index}"],
        }

    def evidence(_audit, *, candidate_id):
        return {"candidate_id": candidate_id, "bound": True}

    rows = prepare_queue(
        cases=cases,
        candidate_manifest=candidate_manifest,
        receipts_by_index=receipts,
        output=tmp_path / "out",
        evidence_builder=evidence,
    )
    assert rows[0]["status"] == "ready_for_independent_mask_review"
    assert len(rows[0]["cases"]) == 4
    assert rows[0]["cases"][0]["candidates"][0]["candidate_id"] == "cand:0"


def test_missing_case_keeps_patient_not_ready(tmp_path: Path):
    labels = ("u1_moderate", "u1_significant", "u2_moderate", "u2_significant")
    cases = [{"case_id": f"R0__{label}"} for label in labels]
    manifest = {"selected_candidates": [{
        "patient_order": 0, "patient_id": "P0", "reference_id": "R0",
        "case_indices": [0, 1, 2, 3],
    }]}
    rows = prepare_queue(
        cases=cases,
        candidate_manifest=manifest,
        receipts_by_index={},
        output=tmp_path / "out",
        evidence_builder=lambda *_args, **_kwargs: {},
    )
    assert rows[0]["status"] == "not_ready_for_independent_mask_review"
    assert {row["status"] for row in rows[0]["cases"]} == {"not_compiled"}
