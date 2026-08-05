import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.replay_product_quality_evaluator import (
    _discover_case_summaries,
    main,
    select_replayed_attempt,
)


def _save_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _save_probabilities(path: Path, mask: np.ndarray) -> None:
    probabilities = np.full((8, *mask.shape), 0.04 / 7, dtype=np.float32)
    for class_id in range(8):
        probabilities[class_id][mask == class_id] = 0.96
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probabilities=probabilities,
        class_ids=np.arange(8, dtype=np.int64),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replay_marks_second_candidate_after_verifier_error_as_recovered():
    attempts = [
        {"attempt_index": 1, "verification": None, "error": "verifier failed"},
        {
            "attempt_index": 2,
            "verification": {"passed": True},
            "error": None,
        },
    ]

    selected, status = select_replayed_attempt(attempts)

    assert selected["attempt_index"] == 2
    assert status == "recovered"


def test_nested_replay_apply_preserves_nuclei_evidence_and_final_hash(tmp_path):
    case_id = "g2_001_breast_tumor_increase_test"
    instruction = tmp_path / "run" / "shard_00" / case_id / "instruction"
    agentic = instruction / "agentic_generation"
    inputs = instruction / "inputs"
    source = np.ones((96, 96), dtype=np.uint8)
    source[:, 48:] = 2
    target = source.copy()
    semantic = np.zeros_like(source, dtype=bool)
    semantic[24:56, 8:40] = True
    target[semantic] = 2
    generation = np.zeros_like(source, dtype=bool)
    generation[20:60, 4:44] = True
    nuclei = np.zeros_like(source, dtype=np.uint8)

    _save_mask(inputs / "source_tissue_mask.png", source)
    _save_mask(inputs / "source_image.png", np.full_like(source, 180))
    _save_mask(instruction / "approved_target_mask.png", target)
    _save_mask(instruction / "target_nuclei_mask.png", nuclei)
    _save_mask(agentic / "semantic_change_region.png", semantic * 255)
    _save_mask(agentic / "generation_change_region.png", generation * 255)
    _save_mask(agentic / "source_verification" / "coarse_mask.png", source)
    _save_probabilities(
        agentic / "source_verification" / "coarse_probabilities.npz",
        source,
    )
    calibration = {
        "changed_region": {"reference": {}, "predicted": {}},
        "full_image": {"reference": {}, "predicted": {}},
    }
    calibration_path = (
        agentic
        / "source_nuclei_verification"
        / "evaluator_calibration_counts.json"
    )
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")

    attempts = []
    for index, (mode, prediction, pixel) in enumerate(
        (("inpaint", target, 91), ("cross-v1-no-ip-pix2pix-v2", source, 173)),
        start=1,
    ):
        attempt_dir = agentic / f"attempt_{index:02d}_{mode.replace('-', '_')}"
        image_path = attempt_dir / "generated_image.png"
        _save_mask(image_path, np.full_like(source, pixel))
        verification_dir = attempt_dir / "verification"
        _save_mask(verification_dir / "coarse_mask_raw.png", prediction)
        _save_probabilities(
            verification_dir / "coarse_probabilities.npz", prediction
        )
        predicted_nuclei = verification_dir / "predicted_nuclei.png"
        _save_mask(predicted_nuclei, nuclei)
        audit_path = verification_dir / "online_semantic_audit.json"
        audit_path.write_text(
            json.dumps({"source_quality": {}, "decision_input": "raw"}),
            encoding="utf-8",
        )
        verification = {
            "semantic_decision_input": "raw",
            "online_semantic_audit": str(audit_path),
            "predicted_nuclei_mask": str(predicted_nuclei),
            "target_nuclei_instance_counts": {},
            "predicted_nuclei_instance_counts": {},
            "source_nuclei_evaluator_calibration": calibration,
        }
        verification_path = verification_dir / "verification.json"
        verification_path.write_text(json.dumps(verification), encoding="utf-8")
        controlnet_dir = attempt_dir / "controlnet"
        controlnet_dir.mkdir(parents=True, exist_ok=True)
        (controlnet_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "reference_tissue_mask": str(
                        inputs / "source_tissue_mask.png"
                    ),
                    "target_tissue_mask": str(
                        instruction / "approved_target_mask.png"
                    ),
                    "target_nuclei_mask": str(
                        instruction / "target_nuclei_mask.png"
                    ),
                }
            ),
            encoding="utf-8",
        )
        attempts.append(
            {
                "attempt_index": index,
                "requested_mode": mode,
                "decision_reason": "test route",
                "error": None,
                "artifact": {
                    "mode": mode,
                    "image_path": str(image_path),
                    "metadata": {
                        "selected_mode": mode,
                        "controlnet_output_dir": str(controlnet_dir),
                    },
                },
                "verification": {
                    "passed": False,
                    "quality_score": 0.0,
                },
            }
        )

    workflow = {
        "status": "evaluator_uncertain",
        "route": {
            "primary_mode": "inpaint",
            "reason": "test route",
        },
        "attempts": attempts,
        "selected_attempt": attempts[1],
        "output_dir": str(agentic),
    }
    (agentic / "pipeline_summary.json").write_text(
        json.dumps(workflow), encoding="utf-8"
    )
    (instruction / "pipeline_summary.json").write_text(
        json.dumps({"status": "complete", "generation": {}}),
        encoding="utf-8",
    )
    _save_mask(agentic / "generated_image.png", np.zeros_like(source))
    _save_mask(instruction / "generated_image.png", np.zeros_like(source))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": case_id,
                        "profile": "BCSS",
                        "primitive": "tumor_increase",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    initial_agentic_hash = _sha256(agentic / "generated_image.png")
    initial_instruction_hash = _sha256(instruction / "generated_image.png")
    assert (
        main(
            [
                "--run-root",
                str(tmp_path / "run"),
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "replay"),
                "--expected-count",
                "1",
            ]
        )
        == 0
    )
    assert _sha256(agentic / "generated_image.png") == initial_agentic_hash
    assert _sha256(instruction / "generated_image.png") == initial_instruction_hash

    assert (
        main(
            [
                "--run-root",
                str(tmp_path / "run"),
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "replay"),
                "--expected-count",
                "1",
                "--apply",
            ]
        )
        == 0
    )

    updated = json.loads((agentic / "pipeline_summary.json").read_text())
    assert updated["status"] == "validated_first_pass"
    assert len(updated["attempts"]) == 1
    assert updated["selected_attempt"]["attempt_index"] == 1
    assert updated["image_generation_provenance"]["selected_attempt"] == 1
    assert updated["image_generation_provenance"]["selected_mode"] == "inpaint"
    assert updated["image_generation_contract"]["quality_evaluator"] == {
        "policy_id": "online-quality-evaluator-v2.4",
        "preservation_exclusion_region": "full_generation_change_region",
    }
    assert updated["online_self_audit"]["artifact_replay_only"] is True
    superseded = updated["artifact_replay"]["superseded_historical_attempts"]
    assert len(superseded) == 1
    assert superseded[0]["attempt_index"] == 2
    verification = json.loads(
        (
            Path(attempts[0]["artifact"]["image_path"]).parent
            / "verification"
            / "verification.json"
        ).read_text()
    )
    assert verification["quality_policy"]["policy_id"] == (
        "online-quality-evaluator-v2.4"
    )
    assert verification["source_nuclei_evaluator_calibration"] == calibration
    assert verification["raw_audit_metrics"]["region_pixels"][
        "P_exclude"
    ] == int(np.count_nonzero(generation))
    selected_hash = _sha256(Path(attempts[0]["artifact"]["image_path"]))
    assert _sha256(agentic / "generated_image.png") == selected_hash
    assert _sha256(instruction / "generated_image.png") == selected_hash
    outer = json.loads((instruction / "pipeline_summary.json").read_text())
    assert outer["generation"]["quality_score"] == updated[
        "selected_attempt"
    ]["verification"]["quality_score"]
    assert outer["generation"]["image_generation_provenance"] == updated[
        "image_generation_provenance"
    ]
    assert outer["generation"]["generation_report"]["content"][
        "alternate_model_triggered"
    ] is False
    replay_summary = json.loads(
        (
            tmp_path
            / "replay"
            / "quality_evaluator_v2_4_replay_summary.json"
        ).read_text()
    )
    assert replay_summary["preflight"]["completed"] is True
    assert replay_summary["preflight"]["case_count"] == 1


def test_apply_requires_an_exact_expected_count(tmp_path):
    with pytest.raises(ValueError, match="--apply requires --expected-count"):
        main(
            [
                "--run-root",
                str(tmp_path / "run"),
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--output",
                str(tmp_path / "replay"),
                "--apply",
            ]
        )


def test_discovery_prefers_repaired_complete_summary_for_duplicate_case(tmp_path):
    case_id = "g2_301_oral_tumor_increase_test"
    old_summary = (
        tmp_path
        / "run"
        / "shard_00"
        / case_id
        / "instruction"
        / "agentic_generation"
        / "pipeline_summary.json"
    )
    repair_summary = (
        tmp_path
        / "run"
        / "repair_01_shard_00"
        / case_id
        / "instruction"
        / "agentic_generation"
        / "pipeline_summary.json"
    )
    old_summary.parent.mkdir(parents=True)
    old_summary.write_text(
        json.dumps({"status": "failed", "selected_attempt": None}),
        encoding="utf-8",
    )
    selected_image = repair_summary.parent / "attempt_01" / "generated_image.png"
    _save_mask(selected_image, np.zeros((8, 8), dtype=np.uint8))
    repair_summary.parent.mkdir(parents=True, exist_ok=True)
    repair_summary.write_text(
        json.dumps(
            {
                "status": "evaluator_uncertain",
                "selected_attempt": {
                    "attempt_index": 1,
                    "artifact": {"image_path": str(selected_image)},
                },
            }
        ),
        encoding="utf-8",
    )

    summaries = _discover_case_summaries(
        run_root=tmp_path / "run",
        output_root=tmp_path / "replay",
    )

    assert summaries == [repair_summary]
