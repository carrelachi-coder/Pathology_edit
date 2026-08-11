from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from phase3_joint_edit_refine.g2_pair_replacements import (
    PAIR_REPLACEMENT_LEDGER_SCHEMA,
    apply_g2_v2_pair_replacements,
)
from phase3_joint_edit_refine.g2_v2_manifest import (
    G2_V2_MANIFEST_SCHEMA,
    PRIMITIVE_ONTOLOGY_VERSION,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    image = tmp_path / "image.png"
    tissue = tmp_path / "tissue.png"
    nuclei = tmp_path / "nuclei.png"
    board = tmp_path / "board.jpg"
    for path, payload in (
        (image, b"image"),
        (tissue, b"tissue"),
        (nuclei, b"nuclei"),
        (board, b"board"),
    ):
        path.write_bytes(payload)
    source_digests = {
        "image_sha256": _sha(image),
        "tissue_mask_sha256": _sha(tissue),
        "nuclei_mask_sha256": _sha(nuclei),
        "nuclei_instances_sha256": None,
        "auxiliary_structure_sha256": {},
    }
    shared = {
        "source_index": 0,
        "organ": "oral",
        "dataset": "ORCA",
        "pathology_domain_id": "oral-squamous-cell-carcinoma-v1",
        "annotation_profile_id": "orca-semantic-v1",
        "cell_observation_profile_id": "cellvit-5-v1",
        "cell_population_profile_id": "oral-scc-population-v1",
        "legacy_primitive": "tumor_burden_increase",
        "original_instruction": "increase tumor burden",
        "seed": 7,
    }
    base_row = {
        **shared,
        "case_id": "old-case",
        "execution_allowed": False,
        "decision_status": "abstain",
        "source_digests": source_digests,
    }
    base = {
        "schema_version": G2_V2_MANIFEST_SCHEMA,
        "primitive_ontology_version": PRIMITIVE_ONTOLOGY_VERSION,
        "case_count": 1,
        "execution_case_count": 0,
        "abstain_case_count": 1,
        "decision_counts": {"abstain": 1},
        "manifest_id": "base",
        "source_chain": {},
        "cases": [base_row],
    }
    base_path = tmp_path / "base.json"
    _write_json(base_path, base)

    semantic = {
        "schema_version": "joint-semantic-intent-v3",
        "instruction": "increase tumor burden",
        "instruction_mode": "direct_edit",
        "scenario": "direct_edit",
        "clinical_direction": "unspecified",
        "treatment_context": "none",
        "scenario_target": "tumor",
        "explicit_edit_scope": "tissue_burden",
        "primitive_id": "tumor-burden-increase-v1",
        "subject": "tumor",
        "direction": "increase",
        "explicit_cell_class": None,
        "explicit_location": None,
        "user_constraints": [],
        "uncertainties": [],
        "parser": "screen",
        "primitive_hypotheses": [
            {
                "primitive_id": "tumor-burden-increase-v1",
                "semantic_fit": "explicit",
                "priority": 0,
                "rationale": "explicit request",
            }
        ],
        "parser_metadata": {"mode": "screen"},
    }
    candidate_row = {
        **shared,
        "case_id": "candidate-case",
        "sample_id": "candidate-sample",
        "source_image_uri": str(image),
        "source_tissue_mask_uri": str(tissue),
        "source_nuclei_mask_uri": str(nuclei),
        "source_nuclei_instances_uri": None,
        "auxiliary_structure_uris": {},
        "source_digests": source_digests,
        "instruction": "increase tumor burden",
        "primitive_id": "tumor-burden-increase-v1",
        "mechanism_id": "oral-scc-cohesive-nest-cord",
        "prebound_semantic_intent": semantic,
        "prebound_semantic_intent_sha256": "screen-only",
        "execution_allowed": True,
        "decision_status": "deterministic_pool_screen_only",
        "joint_area_budget": {"target_fraction": 0.19},
        "budget_contract": {"mode": "joint"},
        "source_manifest_metadata": {},
    }
    candidate = {
        "schema_version": G2_V2_MANIFEST_SCHEMA,
        "primitive_ontology_version": PRIMITIVE_ONTOLOGY_VERSION,
        "case_count": 1,
        "cases": [candidate_row],
    }
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, candidate)
    exact = {
        "schema_version": "g2-v2-read-only-execution-qualification-v1",
        "case_id": "candidate-case",
        "status": "executable_preflight_passed",
        "failure_reasons": [],
        "source_manifest_sha256": _sha(candidate_path),
        "llm_api_used": False,
    }
    exact_path = tmp_path / "exact.jsonl"
    exact_path.write_text(json.dumps(exact, sort_keys=True) + "\n")
    ledger = {
        "schema_version": PAIR_REPLACEMENT_LEDGER_SCHEMA,
        "base_manifest_sha256": _sha(base_path),
        "reviewer": "current_codex_session",
        "llm_api_used": False,
        "replacements": [
            {
                "target_case_id": "old-case",
                "final_case_id": "replacement-case",
                "candidate_manifest": str(candidate_path),
                "candidate_manifest_sha256": _sha(candidate_path),
                "candidate_case_id": "candidate-case",
                "execution_qualification_jsonl": str(exact_path),
                "execution_qualification_sha256": _sha(exact_path),
                "visual_review": {
                    "reviewer": "current_codex_session",
                    "llm_api_used": False,
                    "decision_status": "supported_mechanism",
                    "board_path": str(board),
                    "board_sha256": _sha(board),
                    "board_page": 1,
                    "board_position": 0,
                    "visual_observations": [
                        "cohesive SCC front and a legal tissue interface are visible"
                    ],
                },
            }
        ],
    }
    ledger_path = tmp_path / "replacement.json"
    _write_json(ledger_path, ledger)
    return base_path, ledger_path, exact_path


def test_digest_bound_pair_replacement_preserves_order_and_recomputes_semantics(
    tmp_path: Path,
) -> None:
    base, ledger, _exact = _fixture(tmp_path)
    result = apply_g2_v2_pair_replacements(
        base, ledger, output_dir=tmp_path / "out"
    )
    payload = json.loads(Path(result["manifest"]).read_text())
    row = payload["cases"][0]
    assert result["execution_case_count"] == 1
    assert row["case_id"] == "replacement-case"
    assert row["source_index"] == 0
    assert row["execution_allowed"] is True
    assert row["review_basis"]["reviewer"] == "current_codex_session"
    semantic = row["prebound_semantic_intent"]
    assert semantic["parser_metadata"]["case_id"] == "replacement-case"
    expected = hashlib.sha256(
        json.dumps(
            semantic,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert row["prebound_semantic_intent_sha256"] == expected
    assert payload["pair_replacement_count"] == 1


def test_pair_replacement_rejects_failed_exact_qualification(
    tmp_path: Path,
) -> None:
    base, ledger, exact = _fixture(tmp_path)
    record = json.loads(exact.read_text())
    record["status"] = "execution_requalification_required"
    record["failure_reasons"] = ["not executable"]
    exact.write_text(json.dumps(record, sort_keys=True) + "\n")
    payload = json.loads(ledger.read_text())
    payload["replacements"][0]["execution_qualification_sha256"] = _sha(exact)
    _write_json(ledger, payload)
    with pytest.raises(ValueError, match="clean exact qualification"):
        apply_g2_v2_pair_replacements(
            base, ledger, output_dir=tmp_path / "out"
        )
