import hashlib

import pytest

from scripts.preflight_formal_r3_candidate_assets import verify


def manifest(tmp_path):
    inputs = {}
    for name in ("reference_image", "reference_tissue_mask", "reference_nuclei_mask"):
        path = tmp_path / f"TCGA-A2-A0D0_{name}.png"
        path.write_bytes(name.encode())
        inputs[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema_version": "reserved-four-target-candidate-freeze-v1",
        "unique_reserved_topology_pass_patients": 1,
        "candidate_case_count": 4,
        "selected_candidates": [
            {"patient_id": "TCGA-A2-A0D0", "reference_id": "R0", "verified_source_inputs": inputs}
        ],
    }


def test_all_three_source_assets_are_verified(tmp_path):
    result = verify(manifest(tmp_path))
    assert result["candidate_patients"] == 1
    assert result["candidate_cases"] == 4
    assert result["verified_assets"] == 3


def test_content_drift_is_rejected(tmp_path):
    frozen = manifest(tmp_path)
    path = tmp_path / "TCGA-A2-A0D0_reference_image.png"
    path.write_bytes(b"drift")
    with pytest.raises(ValueError, match="content drift"):
        verify(frozen)


def test_case_count_drift_is_rejected(tmp_path):
    frozen = manifest(tmp_path)
    frozen["candidate_case_count"] = 3
    with pytest.raises(ValueError, match="four-target case count"):
        verify(frozen)
