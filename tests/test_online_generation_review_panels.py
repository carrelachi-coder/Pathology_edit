from __future__ import annotations

import json

import numpy as np
from PIL import Image

from scripts.build_online_generation_review_panels import (
    draw_semantic_generation_boundaries,
    resolve_records,
)


def test_semantic_and_generation_boundaries_use_distinct_colors():
    image = Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8))
    semantic = np.zeros((64, 64), dtype=bool)
    semantic[24:40, 24:40] = True
    generation = np.zeros((64, 64), dtype=bool)
    generation[16:48, 16:48] = True

    result = np.asarray(
        draw_semantic_generation_boundaries(image, semantic, generation)
    )

    assert np.any(np.all(result == [255, 220, 0], axis=-1))
    assert np.any(np.all(result == [0, 235, 255], axis=-1))
    assert result[32, 32].tolist() == [180, 180, 180]


def test_resolve_records_reads_context_policy_from_pipeline_summary(tmp_path):
    generation_dir = tmp_path / "agentic_generation"
    generation_dir.mkdir()
    workflow = {
        "status": "validated_first_pass",
        "selected_attempt": {
            "requested_mode": "inpaint",
            "verification": {"quality_score": 0.98},
        },
    }
    pipeline_summary = {
        "change_regions": {
            "semantic_matches_tissue_difference": True,
            "generation_context_policy": {
                "policy": "bounded_generation_context_v2",
                "primitive_id": "invasive-cord-formation-v1",
                "max_extra_fraction": 1.5,
            }
        }
    }
    (generation_dir / "agentic_workflow.json").write_text(
        json.dumps(workflow), encoding="utf-8"
    )
    (generation_dir / "pipeline_summary.json").write_text(
        json.dumps(pipeline_summary), encoding="utf-8"
    )
    (generation_dir / "generation_report.json").write_text(
        "{}", encoding="utf-8"
    )
    for name in (
        "source_image.png",
        "source_tissue.png",
        "source_nuclei.png",
        "target_tissue.png",
        "target_nuclei.png",
    ):
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / name)
    for name in (
        "semantic_change_region.png",
        "generation_change_region.png",
        "generated_image.png",
    ):
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(
            generation_dir / name
        )
    manifest = {
        "records": [
            {
                "case_id": "cord_01",
                "category": "cord",
                "primitive_id": "invasive-cord-formation-v1",
                "agentic_generation_dir": str(generation_dir),
                "source_image": str(tmp_path / "source_image.png"),
                "source_tissue_mask": str(tmp_path / "source_tissue.png"),
                "source_nuclei_mask": str(tmp_path / "source_nuclei.png"),
                "target_tissue_mask": str(tmp_path / "target_tissue.png"),
                "target_nuclei_mask": str(tmp_path / "target_nuclei.png"),
            }
        ]
    }

    records = resolve_records(manifest)

    assert records[0]["generation_context_policy"]["max_extra_fraction"] == 1.5
    assert records[0]["semantic_matches_tissue_difference"] is True
    assert records[0]["pipeline_summary"] == str(
        (generation_dir / "pipeline_summary.json").resolve()
    )
    assert len(records[0]["selected_image_sha256"]) == 64
