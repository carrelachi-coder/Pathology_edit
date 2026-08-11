from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_joint_edit_refine.evidence_freeze import (
    DATASET_PROFILES,
    freeze_dataset_evidence,
    verify_frozen_evidence_index,
)
from phase3_mask_edit_refine.evidence import EvidenceManifest


def test_six_dataset_evidence_freeze_is_complete_and_replayable(tmp_path: Path):
    grouped = {
        "seed": 42,
        "strategy": "fixture group-disjoint split",
        "preprocessing_revision": "fixture-preprocessing-v1",
        "train": [],
        "val": [],
        "test": [],
    }
    for dataset_id in DATASET_PROFILES:
        root = tmp_path / dataset_id
        for subdir in ("images", "tissue_masks", "nuclei_masks"):
            (root / subdir).mkdir(parents=True, exist_ok=True)
        for split_index, split in enumerate(("train", "val", "test")):
            name = f"{dataset_id}-{split}.png"
            image = np.full((8, 8, 3), 30 + split_index, dtype=np.uint8)
            mask = np.full((8, 8), split_index, dtype=np.uint8)
            nuclei = np.zeros((8, 8), dtype=np.uint8)
            nuclei[2:5, 2:5] = split_index + 1
            Image.fromarray(image).save(root / "images" / name)
            Image.fromarray(mask).save(root / "tissue_masks" / name)
            Image.fromarray(nuclei).save(root / "nuclei_masks" / name)
            grouped[split].append(
                {
                    "dataset_id": dataset_id,
                    "dataset_root": str(root),
                    "images_dir": "images",
                    "masks_dir": "tissue_masks",
                    "nuclei_dir": "nuclei_masks",
                    "image": name,
                    "mask": name,
                    "nuclei": name,
                    "sample_id": f"{dataset_id}:{split}",
                    "group_id": f"{dataset_id}-group-{split}",
                }
            )
    grouped_path = tmp_path / "grouped.json"
    grouped_path.write_text(json.dumps(grouped), encoding="utf-8")

    result = freeze_dataset_evidence(
        grouped_path,
        output_root=tmp_path / "frozen",
        code_revision="fixture-commit",
        workers=2,
    )
    verification = verify_frozen_evidence_index(result["index_path"])

    assert verification["passed"]
    assert verification["verified_records"] == 18
    assert len(result["datasets"]) == 6
    for dataset in result["datasets"]:
        manifest = EvidenceManifest.load(dataset["manifest_path"])
        assert len(manifest.records) == 3
        assert manifest.dataset_revision == dataset["dataset_revision"]
