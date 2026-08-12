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
        (root / "metadata.jsonl").write_text(
            '{"image":"images/sample.png","conditioning_image":"conditioning/sample.png"}\n',
            encoding="utf-8",
        )
        (root / "stats.txt").write_text(
            "Fixture Dataset Statistics\n"
            "Source: fixture source images\n"
            "Resize: no resize\n"
            "Filter: no filtering\n"
            "Label remap: fixture labels\n",
            encoding="utf-8",
        )
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
        raw = json.loads(Path(dataset["manifest_path"]).read_text())
        assert raw["preprocessing"]["materialization_evidence_complete"]
        assert raw["materialization_evidence"]["metadata_jsonl"]["sha256"]
        assert raw["materialization_evidence"]["stats_txt"]["sha256"]
        assert raw["preprocessing"]["preprocessing_statements"] == [
            "Source: fixture source images",
            "Resize: no resize",
            "Filter: no filtering",
            "Label remap: fixture labels",
        ]


def test_evidence_freeze_records_absent_test_without_relabeling_validation(
    tmp_path: Path,
):
    grouped = {
        "seed": 42,
        "strategy": "fixture group-disjoint split",
        "train": [],
        "val": [],
    }
    for dataset_id in DATASET_PROFILES:
        root = tmp_path / dataset_id
        for subdir in ("images", "tissue_masks", "nuclei_masks"):
            (root / subdir).mkdir(parents=True, exist_ok=True)
        (root / "metadata.jsonl").write_text("{}\n", encoding="utf-8")
        (root / "stats.txt").write_text(
            "Source: fixture\nFilter: none\n", encoding="utf-8"
        )
        for split in ("train", "val"):
            name = f"{dataset_id}-{split}.png"
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                root / "images" / name
            )
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
                root / "tissue_masks" / name
            )
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
                root / "nuclei_masks" / name
            )
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
                    "group_id": f"{dataset_id}:{split}",
                }
            )
    path = tmp_path / "grouped.json"
    path.write_text(json.dumps(grouped), encoding="utf-8")
    result = freeze_dataset_evidence(
        path,
        output_root=tmp_path / "frozen",
        code_revision="fixture-commit",
        workers=2,
    )
    assert result["split_contract"]["test_partition_status"] == "not_materialized"
    assert result["split_contract"]["absent_partitions"] == ["test"]
    for dataset in result["datasets"]:
        assert dataset["split_counts"] == {"train": 1, "validation": 1}


def test_evidence_freeze_derives_nuclei_and_groups_from_stage4_records(
    tmp_path: Path,
):
    grouped = {"train": [], "val": []}
    for dataset_id in DATASET_PROFILES:
        root = tmp_path / dataset_id
        for subdir in ("images", "tissue_masks", "nuclei_masks"):
            (root / subdir).mkdir(parents=True, exist_ok=True)
        (root / "metadata.jsonl").write_text("{}\n", encoding="utf-8")
        (root / "stats.txt").write_text(
            "Source: fixture\nFilter: none\n", encoding="utf-8"
        )
        for split, suffix in (("train", "caseA"), ("val", "caseB")):
            name = f"{suffix}_py0_px0.png"
            for subdir, array in (
                ("images", np.zeros((4, 4, 3), dtype=np.uint8)),
                ("tissue_masks", np.zeros((4, 4), dtype=np.uint8)),
                ("nuclei_masks", np.zeros((4, 4), dtype=np.uint8)),
            ):
                Image.fromarray(array).save(root / subdir / name)
            grouped[split].append(
                {
                    "dataset_id": dataset_id,
                    "dataset_root": str(root),
                    "images_dir": "images",
                    "masks_dir": "tissue_masks",
                    "image": name,
                    "mask": name,
                    "sample_id": f"{dataset_id}:{split}",
                }
            )
    path = tmp_path / "stage4.json"
    path.write_text(json.dumps(grouped), encoding="utf-8")
    result = freeze_dataset_evidence(
        path,
        output_root=tmp_path / "frozen",
        code_revision="fixture-commit",
        workers=2,
    )
    assert result["record_normalization"]["group_disjointness"] == (
        "required_across_materialized_partitions"
    )
    for dataset in result["datasets"]:
        raw = json.loads(Path(dataset["manifest_path"]).read_text())
        assert all(
            "/nuclei_masks/" in record["provenance"]["nuclei_mask_uri"]
            for record in raw["records"]
        )
