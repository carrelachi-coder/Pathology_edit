#!/usr/bin/env python3
"""Validate the complete target-condition bank used by generation baselines."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from phase3_mask_edit.benchmark.pathokid import sha256_file, stable_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cellvit-instance-root", type=Path, required=True)
    parser.add_argument("--conic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-items", type=int)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"unsupported manifest: {path}")
    return records


def image_array(path: Path) -> tuple[tuple[int, int], str, np.ndarray]:
    with Image.open(path) as image:
        return image.size, image.mode, np.asarray(image)


def main() -> int:
    args = parse_args()
    records = load_records(args.manifest)
    if args.max_items is not None:
        records = records[: args.max_items]
    unique = {}
    for record in records:
        unique.setdefault(record["target_annotation_id"], record)
    failures = []
    organ_counts = Counter()
    tissue_id_counts = Counter()
    cellvit_id_counts = Counter()
    conic_type_counts = Counter()
    conic_checkpoint_hashes = Counter()
    cellvit_checkpoint_hashes = Counter()
    for index, (annotation_id, record) in enumerate(unique.items(), start=1):
        organ_counts[record["organ"]] += 1
        try:
            image_path = Path(record["target_image"])
            tissue_path = Path(record["target_tissue_mask"])
            cellvit_mask_path = Path(record["target_nuclei_mask"])
            for path in (image_path, tissue_path, cellvit_mask_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            image_size, image_mode, _ = image_array(image_path)
            if image_size != (512, 512) or image_mode != "RGB":
                raise ValueError(f"invalid RGB image geometry/mode: {image_size}, {image_mode}")
            tissue_size, _, tissue = image_array(tissue_path)
            tissue_ids = set(int(value) for value in np.unique(tissue))
            if tissue_size != (512, 512) or not tissue_ids.issubset(set(range(8)) | {255}):
                raise ValueError(f"invalid tissue mask: size={tissue_size}, ids={sorted(tissue_ids)}")
            tissue_id_counts.update(tissue_ids)
            cellvit_size, _, cellvit = image_array(cellvit_mask_path)
            cellvit_ids = set(int(value) for value in np.unique(cellvit))
            if cellvit_size != (512, 512) or not cellvit_ids.issubset({0, 101, 102, 103, 104, 105}):
                raise ValueError(
                    f"invalid CellViT mask: size={cellvit_size}, ids={sorted(cellvit_ids)}"
                )
            cellvit_id_counts.update(cellvit_ids)

            cellvit_json_path = args.cellvit_instance_root / f"{annotation_id}.json"
            cellvit_payload = json.loads(cellvit_json_path.read_text(encoding="utf-8"))
            if not isinstance(cellvit_payload.get("cells"), list):
                raise ValueError("canonical CellViT JSON has no cells list")
            cellvit_provenance = cellvit_payload.get("benchmark_provenance", {})
            if not cellvit_provenance.get("semantic_mask_exact_match"):
                raise ValueError("CellViT instance JSON was not matched to packaged semantic mask")
            cellvit_checkpoint_hashes[str(cellvit_provenance.get("checkpoint_sha256"))] += 1

            conic_dir = args.conic_root / annotation_id
            required = (
                "conic.npy",
                "type_mask.png",
                "instance_edges.png",
                "pathdiff_condition.png",
                "pathdiff_condition_6ch.npy",
                "metadata.json",
            )
            missing = [name for name in required if not (conic_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"missing CoNIC files: {missing}")
            conic = np.load(conic_dir / "conic.npy", allow_pickle=False)
            hint = np.load(conic_dir / "pathdiff_condition_6ch.npy", allow_pickle=False)
            if conic.shape != (256, 256, 2) or hint.shape != (256, 256, 6):
                raise ValueError(f"invalid CoNIC/hint shape: {conic.shape}, {hint.shape}")
            conic_ids = set(int(value) for value in np.unique(conic[..., 1]))
            if not conic_ids.issubset(set(range(7))):
                raise ValueError(f"invalid CoNIC type IDs: {sorted(conic_ids)}")
            conic_type_counts.update(conic_ids)
            metadata = json.loads((conic_dir / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("annotation_id") != annotation_id:
                raise ValueError("CoNIC metadata annotation ID mismatch")
            expected_mpp = (metadata.get("source_mpp"), metadata.get("model_mpp"))
            if expected_mpp != (0.25, 0.5) or metadata.get("crop_applied") is not False:
                raise ValueError(f"invalid CoNIC physical provenance: {expected_mpp}")
            conic_checkpoint_hashes[str(metadata.get("checkpoint_sha256"))] += 1
        except Exception as exc:
            failures.append(
                {"annotation_id": annotation_id, "error": f"{type(exc).__name__}: {exc}"}
            )
        if index % 100 == 0 or index == len(unique):
            print(f"[{index}/{len(unique)}] failures={len(failures)}", flush=True)

    report = {
        "schema_version": 1,
        "status": "failed" if failures else "completed",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "direction_count": len(records),
        "unique_condition_count": len(unique),
        "sample_ids_sha256": stable_digest([row["sample_id"] for row in records]),
        "organ_counts": dict(sorted(organ_counts.items())),
        "coverage": {
            "rgb": len(unique) - len(failures),
            "tissue_semantic": len(unique) - len(failures),
            "cellvit_semantic": len(unique) - len(failures),
            "cellvit_instances": len(unique) - len(failures),
            "conic_instances_and_types": len(unique) - len(failures),
        },
        "observed_ids": {
            "tissue": sorted(tissue_id_counts),
            "cellvit": sorted(cellvit_id_counts),
            "conic": sorted(conic_type_counts),
        },
        "checkpoint_hash_counts": {
            "cellvit": dict(cellvit_checkpoint_hashes),
            "conic": dict(conic_checkpoint_hashes),
        },
        "physical_frame": {
            "source": "512x512 @ 0.25 MPP = 128 um",
            "pixcell_pathdiff_condition": "full-field 256x256 @ 0.5 MPP = 128 um",
            "condition_crop": "none",
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "direction_count", "unique_condition_count", "coverage")}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
