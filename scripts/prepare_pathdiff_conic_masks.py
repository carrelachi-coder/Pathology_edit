#!/usr/bin/env python3
"""Generate official CoNIC-style instance/type masks for PathDiff inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase3_mask_edit.benchmark.conic_hovernet import (  # noqa: E402
    CONIC_CLASS_NAMES,
    HoVerNetConic,
    colorize_pathdiff_mask,
    infer_raw_maps,
    pathdiff_edge_map,
    postprocess_conic_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--image-field", default="target_image")
    parser.add_argument("--id-field", default="target_annotation_id")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"Unsupported manifest structure: {path}")
    return records


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_is_complete(sample_dir: Path) -> bool:
    required = (
        sample_dir / "conic.npy",
        sample_dir / "type_mask.png",
        sample_dir / "instance_edges.png",
        sample_dir / "pathdiff_condition.png",
        sample_dir / "metadata.json",
    )
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def mpp_provenance() -> dict:
    return {
        "source_resolution": 512,
        "source_mpp": 0.25,
        "source_fov_um": 128,
        "model_resolution": 256,
        "model_mpp": 0.5,
        "model_fov_um": 128,
        "output_resolution": 256,
        "preprocessing": "full_field_downsample_512_at_0.25_to_256_at_0.5",
        "crop_applied": False,
    }


def save_condition(
    record: dict,
    annotation_id: str,
    source_image: str,
    conic: np.ndarray,
    sample_dir: Path,
    checkpoint: Path,
    checkpoint_hash: str,
    runtime_seconds: float,
) -> dict:
    sample_dir.mkdir(parents=True, exist_ok=True)
    instances = conic[..., 0].astype(np.int32)
    types = conic[..., 1].astype(np.uint8)
    edges = pathdiff_edge_map(instances)
    colored_types = colorize_pathdiff_mask(types)
    colored_edges = np.repeat((edges * 255)[..., None], 3, axis=2)
    condition = np.concatenate([colored_types, colored_edges], axis=2)

    np.save(sample_dir / "conic.npy", conic.astype(np.int32))
    Image.fromarray(types, mode="L").save(sample_dir / "type_mask.png")
    Image.fromarray(edges * 255, mode="L").save(sample_dir / "instance_edges.png")
    Image.fromarray(colored_types).save(sample_dir / "pathdiff_condition.png")
    np.save(sample_dir / "pathdiff_condition_6ch.npy", condition)

    class_counts = {
        CONIC_CLASS_NAMES[class_id]: int(np.count_nonzero(types == class_id))
        for class_id in sorted(CONIC_CLASS_NAMES)
    }
    metadata = {
        "status": "completed",
        "annotation_id": annotation_id,
        "source_image": source_image,
        **mpp_provenance(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "schema": {
            "channel_0": "instance_id",
            "channel_1": "conic_type_id",
            "type_ids": CONIC_CLASS_NAMES,
            "pathdiff_hint": "RGB type color + RGB instance-edge map",
        },
        "instance_count": int(instances.max()),
        "type_pixel_counts": class_counts,
        "foreground_fraction": float(np.count_nonzero(instances)) / instances.size,
        "runtime_seconds": round(runtime_seconds, 3),
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def main() -> int:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    records = load_records(args.manifest)
    if args.max_items is not None:
        records = records[: args.max_items]
    unique_records: dict[str, dict] = {}
    for record in records:
        annotation_id = str(record.get(args.id_field) or "")
        source_image = str(record.get(args.image_field) or "")
        if not annotation_id or not source_image:
            raise ValueError(
                f"manifest row missing {args.id_field!r} or {args.image_field!r}"
            )
        unique_records.setdefault(
            annotation_id,
            {"record": record, "annotation_id": annotation_id, "source_image": source_image},
        )
    records = [
        value
        for index, value in enumerate(unique_records.values())
        if index % args.num_shards == args.shard_index
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        updated = 0
        missing = []
        for item in records:
            metadata_path = args.output_root / item["annotation_id"] / "metadata.json"
            if not metadata_path.exists():
                missing.append(item["annotation_id"])
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(mpp_provenance())
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            updated += 1
        print(
            json.dumps(
                {"metadata_updated": updated, "missing": missing},
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1 if missing else 0
    pending = [
        item
        for item in records
        if args.overwrite
        or not output_is_complete(args.output_root / item["annotation_id"])
    ]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = HoVerNetConic(num_types=7)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    checkpoint_hash = checkpoint_sha256(args.checkpoint)

    completed = len(records) - len(pending)
    failed = []
    for offset in range(0, len(pending), args.batch_size):
        batch_items = pending[offset : offset + args.batch_size]
        images = []
        valid_items = []
        for item in batch_items:
            try:
                image = Image.open(item["source_image"]).convert("RGB")
                if image.size != (512, 512):
                    raise ValueError(
                        f"Expected 512x512 target at 0.25 MPP, got {image.size}"
                    )
                image = image.resize((256, 256), Image.Resampling.BILINEAR)
                images.append(np.asarray(image, dtype=np.uint8))
                valid_items.append(item)
            except Exception as exc:
                failed.append(
                    {
                        "annotation_id": item["annotation_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if not valid_items:
            continue

        batch_started = time.time()
        np_maps, hv_maps, type_maps = infer_raw_maps(
            model, torch.from_numpy(np.stack(images)), device
        )
        per_item_runtime = (time.time() - batch_started) / len(valid_items)
        for item, np_map, hv_map, type_map in zip(
            valid_items, np_maps, hv_maps, type_maps
        ):
            annotation_id = item["annotation_id"]
            try:
                conic = postprocess_conic_maps(np_map, hv_map, type_map)
                metadata = save_condition(
                    item["record"],
                    annotation_id,
                    item["source_image"],
                    conic,
                    args.output_root / annotation_id,
                    args.checkpoint,
                    checkpoint_hash,
                    per_item_runtime,
                )
                completed += 1
                print(
                    f"[{completed}/{len(records)}] {annotation_id}: "
                    f"instances={metadata['instance_count']}",
                    flush=True,
                )
            except Exception as exc:
                failed.append(
                    {
                        "annotation_id": annotation_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    summary = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "output_root": str(args.output_root),
        "requested": len(records),
        "completed": completed,
        "failed": failed,
        "image_field": args.image_field,
        "id_field": args.id_field,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    summary_name = (
        f"summary_shard{args.shard_index}of{args.num_shards}.json"
        if args.num_shards != 1
        else "summary.json"
    )
    (args.output_root / summary_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
