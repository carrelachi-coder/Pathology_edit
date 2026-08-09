"""Portable source-nucleus instance ledger shared by joint and mature tools."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

INSTANCE_AUTHORITY_VERSION = "source-nucleus-instance-authority-v1"


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def binary_mask_sha256(value: np.ndarray) -> str:
    packed = np.packbits(np.asarray(value, dtype=np.uint8), axis=None)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def build_instance_authority(
    *,
    shape: tuple[int, int],
    source_nuclei_raw: np.ndarray,
    observation_quality: str,
    instances: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a canonical center ledger without re-segmenting semantic nuclei."""

    height, width = (int(shape[0]), int(shape[1]))
    records = []
    seen = set()
    for raw in instances:
        instance_id = str(raw["instance_id"])
        raw_class_id = int(raw["raw_class_id"])
        row = float(raw["row"])
        col = float(raw["col"])
        if instance_id in seen:
            raise ValueError(f"duplicate source nucleus instance ID: {instance_id}")
        if raw_class_id not in {101, 102, 103, 104, 105}:
            raise ValueError(f"invalid raw nucleus class in authority: {raw_class_id}")
        if not (0.0 <= row < height and 0.0 <= col < width):
            raise ValueError(f"source nucleus center is outside the patch: {instance_id}")
        seen.add(instance_id)
        records.append(
            {
                "instance_id": instance_id,
                "raw_class_id": raw_class_id,
                "row": row,
                "col": col,
                "tissue_fine_id": int(raw["tissue_fine_id"]),
                "completeness_status": str(
                    raw.get("completeness_status", "unknown")
                ),
                "source": str(raw.get("source", observation_quality)),
                "area_px": int(raw["area_px"]),
                "bbox_xyxy": [int(value) for value in raw["bbox_xyxy"]],
                "footprint_sha256": str(raw["footprint_sha256"]),
            }
        )
    records.sort(
        key=lambda item: (
            item["raw_class_id"],
            item["row"],
            item["col"],
            item["instance_id"],
        )
    )
    payload = {
        "schema_version": INSTANCE_AUTHORITY_VERSION,
        "shape": [height, width],
        "source_nuclei_raw_sha256": array_sha256(source_nuclei_raw),
        "observation_quality": str(observation_quality),
        "instances": records,
    }
    payload["authority_sha256"] = _canonical_digest(payload)
    return payload


def write_instance_authority(path: str | Path, payload: Mapping[str, Any]) -> None:
    validated = validate_instance_authority(payload)
    Path(path).write_text(
        json.dumps(validated, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_instance_authority(
    path: str | Path,
    *,
    expected_shape: tuple[int, int],
    source_nuclei_raw: np.ndarray,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validated = validate_instance_authority(payload)
    if tuple(validated["shape"]) != tuple(int(value) for value in expected_shape):
        raise ValueError("source instance authority shape does not match the patch")
    if validated["source_nuclei_raw_sha256"] != array_sha256(source_nuclei_raw):
        raise ValueError(
            "source instance authority is not bound to the supplied nuclei mask"
        )
    return validated


def validate_instance_authority(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("source instance authority root must be an object")
    result = dict(payload)
    if result.get("schema_version") != INSTANCE_AUTHORITY_VERSION:
        raise ValueError("unsupported source instance authority schema")
    shape = result.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise ValueError("source instance authority has an invalid shape")
    records = result.get("instances")
    if not isinstance(records, list) or not records:
        raise ValueError("source instance authority contains no instances")
    seen = set()
    height, width = shape
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("source instance authority record must be an object")
        instance_id = record.get("instance_id")
        bbox = record.get("bbox_xyxy")
        row = record.get("row")
        col = record.get("col")
        if (
            not isinstance(instance_id, str)
            or not instance_id
            or instance_id in seen
            or int(record.get("raw_class_id", 0))
            not in {101, 102, 103, 104, 105}
            or int(record.get("area_px", 0)) <= 0
            or not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, int) for value in bbox)
            or not (0 <= bbox[0] < bbox[2] <= width)
            or not (0 <= bbox[1] < bbox[3] <= height)
            or not isinstance(row, (int, float))
            or not isinstance(col, (int, float))
            or not (0 <= float(row) < height and 0 <= float(col) < width)
            or not isinstance(record.get("footprint_sha256"), str)
            or len(record["footprint_sha256"]) != 64
        ):
            raise ValueError("source instance authority record is malformed")
        seen.add(instance_id)
    source_digest = result.get("source_nuclei_raw_sha256")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise ValueError("source instance authority source-mask digest is malformed")
    expected_digest = str(result.get("authority_sha256") or "")
    unsigned = dict(result)
    unsigned.pop("authority_sha256", None)
    if not expected_digest or expected_digest != _canonical_digest(unsigned):
        raise ValueError("source instance authority digest is missing or invalid")
    return result


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
