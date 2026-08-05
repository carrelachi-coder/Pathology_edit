"""Read-only evidence manifests, mask statistics, and provenance verification."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit_refine.models import RefineContractError


@dataclass(frozen=True)
class EvidenceRecord:
    record_id: str
    mask_uri: str
    patient_id: str
    wsi_id: str
    split: str
    image_uri: str | None = None
    pixel_size_um: float | None = None
    mask_sha256: str | None = None
    image_sha256: str | None = None
    provenance: dict[str, Any] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> EvidenceRecord:
        required = ("record_id", "mask_uri", "patient_id", "wsi_id", "split")
        values: dict[str, str] = {}
        for key in required:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                raise RefineContractError(f"evidence record {key} must be a non-empty string")
            values[key] = value.strip()
        split = values["split"]
        if split not in {"train", "validation", "test"}:
            raise RefineContractError(f"unknown evidence split: {split}")
        pixel_size = payload.get("pixel_size_um")
        if pixel_size is not None and (
            not isinstance(pixel_size, (int, float)) or float(pixel_size) <= 0
        ):
            raise RefineContractError("pixel_size_um must be positive when provided")
        provenance = payload.get("provenance")
        if provenance is not None and not isinstance(provenance, Mapping):
            raise RefineContractError("record provenance must be a mapping")
        return cls(
            **values,
            image_uri=_optional_string(payload.get("image_uri")),
            pixel_size_um=float(pixel_size) if pixel_size is not None else None,
            mask_sha256=_optional_string(payload.get("mask_sha256")),
            image_sha256=_optional_string(payload.get("image_sha256")),
            provenance=dict(provenance) if provenance is not None else None,
        )


@dataclass(frozen=True)
class EvidenceManifest:
    annotation_profile_id: str
    dataset_revision: str
    protocol_sources: tuple[str, ...]
    records: tuple[EvidenceRecord, ...]
    manifest_path: str

    @classmethod
    def load(cls, path: str | Path) -> EvidenceManifest:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RefineContractError(f"could not load evidence manifest {source}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise RefineContractError("evidence manifest root must be an object")
        records = payload.get("records")
        sources = payload.get("protocol_sources")
        if not isinstance(records, list) or not records:
            raise RefineContractError("evidence manifest requires records")
        if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
            raise RefineContractError("protocol_sources must be a list of strings")
        manifest = cls(
            annotation_profile_id=_required_string(payload, "annotation_profile_id"),
            dataset_revision=_required_string(payload, "dataset_revision"),
            protocol_sources=tuple(sources),
            records=tuple(EvidenceRecord.from_mapping(item) for item in records),
            manifest_path=str(source.resolve()),
        )
        validate_split_integrity(manifest)
        return manifest


def validate_split_integrity(manifest: EvidenceManifest) -> None:
    seen_record_ids: set[str] = set()
    patient_splits: dict[str, set[str]] = defaultdict(set)
    wsi_splits: dict[str, set[str]] = defaultdict(set)
    for record in manifest.records:
        if record.record_id in seen_record_ids:
            raise RefineContractError(f"duplicate evidence record_id: {record.record_id}")
        seen_record_ids.add(record.record_id)
        patient_splits[record.patient_id].add(record.split)
        wsi_splits[record.wsi_id].add(record.split)
    leaking_patients = sorted(key for key, splits in patient_splits.items() if len(splits) > 1)
    leaking_wsis = sorted(key for key, splits in wsi_splits.items() if len(splits) > 1)
    if leaking_patients or leaking_wsis:
        raise RefineContractError(
            f"split leakage: patients={leaking_patients[:10]}, wsis={leaking_wsis[:10]}"
        )


def verify_evidence_files(
    manifest: EvidenceManifest, *, require_digests: bool = True
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    verified = 0
    for record in manifest.records:
        for kind, uri, expected in (
            ("mask", record.mask_uri, record.mask_sha256),
            ("image", record.image_uri, record.image_sha256),
        ):
            if uri is None:
                continue
            path = Path(uri)
            if not path.is_file():
                failures.append({"record_id": record.record_id, "kind": kind, "error": "missing"})
                continue
            if require_digests and not expected:
                failures.append(
                    {"record_id": record.record_id, "kind": kind, "error": "digest_missing"}
                )
                continue
            if expected:
                observed = sha256_file(path)
                if observed.lower() != expected.lower():
                    failures.append(
                        {"record_id": record.record_id, "kind": kind, "error": "digest_mismatch"}
                    )
                    continue
            verified += 1
    return {
        "passed": not failures,
        "verified_files": verified,
        "failures": failures,
        "manifest_sha256": sha256_file(manifest.manifest_path),
    }


def verify_case_run_bundle(path: str | Path) -> dict[str, Any]:
    """Verify a case provenance bundle without inferring absent run artifacts."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RefineContractError(f"could not load run-bundle manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RefineContractError("run-bundle manifest root must be an object")
    if payload.get("schema_version") != "mask-edit-refine-run-bundle-v1":
        raise RefineContractError("unsupported run-bundle schema_version")
    case_id = _required_string(payload, "case_id")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RefineContractError("run-bundle artifacts must be an object")
    required = {
        "source_image",
        "source_mask",
        "target_mask",
        "instruction",
        "planner_response",
        "run_manifest",
        "code_snapshot",
    }
    missing = sorted(required - set(artifacts))
    failures: list[dict[str, str]] = []
    if missing:
        failures.extend({"artifact": key, "error": "manifest_entry_missing"} for key in missing)
    verified: dict[str, dict[str, Any]] = {}
    for name, raw in artifacts.items():
        if not isinstance(raw, Mapping):
            failures.append({"artifact": str(name), "error": "entry_not_object"})
            continue
        uri = raw.get("path")
        expected = raw.get("sha256")
        if not isinstance(uri, str) or not uri or not isinstance(expected, str) or not expected:
            failures.append({"artifact": str(name), "error": "path_or_digest_missing"})
            continue
        artifact_path = Path(uri)
        if not artifact_path.is_file():
            failures.append({"artifact": str(name), "error": "file_missing"})
            continue
        observed = sha256_file(artifact_path)
        if observed.lower() != expected.lower():
            failures.append({"artifact": str(name), "error": "digest_mismatch"})
            continue
        if name in {"instruction", "planner_response", "run_manifest"}:
            try:
                json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                failures.append({"artifact": str(name), "error": "invalid_json"})
                continue
        verified[str(name)] = {
            "path": str(artifact_path),
            "sha256": observed,
            "bytes": artifact_path.stat().st_size,
        }
    return {
        "case_id": case_id,
        "passed": not failures,
        "verified_artifacts": verified,
        "failures": failures,
        "bundle_manifest_sha256": sha256_file(manifest_path),
    }


def build_annotation_profile_statistics(
    manifest: EvidenceManifest,
    *,
    schema: MaskProfileSchema,
    split: str = "train",
) -> dict[str, Any]:
    """Compute full-mask empirical statistics without using an LLM."""

    records = [record for record in manifest.records if record.split == split]
    if not records:
        raise RefineContractError(f"evidence manifest has no records for split={split}")
    label_fractions: dict[str, list[float]] = defaultdict(list)
    component_counts: dict[str, list[float]] = defaultdict(list)
    component_areas: dict[str, list[float]] = defaultdict(list)
    perimeter_area_ratios: dict[str, list[float]] = defaultdict(list)
    hole_counts: dict[str, list[float]] = defaultdict(list)
    adjacency_counts: dict[str, int] = defaultdict(int)
    background_components_per_mpx: list[float] = []
    background_fractions: list[float] = []
    background_border_fraction: list[float] = []
    internal_background_fraction: list[float] = []
    observed_ids: set[int] = set()
    pixel_sizes: list[float] = []
    total_pixels = 0

    for record in records:
        mask = load_id_mask(record.mask_uri)
        total_pixels += int(mask.size)
        observed_ids.update(int(value) for value in np.unique(mask))
        if record.pixel_size_um is not None:
            pixel_sizes.append(record.pixel_size_um)
        label_masks: dict[str, np.ndarray] = {}
        for label in sorted(schema.readable_labels):
            binary = np.isin(mask, schema.resolve_fine_ids(label))
            label_masks[label] = binary
            label_fractions[label].append(float(np.mean(binary)))
            labeled, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=bool))
            component_counts[label].append(float(count))
            holes = ndimage.binary_fill_holes(binary) & ~binary
            _, hole_count = ndimage.label(holes)
            hole_counts[label].append(float(hole_count))
            for component_id in range(1, count + 1):
                component = labeled == component_id
                area = int(np.count_nonzero(component))
                boundary = component & ~ndimage.binary_erosion(component)
                perimeter = int(np.count_nonzero(boundary))
                component_areas[label].append(float(area))
                perimeter_area_ratios[label].append(perimeter / max(area, 1))
        labels = sorted(label_masks)
        for index, left in enumerate(labels):
            dilated = ndimage.binary_dilation(label_masks[left])
            for right in labels[index + 1 :]:
                if np.any(dilated & label_masks[right]):
                    adjacency_counts[f"{left}|{right}"] += 1

        background = np.isin(mask, tuple(schema.skip_fine_ids))
        background_fractions.append(float(np.mean(background)))
        bg_labeled, bg_count = ndimage.label(background, structure=np.ones((3, 3), dtype=bool))
        background_components_per_mpx.append(bg_count / max(mask.size / 1_000_000.0, 1e-9))
        border_component_ids = {
            int(value)
            for value in np.unique(
                np.concatenate(
                    [
                        bg_labeled[0, :],
                        bg_labeled[-1, :],
                        bg_labeled[:, 0],
                        bg_labeled[:, -1],
                    ]
                )
            )
        } - {0}
        border_pixels = int(np.count_nonzero(np.isin(bg_labeled, tuple(border_component_ids))))
        background_pixels = int(np.count_nonzero(background))
        background_border_fraction.append(border_pixels / max(background_pixels, 1))
        internal_background_fraction.append(
            max(0, background_pixels - border_pixels) / max(mask.size, 1)
        )

    return {
        "schema_version": "annotation-profile-statistics-v1",
        "annotation_profile_id": manifest.annotation_profile_id,
        "dataset_revision": manifest.dataset_revision,
        "split": split,
        "manifest_path": manifest.manifest_path,
        "manifest_sha256": sha256_file(manifest.manifest_path),
        "record_count": len(records),
        "patient_count": len({record.patient_id for record in records}),
        "wsi_count": len({record.wsi_id for record in records}),
        "total_pixels": total_pixels,
        "observed_fine_ids": sorted(observed_ids),
        "pixel_size_um": _quantiles(pixel_sizes),
        "label_fraction": {key: _quantiles(values) for key, values in label_fractions.items()},
        "component_count_per_patch": {
            key: _quantiles(values) for key, values in component_counts.items()
        },
        "component_area_px": {key: _quantiles(values) for key, values in component_areas.items()},
        "perimeter_area_ratio": {
            key: _quantiles(values) for key, values in perimeter_area_ratios.items()
        },
        "hole_count_per_patch": {key: _quantiles(values) for key, values in hole_counts.items()},
        "adjacency_patch_counts": dict(sorted(adjacency_counts.items())),
        "background_components_per_mpx": _quantiles(background_components_per_mpx),
        "background_fraction": _quantiles(background_fractions),
        "background_border_connected_fraction": _quantiles(background_border_fraction),
        "internal_background_fraction": _quantiles(internal_background_fraction),
        "protocol_sources": list(manifest.protocol_sources),
        "review_status": "empirically_validated_pending_internal_review",
    }


def load_id_mask(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        mask = np.load(source, allow_pickle=False)
    else:
        with Image.open(source) as image:
            mask = np.asarray(image)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise RefineContractError(f"mask must be a 2D integer array: {source}")
    return mask


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        return {
            "count": 0,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "mean": None,
        }
    return {
        "count": int(array.size),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(np.mean(array)),
    }


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RefineContractError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RefineContractError("optional URI/digest fields must be non-empty strings")
    return value.strip()
