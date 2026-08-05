"""Read-only, cohort-level annotation-profile statistics builder."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from phase3_mask_edit_refine.evidence import load_id_mask

from .models import JointContractError
from .skills.repository import JointSkillRepository


STATISTICS_SCHEMA_VERSION = "joint-annotation-profile-statistics-v1"


def build_annotation_profile_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    annotation_profile_id: str,
    data_revision: str,
    evidence_manifest_sha256: str,
    repository: JointSkillRepository | None = None,
) -> dict[str, Any]:
    """Compute auditable mask statistics without modifying source datasets."""

    if not records:
        raise JointContractError("annotation statistics require at least one case")
    repository = repository or JointSkillRepository()
    schema = repository.annotation_schema(annotation_profile_id)
    profile = repository.annotation_profiles[annotation_profile_id]
    patient_splits: dict[str, set[str]] = defaultdict(set)
    wsi_splits: dict[str, set[str]] = defaultdict(set)
    patch_coverages: dict[int, list[float]] = defaultdict(list)
    component_areas: dict[int, list[int]] = defaultdict(list)
    component_holes: dict[int, list[int]] = defaultdict(list)
    component_compactness: dict[int, list[float]] = defaultdict(list)
    adjacency_pixels: Counter[tuple[int, int]] = Counter()
    shapes: Counter[str] = Counter()
    background_component_density = []
    internal_background_fraction = []
    internal_background_components = []
    observed_ids: set[int] = set()
    case_digests = []

    declared_ids = {
        int(value)
        for values in schema.label_to_fine_ids.values()
        for value in values
    }.union(int(value) for value in schema.skip_fine_ids)
    for case_index, raw in enumerate(records):
        case_id = _required_string(raw, "case_id")
        patient_id = _required_string(raw, "patient_id")
        wsi_id = _required_string(raw, "wsi_id")
        split = _required_string(raw, "split")
        path = Path(_required_string(raw, "mask_uri"))
        if not path.is_file():
            raise JointContractError(f"statistics mask does not exist: {path}")
        declared_digest = _required_string(raw, "mask_sha256")
        actual_digest = _sha256(path)
        if actual_digest != declared_digest:
            raise JointContractError(f"statistics mask digest mismatch: {case_id}")
        patient_splits[patient_id].add(split)
        wsi_splits[wsi_id].add(split)
        case_digests.append((case_id, actual_digest))
        mask = load_id_mask(path)
        if mask.ndim != 2 or not mask.size:
            raise JointContractError(f"statistics mask is not a non-empty 2-D array: {case_id}")
        height, width = mask.shape
        shapes[f"{height}x{width}"] += 1
        ids, counts = np.unique(mask, return_counts=True)
        current_ids = {int(value) for value in ids}
        for fine_id in current_ids - observed_ids - declared_ids:
            patch_coverages[fine_id].extend([0.0] * case_index)
        observed_ids.update(current_ids)
        total = float(mask.size)
        count_by_id = {int(value): int(count) for value, count in zip(ids, counts)}
        for fine_id in observed_ids.union(declared_ids):
            patch_coverages[fine_id].append(count_by_id.get(fine_id, 0) / total)
        _accumulate_adjacency(mask, adjacency_pixels)
        for fine_id in ids:
            fine_id = int(fine_id)
            region = mask == fine_id
            labels, count = ndimage.label(region, structure=np.ones((3, 3), dtype=bool))
            for component_id in range(1, count + 1):
                component = labels == component_id
                area = int(np.count_nonzero(component))
                boundary = component & ~ndimage.binary_erosion(component)
                perimeter = int(np.count_nonzero(boundary))
                filled = ndimage.binary_fill_holes(component)
                holes = int(
                    ndimage.label(filled & ~component, structure=np.ones((3, 3), dtype=bool))[1]
                )
                component_areas[fine_id].append(area)
                component_holes[fine_id].append(holes)
                component_compactness[fine_id].append(
                    float(perimeter * perimeter / max(1.0, 4.0 * np.pi * area))
                )
        background = np.isin(mask, tuple(profile.prohibited_fine_ids))
        labels, count = ndimage.label(background, structure=np.ones((3, 3), dtype=bool))
        internal_pixels = 0
        internal_count = 0
        for component_id in range(1, count + 1):
            component = labels == component_id
            touches = (
                np.any(component[0])
                or np.any(component[-1])
                or np.any(component[:, 0])
                or np.any(component[:, -1])
            )
            if not touches:
                internal_pixels += int(np.count_nonzero(component))
                internal_count += 1
        background_component_density.append(count * 1_000_000.0 / total)
        internal_background_fraction.append(internal_pixels / total)
        internal_background_components.append(internal_count)

    leaking_patients = sorted(key for key, value in patient_splits.items() if len(value) > 1)
    leaking_wsis = sorted(key for key, value in wsi_splits.items() if len(value) > 1)
    if leaking_patients or leaking_wsis:
        raise JointContractError(
            "patient/WSI split leakage in evidence manifest: "
            f"patients={leaking_patients[:5]}, wsis={leaking_wsis[:5]}"
        )
    all_ids = sorted(observed_ids.union(declared_ids))
    canonical = {
        label: list(ids) for label, ids in sorted(schema.label_to_fine_ids.items())
    }
    return {
        "schema_version": STATISTICS_SCHEMA_VERSION,
        "annotation_profile_id": annotation_profile_id,
        "data_revision": data_revision,
        "evidence_manifest_sha256": evidence_manifest_sha256,
        "case_set_digest": _digest_pairs(case_digests),
        "review_status": "draft",
        "sample_counts": {
            "patches": len(records),
            "patients": len(patient_splits),
            "wsis": len(wsi_splits),
            "splits": dict(sorted(Counter(_required_string(item, "split") for item in records).items())),
            "shapes": dict(sorted(shapes.items())),
        },
        "ontology": {
            "canonical_to_fine_ids": canonical,
            "skip_or_prohibited_fine_ids": sorted(profile.prohibited_fine_ids),
            "observed_fine_ids": all_ids,
            "semantic_warnings": schema.semantic_warnings,
        },
        "per_fine_id": {
            str(fine_id): {
                "patch_coverage_fraction": _quantiles(patch_coverages[fine_id]),
                "component_area_px": _quantiles(component_areas[fine_id]),
                "holes_per_component": _quantiles(component_holes[fine_id]),
                "boundary_compactness": _quantiles(component_compactness[fine_id]),
                "component_count": len(component_areas[fine_id]),
            }
            for fine_id in all_ids
        },
        "adjacency_contact_pixels": {
            f"{left}->{right}": count
            for (left, right), count in sorted(adjacency_pixels.items())
        },
        "background_fragmentation": {
            "components_per_million_pixels": _quantiles(background_component_density),
            "internal_background_fraction": _quantiles(internal_background_fraction),
            "internal_component_count": _quantiles(internal_background_components),
        },
        "split_leakage_audit": {
            "patient_leakage_count": 0,
            "wsi_leakage_count": 0,
        },
        "known_limitations": [
            "statistics describe annotation geometry, not unlabelled biological truth",
            "curvature and physical-distance thresholds require pixel_size_um calibration",
            "draft output requires dataset-owner and pathology review before production",
        ],
    }


def _accumulate_adjacency(mask: np.ndarray, output: Counter[tuple[int, int]]) -> None:
    for first, second in ((mask[:, :-1], mask[:, 1:]), (mask[:-1, :], mask[1:, :])):
        changed = first != second
        left = first[changed].astype(int)
        right = second[changed].astype(int)
        for one, two in zip(left, right):
            a, b = sorted((int(one), int(two)))
            output[(a, b)] += 1


def _quantiles(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p05": None, "p50": None, "p95": None, "mean": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "mean": float(np.mean(array)),
    }


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JointContractError(f"statistics record {key} must be a non-empty string")
    return value.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_pairs(values: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for case_id, value in sorted(values):
        digest.update(case_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
