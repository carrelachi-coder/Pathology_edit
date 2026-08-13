"""Digest-bound reference-shape authority for deterministic cell additions.

Source-mask instances remain the sole authority for baseline counts, density,
occupancy and removals.  A calibrated dataset instance library is a separate,
read-only authority for reusable nucleus footprints and their biological size
ruler when the source provides only a semantic (watershed-reconstructed)
instance mask.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .models import JointContractError
from .nuclei import RAW_TO_INTERNAL

REFERENCE_SHAPE_AUTHORITY_VERSION = "calibrated-reference-shapes-v1"
DEFAULT_REFERENCE_SHAPES_PER_CLASS = 9
_PREFERRED_TISSUE_NAME_BY_CLASS = {
    1: "Tumor",
    2: "Immune infiltrate",
    3: "Stroma",
    4: "Necrosis",
    5: "Normal epithelium",
}


@dataclass(frozen=True)
class ReferenceNucleusShape:
    instance_id: str
    class_id: int
    mask: np.ndarray
    source: str
    area_px: int


@dataclass(frozen=True)
class ReferenceShapeAuthority:
    version: str
    dataset_name: str
    statistics_sha256: str
    authority_sha256: str
    nominal_area_by_class: dict[int, float]
    shapes_by_class: dict[int, tuple[ReferenceNucleusShape, ...]]

    def to_metadata(self) -> dict:
        return {
            "version": self.version,
            "dataset_name": self.dataset_name,
            "statistics_sha256": self.statistics_sha256,
            "authority_sha256": self.authority_sha256,
            "nominal_area_by_class": {
                str(key): value
                for key, value in sorted(self.nominal_area_by_class.items())
            },
            "shapes_by_class": {
                str(class_id): [
                    {
                        "instance_id": item.instance_id,
                        "area_px": item.area_px,
                        "mask_shape": list(item.mask.shape),
                        "mask_sha256": _mask_sha256(item.mask),
                        "source": item.source,
                    }
                    for item in items
                ]
                for class_id, items in sorted(self.shapes_by_class.items())
            },
        }

    def nominal_diameter_px(self, class_ids: tuple[int, ...]) -> float | None:
        areas = [
            self.nominal_area_by_class[class_id]
            for class_id in class_ids
            if class_id in self.nominal_area_by_class
            and self.nominal_area_by_class[class_id] > 0
        ]
        if not areas:
            return None
        return max(3.0, 2.0 * np.sqrt(float(np.median(areas)) / np.pi))


def load_reference_shape_authority(
    library_root: str | Path,
    *,
    dataset_name: str,
    class_ids: tuple[int, ...],
    shapes_per_class: int = DEFAULT_REFERENCE_SHAPES_PER_CLASS,
) -> ReferenceShapeAuthority:
    """Load central complete footprints from the configured dataset library.

    The complete library tree is already frozen and verified by the formal
    runner.  This loader additionally binds the exact statistics file and the
    exact small set of footprints exposed to the compiler/executor.
    """

    root = Path(library_root).resolve()
    statistics_path = root / "statistics.json"
    instances_root = root / "nuclei_instances"
    if not statistics_path.is_file() or not instances_root.is_dir():
        raise JointContractError(
            f"calibrated nucleus instance library is incomplete: {root}"
        )
    statistics_bytes = statistics_path.read_bytes()
    statistics_sha256 = hashlib.sha256(statistics_bytes).hexdigest()
    payload = json.loads(statistics_bytes.decode("utf-8"))
    declared_dataset = str(payload.get("dataset") or dataset_name)
    if declared_dataset.lower() != str(dataset_name).lower():
        raise JointContractError(
            "reference-shape library dataset differs from the population profile"
        )
    statistics = payload.get("statistics", payload)
    if not isinstance(statistics, dict):
        raise JointContractError("reference-shape statistics are malformed")

    selected_by_class: dict[int, tuple[ReferenceNucleusShape, ...]] = {}
    nominal_area_by_class: dict[int, float] = {}
    authority_records: list[dict] = []
    for class_id in tuple(sorted({int(value) for value in class_ids})):
        raw_class_id = 100 + class_id
        preferred_name = _PREFERRED_TISSUE_NAME_BY_CLASS.get(class_id)
        tissue_candidates = []
        for tissue_id, info in statistics.items():
            if not isinstance(info, dict):
                continue
            type_info = (info.get("nuclei_types") or {}).get(
                str(raw_class_id), {}
            )
            stored = int(type_info.get("stored_count", 0) or 0)
            mean_area = float(type_info.get("mean_area", 0.0) or 0.0)
            if stored <= 0 or mean_area <= 0:
                continue
            tissue_candidates.append(
                (
                    0 if info.get("name") == preferred_name else 1,
                    -stored,
                    int(tissue_id),
                    mean_area,
                )
            )
        if not tissue_candidates:
            continue
        tissue_candidates.sort()
        _, _, tissue_id, nominal_area = tissue_candidates[0]
        nominal_area_by_class[class_id] = nominal_area
        bucket_matches = tuple(
            sorted(instances_root.glob(f"tissue_{tissue_id:02d}_*"))
        )
        if len(bucket_matches) != 1:
            raise JointContractError(
                f"reference-shape bucket is ambiguous for tissue {tissue_id}"
            )
        candidates = []
        for path in sorted(bucket_matches[0].glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as record:
                    raw_type = int(record["type"])
                    resolved_class = RAW_TO_INTERNAL.get(raw_type)
                    if resolved_class != class_id:
                        continue
                    mask = np.ascontiguousarray(record["mask"], dtype=bool)
                    area = int(record["area"])
            except (KeyError, OSError, ValueError):
                continue
            if (
                mask.ndim != 2
                or not np.any(mask)
                or int(np.count_nonzero(mask)) != area
                or ndimage.label(
                    mask, structure=np.ones((3, 3), dtype=np.uint8)
                )[1]
                != 1
            ):
                continue
            relative = path.relative_to(root).as_posix()
            mask_digest = _mask_sha256(mask)
            candidates.append(
                (
                    abs(area - nominal_area),
                    area,
                    relative,
                    mask_digest,
                    mask,
                )
            )
        candidates.sort(key=lambda item: item[:4])
        selected = []
        # Deterministic central shapes avoid a single giant semantic component
        # becoming the universal containment margin while retaining genuine
        # morphology around the dataset-calibrated size ruler.
        for _, area, relative, mask_digest, mask in candidates[
            : max(1, int(shapes_per_class))
        ]:
            instance_id = (
                f"library:{declared_dataset}:{relative}:{mask_digest[:12]}"
            )
            selected.append(
                ReferenceNucleusShape(
                    instance_id=instance_id,
                    class_id=class_id,
                    mask=mask,
                    source="calibrated_dataset_instance_library",
                    area_px=area,
                )
            )
            authority_records.append(
                {
                    "instance_id": instance_id,
                    "class_id": class_id,
                    "area_px": area,
                    "mask_sha256": mask_digest,
                }
            )
        if selected:
            selected_by_class[class_id] = tuple(selected)

    if not selected_by_class:
        raise JointContractError(
            "calibrated nucleus instance library has no requested class shapes"
        )
    digest_payload = {
        "version": REFERENCE_SHAPE_AUTHORITY_VERSION,
        "dataset_name": declared_dataset,
        "statistics_sha256": statistics_sha256,
        "nominal_area_by_class": nominal_area_by_class,
        "selected_shapes": authority_records,
    }
    authority_sha256 = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ReferenceShapeAuthority(
        version=REFERENCE_SHAPE_AUTHORITY_VERSION,
        dataset_name=declared_dataset,
        statistics_sha256=statistics_sha256,
        authority_sha256=authority_sha256,
        nominal_area_by_class=nominal_area_by_class,
        shapes_by_class=selected_by_class,
    )


def _mask_sha256(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()
