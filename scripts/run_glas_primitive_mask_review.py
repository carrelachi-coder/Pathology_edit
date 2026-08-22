#!/usr/bin/env python3
"""Generate five reviewed GLaS mask edits for every joint primitive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.authority import (
    summarize_nucleus_instance_authority,
    validate_nucleus_authority_floor,
)
from phase3_joint_edit_refine.auxiliary import materialize_profile_auxiliaries
from phase3_joint_edit_refine.models import CellCountExtentBudget, JointCaseContext
from phase3_joint_edit_refine.nuclei import (
    _local_contour,
    _semantic_instance_labels,
    iter_instances,
    load_native_instances,
    load_nuclei_mask,
    touches_border,
)
from phase3_joint_edit_refine.semantic_parser import RuleBasedSemanticParser
from phase3_joint_edit_refine.visualization import NUCLEI_RGB
from phase3_mask_edit_refine.evidence import load_id_mask, sha256_file
from phase3_mask_edit_refine.visualization import id_mask_to_rgb

SCHEMA_VERSION = "glas-cross-meta-primitive-mask-review-v3"
NATIVE_PARTITION_ALGORITHM = (
    "cellvit-clipped-semantic-exact-raster-partition-v3"
)
GRADE_BY_FINE_ID = {
    5: "normal",
    11: "adenomatous",
    12: "moderately_differentiated",
    13: "poorly_differentiated",
}
GLAND_FINE_IDS = tuple(GRADE_BY_FINE_ID)
MALIGNANT_GLAND_FINE_IDS = (11, 12, 13)
PERIGLANDULAR_PRIMITIVES = frozenset(
    {
        "peritumoral-neoplastic-scatter-increase-v1",
        "peritumoral-small-cluster-increase-v1",
    }
)

# Post-7c6 review budgets tune only count and spatial extent; add/remove/layout
# semantics are unchanged. Depletion keeps the primitive contract's twelve-cell
# floor while allowing the compiler to choose a larger safe local population.
MASK_REVIEW_CELL_BUDGETS = {
    "cell-type-abundance-increase-v1": CellCountExtentBudget(
        20, 12, 28, 384, 0, 48, 0, 0
    ),
    "cell-type-abundance-decrease-v1": CellCountExtentBudget(
        16, 12, 24, 384, 0, 48, 0, 0
    ),
    "cellularity-increase-v1": CellCountExtentBudget(
        20, 12, 28, 384, 0, 48, 0, 0
    ),
    "cellularity-decrease-v1": CellCountExtentBudget(
        16, 12, 24, 384, 0, 48, 0, 0
    ),
    "neoplastic-cell-abundance-increase-v1": CellCountExtentBudget(
        20, 12, 28, 384, 0, 48, 0, 0
    ),
    "neoplastic-cell-abundance-decrease-v1": CellCountExtentBudget(
        24, 12, 36, 384, 0, 48, 0, 0
    ),
    "peritumoral-neoplastic-scatter-increase-v1": CellCountExtentBudget(
        10, 4, 14, 144, 4, 48, 32, 4
    ),
    "peritumoral-small-cluster-increase-v1": CellCountExtentBudget(
        8, 6, 12, 160, 4, 48, 32, 2
    ),
}


@dataclass(frozen=True)
class Evaluation:
    mechanism_id: str
    primitive_id: str
    instruction: str


EVALUATIONS = (
    Evaluation(
        "colorectal-local-population-modulation",
        "cell-type-abundance-increase-v1",
        "Increase immune cells in the selected region.",
    ),
    Evaluation(
        "colorectal-local-population-modulation",
        "cell-type-abundance-decrease-v1",
        "Decrease immune cells in the selected region.",
    ),
    Evaluation(
        "colorectal-local-population-modulation",
        "cellularity-increase-v1",
        "Increase local cellularity.",
    ),
    Evaluation(
        "colorectal-local-population-modulation",
        "cellularity-decrease-v1",
        "Decrease local cellularity.",
    ),
    Evaluation(
        "colorectal-local-population-modulation",
        "neoplastic-cell-abundance-increase-v1",
        "Increase neoplastic cells.",
    ),
    Evaluation(
        "colorectal-local-population-modulation",
        "neoplastic-cell-abundance-decrease-v1",
        "Decrease neoplastic cells.",
    ),
    Evaluation(
        "colorectal-tumor-budding-front",
        "peritumoral-neoplastic-scatter-increase-v1",
        "Add scattered tumor cells near the tumor boundary.",
    ),
    Evaluation(
        "colorectal-tumor-budding-front",
        "peritumoral-small-cluster-increase-v1",
        "Add peritumoral small tumor-cell clusters.",
    ),
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _cross_meta_targets(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs") if isinstance(payload, dict) else None
    if not isinstance(pairs, list):
        raise TypeError("cross-meta JSON must contain a pairs list")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        if str(row.get("dataset") or "").upper() != "GLAS":
            continue
        grouped[str(row["target_tissue_mask"])].append(row)
    records = []
    for tissue_path, rows in sorted(grouped.items()):
        first = rows[0]
        image = Path(str(first["target_image"]))
        tissue = Path(tissue_path)
        nuclei = Path(str(first["target_nuclei_mask"]))
        for asset in (image, tissue, nuclei):
            if not asset.is_file():
                raise FileNotFoundError(f"cross-meta GLaS asset is missing: {asset}")
        records.append(
            {
                "sample_id": str(first["sample_id"]),
                "cross_meta_case_id": str(first["case_id"]),
                "source_image": str(image.resolve()),
                "source_tissue_mask": str(tissue.resolve()),
                "source_nuclei_mask": str(nuclei.resolve()),
                "cross_meta_pair_count": len(rows),
                "cross_meta_reference_sample_ids": sorted(
                    {str(item.get("reference_sample_id")) for item in rows}
                ),
            }
        )
    if len(records) < 5:
        raise ValueError("cross-meta contains fewer than five unique GLaS targets")
    return records


def _candidate_metrics(row: dict[str, Any]) -> dict[str, Any]:
    tissue = load_id_mask(row["source_tissue_mask"])
    nuclei = load_nuclei_mask(row["source_nuclei_mask"])
    if tissue.shape != nuclei.shape:
        raise ValueError(f"tissue/nuclei shape mismatch: {row['sample_id']}")
    gland = np.isin(tissue, GLAND_FINE_IDS)
    malignant = np.isin(tissue, MALIGNANT_GLAND_FINE_IDS)
    distance = ndimage.distance_transform_edt(~gland)
    annulus = (tissue == 2) & (distance >= 4) & (distance <= 48)
    occupied = ndimage.binary_dilation(nuclei > 0, iterations=3)
    counts = Counter()
    gland_counts = Counter()
    malignant_counts = Counter()
    stroma_counts = Counter()
    centroids: dict[int, list[tuple[float, float]]] = defaultdict(list)
    areas: dict[int, list[int]] = defaultdict(list)
    for _, class_id, component in iter_instances(nuclei):
        if touches_border(component):
            continue
        area = int(np.count_nonzero(component))
        if not area:
            continue
        counts[int(class_id)] += 1
        areas[int(class_id)].append(area)
        cy, cx = ndimage.center_of_mass(component)
        centroids[int(class_id)].append((float(cx), float(cy)))
        if np.count_nonzero(component & gland) >= max(1, int(area * 0.5)):
            gland_counts[int(class_id)] += 1
        if np.count_nonzero(component & malignant) >= max(1, int(area * 0.5)):
            malignant_counts[int(class_id)] += 1
        if np.count_nonzero(component & (tissue == 2)) >= max(1, int(area * 0.5)):
            stroma_counts[int(class_id)] += 1
    class_spans = {
        class_id: _point_span(np.asarray(points, dtype=float))
        for class_id, points in centroids.items()
    }
    class_median_diameters = {
        class_id: float(np.sqrt(4.0 * np.median(values) / np.pi))
        for class_id, values in areas.items()
    }
    class_local_counts = {
        class_id: _maximum_local_count(
            np.asarray(points, dtype=float),
            radius_px=6.0 * class_median_diameters[class_id],
        )
        for class_id, points in centroids.items()
    }
    all_points = np.asarray(
        [point for values in centroids.values() for point in values], dtype=float
    )
    all_areas = [area for values in areas.values() for area in values]
    all_median_diameter = (
        float(np.sqrt(4.0 * np.median(all_areas) / np.pi))
        if all_areas
        else 0.0
    )
    grade_ids = sorted(
        int(value) for value in np.unique(tissue) if int(value) in GRADE_BY_FINE_ID
    )
    return {
        **row,
        "shape": list(tissue.shape),
        "grade_fine_ids": grade_ids,
        "gland_pixels": int(np.count_nonzero(gland)),
        "malignant_gland_pixels": int(np.count_nonzero(malignant)),
        "gland_free_pixels": int(np.count_nonzero(gland & ~occupied)),
        "malignant_gland_free_pixels": int(
            np.count_nonzero(malignant & ~occupied)
        ),
        "stroma_free_pixels": int(np.count_nonzero((tissue == 2) & ~occupied)),
        "annulus_pixels": int(np.count_nonzero(annulus)),
        "annulus_free_pixels": int(np.count_nonzero(annulus & ~occupied)),
        "complete_instance_counts": {
            str(key): int(value) for key, value in sorted(counts.items())
        },
        "gland_instance_counts": {
            str(key): int(value) for key, value in sorted(gland_counts.items())
        },
        "malignant_gland_instance_counts": {
            str(key): int(value)
            for key, value in sorted(malignant_counts.items())
        },
        "stroma_instance_counts": {
            str(key): int(value) for key, value in sorted(stroma_counts.items())
        },
        "class_spans_px": {
            str(key): round(float(value), 4)
            for key, value in sorted(class_spans.items())
        },
        "class_median_diameter_px": {
            str(key): round(float(value), 4)
            for key, value in sorted(class_median_diameters.items())
        },
        "class_local_count_radius_6d_max": {
            str(key): int(value)
            for key, value in sorted(class_local_counts.items())
        },
        "all_median_diameter_px": round(all_median_diameter, 4),
        "all_local_count_radius_6d_max": _maximum_local_count(
            all_points,
            radius_px=6.0 * all_median_diameter,
        ),
    }


def _point_span(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    seed = points[0]
    first = points[int(np.argmax(np.sum((points - seed) ** 2, axis=1)))]
    return float(np.sqrt(np.max(np.sum((points - first) ** 2, axis=1))))


def _maximum_local_count(points: np.ndarray, *, radius_px: float) -> int:
    if not len(points) or radius_px <= 0:
        return 0
    deltas = points[:, None, :] - points[None, :, :]
    return int(np.max(np.sum(np.sum(deltas**2, axis=2) <= radius_px**2, axis=1)))


def _count(row: dict[str, Any], field: str, class_id: int | None = None) -> int:
    values = row[field]
    if class_id is None:
        return sum(int(value) for value in values.values())
    return int(values.get(str(class_id), 0))


def _eligible_and_score(
    primitive_id: str, row: dict[str, Any]
) -> tuple[bool, float]:
    total = _count(row, "complete_instance_counts")
    immune = _count(row, "complete_instance_counts", 2)
    neoplastic = _count(row, "malignant_gland_instance_counts", 1)
    span_all = max((float(v) for v in row["class_spans_px"].values()), default=0.0)
    span_immune = float(row["class_spans_px"].get("2", 0.0))
    local_by_class = row["class_local_count_radius_6d_max"]
    immune_local = int(local_by_class.get("2", 0))
    neoplastic_local = int(local_by_class.get("1", 0))
    if primitive_id == "cell-type-abundance-increase-v1":
        return immune >= 4, row["stroma_free_pixels"] + 160 * immune - 22 * total
    if primitive_id == "cell-type-abundance-decrease-v1":
        # This inexpensive screen observes semantic connected components, while
        # the final compiler counts exact CellViT-native instances. Dense immune
        # fields often merge semantically, so keep the screen permissive and let
        # the 12-instance depletion contract remain the hard authority.
        return immune >= 1, (
            5000 * immune_local + 500 * immune + 20 * span_immune
        )
    if primitive_id == "cellularity-increase-v1":
        diameters = [
            float(value)
            for class_id, value in row["class_median_diameter_px"].items()
            if _count(row, "complete_instance_counts", int(class_id)) >= 4
        ]
        minimum_diameter = min(diameters, default=np.inf)
        scale_capacity = (
            (row["stroma_free_pixels"] + row["gland_free_pixels"])
            / max(1.0, minimum_diameter**2)
        )
        return total >= 12 and np.isfinite(minimum_diameter), (
            1_000_000 / minimum_diameter
            + 250 * row["all_local_count_radius_6d_max"]
            + scale_capacity
        )
    if primitive_id == "cellularity-decrease-v1":
        return total >= 40 and span_all >= 48, (
            4000 * row["all_local_count_radius_6d_max"]
            + 300 * total
            + 10 * span_all
        )
    if primitive_id == "neoplastic-cell-abundance-increase-v1":
        return neoplastic >= 4 and row["malignant_gland_pixels"] > 0, (
            row["malignant_gland_free_pixels"] + 350 * neoplastic
        )
    if primitive_id == "neoplastic-cell-abundance-decrease-v1":
        return neoplastic >= 12, (
            5000 * neoplastic_local
            + 500 * neoplastic
            + row["malignant_gland_pixels"]
        )
    if primitive_id in PERIGLANDULAR_PRIMITIVES:
        return neoplastic >= 6 and row["annulus_free_pixels"] >= 6144, (
            row["annulus_free_pixels"] + 400 * neoplastic
        )
    raise ValueError(f"unsupported GLaS primitive: {primitive_id}")


def _diverse_ranked(
    primitive_id: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        eligible, score = _eligible_and_score(primitive_id, row)
        if eligible:
            ranked.append({**row, "selection_score": round(float(score), 4)})
    ranked.sort(key=lambda item: (-item["selection_score"], item["sample_id"]))
    # Capacity is patch-local. Interleaving source cases pushed the second
    # high-capacity patch from rank 2 to rank 8 and made sparse depletion
    # candidates consume most of the execution window. Cross-meta target
    # deduplication already guarantees distinct images, so preserve strict
    # executable-capacity order here.
    return ranked


def _grade(tissue: np.ndarray) -> str:
    values = sorted(
        int(value) for value in np.unique(tissue) if int(value) in GRADE_BY_FINE_ID
    )
    if len(values) != 1:
        raise ValueError(f"GLaS patch must contain one grade fine ID, got {values}")
    return GRADE_BY_FINE_ID[values[0]]


def _semantic_intent(evaluation: Evaluation) -> dict[str, Any]:
    intent = RuleBasedSemanticParser().parse(evaluation.instruction).to_metadata()
    hypotheses = {
        str(item["primitive_id"]) for item in intent["primitive_hypotheses"]
    }
    if evaluation.primitive_id not in hypotheses:
        raise ValueError(
            f"instruction does not bind {evaluation.primitive_id}: {hypotheses}"
        )
    intent["selected_primitive_id"] = evaluation.primitive_id
    return intent


def _native_authority(
    row: dict[str, Any],
    *,
    output_root: Path,
    cellvit_model: Path,
    cellvit_root: Path,
    cellvit_python: Path,
    gpu: int,
    timeout_seconds: int,
    cache: dict[str, Any],
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    cached = cache.get(sample_id)
    if (
        isinstance(cached, dict)
        and Path(str(cached.get("cells_json"))).is_file()
        and (cached.get("class_binding") or {}).get("algorithm")
        == NATIVE_PARTITION_ALGORITHM
    ):
        return cached
    root = output_root / "native_authority" / sample_id
    output_mask = root / "cellvit_native_mask.png"
    summary_path = output_mask.with_suffix(".cellvit_single_patch.json")
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_cellvit_single_patch.py"),
        "--image",
        str(row["source_image"]),
        "--output-mask",
        str(output_mask),
        "--model",
        str(cellvit_model),
        "--cellvit-root",
        str(cellvit_root),
        "--cellvit-python",
        str(cellvit_python),
        "--raw-outdir",
        str(root / "raw"),
        "--gpu",
        str(gpu),
        "--batch-size",
        "8",
        "--mpp",
        "0.25",
        "--magnification",
        "40",
        "--resolution",
        "0.25",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "cellvit_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (root / "cellvit_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode or not summary_path.is_file():
        raise RuntimeError(
            f"CellViT failed for {sample_id}: return_code={completed.returncode}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_cells = Path(str(summary["cells_json"]))
    semantic = Path(str(row["source_nuclei_mask"]))
    bound_cells = root / "semantic_bound_native_cells.json"
    binding = _bind_native_geometry(
        raw_cells_json=raw_cells,
        semantic_path=semantic,
        output_path=bound_cells,
    )
    validation = _validate_native_authority(bound_cells, semantic)
    if validation["status"] != "verified":
        raise RuntimeError(
            f"native instance authority rejected for {sample_id}: "
            + ", ".join(validation["failure_codes"])
        )
    record = {
        "status": "verified",
        "cells_json": str(bound_cells.resolve()),
        "cells_json_sha256": sha256_file(bound_cells),
        "raw_cells_json": str(raw_cells.resolve()),
        "raw_cells_json_sha256": sha256_file(raw_cells),
        "cellvit_mask": str(output_mask.resolve()),
        "cellvit_mask_sha256": sha256_file(output_mask),
        "class_binding": binding,
        "validation": validation,
    }
    cache[sample_id] = record
    return record


def _bind_native_geometry(
    *, raw_cells_json: Path, semantic_path: Path, output_path: Path
) -> dict[str, Any]:
    payload = json.loads(raw_cells_json.read_text(encoding="utf-8"))
    raw_cells = payload.get("cells") if isinstance(payload, dict) else None
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError("CellViT native JSON contains no cells")
    semantic = load_nuclei_mask(semantic_path)
    metadata = payload.get("wsi_metadata") or {}
    accepted = []
    accepted_components = []
    rejected = []
    transitions = Counter()
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, dict):
            continue
        contour = raw.get("contour")
        if not isinstance(contour, list) or len(contour) < 3:
            continue
        points = _local_contour(
            contour, info=raw, metadata=metadata, shape=semantic.shape
        )
        if len(points) < 3:
            continue
        canvas = Image.new("1", (semantic.shape[1], semantic.shape[0]), 0)
        ImageDraw.Draw(canvas).polygon(points, fill=1)
        component = np.asarray(canvas, dtype=bool)
        counts = np.bincount(semantic[component], minlength=6)
        class_id = int(np.argmax(counts[1:6]) + 1)
        agreement = float(counts[class_id] / max(np.count_nonzero(component), 1))
        if agreement < 0.80:
            rejected.append(
                {
                    "index": index,
                    "best_class_id": class_id,
                    "best_agreement": round(agreement, 8),
                }
            )
            continue
        original = int(raw.get("type", 0))
        transitions[f"{original}->{class_id}"] += 1
        accepted.append({**raw, "type": class_id})
        accepted_components.append((class_id, component))
    if not accepted:
        raise ValueError("no CellViT geometry agrees with frozen semantic classes")
    label_map, instance_records, native_seed_count = _semantic_partition_from_native_seeds(
        semantic,
        accepted_components,
    )
    label_map_path = output_path.with_suffix(".instance_labels.npy")
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(label_map_path, label_map, allow_pickle=False)
    output = {
        "raster_instance_authority": {
            "label_map_uri": str(label_map_path.resolve()),
            "label_map_sha256": sha256_file(label_map_path),
            "instances": instance_records,
        },
        "wsi_metadata": metadata,
        "authority": {
            "algorithm": NATIVE_PARTITION_ALGORITHM,
            "minimum_per_contour_class_agreement": 0.80,
            "raw_cells_json": str(raw_cells_json.resolve()),
            "source_semantic_nuclei_mask": str(semantic_path.resolve()),
            "semantic_foreground_exactly_partitioned": True,
        },
    }
    _write_json(output_path, output)
    return {
        "algorithm": output["authority"]["algorithm"],
        "raw_instance_count": len(raw_cells),
        "accepted_instance_count": len(accepted),
        "native_seed_count": native_seed_count,
        "partition_instance_count": len(instance_records),
        "rejected_instance_count": len(rejected),
        "original_to_bound_class_counts": dict(sorted(transitions.items())),
        "rejected_instances": rejected,
    }


def _semantic_partition_from_native_seeds(
    semantic: np.ndarray,
    accepted_components: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, int | str]], int]:
    """Preserve clipped CellViT shapes and partition all residual semantics."""

    semantic = np.asarray(semantic, dtype=np.uint8)
    label_map = np.zeros(semantic.shape, dtype=np.int32)
    records: list[dict[str, int | str]] = []
    next_label = 0
    native_seed_count = 0
    for class_id, component in accepted_components:
        clipped = (
            np.asarray(component, dtype=bool)
            & (semantic == int(class_id))
            & (label_map == 0)
        )
        if np.count_nonzero(clipped) < 3:
            continue
        next_label += 1
        label_map[clipped] = next_label
        records.append(
            {"label_id": next_label, "type": int(class_id), "seed_source": "cellvit"}
        )
        native_seed_count += 1
    for class_id in range(1, 6):
        semantic_instances, _ = _semantic_instance_labels(
            semantic == class_id
        )
        seeded_semantic_ids = {
            int(value)
            for value in np.unique(
                semantic_instances[
                    (semantic == class_id) & (label_map > 0)
                ]
            )
            if int(value) > 0
        }
        residual = (semantic == class_id) & (label_map == 0)
        fallback, fallback_count = _semantic_instance_labels(residual)
        for fallback_id in range(1, fallback_count + 1):
            component = fallback == fallback_id
            if not np.any(component):
                continue
            next_label += 1
            label_map[component] = next_label
            component_semantic_ids = {
                int(value)
                for value in np.unique(semantic_instances[component])
                if int(value) > 0
            }
            seed_source = (
                "semantic_seeded_residual"
                if component_semantic_ids & seeded_semantic_ids
                else "semantic_unseeded"
            )
            records.append(
                {
                    "label_id": next_label,
                    "type": class_id,
                    "seed_source": seed_source,
                }
            )
    if not np.array_equal(label_map > 0, semantic > 0):
        raise ValueError("CellViT-seeded partition does not cover semantic foreground")
    return label_map, records, native_seed_count


def _validate_native_authority(cells_json: Path, semantic_path: Path) -> dict[str, Any]:
    semantic = load_nuclei_mask(semantic_path)
    instances = load_native_instances(
        cells_json, shape=semantic.shape, semantic_mask=semantic
    )
    raster = np.zeros(semantic.shape, dtype=np.uint8)
    counts = Counter()
    for _, class_id, component in instances:
        raster[component] = int(class_id)
        counts[int(class_id)] += 1
    native = raster > 0
    observed = semantic > 0
    intersection = int(np.count_nonzero(native & observed))
    native_pixels = int(np.count_nonzero(native))
    observed_pixels = int(np.count_nonzero(observed))
    precision = intersection / native_pixels if native_pixels else 0.0
    recall = intersection / observed_pixels if observed_pixels else 0.0
    dice = (
        2 * intersection / (native_pixels + observed_pixels)
        if native_pixels + observed_pixels
        else 0.0
    )
    agreement = float(np.mean(raster[native] == semantic[native])) if native_pixels else 0.0
    authority_summary = summarize_nucleus_instance_authority(instances)
    failures = []
    thresholds = {
        "minimum_instance_count": 16,
        "minimum_foreground_precision": 0.85,
        "minimum_foreground_recall": 0.75,
        "minimum_foreground_dice": 0.78,
        "minimum_partition_pixel_class_agreement": 0.90,
        "minimum_native_seed_count": 4,
        "minimum_native_seed_instance_fraction": 0.05,
        "minimum_native_seed_pixel_fraction": 0.05,
    }
    metrics = {
        "partition_instance_count": len(instances),
        "partition_class_counts": {
            str(class_id): int(counts[class_id]) for class_id in range(1, 6)
        },
        "partition_foreground_pixels": native_pixels,
        "semantic_foreground_pixels": observed_pixels,
        "foreground_precision": round(precision, 8),
        "foreground_recall": round(recall, 8),
        "foreground_dice": round(dice, 8),
        "partition_pixel_class_agreement": round(agreement, 8),
        **authority_summary,
    }
    if len(instances) < thresholds["minimum_instance_count"]:
        failures.append("partition_instance_count_below_threshold")
    for key, value in (
        ("foreground_precision", precision),
        ("foreground_recall", recall),
        ("foreground_dice", dice),
        ("partition_pixel_class_agreement", agreement),
    ):
        if value < thresholds[f"minimum_{key}"]:
            failures.append(f"{key}_below_threshold")
    failures.extend(
        validate_nucleus_authority_floor(
            authority_summary,
            minimum_native_seed_count=thresholds[
                "minimum_native_seed_count"
            ],
            minimum_native_seed_instance_fraction=thresholds[
                "minimum_native_seed_instance_fraction"
            ],
            minimum_native_seed_pixel_fraction=thresholds[
                "minimum_native_seed_pixel_fraction"
            ],
        )
    )
    return {
        "status": "verified" if not failures else "rejected",
        "thresholds": thresholds,
        "metrics": metrics,
        "failure_codes": failures,
        "he_consumed_by": "frozen_cellvit_only",
        "llm_api_used": False,
    }


def _validate_primitive_native_authority(
    evaluation: Evaluation,
    native: dict[str, Any],
) -> None:
    """Apply primitive-specific morphology floors after generic validation."""

    required_by_class = (
        {1: 4}
        if evaluation.primitive_id
        in {
            "peritumoral-neoplastic-scatter-increase-v1",
            "peritumoral-small-cluster-increase-v1",
        }
        else {}
    )
    metrics = (native.get("validation") or {}).get("metrics") or {}
    reasons = validate_nucleus_authority_floor(
        metrics,
        minimum_native_seed_count=4,
        minimum_native_seed_instance_fraction=0.05,
        minimum_native_seed_pixel_fraction=0.05,
        minimum_native_references_by_class=required_by_class,
    )
    if reasons:
        raise RuntimeError(
            "primitive-specific native nucleus authority rejected: "
            + ", ".join(reasons)
        )


def _build_case(
    evaluation: Evaluation,
    row: dict[str, Any],
    *,
    output_root: Path,
    native: dict[str, Any] | None,
    attempt_index: int,
    portfolio_index: int = 0,
    removal_variant: int = 0,
) -> dict[str, Any]:
    image = Path(str(row["source_image"]))
    tissue_path = Path(str(row["source_tissue_mask"]))
    nuclei = Path(str(row["source_nuclei_mask"]))
    intent = _semantic_intent(evaluation)
    variant_key = (
        f"::portfolio-{portfolio_index}" if portfolio_index else ""
    )
    if removal_variant:
        variant_key += f"::removal-{removal_variant}"
    short = hashlib.sha256(
        f"{evaluation.primitive_id}::{row['sample_id']}{variant_key}".encode()
    ).hexdigest()[:12]
    case_id = f"glas_mv_{evaluation.primitive_id.removesuffix('-v1')}_{short}"
    provenance = {
        "source_image_sha256": sha256_file(image),
        "source_tissue_mask_sha256": sha256_file(tissue_path),
        "source_nuclei_mask_sha256": sha256_file(nuclei),
        "preprocessing_revision": "cross-meta-eval-mask-review-v1",
        "original_label_map_digest": sha256_file(tissue_path),
        "patch_grade": _grade(load_id_mask(tissue_path)),
        "provider": "GLaS",
        "joint_mechanism_id": evaluation.mechanism_id,
        "joint_primitive_id": evaluation.primitive_id,
        "joint_mechanism_assignment_reason": "primitive_specific_mask_capacity_screening",
        "cross_meta_source_authority": str(row["source_tissue_mask"]),
        "cross_meta_sample_id": str(row["sample_id"]),
        "require_mature_probnet_regeneration": True,
    }
    if portfolio_index:
        provenance["cell_portfolio_candidate_index"] = portfolio_index
    if removal_variant:
        provenance["depletion_removal_selection_variant"] = removal_variant
    instances_uri = None
    if native is not None:
        instances_uri = str(native["cells_json"])
        provenance["source_nuclei_instances_sha256"] = str(
            native["cells_json_sha256"]
        )
        authority_metrics = dict(native["validation"]["metrics"])
        provenance["instance_authority_source"] = str(
            authority_metrics["observation_quality"]
        )
        provenance["nucleus_instance_authority_summary"] = (
            authority_metrics
        )
    else:
        provenance["instance_authority_source"] = (
            "semantic_nuclei_mask_distance_watershed_v1"
        )
    case = JointCaseContext(
        case_id=case_id,
        instruction=evaluation.instruction,
        source_image_uri=str(image),
        source_tissue_mask_uri=str(tissue_path),
        source_nuclei_mask_uri=str(nuclei),
        pathology_domain_id="colorectal-adenocarcinoma-v1",
        annotation_profile_id="glas-gland-v1",
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id="colorectal-cellvit-source-first-v1",
        primitive_id=evaluation.primitive_id,
        joint_area_budget=None,
        cell_count_extent_budget=MASK_REVIEW_CELL_BUDGETS[
            evaluation.primitive_id
        ],
        seed=42 + attempt_index,
        provenance=provenance,
        source_nuclei_instances_uri=instances_uri,
        pixel_size_um=0.25,
        semantic_intent=intent,
    )
    case, _produced = materialize_profile_auxiliaries(
        case,
        source_tissue=load_id_mask(tissue_path),
        source_image=np.asarray(Image.open(image).convert("RGB"), dtype=np.uint8),
        source_nuclei=load_nuclei_mask(nuclei),
        output_dir=output_root / "profile_auxiliary" / case_id,
    )
    payload = case.to_metadata()
    payload["prebound_semantic_intent"] = intent
    return payload


def _run_case(
    payload: dict[str, Any],
    *,
    output_root: Path,
    checkpoint: Path,
    library: Path,
    timeout_seconds: int,
    threads: int,
    device: str,
) -> dict[str, Any]:
    case_id = str(payload["case_id"])
    primitive_id = str(payload["primitive_id"])
    case_root = output_root / "runs" / primitive_id / case_id
    manifest = case_root / "manifest.json"
    _write_json(manifest, [payload])
    command = [
        sys.executable,
        "-m",
        "phase3_joint_edit_refine.cli",
        "--manifest",
        str(manifest),
        "--output-root",
        str(case_root),
        "--agent-mode",
        "offline",
        "--semantic-parser",
        "prebound",
        "--cell-executor",
        "mature",
        "--probnet-checkpoint",
        str(checkpoint),
        "--nuclei-instance-library",
        str(library),
        "--probnet-dataset",
        "GlaS",
        "--device",
        str(device),
        "--meta-eval",
    ]
    environment = os.environ.copy()
    for key in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[key] = str(threads)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return_code = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "stdout.log").write_text(stdout, encoding="utf-8")
    (case_root / "stderr.log").write_text(stderr, encoding="utf-8")
    summary_path = case_root / "joint_run_summary.json"
    summary = None
    if summary_path.is_file():
        values = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(values, list) and len(values) == 1:
            summary = values[0]
    status = summary.get("status") if summary else "missing_summary"
    selected_candidate_id = (
        summary.get("selected_candidate_id") if summary else None
    )
    artifact_paths = dict(summary.get("artifact_paths") or {}) if summary else {}
    if status == "review_required" and summary is not None:
        selected_candidate_id = _compiled_review_candidate(
            artifact_paths=artifact_paths,
            abstain_reasons=summary.get("abstain_reasons") or (),
        )
        if selected_candidate_id is not None:
            status = "compiled_pending_visual_review"
    return {
        "case_id": case_id,
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "summary_path": str(summary_path),
        "status": status,
        "selected_candidate_id": selected_candidate_id,
        "abstain_reasons": list(summary.get("abstain_reasons") or ()) if summary else [],
        "artifact_paths": artifact_paths,
    }


def _compiled_review_candidate(
    *, artifact_paths: dict[str, Any], abstain_reasons: Any
) -> str | None:
    """Return one gate-passing candidate for this script's visual review."""

    if list(abstain_reasons) != [
        "independent_mask_condition_critic_approval_required"
    ]:
        return None
    candidates_path = Path(str(artifact_paths.get("candidates.json") or ""))
    gates_path = Path(str(artifact_paths.get("joint_gate_reports.json") or ""))
    critic_path = Path(
        str(
            artifact_paths.get("joint_critic.json")
            or (candidates_path.parent / "joint_critic.json")
        )
    )
    if not all(path.is_file() for path in (candidates_path, gates_path, critic_path)):
        return None
    critic = json.loads(critic_path.read_text(encoding="utf-8"))
    rankings = critic.get("rankings") if isinstance(critic, dict) else None
    if not isinstance(rankings, list) or not rankings:
        return None
    ranking = rankings[0]
    if not isinstance(ranking, dict) or ranking.get("veto_reasons"):
        return None
    candidate_id = str(ranking.get("candidate_id") or "")
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    reports = json.loads(gates_path.read_text(encoding="utf-8"))
    candidate_exists = any(
        isinstance(item, dict) and item.get("candidate_id") == candidate_id
        for item in candidates
    )
    passing_report = next(
        (
            item
            for item in reports
            if isinstance(item, dict) and item.get("candidate_id") == candidate_id
        ),
        None,
    )
    if not candidate_exists or not isinstance(passing_report, dict):
        return None
    failed_hard = [
        item
        for item in passing_report.get("checks", ())
        if isinstance(item, dict)
        and item.get("severity") == "hard"
        and item.get("passed") is not True
    ]
    if passing_report.get("passed") is not True or failed_hard:
        return None
    return candidate_id


def _existing_selected_runs(
    *, output_root: Path, primitive_id: str
) -> dict[str, dict[str, Any]]:
    """Recover gate-passing runs for an explicit continuation batch."""

    recovered = {}
    runs_root = output_root / "runs" / primitive_id
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        summary_path = manifest_path.parent / "joint_run_summary.json"
        if not summary_path.is_file():
            continue
        manifests = json.loads(manifest_path.read_text(encoding="utf-8"))
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        if not (
            isinstance(manifests, list)
            and len(manifests) == 1
            and isinstance(manifests[0], dict)
            and isinstance(summaries, list)
            and len(summaries) == 1
            and isinstance(summaries[0], dict)
        ):
            continue
        payload = manifests[0]
        summary = summaries[0]
        status = str(summary.get("status") or "")
        selected_candidate_id = summary.get("selected_candidate_id")
        artifact_paths = dict(summary.get("artifact_paths") or {})
        if status == "review_required":
            selected_candidate_id = _compiled_review_candidate(
                artifact_paths=artifact_paths,
                abstain_reasons=summary.get("abstain_reasons") or (),
            )
            if selected_candidate_id is not None:
                status = "compiled_pending_visual_review"
        if status not in {"selected_research", "compiled_pending_visual_review"}:
            continue
        sample_id = str(
            (payload.get("provenance") or {}).get("cross_meta_sample_id") or ""
        )
        if not sample_id or selected_candidate_id is None:
            continue
        recovered[sample_id] = {
            "row": None,
            "payload": payload,
            "run": {
                "case_id": payload.get("case_id"),
                "return_code": 0,
                "duration_seconds": 0.0,
                "summary_path": str(summary_path),
                "status": status,
                "selected_candidate_id": str(selected_candidate_id),
                "abstain_reasons": list(summary.get("abstain_reasons") or ()),
                "artifact_paths": artifact_paths,
            },
        }
    return recovered


def _render_board(
    evaluation: Evaluation,
    selected: list[dict[str, Any]],
    *,
    output_path: Path,
    tile_size: int,
) -> list[dict[str, Any]]:
    header = 58
    columns = 5
    canvas = Image.new(
        "RGB", (columns * tile_size, len(selected) * (tile_size + header)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    small = _font(13)
    labels = (
        "SOURCE H&E",
        "SOURCE MASK",
        "TARGET MASK",
        "DELTA",
        "DELTA ZOOM",
    )
    records = []
    for row_index, record in enumerate(selected):
        payload = record["payload"]
        run = record["run"]
        source_tissue = load_id_mask(payload["source_tissue_mask_uri"])
        source_nuclei = load_nuclei_mask(payload["source_nuclei_mask_uri"])
        candidates_path = Path(run["artifact_paths"]["candidates.json"])
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        selected_candidate = next(
            item
            for item in candidates
            if item["candidate_id"] == run["selected_candidate_id"]
        )
        target_tissue = load_id_mask(selected_candidate["target_tissue_mask"])
        target_nuclei = load_nuclei_mask(selected_candidate["target_nuclei_mask"])
        source_image = np.asarray(
            Image.open(payload["source_image_uri"]).convert("RGB"),
            dtype=np.uint8,
        )
        source_view = _mask_composite(source_tissue, source_nuclei)
        target_view = _mask_composite(target_tissue, target_nuclei)
        delta_view, changed = _delta_view(
            source_tissue, source_nuclei, target_tissue, target_nuclei
        )
        zoom = _delta_zoom(delta_view, changed)
        panels = (source_image, source_view, target_view, delta_view, zoom)
        y = row_index * (tile_size + header)
        counts = _change_counts(source_nuclei, target_nuclei)
        draw.text(
            (8, y + 5),
            f"{row_index + 1}. {payload['provenance']['cross_meta_sample_id']} | "
            f"candidate={run['selected_candidate_id']}",
            fill="black",
            font=font,
        )
        draw.text(
            (8, y + 31),
            f"added={counts['added_pixels']} removed={counts['removed_pixels']} "
            f"changed={counts['changed_pixels']}",
            fill=(45, 45, 45),
            font=small,
        )
        for column, (label, panel) in enumerate(zip(labels, panels, strict=True)):
            x = column * tile_size
            resized = Image.fromarray(panel).resize(
                (tile_size, tile_size),
                (
                    Image.Resampling.BILINEAR
                    if column == 0
                    else Image.Resampling.NEAREST
                ),
            )
            canvas.paste(resized, (x, y + header))
            ImageDraw.Draw(canvas).text(
                (x + 7, y + header + 6), label, fill="white", font=small, stroke_width=2, stroke_fill="black"
            )
        records.append(
            {
                "case_id": payload["case_id"],
                "sample_id": payload["provenance"]["cross_meta_sample_id"],
                "selected_candidate_id": run["selected_candidate_id"],
                "source_image": payload["source_image_uri"],
                "source_tissue_mask": payload["source_tissue_mask_uri"],
                "source_nuclei_mask": payload["source_nuclei_mask_uri"],
                "target_tissue_mask": selected_candidate["target_tissue_mask"],
                "target_nuclei_mask": selected_candidate["target_nuclei_mask"],
                "joint_change_mask": selected_candidate["joint_change_mask"],
                "change_counts": counts,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return records


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _mask_composite(tissue: np.ndarray, nuclei: np.ndarray) -> np.ndarray:
    tissue_rgb = id_mask_to_rgb(tissue).astype(float)
    view = np.clip(0.55 * tissue_rgb + 0.45 * 255, 0, 255).astype(np.uint8)
    for class_id, color in NUCLEI_RGB.items():
        view[nuclei == int(class_id)] = np.asarray(color, dtype=np.uint8)
    boundaries = _gland_boundaries(tissue)
    view[boundaries] = np.asarray([255, 255, 255], dtype=np.uint8)
    return view


def _gland_boundaries(tissue: np.ndarray) -> np.ndarray:
    gland = np.isin(tissue, GLAND_FINE_IDS)
    return gland ^ ndimage.binary_erosion(gland, structure=np.ones((3, 3), dtype=bool))


def _delta_view(
    source_tissue: np.ndarray,
    source_nuclei: np.ndarray,
    target_tissue: np.ndarray,
    target_nuclei: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    view = _mask_composite(source_tissue, np.zeros_like(source_nuclei))
    unchanged = (source_nuclei == target_nuclei) & (source_nuclei > 0)
    view[unchanged] = np.asarray([120, 120, 120], dtype=np.uint8)
    added = (target_nuclei != source_nuclei) & (target_nuclei > 0)
    removed = (target_nuclei != source_nuclei) & (source_nuclei > 0)
    tissue_changed = source_tissue != target_tissue
    view[removed] = np.asarray([255, 0, 210], dtype=np.uint8)
    view[added] = np.asarray([0, 255, 80], dtype=np.uint8)
    view[tissue_changed] = np.asarray([0, 220, 255], dtype=np.uint8)
    return view, added | removed | tissue_changed


def _delta_zoom(view: np.ndarray, changed: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(changed)
    if not len(rows):
        return view
    padding = 28
    y0 = max(int(rows.min()) - padding, 0)
    y1 = min(int(rows.max()) + padding + 1, view.shape[0])
    x0 = max(int(cols.min()) - padding, 0)
    x1 = min(int(cols.max()) + padding + 1, view.shape[1])
    return np.asarray(
        Image.fromarray(view[y0:y1, x0:x1]).resize(
            (view.shape[1], view.shape[0]), Image.Resampling.NEAREST
        )
    )


def _change_counts(source: np.ndarray, target: np.ndarray) -> dict[str, int]:
    added = (target != source) & (target > 0)
    removed = (target != source) & (source > 0)
    return {
        "added_pixels": int(np.count_nonzero(added)),
        "removed_pixels": int(np.count_nonzero(removed)),
        "changed_pixels": int(np.count_nonzero(target != source)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    nucleus_instance_source = (
        "semantic-mask" if args.no_cellvit else args.nucleus_instance_source
    )
    if nucleus_instance_source == "cellvit":
        missing = [
            name
            for name, value in (
                ("--cellvit-model", args.cellvit_model),
                ("--cellvit-root", args.cellvit_root),
                ("--cellvit-python", args.cellvit_python),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "--nucleus-instance-source cellvit requires "
                + ", ".join(missing)
            )
    evaluations = tuple(
        item
        for item in EVALUATIONS
        if args.primitive is None or item.primitive_id == args.primitive
    )
    targets = _cross_meta_targets(args.cross_meta_eval.resolve())
    metrics_path = output_root / "cross_meta_glas_metrics.json"
    if metrics_path.is_file() and not args.refresh_metrics:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(
            json.dumps({"stage": "mask_screen", "status": "reused", "total": len(metrics)}),
            flush=True,
        )
    else:
        metrics = []
        for index, row in enumerate(targets, start=1):
            metrics.append(_candidate_metrics(row))
            if index % 25 == 0 or index == len(targets):
                print(
                    json.dumps(
                        {
                            "stage": "mask_screen",
                            "completed": index,
                            "total": len(targets),
                        }
                    ),
                    flush=True,
                )
        _write_json(metrics_path, metrics)
    native_cache_path = output_root / "native_authority_cache.json"
    native_cache = (
        json.loads(native_cache_path.read_text(encoding="utf-8"))
        if native_cache_path.is_file()
        else {}
    )
    attempts_path = output_root / "attempts.json"
    attempts = (
        json.loads(attempts_path.read_text(encoding="utf-8"))
        if args.resume_selected and attempts_path.is_file()
        else []
    )
    selected_by_primitive: dict[str, list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        ranked = _diverse_ranked(evaluation.primitive_id, metrics)
        if len(ranked) < args.per_primitive:
            raise ValueError(
                f"not enough mask-screen candidates for {evaluation.primitive_id}"
            )
        recovered = (
            _existing_selected_runs(
                output_root=output_root,
                primitive_id=evaluation.primitive_id,
            )
            if args.resume_selected
            else {}
        )
        selected = [
            recovered[str(row["sample_id"])]
            for row in ranked
            if str(row["sample_id"]) in recovered
        ][: args.per_primitive]
        if len(selected) == args.per_primitive:
            selected_by_primitive[evaluation.primitive_id] = selected
            continue
        attempt_start = max(0, int(args.attempt_offset))
        attempt_stop = attempt_start + int(args.max_attempts)
        for attempt_index, row in enumerate(
            ranked[attempt_start:attempt_stop], start=attempt_start + 1
        ):
            compiler_attempt_index = attempt_index + int(args.seed_offset)
            native = None
            native_error = None
            if nucleus_instance_source == "cellvit":
                try:
                    native = _native_authority(
                        row,
                        output_root=output_root,
                        cellvit_model=args.cellvit_model.resolve(),
                        cellvit_root=args.cellvit_root.resolve(),
                        cellvit_python=args.cellvit_python.resolve(),
                        gpu=args.gpu,
                        timeout_seconds=args.cellvit_timeout_seconds,
                        cache=native_cache,
                    )
                    _validate_primitive_native_authority(
                        evaluation, native
                    )
                    _write_json(native_cache_path, native_cache)
                except Exception as exc:  # noqa: BLE001 - one fail-closed candidate
                    native_error = f"{type(exc).__name__}: {exc}"
            if native_error:
                attempt = {
                    "primitive_id": evaluation.primitive_id,
                    "sample_id": row["sample_id"],
                    "attempt_index": attempt_index,
                    "status": "native_authority_rejected",
                    "reason": native_error,
                }
            else:
                payload = _build_case(
                    evaluation,
                    row,
                    output_root=output_root,
                    native=native,
                    attempt_index=compiler_attempt_index,
                    portfolio_index=int(args.portfolio_index),
                    removal_variant=int(args.removal_variant),
                )
                result = _run_case(
                    payload,
                    output_root=output_root,
                    checkpoint=args.probnet_checkpoint.resolve(),
                    library=args.nuclei_instance_library.resolve(),
                    timeout_seconds=args.case_timeout_seconds,
                    threads=args.threads,
                    device=args.probnet_device,
                )
                attempt = {
                    "primitive_id": evaluation.primitive_id,
                    "sample_id": row["sample_id"],
                    "cross_meta_case_id": row["cross_meta_case_id"],
                    "attempt_index": attempt_index,
                    "compiler_seed": 42 + compiler_attempt_index,
                    "seed_offset": int(args.seed_offset),
                    "portfolio_index": int(args.portfolio_index),
                    "removal_variant": int(args.removal_variant),
                    "selection_score": row["selection_score"],
                    "status": result["status"],
                    "return_code": result["return_code"],
                    "duration_seconds": result["duration_seconds"],
                    "abstain_reasons": result["abstain_reasons"],
                }
                if result["status"] in {
                    "selected_research",
                    "compiled_pending_visual_review",
                }:
                    selected.append({"row": row, "payload": payload, "run": result})
            attempts.append(attempt)
            _write_json(attempts_path, attempts)
            print(json.dumps(attempt, ensure_ascii=False, sort_keys=True), flush=True)
            if len(selected) >= args.per_primitive:
                break
        selected_by_primitive[evaluation.primitive_id] = selected
    boards = {}
    selected_manifest = []
    for evaluation in evaluations:
        selected = selected_by_primitive[evaluation.primitive_id]
        if len(selected) != args.per_primitive:
            continue
        board = output_root / "boards" / f"{evaluation.primitive_id}.png"
        records = _render_board(
            evaluation, selected, output_path=board, tile_size=args.tile_size
        )
        boards[evaluation.primitive_id] = str(board)
        selected_manifest.extend(
            {"primitive_id": evaluation.primitive_id, **item} for item in records
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "cross_meta_eval": str(args.cross_meta_eval.resolve()),
        "cross_meta_eval_sha256": sha256_file(args.cross_meta_eval),
        "cross_meta_glas_unique_target_count": len(targets),
        "per_primitive_required": args.per_primitive,
        "selected_counts": {
            evaluation.primitive_id: len(
                selected_by_primitive[evaluation.primitive_id]
            )
            for evaluation in evaluations
        },
        "all_primitives_complete": all(
            len(selected_by_primitive[evaluation.primitive_id])
            == args.per_primitive
            for evaluation in evaluations
        ),
        "boards": boards,
        "selected_cases": selected_manifest,
        "he_generation_run": False,
        "llm_api_used": False,
        "probnet_device": args.probnet_device,
        "nucleus_instance_source": nucleus_instance_source,
    }
    _write_json(output_root / "mask_review_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-meta-eval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probnet-checkpoint", type=Path)
    parser.add_argument("--nuclei-instance-library", type=Path)
    parser.add_argument("--cellvit-model", type=Path)
    parser.add_argument("--cellvit-root", type=Path)
    parser.add_argument("--cellvit-python", type=Path)
    parser.add_argument(
        "--no-cellvit",
        action="store_true",
        help="skip CellViT inference; use semantic watershed instances from cross-meta nuclei mask",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--nucleus-instance-source",
        choices=("semantic-mask", "cellvit"),
        default="semantic-mask",
        help=(
            "Use the aligned nuclei mask by default; rerun CellViT only "
            "when explicitly requested."
        ),
    )
    parser.add_argument(
        "--probnet-device",
        default="cpu",
        help="torch device for the inner joint ProbNet run, e.g. cpu, cuda, or cuda:0",
    )
    parser.add_argument(
        "--primitive",
        choices=tuple(item.primitive_id for item in EVALUATIONS),
    )
    parser.add_argument("--per-primitive", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--attempt-offset", type=int, default=0)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="shift deterministic compiler seeds without changing candidate ranking",
    )
    parser.add_argument(
        "--portfolio-index",
        type=int,
        default=0,
        help="select one exact-capacity compiler portfolio certificate",
    )
    parser.add_argument(
        "--removal-variant",
        type=int,
        default=0,
        help="select a deterministic whole-instance depletion coverage variant",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--refresh-metrics", action="store_true")
    parser.add_argument("--resume-selected", action="store_true")
    parser.add_argument("--case-timeout-seconds", type=int, default=240)
    parser.add_argument("--cellvit-timeout-seconds", type=int, default=300)
    parser.add_argument("--tile-size", type=int, default=384)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["all_primitives_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
