#!/usr/bin/env python3
"""Build balanced, same-WSI benchmark pairs from the TCGA complex pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin
from scipy import ndimage


PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024
PngImagePlugin.MAX_TEXT_MEMORY = 1024 * 1024 * 1024


TISSUE_COLUMNS = (
    "pix_tumor",
    "pix_stroma",
    "pix_necrosis",
    "pix_immune_infiltrate",
    "pix_normal_epithelium",
    "pix_blood_vessel",
    "pix_other_tissue",
)
TISSUE_NAMES = (
    "tumor",
    "stroma",
    "necrosis",
    "immune_infiltrate",
    "normal_epithelium",
    "blood_vessel",
    "other_tissue",
)
CELL_IDS = (101, 102, 103, 104, 105)
CELL_NAMES = ("neoplastic", "inflammatory", "connective", "dead", "epithelial")
TISSUE_PALETTE = {
    0: (30, 30, 30),
    1: (220, 40, 40),
    2: (45, 170, 75),
    3: (145, 70, 190),
    4: (45, 110, 225),
    5: (245, 145, 35),
    6: (35, 200, 205),
    7: (205, 190, 45),
    255: (255, 255, 255),
}
TISSUE_LABELS = {
    0: "Background",
    1: "Tumor",
    2: "Stroma",
    3: "Necrosis",
    4: "Immune infiltrate",
    5: "Normal epithelium",
    6: "Blood vessel",
    7: "Other tissue",
    255: "Ignore / ambiguous",
}


@dataclass(frozen=True)
class Thresholds:
    tissue_jsd: float = 0.08
    tissue_linf: float = 0.15
    tissue_fraction_diff: float = 0.10
    cell_jsd: float = 0.08
    cell_density_diff: float = 0.25
    cell_spatial_jsd: float = 0.10
    min_cells: int = 30


@dataclass(frozen=True)
class PatchFeatures:
    tissue_proportions: np.ndarray
    dominant_tissue: int
    tissue_fraction: float
    cell_counts: np.ndarray
    cell_density: float
    cell_spatial_profile: np.ndarray


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def _normalized(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum())
    if total <= 0.0:
        return np.zeros_like(array)
    return array / total


def jensen_shannon_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return base-2 Jensen-Shannon divergence in the closed interval [0, 1]."""

    a = _normalized(left)
    b = _normalized(right)
    if not np.any(a) and not np.any(b):
        return 0.0
    midpoint = (a + b) / 2.0

    def kl_divergence(source: np.ndarray) -> float:
        valid = source > 0
        return float(np.sum(source[valid] * np.log2(source[valid] / midpoint[valid])))

    return (kl_divergence(a) + kl_divergence(b)) / 2.0


def symmetric_relative_difference(left: float, right: float) -> float:
    denominator = left + right
    if denominator <= 0.0:
        return 0.0
    return 2.0 * abs(left - right) / denominator


def coordinate_boxes_conflict(
    left_x: int,
    left_y: int,
    right_x: int,
    right_y: int,
    *,
    span: int,
    minimum_gap: int = 0,
) -> bool:
    """Return True when two top-left WSI boxes overlap or violate the requested gap."""

    if span <= 0 or minimum_gap < 0:
        raise ValueError("span must be positive and minimum_gap must be non-negative")
    separated = (
        left_x + span + minimum_gap <= right_x
        or right_x + span + minimum_gap <= left_x
        or left_y + span + minimum_gap <= right_y
        or right_y + span + minimum_gap <= left_y
    )
    return not separated


def _read_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_manifest_hashes(output_root: Path) -> None:
    relative_paths = [
        Path("pairs.csv"),
        Path("directions.csv"),
        Path("manual_review.csv"),
        Path("annotation_package/patch_annotation_manifest.csv"),
        Path("annotation_package/double_annotation_manifest.csv"),
        Path("annotation_package/pair_review.csv"),
        Path("annotation_package/palette.json"),
    ]
    hashes = {}
    for relative_path in relative_paths:
        path = output_root / relative_path
        if not path.exists():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(relative_path)] = digest.hexdigest()
    _write_json(output_root / "manifest_hashes.json", hashes)


def _link_or_copy(source: Path, destination: Path) -> None:
    """Materialize an immutable asset cheaply while remaining filesystem-agnostic."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.samefile(destination):
            return
        raise FileExistsError(f"unexpected existing annotation asset: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _copy_png_without_ancillary_chunks(source: Path, destination: Path) -> None:
    """Copy PNG pixel data while stripping oversized ICC/text metadata."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not source.samefile(destination):
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    signature = b"\x89PNG\r\n\x1a\n"
    keep_chunks = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS"}
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        if input_handle.read(8) != signature:
            raise ValueError(f"not a PNG image: {source}")
        output_handle.write(signature)
        while True:
            length_bytes = input_handle.read(4)
            if not length_bytes:
                break
            if len(length_bytes) != 4:
                raise ValueError(f"truncated PNG chunk length: {source}")
            length = int.from_bytes(length_bytes, "big")
            chunk_type = input_handle.read(4)
            payload_and_crc = input_handle.read(length + 4)
            if len(chunk_type) != 4 or len(payload_and_crc) != length + 4:
                raise ValueError(f"truncated PNG chunk: {source}")
            if chunk_type in keep_chunks:
                output_handle.write(length_bytes)
                output_handle.write(chunk_type)
                output_handle.write(payload_and_crc)
            if chunk_type == b"IEND":
                break
    os.replace(temporary, destination)


def _copy_editable_mask(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)


def _colorize_tissue_mask(mask: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label_id, color in TISSUE_PALETTE.items():
        output[mask == label_id] = np.asarray(color, dtype=np.uint8)
    return output


def _pair_preview(
    pair: Mapping[str, object],
    destination: Path,
    *,
    package_root: Path,
) -> None:
    if destination.exists():
        return
    panels: list[Image.Image] = []
    for side in ("a", "b"):
        filename = f"{pair['pair_id']}-{side}.png"
        with Image.open(package_root / "images" / filename) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(package_root / "tissue_masks_auto" / filename) as mask_image:
            mask = np.asarray(mask_image, dtype=np.uint8)
        overlay = (
            0.60 * rgb.astype(np.float32)
            + 0.40 * _colorize_tissue_mask(mask).astype(np.float32)
        ).astype(np.uint8)
        panels.extend(
            [
                Image.fromarray(rgb).resize((256, 256)),
                Image.fromarray(overlay).resize((256, 256)),
            ]
        )
    preview = Image.new("RGB", (1024, 280), color=(255, 255, 255))
    for index, panel in enumerate(panels):
        preview.paste(panel, (index * 256, 24))
    draw = ImageDraw.Draw(preview)
    draw.text((6, 5), f"{pair['pair_id']} | A image / mask | B image / mask", fill=(0, 0, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    preview.save(destination, quality=90, optimize=True)


def _annotation_patch_rows(pairs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair in pairs:
        for side in ("a", "b"):
            rows.append(
                {
                    "annotation_id": f"{pair['pair_id']}-{side}",
                    "pair_id": pair["pair_id"],
                    "side": side,
                    "organ": pair["organ"],
                    "project_id": pair["project_id"],
                    "case_id": pair["case_id"],
                    "wsi": pair["wsi"],
                    "stem": pair[f"{side}_stem"],
                    "x": pair[f"{side}_x"],
                    "y": pair[f"{side}_y"],
                    "pair_score": pair["pair_score"],
                    "image_path": pair[f"{side}_image_path"],
                    "auto_tissue_mask_path": pair[f"{side}_tissue_mask_path"],
                    "auto_cellvit_mask_path": pair[f"{side}_cellvit_mask_path"],
                }
            )
    return rows


def _select_double_annotation_rows(
    patch_rows: Sequence[Mapping[str, object]],
    *,
    target: int,
) -> list[dict[str, object]]:
    """Select one side from score-stratified pairs with near-balanced organ counts."""

    if target <= 0:
        return []
    organs = sorted({str(row["organ"]) for row in patch_rows})
    base, remainder = divmod(target, len(organs))
    target_by_organ = {
        organ: base + (1 if index < remainder else 0)
        for index, organ in enumerate(organs)
    }
    selected: list[dict[str, object]] = []
    for organ in organs:
        by_pair: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in patch_rows:
            if str(row["organ"]) == organ:
                by_pair[str(row["pair_id"])].append(row)
        ordered_pairs = sorted(
            by_pair.values(),
            key=lambda rows: (float(rows[0]["pair_score"]), str(rows[0]["pair_id"])),
        )
        count = min(target_by_organ[organ], len(ordered_pairs))
        if count == 0:
            continue
        indices = np.linspace(0, len(ordered_pairs) - 1, num=count, dtype=int)
        for selection_index, pair_index in enumerate(indices):
            pair_rows = sorted(ordered_pairs[int(pair_index)], key=lambda row: str(row["side"]))
            selected.append(dict(pair_rows[selection_index % len(pair_rows)]))
    return selected


def _materialize_annotation_package(
    output_root: Path,
    pairs: Sequence[Mapping[str, object]],
    *,
    double_annotation_count: int,
) -> dict[str, object]:
    package_root = output_root / "annotation_package"
    patch_rows = _annotation_patch_rows(pairs)
    materialized_rows: list[dict[str, object]] = []
    for row in patch_rows:
        filename = f"{row['annotation_id']}.png"
        image_path = package_root / "images" / filename
        auto_tissue_path = package_root / "tissue_masks_auto" / filename
        editable_tissue_path = package_root / "labels_primary" / filename
        cellvit_path = package_root / "cellvit_masks_auto" / filename
        _copy_png_without_ancillary_chunks(Path(str(row["image_path"])), image_path)
        _link_or_copy(Path(str(row["auto_tissue_mask_path"])), auto_tissue_path)
        _copy_editable_mask(Path(str(row["auto_tissue_mask_path"])), editable_tissue_path)
        _link_or_copy(Path(str(row["auto_cellvit_mask_path"])), cellvit_path)
        materialized_rows.append(
            {
                **row,
                "package_image_path": str(image_path),
                "package_image_relpath": str(image_path.relative_to(package_root)),
                "package_auto_tissue_mask_path": str(auto_tissue_path),
                "package_auto_tissue_mask_relpath": str(auto_tissue_path.relative_to(package_root)),
                "editable_tissue_mask_path": str(editable_tissue_path),
                "editable_tissue_mask_relpath": str(editable_tissue_path.relative_to(package_root)),
                "package_cellvit_mask_path": str(cellvit_path),
                "package_cellvit_mask_relpath": str(cellvit_path.relative_to(package_root)),
                "patch_image_qc": "pending",
                "tissue_mask_review_status": "pending",
                "reviewer_id": "",
                "review_notes": "",
            }
        )

    double_rows = _select_double_annotation_rows(
        materialized_rows,
        target=double_annotation_count,
    )
    for row in double_rows:
        image_path = Path(str(row["package_image_path"]))
        auto_mask_path = Path(str(row["package_auto_tissue_mask_path"]))
        destination = package_root / "labels_secondary" / image_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with Image.open(auto_mask_path) as mask_image:
                Image.new("L", mask_image.size, color=255).save(destination, compress_level=1)
        row["independent_tissue_mask_path"] = str(destination)
        row["independent_tissue_mask_relpath"] = str(destination.relative_to(package_root))
        row["secondary_reviewer_id"] = ""
        row["secondary_review_status"] = "pending"
        row["secondary_review_notes"] = ""

    pair_review_rows = []
    for pair in pairs:
        preview_path = package_root / "pair_previews" / f"{pair['pair_id']}.jpg"
        _pair_preview(pair, preview_path, package_root=package_root)
        pair_review_rows.append(
            {
                "pair_id": pair["pair_id"],
                "organ": pair["organ"],
                "wsi": pair["wsi"],
                "a_annotation_id": f"{pair['pair_id']}-a",
                "b_annotation_id": f"{pair['pair_id']}-b",
                "pair_score": pair["pair_score"],
                "pair_preview_path": str(preview_path),
                "pair_preview_relpath": str(preview_path.relative_to(package_root)),
                "a_image_qc": "pending",
                "b_image_qc": "pending",
                "pair_similarity_qc": "pending",
                "pair_keep": "pending",
                "pair_reject_reason": "",
                "reviewer_id": "",
                "review_notes": "",
            }
        )

    _write_csv(package_root / "patch_annotation_manifest.csv", materialized_rows)
    _write_csv(package_root / "double_annotation_manifest.csv", double_rows)
    _write_csv(package_root / "pair_review.csv", pair_review_rows)
    _write_json(
        package_root / "palette.json",
        {
            str(label_id): {"name": TISSUE_LABELS[label_id], "rgb": TISSUE_PALETTE[label_id]}
            for label_id in TISSUE_LABELS
        },
    )
    (package_root / "README_zh.txt").write_text(
        "组织 mask 使用 0-7 类别，255 表示 Ignore/无法判定。\n"
        "labels_primary 是 segmentator 预标注的可编辑副本；请修正错误类别和边界。\n"
        "labels_secondary 只用于盲法独立标注，不得查看自动 mask 或 labels_primary。\n"
        "pair_review.csv 的 pair_keep 控制 Patho-KID 去留；单张 patch 是否可用于 segmentator "
        "参考集由 patch_image_qc 和 tissue_mask_review_status 独立决定。\n"
    )
    package_summary = {
        "pairs": len(pairs),
        "patches": len(materialized_rows),
        "unique_stems": len({str(row["stem"]) for row in materialized_rows}),
        "double_annotation_patches": len(double_rows),
        "patches_by_organ": dict(sorted(Counter(str(row["organ"]) for row in materialized_rows).items())),
        "double_annotations_by_organ": dict(
            sorted(Counter(str(row["organ"]) for row in double_rows).items())
        ),
        "preannotation": "segmentator_tissue_mask_copy",
        "nuclei_annotation": "automatic_CellViT_only",
    }
    _write_json(package_root / "summary.json", package_summary)
    return package_summary


def _resolve_asset(
    row: Mapping[str, object],
    *,
    path_key: str,
    root: Path,
    subdirectory: str,
    suffix: str = ".png",
) -> Path:
    recorded = str(row.get(path_key, "")).strip()
    if recorded:
        path = Path(recorded)
        if path.exists():
            return path
    stem = str(row["stem"])
    fallback = root / subdirectory / f"{stem}{suffix}"
    if not fallback.exists():
        raise FileNotFoundError(f"missing {path_key} for {stem}: {fallback}")
    return fallback


def _cell_features(mask: np.ndarray, *, min_component_area: int, grid_size: int = 4) -> tuple[np.ndarray, np.ndarray]:
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D CellViT ID mask, received shape={mask.shape}")
    structure = np.ones((3, 3), dtype=np.uint8)
    counts = []
    for cell_id in CELL_IDS:
        labels, component_count = ndimage.label(mask == cell_id, structure=structure)
        if component_count == 0:
            counts.append(0)
            continue
        areas = np.bincount(labels.ravel())[1:]
        counts.append(int(np.count_nonzero(areas >= min_component_area)))

    foreground = np.isin(mask, CELL_IDS)
    grid_values = []
    for row_indices in np.array_split(np.arange(mask.shape[0]), grid_size):
        for column_indices in np.array_split(np.arange(mask.shape[1]), grid_size):
            grid_values.append(int(np.count_nonzero(foreground[np.ix_(row_indices, column_indices)])))
    # Sorting removes absolute-position dependence while retaining clustered-vs-uniform structure.
    spatial_profile = np.sort(_normalized(grid_values))[::-1]
    return np.asarray(counts, dtype=np.float64), spatial_profile


def _patch_features(
    row: Mapping[str, object],
    *,
    root: Path,
    min_component_area: int,
) -> PatchFeatures:
    tissue = _normalized([_as_float(row, column) for column in TISSUE_COLUMNS])
    dominant_tissue = int(np.argmax(tissue)) if np.any(tissue) else -1
    cell_mask_path = _resolve_asset(
        row,
        path_key="cellvit_id_mask_path",
        root=root,
        subdirectory="cellvit_id_masks",
    )
    with Image.open(cell_mask_path) as image:
        cell_mask = np.asarray(image, dtype=np.uint8)
    cell_counts, spatial_profile = _cell_features(cell_mask, min_component_area=min_component_area)
    cell_total = float(cell_counts.sum())
    tissue_pixels = max(_as_float(row, "tissue_pixels", 1.0), 1.0)
    return PatchFeatures(
        tissue_proportions=tissue,
        dominant_tissue=dominant_tissue,
        tissue_fraction=_as_float(row, "tissue_fraction"),
        cell_counts=cell_counts,
        cell_density=cell_total / tissue_pixels,
        cell_spatial_profile=spatial_profile,
    )


def _asset_path(row: Mapping[str, object], key: str, root: Path, subdirectory: str, suffix: str) -> str:
    recorded = str(row.get(key, "")).strip()
    if recorded:
        return recorded
    return str(root / subdirectory / f"{row['stem']}{suffix}")


def _eligible_candidate(row: Mapping[str, object], *, complexity_floor: float) -> bool:
    return (
        _as_bool(row.get("quality_pass", True))
        and _as_bool(row.get("organ_constraints_pass", True))
        and not _as_bool(row.get("training_overlap", False))
        and _as_float(row, "selection_score") >= complexity_floor
    )


def _pair_metrics(left: PatchFeatures, right: PatchFeatures) -> dict[str, float]:
    return {
        "tissue_jsd": jensen_shannon_distance(left.tissue_proportions, right.tissue_proportions),
        "tissue_linf": float(np.max(np.abs(left.tissue_proportions - right.tissue_proportions))),
        "tissue_fraction_diff": abs(left.tissue_fraction - right.tissue_fraction),
        "cell_jsd": jensen_shannon_distance(left.cell_counts, right.cell_counts),
        "cell_density_diff": symmetric_relative_difference(left.cell_density, right.cell_density),
        "cell_spatial_jsd": jensen_shannon_distance(
            left.cell_spatial_profile, right.cell_spatial_profile
        ),
        "left_cell_count": float(left.cell_counts.sum()),
        "right_cell_count": float(right.cell_counts.sum()),
    }


def _first_failed_threshold(
    metrics: Mapping[str, float],
    thresholds: Thresholds,
    *,
    dominant_tissue_matches: bool,
    require_dominant_tissue_match: bool,
) -> str | None:
    checks = (
        (require_dominant_tissue_match and not dominant_tissue_matches, "dominant_tissue"),
        (metrics["tissue_jsd"] > thresholds.tissue_jsd, "tissue_jsd"),
        (metrics["tissue_linf"] > thresholds.tissue_linf, "tissue_linf"),
        (metrics["tissue_fraction_diff"] > thresholds.tissue_fraction_diff, "tissue_fraction_diff"),
        (metrics["cell_jsd"] > thresholds.cell_jsd, "cell_jsd"),
        (metrics["cell_density_diff"] > thresholds.cell_density_diff, "cell_density_diff"),
        (metrics["cell_spatial_jsd"] > thresholds.cell_spatial_jsd, "cell_spatial_jsd"),
        (min(metrics["left_cell_count"], metrics["right_cell_count"]) < thresholds.min_cells, "min_cells"),
    )
    for failed, name in checks:
        if failed:
            return name
    return None


def _pair_score(metrics: Mapping[str, float], thresholds: Thresholds) -> float:
    return float(
        0.25 * metrics["tissue_jsd"] / thresholds.tissue_jsd
        + 0.15 * metrics["tissue_linf"] / thresholds.tissue_linf
        + 0.05 * metrics["tissue_fraction_diff"] / thresholds.tissue_fraction_diff
        + 0.25 * metrics["cell_jsd"] / thresholds.cell_jsd
        + 0.15 * metrics["cell_density_diff"] / thresholds.cell_density_diff
        + 0.15 * metrics["cell_spatial_jsd"] / thresholds.cell_spatial_jsd
    )


def _prefixed_patch_fields(
    prefix: str,
    row: Mapping[str, object],
    features: PatchFeatures,
    *,
    root: Path,
    anchor_stems: set[str],
) -> dict[str, object]:
    output: dict[str, object] = {
        f"{prefix}_stem": row["stem"],
        f"{prefix}_filename": row["filename"],
        f"{prefix}_x": int(row["x"]),
        f"{prefix}_y": int(row["y"]),
        f"{prefix}_selection_score": _as_float(row, "selection_score"),
        f"{prefix}_is_complex_anchor": str(row["stem"]) in anchor_stems,
        f"{prefix}_image_path": _asset_path(row, "image_path", root, "images", ".png"),
        f"{prefix}_text_path": _asset_path(row, "text_path", root, "txts", ".txt"),
        f"{prefix}_tissue_mask_path": _asset_path(row, "id_mask_path", root, "id_masks", ".png"),
        f"{prefix}_cellvit_mask_path": _asset_path(
            row, "cellvit_id_mask_path", root, "cellvit_id_masks", ".png"
        ),
        f"{prefix}_tissue_fraction": features.tissue_fraction,
        f"{prefix}_dominant_tissue": (
            TISSUE_NAMES[features.dominant_tissue] if features.dominant_tissue >= 0 else "none"
        ),
        f"{prefix}_cell_total": int(features.cell_counts.sum()),
        f"{prefix}_cell_density_per_tissue_pixel": features.cell_density,
    }
    for name, value in zip(TISSUE_NAMES, features.tissue_proportions):
        output[f"{prefix}_tissue_prop_{name}"] = float(value)
    for name, value in zip(CELL_NAMES, features.cell_counts):
        output[f"{prefix}_cell_count_{name}"] = int(value)
    return output


def _select_non_conflicting_wsi_pairs(
    edges: Sequence[dict[str, object]],
    *,
    coordinate_span: int,
    coordinate_gap: int,
    max_pairs: int,
) -> list[dict[str, object]]:
    """Find a maximum-cardinality, minimum-cost spatially disjoint WSI matching."""

    if not edges:
        return []
    patch_coordinates: dict[str, tuple[int, int]] = {}
    edge_by_nodes: dict[tuple[str, str], dict[str, object]] = {}
    for edge in edges:
        left = str(edge["a_stem"])
        right = str(edge["b_stem"])
        patch_coordinates[left] = (int(edge["a_x"]), int(edge["a_y"]))
        patch_coordinates[right] = (int(edge["b_x"]), int(edge["b_y"]))
        key = tuple(sorted((left, right)))
        previous = edge_by_nodes.get(key)
        if previous is None or float(edge["pair_score"]) < float(previous["pair_score"]):
            edge_by_nodes[key] = edge

    nodes = sorted(patch_coordinates)
    node_index = {node: index for index, node in enumerate(nodes)}
    indexed_edges = sorted(
        edge_by_nodes.items(),
        key=lambda item: (float(item[1]["pair_score"]), item[0]),
    )
    incident_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge_index, ((left, right), _edge) in enumerate(indexed_edges):
        left_index = node_index[left]
        right_index = node_index[right]
        incident_edges[left_index].append((right_index, edge_index))
        incident_edges[right_index].append((left_index, edge_index))

    conflict_masks = []
    for left_index, left in enumerate(nodes):
        left_x, left_y = patch_coordinates[left]
        mask = 1 << left_index
        for right_index, right in enumerate(nodes):
            if left_index == right_index:
                continue
            right_x, right_y = patch_coordinates[right]
            if coordinate_boxes_conflict(
                left_x,
                left_y,
                right_x,
                right_y,
                span=coordinate_span,
                minimum_gap=coordinate_gap,
            ):
                mask |= 1 << right_index
        conflict_masks.append(mask)

    pair_limit = len(nodes) // 2 if max_pairs <= 0 else min(max_pairs, len(nodes) // 2)

    def better(
        left: tuple[int, float, tuple[int, ...]],
        right: tuple[int, float, tuple[int, ...]],
    ) -> tuple[int, float, tuple[int, ...]]:
        if left[0] != right[0]:
            return left if left[0] > right[0] else right
        if not np.isclose(left[1], right[1]):
            return left if left[1] < right[1] else right
        return left if left[2] < right[2] else right

    @lru_cache(maxsize=None)
    def solve(available_mask: int, remaining_pairs: int) -> tuple[int, float, tuple[int, ...]]:
        if available_mask == 0 or remaining_pairs == 0:
            return (0, 0.0, ())
        left_bit = available_mask & -available_mask
        left_index = left_bit.bit_length() - 1
        best = solve(available_mask & ~left_bit, remaining_pairs)
        for right_index, edge_index in incident_edges.get(left_index, []):
            right_bit = 1 << right_index
            if not available_mask & right_bit:
                continue
            next_mask = available_mask & ~(conflict_masks[left_index] | conflict_masks[right_index])
            count, cost, selected_edges = solve(next_mask, remaining_pairs - 1)
            candidate = (
                count + 1,
                cost + float(indexed_edges[edge_index][1]["pair_score"]),
                tuple(sorted((*selected_edges, edge_index))),
            )
            best = better(best, candidate)
        return best

    _count, _cost, selected_indices = solve((1 << len(nodes)) - 1, pair_limit)
    return sorted(
        (indexed_edges[index][1] for index in selected_indices),
        key=lambda row: (float(row["pair_score"]), str(row["a_stem"]), str(row["b_stem"])),
    )


def _validate_selected_pair_set(
    pairs: Sequence[Mapping[str, object]],
    *,
    coordinate_span: int,
    coordinate_gap: int,
) -> None:
    seen_stems: set[str] = set()
    patches_by_wsi: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for pair in pairs:
        for side in ("a", "b"):
            stem = str(pair[f"{side}_stem"])
            if stem in seen_stems:
                raise RuntimeError(f"selected patch was reused: {stem}")
            seen_stems.add(stem)
            patches_by_wsi[str(pair["wsi"])].append(
                (stem, int(pair[f"{side}_x"]), int(pair[f"{side}_y"]))
            )
    for wsi, patches in patches_by_wsi.items():
        for index, (left_stem, left_x, left_y) in enumerate(patches):
            for right_stem, right_x, right_y in patches[index + 1 :]:
                if coordinate_boxes_conflict(
                    left_x,
                    left_y,
                    right_x,
                    right_y,
                    span=coordinate_span,
                    minimum_gap=coordinate_gap,
                ):
                    raise RuntimeError(
                        f"selected coordinates conflict in {wsi}: {left_stem} vs {right_stem}"
                    )


def build_pairs(
    anchors: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    *,
    anchor_root: Path,
    candidate_root: Path,
    thresholds: Thresholds,
    coordinate_span: int,
    coordinate_gap: int,
    pairs_per_organ: int,
    total_pairs: int,
    min_component_area: int,
    require_dominant_tissue_match: bool,
    complexity_floor_quantile: float,
    max_pairs_per_wsi: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not 0.0 <= complexity_floor_quantile <= 1.0:
        raise ValueError("complexity_floor_quantile must be between 0 and 1")
    anchor_stems = {str(row["stem"]) for row in anchors}
    anchors_by_wsi: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    candidates_by_wsi: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in anchors:
        anchors_by_wsi[str(row["wsi"])].append(row)
    for row in candidates:
        if str(row["wsi"]) in anchors_by_wsi:
            candidates_by_wsi[str(row["wsi"])].append(row)

    organs = sorted({str(row["organ"]) for row in anchors})
    complexity_floor = {}
    for organ in organs:
        scores = [
            _as_float(row, "selection_score")
            for row in candidates
            if str(row["wsi"]) in anchors_by_wsi
            and str(row["organ"]) == organ
            and _as_bool(row.get("quality_pass", True))
            and _as_bool(row.get("organ_constraints_pass", True))
            and not _as_bool(row.get("training_overlap", False))
        ]
        if not scores:
            raise RuntimeError(f"no eligible candidate scores for organ={organ}")
        complexity_floor[organ] = float(np.quantile(scores, complexity_floor_quantile))
    feature_cache: dict[str, PatchFeatures] = {}

    def features(row: Mapping[str, object], root: Path) -> PatchFeatures:
        stem = str(row["stem"])
        if stem not in feature_cache:
            feature_cache[stem] = _patch_features(
                row,
                root=root,
                min_component_area=min_component_area,
            )
        return feature_cache[stem]

    exclusions: Counter[str] = Counter()
    matched_by_wsi: dict[str, list[dict[str, object]]] = {}
    eligible_edge_count = 0
    for wsi in sorted(anchors_by_wsi):
        wsi_edges: list[dict[str, object]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for anchor in anchors_by_wsi[wsi]:
            organ = str(anchor["organ"])
            for candidate in candidates_by_wsi.get(wsi, []):
                if str(candidate["organ"]) != organ:
                    exclusions["organ_mismatch"] += 1
                    continue
                if str(candidate["stem"]) == str(anchor["stem"]):
                    exclusions["same_patch"] += 1
                    continue
                pair_key = tuple(sorted((str(anchor["stem"]), str(candidate["stem"]))))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                if not _eligible_candidate(candidate, complexity_floor=complexity_floor[organ]):
                    exclusions["candidate_not_complex"] += 1
                    continue
                if coordinate_boxes_conflict(
                    int(anchor["x"]),
                    int(anchor["y"]),
                    int(candidate["x"]),
                    int(candidate["y"]),
                    span=coordinate_span,
                    minimum_gap=coordinate_gap,
                ):
                    exclusions["coordinate_conflict"] += 1
                    continue

                anchor_features = features(anchor, anchor_root)
                candidate_features = features(candidate, candidate_root)
                metrics = _pair_metrics(anchor_features, candidate_features)
                failure = _first_failed_threshold(
                    metrics,
                    thresholds,
                    dominant_tissue_matches=(
                        anchor_features.dominant_tissue == candidate_features.dominant_tissue
                    ),
                    require_dominant_tissue_match=require_dominant_tissue_match,
                )
                if failure:
                    exclusions[failure] += 1
                    continue
                eligible_edge_count += 1
                wsi_edges.append(
                    {
                        "organ": organ,
                        "project_id": anchor["project_id"],
                        "case_id": anchor["case_id"],
                        "wsi": wsi,
                        "coordinate_span": coordinate_span,
                        "coordinate_gap": coordinate_gap,
                        "coordinate_overlap": False,
                        "pair_score": _pair_score(metrics, thresholds),
                        **metrics,
                        **_prefixed_patch_fields(
                            "a", anchor, anchor_features, root=anchor_root, anchor_stems=anchor_stems
                        ),
                        **_prefixed_patch_fields(
                            "b", candidate, candidate_features, root=candidate_root, anchor_stems=anchor_stems
                        ),
                    }
                )
        if wsi_edges:
            matched_by_wsi[wsi] = _select_non_conflicting_wsi_pairs(
                wsi_edges,
                coordinate_span=coordinate_span,
                coordinate_gap=coordinate_gap,
                max_pairs=max_pairs_per_wsi,
            )

    ordered_by_organ: dict[str, list[dict[str, object]]] = {}
    for organ in organs:
        organ_matchings = [
            rows
            for wsi, rows in sorted(matched_by_wsi.items())
            if rows and str(rows[0]["organ"]) == organ
        ]
        organ_target = sum(len(rows) for rows in organ_matchings)
        round_index = 0
        organ_ordered: list[dict[str, object]] = []
        while len(organ_ordered) < organ_target:
            round_rows = sorted(
                (rows[round_index] for rows in organ_matchings if len(rows) > round_index),
                key=lambda row: (float(row["pair_score"]), str(row["wsi"])),
            )
            if not round_rows:
                break
            organ_ordered.extend(round_rows[: organ_target - len(organ_ordered)])
            round_index += 1
        ordered_by_organ[organ] = organ_ordered

    selected: list[dict[str, object]] = []
    if total_pairs > 0:
        # Water-fill across organs so smaller pools retain their full representation.
        organ_offsets = {organ: 0 for organ in organs}
        while len(selected) < total_pairs:
            added = False
            for organ in organs:
                offset = organ_offsets[organ]
                if offset >= len(ordered_by_organ[organ]):
                    continue
                selected.append(ordered_by_organ[organ][offset])
                organ_offsets[organ] += 1
                added = True
                if len(selected) == total_pairs:
                    break
            if not added:
                break
    else:
        for organ in organs:
            organ_rows = ordered_by_organ[organ]
            selected.extend(organ_rows[:pairs_per_organ] if pairs_per_organ > 0 else organ_rows)
    selected.sort(key=lambda row: (str(row["organ"]), float(row["pair_score"]), str(row["wsi"])))
    _validate_selected_pair_set(
        selected,
        coordinate_span=coordinate_span,
        coordinate_gap=coordinate_gap,
    )
    organ_index: Counter[str] = Counter()
    for row in selected:
        organ = str(row["organ"])
        organ_index[organ] += 1
        row["pair_id"] = f"{organ}-{organ_index[organ]:04d}"

    selected_counts = Counter(str(row["organ"]) for row in selected)
    selected_wsi_counts = Counter(str(row["wsi"]) for row in selected)
    available_counts = Counter(
        str(rows[0]["organ"])
        for rows in matched_by_wsi.values()
        for _row in rows
        if rows
    )
    summary = {
        "anchor_rows": len(anchors),
        "anchor_wsis": len(anchors_by_wsi),
        "candidate_rows": len(candidates),
        "candidate_rows_on_anchor_wsis": sum(len(rows) for rows in candidates_by_wsi.values()),
        "feature_rows_loaded": len(feature_cache),
        "eligible_edges": eligible_edge_count,
        "wsis_with_eligible_pair": len(matched_by_wsi),
        "available_disjoint_pairs_by_organ": dict(sorted(available_counts.items())),
        "selected_pairs": len(selected),
        "selected_directions": 2 * len(selected),
        "selected_wsis": len(selected_wsi_counts),
        "selected_pairs_per_wsi_histogram": dict(
            sorted(Counter(selected_wsi_counts.values()).items())
        ),
        "selected_pairs_by_organ": dict(sorted(selected_counts.items())),
        "pair_deficits_by_organ": {
            organ: max(0, pairs_per_organ - selected_counts.get(organ, 0)) if pairs_per_organ > 0 else 0
            for organ in sorted(complexity_floor)
        },
        "total_pair_deficit": max(0, total_pairs - len(selected)) if total_pairs > 0 else 0,
        "total_pairs_target": total_pairs if total_pairs > 0 else "not_set",
        "pairs_per_organ_target": (
            pairs_per_organ
            if pairs_per_organ > 0
            else ("water_filled_to_total" if total_pairs > 0 else "all_eligible")
        ),
        "complexity_floor_by_organ": complexity_floor,
        "complexity_floor_quantile": complexity_floor_quantile,
        "exclusions": dict(sorted(exclusions.items())),
        "coordinate_convention": "x_y_are_top_left_wsi_coordinates",
        "coordinate_span": coordinate_span,
        "coordinate_gap": coordinate_gap,
        "max_pairs_per_wsi": max_pairs_per_wsi,
        "all_selected_patches_unique": True,
        "all_selected_wsi_coordinates_nonoverlapping": True,
        "thresholds": thresholds.__dict__,
        "require_dominant_tissue_match": require_dominant_tissue_match,
        "cell_count_source": "8_connected_components_on_CellViT_ID_mask",
        "cell_spatial_source": "sorted_4x4_CellViT_foreground_pixel_profile",
        "min_cell_component_area": min_component_area,
    }
    return selected, summary


def _direction_rows(pairs: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for pair in pairs:
        for direction, reference, target in (("a_to_b", "a", "b"), ("b_to_a", "b", "a")):
            output.append(
                {
                    "sample_id": f"{pair['pair_id']}-{direction}",
                    "pair_id": pair["pair_id"],
                    "direction": direction,
                    "organ": pair["organ"],
                    "project_id": pair["project_id"],
                    "case_id": pair["case_id"],
                    "wsi": pair["wsi"],
                    "reference_stem": pair[f"{reference}_stem"],
                    "reference_x": pair[f"{reference}_x"],
                    "reference_y": pair[f"{reference}_y"],
                    "reference_image_path": pair[f"{reference}_image_path"],
                    "reference_text_path": pair[f"{reference}_text_path"],
                    "reference_tissue_mask_path": pair[f"{reference}_tissue_mask_path"],
                    "reference_cellvit_mask_path": pair[f"{reference}_cellvit_mask_path"],
                    "target_stem": pair[f"{target}_stem"],
                    "target_x": pair[f"{target}_x"],
                    "target_y": pair[f"{target}_y"],
                    "target_image_path": pair[f"{target}_image_path"],
                    "target_text_path": pair[f"{target}_text_path"],
                    "target_tissue_mask_path": pair[f"{target}_tissue_mask_path"],
                    "target_cellvit_mask_path": pair[f"{target}_cellvit_mask_path"],
                    "pair_score": pair["pair_score"],
                    "tissue_jsd": pair["tissue_jsd"],
                    "tissue_linf": pair["tissue_linf"],
                    "tissue_fraction_diff": pair["tissue_fraction_diff"],
                    "cell_jsd": pair["cell_jsd"],
                    "cell_density_diff": pair["cell_density_diff"],
                    "cell_spatial_jsd": pair["cell_spatial_jsd"],
                    "coordinate_span": pair["coordinate_span"],
                    "coordinate_gap": pair["coordinate_gap"],
                    "coordinate_overlap": False,
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build balanced same-organ, same-WSI, non-overlapping complex benchmark pairs."
    )
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--anchor-root", type=Path, default=None)
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--pairs-per-organ",
        type=int,
        default=50,
        help="Per-organ pair cap; 0 keeps every eligible spatially disjoint pair.",
    )
    parser.add_argument(
        "--max-pairs-per-wsi",
        type=int,
        default=0,
        help="Maximum spatially disjoint pairs per WSI; 0 uses every eligible disjoint pair.",
    )
    parser.add_argument(
        "--total-pairs",
        type=int,
        default=0,
        help="Global pair target selected by organ water-filling; requires --pairs-per-organ 0.",
    )
    parser.add_argument("--coordinate-span", type=int, default=512)
    parser.add_argument("--coordinate-gap", type=int, default=0)
    parser.add_argument("--min-component-area", type=int, default=5)
    parser.add_argument("--max-tissue-jsd", type=float, default=0.08)
    parser.add_argument("--max-tissue-linf", type=float, default=0.15)
    parser.add_argument("--max-tissue-fraction-diff", type=float, default=0.10)
    parser.add_argument("--max-cell-jsd", type=float, default=0.08)
    parser.add_argument("--max-cell-density-diff", type=float, default=0.25)
    parser.add_argument("--max-cell-spatial-jsd", type=float, default=0.10)
    parser.add_argument("--min-cells", type=int, default=30)
    parser.add_argument("--require-dominant-tissue-match", action="store_true")
    parser.add_argument("--complexity-floor-quantile", type=float, default=0.25)
    parser.add_argument("--allow-deficit", action="store_true")
    parser.add_argument(
        "--materialize-annotation-package",
        action="store_true",
        help="Create doctor-ready images, editable tissue pre-annotations, pair previews, and review CSVs.",
    )
    parser.add_argument(
        "--double-annotation-count",
        type=int,
        default=150,
        help="Number of score- and organ-stratified patches prepared for blind independent redraw.",
    )
    args = parser.parse_args()
    if args.total_pairs > 0 and args.pairs_per_organ > 0:
        parser.error("--total-pairs requires --pairs-per-organ 0")

    anchor_root = args.anchor_root or args.anchor_manifest.parent
    candidate_root = args.candidate_root or args.candidate_manifest.parent
    thresholds = Thresholds(
        tissue_jsd=args.max_tissue_jsd,
        tissue_linf=args.max_tissue_linf,
        tissue_fraction_diff=args.max_tissue_fraction_diff,
        cell_jsd=args.max_cell_jsd,
        cell_density_diff=args.max_cell_density_diff,
        cell_spatial_jsd=args.max_cell_spatial_jsd,
        min_cells=args.min_cells,
    )
    pairs, summary = build_pairs(
        _read_manifest(args.anchor_manifest),
        _read_manifest(args.candidate_manifest),
        anchor_root=anchor_root,
        candidate_root=candidate_root,
        thresholds=thresholds,
        coordinate_span=args.coordinate_span,
        coordinate_gap=args.coordinate_gap,
        pairs_per_organ=args.pairs_per_organ,
        total_pairs=args.total_pairs,
        min_component_area=args.min_component_area,
        require_dominant_tissue_match=args.require_dominant_tissue_match,
        complexity_floor_quantile=args.complexity_floor_quantile,
        max_pairs_per_wsi=args.max_pairs_per_wsi,
    )
    directions = _direction_rows(pairs)
    output_root = args.output_root
    _write_csv(output_root / "pairs.csv", pairs)
    _write_json(output_root / "pairs.json", pairs)
    _write_csv(output_root / "directions.csv", directions)
    _write_json(output_root / "directions.json", directions)
    _write_csv(
        output_root / "manual_review.csv",
        [
            {
                "pair_id": row["pair_id"],
                "organ": row["organ"],
                "wsi": row["wsi"],
                "a_stem": row["a_stem"],
                "b_stem": row["b_stem"],
                "review_status": "pending",
                "review_reason": "",
            }
            for row in pairs
        ],
    )
    if args.materialize_annotation_package:
        summary["annotation_package"] = _materialize_annotation_package(
            output_root,
            pairs,
            double_annotation_count=args.double_annotation_count,
        )
    _write_manifest_hashes(output_root)
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not args.allow_deficit and (
        any(summary["pair_deficits_by_organ"].values()) or summary["total_pair_deficit"]
    ):
        raise RuntimeError(
            "pair quota deficits: "
            f"by_organ={summary['pair_deficits_by_organ']} total={summary['total_pair_deficit']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
