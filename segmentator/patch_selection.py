from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class OrganRule:
    complexity_labels: tuple[int, ...]
    other_cap: float
    min_valid_classes: int
    head_neck_boundary_only: bool = False


ORGAN_RULES: dict[str, OrganRule] = {
    "breast": OrganRule((1, 2, 3, 4, 5, 6), 0.05, 2),
    "prostate": OrganRule((1, 2, 5), 0.05, 2),
    "colorectal": OrganRule((1, 2, 5), 0.05, 2),
    "lung": OrganRule((1, 2, 3, 4, 5), 0.10, 2),
    "skin": OrganRule((1, 2, 3, 5, 6), 0.05, 2),
    "head_neck": OrganRule((1,), 0.15, 1, head_neck_boundary_only=True),
}

ORGAN_EVAL_LABELS: dict[str, tuple[int, ...]] = {
    "breast": (0, 1, 2, 3, 4, 5, 6, 7),
    "prostate": (0, 1, 2, 5),
    "colorectal": (0, 1, 2, 5),
    "lung": (0, 1, 2, 3, 4, 5, 7),
    "skin": (0, 1, 2, 3, 5, 6),
    "head_neck": (0, 1, 7),
}

PROJECT_TO_ORGAN = {
    "TCGA-BRCA": "breast",
    "TCGA-PRAD": "prostate",
    "TCGA-COAD": "colorectal",
    "TCGA-READ": "colorectal",
    "TCGA-LUAD": "lung",
    "TCGA-LUSC": "lung",
    "TCGA-SKCM": "skin",
    "TCGA-HNSC": "head_neck",
}

TCGA_CASE_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)
PATCH_COORD_RE = re.compile(r"^(?P<wsi>.+)_(?P<x>\d+)_(?P<y>\d+)\.[^.]+$")


def organ_from_project(project_id: str) -> str:
    try:
        return PROJECT_TO_ORGAN[project_id.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported TCGA project: {project_id}") from exc


@dataclass(frozen=True)
class ParsedPatchName:
    case_id: str
    wsi: str
    x: int
    y: int


def parse_tcga_patch_name(filename: str) -> ParsedPatchName:
    case_match = TCGA_CASE_RE.search(filename)
    coord_match = PATCH_COORD_RE.match(filename)
    if not case_match or not coord_match:
        raise ValueError(f"unsupported TCGA patch filename: {filename}")
    return ParsedPatchName(
        case_id=case_match.group(1).upper(),
        wsi=coord_match.group("wsi"),
        x=int(coord_match.group("x")),
        y=int(coord_match.group("y")),
    )


@dataclass(frozen=True)
class ImageQuality:
    tissue_fraction: float
    laplacian_variance: float
    tenengrad: float
    dynamic_range: float
    near_black_fraction: float
    near_white_tissue_fraction: float
    mean_saturation: float


def compute_image_quality(image: np.ndarray, tissue_mask: np.ndarray | None = None) -> ImageQuality:
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected RGB image, got shape {rgb.shape}")

    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    optical_density = -np.log((rgb + 1.0) / 256.0).sum(axis=2)
    if tissue_mask is None:
        tissue = optical_density > 0.15
    else:
        tissue = np.asarray(tissue_mask, dtype=bool)
        if tissue.shape != gray.shape:
            raise ValueError(f"tissue mask shape mismatch: {tissue.shape} vs {gray.shape}")
    focus_domain = tissue if np.any(tissue) else np.ones(gray.shape, dtype=bool)

    laplacian = ndimage.laplace(gray)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    gradient_sq = gx * gx + gy * gy
    p5, p95 = np.percentile(gray[focus_domain], [5, 95])

    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1.0)
    near_white = np.all(rgb >= 250.0, axis=2)

    return ImageQuality(
        tissue_fraction=float(tissue.mean()),
        laplacian_variance=float(np.var(laplacian[focus_domain])),
        tenengrad=float(np.mean(gradient_sq[focus_domain])),
        dynamic_range=float(p95 - p5),
        near_black_fraction=float(np.mean(np.all(rgb <= 5.0, axis=2))),
        near_white_tissue_fraction=float(np.count_nonzero(near_white & tissue) / max(int(tissue.sum()), 1)),
        mean_saturation=float(np.mean(saturation[focus_domain])),
    )


@dataclass(frozen=True)
class MaskFeatures:
    organ: str
    tissue_pixels: int
    other_fraction: float
    valid_class_count: int
    interface_density: float
    class_entropy: float
    shape_irregularity: float
    speckle_fraction: float
    positive_complexity: float
    pre_normalized_score: float


def _remove_small_components(binary: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(binary)
    if count == 0:
        return np.zeros_like(binary, dtype=bool), 0
    areas = np.bincount(labels.reshape(-1))
    keep_ids = np.flatnonzero(areas >= min_area)
    keep_ids = keep_ids[keep_ids != 0]
    cleaned = np.isin(labels, keep_ids)
    return cleaned, int(np.count_nonzero(binary & ~cleaned))


def _component_irregularity(binary: np.ndarray) -> float:
    labels, count = ndimage.label(binary)
    if count == 0:
        return 0.0
    weighted = 0.0
    total_area = 0
    for component_id in range(1, count + 1):
        component = labels == component_id
        area = int(component.sum())
        if area == 0:
            continue
        perimeter = int(np.count_nonzero(component & ~ndimage.binary_erosion(component)))
        compactness = max(0.0, perimeter * perimeter / max(4.0 * math.pi * area, 1.0) - 1.0)
        weighted += min(compactness / 10.0, 1.0) * area
        total_area += area
    return weighted / max(total_area, 1)


def _interface_density(mask: np.ndarray, allowed_pairs: set[tuple[int, int]], denominator: int) -> float:
    horizontal_a, horizontal_b = mask[:, :-1], mask[:, 1:]
    vertical_a, vertical_b = mask[:-1, :], mask[1:, :]

    def count_pairs(a: np.ndarray, b: np.ndarray) -> int:
        changed = a != b
        if not np.any(changed):
            return 0
        left = a[changed].astype(np.int16)
        right = b[changed].astype(np.int16)
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        return sum((int(x), int(y)) in allowed_pairs for x, y in zip(low, high))

    transitions = count_pairs(horizontal_a, horizontal_b) + count_pairs(vertical_a, vertical_b)
    return float(transitions / max(denominator, 1))


def compute_mask_features(mask: np.ndarray, organ: str) -> MaskFeatures:
    if organ not in ORGAN_RULES:
        raise ValueError(f"unsupported organ: {organ}")
    array = np.asarray(mask, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"expected 2D label mask, got shape {array.shape}")

    rule = ORGAN_RULES[organ]
    tissue_pixels = int(np.count_nonzero(array != 0))
    other_fraction = float(np.count_nonzero(array == 7) / max(tissue_pixels, 1))
    min_component_area = max(64, int(math.ceil(tissue_pixels * 0.0005)))

    scoring_array = array.copy()
    if not rule.head_neck_boundary_only:
        other_pixels = scoring_array == 7
        valid_pixels = np.isin(scoring_array, rule.complexity_labels)
        if np.any(other_pixels) and np.any(valid_pixels):
            _, nearest_indices = ndimage.distance_transform_edt(~valid_pixels, return_indices=True)
            nearest_labels = scoring_array[tuple(nearest_indices)]
            scoring_array[other_pixels] = nearest_labels[other_pixels]

    cleaned = np.zeros_like(array)
    shape_cleaned = np.zeros_like(array)
    removed = 0
    labels_to_clean = set(rule.complexity_labels)
    if rule.head_neck_boundary_only:
        labels_to_clean.add(7)
    for class_id in sorted(labels_to_clean):
        kept, removed_count = _remove_small_components(array == class_id, min_component_area)
        cleaned[kept] = class_id
        removed += removed_count
        shape_kept, _ = _remove_small_components(scoring_array == class_id, min_component_area)
        shape_cleaned[shape_kept] = class_id

    class_areas = np.array([np.count_nonzero(cleaned == class_id) for class_id in rule.complexity_labels])
    valid_threshold = max(1, int(math.ceil(tissue_pixels * 0.005)))
    valid_class_count = int(np.count_nonzero(class_areas >= valid_threshold))
    positive_total = int(class_areas.sum())
    nonzero_areas = class_areas[class_areas > 0].astype(np.float64)
    if len(nonzero_areas) <= 1:
        class_entropy = 0.0
    else:
        fractions = nonzero_areas / nonzero_areas.sum()
        class_entropy = float(-(fractions * np.log(fractions)).sum() / math.log(len(fractions)))

    shape_parts = [_component_irregularity(shape_cleaned == class_id) for class_id in rule.complexity_labels]
    shape_irregularity = float(np.mean(shape_parts)) if shape_parts else 0.0

    if rule.head_neck_boundary_only:
        allowed_pairs = {(1, 7)}
        interface_density = _interface_density(cleaned, allowed_pairs, max(positive_total, 1))
        positive_complexity = 0.60 * interface_density + 0.25 * shape_irregularity + 0.15 * interface_density
    else:
        allowed_pairs = {
            (min(a, b), max(a, b))
            for index, a in enumerate(rule.complexity_labels)
            for b in rule.complexity_labels[index + 1 :]
        }
        interface_density = _interface_density(cleaned, allowed_pairs, max(positive_total, 1))
        normalized_count = valid_class_count / max(len(rule.complexity_labels), 1)
        positive_complexity = (
            0.45 * interface_density
            + 0.30 * class_entropy
            + 0.20 * shape_irregularity
            + 0.05 * normalized_count
        )

    speckle_fraction = float(removed / max(tissue_pixels, 1))
    other_penalty = 0.20 * other_fraction / rule.other_cap
    pre_normalized_score = positive_complexity - other_penalty - 0.20 * speckle_fraction
    return MaskFeatures(
        organ=organ,
        tissue_pixels=tissue_pixels,
        other_fraction=other_fraction,
        valid_class_count=valid_class_count,
        interface_density=interface_density,
        class_entropy=class_entropy,
        shape_irregularity=shape_irregularity,
        speckle_fraction=speckle_fraction,
        positive_complexity=float(positive_complexity),
        pre_normalized_score=float(pre_normalized_score),
    )


def passes_organ_constraints(features: MaskFeatures, organ: str) -> bool:
    rule = ORGAN_RULES[organ]
    return (
        features.tissue_pixels > 0
        and features.other_fraction <= rule.other_cap + 1e-12
        and features.valid_class_count >= rule.min_valid_classes
    )


def _percentile_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    if len(array) == 1:
        return np.ones(1, dtype=np.float64)
    return ranks / (len(array) - 1)


def finalize_row_scores(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    output = [dict(row) for row in rows]
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(output):
        organ = str(row["organ"])
        if organ not in ORGAN_RULES:
            raise ValueError(f"unsupported organ: {organ}")
        grouped_indices[organ].append(index)

    for organ, indices in grouped_indices.items():
        focus_values = [
            math.sqrt(
                max(float(output[index]["laplacian_variance"]), 0.0)
                * max(float(output[index]["tenengrad"]), 0.0)
            )
            for index in indices
        ]
        focus_threshold = float(np.quantile(focus_values, 0.15))
        component_names = ("interface_density", "class_entropy", "shape_irregularity", "valid_class_count")
        component_ranks = {
            name: _percentile_ranks([float(output[index][name]) for index in indices])
            for name in component_names
        }
        rule = ORGAN_RULES[organ]
        for local_index, row_index in enumerate(indices):
            row = output[row_index]
            focus_score = focus_values[local_index]
            quality_pass = (
                float(row["tissue_fraction"]) >= 0.70
                and float(row["dynamic_range"]) >= 40.0
                and float(row["near_black_fraction"]) <= 0.02
                and float(row["near_white_tissue_fraction"]) <= 0.05
                and focus_score >= focus_threshold
            )
            constraints_pass = (
                float(row["other_fraction"]) <= rule.other_cap + 1e-12
                and int(row["valid_class_count"]) >= rule.min_valid_classes
            )
            interface_rank = float(component_ranks["interface_density"][local_index])
            entropy_rank = float(component_ranks["class_entropy"][local_index])
            shape_rank = float(component_ranks["shape_irregularity"][local_index])
            count_rank = float(component_ranks["valid_class_count"][local_index])
            if rule.head_neck_boundary_only:
                boundary_tortuosity_rank = 0.5 * interface_rank + 0.5 * shape_rank
                positive_score = 0.60 * boundary_tortuosity_rank + 0.25 * shape_rank + 0.15 * interface_rank
            else:
                positive_score = (
                    0.45 * interface_rank + 0.30 * entropy_rank + 0.20 * shape_rank + 0.05 * count_rank
                )
            selection_score = (
                positive_score
                - 0.20 * float(row["other_fraction"]) / rule.other_cap
                - 0.20 * float(row["speckle_fraction"])
            )
            row.update(
                {
                    "focus_score": focus_score,
                    "focus_threshold_p15": focus_threshold,
                    "quality_pass": quality_pass,
                    "organ_constraints_pass": constraints_pass,
                    "interface_rank": interface_rank,
                    "entropy_rank": entropy_rank,
                    "shape_rank": shape_rank,
                    "valid_class_count_rank": count_rank,
                    "positive_complexity_score": positive_score,
                    "selection_score": selection_score,
                }
            )
    return output


@dataclass(frozen=True)
class SelectionResult:
    complex_rows: list[dict[str, object]]
    random_rows: list[dict[str, object]]
    deficits: dict[str, dict[str, int]]


def select_candidate_pool(
    rows: Iterable[Mapping[str, object]],
    *,
    target: int,
    organ_floor: int,
    case_cap: int,
    required_rows: Iterable[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    eligible = [
        dict(row)
        for row in rows
        if bool(row.get("quality_pass"))
        and bool(row.get("organ_constraints_pass"))
        and not bool(row.get("training_overlap"))
    ]
    eligible.sort(key=lambda row: (-float(row["selection_score"]), str(row["filename"])))
    eligible_by_name = {str(row["filename"]): row for row in eligible}
    selected: list[dict[str, object]] = []
    selected_names: set[str] = set()
    case_counts: dict[str, int] = defaultdict(int)
    organ_counts: dict[str, int] = defaultdict(int)

    def add(row: Mapping[str, object]) -> bool:
        filename = str(row["filename"])
        case_id = str(row["case_id"])
        if filename in selected_names or case_counts[case_id] >= case_cap:
            return False
        materialized = eligible_by_name.get(filename)
        if materialized is None:
            return False
        selected.append(materialized)
        selected_names.add(filename)
        case_counts[case_id] += 1
        organ_counts[str(materialized["organ"])] += 1
        return True

    for row in required_rows:
        add(row)

    for organ in ORGAN_RULES:
        for row in eligible:
            if organ_counts[organ] >= organ_floor:
                break
            if str(row["organ"]) == organ:
                add(row)

    for row in eligible:
        if len(selected) >= target:
            break
        add(row)
    return selected[:target]


def _coordinate_distance(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    return math.hypot(float(a.get("x", 0)) - float(b.get("x", 0)), float(a.get("y", 0)) - float(b.get("y", 0)))


def _round_robin_random(
    grouped: Mapping[str, Sequence[dict[str, object]]],
    target: int,
    per_case_cap: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    case_ids = sorted(grouped)
    rng.shuffle(case_ids)
    shuffled = {
        case_id: sorted((dict(row) for row in grouped[case_id]), key=lambda row: str(row["filename"]))
        for case_id in case_ids
    }
    for rows in shuffled.values():
        rng.shuffle(rows)
    selected: list[dict[str, object]] = []
    for round_index in range(per_case_cap):
        for case_id in case_ids:
            rows = shuffled[case_id]
            if round_index < len(rows):
                selected.append(rows[round_index])
                if len(selected) >= target:
                    return selected
    return selected


def select_case_disjoint_sets(
    rows: Iterable[Mapping[str, object]],
    *,
    complex_per_organ: int,
    random_per_organ: int,
    seed: int,
    case_caps: Mapping[str, int],
    random_case_caps: Mapping[str, int],
    min_coordinate_distance: float = 1024.0,
) -> SelectionResult:
    eligible: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        if not bool(row.get("quality_pass")) or not bool(row.get("organ_constraints_pass")):
            continue
        if bool(row.get("training_overlap")):
            continue
        organ = str(row["organ"])
        if organ in ORGAN_RULES:
            eligible[organ].append(row)

    complex_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    deficits: dict[str, dict[str, int]] = {}
    rng = random.Random(seed)

    for organ in ORGAN_RULES:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in eligible.get(organ, []):
            grouped[str(row["case_id"])].append(row)

        random_cap = int(random_case_caps[organ])
        case_ids = sorted(grouped)
        rng.shuffle(case_ids)
        random_case_ids: set[str] = set()
        random_capacity = 0
        for case_id in case_ids:
            remaining_case_ids = [candidate for candidate in case_ids if candidate not in random_case_ids and candidate != case_id]
            remaining_complex_capacity = sum(
                min(len(grouped[candidate]), int(case_caps[organ])) for candidate in remaining_case_ids
            )
            if remaining_complex_capacity < complex_per_organ:
                continue
            random_case_ids.add(case_id)
            random_capacity += min(len(grouped[case_id]), random_cap)
            if random_capacity >= random_per_organ:
                break
        random_grouped = {case_id: grouped[case_id] for case_id in random_case_ids}
        organ_random = _round_robin_random(random_grouped, random_per_organ, random_cap, rng)

        complex_cap = int(case_caps[organ])
        candidates = [row for case_id, case_rows in grouped.items() if case_id not in random_case_ids for row in case_rows]
        candidates.sort(key=lambda row: (-float(row["selection_score"]), str(row["filename"])))
        selected_by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
        organ_complex: list[dict[str, object]] = []
        for row in candidates:
            case_id = str(row["case_id"])
            prior = selected_by_case[case_id]
            if len(prior) >= complex_cap:
                continue
            if prior and any(_coordinate_distance(row, existing) < min_coordinate_distance for existing in prior):
                continue
            prior.append(row)
            organ_complex.append(row)
            if len(organ_complex) >= complex_per_organ:
                break

        complex_rows.extend(organ_complex)
        random_rows.extend(organ_random)
        deficits[organ] = {
            "complex": max(0, complex_per_organ - len(organ_complex)),
            "random": max(0, random_per_organ - len(organ_random)),
        }

    return SelectionResult(complex_rows=complex_rows, random_rows=random_rows, deficits=deficits)


def organ_valid_confusion(prediction: np.ndarray, target: np.ndarray, organ: str) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.int64)
    gt = np.asarray(target, dtype=np.int64)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} vs {gt.shape}")
    if organ not in ORGAN_EVAL_LABELS:
        raise ValueError(f"unsupported organ: {organ}")
    valid = np.isin(gt, ORGAN_EVAL_LABELS[organ]) & (pred >= 0) & (pred < 8)
    indices = gt[valid] * 8 + pred[valid]
    return np.bincount(indices, minlength=64).reshape(8, 8)
