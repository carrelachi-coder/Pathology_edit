#!/usr/bin/env python3
"""Build G2-600 from fixed validation metadata backed by dataset annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_config import get_config
from phase3_mask_edit.audit.labels import to_coarse_mask
from phase3_mask_edit.benchmark.intents import legal_target_labels_for_primitive
from phase3_mask_edit.core.config import (
    default_recipe_path_for_profile,
    load_recipe,
)
from phase3_mask_edit.core.labels import MaskProfileSchema


DEFAULT_EVAL_METADATA = Path(
    "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/"
    "phase5_runs/cross_meta/metadata_cross_val.json"
)
DEFAULT_OUTPUT = Path("/data1/zhao/wqx/benchmark_v1/g2_600")
DATASET_TO_ORGAN = {
    "BCSS": "breast",
    "GLAS": "colorectal",
    "IGNITE": "lung",
    "ORCA": "oral",
    "PANDA": "prostate",
    "PUMA": "skin",
}
ORGAN_TO_DATASET = {organ: dataset for dataset, organ in DATASET_TO_ORGAN.items()}
ORGAN_QUOTA = 100
DESIRED_PRIMITIVE_QUOTA = 75
MIN_EXISTING_TARGET_FRACTION = 0.01
MIN_MODERATE_SOURCE_FRACTION = 0.14
MIN_TUMOR_DECREASE_NAMED_BACKFILL_FRACTION = 0.05
MIN_OTHER_ONLY_TUMOR_CONTEXT_FRACTION = 0.20
MIN_TUMOR_NON_TUMOR_BOUNDARY_PIXELS = 64
TUMOR_NON_TUMOR_BOUNDARY_RADIUS = 4
MIN_CHANGED_AREA_FRACTION_IMAGE = 0.05
DESIRED_MIN_ORGANS_PER_PRIMITIVE = 3
DESIRED_MIN_PRIMITIVES_PER_ORGAN = 4
DESIRED_MAX_PATCHES_PER_WSI = 2
MIN_RESERVES_PER_ACTIVE_CELL = 5
MIN_EFFECTIVE_PRIMITIVE_QUOTA = 30

COARSE_ID_BY_NAME = {
    "tumor": 1,
    "stroma": 2,
    "necrosis": 3,
    "immune": 4,
    "normal": 5,
    "vessel": 6,
    "other": 7,
}


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    internal_primitive: str
    source_labels: tuple[str, ...]
    target_label: str
    instruction: str


PRIMITIVES = (
    PrimitiveSpec(
        "tumor_increase",
        "tumor_burden_increase",
        ("Stroma", "Normal epithelium", "Other tissue", "Immune infiltrate"),
        "Tumor",
        (
            "Moderately increase existing tumor burden by organically expanding "
            "it into adjacent editable non-tumor tissue; preserve all "
            "unrequested labels and do not create disconnected tumor."
        ),
    ),
    PrimitiveSpec(
        "tumor_decrease",
        "tumor_burden_decrease",
        ("Tumor",),
        "Stroma",
        (
            "Moderately decrease existing tumor burden and organically backfill "
            "the changed area with adjacent valid tissue; preserve all "
            "unrequested labels and do not create background."
        ),
    ),
    PrimitiveSpec(
        "stroma_increase",
        "stroma_increase",
        ("Other tissue", "Normal epithelium", "Immune infiltrate"),
        "Stroma",
        (
            "Moderately increase existing stroma by organically expanding it "
            "into adjacent eligible tissue; preserve tumor and all unrequested "
            "labels. Do not create disconnected stroma."
        ),
    ),
    PrimitiveSpec(
        "stroma_decrease",
        "stroma_decrease",
        ("Stroma",),
        "Tumor",
        (
            "Moderately decrease existing stroma using adjacent valid tissue as "
            "backfill; preserve all unrequested labels and do not create "
            "background."
        ),
    ),
    PrimitiveSpec(
        "immune_increase",
        "stromal_immune_infiltration",
        ("Stroma",),
        "Immune infiltrate",
        (
            "Moderately increase the existing immune infiltrate within eligible "
            "stroma using an organic patchy pattern; preserve all unrequested "
            "labels and do not create a de-novo immune compartment."
        ),
    ),
    PrimitiveSpec(
        "immune_decrease",
        "immune_infiltration_decrease",
        ("Immune infiltrate",),
        "Stroma",
        (
            "Moderately decrease the existing immune infiltrate and organically "
            "repair the changed area with adjacent valid tissue; preserve all "
            "unrequested labels."
        ),
    ),
    PrimitiveSpec(
        "necrosis_increase",
        "necrosis_appearance",
        ("Tumor",),
        "Necrosis",
        (
            "Moderately increase the already present necrotic compartment by "
            "organically extending it within existing tumor; preserve all "
            "unrequested labels and do not create de-novo necrosis."
        ),
    ),
    PrimitiveSpec(
        "necrosis_decrease",
        "necrosis_resolution",
        ("Necrosis",),
        "Stroma",
        (
            "Moderately decrease existing necrosis and organically repair it "
            "with adjacent viable tumor or stroma; preserve all unrequested "
            "labels and do not create background."
        ),
    ),
)
PRIMITIVE_BY_NAME = {item.name: item for item in PRIMITIVES}


@lru_cache(maxsize=None)
def _allowed_target_labels(
    primitive: PrimitiveSpec,
    profile: str,
) -> tuple[str, ...]:
    schema = MaskProfileSchema.from_reference_profile(profile)
    recipe = load_recipe(default_recipe_path_for_profile(profile))
    primitive_config = next(
        (
            item
            for item in recipe.get("primitives", ())
            if isinstance(item, Mapping)
            and str(item.get("name")) == primitive.internal_primitive
        ),
        None,
    )
    if primitive_config is None:
        raise ValueError(
            f"Product recipe lacks G2 primitive {primitive.internal_primitive!r} "
            f"for profile {profile!r}."
        )
    labels = legal_target_labels_for_primitive(primitive_config, schema)
    return labels or (primitive.target_label,)


def load_candidate_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load unique target patches and derive eligibility from annotation masks."""

    metadata_path = Path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs") if isinstance(payload, dict) else None
    if not isinstance(pairs, list):
        raise ValueError("Eval metadata must contain a top-level 'pairs' list.")

    unique: dict[tuple[str, str], Mapping[str, Any]] = {}
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("Every eval metadata pair must be an object.")
        dataset = str(pair.get("dataset", "")).upper()
        sample_id = str(pair.get("sample_id", ""))
        if dataset not in DATASET_TO_ORGAN or not sample_id:
            continue
        unique.setdefault((dataset, sample_id), pair)

    rows = [
        _candidate_from_eval_pair(pair, metadata_path=metadata_path)
        for _, pair in sorted(unique.items())
    ]
    if not rows:
        raise ValueError("No supported human-annotation candidates were found.")
    return rows


def _candidate_from_eval_pair(
    pair: Mapping[str, Any],
    *,
    metadata_path: Path,
) -> dict[str, Any]:
    dataset = str(pair["dataset"]).upper()
    sample_id = str(pair["sample_id"])
    image_path = Path(str(pair["target_image"]))
    tissue_path = Path(str(pair["target_tissue_mask"]))
    nuclei_path = Path(str(pair["target_nuclei_mask"]))
    _validate_annotation_paths(
        dataset,
        image_path=image_path,
        tissue_path=tissue_path,
        nuclei_path=nuclei_path,
    )
    if not tissue_path.is_file():
        raise FileNotFoundError(f"Missing dataset annotation mask: {tissue_path}")

    raw_mask = np.asarray(Image.open(tissue_path))
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[..., 0]
    coarse = to_coarse_mask(raw_mask.astype(np.int64, copy=False))
    valid = coarse != 255
    valid_pixels = int(valid.sum())
    if valid_pixels <= 0:
        raise ValueError(f"Annotation mask has no valid pixels: {tissue_path}")
    counts = {
        name: int(np.count_nonzero(valid & (coarse == class_id)))
        for name, class_id in COARSE_ID_BY_NAME.items()
    }
    tumor = valid & (coarse == COARSE_ID_BY_NAME["tumor"])
    non_tumor_context = valid & np.isin(
        coarse,
        (
            COARSE_ID_BY_NAME["stroma"],
            COARSE_ID_BY_NAME["immune"],
            COARSE_ID_BY_NAME["normal"],
            COARSE_ID_BY_NAME["other"],
        ),
    )
    tumor_boundary_neighborhood = binary_dilation(
        tumor,
        iterations=TUMOR_NON_TUMOR_BOUNDARY_RADIUS,
    ) & ~tumor
    tumor_non_tumor_boundary_pixels = int(
        np.count_nonzero(non_tumor_context & tumor_boundary_neighborhood)
    )
    native_ids = sorted(
        {
            int(value)
            for value in get_config(dataset).to_coarse_map.values()
            if int(value) not in {0, 7}
        }
    )
    foreground_fraction = 1.0 - counts.get("other", 0) / valid_pixels
    return {
        "stem": sample_id,
        "sample_id": sample_id,
        "wsi": str(pair.get("case_id") or sample_id),
        "case_id": str(pair.get("case_id") or sample_id),
        "dataset": dataset,
        "organ": DATASET_TO_ORGAN[dataset],
        "image_path": str(image_path),
        "id_mask_path": str(tissue_path),
        "cellvit_id_mask_path": str(nuclei_path),
        "width": int(coarse.shape[1]),
        "height": int(coarse.shape[0]),
        "valid_pixels": valid_pixels,
        "pix_tumor": counts["tumor"],
        "pix_stroma": counts["stroma"],
        "pix_necrosis": counts["necrosis"],
        "pix_immune_infiltrate": counts["immune"],
        "pix_normal_epithelium": counts["normal"],
        "pix_blood_vessel": counts["vessel"],
        "pix_other_tissue": counts["other"],
        "pix_tumor_edit_non_tumor_context": int(
            np.count_nonzero(non_tumor_context)
        ),
        "pix_tumor_non_tumor_boundary_support": (
            tumor_non_tumor_boundary_pixels
        ),
        "selection_score": float(foreground_fraction),
        "annotated_coarse_ids": native_ids,
        "annotation_provenance": "human_dataset_annotation",
        "annotation_metadata": str(metadata_path),
        "annotation_mask_sha256": _sha256_file(tissue_path),
        "de_novo_edit": False,
    }


def _validate_annotation_paths(
    dataset: str,
    *,
    image_path: Path,
    tissue_path: Path,
    nuclei_path: Path,
) -> None:
    lower_parts = {part.lower() for part in tissue_path.parts}
    forbidden = {"model_masks", "predictions", "segmentator_outputs"}
    if lower_parts & forbidden:
        raise ValueError(
            f"G2 forbids machine-generated tissue masks: {tissue_path}"
        )
    if tissue_path.parent.name != "tissue_masks":
        raise ValueError(
            "G2 tissue masks must come from a dataset tissue_masks directory: "
            f"{tissue_path}"
        )
    patch_root = f"{dataset}_PATCHES".lower()
    if patch_root not in lower_parts:
        raise ValueError(
            f"{dataset} mask is outside its dataset annotation package: "
            f"{tissue_path}"
        )
    if image_path.parent.name != "images":
        raise ValueError(f"Unexpected dataset image path: {image_path}")
    if nuclei_path.parent.name != "nuclei_masks":
        raise ValueError(f"Unexpected dataset nuclei path: {nuclei_path}")


def eligible_primitives(row: Mapping[str, Any]) -> tuple[str, ...]:
    fractions = _coarse_fractions(row)
    tumor = fractions["tumor"]
    stroma = fractions["stroma"]
    necrosis = fractions["necrosis"]
    immune = fractions["immune"]
    normal = fractions["normal"]
    other = fractions["other"]
    named_non_tumor_context = stroma + normal + immune
    non_tumor_context = named_non_tumor_context + other
    tumor_increase_context_sufficient = bool(
        non_tumor_context >= MIN_MODERATE_SOURCE_FRACTION
        and (
            named_non_tumor_context >= MIN_EXISTING_TARGET_FRACTION
            or non_tumor_context >= MIN_OTHER_ONLY_TUMOR_CONTEXT_FRACTION
        )
    )
    tumor_decrease_context_sufficient = bool(
        named_non_tumor_context
        >= MIN_TUMOR_DECREASE_NAMED_BACKFILL_FRACTION
        or non_tumor_context >= MIN_OTHER_ONLY_TUMOR_CONTEXT_FRACTION
    )
    tumor_boundary_support = int(
        row.get(
            "pix_tumor_non_tumor_boundary_support",
            MIN_TUMOR_NON_TUMOR_BOUNDARY_PIXELS,
        )
        or 0
    )
    tumor_non_tumor_adjacent = bool(
        tumor_boundary_support >= MIN_TUMOR_NON_TUMOR_BOUNDARY_PIXELS
    )
    native = {int(value) for value in row.get("annotated_coarse_ids", ())}

    eligibility = {
        "tumor_increase": (
            1 in native
            and tumor >= MIN_EXISTING_TARGET_FRACTION
            and tumor_increase_context_sufficient
            and tumor_non_tumor_adjacent
        ),
        "tumor_decrease": (
            1 in native
            and tumor >= MIN_MODERATE_SOURCE_FRACTION
            and tumor_decrease_context_sufficient
            and tumor_non_tumor_adjacent
        ),
        "stroma_increase": (
            2 in native
            and stroma >= MIN_EXISTING_TARGET_FRACTION
            and normal + other + immune >= MIN_MODERATE_SOURCE_FRACTION
        ),
        "stroma_decrease": (
            2 in native
            and stroma >= MIN_MODERATE_SOURCE_FRACTION
            and tumor + normal + other >= MIN_EXISTING_TARGET_FRACTION
        ),
        "immune_increase": (
            4 in native
            and immune >= MIN_EXISTING_TARGET_FRACTION
            and stroma >= MIN_MODERATE_SOURCE_FRACTION
        ),
        "immune_decrease": (
            4 in native
            and immune >= MIN_MODERATE_SOURCE_FRACTION
            and stroma + other + tumor >= MIN_EXISTING_TARGET_FRACTION
        ),
        "necrosis_increase": (
            3 in native
            and necrosis >= MIN_EXISTING_TARGET_FRACTION
            and tumor >= MIN_MODERATE_SOURCE_FRACTION
        ),
        "necrosis_decrease": (
            3 in native
            and necrosis >= MIN_MODERATE_SOURCE_FRACTION
            and stroma + tumor >= MIN_EXISTING_TARGET_FRACTION
        ),
    }
    return tuple(name for name in PRIMITIVE_BY_NAME if eligibility[name])


def select_cohort(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    return_policy: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible_rows = [dict(row) for row in rows if eligible_primitives(row)]
    policy = _selection_policy(eligible_rows)
    policy["original_exact_quota_issues"] = _quota_feasibility_issues(policy)
    assignments = [
        (row_index, primitive)
        for row_index, row in enumerate(eligible_rows)
        for primitive in eligible_primitives(row)
    ]
    if not assignments:
        raise RuntimeError("No eligible G2 assignments exist.")

    selected, used_caps = _solve_selection(
        eligible_rows,
        assignments,
        policy=policy,
        seed=seed,
    )
    policy["actual_wsi_caps"] = used_caps
    policy["primitive_quotas"] = _counts(selected, "g2_primitive")
    policy["minimum_primitive_quota"] = min(
        policy["primitive_quotas"].values()
    )
    policy["constraint_adjustments"] = _constraint_adjustments(policy)
    validate_selection(selected, policy=policy)
    ordered = sorted(
        selected,
        key=lambda row: (
            tuple(ORGAN_TO_DATASET).index(str(row["organ"])),
            tuple(PRIMITIVE_BY_NAME).index(str(row["g2_primitive"])),
            _stable_hash(str(row["stem"]), seed),
        ),
    )
    if return_policy:
        return ordered, policy
    return ordered


def _selection_policy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    capacities = {
        (organ, primitive): sum(
            row["organ"] == organ and primitive in eligible_primitives(row)
            for row in rows
        )
        for organ in ORGAN_TO_DATASET
        for primitive in PRIMITIVE_BY_NAME
    }
    primitive_support = {
        primitive: [
            organ
            for organ in ORGAN_TO_DATASET
            if capacities[(organ, primitive)] > MIN_RESERVES_PER_ACTIVE_CELL
        ]
        for primitive in PRIMITIVE_BY_NAME
    }
    organ_support = {
        organ: [
            primitive
            for primitive in PRIMITIVE_BY_NAME
            if capacities[(organ, primitive)] > MIN_RESERVES_PER_ACTIVE_CELL
        ]
        for organ in ORGAN_TO_DATASET
    }
    wsi_counts = {
        organ: len(
            {
                str(row["wsi"])
                for row in rows
                if row["organ"] == organ
            }
        )
        for organ in ORGAN_TO_DATASET
    }
    base_caps = {
        organ: max(
            DESIRED_MAX_PATCHES_PER_WSI,
            math.ceil(ORGAN_QUOTA / max(1, wsi_counts[organ])),
        )
        for organ in ORGAN_TO_DATASET
    }
    policy = {
        "annotation_policy": "human_dataset_annotation_only",
        "capacities": {
            organ: {
                primitive: capacities[(organ, primitive)]
                for primitive in PRIMITIVE_BY_NAME
            }
            for organ in ORGAN_TO_DATASET
        },
        "primitive_supported_organs": primitive_support,
        "organ_supported_primitives": organ_support,
        "min_organs_per_primitive": {
            primitive: min(
                DESIRED_MIN_ORGANS_PER_PRIMITIVE,
                len(primitive_support[primitive]),
            )
            for primitive in PRIMITIVE_BY_NAME
        },
        "min_primitives_per_organ": {
            organ: min(
                DESIRED_MIN_PRIMITIVES_PER_ORGAN,
                len(organ_support[organ]),
            )
            for organ in ORGAN_TO_DATASET
        },
        "unique_wsi_counts": wsi_counts,
        "base_wsi_caps": base_caps,
        "desired_primitive_quota": DESIRED_PRIMITIVE_QUOTA,
        "minimum_effective_primitive_quota": (
            MIN_EFFECTIVE_PRIMITIVE_QUOTA
        ),
        "reserve_per_active_cell": MIN_RESERVES_PER_ACTIVE_CELL,
    }
    policy["balanced_cell_targets"] = _balanced_cell_targets(policy)
    return policy


def _balanced_cell_targets(
    policy: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Allocate each organ across annotation-supported primitives.

    Each active cell keeps the configured same-cell reserve. Capacity-limited
    cells are filled first, then the remaining organ quota is distributed as
    evenly as possible across the other supported primitives.
    """

    targets: dict[str, dict[str, int]] = {}
    for organ in ORGAN_TO_DATASET:
        supported = list(policy["organ_supported_primitives"][organ])
        usable = {
            primitive: max(
                0,
                int(policy["capacities"][organ][primitive])
                - MIN_RESERVES_PER_ACTIVE_CELL,
            )
            for primitive in supported
        }
        assigned = {primitive: 0 for primitive in PRIMITIVE_BY_NAME}
        remaining = ORGAN_QUOTA
        active = list(supported)
        while active:
            share, remainder = divmod(remaining, len(active))
            limited = [
                primitive
                for primitive in active
                if usable[primitive] < share
            ]
            if limited:
                for primitive in limited:
                    assigned[primitive] = usable[primitive]
                    remaining -= usable[primitive]
                    active.remove(primitive)
                continue
            for index, primitive in enumerate(active):
                assigned[primitive] = share + int(index < remainder)
            remaining = 0
            break
        if remaining:
            raise RuntimeError(
                f"{organ} lacks annotation capacity for {ORGAN_QUOTA} cases."
            )
        targets[organ] = assigned
    return targets


def _quota_feasibility_issues(
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Report annotation-support bottlenecks before invoking the MILP."""

    grouped: dict[tuple[str, ...], list[str]] = {}
    for primitive, organs in policy["primitive_supported_organs"].items():
        key = tuple(sorted(str(organ) for organ in organs))
        grouped.setdefault(key, []).append(str(primitive))
    issues: list[dict[str, Any]] = []
    for organs, primitives in sorted(grouped.items()):
        required = len(primitives) * DESIRED_PRIMITIVE_QUOTA
        available = len(organs) * ORGAN_QUOTA
        if required > available:
            issues.append(
                {
                    "constraint": "annotation_supported_organ_capacity",
                    "primitives": sorted(primitives),
                    "supported_organs": list(organs),
                    "required_assignments": required,
                    "available_organ_slots": available,
                    "deficit": required - available,
                }
            )
    for organ, primitives in policy["organ_supported_primitives"].items():
        available = len(primitives) * DESIRED_PRIMITIVE_QUOTA
        if available < ORGAN_QUOTA:
            issues.append(
                {
                    "constraint": "annotation_supported_primitive_capacity",
                    "organ": organ,
                    "supported_primitives": sorted(primitives),
                    "required_organ_slots": ORGAN_QUOTA,
                    "available_primitive_slots": available,
                    "deficit": ORGAN_QUOTA - available,
                }
            )
    return issues


def _solve_selection(
    rows: Sequence[dict[str, Any]],
    assignments: Sequence[tuple[int, str]],
    *,
    policy: Mapping[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    base_caps = {
        organ: int(value)
        for organ, value in policy["base_wsi_caps"].items()
    }
    last_message = "not attempted"
    for increment in range(0, 9):
        caps = {organ: cap + increment for organ, cap in base_caps.items()}
        result = _solve_selection_once(
            rows,
            assignments,
            policy=policy,
            wsi_caps=caps,
            seed=seed,
        )
        if result[0] is not None:
            return result[0], caps
        last_message = result[1]
    raise RuntimeError(
        "Exact human-annotation G2 assignment is infeasible even after "
        f"annotation-aware WSI balancing: {last_message}"
    )


def _solve_selection_once(
    rows: Sequence[dict[str, Any]],
    assignments: Sequence[tuple[int, str]],
    *,
    policy: Mapping[str, Any],
    wsi_caps: Mapping[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]] | None, str]:
    patch_indices: dict[int, list[int]] = {}
    wsi_indices: dict[tuple[str, str], list[int]] = {}
    organ_indices = {organ: [] for organ in ORGAN_TO_DATASET}
    primitive_indices = {primitive: [] for primitive in PRIMITIVE_BY_NAME}
    cell_indices = {
        (organ, primitive): []
        for organ in ORGAN_TO_DATASET
        for primitive in PRIMITIVE_BY_NAME
    }
    for index, (row_index, primitive) in enumerate(assignments):
        row = rows[row_index]
        organ = str(row["organ"])
        patch_indices.setdefault(row_index, []).append(index)
        wsi_indices.setdefault((organ, str(row["wsi"])), []).append(index)
        organ_indices[organ].append(index)
        primitive_indices[primitive].append(index)
        cell_indices[(organ, primitive)].append(index)

    cell_keys = list(cell_indices)
    assignment_count = len(assignments)
    cell_offset = assignment_count
    positive_deviation_offset = cell_offset + len(cell_keys)
    negative_deviation_offset = positive_deviation_offset + len(cell_keys)
    maximum_deviation_index = negative_deviation_offset + len(cell_keys)
    variable_count = maximum_deviation_index + 1
    objective = np.zeros(variable_count, dtype=float)
    for index, (row_index, primitive) in enumerate(assignments):
        row = rows[row_index]
        objective[index] = (
            -1e-3 * float(row.get("selection_score", 0.0))
            + 1e-9
            * _hash_fraction(f"{row['stem']}|{primitive}", seed)
        )
    objective[
        positive_deviation_offset:negative_deviation_offset
    ] = 1.0
    objective[
        negative_deviation_offset:maximum_deviation_index
    ] = 1.0
    objective[maximum_deviation_index] = 10000.0

    constraints = (
        len(patch_indices)
        + len(wsi_indices)
        + len(ORGAN_TO_DATASET)
        + 2 * len(cell_keys)
        + len(cell_keys)
        + len(cell_keys)
        + len(PRIMITIVE_BY_NAME)
        + len(cell_keys)
        + len(cell_keys)
    )
    matrix = lil_matrix((constraints, variable_count), dtype=float)
    lower = np.full(constraints, -np.inf, dtype=float)
    upper = np.full(constraints, np.inf, dtype=float)
    cursor = 0

    for indices in patch_indices.values():
        matrix[cursor, indices] = 1.0
        upper[cursor] = 1.0
        cursor += 1
    for (organ, _wsi), indices in wsi_indices.items():
        matrix[cursor, indices] = 1.0
        upper[cursor] = float(wsi_caps[organ])
        cursor += 1
    for organ in ORGAN_TO_DATASET:
        matrix[cursor, organ_indices[organ]] = 1.0
        lower[cursor] = upper[cursor] = ORGAN_QUOTA
        cursor += 1
    cell_variable = {
        key: cell_offset + index for index, key in enumerate(cell_keys)
    }
    for key, indices in cell_indices.items():
        variable = cell_variable[key]
        if indices:
            matrix[cursor, indices] = 1.0
        matrix[cursor, variable] = -float(ORGAN_QUOTA)
        upper[cursor] = 0.0
        cursor += 1
        if indices:
            matrix[cursor, indices] = -1.0
        matrix[cursor, variable] = 1.0
        upper[cursor] = 0.0
        cursor += 1
    reserve_consumption: dict[tuple[str, str], set[int]] = {
        key: set() for key in cell_keys
    }
    for index, (row_index, _assigned_primitive) in enumerate(assignments):
        row = rows[row_index]
        organ = str(row["organ"])
        for reserve_primitive in eligible_primitives(row):
            reserve_consumption[(organ, reserve_primitive)].add(index)
    capacities = policy["capacities"]
    for key in cell_keys:
        organ, primitive = key
        matrix[cursor, sorted(reserve_consumption[key])] = 1.0
        matrix[cursor, cell_variable[key]] = float(
            MIN_RESERVES_PER_ACTIVE_CELL
        )
        upper[cursor] = float(capacities[organ][primitive])
        cursor += 1
    for key in cell_keys:
        organ, primitive = key
        supported = primitive in policy["organ_supported_primitives"][organ]
        matrix[cursor, cell_variable[key]] = 1.0
        lower[cursor] = upper[cursor] = float(supported)
        cursor += 1
    for primitive in PRIMITIVE_BY_NAME:
        matrix[cursor, primitive_indices[primitive]] = 1.0
        lower[cursor] = float(MIN_EFFECTIVE_PRIMITIVE_QUOTA)
        cursor += 1
    for cell_index, key in enumerate(cell_keys):
        organ, primitive = key
        matrix[cursor, cell_indices[key]] = 1.0
        matrix[
            cursor,
            positive_deviation_offset + cell_index,
        ] = -1.0
        matrix[
            cursor,
            negative_deviation_offset + cell_index,
        ] = 1.0
        target = policy["balanced_cell_targets"][organ][primitive]
        lower[cursor] = upper[cursor] = float(target)
        cursor += 1
    for cell_index, _key in enumerate(cell_keys):
        matrix[
            cursor,
            positive_deviation_offset + cell_index,
        ] = 1.0
        matrix[
            cursor,
            negative_deviation_offset + cell_index,
        ] = 1.0
        matrix[cursor, maximum_deviation_index] = -1.0
        upper[cursor] = 0.0
        cursor += 1
    if cursor != constraints:
        raise AssertionError("Internal G2 selection constraint mismatch.")

    result = milp(
        c=objective,
        integrality=np.concatenate(
            (
                np.ones(assignment_count + len(cell_keys), dtype=int),
                np.zeros(2 * len(cell_keys) + 1, dtype=int),
            )
        ),
        bounds=Bounds(
            np.zeros(variable_count, dtype=float),
            np.concatenate(
                (
                    np.ones(assignment_count + len(cell_keys)),
                    np.full(2 * len(cell_keys), float(ORGAN_QUOTA)),
                    np.asarray([float(ORGAN_QUOTA)]),
                )
            ),
        ),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 180.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        return None, str(result.message)

    selected = []
    for index, value in enumerate(result.x[:assignment_count]):
        if value <= 0.5:
            continue
        row_index, primitive = assignments[index]
        selected.append(
            {
                **rows[row_index],
                "g2_organ": rows[row_index]["organ"],
                "g2_primitive": primitive,
            }
        )
    return selected, str(result.message)


def validate_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> None:
    expected_total = len(ORGAN_TO_DATASET) * ORGAN_QUOTA
    if len(rows) != expected_total:
        raise ValueError(f"G2 cohort must contain {expected_total} rows.")
    stems = [f"{row['dataset']}:{row['stem']}" for row in rows]
    if len(set(stems)) != len(stems):
        raise ValueError("G2 cohort contains duplicate patches.")
    if any(
        row.get("annotation_provenance") != "human_dataset_annotation"
        for row in rows
    ):
        raise ValueError("Every G2 source mask must be a dataset annotation.")
    organ_counts = _counts(rows, "g2_organ")
    primitive_counts = _counts(rows, "g2_primitive")
    if any(organ_counts.get(name) != ORGAN_QUOTA for name in ORGAN_TO_DATASET):
        raise ValueError(f"Organ quotas are not exact: {organ_counts}")
    if sum(primitive_counts.values()) != expected_total:
        raise ValueError(f"Primitive quotas do not sum to 600: {primitive_counts}")
    for row in rows:
        if str(row["g2_primitive"]) not in eligible_primitives(row):
            raise ValueError(
                f"Ineligible or de-novo edit selected for {row['stem']}."
            )
    if policy is None:
        return
    if primitive_counts != dict(policy["primitive_quotas"]):
        raise ValueError(
            "Selection primitive counts differ from frozen effective quotas: "
            f"{primitive_counts} != {policy['primitive_quotas']}"
        )
    for primitive, minimum in policy["min_organs_per_primitive"].items():
        covered = {
            str(row["g2_organ"])
            for row in rows
            if row["g2_primitive"] == primitive
        }
        if len(covered) < int(minimum):
            raise ValueError(
                f"{primitive} covers {len(covered)} organs, expected {minimum}."
            )
    for organ, minimum in policy["min_primitives_per_organ"].items():
        covered = {
            str(row["g2_primitive"])
            for row in rows
            if row["g2_organ"] == organ
        }
        if len(covered) < int(minimum):
            raise ValueError(
                f"{organ} covers {len(covered)} primitives, expected {minimum}."
            )
    wsi_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["g2_organ"]), str(row["wsi"]))
        wsi_counts[key] = wsi_counts.get(key, 0) + 1
        if wsi_counts[key] > int(
            policy["actual_wsi_caps"][str(row["g2_organ"])]
        ):
            raise ValueError(f"WSI cap exceeded for {key}.")


def build_product_manifest(
    selected: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    source_manifest: str | Path,
    release_path: str,
    selection_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = []
    for index, row in enumerate(selected, start=1):
        primitive = PRIMITIVE_BY_NAME[str(row["g2_primitive"])]
        source_fractions = _coarse_fractions(row)
        named_non_tumor_fraction = sum(
            source_fractions[name]
            for name in ("stroma", "normal", "immune")
        )
        non_tumor_context_fraction = (
            named_non_tumor_fraction + source_fractions["other"]
        )
        case_hash = _stable_hash(
            f"{row['dataset']}|{row['stem']}|{primitive.name}",
            seed,
        )[:10]
        cases.append(
            {
                "case_id": (
                    f"g2_{index:03d}_{row['organ']}_"
                    f"{primitive.name}_{case_hash}"
                ),
                "condition_id": primitive.name,
                "sample_id": str(row["stem"]),
                "dataset": str(row["dataset"]),
                "profile": str(row["dataset"]),
                "organ": str(row["organ"]),
                "primitive_group": primitive.name.rsplit("_", 1)[0],
                "primitive": primitive.internal_primitive,
                "g2_primitive": primitive.name,
                "expected_primitives": [primitive.internal_primitive],
                "source_image": str(row["image_path"]),
                "source_tissue_mask": str(row["id_mask_path"]),
                "source_nuclei_mask": str(row["cellvit_id_mask_path"]),
                "source_mask_annotation_provenance": (
                    "human_dataset_annotation"
                ),
                "source_mask_sha256": str(row["annotation_mask_sha256"]),
                "instruction": primitive.instruction,
                "old_prompt": "",
                "new_prompt": "",
                "source_labels": list(primitive.source_labels),
                "target_label": primitive.target_label,
                "target_labels": list(
                    _allowed_target_labels(primitive, str(row["dataset"]))
                ),
                "expected_area_bucket": [0.14, 0.24],
                "minimum_changed_area_fraction_image": (
                    MIN_CHANGED_AREA_FRACTION_IMAGE
                ),
                "strength": "moderate",
                "projection_mode": "organic_v2",
                "organic_seed": seed,
                "api_model": "gpt-4.1-mini",
                "wsi": str(row["wsi"]),
                "source_coarse_fractions": source_fractions,
                "tumor_edit_context": {
                    "named_non_tumor_fraction": named_non_tumor_fraction,
                    "non_tumor_context_fraction": (
                        non_tumor_context_fraction
                    ),
                    "tumor_boundary_support_pixels": int(
                        row.get(
                            "pix_tumor_non_tumor_boundary_support",
                            0,
                        )
                        or 0
                    ),
                },
                "annotated_coarse_ids": _as_int_list(
                    row["annotated_coarse_ids"]
                ),
                "de_novo_edit": False,
                "status": "selected",
            }
        )
    return {
        "schema_version": "2.0",
        "description": (
            "Deterministic G2-600 cohort sourced only from original dataset "
            "annotation masks. Generation and evaluation are delegated to the "
            "online product workflow."
        ),
        "source_eval_metadata": str(source_manifest),
        "source_mask_policy": "human_dataset_annotation_only",
        "selection_seed": seed,
        "datasets": list(DATASET_TO_ORGAN),
        "organs": list(ORGAN_TO_DATASET),
        "selection_policy": dict(selection_policy or {}),
        "defaults": {
            "projection_mode": "organic_v2",
            "organic_seed": seed,
            "status": "selected",
        },
        "mask_review_policy": {
            "minimum_changed_area_fraction_image": (
                MIN_CHANGED_AREA_FRACTION_IMAGE
            ),
            "purpose": "human_visible_edit_review",
        },
        "runtime": {
            "edit_variants": [
                {"variant_id": "instruction", "edit_mode": "instruction"}
            ],
            "parser": {
                "instruction_parser": "api",
                "api_base_url": "https://api.cursorai.art/v1",
                "api_key_env": "OPENAI_API_KEY",
                "api_model": "gpt-4.1-mini",
            },
            "contour": {
                "provider": "api-multimodal",
                "api_base_url": "https://api.cursorai.art/v1",
                "api_key_env": "OPENAI_API_KEY",
                "api_model": "gpt-4.1-mini",
                "api_image_detail": "high",
                "max_attempts": 4,
                "max_regions": 8,
                "max_points_per_region": 64,
            },
            "cell": {
                "cell_fill_mode": "probnet",
                "crossing_cell_policy": "delete",
                "probnet_device": "auto",
                "probnet_gamma_values": "3.0",
            },
            "generation": {
                "generation_mode": "agentic",
                "cross_backend": "cross-v1",
                "route_threshold": 0.30,
                "joint_force_cross_min_generation_support_fraction": 0.50,
                "device": "cuda",
            },
            "verification": {"product_release": release_path},
        },
        "g2_constraints": {
            "cohort_size": 600,
            "organ_quota": ORGAN_QUOTA,
            "desired_primitive_quota": DESIRED_PRIMITIVE_QUOTA,
            "minimum_effective_primitive_quota": (
                MIN_EFFECTIVE_PRIMITIVE_QUOTA
            ),
            "effective_primitive_quotas": dict(
                (selection_policy or {}).get("primitive_quotas") or {}
            ),
            "primitive_quota_policy": (
                "organ_stratified_minimax_with_annotation_capacity"
            ),
            "strength": "moderate",
            "de_novo_edits": "forbidden",
            "grade_and_fine_edits": "forbidden",
            "background_edits": "forbidden",
            "mask_source": "human_dataset_annotation_only",
            "tumor_edit_context_policy": {
                "tumor_increase_non_tumor_fraction_min": (
                    MIN_MODERATE_SOURCE_FRACTION
                ),
                "tumor_decrease_named_backfill_fraction_min": (
                    MIN_TUMOR_DECREASE_NAMED_BACKFILL_FRACTION
                ),
                "other_only_non_tumor_fraction_min": (
                    MIN_OTHER_ONLY_TUMOR_CONTEXT_FRACTION
                ),
                "boundary_radius_pixels": (
                    TUMOR_NON_TUMOR_BOUNDARY_RADIUS
                ),
                "boundary_support_pixels_min": (
                    MIN_TUMOR_NON_TUMOR_BOUNDARY_PIXELS
                ),
                "other_background_ignore_equivalence": False,
                "dataset_or_organ_specific_exceptions": False,
            },
        },
        "cases": cases,
    }


def build_reserves(
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    per_cell: int = 25,
) -> list[dict[str, Any]]:
    selected_stems = {
        (str(row["dataset"]), str(row["stem"])) for row in selected
    }
    reserves: list[dict[str, Any]] = []
    for organ in ORGAN_TO_DATASET:
        for primitive in PRIMITIVE_BY_NAME:
            candidates = [
                row
                for row in rows
                if row["organ"] == organ
                and (str(row["dataset"]), str(row["stem"]))
                not in selected_stems
                and primitive in eligible_primitives(row)
            ]
            candidates.sort(
                key=lambda row: (
                    -float(row.get("selection_score", 0.0)),
                    _stable_hash(
                        f"{row['dataset']}|{row['stem']}|{primitive}|reserve",
                        seed,
                    ),
                )
            )
            for rank, row in enumerate(candidates[:per_cell], start=1):
                reserves.append(
                    {
                        **dict(row),
                        "g2_organ": organ,
                        "g2_primitive": primitive,
                        "reserve_rank": rank,
                    }
                )
    return reserves


def validate_reserves(
    selected: Sequence[Mapping[str, Any]],
    reserves: Sequence[Mapping[str, Any]],
    *,
    minimum_per_active_cell: int = MIN_RESERVES_PER_ACTIVE_CELL,
) -> None:
    active_cells = {
        (str(row["g2_organ"]), str(row["g2_primitive"]))
        for row in selected
    }
    reserve_counts = _cell_counts(reserves)
    missing = {
        f"{organ}:{primitive}": reserve_counts.get((organ, primitive), 0)
        for organ, primitive in sorted(active_cells)
        if reserve_counts.get((organ, primitive), 0)
        < minimum_per_active_cell
    }
    if missing:
        raise ValueError(
            "Active G2 organ/primitive cells lack frozen same-cell reserves: "
            f"{missing}; required={minimum_per_active_cell}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-metadata",
        "--candidate-manifest",
        dest="eval_metadata",
        type=Path,
        default=DEFAULT_EVAL_METADATA,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--release",
        default="benchmark_configs/releases/online_agent_product_v1.json",
    )
    parser.add_argument("--validate-paths", action="store_true")
    parser.add_argument(
        "--reserves-per-active-cell",
        type=int,
        default=25,
        help="Maximum deterministic reserve depth for each organ/primitive cell.",
    )
    args = parser.parse_args(argv)

    rows = load_candidate_rows(args.eval_metadata)
    selected, policy = select_cohort(
        rows,
        seed=args.seed,
        return_policy=True,
    )
    if args.validate_paths:
        _validate_paths(selected)
    if args.reserves_per_active_cell < MIN_RESERVES_PER_ACTIVE_CELL:
        raise ValueError(
            "--reserves-per-active-cell must be at least "
            f"{MIN_RESERVES_PER_ACTIVE_CELL}."
        )
    reserves = build_reserves(
        rows,
        selected,
        seed=args.seed,
        per_cell=args.reserves_per_active_cell,
    )
    validate_reserves(selected, reserves)
    product_manifest = build_product_manifest(
        selected,
        seed=args.seed,
        source_manifest=args.eval_metadata,
        release_path=args.release,
        selection_policy=policy,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "g2_600_product_manifest.json"
    selection_path = args.output / "g2_600_selection.csv"
    reserves_path = args.output / "g2_600_reserves.csv"
    summary_path = args.output / "g2_600_selection_summary.json"
    _write_json(manifest_path, product_manifest)
    _write_csv(selection_path, selected)
    _write_csv(reserves_path, reserves)
    summary = {
        "schema_version": 2,
        "eval_metadata": str(args.eval_metadata),
        "source_mask_policy": "human_dataset_annotation_only",
        "selection_seed": args.seed,
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "reserve_count": len(reserves),
        "requested_reserves_per_active_cell": args.reserves_per_active_cell,
        "organ_counts": _counts(selected, "g2_organ"),
        "primitive_counts": _counts(selected, "g2_primitive"),
        "quota_matrix": _quota_matrix_from_selection(selected),
        "reserve_counts_by_cell": {
            f"{organ}:{primitive}": count
            for (organ, primitive), count in sorted(
                _cell_counts(reserves).items()
            )
        },
        "selection_policy": policy,
        "selection_sha256": _sha256_file(selection_path),
        "product_manifest": str(manifest_path),
        "product_manifest_sha256": _sha256_file(manifest_path),
        "reserves": str(reserves_path),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _constraint_adjustments(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    adjustments: list[dict[str, Any]] = list(
        policy.get("original_exact_quota_issues") or []
    )
    for primitive, actual in policy["primitive_quotas"].items():
        if int(actual) != DESIRED_PRIMITIVE_QUOTA:
            adjustments.append(
                {
                    "constraint": "primitive_quota",
                    "item": primitive,
                    "desired": DESIRED_PRIMITIVE_QUOTA,
                    "effective": int(actual),
                    "reason": "human_annotation_support_balancing",
                }
            )
    for primitive, actual in policy["min_organs_per_primitive"].items():
        if int(actual) < DESIRED_MIN_ORGANS_PER_PRIMITIVE:
            adjustments.append(
                {
                    "constraint": "min_organs_per_primitive",
                    "item": primitive,
                    "desired": DESIRED_MIN_ORGANS_PER_PRIMITIVE,
                    "effective": int(actual),
                    "reason": "ground_truth_annotation_support",
                    "supported_organs": policy["primitive_supported_organs"][
                        primitive
                    ],
                }
            )
    for organ, actual in policy["min_primitives_per_organ"].items():
        if int(actual) < DESIRED_MIN_PRIMITIVES_PER_ORGAN:
            adjustments.append(
                {
                    "constraint": "min_primitives_per_organ",
                    "item": organ,
                    "desired": DESIRED_MIN_PRIMITIVES_PER_ORGAN,
                    "effective": int(actual),
                    "reason": "ground_truth_annotation_support",
                    "supported_primitives": policy[
                        "organ_supported_primitives"
                    ][organ],
                }
            )
    for organ, actual in policy["actual_wsi_caps"].items():
        if int(actual) > DESIRED_MAX_PATCHES_PER_WSI:
            adjustments.append(
                {
                    "constraint": "max_patches_per_wsi",
                    "item": organ,
                    "desired": DESIRED_MAX_PATCHES_PER_WSI,
                    "effective": int(actual),
                    "reason": "insufficient_unique_wsi_in_fixed_eval_metadata",
                    "unique_wsi": policy["unique_wsi_counts"][organ],
                }
            )
    return adjustments


def _coarse_fractions(row: Mapping[str, Any]) -> dict[str, float]:
    pixels = max(1.0, float(row.get("valid_pixels", 0)))
    return {
        "tumor": _float(row.get("pix_tumor")) / pixels,
        "stroma": _float(row.get("pix_stroma")) / pixels,
        "necrosis": _float(row.get("pix_necrosis")) / pixels,
        "immune": _float(row.get("pix_immune_infiltrate")) / pixels,
        "normal": _float(row.get("pix_normal_epithelium")) / pixels,
        "vessel": _float(row.get("pix_blood_vessel")) / pixels,
        "other": _float(row.get("pix_other_tissue")) / pixels,
    }


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [
                item.strip()
                for item in value.strip("[]").split(",")
                if item.strip()
            ]
    if not isinstance(value, (list, tuple, set)):
        return []
    return [int(item) for item in value]


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _hash_fraction(value: str, seed: int) -> float:
    return int(_stable_hash(value, seed)[:12], 16) / float(16**12 - 1)


def _quota_matrix_from_selection(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    matrix = {
        organ: {primitive: 0 for primitive in PRIMITIVE_BY_NAME}
        for organ in ORGAN_TO_DATASET
    }
    for row in rows:
        matrix[str(row["g2_organ"])][str(row["g2_primitive"])] += 1
    return matrix


def _counts(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        name = str(row[field])
        result[name] = result.get(name, 0) + 1
    return dict(sorted(result.items()))


def _cell_counts(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], int]:
    result: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["g2_organ"]), str(row["g2_primitive"]))
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _validate_paths(rows: Iterable[Mapping[str, Any]]) -> None:
    for row in rows:
        for field in ("image_path", "id_mask_path", "cellvit_id_mask_path"):
            path = Path(str(row[field]))
            if not path.is_file():
                raise FileNotFoundError(f"Missing {field}: {path}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            writer.writerow(serializable)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    sys.exit(main())
