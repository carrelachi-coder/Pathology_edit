#!/usr/bin/env python3
"""Run five mask-only edits per executable lung/oral/skin primitive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase3_joint_edit_refine.models import (
    CellCountExtentBudget,
    JointAreaBudget,
    JointCaseContext,
)
from phase3_joint_edit_refine.nuclei import load_nuclei_mask
from phase3_joint_edit_refine.semantic_parser import RuleBasedSemanticParser
from phase3_joint_edit_refine.visualization import NUCLEI_RGB
from phase3_mask_edit_refine.evidence import load_id_mask, sha256_file
from phase3_mask_edit_refine.visualization import id_mask_to_rgb
from scripts.run_glas_primitive_mask_review import _native_authority

SCHEMA_VERSION = "lung-oral-skin-cross-meta-mask-review-v4"
DATASET_CONFIG = {
    "lung": ("IGNITE", "lung-carcinoma-v1", "ignite-semantic-v1", "lung-cellvit-source-first-v1"),
    "oral": ("ORCA", "oral-squamous-cell-carcinoma-v1", "orca-semantic-v1", "oral-scc-cellvit-source-first-v1"),
    "skin": ("PUMA", "melanoma-v1", "puma-semantic-v1", "melanoma-cellvit-source-first-v1"),
}
INSTRUCTIONS = {
    "cell-type-abundance-increase-v1": "Increase immune cells in the selected region.",
    "cell-type-abundance-decrease-v1": "Decrease immune cells in the selected region.",
    "cellularity-increase-v1": "Increase local cellularity.",
    "cellularity-decrease-v1": "Decrease local cellularity.",
    "neoplastic-cell-abundance-increase-v1": "Increase neoplastic cells.",
    "neoplastic-cell-abundance-decrease-v1": "Decrease neoplastic cells.",
    "generic-inflammatory-cell-abundance-increase-v1": "Increase generic inflammatory-cell abundance.",
    "generic-inflammatory-cell-abundance-decrease-v1": "Decrease generic inflammatory-cell abundance.",
    "tumor-burden-increase-v1": "Increase tumor burden.",
    "cohesive-boundary-expansion-v1": "Expand the tumor boundary locally.",
    "generic-immune-infiltrate-increase-v1": "Increase the generic immune infiltrate.",
    "generic-immune-infiltrate-decrease-v1": "Decrease the generic immune infiltrate.",
    "necrosis-appearance-v1": "Increase tumor necrosis.",
    "necrosis-resolution-v1": "Reduce tumor necrosis.",
    "infiltrative-nest-cord-extension-v1": "Add a narrow connected tumor cord.",
    "invasive-tumor-footprint-decrease-v1": "Simulate a post-treatment response by decreasing tumor area.",
    "stroma-increase-v1": "Simulate a post-treatment response by increasing operational stroma.",
    "residual-tumor-fragmentation-v1": "Simulate post-treatment residual disease by fragmenting residual tumor into controlled foci.",
    "peritumoral-neoplastic-scatter-increase-v1": "Add scattered tumor cells near the tumor boundary.",
    "peritumoral-small-cluster-increase-v1": "Add peritumoral small tumor-cell clusters.",
}
LOCAL_CELL = frozenset(
    {
        "cell-type-abundance-increase-v1",
        "cell-type-abundance-decrease-v1",
        "cellularity-increase-v1",
        "cellularity-decrease-v1",
        "neoplastic-cell-abundance-increase-v1",
        "neoplastic-cell-abundance-decrease-v1",
        "generic-inflammatory-cell-abundance-increase-v1",
        "generic-inflammatory-cell-abundance-decrease-v1",
    }
)
PERITUMORAL = frozenset(
    {
        "peritumoral-neoplastic-scatter-increase-v1",
        "peritumoral-small-cluster-increase-v1",
    }
)
CELL_BUDGETS = {
    primitive: CellCountExtentBudget(16, 12, 24, 384, 0, 64, 48, 4)
    for primitive in LOCAL_CELL
}
CELL_BUDGETS.update(
    {
        "cellularity-decrease-v1": CellCountExtentBudget(16, 12, 24, 384, 0, 64, 48, 4),
        "peritumoral-neoplastic-scatter-increase-v1": CellCountExtentBudget(6, 4, 8, 160, 4, 64, 48, 4),
        # Two compact two-cell foci satisfy the primitive's 2.5-diameter span
        # contract at the native IGNITE/PUMA scale.  The former 48 px review
        # override rejected otherwise valid localized interfaces.
        "peritumoral-small-cluster-increase-v1": CellCountExtentBudget(8, 4, 10, 176, 4, 64, 32, 2),
    }
)
LARGE_TISSUE_BUDGET = JointAreaBudget(
    target_fraction=0.12,
    min_fraction=0.08,
    max_fraction=0.17,
    tissue_min_fraction=0.08,
    minimum_effective_fraction=0.08,
    fallback_policy="max_feasible_below_target",
    capacity_floor_policy="lower_to_proven_max_safe",
)
COMPARTMENT_TISSUE_BUDGET = JointAreaBudget(
    target_fraction=0.12,
    min_fraction=0.08,
    max_fraction=0.17,
    tissue_min_fraction=0.08,
    minimum_effective_fraction=0.05,
    fallback_policy="max_feasible_below_target",
    capacity_floor_policy="lower_to_proven_max_safe",
)
INTERFACE_TISSUE_BUDGET = JointAreaBudget(
    target_fraction=0.08,
    min_fraction=0.05,
    max_fraction=0.12,
    tissue_min_fraction=0.05,
    minimum_effective_fraction=0.04,
    fallback_policy="max_feasible_below_target",
    capacity_floor_policy="lower_to_proven_max_safe",
)
CORD_TISSUE_BUDGET = JointAreaBudget(
    # A connected invasive cord is intentionally narrow.  Reusing the broad
    # compartment budget made every topology-safe cord fail a 5% tissue floor
    # even when its elongated extension was clearly visible.  Keep the target
    # near the topology-safe narrow-footprint scale so candidate compilation
    # does not spend minutes chasing a broad fill that would no longer be a
    # cord.
    target_fraction=0.009,
    min_fraction=0.004,
    max_fraction=0.015,
    tissue_min_fraction=0.004,
    minimum_effective_fraction=0.004,
    fallback_policy="max_feasible_below_target",
    capacity_floor_policy="lower_to_proven_max_safe",
)


@dataclass(frozen=True)
class Evaluation:
    organ: str
    dataset: str
    pathology_domain_id: str
    annotation_profile_id: str
    population_profile_id: str
    mechanism_id: str
    primitive_id: str
    instruction: str


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _evaluations(organ: str, primitive: str | None) -> tuple[Evaluation, ...]:
    audit = json.loads(
        (ROOT / "phase3_joint_edit_refine/resources/lung_oral_skin_primitive_audit_v1.json").read_text(
            encoding="utf-8"
        )
    )
    organ_row = next(item for item in audit["organs"] if item["organ"] == organ)
    dataset, domain, profile, population = DATASET_CONFIG[organ]
    by_primitive: dict[str, str] = {}
    for item in organ_row["pairs"]:
        if item["status"] != "executable_mask_only":
            continue
        current = by_primitive.setdefault(item["primitive_id"], item["mechanism_id"])
        if current != item["mechanism_id"]:
            raise ValueError(f"ambiguous open mechanism for {organ}/{item['primitive_id']}")
    return tuple(
        Evaluation(organ, dataset, domain, profile, population, mechanism, primitive_id, INSTRUCTIONS[primitive_id])
        for primitive_id, mechanism in sorted(by_primitive.items())
        if primitive is None or primitive_id == primitive
    )


def _cross_meta_targets(path: Path, dataset: str) -> list[dict[str, str]]:
    pairs = json.loads(path.read_text(encoding="utf-8"))["pairs"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        if str(row.get("dataset", "")).upper() == dataset:
            grouped[str(row["target_tissue_mask"])].append(row)
    records = []
    for tissue_path, rows in sorted(grouped.items()):
        row = rows[0]
        assets = {
            "source_image": str(row["target_image"]),
            "source_tissue_mask": tissue_path,
            "source_nuclei_mask": str(row["target_nuclei_mask"]),
        }
        if not all(Path(value).is_file() for value in assets.values()):
            continue
        records.append(
            {
                "sample_id": str(row["sample_id"]),
                "cross_meta_case_id": str(row["case_id"]),
                **assets,
            }
        )
    return records


def _metrics(row: dict[str, str]) -> dict[str, Any]:
    tissue = load_id_mask(row["source_tissue_mask"])
    nuclei = load_nuclei_mask(row["source_nuclei_mask"])
    if tissue.shape != nuclei.shape:
        raise ValueError(f"unaligned masks: {row['sample_id']}")
    # Screening deliberately uses only a cheap conservative pixel-to-count
    # proxy. Exact instances, completeness and packing are rebuilt and gated
    # by the workflow for every attempted case; this prefilter certifies
    # nothing and exists only to rank thousands of cross-meta patches quickly.
    class_pixels = np.bincount(nuclei.ravel(), minlength=6)
    counts = {
        class_id: int(class_pixels[class_id] // 64)
        for class_id in range(1, 6)
    }
    component_counts = {
        class_id: int(ndimage.label(nuclei == class_id)[1])
        for class_id in range(1, 6)
    }
    local_counts_radius_64 = {}
    for class_id in range(1, 6):
        labels, count = ndimage.label(
            nuclei == class_id,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        centers = np.asarray(
            ndimage.center_of_mass(
                np.ones_like(labels, dtype=np.uint8),
                labels,
                range(1, count + 1),
            ),
            dtype=float,
        )
        local_counts_radius_64[class_id] = (
            max(
                (len(items) for items in cKDTree(centers).query_ball_point(centers, 64.0)),
                default=0,
            )
            if centers.size
            else 0
        )
    areas = {int(label): int(np.count_nonzero(tissue == label)) for label in np.unique(tissue)}
    contacts = {}
    for left in range(1, 8):
        boundary = ndimage.binary_dilation(tissue == left, structure=np.ones((3, 3), bool))
        for right in range(left + 1, 8):
            contacts[f"{left}:{right}"] = int(np.count_nonzero(boundary & (tissue == right)))
    occupied = ndimage.binary_dilation(nuclei > 0, iterations=2)
    free = {label: int(np.count_nonzero((tissue == label) & ~occupied)) for label in areas}
    tumor_distance = ndimage.distance_transform_edt(tissue != 1)
    peritumoral_free_stroma = (
        (tissue == 2)
        & ~occupied
        & (tumor_distance >= 4.0)
        & (tumor_distance <= 64.0)
    )
    local_labels, local_count = ndimage.label(
        peritumoral_free_stroma,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    local_sizes = np.bincount(local_labels.ravel(), minlength=local_count + 1)
    if local_count:
        largest_label = int(np.argmax(local_sizes[1:]) + 1)
        rows, cols = np.where(local_labels == largest_label)
        largest_area = int(local_sizes[largest_label])
        largest_span = float(
            np.hypot(int(rows.max()) - int(rows.min()), int(cols.max()) - int(cols.min()))
        )
    else:
        largest_area = 0
        largest_span = 0.0
    return {
        **row,
        "shape": list(tissue.shape),
        "areas": {str(k): v for k, v in areas.items()},
        "free": {str(k): v for k, v in free.items()},
        "peritumoral_free_stroma": {
            "area": int(np.count_nonzero(peritumoral_free_stroma)),
            "largest_component_area": largest_area,
            "largest_component_span_px": largest_span,
        },
        "counts": {str(k): v for k, v in counts.items()},
        "component_counts": {
            str(k): v for k, v in component_counts.items()
        },
        "local_counts_radius_64": {
            str(k): v for k, v in local_counts_radius_64.items()
        },
        "contacts": contacts,
    }


def _value(row: dict[str, Any], field: str, key: int) -> int:
    return int(row[field].get(str(key), 0))


def _contact(row: dict[str, Any], left: int, right: int) -> int:
    return int(row["contacts"].get(f"{min(left, right)}:{max(left, right)}", 0))


def _eligible_score(organ: str, primitive: str, row: dict[str, Any]) -> tuple[bool, float]:
    total_pixels = int(np.prod(row["shape"]))
    tumor = _value(row, "areas", 1)
    stroma = _value(row, "areas", 2)
    tumor_stroma = _contact(row, 1, 2)
    total_cells = sum(map(int, row["counts"].values()))
    if primitive in LOCAL_CELL:
        target_class = 1 if primitive.startswith("neoplastic-") else 2 if (
            primitive.startswith("cell-type-") or primitive.startswith("generic-inflammatory-")
        ) else None
        component_field = (
            "component_counts" if "component_counts" in row else "counts"
        )
        class_count = (
            _value(row, component_field, target_class)
            if target_class
            else sum(map(int, row[component_field].values()))
        )
        local_count = (
            _value(row, "local_counts_radius_64", target_class)
            if target_class and "local_counts_radius_64" in row
            else class_count
        )
        decrease = primitive.endswith("decrease-v1")
        free = sum(map(int, row["free"].values()))
        if target_class == 1:
            # Neoplastic additions are compiled inside a tumor component, so
            # rank by actual mask-derived free tumor capacity rather than by
            # the densest source population.  The previous density-heavy rank
            # systematically selected saturated ORCA tumor rasters.
            free = _value(row, "free", 1)
        eligible = (
            class_count >= (16 if decrease else 4)
            and (not decrease or local_count >= 12)
            and (decrease or free >= 8192)
        )
        # Prefer masks with many distinct observable nuclei components; free
        # area is a secondary packing signal.  Pixel-area-dominant ranking
        # selected a few merged semantic blobs and repeatedly failed native
        # instance authority despite apparently large nuclei-mask area.
        if decrease:
            score = min(local_count, 64) * 1_000_000 + min(class_count, 128) * 8192 + free
        else:
            score = (
                min(class_count, 48) * 8192
                - max(0, class_count - 64) * 16384
                + free
            )
        return eligible, score
    if primitive in PERITUMORAL:
        class_one = _value(row, "counts", 1)
        free_stroma = _value(row, "free", 2)
        local = row.get("peritumoral_free_stroma") or {}
        local_area = int(local.get("area", free_stroma))
        local_component = int(local.get("largest_component_area", local_area))
        local_span = float(local.get("largest_component_span_px", 0.0))
        return (
            class_one >= 6
            and tumor_stroma >= 128
            and local_area >= 1024
            and local_component >= 512
            and local_span >= 32.0,
            # Independent peritumoral foci need a long receiving interface and
            # connected free host capacity near that interface; global stroma
            # area is a poor proxy.  This ranking remains mask-only.
            100_000 * min(local_span, 128.0)
            + 100 * local_component
            + 10 * local_area
            + 50 * tumor_stroma
            + min(class_one, 64) * 50,
        )
    if primitive.startswith("generic-immune-infiltrate-"):
        source = 2 if primitive.endswith("increase-v1") else 4
        return (
            _contact(row, 2, 4) >= 96 and _value(row, "areas", source) >= int(0.08 * total_pixels),
            _value(row, "areas", source) + 30 * _contact(row, 2, 4),
        )
    if primitive == "necrosis-appearance-v1":
        return (_contact(row, 1, 3) >= 96 and tumor >= int(0.08 * total_pixels), tumor + 30 * _contact(row, 1, 3))
    if primitive == "necrosis-resolution-v1":
        necrosis = _value(row, "areas", 3)
        return (_contact(row, 1, 3) >= 96 and necrosis >= int(0.08 * total_pixels), necrosis + 30 * _contact(row, 1, 3))
    if primitive == "infiltrative-nest-cord-extension-v1":
        local = row.get("peritumoral_free_stroma") or {}
        local_area = int(local.get("area", _value(row, "free", 2)))
        local_component = int(local.get("largest_component_area", local_area))
        local_span = float(local.get("largest_component_span_px", 0.0))
        return (
            tumor_stroma >= 128
            and local_component >= 1024
            and local_span >= 48.0,
            100_000 * min(local_span, 192.0)
            + 100 * local_component
            + 10 * local_area
            + 50 * tumor_stroma,
        )
    if primitive in {"tumor-burden-increase-v1", "cohesive-boundary-expansion-v1"}:
        minimum = 0.05 if (
            organ == "lung"
            and primitive in {"tumor-burden-increase-v1", "cohesive-boundary-expansion-v1"}
        ) else 0.08 if (
            organ == "skin" and primitive == "cohesive-boundary-expansion-v1"
        ) else 0.14
        return (tumor_stroma >= 128 and stroma >= int(minimum * total_pixels), stroma + 40 * tumor_stroma)
    if primitive in {"invasive-tumor-footprint-decrease-v1", "stroma-increase-v1", "residual-tumor-fragmentation-v1"}:
        minimum = 0.08
        return (tumor_stroma >= 128 and tumor >= int(minimum * total_pixels), tumor + 40 * tumor_stroma)
    raise ValueError(primitive)


def _semantic_intent(evaluation: Evaluation) -> dict[str, Any]:
    intent = RuleBasedSemanticParser().parse(evaluation.instruction).to_metadata()
    hypotheses = {item["primitive_id"] for item in intent["primitive_hypotheses"]}
    if evaluation.primitive_id not in hypotheses:
        raise ValueError(f"parser did not bind {evaluation.primitive_id}: {hypotheses}")
    intent["selected_primitive_id"] = evaluation.primitive_id
    return intent


def _case(
    evaluation: Evaluation,
    row: dict[str, Any],
    index: int,
    native_authority: dict[str, Any],
) -> dict[str, Any]:
    primitive = evaluation.primitive_id
    short = hashlib.sha256(f"{evaluation.organ}:{primitive}:{row['sample_id']}".encode()).hexdigest()[:12]
    tissue_primitive = primitive not in CELL_BUDGETS
    budget = (
        INTERFACE_TISSUE_BUDGET
        if evaluation.organ == "lung" and primitive in {
            "tumor-burden-increase-v1",
            "cohesive-boundary-expansion-v1",
        }
        else CORD_TISSUE_BUDGET
        if primitive == "infiltrative-nest-cord-extension-v1"
        else COMPARTMENT_TISSUE_BUDGET
        if (
            primitive.startswith("generic-immune-")
            or primitive.startswith("necrosis-")
            or (evaluation.organ == "skin" and primitive == "cohesive-boundary-expansion-v1")
            or (
                evaluation.organ == "skin"
                and primitive
                in {
                    "invasive-tumor-footprint-decrease-v1",
                    "stroma-increase-v1",
                    "residual-tumor-fragmentation-v1",
                }
            )
        )
        else LARGE_TISSUE_BUDGET
    ) if tissue_primitive else None
    intent = _semantic_intent(evaluation)
    cell_budget = CELL_BUDGETS.get(primitive)
    if primitive == "generic-inflammatory-cell-abundance-decrease-v1" or (
        evaluation.organ in {"oral", "skin"}
        and primitive == "cell-type-abundance-decrease-v1"
    ):
        cell_budget = CellCountExtentBudget(10, 6, 14, 384, 0, 64, 48, 3)
    provenance = {
        "source_image_sha256": sha256_file(row["source_image"]),
        "source_tissue_mask_sha256": sha256_file(row["source_tissue_mask"]),
        "source_nuclei_mask_sha256": sha256_file(row["source_nuclei_mask"]),
        "source_nuclei_instances_sha256": native_authority["cells_json_sha256"],
        "preprocessing_revision": "cross-meta-mask-review-v1",
        "original_label_map_digest": sha256_file(row["source_tissue_mask"]),
        "provider": evaluation.dataset,
        "source_site": evaluation.organ,
        "specimen_type": "cross_validation_patch",
        "primary_or_metastatic": "metadata_filename_bound" if evaluation.organ == "skin" else "not_applicable",
        "joint_mechanism_id": evaluation.mechanism_id,
        "joint_primitive_id": primitive,
        "cross_meta_sample_id": row["sample_id"],
        "cross_meta_case_id": row["cross_meta_case_id"],
        "planner_input_policy": "mask_rasters_only_no_he",
        "require_mature_probnet_regeneration": True,
        "instance_authority_source": "frozen_cellvit_geometry_semantic_class_bound_v1",
    }
    case = JointCaseContext(
        case_id=f"{evaluation.organ}_mv_{primitive.removesuffix('-v1')}_{short}",
        instruction=evaluation.instruction,
        source_image_uri=row["source_image"],
        source_tissue_mask_uri=row["source_tissue_mask"],
        source_nuclei_mask_uri=row["source_nuclei_mask"],
        source_nuclei_instances_uri=native_authority["cells_json"],
        pathology_domain_id=evaluation.pathology_domain_id,
        annotation_profile_id=evaluation.annotation_profile_id,
        cell_observation_profile_id="cellvit-five-class-v1",
        cell_population_profile_id=evaluation.population_profile_id,
        primitive_id=primitive,
        joint_area_budget=budget,
        cell_count_extent_budget=cell_budget,
        seed=20260820 + index,
        provenance=provenance,
        pixel_size_um=0.25,
        semantic_intent=intent,
    )
    payload = case.to_metadata()
    payload["prebound_semantic_intent"] = intent
    return payload


def _run_case(payload: dict[str, Any], evaluation: Evaluation, args: argparse.Namespace) -> dict[str, Any]:
    case_root = (
        args.output_dir
        / "runs"
        / SCHEMA_VERSION
        / evaluation.organ
        / evaluation.primitive_id
        / payload["case_id"]
    )
    manifest = case_root / "manifest.json"
    _write_json(manifest, [payload])
    library = args.nuclei_instance_library
    if (library / evaluation.dataset / ".complete").is_file():
        library = library / evaluation.dataset
    command = [
        sys.executable, "-m", "phase3_joint_edit_refine.cli",
        "--manifest", str(manifest), "--output-root", str(case_root),
        "--agent-mode", "offline", "--semantic-parser", "prebound",
        "--cell-executor", "mature", "--probnet-checkpoint", str(args.probnet_checkpoint),
        "--nuclei-instance-library", str(library),
        "--probnet-dataset", evaluation.dataset, "--device", args.device, "--meta-eval",
    ]
    env = os.environ.copy()
    for key in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(args.threads)
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=args.timeout_seconds)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        code, stdout, stderr = 124, exc.stdout or "", exc.stderr or ""
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "stdout.log").write_text(stdout, encoding="utf-8")
    (case_root / "stderr.log").write_text(stderr, encoding="utf-8")
    summary_path = case_root / "joint_run_summary.json"
    summary = None
    if summary_path.is_file():
        values = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = values[0] if isinstance(values, list) and len(values) == 1 else None
    selected = summary.get("selected_candidate_id") if summary else None
    artifacts = dict(summary.get("artifact_paths") or {}) if summary else {}
    if summary and summary.get("status") == "review_required":
        selected = _compiled_candidate(artifacts, summary.get("abstain_reasons") or [])
    return {
        "return_code": code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "status": summary.get("status") if summary else "missing_summary",
        "selected_candidate_id": selected,
        "abstain_reasons": list(summary.get("abstain_reasons") or []) if summary else [],
        "artifact_paths": artifacts,
    }


def _compiled_candidate(artifacts: dict[str, Any], reasons: list[str]) -> str | None:
    if reasons != ["independent_mask_condition_critic_approval_required"]:
        return None
    candidates_path = Path(str(artifacts.get("candidates.json", "")))
    gates_path = Path(str(artifacts.get("joint_gate_reports.json", "")))
    critic_path = Path(str(artifacts.get("joint_critic.json") or candidates_path.parent / "joint_critic.json"))
    if not all(path.is_file() for path in (candidates_path, gates_path, critic_path)):
        return None
    rankings = json.loads(critic_path.read_text(encoding="utf-8")).get("rankings", [])
    if not rankings or rankings[0].get("veto_reasons"):
        return None
    candidate_id = rankings[0].get("candidate_id")
    report = next((item for item in json.loads(gates_path.read_text(encoding="utf-8")) if item.get("candidate_id") == candidate_id), None)
    return str(candidate_id) if report and report.get("passed") is True else None


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _composite(tissue: np.ndarray, nuclei: np.ndarray) -> np.ndarray:
    rgb = np.clip(0.55 * id_mask_to_rgb(tissue).astype(float) + 0.45 * 255, 0, 255).astype(np.uint8)
    for class_id, color in NUCLEI_RGB.items():
        rgb[nuclei == class_id] = np.asarray(color, dtype=np.uint8)
    boundary = np.zeros_like(tissue, dtype=bool)
    boundary[1:, :] |= tissue[1:, :] != tissue[:-1, :]
    boundary[:, 1:] |= tissue[:, 1:] != tissue[:, :-1]
    rgb[boundary] = 255
    return rgb


def _render(evaluation: Evaluation, records: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    tile, header = 320, 62
    canvas = Image.new("RGB", (tile * 5, len(records) * (tile + header)), "white")
    draw, font, small = ImageDraw.Draw(canvas), _font(15), _font(12)
    review = []
    for index, record in enumerate(records):
        payload, run = record["payload"], record["run"]
        candidates = json.loads(Path(run["artifact_paths"]["candidates.json"]).read_text(encoding="utf-8"))
        candidate = next(item for item in candidates if item["candidate_id"] == run["selected_candidate_id"])
        st, sn = load_id_mask(payload["source_tissue_mask_uri"]), load_nuclei_mask(payload["source_nuclei_mask_uri"])
        tt, tn = load_id_mask(candidate["target_tissue_mask"]), load_nuclei_mask(candidate["target_nuclei_mask"])
        source_he = np.asarray(Image.open(payload["source_image_uri"]).convert("RGB"))
        source, target = _composite(st, sn), _composite(tt, tn)
        changed_t, changed_n = st != tt, sn != tn
        delta = _composite(st, np.zeros_like(sn))
        delta[(sn > 0) & ~changed_n] = (120, 120, 120)
        delta[changed_n & (sn > 0)] = (255, 0, 210)
        delta[changed_n & (tn > 0)] = (0, 255, 80)
        delta[changed_t] = (0, 220, 255)
        changed = changed_t | changed_n
        rows, cols = np.nonzero(changed)
        zoom = delta
        if len(rows):
            pad = 28
            crop = delta[max(0, rows.min()-pad):min(delta.shape[0], rows.max()+pad+1), max(0, cols.min()-pad):min(delta.shape[1], cols.max()+pad+1)]
            zoom = np.asarray(Image.fromarray(crop).resize((delta.shape[1], delta.shape[0]), Image.Resampling.NEAREST))
        y = index * (tile + header)
        counts = {"tissue_changed_pixels": int(changed_t.sum()), "nuclei_changed_pixels": int(changed_n.sum()), "joint_changed_pixels": int(changed.sum()), "joint_changed_fraction": round(float(changed.mean()), 6)}
        draw.text((8, y + 5), f"{index+1}. {payload['provenance']['cross_meta_sample_id']} | {run['selected_candidate_id']}", fill="black", font=font)
        draw.text((8, y + 33), f"joint={counts['joint_changed_pixels']} ({counts['joint_changed_fraction']:.1%}) tissue={counts['tissue_changed_pixels']} nuclei={counts['nuclei_changed_pixels']}", fill=(40, 40, 40), font=small)
        for column, (label, panel) in enumerate(zip(("SOURCE H&E (review only)", "SOURCE MASK", "TARGET MASK", "DELTA", "DELTA ZOOM"), (source_he, source, target, delta, zoom), strict=True)):
            resized = Image.fromarray(panel).resize((tile, tile), Image.Resampling.BILINEAR if column == 0 else Image.Resampling.NEAREST)
            canvas.paste(resized, (column * tile, y + header))
            ImageDraw.Draw(canvas).text((column * tile + 6, y + header + 5), label, fill="white", font=small, stroke_width=2, stroke_fill="black")
        review.append({"case_id": payload["case_id"], "sample_id": payload["provenance"]["cross_meta_sample_id"], "selected_candidate_id": run["selected_candidate_id"], "source_image": payload["source_image_uri"], "source_tissue_mask": payload["source_tissue_mask_uri"], "source_nuclei_mask": payload["source_nuclei_mask_uri"], "target_tissue_mask": candidate["target_tissue_mask"], "target_nuclei_mask": candidate["target_nuclei_mask"], "joint_change_mask": candidate["joint_change_mask"], "change_counts": counts})
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return review


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = _evaluations(args.organ, args.primitive)
    targets = _cross_meta_targets(args.cross_meta.resolve(), DATASET_CONFIG[args.organ][0])
    metrics_path = args.output_dir / f"{args.organ}_cross_meta_metrics.json"
    if metrics_path.is_file() and not args.refresh_metrics:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = [_metrics(row) for row in targets]
        _write_json(metrics_path, metrics)
    cache_path = args.output_dir / "native_authority_cache.json"
    native_cache = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.is_file()
        else {}
    )
    attempts_path = args.output_dir / "attempts.json"
    review_path = args.output_dir / "review_manifest.json"
    all_attempts = (
        json.loads(attempts_path.read_text(encoding="utf-8"))
        if attempts_path.is_file()
        else []
    )
    reviews = (
        dict(json.loads(review_path.read_text(encoding="utf-8")).get("reviews") or {})
        if review_path.is_file()
        else {}
    )
    for evaluation in evaluations:
        prior_review = reviews.get(evaluation.primitive_id) or {}
        if (
            len(prior_review.get("cases") or []) == args.per_primitive
            and Path(str(prior_review.get("board", ""))).is_file()
        ):
            continue
        ranked = []
        for row in metrics:
            eligible, score = _eligible_score(evaluation.organ, evaluation.primitive_id, row)
            if eligible:
                if evaluation.primitive_id in {
                    "cohesive-boundary-expansion-v1",
                    "infiltrative-nest-cord-extension-v1",
                    "peritumoral-neoplastic-scatter-increase-v1",
                    "peritumoral-small-cluster-increase-v1",
                    "tumor-burden-increase-v1",
                }:
                    cached_native = native_cache.get(str(row["sample_id"])) or {}
                    cached_metrics = (
                        (cached_native.get("validation") or {}).get("metrics") or {}
                    )
                    reference_counts = (
                        cached_metrics.get("native_complete_reference_count_by_class")
                        or {}
                    )
                    if cached_native and int(reference_counts.get("1", 0)) <= 0:
                        continue
                    score += 200_000 * min(
                        int(reference_counts.get("1", 0)), 64
                    )
                ranked.append((score, row))
        ranked.sort(key=lambda item: (-item[0], item[1]["sample_id"]))
        selected = [
            item
            for item in all_attempts
            if item.get("primitive_id") == evaluation.primitive_id
            and (item.get("run") or {}).get("selected_candidate_id")
        ][: args.per_primitive]
        attempted_samples = {
            str(item.get("sample_id"))
            for item in all_attempts
            if item.get("primitive_id") == evaluation.primitive_id
        }
        if args.retry_failed:
            # Re-run previously rejected samples after an executor repair,
            # while retaining already selected cases and avoiding duplicates.
            attempted_samples = {
                str(item.get("sample_id")) for item in selected
            }
        new_attempts = 0
        for rank_index, (_score, row) in enumerate(ranked, start=1):
            if row["sample_id"] in attempted_samples:
                continue
            if new_attempts >= args.max_attempts:
                break
            new_attempts += 1
            attempt = len(attempted_samples) + new_attempts
            try:
                native = _native_authority(
                    row,
                    output_root=args.output_dir,
                    cellvit_model=args.cellvit_model,
                    cellvit_root=args.cellvit_root,
                    cellvit_python=args.cellvit_python,
                    gpu=args.cellvit_gpu,
                    timeout_seconds=args.cellvit_timeout_seconds,
                    cache=native_cache,
                )
                _write_json(cache_path, native_cache)
                payload = _case(evaluation, row, attempt, native)
                outcome = _run_case(payload, evaluation, args)
            except (OSError, RuntimeError, ValueError) as exc:
                # One unusable patch must not abort an organ-wide review.  The
                # failed mask-only-ranked sample remains explicit in attempts
                # and the runner continues to the next ranked sample.
                short = hashlib.sha256(
                    f"{evaluation.organ}:{evaluation.primitive_id}:{row['sample_id']}".encode()
                ).hexdigest()[:12]
                payload = {
                    "case_id": f"{evaluation.organ}_mv_{evaluation.primitive_id.removesuffix('-v1')}_{short}",
                    "source_image_uri": row["source_image"],
                    "source_tissue_mask_uri": row["source_tissue_mask"],
                    "source_nuclei_mask_uri": row["source_nuclei_mask"],
                    "provenance": {"cross_meta_sample_id": row["sample_id"]},
                }
                outcome = {
                    "return_code": 1,
                    "duration_seconds": 0.0,
                    "status": "preprocessing_failed",
                    "selected_candidate_id": None,
                    "abstain_reasons": [f"{type(exc).__name__}: {exc}"],
                    "artifact_paths": {},
                }
            record = {"organ": evaluation.organ, "primitive_id": evaluation.primitive_id, "mechanism_id": evaluation.mechanism_id, "sample_id": row["sample_id"], "payload": payload, "run": outcome}
            all_attempts.append(record)
            _write_json(attempts_path, all_attempts)
            print(json.dumps({"organ": evaluation.organ, "primitive": evaluation.primitive_id, "attempt": attempt, "selected": len(selected), "status": outcome["status"], "case": payload["case_id"]}), flush=True)
            if outcome["selected_candidate_id"]:
                selected.append(record)
            if len(selected) == args.per_primitive:
                break
        if len(selected) != args.per_primitive:
            reviews[evaluation.primitive_id] = {
                "mechanism_id": evaluation.mechanism_id,
                "board": None,
                "cases": [],
                "failure": (
                    f"obtained {len(selected)} of {args.per_primitive} after "
                    f"{len(attempted_samples) + new_attempts} ranked attempts"
                ),
            }
            _write_json(review_path, {"schema_version": SCHEMA_VERSION, "organ": args.organ, "reviews": reviews})
            continue
        board = args.output_dir / "boards" / evaluation.organ / f"{evaluation.primitive_id}.png"
        cases = _render(evaluation, selected, board)
        reviews[evaluation.primitive_id] = {"mechanism_id": evaluation.mechanism_id, "board": str(board), "cases": cases}
        _write_json(review_path, {"schema_version": SCHEMA_VERSION, "organ": args.organ, "reviews": reviews})
    complete = sum(
        len((reviews.get(item.primitive_id) or {}).get("cases") or [])
        == args.per_primitive
        for item in evaluations
    )
    result = {"schema_version": SCHEMA_VERSION, "organ": args.organ, "executable_primitive_count": len(evaluations), "complete_primitive_count": complete, "all_primitives_complete": complete == len(evaluations), "cases_per_primitive": args.per_primitive, "selected_case_count": sum(len(item.get("cases") or []) for item in reviews.values()), "reviews": reviews}
    _write_json(args.output_dir / f"{args.organ}_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--organ", required=True, choices=tuple(DATASET_CONFIG))
    parser.add_argument("--cross-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probnet-checkpoint", type=Path, required=True)
    parser.add_argument("--nuclei-instance-library", type=Path, required=True)
    parser.add_argument("--cellvit-model", type=Path, required=True)
    parser.add_argument("--cellvit-root", type=Path, required=True)
    parser.add_argument("--cellvit-python", type=Path, required=True)
    parser.add_argument("--cellvit-gpu", type=int, default=0)
    parser.add_argument("--cellvit-timeout-seconds", type=int, default=300)
    parser.add_argument("--primitive")
    parser.add_argument("--per-primitive", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--refresh-metrics", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: value for key, value in result.items() if key != "reviews"}, indent=2))
    return 0 if result["all_primitives_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
