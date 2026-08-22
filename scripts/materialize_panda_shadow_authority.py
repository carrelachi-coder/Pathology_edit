#!/usr/bin/env python3
"""Materialize frozen PANDA source, native-instance, ROI, and runtime authority.

The only H&E consumer in this stage is the explicitly named frozen CellViT
checkpoint.  Candidate selection, ROI construction, semantic binding, and all
validation are deterministic; no LLM or API observes source H&E pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from phase3_joint_edit_refine.nuclei import (
    _local_contour,
    load_native_instances,
    load_nuclei_mask,
)
from phase3_joint_edit_refine.portfolio_authority import (
    array_sha256,
    canonical_metadata_sha256,
)
from phase3_joint_edit_refine.semantic_parser import RuleBasedSemanticParser
from phase3_mask_edit_refine.evidence import sha256_file

SCHEMA_VERSION = "panda-primitive-shadow-authority-v1"
QUALIFICATION_MANIFEST_SCHEMA = "g2-v2-image-instruction-mechanism-manifest-v2"
CELLVIT_PRODUCER_ID = (
    "frozen-cellvit-geometry-plus-frozen-semantic-class-authority-v1"
)
NATIVE_AUTHORITY_ALGORITHM = (
    "cellvit-native-contour-semantic-majority-class-binding-v1"
)
ROI_PRODUCER_ID = "user-delegated-fixed-central-roi-v1"
DEFAULT_USER_AUTHORITY_TEXT = (
    "按 primitive 重新筛选各自 5 个可行病例、补齐 native instances/ROI、"
    "绑定 Frozen ProbNet ranker，并完成 authority → candidate → full joint "
    "execution → gates 的冻结 shadow replay。"
)
JOINT_AREA_BUDGET_BY_PRIMITIVE = {
    # These are the repository's already exercised cross-organ/PANDA geometry
    # contracts.  A narrow cord must never inherit a bulk-growth floor.
    "local-invasive-clearance-v1": (0.04, 0.02, 0.12, 0.02),
    "stroma-increase-v1": (0.03, 0.02, 0.12, 0.02),
    "invasive-tumor-footprint-decrease-v1": (0.03, 0.02, 0.12, 0.02),
    # These are local topology edits, not whole-patch tumor-burden edits.
    # A 1.5% floor remains plainly visible at 512x512 while avoiding the
    # forced bulk cuts/growth that damaged otherwise intact gland units.
    "residual-tumor-fragmentation-v1": (0.035, 0.015, 0.05, 0.015),
    "cohesive-boundary-expansion-v1": (0.02, 0.015, 0.06, 0.015),
    # One PANDA Pattern-5 cord must be large enough to contain complete
    # source-calibrated nuclei while remaining a single narrow projection.
    # On a 512x512 patch this is about 786--3,146 pixels, not bulk growth.
    "infiltrative-nest-cord-extension-v1": (0.0053, 0.003, 0.012, 0.003),
}
FROZEN_EXECUTION_INSTRUCTIONS = {
    ("prostate-local-population-modulation", "cell-type-abundance-increase-v1"): (
        "Increase connective tissue cells in the selected region."
    ),
    ("prostate-local-population-modulation", "cell-type-abundance-decrease-v1"): (
        "Decrease connective tissue cells in the selected region."
    ),
    ("prostate-local-population-modulation", "cellularity-increase-v1"): (
        "Increase local cellularity."
    ),
    ("prostate-local-population-modulation", "cellularity-decrease-v1"): (
        "Decrease local cellularity."
    ),
    (
        "prostate-local-population-modulation",
        "neoplastic-cell-abundance-increase-v1",
    ): "Increase neoplastic cells.",
    (
        "prostate-local-population-modulation",
        "neoplastic-cell-abundance-decrease-v1",
    ): "Decrease neoplastic cells.",
    ("prostate-local-tumor-clearance", "local-invasive-clearance-v1"): (
        "Clear tumor in this local ROI."
    ),
    ("prostate-operational-tumor-retreat", "stroma-increase-v1"): (
        "Simulate a post-treatment response by increasing operational stroma."
    ),
    (
        "prostate-operational-tumor-retreat",
        "invasive-tumor-footprint-decrease-v1",
    ): "Simulate a post-treatment response by decreasing tumor area.",
    (
        "prostate-operational-tumor-retreat",
        "residual-tumor-fragmentation-v1",
    ): (
        "Simulate post-treatment residual disease by fragmenting residual tumor "
        "into controlled foci."
    ),
    ("prostate-pattern-4-growth", "cohesive-boundary-expansion-v1"): (
        "Expand the tumor boundary locally."
    ),
    ("prostate-pattern-5-growth", "cohesive-boundary-expansion-v1"): (
        "Expand the tumor boundary locally."
    ),
    (
        "prostate-pattern-5-infiltrative-front",
        "infiltrative-nest-cord-extension-v1",
    ): "Add a narrow connected tumor cord.",
    (
        "prostate-pattern-5-peripheral-scatter",
        "peritumoral-neoplastic-scatter-increase-v1",
    ): "Add scattered tumor cells near the tumor boundary.",
}
RUNTIME_CODE_RELATIVE_PATHS = (
    "scripts/prepare_panda_primitive_shadow_selection.py",
    "scripts/compose_panda_shadow_rescreen_pool.py",
    "scripts/materialize_panda_shadow_authority.py",
    "scripts/qualify_g2_v2_execution.py",
    "scripts/qualify_g2_v2_execution_bounded.py",
    "scripts/run_panda_primitive_shadow_replay.py",
    "phase3_joint_edit_refine/g2_v2_shadow.py",
    "phase3_joint_edit_refine/g2_execution_qualification.py",
    "phase3_joint_edit_refine/auxiliary.py",
    "phase3_joint_edit_refine/candidate_feasibility.py",
    "phase3_joint_edit_refine/cell_layouts.py",
    "phase3_joint_edit_refine/cell_programs.py",
    "phase3_joint_edit_refine/cli.py",
    "phase3_joint_edit_refine/executable_contract.py",
    "phase3_joint_edit_refine/feasibility.py",
    "phase3_joint_edit_refine/gates.py",
    "phase3_joint_edit_refine/ledger.py",
    "phase3_joint_edit_refine/mature_probnet_adapter.py",
    "phase3_joint_edit_refine/models.py",
    "phase3_joint_edit_refine/nuclei.py",
    "phase3_joint_edit_refine/packing.py",
    "phase3_joint_edit_refine/planner.py",
    "phase3_joint_edit_refine/probnet_adapter.py",
    "phase3_joint_edit_refine/scene.py",
    "phase3_joint_edit_refine/seam.py",
    "phase3_joint_edit_refine/semantic_parser.py",
    "phase3_joint_edit_refine/skills/repository.py",
    "phase3_joint_edit_refine/skills/schema.py",
    "phase3_joint_edit_refine/spatial_contracts.py",
    "phase3_joint_edit_refine/tissue_planner.py",
    "phase3_joint_edit_refine/tissue_tools.py",
    "phase3_joint_edit_refine/workflow.py",
    "phase3_mask_edit_refine/candidates.py",
    "phase3_mask_edit_refine/execution.py",
    "phase3_mask_edit_refine/gates.py",
)


def _canonical_sha256(value: Any) -> str:
    return canonical_metadata_sha256(value)


def _parse_evaluation_count_overrides(value: str | None) -> dict[int, int]:
    overrides: dict[int, int] = {}
    for token in (value or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            index_text, count_text = token.split(":", 1)
            index, count = int(index_text), int(count_text)
        except ValueError as exc:
            raise ValueError(
                "evaluation count overrides must use INDEX:COUNT entries"
            ) from exc
        if index < 0 or count < 5:
            raise ValueError(
                "evaluation count overrides require nonnegative indices and "
                "at least five candidates"
            )
        if index in overrides:
            raise ValueError(f"duplicate evaluation count override: {index}")
        overrides[index] = count
    return overrides


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _directory_sha256(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"runtime asset directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _runtime_code_inventory() -> tuple[list[dict[str, str]], str]:
    inventory = []
    for relative in RUNTIME_CODE_RELATIVE_PATHS:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"runtime code authority is missing: {path}")
        inventory.append({"path": relative, "sha256": sha256_file(path)})
    return inventory, _canonical_sha256(inventory)


def _semantic_intent(instruction: str, primitive_id: str) -> tuple[dict, str]:
    intent = RuleBasedSemanticParser().parse(instruction)
    hypotheses = {item.primitive_id for item in intent.primitive_hypotheses}
    if primitive_id not in hypotheses:
        raise ValueError(
            f"instruction does not bind requested primitive {primitive_id}: "
            f"{sorted(hypotheses)}"
        )
    metadata = intent.to_metadata()
    # The frozen shadow evaluates one reviewed primitive per row.  A parsed
    # instruction may legitimately expose more than one hypothesis (for
    # example, increasing stroma versus reducing tumour footprint), so the
    # authority must bind the reviewed choice explicitly instead of relying
    # on the parser's first hypothesis.
    metadata["selected_primitive_id"] = primitive_id
    return metadata, _canonical_sha256(metadata)


def _joint_area_budget(primitive_id: str) -> dict[str, Any]:
    try:
        target, minimum, maximum, tissue_minimum = (
            JOINT_AREA_BUDGET_BY_PRIMITIVE[primitive_id]
        )
    except KeyError as exc:
        raise ValueError(
            f"no frozen joint-area budget for tissue primitive {primitive_id}"
        ) from exc
    return {
        "target_fraction": target,
        "min_fraction": minimum,
        "max_fraction": maximum,
        "tissue_min_fraction": tissue_minimum,
        "basis": "whole_patch",
        "relative_tolerance": 0.02,
        "fallback_policy": "max_feasible_below_target",
        "capacity_floor_policy": "strict",
        "minimum_effective_fraction": 0.0,
    }


def _fixed_roi(
    *, shape: tuple[int, int], path: Path, source_tissue_sha256: str,
    user_authority_sha256: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]]]:
    height, width = shape
    y0, y1 = height // 8, height - height // 8
    x0, x1 = width // 8, width - width // 8
    roi = np.zeros(shape, dtype=np.uint8)
    roi[y0:y1, x0:x1] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(roi, mode="L").save(path)
    file_digest = sha256_file(path)
    decoded_digest = array_sha256(roi)
    provenance = {
        "producer_id": ROI_PRODUCER_ID,
        "producer_version": "1.0.0",
        "authority_type": "digest_bound_user_local_roi",
        "delegation_scope": "fixed_geometry_only_no_h_e_or_mask_inference",
        "user_authority_text_sha256": user_authority_sha256,
        "source_tissue_mask_sha256": source_tissue_sha256,
        "output_sha256": file_digest,
        "decoded_array_sha256": decoded_digest,
        "geometry": {
            "kind": "central_rectangle",
            "normalized_bounds": [0.125, 0.125, 0.875, 0.875],
            "pixel_bounds_xyxy": [x0, y0, x1, y1],
        },
        "he_or_llm_used": False,
    }
    return (
        {"local_clearance_roi": str(path.resolve())},
        {"local_clearance_roi": file_digest},
        {"local_clearance_roi": provenance},
    )


def _bind_native_geometry_to_frozen_semantic_classes(
    *, raw_cells_json: Path, semantic_path: Path, output_path: Path,
) -> dict[str, Any]:
    """Keep CellViT contours but bind class IDs to the frozen semantic mask.

    PANDA's frozen semantic raster is the class authority used by the editor.
    CellViT supplies native instance geometry.  A contour is retained only if
    at least 80% of its pixels agree on one non-zero frozen semantic class; the
    original CellViT type is preserved as provenance, never trusted silently.
    """

    payload = json.loads(raw_cells_json.read_text(encoding="utf-8"))
    raw_cells = payload.get("cells") if isinstance(payload, dict) else None
    if not isinstance(raw_cells, list):
        raise TypeError("CellViT native JSON lacks a top-level cells list")
    semantic = load_nuclei_mask(semantic_path)
    metadata = (
        payload.get("wsi_metadata")
        if isinstance(payload.get("wsi_metadata"), dict)
        else {}
    )
    accepted = []
    rejected = []
    transitions = Counter()
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, dict):
            rejected.append({"index": index, "reason": "cell_record_not_object"})
            continue
        contour = raw.get("contour")
        if not isinstance(contour, list) or len(contour) < 3:
            rejected.append({"index": index, "reason": "contour_missing"})
            continue
        points = _local_contour(
            contour, info=raw, metadata=metadata, shape=semantic.shape
        )
        if len(points) < 3:
            rejected.append({"index": index, "reason": "contour_out_of_patch"})
            continue
        canvas = Image.new("1", (semantic.shape[1], semantic.shape[0]), 0)
        ImageDraw.Draw(canvas).polygon(points, fill=1)
        component = np.asarray(canvas, dtype=bool)
        area = int(np.count_nonzero(component))
        if not area:
            rejected.append({"index": index, "reason": "empty_contour"})
            continue
        counts = np.bincount(semantic[component], minlength=6)
        class_id = int(np.argmax(counts[1:6]) + 1)
        agreement = float(counts[class_id] / area)
        if agreement < 0.80:
            rejected.append(
                {
                    "index": index,
                    "reason": "no_semantic_class_reaches_0.80",
                    "best_class_id": class_id,
                    "best_agreement": round(agreement, 8),
                }
            )
            continue
        original_type = int(raw.get("type", 0))
        bound = dict(raw)
        bound["type"] = class_id
        bound["authority_binding"] = {
            "algorithm": NATIVE_AUTHORITY_ALGORITHM,
            "cellvit_original_type": original_type,
            "frozen_semantic_class_id": class_id,
            "pixel_agreement": round(agreement, 8),
        }
        transitions[f"{original_type}->{class_id}"] += 1
        accepted.append(bound)
    derived = {
        **payload,
        "cells": accepted,
        "native_instance_authority": {
            "algorithm": NATIVE_AUTHORITY_ALGORITHM,
            "geometry_authority": "frozen_cellvit_contour",
            "class_authority": "frozen_source_semantic_nuclei_mask",
            "minimum_per_contour_class_agreement": 0.80,
            "raw_cells_json": str(raw_cells_json.resolve()),
            "raw_cells_json_sha256": sha256_file(raw_cells_json),
            "source_semantic_nuclei_mask": str(semantic_path.resolve()),
            "source_semantic_nuclei_mask_sha256": sha256_file(semantic_path),
            "raw_instance_count": len(raw_cells),
            "accepted_instance_count": len(accepted),
            "rejected_instance_count": len(rejected),
            "original_to_bound_class_counts": dict(sorted(transitions.items())),
            "rejected_instances": rejected,
            "he_interpreter": "frozen_cellvit_geometry_only",
            "llm_api_used": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(derived, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return derived["native_instance_authority"]


def _validate_native_instances(
    *, cells_json: Path, semantic_path: Path,
) -> dict[str, Any]:
    semantic = load_nuclei_mask(semantic_path)
    instances = load_native_instances(
        cells_json,
        shape=semantic.shape,
        semantic_mask=semantic,
    )
    raster = np.zeros(semantic.shape, dtype=np.uint8)
    counts = Counter()
    for _instance_id, class_id, component in instances:
        raster[component] = int(class_id)
        counts[int(class_id)] += 1
    native_fg = raster > 0
    semantic_fg = semantic > 0
    intersection = int(np.count_nonzero(native_fg & semantic_fg))
    native_pixels = int(np.count_nonzero(native_fg))
    semantic_pixels = int(np.count_nonzero(semantic_fg))
    class_agreement = (
        float(np.mean(raster[native_fg] == semantic[native_fg]))
        if native_pixels
        else 0.0
    )
    precision = intersection / native_pixels if native_pixels else 0.0
    recall = intersection / semantic_pixels if semantic_pixels else 0.0
    dice = (
        2.0 * intersection / (native_pixels + semantic_pixels)
        if native_pixels + semantic_pixels
        else 0.0
    )
    thresholds = {
        "minimum_instance_count": 16,
        "minimum_foreground_precision": 0.85,
        "minimum_foreground_recall": 0.75,
        "minimum_foreground_dice": 0.78,
        "minimum_native_pixel_class_agreement": 0.90,
    }
    metrics = {
        "native_instance_count": len(instances),
        "native_class_counts": {
            str(class_id): int(counts[class_id]) for class_id in range(1, 6)
        },
        "native_foreground_pixels": native_pixels,
        "semantic_foreground_pixels": semantic_pixels,
        "foreground_precision": round(precision, 8),
        "foreground_recall": round(recall, 8),
        "foreground_dice": round(dice, 8),
        "native_pixel_class_agreement": round(class_agreement, 8),
    }
    failures = []
    if len(instances) < thresholds["minimum_instance_count"]:
        failures.append("native_instance_count_below_threshold")
    for name, metric in (
        ("foreground_precision", precision),
        ("foreground_recall", recall),
        ("foreground_dice", dice),
        ("native_pixel_class_agreement", class_agreement),
    ):
        if metric < thresholds[f"minimum_{name}"]:
            failures.append(f"{name}_below_threshold")
    result = {
        "validator_id": "cellvit-native-to-frozen-semantic-agreement-v1",
        "status": "verified" if not failures else "rejected",
        "thresholds": thresholds,
        "metrics": metrics,
        "failure_codes": failures,
        "he_consumed_by": "frozen_cellvit_only",
        "llm_api_used": False,
    }
    result["validation_sha256"] = _canonical_sha256(result)
    return result


def _run_cellvit(
    *, candidate: dict[str, Any], output_root: Path, model: Path,
    cellvit_root: Path, cellvit_python: Path, gpu: int, timeout_seconds: int,
    resume: bool,
) -> dict[str, Any]:
    filename = str(candidate["filename"])
    key = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:12]
    root = output_root / "cellvit" / f"{Path(filename).stem}_{key}"
    root.mkdir(parents=True, exist_ok=True)
    output_mask = root / "cellvit_native_mask.png"
    summary_path = output_mask.with_suffix(".cellvit_single_patch.json")
    stdout_path = root / "cellvit_wrapper_stdout.log"
    stderr_path = root / "cellvit_wrapper_stderr.log"
    if not (resume and summary_path.is_file()):
        command = [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_cellvit_single_patch.py"),
            "--image",
            str(candidate["source_image"]),
            "--output-mask",
            str(output_mask),
            "--model",
            str(model),
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
        try:
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            stderr_path.write_text(completed.stderr or "", encoding="utf-8")
            if completed.returncode:
                return {
                    "status": "rejected",
                    "failure_codes": [
                        f"cellvit_subprocess_exit_{completed.returncode}"
                    ],
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "llm_api_used": False,
                }
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            return {
                "status": "rejected",
                "failure_codes": ["cellvit_subprocess_timeout"],
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "llm_api_used": False,
            }
    if not summary_path.is_file():
        return {
            "status": "rejected",
            "failure_codes": ["cellvit_summary_missing"],
            "llm_api_used": False,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cells_json = Path(str(summary["cells_json"]))
    if not cells_json.is_file():
        return {
            "status": "rejected",
            "failure_codes": ["cellvit_native_json_missing"],
            "llm_api_used": False,
        }
    semantic_path = Path(str(candidate["source_nuclei_mask"]))
    bound_cells_json = root / "semantic_bound_native_cells.json"
    try:
        class_binding = _bind_native_geometry_to_frozen_semantic_classes(
            raw_cells_json=cells_json,
            semantic_path=semantic_path,
            output_path=bound_cells_json,
        )
        validation = _validate_native_instances(
            cells_json=bound_cells_json,
            semantic_path=semantic_path,
        )
    except Exception as exc:  # noqa: BLE001 - one fail-closed authority record
        return {
            "status": "rejected",
            "raw_cells_json": str(cells_json.resolve()),
            "cells_json": str(bound_cells_json.resolve()),
            "failure_codes": [f"{type(exc).__name__}: {exc}"],
            "authority_algorithm": NATIVE_AUTHORITY_ALGORITHM,
            "llm_api_used": False,
        }
    return {
        "status": validation["status"],
        "authority_algorithm": NATIVE_AUTHORITY_ALGORITHM,
        "raw_cells_json": str(cells_json.resolve()),
        "raw_cells_json_sha256": sha256_file(cells_json),
        "cells_json": str(bound_cells_json.resolve()),
        "cells_json_sha256": sha256_file(bound_cells_json),
        "cellvit_mask": str(output_mask.resolve()),
        "cellvit_mask_sha256": sha256_file(output_mask),
        "class_binding": class_binding,
        "validation": validation,
        "failure_codes": list(validation["failure_codes"]),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "llm_api_used": False,
    }


def _prefill_cellvit_cache(
    *, pool: dict[str, Any], cache: dict[str, dict[str, Any]],
    cache_path: Path, output_root: Path, model: Path, cellvit_root: Path,
    cellvit_python: Path, gpu_ids: tuple[int, ...], workers: int,
    timeout_seconds: int, resume: bool,
) -> None:
    """Materialize independent patch authorities concurrently across GPUs."""

    unique = {}
    for evaluation in pool["evaluations"]:
        for candidate in evaluation["candidates"]:
            unique[str(candidate["filename"])] = candidate
    pending = [
        candidate
        for filename, candidate in sorted(unique.items())
        if filename not in cache
        or cache[filename].get("authority_algorithm")
        != NATIVE_AUTHORITY_ALGORITHM
    ]
    if not pending:
        return
    active_gpu_ids = tuple(
        gpu_ids[index % len(gpu_ids)] for index in range(workers)
    )
    executors = [
        (gpu_id, ThreadPoolExecutor(max_workers=1))
        for gpu_id in active_gpu_ids
    ]
    try:
        futures = {
            executor.submit(
                _run_cellvit,
                candidate=candidate,
                output_root=output_root,
                model=model,
                cellvit_root=cellvit_root,
                cellvit_python=cellvit_python,
                gpu=gpu_id,
                timeout_seconds=timeout_seconds,
                resume=resume,
            ): str(candidate["filename"])
            for index, candidate in enumerate(pending)
            for gpu_id, executor in (executors[index % len(executors)],)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            try:
                cache[filename] = future.result()
            except Exception as exc:  # noqa: BLE001 - one fail-closed patch record
                cache[filename] = {
                    "status": "rejected",
                    "authority_algorithm": NATIVE_AUTHORITY_ALGORITHM,
                    "failure_codes": [
                        f"cellvit_controller_{type(exc).__name__}: {exc}"
                    ],
                    "llm_api_used": False,
                }
            _write_json_atomic(cache_path, cache)
            print(
                json.dumps(
                    {
                        "stage": "parallel_cellvit_native_authority",
                        "completed": completed,
                        "total": len(pending),
                        "filename": filename,
                        "status": cache[filename].get("status"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        for _gpu_id, executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def _qualification_row(
    *, evaluation: dict[str, Any], candidate: dict[str, Any], rank: int,
    authority: dict[str, Any], output_root: Path, user_authority_sha256: str,
) -> dict[str, Any]:
    image = Path(str(candidate["source_image"]))
    tissue = Path(str(candidate["source_tissue_mask"]))
    nuclei = Path(str(candidate["source_nuclei_mask"]))
    source_digests = {
        "image_sha256": sha256_file(image),
        "tissue_mask_sha256": sha256_file(tissue),
        "nuclei_mask_sha256": sha256_file(nuclei),
        "nuclei_instances_sha256": authority["cells_json_sha256"],
    }
    primitive_id = str(evaluation["primitive_id"])
    mechanism_id = str(evaluation["mechanism_id"])
    instruction = FROZEN_EXECUTION_INSTRUCTIONS[(mechanism_id, primitive_id)]
    semantic, semantic_digest = _semantic_intent(instruction, primitive_id)
    evaluation_index = int(evaluation["evaluation_index"])
    short = hashlib.sha256(
        f"{evaluation['evaluation_id']}::{candidate['filename']}".encode()
    ).hexdigest()[:12]
    case_id = f"panda_ps_{evaluation_index:02d}_{rank:02d}_{short}"
    auxiliary_uris: dict[str, str] = {}
    auxiliary_digests: dict[str, str] = {}
    auxiliary_provenance: dict[str, dict[str, Any]] = {}
    if primitive_id == "local-invasive-clearance-v1":
        auxiliary_uris, auxiliary_digests, auxiliary_provenance = _fixed_roi(
            shape=(512, 512),
            path=output_root / "roi" / f"{case_id}_local_clearance_roi.png",
            source_tissue_sha256=source_digests["tissue_mask_sha256"],
            user_authority_sha256=user_authority_sha256,
        )
    return {
        "case_id": case_id,
        "source_index": evaluation_index * 100 + rank,
        "evaluation_index": evaluation_index,
        "evaluation_id": str(evaluation["evaluation_id"]),
        "organ": "prostate",
        "dataset": "PANDA",
        "source_image_uri": str(image.resolve()),
        "source_tissue_mask_uri": str(tissue.resolve()),
        "source_nuclei_mask_uri": str(nuclei.resolve()),
        "source_nuclei_instances_uri": authority["cells_json"],
        "source_digests": source_digests,
        "source_manifest_metadata": {
            "provider": "PANDA",
            "patch_grade": "annotation_defined_not_clinically_inferred",
        },
        "pathology_domain_id": "prostate-adenocarcinoma-v1",
        "annotation_profile_id": "panda-gleason-v1",
        "cell_observation_profile_id": "cellvit-five-class-v1",
        "cell_population_profile_id": "prostate-cellvit-source-first-v1",
        "mechanism_id": mechanism_id,
        "primitive_id": primitive_id,
        "instruction": instruction,
        "prebound_semantic_intent": semantic,
        "prebound_semantic_intent_sha256": semantic_digest,
        "joint_area_budget": (
            None
            if mechanism_id
            in {
                "prostate-local-population-modulation",
                "prostate-pattern-5-peripheral-scatter",
            }
            else _joint_area_budget(primitive_id)
        ),
        "seed": 42 + rank,
        "pixel_size_um": 0.25,
        "execution_allowed": True,
        "decision_status": "eligible",
        "decision_reason_code": "mask_only_candidate_plus_native_authority",
        "review_basis": "typed_source_masks_and_frozen_cellvit_native_instances",
        "visual_observations": {
            "policy": "mask_only_no_h_e_for_planning",
            "candidate_metrics": candidate,
        },
        "instance_authority_source": CELLVIT_PRODUCER_ID,
        "native_instance_authority": authority,
        "auxiliary_structure_uris": auxiliary_uris,
        "auxiliary_structure_sha256": auxiliary_digests,
        "auxiliary_structure_provenance": auxiliary_provenance,
        "candidate_pool_rank": rank,
        "source_slide_id": candidate["slide_id"],
        "source_sample_id": candidate["sample_id"],
        "source_cross_meta_case_id": candidate["cross_meta_case_id"],
        "user_authority_text_sha256": user_authority_sha256,
        "llm_api_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cellvit-root", type=Path, required=True)
    parser.add_argument("--cellvit-model", type=Path, required=True)
    parser.add_argument("--cellvit-python", type=Path, required=True)
    parser.add_argument("--probnet-checkpoint", type=Path, required=True)
    parser.add_argument("--nuclei-library", type=Path, required=True)
    parser.add_argument("--valid-per-evaluation", type=int, default=8)
    parser.add_argument(
        "--valid-per-evaluation-overrides",
        help="Comma-separated INDEX:COUNT overrides for targeted rescreening.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cellvit-workers", type=int, default=1)
    parser.add_argument(
        "--gpus",
        help="Comma-separated GPU IDs used by parallel CellViT prefill.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="Populate/validate native CellViT cache without writing authority.",
    )
    parser.add_argument(
        "--user-authority-text", default=DEFAULT_USER_AUTHORITY_TEXT
    )
    args = parser.parse_args()
    if args.valid_per_evaluation < 5:
        raise ValueError("at least five valid candidates per evaluation are required")
    valid_count_overrides = _parse_evaluation_count_overrides(
        args.valid_per_evaluation_overrides
    )
    pool = json.loads(args.candidate_pool.read_text(encoding="utf-8"))
    if pool.get("schema_version") != (
        "panda-cross-meta-eval-primitive-candidate-pool-v1"
    ):
        raise ValueError("unsupported PANDA candidate pool")
    declared_pool_digest = pool.get("candidate_pool_sha256")
    unsigned_pool = dict(pool)
    unsigned_pool.pop("candidate_pool_sha256", None)
    if declared_pool_digest != _canonical_sha256(unsigned_pool):
        raise ValueError("PANDA candidate pool digest mismatch")
    cross_meta_eval = Path(str(pool.get("cross_meta_eval") or ""))
    if (
        not cross_meta_eval.is_file()
        or sha256_file(cross_meta_eval) != pool.get("cross_meta_eval_sha256")
    ):
        raise ValueError("frozen cross-meta eval source is missing or drifted")
    for path in (
        args.cellvit_root,
        args.cellvit_model,
        args.cellvit_python,
        args.probnet_checkpoint,
        args.nuclei_library,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    user_authority_sha256 = hashlib.sha256(
        args.user_authority_text.encode("utf-8")
    ).hexdigest()
    evaluations = []
    qualification_rows = []
    cache: dict[str, dict[str, Any]] = {}
    cache_path = root / "native_instance_cache.json"
    if args.resume and cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if args.cellvit_workers <= 0:
        raise ValueError("CellViT workers must be positive")
    gpu_ids = tuple(
        int(value.strip())
        for value in (args.gpus or str(args.gpu)).split(",")
        if value.strip()
    )
    if not gpu_ids:
        raise ValueError("at least one CellViT GPU ID is required")
    if args.cellvit_workers > 1:
        _prefill_cellvit_cache(
            pool=pool,
            cache=cache,
            cache_path=cache_path,
            output_root=root,
            model=args.cellvit_model.resolve(),
            cellvit_root=args.cellvit_root.resolve(),
            cellvit_python=args.cellvit_python.resolve(),
            gpu_ids=gpu_ids,
            workers=args.cellvit_workers,
            timeout_seconds=args.timeout_seconds,
            resume=args.resume,
        )
    if args.prefill_only:
        if args.cellvit_workers <= 1:
            raise ValueError("prefill-only requires parallel CellViT prefill")
        status_counts = Counter(
            str(item.get("status") or "missing") for item in cache.values()
        )
        print(
            json.dumps(
                {
                    "stage": "native_authority_prefill_complete",
                    "cache": str(cache_path),
                    "cache_record_count": len(cache),
                    "status_counts": dict(sorted(status_counts.items())),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    for evaluation_index, evaluation in enumerate(pool["evaluations"]):
        required_valid_count = valid_count_overrides.get(
            evaluation_index, args.valid_per_evaluation
        )
        current = {**evaluation, "evaluation_index": evaluation_index}
        current["candidate_screening_instruction"] = evaluation["instruction"]
        current["instruction"] = FROZEN_EXECUTION_INSTRUCTIONS[
            (str(evaluation["mechanism_id"]), str(evaluation["primitive_id"]))
        ]
        accepted = []
        rejected = []
        for rank, candidate in enumerate(evaluation["candidates"], start=1):
            filename = str(candidate["filename"])
            if (
                filename not in cache
                or cache[filename].get("authority_algorithm")
                != NATIVE_AUTHORITY_ALGORITHM
            ):
                print(
                    json.dumps(
                        {
                            "stage": "cellvit_native_authority",
                            "evaluation_index": evaluation_index,
                            "candidate_rank": rank,
                            "filename": filename,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                cache[filename] = _run_cellvit(
                    candidate=candidate,
                    output_root=root,
                    model=args.cellvit_model.resolve(),
                    cellvit_root=args.cellvit_root.resolve(),
                    cellvit_python=args.cellvit_python.resolve(),
                    gpu=args.gpu,
                    timeout_seconds=args.timeout_seconds,
                    resume=args.resume,
                )
                _write_json_atomic(cache_path, cache)
            authority = cache[filename]
            if authority.get("status") != "verified":
                rejected.append(
                    {
                        "candidate_rank": rank,
                        "filename": filename,
                        "failure_codes": authority.get("failure_codes", []),
                    }
                )
                continue
            row = _qualification_row(
                evaluation=current,
                candidate=candidate,
                rank=rank,
                authority=authority,
                output_root=root,
                user_authority_sha256=user_authority_sha256,
            )
            qualification_rows.append(row)
            accepted.append(
                {
                    "case_id": row["case_id"],
                    "candidate_rank": rank,
                    "filename": filename,
                    "slide_id": candidate["slide_id"],
                    "native_instance_validation_sha256": authority["validation"][
                        "validation_sha256"
                    ],
                }
            )
            if len(accepted) == required_valid_count:
                break
        if len(accepted) < required_valid_count:
            raise RuntimeError(
                f"{evaluation['evaluation_id']} has only {len(accepted)} "
                f"native-authority candidates; required {required_valid_count}"
            )
        evaluations.append(
            {
                "evaluation_index": evaluation_index,
                "evaluation_id": evaluation["evaluation_id"],
                "mechanism_id": evaluation["mechanism_id"],
                "primitive_id": evaluation["primitive_id"],
                "final_diversity_policy": evaluation["final_diversity_policy"],
                "candidate_screening_instruction": evaluation["instruction"],
                "frozen_execution_instruction": current["instruction"],
                "required_native_authority_candidate_count": (
                    required_valid_count
                ),
                "accepted": accepted,
                "rejected": rejected,
            }
        )
    probnet_digest = sha256_file(args.probnet_checkpoint)
    code_inventory, code_inventory_digest = _runtime_code_inventory()
    runtime_authority = {
        "mature_probnet_executor": {
            "path": str(args.probnet_checkpoint.resolve()),
            "sha256": probnet_digest,
        },
        "frozen_probnet_spatial_ranker": {
            "path": str(args.probnet_checkpoint.resolve()),
            "sha256": probnet_digest,
        },
        "executor_ranker_same_checkpoint": True,
        "cellvit_native_instance_producer": {
            "path": str(args.cellvit_model.resolve()),
            "sha256": sha256_file(args.cellvit_model),
        },
        "panda_nucleus_instance_library": {
            "path": str(args.nuclei_library.resolve()),
            "sha256": _directory_sha256(args.nuclei_library.resolve()),
        },
        "joint_skill_catalog": {
            "path": str(
                (
                    REPOSITORY_ROOT
                    / "phase3_joint_edit_refine"
                    / "skills"
                    / "catalog"
                ).resolve()
            ),
            "sha256": _directory_sha256(
                REPOSITORY_ROOT
                / "phase3_joint_edit_refine"
                / "skills"
                / "catalog"
            ),
        },
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "runtime_code_files": code_inventory,
        "runtime_code_inventory_sha256": code_inventory_digest,
        "llm_api_used": False,
        "he_observation_policy": "frozen_cellvit_only_never_llm",
    }
    runtime_authority["runtime_authority_sha256"] = _canonical_sha256(
        runtime_authority
    )
    qualification_manifest = {
        "schema_version": QUALIFICATION_MANIFEST_SCHEMA,
        "case_count": len(qualification_rows),
        "cases": qualification_rows,
    }
    qualification_manifest["manifest_sha256"] = _canonical_sha256(
        qualification_manifest
    )
    qualification_path = root / "candidate_qualification_manifest.json"
    qualification_path.write_text(
        json.dumps(
            qualification_manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    authority_payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_pool_sha256": pool["candidate_pool_sha256"],
        "user_authority_text": args.user_authority_text,
        "user_authority_text_sha256": user_authority_sha256,
        "evaluation_count": len(evaluations),
        "authority_candidate_count": len(qualification_rows),
        "default_valid_per_evaluation": args.valid_per_evaluation,
        "valid_per_evaluation_overrides": {
            str(index): count
            for index, count in sorted(valid_count_overrides.items())
        },
        "evaluations": evaluations,
        "runtime_authority": runtime_authority,
        "candidate_qualification_manifest": str(qualification_path),
        "candidate_qualification_manifest_sha256": sha256_file(
            qualification_path
        ),
        "freeze_status": "authority_materialized_pending_candidate_compilation",
        "source_h_e_pixel_consumers": [
            "frozen_cellvit_native_instance_producer",
            "deterministic_audit_board_renderer_during_later_execution",
        ],
        "source_h_e_semantic_interpreters": [
            "frozen_cellvit_native_instance_producer"
        ],
        "llm_h_e_exposure": False,
        "llm_api_used": False,
    }
    authority_payload["authority_manifest_sha256"] = _canonical_sha256(
        authority_payload
    )
    authority_path = root / "authority_manifest.json"
    authority_path.write_text(
        json.dumps(
            authority_payload, indent=2, ensure_ascii=False, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "authority_manifest": str(authority_path),
                "authority_manifest_sha256": authority_payload[
                    "authority_manifest_sha256"
                ],
                "qualification_case_count": len(qualification_rows),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
