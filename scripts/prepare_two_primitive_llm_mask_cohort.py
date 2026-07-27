#!/usr/bin/env python3
"""Run native LLM-contour mask edits for the two-primitive trajectory cohort.

Each candidate is edited independently with the formal BCSS implementations of
``tumor_burden_increase`` and ``stromal_immune_infiltration``.  Moderate and
Significant are proposed independently by the multimodal contour agent.  Their
overlap is measured but never altered or used for cohort selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Mapping

import numpy as np
from phase3_mask_edit.backends.fixture_contour import STATUS_VALIDATED
from phase3_mask_edit.backends.llm_agent import (
    LLMContourAgentResult,
    OpenAICompatibleMultimodalContourProvider,
    execute_llm_contour_agent,
)
from phase3_mask_edit.backends.llm_contour import PROJECTION_MODE_ORGANIC_V2
from phase3_mask_edit.benchmark.intents import (
    primitive_config_by_name,
    source_target_labels_for_primitive,
)
from phase3_mask_edit.benchmark.runner import _BoundaryTolerantContourProvider
from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import (
    load_id_mask,
    save_change_region,
    save_id_mask,
    save_metadata,
    save_rgb_mask,
)
from phase3_mask_edit.core.validation import ValidationResult


PRIMITIVES = (
    "tumor_burden_increase",
    "stromal_immune_infiltration",
)
STRENGTHS = ("moderate", "significant")
SEED_NAMESPACE = "two_primitive_llm_native_v1"
AUTH_PATTERN = re.compile(r"(?i)(authorization[^\n]*?bearer\s+)[^\s,'\"]+")
TOKEN_PATTERN = re.compile(r"(?i)\b(sk-[A-Za-z0-9_-]{8,})\b")
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", default="BCSS")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-model", default="gpt-4.1-mini")
    parser.add_argument("--api-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--api-image-detail",
        choices=("low", "high", "auto"),
        default="high",
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--max-points-per-region", type=int, default=64)
    parser.add_argument("--coordinate-tolerance-px", type=float, default=16.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--final-count", type=int, default=300)
    parser.add_argument("--minimum-dose-increase", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in (SEED_NAMESPACE, *parts))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def safe_text(value: Any) -> str:
    text = str(value)
    text = AUTH_PATTERN.sub(r"\1<redacted>", text)
    return TOKEN_PATTERN.sub("<redacted>", text)


def sanitize_api_environment(name: str) -> None:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing API key environment variable: {name}")
    sanitized = value.strip().replace("\r", "").replace("\n", "")
    if not sanitized:
        raise RuntimeError(f"empty API key environment variable: {name}")
    os.environ[name] = sanitized


def validation_metadata(result: ValidationResult) -> dict[str, Any]:
    return {
        "passed": bool(result.passed),
        "primitive": result.primitive,
        "checks": [asdict(check) for check in result.checks],
        "warnings": list(result.warnings),
    }


def result_attempt_metadata(result: LLMContourAgentResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "attempt_count": len(result.attempts),
        "attempt_statuses": [attempt.status for attempt in result.attempts],
        "final_validation": (
            validation_metadata(result.validation)
            if result.validation is not None
            else None
        ),
        "error": safe_text(result.error) if result.error else None,
        "artifact_paths": dict(result.artifact_paths),
    }


def load_existing_native(output_dir: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    summary_path = output_dir / "execution_summary.json"
    target_path = output_dir / "final_target_mask.png"
    if not summary_path.is_file() or not target_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != STATUS_VALIDATED:
        return None
    attempts = summary.get("attempts") or []
    return load_id_mask(target_path), {
        "status": summary.get("status"),
        "attempt_count": len(attempts),
        "attempt_statuses": [
            str(attempt.get("status", "")) for attempt in attempts
        ],
        "final_validation": (
            attempts[-1].get("validation") if attempts else None
        ),
        "error": None,
        "artifact_paths": dict(summary.get("artifact_paths") or {}),
        "resumed": True,
    }


def run_native_edit(
    *,
    source_mask: np.ndarray,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    primitive: str,
    strength: str,
    provider: OpenAICompatibleMultimodalContourProvider,
    output_dir: Path,
    args: argparse.Namespace,
    sample_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    existing = load_existing_native(output_dir)
    if existing is not None:
        return existing

    source_labels, target_label = source_target_labels_for_primitive(
        primitive_config, schema
    )
    intent = EditIntent(
        primitive=primitive,
        strength=strength,
        reference_profile=args.profile,
        source_labels=source_labels,
        target_label=target_label,
        seed=stable_seed(args.seed, sample_id, primitive, strength),
    )
    tolerant_provider = _BoundaryTolerantContourProvider(
        provider,
        mask_shape=tuple(source_mask.shape),
        tolerance_px=args.coordinate_tolerance_px,
    )
    result = execute_llm_contour_agent(
        old_mask=source_mask,
        schema=schema,
        intent=intent,
        primitive_config=primitive_config,
        provider=tolerant_provider,
        output_dir=output_dir,
        allowed_source_labels=source_labels,
        max_attempts=args.max_attempts,
        max_regions=args.max_regions,
        max_points_per_region=args.max_points_per_region,
        projection_mode=PROJECTION_MODE_ORGANIC_V2,
        organic_seed=stable_seed(args.seed, sample_id, primitive, strength, "organic"),
    )
    metadata = result_attempt_metadata(result)
    if result.status != STATUS_VALIDATED or result.edit_result is None:
        raise RuntimeError(
            f"{primitive}/{strength} contour failed: "
            f"{safe_text(result.error or result.status)}"
        )
    return np.asarray(result.edit_result.target_mask), metadata


def dose_denominator(
    source_mask: np.ndarray,
    *,
    schema: MaskProfileSchema,
    primitive: str,
) -> int:
    if primitive == "stromal_immune_infiltration":
        ids = (
            schema.resolve_fine_ids("Stroma")
            + schema.resolve_fine_ids("Immune infiltrate")
        )
        return int(np.count_nonzero(np.isin(source_mask, ids)))
    return int(source_mask.size)


def native_strength_metrics(
    *,
    source_mask: np.ndarray,
    moderate_target: np.ndarray,
    native_significant_target: np.ndarray,
    schema: MaskProfileSchema,
    primitive: str,
    minimum_dose_increase: float,
) -> dict[str, Any]:
    moderate = source_mask != moderate_target
    native_significant = source_mask != native_significant_target
    denominator = dose_denominator(
        source_mask, schema=schema, primitive=primitive
    )
    moderate_count = int(np.count_nonzero(moderate))
    native_significant_count = int(np.count_nonzero(native_significant))
    union_count = int(np.count_nonzero(moderate | native_significant))
    native_intersection = int(np.count_nonzero(moderate & native_significant))
    moderate_dose = moderate_count / denominator
    significant_dose = native_significant_count / denominator
    dose_increase = significant_dose - moderate_dose
    if dose_increase < minimum_dose_increase:
        raise RuntimeError(
            "native Significant dose does not sufficiently exceed Moderate: "
            f"increase={dose_increase:.6f}, required={minimum_dose_increase:.6f}"
        )
    return {
        "policy": "independent_native_contours_no_geometric_reconciliation",
        "denominator_pixels": denominator,
        "moderate_pixels": moderate_count,
        "significant_pixels": native_significant_count,
        "intersection_pixels": native_intersection,
        "union_pixels": union_count,
        "moderate_containment_in_significant": (
            native_intersection / moderate_count if moderate_count else 0.0
        ),
        "iou": (
            native_intersection / union_count if union_count else 0.0
        ),
        "moderate_dose": moderate_dose,
        "significant_dose": significant_dose,
        "dose_increase": dose_increase,
        "minimum_required_dose_increase": minimum_dose_increase,
    }


def save_final_masks(
    *,
    output_dir: Path,
    source_mask: np.ndarray,
    moderate_target: np.ndarray,
    significant_target: np.ndarray,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_mask": str(save_id_mask(source_mask, output_dir / "source_mask.png")),
        "source_mask_rgb": str(
            save_rgb_mask(source_mask, output_dir / "source_mask_rgb.png")
        ),
        "moderate_target_mask": str(
            save_id_mask(moderate_target, output_dir / "moderate_target_mask.png")
        ),
        "moderate_target_mask_rgb": str(
            save_rgb_mask(
                moderate_target, output_dir / "moderate_target_mask_rgb.png"
            )
        ),
        "moderate_change_region": str(
            save_change_region(
                source_mask != moderate_target,
                output_dir / "moderate_change_region.png",
            )
        ),
        "significant_target_mask": str(
            save_id_mask(
                significant_target, output_dir / "significant_target_mask.png"
            )
        ),
        "significant_target_mask_rgb": str(
            save_rgb_mask(
                significant_target,
                output_dir / "significant_target_mask_rgb.png",
            )
        ),
        "significant_change_region": str(
            save_change_region(
                source_mask != significant_target,
                output_dir / "significant_change_region.png",
            )
        ),
    }
    return paths


def primitive_overlap(
    u1: Mapping[str, np.ndarray],
    u2: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for strength in STRENGTHS:
        first = np.asarray(u1[strength], dtype=bool)
        second = np.asarray(u2[strength], dtype=bool)
        intersection = int(np.count_nonzero(first & second))
        union = int(np.count_nonzero(first | second))
        first_count = int(np.count_nonzero(first))
        second_count = int(np.count_nonzero(second))
        output[strength] = {
            "intersection_pixels": intersection,
            "union_pixels": union,
            "iou": intersection / union if union else 0.0,
            "u1_contained_by_u2": intersection / first_count if first_count else 0.0,
            "u2_contained_by_u1": intersection / second_count if second_count else 0.0,
        }
    return output


def run_case(
    row: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    schema: MaskProfileSchema,
    primitive_configs: Mapping[str, Mapping[str, Any]],
    provider: OpenAICompatibleMultimodalContourProvider,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    case_dir = args.output_root / "cases" / sample_id
    result_path = case_dir / "case_result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "eligible" or not args.retry_failed:
            existing["resumed"] = True
            return existing

    source_mask = load_id_mask(row["reference_tissue_mask"])
    native_metadata: dict[str, Any] = {}
    final_targets: dict[str, dict[str, np.ndarray]] = {}
    primitive_metrics: dict[str, Any] = {}
    artifact_paths: dict[str, Any] = {}
    try:
        for primitive in PRIMITIVES:
            config = primitive_configs[primitive]
            native_targets: dict[str, np.ndarray] = {}
            native_metadata[primitive] = {}
            for strength in STRENGTHS:
                target, metadata = run_native_edit(
                    source_mask=source_mask,
                    schema=schema,
                    primitive_config=config,
                    primitive=primitive,
                    strength=strength,
                    provider=provider,
                    output_dir=case_dir / "native" / primitive / strength,
                    args=args,
                    sample_id=sample_id,
                )
                native_targets[strength] = target
                native_metadata[primitive][strength] = metadata

            significant_target = native_targets["significant"]
            metrics = native_strength_metrics(
                source_mask=source_mask,
                moderate_target=native_targets["moderate"],
                native_significant_target=native_targets["significant"],
                schema=schema,
                primitive=primitive,
                minimum_dose_increase=args.minimum_dose_increase,
            )
            moderate_change = source_mask != native_targets["moderate"]
            significant_change = source_mask != significant_target
            final_targets[primitive] = {
                "moderate": moderate_change,
                "significant": significant_change,
            }
            primitive_metrics[primitive] = metrics
            artifact_paths[primitive] = save_final_masks(
                output_dir=case_dir / "final" / primitive,
                source_mask=source_mask,
                moderate_target=native_targets["moderate"],
                significant_target=significant_target,
            )

        overlap = primitive_overlap(
            final_targets["tumor_burden_increase"],
            final_targets["stromal_immune_infiltration"],
        )
        result = {
            "schema_version": 1,
            "status": "eligible",
            "sample_id": sample_id,
            "wsi_id": row["wsi_id"],
            "patient_id": row.get("patient_id", row["wsi_id"]),
            "profile": args.profile,
            "reference_image": row["reference_image"],
            "reference_tissue_mask": row["reference_tissue_mask"],
            "reference_nuclei_mask": row["reference_nuclei_mask"],
            "source_fractions": row["source_fractions"],
            "native_agent": native_metadata,
            "primitive_metrics": primitive_metrics,
            "cross_primitive_overlap_descriptive_only": overlap,
            "artifact_paths": artifact_paths,
            "selection_key": row["selection_key"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "sample_id": sample_id,
            "wsi_id": row["wsi_id"],
            "patient_id": row.get("patient_id", row["wsi_id"]),
            "reference_image": row["reference_image"],
            "reference_tissue_mask": row["reference_tissue_mask"],
            "reference_nuclei_mask": row["reference_nuclei_mask"],
            "source_fractions": row["source_fractions"],
            "native_agent": native_metadata,
            "error": safe_text(exc),
            "selection_key": row["selection_key"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    case_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(result, result_path)
    return result


def round_robin(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["wsi_id"]), []).append(row)
    for items in grouped.values():
        items.sort(key=lambda item: str(item["selection_key"]))
    names = sorted(
        grouped,
        key=lambda name: hashlib.sha256(
            f"{SEED_NAMESPACE}|wsi|{name}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < min(count, len(rows)):
        added = 0
        for name in names:
            items = grouped[name]
            if depth >= len(items):
                continue
            selected.append(items[depth])
            added += 1
            if len(selected) == min(count, len(rows)):
                break
        if added == 0:
            break
        depth += 1
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def summarize(
    *,
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [row for row in rows if row["status"] == "eligible"]
    failures = [row for row in rows if row["status"] == "failed"]
    final = round_robin(eligible, args.final_count)
    write_jsonl(args.output_root / "mask_audit_manifest.jsonl", rows)
    write_jsonl(args.output_root / "eligible_manifest.jsonl", eligible)
    write_jsonl(args.output_root / "final_cohort_manifest.jsonl", final)
    failure_counts = Counter(
        str(row.get("error", "unknown")).split(":", 1)[0] for row in failures
    )
    overlap_values = {
        strength: [
            float(
                row["cross_primitive_overlap_descriptive_only"][strength]["iou"]
            )
            for row in eligible
        ]
        for strength in STRENGTHS
    }
    summary = {
        "schema_version": 1,
        "status": (
            "complete"
            if len(rows) == len(candidates)
            else "partial"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest": str(args.candidate_manifest),
        "candidate_count_requested": len(candidates),
        "processed_count": len(rows),
        "eligible_count": len(eligible),
        "failed_count": len(failures),
        "eligible_wsi_count": len({row["wsi_id"] for row in eligible}),
        "final_cohort_count": len(final),
        "final_cohort_wsi_count": len({row["wsi_id"] for row in final}),
        "final_cohort_complete": len(final) >= args.final_count,
        "selection_policy": (
            "mask_only_primitive_validation_then_deterministic_wsi_round_robin"
        ),
        "cross_primitive_overlap_used_for_selection": False,
        "failure_reason_prefix_counts": dict(sorted(failure_counts.items())),
        "cross_primitive_iou": {
            strength: (
                {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q95": float(np.quantile(values, 0.95)),
                }
                if values
                else {}
            )
            for strength, values in overlap_values.items()
        },
        "configuration": {
            "profile": args.profile,
            "primitives": list(PRIMITIVES),
            "strengths": list(STRENGTHS),
            "api_base_url": args.api_base_url,
            "api_key_env": args.api_key_env,
            "api_model": args.api_model,
            "api_image_detail": args.api_image_detail,
            "max_attempts": args.max_attempts,
            "coordinate_tolerance_px": args.coordinate_tolerance_px,
            "projection_mode": PROJECTION_MODE_ORGANIC_V2,
            "minimum_dose_increase": args.minimum_dose_increase,
            "seed": args.seed,
            "strength_geometry_policy": (
                "independent_native_contours_no_geometric_reconciliation"
            ),
        },
    }
    save_metadata(summary, args.output_root / "run_summary.json")
    return summary


def main() -> int:
    args = parse_args()
    sanitize_api_environment(args.api_key_env)
    candidates = read_jsonl(args.candidate_manifest)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        raise RuntimeError("candidate manifest is empty")
    args.output_root.mkdir(parents=True, exist_ok=True)

    schema = MaskProfileSchema.from_reference_profile(args.profile)
    recipe = load_recipe(default_recipe_path_for_profile(args.profile))
    primitive_configs = {
        name: copy.deepcopy(dict(primitive_config_by_name(recipe, name)))
        for name in PRIMITIVES
    }
    provider = OpenAICompatibleMultimodalContourProvider(
        model=args.api_model,
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        timeout_sec=args.api_timeout_sec,
        temperature=0.0,
        image_detail=args.api_image_detail,
    )

    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for index, row in enumerate(candidates, start=1):
            result = run_case(
                row,
                args=args,
                schema=schema,
                primitive_configs=primitive_configs,
                provider=provider,
            )
            rows.append(result)
            with PRINT_LOCK:
                print(
                    f"[{index}/{len(candidates)}] {result['sample_id']} "
                    f"{result['status']}",
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    run_case,
                    row,
                    args=args,
                    schema=schema,
                    primitive_configs=primitive_configs,
                    provider=provider,
                ): row
                for row in candidates
            }
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - outer safety net.
                    row = futures[future]
                    result = {
                        "status": "failed",
                        "sample_id": row["sample_id"],
                        "wsi_id": row["wsi_id"],
                        "error": safe_text(exc),
                    }
                rows.append(result)
                with PRINT_LOCK:
                    print(
                        f"[{index}/{len(candidates)}] {result['sample_id']} "
                        f"{result['status']}",
                        flush=True,
                    )
    rows.sort(key=lambda row: str(row["sample_id"]))
    summary = summarize(args=args, candidates=candidates, rows=rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "processed_count": summary["processed_count"],
                "eligible_count": summary["eligible_count"],
                "failed_count": summary["failed_count"],
                "final_cohort_count": summary["final_cohort_count"],
                "output_root": str(args.output_root),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if summary["eligible_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
