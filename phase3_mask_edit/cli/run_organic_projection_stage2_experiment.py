"""Run the Phase 3 organic projection Stage 2 experiment.

The experiment is intentionally narrow:
1. Re-run the real proposal through organic_v2 smoke.
2. Sweep organic_v2 score weights and peritumoral decay over seeds.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from phase3_mask_edit.backends.llm_contour import (
    PROJECTION_MODE_ORGANIC_V2,
    execute_contour_proposal_write,
    load_contour_proposal_json,
    rasterize_contour_proposal,
    validate_contour_proposal,
)
from phase3_mask_edit.backends.organic_projection import (
    ORGANIC_PROJECTION_BACKEND,
    apply_organic_projected_label_write,
)
from phase3_mask_edit.core.config import load_recipe
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import (
    load_id_mask,
    save_change_region,
    save_id_mask,
    save_metadata,
    save_rgb_mask,
)
from phase3_mask_edit.core.validation import validate_edit_result


DEFAULT_WEIGHT_COMBOS: tuple[tuple[str, float, float, float], ...] = (
    ("template_leaning", 0.70, 0.25, 0.05),
    ("balanced", 0.45, 0.45, 0.10),
    ("spatial_leaning", 0.25, 0.65, 0.10),
    ("noisy_balanced", 0.35, 0.45, 0.20),
)
DEFAULT_DECAYS: tuple[float, ...] = (24.0, 48.0, 72.0, 96.0)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    schema = MaskProfileSchema.from_reference_profile(args.profile)
    recipe = load_recipe(args.recipe)
    primitive_config = _primitive_config(recipe, args.primitive)
    source_labels, target_label = _operation_labels(primitive_config)
    if args.source_label:
        source_labels = tuple(args.source_label)
    if args.target_label:
        target_label = args.target_label

    mask = load_id_mask(args.mask)
    proposal = validate_contour_proposal(
        load_contour_proposal_json(args.fixture),
        schema=schema,
        mask_shape=tuple(mask.shape),
        primitive=args.primitive,
        reference_profile=args.profile,
        target_label=target_label,
        allowed_source_labels=source_labels,
        max_regions=args.max_regions,
        max_points_per_region=args.max_points_per_region,
    )
    raw_candidate = rasterize_contour_proposal(proposal)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    smoke = run_organic_v2_smoke(
        mask,
        proposal,
        schema=schema,
        primitive_config=primitive_config,
        organic_seed=args.organic_seed,
        output_dir=out / "smoke",
    )
    save_metadata(smoke, out / "smoke_summary.json")

    grid_rows = run_weight_decay_seed_grid(
        mask,
        raw_candidate,
        schema=schema,
        source_labels=source_labels,
        target_label=target_label,
        primitive_config=primitive_config,
        weight_combos=_parse_weight_combos(args.weight_combo),
        decays=_parse_float_list(args.decay_px) or DEFAULT_DECAYS,
        seeds=_parse_int_list(args.seed) or DEFAULT_SEEDS,
        template_sigma=args.template_sigma,
        noise_sigma=args.noise_sigma,
        noise_amplitude=args.noise_amplitude,
    )
    save_metadata({"rows": grid_rows}, out / "grid_results.json")
    _write_csv(grid_rows, out / "grid_results.csv")

    trend = summarize_grid_trends(grid_rows)
    save_metadata(trend, out / "trend_summary.json")

    if args.print_summary:
        print(
            f"smoke organic_v2_pass={smoke['organic_v2']['validation_passed']} "
            f"grid_runs={len(grid_rows)}"
        )
        print(f"trend_summary={out / 'trend_summary.json'}")

    return 0


def run_organic_v2_smoke(
    mask: np.ndarray,
    proposal,
    *,
    schema: MaskProfileSchema,
    primitive_config: Mapping[str, Any],
    organic_seed: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run organic_v2 once and return compact smoke metrics."""

    summary: dict[str, Any] = {}
    mode = PROJECTION_MODE_ORGANIC_V2
    edit = execute_contour_proposal_write(
        mask,
        proposal,
        schema=schema,
        primitive_config=primitive_config,
        projection_mode=mode,
        organic_seed=organic_seed,
    )
    validation = validate_edit_result(
        src_mask=mask,
        target_mask=edit.target_mask,
        change_region=edit.change_region,
        schema=schema,
        primitive_config=primitive_config,
        changed_area_fraction=edit.changed_area_fraction,
    )
    metrics = projection_metrics(
        mask,
        edit.change_region,
        schema=schema,
        source_labels=tuple(edit.ops_log.get("source_labels", ())),
        target_label=str(edit.ops_log.get("target_label", "")),
    )
    entry = {
        "projection_mode": mode,
        "validation_passed": bool(validation.passed),
        "failed_checks": [check.name for check in validation.failed_checks],
        "selected_pixels": int(edit.selected_pixels),
        "changed_area_fraction": float(edit.changed_area_fraction),
        "warnings": list(edit.warnings),
        "ops_log": edit.ops_log,
        "metrics": metrics,
        "validation": _jsonable_dataclass(validation),
    }
    summary[mode] = entry

    if output_dir is not None:
        mode_dir = output_dir / mode
        save_change_region(edit.change_region, mode_dir / "change_region.png")
        save_id_mask(edit.target_mask, mode_dir / "target_mask.png")
        save_rgb_mask(edit.target_mask, mode_dir / "target_mask_rgb.png")
        save_metadata(entry, mode_dir / "summary.json")

    return summary


run_v1_v2_smoke = run_organic_v2_smoke


def run_weight_decay_seed_grid(
    mask: np.ndarray,
    raw_candidate: np.ndarray,
    *,
    schema: MaskProfileSchema,
    source_labels: Sequence[str],
    target_label: str,
    primitive_config: Mapping[str, Any],
    weight_combos: Sequence[tuple[str, float, float, float]],
    decays: Sequence[float],
    seeds: Sequence[int],
    template_sigma: float,
    noise_sigma: float,
    noise_amplitude: float,
) -> list[dict[str, Any]]:
    """Run the reduced 4 x 4 x 5 Stage 2B grid."""

    rows: list[dict[str, Any]] = []
    for combo_name, w_template, w_spatial, w_noise in weight_combos:
        for decay_px in decays:
            for seed in seeds:
                edit = apply_organic_projected_label_write(
                    mask,
                    raw_candidate,
                    schema=schema,
                    source_labels=source_labels,
                    target_label=target_label,
                    primitive_config=primitive_config,
                    seed=seed,
                    template_sigma=template_sigma,
                    noise_sigma=noise_sigma,
                    noise_amplitude=noise_amplitude,
                    w_template=w_template,
                    w_spatial=w_spatial,
                    w_noise=w_noise,
                    decay_px=decay_px,
                )
                validation = validate_edit_result(
                    src_mask=mask,
                    target_mask=edit.target_mask,
                    change_region=edit.change_region,
                    schema=schema,
                    primitive_config=primitive_config,
                    changed_area_fraction=edit.changed_area_fraction,
                )
                metrics = projection_metrics(
                    mask,
                    edit.change_region,
                    schema=schema,
                    source_labels=source_labels,
                    target_label=target_label,
                )
                rows.append(
                    {
                        "projection_backend": ORGANIC_PROJECTION_BACKEND,
                        "weight_combo": combo_name,
                        "w_template": float(w_template),
                        "w_spatial": float(w_spatial),
                        "w_noise": float(w_noise),
                        "decay_px": float(decay_px),
                        "seed": int(seed),
                        "validation_passed": bool(validation.passed),
                        "failed_checks": ";".join(
                            check.name for check in validation.failed_checks
                        ),
                        "selected_pixels": int(edit.selected_pixels),
                        "target_pixels": int(edit.ops_log.get("target_pixels", 0)),
                        "area_shortfall": int(edit.ops_log.get("area_shortfall", 0)),
                        "selected_to_target_ratio": _safe_ratio(
                            edit.selected_pixels,
                            int(edit.ops_log.get("target_pixels", 0)),
                        ),
                        "template_overlap_with_legal_domain": float(
                            edit.ops_log.get("template_overlap_with_legal_domain", 0.0)
                        ),
                        "legal_domain_pixels": int(
                            edit.ops_log.get("legal_domain_pixels", 0)
                        ),
                        **metrics,
                    }
                )
    return rows


def projection_metrics(
    mask: np.ndarray,
    change_region: np.ndarray,
    *,
    schema: MaskProfileSchema,
    source_labels: Sequence[str],
    target_label: str,
) -> dict[str, Any]:
    """Compute label-safety, tumor-distance, and component metrics."""

    change = np.asarray(change_region, dtype=bool)
    selected_pixels = int(np.count_nonzero(change))
    source_mask = np.zeros(mask.shape, dtype=bool)
    for label in source_labels:
        if label:
            source_mask |= np.isin(mask, schema.resolve_fine_ids(label))
    illegal_pixels = int(np.count_nonzero(change & ~source_mask)) if source_labels else 0

    tumor = np.isin(mask, schema.tumor_fine_ids)
    if selected_pixels > 0 and np.any(tumor):
        dist_to_tumor = ndimage.distance_transform_edt(~tumor)
        values = dist_to_tumor[change]
        mean_dist = float(np.mean(values))
        p90_dist = float(np.percentile(values, 90))
    else:
        mean_dist = None
        p90_dist = None

    labeled, component_count = ndimage.label(change, structure=_four_neighbor_structure())
    if component_count:
        areas = ndimage.sum(change, labeled, range(1, component_count + 1))
        largest = int(np.max(areas))
        small_count = int(np.count_nonzero(np.asarray(areas) < max(1, selected_pixels * 0.05)))
    else:
        largest = 0
        small_count = 0

    target_ids = schema.resolve_fine_ids(target_label) if target_label else ()
    return {
        "illegal_source_pixels": illegal_pixels,
        "label_safe": illegal_pixels == 0,
        "target_label": target_label,
        "target_fine_id": int(target_ids[0]) if target_ids else None,
        "mean_dist_to_tumor_boundary_px": mean_dist,
        "p90_dist_to_tumor_boundary_px": p90_dist,
        "component_count": int(component_count),
        "largest_component_pixels": largest,
        "largest_component_fraction": _safe_ratio(largest, selected_pixels),
        "small_component_count": small_count,
    }


def summarize_grid_trends(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the reduced grid by weight combo and decay."""

    by_combo_decay: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["weight_combo"]), float(row["decay_px"]))
        by_combo_decay.setdefault(key, []).append(row)

    groups = []
    for (combo, decay), group_rows in sorted(by_combo_decay.items()):
        mean_dist_values = [
            float(row["mean_dist_to_tumor_boundary_px"])
            for row in group_rows
            if row["mean_dist_to_tumor_boundary_px"] is not None
        ]
        p90_values = [
            float(row["p90_dist_to_tumor_boundary_px"])
            for row in group_rows
            if row["p90_dist_to_tumor_boundary_px"] is not None
        ]
        pass_rate = sum(1 for row in group_rows if row["validation_passed"]) / len(group_rows)
        groups.append(
            {
                "weight_combo": combo,
                "decay_px": decay,
                "runs": len(group_rows),
                "validation_pass_rate": pass_rate,
                "mean_dist_to_tumor_boundary_px_mean": _mean_or_none(mean_dist_values),
                "mean_dist_to_tumor_boundary_px_std": _std_or_none(mean_dist_values),
                "p90_dist_to_tumor_boundary_px_mean": _mean_or_none(p90_values),
                "selected_to_target_ratio_mean": _mean_or_none(
                    [float(row["selected_to_target_ratio"]) for row in group_rows]
                ),
                "component_count_mean": _mean_or_none(
                    [float(row["component_count"]) for row in group_rows]
                ),
                "largest_component_fraction_mean": _mean_or_none(
                    [float(row["largest_component_fraction"]) for row in group_rows]
                ),
            }
        )

    by_combo: dict[str, list[Mapping[str, Any]]] = {}
    for group in groups:
        by_combo.setdefault(str(group["weight_combo"]), []).append(group)

    monotonic_notes = []
    for combo, combo_groups in sorted(by_combo.items()):
        ordered = sorted(combo_groups, key=lambda item: float(item["decay_px"]))
        distances = [
            item["mean_dist_to_tumor_boundary_px_mean"] for item in ordered
        ]
        has_trend = all(
            distances[index] is not None
            and distances[index + 1] is not None
            and float(distances[index]) <= float(distances[index + 1]) + 1e-9
            for index in range(len(distances) - 1)
        )
        monotonic_notes.append(
            {
                "weight_combo": combo,
                "decays": [item["decay_px"] for item in ordered],
                "mean_distances": distances,
                "mean_distance_non_decreasing_with_decay": bool(has_trend),
            }
        )

    return {
        "grid_rows": len(rows),
        "groups": groups,
        "decay_trend_by_weight_combo": monotonic_notes,
    }


def _primitive_config(recipe: Mapping[str, Any], primitive_name: str) -> Mapping[str, Any]:
    for primitive in recipe.get("primitives", []):
        if isinstance(primitive, Mapping) and primitive.get("name") == primitive_name:
            return primitive
    raise ValueError(f"Unknown primitive: {primitive_name}")


def _operation_labels(primitive_config: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    operation = primitive_config.get("mask_operation", {})
    if not isinstance(operation, Mapping):
        operation = {}
    source = operation.get("source")
    target = operation.get("target")
    if isinstance(source, str) and isinstance(target, str):
        return (source,), target
    raise ValueError(
        f"Primitive {primitive_config.get('name')} needs explicit source/target labels."
    )


def _parse_weight_combos(values: Sequence[str] | None) -> tuple[tuple[str, float, float, float], ...]:
    if not values:
        return DEFAULT_WEIGHT_COMBOS
    combos = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError(
                "--weight-combo must be name,w_template,w_spatial,w_noise"
            )
        combos.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return tuple(combos)


def _parse_float_list(value: str | None) -> tuple[float, ...]:
    if not value:
        return ()
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_int_list(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _jsonable_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator <= 0:
        return 0.0
    return float(numerator) / denominator


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _std_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.std(values))


def _four_neighbor_structure() -> np.ndarray:
    return np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
        dtype=bool,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reduced Stage 2 organic projection experiments."
    )
    parser.add_argument("--profile", default="BCSS")
    parser.add_argument("--primitive", default="stromal_immune_infiltration")
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("phase3_mask_edit/recipes/generic.yaml"),
    )
    parser.add_argument("--source-label", action="append")
    parser.add_argument("--target-label")
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument("--max-points-per-region", type=int, default=64)
    parser.add_argument("--organic-seed", type=int, default=11)
    parser.add_argument(
        "--weight-combo",
        action="append",
        help="Override/add combo as name,w_template,w_spatial,w_noise.",
    )
    parser.add_argument(
        "--decay-px",
        help="Comma-separated decay values. Default: 24,48,72,96.",
    )
    parser.add_argument("--seed", help="Comma-separated seeds. Default: 0,1,2,3,4.")
    parser.add_argument("--template-sigma", type=float, default=3.0)
    parser.add_argument("--noise-sigma", type=float, default=18.0)
    parser.add_argument("--noise-amplitude", type=float, default=0.18)
    parser.add_argument("--print-summary", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
