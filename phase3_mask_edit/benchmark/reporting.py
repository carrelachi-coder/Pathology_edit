"""Formal aggregation and clustered confidence intervals for mask-edit benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from phase3_mask_edit.benchmark.models import BenchmarkIntent
from phase3_mask_edit.benchmark.metrics import mode_aware_score_fields


STRENGTH_RANK = {"mild": 1, "moderate": 2, "significant": 3, "xlarge_deid": 4}
FORMAL_METRICS = {
    "direction_hit_rate": ("direction_hit", "binary"),
    "on_target_transition_ratio": ("on_target_transition_ratio", "continuous"),
    "off_target_change_ratio": ("off_target_change_ratio", "continuous"),
    "spatial_containment_ratio": ("spatial_containment_ratio", "continuous"),
    "semantic_core_pass_rate": ("semantic_core_ok", "binary"),
    "primary_pass_rate": ("primary_ok", "binary"),
    "intended_magnitude_bucket_agreement_rate": (
        "intended_magnitude_bucket_agreement",
        "binary",
    ),
    "strict_all_pass_rate": ("strict_all_ok", "binary"),
    # Legacy aliases retained so existing report consumers continue to work.
    "magnitude_bucket_pass_rate": ("magnitude_bucket_pass", "binary"),
    "all_pass_rate": ("all_ok", "binary"),
}


def summarize_semantic_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 13,
) -> dict[str, Any]:
    scored_rows = [_with_mode_aware_scores(row) for row in rows]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    groups["overall"].extend(scored_rows)
    for row in scored_rows:
        for key in ("mode", "organ", "primitive", "strength"):
            groups[f"{key}:{row.get(key, '')}"].append(row)
        combo = "|".join(
            str(row.get(key, "")) for key in ("organ", "primitive", "strength", "mode")
        )
        groups[f"cell:{combo}"].append(row)
    return {
        "bootstrap": {
            "iterations": int(bootstrap_iterations),
            "seed": int(seed),
            "preferred_unit": "wsi_id",
        },
        "strength_evaluation_policy": {
            "gt": "strict intended magnitude bucket is part of primary_ok",
            "instruction": "explicit instruction magnitude bucket is part of primary_ok",
            "prompt": (
                "hidden intended bucket agreement is diagnostic only; primary_ok uses "
                "class, direction, and location, with strength evaluated by same-reference "
                "ordinal metrics"
            ),
            "legacy_fields": ["strength_ok", "all_ok", "magnitude_bucket_pass"],
        },
        "groups": {
            name: _summarize_group(
                items,
                bootstrap_iterations=bootstrap_iterations,
                seed=_stable_seed(seed, name),
            )
            for name, items in sorted(groups.items())
        },
        "ordinal": summarize_ordinal_groups(scored_rows),
    }


def summarize_ordinal_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get("ordinal_group_id") or "")
        mode = str(row.get("mode") or "")
        if group_id and str(row.get("status") or "") == "completed":
            grouped[(group_id, mode)].append(row)

    details: list[dict[str, Any]] = []
    for (group_id, mode), items in sorted(grouped.items()):
        by_strength: dict[str, list[float]] = defaultdict(list)
        for row in items:
            strength = str(row.get("strength") or "")
            if strength not in STRENGTH_RANK:
                continue
            value = _optional_float(row.get("measured_area_fraction"))
            if value is not None:
                by_strength[strength].append(value)
        strengths = sorted(by_strength, key=STRENGTH_RANK.get)
        if len(strengths) < 2:
            continue
        ranks = [float(STRENGTH_RANK[item]) for item in strengths]
        values = [float(np.mean(by_strength[item])) for item in strengths]
        rho = _spearman(ranks, values)
        pairwise_concordant = 0
        pairwise_tied = 0
        pairwise_reversed = 0
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                if right > left:
                    pairwise_concordant += 1
                elif right == left:
                    pairwise_tied += 1
                else:
                    pairwise_reversed += 1
        first = items[0]
        details.append(
            {
                "ordinal_group_id": group_id,
                "mode": mode,
                "organ": first.get("organ", ""),
                "primitive": first.get("primitive", ""),
                "strengths": strengths,
                "n_strengths": len(strengths),
                "measured_area_fractions": values,
                "spearman_rho": rho,
                "strictly_monotonic": all(
                    right > left for left, right in zip(values, values[1:])
                ),
                "nondecreasing_monotonic": all(
                    right >= left for left, right in zip(values, values[1:])
                ),
                "pairwise_concordant": pairwise_concordant,
                "pairwise_tied": pairwise_tied,
                "pairwise_reversed": pairwise_reversed,
            }
        )
    summary = _summarize_ordinal_details(details)
    summary["by_mode"] = {
        mode: _summarize_ordinal_details(
            [item for item in details if str(item.get("mode") or "") == mode]
        )
        for mode in sorted({str(item.get("mode") or "") for item in details})
    }
    summary["by_mode_and_n_strengths"] = {
        f"{mode}|n_strengths:{n_strengths}": _summarize_ordinal_details(
            [
                item
                for item in details
                if str(item.get("mode") or "") == mode
                and int(item.get("n_strengths") or 0) == n_strengths
            ]
        )
        for mode, n_strengths in sorted(
            {
                (
                    str(item.get("mode") or ""),
                    int(item.get("n_strengths") or 0),
                )
                for item in details
            }
        )
    }
    summary["groups"] = details
    return summary


def _summarize_ordinal_details(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rhos = [
        float(item["spearman_rho"])
        for item in details
        if item.get("spearman_rho") is not None
    ]
    pairwise_concordant = sum(
        int(item.get("pairwise_concordant") or 0) for item in details
    )
    pairwise_tied = sum(int(item.get("pairwise_tied") or 0) for item in details)
    pairwise_reversed = sum(int(item.get("pairwise_reversed") or 0) for item in details)
    pairwise_total = pairwise_concordant + pairwise_tied + pairwise_reversed
    return {
        "n_groups": len(details),
        "mean_spearman_rho": float(np.mean(rhos)) if rhos else None,
        "median_spearman_rho": float(np.median(rhos)) if rhos else None,
        "strict_monotonicity_rate": (
            sum(bool(item.get("strictly_monotonic")) for item in details) / len(details)
            if details
            else None
        ),
        "nondecreasing_monotonicity_rate": (
            sum(bool(item.get("nondecreasing_monotonic")) for item in details)
            / len(details)
            if details
            else None
        ),
        "pairwise_total": pairwise_total,
        "pairwise_concordance_rate": (
            pairwise_concordant / pairwise_total if pairwise_total else None
        ),
        "pairwise_tie_rate": pairwise_tied / pairwise_total if pairwise_total else None,
        "pairwise_reversal_rate": (
            pairwise_reversed / pairwise_total if pairwise_total else None
        ),
    }


def write_semantic_report(
    report: Mapping[str, Any], output_dir: str | Path
) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "benchmark_semantic_report.json"
    csv_path = output / "benchmark_semantic_report.csv"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for group, payload in report.get("groups", {}).items():
        row = {
            "group": group,
            "n": payload.get("n", 0),
            "completed": payload.get("completed", 0),
            "cluster_unit": payload.get("cluster_unit", ""),
            "n_clusters": payload.get("n_clusters", 0),
            "completion_rate": payload.get("completion_rate", 0.0),
        }
        for metric in FORMAL_METRICS:
            metric_payload = payload.get(metric, {})
            row[metric] = metric_payload.get("value")
            row[f"{metric}_ci_low"] = metric_payload.get("ci_low")
            row[f"{metric}_ci_high"] = metric_payload.get("ci_high")
        rows.append(row)
    fieldnames = (
        list(rows[0])
        if rows
        else ["group", "n", "completed", "cluster_unit", "n_clusters"]
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def ordinal_group_id_for_intent(intent: BenchmarkIntent) -> str:
    if intent.ordinal_group_id:
        return intent.ordinal_group_id
    region = intent.region_hint or {}
    region_key = json.dumps(
        {
            "bbox_xyxy": region.get("bbox_xyxy"),
            "centroid_xy": region.get("centroid_xy"),
            "source_labels": region.get("source_labels"),
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(
        f"{intent.organ}|{intent.primitive}|{intent.mask_path}|{region_key}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"{intent.profile}_{intent.primitive}_{digest}"


def _summarize_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    completed = [row for row in rows if str(row.get("status") or "") == "completed"]
    cluster_unit = (
        "wsi_id"
        if rows and all(str(row.get("wsi_id") or "") for row in rows)
        else "sample_id"
    )
    cluster_ids = {
        str(row.get(cluster_unit) or row.get("sample_id") or index)
        for index, row in enumerate(rows)
    }
    payload: dict[str, Any] = {
        "n": len(rows),
        "completed": len(completed),
        "cluster_unit": cluster_unit,
        "n_clusters": len(cluster_ids),
    }
    for output_name, (field, kind) in FORMAL_METRICS.items():
        metric_rows = list(rows) if kind == "binary" else completed
        clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for index, row in enumerate(metric_rows):
            cluster = str(row.get(cluster_unit) or row.get("sample_id") or index)
            clusters[cluster].append(row)
        values = [_metric_value(row.get(field), kind=kind) for row in metric_rows]
        values = [value for value in values if value is not None]
        value = float(np.mean(values)) if values else None
        low, high = _cluster_bootstrap_ci(
            clusters,
            field=field,
            kind=kind,
            iterations=bootstrap_iterations,
            seed=_stable_seed(seed, field),
        )
        payload[output_name] = {
            "value": value,
            "ci_low": low,
            "ci_high": high,
            "metric_n": len(values),
        }
    payload["completion_rate"] = len(completed) / len(rows) if rows else 0.0
    return payload


def _cluster_bootstrap_ci(
    clusters: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    field: str,
    kind: str,
    iterations: int,
    seed: int,
) -> tuple[float | None, float | None]:
    keys = sorted(clusters)
    if not keys:
        return None, None
    cluster_sums: list[float] = []
    cluster_counts: list[int] = []
    for key in keys:
        values = [
            value
            for value in (
                _metric_value(row.get(field), kind=kind) for row in clusters[key]
            )
            if value is not None
        ]
        cluster_sums.append(float(np.sum(values)))
        cluster_counts.append(len(values))
    sums = np.asarray(cluster_sums, dtype=float)
    counts = np.asarray(cluster_counts, dtype=np.int64)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    remaining = max(1, int(iterations))
    while remaining > 0:
        batch = min(256, remaining)
        sampled = rng.integers(0, len(keys), size=(batch, len(keys)))
        sampled_counts = np.sum(counts[sampled], axis=1)
        sampled_sums = np.sum(sums[sampled], axis=1)
        valid = sampled_counts > 0
        estimates.extend((sampled_sums[valid] / sampled_counts[valid]).tolist())
        remaining -= batch
    if not estimates:
        return None, None
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _metric_value(value: Any, *, kind: str) -> float | None:
    if kind == "binary":
        if isinstance(value, bool):
            return float(value)
        if str(value).lower() in {"true", "1", "yes"}:
            return 1.0
        if value is None or str(value).lower() in {"false", "0", "no", "", "none"}:
            return 0.0
        return None
    return _optional_float(value)


def _with_mode_aware_scores(row: Mapping[str, Any]) -> dict[str, Any]:
    scored = dict(row)
    scored.update(
        mode_aware_score_fields(
            scored,
            mode=str(scored.get("mode") or ""),
        )
    )
    return scored


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order) and values_array[order[end]] == values_array[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha1(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
