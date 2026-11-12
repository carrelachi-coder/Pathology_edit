#!/usr/bin/env python3
"""Analyze the two frozen R3 physician studies without changing eligibility.

The realism study uses the patient as the independent resampling unit and keeps
reader-level repeated scores paired within patient.  The formal order study
scores Sham < Moderate < Significant as its primary ordering target; the
Original-versus-Sham relation is reported separately because Sham is the
pre-specified generated baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from openpyxl import load_workbook


SCORE_COLUMNS = {
    "A": "A分值",
    "B": "B分值",
    "C": "C分值",
    "D": "D分值",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_score_table(path: Path, required_header: str) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        return _read_csv(path)
    if path.suffix.lower() != ".xlsx":
        raise ValueError(f"Unsupported score file format: {path}")
    sheet = load_workbook(path, read_only=True, data_only=True).active
    headers = None
    rows = []
    for row_index, values_tuple in enumerate(sheet.iter_rows(values_only=True), start=1):
        values = list(values_tuple)
        if headers is None:
            if row_index <= 30 and required_header in values:
                headers = [str(value).strip() if value is not None else "" for value in values]
            continue
        if not any(value not in (None, "") for value in values):
            continue
        rows.append(
            {
                header: "" if value is None else str(value).strip()
                for header, value in zip(headers, values)
                if header
            }
        )
    if headers is None:
        raise ValueError(f"Could not find header {required_header!r} in {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def patient_bootstrap_interval(
    patient_values: dict[str, float], *, draws: int, seed: int
) -> list[float] | None:
    if not patient_values:
        return None
    patients = sorted(patient_values)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        selected = [rng.choice(patients) for _ in patients]
        samples.append(statistics.fmean(patient_values[patient] for patient in selected))
    samples.sort()
    return [_quantile(samples, 0.025), _quantile(samples, 0.975)]


def sign_flip_pvalue(
    patient_values: dict[str, float], *, draws: int, seed: int
) -> float | None:
    values = list(patient_values.values())
    if not values:
        return None
    observed = abs(statistics.fmean(values))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        statistic = abs(statistics.fmean(value * rng.choice((-1, 1)) for value in values))
        extreme += statistic >= observed - 1e-15
    return (extreme + 1) / (draws + 1)


def holm_adjust(raw: dict[str, float | None]) -> dict[str, float | None]:
    available = sorted(
        ((key, value) for key, value in raw.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float | None] = {key: None for key in raw}
    running = 0.0
    count = len(available)
    for index, (key, value) in enumerate(available):
        running = max(running, min(1.0, value * (count - index)))
        adjusted[key] = running
    return adjusted


def quadratic_weighted_kappa(pairs: list[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    categories = range(1, 6)
    observed = Counter(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    total = len(pairs)
    denominator = 0.0
    numerator = 0.0
    for a in categories:
        for b in categories:
            weight = ((a - b) / 4) ** 2
            numerator += weight * observed[(a, b)] / total
            denominator += weight * (left[a] * right[b]) / (total * total)
    if denominator == 0:
        return 1.0 if numerator == 0 else None
    return 1 - numerator / denominator


def analyze_realism(
    mapping_path: Path,
    score_paths: list[Path],
    *,
    bootstrap_draws: int,
    permutation_draws: int,
    seed: int,
) -> dict[str, Any]:
    mapping_payload = json.loads(mapping_path.read_text())
    mapping = {row["blind_id"]: row for row in mapping_payload["records"]}
    if len(mapping) != 480:
        raise ValueError(f"Expected 480 mapped images, found {len(mapping)}")
    observations = []
    seen = set()
    for reader_index, path in enumerate(score_paths, start=1):
        rows = _read_score_table(path, "匿名编号")
        file_ids = [row["匿名编号"].strip() for row in rows]
        if len(file_ids) != len(mapping) or set(file_ids) != set(mapping):
            raise ValueError(
                f"{path} must contain each of the {len(mapping)} blind IDs exactly once"
            )
        for row in rows:
            blind_id = row["匿名编号"].strip()
            if blind_id not in mapping:
                raise ValueError(f"Unknown blind ID {blind_id} in {path}")
            reader = row.get("医生", "").strip() or str(reader_index)
            key = (reader, blind_id)
            if key in seen:
                raise ValueError(f"Duplicate reader/image row {key}")
            seen.add(key)
            score_text = row.get("真实性评分", "").strip()
            unable = row.get("无法判断", "").strip().lower() in {"1", "true", "yes", "是"}
            if unable and score_text:
                raise ValueError(f"Both score and unable-to-judge set for {key}")
            score = int(score_text) if score_text else None
            if score is not None and score not in range(1, 6):
                raise ValueError(f"Invalid realism score {score} for {key}")
            observations.append({**mapping[blind_id], "reader": reader, "score": score, "unable": unable})

    summaries = []
    by_model = defaultdict(list)
    for row in observations:
        by_model[row["model_label"]].append(row)
    for model in sorted(by_model):
        rows = by_model[model]
        scores = [row["score"] for row in rows if row["score"] is not None]
        distribution = Counter(scores)
        summaries.append(
            {
                "model": model,
                "rows": len(rows),
                "scored": len(scores),
                "unable": sum(row["unable"] for row in rows),
                "missing": sum(row["score"] is None and not row["unable"] for row in rows),
                "mean": _mean(scores),
                "median": _median(scores),
                "score_counts": {str(score): distribution[score] for score in range(1, 6)},
            }
        )

    scores_by = {
        (row["case"], row["reader"], row["model_label"]): row["score"]
        for row in observations
        if row["score"] is not None
    }
    model_names = sorted(by_model)
    ours = "Ours global recon"
    comparisons = []
    raw_p = {}
    for comparison_index, model in enumerate(model for model in model_names if model != ours):
        patient_differences = {}
        for case in sorted({row["case"] for row in observations}):
            reader_differences = []
            readers = sorted({row["reader"] for row in observations if row["case"] == case})
            for reader in readers:
                ours_score = scores_by.get((case, reader, ours))
                other_score = scores_by.get((case, reader, model))
                if ours_score is not None and other_score is not None:
                    reader_differences.append(ours_score - other_score)
            if reader_differences:
                patient_differences[case] = statistics.fmean(reader_differences)
        p_value = sign_flip_pvalue(
            patient_differences,
            draws=permutation_draws,
            seed=seed + 1000 + comparison_index,
        )
        raw_p[model] = p_value
        comparisons.append(
            {
                "contrast": f"{ours} minus {model}",
                "comparison_model": model,
                "valid_patients": len(patient_differences),
                "mean_paired_difference": _mean(list(patient_differences.values())),
                "patient_bootstrap_95_interval": patient_bootstrap_interval(
                    patient_differences,
                    draws=bootstrap_draws,
                    seed=seed + 2000 + comparison_index,
                ),
                "raw_sign_flip_p_value": p_value,
            }
        )
    adjusted = holm_adjust(raw_p)
    for row in comparisons:
        row["holm_adjusted_p_value"] = adjusted[row["comparison_model"]]

    readers = sorted({row["reader"] for row in observations})
    agreement = None
    if len(readers) == 2:
        reader_scores = {
            (row["reader"], row["blind_id"]): row["score"]
            for row in observations
            if row["score"] is not None
        }
        pairs = [
            (reader_scores[(readers[0], blind_id)], reader_scores[(readers[1], blind_id)])
            for blind_id in mapping
            if (readers[0], blind_id) in reader_scores and (readers[1], blind_id) in reader_scores
        ]
        agreement = {
            "paired_images": len(pairs),
            "quadratic_weighted_kappa": quadratic_weighted_kappa(pairs),
        }
    return {
        "study": "r3_multimodel_realism",
        "independent_unit": "patient_case",
        "mapping_sha256": _sha256(mapping_path),
        "score_files": [{"path": str(path), "sha256": _sha256(path)} for path in score_paths],
        "model_summaries": summaries,
        "ours_paired_comparisons": comparisons,
        "reader_agreement": agreement,
        "doctor_scores_complete": all(row["score"] is not None or row["unable"] for row in observations),
    }


def analyze_order(key_path: Path, score_paths: list[Path]) -> dict[str, Any]:
    key_rows = _read_csv(key_path)
    key = {(row["reader"], row["group_id"], row["display_letter"]): row for row in key_rows}
    groups = []
    for reader_index, path in enumerate(score_paths, start=1):
        reader = str(reader_index)
        score_rows = _read_score_table(path, "组编号")
        expected_groups = {
            group_id
            for key_reader, group_id, _letter in key
            if key_reader == reader
        }
        file_groups = [row["组编号"].strip() for row in score_rows]
        if len(file_groups) != len(expected_groups) or set(file_groups) != expected_groups:
            raise ValueError(
                f"{path} must contain each of the {len(expected_groups)} reader groups exactly once"
            )
        for row in score_rows:
            group_id = row["组编号"].strip()
            unable = row.get("无法判断", "").strip().lower() in {"1", "true", "yes", "是"}
            state_scores = {}
            for letter, column in SCORE_COLUMNS.items():
                mapping = key.get((reader, group_id, letter))
                if mapping is None:
                    raise ValueError(f"No key for reader={reader} group={group_id} image={letter}")
                text = row.get(column, "").strip()
                score = int(text) if text else None
                if score is not None and score not in range(1, 5):
                    raise ValueError(f"Invalid order score {score} in {group_id}/{letter}")
                state_scores[mapping["state"]] = score
            if unable and any(score is not None for score in state_scores.values()):
                raise ValueError(f"Unable-to-judge and scores both set for {group_id}")
            complete = not unable and all(score is not None for score in state_scores.values())
            sham, moderate, significant = (state_scores[name] for name in ("sham", "moderate", "significant"))
            groups.append(
                {
                    "reader": reader,
                    "group_id": group_id,
                    "patient_id": key[(reader, group_id, "A")]["patient_id"],
                    "operation": key[(reader, group_id, "A")]["operation"],
                    "unable": unable,
                    "complete": complete,
                    "strict_edit_order": bool(complete and sham < moderate < significant),
                    "nondecreasing_edit_order": bool(complete and sham <= moderate <= significant),
                    "original_sham_tie": bool(complete and state_scores["original"] == sham),
                    "original_sham_absolute_rank_difference": abs(state_scores["original"] - sham) if complete else None,
                    "scores": state_scores,
                }
            )
    summaries = []
    for operation in ("u1", "u2"):
        rows = [row for row in groups if row["operation"] == operation]
        valid = [row for row in rows if row["complete"]]
        patient_strict = defaultdict(list)
        for row in valid:
            patient_strict[row["patient_id"]].append(float(row["strict_edit_order"]))
        patient_values = {patient: statistics.fmean(values) for patient, values in patient_strict.items()}
        summaries.append(
            {
                "operation": operation,
                "groups_requested": len(rows),
                "groups_complete": len(valid),
                "unable": sum(row["unable"] for row in rows),
                "strict_edit_order": sum(row["strict_edit_order"] for row in valid),
                "strict_edit_order_rate": _mean([float(row["strict_edit_order"]) for row in valid]),
                "nondecreasing_edit_order": sum(row["nondecreasing_edit_order"] for row in valid),
                "original_sham_ties": sum(row["original_sham_tie"] for row in valid),
                "mean_original_sham_absolute_rank_difference": _mean(
                    [row["original_sham_absolute_rank_difference"] for row in valid]
                ),
                "patient_cluster_bootstrap_95_interval_for_strict_rate": patient_bootstrap_interval(
                    patient_values, draws=10000, seed=9417 if operation == "u1" else 9418
                ),
            }
        )
    return {
        "study": "r3_formal_seven_patient_ordering",
        "independent_unit": "patient_id",
        "primary_order_definition": "sham < moderate < significant",
        "original_sham_relation_is_secondary": True,
        "key_sha256": _sha256(key_path),
        "score_files": [{"path": str(path), "sha256": _sha256(path)} for path in score_paths],
        "operation_summaries": summaries,
        "group_records": groups,
        "doctor_scores_complete": all(row["complete"] or row["unable"] for row in groups),
        "qualification_updated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="study", required=True)
    realism = subparsers.add_parser("realism")
    realism.add_argument("--mapping", type=Path, required=True)
    realism.add_argument("--scores", type=Path, nargs="+", required=True)
    realism.add_argument("--bootstrap-draws", type=int, default=10000)
    realism.add_argument("--permutation-draws", type=int, default=100000)
    realism.add_argument("--seed", type=int, default=9417)
    realism.add_argument("--output", type=Path, required=True)
    ordering = subparsers.add_parser("ordering")
    ordering.add_argument("--key", type=Path, required=True)
    ordering.add_argument("--scores", type=Path, nargs="+", required=True)
    ordering.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.study == "realism":
        result = analyze_realism(
            args.mapping,
            args.scores,
            bootstrap_draws=args.bootstrap_draws,
            permutation_draws=args.permutation_draws,
            seed=args.seed,
        )
    else:
        result = analyze_order(args.key, args.scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
