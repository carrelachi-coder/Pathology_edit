#!/usr/bin/env python3
"""Analyze the frozen formal R3 sham/Moderate/Significant paired endpoint series."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


VALID = {"valid_nonzero", "valid_zero_with_completion"}
STATES = (
    "original",
    "sham",
    "u1_moderate",
    "u1_significant",
    "u2_moderate",
    "u2_significant",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def stable_seed(seed: int, label: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    return (seed + suffix) % (2**63 - 1)


def percentile_interval(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    sampled = values[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    return [float(value) for value in np.percentile(sampled, [2.5, 97.5])]


def paired_sign_permutation(values: np.ndarray, seed: int, draws: int) -> float:
    observed = abs(float(values.mean()))
    if observed == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = draws
    while remaining:
        count = min(remaining, 4096)
        signs = rng.integers(0, 2, size=(count, len(values)), dtype=np.int8) * 2 - 1
        null = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(null >= observed))
        remaining -= count
    return float((extreme + 1) / (draws + 1))


def holm_adjust(rows: list[dict], alpha: float = 0.05) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["raw_p_value"])
    running = 0.0
    m = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * row["raw_p_value"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p_value"] = running
        rows[index]["holm_reject_alpha_0_05"] = running <= alpha


def readout(record: dict, endpoint: str) -> tuple[str, float | None]:
    if endpoint in {"u1", "u2"}:
        item = record.get("tissue", {}).get(endpoint, {})
    elif endpoint == "secondary_inflammatory_density":
        item = record.get("inflammatory_density", {})
    else:
        raise ValueError(f"unknown endpoint: {endpoint}")
    status = str(item.get("status", record.get("cell_status", "missing")))
    value = item.get("value")
    if status in VALID:
        if value is None or not np.isfinite(float(value)):
            raise ValueError("valid measurement lacks a finite value")
        return status, float(value)
    return status, None


def validate_report(report: dict) -> tuple[list[str], dict[str, dict[str, dict]]]:
    if report.get("scope") != "formal_r3" or report.get("formal_launch_permitted") is not True:
        raise ValueError("input is not a formal R3 measurement report")
    if report.get("status") != "complete" or report.get("chain_integrity_accepted") is not True:
        raise ValueError("formal measurement chain is incomplete")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("formal report contains no measurement records")
    grouped: dict[str, dict[str, dict]] = {}
    references: dict[str, str] = {}
    for row in records:
        patient = str(row.get("patient_id", ""))
        reference = str(row.get("reference_id", ""))
        state = str(row.get("state", ""))
        if not patient or not reference or state not in STATES:
            raise ValueError("measurement identity is incomplete")
        if patient in references and references[patient] != reference:
            raise ValueError("formal R3 permits only one source per patient")
        references[patient] = reference
        patient_rows = grouped.setdefault(patient, {})
        if state in patient_rows:
            raise ValueError("patient state is duplicated")
        patient_rows[state] = row
    for patient, rows in grouped.items():
        if set(rows) != set(STATES):
            raise ValueError(f"patient lacks the complete six-state series: {patient}")
    return sorted(grouped), grouped


def summarize_comparison(
    patients: list[str],
    grouped: dict[str, dict[str, dict]],
    *,
    endpoint: str,
    edit_state: str,
    seed: int,
    bootstrap_draws: int,
    permutation_draws: int,
) -> dict:
    deltas = []
    excluded: dict[str, int] = {}
    for patient in patients:
        sham_status, sham = readout(grouped[patient]["sham"], endpoint)
        edit_status, edit = readout(grouped[patient][edit_state], endpoint)
        if sham is None or edit is None:
            key = f"sham:{sham_status}|edit:{edit_status}"
            excluded[key] = excluded.get(key, 0) + 1
            continue
        deltas.append(edit - sham)
    if not deltas:
        raise ValueError(f"no valid pairs for {endpoint} {edit_state}")
    values = np.asarray(deltas, dtype=float)
    label = endpoint + "__" + edit_state
    return {
        "endpoint": endpoint,
        "edit_state": edit_state,
        "requested_patients": len(patients),
        "valid_pairs": len(values),
        "excluded_pairs_by_status": excluded,
        "mean_edit_minus_sham": float(values.mean()),
        "median_edit_minus_sham": float(np.median(values)),
        "patient_bootstrap_95_interval": percentile_interval(
            values, stable_seed(seed, "bootstrap__" + label), bootstrap_draws
        ),
        "raw_p_value": paired_sign_permutation(
            values, stable_seed(seed, "permutation__" + label), permutation_draws
        ),
        "positive_delta": int(np.count_nonzero(values > 0)),
        "zero_delta": int(np.count_nonzero(values == 0)),
        "negative_delta": int(np.count_nonzero(values < 0)),
    }


def summarize_order(
    patients: list[str], grouped: dict[str, dict[str, dict]], endpoint: str
) -> dict:
    moderate = endpoint + "_moderate"
    significant = endpoint + "_significant"
    counts = {"strict_order": 0, "adjacent_tie": 0, "reversal": 0, "incomplete": 0}
    for patient in patients:
        values = [readout(grouped[patient][state], endpoint)[1] for state in ("sham", moderate, significant)]
        if any(value is None for value in values):
            counts["incomplete"] += 1
        elif values[0] < values[1] < values[2]:
            counts["strict_order"] += 1
        elif values[0] == values[1] or values[1] == values[2]:
            counts["adjacent_tie"] += 1
        else:
            counts["reversal"] += 1
    requested = len(patients)
    valid = requested - counts["incomplete"]
    return {
        "endpoint": endpoint,
        "requested_patients": requested,
        "valid_triplets": valid,
        **counts,
        "strict_order_rate_among_valid": counts["strict_order"] / valid if valid else None,
        "strict_order_all_requested_lower_bound": counts["strict_order"] / requested,
        "strict_order_all_requested_upper_bound": (counts["strict_order"] + counts["incomplete"]) / requested,
    }


def summarize_sham_drift(
    patients: list[str],
    grouped: dict[str, dict[str, dict]],
    endpoint: str,
    *,
    seed: int,
    bootstrap_draws: int,
) -> dict:
    values = []
    for patient in patients:
        original = readout(grouped[patient]["original"], endpoint)[1]
        sham = readout(grouped[patient]["sham"], endpoint)[1]
        if original is not None and sham is not None:
            values.append(sham - original)
    array = np.asarray(values, dtype=float)
    return {
        "endpoint": endpoint,
        "requested_patients": len(patients),
        "valid_pairs": len(array),
        "mean_sham_minus_original": float(array.mean()) if len(array) else None,
        "median_sham_minus_original": float(np.median(array)) if len(array) else None,
        "patient_bootstrap_95_interval": (
            percentile_interval(
                array, stable_seed(seed, "sham_drift__" + endpoint), bootstrap_draws
            )
            if len(array)
            else None
        ),
    }


def analyze(report: dict, *, seed: int, bootstrap_draws: int, permutation_draws: int) -> dict:
    patients, grouped = validate_report(report)
    primary = [
        summarize_comparison(
            patients,
            grouped,
            endpoint=endpoint,
            edit_state=f"{endpoint}_{strength}",
            seed=seed,
            bootstrap_draws=bootstrap_draws,
            permutation_draws=permutation_draws,
        )
        for endpoint in ("u1", "u2")
        for strength in ("moderate", "significant")
    ]
    holm_adjust(primary)
    secondary = [
        summarize_comparison(
            patients,
            grouped,
            endpoint="secondary_inflammatory_density",
            edit_state=f"u2_{strength}",
            seed=seed,
            bootstrap_draws=bootstrap_draws,
            permutation_draws=permutation_draws,
        )
        for strength in ("moderate", "significant")
    ]
    return {
        "schema_version": "formal-r3-paired-analysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "independent_unit": "patient_id",
        "seed": seed,
        "bootstrap_draws": bootstrap_draws,
        "permutation_draws": permutation_draws,
        "requested_patients": len(patients),
        "patient_ids": patients,
        "primary_holm_family": primary,
        "order_outcomes": [summarize_order(patients, grouped, endpoint) for endpoint in ("u1", "u2")],
        "sham_drift": [
            summarize_sham_drift(
                patients,
                grouped,
                endpoint,
                seed=seed,
                bootstrap_draws=bootstrap_draws,
            )
            for endpoint in ("u1", "u2")
        ],
        "secondary_inflammatory_density": secondary,
        "qualification_updated": False,
        "physician_claim_created": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--permutation-draws", type=int, default=20000)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; formal analyses are append-only")
    if args.bootstrap_draws <= 0 or args.permutation_draws <= 0:
        raise ValueError("analysis draw counts must be positive")
    report = json.loads(args.measurement_report.read_text())
    result = analyze(
        report,
        seed=args.seed,
        bootstrap_draws=args.bootstrap_draws,
        permutation_draws=args.permutation_draws,
    )
    result["measurement_report"] = identity(args.measurement_report)
    args.output.mkdir(parents=True)
    (args.output / "formal_r3_analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"requested_patients": result["requested_patients"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
