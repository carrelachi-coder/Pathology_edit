"""Blind model/candidate evaluation metrics and release gates."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phase3_mask_edit_refine.models import RefineContractError


@dataclass(frozen=True)
class EvaluationRecord:
    case_id: str
    model_config: str
    pathology_domain_id: str
    annotation_profile_id: str
    primitive_id: str
    schema_valid: bool
    predicted_interface_ids: tuple[str, ...]
    legal_interface_ids: tuple[str, ...]
    hard_violation_count: int
    changed_area_passed: bool
    unrequested_label_violation_pixels: int
    expert_morphology_accepted: bool | None
    abstained: bool
    cost_usd: float
    latency_sec: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> EvaluationRecord:
        required_strings = {
            key: _required_string(payload, key)
            for key in (
                "case_id",
                "model_config",
                "pathology_domain_id",
                "annotation_profile_id",
                "primitive_id",
            )
        }
        expert = payload.get("expert_morphology_accepted")
        if expert is not None and not isinstance(expert, bool):
            raise RefineContractError("expert_morphology_accepted must be bool or null")
        return cls(
            **required_strings,
            schema_valid=_required_bool(payload, "schema_valid"),
            predicted_interface_ids=_strings(payload.get("predicted_interface_ids")),
            legal_interface_ids=_strings(payload.get("legal_interface_ids")),
            hard_violation_count=_nonnegative_int(payload, "hard_violation_count"),
            changed_area_passed=_required_bool(payload, "changed_area_passed"),
            unrequested_label_violation_pixels=_nonnegative_int(
                payload, "unrequested_label_violation_pixels"
            ),
            expert_morphology_accepted=expert,
            abstained=_required_bool(payload, "abstained"),
            cost_usd=_nonnegative_number(payload, "cost_usd"),
            latency_sec=_nonnegative_number(payload, "latency_sec"),
        )


def load_evaluation_jsonl(path: str | Path) -> tuple[EvaluationRecord, ...]:
    records: list[EvaluationRecord] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RefineContractError(f"invalid JSONL line {line_number}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise RefineContractError(f"evaluation line {line_number} must be an object")
        records.append(EvaluationRecord.from_mapping(payload))
    if not records:
        raise RefineContractError("evaluation JSONL contains no records")
    return tuple(records)


def score_evaluation(records: Iterable[EvaluationRecord]) -> dict[str, Any]:
    values = tuple(records)
    if not values:
        raise RefineContractError("cannot score an empty evaluation")
    models: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in values:
        models[record.model_config].append(record)
    return {
        "schema_version": "mask-edit-refine-evaluation-v1",
        "models": {
            model: _score_model(tuple(records_for_model))
            for model, records_for_model in sorted(models.items())
        },
    }


def _score_model(records: tuple[EvaluationRecord, ...]) -> dict[str, Any]:
    strata: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        strata[f"{record.pathology_domain_id}|{record.annotation_profile_id}"].append(record)
    overall = _metrics(records)
    by_stratum = {key: _metrics(tuple(items)) for key, items in sorted(strata.items())}
    expert_strata = [
        value["expert_morphology_accept_rate"]
        for value in by_stratum.values()
        if value["expert_reviewed_count"] > 0
    ]
    release_checks = {
        "schema_valid_rate_100pct": overall["schema_valid_rate"] == 1.0,
        "top3_interface_recall_ge_95pct": overall["top3_interface_recall"] >= 0.95,
        "each_stratum_top3_ge_90pct": all(
            value["top3_interface_recall"] >= 0.90 for value in by_stratum.values()
        ),
        "zero_hard_violations": overall["hard_violation_count"] == 0,
        "zero_unrequested_label_violations": overall[
            "unrequested_label_violation_pixels"
        ]
        == 0,
        "changed_area_pass_rate_ge_99_5pct": overall["changed_area_pass_rate"] >= 0.995,
        "expert_accept_rate_ge_95pct": (
            overall["expert_reviewed_count"] > 0
            and overall["expert_morphology_accept_rate"] >= 0.95
        ),
        "each_reviewed_stratum_expert_accept_ge_90pct": bool(expert_strata)
        and all(value >= 0.90 for value in expert_strata),
        "average_cost_below_0_15_usd": overall["average_cost_usd"] < 0.15,
    }
    return {
        "overall": overall,
        "by_domain_profile": by_stratum,
        "release_checks": release_checks,
        "release_passed": all(release_checks.values()),
    }


def _metrics(records: tuple[EvaluationRecord, ...]) -> dict[str, Any]:
    count = len(records)
    expert_records = [
        record for record in records if record.expert_morphology_accepted is not None
    ]
    return {
        "case_count": count,
        "schema_valid_rate": float(np.mean([record.schema_valid for record in records])),
        "top1_interface_recall": float(np.mean([_topk_hit(record, 1) for record in records])),
        "top3_interface_recall": float(np.mean([_topk_hit(record, 3) for record in records])),
        "hard_violation_count": int(sum(record.hard_violation_count for record in records)),
        "changed_area_pass_rate": float(
            np.mean([record.changed_area_passed for record in records])
        ),
        "unrequested_label_violation_pixels": int(
            sum(record.unrequested_label_violation_pixels for record in records)
        ),
        "expert_reviewed_count": len(expert_records),
        "expert_morphology_accept_rate": float(
            np.mean([record.expert_morphology_accepted for record in expert_records])
        )
        if expert_records
        else 0.0,
        "abstain_rate": float(np.mean([record.abstained for record in records])),
        "average_cost_usd": float(np.mean([record.cost_usd for record in records])),
        "p95_latency_sec": float(np.percentile([record.latency_sec for record in records], 95)),
    }


def _topk_hit(record: EvaluationRecord, k: int) -> bool:
    legal = set(record.legal_interface_ids)
    return bool(legal.intersection(record.predicted_interface_ids[:k]))


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RefineContractError(f"{key} must be a non-empty string")
    return value


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise RefineContractError(f"{key} must be boolean")
    return value


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RefineContractError("interface IDs must be lists of strings")
    return tuple(value)


def _nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value < 0:
        raise RefineContractError(f"{key} must be a non-negative integer")
    return value


def _nonnegative_number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or float(value) < 0:
        raise RefineContractError(f"{key} must be a non-negative number")
    return float(value)
