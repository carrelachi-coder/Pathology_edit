"""Data models and serialization helpers for mask-edit semantic benchmarks."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class BenchmarkIntent:
    sample_id: str
    organ: str
    profile: str
    image_path: str | None
    mask_path: str
    primitive: str
    strength: str
    region_hint: dict[str, Any]
    source_labels: tuple[str, ...]
    target_label: str | None
    expected_direction: str
    expected_area_bucket: tuple[float, float] | None
    seed: int
    specialized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BenchmarkIntent":
        bucket = payload.get("expected_area_bucket")
        if bucket is not None:
            if not isinstance(bucket, (list, tuple)) or len(bucket) != 2:
                raise ValueError("expected_area_bucket must be null or a two-item list.")
            bucket = (float(bucket[0]), float(bucket[1]))
        return cls(
            sample_id=str(payload["sample_id"]),
            organ=str(payload["organ"]),
            profile=str(payload["profile"]),
            image_path=_optional_str(payload.get("image_path")),
            mask_path=str(payload["mask_path"]),
            primitive=str(payload["primitive"]),
            strength=str(payload["strength"]),
            region_hint=dict(payload.get("region_hint") or {}),
            source_labels=tuple(str(item) for item in payload.get("source_labels") or ()),
            target_label=_optional_str(payload.get("target_label")),
            expected_direction=str(payload["expected_direction"]),
            expected_area_bucket=bucket,
            seed=int(payload["seed"]),
            specialized=bool(payload.get("specialized", False)),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_labels"] = list(self.source_labels)
        payload["expected_area_bucket"] = (
            list(self.expected_area_bucket) if self.expected_area_bucket is not None else None
        )
        return payload


@dataclass(frozen=True)
class BenchmarkPrompt:
    sample_id: str
    old_prompt: str
    new_prompt: str
    instruction: str
    generator_model: str = "template"
    checker_model: str = "not_checked"
    checker_status: str = "accepted"
    checker_reason: str = "template_generation"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BenchmarkPrompt":
        return cls(
            sample_id=str(payload["sample_id"]),
            old_prompt=str(payload.get("old_prompt") or ""),
            new_prompt=str(payload.get("new_prompt") or ""),
            instruction=str(payload.get("instruction") or ""),
            generator_model=str(payload.get("generator_model") or ""),
            checker_model=str(payload.get("checker_model") or ""),
            checker_status=str(payload.get("checker_status") or ""),
            checker_reason=str(payload.get("checker_reason") or ""),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def read_intents_jsonl(path: str | Path) -> list[BenchmarkIntent]:
    items: list[BenchmarkIntent] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                items.append(BenchmarkIntent.from_mapping(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"Invalid benchmark intent at line {line_number}: {exc}") from exc
    return items


def write_intents_jsonl(intents: Iterable[BenchmarkIntent], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for intent in intents:
            stream.write(json.dumps(intent.to_mapping(), ensure_ascii=False, sort_keys=True) + "\n")
    return output_path


def write_intents_csv(intents: Iterable[BenchmarkIntent], path: str | Path) -> Path:
    rows = [intent.to_mapping() for intent in intents]
    fieldnames = [
        "sample_id",
        "organ",
        "profile",
        "image_path",
        "mask_path",
        "primitive",
        "strength",
        "region_hint",
        "source_labels",
        "target_label",
        "expected_direction",
        "expected_area_bucket",
        "seed",
        "specialized",
        "metadata",
    ]
    return _write_csv_rows(rows, fieldnames, path)


def read_prompts_csv(path: str | Path) -> dict[str, BenchmarkPrompt]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return {
            str(row["sample_id"]): BenchmarkPrompt.from_mapping(row)
            for row in reader
            if row.get("sample_id")
        }


def write_prompts_csv(prompts: Iterable[BenchmarkPrompt], path: str | Path) -> Path:
    rows = [prompt.to_mapping() for prompt in prompts]
    fieldnames = [
        "sample_id",
        "old_prompt",
        "new_prompt",
        "instruction",
        "generator_model",
        "checker_model",
        "checker_status",
        "checker_reason",
    ]
    return _write_csv_rows(rows, fieldnames, path)


def write_eval_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    materialized = [dict(row) for row in rows]
    fieldnames = [
        "sample_id",
        "organ",
        "profile",
        "primitive",
        "strength",
        "mode",
        "status",
        "parsed_semantic_diff",
        "planned_primitive",
        "measured_class_delta",
        "measured_area_fraction",
        "measured_location",
        "class_ok",
        "direction_ok",
        "strength_ok",
        "location_ok",
        "all_ok",
        "error",
        "output_dir",
    ]
    return _write_csv_rows(materialized, fieldnames, path)


def _write_csv_rows(rows: list[Mapping[str, Any]], fieldnames: list[str], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return output_path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False)
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
