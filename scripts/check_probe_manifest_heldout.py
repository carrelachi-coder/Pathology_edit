"""Check whether generation probes are held out from training metadata.

The comparison is done at source-slide/case level, not just patch level.  A
probe is considered overlapping when any target, paired-reference, or alternate
reference source id appears as a target or reference source id in the training
metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PATCH_RE = re.compile(r"^(?P<case_id>.+)_py(?P<py>\d+)_px(?P<px>\d+)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check probe source-slide overlap with training metadata.")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument(
        "--train-metadata",
        action="append",
        required=True,
        help="Training metadata JSON. May be repeated.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--ignore-dataset",
        action="store_true",
        help="Compare case ids without dataset prefixes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = json.loads(Path(args.selection_manifest).read_text(encoding="utf8"))
    train_records = []
    for path in args.train_metadata:
        train_records.extend(read_metadata(Path(path)))

    train_sources = collect_train_sources(
        train_records,
        ignore_dataset=bool(args.ignore_dataset),
    )
    probe_rows = build_probe_rows(
        manifest,
        train_sources=train_sources,
        ignore_dataset=bool(args.ignore_dataset),
    )
    summary = summarize(probe_rows, train_sources=train_sources)
    summary.update(
        {
            "selection_manifest": str(args.selection_manifest),
            "train_metadata": [str(path) for path in args.train_metadata],
            "ignore_dataset": bool(args.ignore_dataset),
        }
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    if args.output_csv:
        write_csv(Path(args.output_csv), probe_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def read_metadata(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if isinstance(payload, dict):
        records = payload.get("pairs")
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a 'pairs' list")
        return records
    if isinstance(payload, list):
        return payload
    raise TypeError(f"unsupported metadata payload in {path}: {type(payload)!r}")


def collect_train_sources(
    records: list[dict[str, Any]],
    *,
    ignore_dataset: bool,
) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        for role, source_id in record_source_ids(record, ignore_dataset=ignore_dataset).items():
            entry = sources.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "count": 0,
                    "target_count": 0,
                    "reference_count": 0,
                    "example_metadata_index": index,
                    "example_sample_id": str(record.get("sample_id") or ""),
                    "example_reference_sample_id": str(record.get("reference_sample_id") or ""),
                    "roles": set(),
                },
            )
            entry["count"] += 1
            entry["roles"].add(role)
            if role == "target":
                entry["target_count"] += 1
            elif role == "reference":
                entry["reference_count"] += 1
    for entry in sources.values():
        entry["roles"] = sorted(entry["roles"])
    return sources


def build_probe_rows(
    manifest: list[dict[str, Any]],
    *,
    train_sources: dict[str, dict[str, Any]],
    ignore_dataset: bool,
) -> list[dict[str, Any]]:
    rows = []
    for probe_index, item in enumerate(manifest):
        dataset = "" if ignore_dataset else str(item.get("dataset") or "")
        probe_sources = {
            "target": source_key(
                dataset=dataset,
                explicit=item.get("target_case_id") or item.get("target_wsi_id"),
                sample_id=item.get("sample_id"),
                image_path=item.get("target_image"),
                ignore_dataset=ignore_dataset,
            ),
            "paired_reference": source_key(
                dataset=dataset,
                explicit=item.get("paired_reference_case_id") or item.get("paired_reference_wsi_id"),
                sample_id=item.get("paired_reference_sample_id"),
                image_path=item.get("paired_reference_image"),
                ignore_dataset=ignore_dataset,
            ),
        }
        for mode, payload in sorted((item.get("alternates") or {}).items()):
            probe_sources[f"alternate_{mode}"] = source_key(
                dataset="" if ignore_dataset else str(payload.get("dataset") or item.get("dataset") or ""),
                explicit=payload.get("reference_case_id") or payload.get("reference_wsi_id"),
                sample_id=payload.get("reference_sample_id"),
                image_path=payload.get("reference_image"),
                ignore_dataset=ignore_dataset,
            )
        overlaps = {
            role: train_sources.get(source_id)
            for role, source_id in probe_sources.items()
            if source_id and source_id in train_sources
        }
        rows.append(
            {
                "probe_index": probe_index,
                "sample_id": str(item.get("sample_id") or ""),
                "dataset": str(item.get("dataset") or ""),
                "target_source_id": probe_sources.get("target", ""),
                "paired_reference_source_id": probe_sources.get("paired_reference", ""),
                "alternate_source_ids": ";".join(
                    f"{role}={source_id}"
                    for role, source_id in sorted(probe_sources.items())
                    if role.startswith("alternate_")
                ),
                "overlap": bool(overlaps),
                "overlap_roles": ",".join(sorted(overlaps)),
                "overlap_source_ids": ",".join(sorted({source_id for role, source_id in probe_sources.items() if role in overlaps})),
                "overlap_train_counts": ",".join(
                    f"{role}:{overlaps[role]['count']}" for role in sorted(overlaps)
                ),
            }
        )
    return rows


def record_source_ids(record: dict[str, Any], *, ignore_dataset: bool) -> dict[str, str]:
    dataset = "" if ignore_dataset else str(record.get("dataset") or "")
    return {
        "target": source_key(
            dataset=dataset,
            explicit=first_present(record, ("target_case_id", "case_id", "target_wsi_id", "slide_id")),
            sample_id=record.get("sample_id"),
            image_path=record.get("target_image"),
            ignore_dataset=ignore_dataset,
        ),
        "reference": source_key(
            dataset=dataset,
            explicit=first_present(record, ("reference_case_id", "reference_wsi_id", "reference_slide_id")),
            sample_id=record.get("reference_sample_id"),
            image_path=record.get("reference_image"),
            ignore_dataset=ignore_dataset,
        ),
    }


def source_key(
    *,
    dataset: str,
    explicit: Any,
    sample_id: Any,
    image_path: Any,
    ignore_dataset: bool,
) -> str:
    case_id = str(explicit or "").strip()
    if not case_id:
        sample = str(sample_id or "").strip()
        if not sample and image_path:
            sample = Path(str(image_path)).stem
        case_id = parse_case_id(sample)
    if ignore_dataset:
        return case_id
    return f"{dataset}::{case_id}" if dataset else case_id


def parse_case_id(sample_id: str) -> str:
    match = PATCH_RE.match(str(sample_id))
    if match:
        return match.group("case_id")
    return str(sample_id)


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def summarize(
    probe_rows: list[dict[str, Any]],
    *,
    train_sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    overlap_rows = [row for row in probe_rows if row["overlap"]]
    role_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    for row in overlap_rows:
        for role in str(row["overlap_roles"]).split(","):
            if role:
                role_counter[role] += 1
        for source_id in str(row["overlap_source_ids"]).split(","):
            if source_id:
                source_counter[source_id] += 1
    return {
        "status": "FAIL" if overlap_rows else "PASS",
        "num_probes": len(probe_rows),
        "overlap_probes": len(overlap_rows),
        "overlap_rate": len(overlap_rows) / len(probe_rows) if probe_rows else None,
        "train_source_count": len(train_sources),
        "overlap_by_probe_role": dict(sorted(role_counter.items())),
        "top_overlap_sources": source_counter.most_common(20),
        "overlap_preview": overlap_rows[:50],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
