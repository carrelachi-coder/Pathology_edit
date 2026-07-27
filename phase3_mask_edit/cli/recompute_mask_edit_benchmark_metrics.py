"""Recompute formal mask-edit metrics from saved source and target masks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase3_mask_edit.benchmark.metrics import evaluate_mask_edit
from phase3_mask_edit.benchmark.intents import infer_patient_id, infer_wsi_id
from phase3_mask_edit.benchmark.models import read_intents_jsonl, write_eval_csv
from phase3_mask_edit.benchmark.reporting import (
    ordinal_group_id_for_intent,
    summarize_semantic_rows,
    write_semantic_report,
)
from phase3_mask_edit.core.mask_io import load_id_mask, save_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", required=True, type=Path)
    parser.add_argument("--eval-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--path-root",
        type=Path,
        default=Path.cwd(),
        help="Root used to resolve relative output_dir values in the evaluation CSV.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    intents = {item.sample_id: item for item in read_intents_jsonl(args.intents)}
    with args.eval_results.open("r", encoding="utf-8", newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]

    args.output.mkdir(parents=True, exist_ok=True)
    missing: list[dict[str, str]] = []
    target_bank: list[dict[str, Any]] = []
    recomputed = 0
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        intent = intents.get(sample_id)
        if intent is None:
            missing.append({"sample_id": sample_id, "reason": "intent_not_found"})
            continue
        ordinal_group_id = ordinal_group_id_for_intent(intent)
        wsi_id = intent.wsi_id or infer_wsi_id(intent.mask_path)
        patient_id = intent.patient_id or infer_patient_id(
            intent.mask_path, wsi_id=wsi_id
        )
        row.update(
            {
                "source_dataset": intent.source_dataset or intent.profile,
                "wsi_id": wsi_id,
                "patient_id": patient_id,
                "ordinal_group_id": ordinal_group_id,
            }
        )
        if str(row.get("status") or "") != "completed":
            continue
        sample_dir = _resolve_path(
            str(row.get("output_dir") or ""), root=args.path_root
        )
        source_path = sample_dir / "source_mask.png"
        target_path = sample_dir / "target_mask.png"
        if not source_path.is_file() or not target_path.is_file():
            missing.append(
                {
                    "sample_id": sample_id,
                    "reason": "saved_mask_missing",
                    "output_dir": str(sample_dir),
                }
            )
            continue
        metrics = evaluate_mask_edit(
            load_id_mask(source_path), load_id_mask(target_path), intent
        )
        _update_eval_row(row, metrics)
        recomputed += 1
        target_bank.append(
            {
                "sample_id": sample_id,
                "mode": row.get("mode", ""),
                "organ": intent.organ,
                "profile": intent.profile,
                "primitive": intent.primitive,
                "strength": intent.strength,
                "source_dataset": intent.source_dataset or intent.profile,
                "wsi_id": wsi_id,
                "patient_id": patient_id,
                "ordinal_group_id": ordinal_group_id,
                "source_mask_path": str(source_path.resolve()),
                "target_mask_path": str(target_path.resolve()),
                "change_region_path": str((sample_dir / "change_region.png").resolve()),
                "metrics_path": str((sample_dir / "metrics.json").resolve()),
            }
        )

    enriched_path = write_eval_csv(
        rows, args.output / "benchmark_eval_results.enriched.csv"
    )
    report = summarize_semantic_rows(
        rows,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    report_paths = write_semantic_report(report, args.output)
    target_paths = _write_target_bank(target_bank, args.output)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "intents_path": str(args.intents.resolve()),
        "intents_sha256": _sha256(args.intents),
        "eval_results_path": str(args.eval_results.resolve()),
        "eval_results_sha256": _sha256(args.eval_results),
        "total_rows": len(rows),
        "recomputed_rows": recomputed,
        "missing_count": len(missing),
        "missing": missing,
        "outputs": [
            str(enriched_path),
            *(str(item) for item in report_paths),
            *(str(item) for item in target_paths),
        ],
    }
    save_metadata(summary, args.output / "recompute_summary.json")
    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not missing else 2


def _update_eval_row(row: dict[str, Any], metrics: dict[str, Any]) -> None:
    json_fields = {"measured_class_delta", "measured_location"}
    for key, value in metrics.items():
        row[key] = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if key in json_fields
            else value
        )


def _write_target_bank(rows: list[dict[str, Any]], output: Path) -> tuple[Path, Path]:
    csv_path = output / "target_mask_bank.csv"
    jsonl_path = output / "target_mask_bank.jsonl"
    fieldnames = list(rows[0]) if rows else ["sample_id", "mode", "target_mask_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return csv_path, jsonl_path


def _resolve_path(value: str, *, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
