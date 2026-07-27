"""Build structured GT intents for the mask-edit semantic benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

from phase3_mask_edit.benchmark.intents import (
    BuildConfig,
    build_benchmark_intents,
    ordinal_groups_from_intents,
)
from phase3_mask_edit.benchmark.models import write_intents_csv, write_intents_jsonl
from phase3_mask_edit.core.mask_io import save_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="Benchmark YAML config."
    )
    parser.add_argument("--output", type=Path, help="Override config output_dir.")
    parser.add_argument("--patches-per-combo", type=int)
    parser.add_argument("--max-masks-per-profile", type=int)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    config = BuildConfig.from_yaml(args.config)
    overrides = {}
    if args.output is not None:
        overrides["output_dir"] = args.output
    if args.patches_per_combo is not None:
        overrides["patches_per_combo"] = args.patches_per_combo
    if args.max_masks_per_profile is not None:
        overrides["max_masks_per_profile"] = args.max_masks_per_profile
    if overrides:
        config = replace(config, **overrides)
    intents, summary = build_benchmark_intents(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_intents_jsonl(intents, config.output_dir / "benchmark_intents.jsonl")
    write_intents_csv(intents, config.output_dir / "benchmark_intents.csv")
    write_intents_jsonl(intents, config.output_dir / "mask_semantic_intents.jsonl")
    write_intents_csv(intents, config.output_dir / "mask_semantic_intents.csv")
    write_intents_csv(
        [intent for intent in intents if intent.qc_status != "accepted"],
        config.output_dir / "intent_qc.manual_review.csv",
    )
    _write_shortfalls(
        summary.get("shortfalls", []), config.output_dir / "shortfalls.csv"
    )
    _write_jsonl(
        ordinal_groups_from_intents(intents), config.output_dir / "ordinal_groups.jsonl"
    )
    shutil.copy2(args.config, config.output_dir / "source_benchmark_config.yaml")
    save_metadata(
        summary.get("config", {}), config.output_dir / "effective_build_config.json"
    )
    save_metadata(summary, config.output_dir / "build_summary.json")
    if args.print_summary:
        print(
            json.dumps(
                {"num_intents": len(intents), **summary}, indent=2, ensure_ascii=False
            )
        )
    return 0


def _write_shortfalls(rows: list[dict[str, object]], path: Path) -> Path:
    fieldnames = ["organ", "primitive", "strength", "available", "requested"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_jsonl(rows: list[dict[str, object]], path: Path) -> Path:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
