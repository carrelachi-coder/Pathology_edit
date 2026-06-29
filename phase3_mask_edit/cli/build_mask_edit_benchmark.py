"""Build structured GT intents for the mask-edit semantic benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_mask_edit.benchmark.intents import BuildConfig, build_benchmark_intents
from phase3_mask_edit.benchmark.models import write_intents_csv, write_intents_jsonl
from phase3_mask_edit.core.mask_io import save_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Benchmark YAML config.")
    parser.add_argument("--output", type=Path, help="Override config output_dir.")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    config = BuildConfig.from_yaml(args.config)
    if args.output is not None:
        config = BuildConfig(
            data_root=config.data_root,
            output_dir=args.output,
            profiles=config.profiles,
            patches_per_combo=config.patches_per_combo,
            strengths=config.strengths,
            allowed_primitives=config.allowed_primitives,
            excluded_primitives=config.excluded_primitives,
            seed=config.seed,
            max_masks_per_profile=config.max_masks_per_profile,
        )
    intents, summary = build_benchmark_intents(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_intents_jsonl(intents, config.output_dir / "benchmark_intents.jsonl")
    write_intents_csv(intents, config.output_dir / "benchmark_intents.csv")
    save_metadata(summary, config.output_dir / "build_summary.json")
    if args.print_summary:
        print(json.dumps({"num_intents": len(intents), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
