#!/usr/bin/env python3
"""Run portable Phase 3 real-mask smoke tests.

This script is meant to be copied/run on the machine that has
`edit_datasets/`.  On machines without real data it exits cleanly and writes
summary files with `no_samples` rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phase3_real_smoke_utils import (
    DEFAULT_DATA_ROOT,
    load_default_recipe,
    parse_profiles,
    parse_strengths,
    run_primitive_case,
    safe_case_name,
    save_case_artifacts,
    select_samples,
    write_summary_files,
)


DEFAULT_PRIMITIVES = [
    "tumor_burden_increase",
    "tumor_burden_decrease",
    "necrosis_appearance",
    "stromal_immune_infiltration",
]
DEFAULT_OUTPUT = Path("phase3_mask_edit/previews/real_mask_smoke")


def main() -> None:
    args = _parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    recipe = load_default_recipe(Path(args.recipe))
    rows = []

    profiles = parse_profiles(args.profiles)
    strengths = parse_strengths(args.strengths)
    primitives = args.primitives or DEFAULT_PRIMITIVES

    for primitive in primitives:
        primitive_dir = output_dir / primitive
        for profile in profiles:
            samples = select_samples(
                data_root=data_root,
                profile=profile,
                primitive=primitive,
                limit=args.limit_per_profile,
                include_rejected=args.include_rejected,
            )
            if not samples:
                rows.append({
                    "primitive": primitive,
                    "profile": profile,
                    "status": "no_samples",
                    "failure_reason": f"no usable masks under {data_root / profile}",
                })
                continue

            for sample_index, sample in enumerate(samples):
                if sample.get("load_error"):
                    rows.append({
                        "primitive": primitive,
                        "profile": profile,
                        "mask_path": sample.get("mask_path"),
                        "status": "load_failed",
                        "failure_reason": sample["load_error"],
                    })
                    continue

                for strength in strengths:
                    seed = args.seed + sample_index
                    metadata = run_primitive_case(
                        old_mask=sample["old_mask"],
                        schema=sample["schema"],
                        context=sample["context"],
                        recipe=recipe,
                        primitive=primitive,
                        strength=strength,
                        seed=seed,
                    )
                    case_name = safe_case_name(
                        profile,
                        primitive,
                        strength,
                        f"sample{sample_index:03d}",
                        Path(sample["mask_path"]).stem,
                    )
                    artifacts = save_case_artifacts(
                        output_dir=primitive_dir,
                        case_name=case_name,
                        old_mask=sample["old_mask"],
                        target_mask=metadata["target_mask"],
                        change_region=metadata["change_region"],
                        metadata={
                            **metadata,
                            "mask_path": str(sample["mask_path"]),
                            "sample_score": sample.get("sample_score"),
                        },
                    )
                    rows.append({
                        **metadata,
                        "mask_path": str(sample["mask_path"]),
                        "sample_score": sample.get("sample_score"),
                        "panel": artifacts["panel"],
                        "metadata_path": artifacts["metadata"],
                    })

    write_summary_files(rows, output_dir)
    print(
        f"Wrote {len(rows)} Phase3 smoke rows for "
        f"{len(primitives)} primitives to {output_dir}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run portable Phase3 real-mask smoke tests."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=["BCSS", "IGNITE", "PUMA", "PANDA", "ORCA", "GlaS"],
    )
    parser.add_argument("--primitives", nargs="*", default=DEFAULT_PRIMITIVES)
    parser.add_argument("--strengths", nargs="*", default=["mild", "moderate", "significant"])
    parser.add_argument("--limit-per-profile", type=int, default=10)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--recipe", default="phase3_mask_edit/recipes/generic.yaml")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include low-priority/rejected samples in the summary for debugging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

