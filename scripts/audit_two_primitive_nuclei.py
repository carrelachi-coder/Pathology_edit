#!/usr/bin/env python3
"""Audit the frozen 1,200-target nuclei bank for the paired U1/U2 study."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from inpaint_cells.utils.mask_utils import load_nuclei_mask, load_tissue_mask
from scripts.prepare_embedding_utility_nuclei import _retain_complete_reference_cells


EXPECTED_CHECKPOINT_SHA256 = (
    "c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--minimum-placement-completion", type=float, default=0.98)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    failures: list[str] = []
    if len(rows) != args.expected_count:
        failures.append(f"manifest count {len(rows)} != {args.expected_count}")
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    if not all(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        failures.append("sample IDs must be non-empty and unique")

    group_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "rows": 0,
            "requested": 0,
            "placed": 0,
            "unfilled": 0,
            "reference_source_components": 0,
            "reference_kept_components": 0,
        }
    )
    checked = 0
    for row in rows:
        sample_id = str(row.get("sample_id") or "unknown")
        group = f"{row.get('primitive')}/{row.get('strength')}"
        try:
            nuclei_path = Path(row["target_nuclei_mask"])
            metadata_path = Path(row["target_nuclei_metadata"])
            support_path = Path(row["generation_change_region"])
            for path in (nuclei_path, metadata_path, support_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("sample_id") != sample_id:
                raise ValueError("metadata sample_id mismatch")
            if metadata.get("generation_mode") != "direct_full_change":
                raise ValueError(
                    f"generation_mode={metadata.get('generation_mode')!r}"
                )
            if metadata.get("base_sample_id") is not None:
                raise ValueError("independent-strength run unexpectedly has a base")
            if metadata.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
                raise ValueError("unexpected ProbNet checkpoint SHA256")
            if metadata.get("checkpoint_role") != "P(nucleus)_spatial_placement_only":
                raise ValueError("unexpected ProbNet checkpoint role")

            priors = metadata["patch_adaptive_priors"]
            if (
                priors.get("checkpoint_role")
                != "spatial_placement_probability_only"
            ):
                raise ValueError("patch prior checkpoint role is not spatial-only")
            if (
                priors.get("count_policy")
                != "reliable patch-local density else area-weighted dataset shrinkage"
            ):
                raise ValueError("unexpected count policy")
            if (
                priors.get("type_policy")
                != "reliable patch-local quota else dataset tissue prior"
            ):
                raise ValueError("unexpected type policy")

            target_raw = load_nuclei_mask(nuclei_path, remap=False)
            reference_raw = load_nuclei_mask(
                row["reference_nuclei_mask"], remap=False
            )
            target_tissue = load_tissue_mask(row["target_tissue_mask"])
            support_raw = cv2.imread(str(support_path), cv2.IMREAD_GRAYSCALE)
            if support_raw is None:
                raise FileNotFoundError(support_path)
            support = support_raw > 128
            change_raw = cv2.imread(str(row["change_region"]), cv2.IMREAD_GRAYSCALE)
            if change_raw is None:
                raise FileNotFoundError(row["change_region"])
            semantic_change = change_raw > 128
            if not np.all(support[semantic_change]):
                raise ValueError("generation support does not contain semantic change")
            if not (
                target_raw.shape
                == reference_raw.shape
                == target_tissue.shape
                == support.shape
                == (512, 512)
            ):
                raise ValueError("unexpected or misaligned mask shape")

            retained, integrity = _retain_complete_reference_cells(
                reference_raw, support
            )
            retained_pixels = retained > 0
            if not np.array_equal(target_raw[retained_pixels], retained[retained_pixels]):
                raise ValueError("retained source nuclei are not bitwise preserved")
            saved_integrity = metadata["source_cell_integrity"]
            for key in (
                "source_components",
                "kept_components",
                "deleted_components",
                "crossing_components",
            ):
                if int(saved_integrity.get(key, -1)) != int(integrity[key]):
                    raise ValueError(f"source integrity mismatch for {key}")

            requested = 0
            placed = 0
            diagnostics = metadata["probnet"]
            for tissue in diagnostics["tissues"].values():
                target_count = int(tissue["target_count"])
                placed_count = int(tissue["placed"])
                requested += target_count
                placed += placed_count
                if sum(int(value) for value in tissue["target_by_type"].values()) != target_count:
                    raise ValueError("type target quota does not sum to tissue target")
                if sum(int(value) for value in tissue["placed_by_type"].values()) != placed_count:
                    raise ValueError("placed type quota does not sum to tissue placed")
                components = tissue.get("component_sampling", {})
                if components:
                    if sum(int(value["quota"]) for value in components.values()) != target_count:
                        raise ValueError("component quota does not sum to tissue target")
                    if sum(int(value["placed"]) for value in components.values()) != placed_count:
                        raise ValueError("component placement does not sum to tissue placed")
            if int(diagnostics.get("placed", -1)) != placed:
                raise ValueError("global placed count mismatch")
            unfilled = requested - placed
            if unfilled < 0:
                raise ValueError("placed count exceeds requested count")

            totals = group_totals[group]
            totals["rows"] += 1
            totals["requested"] += requested
            totals["placed"] += placed
            totals["unfilled"] += unfilled
            totals["reference_source_components"] += int(
                integrity["source_components"]
            )
            totals["reference_kept_components"] += int(
                integrity["kept_components"]
            )
            checked += 1
        except Exception as exc:
            failures.append(f"{sample_id}: {type(exc).__name__}: {exc}")

    groups = {}
    total_requested = 0
    total_placed = 0
    for group, totals in sorted(group_totals.items()):
        total_requested += totals["requested"]
        total_placed += totals["placed"]
        groups[group] = {
            **totals,
            "placement_completion": (
                totals["placed"] / totals["requested"]
                if totals["requested"]
                else 1.0
            ),
        }
    overall_completion = (
        total_placed / total_requested if total_requested else 1.0
    )
    if overall_completion < args.minimum_placement_completion:
        failures.append(
            f"overall placement completion {overall_completion:.6f} below "
            f"{args.minimum_placement_completion:.6f}"
        )

    report = {
        "schema_version": 1,
        "status": "complete" if not failures and checked == len(rows) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "expected_count": args.expected_count,
        "checked_count": checked,
        "wsi_count": len({str(row.get("wsi_id") or "") for row in rows}),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "strength_policy": "independent_direct_from_original_reference",
        "requested_instances": total_requested,
        "placed_instances": total_placed,
        "unfilled_instances": total_requested - total_placed,
        "placement_completion": overall_completion,
        "minimum_placement_completion": args.minimum_placement_completion,
        "groups": groups,
        "failure_count": len(failures),
        "failures": failures[:200],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
