#!/usr/bin/env python3
"""Build audited embedding manifests for the paired two-primitive cohort."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image

from phase3_mask_edit.benchmark.pathokid import sha256_file, stable_digest


PRIMITIVE_CODES = {
    "tumor_burden_increase": "u1",
    "stromal_immune_infiltration": "u2",
}
STRENGTHS = ("moderate", "significant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuclei-manifest", type=Path, required=True)
    parser.add_argument("--mask-cohort-manifest", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-references", type=int, default=300)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_image(path: Path, *, sample_id: str, field: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{sample_id}: missing {field}: {path}")
    with Image.open(path) as image:
        if image.size != (512, 512):
            raise ValueError(
                f"{sample_id}: {field} has size {image.size}, expected (512, 512)"
            )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def main() -> int:
    args = parse_args()
    nuclei_rows = read_jsonl(args.nuclei_manifest)
    cohort_rows = read_jsonl(args.mask_cohort_manifest)
    cohort_by_reference = {
        str(row["sample_id"]): row for row in cohort_rows
    }
    if len(cohort_by_reference) != args.expected_references:
        raise ValueError(
            f"expected {args.expected_references} mask-cohort references, "
            f"found {len(cohort_by_reference)}"
        )
    expected_rows = args.expected_references * len(PRIMITIVE_CODES) * len(STRENGTHS)
    if len(nuclei_rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} nuclei rows, found {len(nuclei_rows)}")

    row_by_key: dict[tuple[str, str, str], dict] = {}
    for row in nuclei_rows:
        key = (
            str(row["reference_id"]),
            str(row["primitive"]),
            str(row["strength"]),
        )
        if key in row_by_key:
            raise ValueError(f"duplicate nuclei row: {key}")
        row_by_key[key] = row

    output: dict[tuple[str, str], list[dict]] = {
        (code, strength): []
        for code in PRIMITIVE_CODES.values()
        for strength in STRENGTHS
    }
    failures: list[dict] = []
    for reference_id in sorted(cohort_by_reference):
        cohort = cohort_by_reference[reference_id]
        for primitive, code in PRIMITIVE_CODES.items():
            primitive_metrics = cohort["primitive_metrics"][primitive]
            moderate_dose = float(primitive_metrics["moderate_dose"])
            significant_dose = float(primitive_metrics["significant_dose"])
            dose_increase = float(primitive_metrics["dose_increase"])
            overlap = cohort["cross_primitive_overlap_descriptive_only"]
            for strength in STRENGTHS:
                key = (reference_id, primitive, strength)
                row = row_by_key.get(key)
                if row is None:
                    failures.append({"key": key, "error": "missing nuclei row"})
                    continue
                sample_id = str(row["sample_id"])
                paths = {
                    "reference_image": Path(row["reference_image"]),
                    "inpaint_image": (
                        args.generated_root
                        / "inpaint"
                        / sample_id
                        / "generated_image.png"
                    ),
                    "cross_image": (
                        args.generated_root
                        / "cross-v1"
                        / sample_id
                        / "generated_image.png"
                    ),
                }
                try:
                    for field, path in paths.items():
                        validate_image(path, sample_id=sample_id, field=field)
                except Exception as exc:
                    failures.append(
                        {
                            "sample_id": sample_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                realized_dose = (
                    moderate_dose if strength == "moderate" else significant_dose
                )
                final = {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "reference_id": reference_id,
                    "pair_id": str(row["pair_id"]),
                    "moderate_sample_id": str(row["moderate_sample_id"]),
                    "patient_id": str(row["patient_id"]),
                    "wsi_id": str(row["wsi_id"]),
                    "profile": str(row["profile"]),
                    "primitive": primitive,
                    "primitive_code": code,
                    "strength": strength,
                    "reference_image": str(paths["reference_image"]),
                    "inpaint_image": str(paths["inpaint_image"]),
                    "cross_image": str(paths["cross_image"]),
                    "target_tissue_mask": str(row["target_tissue_mask"]),
                    "target_nuclei_mask": str(row["target_nuclei_mask"]),
                    "change_region": str(row["change_region"]),
                    "generation_change_region": str(
                        row["generation_change_region"]
                    ),
                    "generation_seed": int(row["generation_seed"]),
                    "changed_area_fraction_of_patch": float(
                        row["changed_area_fraction"]
                    ),
                    "realized_dose_fraction": realized_dose,
                    "moderate_realized_dose_fraction": moderate_dose,
                    "significant_realized_dose_fraction": significant_dose,
                    "dose_increase_fraction": dose_increase,
                    "moderate_containment_in_significant": float(
                        primitive_metrics["moderate_containment_in_significant"]
                    ),
                    "moderate_significant_mask_iou": float(
                        primitive_metrics["iou"]
                    ),
                    "cross_primitive_mask_iou_moderate": float(
                        overlap["moderate"]["iou"]
                    ),
                    "cross_primitive_mask_iou_significant": float(
                        overlap["significant"]["iou"]
                    ),
                    "sha256": {
                        field: sha256_file(path) for field, path in paths.items()
                    },
                }
                output[(code, strength)].append(final)

    if failures:
        preview = json.dumps(failures[:20], indent=2, ensure_ascii=False)
        raise RuntimeError(f"{len(failures)} manifest failures:\n{preview}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    for (code, strength), rows in output.items():
        if len(rows) != args.expected_references:
            raise ValueError(
                f"{code}/{strength}: expected {args.expected_references}, "
                f"found {len(rows)}"
            )
        path = args.output_dir / f"{code}_{strength}_evaluation_manifest.jsonl"
        write_jsonl(path, rows)
        summary = {
            "status": "complete",
            "primitive_code": code,
            "primitive": rows[0]["primitive"],
            "strength": strength,
            "row_count": len(rows),
            "reference_count": len({row["reference_id"] for row in rows}),
            "wsi_count": len({row["wsi_id"] for row in rows}),
            "manifest": str(path),
            "manifest_sha256": sha256_file(path),
            "rows_digest": stable_digest(rows),
        }
        path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summaries[f"{code}_{strength}"] = summary

    cell_counts = Counter(
        (row["primitive_code"], row["strength"])
        for rows in output.values()
        for row in rows
    )
    combined = {
        "status": "complete",
        "expected_references": args.expected_references,
        "total_rows": sum(cell_counts.values()),
        "cell_counts": {
            f"{code}_{strength}": cell_counts[(code, strength)]
            for code in PRIMITIVE_CODES.values()
            for strength in STRENGTHS
        },
        "manifests": summaries,
    }
    (args.output_dir / "embedding_manifest_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(combined, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
