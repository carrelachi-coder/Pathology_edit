#!/usr/bin/env python3
"""Split the enriched paired U1/U2 nuclei manifest into frozen analysis groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GROUPS = (
    ("tumor_burden_increase", "moderate", "u1_moderate"),
    ("tumor_burden_increase", "significant", "u1_significant"),
    ("stromal_immune_infiltration", "moderate", "u2_moderate"),
    ("stromal_immune_infiltration", "significant", "u2_significant"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1200)
    parser.add_argument("--expected-group-count", type=int, default=300)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    if len(rows) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} rows, found {len(rows)}")
    for row in rows:
        for field in (
            "target_nuclei_mask",
            "target_nuclei_metadata",
            "generation_change_region",
            "inpaint_change_region",
        ):
            path = Path(row.get(field) or "")
            if not path.is_file():
                raise FileNotFoundError(
                    f"{row.get('sample_id')}: missing {field}: {path}"
                )

    outputs = {}
    for primitive, strength, label in GROUPS:
        group_rows = [
            row
            for row in rows
            if row["primitive"] == primitive and row["strength"] == strength
        ]
        if len(group_rows) != args.expected_group_count:
            raise ValueError(
                f"{label}: expected {args.expected_group_count}, found {len(group_rows)}"
            )
        path = args.output_root / f"{label}_nuclei_manifest.jsonl"
        write_jsonl(path, group_rows)
        outputs[label] = str(path)

    first_reference = str(rows[0]["reference_id"])
    smoke_rows = [row for row in rows if str(row["reference_id"]) == first_reference]
    if len(smoke_rows) != len(GROUPS):
        raise ValueError(
            f"smoke reference {first_reference} has {len(smoke_rows)} rows"
        )
    smoke_path = args.output_root / "smoke_nuclei_manifest.jsonl"
    write_jsonl(smoke_path, smoke_rows)
    outputs["smoke"] = str(smoke_path)

    summary = {
        "schema_version": 1,
        "status": "complete",
        "source_manifest": str(args.manifest),
        "source_count": len(rows),
        "group_count": args.expected_group_count,
        "smoke_reference": first_reference,
        "outputs": outputs,
    }
    (args.output_root / "nuclei_manifest_split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
