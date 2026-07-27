"""Select a deterministic organ/primitive/strength benchmark preflight subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from phase3_mask_edit.benchmark.models import (
    BenchmarkIntent,
    read_intents_jsonl,
    write_intents_jsonl,
)


Cell = tuple[str, str, str]


def select_preflight_intents(
    intents: Iterable[BenchmarkIntent],
    *,
    per_cell: int = 1,
    seed: int = 13,
) -> list[BenchmarkIntent]:
    """Select up to ``per_cell`` intents from every represented benchmark cell."""
    if per_cell < 1:
        raise ValueError("per_cell must be at least 1")

    grouped: dict[Cell, list[BenchmarkIntent]] = defaultdict(list)
    seen_ids: set[str] = set()
    for intent in intents:
        if intent.sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {intent.sample_id}")
        seen_ids.add(intent.sample_id)
        grouped[(intent.organ, intent.primitive, intent.strength)].append(intent)

    selected: list[BenchmarkIntent] = []
    for cell in sorted(grouped):
        candidates = sorted(
            grouped[cell],
            key=lambda item: (
                _qc_rank(item.qc_status),
                _stable_rank(seed, item.sample_id),
                item.sample_id,
            ),
        )
        selected.extend(candidates[:per_cell])
    return sorted(
        selected,
        key=lambda item: (item.organ, item.primitive, item.strength, item.sample_id),
    )


def summarize_selection(
    source: list[BenchmarkIntent], selected: list[BenchmarkIntent]
) -> dict[str, object]:
    source_cells = {(item.organ, item.primitive, item.strength) for item in source}
    selected_counts = Counter(
        (item.organ, item.primitive, item.strength) for item in selected
    )
    return {
        "num_source_intents": len(source),
        "num_selected_intents": len(selected),
        "num_source_cells": len(source_cells),
        "num_selected_cells": len(selected_counts),
        "missing_cells": [list(cell) for cell in sorted(source_cells - selected_counts.keys())],
        "duplicate_cells": [
            [*cell, count]
            for cell, count in sorted(selected_counts.items())
            if count > 1
        ],
        "organs": sorted({item.organ for item in selected}),
        "primitives": sorted({item.primitive for item in selected}),
        "strengths": sorted({item.strength for item in selected}),
        "specialized_primitives": sorted(
            {item.primitive for item in selected if item.specialized}
        ),
        "qc_status_counts": dict(
            sorted(Counter(item.qc_status for item in selected).items())
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--per-cell", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    source = read_intents_jsonl(args.intents)
    selected = select_preflight_intents(
        source,
        per_cell=args.per_cell,
        seed=args.seed,
    )
    summary = summarize_selection(source, selected)
    if summary["missing_cells"]:
        raise RuntimeError("Preflight selection did not cover every source cell")

    write_intents_jsonl(selected, args.output)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}|{sample_id}".encode("utf-8")).hexdigest()


def _qc_rank(status: str) -> int:
    return {
        "accepted": 0,
        "manual_review": 1,
        "pending": 2,
        "rejected": 4,
    }.get(status.lower(), 3)


if __name__ == "__main__":
    raise SystemExit(main())
