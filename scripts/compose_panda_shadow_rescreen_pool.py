#!/usr/bin/env python3
"""Compose a targeted PANDA rescreen pool without expanding unrelated cases."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.prepare_panda_primitive_shadow_selection import (
    _canonical_sha256,
    _sha256,
    validate_candidate_pool,
)

PRODUCER_ID = "panda-shadow-targeted-rescreen-pool-composer-v1"


def compose_rescreen_pool(
    *,
    base: dict,
    replacement: dict,
    evaluation_indices: tuple[int, ...],
    base_path: Path,
    replacement_path: Path,
) -> dict:
    """Replace only named evaluation rows and retain source-pool evidence."""

    validate_candidate_pool(base)
    validate_candidate_pool(replacement)
    if len(base["evaluations"]) != len(replacement["evaluations"]):
        raise ValueError("candidate pools have different evaluation counts")
    valid_indices = set(range(len(base["evaluations"])))
    requested = set(evaluation_indices)
    if not requested or not requested <= valid_indices:
        raise ValueError("target evaluation indices are empty or out of range")
    payload = copy.deepcopy(base)
    for index in sorted(requested):
        left = base["evaluations"][index]
        right = replacement["evaluations"][index]
        for identity in ("evaluation_id", "mechanism_id", "primitive_id"):
            if left[identity] != right[identity]:
                raise ValueError(
                    f"evaluation {index} identity drifted for {identity}"
                )
        payload["evaluations"][index] = copy.deepcopy(right)
    payload.pop("candidate_pool_sha256", None)
    payload["producer_id"] = PRODUCER_ID
    payload["candidate_count_per_evaluation"] = min(
        len(item["candidates"]) for item in payload["evaluations"]
    )
    payload["composition"] = {
        "mode": "targeted_evaluation_replacement",
        "base_pool": str(base_path.resolve()),
        "base_pool_file_sha256": _sha256(base_path),
        "base_candidate_pool_sha256": base["candidate_pool_sha256"],
        "replacement_pool": str(replacement_path.resolve()),
        "replacement_pool_file_sha256": _sha256(replacement_path),
        "replacement_candidate_pool_sha256": replacement[
            "candidate_pool_sha256"
        ],
        "replaced_evaluation_indices": sorted(requested),
        "unrelated_evaluations_preserved_exactly": True,
    }
    payload["freeze_status"] = (
        "targeted_rescreen_pool_pending_native_authority_and_live_compiler"
    )
    payload["candidate_pool_sha256"] = _canonical_sha256(payload)
    validate_candidate_pool(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--replacement-pool", type=Path, required=True)
    parser.add_argument("--evaluation-indices", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indices = tuple(
        sorted(
            {
                int(value.strip())
                for value in args.evaluation_indices.split(",")
                if value.strip()
            }
        )
    )
    base = json.loads(args.base_pool.read_text(encoding="utf-8"))
    replacement = json.loads(
        args.replacement_pool.read_text(encoding="utf-8")
    )
    payload = compose_rescreen_pool(
        base=base,
        replacement=replacement,
        evaluation_indices=indices,
        base_path=args.base_pool,
        replacement_path=args.replacement_pool,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "candidate_pool_sha256": payload["candidate_pool_sha256"],
                "replaced_evaluation_indices": list(indices),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
