#!/usr/bin/env python3
"""Build an online mask-only manifest from a frozen 18-condition cohort.

The frozen target masks are never loaded or copied. Only the source patch and
the original edit-intent metadata are reused so the current instruction parser,
gpt-4.1-mini contour agent, and organic_v2 projection generate fresh targets.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from phase3_mask_edit.benchmark.intents import (
    primitive_config_by_name,
    strength_interval,
)
from phase3_mask_edit.core.config import (
    default_recipe_path_for_profile,
    load_recipe,
)


DEFAULT_API_BASE_URL = "https://api.cursorai.art/v1"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = list(_read_jsonl(args.canary_manifest))
    if len(rows) != 18:
        raise ValueError(f"Expected exactly 18 canary rows, found {len(rows)}.")
    cases = [
        _build_case(row, review_index=index, api_model=args.api_model)
        for index, row in enumerate(rows, start=1)
    ]
    manifest = {
        "schema_version": "0.3",
        "description": (
            "Fresh online mask-only review for the original 18 canary sources "
            "and edit intents. Frozen target masks are explicitly excluded."
        ),
        "datasets": sorted({str(case["dataset"]) for case in cases}),
        "defaults": {
            "parser_provider": "api",
            "contour_provider": "api-multimodal",
            "projection_mode": "organic_v2",
            "status": "stage_1_mask_review",
        },
        "runtime": {
            "edit_variants": [
                {"variant_id": "instruction", "edit_mode": "instruction"}
            ],
            "parser": {
                "instruction_parser": "api",
                "api_base_url": args.api_base_url,
                "api_key_env": args.api_key_env,
                "api_model": args.api_model,
                "no_few_shot": False,
            },
            "contour": {
                "provider": "api-multimodal",
                "api_base_url": args.api_base_url,
                "api_key_env": args.api_key_env,
                "api_model": args.api_model,
                "api_image_detail": "high",
                "max_attempts": 4,
                "max_regions": 8,
                "max_points_per_region": 64,
            },
            "cell": {"cell_fill_mode": "preserve"},
            "generation": {"generation_mode": "dry-run"},
            "model_paths": {},
        },
        "provenance": {
            "source_canary_manifest": str(args.canary_manifest),
            "frozen_target_mask_consumed": False,
            "intent_source": "sibling_gt_intent_json_metadata_only",
            "api_model": args.api_model,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case_count": len(cases),
                "frozen_target_mask_consumed": False,
            },
            indent=2,
        )
    )
    return 0


def _build_case(
    row: dict[str, Any],
    *,
    review_index: int,
    api_model: str,
) -> dict[str, Any]:
    intent_path = _intent_path_from_canary_row(row)
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    instruction = _canonical_instruction(intent)
    profile = str(row["profile"])
    primitive = str(intent["primitive"])
    strength = str(intent["strength"])
    recipe = load_recipe(default_recipe_path_for_profile(profile))
    primitive_config = primitive_config_by_name(recipe, primitive)
    current_bucket = strength_interval(primitive_config, strength)
    return {
        "review_index": f"{review_index:02d}",
        "case_id": f"{review_index:02d}_{row['condition_id']}",
        "condition_id": str(row["condition_id"]),
        "sample_id": str(row["sample_id"]),
        "dataset": profile,
        "profile": profile,
        "organ": row.get("organ"),
        "primitive": primitive,
        "strength": strength,
        "source_labels": list(intent.get("source_labels") or ()),
        "target_label": intent.get("target_label"),
        "expected_area_bucket": (
            list(current_bucket)
            if current_bucket is not None
            else intent.get("expected_area_bucket")
        ),
        "expected_area_bucket_source": (
            "current_product_recipe"
            if current_bucket is not None
            else "intent_metadata_fallback"
        ),
        "instruction": instruction,
        "organic_seed": int(intent.get("seed", 0)),
        "api_model": api_model,
        "source_image": str(row["reference_image_path"]),
        "source_tissue_mask": str(row["reference_tissue_mask_path"]),
        "source_nuclei_mask": str(row["reference_nuclei_mask_path"]),
        "intent_metadata_path": str(intent_path),
        "frozen_target_mask_consumed": False,
    }


def _intent_path_from_canary_row(row: dict[str, Any]) -> Path:
    # The path is used only to locate sibling metadata. The target mask itself
    # is never opened, hashed, copied, or included in the fresh manifest.
    target_path = Path(str(row["target_tissue_mask_path"]))
    candidates = (
        target_path.parent / "gt_intent.json",
        target_path.parent.parent / "gt_intent.json",
        target_path.parent.parent.parent / "gt_intent.json",
    )
    for intent_path in candidates:
        if intent_path.exists():
            return intent_path
    tried = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"Missing original intent metadata. Tried:\n{tried}")


def _canonical_instruction(intent: dict[str, Any]) -> str:
    primitive = _primitive_instruction_phrase(str(intent["primitive"]))
    strength = str(intent.get("strength") or "moderate")
    sources = ", ".join(str(item) for item in intent.get("source_labels") or ())
    target = str(intent.get("target_label") or "")
    region = intent.get("region_hint") or {}
    location = str(region.get("description") or region.get("location") or "")
    parts = [f"Apply a {strength} {primitive} edit"]
    if sources and target:
        parts.append(f"by changing {sources} to {target}")
    elif target:
        parts.append(f"with target tissue {target}")
    if location:
        parts.append(f"in the {location}")
    parts.append("Preserve all unrequested tissue labels and regions")
    return "; ".join(parts) + "."


def _primitive_instruction_phrase(primitive: str) -> str:
    match = re.fullmatch(
        r"gleason_(upgrade|downgrade)_(\d+)to(\d+)",
        primitive.strip().lower(),
    )
    if match:
        direction, source, target = match.groups()
        return (
            f"Gleason pattern {source} to Gleason pattern {target} "
            f"{direction}"
        )
    return primitive.replace("_", " ")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("Each JSONL row must be an object.")
                yield value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the fresh online 18-condition mask review manifest."
    )
    parser.add_argument("--canary-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-model", default="gpt-4.1-mini")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
