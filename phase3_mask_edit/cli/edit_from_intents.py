"""Execute Phase 3 EditIntent JSON against an id mask."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
from phase3_mask_edit.core.context import MaskEditContext
from phase3_mask_edit.core.intent import EditIntent
from phase3_mask_edit.core.labels import MaskProfileSchema
from phase3_mask_edit.core.mask_io import (
    load_id_mask,
    load_metadata,
    save_change_region,
    save_id_mask,
    save_metadata,
    save_rgb_mask,
)
from phase3_mask_edit.generic.executor import EditExecutionResult, execute_edit


@dataclass(frozen=True)
class SequentialExecutionResult:
    """Result of applying one or more intents to a mask."""

    source_mask: np.ndarray
    target_mask: np.ndarray
    change_region: np.ndarray
    steps: tuple[dict[str, Any], ...]
    status: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "num_steps": len(self.steps),
            "executed_steps": sum(1 for step in self.steps if step["executed"]),
            "changed_pixels": int(np.count_nonzero(self.change_region)),
            "changed_area_fraction": (
                float(np.count_nonzero(self.change_region)) / int(self.change_region.size)
                if self.change_region.size
                else 0.0
            ),
            "steps": list(self.steps),
        }


def execute_intents_on_mask(
    mask: np.ndarray,
    intents: Sequence[EditIntent],
    *,
    reference_profile: str,
    recipe: Mapping[str, Any] | None = None,
    stop_on_failure: bool = True,
) -> SequentialExecutionResult:
    """Apply intents sequentially, updating context after each successful edit."""

    if recipe is None:
        recipe = load_recipe(default_recipe_path_for_profile(reference_profile))

    schema = MaskProfileSchema.from_reference_profile(reference_profile)
    source_mask = np.asarray(mask)
    current_mask = np.array(source_mask, copy=True)
    union_change = np.zeros(current_mask.shape, dtype=bool)
    steps: list[dict[str, Any]] = []
    overall_status = "no_intents" if not intents else "executed"

    for index, intent in enumerate(intents, start=1):
        context = MaskEditContext.from_mask(current_mask, schema)
        result = execute_edit(current_mask, intent, recipe, schema, context)
        step = _step_metadata(index, intent, result)

        if result.edit_result is None:
            step["executed"] = False
            steps.append(step)
            overall_status = result.status
            if stop_on_failure:
                break
            continue

        step["executed"] = True
        step["changed_pixels"] = int(np.count_nonzero(result.edit_result.change_region))
        step["changed_area_fraction"] = result.edit_result.changed_area_fraction
        steps.append(step)
        union_change |= result.edit_result.change_region
        current_mask = np.array(result.edit_result.target_mask, copy=True)

        if result.status not in {
            "executed_validated",
            "degraded_executed",
        }:
            overall_status = result.status
            if stop_on_failure:
                break

    if intents and all(step.get("executed") for step in steps):
        if any(str(step["status"]).startswith("degraded") for step in steps):
            overall_status = "degraded_executed"
        else:
            overall_status = "executed"

    return SequentialExecutionResult(
        source_mask=np.array(source_mask, copy=True),
        target_mask=current_mask,
        change_region=union_change,
        steps=tuple(steps),
        status=overall_status,
    )


def save_sequential_execution_output(
    result: SequentialExecutionResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save standard sequential-execution artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    paths["source_mask"] = save_id_mask(result.source_mask, out / "source_mask.png")
    paths["target_mask"] = save_id_mask(result.target_mask, out / "target_mask.png")
    paths["change_region"] = save_change_region(
        result.change_region,
        out / "change_region.png",
    )
    paths["source_mask_rgb"] = save_rgb_mask(
        result.source_mask,
        out / "source_mask_rgb.png",
    )
    paths["target_mask_rgb"] = save_rgb_mask(
        result.target_mask,
        out / "target_mask_rgb.png",
    )
    paths["summary"] = save_metadata(
        result.to_metadata(),
        out / "execution_summary.json",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if not args.allow_legacy_deterministic_executor:
        parser.error(
            "edit_from_intents uses the legacy deterministic primitive executor. "
            "Use the LLM contour organic_v2 UI/CLI instead, or pass "
            "--allow-legacy-deterministic-executor for debugging old primitive tests."
        )

    mask = load_id_mask(args.mask)
    intents = load_intents(args.intents)
    result = execute_intents_on_mask(
        mask,
        intents,
        reference_profile=args.profile,
        stop_on_failure=not args.continue_on_failure,
    )
    save_sequential_execution_output(result, args.output)

    if args.print_summary:
        import json

        print(json.dumps(result.to_metadata(), indent=2, ensure_ascii=False))

    return 0


def load_intents(path: str | Path) -> list[EditIntent]:
    """Load intents from `edit_intents.json` or a raw list JSON."""

    payload = load_metadata(path)
    if isinstance(payload, Mapping) and "intents" in payload:
        raw_intents = payload["intents"]
    else:
        raw_intents = payload

    if not isinstance(raw_intents, list):
        raise ValueError("intent file must contain a list or {'intents': [...]} mapping.")
    return [EditIntent.from_mapping(item) for item in raw_intents]


def _step_metadata(
    index: int,
    intent: EditIntent,
    result: EditExecutionResult,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "index": index,
        "primitive": intent.primitive,
        "strength": intent.strength,
        "status": result.status,
        "applicability": {
            "status": result.applicability.status,
            "reasons": list(result.applicability.reasons),
            "warnings": list(result.applicability.warnings),
            "fallback_actions": list(result.applicability.fallback_actions),
        },
        "intent": intent.to_metadata(),
    }
    if result.validation is not None:
        metadata["validation"] = _jsonable_dataclass(result.validation)
    if result.edit_result is not None:
        metadata["ops_log"] = result.edit_result.ops_log
    return metadata


def _jsonable_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute Phase3 edit_intents.json against an id mask."
    )
    parser.add_argument("--profile", required=True, help="Reference profile, e.g. BCSS.")
    parser.add_argument("--mask", required=True, type=Path, help="Original id mask PNG.")
    parser.add_argument(
        "--intents",
        required=True,
        type=Path,
        help="edit_intents.json from parse_prompts.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument(
        "--allow-legacy-deterministic-executor",
        action="store_true",
        help="Permit the old non-LLM execute_edit path for legacy/debug use.",
    )
    parser.add_argument("--print-summary", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

"""
export CURSOR_API_KEY="sk-dz8Ubk5NyvxyjG3384xyygUhgF18gBIFozee90C8FlI7vhhf"

python -m phase3_mask_edit.cli.parse_prompts \
  --profile BCSS \
  --mask phase3_mask_edit/previews/test_masks/bcss_necrosis_api_smoke/source_mask.png \
  --old-prompt "High-grade carcinoma without necrosis." \
  --new-prompt "High-grade carcinoma with focal necrosis." \
  --parser api \
  --api-base-url "https://api.cursorai.art/v1" \
  --api-key-env CURSOR_API_KEY \
  --api-model gpt-4o \
  --output phase3_mask_edit/previews/api_prompt_to_mask/necrosis_add_focal \
  --execute \
  --print-summary
"""
