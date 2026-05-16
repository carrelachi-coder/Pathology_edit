"""Execute Phase 3 EditIntent JSON against an id mask."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from phase3_mask_edit.core.config import default_recipe_path_for_profile, load_recipe
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


LEGACY_EXECUTOR_RETIRED_MESSAGE = (
    "The old non-LLM deterministic primitive executor has been retired from "
    "active Phase 3 entrypoints. Use the LLM contour organic_v2 UI/API/fixture "
    "backend to execute mask edits."
)


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
    """Retired compatibility hook for the old non-LLM primitive executor."""

    del mask, intents, reference_profile, recipe, stop_on_failure
    raise RuntimeError(LEGACY_EXECUTOR_RETIRED_MESSAGE)


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
    parser.error(LEGACY_EXECUTOR_RETIRED_MESSAGE)

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
