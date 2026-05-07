from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phase3_mask_edit.core.mask_io import load_id_mask, save_metadata
from phase3_mask_edit.parser.api_parser import ApiParserConfig, parse_prompts_with_api
from phase3_mask_edit.parser.qwen_local_parser import (
    QwenLocalParserConfig,
    parse_prompts_with_qwen_local,
)
from phase3_mask_edit.parser.semantic_diff import (
    load_semantic_diff,
    save_semantic_diff,
)
from phase3_mask_edit.rules.semantic_to_intent import plan_edit_intents


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    old_prompt = _read_text_arg(args.old_prompt, args.old_prompt_file)
    new_prompt = _read_text_arg(args.new_prompt, args.new_prompt_file)

    if args.semantic_diff:
        semantic_diff = load_semantic_diff(args.semantic_diff)
        parser_info = {"mode": "semantic_diff_file", "path": str(args.semantic_diff)}
    elif args.parser == "api":
        if not args.api_model:
            parser.error("--api-model is required when --parser api is used")
        if old_prompt is None or new_prompt is None:
            parser.error("--old-prompt/--new-prompt or prompt files are required")
        config = ApiParserConfig(
            model=args.api_model,
            api_base_url=args.api_base_url,
            api_key_env=args.api_key_env,
            timeout_sec=args.api_timeout_sec,
            temperature=args.api_temperature,
            use_few_shot=not args.no_few_shot,
        )
        semantic_diff = parse_prompts_with_api(
            old_prompt,
            new_prompt,
            config=config,
        )
        parser_info = {
            "mode": "api",
            "api_base_url": args.api_base_url,
            "api_key_env": args.api_key_env,
            "api_model": args.api_model,
            "use_few_shot": not args.no_few_shot,
        }
    elif args.parser == "qwen-local":
        if not args.qwen_model_path:
            parser.error("--qwen-model-path is required when --parser qwen-local is used")
        if old_prompt is None or new_prompt is None:
            parser.error("--old-prompt/--new-prompt or prompt files are required")
        config = QwenLocalParserConfig(
            model_path=args.qwen_model_path,
            device=args.qwen_device,
            max_new_tokens=args.qwen_max_new_tokens,
            temperature=args.qwen_temperature,
            top_p=args.qwen_top_p,
            do_sample=not args.qwen_greedy,
            use_few_shot=not args.no_few_shot,
        )
        semantic_diff = parse_prompts_with_qwen_local(
            old_prompt,
            new_prompt,
            config=config,
        )
        parser_info = {
            "mode": "qwen-local",
            "model_path": args.qwen_model_path,
            "device": args.qwen_device,
            "max_new_tokens": args.qwen_max_new_tokens,
            "temperature": args.qwen_temperature,
            "top_p": args.qwen_top_p,
            "do_sample": not args.qwen_greedy,
            "use_few_shot": not args.no_few_shot,
        }
    else:
        parser.error("--semantic-diff is required when --parser fixture is used")

    old_mask = load_id_mask(args.mask) if args.mask else None
    plan = plan_edit_intents(
        semantic_diff,
        reference_profile=args.profile,
        old_mask=old_mask,
        old_prompt=old_prompt,
        new_prompt=new_prompt,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_semantic_diff(semantic_diff, output_dir / "semantic_diff.json")
    save_metadata(
        {"intents": [intent.to_metadata() for intent in plan.intents]},
        output_dir / "edit_intents.json",
    )
    planning_summary = plan.to_metadata()
    planning_summary["parser"] = parser_info
    save_metadata(planning_summary, output_dir / "planning_summary.json")

    if args.print_summary:
        print(json.dumps(planning_summary, indent=2, ensure_ascii=False))

    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse old/new pathology prompts into Phase3 EditIntent plans."
    )
    parser.add_argument("--profile", required=True, help="Reference mask profile, e.g. BCSS.")
    parser.add_argument("--old-prompt", help="Original pathology-report prompt.")
    parser.add_argument("--new-prompt", help="Edited pathology-report prompt.")
    parser.add_argument("--old-prompt-file", type=Path, help="File containing old prompt.")
    parser.add_argument("--new-prompt-file", type=Path, help="File containing new prompt.")
    parser.add_argument(
        "--semantic-diff",
        type=Path,
        help="Existing semantic_diff JSON. Useful for fixture/offline planning.",
    )
    parser.add_argument(
        "--parser",
        choices=("api", "qwen-local", "fixture"),
        default="fixture",
        help=(
            "Parser mode. fixture requires --semantic-diff; api calls a model API; "
            "qwen-local loads a local Qwen model."
        ),
    )
    parser.add_argument("--mask", type=Path, help="Optional original id mask for applicability.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-model", help="Model name for API parser mode.")
    parser.add_argument("--api-timeout-sec", type=float, default=60.0)
    parser.add_argument("--api-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-model-path", help="Local Qwen model path.")
    parser.add_argument("--qwen-device", default="cuda")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=256)
    parser.add_argument("--qwen-temperature", type=float, default=0.1)
    parser.add_argument("--qwen-top-p", type=float, default=0.9)
    parser.add_argument("--qwen-greedy", action="store_true")
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    return parser


def _read_text_arg(value: str | None, path: Path | None) -> str | None:
    if value is not None and path is not None:
        raise SystemExit("Provide either direct prompt text or prompt file, not both.")
    if path is None:
        return value
    return path.read_text(encoding="utf-8").strip()


if __name__ == "__main__":
    raise SystemExit(main())
"""
python -m phase3_mask_edit.cli.parse_prompts \
  --profile BCSS \
  --old-prompt "High-grade carcinoma without necrosis." \
  --new-prompt "High-grade carcinoma with focal necrosis." \
  --parser api \
  --api-base-url "https://api.cursorai.art/console/v1" \
  --api-key-env "sk-dz8Ubk5NyvxyjG3384xyygUhgF18gBIFozee90C8FlI7vhhf" \
  --api-model gpt-4o \
  --output /Users/wangqinxin/Documents/GitHub/Pathology_edit/phase3_mask_edit/previews \
  --print-summary
"""