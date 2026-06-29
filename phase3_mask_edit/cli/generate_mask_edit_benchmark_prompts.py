"""Generate and optionally LLM-check prompts for benchmark GT intents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase3_mask_edit.benchmark.models import read_intents_jsonl, write_prompts_csv
from phase3_mask_edit.benchmark.prompts import LLMConfig, accepted_prompts, generate_prompts, write_manual_review_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intents", required=True, type=Path, help="benchmark_intents.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--use-llm-generator", action="store_true")
    parser.add_argument("--generator-model", default="template")
    parser.add_argument("--generator-api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--generator-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--use-llm-checker", action="store_true")
    parser.add_argument("--checker-model", default="not_checked")
    parser.add_argument("--checker-api-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--checker-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--manual-review-per-group", type=int, default=3)
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args(argv)

    intents = read_intents_jsonl(args.intents)
    generator = LLMConfig(
        model=args.generator_model,
        api_base_url=args.generator_api_base_url,
        api_key_env=args.generator_api_key_env,
    )
    checker = LLMConfig(
        model=args.checker_model,
        api_base_url=args.checker_api_base_url,
        api_key_env=args.checker_api_key_env,
    )
    prompts = generate_prompts(
        intents,
        generator=generator,
        checker=checker,
        use_llm_generator=args.use_llm_generator,
        use_llm_checker=args.use_llm_checker,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_prompts_csv(prompts, args.output / "benchmark_prompts.csv")
    accepted = accepted_prompts(prompts)
    write_prompts_csv(accepted, args.output / "benchmark_prompts.accepted.csv")
    write_manual_review_csv(
        accepted,
        args.output / "benchmark_prompts.manual_review.csv",
        per_group=args.manual_review_per_group,
    )
    if args.print_summary:
        print(json.dumps({"num_prompts": len(prompts), "num_accepted": len(accepted)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
