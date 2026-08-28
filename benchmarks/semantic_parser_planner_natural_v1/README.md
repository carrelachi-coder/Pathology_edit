# Natural Semantic Parser / Planner Benchmark v1

This benchmark evaluates the product API Parser and deterministic Planner on
natural Chinese and English requests derived from the reviewed ontology and
open primitive catalog.

- `benchmark.jsonl`: 356 frozen cases with immutable structured gold labels.
- `manifest.json`: source lineage, digest, catalog coverage and split counts.
- `generation_report.json`: generator model, token accounting and fallback count.
- `natural_language_samples.md`: stratified rewrite examples for human review.
- `results_api_gpt_4_1_mini_development_final/`: final development result.
- `results_api_gpt_4_1_mini_final_holdout/`: one-shot final holdout result.
- `benchmark_review.md`: bilingual design and result review.

The generator rewrites wording only. It receives semantic labels but no
primitive identifiers, and the labels are inherited unchanged from
`semantic_parser_planner_v1`. A semantic-surface gate rejects changed polarity,
strength, morphology, language, clinical context, or intent ordering. Cases
that cannot be safely rewritten fall back to the source template and remain in
the development split.

The API key is read from `OPENAI_API_KEY` and is never stored in benchmark
artifacts. Example reproduction:

```bash
python scripts/run_semantic_parser_planner_benchmark.py \
  --benchmark benchmarks/semantic_parser_planner_natural_v1/benchmark.jsonl \
  --manifest benchmarks/semantic_parser_planner_natural_v1/manifest.json \
  --parser api \
  --api-protocol chat-completions \
  --model gpt-4.1-mini \
  --api-base-url https://api.cursorai.art/v1 \
  --evaluation-split final_holdout \
  --workers 8
```

This is a synthetic language-interface benchmark. It does not evaluate mask
quality and must not be presented as independent clinician-authored validation.
