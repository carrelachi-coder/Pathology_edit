# Semantic Parser / Planner benchmark v1

This frozen synthetic benchmark is generated from the v4 closed semantic ontology and the currently open primitive catalog.

## Composition

- 356 instructions: 178 Chinese and 178 English.
- 296 catalog-bound single-intent cases: four language forms for each of 74 open organ–profile primitive bindings.
- 24 explicitly ordered two-intent cases.
- 12 underspecified-invasion cases.
- 12 negated-edit cases.
- 12 unordered-conflict cases.
- Six canonical profiles: BCSS, PANDA, GLaS, IGNITE, ORCA and PUMA.

The benchmark tests intent count, closed-ontology fields, polarity, explicit order, primitive leakage and the resulting organ-compatible Planner program. It does not test image-conditioned feasibility, mask quality, clinical realism or unrestricted natural-language generalization.

## Reproduce

```bash
python scripts/build_semantic_parser_planner_benchmark.py
python scripts/run_semantic_parser_planner_benchmark.py --parser gold
python scripts/run_semantic_parser_planner_benchmark.py --parser rule-based
```

Run the product API Parser after credentials and the release model are fixed:

```bash
python scripts/run_semantic_parser_planner_benchmark.py \
  --parser api \
  --model gpt-5.6-luna \
  --reasoning-effort low \
  --api-key-env OPENAI_API_KEY
```

Use `--write-predictions` only when per-case outputs are needed for error analysis. The committed summary reports are sufficient for routine catalog regression.

## Current results

- Gold structured-request → Planner replay: 356/356 exact.
- Offline rule-based Parser → Planner pipeline: 356/356 exact on the frozen templates after fixing English connector tokenization and catalog-phrase precedence.
- Product API Parser: not run in this environment because `OPENAI_API_KEY` was absent.

The 100% offline result is an interface-conformance ceiling on known templates, not a paper-level language-understanding result.
