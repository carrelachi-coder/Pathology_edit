# Semantic Parser / Planner benchmark v1

## Scope / 范围

This run evaluated **269** catalog-derived Chinese and English instructions with parser mode **api**.

本次运行使用 **api** 模式，评估了 **269** 条由 ontology 与 primitive catalog 派生的中英文指令。

> This is a synthetic interface-conformance and regression benchmark. It is not evidence of unrestricted clinical-language understanding or mask quality.

> 这是合成的接口一致性与回归测试，不代表系统已经具备不受限制的临床语言理解能力，也不评价最终 mask 的病理质量。

## Overall results / 总体结果

| Metric | Result | Interpretation |
|---|---:|---|
| Parse success | 100.00% | Returned a schema-valid semantic request |
| Intent-count exact | 100.00% | Preserved the number of user goals |
| Closed-ontology semantic exact | 99.26% | All scored intent fields and relations matched |
| Relation exact | 100.00% | Preserved ordered versus unordered composition |
| Primitive leakage | 0 | Parser output must contain no primitive or mechanism IDs |
| Gold-request Planner replay exact | 100.00% | Frozen catalog expectations reproduce in the current tree |
| Parsed end-to-end Planner exact | 100.00% | Parser output led to the expected organ-compatible program |

## Intent-field accuracy / 意图字段准确率

| Field | Accuracy |
|---|---:|
| `intent_type` | 100.00% |
| `target` | 100.00% |
| `operation` | 100.00% |
| `polarity` | 100.00% |
| `clinical_context` | 100.00% |
| `spatial_scope` | 99.32% |
| `morphology` | 100.00% |
| `cell_class` | 100.00% |
| `strength` | 99.66% |

## Breakdown by language / 按语言分层

| Language | n | Parse | Semantic exact | Planner exact |
|---|---:|---:|---:|---:|
| en | 134 | 100.00% | 98.51% | 100.00% |
| zh | 135 | 100.00% | 100.00% | 100.00% |

## Breakdown by case type / 按测试类型分层

| Category | n | Semantic exact | Planner exact |
|---|---:|---:|---:|
| `catalog_single_intent` | 226 | 99.56% | 100.00% |
| `negation` | 8 | 100.00% | 100.00% |
| `ordered_multi_intent` | 18 | 100.00% | 100.00% |
| `underspecified_intent` | 9 | 100.00% | 100.00% |
| `unordered_conflict` | 8 | 87.50% | 100.00% |

## Catalog audit / 目录审计

The benchmark covers **20** open primitive types and **74** organ–profile bindings.

测试覆盖 **20** 种当前开放的 primitive，以及 **74** 个器官—标注体系绑定。

Executable-scope identifiers with no open canonical profile binding: `invasive-front-expansion-v1`, `neoplastic-microinfiltration-increase-v1`, `tumor-burden-increase-v1`.

These identifiers are reported as catalog-consistency warnings and are not counted as supported benchmark primitives.

这些标识仅作为目录一致性警告，不计入当前受支持的 benchmark primitive。

## Interpretation / 解读

The API run measures the frozen Parser prompt and model on synthetic catalog-derived language. An independent clinician-authored test set is still required for an external language-generalization claim.

API 运行衡量的是冻结 prompt 与模型在目录派生合成语言上的表现；若要提出外部语言泛化结论，仍需独立的临床专家原创指令集。

Benchmark SHA-256: `29bcfa4dae2abb522d315a1bba88713b0de52a5bd5c24543281912adae093db4`
