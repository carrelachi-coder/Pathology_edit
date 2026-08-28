# Semantic Parser / Planner benchmark v1

## Scope / 范围

This run evaluated **356** catalog-derived Chinese and English instructions with parser mode **rule-based**.

本次运行使用 **rule-based** 模式，评估了 **356** 条由 ontology 与 primitive catalog 派生的中英文指令。

> This is a synthetic interface-conformance and regression benchmark. It is not evidence of unrestricted clinical-language understanding or mask quality.

> 这是合成的接口一致性与回归测试，不代表系统已经具备不受限制的临床语言理解能力，也不评价最终 mask 的病理质量。

## Overall results / 总体结果

| Metric | Result | Interpretation |
|---|---:|---|
| Parse success | 100.00% | Returned a schema-valid semantic request |
| Intent-count exact | 100.00% | Preserved the number of user goals |
| Closed-ontology semantic exact | 100.00% | All scored intent fields and relations matched |
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
| `spatial_scope` | 100.00% |
| `morphology` | 100.00% |
| `cell_class` | 100.00% |
| `strength` | 100.00% |

## Breakdown by language / 按语言分层

| Language | n | Parse | Semantic exact | Planner exact |
|---|---:|---:|---:|---:|
| en | 178 | 100.00% | 100.00% | 100.00% |
| zh | 178 | 100.00% | 100.00% | 100.00% |

## Breakdown by case type / 按测试类型分层

| Category | n | Semantic exact | Planner exact |
|---|---:|---:|---:|
| `catalog_single_intent` | 296 | 100.00% | 100.00% |
| `negation` | 12 | 100.00% | 100.00% |
| `ordered_multi_intent` | 24 | 100.00% | 100.00% |
| `underspecified_intent` | 12 | 100.00% | 100.00% |
| `unordered_conflict` | 12 | 100.00% | 100.00% |

## Catalog audit / 目录审计

The benchmark covers **20** open primitive types and **74** organ–profile bindings.

测试覆盖 **20** 种当前开放的 primitive，以及 **74** 个器官—标注体系绑定。

Executable-scope identifiers with no open canonical profile binding: `invasive-front-expansion-v1`, `neoplastic-microinfiltration-increase-v1`, `tumor-burden-increase-v1`.

These identifiers are reported as catalog-consistency warnings and are not counted as supported benchmark primitives.

这些标识仅作为目录一致性警告，不计入当前受支持的 benchmark primitive。

## Interpretation / 解读

The offline rule-based parser passed every frozen generated form after catalog-phrase and connector regressions were corrected. This is a known-template conformance ceiling, not a measurement of the product API Parser.

在修复目录短语与连接词回归后，离线规则解析器通过了全部冻结的生成式语句；这是已知模板上的一致性上限，不是产品 API Parser 的性能测量。

Benchmark SHA-256: `3932287495dd7f78b2feb2b0b087f7bb2c0990c3d93e451015cb8593d8acd106`
