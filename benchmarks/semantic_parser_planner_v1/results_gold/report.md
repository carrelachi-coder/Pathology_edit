# Semantic Parser / Planner benchmark v1

## Scope / 范围

This run evaluated **356** catalog-derived Chinese and English instructions with parser mode **gold**.

本次运行使用 **gold** 模式，评估了 **356** 条由 ontology 与 primitive catalog 派生的中英文指令。

> This is a synthetic interface-conformance and regression benchmark. It is not evidence of unrestricted clinical-language understanding or mask quality.

> 这是合成的接口一致性与回归测试，不代表系统已经具备不受限制的临床语言理解能力，也不评价最终 mask 的病理质量。

## Overall results / 总体结果

| Metric | Result | Interpretation |
|---|---:|---|
| Parse success | n/a | Returned a schema-valid semantic request |
| Intent-count exact | n/a | Preserved the number of user goals |
| Closed-ontology semantic exact | n/a | All scored intent fields and relations matched |
| Relation exact | n/a | Preserved ordered versus unordered composition |
| Primitive leakage | 0 | Parser output must contain no primitive or mechanism IDs |
| Gold-request Planner replay exact | 100.00% | Frozen catalog expectations reproduce in the current tree |
| Parsed end-to-end Planner exact | n/a | Parser output led to the expected organ-compatible program |

## Intent-field accuracy / 意图字段准确率

| Field | Accuracy |
|---|---:|
| `intent_type` | n/a |
| `target` | n/a |
| `operation` | n/a |
| `polarity` | n/a |
| `clinical_context` | n/a |
| `spatial_scope` | n/a |
| `morphology` | n/a |
| `cell_class` | n/a |
| `strength` | n/a |

## Breakdown by language / 按语言分层

| Language | n | Parse | Semantic exact | Planner exact |
|---|---:|---:|---:|---:|
| en | 178 | n/a | n/a | n/a |
| zh | 178 | n/a | n/a | n/a |

## Breakdown by case type / 按测试类型分层

| Category | n | Semantic exact | Planner exact |
|---|---:|---:|---:|
| `catalog_single_intent` | 296 | n/a | n/a |
| `negation` | 12 | n/a | n/a |
| `ordered_multi_intent` | 24 | n/a | n/a |
| `underspecified_intent` | 12 | n/a | n/a |
| `unordered_conflict` | 12 | n/a | n/a |

## Catalog audit / 目录审计

The benchmark covers **20** open primitive types and **74** organ–profile bindings.

测试覆盖 **20** 种当前开放的 primitive，以及 **74** 个器官—标注体系绑定。

Executable-scope identifiers with no open canonical profile binding: `invasive-front-expansion-v1`, `neoplastic-microinfiltration-increase-v1`, `tumor-burden-increase-v1`.

These identifiers are reported as catalog-consistency warnings and are not counted as supported benchmark primitives.

这些标识仅作为目录一致性警告，不计入当前受支持的 benchmark primitive。

## Interpretation / 解读

This run validates only the deterministic Planner projection from reviewed structured requests; it does not score natural-language parsing.

本次运行仅验证已审查结构化请求到确定性 Planner 输出的映射，不评价自然语言解析。

Benchmark SHA-256: `3932287495dd7f78b2feb2b0b087f7bb2c0990c3d93e451015cb8593d8acd106`
