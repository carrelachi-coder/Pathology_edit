# Natural-language Parser / Planner benchmark review

## 结论 / Conclusion

当前 Parser prompt 与确定性 Planner 可以固定为论文和产品的这一版本。最终未见留出集共 87 条，结构化语义完全一致率、意图数量一致率、关系一致率及端到端 Planner 一致率均为 100%，且没有出现 primitive/mechanism ID 泄漏。

The current Parser prompt and deterministic Planner are suitable for freezing as the paper/product version. On the 87-case unseen final holdout, closed-ontology semantic exact match, intent-count exact match, relation exact match, and end-to-end Planner exact match were all 100%, with no primitive or mechanism identifier leakage.

## Benchmark design / 测试设计

| Item | Value |
|---|---:|
| Total cases | 356 |
| Chinese / English | 178 / 178 |
| Catalog single intent | 296 |
| Ordered multi-intent | 24 |
| Underspecified intent | 12 |
| Negation | 12 |
| Unordered conflict | 12 |
| Development / final holdout | 269 / 87 |
| Substantive model rewrites | 343 |
| Safe template fallbacks | 3, all in development |

`gpt-5.4-mini-2026-03-17` generated natural surface forms without seeing primitive IDs. `gpt-4.1-mini` was evaluated as the product Parser through strict JSON-schema output. Gold semantic labels and Planner expectations came from the frozen ontology/catalog benchmark rather than from the language generator.

自然语言生成模型只负责改写表达，不生成 gold label，也看不到 primitive ID。生成后还会检查语言、极性、强度、临床语境、形态和多意图顺序是否漂移；不合格句子会重生成，仍无法保真的句子回退到原模板并排除出最终留出集。

## Results / 结果

| Split | n | Parse success | Intent count | Semantic exact | Relation exact | Planner exact | Leakage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Development, initial prompt | 269 | 100% | 98.51% | 61.34% | 96.28% | 79.93% | 0 |
| Development, frozen prompt | 269 | 100% | 100% | 99.26% | 100% | 100% | 0 |
| Final holdout, one shot | 87 | 100% | 100% | 100% | 100% | 100% | 0 |

The initial development errors exposed real ontology-boundary problems: invasion morphology versus generic tumour-cell abundance, negated direction, necrosis operation normalization, cohesive expansion versus topology, and ordered versus unordered composition. The prompt was revised only for these general boundaries and then frozen before the final holdout run.

开发集冻结版仍有两条字段级差异，但均不改变 Planner：一句把 `mild` 读成 `moderate`；一句把未指定范围的 “tumor area” 读成 `whole_lesion`。因此开发集 semantic exact 为 99.26%，而最终 primitive program 仍为 100%。这些差异保留在报告中，没有继续按单句硬编码。

## Input → Parser → Planner examples / 示例

| User input | Parser projection | Planner result |
|---|---|---|
| 请在局部减少炎症细胞。 | `selected_cell_population`, `decrease`, `inflammatory`, `local` | `cell-type-abundance-decrease-v1` |
| Please make the scattered single tumor cells in the peritumoral tissue more numerous. | `invasion_pattern`, `increase`, `single_cell`, `peritumoral` | `peritumoral-neoplastic-scatter-increase-v1` |
| 先把肿瘤边界连续向外扩展，再减少局部肿瘤细胞数量。 | two intents with `explicit_sequence` | cohesive expansion → confirm → neoplastic-cell decrease |
| Do not increase neoplastic cells locally. | `increase` + `negated` (not `decrease`) | clarification required; no edit is executed |
| Could you increase the tumor area and decrease the tumor area? | opposing intents with `unordered` relation | conflict detected; clarification required |

## Paper recommendation / 文章写法建议

This benchmark should be mentioned briefly in the main Results as an interface-validation result and reported in detail in the supplement. It is useful evidence that free-form bilingual instructions are normalized into the intended closed ontology and executable programs, but it is not a primary mask-edit quality result.

建议正文只用一小段或一句话报告：双语自然语言经结构化 Parser 后，在未见合成留出集上保持了意图与 Planner 映射。详细的类别、prompt、样例和字段级指标放补充材料。不要把 100% 写成临床语言泛化结论；下一阶段如果需要更强主张，应增加病理医生独立原创且盲标的外部测试集。
