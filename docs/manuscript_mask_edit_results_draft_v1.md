# Mask-edit Results draft v2

Status: revised four-paragraph manuscript draft. Quantitative mask-quality and pathologist-review values should be inserted only after the corresponding frozen evaluation is complete.

## Proposed subsection title

**An agentic mask-editing framework integrates pathology and dataset-specific annotation knowledge bases**

**融合病理与数据集特异标注知识库的智能体式 mask 编辑框架**

## Sentence-by-sentence draft

### Paragraph 1 — Agentic mask-editing framework

**E1.** We developed an agentic mask-editing framework that converts free-form editing instructions into auditable changes of tissue and cellular states by integrating two complementary knowledge bases: one encoding pathology and tissue architecture, and the other encoding the label semantics of dataset-specific expert annotations.

**中1.** 我们构建了一个智能体式 mask 编辑框架，通过融合两个互补的知识库——描述病理过程与组织结构的病理知识库，以及描述特定数据集专家标注语义的标注知识库——将自由形式的编辑指令转化为可审计的组织和细胞状态变化。

**E2.** An instruction-only Parser agent decomposes each Chinese or English request into one or more biological intents and represents their target, direction, polarity, spatial scope, morphology and strength in a closed semantic ontology.

**中2.** 仅接收文本的 Parser agent 将每条中文或英文请求分解为一个或多个生物学意图，并使用封闭语义本体表示其作用对象、变化方向、极性、空间范围、形态和强度。

**E3.** A Planner agent then combines the structured request with organ identity and annotation-profile semantics to select one compatible primitive for each intent from the pathology-informed capability catalog.

**中3.** 随后，Planner agent 将结构化请求与器官身份及标注体系语义结合，从病理知识驱动的能力目录中为每个意图选择一个兼容的 primitive。

**E4.** The selected program is passed to a mask executor and deterministic evaluator, which together form a closed loop of planning, editing, validation and state update rather than treating mask manipulation as an isolated pixel operation.

**中4.** 选定的程序随后被传递给 mask 执行器和确定性 evaluator，由此形成“规划—编辑—验证—状态更新”的闭环，而不是将 mask 编辑视为孤立的像素操作。

### Paragraph 2 — Pathology-informed operational levels

**E5.** The executable catalog is organized across three complementary operational levels: lesion and tissue-compartment geometry, local invasive architecture and tumour topology, and cellular composition.

**中5.** 可执行目录由三个相互补充的操作层级构成：病灶与组织区室几何、局部浸润结构与肿瘤拓扑，以及细胞组成。

**E6.** Lesion- and compartment-level primitives describe spatial changes such as cohesive boundary expansion, reduction of the invasive footprint, bounded local clearance, fragmentation into residual foci, and remodelling of necrotic, stromal or immune-rich regions.

**中6.** 病灶与区室层级的 primitive 描述连续性边界扩张、侵袭性肿瘤范围缩小、局部限定区域清除、形成残余小灶的碎片化，以及坏死、间质或免疫富集区重塑等空间变化。

**E7.** Architecture-level primitives encode the manner in which tumour extends through surrounding tissue, including invasive fronts, cords, nest–cord patterns, discrete nests, small clusters and single-cell dissemination.

**中7.** 结构层级的 primitive 描述肿瘤向周围组织延伸的方式，包括浸润前沿、条索、巢索结构、离散肿瘤巢、小细胞簇和单细胞播散。

**E8.** The current catalog contains 20 semantic primitive types represented through 74 organ–annotation-profile bindings, with each capability exposed only when its required tissue labels, cell observations and spatial constraints are available.

**中8.** 当前目录包含 20 种语义 primitive，并形成 74 个“器官—标注体系—primitive”绑定；只有当所需的组织标签、细胞观测和空间约束均可用时，相应能力才会被开放。

### Paragraph 3 — Cellular states and annotation-profile knowledge

**E9.** At the cellular level, overall-cellularity primitives modulate total local nucleus density while maintaining the existing class composition, whereas cell-type-abundance primitives selectively alter an explicitly specified cell population.

**中9.** 在细胞层级，总体细胞密度 primitive 调节局部细胞核总密度并维持原有类别构成，而细胞类型丰度 primitive 则选择性地改变用户明确指定的细胞群体。

**E10.** Neoplastic-cell-abundance primitives operate within tumour-compatible tissue support, while immune-compartment primitives represent changes in the spatial extent of an annotated immune-rich region.

**中10.** 肿瘤细胞丰度 primitive 在与肿瘤兼容的组织支撑区域内进行操作，而免疫区室 primitive 表达带标注免疫富集区域在空间范围上的变化。

**E11.** Dataset-specific annotation knowledge further resolves structures whose label semantics differ across cohorts: in the masks used here, PANDA assigns Gleason-pattern labels to cancerous epithelium, leaving the glandular lumen outside the tumour label, whereas GLaS represents the complete gland unit, including its lumen, as one annotated object.

**中11.** 数据集特异的标注知识还用于解析不同队列中标签语义不一致的结构：在本研究使用的 mask 中，PANDA 的 Gleason pattern 标签对应癌性上皮，腺腔不属于肿瘤标签；GLaS 则将包括腺腔在内的完整腺体单元作为一个标注对象。

**E12.** Accordingly, the editor identifies likely glandular lumina from their appearance in the H&E image and the scarcity of nearby nuclei, without requiring a lumen to be fully enclosed within the cropped patch.

**中12.** 因此，编辑器根据 H&E 图像中的腔隙外观和周围较低的细胞核密度识别可能的腺腔；即使腺腔在 patch 边缘被截断，也不要求它在当前图像中完全闭合。

**E13.** The detected lumen and its surrounding epithelial wall are then protected from edits that do not target gland structure, while cell placement is restricted to cellular tissue outside the lumen.

**中13.** 随后，系统会保护识别出的腺腔及其周围上皮壁，避免与腺体结构无关的编辑破坏它们；需要放置细胞时，也只会选择腺腔外实际含有细胞的组织区域。

### Paragraph 4 — Closed-loop execution and compositional editing

**E14.** Before execution, each planned primitive is evaluated against the current tissue and nucleus state, and a candidate edit is committed only when it satisfies the requested direction and magnitude, legal source-to-target transitions, spatial requirements and structure-preservation constraints.

**中14.** 在执行前，每个规划得到的 primitive 都会结合当前组织和细胞核状态进行评估；只有当候选编辑满足请求的变化方向与幅度、合法的源到目标转换、空间要求和结构保护约束时，结果才会被提交。

**E15.** Requests containing multiple intents are represented as multi-step programs and executed sequentially through edit, validation and confirmation, after which the updated mask state is re-analysed before planning the next step.

**中15.** 包含多个意图的请求被表示为多步骤程序，并按照“编辑—验证—确认”的顺序连续执行；每一步完成后，系统都会重新分析更新后的 mask 状态，再规划下一步。

**E16.** Unsupported, ambiguous or conflicting requests produce an explicit clarification or review state, preventing the framework from silently substituting a different biological operation.

**中16.** 对于不受支持、存在歧义或相互冲突的请求，系统会明确进入澄清或审查状态，从而避免在未说明的情况下替换为另一种生物学操作。

**E17.** In a bilingual synthetic interface audit, the frozen Parser–Planner pipeline exactly recovered the structured semantic request and organ-compatible edit program for all 87 held-out natural-language cases (Supplementary Methods and Supplementary Table X).

**中17.** 在一项中英文合成接口测试中，冻结后的 Parser–Planner 流程在 87 条未见自然语言样本上均准确恢复了结构化语义请求及与器官兼容的编辑程序（补充方法和补充表 X）。
