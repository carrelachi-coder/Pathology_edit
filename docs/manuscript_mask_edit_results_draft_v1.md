# Mask-edit Results draft v1

Status: working manuscript text for review. The catalog counts below are reproduced from the current `final-mask-edit` capability registry. Quantitative mask-quality and pathologist-review values remain deliberately unfilled until the corresponding frozen evaluation is complete.

## Proposed subsection title

**Pathology-aware primitives produce distinguishable and annotation-compatible target states**

**病理感知的编辑原语生成具有区分度且与标注体系兼容的目标状态**

## Sentence-by-sentence draft

### Paragraph 1 — Refined action space

**E1.** We refined the mask-edit action space around biological state variables that can be represented and verified in each dataset, rather than imposing one universal set of pixel transformations on every organ.

**中1.** 我们围绕每个数据集中能够被表达和验证的生物学状态变量重新构建了 mask-edit 动作空间，而不是将同一组像素变换强行应用于所有器官。

**E2.** The resulting catalog exposes 20 distinct semantic primitive types through 74 organ–annotation-profile bindings across breast, prostate, colorectal, lung, oral and melanoma data.

**中2.** 最终目录在乳腺、前列腺、结直肠、肺、口腔和黑色素瘤数据中开放了 20 种语义不同的 primitive，共形成 74 个“器官—标注体系—primitive”绑定。

**E3.** Capability is profile specific: an operation is available only when the tissue labels, cell observations and pathology-specific spatial constraints required to evaluate that operation are present.

**中3.** 这些能力是标注体系特异的：只有当相应的组织标签、细胞观测以及病理特异的空间约束均可用时，一个操作才会被开放。

**E4.** Unsupported combinations remain closed rather than being approximated by a superficially similar edit.

**中4.** 对于不受支持的组合，系统会保持关闭，而不会用表面相似的编辑来替代。

### Paragraph 2 — Why the primitives are separated

**E5.** We organized the primitives into three operational strata according to the biological state that they alter: tissue extent and compartment geometry, local invasive architecture and topology, and cellular composition.

**中5.** 我们按照 primitive 所改变的生物学状态，将其组织为三个操作层级：组织范围与区室几何、局部浸润结构与拓扑，以及细胞组成。

**E6.** Tissue-extent operations distinguish contiguous boundary expansion, coherent reduction of the invasive footprint, bounded local clearance and fragmentation into residual foci, because these changes encode different spatial outcomes despite all changing tumour area.

**中6.** 组织范围类操作区分连续性的边界扩张、侵袭性肿瘤足迹的整体缩小、局部限定区域的清除，以及形成残余小灶的碎片化；虽然这些操作都会改变肿瘤面积，但它们表达的是不同的空间结局。

**E7.** In particular, the generic tumour-burden-increase alias was retired where it was operationally indistinguishable from cohesive boundary expansion.

**中7.** 尤其是，当笼统的“肿瘤负荷增加”与“连续性边界扩张”在实际操作上无法区分时，我们移除了前者这一别名。

**E8.** Invasion-pattern primitives instead encode how tumour cells cross the existing boundary, including cords, mixed nest–cord extensions, discrete nests, small clusters and single-cell scatter.

**中8.** 相比之下，浸润模式类 primitive 描述肿瘤细胞如何越过现有边界，包括条索、巢索混合延伸、离散肿瘤巢、小细胞簇和单细胞散布。

**E9.** Their geometry is constrained to form smooth, tissue-connected or biologically separated structures of sufficient scale, avoiding the small triangular protrusions and arbitrary gland splitting produced by earlier implementations.

**中9.** 这些结构在几何上被限制为具有足够尺度、边缘平滑，并与组织连续或具有合理生物学间隔的形态，从而避免早期实现中出现的小三角形凸起和任意切断腺体的问题。

### Paragraph 3 — Cell-level distinctions

**E10.** Cell-level operations were separated by denominator and biological subject rather than by the shared act of adding or removing nuclei.

**中10.** 细胞层面的操作不是按照“增加或删除细胞”这一共同动作来划分，而是依据统计分母和生物学对象进行区分。

**E11.** Overall-cellularity edits change the total local nucleus density while approximately preserving the existing class mixture, whereas cell-type-abundance edits selectively alter one explicitly named class.

**中11.** 总体细胞密度编辑改变局部细胞核的总密度，同时尽量保持原有类别构成；而特定细胞类型丰度编辑只改变用户明确指定的一类细胞。

**E12.** Neoplastic-cell-abundance edits are a tumour-cell-specific subset with separate placement and preservation constraints, and tissue-scale immune-compartment edits change an annotated immune-rich region rather than merely inserting inflammatory nuclei.

**中12.** 肿瘤细胞丰度编辑是面向肿瘤细胞的特定操作，具有独立的放置与保护约束；组织尺度的免疫区室编辑改变的是带标注的免疫富集区域，而不只是简单插入炎症细胞核。

**E13.** These definitions make the primitives distinguishable by their intended endpoint even when their low-level implementation shares nucleus insertion or deletion routines.

**中13.** 因此，即使不同 primitive 在底层共享细胞核插入或删除程序，它们仍可通过预期终点清楚地区分。

### Paragraph 4 — Annotation-aware gland handling

**E14.** Gland-containing profiles required dataset-specific treatment because lumen semantics differ between PANDA and GlaS.

**中14.** 含腺体的数据需要采用数据集特异的处理方式，因为 PANDA 与 GLaS 对腺腔的标注语义不同。

**E15.** PANDA represents gland lumina as stroma, whereas GlaS includes luminal space within the gland annotation; consequently, the same complement region cannot be interpreted identically in the two datasets.

**中15.** PANDA 将腺腔标注为间质，而 GLaS 将腺腔空间包含在腺体标注之中，因此两个数据集中的相同补集区域不能采用同一种解释。

**E16.** The refined implementation therefore combines colour-based low-content evidence, connected-region structure and nucleus density to identify candidate luminal regions, while allowing lumina truncated by the patch boundary.

**中16.** 因此，改进后的实现综合使用基于颜色的低内容证据、连通区域结构和细胞核密度来识别候选腺腔，同时允许腺腔被 patch 边缘截断。

**E17.** Gland walls and protected luminal topology are then treated as invariants for edits that do not explicitly request gland destruction or remodelling.

**中17.** 对于没有明确要求破坏或重塑腺体的编辑，腺体壁和受保护的腺腔拓扑随后被视为不可破坏的约束。

**E18.** This prevents cell scatter from being placed into empty lumina and prevents tumour-retreat operations from being realized as arbitrary epithelial-wall thinning.

**中18.** 这可以避免散落细胞被错误放入空腔，也可以避免肿瘤退缩被实现为任意削薄腺体上皮壁。

### Paragraph 5 — Closed-loop execution and composition

**E19.** Each structured intent is first mapped to a bounded set of organ-compatible primitives and is then resolved against the current tissue and nucleus state by deterministic preflight checks.

**中19.** 每个结构化意图首先被映射到一组有限的、与器官兼容的 primitive，随后再通过确定性的预检规则，结合当前组织和细胞核状态完成解析。

**E20.** Candidate masks are accepted only when they satisfy the requested direction and magnitude, legal source-to-target transitions, spatial requirements and invariant-preservation gates.

**中20.** 候选 mask 只有在同时满足请求的变化方向与幅度、合法的源到目标转换、空间要求和不变量保护门控后才会被接受。

**E21.** Failure is retained as an explicit clarification, review or execution outcome rather than being hidden by substituting a different primitive.

**中21.** 当系统失败时，会明确返回需要澄清、需要审查或执行失败，而不是通过替换为另一个 primitive 来掩盖问题。

**E22.** For multi-intent instructions, the system constructs one program step per user intention and executes the steps transactionally: edit, validate, commit, re-analyse the updated state and only then continue.

**中22.** 对于包含多个意图的指令，系统为每个用户意图建立一个程序步骤，并以事务方式依次执行：编辑、验证、提交、重新分析更新后的状态，然后才继续下一步。

**E23.** This design preserves explicit order and prevents a later primitive from being planned against a stale pre-edit mask.

**中23.** 这一设计既保留了用户明确指定的顺序，也避免后续 primitive 基于已经过时的编辑前 mask 进行规划。

### Paragraph 6 — Evidence to report after the frozen run

**E24.** In the frozen mask benchmark, we will report per-primitive executability, direction and transition accuracy, off-target change, invariant violations and source-relative magnitude, stratified by organ and annotation profile.

**中24.** 在冻结的 mask benchmark 中，我们将按照器官和标注体系，分别报告每个 primitive 的可执行率、方向与转换准确性、非目标区域变化、不变量违规情况，以及相对于源区域的变化幅度。

**E25.** Representative panels will provide qualitative evidence of morphological distinction, but they will not replace cohort-level measurements or blinded pathology review.

**中25.** 代表性 panel 将用于展示不同形态操作之间的视觉区分，但不会替代队列层面的量化指标或盲法病理审查。

**E26.** The final manuscript should insert the frozen values for these endpoints here and report the natural-language Parser/Planner audit separately as an interface validation rather than as the primary biological result.

**中26.** 最终论文应在此处填入上述终点的冻结评估数值，并将自然语言 Parser/Planner 测试作为独立的接口验证报告，而不是主要生物学结果。

## Recommended use in the paper

Use E1–E23 as the conceptual Results narrative after light compression. Replace E24–E26 with measured values once the final mask cohort is frozen. The current manuscript's legacy claims of 18 primitives, 231 settings, 14,288 cases and 99.92% success should not be carried forward until the refined catalog has been rerun under an equivalent frozen protocol.

For the main paper, include only one short sentence about the language benchmark, for example:

> A bilingual template-and-paraphrase audit was used to verify that the instruction-only Parser preserved intent number, order, polarity and closed-ontology fields, and that the Planner selected only organ-compatible primitives (Supplementary Methods and Supplementary Table X).

Suggested Chinese interpretation:

> 我们使用中英文模板与释义改写测试，验证仅接收文本的 Parser 能够保留意图数量、顺序、否定关系和封闭本体字段，并验证 Planner 只选择与器官兼容的 primitive（补充方法和补充表 X）。

Do not present this synthetic audit as evidence of unrestricted clinical language understanding. If an independent, clinician-authored instruction set is later collected, that result can be upgraded to a short secondary Results paragraph.
