# Breast / Prostate / Colon Primitive 病理生理与执行区分度审计

审计分支：`golden/breast-all-primitives-glas-panda`  
审计日期：2026-08-23

## 结论口径

- 这里的 primitive 是受标注约束的反事实形态编辑，不等于从一张小 patch 直接推断患者真实的纵向疾病进展、疗效或预后。
- 当前启用的 42 个 organ × mechanism 项中，没有两项是“目标类别、宿主区域、组织转换、空间布局和后条件均相同”的无条件完全重复操作。
- 3 个关闭项应继续关闭：Colon/GLaS `tumor-burden-increase-v1`、Prostate/PANDA `architecture-progression-v1`、Prostate Pattern 3 的 `cohesive-boundary-expansion-v1`。
- 需要重点治理的条件性重叠有四类：
  1. Breast 的两种 cord primitive 临床含义过近；实际编译顺序不同，但目录层面的区分度偏低。
  2. Breast `cell-type-abundance-*` 若允许显式选择 neoplastic class，会与 `neoplastic-cell-abundance-*` 条件性重叠；应把 neoplastic 指令统一路由到后者。
  3. 三个器官的 `cellularity-*` 若落在单一细胞类别 patch，可能退化成 `cell-type-abundance-*`；应要求多类别组成或明确记录退化。
  4. Colon 的 scatter 与 small-cluster 都属于 tumor budding 生物学谱系，但分别控制单细胞与 2–4 细胞簇，操作并不相同。

## 病理依据

- Breast 新辅助治疗后，残余肿瘤床面积、残余浸润癌细胞密度以及散在多灶残余癌都是规范化疗效评估的重要形态变量；因此 footprint、cellularity、fragmentation 三类编辑有依据，但不能由此宣称 pCR 或临床获益。[CAP Breast protocol](https://documents.cap.org/protocols/Breast.Invasive_4.10.0.0.REL.CAPCP_R.pdf)、[International post-neoadjuvant pathology recommendations](https://pubmed.ncbi.nlm.nih.gov/26205180/)
- Breast 肿瘤芽生/小细胞簇位于浸润前沿并与侵袭性表型相关；TIL 的临床意义依亚型和空间位置而变，不能把任意 immune-cell 增减直接解释成疗效。[Breast tumor budding study](https://pubmed.ncbi.nlm.nih.gov/33669393/)、[International TILs Working Group](https://pubmed.ncbi.nlm.nih.gov/25214542/)
- Breast 肿瘤坏死可反映缺氧和侵袭性，但“坏死消退后直接变为活肿瘤”不是治疗愈合的标准表述，应解释为反事实的 viable-tumor repopulation。[Breast necrosis cohort](https://pubmed.ncbi.nlm.nih.gov/8410120/)
- Prostate Gleason Pattern 4 包括融合、筛状及形成不良腺体；Pattern 5 包括实性结构、cord/single-cell 等非成腺形态，所以 P4 与 P5 的边界扩张必须保持不同 fine label 和结构后条件。[ISUP 2014 consensus](https://pubmed.ncbi.nlm.nih.gov/26492179/)、[ISUP 2019 consensus](https://pubmed.ncbi.nlm.nih.gov/32459716/)
- 前列腺癌经雄激素剥夺后可出现缩小的肿瘤腺体被间质分隔、残余肿瘤体积和细胞密度下降，因此 footprint retreat 与 residual fragmentation 有治疗反应依据；完整腺体单元不应被任意切半。[ADT pathology study](https://pubmed.ncbi.nlm.nih.gov/8826921/)、[Post-neoadjuvant prostate residual disease study](https://pubmed.ncbi.nlm.nih.gov/27273062/)
- Colon tumor budding 的国际定义是浸润前沿的单个肿瘤细胞或不超过 4 个细胞的小簇；immune density 具有预后信息，但 GLaS 的 generic immune class 不能等同于临床 Immunoscore 的 CD3/CD8 测量。[ITBCC consensus](https://pubmed.ncbi.nlm.nih.gov/28548122/)、[Consensus Immunoscore validation](https://pubmed.ncbi.nlm.nih.gov/29754777/)

## 总表

“区分度”评价的是相对于最接近 primitive 的实际 mask 操作，而不是仅比较名字。

| Organ | Primitive / mechanism | 当前状态 | 病理生理或治疗意义 | 实际操作 | 与相关 primitive 的区分度与结论 |
|---|---|---|---|---|---|
| Breast | `tumor-burden-increase-v1` | 条件支持 / shadow | 未治疗进展或治疗后进展导致浸润癌总体占据面积增加 | 从认证外边界把 BCSS Stroma 2 转为 Tumor 1，并重建完整 class-1 肿瘤细胞 | **中**：与 cohesive expansion 共用边界生长机制，但本项以总体 burden/面积为终点；不完全重复 |
| Breast | `cohesive-boundary-expansion-v1` | 条件支持 / shadow | 黏连性肿瘤前沿局部推进 | 浅表、连续、多叶状 Stroma 2→Tumor 1 扩张，伴整实例细胞替换 | **中高**：强调局部前沿几何；区别于更广义 tumor burden |
| Breast | `invasive-cord-formation-v1` | 条件支持 / shadow | 狭窄肿瘤细胞 cord 向间质延伸，模拟浸润形态 | 先排布至少 6 个 class-1 细胞，主体 1–2 细胞宽，再从细胞轮廓派生 Tumor 支持 | **中低**：与下一项病理含义很近；本项是 cell-first 且宽度/细胞数约束更强 |
| Breast | `infiltrative-nest-cord-extension-v1` | 条件支持 / shadow | 从既有肿瘤边界形成锥形浸润突起 | 先生成窄而渐细的 Tumor 组织投影，再用 ProbNet 重建 class-1 细胞 | **低**：与 invasive-cord 目录语义近似；建议合并为一个 cord primitive 加 layout mode，或强制 nest-anchor 后再保留 |
| Breast | `peritumoral-tumor-nest-formation-v1` | 条件支持 / shadow | 邻近间质内形成较大的脱离肿瘤岛 | 新建 1 个不规则、脱离的 Tumor 1 岛，填入 6–12 个完整 class-1 细胞 | **高**：有组织标签且 ≥6 细胞；区别于 ≤4 细胞 bud/cluster 和单细胞 scatter |
| Breast | `peritumoral-neoplastic-scatter-increase-v1` | 条件支持 / shadow | 浸润前沿离散单细胞播散 | 组织标签不变；只在 Tumor 外侧认证 Stroma 环带加入分离的完整 class-1 单细胞 | **高**：single-cell；区别于小簇和带 Tumor 组织岛的 nest |
| Breast | `peritumoral-small-cluster-increase-v1` | 条件支持 / shadow | 浸润前沿多个小型 tumor buds | 组织标签不变；外侧环带形成多个 2–4 个 class-1 细胞的紧凑小簇 | **中高**：与 scatter 同属 budding 谱系，但布局和每灶细胞数不同 |
| Breast | `neoplastic-cell-abundance-increase-v1` | 条件支持 / shadow | 既有肿瘤区内癌细胞密度升高，可代表增殖/较高残余癌细胞量 | 不改组织；在一个有界 Tumor 1 区内跨多个局部位点加入完整 class-1 细胞 | **高**：区别于 tumor-burden（不扩组织面积）和 scatter（不放到肿瘤外） |
| Breast | `invasive-tumor-footprint-decrease-v1` | 条件支持 / shadow | 治疗后浸润癌占据面积缩小 | 一条长而浅、不规则的外沿 Tumor 1→Stroma 2 退缩；保留残余肿瘤面积下限 | **中高**：连续外沿退缩；区别于 fragmentation、ROI clearance 和 stroma replacement |
| Breast | `neoplastic-cell-abundance-decrease-v1` | 条件支持 / shadow | 治疗后残余癌细胞密度下降但肿瘤床/组织范围未必缩小 | 组织像素完全不变；只按局部密度梯度移除完整 class-1 细胞 | **高**：区别于 footprint decrease 的组织面积变化 |
| Breast | `residual-tumor-fragmentation-v1` | 条件支持 / shadow | 治疗后残余浸润癌呈多个散在灶 | Tumor 1 内形成弯曲、宽度不一的 breakup 区并转为 Stroma 2，同时保留多个有间距的残余岛 | **高**：内部断裂并保留多灶；不是连续外沿退缩 |
| Breast | `stroma-increase-v1` | 条件支持 / shadow | 治疗、修复或纤维化相关的间质替代的操作代理 | 在合法原生界面做 Tumor 1→Stroma 2，并把肿瘤细胞替换为 stromal-compatible 细胞 | **中**：与 footprint 都做 Tumor→Stroma，但本项以“间质替代”身份和界面为核心；BCSS generic stroma **不能**诊断纤维化 |
| Breast | `local-invasive-clearance-v1` | 条件支持 / shadow | 局部消融/切除或指定局部治疗清除的反事实 | 只在用户提供 ROI 内 Tumor 1→Stroma 2，并移除 ROI 内完整 class-1 细胞 | **高**：空间权威来自用户 ROI；不能宣称全病灶清除或 pCR |
| Breast | `necrosis-appearance-v1` | 条件支持 / shadow | 缺氧/快速生长相关肿瘤坏死增加 | 仅从既有 Tumor–Necrosis 接触处将 Tumor 1 转入相邻 Necrosis 3，并使用稀疏坏死相容细胞群 | **高**：接收标签为 Necrosis，不是 Stroma，也不是单纯去细胞 |
| Breast | `necrosis-resolution-v1` | 条件支持 / shadow | 更准确应解释为“活肿瘤向既有坏死区再占据”，不是治疗性愈合 | 从既有接触界面把 Necrosis 3 转为 Tumor 1，并重建完整 class-1 细胞 | **中**：是 necrosis appearance 的方向逆转；名称有生理歧义，建议后续改名为 viable-tumor-repopulation |
| Breast | `generic-immune-infiltrate-increase-v1` | 条件支持 / shadow | 大片免疫浸润 compartment 扩大；可能与抗肿瘤免疫或炎症有关，但不能直接推断疗效 | Stroma 2→Immune infiltrate 4；class-3 connective 细胞替换为完整 class-2 inflammatory 细胞 | **高**：改变组织 compartment；区别于只增加若干 immune cells 的 cell-type abundance |
| Breast | `generic-immune-infiltrate-decrease-v1` | 条件支持 / shadow | 免疫浸润 compartment 退缩 | Immune infiltrate 4→Stroma 2；class-2 inflammatory 替换为 class-3 connective | **高**：组织级逆过程；非单纯细胞计数减少 |
| Breast | `cell-type-abundance-increase-v1` | 条件支持 / shadow | 指定一种可观察细胞群局部增多；意义取决于细胞类型 | 组织不变；在相容有界 component 中加入一种显式类别的完整细胞 | **中**：若类别是 neoplastic 会与专用 neoplastic primitive 重叠；应将 class-1 指令统一路由到专用项 |
| Breast | `cell-type-abundance-decrease-v1` | 条件支持 / shadow | 指定一种可观察细胞群局部减少 | 组织不变；沿有界梯度只移除该类别完整实例 | **中**：同上，neoplastic class 存在条件性重叠 |
| Breast | `cellularity-increase-v1` | 条件支持待病理复核 / shadow | 局部总细胞密度增加，而非某一谱系特异增殖 | 组织不变；按源 patch 观察到的多类别组成加入相容完整细胞 | **中高**：应保持多类别比例；若只有一个类别则会退化为 cell-type abundance |
| Breast | `cellularity-decrease-v1` | 条件支持待病理复核 / shadow | 局部总细胞密度下降，可作治疗效应/低细胞区代理 | 组织不变；按源组成跨类别移除完整实例，形成强核心到弱过渡的梯度 | **中高**：区别在 mixed-population；单类别 patch 上存在条件性退化 |
| Prostate / PANDA | `architecture-progression-v1` | **关闭** | 从低级别腺体结构向更高级 Gleason architecture 转变 | 设计上需执行 fine-label 转换、完整 gland-unit 与 lumen 后条件；当前生成器未验证，因此不执行 | **应关闭**：不是简单边界膨胀；开启会把级别变化与面积增长混淆 |
| Prostate / PANDA | `cell-type-abundance-increase-v1` | 条件支持 / shadow | selected stroma 内 connective/fibroblast-like 细胞增多，反映 reactive stroma 的组成变化，不等于肿瘤细胞增多 | 组织不变；在外部 cellular stroma 中加入完整 class-3 connective 细胞，避开 lumen/空白 | **高**：目标 class-3；区别于总 cellularity 和 class-1 neoplastic abundance |
| Prostate / PANDA | `cell-type-abundance-decrease-v1` | 条件支持 / shadow | connective 细胞局部减少 | 组织不变；只移除足量完整 class-3 实例，并扩大可见 change region | **高**：只针对 class-3；不是任意细胞减少 |
| Prostate / PANDA | `cellularity-increase-v1` | 条件支持 / shadow | 局部混合细胞密度增加 | 组织不变；按源组成加入多个相容类别的完整细胞，避开 lumen 与非组织空白 | **中高**：mixed-class；不同于只加 connective 或 neoplastic |
| Prostate / PANDA | `cellularity-decrease-v1` | 条件支持 / shadow | 局部混合细胞密度降低 | 组织不变；跨源观察类别移除完整实例并形成局部密度梯度 | **中高**：mixed-class；单类别 patch 时需标记退化 |
| Prostate / PANDA | `neoplastic-cell-abundance-increase-v1` | 条件支持 / shadow | 既有肿瘤区内肿瘤上皮细胞密度升高 | 组织不变；仅在 Tumor/pattern-compatible 区加入完整 class-1 细胞 | **高**：不扩张 tumor footprint，也不向 stroma scatter |
| Prostate / PANDA | `neoplastic-cell-abundance-decrease-v1` | 条件支持 / shadow | 治疗相关残余肿瘤细胞量下降但组织轮廓保持 | 组织不变；只移除 Tumor 内完整 class-1 实例 | **高**：区别于 footprint retreat 的组织转换 |
| Prostate / PANDA | `local-invasive-clearance-v1` | 条件支持 / shadow | 指定局部消融/清除的反事实代理 | 只在显式 ROI 内清除 Tumor 并置换为相容 stroma；ROI 外保持稳定 | **高**：ROI 驱动；不能等同于整体治疗反应 |
| Prostate / PANDA | `invasive-tumor-footprint-decrease-v1` | 条件支持 / shadow | ADT/放疗等治疗后可见肿瘤占据范围退缩 | 仅沿 lumen-distant 的 solid 接收边界做长而浅的 Tumor→Stroma 退缩；完整 lumen-associated gland component 像素稳定 | **高**：外沿退缩且保护完整腺体壁；不再允许“削薄腺体壁” |
| Prostate / PANDA | `residual-tumor-fragmentation-v1` | 条件支持 / shadow | 治疗后残余肿瘤以多个被间质分隔的灶存在 | 仅在 solid、lumen-distant 区形成 breakup，保留多个残余 Tumor focus；完整 lumen-associated gland unit 原子保护 | **高**：产生多残余灶；不得把完整腺体左右切开 |
| Prostate / PANDA | Pattern 3 `cohesive-boundary-expansion-v1` | **关闭** | Pattern 3 的生长应表现为新增分离、完整、成形良好的腺体 | 当前通用界面 band 无法可靠插入 whole-gland units，因此不执行 | **应关闭**：与 P4/P5 边界生长不能共用同一几何实现 |
| Prostate / PANDA | Pattern 4 `cohesive-boundary-expansion-v1` | 条件支持 / shadow | 融合、筛状或形成不良腺体成分的局部扩展 | 将相邻 stroma 转为 Pattern-4 fine label；保持 pattern-compatible 连续性和原生 lumen/internal spaces | **高（对 P5）**：虽复用 cohesive executor，但目标 fine label 与 gland/lumen 后条件不同，不是完全重复 |
| Prostate / PANDA | Pattern 5 `cohesive-boundary-expansion-v1` | 条件支持 / shadow | 非成腺的实性 Pattern-5 肿瘤边界推进 | 将相邻 stroma 转为 Pattern-5 fine label，形成连续、非成腺、细胞化的 solid front | **高（对 P4）**：同一 primitive ID 的不同 organ mechanism；实际目标标签与形态后条件不同 |
| Prostate / PANDA | `infiltrative-nest-cord-extension-v1` | 条件支持 / shadow | GP5 可见 cord/cluster/single-cell 型浸润前沿 | 从 Pattern-5 边界做较大、圆滑、不规则、连续且包含多个细胞的窄 cord；保护 lumen 和完整 gland unit | **高**：区别于 cohesive sheet expansion；不是三角形小突起 |
| Prostate / PANDA | `peritumoral-neoplastic-scatter-increase-v1` | 条件支持 / shadow | GP5 单细胞/离散小尺度浸润的操作代理 | 组织 fine label 不变；只在 Pattern-5 外侧、真实 cellular stroma 内加入稀疏分离 class-1 细胞，排除 lumen、白色背景和玻片空白 | **高**：cell-only 且离散；区别于 cord 的连续 Tumor 支持 |
| Colon / GLaS | `tumor-burden-increase-v1` | **关闭** | 腺癌 gland-forming front 扩张本有疾病进展意义 | GLaS 把 gland epithelium 与 lumen 同标；当前没有足够可靠的显式 stroma/whole-gland-unit authority，因此不做组织级扩张 | **应关闭**：否则容易填 lumen、连接腺体或把裂隙误当 stroma |
| Colon / GLaS | `cell-type-abundance-increase-v1` | 条件支持 / shadow | generic immune-cell 局部增多；可表示免疫微环境增强，但不能等同 Immunoscore 或疗效 | 组织不变；在合法非 lumen 区加入完整 immune class-2 细胞 | **高**：只改 immune class；区别于 total cellularity 和 neoplastic abundance |
| Colon / GLaS | `cell-type-abundance-decrease-v1` | 条件支持 / shadow | generic immune-cell 局部减少 | 组织不变；只移除足量完整 immune class-2 实例 | **高**：目标类别固定，非总细胞减少 |
| Colon / GLaS | `cellularity-increase-v1` | 条件支持 / shadow | 局部总细胞密度增加；可反映肿瘤/间质/炎症综合变化，但不是特定进展事件 | 组织不变；按源组成多类别加入完整细胞，并排除 confirmed/uncertain lumen | **中高**：mixed-class；与单类 abundance 的区别依赖多类别约束 |
| Colon / GLaS | `cellularity-decrease-v1` | 条件支持 / shadow | 局部总体低细胞性，可作治疗效应或组织稀疏化代理 | 组织不变；跨类别移除完整实例并形成可见局部梯度 | **中高**：mixed-class；单类别区域存在条件性退化 |
| Colon / GLaS | `neoplastic-cell-abundance-increase-v1` | 条件支持 / shadow | 既有恶性腺体上皮范围内肿瘤细胞密度升高，而非腺体变大 | 组织不变；只在 malignant-gland 的合法 epithelial host 中加入 class-1，three-layer observer 的 confirmed/uncertain lumen 均为保护区 | **高**：区别于关闭的 gland footprint growth 和腺体外 budding |
| Colon / GLaS | `neoplastic-cell-abundance-decrease-v1` | 条件支持 / shadow | 既有恶性腺体范围内肿瘤细胞密度降低 | 组织不变；只移除 malignant-gland host 中完整 class-1，保留 gland/lumen 组织标签 | **高**：不缩小 gland，也不把腺体壁切开 |
| Colon / GLaS | `peritumoral-neoplastic-scatter-increase-v1` | 条件支持 / shadow | ITBCC tumor budding 谱系中的单个肿瘤细胞 | 组织不变；在 typed gland exterior annulus 的 external cellular stroma 中放置多个相互分离的 class-1 单细胞，排除 lumen/低细胞空腔/背景 | **中高**：与 small-cluster 同病理谱系，但本项严格控制 single-cell foci |
| Colon / GLaS | `peritumoral-small-cluster-increase-v1` | 条件支持 / shadow | ITBCC tumor budding 谱系中的 2–4 细胞小簇 | 组织不变；在同类外侧 cellular stroma 环带放置多个紧凑 2–4-cell foci | **中高**：生物学意义相同、形态操作不同；也可合并成一个 budding primitive 的 layout 参数 |

## 最终重复性判断

| 相关组 | 是否“完全一致” | 判断 |
|---|---:|---|
| Breast tumor burden vs cohesive boundary expansion | 否 | 共享边界生长家族，但面积目标和局部几何不同 |
| Breast invasive cord vs infiltrative nest/cord | 否，但区分度低 | cell-first 与 tissue-first 不同；目录语义建议合并或增加 nest-anchor 硬约束 |
| Breast/各器官 cell-type abundance vs neoplastic abundance | 条件性 | 当显式类别为 class-1 时会重叠；应由解析器统一路由到 neoplastic primitive |
| 各器官 cellularity vs cell-type abundance | 条件性 | 多类别时不同；单类别 patch 会退化，需 gate 或显式审计标记 |
| Breast/Prostate footprint decrease vs fragmentation | 否 | 连续外沿退缩 vs 内部 breakup 后的多灶残余 |
| Breast footprint decrease vs stroma increase | 否，但共享转换 | 都可 Tumor→Stroma；前者定义退缩几何，后者定义间质替代机制/界面 |
| Prostate Pattern 4 vs Pattern 5 cohesive expansion | 否 | 复用执行家族，但 fine label、成腺/非成腺结构和 lumen 后条件不同 |
| Colon scatter vs small cluster | 否，但同一疾病构型谱系 | 单细胞 vs 2–4 细胞簇；可保留为形态控制，也可合并为 budding + layout |

