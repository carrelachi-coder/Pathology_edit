# mask edit refine

`phase3_mask_edit_refine` 是与现有 `phase3_mask_edit` 并列的 Architecture-B 实现。它只通过
`tool_adapters.py` 调用旧包中的确定性绘制函数，不修改旧入口、旧配置或旧产物。

## 当前安全状态

- 默认模型路由：Terra Planner、Terra Critic；可选 Sol 单次升级。
- 工作流：场景图 → 多模态结构化规划 → 12 个确定性候选 → fail-closed gates → 独立 critic → 选择或拒绝。
- `pathology_domain_id` 与 `annotation_profile_id` 均为必填，禁止由 organ、dataset 或旧 profile 字段互相推导。
- catalog 中的 15 个 skills 当前均为 `draft`。原因是完整训练集统计、case 152/175 原始 run bundle 与内部病理审核尚未挂载。生产模式会拒绝执行；只有 `--research` 可用于 shadow/evaluation。
- heuristic provider 不具备病理视觉审查能力。若 active hard rules 需要视觉 veto，它会拒绝最终选择；测试中的 fixture critic 仅用于验证编排。

## 输入

以 `configs/mask_edit_refine/case_context.example.json` 为模板创建 case 文件。输入必须包含真实的
source image/mask SHA256、显式双轴 metadata、primitive、面积预算和 seed。

面积预算由调用方提供，Planner 无权修改。`relative_tolerance` 默认是目标像素数的 2%；
`min_fraction/max_fraction` 可表达任务本身允许的范围。`fallback_policy=exact` 要求目标面积；
`fallback_policy=max_feasible_below_target` 则先尝试 `target_fraction`，若其违反合法界面、source
保留或拓扑约束，就在不低于 `min_fraction` 的前提下选择最大已验证安全面积。系统记录 desired、
resolved、差值、搜索点和 binding constraint；低于硬下限时仍然拒绝。旧输入未显式携带该字段时，
非退化的 `min_fraction < target_fraction` 会按后一策略解析，精确预算仍保持 `exact`。

## Skills 与 references 如何组合

每个 skill 的 `SKILL.md` 只规定加载顺序和执行动作。运行时先加载同目录的
`references/mask_contract.json`，再按需加载 `references/rules.json`：前者只放 mask
阶段可观察、可执行或必须显式 abstain 的约束，后者放病理解释、视觉反例和生成阶段要求。
系统独立加载一份 pathology-domain、一份
annotation-profile 和一份 edit-primitive，再取三者能力交集：

1. annotation profile 决定当前 mask 实际有哪些标签、零标签的语义、必须携带的 remap/
   provider/site provenance，以及哪些细分组织绝对不能从粗标签反推。
2. pathology domain 根据 H&E 识别局部结构，并规定该癌种下结构应有和禁止出现的形态。
3. primitive 固定 source→target、面积语义、可调用工具和通用拓扑合同。
4. Planner 只能选择场景图中已存在的合法 interface/component ID，并在结构化计划中引用
   所有 active `constraint_id` 及相关 `rule_id`；工具只在 source label 和允许 band 内生成候选。
5. mask contract 明确记录 `observability`、`enforcement`、`enforcement_stages`、checker、
   failure action 和 `generation_handoff`。例如“保留 Gleason 4 fine ID”是 deterministic，
   “新增区域真的呈筛状结构”则交给生成与生成后审核，不能冒充 mask 保证。
6. 每条 hard rule/constraint 必须绑定确定性 checker 或视觉 critic veto。未实现 checker、缺 profile
   provenance、背景被当作种子、或 critic 未确认 hard visual rule 时一律拒绝。

例如 `breast-invasive-carcinoma-v1 + orca-semantic-v1` 是合法组合：乳腺 skill 负责乳腺
H&E 形态，ORCA profile 仍只允许 `Carcinoma / Other tissue / Non-tissue` 三类语义。
ORCA 的碎片 non-tissue 必须逐像素保持，不能作为 Tumor 或 Other tissue 的种子，不能被
形态学填洞，也不能被跨越后桥接两个组织组件。

## Planner 意图如何落到工具

`mask-edit-refine-plan-v2` 不再把自由文本 anchor 当作绘制指令。场景分析会把每条 directed
interface 拆成带可视编号的 `anchor_segment_id`，Planner 必须在每条计划界面中输出
`execution_contract`：

- 要执行的 `anchor_segment_ids`；
- 每条界面的 `area_allocation_fraction`，全计划必须精确求和为 1；
- tapered/uniform/multi-lobe 深度轮廓、端点 taper、lobe 数和受限 noise 参数；
- anchor 覆盖、off-anchor 接触及面积分配容差。

确定性 execution compiler 保留 Planner 选择的界面、anchor 和相对深度轮廓，并联合求解安全面积、
界面间实际分配和绝对深度。它先保护 source 的窄颈/残余面积，阻止接触未选择 target component，
再对 whole-mask component/hole topology 做验证；若 19% 不可行但任务允许 14–24%，则选不低于
14% 的最大安全可行量，而不是因达不到 19% 直接失败。求解器可以在 Planner 已选界面之间重分配
面积，但不能扩展到未选区域；重分配和零容量界面均写入 compiler audit。候选生成只能在选中
anchor 的 Voronoi 影响域内生长；相邻 anchor 会先合并为连续弧，只在真实外端 taper。

`execution_contract_fidelity` gate 会从最终像素重新计算每条界面的面积、anchor coverage、
off-anchor pixels、深度包络和组件数，不信任 tool trace 自报结果。Planner 不能放宽系统的 2%
面积容差、50% 最低 anchor coverage 或 2% 最大 off-anchor contact。

## 常用只读准备命令

```bash
python scripts/run_mask_edit_refine.py list-skills
python scripts/run_mask_edit_refine.py validate-skills
python scripts/run_mask_edit_refine.py verify-run-bundle \
  --manifest configs/mask_edit_refine/run_bundle_manifest.example.json
python scripts/run_mask_edit_refine.py verify-evidence \
  --manifest configs/mask_edit_refine/evidence_manifest.example.json
python scripts/run_mask_edit_refine.py profile-stats \
  --manifest /path/to/evidence_manifest.json \
  --split train \
  --output /path/to/read_only_statistics.json
```

`profile-stats` 会按完整 mask 计算标签覆盖、连通域、孔洞、邻接、周长面积比、背景碎片度、
内部背景比例、患者/WSI 数量和分位数。生成结果仍需内部审核后才能标记为
`internally_reviewed` 并接入 production skill。

## Research shadow run

```bash
OPENAI_API_KEY=... OPENAI_API_BASE_URL=https://api.openai.com/v1 \
python scripts/run_mask_edit_refine.py run \
  --case /path/to/case_context.json \
  --output-root /path/to/refine_artifacts \
  --research \
  --provider openai
```

API key 只从 `--api-key-env` 指定的环境变量读取，不应写入 case、config、命令历史或审计产物。
兼容服务可用 `--api-base-url` 或 `OPENAI_API_BASE_URL` 指定 endpoint。

没有合格候选、证据不完整、skill 未认证、profile 签名异常、模型失败或 critic 不确定时，
工作流会输出结构化 abstain；不会回落旧方案。

## 生产放行前置条件

1. 使用 `verify-run-bundle` 冻结并核验 case 152/175 的原始输入、target、raw Planner response、manifest 和代码快照。
2. 为 BCSS、PANDA、GLaS、IGNITE、PUMA、ORCA 挂载无患者/WSI split 泄漏的 evidence manifest，并生成全训练集统计。
3. 对 domain/profile/primitive rules 和统计产物完成内部病理审核，将状态提升为 `internally_reviewed`。
4. 用 150–200 例盲评运行 `score-eval`，所有 release gates 通过后才允许去掉 research 标记。
