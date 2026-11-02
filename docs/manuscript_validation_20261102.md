# 2026-11-02 实现与验收记录

修复基线为 `6f481a989b57648449340a17d17b8cda3c9c818f`。代码在独立分支 `codex/manuscript-validation-20261102` 实现。原始工作计划分支和服务器原工作目录未被改写。

[协议与实际版本表](../benchmarks/manuscript_validation_20261102/protocol.json) 已填写正式入口、主终点、实际权重哈希、后处理区域规则、评价标签、ROI、缺失状态及分析单位。协议当前未冻结，`formal_launch_permitted=false`；正式 30 WSI 尚未启动。

## 已提交的行为修复

| 问题 | 实现与验收范围 |
| --- | --- |
| F1 sham 与输出阶段 | Cross 固定路线直接调用真实生成；显式选择 `postprocessed_precomposite`，同时保留原始、后处理与合成输出。空 support 合成仍可按原语义返回源图，但该文件不能充当全局 sham 终点。正式入口另外核对源条件、prompt、seed、五种状态及批准的 joint handoff。 |
| 后处理区域 | 低染色保护记录实际配置、允许区域、核排除区域与应用 alpha mask 的哈希。正式入口使用独立的共同全视野后处理允许区域，保留批准的 joint generation support；二者不能混用。相同 policy 名称且作用区域不同的反例已测试。 |
| F2 缺失与零检测 | CellViT 缺少 raw JSON 时记录 missing、非成功退出，不写零 mask。消费者拒绝缺失状态和无完成凭据的空检测。续跑核对源图、模型、mask 内容及参数；每次检测使用独立 raw 结果目录，避免串用旧结果。 |
| F3 组织分割尾批 | 先建立待处理清单再执行分批，修复 A 待处理、B/C 已存在时漏 A 的反例；结束后逐文件检查，缺产物时非成功退出。 |
| F4 统计 | 操作标签在原始配对数据上按冻结独立单位交换，每次重新拟合留出方向；小数据枚举对照通过。旧分数符号翻转以其真实名称另报。相关性提供聚类 bootstrap 区间，去掉普通 IID p 值。 |
| 统一统计单位 | 表征脚本用 `--cluster-unit patient_id` 同时控制留出、置换、bootstrap 与相关性；缺失或矛盾映射直接拒绝。KID 使用现有 `--bootstrap-group-field patient_id`，形态评价使用 `--group-field patient_id`；后者已移除静默退回 pair/sample 的逻辑。医生分析遵守同一协议单位，尚未开展。 |
| F5 实验身份 | 计算实际 ControlNet 权重、conditioning 和配置内容哈希；两个 handoff 生产端写源哈希，消费端重新核对。生成续跑绑定完整模型、代码、输入、目标、support、后处理区域、参数及队列身份，并验证输出内容；不匹配时保留旧产物并拒绝复用。 |
| F6 Inpaint 参数 | seed、steps、guidance、conditioning scale 和 dtype 接入实际 bundle；VAE posterior 和扩散噪声共用显式 generator。缓存区分采样参数和配置/conditioning 内容；重复 seed 的 VAE 抽样及参数传递测试通过。真实 GPU Inpaint 仍待验收。 |
| F7 联合变化范围 | 通用入口默认计算组织与细胞联合变化，并拒绝遗漏实际变化的外部 semantic region；原有 joint 门槛保持。 |

针对性测试位于 [test_manuscript_validation.py](../tests/test_manuscript_validation.py)，另有原生产权重测试中的同长度篡改权重反例。合成数据和 mock 仅证明控制流与实现行为，不是论文结果。

## 首轮回归与基线问题（历史记录）

- 相关入口、评价、权重、采样、后处理、handoff 和消融的针对性回归通过；精确数量与代码身份在 [机器可读验收记录](../benchmarks/manuscript_validation_20261102/validation_report.json) 中记录。
- 扩大联合编辑回归：211 项通过、334 个子测试通过；14 项已知失败单独排除于这次通过数之外。
- 这 14 项已在未经修改的 `6f481a98` 上重现：12 项为过时乳腺机制/primitive fixture、1 项为 planner 场景 mock 不完整，另 1 项为非乳腺 cellularity decrease 的 `selected_research` / `abstained` 预期差异。详见 [逐项清单](../benchmarks/manuscript_validation_20261102/baseline_failures.json)。它们没有被算作通过，也没有通过放宽研究门槛来消除。
- 首次扩大回归因人工中断，只能作为发现问题的记录；上述 211 项来自后续完整结束的回归。两处新增源哈希接线引起的正向测试失败已修复，相关 handoff 测试单独通过。

## 非正式链路与正式启动条件

技术 pilot 固定选择历史 manifest 中 sample_id 字典序第一项，生成 sham 与一个编辑状态，再测量原图及两张生成图。该来源排除于后续正式主队列，不按 pilot 表现调整终点或选病例。历史目标只用于接线，不构成新 joint 正式队列的验收。

服务器独立导出目录为 `/home/lyw/wqx-DL/flow-edit/manuscript-validation-20261102`，研究输出位于 `/data1/zhao/wqx/benchmarks/runs/manuscript_chain_pilot_20261102_v2`。实际生成快照为 `f7b47139336f9d1e5f85003da3bf852c223cdf6f`，测量快照为 `8ed6882fd35e907cb5132f29676a4fac52025ab4`。两张真实生成图及三张图的组织/细胞测量均已完成，三份细胞结果均为有效非零检测。生成环境缺少 MMSegmentation 导致首次分割失败后，使用独立分割环境并校验原有输出哈希继续测量，旧失败日志保留。代码与环境按阶段记录，不能用当前分支名替代真实运行版本。v1 在生成前为修正标签映射而主动终止，旧目录保留；不计为生成失败或论文结果。

尚须冻结适用于两种操作的正式 joint 目标及 ROI、核查所选来源的训练重叠，并完成最终运行版本的完整验收，才能把协议转为 frozen。历史 2,400 张图的统计尚未重算；本次没有覆盖历史报告。医生研究仍放在最后。

## 第二轮审查后的修复与验收

实现快照为 `6b85b70c138e4e85709f4468dbe9f79f1a3c6b7a`。[本轮记录](../benchmarks/manuscript_validation_20261102/round2_validation_report.json) 与日志另存，首轮 pilot 与测试日志保留。

- 正式操作从协议读取：u1 为 `cohesive-boundary-expansion-v1` / `breast-annotation-anchored-boundary-growth`；u2 为 `generic-immune-infiltrate-increase-v1` / `breast-generic-immune-compartment-turnover`。两档绑定批准 handoff、预算与目标哈希；在固定 ROI 检查相对源图的方向及两档目标剂量，拒绝重复或逆向目标。这不限制生成 H&E 出现并列或逆序。
- 必需资产采用 `{status, path, sha256}`。入口核对来源、患者/WSI、pilot 排除记录、protected ROI、实际 prompt，以及仓库根目录的 HEAD、未提交/未跟踪文件、解释器和冻结依赖版本。把标志改为 frozen 不足以启动。协议中目标、ROI、映射与训练核查资产仍 unresolved；两档区间和最小剂量差仍待生成前冻结。
- CellViT 的独立 completion 文件绑定检测 JSON；仅修改 cells 列表不能复用。Segmentator 复用须解码、核对尺寸和标签，并验证输入、checkpoint、配置及输出凭据；旧无凭据文件须另存重测。
- 两个生成入口在首个产物写入前记录 started 身份，失败保留 failed 状态；非空失败目录不能当新任务。技术重试使用新目录。
- 效应量和区间改为先对患者/WSI 内来源求均值，再让独立单位等权，与置换观测统计量一致；报告显式输出该观测量。历史结果尚未重算。
- 12 个早退 fixture 已迁移到当前支持机制，原预算和权限断言实际执行；补齐 planner mock。cellularity 的当前 selected_research 行为增加了组织不变、12–15 个实例删除、区域外不变和 gate 通过的后置断言。生产门槛与旧机制关闭规则未放宽。

完整回归 **316 项、346 个子测试通过，无 deselect**；最终补充反例 **10 项通过**（与完整回归部分重叠，不相加计数）。这轮未重跑 GPU，也未启动正式队列。原技术 pilot 的 165/177/188 是总细胞检测数；两张生成图的低染色保护均未实际触发混合，不能当成炎症终点或保护效果验证。正式 joint 两操作、两强度与保护 ROI 的完整链路仍待验收。

## 第三轮：ROI 合同一致性

代码快照为 `4c61dcb5f0fb0fef425eb2a78c3b33986b1a1738`，证据另存于 [ROI 验收记录](../benchmarks/manuscript_validation_20261102/round3_roi_validation_report.json)。本轮相关集合 **99 项通过**，包含 6 个新增 ROI 场景；没有重新运行整个仓库或 GPU。

测量 ROI 必须逐像素等于源 mask 经 fine→coarse 映射后的协议区域：U1 为有效组织，U2 为 Stroma ∪ Immune。哈希正确的局部裁剪也会拒绝。保护区域则从四个已验证 handoff 的 `joint_change` 取并集，再膨胀 8 像素后从源有效组织排除，保留完整 native footprint 和 nuclei 编码统一语义。入口通过全部检查后，保存每个 ROI 的像素分母、形状、文件哈希和二值内容哈希。

读出 helper 对源 255 作 ignore，允许目标在固定 ROI 外保留 255，在 ROI 内则拒绝计量。实际 HTE 生成器只接受 0–15，因此正式源/目标生成条件中的 255 会在模型调用前拒绝；必须先准备合法且单独冻结的条件资产，不静默改写批准目标。

完整五输出正向测试使用实际 joint ledger 和 handoff 校验，并经过结构化资产、干净 fixture 仓库、环境及正式入口检查；ML 依赖与生成明确使用 CPU stub。这证明接线与校验行为，不等于最终 joint GPU 验收。原始像素差异与批准 J 不等价的保护区、哈希正确但定义错误的测量 ROI，均在生成调用前停止。

正式资产与剂量规则仍未冻结，30 WSI 继续暂不启动。历史统计按独立单位等权重算时，点估计也可能变化，须在旧新差异表中注明；本轮未重算这些结果。医生研究仍放在最后。

## 资产准备：用户固定 19%／32%

用户在新目标与 H&E 生成前明确选择沿用历史的整图 joint 预算：Moderate 19%，Significant 32%。对应范围固定为 14%–24% 与 27%–37%，组织变化下限分别为 14% 与 27%，严格容量下限不降低。ROI 剂量必须相对源图增加，两档目标至少相差 0.05；这些是标注条件预算，不是临床分级。

初始脚本保存的 30 个独立患者输入只是候选，不是已经验收的正式队列。原技术源的 Significant 容量仅 68,645 像素，低于 70,779 的下限；下一技术源虽通过必要容量检查，完整拓扑仍只能实现约 49,148 像素。两次未完整结束的 U2 编译均保留超时与中间产物，不计为目标成功，不降低预算补齐配对。

因此改用既有 300 来源、90 患者的完整池逐项进行源 mask 资格检查，保留全分母。在 sample_id 顺序中，每位患者取首个通过容量、拓扑、四目标门槛与独立条件审查的来源；所有用于技术尝试的患者均排除。没有按生成 H&E、表征或测量表现挑选。必要容量或单个拓扑 witness 通过均不能替代完整四目标批准。源池、映射和训练清单核查的实际文件留在服务器，公开记录仅含身份哈希与汇总。

300 个来源的保留清单核查：Cross train/val 精确文件名引用分别 270/30；Segmentator 原 coarse train 精确引用 268 个，患者组覆盖全部 300 来源；后续 fine train/val/test 精确引用分别 217/31/52。它们是保留清单中的引用证据，不证明权重的全部历史训练来源。Pix2pix 仍有历史配置到最终权重的完整谱系限制；FLUX 和 CellViT 缺少可逐患者核查的训练表，重叠未知。自动评价仍属于耦合计算读出，不是独立临床验证。

映射核对使用保留数据集的原 split-group 记录；其中 wsi_id 是患者/WSI 分组别名，完整 TCGA slide barcode 未恢复，不能称为已核实完整切片身份。最终独立单位固定为患者。

当前编译代码为 `18258d49` 的真实、干净 Git 导出；新增池清点与审计脚本版本另记，不能把后续分支 HEAD 当作已运行版本。真实 mature ProbNet checkpoint 内容哈希为 `c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211`，实例库树哈希为 `f00d5c4b9afb3cae7ca2e22f41f698d51e36af4f3ad216599d47bb7b058433e4`，均来自实际执行 trace。

第三技术来源通过两个 Significant 源拓扑 witness 后启动完整四目标编译；已完成的 U1 Moderate 实现 joint 18.739%，硬性检查通过，确定性 critic 按设计返回 review_required，尚无批准 handoff。其他目标的运行状态须以终态记录为准。最终 joint GPU 技术验收和正式 30 WSI 均未启动；本节是资产准备进展，不能当作验收通过记录。

全池必要容量检查已完成：**300/300 来源检查结束，74 来源、43 患者通过两操作 Significant 必要容量，226 来源未通过，无执行异常**。结束后 900 个输入文件重新计算内容哈希，全部仍与冻结 census 一致；这份后验核对另存，不能改写成旧批次在读取前运行了新增哈希检查。完整拓扑与四目标资格仍待完成，74 不是正式配对数。

第三技术来源两个 Moderate 均为 review_required；两个 Significant 在 1,800 秒控制时限内未完成，退出 124。保留中间产物和独立控制记录，不伪造缺失的 producer result。另一个已通过源拓扑的技术来源也在相同预算和 1,800 秒时限下编译；所有四个尝试患者均进入患者级排除表。

为最终 GPU 技术验收补上同一入口的 `--technical-validation` 模式：必须使用单独 frozen 技术协议，scope 为 joint_gpu_technical_acceptance，formal_launch_permitted 必须为 false，来源仅 1–2 个且同时属于预留技术名单和源/患者排除表。继续执行同一 handoff、剂量、J 保护 ROI、干净代码、环境和真实模型内容检查；写出 scope 与协议/manifest 哈希记录。正式模式也拒绝来自技术患者的其他 patch，缺少患者排除字段时停止。主正式协议仍未冻结。

这轮 **42 项相关 CPU 检查通过**，包括 7 个技术范围场景和真实子进程超时保留中间产物的反例；这些 CPU 检查不是 H&E 生成／评价的 GPU 验收；真实 ProbNet 条件编译另外留痕，不计为正式结果。[资产准备记录](../benchmarks/manuscript_validation_20261102/asset_preparation_report.json) 与日志单独保存。

用户明确委托当前 Codex 聊天独立审查 mask 条件图。新增独立模型审查入口与原有 user-only 批准入口分开，保留原入口的严格用户身份校验。模型审查绑定实际源输入、全部展示的 passing candidates、gate/plan 文件和 board 哈希，并从源/目标 mask 重建 board 逐像素核对；不读取原始 H&E，不生成批准默认值。只有已记录的 approved 决策、至少 0.7 confidence 和完整观察说明可进入原有严格 handoff 检查；新增/失效 hard gate、目标或证据变更、伪造 board 均拒绝。provenance 标记 independent_critic_review，不写 user_approval 或医生批准。**55 项相关 CPU 测试通过**；其中端到端持久化重构测试明确使用合成 gate fixture，不能当作实际病理/生成验收。

初始池的源拓扑检查已证明：即使剩余或异常来源全部可能合格，排除技术患者后仍达不到 30 位患者。用户已明确选择从同一 BCSS 数据集补充来源，保留 30 位患者目标及 19%／32% 预算。旧 300 来源的全分母和失败记录保留；新增池必须在任何新 H&E 生成前按源条件规则单独冻结和留痕。

初始池最终 74/74 个必要容量合格来源完成两操作源拓扑检查：**14 来源、10 患者通过；排除全部技术患者后，只剩 10 来源、8 患者，且尚未通过完整四目标与模型审查**。无超时或异常结果冒充失败；完整控制记录已另存为不可变快照。原池不能提供正式 30 位患者。

两个实际 Moderate 条件的独立 Codex 决策已绑定此前确实查看的 mask-only board 哈希，通过重建、原有 ledger/实例/合同/S>J 校验后输出 handoff，并经实际消费端重新核验。晋升与消费代码为 `3d7088ae`；没有调用图像生成。模型记录不带 user_approval。另一个技术来源两个 Moderate 为 review_required，U2 Significant 因 deterministic_replan_stalled abstained，U1 Significant 超时，均未被补齐为成功配对。第三来源的两个超时 Significant 仅各做一次同输入运行时重试，时限 7,200 秒，不是看过 H&E 后的质量重试。

用户授权的同 BCSS 补充 census 在 `d4ed240f` 冻结：原导出 **22,870 张 patch、151 患者组**，每位非技术患者按文件名取首 12 个满足廉价源条件的新增来源，得到 **1,219 来源、115 患者组**。廉价条件只包含已知 tissue/nucleus ID、512 对齐、已有 fine Tumor1/Stroma2/Immune4，以及至少 27% Stroma 的必要下限。原 300 和其失败不删除；技术患者整组排除，达到 cap 后的未检查来源单独记录，不能叫作资格失败。新 census 仍需两操作容量、拓扑、四目标和独立模型审查，不是正式队列。两项新增 census CPU 测试通过。

第一批补充 census 的 1,219 个来源完成必要容量检查，145 来源、68 患者通过，容量检查没有未知结果；随后 145 个源拓扑检查中 5 来源、5 患者通过，另有 **2 个 1,200 秒超时保留为 unknown**，不当作形态失败。原池与第一批仍不足 30 个完整来源，因此在用户已授权的同一 BCSS 范围继续按固定顺序追加剩余廉价条件合格来源，旧池不删除，不按 H&E/读出筛选。

第三来源的 U1 Significant 运行时重试完成，实际 J 为 31.8493%；Codex 在真实 mask-only board 上记录角状 sector 边界限制，选择候选 01，经原有硬门槛和精确合同检查晋升并通过消费端校验。两个 Moderate 加这个 Significant 共三个条件可用，U2 Significant 仍未完成。另按完整初始 census 的资格顺序预留第五技术患者并编译四状态，全部五位技术患者均排除于正式分析；旧四患者排除快照不覆盖。

新增实际技术资产与测量入口，要求四个已审查、同源、预算精确一致的 handoff，再冻结 J 并集保护 ROI、共同全幅后处理、实际 checkpoint 内容、CellViT 实现包和两套 Python/依赖。冻结协议仅限技术来源，formal_launch_permitted 保持 false。测量端拒绝 source_composite 冒充 global sham，核对实际 endpoint/seed/权重/alpha 与 nuclei-exclusion 区域，按固定源 ROI 计组织分母和 type-2 局部多边形面积质心；密度明确使用 nominal 0.25 um/px，不宣称恢复了逐切片物理校准。缺失、失败、有效零和不可定位的 uncertain 分开记录，不要求图像读出单调。

还消除了 CellViT 转换对 live 原仓库的隐式导入：改为读取当前执行快照的确定文件路径。当前原仓库与快照该文件内容哈希相同（6c210d1b…），转换算法未改；新增反例证实外部 namespace 缓存不能串入。**75 项相关 CPU 检查通过**，包括真实后处理 mask 验证和明确标为合成的 producer 记账场景。尚未运行本轮五状态 H&E 生成与六图评价，不能把这些检查记为最终 GPU 通过。

第五技术来源的四个状态均通过硬门槛，Codex 已实际查看全部 passing-candidate mask-only board，记录独立决定并晋升四个 handoff。按 U1 Moderate/Significant、U2 Moderate/Significant 排列，实际 J 分别为 16.3239%、31.6746%、18.4662%、30.5122%。名义 19%／32% 没有改写；其中三个状态使用生成前已经声明的 batch-max-safe 回退，严格组织下限 14%／27% 不降低，不宣称实际 J 等于名义目标。审查记录保留角状边界及 mask-only 局限，未赋予医生或用户批准。技术协议正在核对实际模型和环境，尚无本轮 H&E 验收结果。

在同一 BCSS 导出内追加全部剩余廉价源条件合格来源，第二补充 census 冻结 1,918 个来源、79 个患者组，保持旧 300 和第一批 1,219 的记录；五位技术患者整组排除。映射及训练清单引用审计完成，必要容量筛查正在运行；不能把新增候选数当作正式合格患者数。

## 最终 joint 技术 GPU 验收（单个排除来源）

真实生成和测量均使用干净快照 `20ace976e92b7d3e3de091590e4e2d12653f5def`，完整证据见 [本轮技术 GPU 记录](../benchmarks/manuscript_validation_20261102/final_joint_gpu_validation_report.json)。冻结的技术协议、四个审查目标、固定源 ROI、J 并集保护区、共同全幅后处理、真实权重和环境均经实际入口核对。五状态 H&E 全部真实生成；原图加五状态共六图完成 Segmentator 与 CellViT，七个测量控制进程均为退出 0，无 missing/uncertain 被改成零。本轮只验收链路与身份完整性，正式 30 位患者队列仍未冻结、未启动。

首次生成因缺少底层要求的显式 Pix2pix 环境选择器，五状态在模型调用前全部失败；旧失败目录保留。随后仅补全 `PATHOLOGY_PIX2PIX_CHECKPOINT`，指向协议已冻结的同一 epoch26/step214895 权重，在新目录重试；代码、目标、预算和采样未变化。重试为 5/5 完成，不能把首轮失败抹掉。

固定 U1 分母为 262,144 像素；sham/Moderate/Significant 的 Tumor 计数为 139,637/158,111/180,098。U2 固定源 Stroma∪Immune 分母为 193,237 像素；主组织读出为 **0/0/0 并列**。核对了实际共享标签定义（4=Immune infiltrate）和六张完整 coarse 输出，均没有 class-4 像素，未发现本轮消费者标签错配；这是该评价器在本技术来源上的实际输出，不声称源或生成组织没有免疫浸润。次要 type-2 ROI 计数为 10/42/45，总检测数另存，不能用次要结果替代 U2 主终点。原图 U1 预测与标注也有明显偏差，所有误差和 sham 漂移按原口径保留，不调评价器、不换技术来源。

五状态低染色保护实际 alpha 全零；本轮证明配置、区域和未触发分支完整，不能宣称已验证保护混合的效果。图像终点全部采用真实 `postprocessed_precomposite`，空 support 的源图合成文件没有充当 sham。独立单位、源分母及名义物理尺度均按冻结协议执行。本轮通过技术验收，不构成正式队列效应或临床有效性的结论。
