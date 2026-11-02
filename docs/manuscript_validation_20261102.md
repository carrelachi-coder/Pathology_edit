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

## 回归与基线问题

- 相关入口、评价、权重、采样、后处理、handoff 和消融的针对性回归通过；精确数量与代码身份在 [机器可读验收记录](../benchmarks/manuscript_validation_20261102/validation_report.json) 中记录。
- 扩大联合编辑回归：211 项通过、334 个子测试通过；14 项已知失败单独排除于这次通过数之外。
- 这 14 项已在未经修改的 `6f481a98` 上重现：13 项与乳腺机制/primitive 范围及其旧测试预期有关，1 项为非乳腺 cellularity decrease 的 `selected_research` / `abstained` 预期差异。详见 [逐项清单](../benchmarks/manuscript_validation_20261102/baseline_failures.json)。它们没有被算作通过，也没有通过放宽研究门槛来消除。
- 首次扩大回归因人工中断，只能作为发现问题的记录；上述 211 项来自后续完整结束的回归。两处新增源哈希接线引起的正向测试失败已修复，相关 handoff 测试单独通过。

## 非正式链路与正式启动条件

技术 pilot 固定选择历史 manifest 中 sample_id 字典序第一项，生成 sham 与一个编辑状态，再测量原图及两张生成图。该来源排除于后续正式主队列，不按 pilot 表现调整终点或选病例。历史目标只用于接线，不构成新 joint 正式队列的验收。

服务器独立导出目录为 `/home/lyw/wqx-DL/flow-edit/manuscript-validation-20261102`，研究输出位于 `/data1/zhao/wqx/benchmarks/runs/manuscript_chain_pilot_20261102_v2`。实际生成快照为 `f7b47139336f9d1e5f85003da3bf852c223cdf6f`，测量快照为 `8ed6882fd35e907cb5132f29676a4fac52025ab4`。两张真实生成图及三张图的组织/细胞测量均已完成，三份细胞结果均为有效非零检测。生成环境缺少 MMSegmentation 导致首次分割失败后，使用独立分割环境并校验原有输出哈希继续测量，旧失败日志保留。代码与环境按阶段记录，不能用当前分支名替代真实运行版本。v1 在生成前为修正标签映射而主动终止，旧目录保留；不计为生成失败或论文结果。

尚须解决基线回归、冻结适用于两种操作的正式 joint 目标及 ROI、核查所选来源的训练重叠，并完成最终运行版本的完整验收，才能把协议转为 frozen。历史 2,400 张图的统计尚未重算；本次没有覆盖历史报告。医生研究仍放在最后。
