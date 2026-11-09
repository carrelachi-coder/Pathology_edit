# S07 有限诊断结案与 R3 后续证据边界

本次与 manifest `303944169cc5aefc848f61fbedf70eebfd40e5a17ead578c6fa34cd1ef799b4e` 绑定的授权已经消费并结案。Controller 返回 `mismatch_hold`，退出码 78。A 完成并通过历史完整业务 receipt 比较，B 在 1,200 秒门槛触发超时，C 未启动。`mismatch_hold` 是停止状态名，本轮没有取得 B/C 业务或语义不一致的证据。

## 已证实的执行结果

A 用时 113.0759 秒。B 的进程门禁记录 1,200.0242 秒，包括截止后的观测开销。整次启动用时 1,321.7760 秒。共保留四次逻辑 witness 的启动额度，不能据此声称四次 witness 全部结束。没有自动重试。

B 仅有部分提交产物。只读核查覆盖全部 234 个已索引块，逐块验证文件 SHA、压缩载荷 SHA、解压载荷 SHA 与逻辑行数，共 7,685,156 条事件。解压内容 981,452,916 字节，压缩载荷 73,908,422 字节，按唯一 inode 计算的文件分配为 76,107,776 字节。这些数字不含尚未提交的内存尾部，也不能预测完整 C 的容量。

部分流有一次 `qualification.begin`，没有 `qualification.complete`。因此 B 尚未记录完第一个操作，不能解释 U2 的真实观察结果。已有事件包含 96 次 `grow.begin`、95 次 `grow.end`、9 次 `compiler.attempt` 和 8 次 `compiler.exception`。这些是部分业务轨迹中的事件数量，不是额外的外层自动重试，也不能与缺失内部轨迹的 A 逐项比较。

Guard 相关事件共 7,684,542 条，约占该部分流的 99.992%。它们仍按原顺序逐条保存，没有抽样或合并。部分块完整，不代表业务过程完整。缺少最终业务 receipt、完整封存和完成标记，故不能发布接受证明。

## 当前能够与不能够定位的问题

本轮停止由执行时间门槛触发，已有文件的分配远低于 4 GiB。没有证据支持将这次停止归因为旧版 200 万事件上限或已发生的存储预算拒绝。

目前没有分离业务计算、Adapter 编码、Codec 检查、压缩及文件系统同步的耗时。因此不能将 1,200 秒归因于某个函数，也不能用 A 与 B 的总时长比证明随机轨迹或业务决策发生改变。B 未完成，更无法用它推算完整事件数或完成时间。

已有证据不足以解释源拓扑欠填的全部原因。需要的完整合法域、ownership、逐步 mask 与两个操作的终态仍未齐全。

## 下一阶段范围

保留本次目录及旧 HOLD，不重新运行 S07，不提高冻结运行的时限。工程定位可先使用独立合成事件或已保存字节做只读、无 witness 的计时分解，区分 Adapter 编码、Codec 校验、预算登记、压缩和持久化成本。任何新实现须继续保留全部逻辑事件、数组 occurrence、原始顺序、身份绑定和故障隔离。只有具体改进通过独立验收后，才准备新的真实诊断方案与授权。

R3 历史表征统计采用另一个独立工作流。它仍按患者聚类、保留 5,000 次 bootstrap 和 20,000 次原始操作标签置换，每次重新拟合方向。它不能替代正式 sham 配对形态终点。

患者资格仍为 17 位源拓扑通过、含 unknown 上限 19 位、完整四目标批准 0 位。30 位患者目标、19%／32% 名义预算、14%／27% 组织硬下限及最小两档差保持不变。正式 H&E、批量四目标和医生研究继续关闭。

## 归档证据

- `benchmarks/manuscript_validation_20261109/adapter_controller_v1/actual_s07_terminal_report_v1.json`
- `benchmarks/manuscript_validation_20261109/adapter_controller_v1/actual_s07_partial_storage_audit_v1.json`
- `benchmarks/manuscript_validation_20261109/adapter_controller_v1/actual_s07_committed_blocks_audit_v1.json`

## 本地独立成本分解

新增 `scripts/profile_adapter_synthetic_cost_v1.py` 只生成合成 `guard.stage`，不导入 planner、guard 或 witness。一次 100,000 事件的 cProfile 运行耗时 7.34 秒，全局 Python／NumPy RNG 保持不变。该测试故意不发布完成标记。

本地调用统计显示 `pathlib.parse_parts` 被调用 700,024 次，自身耗时约 1.40 秒；预算 `reservation` 被调用 600,012 次。预算路径处理可作为独立优化候选，但这不是目标服务器上的耗时归因。Profiler 本身也增加运行成本，不能把 7.34 秒线性外推为 S07 的预期完成时间。源码、调用次数和结果已归档，现有生产 Codec 与 Adapter 未改动。

## 独立预算候选

`diagnostic_storage/guard_budget_candidate_v1.py` 在独立文件中实现候选预算类。它直接读取已构造的具体 Path 的文件名，并在单次 claim 内复用同一个 reservation 计算结果。没有缓存磁盘状态，没有修改页取整、索引增长余量、控制文件预留、实际分配检查或超预算行为。原 Codec、Adapter 与服务器快照保持原样。

6 项差分测试覆盖三档预算下共 6,000 次预算动作、预算异常和状态、hash/full 模式完整产物逐字节相同及实际分配留痕。另在独立测试进程中将 Codec 的预算类替换为候选，现有 Codec 测试文件的 52 项测试全部通过。该替换只存在于测试进程。

同一 100,000 合成事件场景的单次本地剖析观测为原版 7.34 秒、候选 5.24 秒。两次有先后顺序，均带 profiler，缓存与并行统计计算负载未受控。这不是正式性能验收，不是服务器提速保证。候选尚未接入生产，也没有新的 S07 执行授权。

## 部分轨迹中的拓扑标量

只读提取已提交前缀中的全部 `execution.works` 和 `probe.complete` 事件后，得到 9 组执行工作项和 17 个试验终态。17 个试验均为 `valid=false`、`topology.passed=false`。最大记录试验面积为 70,729 像素，同时源孔洞数为 13→10，不能称为合法安全面积，也不能解释为只差 50 像素即可通过。

工作项的 `item_capacity_px` 合计在不同计划中为 39,318 或 72,026。这是 ownership 后工作项记录的容量，不是满足最终拓扑的可实现面积。容量合计与试验面积的差值仍不构成各步骤损失分解。当前前缀没有拓扑通过的 probe，不能从中估计该来源的最优合法面积，更不能推断 U2。完整标量与摘要分别保存为 `actual_s07_partial_trial_scalars_v1.json` 与 `actual_s07_partial_trial_summary_v1.json`。

## 搜索范围及 Full Capture 的已知工作量

第一块中保存的 `portfolio.preferences` 预先列出 U1 的 13 个 preference order。超时前记录了 3 次 preference 入口及 9 次 compiler attempt，不能声称已经覆盖全部搜索，更不能把已记录数量当作完整事件量上限。

已提交 B 前缀的数组索引包含 2,767 个 occurrence，全部属于已提交事件。按各 occurrence 的 `decoded_bytes` 求和为 924,738,656 字节。这给出相同前缀若逐 occurrence 原样保存时的原始数组载荷量，不包含 C 的控制文件、对齐和事件块，也不是完整 C 的容量预测。它进一步说明 683 个合成数组不能代表真实全流程。当前没有据此提高存储或运行预算。
