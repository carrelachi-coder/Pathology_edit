# Observer 预算候选：生产接入前审查材料

候选代码为 `diagnostic_storage/guard_budget_candidate_v1.py`，截至提交 `950ed30f` 未被生产 Codec 或 Adapter 导入。目标仅是减少记录器在高频事件上的路径解析与重复容量计算；它不修改逻辑事件、压缩格式、拓扑规划、guard 判定、面积阈值或 RNG。

## 语义对照

原版 `Budget.claim()` 先调用 `check()`，然后再次调用 `reservation()`。候选在 `str`、原生 `PosixPath`、原生 `WindowsPath` 三种路径类型上只计算一次 reservation，然后按原版公式计算允许额度、更新 `total`、`charges` 和 `sizes`。其他路径类型回退原版 `claim()`。对于原生 Path，直接读取 `.name` 与原版 `Path(path).name` 具有相同结果。`.index.pending` 的 `2×size+65536` 预留、4 KiB 向上取整、历史 charge 下限、`CONTROL_RESERVE` 和错误类型／文本保持一致。`actual()`、`seal()`、`claim_many()` 继承原版；`claim_many()` 通过动态派发调用候选 `claim()`。

证据范围：6 项差分测试覆盖 6,000 次混合预算操作及三档上限；独立进程将候选替入原版 Codec 后，52 项 Codec 测试和 55 项 Adapter 测试通过；Hash-only 和 Full Capture 小型完整流产生逐字节相同文件。目标服务器一次固定合成验收中，原版与候选各完成 100,000 事件并得到相同逻辑流 SHA。含进程开销耗时分别为 31.03 秒和 16.48 秒。完整报告见 [`budget_cost_trial_v4/closeout_report.json`](../benchmarks/manuscript_validation_20261109/adapter_controller_v1/budget_cost_trial_v4/closeout_report.json)。

## 接入前仍需核验

1. 对生产 Codec/Adapter 的实际接线差异单独审查，并重新冻结代码身份；历史 S07 授权及输出不可复用。
2. 在真实业务路径上确认记录器故障仍被隔离，B/C 的业务 receipt、事件序列、数组和 RNG 比较标准不变。
3. 明确 Full Capture 的真实数组数量和整树 4 GiB 预算风险；合成 Hash-only 加速不能代替该检查。
4. 固定新的真实 S07 诊断清单后，单独申请一次性授权。即使过程证据完整，也不能自动增加患者资格。

本材料支持候选进入接入审查，不构成生产接入批准，也不证明 S07 能在 1,200 秒内完成。正式 30 位患者实验、批量四目标编译、H&E 和医生研究仍关闭。

## 直接改 Codec v2 的否决证据

在隔离工作树中，仅替换 `guard_blocks_v2.Budget` 的 `reservation()` 和 `claim()` 后，新增的冻结旧算法对照测试通过，但原有 Adapter 测试出现 31 项失败。失败主因是 `observer_adapter_contract_v1.CODEC_PIN` 固定了原 Codec v2 文件 SHA，`code_identity()` 拒绝任何直接修改，Adapter 因此无法初始化。这是预期的身份保护，不能为了让测试通过而偷偷改写旧 pin。该试验改动未提交、未部署，也未改变当前生产快照。

后续接入应采用新的版本化 Adapter/合同身份，明确将候选预算模块 SHA 纳入捕获与验证清单，同时保留旧 Codec v2 和 Adapter v1 的原始身份。新版本需在本地完整回归后独立审查，再冻结真实 S07 新清单；不得复用 v4 合成授权。
