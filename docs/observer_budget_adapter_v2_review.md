# Observer Adapter v2 预算候选接入原型：独立审查材料

本分支只新增版本化文件，不修改旧 `guard_blocks_v2.py`、`observer_adapter_v1.py`、原始拓扑业务文件或真实 S07 Controller。该原型未部署、未用于真实来源、未获得新 S07 授权。

## 版本边界

- `observer_adapter_contract_v2.py` 使用独立格式 `observer-codec-adapter-v2-budget`。它继续钉住原 Codec v2 SHA，并新增预算候选模块 SHA pin；代码身份还包含新的 Adapter 与 Validator 文件。
- `observer_adapter_v2.py` 在原 Writer 初始化且事件数为零时，显式将预算对象替换为已验证的 `CandidateBudget`。旧 Adapter 继续使用旧 `Budget`。
- `observer_adapter_validation_v2_budget.py` 使用相同的新格式及代码身份核验，不读取旧版完成标记作为新版证据。
- 旧 Codec/Adapter 文件均保持原始 SHA，避免修改历史捕获及旧授权的身份。

## 本地证据

在此隔离分支执行 `pytest` 的指定集合：**171 项通过**。集合包括旧 Adapter 测试、新 Adapter 对应测试、Codec 测试、预算候选差分测试，以及新版本身份边界测试。新测试确认：候选模块哈希改变时新 Adapter 初始化拒绝；旧 Adapter 仍使用原 Budget，新 Adapter 使用 CandidateBudget。此前目标服务器固定合成验收的 100,000 条事件逻辑流哈希相等，结果不替代本分支的真实运行验收。

## 尚未完成

1. 独立审查三份新文件的完整差异和失败隔离；本地通过不等于审查批准。
2. 将新 Adapter/Validator 接入一个**新版本**真实诊断 Controller，更新 B/C 完整比较与运行身份。旧 Controller 和已消费的 S07 授权不可复用。
3. 在目标环境进行必要的只读身份与资源验收，再单独申请有限 S07 诊断授权。
4. 真实完整证据出现后，仍需单独研究拓扑可执行性和 30 位患者资格。此原型本身不改变 17/19 来源记录。

## 目标环境准备请求

新增 setup-only 脚本与冻结清单 `observer_budget_adapter_v2_target_prep_manifest_v1.json`。它在新隔离目录核对代码 SHA、Python/平台、220 个 distribution occurrence、环境和 no-replace 硬链接，然后只初始化一个零事件合成 Adapter，确认候选预算选择、格式／代码身份、默认 RNG、线程以及 Torch/业务模块导入边界。外层 90 秒、setup 60 秒、零重试；无 witness、S07、H&E 或接受证明。本地包含授权拒绝和仓库外工作目录烟雾测试的指定集合 **173 项通过**。该清单尚未获得独立的 SHA 绑定执行授权。

## 目标环境零事件验收结果

SHA 绑定清单 `150f4b0505b33bc125c6a44eccae27a9e3459ec170539feb89c045765bccc910` 的一次授权已消费并结案。目标服务器隔离快照在 3.31 秒完成：身份／磁盘检查、Adapter v2 零事件初始化、候选预算选择、新格式与代码 SHA 核验均通过；Python/NumPy 默认 RNG 不变，线程数 1→1，Torch 未导入。服务器输出目录没有 `adapter_complete.json` 或 Codec 完成标记。真实 witness、S07、H&E 和实验接受证明均为零。回执见 [`observer_budget_adapter_v2_target_prep_result_v1/closeout_report.json`](../benchmarks/manuscript_validation_20261109/adapter_controller_v1/observer_budget_adapter_v2_target_prep_result_v1/closeout_report.json)。

这只关闭目标环境导入与身份准备，不证明真实 B/C 非干扰、Full Capture 容量或 S07 能按时完成。下一步仍是对草稿 PR 进行独立代码审查，再设计新版本真实诊断 Controller 和完整比较清单。
