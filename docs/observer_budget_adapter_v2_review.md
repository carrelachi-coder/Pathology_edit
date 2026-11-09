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
