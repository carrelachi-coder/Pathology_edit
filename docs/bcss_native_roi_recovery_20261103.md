# BCSS 原始 ROI 固定裁块恢复：2026-11-03

本阶段只恢复和核验源条件，不运行 ProbNet、四目标编译或 H&E。正式 30 位患者尚未冻结。19%／32% 名义编辑预算、14%／27% 组织编辑硬下限和患者级独立单位不变。

## 生成前冻结的规则

[裁块协议](../benchmarks/manuscript_validation_20261103/native_roi_crop_protocol_v1.json) 在全量扫描前提交。窗口为 512×512，stride 256，从每个原始 ROI 左上角起算，按患者、ROI、y、x 固定排序。只取完整落在 ROI 内的窗口，不补边、不缩放、不加末端平移窗口。RGB、22 类组织标签经固定表映射后的统一 mask、历史核 mask 使用相同坐标。

廉价条件沿用既有源资格：统一标签合法、核编码合法，组织 ID1/2/4 均存在，Stroma 至少 70,779 像素（27%）。历史训练导出的背景过滤只记录，不在用户本次列出的廉价条件之外另加门槛。各原因作为独立布尔失败记录，不能将多标签计数相加当作独立窗口总数。技术患者整组排除。通过廉价筛选不等于容量、拓扑或四目标通过。

最终来源顺序保留三个旧冻结池及其全部历史失败，再追加本轮固定序列；每位患者只有四目标及独立 mask-only 审查全部完成后才可能计为一位合格患者。不按 H&E、评价器读出或实际 J 接近名义值的程度重排。

## 两位真实患者的几何预检

固定使用原始文件名排序最前的两位患者，仅核对几何，不按观察结果调整窗口或资格规则。[预检记录](../benchmarks/manuscript_validation_20261103/native_roi_geometry_pilot_v1.json) 覆盖 300 个完整网格窗口及 286 个旧导出坐标。其中 285 个可解码旧三联全部与原始同位裁块逐像素一致。

第一版预检发现一张旧 RGB PNG 截断，失败目录和日志保留。服务器同名 PNG 与挂载副本 SHA 相同（4dde4eb16c994876389cfe39e27967e20f6fc2b801ef6a873e75af816f5e7dca），两边均无法正常解码；并非挂载副本独有问题。原始 ROI 仍可正常解码，旧组织和核组件与对应原始坐标一致。

无效旧 PNG 不修复、不覆盖，也不记为像素验证通过。其坐标仍视为旧来源；另以原始同位裁块内容作保守排除别名，阻止相同内容被重新计成新来源。可读旧文件的实际解码内容哈希与无效旧文件的原始坐标别名分字段记录，不混称为原文件内容。该处理只增加排除，不放宽资格。几何边界、内容身份、严格下限及无效旧文件保留的四项针对性 CPU 检查通过。

## 全量分母及去重

151 个原始 ROI 全部按冻结规则枚举。每个窗口保存原始 ROI、患者、坐标、三路解码内容 SHA、旧导出坐标匹配、可读旧文件内容匹配、无效旧文件保守别名、同内容新窗口及重叠关系。哈希绑定数组形状和 uint8 内容，不依赖 PNG 压缩或文件名。网格的全部正面积相邻重叠窗口及重叠像素数明确保存；它们不作为独立患者。

旧目录完整索引先建立，再扫描原始 ROI。原始三联在读取时按已审计文件 SHA 验证；可读旧导出的像素对应关系逐项检查，无效旧文件另列。窗口 manifest、旧导出索引和逐患者计数分别保存哈希。新窗口只有同时满足廉价条件、技术排除和新旧去重，才另存为新的候选源三联；原始盘不写入。

输出状态区分旧内容重复、无效旧来源的保守排除、新内容重复、廉价条件失败、技术患者排除和“新廉价通过但尚未容量合格”。容量／拓扑／四目标的未运行计数明确为零运行，不冒充失败或通过；扫描异常保存失败状态和已有产物，不补零。

## 历史 CellViT 伪标签

151 份原始 JSON、转换脚本和原始核 mask 已在前轮审计中绑定哈希并逐像素重建。挂载盘进一步找到 AMP-001 权重，内容 SHA 与当前冻结评价权重相同（356418f19d9d478f164c7a31f85274584fefaa02355815c09f52346c658c8ec4）。这仅证明该权重文件存在；尚无历史运行凭据将它绑定至这 151 份 JSON，因此仍记为 historical CellViT-derived pseudo-labels，原生成 checkpoint 身份未证实。脚本默认 AMP-002 也不是执行证据。

本轮不重新生成或覆盖核 mask，不将其表述为人工核真值、独立临床验证或已证明训练无重叠。新裁块仍来自同一 151 位患者；只有此前尚未计入合格队列的患者获得完整资格，才可能增加独立患者数。

## 全量终态

[扫描终态及证据哈希](../benchmarks/manuscript_validation_20261103/native_roi_source_scan_report_v1.json) 已另存。151/151 ROI 完成，**32,753 个窗口**全部有记录。旧 22,870 个坐标全部核对：22,869 个可解码旧三联逐像素匹配，1 个无效旧 RGB 保留错误并作保守排除。新旧比较不是只按文件名。

| 互斥窗口状态 | 数量 |
| --- | ---: |
| 与可读旧导出内容重复 | 22,148 |
| 无效旧文件原始内容别名保守排除 | 1 |
| 新位置但廉价条件失败 | 9,369 |
| 技术患者排除（含其旧、新位置） | 1,019 |
| 新位置、去重后廉价通过 | 216 |
| 合计 | 32,753 |

独立失败原因按全窗口另报，可能一窗多因：缺 Tumor 11,479、缺 Stroma 7,024、Stroma 不足 27% 18,942、缺 Immune 21,409。它们不能相加作为互斥分母。151 位患者的窗口数和原因矩阵、237,206 条有向正面积重叠关系、旧导出解码索引、新三联及全部哈希留于私有研究资产。216 个新三联的 648 个保存后文件已重新计算 SHA、解码并复核廉价条件。

216 个新候选覆盖 **67 位患者**，其中 10 位已在原 17 位拓扑通过集合，另 **57 位**不在该集合。合并旧通过和 unknown 后，假设所有新廉价候选最终合格的极乐观并集上限为 **76 位**；这不是已合格人数。旧池自身仍为 17 位通过、最多 19 位；本轮新增完整资格患者为 **0**。容量、拓扑、四目标和独立审查均未运行，记录为 not_run，passed/failed/timeout 为 null，不把未运行伪装成失败或有效零。

这216项均因历史训练预处理的额外条件未入旧导出：94 项 BG>=30%，137 项编辑相关组织最大比例<5%，两者交集15项。本轮冻结的廉价规则未添加这些训练过滤；分布差异必须保留，不能把216项直接接受为正式病例。本轮没有依据 H&E 或评价结果改规则。

只读训练引用审计同时检查新文件名与原始坐标编码别名。各保留清单的精确别名引用为0，但患者级引用并非0：Cross train/val 覆盖205/11个新候选来源，Segmentator coarse train覆盖全部216，fine train/val/test分别157/28/31。这里计数单位是216个候选来源，不是患者数；完整逐来源和模型身份记录另存。FLUX、CellViT患者重叠及Pix2pix完整checkpoint训练谱系仍未知，未发现精确文件引用不等于独立验证。

下一阶段先对新源条件执行必要容量及源拓扑检查，区分真正不满足条件、执行异常和超时，再按固定顺序处理四目标与独立审查。本轮没有启动批量 ProbNet、修改19%／32%预算或开放正式H&E。

216组三联及完整私有清单已迁移服务器独立目录，648个文件全部SHA复核一致。服务器census仅重定位路径，保留Mac原始manifest与archive的父哈希；未运行任何容量、拓扑或模型编译任务。完整逐患者分母与not_run阶段账本均已归档。


## 216 个候选的容量与拓扑终态

[两阶段冻结协议](../benchmarks/manuscript_validation_20261103/native_source_qualification_protocol_v1.json) 在调用前提交，实际执行复用干净 `20ace976e92b7d3e3de091590e4e2d12653f5def`。完整性复核再次验证216组三联、648个文件，SHA、解码内容、尺寸、标签和廉价条件全部一致。使用8个CPU worker、每worker数值线程1，GPU不可见；容量每来源300秒、拓扑每来源1200秒，均涵盖U1+U2，不自动重试。归档的驱动源码与冻结SHA相同，没有修改扫描器或既有资格算法。

[本轮完整验收报告](../benchmarks/manuscript_validation_20261103/native_source_qualification_report_v1.json) 与私有逐来源ledger、两个阶段的全部控制凭据、输出和日志均保留。

| 阶段 | 完成 | 双操作通过 | 已知条件不满足 | 超时／执行异常 |
| --- | ---: | ---: | ---: | ---: |
| Significant 必要容量 | 216/216 | 9 来源、5 患者 | 207 来源 | 0/0 |
| 源拓扑（只进入容量通过项） | 9/9 | 0 来源、0 患者 | 9 来源 | 0/0 |

其余207个来源的拓扑为 `not_run_capacity_not_passed`，不是拓扑失败。225个来源控制进程均正常退出；不存在把未运行、超时或执行异常填成有效零。U1容量单项通过22/216，U2单项通过50/216，交集仅9；9个拓扑来源中，U1和U2各0/9通过，不能把单项成功当完整资格。

容量阶段的9个来源来自5位患者，其中4位不在旧17位通过集合。即使它们全部通过拓扑、加上旧unknown，容量阶段的中间乐观并集上限也仅23位；实际拓扑终态为0新增，因此合并旧池后仍为 **17位源拓扑通过、包括旧unknown的上限19位**。本轮无新unknown，新增四目标完整批准患者0位，正式30位尚未组成。

18个操作witness均为 `whole_mask_topology_witness_not_executable`：17项诊断含 `TopologySafeAreaUnderfillError`，1项为无定向界面同时满足组织和预检核合同的 `RefineContractError`。这些是预期合同拒绝，不是前轮模块找不到的环境错误。记录中的一个组织试验实现值70732仍低于硬下限70779，未豁免47像素；该试验面积不是最终joint J。必要容量聚合虽通过，保持整图拓扑后仍未找到达到组织硬下限的可执行witness。

描述性分层未用于重新排序或筛选：历史高背景单项79来源、低编辑组织单项122来源、两项同时15来源；所有9个容量通过项均来自低编辑组织单项组，拓扑全部未通过。这只是当前固定检查器和有界搜索的源条件结果，不是患者组织缺失、临床结论或所有可能编辑策略均不可能的数学证明。

本轮工作按已冻结规则收尾：不再扫描同一窗口、不重复编译失败条件、不降低19%／32%预算及14%／27%硬下限，也不启动批量四目标、H&E或医生研究。当前数据和固定网格恢复未突破30位独立患者的来源瓶颈。


## 拓扑瓶颈只读诊断：解释修正与证据边界

本轮没有扩池、重跑 witness、编译四目标或生成 H&E。只读取已冻结的源 mask、预检元数据与18项保存的失败错误串。原始资格结果保持17位通过、含旧unknown的上限19位。[逐项诊断及源码身份](../benchmarks/manuscript_validation_20261103/topology_bottleneck_diagnosis_v1.json)、[18行诊断矩阵](../benchmarks/manuscript_validation_20261103/topology_bottleneck_matrix_v1.csv) 与 [比较图](../benchmarks/manuscript_validation_20261103/topology_preflight_vs_logged_trials_v1.png) 已另存。

**纠正70732的“近乎通过”解释。**该试验的 `topology.passed=false`、源孔洞13→10，不能称为拓扑安全面积；只强调距70779差47像素不完整。此前原始诊断字段 `observed_realized_safe_area_values` 的命名容易误读，其中含拓扑不通过的试验，现明确要求按 `topology.passed` 分开解释。旧控制记录、原始数值和通过数不改写。

同一S07/U1 witness中，已保存且拓扑审计通过的最大floor trial仅22287，距组织硬下限48492；它也不是完整witness通过。源Stroma中2像素小组分在编辑前已经存在，并且同时见于拓扑通过和不通过的试验，因此不能将该小组分本身当作新增碎片或确定失败原因。可确认的70732状态差异是孔洞13→10，逐像素过程因失败mask未保存而不能重构。

### 18项逐项面积记录

以下单位均为组织像素，不是最终joint J。只取保存日志中 `topology.passed=true` 的floor probe；重复错误串去重的是诊断状态，不是执行次数。日志中最大值不构成全局最优或可实现面积上界证明。

| 来源 | U1：日志中拓扑通过的最大 floor trial / 距硬下限 | U2：同口径 |
| --- | ---: | ---: |
| S00 | 32,036 / 38,743 | 36,120 / 34,659 |
| S01 | 28,862 / 41,917 | 55,617 / 15,162 |
| S02 | 38,327 / 32,452 | 49,208 / 21,571 |
| S03 | 23,222 / 47,557 | 25,263 / 45,516 |
| S04 | 19,886 / 50,893 | 16,847 / 53,932 |
| S05 | 29,664 / 41,115 | 55,074 / 15,705 |
| S06 | 34,296 / 36,483 | 57,966 / 12,813 |
| S07 | 22,287 / 48,492 | 未留存面积；联合界面合同拒绝 |
| S08 | 23,897 / 46,882 | 14,775 / 56,004 |


U1跨这9个来源的最大已记录拓扑通过试验为38327，U2为57966，均低于70779。另有1项联合界面拒绝没有可恢复的试验面积，明确为null，不填0。

### 已确认与尚不能确认的原因

- **源Stroma总量及整体碎片化不是共同直接证据。**9个源Stroma为233833–248784像素，8连通组分仅1–2个，最大组分占比99.8334%–100%，均远大于硬下限。受anchor、深度、保护区和所有权约束的局部合法域可能完全不同；其mask未存，不能据全源连通性断言局部可达性。
- **预检anchor并非全部没有覆盖。**18项均有3–12个cell-feasible anchor，允许的预检报告覆盖率最低约75.27%，并非直接观测到终端anchor覆盖失败。但缺少失败目标及实际selected anchor mask，无法验定终端接缝连续性，也不能排除joint约束对可达域的间接作用。
- **17项最终错误为安全面积欠填，1项为联合定向界面合同拒绝。**4项witness的部分无效诊断状态记录了不获允许的源孔洞变化；不能据此把全部17项欠填都归因于孔洞。其余通过拓扑但面积不足的状态表明执行停止在安全前缀，而不是整图组织面积不足。
- **预检面积与试验面积之差不是逐步损失分解。**预检是聚合潜在容量；失败probe经过具体计划、合法域、所有权、删除预算和拓扑保护。没有中间mask就无法把该差额精确分给anchor、保护区、界面所有权或某一种局部拓扑检查。

### 遥测缺口与代码证据

核对了实际执行`20ace976`的6个相关源码文件，内容SHA与只读分析所用文件一致。`_qualify_tissue_case`失败出口只保留schema、错误串和`persisted_target_mask=false`，没有序列化失败candidate、原始计划、结构化underfill反馈或各guard拒绝计数。即使错误类型本身携带feedback，最外层汇总后也不能从日志完整找回逐界面贡献。

因此，本轮不能给出可审计的逐步骤像素损失waterfall，未用猜测或0补齐。已保留的源mask图只是源条件，不是重建的失败目标mask。归因边界、null字段与每项输入SHA均写入机器记录。

### 现有路径与研究方案的界限

执行代码已包含全局偏好、对允许的interface/anchor逐项偏好、有界3次重规划，以及欠填后的剩余界面扩展和选中front之间的面积再分配。因此，“换一个anchor”或“增加timeout”不能自动作为新证据；动态逐计划指纹未保存，也不能声称所有anchor组合和所有执行顺序已被数学穷尽。

合法域的`argmin`所有权分配按代码保留其并集；它可能改变各front可达性，但不能未经证据称为直接丢掉并集像素。未来可以把不改变密封合法域和硬检查的联合front调度／所有权可达性作为统一算法研究假设；本轮没有实现或验证任何新的合法survivor。

允许靶区合并、孔洞变化、源组分消失／分裂、远处新建免疫区域或改变操作定义均属于机制／研究设计变更，不是就地调参。能力矩阵中的breast `tumor-burden-increase-v1`已经作为cohesive操作的重复绑定退役，改名不是新的执行路径。这里未评价或批准替代机制的临床合理性。

[后续证据采集与统一算法研究草案](../benchmarks/manuscript_validation_20261103/topology_diagnostic_followup_proposal_v1.json) 仅供审查，状态为draft、未冻结且execution_permitted=false。若未来获准采集缺失遥测，应保存每步源/目标/合法域mask、计划、anchor及guard事件；拒绝计数可多次涉及同一像素，不能直接相加成独立损失面积。若利用失败患者开发新策略，还需预先冻结开发／独立验收患者划分，并核查是否增加正式队列排除，不能调完再把全部开发病例当独立验证。

### 对30位患者的含义

只修复当前9个来源的所有18项witness，患者并集也最多23，仍不足30。只读合并旧、新必要容量通过清单可得到538来源记录，排除技术患者后527来源、84患者组；这是历史必要容量集合，不是本轮重新验证或新的拓扑／四目标通过数。它说明统一、可泛化的机制研究可能需要覆盖更广的已知容量来源，而不是只优化这9个窗口；不能保证其中30位最终合格。

本轮只读诊断不改变17／19资格结论，不重跑同批失败，不放宽预算或硬下限。批量四目标、正式H&E和医生研究继续关闭。


## 有限过程证据采集协议冻结：尚未授权执行

[冻结协议](../benchmarks/manuscript_validation_20261103/limited_topology_capture_protocol_v1.json) 现在为 `frozen_protocol_execution_not_authorized`，`execution_permitted=false`。此前的 draft 原样保留。冻结的是采集范围、非干扰合同和比较验收规则；记录代码尚未实现或验收，本轮没有重跑资格搜索。

预选对象仅为 S07 的 U1/U2 Significant，绑定原始9来源子集中的索引7、原始模板1/3、三路源文件SHA和重建的输入映射SHA。重建映射不是旧计划哈希；旧运行未保存的计划和完整随机状态仍标记缺失。19%／32%名义预算、14%／27%组织硬下限及全部拓扑／联合合同不变。

若未来单独获准执行，采用依次进行的三个比较臂：A为原始20ace976执行；B为仅增加被动记录的构建，保存语义序列和哈希；C为与B相同的构建，同时保存完整mask和事件。每臂只执行同一来源的U1/U2，共最多3个来源进程、6次逻辑witness调用；每个来源进程上限1200秒，总进程预算3600秒，CPU单并发、零自动重试。不会在本次冻结后自动运行这些比较臂。

A先比较旧记录中实际保存的结果、错误和诊断状态；B必须与A的业务结果／异常一致；C必须与B的有序计划、分配、mask、已有随机状态和业务结果一致。旧计划未保存，因此不能声称证明了与旧运行的内部计划完全一致。任何差异、记录故障、证据缺失或超时均进入 `mismatch_hold`，停止后续比较臂并解释原因；不换seed、anchor、参数或时限，也不覆盖旧结论。

记录器只能读取已经计算出的值，将数组复制到独立存储；不得参与候选选择、排序、面积分配、随机数和通过判定，不得重复调用guard获取记录。记录器返回值不能进入业务逻辑；记录故障不能被业务异常处理捕获后触发重规划。事件编号不消耗随机数，不能为记录初始化CUDA或额外随机生成器。输出位于独立的新目录，不进入业务缓存、续跑或文件发现；目录存在即拒绝，不覆盖。增加记录可能影响耗时，这类超时只使诊断无效，不能作为生物学失败。

执行前还必须冻结observer构建commit／补丁SHA、输入与合同、运行环境以及比较字段；审查仅观测的差异，并用合成输入检验关闭／哈希／完整记录及记录故障下的业务结果、计划和随机状态一致。不能执行后再删去不匹配字段。原始完整环境和随机状态缺失的限制必须保留。

采集范围包括每次计划和anchor顺序、合法域与所有权mask及并集、保护区、desired／floor探测的失败目标mask、逐界面贡献、已有guard事件及唯一像素集合、拓扑审计和未被外层压成错误串的结构化反馈。未评估、不可获取、零事件分别记录。拒绝次数可能重复涉及同一像素，各mask也可能重叠，不能相加成独立损失；预检容量与试验面积之差也不能变成因果waterfall。

该协议不允许新anchor搜索、算法或机制变更。诊断意外通过也不增加资格人数。17位源拓扑通过、含旧unknown上限19位以及新增四目标批准0位均保持不变；正式30位患者、批量四目标、H&E和医生研究继续关闭。后续若进入方法研究，须另行冻结开发／独立验收患者划分并审计正式队列排除。

## P1记录器实现与P2本地非干扰验证：不构成复验授权

[实现及验收证据](../benchmarks/manuscript_validation_20261103/passive_observer_implementation_report_v1.json) 已另存。冻结采集协议和旧draft均逐字节保留；协议中的`not_implemented_or_accepted`是冻结时的状态，本节和新报告记录后续实现进度，不能将进度更新当作执行授权。

记录器默认关闭，只能在显式`observation_session`中激活，未添加命令行／环境自动启动路径，也未修改原source probe或控制器。没有向服务器安装构建，没有读取S07运行资格搜索。A/B/C均未执行，`execution_permitted=false`继续有效。

业务入口只增加被动观测调用，覆盖源输入与preflight、有序偏好和重规划、planner的label／preflight／front过滤、方向性anchor数量限制的已计算输入、排序与分配、独立合法envelope、ownership、逐front贡献及desired／floor探测。U2在计划生成前就拒绝时，也会留下过滤前后状态；不为取记录重复调用筛选器。逐像素记录保留已通过的guard阶段、首个拒绝原因／坐标和接受顺序；下游短路条件不额外求值。

Hash-only与full-capture使用完全相同的语义序列；完整模式额外写入独立复制的无损数组。记录器不重新渲染业务异常，保留原先已经生成的错误文本及结构化feedback。写盘、空间、序列化、复制、队列或记录器故障均在业务异常处理之外标记证据无效，不触发重规划。已有目录拒绝写入，不进入业务缓存或resume。关闭时无法保存状态文件的故障仍保留在内存状态中，不能据部分落盘内容判定采集成功。

[静态审查](../benchmarks/manuscript_validation_20261103/passive_observer_validation_v1/static_review.json) 将5个业务文件与`20ace976`比较：剥离observer导入和38处仅使用已有变量／属性／常量的观测调用后，业务AST一致。这排除了本轮业务算法表达式和控制流的改动，但不能证明记录开销没有任何运行时影响。

[测试日志](../benchmarks/manuscript_validation_20261103/passive_observer_validation_v1/pytest.txt) 为 **120项通过，另85项subtest通过**，其中新增观测器测试43项。包括真实确定性mask执行器的合成成功与欠填路径、采用明确mock依赖的真实有界重试处理器、合成preflight下的实际界面过滤拒绝路径。六类要求的故障注入均未改变被测业务结果，重试处理器的三次执行及anchor偏好也完全相同；额外覆盖raster ENOSPC、关闭失败、存储／事件上限和旧目录碰撞。

严格B/C比较器不排除任何语义字段。输入、合同、计划或anchor顺序、面积分配、mask、拓扑、异常和RNG摘要任一差异都会得到`mismatch_hold`；截断日志、缺失或损坏数组、缺失manifest项也拒绝比较。局部生成器只读状态不前进，不额外初始化CUDA／生成器，也不消费惰性迭代器。拒绝次数与唯一像素集合／交集分开处理，不相加为面积损失。

本轮结果只支持“记录器已实现、本地静态及合成检查通过，待独立审查”。不证明S07实跑的非干扰性，不补齐旧运行缺失的随机状态或计划；CUDA和未传入的局部生成器状态仍明确不可获取。失败probe若未生成完整fine-label目标图，只记录实际存在的变化mask及源／目标二值状态，后续重建必须标为derived，不能声称捕获了不存在的历史目标。合法域／可达性和guard唯一像素汇总留给离线分析，不增加业务判定。

进入P3前仍须独立审查实现、冻结observer完整commit与运行环境／输入合同／比较字段／存储配置，并另行取得有限复验授权。若未来利用S07开发算法，须另定患者级开发／独立验收划分。当前仍为17位源拓扑通过、含旧unknown上限19位、新增四目标批准0位；30位目标、19%／32%名义预算和14%／27%组织硬下限不变，批量四目标、H&E和医生研究继续关闭。

## 记录器资源保护及完成证据补强：本地验收，不执行S07

[本轮报告](../benchmarks/manuscript_validation_20261103/passive_observer_resource_report_v1.json) 将资源保护与前轮实现验收分开留存；冻结协议、前轮120项测试日志及报告均保持原样。

复制数组前先核查逻辑`nbytes`、剩余存储、单个独立数组上限和artifact数量上限，连巨大broadcast view也不能先复制再检查。默认独立数组payload上限64 MiB、每事件metadata保守计费上限8 MiB、artifact库存上限100,000。文件哈希使用1 MiB分块读取，数组哈希使用分块memoryview，消除`read_bytes()`和`tobytes()`产生的整块副本。比较器对raster采用只读mmap校验。超限仍只使诊断无效，不改变业务结果。上述限额约束受控payload与库存，不能解释成整个进程RSS或OS资源压力已有硬保证；离线比较仍保留解码后的事件列表。

本轮还关闭了一个实际完成状态漏洞：summary写出后、关闭时报错，磁盘上可能已留下`valid=true`内容。因此v2证据必须额外具有完成标记，绑定summary SHA。只有summary成功关闭、pending完成标记写出／fsync／关闭成功后才发布最终标记；任何一步失败均不得据有效外观的summary接受证据。异常离开观测session也标记无效，并原样重新抛出业务异常。

比较器要求语义数组计数、summary计数与完整捕获manifest逐项对应，拒绝重复引用、孤立／缺失raster、符号链接、错误序号及有fault却标为valid的summary。每个文件和解码内容仍检查哈希、dtype、shape。旧v1 summary不会被静默升级为新v2完成证据。完成标记用于检测意外缺失和损坏，不是防篡改签名。

[本轮指定测试集合](../benchmarks/manuscript_validation_20261103/passive_observer_resource_validation_v1/pytest.txt) **137项通过，另85项subtest通过**；观测器测试60项，新增17个资源与完成状态边界场景，包括复制回调前拒绝、禁止整文件读取、部分写入、重复引用、孤立文件、关闭失败和异常退出。专门验证：即使summary内容已完整写出且写着`valid=true`，summary关闭或完成标记关闭失败也不会被比较器接受；自洽更新summary哈希也无法使重复artifact引用合法。

[静态审查](../benchmarks/manuscript_validation_20261103/passive_observer_resource_validation_v1/static_review.json) 仍为5个业务文件、38处观测调用；本轮没有修改这些业务文件。上述结果是本地静态／合成及指定回归集合，不是全仓CI，也不是Vercel部署状态的替代解释。未运行真实S07、未验证真实B/C一致性或OS级终止／严重压力。

当前实现待独立代码终审。P3前还须另存并冻结实际observer完整commit、源输入和case mapping／合同、Python与依赖、资源限额及存储身份，并取得单独执行授权；本轮未将这些执行准备描述为已经完成。S07 A/B/C仍未授权、未执行；17／19资格记录、30位患者目标、19%／32%名义预算、14%／27%组织硬下限及正式实验关闭状态均不变。

## 原子发布收尾与S07执行身份准备：仍未授权

[原子发布修复](../benchmarks/manuscript_validation_20261103/passive_observer_atomic_report_v1.json) 使用`os.link(pending, final)`替代exists检查加rename：目标若已存在，由单次系统调用返回EEXIST，绝不覆盖；pending保留为同inode的不可变证据，不在发布成功后增加可能失败的清理步骤。新增故障测试在link前制造竞争目标，确认原字节不变、诊断无效；正常发布也验证同inode及第二次发布拒绝。指定本地集合为 **139项通过，另85项subtest通过**，观测器测试62项。5个业务文件及38处观测点未改；这不是全仓CI或真实S07验收。

[独立execution manifest](../benchmarks/manuscript_validation_20261103/limited_topology_execution_manifest_v1.json) 绑定observer完整commit `c119c600e7479aef9046087745e7be34f57cbc29`，A仍为原始`20ace976`。代码修复先提交，身份manifest后提交，因此不会将自身文档commit误称为运行代码commit。服务器只读核对确认：原始快照tracked clean、S07三路文件SHA和两项规范化case mapping SHA与冻结协议一致；521个resource／skill文件与observer构建内容一致。[环境快照](../benchmarks/manuscript_validation_20261103/limited_topology_execution_environment_v1.json) 保存Python二进制SHA、版本／平台、220项distribution版本和RECORD摘要以及相关环境；[合同清单](../benchmarks/manuscript_validation_20261103/limited_topology_execution_contracts_v1.json) 单列全部资源摘要。它们不是旧运行完整环境恢复或全部二进制依赖的密封证明。

本轮在同一存储设备另建小型preflight目录，确认支持hardlink，已存在目标拒绝且内容保留；未创建A/B/C输出目录。约15 TiB可用空间只是审计时快照，不等于物理预留。manifest固定每臂1200秒、最多3进程／6次witness、CPU单并发／零重试、记录器限额及12 GiB整体存储预算；实际有限配额／预留和启动前身份重验仍须落实。observer服务器快照尚未安装。未来B/C显式wrapper源码和SHA写入manifest，当前没有执行。

**身份准备发现RNG前置条件尚未解决。**两个独立CPU启动探针仅导入Python random／NumPy并读取状态，未调用业务模块或任何witness；它们的默认全局RNG状态摘要不同。案例seed=42不等于这些全局状态天然一致，而冻结协议和比较器要求B/C比较全部可用RNG摘要。因此manifest为`identity_manifest_frozen_execution_on_hold`、`ready_for_execution=false`：需要在授权前明确并审查与原合同兼容的初始状态控制规则，不能擅自重置／恢复状态、事后排除RNG字段或声称恢复了旧状态。本轮未实施任何此类改变。

比较字段已事先列明：A对照旧保存的业务结果／错误列表SHA和既有trial状态，不比较从未保存的旧内部计划；A/B比较业务终态和确定性指标；B/C完整语义序列比较，不排除任何语义字段。输出／控制器时间等账务字段只按各臂身份另核验。任一差异、身份漂移、超时或证据不全进入`mismatch_hold`，不启动下一臂。raw路径若造成未解释差异，同样暂停，不在观察后增加排除。

以上是执行身份与启动规格的冻结，不是执行准备全部完成：RNG规则、observer不可变快照安装核验、资源预留／live preflight及单独执行授权仍为显式门槛。S07 A/B/C未授权、未执行；17／19患者资格、新增四目标批准0、30位目标及预算／硬下限均不变，批量四目标、H&E和医生研究继续关闭。

## RNG决策、外层controller及部署／预留准备

本轮交付三件事：[独立RNG决策记录](../benchmarks/manuscript_validation_20261103/rng_initialization_decision_record_v1.json)、[最终controller及准备报告](../benchmarks/manuscript_validation_20261103/controller_startup_preparation_report_v1.json)、[候选execution manifest v2](../benchmarks/manuscript_validation_20261103/limited_topology_execution_manifest_v2.json)。旧冻结协议、execution manifest v1及Observer主体未改；启动策略仍待单独审查，不能将这些交付物解释为执行授权。

### C方案的技术结论与科学边界

目标服务器仅运行了不涉及业务来源的启动探针：父进程为1个native线程，预加载NumPy和CPU Torch，CUDA未初始化，没有导入pathology业务包。直接fork的两个子进程保留了hash／NumPy／Torch CPU状态，但Python random被子进程初始化钩子改变，完整初始状态不相等。因此原始“自然继承即可”方案不成立。

候选改为 **共同父进程＋显式postfork恢复**：固定导入顺序、父进程无业务模块／单线程／无CUDA，保存一份新的共同启动状态；A/B/C顺序fork、不exec，每个子进程在业务导入前显式恢复Python random、NumPy legacy及CPU Torch默认生成器，并检查共同状态摘要和继承的hash指纹。两项非业务探针的恢复后摘要均与父进程一致。`PYTHONHASHSEED`仍可为null，因为同一次父进程的hash secret被继承；指纹不是旧secret恢复，也不保证不同控制实验批次的secret相同。业务局部Generator不被发现、重置或替换，原案例seed不改；不可访问状态仍标记不可获取。

这不是纯fork，也不是对历史状态的恢复，而是必须披露和单独批准的实验控制变更，**同样适用于使用旧业务代码的A**。因此A应解释为controlled legacy baseline；若它与历史保存业务结果不同，仍进入hold。恢复初始状态仅去除一个混杂因素，不能证明记录器运行时无干扰；完整语义比较仍不排除RNG字段。`rng_decision_approved=false`、`execution_permitted=false`继续有效，未为S07实施该控制。

### Controller的强制停止规则

生产controller固定为`80fbbfc28a4d915680211e749f42cbea9e2a891a`。三个门槛——允许执行、RNG决策获批、准备检查完成——以及独立manifest／decision哈希绑定的授权均在输出创建和状态恢复之前检查。来源限定为原probe、subset索引7、limit1、原模板，严格校验参数和模式，最多三次来源进程／六次逻辑witness预算；每次1200秒，零自动重试。

只有A完整业务receipt与历史一致，才启动B；B须与A业务receipt一致、证据有效，才启动C；C需通过完整B/C比较。任一异常、超时、信号退出、不一致或证据无效都会停止后续阶段。比较器只在独立的离线helper子进程中导入，不在共同父进程中导入，以免改变B/C的模块可用性和导入顺序；最多两个helper、各30秒、零witness调用。controller保留PID、退出码、超时和耗时记录；父进程中断会终止并回收子进程。所有业务receipt严格相等，尚未增加路径规范化或事后字段排除；辅助输出路径若不一致，同样hold。

新增38项controller／startup合成测试通过；指定组合回归177项通过、另85项subtest通过，最终controller的38项定向测试也通过，几个数字不相加。目标服务器还使用完全独立的假来源JSON和合成probe脚本验证了实际CLI：成功时A/B/C完整通过真实Observer比较；A退出时仅启动A；B记录无效时仅启动A、B。实际使用Observer／比较器及其包内类型定义，未读取BCSS图像／mask，未调用真实规划、资格搜索或witness。合成授权记录只覆盖这些fixture，绝不覆盖S07。

### 代码部署、物理预留及拒绝授权核验

A的`20ace976`快照、B/C的`c119c600`快照以及controller的`80fbbfc2`快照已置于三个独立目录，按归档内容核验共4066个文件，未改原代码目录或共享Python环境。目标服务器直接身份／scope检查通过，220项distribution身份与冻结环境一致；这不是OS／二进制依赖的完整hermetic证明。完整source inventory在v2 manifest内固定，来源文件、原合同及比较器身份在每臂前重验。

已在同设备`execution_preparation_v2/reservation_12gib.bin`用`posix_fallocate`实际预留 **12 GiB**，并核对size／allocated blocks，不是稀疏文件或仅free-space数字。该文件是有限headroom，不是独占quota，无法保证其他任务不会耗尽未预留空间；controller在每臂前核对预留和可用空间，ENOSPC仍使诊断无效并停止。此预留当前保留，后续应明确释放，不会自动删除历史数据。

S07实际输出根目录仍未创建。真实部署的controller以关闭授权记录核验，确实在输出创建及业务导入前拒绝执行；直接只读preflight不调用任何来源witness。实际输出位置的hardlink不覆盖测试会在获准后、A启动前再次进行，不能以准备目录的结果替代。v2的`preflight_complete=true`仅表示本轮准备核验完成，`ready_for_execution=false`和另外两个授权门槛仍关闭；后续必须另存获批版本，不能修改已冻结的v1或本次review候选来伪装授权历史。

当前仍为17位源拓扑通过、含unknown上限19位、新增四目标批准0位。30位患者目标、19%／32%名义预算及14%／27%组织硬下限不变；S07 A/B/C未获授权、未执行，批量四目标、正式H&E和医生研究继续关闭。

## 最后两项准备验收：RNG／存储通过，三臂导入顺序仍需明确判断

[本轮审查报告](../benchmarks/manuscript_validation_20261103/controlled_startup_final_review_report_v1.json) 仅补充用户要求的import-only烟雾测试和实际输出位置存储检查。Observer、controller `80fbbfc2`、原冻结协议及execution manifest v1／v2均未改；本轮没有增加生产功能或执行S07 witness。这里提供的是审查证据，不冒充已经取得独立科学批准。

### 业务导入烟雾测试

[完整导入证据](../benchmarks/manuscript_validation_20261103/controlled_startup_final_review_v1/import_smoke_report.json) 使用冻结环境、三个实际快照和准确的probe argv，`runpy`采用非`__main__`名称，仅加载定义；profile守卫阻止main／probe／资格和witness入口被调用。父进程始终无业务模块、1个native线程。三个子进程恢复后的Python random、NumPy legacy、Torch CPU及hash摘要一致；导入结束后仍一致，所有已记录source-module入口的摘要也均等于初始值。三个子进程导入后均为1个native线程，CUDA均未初始化，未触发任何受阻业务调用。

这里的“无来源处理”不等于完全不读取文件：父进程的只读身份检查仍会对冻结输入计算哈希并读取subset metadata；没有解码来源图像／mask，也没有进入source probe、规划或witness。初步开发烟雾测试曾保留helper argv，最终验收另用准确的业务argv完成三次子进程导入，两次记录均保留，真实witness调用始终为0。

**不能把三臂业务导入顺序记为全部通过。**B/C顺序一致，但A与B/C不同，即使在描述性比较中单独去掉Observer文件本身，差异仍存在。三者加载相同的96个source文件，只有两个文件提前：`phase3_mask_edit_refine/__init__.py`和`phase3_mask_edit_refine/models.py`。原因是B/C在probe前导入Observer，Python会先执行包初始化及其中的models导入；A在legacy正常依赖链中较晚加载它们。[逐项顺序差异](../benchmarks/manuscript_validation_20261103/controlled_startup_final_review_v1/A_vs_B_import_order.diff) 已保存。

该差异未伴随已观测默认RNG、hash、线程或CUDA状态变化，但这不能证明任意导入副作用都不存在，更不能把“顺序相同”改写为真。生产比较器没有排除这两个模块、没有放宽A历史比较，也没有增加路径归一化。导入记录使用相对source路径只是说明此次顺序差异，不是修改生产业务／语义比较。

### 实际启动位置的存储与授权检查

[存储证据](../benchmarks/manuscript_validation_20261103/controlled_startup_final_review_v1/exact_output_storage_report.json) 确实在manifest v2指定的最终输出根目录临时执行：hardlink发布成功，第二次发布得到EEXIST并保留原字节；A/B/C三个空目录及各自测试marker的path／inode均独立，且和原预留文件同设备。只移除本轮独占创建、inode和内容未变的测试文件，以及仍为空的目录，不递归清理、不删除外来内容。完成后实际输出根目录恢复为不存在，满足后续fresh-output门槛，没有生成任何业务结果。

原12 GiB预留文件inode、size及实际allocated blocks核验保持有效，没有释放或改写预留文件。检查前后均重新通过完整快照／输入／runtime／scope身份核验；再次调用关闭的run门槛，在输出创建和业务导入前拒绝。远端[负授权记录](../benchmarks/manuscript_validation_20261103/controlled_startup_final_review_v1/authorization_denied_bound_to_v2.json) 仍为`approved=false`，准确绑定v2 SHA `e8f05fe392079e329fb1deea4b8c14fc6a84606c71bd9b2322a69dfac66f720c`及RNG决策SHA。不存在正授权记录；物理预留仍不是独占quota，未来获准启动前还需再次执行live检查。

### 终审结论边界

RNG恢复和导入兼容性、CPU单线程／无CUDA以及实际位置存储检查获得正面证据；**严格的A/B/C导入顺序一致性未满足**。当前状态为`hold_for_explicit_import_order_acceptance_decision`：需要独立明确是否接受这两个文件提前导入，作为披露过的instrumentation／startup差异；若要求顺序完全一致，则须另行审查初始化安排，不能擅自预加载A来让测试通过。本轮没有选择或实施任何新的初始化修正，也没有给出S07执行批准。

A与历史完整业务receipt的严格比较不变，任何不一致仍在B前停止。当前17位源拓扑通过、含unknown上限19位、新增四目标批准0位及30位目标均不变；19%／32%预算和14%／27%硬下限不变。S07诊断、批量四目标、正式H&E及医生研究继续关闭。

## 最终执行前技术审查：导入差异已接受，正式批准仍未签发

用户对`36e04234`的后续独立技术审查明确接受两个提前导入模块，因此这项技术差异从待返工清单关闭；事实仍为B/C顺序一致、A与B/C不同，不改写成三臂完全一致。本轮[最终执行前审查](../benchmarks/manuscript_validation_20261103/final_execution_authorization_review_v1.json)只核对用户指定的四项，不修改Observer／controller／业务代码，不再增加导入测试或资源保护功能，也不冒充新的外部独立签署。

Controller技术复核在既有正常流程及已覆盖故障范围内未发现新增实质阻断：严格A→B→C、A对历史及A/B完整receipt相等、B证据有效后才开放C、完整B/C语义比较，以及超时／信号／异常／不一致立即停止均保留。无自动重试或失败后resume，RNG恢复仅用于获批后的初始条件控制；不用于让失败结果变成匹配。原38项定向测试和三个已部署CLI合成场景作为既有证据引用，没有重跑实际S07。

[本次服务器只读核验](../benchmarks/manuscript_validation_20261103/final_execution_authorization_review_v1/readonly_server_audit.json)重新通过4066个部署文件、220项distribution及两份case mapping身份；实际输出根目录和三个arm路径均不存在，12 GiB物理预留仍有效、inode与上轮实际位置检查一致。它不是独占quota；实际输出位置hardlink检查仍须在未来启动时重做，每臂前还须重验身份、预留与剩余空间。本轮未创建输出目录、未导入业务模块、未恢复RNG，真实witness调用0次。

**授权绑定有一个必须保持清楚的操作条件。**已冻结v2的SHA为`e8f05fe392079e329fb1deea4b8c14fc6a84606c71bd9b2322a69dfac66f720c`，其两个执行／RNG批准布尔值均为false。只在内存中做的gate-only检查确认：即使假设存在匹配v2 SHA的正记录，v2也仍会拒绝；本轮没有创建正授权文件。因此未来若正式批准，应保留v2，另存引用v2、RNG决策和导入差异接受SHA的新版本launch manifest，再把授权绑定到新文件的准确SHA。不能修改v2后继续使用旧SHA，也不能将此审查写成已颁发执行许可。本轮没有生成launch-enabled manifest。

当前技术资料可供最终批准审查，但整体RNG新实验条件尚未正式批准，S07执行授权也未签发，独立签署不由实现者自造。仍为17位源拓扑通过、含unknown上限19位、新增四目标批准0位；30位目标、19%／32%预算和14%／27%硬下限不变。诊断成功≠拓扑资格，拓扑资格≠四目标批准，有限S07许可也≠正式H&E许可；正式实验及医生研究继续关闭。

## 已获批的S07有限诊断：A通过历史比较，B触及事件上限后停止

用户通过两个独立回复正式批准了RNG实验条件和这一次S07有限诊断。[RNG批准记录](../benchmarks/manuscript_validation_20261103/rng_controlled_startup_approval_v1.json)、[独立执行授权](../benchmarks/manuscript_validation_20261103/s07_execution_authorization_v1.json)及[启动manifest v3](../benchmarks/manuscript_validation_20261103/limited_topology_launch_manifest_v3.json)均另存，不覆盖v2。v3准确SHA为`e41815ae705efe05b2075dffe7fd6599b6186782c2f606fb2398922a5ad32032`，与用户批准的候选字节一致；现场重验代码、输入、环境、scope和授权后，仅启动一次controller。

[本次结果](../benchmarks/manuscript_validation_20261103/s07_authorized_diagnostic_report_v1.json) 为 **`mismatch_hold_B_observer_event_cap`**，不是完整过程采集验收通过：

| 臂 | 运行结果 | 后续判定 |
| --- | --- | --- |
| A Controlled Legacy | 约113.29秒，退出0，完整业务receipt严格等于历史 | Controller接受，开放B |
| B Hash-only | 约152.62秒，退出78，未超时；Observer达到冻结的2,000,000事件上限 | 记录无效，停止后续臂 |
| C Full Capture | 未启动 | 无Full Capture或B/C一致性结果 |

完成4次逻辑witness调用／4次预算预留，未使用C的2次余量，无自动重试或新启动目录。Controller日志的直接hold理由是B非零退出；事后核对明确为Observer事件限额触发，不是1200秒超时、4 GiB磁盘限额或新增生物学失败。B记录器故障后按既有隔离政策让业务调用完成，再拒绝接受诊断证据。

离线对照确认B完整业务receipt也与A、历史结果相同，两个已运行子进程的初始状态摘要与共同父进程一致，hash fingerprint一致；U1/U2仍保留原来的witness失败及相同错误列表SHA。**这些只支持本例已保存业务终态一致，不使B重新获得controller接受，也不证明完整非干扰性。**B证据`valid=false`、`interpretation_permitted=false`、无完成标记；C未执行。

保存的约243 MiB trace包含序号0至1,999,999的完整连续前缀，链哈希匹配状态文件，因此不是传输损坏。但它只有1个qualification开始记录，不能覆盖完整U1/U2过程；本轮没有利用这个无效前缀作合法域面积损失分解或拓扑统一根因推断，也没有声称找到新的合法路径。

原始启动RNG状态、含患者映射的业务receipt及全部trace保持私有；完整归档约15.3 MB，16个文件均对照私有清单核验。Git只保存经检查的结果／状态、启动摘要、错误和文件哈希及审计元数据。既有12 GiB预留保持未释放，未自动删除任何证据。

[授权使用终态记录](../benchmarks/manuscript_validation_20261103/s07_execution_authorization_consumption_v1.json) 明确本次一次性序列已使用并以hold结束。不得将剩余2次余量改作重跑，不提高事件上限、不采样丢弃记录、不修改算法或复用新目录；若研究新的采集策略或资源配置，应另行审查和授权。本轮到此结束，不启动额外实验。

本次没有获得合格的完整过程证据，也没有改变科学资格：仍为17位源拓扑通过、含unknown上限19位、新增四目标批准0位，正式30位队列未成立。19%／32%名义预算、14%／27%组织硬下限均未改变；正式H&E、批量四目标及医生研究继续关闭。

## S07授权正式结案：只分析保存前缀并设计无损记录格式

按用户明确指示，[本次授权正式结案](../benchmarks/manuscript_validation_20261103/s07_authorization_closure_v1.json)，不重跑S07，不使用未启动C的余量。旧授权、消费记录、原始trace及报告均不覆盖；正式H&E、批量四目标和医生研究继续关闭。本轮没有导入或调用pathology业务模块，新增witness调用0次，也未改Observer、controller、算法或原限额。

[离线容量分析](../benchmarks/manuscript_validation_20261103/saved_guard_capacity_analysis_report_v1.json)逐条处理全部2,000,000个保存事件，先验证原trace SHA、连续序号及状态链；不是抽样。输入总计254,793,387字节（约243 MiB），分布如下：

| 事件类型 | 事件数 | 原始字节 |
| --- | ---: | ---: |
| guard.stage | 1,424,937 | 188,253,069 |
| guard.local_result | 292,147 | 32,968,023 |
| guard.accept | 281,060 | 32,260,730 |
| guard.reject | 1,693 | 264,664 |
| 其余全部 | 163 | 1,046,901 |

Guard类占事件 **99.99185%**、字节 **99.5891%**。仅重复的sequence／event／values／mapping外壳就对应135,205,946字节，约占输入53.06%；这用于解释格式开销，不意味着允许删除这些信息。单条长度中位数132字节、P99为141字节、最大112,869字节，说明高频小记录主导了计数配额。这些是跨guard阶段、重复访问及多次试验的事件计数，不是唯一像素数或独立面积损失。

对相同输入做了两个预先限定的离线实验：每块目标4 MiB、保持完整记录边界，用zlib level1／6无损编码，各61块。压缩载荷分别为 **19,227,635字节（7.55%）** 和 **15,228,134字节（5.98%）**；两种方式均解码、逐条检查全部200万事件并恢复完全相同的原始字节SHA。所有字段、Unicode、非有限值标记、顺序和序号均未抽样、删除或规范化。[完整统计与块校验摘要](../benchmarks/manuscript_validation_20261103/guard_capacity_analysis_v1/capacity_report.json)／[逐类型字节表](../benchmarks/manuscript_validation_20261103/guard_capacity_analysis_v1/event_bytes.csv)已保存，实际压缩payload保持私有。

这些数字仅是**保存前缀的压缩载荷**，尚不包括未来索引／footer及Full Capture数组；离线主机的编码耗时也不是在线业务开销。不能用此比例预测完整U1/U2流量，更不能说原B证据已经有效：前缀仍只有1次qualification开始、0次完成、22次grow开始／21次结束，尾部缺失。本轮没有恢复尾部、生成生产完成标记或进行拓扑因果归因。4项小型合成测试验证了字节／字段／顺序守恒及坏序号、坏链、损坏codec拒绝，未运行真实来源。

[下一版无损块格式草案](../benchmarks/manuscript_validation_20261103/lossless_guard_format_proposal_v1.json)以这些证据为依据，**未实现、未批准上线、未获得执行授权**：保留每一条逻辑事件和原JSONL字节，只改变物理存储方式；逻辑sequence与物理block编号严格分开，不把61个块偷换成61个guard事件。每块记录序号范围／计数、原始及压缩字节数、双重SHA和前块链；pending文件关闭后才原子、不覆盖式提交。全局完成必须另证两项操作终结、全部事件／数组出现次数、无fault、完整receipt和最后digest，块完整不能冒充业务过程完整。

资源政策同样只作为新版本提案：不把旧`max_events`直接调大，不把它悄悄改成block计数。拟在独立审查后，使用明确的物理字节、在途buffer和原运行时预算核算，并保留真实逻辑事件总数；先在现有4 GiB／1200秒范围评估，不增加磁盘／时间额度。解码和B/C比较必须流式进行，不能把压缩后的巨大事件流一次解开为内存对象列表。现有c119比较器不会被默认为支持新格式；新producer／reader需另行固定代码和验收。Full Capture数组体积／压缩性目前没有数据，草案保留逐快照身份及严格库存要求，不私自启用dedup。

下一步仅供审查新格式和资源政策；不能把离线codec往返成功写成完整S07采集成功，也不能直接开启下一轮实验。17／19资格记录、新增四目标批准0、30位目标、19%／32%名义预算和14%／27%组织硬下限均不变。


## 独立 v2 分块无损编码器：本地合成验收完成，尚未接入 Observer

按用户对离线分析的验收意见，本轮直接实现独立存储模块，不重新分析旧200万事件，不调用 pathology planner、guard 或 witness。[格式协议](../benchmarks/manuscript_validation_20261103/guard_block_format_v2.json)只冻结独立实现与合成验证范围；[完整验收报告](../benchmarks/manuscript_validation_20261103/guard_block_codec_v2_validation_report.json)明确生产接入和S07重跑均未获批。旧授权继续结案，原始失败记录、协议、manifest、Observer和controller均不改动。

[编码器／流式解码器](../diagnostic_storage/guard_blocks_v2.py)以约4 MiB为物理块目标，保留每一条原JSONL字节，包括字段、空白、顺序、sequence和编码后的类型／RNG信息；zlib固定level1，不采样、不合并逻辑事件，也不把块数当事件配额。块头、双重SHA、前块链及append-only索引分别绑定逻辑sequence范围和物理产物。解码器按块验证、逐事件产出，数组按出现顺序逐项校验；严格比较不排除字段。只有迭代器完全耗尽后才返回完整校验结论，不积累全部事件或全部索引。

最终同一代码的本机合成结果如下；编码耗时包括合成生成和独立参考哈希，不是在线记录器开销：

| 逻辑事件 | 物理块 | 编码秒 | 独立解码秒 | 编码峰值MiB | 解码峰值MiB | 实际文件占用MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000,000 | 62 | 27.09 | 7.60 | 68.28 | 83.66 | 13.88 |
| 3,000,000 | 92 | 38.62 | 11.54 | 69.75 | 86.42 | 20.91 |
| 5,000,000 | 154 | 65.69 | 20.93 | 82.69 | 85.94 | 34.97 |

三种规模均验证全部逻辑事件、连续sequence和完全相同的原始字节SHA。另以两个独立Reader实例对同一500万事件合成流做严格比较，峰值97.50 MiB；这是流式比较内存验证，不是真实B/C非干扰性验收。小型测试另行核对Hash-only／Full Capture间相同数组token和逻辑字节。

**Full Capture已实际写盘，但仍只证明合成场景。**200万事件加683个512×512 float64快照，原始数组共1,432,354,816字节，所有文件实际占用1,450,278,912字节，编码／解码峰值99.625／91.50 MiB。独立[参考数组核验脚本](../scripts/verify_synthetic_guard_arrays_v2.py)未导入codec，重新生成相同固定种子的683个数组，对每份blob逐字节核对，共约1.43 GB全部一致。数组采用原始无损blob，未压缩、未dedup；不能因快照数量恰好也是683，就推断它代表原S07数组的shape、dtype或容量。

**4 GiB边界测试曾真实失败，保留失败后再修正。**首次压力运行暴露出持续增长索引的文件系统预分配核算不足：失败后无法写出摘要，但未发布完成标记，实际文件占用仍低于4 GiB。[首次失败记录](../benchmarks/manuscript_validation_20261103/guard_block_codec_v2_validation/initial_budget_attempt_failure.json)和私有失败目录保留。修复在统一预算内预留索引2倍逻辑长度加64 KiB，并保留先前观测到的实际分配，不增加总额度；增加反例测试后，五个压力场景全部用最终代码重新运行。

最终边界运行实际保存2,038个2 MiB快照；下一份在复制前被拒绝，实际文件分配4,284,506,112字节，低于4,294,967,296字节上限。故障摘要成功落盘，`block_integrity_complete=false`、`business_process_complete=false`、无正式完成标记，严格读取拒绝。统一核算包括数组、块头、索引、控制文件和pending；pending／final是同inode硬链接，物理占用不重复计算。每数组64 MiB、每事件8 MiB和有限块buffer另有上限；这是有界核算与实测，不是独占quota或操作系统硬RSS保证。

[56项定向测试](../benchmarks/manuscript_validation_20261103/guard_block_codec_v2_validation/pytest.txt)通过，其中52项新codec测试、4项既有离线回归；包括写入／ENOSPC、部分块与关闭失败、索引缺失／重复、数组损坏／孤立、进程中断、无覆盖发布竞争、复制前拒绝及索引预分配反例。块完整与业务过程完整保持独立；必须两者同时满足，才能原子、不覆盖式发布绑定summary SHA的完成标记。完整块但缺业务尾部的前缀永远不能升级为完整诊断。哈希只能发现损坏／不一致，不是抵抗整套目录恶意重写的数字签名。

[复核命令](../benchmarks/manuscript_validation_20261103/guard_block_codec_v2_validation/reproduce_commands.txt)、代码／运行时身份和各场景报告均入库；大体积合成流、数组及失败目录留在私有`.codex-artifacts`，不进入Git。本轮是本地指定测试，不声称全仓CI或1200秒真实业务验收通过。

P0–P3的独立实现与合成验证已完成；P4生产Observer接入尚未开始、尚未批准，独立模块的异常仍传给调用者，未来故障隔离adapter须单独审查。P5仍需要新代码／运行manifest／资源政策的独立验收和新授权。原S07截断证据仍无效，不重跑、不解释缺失尾部。17位拓扑通过、含unknown上限19位、新增四目标批准0位、正式30位目标、19%／32%预算及14%／27%组织硬下限均保持不变；正式H&E、批量四目标和医生研究继续关闭。


## Observer Adapter：独立设计提交审查，尚未实现或接入

用户对`c2fd0741`的后续审查认可独立Codec和合成证据，并要求下一步先设计安全Adapter。本轮只形成[设计方案](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_design_v1.json)和[只读接口核对](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_interface_audit_v1.json)，没有创建Adapter实现、修改Codec／旧Observer／Controller、部署或重跑S07。独立Codec的56项测试和压力结果直接引用，未重复运行；本次只读AST核对仍为5个业务文件、38个既有observe调用。

方案以独立Adapter core处理编码和存储，再由一个小型`FailIsolatedSink`接入**原有**`observation_session`／ContextVar。Sink的`record`、`invalidate`、`close`分别隔离普通异常，先写入预初始化的sticky invalid位，再尝试可失败的core通知。验收必须注入“record与invalidate同时失败”“quiesce与invalidate同时失败”，避免故障处理自身抛错进入业务重规划分支。OS级终止、无法继续记账的进程级OOM和外部控制信号不声称可被完全隔离；缺完整证明即诊断无效。

接口核对识别出以下设计边界，不能直接替换旧类就开始运行：

| 接口 | 设计决定 |
| --- | --- |
| 逻辑事件编码 | 保留旧typed encoder、字段／遍历顺序和JSONL格式；Codec的`dump()`会排序key，不能用于事件编码。生命周期信息放在外部，不添加新逻辑事件或RNG检查点。 |
| 数组交接 | 直接把已计算ndarray交给公开`Writer.add_array()`，只由Codec在预算检查后复制一次；不先调用旧Observer复制，不去重或转换dtype。 |
| session退出 | 旧session会在finally调用close，因此Sink.close只停止事件接收；业务receipt核验和显式finalize在业务调用结束后完成，不能把close当作完整诊断。 |
| 操作身份 | Codec的expected_operations绑定冻结的两个完整case_id，而不是合成默认值U1／U2；complete事件没有operation_id，沿用当前active operation，不改原事件。 |
| 完成证明 | Codec子目录的内层marker仅证明候选流；独立验证器还须核对Sink/core无fault、receipt、身份、模式和完整流，才可发布根目录`adapter_complete.json`。内层marker不能掩盖外层失败。 |
| 比较入口 | 旧`_load()`读取全部事件且要求JSONL／npy，不能用于v2。新外层校验必须核对B=hash-only、C=full-capture，再调用Codec流式严格比较。 |

正常结束的负面witness仍可构成完整业务记录，不会变成拓扑通过；进程退出0则不够，因为原probe可捕获异常后继续写部分checks。最终必须同时核对两项qualification开始／结束、原始source／primitive／case映射、已有receipt及两项完整性状态。A对历史和A/B仍沿用完整decoded JSON receipt相等，不删除字段、不归一化路径；各臂文件SHA用于绑定，不擅自将旧业务比较升级为JSON空白也必须相等。

资源分区目前是**待审查提案**：总观察数据仍限4 GiB，其中256 KiB留给有大小上限的外层控制文件，其余4,294,705,152字节交给Codec；Codec内部原64 KiB完成控制预留、索引预分配及数组／块上限不变。所有root控制文件、索引、blob和pending均在同一总额内核算，不另存完整JSONL或artifact列表。旧Observer的200万事件默认限制不动，新Adapter也不把块数偷换为逻辑事件配额。该新分区尚未经过Adapter整链合成验证，不声称已经可用。

另有一个明确的执行前资源待决项：现有Controller的验证helper固定30秒，而已入库的本机500万事件双Reader比较耗时45.6946秒。这不证明真实S07必定超时，却足以否定“可以直接沿用旧helper并宣称准备就绪”。本轮不提高限额或改Controller；未来须先测量目标环境的合成Adapter校验，再单独冻结helper及总wall预算口径。1200秒来源对、最多6次witness和零重试不因本设计扩大，未解决或超时均hold。

未来实现验收矩阵已列入方案：旧编码器合成差分；Sink/core双重故障隔离；业务返回／异常／重规划／RNG不变；缺终态、错误映射和负面完整receipt；真正Adapter链路的超200万事件和整树4 GiB压力；流式wrapper的模式、身份、完整迭代、外层marker及发布竞争。届时复用已验收Codec，不重复分析原S07前缀，也不能将这些合成验收称为真实B/C非干扰性结果。

当前仅为**P4设计完成、待审查；Adapter未实现，P5未获执行授权**。既有受控RNG条件批准在初始化策略不变时继续作为依据，不因新设计重复申请；新模块导入兼容性和新运行身份仍需审查，初始化策略若改变才另作决定。原S07一次性授权已消费，新诊断必须另行绑定新manifest授权；本轮不制作可启动manifest、不复用旧授权。17／19患者资格、新增四目标批准0、30位目标、19%／32%预算、14%／27%组织硬下限均不变，正式H&E、批量四目标和医生研究继续关闭。


## 独立 Observer Adapter 已实现：定向测试通过，大规模比较保留时间 hold

按用户最新授权，P1–P3推进独立Adapter实现与合成验收，不接入生产Observer、不改变旧Codec或业务逻辑。本轮[实现接口与资源合同](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_implementation_contract_v1.json)另存，保留上一版设计原文和所有历史协议／授权。[完整实现报告](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_implementation_report_v1.json)的整体状态是 **`independent_adapter_implemented_local_tests_passed_large_comparison_timeout_hold`**，不是已可执行S07。

三个新模块分别负责[有界文件／receipt合同](../diagnostic_storage/observer_adapter_contract_v1.py)、[类型编码core和故障隔离Sink](../diagnostic_storage/observer_adapter_v1.py)、[独立外层流式验证器](../diagnostic_storage/observer_adapter_validation_v1.py)。Adapter不继承旧Observer，不修改其ContextVar、2M默认限制、38个业务观测点、Controller或RNG控制。只有显式把Sink交给原`observation_session`才启用；Codec继续固定为`c2fd0741`对应文件SHA，原56项Codec及2M／3M／5M证据复用，没有再跑一遍。

Sink先锁定自己的无效位，再通知core；`record`或`quiesce`与core故障处理同时抛错时，也不会把普通记录器异常交给业务catch／重规划。core类型编码AST与旧版只有ndarray分支委托给公开`Writer.add_array()`这一处变化；metadata计费与`_rng`的AST一致、检查点集合一致。数组不预先复制，不去重、不改dtype，Codec承担唯一一次预算检查后的独立拷贝。异常消息不通过自定义str/repr写故障日志，原业务返回／异常继续沿原路径处理。

session退出只quiesce。显式`finalize(normal_invocation_end, business_receipt_path)`在业务调用返回后检查原文件，绑定真实SHA和完整case ID。后续重复finalize是无写入的幂等返回，不是重试。外层验证器独立核对完整case／primitive映射、两个qualification开始／结束、终态failures／witness与receipt对应内容、Sink/core无fault、模式与身份、完整迭代的Codec Reader及整树预算，再发布根目录接受标记。内层Codec marker只是必要条件；其存在不能掩盖外层缺失或错误。失败witness可以被完整记录，但永远不等于患者合格。

[113项定向测试](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_validation_v1/pytest.txt)通过：51项新Adapter场景、62项旧Observer回归，不是全仓CI。覆盖普通写入／ENOSPC／copy／编码／队列／RNG读取故障，record+invalidate及quiesce+invalidate双重故障，原业务异常对象保留，receipt错配／缺终态／exit0部分结果，数组预算前拒绝，封存／控制文件关闭失败、发布竞争、进程中断、身份／模式／外层proof损坏，以及硬链接统计反例。64×64合成执行器与有界重规划fixture比较了off/hash/full/fault结果、mask、ownership／guard／计划有序事件、RNG和固定anchor／重规划序列；这些是合成函数测试，未调用BCSS／S07来源资格入口，不能替代真实非干扰性验收。

[210万事件Adapter整链压力](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_validation_v1/pressure_2100000.json)经过新的类型编码、单槽队列、Codec、receipt和独立验证进程：

| 合成模式 | 事件／块 | 数组occurrence | 编码秒 | 单流外层验证秒 |
| --- | ---: | ---: | ---: | ---: |
| Hash-only | 2,100,000／54 | 685 | 37.76 | 7.99 |
| Full Capture | 2,100,000／54 | 685 | 52.01 | 21.54 |

两流完整sequence和字节SHA一致，Python／NumPy全局RNG状态未改变。685包含683份新512×512 float64快照和2份既有NumPy legacy RNG状态快照；不能把它混写成原S07的683份Raster。独立重新生成683份合成快照，1,432,354,816字节原始payload逐字节一致。生产者进程累计峰值约106.47 MiB，两个验证子进程峰值约95.98／78.48 MiB；生产者B/C在同一进程运行，因此其峰值不是每模式隔离测量，更不是硬RSS保证。

**大规模完整包装比较没有通过原时间门槛。**两份单流均在30秒内完成，但重新验证两流并做严格流式比较的完整wrapper耗时 **41.55秒**，超过保留的30秒，返回`timeout_hold`，没有`semantic_match`，未增加限额或自动重试。各臂根marker仅表示单流完整性，不代表B/C或整个实验已接受。压力脚本返回0表示完成报告生成，不能据此覆盖报告内的hold。库内比较deadline是协作式前后检查，不能强行中断正在执行的公开Codec.compare；未来Controller仍须单独提供硬进程超时并审查helper／总wall预算，本轮没有实现或部署新的Controller。1200秒来源对和6次／0重试范围不变。

[整树4 GiB边界](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_validation_v1/whole_tree_budget.json)将外层控制文件也包括在内：256 KiB控制分区＋Codec剩余分区仍在同一4 GiB内。实际接受2,038份2 MiB快照后，下一份在复制前拒绝；两项Codec完整性状态均false，内外完成标记均未产生，外层验证拒绝，故障摘要仍保存。原现场统计即使高估也低于4 GiB。

汇总阶段发现**测试脚本的占用观察器**把`.index.pending`／`.index.jsonl`误当两个文件，Adapter／Reader自身预算核算未受影响。原报告保留，另有[inode去重统计更正](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_validation_v1/allocation_observation_correction.json)，并增加反例测试；没有重跑大规模事件。同一轮后续stat中，Hash／Full整树占用约15.72 MB／1.451 GB，预算失败树约4.285 GB。原现场与后来stat不是同一时刻，文件系统extent／预分配回收也可能改变st_blocks，因此不能把全部字节差都解释为重复计数；更正记录另列同一pass旧统计与inode去重结果。初次1000事件smoke的合成调度器还曾把快照位置排到qualification边界，导致少一份数组；其失败目录与[说明](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_validation_v1/smoke_generator_failure.json)也保留，修复只涉及合成guard ordinal排程，不涉及Adapter／Codec或任何真实来源。

代码／运行时与旧文件未变哈希、只读AST结果、复核命令及原始压力摘要均已入库；大体积合成流／数组／receipt及失败目录保持私有。所有类型错配或无法完整封存都hold，不作事后tuple／路径／字段归一化；OS终止、进程级OOM和整套目录恶意重写不声称可完全隔离或防护。

当前更新为：**独立Adapter实现完成，指定本地故障和合成非干扰场景通过；大规模比较时间门槛hold；待独立终审、新启动／执行身份及新授权。**尚未接入生产、未部署、未重跑S07，原授权不复用。17位源拓扑通过、含unknown上限19位、新增四目标批准0、30位目标、19%／32%预算及14%／27%硬下限均未改变，正式H&E、批量四目标和医生研究继续关闭。


## 单遍双流比较：保留全部校验，本机合成通过，仍待独立终审

初版实现及41.55秒hold已提交为`44eea864`，旧报告／合同／目录不覆盖。代码检查确认初版先分别完整验证两流，再由Codec.compare完整重读；本轮只改[外层验证器](../diagnostic_storage/observer_adapter_validation_v1.py)，将元数据准备、checked-events和finish检查共用，让两个**公开Codec.Reader迭代器各遍历一次**，在同一遍完成全部字节比较、数组／索引／哈希／sequence、终态与receipt、外层proof和整树预算核验。没有相信旧marker跳过校验，也没有改Codec、core／Sink、编码／RNG、业务文件或30秒限额；另在流耗尽后重读有界控制文件和receipt绑定，拒绝校验期间被修改的文件。

[117项定向测试](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_validation_v2/pytest.txt)通过：55项Adapter、62项旧Observer回归。新增检查证明每流仅一次Reader遍历、已有接受marker也不能掩盖数组损坏、流中receipt变更会拒绝、未知字段不被排除、deadline在途中仍生效。既有原始异常、ownership／guard／计划顺序、重规划和RNG合成对照全部保留。

按新的准确代码身份，在**新目录**重新运行210万事件×两个模式、683份2 MiB新快照（另有2份RNG数组）的Adapter压力，未修改旧目录或用新代码冒充旧manifest身份。[最新完整结果](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_implementation_report_v2.json)／[v2接口合同](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_implementation_contract_v2.json)显示：单流验证约8.95／15.18秒，完整单遍双流比较 **27.569秒**，返回`semantic_match`，全部2,100,000事件连续、字节SHA相同、数组与receipt完整，排除字段为空。683份合成快照约1.43 GB再次独立重新生成、逐字节一致；进程前后Python／NumPy全局状态不变。v1／v2各自自然启动的全局RNG初态可能不同，本轮不重置它们，也不声称恢复了历史随机轨迹。

另外只运行一次[新进程只读比较](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_validation_v2/fresh_process_comparison.json)，不生成事件、数组或witness：比较函数29.808秒，整个子进程 **29.982秒**，在父进程硬30秒限制内正常完成，峰值 **102 MiB**。这是非常小的本机余量，不能推断目标服务器或更长的真实S07轨迹一定满足同一限制；未为获得更快样本而重试。源码消除了重复遍历，但两次本机时间并不是受控性能研究，不将全部耗时差归因于这一个因素。

新代码的[整树4 GiB边界](../benchmarks/manuscript_validation_20261103/observer_codec_adapter_validation_v2/whole_tree_budget.json)也重新核验：2,038份快照后下一份复制前拒绝，实际inode去重分配4,284,502,016字节，低于4,294,967,296上限；Codec两项完整性状态false，内外marker不存在，外层拒绝。最新测试脚本已正确按inode计数，之前的统计和更正仍作为独立历史证据保留。

因此，最新状态是**独立Adapter及指定本地合成／故障／压力验收通过，待独立终审、目标运行环境与硬超时／导入身份核验、新manifest及新S07授权**。本轮没有提供或部署新Controller，更没有启动真实诊断。真实完整trace的数量／时长仍未知，本机27.57／29.98秒不构成正式执行放行。生产Observer、旧2M限制及旧授权不变；17／19患者资格、四目标批准0位、30位目标、19%／32%预算和14%／27%组织硬下限不变，正式H&E、批量四目标和医生研究继续关闭。


## 目标服务器技术准备：硬超时门禁有效，完整比较维持 HOLD

用户对`32dc6007`的独立审查通过后，本轮只做生产接入前准备，不修改已验收Codec／Adapter／Validator，也不接入生产或重跑S07。新增[只读进程门禁](../diagnostic_storage/validator_process_gate_v1.py)与[固定worker](../scripts/validator_readonly_worker_v1.py)，代码封存于`dfeee1c2`；旧Observer／Controller／startup文件以及既有2M默认限制均未改动。[本轮总报告](../benchmarks/manuscript_validation_20261103/adapter_target_preflight_report_v1.json)结论为 **`HOLD_target_full_pair_validation_exceeded_frozen30s`**。

门禁父进程仅用标准库，把固定的只读验证worker放入独立进程组。30秒从创建进程前开始，包含输入传递、执行、输出和退出等待；接收JSON后再次核对时间，晚到exit0也hold。超时／非零／信号／响应错绑／输出超限／可捕获取消触发终止与回收，不重试；worker没有publish开关，单流调用固定`publish=False`，父门禁也不发布接受证明。外层生产Controller的实际接线与最终父进程发布路径尚未实现或批准，不能把这个准备组件当成已完成生产接入。不可捕获的父进程SIGKILL／内核故障仍不能保证全部清理；只读worker即便残留也没有发布权限。

[64项本地定向测试](../benchmarks/manuscript_validation_20261103/adapter_target_preflight_v1/local_gate_pytest.txt)通过：26项新门禁、38项既有Controller回归。覆盖进程组超时与后代清理、晚到有效响应、原始信号／非零、stdout／stderr上限、外部取消回收、请求SHA／模式绑定，以及在原Controller.sequence的合成assess中超时后不启动C。这里的execute是合成占位，不是真实来源调用；已接受的117项Adapter和56项Codec证据复用，未再跑一遍。

[目标环境审计](../benchmarks/manuscript_validation_20261103/adapter_target_preflight_v1/server_environment_audit.json)确认Python3.11.15、NumPy1.26.4、zlib1.3.2，解释器SHA和220项distribution记录与此前服务器冻结身份一致；新记录还固定了NumPy本地扩展、zlib扩展及实际加载libz的文件SHA。它是运行身份，不是完整hermetic OS闭包。原A／B快照的10项业务文件哈希／去观测调用AST核对通过，7项真实源输入／模板／旧Controller／RNG决策文件哈希通过，旧Observer／Comparator／startup另行复核未变。所有检查均只读，不重建case，不调用planner或来源资格入口。

实际候选输出父目录位于ext4，隔离技术目录的no-replace硬链接和同inode验证通过，与候选输出父目录同设备；未来S07候选目录尚未创建。已有12 GiB物理预留仅检查、未修改，仍非排他quota。代码只复制到新的只读技术快照目录，9个文件前后哈希一致，没有安装到现有生产快照或服务路径。既有受控RNG条件与两个提前导入模块的接受记录保持原SHA；本轮只读worker未加载business模块或Torch，不冒充新的完整业务启动／导入非干扰性验收。

**保留一次未完成的构建尝试。**初次目标服务器合成生成在预先设置的600秒构建看门狗下约600.89秒后SSH返回255，B已封存、C未封存，无最终profile报告。确切终止信号未留存，不把它伪装成一个完整Validator超时测量；B精确验证耗时等未持久化数据保留null，不补零、不从代码分支反推数值。[原尝试记录](../benchmarks/manuscript_validation_20261103/adapter_target_preflight_v1/first_builder_attempt_hold.json)和全部部分目录保留，没有恢复或重跑该生成任务。

为了把生成成本与验证时限分开，下一项是事先定义的**验证-only实验**：[固定闭合合成输入投影](../benchmarks/manuscript_validation_20261103/adapter_target_preflight_v1/closed_fixture_projection_protocol.json)。复用`32dc6007`已接受的2,100,000事件／685数组（683个新快照＋2个RNG快照）完整B/C数据，809个canonical文件逐字节SHA核对后复制到新目录，再重建同inode pending别名。仅不携带旧外层接受／validation控制文件，要求新验证器完整重验；逻辑事件、数组、索引、receipt、manifest/status及内层封存文件均未改写，原数据和旧证明未删除。这个物理投影不是排除语义字段，也不是恢复截断的旧S07 B。原始全清单、RNG数据、约1.465 GB归档和新拷贝均私有保存，不入Git。

[目标服务器只读结果](../benchmarks/manuscript_validation_20261103/adapter_target_preflight_v1/target_readonly_result.json)每一步立即保存receipt，结果如下：

| 步骤 | OS进程总耗时 | 状态 |
| --- | ---: | --- |
| B单流 | 19.373秒 | validated_readonly |
| C单流 | 23.703秒 | validated_readonly |
| 完整B/C比较 | 30.004秒（含终止／回收） | **HOLD：timeout，SIGKILL退出-9，已回收** |

所有步骤仍用硬30秒，未改60秒，未自动重试超时比较。完整比较没有返回worker receipt或semantic_match，不得根据源SHA相同推断已完成，也不得把缺失结果填零。B/C外层接受标记均不存在；输入数据的既有内层物理封存不因此升级为观察接受。独立目标安全fixture也验证了强制超时阻止下一验证步骤、晚到写入未发生、进程已回收。这些是验证门禁与合成数据证据，不是S07预先验收，也不能证明真实Full Capture必然适合4 GiB。

[执行关闭的准备manifest](../benchmarks/manuscript_validation_20261103/adapter_pre_execution_preparation_manifest_v1.json)另存，固定源码、实际运行环境、文件系统／预留、旧RNG／导入披露／授权结案记录及所有证据SHA；`execution_permitted=false`、`ready_for_execution=false`、`preflight_complete=false`、生产接入false、无新授权。它不是可启动的launch manifest。未来须先独立审查目标时限／资源条件与生产接线，再另冻最终launch manifest并绑定新授权，旧授权和剩余调用槽不能复用。

本轮已明确回答最关键的执行问题：**这一次目标服务器完整比较未能在冻结30秒内结束，因此不能放行。**没有优化或削减检查来隐藏这个负面结果。17位源拓扑通过、含unknown上限19位、新增四目标批准0、30位患者目标、19%／32%名义预算及14%／27%组织硬下限不变；正式H&E、批量四目标和医生研究继续关闭。

## 只读验证预算 v2：候选实现与试验封存，等待审查

依据对 `aabe1577` 的审查，本轮把工程时限与研究资格门槛分开处理：[候选预算](../benchmarks/manuscript_validation_20261103/readonly_budget_v2_review/validation_budget_policy_v2.json)保持单流30秒，为双流完整比较提出60秒硬上限。旧30秒HOLD永久保留，不改写为通过。**60秒尚未获批或在目标环境验证**，本轮未连接服务器、未部署新快照、未启动任何目标测量。

实现提交为 `1c8ad6ff50e1c94719141e17b519ecf5eee93a2c`。新增独立v2门禁、worker及只读验证封装；可信任务映射选择时限，worker显式传入库内部，父进程继续独立硬超时。capture输入不能自行设置时间，父进程还核对worker及库返回的政策、时限和消费端代码身份。不改旧门禁、旧Validator、Codec、Adapter或生产Controller，不增加发布接受证明的入口。

[producer／consumer兼容提案](../benchmarks/manuscript_validation_20261103/readonly_budget_v2_review/producer_consumer_compatibility_v2.json)仅接受精确哈希固定的旧组件：新封装复用旧 `_prepare`、`_checked_events`、`_finish`，每条流只扫描一次，完整检查数组、索引、终态、receipt及控制文件绑定。producer与consumer身份分别记录；旧manifest代码哈希不改，不做字段排除或receipt归一化。此兼容提案仍待独立审查，不是通用跨版本兼容声明。

[本地测试](../benchmarks/manuscript_validation_20261103/readonly_budget_v2_review/pytest.txt)49项通过，包括23项新预算／身份／门禁测试和26项旧门禁回归。真实微型worker测试贯通30／60秒配置，模拟时钟验证新库可越过旧30秒界限，错误政策／身份、输入覆盖时限和进程超时均拒绝。这里没有重跑117项Adapter、56项Codec或真实210万事件压力，也不把本地微型结果当成目标服务器验收。

[封存的固定输入试验manifest](../benchmarks/manuscript_validation_20261103/readonly_budget_v2_review/fixed_synthetic_trial_manifest_v2.json) SHA256 为 `bf47f607b2dab75aad7053256c617d227ee45723f410e5126b5e02f4a440c7ec`。拟复用已有目标目录中的完整合成B/C输入，不重新生成：每流210万事件、685个数组occurrence，809个canonical文件绑定私有完整清单SHA。预定顺序为B单流30秒一次、C单流30秒一次、B/C比较60秒一次；任一步HOLD停止，零自动重试。启动前仅一次完整输入／代码／环境核验，明确披露哈希扫描可能预热文件缓存；不追加择优测速或调整缓存的尝试。实际新输出目录重新做no-replace硬链接检查、空间及既有12 GiB物理预留检查，预留仍非quota。

[审查报告](../benchmarks/manuscript_validation_20261103/readonly_budget_v2_review/review_report_v2.json)将状态限定为“本地候选完成，目标试验未运行”。预算与精确兼容规则获审查接受、并取得与本次manifest SHA绑定的一次合成试验批准后，才允许部署隔离技术快照和运行这三项测量。**这不是新的S07申请，也不沿用旧S07授权。**生产接线、最终接受证明发布仍是后续独立工作。17／19患者记录、新增四目标批准0、30位目标、19%／32%名义预算和14%／27%组织硬下限保持原样，正式H&E、批量四目标和医生研究继续关闭。

## 已批准的一次合成试验：setup超时HOLD，三个测量均未启动

用户明确回复“接受提案并批准这一次合成试验”，已另存[授权记录](../benchmarks/manuscript_validation_20261103/readonly_budget_v2_review/fixed_synthetic_trial_authorization_v2.json)，绑定原封存manifest SHA `bf47f607b2dab75aad7053256c617d227ee45723f410e5126b5e02f4a440c7ec`、30／60秒政策及精确producer／consumer兼容规则。该记录是助手对本聊天用户明确决定的留痕，不伪装成外部审查签名；原提案、manifest和旧报告保持原始字节。

[本次目标证据](../benchmarks/manuscript_validation_20261103/readonly_budget_v2_review/target_trial_evidence_v2.json)显示：11个源码文件及manifest、approval、私有输入清单共14个文件部署到新的隔离只读快照，未修改生产路径。随后只启动了一次封存runner，授权已消费。**启动前核验触及预先冻结的60秒setup预算，runner捕获TimeoutError，约60.371秒总进程时间后以78正常退出并保存HOLD；B单流、C单流和B/C比较全部未运行。**185秒总看门狗未触发，没有自动重试。

[结果报告](../benchmarks/manuscript_validation_20261103/readonly_budget_v2_review/target_trial_report_v2.json)明确将三个测量的耗时与verdict记为null、状态为`not_run_setup_hold`。这次不是“双流60秒比较超时”，也没有得到semantic_match或不一致结论。runner只保存异常类型，没有保存setup内部阶段或逐文件进度，所以不能确定耗时集中在哪一项检查，也不能宣称完整运行环境或809个输入文件已核验通过。60.371秒是含启动／退出开销的总时间，不是精确setup内部耗时。

只读归档核对确认11个部署源码哈希仍匹配，B/C原manifest哈希未变，两个输入根目录均无外层接受证明；实际新输出仅有`trial_result.json`，本次输出目录的hardlink探针尚未到达。既有内层封存不升级为新接受证明。后续读取只收集小型控制文件和源码哈希，没有再次调用验证器、没有全量重扫输入或择优测速。本次授权至此关闭，不能因三个测量尚未启动就复用；保留目录，不resume、不扩时限。下一步先离线审查setup身份核验工作量和阶段遥测缺口，再决定是否提出新的版本化方案。

旧`aabe1577`仍是30秒政策下的完整比较HOLD。本次没有修改验证标准，也没有接入生产Controller、运行S07或生成H&E。患者17／19、新增四目标批准0、30位目标、19%／32%预算和14%／27%组织硬下限均保持不变。

## 2026-11-04：离线工作量分析与 setup 阶段遥测，本地验收完成

依据对 `29206db6` 的审查，本轮只处理准备阶段的可观测性。[离线报告](../benchmarks/manuscript_validation_20261104/setup_telemetry_v3/offline_workload_report_v1.json)只读取本地保存的清单和环境审计JSON，没有连接服务器或打开清单所指向的数据文件。809个canonical文件总计 **1,463,747,425 bytes**：685个数组占1,432,607,288 bytes（97.8726%），108个事件块占30,271,096 bytes，16个控制／索引文件占869,041 bytes。220个distribution中217个保存了RECORD哈希，3个为空；现存记录没有RECORD大小或其他元数据读取量，这些规模保留null。不得从哈希推算文件大小，也不得据此断言旧超时发生在数组扫描或distribution阶段。

新增[准备工具](../scripts/preflight_setup_telemetry_v3.py)，代码提交 `62122a68`，不修改旧runner、Codec、Adapter、Validator、门禁或生产业务文件。按原顺序记录legacy code、runtime code、解释器／环境、distributions、native binaries、input inventory、storage七个阶段。单调时钟记录开始／完成、elapsed_ms、已完成项数、完整哈希读取的字节量及零起始最后项索引。distributions的单项“完成”仅指枚举／哈希完成，必须在阶段末通过整体packages SHA才能宣布该阶段通过。失败或被中断的部分哈希字节不估算；输入inventory控制文件的哈希字节计入该阶段。原身份核验、遍历顺序、deadline检查位置与60秒setup预算均保留。

[遥测协议](../benchmarks/manuscript_validation_20261104/setup_telemetry_v3/setup_telemetry_protocol_v3.json)固定日志上限1 MiB、2048条、单条2 KiB及7个阶段内存状态。每个完成项与阶段边界写一条append-only记录，不写路径、包名、原始异常内容或输入数据。异常时保存当前阶段进度和最后完整阶段；SIGKILL只能留下前缀，缺少最终summary必须视为不完整。写入、容量、flush或close失败不改变身份检查结果，但会使遥测无效、准备结论HOLD；身份通过与遥测有效分别标记。独立summary用exclusive-create发布，不是capture接受证明，工具没有验证器调用或授权入口，也不复用旧输出。

[46项本地定向测试](../benchmarks/manuscript_validation_20261104/setup_telemetry_v3/pytest.txt)通过，包括23项新setup测试与23项既有预算回归。成功路径逐项核对与旧v2相同的哈希读取顺序；模拟慢哈希与慢storage保留原超时，覆盖哈希不匹配、读取异常、中断、遥测故障与身份错误同时发生、部分写入、最终summary失败及异常退出前缀。809个微型输入与220个合成distribution项的场景验证固定容量边界。这些是本地合成证据，不是目标服务器性能验收；没有重跑旧Adapter／Codec压力或全仓CI。

[实现报告](../benchmarks/manuscript_validation_20261104/setup_telemetry_v3/implementation_report_v3.json)和协议封存代码SHA及旧组件SHA。本轮无服务器连接、无部署、无全量输入重扫、无新B/C测量。setup仍60秒，B／C单流各30秒、完整比较60秒，零自动重试。新增日志本身有CPU与I/O开销，不能声称它保证60秒足够；原setup门禁仍是原位置的协作式检查，阻塞读取会延迟检测，未改称OS硬实时保证。方案当前仅供独立审查，新的目标launch manifest和执行授权尚未形成；旧授权保持已消费，旧HOLD永久保留，不补写历史阶段遥测。

患者17／19、新增四目标批准0、30位目标、19%／32%名义预算、14%／27%硬下限保持不变；S07、生产接入、正式H&E、批量四目标和医生研究均未启动。
