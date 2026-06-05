# Cross V5 模型架构设计

Date: 2026-06-05

## 0. 一句话结论

Cross V5 不再把问题定义成“ControlNet 加 reference attention 是否足够强”，而是重新定义为：

**target masks 提供几何结构，reference image 与 reference masks 构建按 mask id 索引的 appearance bank，生成器把 bank 中对应类别的真实病理外观渲染到 target 结构上；当 reference 缺少某类时，使用 learned per-class prior bank 兜底。**

也就是：

- target tissue mask 决定每类组织在哪里。
- target nuclei/cell geometry 决定核在哪里、多大、边界和密度如何。
- reference tissue mask 把 reference image 拆成 tissue-level appearance bank。
- reference cell/nuclei mask 把 reference image 拆成 cell-level 或 nuclei-level appearance bank。
- class-wise router 根据 target mask id 选择对应 bank。
- renderer 只从对应 bank 注入外观，不再让模型自己在全局 reference tokens 里“猜”应该看哪一类。

Cross V5 的核心原则是：

**ControlNet = where / shape / geometry；Reference Bank = what it looks like。**

### 0.1 2026-06-05 设计回写：三条硬决策

本版设计新增三条必须落地的约束。

第一，**min 版不是不要 cell mask，而是只使用 cell mask 的几何信息**。cell mask 的 binary nuclei mask、boundary、distance transform、centroid heatmap、size/radius map 和 density map 从第一版就进入 ControlNet，因为它们是 target 结构的一部分。暂缓的是 cell type id 到 CellBank 的 appearance routing，不是暂缓核几何。

第二，**appearance fidelity loss 使用像素级 stain statistics 加 masked VGG Gram 两路，不使用 UNI2-h 做 appearance 裁判**。颜色/染色是主力监督，优先比较 H&E 解卷积通道的 region mean/std，退而可用 LAB/RGB 统计；纹理使用 VGG 浅层到中层的 masked Gram。UNI2-h 只保留给 semantic 或 geometry consistency，不进入 appearance loss。

第三，**ControlNet 与 reference bank 的竞争用三层防线处理**：class id 从 ControlNet 移出，让 ControlNet 物理上只知道几何；appearance 接 AdaLN-style normalization modulation，而不是和 ControlNet 在同一个 hidden state 上做残差加法；appearance modulation 的 scale/shift 不做 zero-init，让 ref bank 从第一步就有实质影响。

## 1. 从 Cross V4 到 Cross V5 的问题重构

Cross V4 已经实现了 target-only ControlNet、reference local tokens、token metadata、coarse tissue priors 与 mask-guided attention bias。它的问题不是 reference 完全没有进入模型，而是：

**reference 仍然只是一个可被模型忽略的 context，不是不可替代的 appearance source。**

在当前结构中，target mask 加 ControlNet 加 FLUX prior 已经足够生成一个平均合理的病理图像。于是模型可以走 shortcut：

- target masks 负责组织布局、细胞密度、边界和类别。
- ControlNet 残差同时携带结构与局部高频线索。
- FLUX prior 补上平均病理纹理。
- reference tokens 只提供弱染色扰动，甚至在 same-class swap 时 A/B reference 输出几乎一致。

因此 V5 不把失败归因于单个 attention bias、学习率或 swap loss，而是改架构职责：

- 从“target token 更倾向 attend 同类 ref token”升级为“target class 只能从对应 class bank 取 appearance”。
- 从“reference 是辅助上下文”升级为“reference bank 是有类覆盖时的主外观入口”。
- 从“swap sensitivity 证明 reference path 活着”升级为“region-wise appearance fidelity 直接约束生成区域像对应 reference bank”。

## 2. 总体数据流

Cross V5 的高层数据流分成两条路径。

Reference path:

- 输入 reference image。
- 输入 reference tissue mask。
- 输入 reference nuclei/cell mask。
- 由 Reference Appearance Bank Builder 按 mask id 构建 tissue bank 与 cell bank。
- 每个 bank 同时保留低频 prototype、类内多模式 prototype、局部 texture tokens 与可解释 appearance statistics。

Target path:

- 输入 target tissue mask。
- 输入 target nuclei/cell geometry maps。
- Target Geometry Encoder 与 Geometry ControlNet 只编码位置、边界、距离、密度和形状。
- Target 的 tissue id 与 cell id 不作为强 appearance condition 进入 ControlNet，而是作为 router 的索引。

Generation path:

- Class-wise Appearance Router 对每个 target region 或 token 执行 hard class routing。
- Appearance Injection Renderer 将对应 bank 的颜色与纹理 style 全部接入 SEAN 式 spatial AdaLN；高频纹理不再通过残差 cross-attention 注入。
- 对 reference 缺失类别，router 自动切到 learned prior bank。
- 输出符合 target geometry 且 appearance 对齐 reference 同类区域的 generated pathology image。

## 3. 输入与输出定义

### 3.1 训练输入

每个训练样本包含：

- reference image：真实病理 patch。
- reference tissue mask：reference 中每个像素的 tissue class。
- reference nuclei/cell mask：reference 中每个核或细胞的类别、区域或至少空间分布。
- target tissue mask：生成图需要满足的 tissue layout。
- target nuclei/cell geometry：核位置、大小、边界、距离场、质心热图、密度图等几何条件。
- target tissue/cell class ids：仅用于 routing 与 loss 分区，不作为强 appearance condition 直接喂给 ControlNet。
- target image：训练时的真实 GT。

### 3.2 推理输入

推理阶段的 target masks 来自 mask editing pipeline，reference 可以来自：

- 同一 WSI 的真实 patch。
- 用户指定的参考图。
- 检索得到的同类病例 patch。
- 跨病例或跨数据集 reference。

推理输出是一张 target image：

- 几何上服从 target tissue/cell masks。
- 有 reference 覆盖的类别，appearance 来自 reference 对应类别。
- reference 缺类时，appearance 来自 per-class prior，同时尽量保持全局 stain 协调。

## 4. Reference Appearance Bank Builder

Reference Appearance Bank Builder 是 V5 的第一核心模块。它不把 reference image 压成单个全局向量，而是按 mask id 拆成多个可路由的 appearance bank。

### 4.1 Tissue Bank

对每个 tissue class 构建一个 TissueBank。

每个 TissueBank 至少包含：

- class presence flag：reference 是否存在该类。
- low-frequency prototype：该 tissue class 的整体染色、亮度、H/E 比例和低频色调。
- sub-prototypes：类内多模式外观，例如同一个 tumor class 中的实性区、腺体区、坏死边缘区。
- local texture tokens：该类局部区域的高频纹理、细胞排列、纤维结构、坏死颗粒感等。
- stain statistics：H&E 解卷积通道、LAB 或 RGB 空间中的均值、方差、直方图统计。
- texture statistics：浅层 CNN 或 VGG-style Gram 特征，用于纹理监督。
- cell composition summary：该组织区域内不同 cell/nuclei 类型的比例、密度与空间分布。

TissueBank 的目标不是表示“这是 tumor”这个语义，而是表示“reference 中的 tumor 长什么样”。

V5-min 的 bank source 必须与 appearance judge 同源：

- prototype 不使用 UNI2-h。默认 prototype 是 reference RGB 在对应 tissue mask 内的 HED stain statistics：H/E 通道 mean + std，可选 H/E covariance。
- 如果需要更强表达力，可以把浅层 CNN/VGG 的 region-pooled texture feature 拼接到 HED statistics 后面，作为 `hed_stats+token_pool` prototype。
- local tokens 不使用 UNI2-h token，而使用浅层 VGG 或 pathology texture encoder 的 feature tokens，用于保留高频纹理、纤维和坏死颗粒感。
- UNI2-h 只保留给 semantic/geometry consistency，不进入 appearance bank，也不作为 appearance loss 的主裁判。

这样 `source` 与 `judge` 闭环对齐：bank 用 HED/VGG 低层视觉空间表达外观，AdaLN 用这些 prototype 产生 `gamma/beta`，appearance loss 再用 HED 统计和 masked VGG Gram 检查输出。

### 4.2 Cell Bank

CellBank 是 V5 的扩展模块，不作为 V5-min 的首轮实现目标。

对每个 cell/nuclei class 构建一个 CellBank，包含：

- nucleus prototype：该 cell class 的平均核染色、大小、边缘清晰度。
- chromatin statistics：染色质颗粒感、深浅、粗细和异型性。
- local nucleus tokens：核或细胞级局部 texture tokens。
- size/shape distribution：reference 中该 cell class 的大小、长宽比、圆度和边界统计。
- density/context statistics：该类细胞的局部密度与邻域共现模式。

CellBank 的目标是让 target cell geometry 决定核在哪里、多大、什么类型，而核外观来自 reference 中对应 cell class。

但由于 cell-level token attention 的显存、噪声和尺度问题，V5-min 应先只做 tissue-level bank。等 tissue-level 迁移稳定后，再将 CellBank 作为 V5.1 增量接入，且优先采用统计级调制，而不是一开始就做 per-nucleus cross-attention。

### 4.3 Prior Bank

当 reference 缺少 target 所需类别时，router 使用 learned prior bank。

Prior bank 按 class id 组织：

- PriorTissueBank[class]：提供该 tissue class 的默认病理 appearance。
- PriorCellBank[class]：提供该 cell/nuclei class 的默认核外观。

训练时必须加入 class-bank dropout：

- 随机隐藏 reference 中本来存在的某些 class bank。
- 强制模型在这些类别上走 prior bank。
- 防止 prior bank 因为真实 batch 中经常被 ref-derived bank 覆盖而训练不足。

## 5. Target Geometry Encoder

Cross V5 中 target path 只负责几何与 routing，不负责外观。

### 5.1 Geometry ControlNet 输入

ControlNet 应接收 class-agnostic 或弱 class 的几何条件：

- target tissue region boundary。
- target tissue coarse layout。
- binary nuclei mask。
- nuclei boundary map。
- distance transform。
- centroid heatmap。
- radius/size map。
- low-frequency density map。
- optional instance separation map。

这些几何图可以从 cell/nuclei mask 直接生成。也就是说，cell mask 在 V5-min 中必须存在并进入 ControlNet，只是进入方式是几何化后的条件，而不是“cell type appearance shortcut”。

这些条件回答的是：

- 哪里有组织区域。
- 哪里有核。
- 核有多大。
- 边界在哪里。
- 组织和核的密度如何。

它们不回答：

- tumor 核应该多深染。
- stroma 胶原应该多粉。
- necrosis 颗粒应该是什么纹理。
- inflammatory nuclei 应该是什么染色质风格。

### 5.2 Class ID 的角色

target tissue id 和 target cell id 不再作为强 appearance signal 注入 ControlNet。

它们只用于：

- router 选择 TissueBank 或 CellBank。
- loss 按 region/cell 分区计算。
- evaluation 做单类 bank 替换和类别级 appearance 检查。

这样可以避免 ControlNet 仅凭 class id 与 FLUX prior 生成“平均 tumor”“平均 stroma”，从而绕过 reference bank。

## 6. Class-Wise Appearance Router

Class-wise Appearance Router 是 V5 的第二核心模块。

### 6.1 Tissue-Level Hard Routing

对每个 target tissue region 或 packed target token：

- 读取 target tissue id。
- 如果 reference 中存在该 tissue class，则路由到 TissueBank[class]。
- 如果 reference 中缺少该 tissue class，则路由到 PriorTissueBank[class]。
- 类间 routing 必须是 hard routing，不能 soft 混合到其他 class bank。

Hard routing 的原因是：V5 的目标就是堵住跨类泄漏，让 tumor token 不能为了降低 denoise loss 去读取 stroma bank，也不能让 stroma token 偷读 tumor bank。

允许 soft 的地方只在类内：

- 同一个 TissueBank 内的多个 sub-prototypes。
- 同一个 class 内由 local texture tokens 汇总出的 texture statistics 或 style code。
- 同一个 class 内不同 reference patches 或 crops 的 bank entries。

### 6.2 Cell-Level Routing

V5.1 中对每个 target nucleus 或 cell region：

- 读取 target cell id。
- 如果 reference 中存在该 cell class，则路由到 CellBank[class]。
- 如果缺少，则路由到 PriorCellBank[class]。

第一版 cell-level 建议使用统计级调制：

- 将 cell composition、chromatin statistics、nucleus stain statistics 作为 per-region appearance modulation 条件。
- 避免一开始使用 per-nucleus cross-attention。

等统计级 cell bank 证明有效后，再考虑 nucleus-token attention。

## 7. Appearance Injection Renderer

Appearance Injection Renderer 负责把 bank 中的外观注入到生成器。它应同时覆盖低频与高频两种外观。

### 7.1 外观注入：SEAN 式 Spatial AdaLN

V5-min 的外观注入不使用 residual cross-attention。颜色与纹理都走归一化调制通道：

```
style_c = TissueBank[class_id]          # HED stats + optional shallow texture stats
structure_t = target structure token    # tissue layout + geometry maps + xy
gamma_t, beta_t = MLP(style_c, structure_t)
h = LN(h) * (1 + gamma_t) + beta_t
```

这里的 `gamma_t/beta_t` 是 per-token 空间变化的调制图，而不是每个 class 一个常量。它等价于 SEAN/SPADE 风格的 region-wise style modulation：style code 来自 reference bank，空间变化来自 target structure token。

关键约束：

- appearance modulation 不应 zero-init 到完全无效。
- 初始增益要足够让 reference bank 从训练第一步就有影响。
- geometry residual 与 appearance modulation 作用在不同算子上：ControlNet 是唯一结构残差来源，appearance 只走 scale/shift。
- target structure token 只提供布局、几何图和坐标，不携带 reference appearance。

### 7.2 高频外观：禁用 Residual Local Attention

高频外观包括：

- 核纹理。
- 染色质颗粒感。
- 胶原纤维纹理。
- 坏死区域颗粒与碎屑。
- 免疫细胞簇状纹理。
- 局部 H&E pattern。

V5-min 不再用 class-wise local-token cross-attention 注入纹理，因为它会产生：

```
h = h + texture_cross_attention(...)
```

这和 ControlNet 的 `h = h + controlnet_residual` 落在同一个加法通道上，会把 V4 的残差竞争老坑请回来。

local texture tokens 在 V5-min 中只允许用于计算 texture style statistics 或代表性 texture code，再进入 `style_c`；不能直接作为 cross-attention 的 K/V 残差注入。

V5-min 的高频纹理由两件事共同承担：

- spatial AdaLN 根据 `style_c + target structure` 生成空间变化的 `gamma/beta`，提供纹理 pattern 的调制能力。
- masked VGG Gram texture loss 在输出端倒逼生成图的区域纹理统计对齐 reference。

## 8. Loss 设计

V5 不能只依赖 denoise/reconstruction loss，也不能把 swap sensitivity 当主监督。Loss 必须和职责分工一致。

### 8.1 Reconstruction / Denoise Loss

目标：

- 保证生成图整体合理。
- 保证 diffusion training 的基础目标稳定。
- same-WSI target image 可以作为真实 GT。

风险：

- 如果 ref-target 总是 same-WSI 且外观差异小，模型可能无视 reference bank 也能降低重建 loss。

因此 reconstruction/denoise loss 是必要基础，但不是 appearance 迁移的充分监督。

### 8.2 Geometry Consistency Loss

目标：

- 生成图经过 frozen tissue/nuclei predictor 后，应与 target tissue/cell geometry 一致。
- 保证 ControlNet 负责的 where/shape 没有被 appearance bank 破坏。

该 loss 关注：

- tissue region 一致性。
- nuclei position 一致性。
- nuclei size/shape 一致性。
- density map 一致性。

语义编码器如 UNI2-h 更适合辅助这类 semantic/geometry consistency，而不适合作为 appearance fidelity 的主裁判。

### 8.3 Region-Wise Reference Appearance Fidelity Loss

这是 V5 的核心监督。

目标：

- generated tumor region 像 reference tumor bank。
- generated stroma region 像 reference stroma bank。
- generated necrosis region 像 reference necrosis bank。
- generated immune region 像 reference immune bank。

它不要求像素配准，而是要求 region-level appearance statistics 与 features 对齐。

Appearance fidelity 的裁判必须对低层视觉敏感，不能只用深层语义 encoder。

推荐至少分两路：

- 低频颜色与染色：H&E 解卷积通道统计、LAB/RGB 直方图、均值、方差、分位数。
- 高频纹理：VGG-style 或浅层 CNN Gram 特征，或者专门为 stain/texture 敏感训练的 pathology appearance encoder。

首版最直接的实现是：

- `L_color`：在 generated target region 与 reference same-class region 内，比较 H&E 通道的 mean/std；如果 H&E 解卷积不稳定或实现成本过高，使用 LAB/RGB region mean/std/covariance 作为退路。
- `L_texture`：取 VGG `relu1_2`、`relu2_2`、`relu3_3` 等浅层到中层特征，在对应 region mask 内计算 masked Gram，再比较 generated 与 reference 的 Gram 矩阵。
- `L_appearance = lambda_color * L_color + lambda_texture * L_texture`，其中 `lambda_color` 应偏大，先确保 H/E 深浅和整体色调对齐，再让纹理分支补高频。

实现约束：

- HED/OD 变换不能用过小的 `eps` 和硬下界导致深色核区域梯度爆炸或被 clamp 截断。生成图侧应使用平滑的 `-log(x + stain_eps)`，`stain_eps` 可从 `1e-3` 起步。
- masked Gram 默认按 `mask_pixels * channels` 归一化，让不同 feature 分辨率和通道数的层更可比。
- 如果 texture loss 被整体能量或颜色差异主导，可以在 masked Gram 前启用 region 内逐通道标准化，让 Gram 更偏向通道相关结构。

VGG Gram 不能单独承担 appearance loss。它擅长纹理相关性，但可能放过整体偏紫、偏粉、H/E 比例变化等低频颜色偏移；病理 appearance 中颜色是第一优先级，所以必须显式加入 stain statistics。

不建议把 UNI2-h 深层语义 token 作为主 appearance loss，因为它可能对 stain 与 texture 做语义不变量压缩，导致 loss 下降但图像外观不动。

### 8.4 Cell-Level Appearance Fidelity Loss

V5.1 接入 CellBank 后，加入 cell-level appearance loss。

目标：

- generated neoplastic nuclei appearance 像 reference neoplastic nuclei。
- generated inflammatory nuclei appearance 像 reference inflammatory nuclei。
- generated connective nuclei appearance 像 reference connective nuclei。
- generated dead/necrotic nuclei appearance 像 reference dead/necrotic nuclei。

首版 cell-level loss 可优先使用统计监督：

- nucleus stain intensity。
- chromatin texture。
- edge sharpness。
- size/shape distribution。
- cell class density and neighborhood statistics。

在 tissue-level bank 尚未验证前，不启用这一路主监督。

### 8.5 Swap Sensitivity Loss

Swap sensitivity 只能证明：

- 换 reference 后输出应该变化。

它不能证明：

- 输出变得像正确 reference。

因此 V5 中 swap loss 只作为辅助诊断或稀疏约束，而不是主损失。

更推荐的诊断是单类 bank 替换：

- 固定 target。
- 只替换 TissueBank[tumor]。
- 检查生成图中 tumor 区 appearance 是否变化。
- 同时检查 stroma、necrosis 等其他区域是否保持不变。

这个实验比全图 swap 更能证明 class-wise bank routing 的因果链是否成立。

## 9. 训练配对策略

### 9.1 Same-WSI Pair

Same-WSI pair 仍然有价值：

- 提供真实 target image 作为 GT。
- reference 与 target 的 stain 域接近。
- 训练初期更稳定。

但 same-WSI pair 不能是唯一配对方式，因为它可能让模型靠 FLUX prior 与 ControlNet shortcut 降低重建 loss。

### 9.2 Cross-WSI / Cross-Stain Pair

训练中必须加入跨 WSI 或跨染色批次 pair：

- 让 reference 和 target 的 appearance 差异更明显。
- 让“从 reference bank 搬 appearance”成为降低 appearance loss 的必要路径。
- 避免模型只学同一 WSI 的平均风格。

推荐采样策略：

- early stage：same-WSI 为主，保证训练稳定。
- bank-learning stage：增加 cross-WSI / high-appearance-gap pair。
- robustness stage：混合 full coverage、partial coverage、low coverage 和 class-bank dropout。

### 9.3 Coverage-Aware Sampling

每个 batch 应记录并采样：

- full coverage：reference 覆盖 target 所需主要 tissue classes。
- partial coverage：reference 缺少部分 target classes。
- low coverage：reference 与 target class overlap 很低。

Coverage-aware sampling 的目标是同时训练：

- 有类覆盖时使用 ref-derived bank。
- 缺类时使用 prior bank。
- 多类混合时按 hard routing 独立控制各区域 appearance。

## 10. V5-Min 最小可行版本

V5-min 的目标不是一次性做完整 cell-level rendering，而是先证明 tissue-level bank 真的能控制生成外观。

### 10.1 V5-Min 输入

V5-min 使用：

- reference image。
- reference tissue mask。
- target tissue mask。
- target nuclei geometry maps。
- target image GT。

V5-min 暂不把完整 target cell id mask 作为强 ControlNet condition。

### 10.2 V5-Min 模块

V5-min 包含：

- RefBankBuilder：每个 tissue class 一个或多个 stain/texture style prototype，加 K 个 local texture tokens 作为统计源。prototype 默认使用 HED stain statistics，可选拼接 hard/thresholded token pooling；local tokens 不作为 cross-attention K/V 残差注入。
- GeometryControlNet：target tissue geometry 与 nuclei geometry。
- Hard Class Router：target token 按 tissue id 只读取对应 TissueBank。
- Appearance Injection：SEAN 式 spatial AdaLN，由 `style_c + target structure token` 生成 per-token `gamma/beta`；不使用 class-wise residual local attention。
- Prior Fallback：reference 缺类时使用 learned PriorTissueBank。

V5-min 的 loss 四件套是：

- `L_denoise`：基础 diffusion/重建目标。
- `L_color`：region-wise stain/color statistics。
- `L_texture`：masked shallow VGG Gram。
- `L_geometry`：将生成图经过 frozen dense tissue/nuclei predictor 后，用可微 dense logits/maps 对齐 target geometry；tissue/nuclei semantic logits 可用 CE + Dice，binary nucleus map 可用 BCE + Dice，distance/centroid/boundary 等 dense maps 可用 L1。该 loss 必须避开 argmax、watershed、instance postprocess 等不可微步骤。

不包含：

- per-nucleus cross-attention。
- 完整 CellBank token path。
- 复杂 global style token path。
- 让 class id 作为 appearance shortcut 进入 ControlNet。

### 10.3 V5-Min 验证目标

V5-min 首轮只回答三个问题：

- target tumor 区是否像 reference tumor。
- target stroma 区是否像 reference stroma。
- target necrosis/immune 等区域是否能按 reference 对应 class 独立变化。

如果 tissue-level bank 不能证明有效，不应继续堆 cell-level bank。

## 11. 标准诊断与可解释性验证

V5 的训练诊断不能只看 loss 曲线，需要固定诊断面板。

### 11.1 Single-Class Bank Replacement

固定 target masks 与其他 bank，只替换一个 class bank：

- 替换 tumor bank，只允许 tumor 区 appearance 变化。
- 替换 stroma bank，只允许 stroma 区 appearance 变化。
- 替换 necrosis bank，只允许 necrosis 区 appearance 变化。

判断标准：

- 被替换类别区域发生明显 stain/texture 改变。
- 未替换类别区域尽量保持稳定。
- geometry 不应明显漂移。

### 11.2 Bank Ablation

对每类 bank 做 ablation：

- normal bank。
- zero bank。
- prior bank。
- swapped same-class bank。
- wrong-class bank。

判断标准：

- zero bank 应削弱 reference-specific appearance。
- prior bank 应生成合理但不 reference-specific 的平均外观。
- same-class swap 应改变对应区域 appearance。
- wrong-class bank 不应被 hard router 读取。

### 11.3 Region Appearance Metrics

每个诊断样本记录：

- generated-reference region stain distance。
- generated-reference texture distance。
- generated-target geometry consistency。
- per-class bank usage。
- prior fallback usage。
- class-wise replacement sensitivity。

这些指标要按 tissue class 分开记录，不能只给全图均值。

## 12. 主要风险与设计约束

### 12.1 Appearance Encoder 选错

最大风险是用深层语义 encoder 做 appearance fidelity 裁判。

如果 encoder 对 stain/texture 不敏感，模型会出现：

- loss 下降。
- region feature 接近。
- 但可视化图像 appearance 不迁移。

因此 appearance loss 必须使用低层颜色统计与浅层纹理特征；语义 encoder 只作为辅助几何/语义一致性约束。

### 12.2 ControlNet 与 Appearance Modulation 尺度竞争

如果 appearance injection 只是额外残差，并且初始化过弱，它会再次被 ControlNet 残差压制。

设计约束：

- 第一层治本：class id 从 ControlNet 移出，ControlNet 只吃 binary mask、boundary、distance transform、centroid heatmap、density map 等几何条件。
- 第二层关键：appearance 不走 `h = h + ref_residual`，而是走归一化调制入口，例如 `h_norm * (1 + gamma_c) + beta_c`。ControlNet 改内容残差，appearance 改 norm 后的 scale/shift，两者不在同一个加法上抢量级。
- 第三层细节：产生 `gamma_c/beta_c` 的 MLP 不做 zero-init，可使用正常初始化或轻微正向初始增益，避免优化早期 appearance path 被判成无效路径。
- 早期诊断必须检查单类 bank 替换是否立即造成 region-level 可见变化。

### 12.3 Class Routing 软化导致跨类泄漏

如果 router 在类间使用 soft routing，模型可能重新学会偷读其他 class bank。

设计约束：

- 类间 hard routing。
- 类内 soft selection。
- wrong-class bank ablation 必须成为 smoke test。

### 12.4 Prior Bank 训练不足

如果 prior bank 只在真实缺类时使用，常见 class 的 prior 可能训练不足。

设计约束：

- 使用 class-bank dropout。
- 对所有 class 定期强制走 prior。
- 单独记录 prior fallback 区域的质量。

### 12.5 CellBank 过早接入

过早接 per-nucleus attention 会带来显存、噪声和训练不稳定。

设计约束：

- V5-min 只做 tissue bank。
- V5.1 再做 cell-level。
- CellBank 首版优先统计级调制，不直接做大规模 per-nucleus attention。

## 13. 推荐落地顺序

第一阶段：V5-min tissue bank。

- 移除 ControlNet 中的强 appearance shortcut。
- 构建 tissue-level Reference Appearance Bank。
- 使用 hard class router。
- 将 prototype modulation 接到归一化调制入口。
- 加入 region-wise stain statistics 与 shallow texture loss。
- 做 single-class bank replacement smoke test。

第二阶段：coverage 与 prior fallback。

- 加入 coverage-aware sampling。
- 加入 class-bank dropout。
- 验证 reference 缺类时 prior bank 质量。
- 验证有类覆盖时 prior 不抢 ref-derived bank 的职责。

第三阶段：更强的 tissue texture。

- 增加类内 sub-prototypes。
- 增强 local texture statistics、sub-prototype 和 spatial AdaLN style decoder。
- 调整高频 texture fidelity loss。
- 做跨 WSI / 高 appearance gap 诊断。

第四阶段：CellBank V5.1。

- 先加入 cell composition 与 nucleus stain/chromatin statistics 作为 per-region modulation。
- 验证 neoplastic、inflammatory、connective 等 cell class 的统计级 appearance 迁移。
- 最后再考虑 nucleus-token attention。

## 14. 最终架构定义

Cross V5 的模型定义是：

**Geometry-Controlled, Bank-Conditioned Pathology Rendering。**

它把旧的：

- ControlNet。
- reference attention。
- attention bias。
- swap sensitivity。

升级为：

- target geometry-only control。
- mask-id indexed reference appearance bank。
- hard class-wise router。
- SEAN-style spatial AdaLN appearance modulation。
- region-wise reference appearance fidelity。
- learned prior fallback。
- single-class bank replacement diagnostics。

最终目标是让模型满足这条因果链：

**一个 mask id 对应一个 appearance source；target mask 决定放在哪里；reference bank 决定长什么样；缺类时 prior bank 兜底。**

## 15. V5-Min 训练胶水层

V5-min 的训练胶水层应保持在现有 V3/V4 训练循环之外，先作为可插拔接口验证。当前骨架包含：

- `controlnet_train/modules/cross_v5_conditioning.py`：HED stain-stat Reference tissue bank、shallow texture local tokens/statistics、target structure tokens、SEAN-style spatial AdaLN modulation。
- `controlnet_train/data/cross_v5_pairing.py`：metadata-level V5 pairing sampler，按 same/cross-WSI、appearance-gap、coverage bucket 和 bank dropout 抽 reference pair。
- `controlnet_train/training/cross_v5_losses.py`：region-wise color/texture appearance loss 与 dense geometry consistency loss。
- `controlnet_train/training/cross_v5_glue.py`：FLUX train loop 的组装胶水，包括 loss 权重、pairing policy、latent-to-RGB decode bridge、frozen predictor bridge、AdaLN hook spec。
- `controlnet_train/training/cross_v5_flux_adapters.py`：对接真实 diffusers FLUX double/single transformer blocks 的 AdaLN adapter 骨架。
- `scripts/smoke_cross_v5_predictor_bridge.py`：验证 generated RGB 到 frozen predictor 的梯度没有被 `no_grad` 切断。
- `scripts/smoke_cross_v5_adaln_adapter.py`：验证 hook installer、V5-ready block adapter、bank swap 对 hidden 输出的因果影响，并拒绝 zero-init gamma path。
- `scripts/smoke_cross_v5_flux_adapter.py`：验证 FLUX double/single mock blocks 的 patched forward、V5 kwargs 剥离、新旧 single-block 签名兼容，以及缺 bank 时 strict raise。
- `scripts/smoke_cross_v5_decode_and_pairing.py`：验证 latent decode bridge 可微、以及 pairing sampler 按 high-gap/full-coverage 抽样并执行 bank dropout。
- `scripts/smoke_cross_v5_visual_bank.py`：验证默认 bank prototype 来自 HED stain statistics，local tokens 保持浅层 texture-token 维度。
- `scripts/smoke_cross_v5_spatial_adaln.py`：验证同一 class 在不同 target structure token 下产生不同 `gamma/beta`。
- `scripts/smoke_cross_v5_loss_assembly.py`：验证 `denoise + appearance + geometry` 四族 loss 可以联合反传。

推荐默认权重：

- `denoise = 1.0`
- `appearance = 0.75`
- `geometry = 0.25`
- `swap_sensitivity = 0.0`

其中 `appearance` 是外层 family-level 总权重；颜色和纹理的相对比例只由 `CrossV5AppearanceLossConfig.color_weight / texture_weight` 控制，避免同一权重在 glue 层和 loss 内部被重复缩放。

推荐 pairing policy：

- pair mode：`same_wsi / cross_wsi / high_appearance_gap = 0.35 / 0.45 / 0.20`
- coverage：`full / partial / low = 0.55 / 0.35 / 0.10`
- class-bank dropout：`0.15`
- bank presence：至少 `1.0` 个有效 token，token confidence 从 `0.5` 起步。

接入 FLUX train loop 时的关键顺序：

1. 走现有 diffusion forward，得到 `noise_pred` 和 `denoise_loss`。
2. 用 `decode_cross_v5_prediction_rgb(...)` 得到 `prediction_rgb`：先用 `reconstruct_cross_v5_x0_latents(...)` 从 `x_t` 和 `noise_pred` 重建 `x0_hat`，必要时 unpack FLUX packed latents，再按 VAE `scaling_factor/shift_factor` decode。VAE 参数可以 frozen，但 decode 不能放进 `torch.no_grad()`，也不能 detach `x0_hat`。
3. 用 `assemble_cross_v5_step_losses(...)` 做 step-level 门控与 loss 组装。
4. 只有当 `CrossV5LossIntervals.geometry` 与 `geometry_timestep_max` 同时放行时，才将 `prediction_rgb` 输入 frozen tissue/CellViT predictor。predictor 参数 `requires_grad_(False)`，但 forward 不能 no-grad。
5. 对组合后的 `total` 做一次 backward。

配对 sampler 要求：

- `CrossV5PairingSampler` 输入 candidate pair metadata，输出带 `v5_pair_mode`、`v5_coverage_mode`、`v5_reference_bank_keep_tissue_ids` 和 `v5_reference_bank_drop_tissue_ids` 的 record。
- pair mode 按 `same_wsi / cross_wsi / high_appearance_gap` 权重抽；coverage 再按 `full / partial / low` 权重抽。
- high-gap 使用 `appearance_gap` / `appearance_gap_score` / `stain_distance` 等字段，默认取 0.75 quantile 以上作为 high-gap pool。
- bank dropout 在 reference 可覆盖的 tissue ids 上执行，并默认至少保留一个可用 bank class，避免 appearance loss 变成无 reference 可用的假样本。

AdaLN 接入原则：

- 先用显式 V5-ready block wrapper 暴露 `set_cross_v5_adaln_modulator(...)`，再用 `install_cross_v5_adaln_hooks(...)` 安装。
- 不在 glue 层猜 diffusers 私有字段，避免版本差异导致 silent no-op。
- hook 位置应在 norm 后、residual 加法前后的明确 AdaLN-style 入口，不能退化成普通 residual reference injection。
- V5-min 推荐使用 `CrossV5SpatialAdaLNModulator`：`cross_v5_target_structure_tokens` 与 reference bank prototype 共同生成 per-token `gamma/beta`。
- `require_nonzero_gamma=True` 时安装器应拒绝 gamma path 全零初始化，防止 appearance 从第一步就被训练忽略。
- 单测/烟测必须覆盖“hook 被调用后 hidden 改变，换 bank 后输出改变”的因果链；只测 loss assembly 不足以证明 appearance bank 真的进入了 FLUX forward。

真实 diffusers FLUX block 的 V5-min 接入点：

- double-stream `FluxTransformerBlock`：只调制 image stream；插在 `norm2(hidden_states)` 和 FLUX 自带 `scale_mlp/shift_mlp` 之后、`ff(...)` 之前，即 `double_image_post_norm2_before_ff`。
- single-stream `FluxSingleTransformerBlock`：V5-min 默认不安装；adapter 骨架兼容旧版 pre-concat 签名和新版 separated-stream 签名。若 V5.1 打开 single block，只在 `norm(hidden_states, emb=temb)` 后调制 image-token 的 MLP branch，attention branch 继续使用未调制的 norm hidden，避免 appearance 直接污染结构注意力，即 `single_image_post_norm_mlp_only`。
- V5 bank 通过 `joint_attention_kwargs` 传入：`cross_v5_target_class_ids`、`cross_v5_target_structure_tokens`、`cross_v5_bank`、可选 `cross_v5_fallback_prototypes` 和 `cross_v5_image_token_start`。adapter 会在进入 attention processor 前剥离这些 key，避免污染原 attention kwargs。
- `require_conditioning=True` 时，装了 adapter 却没有传 `cross_v5_bank` 或 `cross_v5_target_class_ids` 会直接 raise，防止训练静默退化成无 reference injection。

昂贵 decoded-image branch 可用 step/timestep 双门控：

- `interval <= 0` 表示关闭该 branch。
- `global_step % interval == 0` 才运行。
- 可选 `timestep_min/timestep_max` 限制只在特定噪声区间运行 geometry 或 appearance branch。
- 默认 `CrossV5LossIntervals.geometry = 4` 且 `geometry_timestep_max = 350.0`，也就是 geometry predictor 只在低噪步稀疏运行；若 scheduler 的 timestep 方向不同，应显式改这个 cutoff。
