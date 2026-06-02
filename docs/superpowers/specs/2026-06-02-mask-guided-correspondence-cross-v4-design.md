# Mask-Guided Correspondence Cross V4 模型架构设想

Date: 2026-06-02

## 0. 一句话结论

你现在真正要做的不是“让模型凭空学会某种病理质感”，而是一个更可解、更工程化的任务：

**给定 reference image + reference tissue/cell masks，以及 target tissue/cell masks，让模型按语义把 reference 中真实存在的组织/细胞质感搬到 target 对应区域；当 reference 缺少 target 所需类别时，再由 per-class 病理先验 token 兜底生成。**

因此，完整架构的核心不是全局 style transfer，也不是把 reference 压成一个无空间向量，而是：

1. 保留 reference 的空间 token。
2. 用 reference mask 和 target mask 在 attention logits 上做语义对应约束。
3. 为 reference 缺类区域提供 per-class prior token。
4. 把 cell/nuclei mask 纳入结构控制和 attention 对齐，而不是只当附属通道。

这份文档把这个方向暂命名为 **Cross V4: Mask-Guided Correspondence + Prior Fallback**。

当前仓库里的 Cross V3 已经搭了大约 70%：

- `controlnet_train/modules/cross_v3_conditioning.py` 已经把 `z_ref + ref_tissue_feat + ref_nuclei_feat` 编成 reference context tokens。
- `controlnet_train/training/flux_phase5_cross_v3.py` 已经把 target tissue/nuclei 作为 ControlNet 结构输入，把 reference tokens 拼进 FLUX joint attention。
- `controlnet_train/data/cross.py` 已经有 same-case/cross-reconstruction pairing，并记录 `full/partial/low` coverage、`missing_target_tissue_ids`。
- `controlnet_train/training/cross_v1_losses.py` 已经有 regional stain/style loss 和 ref-swap sensitivity loss。

现在最关键缺口是：**reference tokens 虽然进了 attention，但 target 的每个生成位置并不知道应该 attend reference 的哪个语义类别。** 所以模型很容易退化成只用 reference 的全局染色/色调，不真正搬运组织级、细胞级纹理。

---

## 1. 任务重新定义

### 1.1 旧误区：把目标误解成纯质感迁移

如果任务被定义成：

- 给 target 结构 A；
- 给 reference 质感 B；
- 让模型凭空合成 “A 结构 + B 质感”；
- 且没有真实 GT；

那确实很难，因为模型没有监督信号知道“正确组合”应该长什么样。

但你的真实目标不是这个。

### 1.2 新定义：语义对齐的样例引导生成

你的真实输入是：

- `reference_image`: 一张真实病理 patch。
- `reference_tissue_mask`: reference 中每个像素是什么组织。
- `reference_cell_mask` / `reference_nuclei_mask`: reference 中每个像素是什么细胞/细胞核类别，或者至少是细胞核空间分布。
- `target_tissue_mask`: 你希望生成图中每个位置是什么组织。
- `target_cell_mask` / `target_nuclei_mask`: 你希望生成图中每个位置有什么细胞/细胞核结构。

你的真实输出是：

- `target_image`: 符合 target masks 的病理图像。
- 对于 target 中 reference 也有的类别，图像外观应该尽量来自 reference 同类区域。
- 对于 target 中 reference 没有的类别，图像外观应该由模型学到的病理先验生成，同时尽量保持全局染色风格一致。

这更接近：

- exemplar-guided synthesis
- semantic correspondence
- masked cross-attention
- conditional pathology image generation

### 1.3 为什么 same WSI / same case 有天然 GT

训练时让 reference 和 target 来自同一个 WSI 或同一个 case 的不同 patch。

此时：

- target 图像本身就是真实 GT；
- reference 和 target 的染色、扫描、组织域基本一致；
- reference 和 target 的空间布局不同；
- target 的肿瘤区不一定和 reference 的肿瘤区在同一位置；
- target 的细胞密度/细胞组成也不一定和 reference 完全重合。

所以模型必须学会：

1. 读 target mask，知道当前 target 位置需要什么组织/细胞。
2. 读 reference mask，找到 reference 中同类组织/细胞区域。
3. 从 reference 对应区域提取真实质感。
4. 重排到 target mask 指定的位置。
5. 若 reference 没有对应类别，则切换到 prior。

GT 就是 target 原图。你不需要跨 WSI 配对的人工 GT。

### 1.4 训练和推理的关系

训练阶段：

- 主要使用同 WSI / 同 case reference-target 对。
- 有监督目标：重建真实 target image。
- 通过采样策略构造 full coverage、partial coverage、low coverage。
- 学会“有 reference 同类时听 reference，没有时听 prior”。

推理阶段：

- reference 可以来自当前图、用户选择的图、同病例图、检索库、甚至跨病例图。
- target masks 来自你的 mask editing pipeline；这里的 `target masks` 明确包括 `target_tissue_mask` 和 pipeline 自动生成的 `target_cell_mask` / `target_nuclei_mask`。
- 模型用训练时学到的语义对齐机制，把 reference 的同类外观迁到 target。
- 没有同类 reference 的区域由 prior 兜底。

---

## 2. 当前已有系统如何对应这个目标

### 2.1 当前 Cross V3 的已实现结构

当前 Cross V3 的高层数据流是：

```text
target_tissue_mask
target_nuclei_mask
        |
        v
target tissue/nuclei encoders
        |
        v
ControlNet condition
        |
        v
FLUX denoising image tokens


reference_image
reference_tissue_mask
reference_nuclei_mask
        |
        v
VAE z_ref + ref tissue/nuclei features
        |
        v
CrossV3ReferenceContextEncoder
        |
        v
reference context tokens
        |
        v
append to FLUX joint attention context
```

这已经符合“结构和外观分离”的大方向：

- ControlNet 只看 target 结构。
- reference 不进 ControlNet，避免 reference 布局污染 target 结构。
- reference 通过 joint attention 进入 FLUX transformer，理论上负责外观/纹理。

### 2.2 当前 Cross V3 的关键优点

**优点 1：reference token 保留空间。**

`pack_cross_v3_reference_grid()` 将 `z_ref + ref_tissue_feat + ref_nuclei_feat` 以 2x2 latent patch 的形式打包成 token 序列。这样 reference token 不是一个全局向量，而是保留了局部区域信息。

对你的任务来说，这是正确的。不要把 reference 完全塌缩成无位置 style vector。

**优点 2：reference token 含有 tissue/cell 条件。**

reference token 中包含：

- `z_ref`: reference 图像 latent，含真实像素质感。
- `ref_tissue_feat`: reference 组织类别特征。
- `ref_nuclei_feat`: reference 细胞/细胞核特征。

这给语义对齐提供了原料。

**优点 3：target 结构由 target mask 决定。**

ControlNet 只拿 `tar_tissue_feat + tar_nuclei_feat`。这让生成结构跟着 target mask，而不是被 reference 布局带跑。

**优点 4：数据层已经记录 coverage。**

`build_cross_metadata()` 已经会计算：

- `pair_difficulty`: `full` / `partial` / `low`
- `tissue_coverage_ratio`
- `area_coverage_ratio`
- `missing_target_tissue_ids`
- `covered_target_tissue_ids`

这正好服务于 “reference 有同类 / reference 缺同类” 两种训练场景。

### 2.3 当前 Cross V3 缺的核心机制

当前的问题不是 reference token 没进模型，而是：

**target token attend reference token 时，没有显式语义约束。**

也就是说，target 肿瘤位置的 query 在 attention 里看到的是一堆 reference tokens：

- reference 肿瘤 token
- reference 间质 token
- reference 坏死 token
- reference 免疫 token
- reference 背景 token
- 可能还有 route anchor token

但 attention logits 里没有一条规则说：

```text
target tumor token 应该优先 attend reference tumor token
target stroma token 应该优先 attend reference stroma token
target immune token 应该优先 attend reference immune token
target neoplastic-cell-rich 区域应该优先 attend reference neoplastic-cell-rich 区域
```

所以模型很容易：

- 只利用 reference 的全局色调；
- 忽略具体 tissue/cell 对应关系；
- 继续依赖 FLUX/ControlNet 自身先验；
- 产生“结构对了，但 reference 质感没真正迁移”的结果。

---

## 3. Cross V4 总体架构

### 3.1 总体结构图

```text
                                   ┌─────────────────────────┐
                                   │ reference_image          │
                                   │ reference_tissue_mask    │
                                   │ reference_cell_mask      │
                                   └───────────┬─────────────┘
                                               │
                                               v
                             ┌──────────────────────────────────┐
                             │ Reference Context Encoder         │
                             │                                  │
                             │ z_ref                            │
                             │ ref tissue features              │
                             │ ref cell/nuclei features         │
                             │ ref token semantic IDs/histograms │
                             └───────────┬──────────────────────┘
                                         │
                                         v
           ┌──────────────────────────────────────────────────────────┐
           │ Context Tokens                                            │
           │                                                          │
           │ text tokens                                               │
           │ local reference tokens                                    │
           │ tissue/cell route anchor tokens                           │
           │ per-class prior tokens                                    │
           │ optional global stain/style tokens                        │
           └──────────────────────────────┬───────────────────────────┘
                                          │
                                          v
┌──────────────────────────┐     ┌────────────────────────────────────┐
│ target_tissue_mask        │     │ Mask-Guided Correspondence Bias    │
│ target_cell_mask          │────>│                                    │
└─────────────┬────────────┘     │ target token semantic IDs          │
              │                  │ reference token semantic IDs        │
              v                  │ tissue match bias                   │
┌──────────────────────────┐     │ cell similarity bias                │
│ Target Structure Encoder  │     │ prior fallback bias                 │
│                          │     └────────────────┬───────────────────┘
│ tissue features           │                      │
│ cell/nuclei features      │                      v
└─────────────┬────────────┘     ┌────────────────────────────────────┐
              │                  │ FLUX Transformer                    │
              v                  │                                    │
┌──────────────────────────┐     │ image tokens query context tokens   │
│ ControlNet condition      │────>│ attention logits += semantic bias   │
└──────────────────────────┘     │ denoise target image tokens         │
                                 └────────────────┬───────────────────┘
                                                  │
                                                  v
                                      ┌─────────────────────────┐
                                      │ generated target image   │
                                      └─────────────────────────┘
```

### 3.2 三条信息通路

Cross V4 应该明确分成三条通路。

#### 通路 A：target structure path

输入：

- `target_tissue_mask`
- `target_cell_mask` / `target_nuclei_mask`

作用：

- 决定哪里是肿瘤、间质、坏死、免疫浸润等。
- 决定哪里有 neoplastic / inflammatory / connective / dead / epithelial cells。
- 控制 target 图像的结构、组织区域边界、细胞密度、细胞类型分布。

实现位置：

- 当前已经在 `build_cross_v3_control_condition()` 中部分实现。
- tissue 通过 HTE / one-hot encoder。
- cell/nuclei 通过 `NucleiConditionEncoder`。

这条通路应该继续进入 ControlNet。

#### 通路 B：reference appearance path

输入：

- `reference_image`
- `reference_tissue_mask`
- `reference_cell_mask` / `reference_nuclei_mask`

作用：

- 提供真实的局部病理质感。
- 提供同类组织的颜色、腺体形态、坏死颗粒感、间质纤维感、免疫细胞密度等。
- 提供同类细胞类型的核形态、密度、染色强度和空间分布。

实现位置：

- 当前 `CrossV3ReferenceContextEncoder` 已经实现局部 reference token。
- Cross V4 需要让 encoder 不只返回 token，还返回 token 的语义元信息：
  - tissue ID
  - coarse tissue ID
  - tissue confidence
  - cell class histogram
  - nuclei density
  - optional boundary/contact features

这条通路进入 FLUX joint attention context。

#### 通路 C：semantic correspondence path

输入：

- target token semantic metadata
- reference token semantic metadata
- prior token metadata

作用：

- 修改 attention logits。
- 让 target 每个位置优先读 reference 同类 token。
- 当 reference 同类不存在时，让 target 位置读 per-class prior token。
- 用 cell/nuclei mask 对 tissue 对齐做进一步细化。

这是当前最缺的通路。

它不应该只是把 class embedding 加到 token 上。它应该显式参与 attention logits：

```text
attention_logits[target_i, context_j] += correspondence_bias[target_i, context_j]
```

---

## 4. 输入/输出契约

### 4.1 必需输入

Cross V4 的最小输入应该是：

```text
reference_image:        Float tensor, shape (3, H, W), normalized [0, 1]
reference_tissue_mask:  Long tensor, shape (H, W), unified fine tissue ID
reference_cell_mask:    Long tensor, shape (H, W), cell/nuclei ID
target_tissue_mask:     Long tensor, shape (H, W), unified fine tissue ID
target_cell_mask:       Long tensor, shape (H, W), cell/nuclei ID
```

这里要把“模型输入”和“用户手工输入”分清楚：

- Cross V4 模型运行时确实消费 `target_cell_mask` / `target_nuclei_mask`。
- 但它不是额外要求用户手工提供的新条件，而是由你的上游 editing pipeline 自动生成。
- 换句话说，推理闭环是：用户/编辑器得到 `target_tissue_mask` 后，pipeline 同步生成对应的 `target_cell_mask` / `target_nuclei_mask`，Cross V4 再同时使用 target tissue 和 target cell 条件生成图像。

当前代码中 `reference_cell_mask` / `target_cell_mask` 对应的是：

```text
reference_nuclei_mask
target_nuclei_mask
```

但概念上建议文档和新接口里统一叫：

```text
cell_mask / nuclei_mask
```

原因：

- 你关心的是细胞级微环境；
- 数据上可能是 CellViT nuclei type mask；
- 后续可能从 nuclei ID map 扩展为 instance mask、density map、cell-type probability map。

### 4.2 组织标签空间

当前 unified tissue label 已经是 16 fine classes：

```text
0  Background
1  Tumor
2  Stroma
3  Necrosis
4  Immune infiltrate
5  Normal epithelium
6  Blood vessel
7  Other tissue
8  Gleason 3
9  Gleason 4
10 Gleason 5
11 Adenomatous gland
12 Moderately differentiated
13 Poorly differentiated
14 DCIS
15 Angioinvasion
```

同时有 8 coarse classes：

```text
0 Background
1 Tumor
2 Stroma
3 Necrosis
4 Immune infiltrate
5 Normal epithelium
6 Blood vessel
7 Other tissue
```

Cross V4 建议采用双层语义：

- attention bias 的硬主干先用 coarse。
- fine label 作为附加加分或 subtype refinement。
- per-class prior token 起步用 coarse，更稳。
- fine prior token 后续作为可选增强，不建议第一版就全量启用。

原因：

- fine tumor subtype 分布不均衡；
- 不同数据集 fine subtype 含义不完全等价；
- coarse prior 每类样本更多，先验更容易学稳；
- fine bias 可帮助 Gleason / gland differentiation / DCIS 等任务，但不应一开始成为唯一约束。

### 4.3 细胞标签空间

当前 `dataset_config/unified_labels.py` 中 cell classes 是：

```text
101 Neoplastic
102 Inflammatory
103 Connective
104 Dead
105 Epithelial
```

`NucleiConditionEncoder` 内部会 remap 到：

```text
0 background
1 neoplastic
2 inflammatory
3 connective
4 dead
5 epithelial
```

Cross V4 应该显式使用这些 cell labels：

- ControlNet 继续使用 cell mask 控制 target 细胞结构。
- reference tokens 中保存 cell histogram / density。
- attention bias 中增加 cell similarity。
- prior tokens 中增加 cell-aware prior。

### 4.4 输出

模型主输出：

```text
generated_target_image: Float tensor or PIL image, shape (3, H, W)
```

建议调试输出：

```text
attention_mass_by_source:
  - ref_same_tissue
  - ref_same_coarse
  - ref_same_cell
  - ref_mismatch
  - tissue_prior
  - cell_prior
  - text/global

coverage_report:
  - target_tissue_ids
  - reference_tissue_ids
  - covered_target_tissue_ids
  - missing_target_tissue_ids
  - target_cell_ids
  - reference_cell_ids
  - covered_target_cell_ids
  - missing_target_cell_ids

debug_maps:
  - per-target-token selected source class
  - per-region ref-attention heatmap
  - per-region prior-attention heatmap
```

这些不是训练必须，但对诊断“模型到底有没有听 reference”非常关键。

---

## 5. Token 设计

### 5.1 Target image tokens

FLUX 中 denoising 的 image tokens 来自 noisy target latent。

如果输入 patch 是 `H x W`，VAE latent 通常是：

```text
H_lat = H / 8
W_lat = W / 8
```

FLUX 又做 2x2 packing：

```text
H_tok = H_lat / 2
W_tok = W_lat / 2
N_img = H_tok * W_tok
```

例如：

- 256x256 patch -> latent 32x32 -> packed 16x16 -> 256 image tokens。
- 512x512 patch -> latent 64x64 -> packed 32x32 -> 1024 image tokens。

Cross V4 需要为每个 target image token 建立语义 metadata：

```text
target_token_tissue_fine_id:    (B, N_img)
target_token_tissue_coarse_id:  (B, N_img)
target_token_tissue_confidence: (B, N_img)
target_token_cell_hist:         (B, N_img, NUM_CELL_WITH_BG)
target_token_cell_density:      (B, N_img)
```

这些 metadata 由 target masks 下采样得到。

### 5.2 Reference local tokens

当前 Cross V3 的 reference local token 来自：

```text
[z_ref, ref_tissue_feat, ref_nuclei_feat]
```

拼接后 2x2 packing，再投影到 FLUX context dimension。

Cross V4 保留这条路，但需要让 reference encoder 返回：

```text
reference_tokens:               (B, N_ref, D)
reference_token_tissue_fine_id:  (B, N_ref)
reference_token_tissue_coarse_id:(B, N_ref)
reference_token_tissue_conf:     (B, N_ref)
reference_token_cell_hist:       (B, N_ref, NUM_CELL_WITH_BG)
reference_token_cell_density:    (B, N_ref)
```

这里的 `N_ref` 与 target image tokens 类似，取决于 reference patch 尺寸。

注意：reference token 必须保留空间局部性。不能把它们全部平均池化成一个 style vector。

### 5.3 Route anchor tokens

当前 Cross V3 已经有 route anchor：

- `route_anchor_mode=none/coarse/fine`
- local route embedding
- anchor route embedding
- missing anchor token

但当前 route anchor 更像“类别摘要 token”，还没有真正改变 image token 对 reference token 的连接方式。

Cross V4 可以保留 route anchor，但它的定位应该是：

- 提供每个组织类别在 reference 中的 pooled appearance summary。
- 作为 local reference tokens 的补充。
- 不作为 mask-guided attention 的替代品。

也就是说：

```text
local reference tokens = 细节来源
route anchor tokens    = 类别级摘要来源
attention bias         = 负责让 target query 找对来源
```

### 5.4 Per-class prior tokens

这是 Cross V4 必须新增的部分。

为什么需要 prior tokens？

因为 target 中某个类别可能 reference 里没有。

例子：

```text
target_tissue_mask 有 tumor
reference_tissue_mask 没有 tumor，只有 stroma
```

此时 target tumor token 不应该去 attend reference stroma token 来生成 tumor。它应该改去 attend tumor prior token。

建议新增：

```text
tissue_prior_tokens: (NUM_COARSE, R_tissue, D)
cell_prior_tokens:   (NUM_CELL_WITH_BG, R_cell, D)
global_style_tokens: (R_global, D), optional
```

其中：

- `R_tissue`: 每个组织类别几个 prior tokens，建议 4 或 8 起步。
- `R_cell`: 每个细胞类别几个 prior tokens，建议 2 或 4 起步。
- `R_global`: 全局染色/扫描风格 token，建议 2 或 4。

MVP 建议：

```text
tissue_prior level: coarse 8 classes
R_tissue: 4
cell_prior classes: off
R_cell: 0
global_style_tokens: 0
```

主干验证后再逐项增加：

```text
cell_prior classes: 6 including background
R_cell: 2
global_style_tokens: 2
```

注意：prior token 自身数量很小，但 attention bias 的主要显存成本来自 `N_img x N_ref/context` logits materialization，不来自这几十个 prior tokens。

后续完整配置的额外 prior/style token 数约为：

```text
8 * 4 + 6 * 2 + 2 = 46 tokens
```

显存压力远小于 reference local tokens。

### 5.5 Context token 顺序建议

建议 Cross V4 的 context sequence 结构为：

```text
[text_tokens,
 global_style_tokens,
 tissue_prior_tokens,
 cell_prior_tokens,
 route_anchor_tokens,
 reference_local_tokens]
```

同时保存 segment offsets：

```python
context_segments = {
    "text": (0, n_text),
    "global_style": (...),
    "tissue_prior": (...),
    "cell_prior": (...),
    "route_anchor": (...),
    "reference_local": (...),
}
```

当前 `append_cross_v3_reference_context()` 只是把 reference tokens 拼到 text 后面，没有返回结构化 segment 信息。Cross V4 应该改成返回：

```python
CrossV4Context:
    encoder_hidden_states
    txt_ids
    segments
    reference_metadata
    prior_metadata
```

这样 attention bias builder 才知道每个 context token 是什么来源、什么语义。

---

## 6. Mask-Guided Correspondence Attention

### 6.1 核心公式

FLUX transformer 中某层 cross/joint attention 的 logits 可以抽象为：

```text
logits[i, j] = Q_img[i] dot K_context[j] / sqrt(d)
```

其中：

- `i`: target image token index
- `j`: context token index

Cross V4 要改成：

```text
logits[i, j] = Q_img[i] dot K_context[j] / sqrt(d) + bias[i, j]
```

`bias[i, j]` 由 target mask 和 reference mask 决定。

### 6.2 Tissue correspondence bias

最基础的 tissue bias：

```text
if context_j is reference local token:
    if target_fine[i] == ref_fine[j]:
        bias += lambda_same_fine
    elif target_coarse[i] == ref_coarse[j]:
        bias += lambda_same_coarse
    else:
        bias += lambda_mismatch
```

但 fine/coarse 应该注意顺序：

```text
same fine   -> strongest positive
same coarse -> medium positive
mismatch   -> negative
```

建议：

```text
lambda_same_fine   = +3.0
lambda_same_coarse = +2.0
lambda_mismatch    = -2.0
```

也就是 mismatch 时直接加一个带符号的负值 `-2.0`，不是一开始就 `-inf`。

这里必须用互斥逻辑：same fine 本身必然 also same coarse，所以 same fine 不应该再额外叠加 coarse bonus。否则 `+3.0` 会实际变成 `+5.0`，后续调参会完全对不上预期。

原因：

- 病理区域边界有混合。
- mask 可能有噪声。
- reference token 2x2 packing 后可能跨类别。
- 过硬屏蔽会导致边界不自然。

后续可尝试 hard mask：

```text
mismatch -> -inf
```

但建议只用于干净 mask 或后期 ablation。

### 6.3 Reference presence gate

对每个 target token 的类别 `c`，先判断 reference 中是否存在同类 token：

```text
present_ref_same_class[i] = exists j such that ref_coarse[j] == target_coarse[i]
```

如果存在：

- reference same-class tokens 应该加分；
- prior tokens 可以存在，但不应压过 reference；
- target 应主要听 reference。

如果不存在：

- reference mismatch tokens 不应被强行使用；
- corresponding prior tokens 应加分；
- target 应主要听 prior。

### 6.4 Prior fallback bias

对 tissue prior token：

```text
if context_j is tissue_prior token for class c:
    if target_coarse[i] == c:
        if reference has same class:
            bias += lambda_prior_when_ref_present
        else:
            bias += lambda_prior_when_ref_missing
    else:
        bias += lambda_prior_wrong_class
```

建议初始值：

```text
lambda_prior_when_ref_present = +0.5
lambda_prior_when_ref_missing = +3.0
lambda_prior_wrong_class      = -2.0
```

含义：

- reference 有同类时，prior 只是弱备份。
- reference 没同类时，prior 强力兜底。
- wrong-class prior 被压低。

这就是“自动分流”的关键：

```text
target tumor token
  ├─ reference 有 tumor -> same-class ref logits 高 -> attend ref tumor
  └─ reference 无 tumor -> tumor prior logits 高 -> attend tumor prior
```

不需要手写 if-else 选择生成路径。if-else 只出现在 bias 计算里，最终由 softmax 自己决定 attention mass。

### 6.5 Cell/nuclei correspondence bias

你特别提醒“还有细胞 mask”，这点非常重要。

tissue mask 只能告诉模型：

```text
这里是 tumor / stroma / necrosis / immune
```

但病理质感很大一部分来自细胞层面：

- 肿瘤细胞核大小、深染程度、异型性。
- 免疫细胞密度。
- 间质细胞/成纤维细胞分布。
- 坏死区域 dead cell / debris 形态。
- 上皮细胞排列。

因此 Cross V4 的 attention bias 不应只看 tissue ID，还要看 cell/nuclei mask。

对每个 token，先从 cell mask 下采样得到：

```text
cell_hist[i] = [p_background, p_neoplastic, p_inflammatory, p_connective, p_dead, p_epithelial]
cell_density[i] = 1 - p_background
dominant_cell_id[i] = argmax non-background cell class
```

然后定义 cell similarity：

```text
cell_sim[i, j] = cosine(target_cell_hist[i], ref_cell_hist[j])
```

或更简单：

```text
cell_sim[i, j] = dot(target_cell_hist[i], ref_cell_hist[j])
```

加入 attention bias：

```text
bias[i, j] += lambda_cell_sim * cell_sim[i, j]
bias[i, j] -= lambda_density * abs(target_density[i] - ref_density[j])
```

建议初始值：

```text
lambda_cell_sim = +1.0
lambda_density  = +0.5
```

重要原则：

```text
tissue match 是主约束，cell match 是细化约束。
```

也就是说：

- target tumor 不能因为 cell hist 相似就 attend reference stroma。
- 先按 tissue/coarse 类别筛方向。
- 再在同类 tissue 内用 cell hist/density 挑更像的 reference local tokens。

### 6.6 Cell prior fallback

如果 target 某区域需要某种 cell pattern，但 reference 没有对应 cell 类型或密度很低，也需要 cell prior。

例子：

```text
target_tissue = immune infiltrate
target_cell_hist = high inflammatory
reference_tissue 有 immune，但 reference_cell_mask 里 inflammatory cell 很少
```

此时只靠 reference immune token 可能不够。应该让 target 同时 attend：

- reference immune tokens
- inflammatory cell prior tokens

cell prior bias：

```text
if context_j is cell_prior token for cell class k:
    bias += lambda_cell_prior * target_cell_hist[i, k]
```

如果 `target_cell_hist[i, inflammatory]` 高，则 inflammatory prior tokens 得到加分。

建议：

```text
lambda_cell_prior = +1.0
```

### 6.7 Global stain/style token

reference 缺某个 tissue 类别时，prior 可以生成“典型 tumor”，但还需要尽量跟 reference 的整体染色风格一致。

因此可以加 `global_style_tokens`：

- 主干 MVP 先不启用 global style tokens；
- tissue-only bias 跑通后，第一次启用 global style 时，只建议从 reference local tokens mean-pooling 后投影得到；
- 后续稳定后，才考虑从 reference image 全局 pooled feature 得到；
- 不要在第一版使用可学习 query 对 reference tokens 做 resampling。

原因是 learnable query resampler 很容易变成新的全局 style bottleneck，重新踩回早期 Perceiver/style-collapse 的坑。global style token 只需要提供弱染色兜底，不应该成为强 reference encoder。

它只提供弱全局信息：

- H/E 染色浓淡；
- 扫描亮度；
- stain cast；
- 全局色调。

它不负责 tissue-specific 质感搬运。

建议：

```text
global_style_tokens = weak fallback, not main path
```

这能解决 B 类区域的一个现实问题：

```text
reference 没 tumor，但整体染色偏蓝；
tumor prior 生成 tumor 时，也应该偏向同一张 reference 的蓝调。
```

### 6.8 Attention bias 伪代码

```python
def build_correspondence_bias(
    target_meta,
    context_meta,
    segments,
    *,
    lambda_same_fine=3.0,
    lambda_same_coarse=2.0,
    lambda_mismatch=-2.0,
    lambda_cell_sim=1.0,
    lambda_density=0.5,
    lambda_prior_present=0.5,
    lambda_prior_missing=3.0,
    lambda_prior_wrong=-2.0,
    lambda_cell_prior=1.0,
):
    # MVP 不要构造完整 text+context bias。
    # 只对需要约束的 reference_local / tissue_prior / cell_prior segments
    # 构造局部 bias，再在 attention processor 内加到对应 logits slice。
    # ref_bias: (B, N_img, N_ref)
    # tissue_prior_bias: (B, N_img, N_tissue_prior)
    # cell_prior_bias: (B, N_img, N_cell_prior)
    bias = zeros(...)

    # 1. Reference local tokens
    ref_slice = segments["reference_local"]
    target_coarse = target_meta.tissue_coarse_id      # (B, N_img)
    target_fine = target_meta.tissue_fine_id          # (B, N_img)
    ref_coarse = context_meta.ref_tissue_coarse_id    # (B, N_ref)
    ref_fine = context_meta.ref_tissue_fine_id        # (B, N_ref)

    same_fine = target_fine[:, :, None] == ref_fine[:, None, :]
    same_coarse = target_coarse[:, :, None] == ref_coarse[:, None, :]

    ref_bias = zeros(B, N_img, N_ref)
    ref_bias = where(
        same_fine,
        lambda_same_fine,
        where(same_coarse, lambda_same_coarse, lambda_mismatch),
    )

    cell_sim = cosine_or_dot(
        target_meta.cell_hist[:, :, None, :],
        context_meta.ref_cell_hist[:, None, :, :],
    )
    density_gap = abs(
        target_meta.cell_density[:, :, None]
        - context_meta.ref_cell_density[:, None, :]
    )
    ref_bias = ref_bias + lambda_cell_sim * cell_sim - lambda_density * density_gap

    bias[:, :, ref_slice] += ref_bias

    # 2. Tissue prior tokens
    prior_slice = segments["tissue_prior"]
    for class_id in range(NUM_COARSE):
        token_slice = context_meta.tissue_prior_slices[class_id]
        target_is_class = target_coarse == class_id
        ref_has_class = context_meta.ref_class_presence[:, class_id]

        bonus = where(
            ref_has_class[:, None],
            lambda_prior_present,
            lambda_prior_missing,
        )

        bias[:, :, token_slice] += where(
            target_is_class[:, :, None],
            bonus[:, :, None],
            lambda_prior_wrong,
        )

    # 3. Cell prior tokens
    cell_prior_slice = segments["cell_prior"]
    for cell_id in range(NUM_CELL_WITH_BG):
        token_slice = context_meta.cell_prior_slices[cell_id]
        weight = target_meta.cell_hist[:, :, cell_id]
        bias[:, :, token_slice] += lambda_cell_prior * weight[:, :, None]

    return bias
```

### 6.9 哪些层注入 bias

不一定所有 FLUX 层都要加。

实现注意：这个 bias 会迫使 attention materialize 部分 logits，可能无法继续使用 FlashAttention/SDPA 这类 fused kernel。它的显存风险不在 prior token 数，而在 `N_img x N_context` 的 attention 矩阵本身。

粗略规模：

```text
256x256 patch -> packed image tokens 256, reference tokens 256
512x512 patch -> packed image tokens 1024, reference tokens 1024
```

512 分辨率下，单样本仅 image-to-reference bias 就约 `1024 x 1024`，多 head、多层、多 denoise step 时会明显吃显存；如果 attention processor 因 additive bias 回退到非 fused attention，成本会更高。

所以 MVP 的硬约束是：

- 先用 256x256 patch 验证机制。
- 只在 1 个 double transformer block 的 joint attention 加 bias。
- 优先选中后层的一个 block。
- bias tensor 使用 `bfloat16` / `float16`，不要用 fp32。
- 只对 `reference_local`、`tissue_prior`、后续的 `cell_prior` segment 构造和注入 bias，不对 text tokens 或整段 context 构造稠密 bias。
- single transformer blocks 先不加，除非发现 reference 使用不足。

原因：

- 早期层更偏全局结构；
- 中后层更影响纹理和局部外观；
- 全层强加 bias 可能影响基础生成稳定性。
- additive bias 可能关闭 fused attention，层数每加一层都是真显存/速度成本。

如果使用当前训练脚本中 `--num-double-layers 4 --num-single-layers 4` 的轻量设置，也不建议一开始全部 double layers 加。先单层跑通 attention mass 和 ref-swap 诊断，再扩到 2 层。

### 6.10 bias 强度调度

训练初期模型还没学会 prior/ref token 的语义，如果一开始 bias 太强可能不稳定。尤其是 prior tokens 新初始化时，过早把 attention 强推到 prior，会把随机 token 的噪声放大。

建议：

```text
step 0 - 500:      lambda scale 0.2
step 500 - 1500:   linear warmup to 1.0
step > 1500:       full strength
```

或者：

```text
lambda_effective = lambda_base * min(1, global_step / warmup_steps)
```

更稳的顺序：

```text
step 0 - 300:      prior tokens 先进入 context，但 correspondence bias scale = 0
step 300 - 1000:   tissue/reference bias 从 0 warmup 到 1
step 1000 后:      prior fallback bias 再从低值 warmup
```

同时建议 prior token bank 使用单独学习率，略高于 reference/context projection，例如 `2x - 5x conditioning_learning_rate`，但仍低于会破坏 FLUX 主干的量级。

---

## 7. 细胞 mask 在模型中的完整角色

先明确来源：你的 pipeline 会生成 target 侧的 cell/nuclei mask，所以 Cross V4 不需要在推理时临时猜一个 target cell mask，也不需要把 cell branch 降级成“没有输入时才可选”的弱功能。真正缺的是把这个已经存在的 cell/nuclei mask 用到更深的位置。

### 7.1 细胞 mask 不能只作为 ControlNet 输入

如果 cell/nuclei mask 只进入 ControlNet，那么它主要控制 target 的结构，例如：

- 哪些地方细胞多；
- 哪些地方细胞少；
- 粗略细胞类型分布；
- 局部高频结构。

但它不会直接告诉 reference attention：

```text
target 这个 high inflammatory 区域应该去 reference high inflammatory 区域取质感。
```

所以 Cross V4 必须让 cell mask 同时进入：

1. target structure path；
2. reference appearance path；
3. semantic correspondence bias。

### 7.2 tissue-cell 双层对齐

建议采用两阶段匹配逻辑：

```text
第 1 层：tissue semantic match
target tumor -> ref tumor
target stroma -> ref stroma
target necrosis -> ref necrosis

第 2 层：cell microenvironment match
在同类 tissue 内：
  high neoplastic -> high neoplastic
  high inflammatory -> high inflammatory
  high connective -> high connective
  high dead -> high dead
  similar nuclei density -> similar nuclei density
```

注意不要让 cell match 破坏 tissue match。

错误例子：

```text
target tumor 区有很多 neoplastic cells
reference stroma 区也因为误分割有一些 neoplastic-like cells
模型不应该跨 tissue 去 attend stroma
```

因此：

```text
tissue mismatch penalty > cell similarity bonus
```

### 7.3 细胞 density 特别重要

病理质感很多时候不是由“细胞类别”单独决定，而是由 density 决定：

- dense tumor nuclei -> 深色、高核质比、拥挤。
- loose stroma -> 粉红纤维、细胞少。
- immune infiltrate -> 小圆深染核密集。
- necrosis -> dead/debris/低结构。

所以每个 token 应该有：

```text
cell_density = non_background_cell_pixels / total_pixels
```

attention 中加入：

```text
- lambda_density * abs(target_density - ref_density)
```

这会鼓励 target 高细胞密度区域 attend reference 高细胞密度区域。

### 7.4 instance mask 的未来扩展

如果未来你有细胞 instance mask，而不只是 type mask，可以进一步提取：

- nuclei count per token；
- mean nuclei area；
- mean eccentricity；
- nearest-neighbor distance；
- clustering index；
- boundary density；
- local graph features。

第一版不建议上来就做这么复杂，但文档层面要预留接口：

```python
CellTokenStats:
    class_hist
    density
    instance_count
    mean_area
    mean_eccentricity
    mean_nn_distance
```

当前最重要的是先把 `class_hist + density` 用起来。

### 7.5 cell mask 的噪声处理

CellViT 或其它细胞分割模型的 mask 可能有噪声。

因此 cell bias 应该是 soft bias，不建议 hard mask：

```text
tissue: 可中等强度约束
cell: 软加分/减分
```

如果 cell mask 质量不稳定，建议：

- 对 cell histogram 做 token-level 平滑。
- density clamp 到合理范围。
- 忽略很低 confidence 的 cell token。
- 训练时随机 drop cell bias，让模型不完全依赖 noisy cell mask。

---

## 8. Reference 有同类 vs 缺同类：A/B 区域自动分流

### 8.1 A 类区域：reference 有对应组织/细胞

例子：

```text
target tumor 区域
reference 中也有 tumor 区域
```

期望行为：

- target tumor image tokens attend reference tumor local tokens。
- 若 cell mask 指示 target tumor 区域 neoplastic density 高，则优先 attend reference tumor 中 neoplastic density 高的 tokens。
- route anchor tumor token 提供类别摘要。
- tumor prior token 只弱参与，不抢 reference。

attention 结果应该是：

```text
ref_same_tissue attention mass 高
prior attention mass 低/中
ref_mismatch attention mass 低
```

### 8.2 B 类区域：reference 缺对应组织/细胞

例子：

```text
target tumor 区域
reference 中没有 tumor
```

期望行为：

- target tumor image tokens 不应该 attend reference stroma 伪造 tumor。
- target tumor image tokens 应 attend tumor prior tokens。
- global style tokens 提供 reference 的染色风格。
- cell prior 根据 target cell mask 提供 neoplastic nuclei pattern。

attention 结果应该是：

```text
tissue_prior attention mass 高
cell_prior attention mass 中/高
global_style attention mass 中
ref_mismatch attention mass 低
```

### 8.3 同一张 target 中 A/B 共存

同一张 target 可能有：

- tumor: reference 有 -> 听 reference tumor；
- necrosis: reference 没有 -> 听 necrosis prior；
- stroma: reference 有 -> 听 reference stroma；
- immune: reference 有但 cell density 不匹配 -> reference immune + inflammatory prior 混合。

这就是为什么需要 token-level soft attention，而不是整张图级别的“用 reference / 不用 reference”开关。

---

## 9. 训练数据设计

### 9.1 数据基本原则

训练目标：

```text
model(reference_image, reference_masks, target_masks) -> target_image
```

GT：

```text
target_image
```

reference 和 target 建议来自：

1. 同 WSI / 同 case，不同 patch。
2. 同 dataset，同 label space。
3. 尽可能共享染色域。

这是 supervised cross-reconstruction。

### 9.2 Coverage bucket 必须被显式使用

当前 `controlnet_train/data/cross.py` 已经定义：

```text
full:    target tissue classes 都被 reference 覆盖，或 target 没有效类别
partial: target 部分类别 reference 有，部分类别 reference 没有
low:     target 类别几乎都 reference 没有
```

Cross V4 训练必须显式平衡这些 bucket。

建议 batch sampling 比例：

```text
full coverage:    50%
partial coverage: 35%
low coverage:     15%
```

训练初期可以：

```text
full coverage:    70%
partial coverage: 25%
low coverage:      5%
```

中后期再提高 partial/low。

### 9.3 为什么 full coverage 重要

full coverage 样本训练 A 类行为：

```text
有同类 reference -> 听 reference
```

如果 full coverage 太少，模型学不到真正的 semantic texture retrieval。

full coverage 里还可以设计更强的训练：

- target 和 reference 同 case 但空间布局差异大；
- target/ref 中同类 tissue 区域分布在不同位置；
- cell density 有局部差异；
- 防止模型靠位置对应偷懒。

### 9.4 为什么 partial/low coverage 重要

partial/low coverage 样本训练 B 类行为：

```text
reference 缺类 -> 听 prior
```

如果没有 partial/low：

- prior tokens 学不到东西；
- 推理时 reference 缺类会崩；
- 模型可能错误 attend mismatch reference。

### 9.5 cell coverage 也要记录

当前 metadata 主要记录 tissue coverage。

Cross V4 还应该记录 cell coverage：

```text
target_cell_ids
reference_cell_ids
covered_target_cell_ids
missing_target_cell_ids
cell_coverage_ratio
cell_density_similarity_by_tissue
```

特别是：

```text
target tissue class c 内的 cell histogram
reference tissue class c 内的 cell histogram
```

例子：

```json
{
  "covered_target_tissue_ids": [1, 2],
  "missing_target_tissue_ids": [3],
  "cell_coverage_ratio": 0.75,
  "tissue_cell_hist_similarity": {
    "1": 0.83,
    "2": 0.71
  }
}
```

这能帮助采样器构造：

- tissue full + cell full；
- tissue full + cell mismatch；
- tissue partial + cell full；
- tissue partial + cell partial。

### 9.6 Reference crop 采样策略

对每个 target patch：

1. 计算 target tissue set。
2. 计算 target cell stats。
3. 在同 case 候选 reference 中搜索：
   - full coverage reference；
   - partial coverage reference；
   - low coverage reference；
   - cell-similar reference；
   - cell-dissimilar reference。
4. 按 bucket 概率采样。

建议 scoring：

```text
score =
  10.0 * tissue_coverage_score
+  3.0 * tissue_area_coverage_score
+  2.0 * cell_hist_similarity
+  2.0 * stain_similarity
-  1.0 * distance_penalty
```

当前 `_score_reference()` 已经有 tissue coverage、nuclei hist similarity、stain similarity、distance penalty。Cross V4 可以继续扩展为 per-tissue cell similarity。

### 9.7 自重建 warmup

当前训练脚本已有：

```text
self_reconstruction_warmup_steps
self_reconstruction_sample_prob
```

Cross V4 建议保留，但不要让 warmup 太久。

作用：

- 检查 reference tokens、ControlNet condition、VAE latent 流水线是否通。
- 让模型先学会“reference=target 时可以重建”。

风险：

- 太久会让模型学成复制同位置 reference，而不是跨空间对应。

建议：

```text
self_reconstruction_warmup_steps: 500 - 1000
self_reconstruction_sample_prob after warmup: 0.05 - 0.10
```

### 9.8 Spatial jitter / crop mismatch

为了防止模型靠 ref/target 的局部相似性偷懒，训练时应保证：

- reference 和 target 不同 patch；
- 若同 patch自重建，只用于 warmup/少量 regularization；
- reference crop 可以做随机平移/翻转/旋转；
- target/ref 相对位置不要进入 reference token coordinate；
- 当前 `append_cross_v3_reference_context()` 中 reference `txt_ids` 置零，这一点保留是合理的，避免模型把 reference token 当 target 坐标。

### 9.9 防止 bias 复制捷径

mask-guided bias 会显式告诉 target token 去找 reference 的同类 token。这个机制是必要的，但它也可能制造新的 shortcut：

```text
同 WSI 训练时，reference tumor 和 target tumor 本来就很像；
bias 又把 target tumor 指向 reference tumor；
模型可能学成“按 bias 复制 reference 同类局部内容”，而不是学会稳健的质感检索和重排。
```

这会导致 same-WSI validation 看起来很好，但跨 WSI / 跨病例 reference 推理时崩掉。它和早期 cross-sample spatial misalignment 的问题同构，只是这次 shortcut 藏在 semantic bias 里。

因此训练必须刻意破坏“可直接复制”的条件：

- reference 与 target 尽量选同 case 但远距离 crop，而不是相邻 crop。
- 对 reference image/mask 做同步随机翻转、旋转、轻微仿射、颜色扰动，target 不做同样变换。
- full coverage 样本也要包含布局差异大的 reference-target 对。
- 自重建样本只用于短 warmup 和少量 sanity regularization，不能成为主训练分布。
- ref-swap loss 在这里应保留并加强诊断意义：换成另一个同类 reference 后，covered region 的输出/损失应该发生可测变化。
- validation 必须包含跨 WSI 或至少跨 case reference 的 qualitative set，即使没有严格 GT，也要看模型是否把 target 结构守住、是否只搬 style 不搬 reference 布局。

MVP 成功标准不能只看 same-WSI reconstruction。至少要同时满足：

- same-case full coverage 上 normal ref 优于 zero/random ref。
- cross-case/cross-WSI reference 上 target mask 结构不被 reference 布局污染。
- attention mass 仍然落在 reference 同类 token，而不是退回全局 text/prior。
- 换 reference 后，covered class 的区域外观有方向性变化，missing class 不错误贴 mismatch reference。

---

## 10. 训练阶段建议

### 10.1 Stage 0: plumbing sanity

目标：

- 确认所有 tensor shape、dtype、checkpoint save/load、inference bundle 都可用。
- 确认 reference tokens 进入 transformer 后确实影响 noise prediction。

设置：

```text
self reconstruction only
small dataset
few hundred steps
attention bias scale = 0 or low
prior tokens enabled but weak
```

诊断：

- zero reference vs normal reference 的 noise pred diff。
- reference token grad norm。
- prior token grad norm。
- decoded reconstruction sanity。

### 10.2 Stage 1: full coverage correspondence

目标：

- 训练“有同类 reference 时听 reference”。

设置：

```text
full coverage high ratio
prior weak
mask-guided reference bias enabled
cell bias enabled soft
```

MVP 阶段这里的 `cell bias enabled soft` 应理解为可选项。为了先验证主干，第一轮训练建议只开 tissue-only bias；cell histogram/density bias 放到 Step 4。

损失：

- denoise loss。
- regional stain/style loss。
- optional attention same-class mass regularizer。
- ref-swap sensitivity loss。

期望现象：

- normal reference loss < zero/random reference loss。
- same-class attention mass 上升。
- generated target 的同类区域色调/纹理更接近 reference 同类区域。
- 换成另一个同类 reference 时，covered class 的外观有可观察变化，但 target mask 结构不漂移。

### 10.3 Stage 2: partial coverage + prior fallback

目标：

- 训练“reference 缺类时听 prior”。

设置：

```text
full/partial/low mixed
prior fallback bias enabled
reference dropout
missing class sampling explicit
```

损失：

- denoise loss。
- attention fallback regularizer：
  - if target class present in ref -> encourage ref same-class mass。
  - if target class absent in ref -> encourage matching prior mass。
- prior usage entropy / anti-collapse loss。

期望现象：

- missing classes 的 prior token grad norm 非零。
- missing class target regions 的 prior attention mass 高。
- reference mismatch attention mass 低。
- generation 不再把 stroma reference 硬套成 tumor。

### 10.4 Stage 3: stain/style robustness

目标：

- 保证跨 patch、轻微跨 stain 的外观迁移稳定。
- 让 B 类 prior fallback 也能跟随 reference 全局 stain。

设置：

```text
HED stain augmentation
stain counterfactual pairs
global style tokens enabled
```

损失：

- regional stain/style loss。
- global stain consistency loss。
- decoded image color stats loss。

注意：

- HED augmentation 不能太强，否则 reference-target GT 关系会变脏。
- 第一版建议轻量 HED。

### 10.5 Stage 4: cross-case / cross-WSI eval only

跨 WSI 没有严格 GT，建议先作为 eval/inference，而不是主训练。

可以做：

- reference retrieval eval。
- zero/random/reference ablation。
- 人工视觉 QA。
- segmentation consistency eval。
- cell distribution consistency eval。

若要训练，可用弱监督：

- mask consistency。
- style consistency。
- CLIP/UNI/pathology feature similarity。

但主能力应该先由 same-case supervised training 学出来。

---

## 11. Loss 设计

### 11.1 主 denoise loss

沿用当前 FlowMatch denoise objective：

```text
loss_denoise = MSE(noise_pred, target_velocity)
```

这是主监督。

### 11.2 Regional stain/style loss

当前已有：

```python
regional_stain_style_loss(
    prediction,
    reference,
    target_tissue_mask,
    reference_tissue_mask,
    target_nuclei_mask,
    reference_nuclei_mask,
)
```

它按共享 label 匹配 target/ref 区域的颜色统计。

Cross V4 应继续使用，但要注意：

- 只对 reference 有同类的区域计算强 style loss。
- reference 缺类的 target 区域不能强行和 mismatch reference 匹配。
- cell/nuclei region loss 应该保留，但权重要温和。

建议：

```text
reference_style_loss_weight = 0.5 - 1.0
tissue_weight = 1.0
nuclei/cell_weight = 0.5 - 1.0
cov_weight = 0.1 - 0.25
```

### 11.3 Ref-swap sensitivity loss

当前已有：

```text
normal reference loss 应该低于 zero/random reference loss 至少 margin
```

这是好的，但它是全图级的。

Cross V4 建议升级成 coverage-aware：

- 对 full coverage 样本，normal ref 应明显优于 zero/random。
- 对 partial coverage 样本，只在 covered classes 上要求 normal ref 优于 swapped ref。
- 对 missing classes，不要求 normal ref 优于 swapped ref，因为本来就该靠 prior。

理想版本：

```text
loss_ref_swap_local =
  covered_region_weight * max(0, margin + normal_loss_region - swapped_loss_region)
```

需要能算 region-level denoise/pixel loss，或者 decoded image 后按 mask 计算。

### 11.4 Attention correspondence regularizer

如果能从 FLUX attention processor 中拿到 attention maps，建议加轻量正则：

对 target token `i`：

```text
if reference has same target class:
    encourage sum_attention_to_ref_same_class(i) high
else:
    encourage sum_attention_to_matching_prior(i) high
```

形式：

```text
loss_attn_present =
  - log(attn_mass_ref_same_class + eps)

loss_attn_missing =
  - log(attn_mass_matching_prior + eps)
```

权重建议：

```text
attention_loss_weight = 0.01 - 0.05
```

这不是主损失，只是防止模型绕开 reference/prior token。

### 11.5 Prior anti-collapse loss

Prior tokens 有两个风险：

1. 全部类别 prior 学成一样。
2. 模型过度依赖 prior，reference 有同类也不听 reference。

可以加：

```text
prior_diversity_loss:
    不同 class prior token 的 cosine similarity 不要太高

prior_usage_balance:
    present-ref 区域 prior attention 不要过高
    missing-ref 区域 prior attention 不要过低
```

第一版可以只监控，不一定上损失。

### 11.6 Mask consistency loss

可选，风险较高但很有价值。

训练时 decode `pred_original` 得到 RGB，再用 frozen segmentator / CellViT 预测：

- predicted tissue mask 应接近 target tissue mask；
- predicted cell mask/density 应接近 target cell mask。

问题：

- 解码耗显存和时间；
- segmentator 本身有误差；
- 训练早期图像质量差时 loss 不稳定。

建议先作为 eval metric，后续再上轻量 loss。

### 11.7 总损失建议

第一版总损失：

```text
loss =
  loss_denoise
+ w_style * loss_regional_style
+ w_swap  * loss_ref_swap
+ w_attn  * loss_attention_correspondence
```

初始权重：

```text
w_style = 0.5
w_swap  = 0.1
w_attn  = 0.02
```

如果不方便拿 attention maps：

```text
先不上 w_attn，但必须有 attention bias 和 diagnostics。
```

---

## 12. Inference 设计

### 12.1 Inference 输入

沿用 Phase 5.4 edit pipeline 的空间条件，但明确 cell mask 的来源：`target_cell_mask` / `target_nuclei_mask` 由你的 pipeline 自动生成，并作为 Cross V4 的正式条件输入进入模型。

```text
reference_image
reference_tissue_mask
reference_cell_mask / reference_nuclei_mask
target_tissue_mask
target_cell_mask / target_nuclei_mask
```

因此推理时的接口语义是：

- reference 侧：来自当前图/检索图及其 tissue + cell/nuclei masks。
- target 侧：来自编辑后的 `target_tissue_mask`，以及 pipeline 根据目标组织状态生成的 `target_cell_mask` / `target_nuclei_mask`。
- Cross V4 不把 target cell mask 当作缺失项；它把它当作 target 微环境结构的一部分。

### 12.2 Inference 步骤

```text
1. Encode reference image -> z_ref
2. Encode reference tissue/cell masks -> ref features
3. Build reference local tokens + semantic metadata
4. Build target ControlNet condition from target tissue/cell masks
5. Build target token semantic metadata from target tissue/cell masks
6. Build prior/global tokens
7. Build correspondence bias for configured transformer layer(s)
8. Run FLUX ControlNet sampling
9. Save generated image + debug reports
```

### 12.3 Reference retrieval

第一版可以让用户显式给 reference。

后续如果自动检索，retrieval score 应该和训练采样一致：

```text
score =
  tissue coverage
+ cell coverage
+ per-tissue cell hist similarity
+ stain similarity
+ dataset/case preference
- distance penalty
```

如果 target 某类 reference 检索不到，不必失败，因为 prior fallback 可以兜底。但 debug report 必须告诉用户：

```text
missing_target_tissue_ids: [...]
missing_target_cell_ids: [...]
这些区域主要由 prior 生成，不是 reference 搬运。
```

### 12.4 Debug output 必须保留

Cross V4 不是黑盒越黑越好。每次 inference 最好保存：

```text
run_summary.json
coverage_report.json
attention_source_stats.json
target_mask_overlay.png
reference_mask_overlay.png
generated_vs_reference_region_style.json
```

最重要的统计：

```text
for each target tissue class:
    area_ratio
    ref_present
    attention_mass_ref_same
    attention_mass_ref_mismatch
    attention_mass_tissue_prior
    attention_mass_cell_prior
    attention_mass_global
```

如果生成失败，这些能快速判断：

- 是 reference 没同类；
- 是 attention bias 没生效；
- 是 prior 没学到；
- 是 cell mask 噪声；
- 是 ControlNet 结构没控制住。

---

## 13. 当前缺的东西

下面是最重要的缺口清单，按优先级排序。

### 13.1 缺口 1：真正的 mask-guided attention bias

当前状态：

- `append_cross_v3_reference_context()` 只是把 reference tokens 拼到 prompt 后。
- `flux_transformer(... joint_attention_kwargs=None ...)` 没有传任何 semantic bias。
- target token 不知道哪些 reference token 同类。

需要新增：

- target token semantic grid builder。
- reference token semantic metadata。
- context segment offsets。
- correspondence bias tensor。
- FLUX attention processor / forward patch，使 attention logits 能加 bias。

这是最核心缺口。不做这个，reference 纹理很可能继续只表现为全局色调。

### 13.2 缺口 2：per-class prior fallback tokens

当前状态：

- route anchor 有 missing anchor，但不是明确的 per-target-class generation prior。
- fixed prompt `"histopathology image"` 信息量太低，不能承担“凭空生成某类组织”的兜底。

需要新增：

- coarse tissue prior tokens。
- cell prior tokens。
- optional global style tokens。
- prior token segment metadata。
- present/missing aware prior bias。
- prior token save/load。

### 13.3 缺口 3：pipeline 生成的 cell/nuclei mask 还没有参与 attention 对齐

当前状态：

- 你的 pipeline 会生成 target 侧 cell/nuclei mask，推理输入闭环是成立的。
- `NucleiConditionEncoder` 把 nuclei mask 编成 feature。
- reference token 中也包含 `ref_nuclei_feat`。
- regional style loss 可按 nuclei mask 计算。

但缺少的是更深一层的使用方式：

- target token cell histogram。
- reference token cell histogram。
- nuclei density。
- cell similarity attention bias。
- cell prior fallback。
- cell coverage metadata。

你已经有 pipeline 生成的 cell mask，这应该成为架构优势，而不是只作为普通 ControlNet 控制通道。

### 13.4 缺口 4：coverage-aware sampler 还不够细

当前状态：

- tissue coverage bucket 已有。
- `missing_target_tissue_ids` 已记录。

需要新增：

- 强制 batch 中 full/partial/low 的比例。
- cell coverage bucket。
- per-tissue cell histogram similarity。
- missing class targeted sampling。
- low coverage 样本的 prior-focused sampling。

### 13.5 缺口 5：loss 还没有局部化到 covered/missing 区域

当前状态：

- denoise loss 是全图。
- ref-swap loss 是全图。
- regional style loss 按共享 label 做颜色统计。

需要新增或改造：

- coverage-aware ref-swap。
- attention correspondence regularizer。
- prior fallback regularizer。
- optional region-level decoded loss。

### 13.6 缺口 6：attention diagnostics

当前状态：

- 有 `ref_check_step`，可以比较 zero-ref / with-ref 的 noise pred diff。

但这只能回答：

```text
reference 有没有影响模型？
```

不能回答：

```text
target tumor 有没有 attend reference tumor？
target missing necrosis 有没有 attend necrosis prior？
cell mask 有没有影响选择？
```

需要新增：

- attention capture hooks。
- per-class attention mass。
- per-source attention heatmaps。
- prior usage stats。
- mismatch attention stats。

### 13.7 缺口 7：inference bundle/checkpoint spec 需要升级

当前保存：

- `cross_v3_control_spec`
- `cross_v3_reference_spec`
- condition modules。

Cross V4 需要保存：

- `cross_v4_correspondence_spec`
- prior token weights。
- bias hyperparameters。
- route/prior/context segment policy。
- cell metadata policy。
- attention processor config。

否则训练和推理会不一致。

### 13.8 缺口 8：评估指标还不完整

需要新增：

- Reference usage metric:
  - normal vs zero ref
  - normal vs random ref
  - same-class region style distance
  - mismatch attention mass

- Mask consistency metric:
  - generated image -> segmentator -> tissue mask agreement
  - generated image -> CellViT -> cell mask/density agreement

- Coverage-specific metric:
  - full bucket quality
  - partial bucket quality
  - low bucket quality
  - missing class prior quality

- Cell-aware metric:
  - cell density error
  - cell type histogram error
  - per-tissue cell composition error

### 13.9 缺口 9：pipeline 生成的细胞 mask 质量审计

即使 pipeline 会生成 cell/nuclei masks，也仍然需要审计它们的质量，因为 Cross V4 会把它们用于 attention 对齐和 prior fallback。需要知道：

- 哪些 dataset 有真实 cell labels；
- 哪些是 CellViT 推理结果；
- label ID 是否统一；
- cell mask 和 tissue mask 是否空间对齐；
- cell mask 是否落在合理 tissue 区域内；
- dead cell 是否和 necrosis 对应；
- inflammatory cell 是否和 immune infiltrate 对应。

没有这个审计，cell bias 可能把噪声放大。

### 13.10 缺口 10：ablation matrix

必须设计 ablation，否则很难知道哪个部件有效。

建议最小 ablation：

```text
A0: Cross V3 baseline
A1: + tissue mask-guided ref bias
A2: + tissue prior tokens
A3: + cell histogram/density bias
A4: + cell prior tokens
A5: + global style tokens
A6: + attention regularizer
```

每个 ablation 都评估：

- full coverage。
- partial coverage。
- low coverage。
- zero/random ref sensitivity。
- cell consistency。

---

## 14. 代码落地蓝图

### 14.1 新增模块建议

建议新增：

```text
controlnet_train/modules/cross_v4_correspondence.py
```

包含：

```python
CrossV4CorrespondenceSpec
CrossV4ContextSegments
CrossV4TokenMetadata
build_tissue_token_metadata(...)
build_cell_token_metadata(...)
build_cross_v4_context(...)
build_correspondence_bias(...)
```

### 14.2 修改 reference encoder

当前：

```python
reference_tokens = reference_context_encoder(...)
```

建议改为：

```python
reference_context = reference_context_encoder(...)

reference_context.tokens
reference_context.token_metadata
reference_context.route_anchor_metadata
```

或保持 encoder 返回 tokens，但另写 metadata builder：

```python
reference_tokens = reference_context_encoder(...)
reference_meta = build_reference_token_metadata(
    reference_tissue_mask,
    reference_nuclei_mask,
    token_height,
    token_width,
)
```

第一版建议后者，改动更小。

### 14.3 新增 target metadata builder

```python
target_meta = build_target_token_metadata(
    target_tissue_mask=batch["target_tissue_mask"],
    target_nuclei_mask=batch["target_nuclei_mask"],
    token_height=pixel_latents.shape[2] // 2,
    token_width=pixel_latents.shape[3] // 2,
)
```

注意 token grid 要和 FLUX packed image tokens 对齐。

### 14.4 新增 prior token module

```python
class CrossV4PriorTokenBank(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_tissue_classes: int = NUM_COARSE,
        tissue_tokens_per_class: int = 4,
        num_cell_classes: int = 6,
        cell_tokens_per_class: int = 2,
        global_tokens: int = 2,
    ):
        ...

    def forward(self, batch_size: int, reference_global_feature=None):
        return prior_tokens, prior_metadata
```

如果 global style token 从 reference 动态生成，可以另写：

```python
CrossV4GlobalStyleEncoder
```

第一版可以先用：

```text
global_style_token = projection(mean(reference_tokens))
```

### 14.5 替代 append_cross_v3_reference_context

新增：

```python
append_cross_v4_context(
    prompt_embeds,
    text_ids,
    reference_tokens,
    reference_meta,
    prior_tokens,
    prior_meta,
)
```

返回：

```python
encoder_hidden_states
context_ids
context_meta
segments
```

### 14.6 Attention processor

这是最难的工程点。

当前 diffusers FLUX transformer 调用：

```python
flux_transformer(
    encoder_hidden_states=batch_context,
    txt_ids=context_ids,
    img_ids=latent_image_ids,
    joint_attention_kwargs=None,
)
```

Cross V4 需要让 attention 层接收：

```python
joint_attention_kwargs={
    "correspondence_bias": bias,
    "context_segments": segments,
    "capture_attention": ...
}
```

但 diffusers 当前 attention processor 不一定原生支持 arbitrary additive bias。可能需要：

1. 自定义 attention processor。
2. monkey patch FLUX attention forward。
3. 在指定 transformer blocks 中替换 processor。
4. 让 processor 在 `QK^T` 后、softmax 前加 `bias`。

伪代码：

```python
attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale

if correspondence_bias is not None and query_is_image_to_context:
    attn_scores[..., image_query_slice, context_key_slice] += correspondence_bias

attn_probs = softmax(attn_scores, dim=-1)
hidden = attn_probs @ v
```

注意 FLUX joint attention 可能把 image tokens 和 context tokens 拼在同一个 attention 中，所以 bias 的 shape/slice 要和真实实现对齐。

### 14.7 Training loop 改造

在 `flux_phase5_cross_v3.py` 基础上，Cross V4 training loop 需要新增：

```python
target_meta = build_target_token_metadata(...)
reference_meta = build_reference_token_metadata(...)
prior_tokens, prior_meta = prior_token_bank(...)
batch_context, context_ids, context_meta, segments = append_cross_v4_context(...)
correspondence_bias = build_correspondence_bias(target_meta, context_meta, segments, ...)

noise_pred = flux_transformer(
    ...,
    encoder_hidden_states=batch_context,
    txt_ids=context_ids,
    img_ids=latent_image_ids,
    joint_attention_kwargs={
        "correspondence_bias": correspondence_bias,
        "capture_attention": should_capture,
    },
)
```

### 14.8 Inference loop 改造

`controlnet_train/inference/pipeline_cross_v3.py` 需要对应升级：

- load prior token bank；
- build metadata；
- build context segments；
- build correspondence bias per sampling step；
- pass into transformer during pipeline sampling；
- save debug outputs。

如果 diffusers pipeline 内部不容易传动态 `joint_attention_kwargs`，可能需要自定义 sampling loop，而不是直接调用 pipeline `__call__`。

当前 Cross V3 inference 已经自定义了部分 `_sample_with_flux_controlnet`，可以继续扩展。

### 14.9 Checkpoint 保存

`phase5_conditioning.pt` 需要新增：

```python
"cross_version": "v4",
"cross_v4_correspondence_spec": {
    "tissue_bias_level": "coarse_plus_fine",
    "num_tissue_prior_classes": 8,
    "tissue_tokens_per_class": 4,
    "num_cell_prior_classes": 6,
    "cell_tokens_per_class": 0,
    "global_style_tokens": 0,
    "route_anchor_mode": "none",
    "mvp_patch_size": 256,
    "materialized_attention_layers": 1,
    "bias_segments": ["reference_local", "tissue_prior"],
    "bias_dtype": "bf16",
    "lambda_same_fine": 3.0,
    "lambda_same_coarse": 2.0,
    "lambda_mismatch": -2.0,
    "lambda_cell_sim": 1.0,
    "lambda_density": 0.5,
    "lambda_prior_present": 0.5,
    "lambda_prior_missing": 3.0,
    "lambda_prior_wrong": -2.0
}
```

并保存：

```python
"prior_token_bank": state_dict
"global_style_encoder": state_dict, optional
```

---

## 15. 推荐实现顺序

### Step 1：只做 metadata，不改 attention

目标：

- 确保 target/ref tissue/cell token metadata 都能正确生成。

新增测试：

- mask 下采样 shape 正确；
- fine/coarse ID 正确；
- cell histogram sum 为 1；
- density 范围 `[0, 1]`；
- reference token 数和 reference_tokens 对齐；
- target token 数和 packed latent image tokens 对齐。

### Step 2：实现 prior token bank，但先只拼 context

目标：

- prior tokens 能训练、保存、加载。
- 不加 bias，观察是否有梯度。
- `route_anchor_mode` 在 MVP 中保持 `none`，避免 route anchor 和 prior token 两套 per-class 通道互相分散 attention。

诊断：

- prior token grad norm；
- prior token abs mean/std；
- zero prior ablation。

### Step 3：实现 tissue-only correspondence bias

目标：

- target tissue -> reference same tissue。
- target missing tissue -> tissue prior。

暂时不加 cell bias，先验证主机制。

评估：

- tumor/stroma/necrosis/immune full/partial bucket。
- attention mass same/mismatch/prior。
- normal vs zero/random ref。

### Step 4：加入 cell histogram/density bias

目标：

- 在 tissue 同类内部进一步按 cell 微环境选 reference。

评估：

- immune infiltration。
- necrosis/dead cell。
- tumor/neoplastic density。
- stroma/connective density。

### Step 5：加入 cell prior tokens

目标：

- target cell pattern reference 缺失时有兜底。

### Step 6：加入 global style tokens

目标：

- missing tissue/cell 区域也能跟 reference 整体染色一致。

MVP 后的第一版 global style 只做 `mean(reference_local_tokens) -> projection -> global_style_tokens`。不要先上 learnable query resampler。

### Step 7：加 attention regularizer

目标：

- 防止模型绕过 correspondence bias。
- 提高可解释性。

---

## 16. 关键超参数建议

### 16.1 Token 数

```text
tissue_prior_level: coarse
tissue_tokens_per_class: 4
cell_tokens_per_class: 0 in MVP, then 2 after tissue-only bias works
global_style_tokens: 0 in MVP, then 2 after tissue-only bias works
route_anchor_mode: none in MVP
```

第一版不要上 fine prior token。

route anchor 和 prior token 功能有重叠，都是 per-class/context summary。为了让 attention 诊断干净，MVP 关闭 route anchor，只保留 prior token。等 tissue-only bias 和 prior fallback 明确有效后，再单独 ablate route anchor 是否有增益。

### 16.2 Bias 强度

```text
lambda_same_fine:   3.0
lambda_same_coarse: 2.0
lambda_mismatch:   -2.0
lambda_cell_sim:    1.0
lambda_density:     0.5
lambda_prior_present: 0.5
lambda_prior_missing: 3.0
lambda_prior_wrong:  -2.0
lambda_cell_prior:    1.0
```

训练前 1000-1500 step 做 warmup。

这些 lambda 都按“最终加到 logits 上的带符号数值”记录，不再写成“正数再手动取负”，避免实现时漏掉符号。

### 16.3 Attention 显存安全配置

```text
mvp_patch_size: 256
materialized_attention_layers: 1 double block
bias_segments: reference_local + tissue_prior only
bias_dtype: bf16/fp16
disable_cell_bias_in_mvp: true
disable_global_style_in_mvp: true
route_anchor_mode: none
```

512 patch、2 个以上注入层、cell prior/global style/route anchor 同时开启，都应该放到 MVP 之后逐项 ablate，并记录显存峰值、step time、attention mass。

### 16.4 Loss 权重

```text
denoise: 1.0
regional_style: 0.5
ref_swap: 0.1
attention_regularizer: 0.02
prior_diversity: 0.001, optional
```

ref-swap 的 `0.1` 是起步值。如果诊断显示模型仍然不读 reference，可提高到 `0.2 - 0.3`，但要按 covered/missing region 分开看，避免 missing class 被错误惩罚。

### 16.5 Sampling 比例

```text
Stage 1:
  full: 70
  partial: 25
  low: 5

Stage 2+:
  full: 50
  partial: 35
  low: 15
```

### 16.6 cell bias dropout

为了防止 cell mask 噪声：

```text
cell_bias_dropout_prob: 0.1 - 0.2
```

训练时随机关闭 cell bias，让模型仍能靠 tissue 和 image prior 工作。

---

## 17. 需要重点监控的失败模式

### 17.1 reference 只影响全局色调

表现：

- normal vs zero ref 有差异；
- 但同类纹理没迁移；
- attention mass 没有集中到 same-class ref。

排查：

- correspondence bias 是否真的加到 logits；
- context segment slice 是否错位；
- target/ref token grid 是否对齐；
- lambda 是否太小；
- ref tokens 是否被 projection 初始化太弱。

### 17.2 target 结构被 reference 布局污染

表现：

- 生成图出现 reference 的空间布局；
- target mask 边界不被遵守。

排查：

- reference token 是否带了坐标 ID；
- ControlNet scale 是否太低；
- attention bias 是否过强导致 reference local structure 直接复制；
- target tissue/cell features 是否有效。

### 17.3 reference 缺类时生成错类

表现：

- target 要 tumor，reference 只有 stroma，生成像 stroma。

排查：

- prior fallback bias 是否启用；
- missing class detection 是否正确；
- prior tokens 是否有梯度；
- partial/low coverage 训练比例是否足够。

### 17.4 prior 抢走 reference

表现：

- reference 有同类，但模型仍生成典型平均质感，不像 reference。

排查：

- `lambda_prior_present` 是否太高；
- `lambda_same_coarse/fine` 是否太低；
- prior token 数是否太多；
- full coverage 训练是否不足；
- ref-swap loss 是否太弱。

### 17.5 cell mask 让结果变脏

表现：

- 纹理噪声增加；
- 局部细胞样点过密或乱；
- tissue 对了但 cell pattern 不稳定。

排查：

- cell mask 是否对齐；
- cell IDs 是否 remap 正确；
- cell bias 是否太强；
- density penalty 是否太大；
- CellViT mask 是否有系统性错误。

---

## 18. 最小可行版本

如果要最快验证核心想法，MVP 可以是：

1. 保留 Cross V3 的 ControlNet 和 reference token encoder。
2. 新增 target/ref coarse tissue token metadata。
3. 新增 coarse tissue prior tokens，每类 4 个。
4. 在 FLUX 1 个中后层 double-block joint attention logits 加 tissue-only bias。
5. patch size 先用 256，bias dtype 用 bf16/fp16，只对 reference_local + tissue_prior segment 加 bias。
6. `route_anchor_mode=none`，不加 cell bias，不加 global style，不加 attention regularizer。
7. 用 full/partial/low coverage 平衡训练，并混入布局差异大的 same-case reference-target 对。
8. 做 normal/zero/random ref ablation，以及 cross-case/cross-WSI qualitative reference ablation。

MVP 成功标准：

- full coverage 下，同类 reference 纹理迁移比 Cross V3 明显。
- zero/random reference 明显变差。
- partial coverage 下，covered class 更像 reference，missing class 不错误贴 mismatch reference。
- attention stats 显示 same-class ref mass 高，missing class prior mass 高。
- 256/单层 materialized attention 能稳定训练，显存和 step time 可接受。
- cross-case/cross-WSI reference 不把 reference 布局复制到 target。
- 更换 reference 时 covered class 外观有方向性变化。

第二阶段再加 cell/nuclei：

1. cell histogram/density metadata。
2. cell similarity bias。
3. cell prior tokens。
4. cell-aware metrics。

---

## 19. 对你现在“缺什么”的总清单

按“能不能继续推进模型”的优先级：

1. **缺 attention bias 注入能力。** 这是最大缺口。
2. **缺 target/ref token 级 semantic metadata。** 没有它就建不了 bias。
3. **缺 per-class prior token bank。** reference 缺类时没有可靠兜底。
4. **缺 cell/nuclei token histogram 和 density。** 你的 pipeline 会生成细胞 mask，但还没真正用于 correspondence。
5. **缺 coverage-aware batch sampler。** 当前有 metadata，但训练 loop 还没强约束比例。
6. **缺局部 ref-swap / attention regularizer。** 现在损失还偏全图。
7. **缺 attention diagnostics。** 目前只能知道 ref 有无影响，不能知道听了哪个类。
8. **缺 Cross V4 checkpoint/inference spec。** 训练和推理必须保存同一套 bias/prior 配置。
9. **缺 pipeline 生成 cell mask 的质量审计。** cell mask 若有噪声，bias 会放大问题。
10. **缺系统 ablation。** 没有 ablation 很难判断是 reference token、bias、prior、cell mask 哪个起作用。

---

## 20. 最终架构判断

你的直觉是对的：

```text
ref 图里有什么，target 对应语义区域就应该去 ref 的同类区域找；
ref 图里没有的，target 就应该靠模型先验补；
细胞 mask 不只是辅助，它应该参与微环境级别的对齐。
```

所以最终架构应该是：

```text
Target masks -> ControlNet -> 定结构
Reference image + masks -> reference tokens -> 提供真实局部外观
Target/ref masks -> correspondence bias -> 让 target 找对 reference 来源
Per-class prior tokens -> reference 缺类时兜底
Cell/nuclei masks -> cell-aware bias/prior -> 控制细胞级微环境
```

这比“全局 style vector”更符合你的目标，也比当前 Cross V3 更完整。

一句话收束：

**Cross V4 的核心不是再加一个更大的 reference encoder，而是让 attention 知道 target 每个位置应该从 reference 的哪一类区域取信息，以及 reference 没有这类区域时该去哪个 prior token 兜底。**
