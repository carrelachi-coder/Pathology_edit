# Cross V6-min-0b Gamma-Attention Latent Composer 架构设计

Date: 2026-06-06

## 0. 结论

V6-min-0b 的因果方向保持不变：

```text
target masks 只在 composer 内部做 routing
final condition 只暴露 reference-derived latent + class-agnostic geometry
final condition 不给 target class one-hot
```

但要把设计拆成两个独立层来看：

```text
Layer A: appearance 用什么机制从 ref_latent 取
  masked mean pooling vs same-class attention
  决定能否验证 reference 空间 appearance 迁移

Layer B: nuclei 怎么 merge 进 tissue
  直接加 vs mask-gated residual
  决定核区会不会发脏 / 双倍叠加
```

新的推荐首版是：

```text
ref_image -> frozen VAE -> ref_latent

tissue:
  z_tissue_pool = masked pooling baseline
  z_tissue_attn = same-class attention composer
  z_tissue_target =
    z_tissue_pool
    + gamma * (z_tissue_attn - stopgrad(z_tissue_pool))

nuclei:
  z_nuclei_target = frozen VAE nuclei mask pooling

merge:
  z_ref_to_target =
    z_tissue_target
    + alpha * nuclei_mask_lat
      * (z_nuclei_target - stopgrad(z_tissue_target))

final condition:
  z_ref_to_target
  + nuclei_binary
  + nuclei_boundary
  + nuclei_distance_map
  + tissue_boundary
  + retrieval_confidence_map
  + missing_class_map
```

一句话：

**采纳 nuclei residual merge；同时把 same-class attention 作为 gamma 初始为 0 的 residual 分支挂上首版，否则 mean-pooling baseline 会给出一个注定模糊的 ref-insensitive 负结果。**

## 0.1 保留的 V5 教训

V5 的根本问题是：

```text
target class one-hot -> ControlNet
模型学成 class ID -> average appearance
```

V6-min-0b 保留以下硬约束：

- target tissue/nuclei masks 只在 composer 内部用于 routing。
- final condition 不给 `target_tissue_onehot` / `target_nuclei_type_onehot`。
- appearance value 必须来自 `ref_latent` 中被 reference mask 选中的 token。
- final condition 保留 class-agnostic geometry。
- VAE frozen。
- 不训练 nuclei CNN，不用 random frozen CNN。
- `z_ref_to_target` 只作为 condition，不替代 diffusion noisy latent。

## 0.2 必须修正的核心点

Masked mean pooling：

```text
T_ref_tissue[class] = masked_pool(ref_latent, ref_tissue_mask_lat[class])
z_tissue_pool = sum_class T_ref_tissue[class] * target_tissue_mask_lat[class]
```

会把每个 class 的 reference appearance 压成 per-channel 标量，再铺回 target layout。结果是：

```text
每个 class 区域内部是空间常数
reference 内部纹理、排列、染色梯度被抹掉
```

它只能回答：

```text
全局色调 / 低频 style 是否能进入 condition
routing pipeline 是否接通
```

不能回答：

```text
reference 空间 appearance 能不能迁移
```

因此首版不能只跑 mean pooling。必须保留 pooling 短训版作低频 reference causality 下界，同时在同周期加入 same-class attention residual。

在训练 attention 之前，还要先做一个零成本前置诊断：确认 `ref_latent` 在同类区域内部是否真的有可检索的空间差异。如果 VAE latent 的同类 token 本身几乎常数，attention 再精巧也会退化回 pooling。

## 1. 总体数据流

```text
Reference:
  ref_image
    |
    v
  frozen VAE encoder
    |
    v
  ref_latent

  ref_tissue_mask
  ref_nuclei_mask
    |
    v
  resize_soft to latent resolution


Layer A / tissue appearance:
  z_tissue_pool =
    TissueMaskedPooling(ref_latent, ref_tissue_mask, target_tissue_mask)

  z_tissue_attn =
    TissueSameClassAttention(
      Q = TargetLayoutEncoder(class-agnostic geometry),
      K/V = ref_latent tokens inside same-class ref tissue mask,
      routing = target_tissue_mask,
    )

  z_tissue_target =
    z_tissue_pool
    + gamma * (z_tissue_attn - stopgrad(z_tissue_pool))


Layer B / nuclei low-frequency residual:
  z_nuclei_target =
    NucleiMaskedPooling(ref_latent, ref_nuclei_mask, target_nuclei_mask)

  z_ref_to_target =
    z_tissue_target
    + alpha * target_nuclei_mask_lat
      * (z_nuclei_target - stopgrad(z_tissue_target))


Final condition:
  normalize(z_ref_to_target)
  + nuclei_binary
  + nuclei_boundary
  + nuclei_distance_map
  + tissue_boundary
  + retrieval_confidence_map
  + missing_class_map
    |
    v
  SingleConditionEncoder
    |
    v
  ControlNet residual adapters
```

## 2. Layer A: Tissue Appearance Retrieval

### 2.1 Pooling 下界

Pooling composer 是归因下界：

```text
T_ref_tissue[class] =
  masked_pool(ref_latent, ref_tissue_mask_lat[class])

z_tissue_pool =
  sum_class T_ref_tissue[class][:, :, None, None]
            * target_tissue_mask_lat[class]
```

作用：

```text
验证 mask resize / pooling / scatter / ControlNet condition 线路是否接通
提供稳定低频 condition
作为 attention 版的下界对照
```

限制：

```text
每类区域内部空间常数
无法保留同类区域内部纹理
无法判断 spatial appearance migration 是否成立
```

所以：

```text
pooling-only ref-insensitive != reference path 失败
```

### 2.2 Same-class Attention Residual

Attention 不替代 pooling，而是 residual：

```text
z_tissue_target =
  z_tissue_pool
  + gamma * (z_tissue_attn - stopgrad(z_tissue_pool))
```

`gamma` 初始化：

```text
gamma = 1e-3 - 1e-2
```

好处：

```text
1. 训练初期 ControlNet 先拿稳定 pooling condition。
2. attention 输出分布不会一开始乱跳。
3. attention 子网络从第一步就有梯度流。
4. gamma 本身是 attention 是否贡献 reference 信息的诊断。
```

不要把 gamma 严格初始化为 0。否则 `d(loss)/d(attn_params)` 会被 gamma 乘成 0，attention 子网络在第一步拿不到学习信号，只能等 gamma 自己先爬起来。

诊断也不要看“gamma 是否接近 0”，而要看：

```text
gamma 是否显著偏离初值并稳定增长
attention residual norm 是否随训练变成非零
```

`z_tissue_attn` 必须是 target-layout 的空间特征图：

```text
z_tissue_attn: [B, C_lat, H_lat, W_lat]
```

Attention 必须使用 **per-position query**，不是 per-class query：

```text
target latent 上属于 class c 的每个位置 (x, y)
  都有自己的 Q(x, y)
  去 attend ref class-c token bank
  输出 O(x, y)

O scatter 回 target layout
```

如果每个 class 只生成一个 query，`z_tissue_attn` 会退化成 per-class 常数，和 pooling 没有本质区别。

### 2.3 Q/K/V 约束

Q 端绝对不能泄露 class one-hot：

```text
Q_map = TargetLayoutEncoder(class-agnostic geometry)
Q = Linear_Q(Q_map target positions)
```

允许输入：

```text
nuclei_binary
nuclei_boundary
nuclei_distance_map
tissue_boundary
optional coordinate embedding
optional density / centroid geometry
```

`coord_embedding` 推荐开启，但只能作为 Q 的几何锚点，不能成为 appearance shortcut。约束：

```text
appearance value 只能来自 K/V = ref_latent same-class tokens
coord 不能参与 K/V
pseudo-self 中 ref 与 target 要有独立 crop/flip/jitter，打破绝对坐标抄答案
```

如果使用绝对坐标且 pseudo-self ref/target 坐标完全对齐，模型可能学到：

```text
target coordinate -> fixed appearance
```

从而绕过 reference K/V 检索。

禁止输入：

```text
target_tissue_onehot
target_nuclei_type_onehot
learned class embedding as Q
```

class 信息只能用于 routing：

```text
target class c positions attend only ref bank c
```

K/V 必须来自 reference：

```text
ref_tokens_c = ref_latent tokens inside ref_tissue_mask_lat[class]
K = Linear_K(ref_tokens_c)
V = Linear_V(ref_tokens_c)
```

不能用：

```text
K/V = learned class embedding
```

投影维度必须对齐：

```text
Q: [B, N_target_c, d_k]
K: [B, N_ref_c, d_k]
V: [B, N_ref_c, d_v]
O: [B, N_target_c, d_v] -> Linear_O -> [B, N_target_c, C_lat]
```

由于每个 batch / class 的 `N_ref_c` 不同，K/V bank 必须使用：

```text
padding + attention_mask
```

或等价的 per-class ragged implementation。padding 位置必须在 softmax 前 mask 掉：

```text
attention_logits[padded_ref_tokens] = -inf
```

否则 attention 概率会漏到 pad token。

为了控制显存，必须限制 ref bank token 数：

```text
max_ref_tokens_per_class: K
```

采样方式比 K 值更重要：

```text
sampling = uniform_spatial_or_random
```

不要按空间顺序取前 K 个 token，否则会系统性丢掉 reference 同类区域的 appearance 多样性。attention 相对 pooling 的价值正是利用类内 appearance 多样性。

`gamma` 推荐为 per-channel scalar，类似 LayerScale：

```text
gamma: [1, C_lat, 1, 1]
init = 1e-3 - 1e-2
z_tissue_target =
  z_tissue_pool
  + gamma * (z_tissue_attn - stopgrad(z_tissue_pool))
```

不推荐：

```text
single global scalar gamma: 诊断太粗
per-class gamma: 小类容易学崩，解释也更复杂
```

诊断记录：

```text
||gamma - gamma_init||
mean(abs(gamma - gamma_init))
per-channel gamma histogram
```

### 2.4 Empty Bank Fallback

如果某 class 在 reference 中缺失：

```text
ref_tissue_mask_lat[class].sum() < min_pixels
```

则：

```text
attention bank is empty
fallback to z_tissue_pool / global fallback
missing_class_map[class region] = 1
retrieval_confidence_map[class region] = 0
```

不能让 empty bank 进入 softmax，否则可能 NaN。

### 2.5 Pooling Smoke + Attention 同周期

顺序：

```text
1. pooling smoke / 短训:
   确认 frozen VAE、mask resize、pool/scatter、condition injection 接通。

2. gamma-attention train:
   同周期验证 same-class attention 是否带来额外 reference 信息。
```

如果 attention 版也 ref-insensitive，再排查：

```text
VAE latent 是否对 stain/texture 敏感
pair sampler 是否让 ref A/B 太像
ControlNet 是否忽略 condition
loss 是否没有给 appearance pressure
```

### 2.6 VAE 类内方差前置诊断

在正式训练 attention 前，必须先测 `ref_latent` 同类 token 内部是否有可检索差异：

```text
for each ref patch, each tissue class:
  tokens = ref_latent[ref_tissue_mask_lat[class]]
  token_variance = mean(var(tokens, dim=token))
  pca_energy = explained_variance(tokens)
  pairwise_distance = mean_pairwise_distance(tokens)
```

判读：

```text
类内方差高:
  attention 有东西可取，Layer A 赌注成立。

类内方差极低:
  attention 会退化回 pooling。
  问题在 frozen VAE latent 缺少 class-internal spatial appearance，
  不应归因于 composer 或 ControlNet。
```

这个诊断不需要训练。它和 pooling smoke 共同决定 Layer A 的上限：

```text
pooling smoke:
  routing / condition injection 是否接通

VAE class-internal variance:
  ref bank 内部是否有 differentiated appearance 可检索
```

## 3. Layer B: Nuclei Merge

### 3.1 Nuclei 不上 attention

Nuclei path 首版不做 attention：

```text
nuclei composer = frozen VAE latent + nuclei mask pooling
```

原因：

```text
VAE 8x 下采样后单个 nucleus 可能只有 1-2 latent cells
nuclei K/V bank 空间区分度很低
attention 大概率退化回 pooling
```

核级 high-frequency appearance 留给后续 V6-min-2 high-res composer。

### 3.2 Nuclei Pooling

如果只有 nuclei foreground：

```text
T_ref_nuclei =
  masked_pool(ref_latent, ref_nuclei_mask_lat)

z_nuclei_target =
  T_ref_nuclei[:, :, None, None] * target_nuclei_mask_lat
```

如果有 nuclei type：

```text
T_ref_nuclei[type] =
  masked_pool(ref_latent, ref_nuclei_type_mask_lat[type])

z_nuclei_target =
  sum_type T_ref_nuclei[type][:, :, None, None]
           * target_nuclei_type_mask_lat[type]
```

如果只有 instance ID：

```text
instance ID 不作为语义 embedding
collapse instances -> nuclei foreground mask
```

### 3.3 Residual Merge

不要直接：

```text
z_ref_to_target =
  z_tissue_target
  + alpha * nuclei_mask * z_nuclei_target
```

推荐：

```text
z_nuclei_residual =
  z_nuclei_target - stopgrad(z_tissue_target)

z_ref_to_target =
  z_tissue_target
  + alpha * nuclei_mask_lat * z_nuclei_residual
```

理由不是 receptive field，而是：

```text
z_tissue_target 和 z_nuclei_target 都从同一 ref_latent mean-pool 得到
两者共享 reference 整体染色 DC 分量
直接相加会把 stain base 加两次
residual 去掉共享 DC，只保留 nuclei 相对 tissue 的偏移
```

在纯 pooling 版里 `stopgrad` 基本是 no-op；一旦加入可训练 attention / gamma，`stopgrad` 可以防止 nuclei residual 反向扰动 tissue base。

### 3.4 Alpha 与 Coverage

默认：

```text
alpha = 0.25 或 0.3
```

Ablation：

```text
alpha in {0, 0.2, 0.4}
```

如果 `0.4` 让核区发脏或结构变差，回退。

Nuclei mask 下采样：

```text
area / bilinear
keep soft mask
do not use nearest hard downsample
record nuclei_mask_lat.sum()
```

Coverage gate：

```text
coverage_nuclei =
  sum(ref_nuclei_mask_lat) > min_pixels

if not coverage_nuclei:
  z_nuclei_target = 0
  nuclei_confidence = 0
  missing_class_map[nuclei region] = 1
```

不要从太小的 nuclei bank 硬算噪声 token。

## 4. Final Condition 与 Normalization

Final condition：

```text
z_ref_to_target
nuclei_binary
nuclei_boundary
nuclei_distance_map
tissue_boundary
retrieval_confidence_map
missing_class_map
```

不包含：

```text
target_tissue_onehot
target_nuclei_type_onehot
target_tissue_mask
target_nuclei_mask
```

进入 SingleConditionEncoder 前必须做 norm 对齐：

```text
z_ref_to_target_norm =
  normalize_to_ref_or_target_latent_stats(z_ref_to_target)
```

至少保证：

```text
mean / std 与 ref_latent 或 target_latent condition scale 同量级
```

必须监控：

```text
||z_ref_to_target||
std(z_ref_to_target)
||projected_z_ref_to_target||
||cond_feats[i]||
||ControlNet residual from condition||
controlnet_conditioning_scale = 0 vs 1
```

如果这些 norm 长期接近 0 或远小于 geometry condition，ControlNet 很可能忽略 reference-derived latent。

## 5. 输入与输出定义

```python
@dataclass
class CrossV6Min0bComposerCondition:
    ref_latent: Tensor
    ref_tissue_mask: Tensor
    ref_nuclei_mask: Tensor
    target_tissue_mask: Tensor
    target_nuclei_mask: Tensor
    nuclei_binary: Tensor
    nuclei_boundary: Tensor
    nuclei_distance_map: Tensor
    tissue_boundary: Tensor
```

```python
@dataclass
class CrossV6Min0bComposerOutput:
    z_tissue_pool: Tensor
    z_tissue_attn: Tensor
    gamma: Tensor
    z_tissue_target: Tensor
    z_nuclei_target: Tensor
    z_ref_to_target: Tensor
    retrieval_confidence_map: Tensor
    missing_class_map: Tensor
```

```python
@dataclass
class CrossV6Min0bControlCondition:
    z_ref_to_target: Tensor
    nuclei_binary: Tensor
    nuclei_boundary: Tensor
    nuclei_distance_map: Tensor
    tissue_boundary: Tensor
    retrieval_confidence_map: Tensor
    missing_class_map: Tensor
```

## 6. Loss 与 Coverage

主 loss 保持小集合：

```text
L_v6_min_0b =
  L_denoise
  + lambda_latent  * L_latent_self_recon_pseudo
  + lambda_color   * L_region_color
  + lambda_texture * L_region_texture_light
```

推荐初值：

```text
lambda_color   = 0.05
lambda_texture = 0.005 - 0.01
beta_nuclei    = 0.2
lambda_latent  = 0.01 - 0.03
```

`L_latent_self_recon_pseudo` 只用于 pseudo-self：

```text
target_latent = frozen_vae(target_image)
z_ref_to_target = composer(ref_latent, ref_masks, target_masks)

L_latent_self_recon_pseudo =
  distance(z_ref_to_target, stopgrad(target_latent))
```

额外记录：

```text
dist(z_tissue_target, target_latent)
dist(z_ref_to_target, target_latent)
```

理想：

```text
z_ref_to_target 比 z_tissue_target 更接近 target_latent
```

如果加 nuclei 后反而更差，优先排查：

```text
alpha 太大
mask 下采样错
nuclei token 噪声大
coverage gate 没起作用
```

`retrieval_coverage` 不作为主 loss，用于：

```text
pair filtering
confidence / missing maps
region color / texture loss mask
diagnostic metric
```

reference 缺失 class/type 时，不计算该 class/type 的 region color / texture loss。

## 7. Ablation 与诊断

### 7.1 Layer A: reference 空间 appearance 是否迁移

必须比较：

```text
P: pooling-only short train / gamma disabled
A: gamma-attention residual
```

观察：

```text
same target + ref A
same target + ref B
gamma 是否显著偏离初值并稳定增长
attention residual norm
VAE class-internal token variance
```

判读：

```text
P 只有低频 reference causality, A 有额外类内空间 reference causality:
  reference path 成立，pooling 是信息瓶颈。

P 和 A 都只有低频 ref response 或都 ref-insensitive:
  先看 VAE class-internal variance。
  若方差极低，问题是 VAE latent 没有可迁移空间信息。
  若方差足够，再排查 sampler、normalization、ControlNet residual、loss。
```

### 7.2 Layer B: nuclei 是否有贡献

比较：

```text
A: z_tissue_target + geometry
B: z_ref_to_target + geometry
C: zero ref nuclei memory + geometry
D: zero ref tissue memory + geometry
```

看：

```text
B 比 A 在核区是否更有 reference 染色感
zero nuclei 后核区是否变弱
zero tissue 后整体 tissue appearance 是否变弱
```

注意：

```text
zero ref nuclei memory 可能无明显差异
```

这可能只是 VAE latent 分辨率下 nuclei signal 弱，不一定是实现 bug。

### 7.3 Shortcut 回归判据

比较：

```text
B: z_ref_to_target + geometry
C: z_ref_to_target + geometry + target masks in final condition
```

如果 C 明显更强但 ref swap 不敏感，说明 class shortcut 从 final condition 回来了。

### 7.4 每 checkpoint 标准 sanity

必须每个 checkpoint 跑：

```text
controlnet_conditioning_scale = 0 vs 1
gamma - gamma_init norm / histogram
attention residual norm
||z_ref_to_target||
std(z_ref_to_target)
||projected_z_ref_to_target||
||cond_feats[i]||
||ControlNet residual from condition||
retrieval confidence / missing-class map visualization
```

训练前必须跑：

```text
VAE class-internal token variance / PCA energy / pairwise distance
```

## 8. 推荐默认配置

```text
vae:
  frozen: true

composer:
  tissue_appearance:
    pooling: enabled
    same_class_attention: enabled
    attention_query: per_position
    output_layout: target_layout_feature_map
    gamma:
      type: per_channel_layerscale
      shape: [1, C_lat, 1, 1]
      init: 1e-3 - 1e-2
    q_class_onehot: forbidden
    q_inputs:
      nuclei_binary: enabled
      nuclei_boundary: enabled
      nuclei_distance_map: enabled
      tissue_boundary: enabled
      coord_embedding: recommended
    coord_constraints:
      value_source: K/V_only
      break_pseudo_self_absolute_coord_shortcut: independent_ref_target_crop_flip_jitter
    q_projection: Linear_Q
    kv_source: ref_latent_same_class_tokens
    k_projection: Linear_K
    v_projection: Linear_V
    output_projection: Linear_O_to_C_lat
    variable_bank:
      implementation: padding_plus_attention_mask
      mask_padding_logits: -inf
      max_ref_tokens_per_class: required
      token_sampling: uniform_spatial_or_random
    empty_bank_fallback: pooling + missing_class_map

  nuclei_appearance:
    mode: frozen_vae_mask_pooling
    same_class_attention: disabled
    trainable_cnn: disabled
    random_frozen_cnn: forbidden
    mask_resize: area_or_bilinear_soft
    coverage_gate: zero_below_min_pixels

  merge:
    mode: mask_gated_residual
    formula: z_tissue + alpha * nuclei_mask_lat * (z_nuclei - stopgrad(z_tissue))
    alpha: 0.25 - 0.3
    alpha_ablation: [0, 0.2, 0.4]

final_condition:
  z_ref_to_target: enabled
  nuclei_binary: enabled
  nuclei_boundary: enabled
  nuclei_distance_map: enabled
  tissue_boundary: enabled
  retrieval_confidence_map: enabled
  missing_class_map: enabled
  target_tissue_onehot: disabled
  target_nuclei_type_onehot: disabled

normalization:
  z_ref_to_target: explicit
  target_stats: ref_or_target_latent_stats
  monitor_residual_norm: true

losses:
  denoise: enabled
  lambda_color: 0.05
  lambda_texture: 0.005 - 0.01
  beta_nuclei: 0.2
  lambda_latent: 0.01 - 0.03
  latent_self_recon_pseudo: pseudo_self_only
  retrieval_coverage_as_loss: false
  geometry_consistency_as_loss: false

diagnostics:
  vae_class_internal_variance_precheck: required
  vae_class_internal_pca_energy: required
  vae_class_internal_pairwise_distance: required
  dist_z_tissue_to_target_latent: enabled
  dist_z_ref_to_target_to_target_latent: enabled
  gamma_delta_from_init: enabled
  gamma_l2_norm: enabled
  gamma_histogram: enabled
  attention_residual_norm: enabled
  condition_scale_0_vs_1: every_checkpoint
```

## 9. 非目标

V6-min-0b 暂不做：

- target tissue/nuclei one-hot 直接进入 final ControlNet condition。
- Q 端使用 class embedding。
- learned class embedding 作为 K/V appearance value。
- nuclei same-class attention。
- trainable nuclei CNN。
- random frozen nuclei CNN。
- high-res nuclei detail composer。
- masked image -> latent 作为主 reference memory 路径。
- `z_ref_to_target` 直接替代 diffusion noisy latent。
- warped_ref 主路径。
- delayed reference enable。

## 10. 最终定义

Cross V6-min-0b 的模型定义是：

```text
Cross V6-min-0b
  = frozen VAE(ref_image) -> ref_latent
  + z_tissue_pool
  + gamma * same_class_attention_residual
  + mask-gated nuclei residual
  + normalized z_ref_to_target
  + class-agnostic target geometry
  + retrieval_confidence_map / missing_class_map
  + SingleConditionEncoder
  + ControlNet residual adapters
```

它要验证的核心假设是：

```text
如果 target tissue/nuclei masks 只在 composer 内部用于 frozen VAE reference latent lookup，
Q 不泄露 target class one-hot，
same-class attention residual 能在 pooling 下界之上提供 reference spatial appearance，
并且 nuclei 只作为小权重 residual，
那么模型能建立 reference-specific appearance path，
同时避免 V5 的 class shortcut 和 trainable nuclei CNN 的不稳定。
```

一句话收束：

**Layer B 的 nuclei residual merge 要采纳；真正决定 V6 成败的是 Layer A，所以首版必须用小正值 gamma 的 same-class attention residual 在同周期验证 reference 空间 appearance，而不能只依赖 masked mean pooling。**
