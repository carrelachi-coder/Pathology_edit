ControlNet Train
================

`controlnet_train` 已按 `.claude/plans/plan.md` 的 Phase 5 目标做了第一轮整理。

当前分层
--------

- `hte_embedding.py`
  Phase 5 的核心新模块。负责把 `tissue_mask` 的 `int` ID map 编码成 HTE 特征。
- `tissue_condition_downsampler.py`
  把 full-resolution HTE 特征下采样到 FLUX latent resolution。
- `nuclei_condition_encoder.py`
  把 `nuclei_mask` 的 raw ID map 编码成 learned nuclei condition feature。
- `change_mask_encoder.py`
  把 binary `change_region_mask` 投影成轻量 4-channel learned feature。
- `conditioning.py`
  放置 Phase 5 条件拼接辅助函数；当前已提供 `cross V0` 的单路 spatial concat helper。
- `legacy_rgb_vae/`
  归档旧版 BCSS-only / RGB mask / VAE mask latent 流程，避免和 Phase 5 新方案混用。

为什么要这样整理
----------------

Phase 5 的目标和旧脚本差异很大：

- 旧流程：`mask -> RGB -> VAE latent`
- 新流程：`mask(int ID map) -> HTE embedding`
- 旧流程：主要围绕 BCSS 单数据集
- 新流程：面向 BCSS / PANDA / GlaS / IGNITE / PUMA / ORCA 多数据集
- 旧流程：脚本职责混在一起，且默认使用硬编码路径
- 新流程：应围绕 `dataset_config/`、分层 mask、统一 fine label 空间组织

所以这里先把“旧实现”整体归档，让 Phase 5 根目录只放真正的新方案文件。

legacy_rgb_vae 内容
-------------------

以下文件属于旧方案，现统一放到 `legacy_rgb_vae/`：

- `build_inpaint_dataset.py`
- `generate_training_pairs.py`
- `precompute_vae_latents.py`
- `run_precompute.sh`
- `train.sh`
- `training_pairs.json`
- `train_controlnet_flux.py`
- `train_controlnet_flux_inpaint.py`
- `val_controlnet_flux.py`

这些文件保留是为了方便对照和迁移，但不再代表 Phase 5 的目标实现。

Phase 5 推荐后续落位
---------------------

建议后续把新脚本按下面的职责继续补齐：

1. `build_inpaint_dataset.py`
   扫描多数据集分层数据，直接读取 `tissue_mask.png`，生成训练清单和 prompt 元数据。
2. `generate_training_pairs.py`
   依据多数据集 patch / WSI 分组逻辑，生成 cross-reconstruction pairs。
3. `train_controlnet_flux_inpaint.py`
   用 `ref_image + ref_mask(HTE) + target_mask(HTE)` 训练新的 ControlNet。
4. `val_controlnet_flux.py`
   验证新 HTE conditioning 的推理链路。

当前实现范围
------------

这轮代码先对齐 `.claude/plans/plan.md` 的 `5.1` 基础设施部分：

- 已完成：
  - `HierarchicalTissueEmbedding`
  - `TissueConditionDownsampler`
  - `NucleiConditionEncoder`
  - `ChangeMaskEncoder`
  - `cross V0 spatial concat baseline` 的公共拼接 helper
- 暂未实现：
  - `ReferenceMorphologyEncoder`
  - `cross V1 reference branch`

也就是说，当前 `cross controlnet` 先按计划里的 `V0` 收敛：把
`reference_image_latent + reference_tissue_feat + reference_nuclei_feat + target_tissue_feat + target_nuclei_feat`
拼成单一路 `controlnet_cond`。

迁移原则
--------

- 只把 `tissue_mask` 作为 ControlNet conditioning。
- 不再为 mask 生成 RGB PNG 或 VAE latent。
- 所有标签解释都从 `dataset_config/` 读取。
- HTE 只负责离散 tissue ID 的语义编码，不负责图像重建。
