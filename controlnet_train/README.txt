ControlNet Train
================

`controlnet_train` 已按 `.claude/plans/plan.md` 的 Phase 5 目标做了第一轮整理。

当前分层
--------

- `modules/`
  Phase 5 的条件编码模块层。
  - `hte_embedding.py`: 把 `tissue_mask` 的 `int` ID map 编码成 HTE 特征。
  - `tissue_condition_downsampler.py`: 把 full-resolution HTE 特征下采样到 FLUX latent resolution。
  - `nuclei_condition_encoder.py`: 把 `nuclei_mask` 的 raw ID map 编码成 learned nuclei condition feature。
  - `change_mask_encoder.py`: 把 binary `change_region_mask` 投影成轻量 4-channel learned feature。
  - `conditioning.py`: 放置 Phase 5 条件拼接辅助函数；当前已提供 `cross V0` 的单路 spatial concat helper。
- `data/`
  Phase 5 的新数据层。
  - `common.py`: 共享 layered patch 读取、prompt、nuclei remap、train/val split
  - `inpaint.py`: `local-preservation` / inpaint metadata builder + `InpaintDataset`
  - `cross.py`: `same-WSI cross-reconstruction` metadata builder + `CrossReconstructionDataset`
- `cli/`
  Phase 5 的脚本入口层。
  - `build_inpaint_dataset.py`: 把上游编辑样本清单归一化为 `metadata_inpaint_{train,val}.jsonl`。
  - `generate_training_pairs.py`: 从多数据集 layered patch 根目录生成 `metadata_cross_{train,val}.json`。
- `training/`
  Phase 5 的训练共享层。
  - `conditioning.py`: 记录 `5.3` 的 cond 通道规格与 `controlnet_x_embedder` 宽度补丁。
  - `flux_phase5.py`: 基于 `legacy_rgb_vae/` 中官方 FLUX ControlNet 训练流，接入新的 HTE / nuclei / change-mask 条件。
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

1. `cli/build_inpaint_dataset.py`
   扫描多数据集分层数据，直接读取 `tissue_mask.png`，生成训练清单和 prompt 元数据。
2. `cli/generate_training_pairs.py`
   依据多数据集 patch / WSI 分组逻辑，生成 cross-reconstruction pairs。
3. `cli/train_controlnet_flux_inpaint.py`
   用 `ref_image + ref_mask(HTE) + target_mask(HTE)` 训练新的 ControlNet。
4. `cli/train_controlnet_flux_cross.py`
   用 `reference_image + reference_tissue_mask + reference_nuclei_mask + target_*mask` 训练 `cross V0` baseline。

当前实现范围
------------

这轮代码当前已经覆盖到 `.claude/plans/plan.md` 的 `5.1`、`5.2` 和 `5.3` 第一版骨架：

- 已完成：
  - `HierarchicalTissueEmbedding`
  - `TissueConditionDownsampler`
  - `NucleiConditionEncoder`
  - `ChangeMaskEncoder`
  - `cross V0 spatial concat baseline` 的公共拼接 helper
  - `build_inpaint_condition` 的公共拼接 helper
  - `cli/train_controlnet_flux_inpaint.py`
  - `cli/train_controlnet_flux_cross.py`
  - `training/conditioning.py`
  - `training/flux_phase5.py`
- 暂未实现：
  - `ReferenceMorphologyEncoder`
  - `cross V1 reference branch`
  - `Phase 5` 验证 / 推理脚本

也就是说，当前 `cross controlnet` 先按计划里的 `V0` 收敛：把
`reference_image_latent + reference_tissue_feat + reference_nuclei_feat + target_tissue_feat + target_nuclei_feat`
拼成单一路 `controlnet_cond`。

数据准备
--------

Phase 5 现在明确分成两类 DataLoader：

1. `InpaintDataset`
   服务 `local-preservation / inpaint` 训练。
2. `CrossReconstructionDataset`
   服务 `same-WSI cross-reconstruction` 训练。

两类任务的 metadata schema 分开，但共享同一层底层读取逻辑：

- layered patch 根目录统一约定为：
  - `images/`
  - `tissue_masks/`
  - `nuclei_masks/`
  - `metadata.jsonl`
  - `stats.txt`
- `tissue_mask` 始终是 unified fine tissue ID，值域 `[0, 15]`
- `nuclei_mask` 始终是 raw nuclei ID，值域 `0/101-105`
- 训练时再在 loader 内做 nuclei remap：`0/101-105 -> 0..5`

多数据集原始 patch 根目录
------------------------

当前仓库外部的实际 layered patch 根目录示例：

- `D:\WQX\datasets\BCSS\BCSS_PATCHES`
- `D:\WQX\datasets\PANDA\PANDA_PATCHES`
- `D:\WQX\datasets\GlaS\GlaS_PATCHES`
- `D:\WQX\datasets\IGNITE_PATCHES`
- `D:\WQX\datasets\ORCA\ORCA_PATCHES`
- `D:\WQX\datasets\PUMA\PUMA_PATCHES`

这些目录是 `cross` metadata 构建的直接输入；脚本会从 `metadata.jsonl` 和同名的
`images/tissue_masks/nuclei_masks` 中读取样本。

Inpaint Metadata 约定
---------------------

`cli/build_inpaint_dataset.py` 不再负责 BCSS-only 的旧式 RGB mask 造数，而是负责把上游编辑结果
归一化成统一 schema。输入 `jsonl` 中每条记录至少要有：

- `dataset`
- `source_image`
- `target_image`
- `target_tissue_mask`
- `target_nuclei_mask`
- `change_region_mask`

可选字段：

- `erased_source_image`
- `prompt`
- `edit_type`
- `change_ratio`
- `sample_id`
- `case_id`

输出：

- `metadata_inpaint_train.jsonl`
- `metadata_inpaint_val.jsonl`

如果输入里没有 `erased_source_image`，脚本会根据 `source_image + change_region_mask`
自动生成灰色擦除版本。

Cross Metadata 约定
-------------------

`cli/generate_training_pairs.py` 专门负责 `same-WSI cross-reconstruction`。
它会从多个 layered patch 根目录中读取样本，并输出：

- `metadata_cross_train.json`
- `metadata_cross_val.json`

每个 pair 至少包含：

- `target_image`
- `target_tissue_mask`
- `target_nuclei_mask`
- `reference_image`
- `reference_tissue_mask`
- `reference_nuclei_mask`
- `dataset`
- `sample_id`
- `reference_sample_id`
- `case_id`

当前配对规则：

- 只在同一 `case_id` / WSI 内配对
- `reference` 必须覆盖 `target` 的 tissue 语义集合
- nuclei 分布和 stain 只作为软排序信号

GT synthesis mode
------------------

`cli/build_inpaint_dataset.py` 也可以直接从 layered patch 根目录生成 synthetic GT inpaint metadata：

```bash
python controlnet_train/cli/build_inpaint_dataset.py ^
  --dataset-root PANDA=D:\\WQX\\datasets\\PANDA\\PANDA_PATCHES ^
  --dataset-root BCSS=D:\\WQX\\datasets\\BCSS\\BCSS_PATCHES ^
  --forced-mode replace_like_blob ^
  --output-dir phase5_runs\\inpaint_meta
```

Supported synthesized `mask_mode` values:

- `identity`
- `near_identity`
- `expand_band`
- `shrink_band`
- `replace_like_blob`

Each output row traces the edit with:

- `mask_mode`
- `size_bucket`
- `change_ratio`

推荐命令
--------

inpaint metadata 归一化：

```bash
python controlnet_train/cli/build_inpaint_dataset.py ^
  --input-jsonl path\\to\\edited_samples.jsonl ^
  --output-dir phase5_runs\\inpaint_meta
```

cross metadata 构建：

```bash
python controlnet_train/cli/generate_training_pairs.py ^
  --dataset-root BCSS=D:\\WQX\\datasets\\BCSS\\BCSS_PATCHES ^
  --dataset-root PANDA=D:\\WQX\\datasets\\PANDA\\PANDA_PATCHES ^
  --dataset-root GlaS=D:\\WQX\\datasets\\GlaS\\GlaS_PATCHES ^
  --output-dir phase5_runs\\cross_meta
```

inpaint 训练：

```bash
python controlnet_train/cli/train_controlnet_flux_inpaint.py ^
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev ^
  --train-metadata phase5_runs\\inpaint_meta\\metadata_inpaint_train.jsonl ^
  --output-dir phase5_runs\\controlnet_inpaint
```

cross V0 训练：

```bash
python controlnet_train/cli/train_controlnet_flux_cross.py ^
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev ^
  --train-metadata phase5_runs\\cross_meta\\metadata_cross_train.json ^
  --output-dir phase5_runs\\controlnet_cross
```

Phase 5.3 架构说明
-----------------

- `5.3` 的训练流保留了 `legacy_rgb_vae/` 里基于官方 `diffusers` `FluxControlNetModel` 的训练主干。
- 当前改动集中在两部分：
  - `controlnet_cond` 不再依赖 mask VAE latent，而是改为 HTE / nuclei / change-mask learned features
  - `controlnet_x_embedder` 的输入宽度按新的 packed cond 通道数显式扩展
- 结合当前多数据集 patch 规模：
  - BCSS: `22870`
  - PANDA: `29300`
  - ORCA: `29606`
  - IGNITE: `9097`
  - GlaS: `2185`
  - PUMA: `1844`
  - 合计约 `94902` patches
- 这个数据规模足以先支撑当前 `5.3` 计划里的默认容量；第一版不建议额外增大 ControlNet 层数或引入更重的 morphology branch。

迁移原则
--------

- 只把 `tissue_mask` 作为 ControlNet conditioning。
- 不再为 mask 生成 RGB PNG 或 VAE latent。
- 所有标签解释都从 `dataset_config/` 读取。
- HTE 只负责离散 tissue ID 的语义编码，不负责图像重建。
- `5.2` 的 DataLoader 分为 `inpaint` 和 `cross` 两类，不再混用单一 schema。
Phase 5.4 Unified Inference
---------------------------

Phase 5.4 adds a dedicated `inference/` package so inference logic no longer
lives inside `training/` or ad-hoc validation scripts.

Current Phase 5.4 layout:

- `controlnet_train/inference/router.py`
  - computes `change_region_mask = (reference_tissue_mask != target_tissue_mask)`
  - computes `change_ratio`
  - selects `inpaint` or `cross`
- `controlnet_train/inference/pipeline.py`
  - validates the five required spatial inputs
  - resolves the FLUX prompt
  - dispatches to the selected inference runner
  - saves `final.png`, `change_region_mask.png`, and `run_summary.json`
- `controlnet_train/cli/edit_pipeline.py`
  - unified CLI entrypoint for Phase 5.4 editing

Phase 5.4 required spatial inputs:

- `reference_image`
- `reference_tissue_mask`
- `reference_nuclei_mask`
- `target_tissue_mask`
- `target_nuclei_mask`

Prompt resolution order:

1. `--prompt`
2. `--dataset` -> `default_prompt_for_dataset(...)`
3. fallback prompt: `H&E stained cancer histopathology at 40x magnification`

Default routing thresholds:

- `change_ratio <= 0.12` -> `inpaint`
- `change_ratio >= 0.30` -> `cross`
- middle band -> `inpaint`

First-version scope limits:

- no blending
- no cross V1 morphology branch
- no internal reference retrieval

Recommended command:

```bash
python controlnet_train/cli/edit_pipeline.py ^
  --reference-image path\to\reference.png ^
  --reference-tissue-mask path\to\reference_tissue.png ^
  --reference-nuclei-mask path\to\reference_nuclei.png ^
  --target-tissue-mask path\to\target_tissue.png ^
  --target-nuclei-mask path\to\target_nuclei.png ^
  --pretrained-model-name-or-path black-forest-labs/FLUX.1-dev ^
  --inpaint-checkpoint phase5_runs\controlnet_inpaint ^
  --cross-checkpoint phase5_runs\controlnet_cross ^
  --output-dir phase5_runs\edit_outputs ^
  --dataset BCSS
```
