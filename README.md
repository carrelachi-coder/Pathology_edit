# Pathology Edit：产品代码

本仓库包含病理图像编辑产品的完整可执行链路。论文、实验输出、数据集、
checkpoint 和机器本地配置不进入 Git 历史。

| 产品模块 | 主要代码 | 当前定位 |
| --- | --- | --- |
| 语义 mask 编辑 | `phase3_mask_edit/` | LLM contour + `organic_v2` 生产入口；旧非 LLM 确定性执行器已退役 |
| 组织分割 | `segmentator/` | C 线 Boundary + CellViT Teacher joint epoch 2；严格 release 驱动推理 |
| 细胞布局（CellDistNet） | `inpaint_cells/` | epoch29 只提供落点先验；patch 自适应配额，当前 patch 同类 shape 优先 |
| 图像生成 | `controlnet_train/`、`scripts/` | Inpaint 与 Cross V1 + Pix2pix 两条生产路由 |
| 基准与评估 | `phase3_mask_edit/benchmark/`、`benchmark_configs/` | mask 语义、条件一致性、Patho-KID 与表示分析 |

线上产品的唯一冻结清单见 `docs/online_product_release.md`；生成模型打包
细节见 `docs/generation_model_release.md`，C 线分割性能见
`docs/segmentator_fine_validation.md`。其他 benchmark/plan 文档不定义线上默认值。

## 生成模型生产链路

本仓库保留病理图像编辑中已经确定的四条生成链路：

| 模块 | 用途 | 生产入口 |
| --- | --- | --- |
| Inpaint ControlNet | 局部组织编辑与小范围修补 | `scripts/train_phase5_inpaint.sh`、`controlnet_train/cli/eval_controlnet_flux_inpaint.py` |
| Cross V1 no-IP | 根据目标 mask 生成 Stage 1 图像；生产推理不加载 UNI/IP-Adapter | `scripts/train_phase5_cross_v1.sh`、`scripts/generate_cross_v1_no_ip_strict.py` |
| Pix2pix full-pyramid | 将 Cross V1 输出迁移到参考图的局部纹理与方向，并保护低染色 Cross 结构 | `scripts/train_pix2pix_postprocess.sh`、`scripts/generate_cross_v1_no_ip_strict.py` |
| ProbNet | 在编辑区域预测细胞概率并结合实例库生成 nuclei mask | `scripts/phase4_probnet_workflow.sh`、`scripts/phase4_probnet_workflow_all.sh`、`inpaint_cells/generate.py` |

Cross V0、Cross V2/V3、旧 RGB-VAE、旧 pix2pix v2、旧 ControlNet/Inpaint 原型，以及 steered-texture/window-orientation/WSI-identity 的历史启动脚本不属于生产入口。

## 1. 环境与外部依赖

Inpaint、Cross V1 和 Pix2pix 使用 Phase 5 环境；ProbNet 使用独立的 Phase 4 环境：

```bash
conda env create -f envs/phase5_controlnet_inpaint.yaml
conda env create -f envs/phase4_probnet.yaml
```

以下资产不会随本仓库或下面的模型仓库重复发布：

- `black-forest-labs/FLUX.1-dev` 基础模型；
- Cross V1 训练阶段所需的 UNI 权重；
- 训练数据、optimizer/scheduler 状态；
- ProbNet 推理使用的、按数据集构建的 nuclei instance library。

模型仓库当前均为 private，需要先登录有权限的 Hugging Face 账号：

```bash
hf auth login
hf auth whoami
```

## 2. 生产 checkpoint

| 模型 | 发布来源 | 生产文件/目录 |
| --- | --- | --- |
| Inpaint ControlNet | [Qinxin11/pathology-inpaint-controlnet](https://huggingface.co/Qinxin11/pathology-inpaint-controlnet) | 仓库根目录；含 ControlNet、`config.json`、`phase5_conditioning.pt` |
| Cross V1 + Pix2pix | [Qinxin11/pathology-cross-v1-pix2pix](https://huggingface.co/Qinxin11/pathology-cross-v1-pix2pix) | `cross_v1/`；`pix2pix/pix2pix_epoch26_step214895.pt` |
| ProbNet / CellDistNet | [Qinxin11/pathology-probnet](https://huggingface.co/Qinxin11/pathology-probnet) | `best_epoch29_c29607f1b609accb.pt`；epoch 29 / step 33785；SHA256 `c29607f...571211` |
| Segmentator | 本地冻结 release | `/data1/zhao/wqx/segmentator_fine/legacy_anchor_fine_seed42/best_composite.pt`；release `segmentator-fine-legacy-anchor-v1`；SHA256 `5165e0f...27b3f` |

下载并注册为默认模型路径：

```bash
mkdir -p /models/pathology

hf download Qinxin11/pathology-inpaint-controlnet \
  --local-dir /models/pathology/pathology-inpaint-controlnet
hf download Qinxin11/pathology-cross-v1-pix2pix \
  --local-dir /models/pathology/pathology-cross-v1-pix2pix
hf download Qinxin11/pathology-probnet \
  best_epoch29_c29607f1b609accb.pt \
  --revision add6970449cf3a94997375a665c832e91b188251 \
  --local-dir /models/pathology/pathology-probnet
mkdir -p /models/pathology/pathology-segmentator
install -m 0644 \
  /data1/zhao/wqx/segmentator_fine/legacy_anchor_fine_seed42/best_composite.pt \
  /models/pathology/pathology-segmentator/legacy_anchor_fine_seed42_best_composite.pt

export PATHOLOGY_INPAINT_CHECKPOINT=/models/pathology/pathology-inpaint-controlnet
export PATHOLOGY_CROSS_V1_CHECKPOINT=/models/pathology/pathology-cross-v1-pix2pix/cross_v1
export PATHOLOGY_PIX2PIX_CHECKPOINT=/models/pathology/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt
export PATHOLOGY_PROBNET_CHECKPOINT=/models/pathology/pathology-probnet/best_epoch29_c29607f1b609accb.pt
export PATHOLOGY_SEGMENTATOR_CHECKPOINT=/models/pathology/pathology-segmentator/legacy_anchor_fine_seed42_best_composite.pt
export PATHOLOGY_SEGMENTATOR_PYTHON=/home/lyw/anaconda3/envs/pathology-segmentator-mmseg/bin/python3.10
export PATHOLOGY_CELLVIT_ROOT=/home/lyw/wqx-DL/flow-edit/FlowEdit-main/CellViT-plus-plus-main/CellViT-plus-plus-main
export PATHOLOGY_CELLVIT_MODEL=$PATHOLOGY_CELLVIT_ROOT/checkpoints/CellViT-SAM-H-x40-AMP-001.pth
export PATHOLOGY_CELLVIT_PYTHON=/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python
export FLUX_MODEL=/data/huggingface/FLUX.1-dev
```

代码默认指向 `amax2` 上的 inference-only 打包目录，也允许通过上述环境变量覆盖。Inpaint/Cross 会校验打包 manifest、release commit、权重大小和 SHA 记录；Pix2pix 会校验文件 SHA、epoch 26 / step 214895、full-pyramid steering、identity adapter 和 `nuclei_reference_support_v2`，并在 generation change region 内执行 `cross_rgb_od_low_stain_v1` 低染色结构保护；ProbNet 会强制 epoch29 SHA，先按分数贪心构造 quota-aware coverage prefix，再保留完整稳定降序 tail 处理放置失败；Segmentator 会通过 `segmentator_fine_legacy_anchor.json` 重建架构并严格加载最终 legacy-anchor checkpoint。历史 C-line 仅用于研究对照，不是生产 fallback。

权重 SHA256、文件大小、打包范围和 Hub 往返验证结果见 [生成模型发布说明](docs/generation_model_release.md)。

## 3. 训练

所有脚本都支持通过环境变量覆盖路径和训练参数。建议始终显式设置仓库、数据、输出和恢复 checkpoint，避免依赖服务器默认值。

### 3.1 Inpaint ControlNet

`train_phase5_inpaint.sh` 默认先构建六数据集 metadata，再启动多卡训练：

```bash
conda activate pathology-phase5-inpaint

PROJECT_ROOT="$PWD" \
MODEL_DIR="$FLUX_MODEL" \
BCSS_ROOT=/data/datasets/BCSS/BCSS_PATCHES \
PANDA_ROOT=/data/datasets/PANDA/PANDA_PATCHES \
GLAS_ROOT=/data/datasets/GlaS/GlaS_PATCHES \
IGNITE_ROOT=/data/datasets/IGNITE_PATCHES \
ORCA_ROOT=/data/datasets/ORCA/ORCA_PATCHES \
PUMA_ROOT=/data/datasets/PUMA/PUMA_PATCHES \
INPAINT_META_DIR=/data/runs/inpaint_meta \
INPAINT_OUTPUT_DIR=/data/runs/controlnet_inpaint \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
bash scripts/train_phase5_inpaint.sh
```

如果 metadata 已经存在，可设置 `RUN_DATASET_BUILD=0`；只构建数据而不训练时设置 `RUN_TRAIN=0`。

### 3.2 Cross V1

Cross V1 训练仍使用 UNI/ref conditioning；发布的生产推理导出则是 no-IP/no-UNI：

```bash
conda activate pathology-phase5-inpaint

PROJECT_ROOT="$PWD" \
MODEL_DIR="$FLUX_MODEL" \
UNI_CHECKPOINT=/models/UNI-2h/pytorch_model.bin \
CROSS_META=/data/runs/cross_meta/metadata_cross_train.json \
CONTROLNET_CHECKPOINT=/data/runs/controlnet_cross_v1/checkpoint-20000 \
CROSS_V1_OUTPUT_DIR=/data/runs/controlnet_cross_v1_expB \
GPU_IDS=0,1,2 \
MAX_TRAIN_STEPS=5000 \
bash scripts/train_phase5_cross_v1.sh
```

继续中断训练时设置 `RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N`。脚本默认只训练指定的 ControlNet output projection 与 conditioning/ref 分支，不应换回已删除的 Cross V0/V2/V3 启动脚本。

### 3.3 最终 Pix2pix full-pyramid

最终训练参数已经收敛到唯一入口 `train_pix2pix_postprocess.sh`：

```bash
conda activate pathology-phase5-inpaint

REPO_DIR="$PWD" \
PYTHON_BIN="$(command -v python)" \
TRAIN_METADATA=/data/runs/cross_meta/metadata_cross_train.json \
VAL_METADATA=/data/runs/cross_meta/metadata_cross_val.json \
I0_CACHE_DIR=/data/runs/pix2pix_i0_cache/train \
VAL_I0_CACHE_DIR=/data/runs/pix2pix_i0_cache/val \
RESUME=/data/runs/pix2pix_previous/ckpt/pilot_step002000.pt \
OUTPUT_DIR=/data/runs/pix2pix_full_pyramid \
GPU_IDS=0 \
BATCH_SIZE=2 \
MAX_CONTINUATION_STEPS=1000 \
bash scripts/train_pix2pix_postprocess.sh
```

完成后脚本会检查 `${OUTPUT_DIR}/ckpt/pilot_step001000.pt` 是否存在。不要在推理 CLI 手工覆盖架构、steering 或 nuclei trust 参数；这些信息必须随 checkpoint 保存并由 loader 校验。

### 3.4 ProbNet

单数据集完整 workflow（准备数据、构建实例库、训练、验证推理）：

```bash
conda activate pathology-phase4
bash scripts/phase4_probnet_workflow.sh \
  BCSS /data/edit_datasets/BCSS /data/runs/phase4/BCSS
```

六数据集联合训练：

```bash
conda activate pathology-phase4
bash scripts/phase4_probnet_workflow_all.sh \
  all /data/edit_datasets /data/runs/phase4/all6
```

`phase4_probnet_workflow_all.sh` 的第一个参数也可以单独使用 `prepare_dataset`、`build_library`、`train` 或 `generate`。实例库不会上传到模型仓库，必须用 `inpaint_cells/nuclei_library/build_library.py` 针对目标数据集构建并保存在本地。

## 4. 推理

### 4.1 Inpaint 固定样本/验证集

```bash
conda activate pathology-phase5-inpaint

python controlnet_train/cli/eval_controlnet_flux_inpaint.py \
  --pretrained-model-name-or-path "$FLUX_MODEL" \
  --checkpoint "$PATHOLOGY_INPAINT_CHECKPOINT" \
  --metadata /data/runs/inpaint_meta/metadata_inpaint_val.jsonl \
  --output-dir /data/outputs/inpaint_eval \
  --num-samples 16 \
  --device cuda \
  --torch-dtype bf16
```

### 4.2 Cross V1 no-IP/no-UNI + Pix2pix

推荐用 metadata 选固定样本。`--output` 是 Stage 1，`--pix2pix-output` 是最终结果：

```bash
conda activate pathology-phase5-inpaint

python scripts/generate_cross_v1_no_ip_strict.py \
  --checkpoint "$PATHOLOGY_CROSS_V1_CHECKPOINT" \
  --pretrained-model "$FLUX_MODEL" \
  --metadata /data/runs/cross_meta/metadata_cross_val.json \
  --metadata-index 0 \
  --output /data/outputs/sample_stage1.png \
  --pix2pix-checkpoint "$PATHOLOGY_PIX2PIX_CHECKPOINT" \
  --pix2pix-output /data/outputs/sample_final.png \
  --seed 42 \
  --device cuda
```

也可以不用 metadata，只显式提供目标 mask 运行 Stage 1：

```bash
python scripts/generate_cross_v1_no_ip_strict.py \
  --checkpoint "$PATHOLOGY_CROSS_V1_CHECKPOINT" \
  --pretrained-model "$FLUX_MODEL" \
  --target-tissue-mask /data/input/target_tissue.png \
  --target-nuclei-mask /data/input/target_nuclei.png \
  --output /data/outputs/stage1.png
```

严格入口不会构造 UNI，也不会加载 IP-Adapter。Pix2pix 还需要 metadata 中的参考 RGB 图与 mask，因此只有 metadata 模式可以接 Stage 2；若没有提供 `--pix2pix-output`，最终图默认写为 `<output stem>_pix2pix.png`。

### 4.3 ProbNet 单图 nuclei mask

输入需要 edited tissue mask、edit region，以及对应数据集的本地实例库；原有 nuclei mask 可选：

```bash
conda activate pathology-phase4

python inpaint_cells/generate.py \
  --dataset ORCA \
  --ckpt "$PATHOLOGY_PROBNET_CHECKPOINT" \
  --library /data/runs/phase4/all6/nuclei_library/ORCA \
  --input-tissue /data/input/edited_tissue.png \
  --reference-tissue /data/input/source_tissue.png \
  --input-nuclei /data/input/source_nuclei.png \
  --reference-nuclei-shapes /data/input/source_nuclei.png \
  --edit-region /data/input/edit_region.png \
  --output /data/outputs/generated_nuclei.png \
  --vis-dir /data/outputs/probnet_vis \
  --gamma-values 1.5 \
  --device cuda
```

生产 shape 策略是“当前 patch 同类别 reference shape 不放回优先，耗尽后
才用 library”。Library fallback 会按当前 patch 同类细胞的经验面积缩放；
默认线性缩放限制为 `0.5-2.0`。没有同类参考时保持 library 原尺寸并记录
未校准诊断。Count 和 subtype quota 仍由 patch 统计策略决定。每个
tissue/component 的 primary quota prefix 先按 ProbNet 分数贪心选择，并用
`0.75 * sqrt(area / quota)` 的通用半径防止窄高分带吞掉全部 quota，半径
上限为 48 px；未进入 prefix 的所有合法候选仍保持 ProbNet 分数稳定降序，
供放置失败时完整重试。该规则不增加 organ/dataset-specific 硬约束。

端到端 smoke test（从完整 layered sample 自动擦除一块区域后重建）：

```bash
python scripts/phase4_single_sample_smoke.py \
  --dataset ORCA \
  --input-tissue /data/input/tissue_mask.png \
  --input-nuclei /data/input/nuclei_mask.png \
  --ckpt "$PATHOLOGY_PROBNET_CHECKPOINT" \
  --library /data/runs/phase4/all6/nuclei_library/ORCA \
  --output-dir /data/outputs/probnet_smoke \
  --device cuda
```

### 4.4 端到端路由与 agentic workflow

在线产品入口是完整 UI；其第四阶段直接调用标准 agent runner：

```bash
python scripts/phase3_end_to_end_ui.py
```

`run_phase3_inpaint_pipeline.py` 是底层 generation core，可根据改动面积选择 Inpaint 或 Cross V1 + Pix2pix，并可先用 ProbNet 填充细胞层：

```bash
python scripts/run_phase3_inpaint_pipeline.py \
  --mode gen \
  --profile BCSS \
  --reference-image /data/input/reference.png \
  --reference-tissue-mask /data/input/reference_tissue.png \
  --reference-nuclei-mask /data/input/reference_nuclei.png \
  --target-tissue-mask /data/input/target_tissue.png \
  --cell-fill-mode probnet \
  --nuclei-library /data/runs/phase4/all6/nuclei_library/BCSS \
  --generation-mode auto \
  --pretrained-model-name-or-path "$FLUX_MODEL" \
  --output /data/outputs/phase3 \
  --print-summary
```

已有 target tissue/nuclei mask 时，可运行带一次质量检查和受控 fallback 的 agentic workflow：

```bash
python scripts/run_agentic_edit_workflow.py \
  --profile BCSS \
  --reference-image /data/input/reference.png \
  --reference-tissue-mask /data/input/reference_tissue.png \
  --reference-nuclei-mask /data/input/reference_nuclei.png \
  --target-tissue-mask /data/input/target_tissue.png \
  --target-nuclei-mask /data/input/target_nuclei.png \
  --semantic-change-region /data/input/semantic_change_region.png \
  --generation-change-region /data/input/generation_change_region.png \
  --segmentator-release benchmark_configs/releases/segmentator_fine_legacy_anchor.json \
  --cellvit-root "$PATHOLOGY_CELLVIT_ROOT" \
  --cellvit-model "$PATHOLOGY_CELLVIT_MODEL" \
  --cellvit-python "$PATHOLOGY_CELLVIT_PYTHON" \
  --pretrained-model-name-or-path "$FLUX_MODEL" \
  --output /data/outputs/agentic/run_001
```

Agent runner 对源图和每次生成图执行同一 legacy-anchor Segmentator，并用冻结
CellViT 审计 nuclei consistency；失败时最多切换一次 backend。语义
change region 必须等于 source/target tissue 的真实差分，generation region
可以是其结构性扩张后的超集。完整生产清单、禁止版本和哈希见
[在线产品 release](docs/online_product_release.md)。

在线 UI 和大图 patch 模式都通过同一个
`run_phase3_inpaint_pipeline.py -> inpaint_cells/generate.py` 细胞入口构建
target nuclei，并把每次运行独立的 `cell_fill_log.json` 交给 agent runner
校验。count 只从编辑前 source patch 的 tissue/nuclei 估计；changed tissue
使用 density head 决定 type quota 和精确 target-tissue library shape，
unchanged tissue 保持编辑前 type，并使用对应连通域的 patch shape。删除区
相交的完整旧核会被删除，1.5 倍最大删除核直径的 buffer 只保留外围旧核为
placement obstacle；生成核使用零重叠、完整 tissue containment 和 1 px
间隔。

## 5. 快速检查

```bash
python -m py_compile \
  controlnet_train/cli/eval_controlnet_flux_inpaint.py \
  scripts/generate_cross_v1_no_ip_strict.py \
  scripts/run_phase3_inpaint_pipeline.py \
  scripts/run_agentic_edit_workflow.py \
  inpaint_cells/generate.py

python controlnet_train/cli/eval_controlnet_flux_inpaint.py --help
python scripts/generate_cross_v1_no_ip_strict.py --help
python inpaint_cells/generate.py --help
```

完整发布验收包括：生成模型 Hub 文件 SHA256 对齐、Inpaint 固定样本、Cross V1 + Pix2pix 像素一致性、ProbNet 配合本地实例库生成 mask，以及最终 legacy-anchor Segmentator 的本地 SHA256 校验和严格 release 重建。详细结果记录在 [docs/generation_model_release.md](docs/generation_model_release.md)。
