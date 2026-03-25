#!/usr/bin/env python3
"""
端到端 Pipeline 测试: LLM → 规则引擎 → 原子操作 → ProbNet → Inpaint ControlNet 图像生成

五种模式:
    1. --mode report : 报告 → mask编辑 → 图像生成 (完整pipeline)
    2. --mode diff   : semantic_diff → mask编辑 → 图像生成
    3. --mode batch  : 预设用例批量跑mask编辑 (不生成图像)
    4. --mode gen    : 给定原图+原mask+编辑后mask → 直接生成图像 (inpaint 6ch)
    5. --mode gen9ch : 给定原图+原mask+编辑后mask → 直接生成图像 (9ch)

用法:
    cd /home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/rule_engine

    # 完整pipeline (需要GPU: LLM + ControlNet)
   CUDA_VISIBLE_DEVICES=7 python test_pipeline.py      --mode experiment      --experiment-config /home/lyw/wqx-DL/flow-edit/FlowEdit-main/experiments.json      --output /home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/rule_engine/test_output      --mask /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_y0_x512.png          --image /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/images/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_y0_x512.png         --gen-method auto    --generate

    # 跳过LLM, 直接给diff + 生成图像
    CUDA_VISIBLE_DEVICES=7 python test_pipeline.py --mode diff \
        --mask /path/to/mask.png --image /path/to/image.png \
        --diff '{"tumor_change":{"growth":"increase",...}}' --generate

    # 给定已编辑的mask, 直接生成图像
    CUDA_VISIBLE_DEVICES=7 python test_pipeline.py --mode gen \
        --image /path/to/original_image.png \
        --mask /path/to/original_mask.png \
        --edited-mask /path/to/edited_mask.png
"""

import sys
import os
import json
import argparse
import numpy as np
from PIL import Image

# =============================================================================
# 路径
# =============================================================================

RULE_ENGINE_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/rule_engine"
MASK_GEN_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/mask_data_generate"

sys.path.insert(0, RULE_ENGINE_DIR)
sys.path.insert(0, MASK_GEN_DIR)

from rule_engine import RuleEngine, MaskEditor, MaskAnalyzer, SemanticEditor

# =============================================================================
# 默认配置
# =============================================================================

PRIOR_DB_PATH = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/Prior_knowledge_of_pathology/prior_db.json"
PROB_NET_CKPT = "/data/huggingface/pathology_edit/prob_net/checkpoints/best.pt"
NUCLEI_LIBRARY = "/data/huggingface/pathology_edit/nuclei_library"
LLM_PATH = "/data/huggingface/Qwen2.5-VL-7B-Instruct"

CONTROLNET_INPAINT_PATH = "/data/huggingface/pathology_edit/inpaint_controlnet_output/checkpoint-8000/flux_controlnet"
CONTROLNET_9CH_PATH = "/data/huggingface/controlnet_6ch_v2_output/checkpoint-24000/flux_controlnet"
FLUX_PATH = "/data/huggingface/FLUX.1-dev"

DEFAULT_MASK = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_y0_x256.png"
DEFAULT_IMAGE = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/images/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_y0_x256.png"
OUTPUT_DIR = "/home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_edit/rule_engine/test_output"

# =============================================================================
# 颜色映射
# =============================================================================

COLOR_MAP = {
    0: [30,30,30], 1: [180,60,60], 2: [60,150,60], 3: [140,60,180],
    4: [60,60,180], 5: [180,180,80], 6: [160,40,40], 7: [40,40,40],
    8: [80,150,150], 9: [200,170,100], 10: [180,120,150], 11: [120,120,190],
    12: [100,190,190], 13: [200,140,60], 14: [140,200,100], 15: [140,140,140],
    16: [200,200,130], 17: [150,80,60], 18: [60,140,100], 19: [190,40,40],
    20: [80,60,150], 21: [170,170,170],
    101: [255,0,0], 102: [0,255,0], 103: [0,80,255],
    104: [255,255,0], 105: [255,0,255],
}

_rgb_to_val = {}
for val, rgb in COLOR_MAP.items():
    _rgb_to_val[rgb[0] * 65536 + rgb[1] * 256 + rgb[2]] = val


def load_mask_from_png(path):
    img = np.array(Image.open(path).convert("RGB"))
    encoded = img[:,:,0].astype(np.int64)*65536 + img[:,:,1].astype(np.int64)*256 + img[:,:,2].astype(np.int64)
    result = np.zeros(img.shape[:2], dtype=np.int64)
    for key, val in _rgb_to_val.items():
        result[encoded == key] = val
    return result


def class_map_to_rgb(class_map):
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in COLOR_MAP.items():
        rgb[class_map == val] = color
    return rgb


def save_mask_results(result, original_mask, output_dir, name):
    """保存 mask 编辑结果的可视化"""
    os.makedirs(output_dir, exist_ok=True)
    original_rgb = class_map_to_rgb(original_mask)
    src_rgb = class_map_to_rgb(result["src_mask"])
    tar_rgb = class_map_to_rgb(result["tar_mask"])

    change_region = result["change_region"]

    change_vis = original_rgb.copy()
    if change_region.any():
        change_vis[change_region] = [255, 255, 0]

    Image.fromarray(original_rgb).save(os.path.join(output_dir, f"{name}_00_original.png"))
    Image.fromarray(src_rgb).save(os.path.join(output_dir, f"{name}_01_src.png"))
    Image.fromarray(tar_rgb).save(os.path.join(output_dir, f"{name}_02_tar.png"))
    Image.fromarray(change_vis).save(os.path.join(output_dir, f"{name}_03_change.png"))

    compare = np.concatenate([original_rgb, tar_rgb, change_vis], axis=1)
    Image.fromarray(compare).save(os.path.join(output_dir, f"{name}_compare.png"))


def print_result(result):
    """打印编辑结果摘要"""
    print(f"\n  Operations: {len(result['ops_log'])}")
    for op_log in result["ops_log"]:
        log = op_log["log"]
        accepted = log.get("accepted", False) if isinstance(log, dict) else False
        print(f"    {op_log['op']}: accepted={accepted}, params={op_log['params']}")

    change_region = result["change_region"]
    change_pct = change_region.sum() / change_region.size * 100
    print(f"  Change region (tissue+cell): {change_pct:.1f}%")
    print(f"  has_shrink: {result.get('has_shrink', False)}")

    # 判断组织层是否有变化
    tissue_change = (result["original_tissue"] != result["edited_tissue"])
    if tissue_change.any():
        tissue_pct = tissue_change.sum() / tissue_change.size * 100
        print(f"    - Tissue change: {tissue_pct:.1f}%")
        cell_only_pixels = change_region.sum() - tissue_change.sum()
        if cell_only_pixels > 0:
            print(f"    - Additional cell change: {cell_only_pixels} pixels")
    elif change_region.any():
        print(f"    - Cell-only mode (tissue unchanged, cells adjusted)")

    # auto 模式建议
    if change_pct >= 35:
        print(f"  → Recommended: 9ch (change >= 35%)")
    elif change_pct > 0:
        print(f"  → Recommended: inpaint (change < 35%)")
    else:
        print(f"  → No change detected")

    print(f"  Original tissue: {MaskAnalyzer(result['original_tissue']).summary()}")
    print(f"  Edited tissue:   {MaskAnalyzer(result['edited_tissue']).summary()}")


# =============================================================================
# Inpaint ControlNet 图像生成 (6ch, 原有)
# =============================================================================

def load_inpaint_controlnet(controlnet_path, dtype=None):
    """加载 6ch inpaint ControlNet"""
    import torch
    import torch.nn as nn
    import glob
    from safetensors.torch import load_file
    from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel

    if dtype is None:
        dtype = torch.bfloat16

    with open(os.path.join(controlnet_path, "config.json")) as f:
        config = json.load(f)
    for k in ["_class_name", "_diffusers_version", "_name_or_path"]:
        config.pop(k, None)

    controlnet = FluxControlNetModel(**config)

    # 扩展 controlnet_x_embedder 到 128 维 (6ch: erased_image + mask)
    old_cx = controlnet.controlnet_x_embedder
    new_cx = nn.Linear(128, old_cx.out_features)
    with torch.no_grad():
        new_cx.weight.zero_()
        new_cx.weight[:, :old_cx.in_features] = old_cx.weight
        if old_cx.bias is not None:
            new_cx.bias.copy_(old_cx.bias)
    controlnet.controlnet_x_embedder = new_cx

    shard_files = sorted(glob.glob(os.path.join(controlnet_path, "diffusion_pytorch_model*.safetensors")))
    if not shard_files:
        shard_files = sorted(glob.glob(os.path.join(controlnet_path, "diffusion_pytorch_model*.bin")))

    state_dict = {}
    for f in shard_files:
        if f.endswith(".safetensors"):
            state_dict.update(load_file(f))
        else:
            import torch as _torch
            state_dict.update(_torch.load(f, map_location="cpu"))

    controlnet.load_state_dict(state_dict, strict=True)
    controlnet = controlnet.to(dtype=dtype)
    return controlnet


# =============================================================================
# 9ch ControlNet 加载 (ref_image + ref_mask + target_mask)
# =============================================================================

def load_9ch_controlnet(controlnet_path, dtype=None):
    """
    加载 9ch ControlNet。

    训练时 controlnet_x_embedder 被扩展到 old_in * 3 = 192 维:
      [ref_image_latent(16ch) + ref_mask_latent(16ch) + target_mask_latent(16ch)]
    打包后每个 token 是 64*3 = 192 维。
    """
    import torch
    import torch.nn as nn
    import glob
    from safetensors.torch import load_file
    from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel

    if dtype is None:
        dtype = torch.bfloat16

    with open(os.path.join(controlnet_path, "config.json")) as f:
        config = json.load(f)
    for k in ["_class_name", "_diffusers_version", "_name_or_path"]:
        config.pop(k, None)

    controlnet = FluxControlNetModel(**config)

    # 扩展 controlnet_x_embedder: 64 → 192 (3 × 64, 对应 3 个 16ch latent)
    old_cx = controlnet.controlnet_x_embedder
    old_in = old_cx.in_features    # 64
    new_in = old_in * 3            # 192
    out_features = old_cx.out_features
    new_cx = nn.Linear(new_in, out_features)
    with torch.no_grad():
        new_cx.weight.zero_()
        new_cx.weight[:, :old_in] = old_cx.weight
        if old_cx.bias is not None:
            new_cx.bias.copy_(old_cx.bias)
    controlnet.controlnet_x_embedder = new_cx

    # 加载训练好的权重 (会覆盖上面的初始化)
    shard_files = sorted(glob.glob(os.path.join(controlnet_path, "diffusion_pytorch_model*.safetensors")))
    if not shard_files:
        shard_files = sorted(glob.glob(os.path.join(controlnet_path, "diffusion_pytorch_model*.bin")))

    state_dict = {}
    for f in shard_files:
        if f.endswith(".safetensors"):
            state_dict.update(load_file(f))
        else:
            state_dict.update(torch.load(f, map_location="cpu"))

    controlnet.load_state_dict(state_dict, strict=True)
    controlnet = controlnet.to(dtype=dtype)

    print(f"Loaded 9ch ControlNet: x_embedder {old_in} → {new_in}")
    return controlnet


# =============================================================================
# 通用工具
# =============================================================================

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.16):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


# =============================================================================
# 9ch ControlNet 图像生成 (ref_image + ref_mask + target_mask)
# =============================================================================

def generate_image_with_9ch_controlnet(
    original_image_path,
    original_mask_class_map,
    edited_mask_class_map,
    output_dir,
    name,
    controlnet_path=CONTROLNET_9CH_PATH,
    flux_path=FLUX_PATH,
    device="cuda",
    num_inference_steps=28,
    guidance_scale=3.5,
    conditioning_scale=1.0,
    seed=42,
):
    """
    用 9ch ControlNet 生成编辑后的图像。

    训练时的输入:
      control_9ch = cat([ref_img_latent, ref_mask_latent, target_mask_latent], dim=1)

    推理时对应:
      ref_img_latent     = VAE.encode(original_image)       # 参考图像
      ref_mask_latent    = VAE.encode(original_mask_rgb)     # 参考 mask
      target_mask_latent = VAE.encode(edited_mask_rgb)       # 目标 mask (编辑后)

    ControlNet 学习: 给定参考图+参考mask, 按 target_mask 布局生成新图。
    """
    import torch
    from torchvision import transforms as T
    from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

    VAE_LATENT_CH = 16
    dtype = torch.bfloat16
    os.makedirs(output_dir, exist_ok=True)

    # --- 准备输入图像 ---
    original_image = Image.open(original_image_path).convert("RGB")

    # mask → RGB
    original_mask_rgb = class_map_to_rgb(original_mask_class_map)
    edited_mask_rgb = class_map_to_rgb(edited_mask_class_map)

    ref_image = original_image
    ref_mask_image = Image.fromarray(original_mask_rgb)
    target_mask_image = Image.fromarray(edited_mask_rgb)

    # 保存中间结果
    ref_image.save(os.path.join(output_dir, f"{name}_9ch_00_ref_image.png"))
    ref_mask_image.save(os.path.join(output_dir, f"{name}_9ch_01_ref_mask.png"))
    target_mask_image.save(os.path.join(output_dir, f"{name}_9ch_02_target_mask.png"))

    # --- 加载模型 ---
    print(f"Loading 9ch ControlNet from {controlnet_path}...")
    controlnet = load_9ch_controlnet(controlnet_path, dtype=dtype)

    pipe = FluxControlNetPipeline.from_pretrained(
        flux_path, controlnet=controlnet, torch_dtype=dtype)
    pipe.to(device)
    controlnet.eval()

    # --- 对齐尺寸 (16 的倍数) ---
    w, h = ref_image.size
    w, h = w - w % 16, h - h % 16
    ref_image = ref_image.crop((0, 0, w, h))
    ref_mask_image = ref_mask_image.crop((0, 0, w, h))
    target_mask_image = target_mask_image.crop((0, 0, w, h))

    # --- Prompt ---
    prompt = "a H&E stained breast cancer pathology image"
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=device)

    # --- 图像 → latent ---
    img_transform = T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(), T.Normalize([0.5], [0.5]),
    ])
    mask_transform = T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(), T.Normalize([0.5], [0.5]),
    ])

    ref_img_tensor = img_transform(ref_image).unsqueeze(0).to(device, dtype=dtype)
    ref_mask_tensor = mask_transform(ref_mask_image).unsqueeze(0).to(device, dtype=dtype)
    target_mask_tensor = mask_transform(target_mask_image).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        ref_img_latent = pipe.vae.encode(ref_img_tensor).latent_dist.sample()
        ref_img_latent = (ref_img_latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        ref_mask_latent = pipe.vae.encode(ref_mask_tensor).latent_dist.sample()
        ref_mask_latent = (ref_mask_latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        target_mask_latent = pipe.vae.encode(target_mask_tensor).latent_dist.sample()
        target_mask_latent = (target_mask_latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    # --- 9ch control = [ref_image, ref_mask, target_mask] (和训练一致) ---
    control_9ch = torch.cat([ref_img_latent, ref_mask_latent, target_mask_latent], dim=1)
    # shape: (1, 48, H_lat, W_lat)

    h_lat, w_lat = ref_img_latent.shape[2], ref_img_latent.shape[3]
    num_channels_latents = pipe.transformer.config.in_channels // 4  # 16

    control_image = FluxControlNetPipeline._pack_latents(
        control_9ch, 1, VAE_LATENT_CH * 3, h_lat, w_lat)

    # --- 准备 latents ---
    generator = torch.Generator(device=device).manual_seed(seed)
    latents, latent_image_ids = pipe.prepare_latents(
        1, num_channels_latents, h, w, prompt_embeds.dtype, device, generator, None)

    # --- Timesteps ---
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

    controlnet_blocks_repeat = False if controlnet.input_hint_block is None else True

    # --- 去噪循环 ---
    print(f"Generating image with 9ch ControlNet ({num_inference_steps} steps, "
          f"conditioning_scale={conditioning_scale})...")
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            guidance = None
            if controlnet.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device).expand(latents.shape[0])

            cn_blocks, cn_single = controlnet(
                hidden_states=latents,
                controlnet_cond=control_image,
                controlnet_mode=None,
                conditioning_scale=conditioning_scale,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )

            guidance_t = None
            if pipe.transformer.config.guidance_embeds:
                guidance_t = torch.tensor([guidance_scale], device=device).expand(latents.shape[0])

            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance_t,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                controlnet_block_samples=cn_blocks,
                controlnet_single_block_samples=cn_single,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
            )[0]

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # --- Decode ---
    latents = pipe._unpack_latents(latents, h, w, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    generated_image = pipe.image_processor.postprocess(image.detach(), output_type="pil")[0]

    # --- 保存 ---
    generated_image.save(os.path.join(output_dir, f"{name}_9ch_03_generated.png"))

    # 拼接对比: ref_image | ref_mask | target_mask | generated
    compare = np.concatenate([
        np.array(ref_image.crop((0, 0, w, h))),
        np.array(ref_mask_image),
        np.array(target_mask_image),
        np.array(generated_image),
    ], axis=1)
    Image.fromarray(compare).save(os.path.join(output_dir, f"{name}_9ch_full_compare.png"))
    print(f"Generated image saved to {output_dir}/{name}_9ch_03_generated.png")

    # 清理显存
    del pipe, controlnet
    torch.cuda.empty_cache()

    return generated_image


# =============================================================================
# Inpaint ControlNet 图像生成 (6ch, 原有)
# =============================================================================

def generate_image_with_inpaint_controlnet(
    original_image_path,
    original_mask_class_map,
    edited_mask_class_map,
    change_region,
    output_dir,
    name,
    controlnet_path=CONTROLNET_INPAINT_PATH,
    flux_path=FLUX_PATH,
    device="cuda",
    num_inference_steps=28,
    guidance_scale=3.5,
):
    """
    用 inpaint ControlNet 生成编辑后的图像。

    流程:
      1. 原图在 change_region 处擦除 → erased_image
      2. 编辑后mask → RGB → target_mask_image
      3. [erased_image_latent + mask_latent] → 6ch ControlNet → 生成图像
    """
    import torch
    from torchvision import transforms as T
    from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

    dtype = torch.bfloat16
    os.makedirs(output_dir, exist_ok=True)

    original_image = Image.open(original_image_path).convert("RGB")
    original_np = np.array(original_image)

    erased_np = original_np.copy()
    erased_np[change_region] = [128, 128, 128]
    erased_image = Image.fromarray(erased_np)

    edited_mask_rgb = class_map_to_rgb(edited_mask_class_map)
    mask_image = Image.fromarray(edited_mask_rgb)

    erased_image.save(os.path.join(output_dir, f"{name}_03_erased_image.png"))
    mask_image.save(os.path.join(output_dir, f"{name}_04_target_mask_rgb.png"))
    original_image.save(os.path.join(output_dir, f"{name}_05_original_image.png"))

    print(f"Loading inpaint ControlNet from {controlnet_path}...")
    controlnet = load_inpaint_controlnet(controlnet_path, dtype=dtype)

    pipe = FluxControlNetPipeline.from_pretrained(
        flux_path, controlnet=controlnet, torch_dtype=dtype)
    pipe.to(device)
    controlnet.eval()

    w, h = erased_image.size
    w, h = w - w % 16, h - h % 16
    erased_image = erased_image.crop((0, 0, w, h))
    mask_image = mask_image.crop((0, 0, w, h))

    prompt = "a H&E stained breast cancer pathology image"
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=device)

    img_transform = T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(), T.Normalize([0.5], [0.5]),
    ])
    mask_transform = T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(), T.Normalize([0.5], [0.5]),
    ])

    erased_tensor = img_transform(erased_image).unsqueeze(0).to(device, dtype=dtype)
    mask_tensor = mask_transform(mask_image).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        erased_latents = pipe.vae.encode(erased_tensor).latent_dist.sample()
        erased_latents = (erased_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        mask_latents = pipe.vae.encode(mask_tensor).latent_dist.sample()
        mask_latents = (mask_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    control_latents = torch.cat([erased_latents, mask_latents], dim=1)
    num_channels_latents = pipe.transformer.config.in_channels // 4
    h_lat, w_lat = erased_latents.shape[2], erased_latents.shape[3]

    control_image = FluxControlNetPipeline._pack_latents(
        control_latents, 1, num_channels_latents * 2, h_lat, w_lat)

    generator = torch.Generator(device=device).manual_seed(42)
    latents, latent_image_ids = pipe.prepare_latents(
        1, num_channels_latents, h, w, prompt_embeds.dtype, device, generator, None)

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

    controlnet_blocks_repeat = False if controlnet.input_hint_block is None else True

    print(f"Generating image ({num_inference_steps} steps)...")
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            guidance = None
            if controlnet.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device).expand(latents.shape[0])

            cn_blocks, cn_single = controlnet(
                hidden_states=latents,
                controlnet_cond=control_image,
                controlnet_mode=None,
                conditioning_scale=1.0,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )

            guidance_t = None
            if pipe.transformer.config.guidance_embeds:
                guidance_t = torch.tensor([guidance_scale], device=device).expand(latents.shape[0])

            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance_t,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                controlnet_block_samples=cn_blocks,
                controlnet_single_block_samples=cn_single,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
            )[0]

            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = pipe._unpack_latents(latents, h, w, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    generated_image = pipe.image_processor.postprocess(image.detach(), output_type="pil")[0]

    generated_image.save(os.path.join(output_dir, f"{name}_06_generated.png"))

    orig_resized = original_image.crop((0, 0, w, h))
    compare = np.concatenate([
        np.array(orig_resized),
        np.array(erased_image),
        np.array(mask_image),
        np.array(generated_image),
    ], axis=1)
    Image.fromarray(compare).save(os.path.join(output_dir, f"{name}_full_compare.png"))
    print(f"Generated image saved to {output_dir}/{name}_06_generated.png")

    del pipe, controlnet
    torch.cuda.empty_cache()

    return generated_image


# =============================================================================
# 图像生成统一入口
# =============================================================================

def generate_image(args, result, combined_mask, name,
                   auto_threshold=0.35):
    """
    统一的图像生成入口, 根据 change_region 面积比例自动选择生成方法。

    Returns:
        str: 实际使用的生成方法 ("9ch" / "inpaint" / "skipped")
    """
    change_region = result["change_region"]
    change_ratio = change_region.sum() / change_region.size
    gen_method = args.gen_method

    if gen_method == "auto":
        if change_ratio >= auto_threshold:
            gen_method = "9ch"
            print(f"  [AUTO] Change region {change_ratio:.1%} >= {auto_threshold:.0%} "
                  f"→ using 9ch (full image generation)")
        else:
            gen_method = "inpaint"
            print(f"  [AUTO] Change region {change_ratio:.1%} < {auto_threshold:.0%} "
                  f"→ using inpaint (local inpainting)")

    # 安全兜底: inpaint 需要 change_region 非空
    if gen_method == "inpaint" and not change_region.any():
        print("  [WARN] change_region is empty, nothing to inpaint. Skipping generation.")
        return "skipped"

    if gen_method == "9ch":
        generate_image_with_9ch_controlnet(
            original_image_path=args.image,
            original_mask_class_map=combined_mask,
            edited_mask_class_map=result["tar_mask"],
            output_dir=args.output,
            name=name,
            controlnet_path=args.controlnet_9ch_path,
            device=args.device,
            conditioning_scale=args.conditioning_scale,
            seed=args.seed,
        )
    else:
        generate_image_with_inpaint_controlnet(
            original_image_path=args.image,
            original_mask_class_map=combined_mask,
            edited_mask_class_map=result["tar_mask"],
            change_region=change_region,
            output_dir=args.output,
            name=name,
            controlnet_path=args.controlnet_inpaint_path,
            device=args.device,
        )

    return gen_method


# =============================================================================
# 模式1: report
# =============================================================================

def run_report_mode(args):
    print("=" * 60)
    print("Mode: REPORT")
    print("=" * 60)

    combined_mask = load_mask_from_png(args.mask)
    print(f"Mask: {args.mask}")
    print(f"Shape: {combined_mask.shape}")

    editor = SemanticEditor(
        llm_path=LLM_PATH,
        prior_db_path=PRIOR_DB_PATH,
        prob_net_ckpt=PROB_NET_CKPT if os.path.exists(PROB_NET_CKPT) else None,
        nuclei_library_path=NUCLEI_LIBRARY if os.path.exists(NUCLEI_LIBRARY) else None,
        llm_device=args.device,
    )

    print(f"\nOriginal report:\n  {args.original_report}")
    print(f"\nEdited report:\n  {args.edited_report}")

    result = editor.edit_from_reports(
        original_mask=combined_mask,
        original_report=args.original_report,
        edited_report=args.edited_report,
    )

    print(f"\nSemantic diff (LLM output):")
    print(f"  {json.dumps(result['semantic_diff'], indent=2)}")
    print_result(result)
    save_mask_results(result, combined_mask, args.output, "report")

    if args.generate and args.image:
        print("\nReleasing LLM to free GPU memory...")
        del editor
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()

        print(f"\n--- Image Generation ({args.gen_method}) ---")
        generate_image(args, result, combined_mask, "report",
                       auto_threshold=args.auto_threshold)
    else:
        del editor

    print(f"\nSaved to {args.output}")


# =============================================================================
# 模式2: diff
# =============================================================================

def run_diff_mode(args):
    print("=" * 60)
    print("Mode: DIFF")
    print("=" * 60)

    combined_mask = load_mask_from_png(args.mask)
    print(f"Mask: {args.mask}")
    print(f"Shape: {combined_mask.shape}")

    semantic_diff = json.loads(args.diff)
    print(f"\nSemantic diff:")
    print(f"  {json.dumps(semantic_diff, indent=2)}")

    editor = MaskEditor(
        prior_db_path=PRIOR_DB_PATH,
        prob_net_ckpt=PROB_NET_CKPT if os.path.exists(PROB_NET_CKPT) else None,
        nuclei_library_path=NUCLEI_LIBRARY if os.path.exists(NUCLEI_LIBRARY) else None,
        seed=42,
    )

    result = editor.edit(combined_mask, semantic_diff)
    print_result(result)
    save_mask_results(result, combined_mask, args.output, "diff")

    if args.generate and args.image:
        print(f"\n--- Image Generation ({args.gen_method}) ---")
        generate_image(args, result, combined_mask, "diff",
                       auto_threshold=args.auto_threshold)

    print(f"\nSaved to {args.output}")


# =============================================================================
# 模式3: batch
# =============================================================================

BATCH_CASES = [
    # --- 组织层编辑 ---
    {
        "name": "tumor_expand",
        "desc": "肿瘤中度增长",
        "diff": {
            "tumor_change": {"growth": "increase", "degree": "moderate", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none"},
        },
    },
    {
        "name": "tumor_shrink",
        "desc": "肿瘤中度收缩 (治疗反应)",
        "diff": {
            "tumor_change": {"growth": "decrease", "degree": "moderate", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none"},
        },
    },
    {
        "name": "lymph_increase",
        "desc": "淋巴浸润显著增加",
        "diff": {
            "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "increase", "degree": "significant"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none"},
        },
    },
    {
        "name": "necrosis_add",
        "desc": "新增局灶坏死",
        "diff": {
            "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "add", "extent": "focal"},
            "stroma_change": {"density": "none"},
        },
    },
    {
        "name": "necrosis_fibrosis",
        "desc": "坏死中度纤维化修复",
        "diff": {
            "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "decrease", "extent": "moderate"},
            "stroma_change": {"density": "none"},
        },
    },
    {
        "name": "necrosis_remove",
        "desc": "坏死强力纤维化移除",
        "diff": {
            "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "remove", "extent": "extensive"},
            "stroma_change": {"density": "none"},
        },
    },
    {
        "name": "stromal_fibrosis",
        "desc": "间质纤维化 (促纤维化反应)",
        "diff": {
            "tumor_change": {"growth": "none", "degree": "mild", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "increase", "degree": "moderate"},
        },
    },
    {
        "name": "combined",
        "desc": "肿瘤增长 + 升级 + 淋巴增加",
        "diff": {
            "tumor_change": {"growth": "increase", "degree": "significant", "grade_change": "upgrade"},
            "lymphocyte_change": {"infiltration": "increase", "degree": "moderate"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none"},
        },
    },
    {
        "name": "treatment_response",
        "desc": "治疗后全套 (肿瘤缩小 + 坏死纤维化 + 间质纤维化)",
        "diff": {
            "tumor_change": {"growth": "decrease", "degree": "moderate", "grade_change": "none"},
            "lymphocyte_change": {"infiltration": "increase", "degree": "mild"},
            "necrosis_change": {"action": "decrease", "extent": "moderate"},
            "stroma_change": {"density": "increase", "degree": "significant"},
        },
    },
    # --- Cell-only 场景 ---
    {
        "name": "grade_upgrade_only",
        "desc": "纯 grade upgrade (cell-only, 无组织层变化)",
        "diff": {
            "tumor_change": {"growth": "none", "degree": "moderate", "grade_change": "upgrade"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none"},
        },
    },
    {
        "name": "grade_downgrade_only",
        "desc": "纯 grade downgrade (cell-only)",
        "diff": {
            "tumor_change": {"growth": "none", "degree": "moderate", "grade_change": "downgrade"},
            "lymphocyte_change": {"infiltration": "none", "degree": "mild"},
            "necrosis_change": {"action": "none", "extent": "focal"},
            "stroma_change": {"density": "none"},
        },
    },
]


def run_batch_mode(args):
    print("=" * 60)
    print("Mode: BATCH")
    print("=" * 60)

    combined_mask = load_mask_from_png(args.mask)
    print(f"Mask: {args.mask}")
    print(f"Shape: {combined_mask.shape}")
    print(f"Tissues: {MaskAnalyzer(combined_mask).summary()}")

    editor = MaskEditor(
        prior_db_path=PRIOR_DB_PATH,
        prob_net_ckpt=PROB_NET_CKPT if os.path.exists(PROB_NET_CKPT) else None,
        nuclei_library_path=NUCLEI_LIBRARY if os.path.exists(NUCLEI_LIBRARY) else None,
        seed=42,
    )

    for i, tc in enumerate(BATCH_CASES):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {tc['name']} -- {tc['desc']}")
        print(f"{'='*60}")

        # 先打印规则引擎的规划
        tissue_mask = editor._get_tissue_only(combined_mask)
        ops = editor.rule_engine.plan(tc["diff"], tissue_mask)
        bias, density = editor.rule_engine.compute_cell_adjustments(tc["diff"])
        print(f"  Planned ops: {len(ops)}")
        for op in ops:
            print(f"    {op['op']}: {op['params']}")
        if bias:
            print(f"  Cell type_bias: {bias}")
        if density:
            print(f"  Cell density_scale: {density}")

        # 执行完整编辑
        result = editor.edit(combined_mask, tc["diff"])
        print_result(result)
        save_mask_results(result, combined_mask, args.output, tc["name"])

    print(f"\nAll results saved to {args.output}")


# =============================================================================
# 模式4: gen (inpaint)
# =============================================================================

def run_gen_mode(args):
    print("=" * 60)
    print("Mode: GEN (inpaint ControlNet)")
    print("=" * 60)

    original_mask = load_mask_from_png(args.mask)
    edited_mask = load_mask_from_png(args.edited_mask)
    change_region = (original_mask != edited_mask)

    change_pct = change_region.sum() / change_region.size * 100
    print(f"Original mask: {args.mask}")
    print(f"Edited mask: {args.edited_mask}")
    print(f"Change region: {change_pct:.1f}%")

    if not change_region.any():
        print("WARNING: No change between original and edited mask. "
              "Inpaint mode requires change_region. Use gen9ch mode instead.")
        return

    generate_image_with_inpaint_controlnet(
        original_image_path=args.image,
        original_mask_class_map=original_mask,
        edited_mask_class_map=edited_mask,
        change_region=change_region,
        output_dir=args.output,
        name="gen",
        device=args.device,
    )


# =============================================================================
# 模式5: gen9ch (9ch ControlNet)
# =============================================================================

def run_gen9ch_mode(args):
    print("=" * 60)
    print("Mode: GEN9CH (9ch ControlNet: ref_image + ref_mask + target_mask)")
    print("=" * 60)

    original_mask = load_mask_from_png(args.mask)
    edited_mask = load_mask_from_png(args.edited_mask)

    change_region = (original_mask != edited_mask)
    change_pct = change_region.sum() / change_region.size * 100
    print(f"Original image: {args.image}")
    print(f"Original mask:  {args.mask}")
    print(f"Edited mask:    {args.edited_mask}")
    print(f"Change region:  {change_pct:.1f}%")

    generate_image_with_9ch_controlnet(
        original_image_path=args.image,
        original_mask_class_map=original_mask,
        edited_mask_class_map=edited_mask,
        output_dir=args.output,
        name="gen9ch",
        controlnet_path=args.controlnet_9ch_path,
        device=args.device,
        conditioning_scale=args.conditioning_scale,
        seed=args.seed,
    )


# =============================================================================
# 模式6: experiment (JSON 配置驱动的批量实验)
# =============================================================================

def run_experiment_mode(args):
    """
    JSON 配置驱动的批量实验。

    流程 (三阶段, LLM 用完即释放):
      Phase 1: LLM Parser — 批量解析所有报告对 → semantic_diff
      Phase 2: Rule Engine + Mask Edit — 批量编辑 mask
      Phase 3: Image Generation — 批量生成图像 (可选)

    每个实验保存的文件 (以 exp01_tumor_growth_moderate 为例):
      Phase 1:
        exp01_tumor_growth_moderate_semantic_diff.json   — LLM parser 输出
      Phase 2:
        exp01_tumor_growth_moderate_ops_log.json         — 原子操作流
        exp01_tumor_growth_moderate_original_mask.png     — 原 mask (RGB)
        exp01_tumor_growth_moderate_edited_mask.png       — 新 mask (RGB)
        exp01_tumor_growth_moderate_change_region.png     — change region (白=变化)
        exp01_tumor_growth_moderate_compare.png           — 对比拼接图
      Phase 3:
        exp01_tumor_growth_moderate_original_image.png    — 原图
        exp01_tumor_growth_moderate_generated_image.png   — 生成图
        exp01_tumor_growth_moderate_generation_info.json  — 生成方法 + 参数

    汇总:
      all_semantic_diffs.json                            — 全部 LLM 输出
      experiment_summary.json                            — 全部实验结果汇总
    """
    import gc
    from shutil import copy2

    print("=" * 60)
    print("Mode: EXPERIMENT (JSON-driven batch)")
    print("=" * 60)

    # --- 加载配置 ---
    with open(args.experiment_config, "r") as f:
        config = json.load(f)

    meta = config.get("_meta", {})
    mask_path = args.mask if args.mask != DEFAULT_MASK else meta.get("mask", DEFAULT_MASK)
    image_path = args.image if args.image != DEFAULT_IMAGE else meta.get("image", DEFAULT_IMAGE)
    original_report = config["original_report"]
    experiments = config["experiments"]

    print(f"Config: {args.experiment_config}")
    print(f"Mask:   {mask_path}")
    print(f"Image:  {image_path}")
    print(f"Experiments: {len(experiments)}")

    combined_mask = load_mask_from_png(mask_path)
    print(f"Mask shape: {combined_mask.shape}")
    print(f"Mask tissues: {MaskAnalyzer(combined_mask).summary()}")

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 用于最终汇总
    summary = {
        "config": args.experiment_config,
        "mask_path": mask_path,
        "image_path": image_path,
        "mask_shape": list(combined_mask.shape),
        "experiments": {},
    }

    # =====================================================================
    # Phase 1: LLM Parser — 批量解析
    # =====================================================================
    print(f"\n{'='*60}")
    print("Phase 1: LLM Parser — batch parsing all report pairs")
    print(f"{'='*60}")

    from llm_parser import LLMParser
    llm_parser = LLMParser(
        model_path=args.llm_path,
        device=args.device,
    )

    all_diffs = {}
    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        prefix = f"{exp_id}_{exp_name}"
        print(f"\n  [{exp_id}] {exp_name}: parsing...")

        diff = llm_parser.parse(original_report, exp["edited_report"])
        all_diffs[exp_id] = diff

        print(f"    → {json.dumps(diff)}")

        # 1. 保存 LLM parser 输出
        diff_path = os.path.join(output_dir, f"{prefix}_semantic_diff.json")
        with open(diff_path, "w") as f:
            json.dump({
                "id": exp_id,
                "name": exp_name,
                "desc": exp.get("desc", ""),
                "original_report": original_report,
                "edited_report": exp["edited_report"],
                "semantic_diff": diff,
            }, f, indent=2, ensure_ascii=False)

    # 保存汇总
    summary_path = os.path.join(output_dir, "all_semantic_diffs.json")
    with open(summary_path, "w") as f:
        json.dump(all_diffs, f, indent=2)
    print(f"\nAll diffs saved to {summary_path}")

    # 释放 LLM
    print("\nReleasing LLM to free GPU memory...")
    del llm_parser
    import torch
    torch.cuda.empty_cache()
    gc.collect()

    # =====================================================================
    # Phase 2: Rule Engine + Mask Edit — 批量编辑
    # =====================================================================
    print(f"\n{'='*60}")
    print("Phase 2: Rule Engine + Mask Edit — batch editing")
    print(f"{'='*60}")

    editor = MaskEditor(
        prior_db_path=PRIOR_DB_PATH,
        prob_net_ckpt=PROB_NET_CKPT if os.path.exists(PROB_NET_CKPT) else None,
        nuclei_library_path=NUCLEI_LIBRARY if os.path.exists(NUCLEI_LIBRARY) else None,
        seed=args.seed,
    )

    all_results = {}
    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        prefix = f"{exp_id}_{exp_name}"
        diff = all_diffs[exp_id]

        print(f"\n  [{exp_id}] {exp_name}: editing mask...")

        result = editor.edit(combined_mask, diff)
        all_results[exp_id] = result

        print_result(result)

        change_region = result["change_region"]
        change_pct = float(change_region.sum() / change_region.size * 100)

        # 3. 保存原 mask (RGB)
        original_rgb = class_map_to_rgb(combined_mask)
        Image.fromarray(original_rgb).save(
            os.path.join(output_dir, f"{prefix}_original_mask.png"))

        # 3. 保存新 mask (RGB)
        tar_rgb = class_map_to_rgb(result["tar_mask"])
        Image.fromarray(tar_rgb).save(
            os.path.join(output_dir, f"{prefix}_edited_mask.png"))

        # 4. 保存 change region (独立的二值图: 白=变化, 黑=不变)
        change_img = np.zeros(change_region.shape, dtype=np.uint8)
        change_img[change_region] = 255
        Image.fromarray(change_img).save(
            os.path.join(output_dir, f"{prefix}_change_region.png"))

        # 对比拼接图: 原mask | 新mask | change region叠加
        change_overlay = original_rgb.copy()
        change_overlay[change_region] = [255, 255, 0]
        compare = np.concatenate([original_rgb, tar_rgb, change_overlay], axis=1)
        Image.fromarray(compare).save(
            os.path.join(output_dir, f"{prefix}_compare.png"))

        # 2. 保存原子操作流
        safe_ops_log = []
        for op_entry in result["ops_log"]:
            safe_entry = {
                "op": op_entry["op"],
                "direction": op_entry.get("direction", ""),
                "params": op_entry["params"],
            }
            log = op_entry.get("log", {})
            if isinstance(log, dict):
                safe_entry["log"] = log
            else:
                safe_entry["log"] = {"accepted": getattr(log, "accepted", False),
                                      "reason": str(log)}
            safe_ops_log.append(safe_entry)

        ops_log_path = os.path.join(output_dir, f"{prefix}_ops_log.json")
        with open(ops_log_path, "w") as f:
            json.dump({
                "id": exp_id,
                "name": exp_name,
                "semantic_diff": diff,
                "ops_log": safe_ops_log,
                "change_region_pct": change_pct,
                "original_tissue": MaskAnalyzer(result["original_tissue"]).summary(),
                "edited_tissue": MaskAnalyzer(result["edited_tissue"]).summary(),
            }, f, indent=2, ensure_ascii=False)

        # 汇总记录
        summary["experiments"][exp_id] = {
            "name": exp_name,
            "desc": exp.get("desc", ""),
            "semantic_diff": diff,
            "n_ops": len(safe_ops_log),
            "change_region_pct": change_pct,
        }

    del editor
    gc.collect()

    # =====================================================================
    # Phase 3: Image Generation (可选)
    # =====================================================================
    if not args.generate:
        print(f"\n{'='*60}")
        print("Phase 3: SKIPPED (use --generate to enable)")
        print(f"{'='*60}")

        # 保存汇总 (不含 generation info)
        with open(os.path.join(output_dir, "experiment_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nAll results saved to {output_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Phase 3: Image Generation ({args.gen_method})")
    print(f"{'='*60}")

    # 6. 保存原图 (只复制一次)
    original_image_dst = os.path.join(output_dir, "original_image.png")
    if not os.path.exists(original_image_dst):
        copy2(image_path, original_image_dst)
        print(f"  Original image copied to {original_image_dst}")

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        prefix = f"{exp_id}_{exp_name}"
        result = all_results[exp_id]

        change_pct = result["change_region"].sum() / result["change_region"].size * 100
        print(f"\n  [{exp_id}] {exp_name}: generating image "
              f"(change={change_pct:.1f}%)...")

        gen_args = argparse.Namespace(
            image=image_path,
            output=output_dir,
            gen_method=args.gen_method,
            controlnet_9ch_path=args.controlnet_9ch_path,
            controlnet_inpaint_path=args.controlnet_inpaint_path,
            device=args.device,
            conditioning_scale=args.conditioning_scale,
            seed=args.seed,
            auto_threshold=args.auto_threshold,
        )

        # 生成图像, 获取实际使用的方法
        actual_method = generate_image(
            gen_args, result, combined_mask,
            name=prefix,
            auto_threshold=args.auto_threshold,
        )

        # 5 + 6. 保存生成信息
        gen_info = {
            "id": exp_id,
            "name": exp_name,
            "gen_method_requested": args.gen_method,
            "gen_method_actual": actual_method,
            "change_region_pct": float(change_pct),
            "auto_threshold": args.auto_threshold,
            "conditioning_scale": args.conditioning_scale,
            "seed": args.seed,
            "controlnet_9ch_path": args.controlnet_9ch_path,
            "controlnet_inpaint_path": args.controlnet_inpaint_path,
        }
        gen_info_path = os.path.join(output_dir, f"{prefix}_generation_info.json")
        with open(gen_info_path, "w") as f:
            json.dump(gen_info, f, indent=2)

        # 找到生成的图像并重命名为统一格式
        # 9ch 模式生成的文件: {name}_9ch_03_generated.png
        # inpaint 模式生成的文件: {name}_06_generated.png
        if actual_method == "9ch":
            src_gen = os.path.join(output_dir, f"{prefix}_9ch_03_generated.png")
        elif actual_method == "inpaint":
            src_gen = os.path.join(output_dir, f"{prefix}_06_generated.png")
        else:
            src_gen = None

        if src_gen and os.path.exists(src_gen):
            dst_gen = os.path.join(output_dir, f"{prefix}_generated_image.png")
            if src_gen != dst_gen:
                copy2(src_gen, dst_gen)

        # 更新汇总
        summary["experiments"][exp_id]["gen_method"] = actual_method

    # 保存最终汇总
    with open(os.path.join(output_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"All results saved to {output_dir}")
    print(f"{'='*60}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline test")
    parser.add_argument("--mode", choices=["report", "diff", "batch", "gen", "gen9ch", "experiment"],
                        default="batch")
    parser.add_argument("--mask", default=DEFAULT_MASK, help="Input mask PNG")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Input image PNG")
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generate", action="store_true", help="Also generate image with ControlNet")
    parser.add_argument("--gen-method", choices=["inpaint", "9ch", "auto"], default="auto",
                        help="Image generation method: inpaint (6ch local), 9ch (full image), "
                             "auto (choose by change_region size, default)")
    parser.add_argument("--auto-threshold", type=float, default=0.35,
                        help="Change region ratio threshold for auto mode to switch to 9ch (default: 0.35)")
    parser.add_argument("--controlnet-9ch-path", default=CONTROLNET_9CH_PATH,
                        help="Path to 9ch ControlNet checkpoint")
    parser.add_argument("--controlnet-inpaint-path", default=CONTROLNET_INPAINT_PATH,
                        help="Path to inpaint (6ch) ControlNet checkpoint")
    parser.add_argument("--conditioning-scale", type=float, default=1.0,
                        help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=42)

    # report mode
    parser.add_argument("--original-report", default=None)
    parser.add_argument("--edited-report", default=None)

    # diff mode
    parser.add_argument("--diff", default=None)

    # gen / gen9ch mode
    parser.add_argument("--edited-mask", default=None, help="Edited mask PNG")

    # experiment mode
    parser.add_argument("--experiment-config", default=None,
                        help="Path to experiment JSON config file")
    parser.add_argument("--llm-path", default=LLM_PATH,
                        help="Path to LLM model for experiment mode")

    args = parser.parse_args()

    if args.mode == "report":
        if not args.original_report or not args.edited_report:
            print("ERROR: --original-report and --edited-report required")
            return
        run_report_mode(args)
    elif args.mode == "diff":
        if not args.diff:
            print("ERROR: --diff required")
            return
        run_diff_mode(args)
    elif args.mode == "batch":
        run_batch_mode(args)
    elif args.mode == "gen":
        if not args.edited_mask or not args.image:
            print("ERROR: --image and --edited-mask required for gen mode")
            return
        run_gen_mode(args)
    elif args.mode == "gen9ch":
        if not args.edited_mask or not args.image:
            print("ERROR: --image, --mask, and --edited-mask required for gen9ch mode")
            return
        run_gen9ch_mode(args)
    elif args.mode == "experiment":
        if not args.experiment_config:
            print("ERROR: --experiment-config required for experiment mode")
            return
        run_experiment_mode(args)


if __name__ == "__main__":
    main()