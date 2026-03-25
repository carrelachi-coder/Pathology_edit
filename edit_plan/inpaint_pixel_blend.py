"""
inpaint_pixel_blend_6ch.py
==========================
6ch Inpainting ControlNet + 像素级 Blending

流程:
  1. 读取 原图 + 原mask(original_with_cells.png) + 新mask(final_with_generated_cells.png)
  2. 通过 change_region 确定编辑区域
  3. 将原图在编辑区域 erase → erased_image
  4. erased_image + new_mask 分别 VAE encode → concat 成 6ch 作为 ControlNet 输入
  5. 用 new_mask 生成 prompt
  6. 生成完整图像
  7. 像素空间 blending: 只替换编辑区域，保留非编辑区域

与训练脚本的对应关系:
  训练: erased_image(3ch) + mask_image(3ch) → VAE encode 各得 16ch latent → cat=32ch
        → _pack_latents(32ch) → controlnet_x_embedder(128→3072)
  推理: erased_pil + new_mask_pil → 同样流程

用法:
  python /home/lyw/wqx-DL/flow-edit/FlowEdit-main/edit_plan/inpaint_pixel_blend.py\
      --mask_edit_dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/mask_data_generate/mask_edit_output \
      --original_image /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/images/TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500_y0_x0.png \
      --controlnet_path /data/huggingface/pathology_edit/inpaint_controlnet_output/checkpoint-8000/flux_controlnet \
      --output_dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/edit_result/pixel_blend_results_6ch 
      --strength 0.75
"""

import os
import argparse
import subprocess
import json
import random
import logging
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from torchvision import transforms as T
from safetensors.torch import load_file

from diffusers import FluxPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
#  加载 6ch Inpainting ControlNet
# ============================================================

def load_inpaint_controlnet(controlnet_path, dtype=torch.bfloat16):
    """
    手动加载 6ch inpaint controlnet，绕过 from_pretrained 的 shape 校验。

    checkpoint 里:
      x_embedder.weight:              [3072, 64]   ← 原始, 未改
      controlnet_x_embedder.weight:   [3072, 128]  ← 训练时 patch 过的

    所以初始化时 in_channels=64 (匹配 x_embedder),
    再手动扩展 controlnet_x_embedder 到 128 (匹配 checkpoint).
    """
    with open(os.path.join(controlnet_path, "config.json")) as f:
        config = json.load(f)

    # 保持 in_channels=64 让 x_embedder [3072,64] 能匹配 checkpoint
    config["in_channels"] = 64

    for k in ["_class_name", "_diffusers_version", "_name_or_path"]:
        config.pop(k, None)

    controlnet = FluxControlNetModel(**config)

    # 手动把 controlnet_x_embedder 扩展到 128（和训练脚本一致）
    # 训练脚本: new_in = old_in * 2 = 64 * 2 = 128
    old_cx = controlnet.controlnet_x_embedder
    new_cx = nn.Linear(128, old_cx.out_features)
    with torch.no_grad():
        new_cx.weight.zero_()
        new_cx.weight[:, :old_cx.in_features] = old_cx.weight
        if old_cx.bias is not None:
            new_cx.bias.copy_(old_cx.bias)
    controlnet.controlnet_x_embedder = new_cx

    # 加载权重
    shard_files = sorted(glob.glob(os.path.join(controlnet_path, "diffusion_pytorch_model*.safetensors")))
    if not shard_files:
        shard_files = sorted(glob.glob(os.path.join(controlnet_path, "diffusion_pytorch_model*.bin")))

    if not shard_files:
        raise FileNotFoundError(f"No model files found in {controlnet_path}")

    state_dict = {}
    for f in shard_files:
        if f.endswith(".safetensors"):
            state_dict.update(load_file(f))
        else:
            state_dict.update(torch.load(f, map_location="cpu"))

    controlnet.load_state_dict(state_dict, strict=True)
    controlnet = controlnet.to(dtype=dtype)
    logger.info(f"Loaded 6ch inpaint ControlNet from {controlnet_path}")
    logger.info(f"  x_embedder: {controlnet.x_embedder.weight.shape}")
    logger.info(f"  controlnet_x_embedder: {controlnet.controlnet_x_embedder.weight.shape}")
    return controlnet


# ============================================================
#  颜色映射 & Prompt
# ============================================================

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
TISSUE_NAMES = {
    0: 'outside_roi', 1: 'tumor', 2: 'stroma', 3: 'lymphocytic_infiltrate',
    4: 'necrosis', 5: 'glandular_secretions', 6: 'blood', 7: 'exclude',
    8: 'metaplasia', 9: 'fat', 10: 'plasma_cells', 11: 'other_immune_infiltrate',
    12: 'mucoid_material', 13: 'normal_acinus_or_duct', 14: 'lymphatics',
    15: 'undetermined', 16: 'nerve', 17: 'skin_adnexa', 18: 'blood_vessel',
    19: 'angioinvasion', 20: 'dcis', 21: 'other',
}
NUCLEI_NAMES = {101: 'neoplastic', 102: 'inflammatory', 103: 'connective',
                104: 'dead', 105: 'epithelial'}
SKIP_TISSUES = {0, 7, 15, 21}


def mask_to_prompt(mask_path):
    """从 semantic mask 生成文本 prompt"""
    img = np.array(Image.open(mask_path).convert("RGB"))
    h, w, _ = img.shape
    total = h * w
    encoded = img[:,:,0].astype(np.int64)*65536 + img[:,:,1].astype(np.int64)*256 + img[:,:,2].astype(np.int64)
    rgb_to_val = {}
    for val, rgb in COLOR_MAP.items():
        rgb_to_val[rgb[0]*65536 + rgb[1]*256 + rgb[2]] = val
    class_map = np.zeros((h, w), dtype=np.int64)
    for key, val in rgb_to_val.items():
        class_map[encoded == key] = val
    tissue_parts = []
    for tid in range(22):
        if tid in SKIP_TISSUES:
            continue
        pct = (class_map == tid).sum() / total * 100
        if pct >= 1.0:
            tissue_parts.append((TISSUE_NAMES[tid], pct))
    tissue_parts.sort(key=lambda x: -x[1])
    nuclei_parts = [NUCLEI_NAMES[nid] for nid in [101,102,103,104,105] if (class_map==nid).sum()>0]
    prompt = "H&E stained breast cancer histopathology at 40x magnification"
    if tissue_parts:
        prompt += ", showing " + ", ".join(f"{n} ({p:.0f}%)" for n,p in tissue_parts)
    if nuclei_parts:
        prompt += ", with " + " and ".join(nuclei_parts) + " nuclei"
    return prompt


# ============================================================
#  像素空间 Blending
# ============================================================

def create_feathered_mask(change_mask_bool, feather_radius=15):
    mask_float = change_mask_bool.astype(np.float32)
    feathered = gaussian_filter(mask_float, sigma=feather_radius)
    feathered[change_mask_bool] = np.maximum(feathered[change_mask_bool], 0.95)
    if feathered.max() > 0:
        feathered = feathered / feathered.max()
    return feathered


def pixel_blend(src_np, gen_np, feathered_mask):
    mask_3ch = feathered_mask[:, :, np.newaxis]
    return (mask_3ch * gen_np + (1 - mask_3ch) * src_np).astype(np.uint8)


def poisson_blend(src_np, gen_np, mask_bool):
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return src_np
    center = (int(xs.mean()), int(ys.mean()))
    try:
        return cv2.seamlessClone(gen_np, src_np, mask_uint8, center, cv2.NORMAL_CLONE)
    except Exception as e:
        logger.warning(f"Poisson blend failed ({e}), falling back to feathered")
        return pixel_blend(src_np, gen_np, create_feathered_mask(mask_bool, 15))


# ============================================================
#  Erase 原图
# ============================================================

def erase_image(src_pil, change_mask_bool, erase_value=128):
    """
    将原图在 change_mask 区域填充灰色 (模拟 erase)。
    erase_value: 填充值, 128=中灰色, 需和训练时一致。
    """
    src_np = np.array(src_pil).copy()
    dilated = binary_dilation(change_mask_bool, iterations=3)
    src_np[dilated] = erase_value
    return Image.fromarray(src_np)


# ============================================================
#  辅助函数
# ============================================================

def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.16):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def create_diff_heatmap(img1_np, img2_np):
    """生成两张图的像素差异热力图"""
    diff = np.abs(img1_np.astype(np.float32) - img2_np.astype(np.float32))
    diff_gray = diff.mean(axis=2)
    if diff_gray.max() > 0:
        diff_norm = (diff_gray / diff_gray.max() * 255).astype(np.uint8)
    else:
        diff_norm = diff_gray.astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
    return diff_color


# ============================================================
#  6ch ControlNet 推理 (核心 — 严格对齐训练脚本)
# ============================================================

@torch.no_grad()
def generate_with_6ch_controlnet(
    pipe, controlnet, erased_pil, new_mask_pil, prompt, device,
    src_image_pil=None, strength=0.75,
    num_inference_steps=28, guidance_scale=3.5,
    controlnet_conditioning_scale=1.0, generator=None,
):
    """
    用 6ch inpainting ControlNet 生成图像。

    对齐训练脚本的关键点:
      1. erased_image 和 mask_image 分别 VAE encode
      2. latent cat 在 dim=1 → 32ch (16+16)
      3. _pack_latents 的 num_channels = num_ch_lat * 2
         其中 num_ch_lat = erased_latents.shape[1] (VAE latent channels = 16)
      4. txt_ids: text_ids[0] if text_ids.dim() == 3 else text_ids
      5. timestep / 1000
      6. guidance 用 torch.full 标量展开 (不是 torch.tensor([x]).expand)
    """
    dtype = torch.bfloat16

    w, h = erased_pil.size
    w, h = w - w % 16, h - h % 16
    erased_pil = erased_pil.crop((0, 0, w, h))
    new_mask_pil = new_mask_pil.crop((0, 0, w, h))

    # ---- Encode prompt ----
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=device,
    )

    # ---- 6ch control_image: 分别 VAE encode 再 concat ----
    # 训练脚本: transforms.Normalize([0.5], [0.5])
    img_transform = T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    erased_tensor = img_transform(erased_pil).unsqueeze(0).to(device, dtype=dtype)
    mask_tensor = img_transform(new_mask_pil).unsqueeze(0).to(device, dtype=dtype)

    # 训练脚本:
    #   erased_latents = vae.encode(erased_values).latent_dist.sample()
    #   erased_latents = (erased_latents - vae.config.shift_factor) * vae.config.scaling_factor
    erased_latents = pipe.vae.encode(erased_tensor).latent_dist.sample()
    erased_latents = (erased_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    mask_latents = pipe.vae.encode(mask_tensor).latent_dist.sample()
    mask_latents = (mask_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    # 训练脚本:
    #   control_latents_cat = torch.cat([erased_latents, mask_latents], dim=1)
    #   num_ch_lat = erased_latents.shape[1]  # = 16
    #   control_image = _pack_latents(control_latents_cat, bsz, num_ch_lat * 2, h, w)
    control_latents = torch.cat([erased_latents, mask_latents], dim=1)
    num_ch_lat = erased_latents.shape[1]  # VAE latent channels = 16
    h_lat, w_lat = erased_latents.shape[2], erased_latents.shape[3]

    control_image = FluxControlNetPipeline._pack_latents(
        control_latents, 1, num_ch_lat * 2, h_lat, w_lat,
    )

    # ---- denoising latents ----
    num_channels_latents = num_ch_lat  # = 16, 和训练一致

    latents, latent_image_ids = pipe.prepare_latents(
        1, num_channels_latents, h, w,
        prompt_embeds.dtype, device, generator, None,
    )

    # ---- SDEdit ----
    if src_image_pil is not None and strength < 1.0:
        src_pil_cropped = src_image_pil.crop((0, 0, w, h))
        src_tensor = img_transform(src_pil_cropped).unsqueeze(0).to(device, dtype=dtype)
        x0 = pipe.vae.encode(src_tensor).latent_dist.sample()
        x0 = (x0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        x0_packed = FluxControlNetPipeline._pack_latents(
            x0, 1, num_channels_latents, h_lat, w_lat,
        )
        start_step = max(0, int(num_inference_steps * (1 - strength)))
        logger.info(f"SDEdit: strength={strength}, start_step={start_step}/{num_inference_steps}")
    else:
        x0_packed = None
        start_step = 0

    # ---- Timesteps ----
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
        pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
    )

    # ---- SDEdit: 源图加噪 ----
    # 训练脚本: noisy = (1-sigma)*x0 + sigma*noise
    if x0_packed is not None and start_step > 0:
        noise = latents.clone()
        sigma = pipe.scheduler.sigmas[start_step] if start_step < len(pipe.scheduler.sigmas) else pipe.scheduler.sigmas[-1]
        latents = ((1.0 - sigma) * x0_packed + sigma * noise).to(dtype)
        timesteps = timesteps[start_step:]
        logger.info(f"Injected source latent with sigma={sigma:.4f}, remaining steps={len(timesteps)}")

    # ---- txt_ids 维度处理 (和训练脚本一致) ----
    # 训练: txt_ids=text_ids[0] if text_ids.dim() == 3 else text_ids
    txt_ids_for_model = text_ids[0] if text_ids.dim() == 3 else text_ids

    # ---- input_hint_block 检查 ----
    controlnet_blocks_repeat = (
        hasattr(controlnet, 'input_hint_block')
        and controlnet.input_hint_block is not None
    )

    # ---- 去噪循环 ----
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        # guidance (训练脚本: torch.full((bsz,), guidance_scale, ...))
        if controlnet.config.guidance_embeds:
            guidance = torch.full(
                (latents.shape[0],), guidance_scale,
                device=device, dtype=dtype
            )
        else:
            guidance = None

        # ControlNet forward (训练: timestep=timesteps/1000)
        cn_blocks, cn_single = controlnet(
            hidden_states=latents,
            controlnet_cond=control_image,
            controlnet_mode=None,
            conditioning_scale=controlnet_conditioning_scale,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids_for_model,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )

        # Transformer guidance
        if pipe.transformer.config.guidance_embeds:
            guidance_t = torch.full(
                (latents.shape[0],), guidance_scale,
                device=device, dtype=dtype
            )
        else:
            guidance_t = None

        # Transformer forward
        # 训练: controlnet_block_samples=[s.to(dtype=weight_dtype) for s in ...]
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance_t,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            controlnet_block_samples=[s.to(dtype=dtype) for s in cn_blocks]
                if cn_blocks is not None else None,
            controlnet_single_block_samples=[s.to(dtype=dtype) for s in cn_single]
                if cn_single is not None else None,
            txt_ids=txt_ids_for_model,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )[0]

        # Scheduler step
        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    # ---- Decode ----
    latents = pipe._unpack_latents(latents, h, w, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    return image


# ============================================================
#  完整 Pipeline: 单次编辑
# ============================================================

def run_single_edit(pipe, controlnet, source_image_path, source_mask_path,
                    target_mask_path, change_mask_path, output_dir, edit_name,
                    device, strength=0.75, T_steps=28, guidance_scale=3.5,
                    controlnet_conditioning_scale=1.0,
                    feather_radius=15, blend_mode="feather",
                    erase_value=128, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    # 生成 prompt (基于新 mask)
    tar_prompt = mask_to_prompt(target_mask_path)
    src_prompt = mask_to_prompt(source_mask_path)
    print(f"  Src prompt: {src_prompt}")
    print(f"  Tar prompt: {tar_prompt}")

    # 源图像
    src_image = Image.open(source_image_path).convert("RGB")
    w, h = src_image.size
    w, h = w - w % 16, h - h % 16
    src_image = src_image.crop((0, 0, w, h))
    src_np = np.array(src_image)

    # 新 mask
    tar_mask_pil = Image.open(target_mask_path).convert("RGB").crop((0, 0, w, h))

    # ========== 确定编辑区域 ==========
    # change_region_mask.png: RGB 图, 红色(255,0,0)=变化区域, 黑色(0,0,0)=未变
    change_bool = None
    if change_mask_path and os.path.exists(change_mask_path):
        mask_rgb = np.array(Image.open(change_mask_path).convert("RGB"))
        # 任一 RGB 通道 > 10 就认为是变化区域 (红色=变化, 黑色=未变)
        change_bool = np.any(mask_rgb > 10, axis=-1)
        print(f"  Loaded change_region from png: {change_mask_path}")
        print(f"    changed pixels: {change_bool.sum()}/{change_bool.size}")

    # fallback: 对比原mask和新mask的像素差异
    if change_bool is None:
        src_mask_rgb = np.array(Image.open(source_mask_path).convert("RGB").crop((0,0,w,h)))
        tar_mask_rgb = np.array(tar_mask_pil)
        change_bool = np.any(src_mask_rgb != tar_mask_rgb, axis=-1)
        print(f"  Computed change_region from mask diff")

    if change_bool.shape[0] != h or change_bool.shape[1] != w:
        change_bool = np.array(Image.fromarray(
            change_bool.astype(np.uint8)*255).resize((w, h), Image.NEAREST)) > 128

    change_dilated = binary_dilation(change_bool, iterations=5)
    edit_ratio = change_bool.sum() / change_bool.size * 100
    dilated_ratio = change_dilated.sum() / change_dilated.size * 100
    print(f"  编辑区域: {edit_ratio:.1f}% (膨胀后: {dilated_ratio:.1f}%)")

    if edit_ratio < 0.1:
        print(f"  编辑区域过小 (<0.1%), 跳过")
        return None

    # Debug: 保存 change_bool 可视化, 方便检查解析是否正确
    debug_change_vis = np.zeros((*change_bool.shape, 3), dtype=np.uint8)
    debug_change_vis[change_bool] = [255, 0, 0]       # 红色 = 原始变化区域
    debug_change_vis[change_dilated & ~change_bool] = [255, 165, 0]  # 橙色 = 膨胀部分
    Image.fromarray(debug_change_vis).save(os.path.join(output_dir, f"{edit_name}_debug_change_region.png"))
    print(f"  [Debug] 保存了 change_region 可视化: {edit_name}_debug_change_region.png")

    # ========== Step 1: Erase 原图 ==========
    print(f"  [Step 1] Erase 原图 (编辑区域填充灰色 {erase_value})...")
    erased_pil = erase_image(src_image, change_bool, erase_value=erase_value)

    # ========== Step 2: 6ch ControlNet 生成 ==========
    print(f"  [Step 2] 6ch ControlNet 生成 (erased + new_mask → image)...")
    generator = torch.Generator(device=device).manual_seed(seed)

    gen_pil = generate_with_6ch_controlnet(
        pipe=pipe,
        controlnet=controlnet,
        erased_pil=erased_pil,
        new_mask_pil=tar_mask_pil,
        prompt=tar_prompt,
        device=device,
        src_image_pil=src_image if strength < 1.0 else None,
        strength=strength,
        num_inference_steps=T_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=generator,
    )

    if gen_pil.size != (w, h):
        gen_pil = gen_pil.resize((w, h), Image.LANCZOS)
    gen_np = np.array(gen_pil)

    # ========== Step 3: 像素 Blending ==========
    print(f"  [Step 3] 像素 blending ({blend_mode})...")
    feathered = None
    if blend_mode == "poisson":
        blended_np = poisson_blend(src_np, gen_np, change_dilated)
    else:
        feathered = create_feathered_mask(change_dilated, feather_radius)
        blended_np = pixel_blend(src_np, gen_np, feathered)
    blended_pil = Image.fromarray(blended_np)

    # ========== 保存单独的图 ==========
    src_image.save(os.path.join(output_dir, f"{edit_name}_source.png"))
    erased_pil.save(os.path.join(output_dir, f"{edit_name}_erased.png"))
    gen_pil.save(os.path.join(output_dir, f"{edit_name}_generated_full.png"))
    blended_pil.save(os.path.join(output_dir, f"{edit_name}_edited.png"))
    tar_mask_pil.save(os.path.join(output_dir, f"{edit_name}_target_mask.png"))

    if feathered is not None:
        feathered_vis = (feathered * 255).astype(np.uint8)
        Image.fromarray(feathered_vis).save(os.path.join(output_dir, f"{edit_name}_feather_mask.png"))

    with open(os.path.join(output_dir, f"{edit_name}_config.json"), "w") as f:
        json.dump({
            "src_prompt": src_prompt, "tar_prompt": tar_prompt,
            "strength": strength, "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "blend_mode": blend_mode, "feather_radius": feather_radius,
            "edit_ratio": float(edit_ratio), "seed": seed,
            "erase_value": erase_value,
        }, f, indent=2)

    # ========== 可视化 1: 完整 3×4 对比图 ==========
    print(f"  [Vis] 生成对比图...")
    src_mask_img = Image.open(source_mask_path).convert("RGB").crop((0,0,w,h))

    # 编辑区域 overlay
    overlay = src_np.copy()
    overlay[change_dilated] = (overlay[change_dilated]*0.5 +
                                np.array([255,80,80])*0.5).astype(np.uint8)

    # 差异热力图
    diff_src_gen = create_diff_heatmap(src_np, gen_np)
    diff_src_blend = create_diff_heatmap(src_np, blended_np)
    diff_gen_blend = create_diff_heatmap(gen_np, blended_np)

    # blend 边界可视化
    boundary = change_dilated & ~binary_erosion(change_dilated, iterations=2)
    blend_boundary = blended_np.copy()
    blend_boundary[boundary] = [0, 255, 0]

    fig, axes = plt.subplots(3, 4, figsize=(32, 24))

    row1 = [
        (np.array(src_mask_img), "Source Mask"),
        (np.array(tar_mask_pil), "Target Mask (new)"),
        (overlay, "Edit Region (red)"),
        (np.array(erased_pil), "Erased Image"),
    ]
    row2 = [
        (src_np, "Source Image"),
        (gen_np, "Generated (full)"),
        (blended_np, f"Blended ({blend_mode})"),
        (blend_boundary, "Blend Boundary (green)"),
    ]
    row3 = [
        (diff_src_gen, "Diff: Source vs Generated"),
        (diff_src_blend, "Diff: Source vs Blended"),
        (diff_gen_blend, "Diff: Generated vs Blended"),
        (None, ""),
    ]

    if feathered is not None:
        feather_cm = cv2.applyColorMap((feathered * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        feather_cm = cv2.cvtColor(feather_cm, cv2.COLOR_BGR2RGB)
        row3[3] = (feather_cm, "Feather Mask")

    for row_idx, items in enumerate([row1, row2, row3]):
        for col_idx, (img, title) in enumerate(items):
            ax = axes[row_idx, col_idx]
            if img is not None:
                ax.imshow(img)
                ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')

    plt.suptitle(
        f"6ch Inpaint ControlNet: {edit_name}\n"
        f"strength={strength}, guidance={guidance_scale}, "
        f"cn_scale={controlnet_conditioning_scale}, blend={blend_mode}, "
        f"feather_r={feather_radius}, edit_area={edit_ratio:.1f}%",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{edit_name}_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ========== 可视化 2: Before / After 简洁版 ==========
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))
    axes2[0].imshow(src_np)
    axes2[0].set_title("Before (Source)", fontsize=14, fontweight='bold')
    axes2[0].axis('off')

    axes2[1].imshow(blended_np)
    axes2[1].set_title("After (Blended)", fontsize=14, fontweight='bold')
    axes2[1].axis('off')

    axes2[2].imshow(diff_src_blend)
    axes2[2].set_title("Difference Heatmap", fontsize=14, fontweight='bold')
    axes2[2].axis('off')

    plt.suptitle(f"Before vs After: {edit_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{edit_name}_before_after.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  已保存: {output_dir}")
    return blended_pil


# ============================================================
#  主函数
# ============================================================

def find_free_gpu():
    try:
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader".split()
        ).decode('utf-8')
        gpu_mem = []
        for i, line in enumerate(result.strip().split('\n')):
            used, total = map(int, line.split(','))
            gpu_mem.append((i, total - used))
        gpu_mem.sort(key=lambda x: -x[1])
        best, free = gpu_mem[0]
        print(f"自动选择 GPU {best} (空闲 {free} MiB)")
        return best
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="6ch Inpaint ControlNet + Pixel Blend")
    parser.add_argument("--mask_edit_dir", type=str, default="./mask_edit_output")
    parser.add_argument("--original_image", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True,
                        help="6ch inpaint ControlNet checkpoint 路径")
    parser.add_argument("--flux_path", type=str, default="/data/huggingface/FLUX.1-dev")
    parser.add_argument("--output_dir", type=str, default="./pixel_blend_results_6ch")
    parser.add_argument("--device_number", type=int, default=None)
    parser.add_argument("--strength", type=float, default=0.75,
                        help="SDEdit 强度: 0~1, 1.0=纯噪声生成不用SDEdit")
    parser.add_argument("--T_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blend_mode", type=str, default="feather",
                        choices=["poisson", "feather"])
    parser.add_argument("--feather_radius", type=int, default=15)
    parser.add_argument("--erase_value", type=int, default=128,
                        help="Erase 填充值, 需和训练时一致")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device_number is None:
        device_id = find_free_gpu()
    else:
        device_id = args.device_number
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ========== 加载模型 ==========
    print(">>> 加载 6ch Inpaint ControlNet...")
    controlnet = load_inpaint_controlnet(args.controlnet_path, dtype=torch.bfloat16)
    controlnet = controlnet.to(device)
    controlnet.eval()

    print(">>> 加载 FLUX pipeline...")
    pipe = FluxControlNetPipeline.from_pretrained(
        args.flux_path, controlnet=controlnet, torch_dtype=torch.bfloat16
    )
    pipe = pipe.to(device)
    print(">>> 模型加载完成")

    # ========== 查找编辑方法和文件 ==========
    edit_methods = ['boundary_deform', 'lymphocyte_infiltration', 'necrosis_expansion']

    original_mask_path = None
    for method in edit_methods:
        candidate = os.path.join(args.mask_edit_dir, method, "original_with_cells.png")
        if os.path.exists(candidate):
            original_mask_path = candidate
            break
    if original_mask_path is None:
        print(f"错误：在 {args.mask_edit_dir} 中未找到 original_with_cells.png")
        return

    print(f">>> 原始 mask: {original_mask_path}")
    print(f">>> 原始图像: {args.original_image}")
    os.makedirs(args.output_dir, exist_ok=True)

    for method in edit_methods:
        method_dir = os.path.join(args.mask_edit_dir, method)

        target_mask_path = os.path.join(method_dir, "final_with_generated_cells.png")
        if not os.path.exists(target_mask_path):
            target_mask_path = os.path.join(method_dir, "edited_with_retained_cells.png")
        if not os.path.exists(target_mask_path):
            print(f"\n跳过 {method} (未找到目标 mask)")
            continue

        change_mask_path = os.path.join(method_dir, "change_region_mask.png")
        if not os.path.exists(change_mask_path):
            change_mask_path = None

        print(f"\n{'='*60}")
        print(f"  编辑方式: {method}")
        print(f"  目标 mask: {target_mask_path}")
        print(f"  变化区域: {change_mask_path or '(从 mask diff 计算)'}")
        print(f"{'='*60}")

        run_single_edit(
            pipe=pipe,
            controlnet=controlnet,
            source_image_path=args.original_image,
            source_mask_path=original_mask_path,
            target_mask_path=target_mask_path,
            change_mask_path=change_mask_path,
            output_dir=os.path.join(args.output_dir, method),
            edit_name=method,
            device=device,
            strength=args.strength,
            T_steps=args.T_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            feather_radius=args.feather_radius,
            blend_mode=args.blend_mode,
            erase_value=args.erase_value,
            seed=args.seed,
        )

    print(f"\n>>> 全部完成！结果: {args.output_dir}")


if __name__ == "__main__":
    main()