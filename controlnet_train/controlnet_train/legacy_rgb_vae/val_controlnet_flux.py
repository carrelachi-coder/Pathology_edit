"""
validate_inpaint_controlnet.py
==============================
验证 6ch Inpainting ControlNet (擦除图 + mask → 生成图)

和原版 validate_controlnet.py 的区别:
  - ControlNet 输入从 3ch (mask) 变成 6ch (erased_image + mask)
  - 不能直接用 pipe(control_image=...), 需要手动构造 6ch 输入
  - 从 val jsonl 读取 erased_image 和 mask_image 两个字段
  - 支持 CellViT 分析

用法:
  python /home/lyw/wqx-DL/flow-edit/FlowEdit-main/controlnet_train/val_controlnet_flux.py\
      --controlnet_path /data/huggingface/pathology_edit/inpaint_controlnet_output/checkpoint-8000/flux_controlnet \
      --flux_path /data/huggingface/FLUX.1-dev \
      --val_jsonl /data/huggingface/pathology_edit/inpaint_dataset/metadata_val.jsonl \
      --output_dir /data/huggingface/pathology_edit/inpaint_val_results \
      --num_samples 10 \
      --device cuda:1
"""

import argparse
import json
import os
import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms as T

from diffusers import FluxPipeline
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

import sys
CELLVIT_PATH = "/data/huggingface/pathology_edit/CellViT/CellViT-plus-plus-main"
sys.path.insert(0, CELLVIT_PATH)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# CellViT 颜色映射
NUCLEI_COLORS = np.array([
    [0, 0, 0],       # 0: Background
    [255, 0, 0],     # 1: Neoplastic
    [0, 255, 0],     # 2: Inflammatory
    [0, 0, 255],     # 3: Connective
    [255, 255, 0],   # 4: Dead
    [255, 0, 255],   # 5: Epithelial
], dtype=np.uint8)


from safetensors.torch import load_file
import glob, json
def load_inpaint_controlnet(controlnet_path, dtype=torch.bfloat16):
    """手动加载 6ch inpaint controlnet"""
    with open(os.path.join(controlnet_path, "config.json")) as f:
        config = json.load(f)
    
    # 不要改 in_channels，保持 64 让 x_embedder 能匹配
    # controlnet_x_embedder 的 128 维会自动对上
    
    for k in ["_class_name", "_diffusers_version", "_name_or_path"]:
        config.pop(k, None)
    
    controlnet = FluxControlNetModel(**config)
    
    # 手动把 controlnet_x_embedder 扩展到 128
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
    
    state_dict = {}
    for f in shard_files:
        if f.endswith(".safetensors"):
            state_dict.update(load_file(f))
        else:
            state_dict.update(torch.load(f, map_location="cpu"))
    
    # strict=True 现在应该能过了
    controlnet.load_state_dict(state_dict, strict=True)
    controlnet = controlnet.to(dtype=dtype)
    return controlnet

def init_cellvit(model_path, device):
    print(f"Initializing CellViT from {model_path}...")
    from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    model = CellViTSAM(
        model_path=None,
        num_nuclei_classes=config["data.num_nuclei_classes"],
        num_tissue_classes=config["data.num_tissue_classes"],
        vit_structure="SAM-H",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def get_cellvit_mask(model, config, pil_img, device):
    if pil_img.size != (1024, 1024):
        pil_img = pil_img.resize((1024, 1024), Image.LANCZOS)
    img = np.array(pil_img.convert("RGB"))
    mean = np.array(config["transformations.normalize.mean"])
    std = np.array(config["transformations.normalize.std"])
    img_norm = (img.astype(np.float32) / 255.0 - mean) / std
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(img_tensor)
            nuclei_type = torch.argmax(outputs["nuclei_type_map"], dim=1)[0].cpu().numpy()
    vis_mask = NUCLEI_COLORS[nuclei_type]
    vis_mask_pil = Image.fromarray(vis_mask).resize((512, 512), Image.NEAREST)
    return np.array(vis_mask_pil)


def load_val_data(jsonl_path, num_samples=None):
    entries = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    return entries[:num_samples]


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.16):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


# ============================================================
#  6ch ControlNet 推理
# ============================================================

@torch.no_grad()
def generate_with_inpaint_controlnet(
    pipe, controlnet, entry, device,
    num_inference_steps=28, guidance_scale=3.5,
    controlnet_conditioning_scale=1.0, generator=None,
):
    """
    用 6ch inpainting ControlNet 生成图像。
    手动构造 6ch control_image 并走去噪循环。
    """
    dtype = torch.bfloat16

    # 加载输入
    erased_img = Image.open(entry["erased_image"]).convert("RGB")
    mask_img = Image.open(entry["mask_image"]).convert("RGB")
    prompt = entry["text"]

    # 确保尺寸一致且是 16 的倍数
    w, h = erased_img.size
    w, h = w - w % 16, h - h % 16
    erased_img = erased_img.crop((0, 0, w, h))
    mask_img = mask_img.crop((0, 0, w, h))

    # Encode prompt
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=device,
    )

    # 准备 control_image: 分别 VAE encode 再 concat
    img_transform = T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(), T.Normalize([0.5], [0.5]),
    ])

    erased_tensor = img_transform(erased_img).unsqueeze(0).to(device, dtype=dtype)
    mask_tensor = img_transform(mask_img).unsqueeze(0).to(device, dtype=dtype)

    erased_latents = pipe.vae.encode(erased_tensor).latent_dist.sample()
    erased_latents = (erased_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    mask_latents = pipe.vae.encode(mask_tensor).latent_dist.sample()
    mask_latents = (mask_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    # concat → 6ch latent
    control_latents = torch.cat([erased_latents, mask_latents], dim=1)

    num_channels_latents = pipe.transformer.config.in_channels // 4
    h_lat, w_lat = erased_latents.shape[2], erased_latents.shape[3]

    # pack control
    control_image = FluxControlNetPipeline._pack_latents(
        control_latents, 1, num_channels_latents * 2, h_lat, w_lat,
    )

    # 准备 latents (随机噪声)
    latents, latent_image_ids = pipe.prepare_latents(
        1, num_channels_latents, h, w,
        prompt_embeds.dtype, device, generator, None,
    )

    # Timesteps
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

    controlnet_blocks_repeat = False if controlnet.input_hint_block is None else True

    # 去噪循环
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        # guidance
        use_guidance = controlnet.config.guidance_embeds
        guidance = torch.tensor([guidance_scale], device=device) if use_guidance else None
        if guidance is not None:
            guidance = guidance.expand(latents.shape[0])

        # ControlNet
        cn_blocks, cn_single = controlnet(
            hidden_states=latents,
            controlnet_cond=control_image,
            controlnet_mode=None,
            conditioning_scale=controlnet_conditioning_scale,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )

        # guidance for transformer
        guidance_t = torch.tensor([guidance_scale], device=device) if pipe.transformer.config.guidance_embeds else None
        if guidance_t is not None:
            guidance_t = guidance_t.expand(latents.shape[0])

        # Transformer
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
            joint_attention_kwargs=None,
            return_dict=False,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )[0]

        # Scheduler step
        latents_dtype = latents.dtype
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    # Decode
    latents = pipe._unpack_latents(latents, h, w, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    return image


# ============================================================
#  主函数
# ============================================================

def generate_images(entries, controlnet_path, flux_path, output_dir, device,
                    model_name, cellvit_model=None, cellvit_config=None):
    """加载模型并生成图像"""

    print(f"\n--- [{model_name}] Loading ControlNet from {controlnet_path} ---")
    controlnet = load_inpaint_controlnet(controlnet_path, dtype=torch.bfloat16)
    # === INPAINT PATCH: 修改 x_embedder 维度 (和训练一致) ===
    old_x = controlnet.x_embedder
    old_in = old_x.in_features
    new_in = old_in * 2
    if old_x.in_features != new_in:
        # 检查是否已经是 128 维 (saved checkpoint 可能已经是 128)
        logger.info(f"x_embedder already has {old_in} features, checking if patch needed...")
    # 如果保存的 checkpoint 已经是 128 维就不需要 patch
    # 如果是 64 维则需要 patch (但正常 save/load 应该已经是 128)
    # === END PATCH ===

    pipe = FluxControlNetPipeline.from_pretrained(
        flux_path, controlnet=controlnet, torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    controlnet.eval()

    gen_dir = os.path.join(output_dir, "generated")
    gt_dir = os.path.join(output_dir, "ground_truth")
    erased_dir = os.path.join(output_dir, "erased")
    mask_dir = os.path.join(output_dir, "mask")
    cv_dir = os.path.join(output_dir, "cellvit_analysis")
    for d in [gen_dir, gt_dir, erased_dir, mask_dir, cv_dir]:
        os.makedirs(d, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(42)

    for i, entry in enumerate(tqdm(entries, desc=f"Generating {model_name}")):
        # 从 jsonl 中获取文件名
        gt_name = Path(entry["image"]).stem
        patch_name = f"{gt_name}"

        # 生成
        result = generate_with_inpaint_controlnet(
            pipe, controlnet, entry, device,
            num_inference_steps=28, guidance_scale=3.5,
            controlnet_conditioning_scale=1.0, generator=generator,
        )

        # 保存
        result.save(os.path.join(gen_dir, f"{patch_name}.png"))
        Image.open(entry["image"]).convert("RGB").save(os.path.join(gt_dir, f"{patch_name}.png"))
        Image.open(entry["erased_image"]).convert("RGB").save(os.path.join(erased_dir, f"{patch_name}.png"))
        Image.open(entry["mask_image"]).convert("RGB").save(os.path.join(mask_dir, f"{patch_name}.png"))

        if cellvit_model is not None:
            cv_vis = get_cellvit_mask(cellvit_model, cellvit_config, result, device)
            Image.fromarray(cv_vis).save(os.path.join(cv_dir, f"CV_{patch_name}.png"))

    del pipe
    torch.cuda.empty_cache()
    return gen_dir, gt_dir, erased_dir, mask_dir, cv_dir


def create_comparison_grid(gen_dir, gt_dir, erased_dir, mask_dir, cv_dir,
                            output_path, num_samples=8):
    """5 列对比: erased | mask | generated | cellvit | GT"""
    import matplotlib.pyplot as plt

    files = sorted(os.listdir(gen_dir))[:num_samples]

    fig, axes = plt.subplots(num_samples, 5, figsize=(25, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Erased Input", "Mask", "Generated", "CellViT", "Ground Truth"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14)

    for i, fname in enumerate(files):
        erased = np.array(Image.open(os.path.join(erased_dir, fname)))
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        gen = np.array(Image.open(os.path.join(gen_dir, fname)))
        gt = np.array(Image.open(os.path.join(gt_dir, fname)))

        cv_path = os.path.join(cv_dir, f"CV_{fname}")
        cv = np.array(Image.open(cv_path)) if os.path.exists(cv_path) else np.zeros_like(gen)

        for j, img in enumerate([erased, mask, gen, cv, gt]):
            axes[i, j].imshow(img)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved to {output_path}")


def compute_pixel_metrics(gen_dir, gt_dir):
    """SSIM + LPIPS"""
    import lpips
    from skimage.metrics import structural_similarity as ssim

    loss_fn = lpips.LPIPS(net='alex').cuda()
    gen_files = sorted(os.listdir(gen_dir))
    gt_files = sorted(os.listdir(gt_dir))

    ssim_scores, lpips_scores = [], []
    for gf, gtf in tqdm(zip(gen_files, gt_files), total=len(gen_files), desc="Metrics"):
        gen_img = np.array(Image.open(os.path.join(gen_dir, gf)).convert("RGB"))
        gt_img = np.array(Image.open(os.path.join(gt_dir, gtf)).convert("RGB"))

        s = ssim(gt_img, gen_img, channel_axis=2, data_range=255)
        ssim_scores.append(s)

        gen_t = torch.from_numpy(gen_img).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0
        gt_t = torch.from_numpy(gt_img).permute(2,0,1).unsqueeze(0).float() / 127.5 - 1.0
        with torch.no_grad():
            lp = loss_fn(gen_t.cuda(), gt_t.cuda()).item()
        lpips_scores.append(lp)

    return {
        "SSIM_mean": float(np.mean(ssim_scores)), "SSIM_std": float(np.std(ssim_scores)),
        "LPIPS_mean": float(np.mean(lpips_scores)), "LPIPS_std": float(np.std(lpips_scores)),
    }


def compute_fid_kid(gen_dir, gt_dir):
    try:
        import torch_fidelity
        metrics = torch_fidelity.calculate_metrics(
            input1=gen_dir, input2=gt_dir, cuda=True,
            fid=True, kid=True,
            kid_subset_size=min(100, len(os.listdir(gen_dir))),
            verbose=False,
        )
        return {
            "FID": metrics.get("frechet_inception_distance", None),
            "KID_mean": metrics.get("kernel_inception_distance_mean", None),
            "KID_std": metrics.get("kernel_inception_distance_std", None),
        }
    except ImportError:
        print("torch-fidelity not installed, skipping FID/KID")
        return {"FID": None, "KID_mean": None, "KID_std": None}


def main():
    parser = argparse.ArgumentParser(description="Validate 6ch Inpainting ControlNet")
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--flux_path", type=str, default="/data/huggingface/FLUX.1-dev")
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cellvit_checkpoint", type=str,
                        default="/data/huggingface/pathology_edit/CellViT/CellViT-plus-plus-main/checkpoints/CellViT-SAM-H-x40-AMP-001.pth")
    parser.add_argument("--skip_cellvit", action="store_true")
    parser.add_argument("--compute_metrics", action="store_true")
    args = parser.parse_args()

    # CellViT
    cv_model, cv_config = None, None
    if not args.skip_cellvit:
        try:
            cv_model, cv_config = init_cellvit(args.cellvit_checkpoint, args.device)
            print("CellViT loaded")
        except Exception as e:
            print(f"CellViT 加载失败 ({e}), 跳过")

    os.makedirs(args.output_dir, exist_ok=True)
    entries = load_val_data(args.val_jsonl, args.num_samples)
    print(f"Loaded {len(entries)} validation samples")

    # 查找 checkpoints
    checkpoints = {}
    if os.path.exists(os.path.join(args.controlnet_path, "config.json")):
        # 直接是 controlnet 目录
        checkpoints["model"] = args.controlnet_path
    else:
        # 是训练输出目录，找 checkpoint-*/flux_controlnet
        for d in sorted(os.listdir(args.controlnet_path)):
            if d.startswith("checkpoint-"):
                cn_path = os.path.join(args.controlnet_path, d, "flux_controlnet")
                if os.path.exists(cn_path):
                    checkpoints[d] = cn_path

    if not checkpoints:
        print(f"未找到 checkpoint: {args.controlnet_path}")
        return

    print(f"找到 {len(checkpoints)} 个 checkpoint: {list(checkpoints.keys())}")

    for name, path in checkpoints.items():
        print(f"\n>>>> [{name}] <<<<")
        current_out = os.path.join(args.output_dir, name)

        gen_dir, gt_dir, erased_dir, mask_dir, cv_dir = generate_images(
            entries, path, args.flux_path, current_out, args.device,
            model_name=name, cellvit_model=cv_model, cellvit_config=cv_config,
        )

        # 对比图
        grid_path = os.path.join(args.output_dir, f"grid_{name}.png")
        create_comparison_grid(gen_dir, gt_dir, erased_dir, mask_dir, cv_dir,
                                grid_path, num_samples=min(8, args.num_samples))

        # 指标
        if args.compute_metrics:
            print("Computing metrics...")
            pixel_metrics = compute_pixel_metrics(gen_dir, gt_dir)
            fid_metrics = compute_fid_kid(gen_dir, gt_dir)
            all_metrics = {**pixel_metrics, **fid_metrics}
            print(f"  {all_metrics}")

            with open(os.path.join(current_out, "metrics.json"), "w") as f:
                json.dump(all_metrics, f, indent=2)

    print("\n完成！")


if __name__ == "__main__":
    main()