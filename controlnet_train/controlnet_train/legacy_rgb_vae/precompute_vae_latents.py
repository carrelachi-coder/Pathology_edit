#!/usr/bin/env python
# coding=utf-8
"""
precompute_vae_latents.py
=========================
多卡并行预计算所有 image 和 mask 的 VAE latent。
自动跳过已存在的 .pt 文件 (支持断点续传)。

用法 (4卡并行):
  CUDA_VISIBLE_DEVICES=1,2,4,7 python precompute_vae_latents.py \
    --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
    --image_dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/images \
    --mask_dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/conditioning \
    --latent_dir /home/lyw/wqx-DL/flow-edit/FlowEdit-main/BCSS_dataset/latents \
    --resolution 512 \
    --batch_size 16

输出:
  latents/images/xxx.pt  (float16)
  latents/masks/xxx.pt   (float16)
"""

import argparse
import os
import torch
import torch.multiprocessing as mp
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL


def get_todo_files(file_list, output_dir):
    """过滤掉已存在的文件，返回待处理列表"""
    os.makedirs(output_dir, exist_ok=True)
    existing = set(os.listdir(output_dir))
    todo = [f for f in file_list if f.rsplit(".", 1)[0] + ".pt" not in existing]
    skipped = len(file_list) - len(todo)
    if skipped > 0:
        print(f"  跳过 {skipped} 个已存在文件, 剩余 {len(todo)} 个待处理")
    return todo


def encode_batch(vae, file_list, input_dir, output_dir, img_transforms, batch_size, device, dtype):
    """在单卡上批量 encode 并保存"""
    for i in tqdm(range(0, len(file_list), batch_size), desc=f"  GPU {device}", position=device.index or 0):
        batch_files = file_list[i : i + batch_size]

        images = []
        for fname in batch_files:
            img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
            images.append(img_transforms(img))

        batch_tensor = torch.stack(images).to(device=device, dtype=dtype)

        with torch.no_grad():
            latents = vae.encode(batch_tensor).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

        for j, fname in enumerate(batch_files):
            out_name = fname.rsplit(".", 1)[0] + ".pt"
            torch.save(latents[j].cpu().half(), os.path.join(output_dir, out_name))

    torch.cuda.empty_cache()


def worker(rank, num_gpus, args, image_todo, mask_todo, image_transforms, mask_transforms):
    """单卡 worker 进程"""
    device = torch.device(f"cuda:{rank}")

    # 加载 VAE 到当前卡
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.eval()
    vae.requires_grad_(False)

    # 按 rank 分片
    def shard(file_list, rank, num_gpus):
        return [f for i, f in enumerate(file_list) if i % num_gpus == rank]

    my_images = shard(image_todo, rank, num_gpus)
    my_masks = shard(mask_todo, rank, num_gpus)

    print(f"[GPU {rank}] Images: {len(my_images)}, Masks: {len(my_masks)}")

    # Encode images
    if my_images:
        encode_batch(
            vae, my_images, args.image_dir,
            os.path.join(args.latent_dir, "images"),
            image_transforms, args.batch_size, device, torch.bfloat16,
        )

    # Encode masks
    if my_masks:
        encode_batch(
            vae, my_masks, args.mask_dir,
            os.path.join(args.latent_dir, "masks"),
            mask_transforms, args.batch_size, device, torch.bfloat16,
        )

    print(f"[GPU {rank}] Done!")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU precompute VAE latents")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args()

    # 检测可用 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")

    # Transforms
    image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # 收集文件
    image_files = sorted([
        f for f in os.listdir(args.image_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    mask_files = sorted([
        f for f in os.listdir(args.mask_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ])
    print(f"Total images: {len(image_files)}, masks: {len(mask_files)}")

    # 过滤已存在的
    image_todo = get_todo_files(image_files, os.path.join(args.latent_dir, "images"))
    mask_todo = get_todo_files(mask_files, os.path.join(args.latent_dir, "masks"))

    total_todo = len(image_todo) + len(mask_todo)
    if total_todo == 0:
        print("\n所有 latent 已存在, 无需计算!")
        return

    print(f"\n待计算: {len(image_todo)} images + {len(mask_todo)} masks = {total_todo} total")
    print(f"使用 {num_gpus} 张 GPU 并行计算, 每卡约 {total_todo // num_gpus} 个文件\n")

    # 单卡直接跑, 多卡用 multiprocessing
    if num_gpus == 1:
        worker(0, 1, args, image_todo, mask_todo, image_transforms, mask_transforms)
    else:
        mp.spawn(
            worker,
            args=(num_gpus, args, image_todo, mask_todo, image_transforms, mask_transforms),
            nprocs=num_gpus,
            join=True,
        )

    # 统计结果
    img_dir = os.path.join(args.latent_dir, "images")
    mask_dir = os.path.join(args.latent_dir, "masks")
    n_img = len([f for f in os.listdir(img_dir) if f.endswith(".pt")])
    n_mask = len([f for f in os.listdir(mask_dir) if f.endswith(".pt")])

    total_bytes = 0
    for d in [img_dir, mask_dir]:
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Image latents: {n_img}")
    print(f"  Mask latents:  {n_mask}")
    print(f"  Total size:    {total_bytes / 1024**3:.2f} GB")
    print(f"  Saved to:      {args.latent_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()