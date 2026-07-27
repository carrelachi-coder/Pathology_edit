#!/usr/bin/env python3
"""Resumable batch inference worker for the non-editing generation baselines."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import traceback

import numpy as np
from PIL import Image, PngImagePlugin
import torch


MODEL_TYPES = (
    "pixcell",
    "pathdiff",
    "pathldm",
    "unipath",
    "mupad_text",
    "mupad_image",
)

PATHDIFF_SCALE_PATTERN = re.compile(
    r"(?i)(?:\b(?:magnification|objective|zoom|mpp)\b|\d+(?:\.\d+)?\s*(?:x\b|×))"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--pipeline-dir", type=Path)
    parser.add_argument("--vae-dir", type=Path)
    parser.add_argument("--uni-root", type=Path)
    parser.add_argument("--repo-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--conic-root", type=Path)
    parser.add_argument("--mupad-context-root", type=Path)
    parser.add_argument("--exclude-sample-ids", type=Path)
    parser.add_argument(
        "--pathdiff-mode", choices=("both", "t2i"), default="both"
    )
    parser.add_argument("--pathdiff-text-objective-magnification", type=float)
    parser.add_argument("--rag-root", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--guidance-high", type=float)
    parser.add_argument("--guidance-low", type=float)
    parser.add_argument(
        "--pathldm-tumor-level", choices=("low", "high")
    )
    parser.add_argument(
        "--pathldm-til-level", choices=("low", "high")
    )
    parser.add_argument("--native-mpp", type=float)
    parser.add_argument("--native-resolution", type=int, required=True)
    parser.add_argument("--source-mpp", type=float, default=0.25)
    parser.add_argument("--organs", nargs="+")
    rag_group = parser.add_mutually_exclusive_group()
    rag_group.add_argument("--use-rag", dest="use_rag", action="store_true")
    rag_group.add_argument("--no-use-rag", dest="use_rag", action="store_false")
    parser.set_defaults(use_rag=True)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise TypeError(f"Unsupported manifest structure: {path}")
    return records


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_path(path: Path | None, flag: str) -> Path:
    if path is None:
        raise ValueError(f"{flag} is required")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def nominal_mpp_for_objective(magnification: float) -> float:
    if magnification <= 0:
        raise ValueError("objective magnification must be positive")
    return 10.0 / float(magnification)


def pathdiff_text_effective_prompt(
    prompt: str, objective_magnification: float | None
) -> tuple[str, dict]:
    original_prompt = prompt.strip()
    if not original_prompt:
        raise ValueError("PathDiff text prompt cannot be empty")
    if objective_magnification is None:
        return original_prompt, {
            "kind": "unconstrained_pathcap_caption",
            "physical_scale_status": "unknown_pathcap_no_mpp_normalization",
        }
    if PATHDIFF_SCALE_PATTERN.search(original_prompt):
        raise ValueError(
            "PathDiff source prompt already contains a scale term; "
            "refuse to add a potentially conflicting objective magnification"
        )
    magnification_text = f"{float(objective_magnification):g}x"
    nominal_mpp = nominal_mpp_for_objective(objective_magnification)
    scale_prefix = (
        "H&E-stained histopathology at "
        f"{magnification_text} objective magnification."
    )
    return f"{scale_prefix} {original_prompt}", {
        "kind": "prompt_conditioned_objective_magnification",
        "objective_magnification": float(objective_magnification),
        "nominal_mpp": nominal_mpp,
        "prompt_position": "prefix",
        "scale_prompt": scale_prefix,
        "physical_scale_status": "prompt_conditioned_nominal_scale",
    }


def image_stats(image: Image.Image) -> dict:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pixels = array.reshape(-1, 3)
    return {
        "size": list(image.size),
        "mean": [round(float(value), 3) for value in pixels.mean(axis=0)],
        "std": [round(float(value), 3) for value in pixels.std(axis=0)],
        "min": [int(value) for value in pixels.min(axis=0)],
        "max": [int(value) for value in pixels.max(axis=0)],
    }


def to_pil(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, list):
        return to_pil(value[0])
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu()
        if value.ndim == 4:
            value = value[0]
        if value.ndim == 3 and value.shape[0] in (1, 3):
            value = value.permute(1, 2, 0)
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


class PixCellAdapter:
    allowed_inputs = ("reference_image", "target_nuclei_mask")

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from diffusers import AutoencoderKL, DiffusionPipeline
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.args = args
        self.device = device
        self.dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_dir = require_path(args.model_dir, "--model-dir")
        pipeline_dir = require_path(args.pipeline_dir, "--pipeline-dir")
        vae_dir = require_path(args.vae_dir, "--vae-dir")
        uni_root = require_path(args.uni_root, "--uni-root")

        vae = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=self.dtype)
        sys.modules.setdefault(
            "diffusers.models.controlnet",
            importlib.import_module("diffusers.models.controlnets.controlnet"),
        )
        sys.path[:0] = [
            str(model_dir / "controlnet"),
            str(model_dir / "transformer"),
            str(pipeline_dir),
        ]
        self.pipe = DiffusionPipeline.from_pretrained(
            str(model_dir),
            vae=vae,
            custom_pipeline=str(pipeline_dir),
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(device)

        model_kwargs = dict(
            model_name="vit_giant_patch14_224",
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        self.uni_model = timm.create_model(pretrained=False, **model_kwargs)
        state = torch.load(
            str(uni_root / "pytorch_model.bin"),
            map_location="cpu",
            weights_only=False,
        )
        self.uni_model.load_state_dict(state, strict=True)
        config_path = uni_root / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.uni_model.pretrained_cfg.update(config.get("pretrained_cfg", {}))
        transform_config = resolve_data_config(
            self.uni_model.pretrained_cfg, model=self.uni_model
        )
        self.transform = create_transform(**transform_config)
        self.uni_model.eval().to(device)

    @staticmethod
    def load_binary_mask(path: str) -> tuple[np.ndarray, list[int]]:
        raw = np.asarray(Image.open(path))
        if raw.ndim == 3:
            raw = raw[..., 0]
        if raw.shape != (512, 512):
            raise ValueError(f"Expected a 512x512 mask at 0.25 MPP, got {raw.shape}")
        unique_ids = sorted(int(value) for value in np.unique(raw))
        allowed = {0, 1, 2, 3, 4, 5, 101, 102, 103, 104, 105, 255}
        unknown = set(unique_ids) - allowed
        if unknown:
            raise ValueError(f"Unexpected CellViT mask IDs: {sorted(unknown)}")
        binary = (raw > 0).astype(np.uint8) * 255
        binary = np.asarray(
            Image.fromarray(binary).resize((256, 256), Image.Resampling.NEAREST)
        )
        return np.repeat(binary[..., None], 3, axis=2), unique_ids

    def generate(self, record: dict) -> tuple[Image.Image, dict]:
        image = Image.open(record["reference_image"]).convert("RGB")
        if image.size != (512, 512):
            raise ValueError(
                f"Expected a 512x512 reference at 0.25 MPP, got {image.size}"
            )
        image = image.resize((256, 256), Image.Resampling.BICUBIC)
        mask, mask_ids = self.load_binary_mask(record["target_nuclei_mask"])
        with torch.inference_mode():
            embedding = self.uni_model(
                self.transform(image).unsqueeze(0).to(self.device)
            ).unsqueeze(1)
        uncond = self.pipe.get_unconditional_embedding(embedding.shape[0])
        generator = torch.Generator(device=self.device).manual_seed(record["seed"])
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", enabled=self.device.type == "cuda"
        ):
            generated = self.pipe(
                uni_embeds=embedding,
                controlnet_input=mask,
                negative_uni_embeds=uncond,
                guidance_scale=self.args.guidance,
                num_inference_steps=self.args.steps,
                num_images_per_prompt=1,
                generator=generator,
            ).images[0]
        return generated.convert("RGB"), {
            "source_mask_ids": mask_ids,
            "binary_mask_foreground_fraction": float(np.count_nonzero(mask[..., 0]))
            / mask[..., 0].size,
            "reference_preprocessing": {
                "operation": "full_field_downsample",
                "source_resolution": [512, 512],
                "source_mpp": self.args.source_mpp,
                "model_resolution": [256, 256],
                "model_mpp": self.args.native_mpp,
            },
            "target_mask_preprocessing": {
                "operation": "full_field_nearest_downsample",
                "source_resolution": [512, 512],
                "source_mpp": self.args.source_mpp,
                "model_resolution": [256, 256],
                "model_mpp": self.args.native_mpp,
                "crop_applied": False,
            },
        }


class PathDiffAdapter:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        repo_dir = require_path(args.repo_dir, "--repo-dir")
        config = require_path(args.config, "--config")
        checkpoint = require_path(args.checkpoint, "--checkpoint")
        self.args = args
        self.device = device
        self.mode = args.pathdiff_mode
        self.text_objective_magnification = (
            args.pathdiff_text_objective_magnification
        )
        if self.mode == "t2i":
            self.allowed_inputs = ("prompt",)
            self.conic_root = None
        else:
            self.allowed_inputs = ("prompt", "target_conic_instance_type_mask")
            self.conic_root = require_path(args.conic_root, "--conic-root")
        sys.path.insert(0, str(repo_dir))
        os.chdir(repo_dir)
        import sampling
        from ldm.data.mask_cond.mask_condition import NULL_MASK

        self.null_mask_value = int(NULL_MASK)
        if self.null_mask_value != 10:
            raise RuntimeError(
                "PathDiff official text-only NULL_MASK changed: "
                f"expected 10, got {self.null_mask_value}"
            )

        self.sampling = sampling
        self.model = sampling.get_model(str(config), str(device), str(checkpoint))
        payload = torch.load(str(checkpoint), map_location="cpu")
        checkpoint_state = payload.get("state_dict", payload)
        model_keys = set(self.model.state_dict())
        checkpoint_keys = set(checkpoint_state)
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        if missing or unexpected:
            raise RuntimeError(
                "PathDiff checkpoint/config mismatch: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        del checkpoint_state, payload
        sampling.device = str(device)
        sampling.sampler = sampling.DDIMSampler(self.model)

    def generate(self, record: dict) -> tuple[Image.Image, dict]:
        if self.mode == "t2i":
            effective_prompt, scale_condition = pathdiff_text_effective_prompt(
                record["prompt"], self.text_objective_magnification
            )
            # Official sample_one ignores this placeholder before constructing
            # its 256x256x6 NULL_MASK tensor for mode='t2i'.
            placeholder = np.zeros((256, 256), dtype=np.uint8)
            generated, mask_preview, _ = self.sampling.sample_one(
                self.model,
                placeholder,
                effective_prompt,
                number_of_steps=self.args.steps,
                unconditional_guidance_scale=self.args.guidance,
                mode="t2i",
            )
            return Image.fromarray(generated).convert("RGB"), {
                "condition_preview": mask_preview,
                "pathdiff_inference_mode": "t2i",
                "pathdiff_official_entrypoint": "sampling.py::sample_one(mode='t2i')",
                "pathdiff_control_condition": {
                    "kind": "official_null_mask",
                    "value": self.null_mask_value,
                    "shape": [256, 256, 6],
                },
                "original_prompt": record["prompt"],
                "effective_prompt": effective_prompt,
                "pathdiff_text_scale_condition": scale_condition,
                "native_scale_status": scale_condition["physical_scale_status"],
            }

        assert self.conic_root is not None
        conic_path = self.conic_root / record["target_annotation_id"] / "conic.npy"
        conic = np.load(conic_path)
        if conic.shape != (256, 256, 2):
            raise ValueError(f"Unexpected CoNIC shape {conic.shape}: {conic_path}")
        instance_mask = conic[..., 0]
        type_mask = conic[..., 1]
        labels = sorted(int(value) for value in np.unique(type_mask))
        if not set(labels).issubset(set(range(7))):
            raise ValueError(f"Unexpected CoNIC type IDs {labels}: {conic_path}")
        edges = self.sampling.get_edges(instance_mask)
        generated, mask_preview, _ = self.sampling.sample_one(
            self.model,
            type_mask,
            record["prompt"],
            inst_mask=edges,
            number_of_steps=self.args.steps,
            unconditional_guidance_scale=self.args.guidance,
            mode="",
        )
        return Image.fromarray(generated).convert("RGB"), {
            "pathdiff_inference_mode": "both",
            "conic_mask": str(conic_path),
            "conic_type_ids": labels,
            "instance_count": int(instance_mask.max()),
            "condition_preview": mask_preview,
            "target_condition_preprocessing": {
                "operation": "full_field_downsample_then_conic_segmentation",
                "source_resolution": [512, 512],
                "source_mpp": self.args.source_mpp,
                "model_resolution": [256, 256],
                "model_mpp": self.args.native_mpp,
                "crop_applied": False,
            },
        }


class PathLDMAdapter:
    allowed_inputs = ("prompt",)

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        repo_dir = require_path(args.repo_dir, "--repo-dir")
        config_path = require_path(args.config, "--config")
        checkpoint = require_path(args.checkpoint, "--checkpoint")
        sys.path.insert(0, str(repo_dir))
        from ldm.models.diffusion.ddim import DDIMSampler
        from ldm.util import instantiate_from_config
        from omegaconf import OmegaConf

        config = OmegaConf.load(str(config_path))
        for branch in ("first_stage_config", "unet_config"):
            try:
                del config["model"]["params"][branch]["params"]["ckpt_path"]
            except Exception:
                pass
        payload = torch.load(str(checkpoint), map_location="cpu")
        state = payload.get("state_dict", payload)
        self.model = instantiate_from_config(config.model)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "PathLDM checkpoint mismatch: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.model.eval().to(device)
        self.sampler = DDIMSampler(self.model)
        self.args = args

    def conditioning_prompt(self, record: dict) -> tuple[str, dict]:
        tumor_level = record.get(
            "pathldm_tumor_level", self.args.pathldm_tumor_level
        )
        til_level = record.get("pathldm_til_level", self.args.pathldm_til_level)
        valid_levels = {"low", "high"}
        if tumor_level not in valid_levels or til_level not in valid_levels:
            raise ValueError(
                "PathLDM requires low/high tumor and TIL levels; got "
                f"tumor={tumor_level!r}, TIL={til_level!r}"
            )
        prefix = f"{tumor_level.capitalize()} tumor; {til_level} TIL;"
        return prefix + record["prompt"], {
            "pathldm_conditioning_prefix": prefix,
            "pathldm_tumor_level": tumor_level,
            "pathldm_til_level": til_level,
        }

    def generate(self, record: dict) -> tuple[Image.Image, dict]:
        effective_prompt, conditioning_metadata = self.conditioning_prompt(record)
        with torch.no_grad():
            unconditional = self.model.get_learned_conditioning([""])
            conditioning = self.model.get_learned_conditioning([effective_prompt])
            samples, _ = self.sampler.sample(
                self.args.steps,
                1,
                [3, 64, 64],
                conditioning,
                verbose=False,
                unconditional_guidance_scale=self.args.guidance,
                unconditional_conditioning=unconditional,
                eta=0,
            )
            decoded = self.model.decode_first_stage(samples)
            decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
        return to_pil(decoded[0]), {
            **conditioning_metadata,
            "effective_prompt": effective_prompt,
        }


class UniPathAdapter:
    allowed_inputs = ("prompt",)

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        repo_dir = require_path(args.repo_dir, "--repo-dir")
        model_dir = require_path(args.model_dir, "--model-dir")
        sys.path.insert(0, str(repo_dir / "src"))
        from unipath.conversation import conv_templates
        from unipath.model.builder import load_pretrained_model
        from unipath.utils import disable_torch_init

        disable_torch_init()
        self.conv_templates = conv_templates
        self.tokenizer, self.model, _ = load_pretrained_model(str(model_dir))
        original_sample_images = self.model.sample_images
        steps = args.steps
        guidance = args.guidance

        def sample_images_with_steps(*positional, **kwargs):
            kwargs.setdefault("num_inference_steps", steps)
            kwargs.setdefault("guidance_scale", guidance)
            return original_sample_images(*positional, **kwargs)

        self.model.sample_images = sample_images_with_steps
        self.retriever = None
        if args.use_rag:
            rag_root = require_path(args.rag_root, "--rag-root")
            from unipath.retrieval import MultiModalRetriever

            self.retriever = MultiModalRetriever(
                h5_file=str(rag_root / "selected_8k.h5"),
                vocab_file=str(rag_root / "llm_filtered_vocab_gemini_pro.txt"),
                inverted_index_file=str(rag_root / "keyword_inverted_index.json"),
                image_dir=str(rag_root / "images"),
                device=str(device),
                load_conch=True,
            )

    def generate(self, record: dict) -> tuple[Image.Image, dict]:
        conversation = self.conv_templates["qwen"].copy()
        conversation.append_message(
            conversation.roles[0], f"Analysis of my description: {record['prompt']}"
        )
        conversation.append_message(conversation.roles[1], None)
        image = self.model.generate_image(
            text=[conversation.get_prompt()],
            user_text=[record["prompt"]],
            tokenizer=self.tokenizer,
            retriever=self.retriever,
        )
        return to_pil(image), {"use_rag": self.retriever is not None}


class MuPaDAdapter:
    def __init__(
        self, args: argparse.Namespace, device: torch.device, modality: str
    ) -> None:
        model_dir = require_path(args.model_dir, "--model-dir")
        sys.path.insert(0, str(model_dir))
        from pipeline import SiTPipeline

        self.pipe = SiTPipeline.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        self.pipe.to(device)
        self.args = args
        self.modality = modality
        self.allowed_inputs = (
            ("prompt",) if modality == "text" else ("reference_wsi_context",)
        )
        self.context_root = None
        if modality == "image":
            self.context_root = require_path(
                args.mupad_context_root, "--mupad-context-root"
            )

    def load_image_condition(self, record: dict) -> tuple[Image.Image, dict]:
        assert self.context_root is not None
        sample_dir = (
            self.context_root
            / record["organ"]
            / record["reference_annotation_id"]
        )
        image_path = sample_dir / "context.png"
        metadata_path = sample_dir / "metadata.json"
        if not image_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                f"missing real WSI context for {record['reference_annotation_id']}: "
                f"{sample_dir}"
            )
        PngImagePlugin.MAX_TEXT_CHUNK = max(
            PngImagePlugin.MAX_TEXT_CHUNK, 64 * 1024 * 1024
        )
        image = Image.open(image_path).convert("RGB")
        if image.size != (self.args.native_resolution, self.args.native_resolution):
            raise ValueError(
                f"unexpected MuPaD WSI context size {image.size}: {image_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        central_verified = metadata.get(
            "central_reference_verified", metadata.get("central_reference_exact")
        )
        if central_verified is not True:
            raise ValueError(f"WSI context failed central reference check: {metadata_path}")
        metadata["central_reference_verified"] = central_verified
        if metadata.get("target_overlap") is not False:
            raise ValueError(f"WSI context overlaps target patch: {metadata_path}")
        if metadata.get("context_operation") != (
            "real_wsi_centered_crop_then_downsample"
        ):
            raise ValueError(f"unexpected WSI context operation: {metadata_path}")
        return image, {
            **metadata,
            "reference_wsi_context": str(image_path),
        }

    def generate(self, record: dict) -> tuple[Image.Image, dict]:
        inputs = {
            "modality": self.modality,
            "num_images_per_prompt": 1,
            "num_inference_steps": self.args.steps,
            "guidance_scale": self.args.guidance,
            "guidance_high": self.args.guidance_high,
            "guidance_low": self.args.guidance_low,
            "mode": "sde",
            "path_type": "linear",
            "seed": record["seed"],
        }
        if self.modality == "text":
            inputs["prompt"] = record["prompt"]
            preprocessing = None
        else:
            inputs["image"], preprocessing = self.load_image_condition(record)
        output = self.pipe(**inputs)
        extra = {"modality": self.modality}
        if preprocessing is not None:
            extra["reference_preprocessing"] = preprocessing
        return to_pil(output["images"][0]), extra


def make_adapter(args: argparse.Namespace, device: torch.device):
    if args.model_type == "pixcell":
        return PixCellAdapter(args, device)
    if args.model_type == "pathdiff":
        return PathDiffAdapter(args, device)
    if args.model_type == "pathldm":
        return PathLDMAdapter(args, device)
    if args.model_type == "unipath":
        return UniPathAdapter(args, device)
    if args.model_type == "mupad_text":
        return MuPaDAdapter(args, device, "text")
    if args.model_type == "mupad_image":
        return MuPaDAdapter(args, device, "image")
    raise ValueError(args.model_type)


def output_is_complete(sample_dir: Path) -> bool:
    required = sample_dir / "generated.png", sample_dir / "metadata.json"
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def input_contract(args: argparse.Namespace) -> list[str]:
    contracts = {
        "pixcell": ["reference_image", "target_nuclei_mask"],
        "pathdiff": ["prompt", "target_conic_instance_type_mask"],
        "pathldm": ["prompt"],
        "unipath": ["prompt"],
        "mupad_text": ["prompt"],
        "mupad_image": ["reference_wsi_context"],
    }
    if args.model_type == "pathdiff" and args.pathdiff_mode == "t2i":
        return ["prompt"]
    return contracts[args.model_type]


def execution_provenance(args: argparse.Namespace) -> dict:
    paths = {
        "model_dir": args.model_dir,
        "pipeline_dir": args.pipeline_dir,
        "vae_dir": args.vae_dir,
        "uni_root": args.uni_root,
        "repo_dir": args.repo_dir,
        "model_config": args.config,
        "model_checkpoint": args.checkpoint or args.model_dir,
        "conic_root": args.conic_root,
        "mupad_context_root": args.mupad_context_root,
        "excluded_sample_ids": args.exclude_sample_ids,
        "rag_root": args.rag_root,
    }
    provenance = {
        key: str(value)
        for key, value in paths.items()
        if value is not None
    }
    pathdiff_text_scale_condition = None
    if args.model_type == "pathdiff" and args.pathdiff_mode == "t2i":
        _, pathdiff_text_scale_condition = pathdiff_text_effective_prompt(
            "scale provenance placeholder",
            args.pathdiff_text_objective_magnification,
        )
    provenance.update(
        {
            "worker": str(Path(__file__).resolve()),
            "command": " ".join(sys.argv),
            "native_output_mpp": args.native_mpp,
            "native_output_resolution": [
                args.native_resolution,
                args.native_resolution,
            ],
            "native_output_fov_um": round(
                args.native_resolution * args.native_mpp, 6
            )
            if args.native_mpp is not None
            else None,
            "source_patch_mpp": args.source_mpp,
            "physical_scale_status": (
                pathdiff_text_scale_condition["physical_scale_status"]
                if pathdiff_text_scale_condition is not None
                else "declared_by_benchmark_protocol"
            ),
        }
    )
    if pathdiff_text_scale_condition is not None:
        provenance["pathdiff_text_scale_condition"] = pathdiff_text_scale_condition
    return provenance


def standard_preprocessing_provenance(args: argparse.Namespace) -> dict:
    if args.model_type == "pixcell":
        return {
            "reference_preprocessing": {
                "operation": "full_field_downsample",
                "source_resolution": [512, 512],
                "source_mpp": args.source_mpp,
                "model_resolution": [256, 256],
                "model_mpp": args.native_mpp,
            },
            "target_mask_preprocessing": {
                "operation": "full_field_nearest_downsample",
                "source_resolution": [512, 512],
                "source_mpp": args.source_mpp,
                "model_resolution": [256, 256],
                "model_mpp": args.native_mpp,
                "crop_applied": False,
            },
        }
    if args.model_type == "pathdiff":
        if args.pathdiff_mode == "t2i":
            _, scale_condition = pathdiff_text_effective_prompt(
                "scale provenance placeholder",
                args.pathdiff_text_objective_magnification,
            )
            return {
                "pathdiff_inference_mode": "t2i",
                "pathdiff_official_entrypoint": "sampling.py::sample_one(mode='t2i')",
                "pathdiff_control_condition": {
                    "kind": "official_null_mask",
                    "value": 10,
                    "shape": [256, 256, 6],
                },
                "pathdiff_text_scale_condition": scale_condition,
                "native_scale_status": scale_condition["physical_scale_status"],
            }
        return {
            "target_condition_preprocessing": {
                "operation": "full_field_downsample_then_conic_segmentation",
                "source_resolution": [512, 512],
                "source_mpp": args.source_mpp,
                "model_resolution": [256, 256],
                "model_mpp": args.native_mpp,
                "crop_applied": False,
            }
        }
    return {}


def backfill_metadata(
    metadata_path: Path, args: argparse.Namespace, provenance: dict
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(provenance)
    metadata.update(standard_preprocessing_provenance(args))
    metadata["allowed_generation_inputs"] = input_contract(args)
    metadata["target_image_used_for_generation"] = False
    metadata["steps"] = args.steps
    metadata["guidance"] = args.guidance
    if args.guidance_high is not None:
        metadata["guidance_high"] = args.guidance_high
    if args.guidance_low is not None:
        metadata["guidance_low"] = args.guidance_low
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    records = load_records(args.manifest)
    excluded_sample_ids = set()
    if args.exclude_sample_ids is not None:
        exclusion_path = require_path(args.exclude_sample_ids, "--exclude-sample-ids")
        excluded_sample_ids = {
            line.strip()
            for line in exclusion_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        records = [
            record
            for record in records
            if record["sample_id"] not in excluded_sample_ids
        ]
    if args.native_resolution <= 0:
        raise ValueError("native resolution must be positive")
    if args.native_mpp is not None and args.native_mpp <= 0:
        raise ValueError("native MPP must be positive when specified")
    if args.pathdiff_text_objective_magnification is not None:
        if args.model_type != "pathdiff" or args.pathdiff_mode != "t2i":
            raise ValueError(
                "--pathdiff-text-objective-magnification is only valid for "
                "PathDiff text-only inference"
            )
        expected_mpp = nominal_mpp_for_objective(
            args.pathdiff_text_objective_magnification
        )
        if args.native_mpp is None or not np.isclose(args.native_mpp, expected_mpp):
            raise ValueError(
                "PathDiff text objective magnification requires matching "
                f"--native-mpp {expected_mpp:g}"
            )
    if args.organs:
        allowed_organs = set(args.organs)
        records = [record for record in records if record["organ"] in allowed_organs]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    records = [
        record
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_items is not None:
        records = records[: args.max_items]
    model_root = args.output_root / args.model_id
    model_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    provenance = execution_provenance(args)

    if args.metadata_only:
        updated = 0
        missing = []
        for record in records:
            sample_dir = model_root / record["organ"] / record["sample_id"]
            if output_is_complete(sample_dir):
                backfill_metadata(sample_dir / "metadata.json", args, provenance)
                updated += 1
            else:
                missing.append(record["sample_id"])
        print(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "metadata_updated": updated,
                    "missing": missing,
                },
                indent=2,
            ),
            flush=True,
        )
        return 1 if missing else 0

    load_started = time.time()
    adapter = make_adapter(args, device)
    load_seconds = time.time() - load_started
    completed = 0
    skipped = 0
    failures = []
    for index, record in enumerate(records, start=1):
        sample_dir = model_root / record["organ"] / record["sample_id"]
        if not args.overwrite and output_is_complete(sample_dir):
            backfill_metadata(sample_dir / "metadata.json", args, provenance)
            skipped += 1
            print(f"[{index}/{len(records)}] skip {record['sample_id']}", flush=True)
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        generated_path = sample_dir / "generated.png"
        metadata_path = sample_dir / "metadata.json"
        error_path = sample_dir / "error.json"
        started = time.time()
        try:
            set_seed(int(record["seed"]))
            generated, extra = adapter.generate(record)
            expected_size = (args.native_resolution, args.native_resolution)
            if generated.size != expected_size:
                raise ValueError(
                    f"Unexpected native output size {generated.size}; expected {expected_size}"
                )
            generated.save(generated_path)
            if "condition_preview" in extra:
                Image.fromarray(extra.pop("condition_preview")).save(
                    sample_dir / "condition.png"
                )
            metadata = {
                "status": "completed",
                "model_id": args.model_id,
                "model_type": args.model_type,
                "sample_id": record["sample_id"],
                "pair_id": record["pair_id"],
                "direction": record["direction"],
                "organ": record["organ"],
                "seed": int(record["seed"]),
                "allowed_generation_inputs": list(adapter.allowed_inputs),
                "prompt": record["prompt"]
                if "prompt" in adapter.allowed_inputs
                else None,
                "reference_image": record["reference_image"]
                if "reference_image" in adapter.allowed_inputs
                else None,
                "target_image_used_for_generation": False,
                "output": str(generated_path),
                "steps": args.steps,
                "guidance": args.guidance,
                "runtime_seconds": round(time.time() - started, 3),
                "image_stats": image_stats(generated),
                **provenance,
                **extra,
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            error_path.unlink(missing_ok=True)
            completed += 1
            print(
                f"[{index}/{len(records)}] done {record['sample_id']} "
                f"({metadata['runtime_seconds']}s)",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "status": "failed",
                "model_id": args.model_id,
                "sample_id": record.get("sample_id"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            error_path.write_text(
                json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            failures.append(failure)
            print(
                f"[{index}/{len(records)}] FAIL {record.get('sample_id')}: {exc}",
                flush=True,
            )

    summary = {
        "model_id": args.model_id,
        "model_type": args.model_type,
        "manifest": str(args.manifest),
        "output_root": str(model_root),
        "requested": len(records),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "completed_this_run": completed,
        "skipped_complete": skipped,
        "failed": len(failures),
        "failures": failures,
        "load_seconds": round(load_seconds, 3),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    summary_name = (
        f"batch_summary_shard{args.shard_index}of{args.num_shards}.json"
        if args.num_shards != 1
        else "batch_summary.json"
    )
    (model_root / summary_name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
