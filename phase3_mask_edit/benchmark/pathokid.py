"""Feature extraction and clustered Patho-KID statistics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Sequence

import numpy as np
from PIL import Image


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_transform_repr(transform) -> str:
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", repr(transform))


def input_digest(sample_ids: Sequence[str], image_paths: Sequence[Path]) -> str:
    if len(sample_ids) != len(image_paths):
        raise ValueError("sample_ids and image_paths must have the same length")
    entries = []
    for sample_id, path in zip(sample_ids, image_paths):
        stat = path.stat()
        entries.append(
            {
                "sample_id": sample_id,
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return stable_digest(entries)


def l2_normalize(features: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"features must be rank 2, got {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("features contain a zero-norm row")
    return values / norms


def polynomial_kernel(
    left: np.ndarray,
    right: np.ndarray,
    degree: int = 3,
    offset: float = 1.0,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError(f"incompatible feature shapes: {left.shape}, {right.shape}")
    # Match the standard KID polynomial kernel on raw feature activations.
    return (left @ right.T / left.shape[1] + offset) ** degree


def kid_from_kernels(
    kernel_xx: np.ndarray,
    kernel_yy: np.ndarray,
    kernel_xy: np.ndarray,
) -> float:
    count_x = kernel_xx.shape[0]
    count_y = kernel_yy.shape[0]
    if kernel_xx.shape != (count_x, count_x):
        raise ValueError("kernel_xx must be square")
    if kernel_yy.shape != (count_y, count_y):
        raise ValueError("kernel_yy must be square")
    if kernel_xy.shape != (count_x, count_y):
        raise ValueError("kernel_xy shape mismatch")
    if count_x < 2 or count_y < 2:
        raise ValueError("unbiased KID requires at least two samples per set")
    within_x = (kernel_xx.sum() - np.trace(kernel_xx)) / (
        count_x * (count_x - 1)
    )
    within_y = (kernel_yy.sum() - np.trace(kernel_yy)) / (
        count_y * (count_y - 1)
    )
    cross = kernel_xy.mean()
    return float(within_x + within_y - 2.0 * cross)


def unbiased_kid(real: np.ndarray, generated: np.ndarray) -> float:
    return kid_from_kernels(
        polynomial_kernel(real, real),
        polynomial_kernel(generated, generated),
        polynomial_kernel(real, generated),
    )


def subset_kid(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    subset_size: int,
    repeats: int,
    seed: int,
) -> np.ndarray:
    if repeats <= 0:
        raise ValueError("subset repeats must be positive")
    if subset_size < 2 or subset_size > min(len(real), len(generated)):
        raise ValueError("subset_size must be in [2, min(real, generated)]")
    kernel_xx = polynomial_kernel(real, real)
    kernel_yy = polynomial_kernel(generated, generated)
    kernel_xy = polynomial_kernel(real, generated)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        index_x = rng.choice(len(real), size=subset_size, replace=False)
        index_y = rng.choice(len(generated), size=subset_size, replace=False)
        values.append(
            kid_from_kernels(
                kernel_xx[np.ix_(index_x, index_x)],
                kernel_yy[np.ix_(index_y, index_y)],
                kernel_xy[np.ix_(index_x, index_y)],
            )
        )
    return np.asarray(values, dtype=np.float64)


def _stratified_sample_counts(strata: Sequence[str], sample_size: int) -> dict[str, int]:
    names, population = np.unique(np.asarray(strata, dtype=str), return_counts=True)
    if sample_size < len(names) or sample_size <= 0:
        raise ValueError("sample_size must include at least one row per stratum")
    expected = population / population.sum() * sample_size
    counts = np.floor(expected).astype(int)
    counts[counts == 0] = 1
    remainder = sample_size - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(expected - np.floor(expected)))
        for index in order[:remainder]:
            counts[index] += 1
    elif remainder < 0:
        order = np.argsort(expected - np.floor(expected))
        for index in order:
            if remainder == 0:
                break
            if counts[index] > 1:
                counts[index] -= 1
                remainder += 1
    if int(counts.sum()) != sample_size:
        raise RuntimeError("failed to allocate exact stratified sample counts")
    return {str(name): int(count) for name, count in zip(names, counts)}


def real_vs_real_kid_curve(
    features: np.ndarray,
    strata: Sequence[str],
    *,
    sample_sizes: Sequence[int],
    repeats: int,
    seed: int,
) -> dict[int, dict]:
    """Estimate same-distribution KID at fixed, organ-preserving sample sizes.

    Two disjoint samples are used whenever each stratum can supply both sides.
    Larger requested sizes use independent empirical bootstrap samples and are
    explicitly labeled because overlap changes the interpretation.
    """
    features = np.asarray(features, dtype=np.float64)
    strata = np.asarray(strata, dtype=str)
    if features.ndim != 2 or len(features) != len(strata):
        raise ValueError("feature and stratum rows must align")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    kernel = polynomial_kernel(features, features)
    rng = np.random.default_rng(seed)
    stratum_indices = {
        name: np.flatnonzero(strata == name) for name in sorted(set(strata.tolist()))
    }
    results = {}
    for sample_size in sample_sizes:
        counts = _stratified_sample_counts(strata, int(sample_size))
        disjoint = all(2 * counts[name] <= len(stratum_indices[name]) for name in counts)
        values = []
        overlaps = []
        for _ in range(repeats):
            left_parts = []
            right_parts = []
            for name, count in counts.items():
                available = stratum_indices[name]
                if disjoint:
                    selected = rng.choice(available, size=2 * count, replace=False)
                    left_parts.append(selected[:count])
                    right_parts.append(selected[count:])
                else:
                    left_parts.append(rng.choice(available, size=count, replace=True))
                    right_parts.append(rng.choice(available, size=count, replace=True))
            left = np.concatenate(left_parts)
            right = np.concatenate(right_parts)
            rng.shuffle(left)
            rng.shuffle(right)
            values.append(
                kid_from_kernels(
                    kernel[np.ix_(left, left)],
                    kernel[np.ix_(right, right)],
                    kernel[np.ix_(left, right)],
                )
            )
            overlaps.append(len(set(left.tolist()) & set(right.tolist())))
        results[int(sample_size)] = {
            "sampling_mode": "stratified_disjoint" if disjoint else "stratified_independent_bootstrap",
            "stratum_counts_per_side": counts,
            "values": np.asarray(values, dtype=np.float64),
            "source_overlap_count": np.asarray(overlaps, dtype=np.int64),
        }
    return results


def paired_bootstrap_delta(left: np.ndarray, right: np.ndarray) -> dict:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("paired bootstrap vectors must be same-shape non-empty vectors")
    delta = left - right
    summary = summarize_values(delta)
    return {
        **summary,
        "probability_left_better": float(np.mean(delta < 0)),
        "probability_tie": float(np.mean(delta == 0)),
        "mean_x1000": summary["mean"] * 1000.0,
        "ci95_low_x1000": summary["ci95_low"] * 1000.0,
        "ci95_high_x1000": summary["ci95_high"] * 1000.0,
    }


@dataclass(frozen=True)
class BootstrapDraws:
    group_names: tuple[str, ...]
    counts: np.ndarray
    seed: int

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "group_names": list(self.group_names),
            "draw_counts": self.counts.tolist(),
        }


def make_cluster_bootstrap_draws(
    groups: Sequence[str], *, repeats: int, seed: int
) -> BootstrapDraws:
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    group_names = tuple(sorted(set(str(group) for group in groups)))
    if len(group_names) < 2:
        raise ValueError("cluster bootstrap requires at least two groups")
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(group_names), 1.0 / len(group_names))
    counts = rng.multinomial(len(group_names), probabilities, size=repeats)
    return BootstrapDraws(group_names=group_names, counts=counts, seed=seed)


def cluster_bootstrap_kid(
    real: np.ndarray,
    generated: np.ndarray,
    groups: Sequence[str],
    draws: BootstrapDraws,
) -> np.ndarray:
    if len(real) != len(generated) or len(real) != len(groups):
        raise ValueError("real, generated, and groups must have the same length")
    group_to_index = {name: index for index, name in enumerate(draws.group_names)}
    try:
        group_indices = np.asarray([group_to_index[str(group)] for group in groups])
    except KeyError as exc:
        raise ValueError(f"bootstrap draws do not contain group {exc.args[0]!r}") from exc
    if set(group_to_index) != set(str(group) for group in groups):
        raise ValueError("bootstrap group set does not match the evaluation rows")

    kernel_xx = polynomial_kernel(real, real)
    kernel_yy = polynomial_kernel(generated, generated)
    kernel_xy = polynomial_kernel(real, generated)
    membership = np.zeros((len(real), len(draws.group_names)), dtype=np.float64)
    membership[np.arange(len(real)), group_indices] = 1.0
    group_sizes = membership.sum(axis=0)
    aggregate_xx = membership.T @ kernel_xx @ membership
    aggregate_yy = membership.T @ kernel_yy @ membership
    aggregate_xy = membership.T @ kernel_xy @ membership
    diagonal_xx = membership.T @ np.diag(kernel_xx)
    diagonal_yy = membership.T @ np.diag(kernel_yy)

    values = []
    for counts in draws.counts:
        counts = counts.astype(np.float64, copy=False)
        sample_count = float(counts @ group_sizes)
        if sample_count < 2:
            raise ValueError("bootstrap draw contains fewer than two samples")
        within_x = (
            counts @ aggregate_xx @ counts - counts @ diagonal_xx
        ) / (sample_count * (sample_count - 1.0))
        within_y = (
            counts @ aggregate_yy @ counts - counts @ diagonal_yy
        ) / (sample_count * (sample_count - 1.0))
        cross = (counts @ aggregate_xy @ counts) / (sample_count**2)
        values.append(float(within_x + within_y - 2.0 * cross))
    return np.asarray(values, dtype=np.float64)


def summarize_values(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("cannot summarize an empty value vector")
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def load_feature_cache(
    feature_path: Path,
    metadata_path: Path,
    *,
    expected_sample_ids: Sequence[str],
    expected_input_digest: str,
    expected_extractor_digest: str,
) -> np.ndarray | None:
    if not feature_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("input_digest") != expected_input_digest
        or metadata.get("extractor_digest") != expected_extractor_digest
    ):
        return None
    with np.load(feature_path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str).tolist()
        features = payload["features"].astype(np.float32, copy=False)
    if sample_ids != list(expected_sample_ids):
        return None
    if features.ndim != 2 or len(features) != len(sample_ids):
        return None
    return features


def save_feature_cache(
    feature_path: Path,
    metadata_path: Path,
    *,
    sample_ids: Sequence[str],
    features: np.ndarray,
    metadata: dict,
) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_path,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        features=np.asarray(features, dtype=np.float32),
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


class ImageFeatureExtractor:
    """Small adapter around official UNI-2h and CONCH inference APIs."""

    def __init__(self, name: str, model, transform, device, dtype, metadata: dict):
        self.name = name
        self.model = model
        self.transform = transform
        self.device = device
        self.dtype = dtype
        self.metadata = metadata

    def extract(
        self,
        image_paths: Sequence[Path],
        *,
        batch_size: int,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        import torch

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batches = []
        with torch.inference_mode():
            for start in range(0, len(image_paths), batch_size):
                paths = image_paths[start : start + batch_size]
                tensors = []
                for path in paths:
                    with Image.open(path) as image:
                        tensors.append(self.transform(image.convert("RGB")))
                batch = torch.stack(tensors).to(
                    device=self.device, dtype=self.dtype, non_blocking=True
                )
                if self.name == "conch":
                    features = self.model.encode_image(
                        batch, proj_contrast=False, normalize=False
                    )
                else:
                    features = self.model(batch)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                if features.ndim != 2:
                    raise ValueError(
                        f"{self.name} returned feature shape {tuple(features.shape)}"
                    )
                batches.append(features.float().cpu().numpy())
                if progress is not None:
                    progress(min(start + len(paths), len(image_paths)), len(image_paths))
        output = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        if output.shape[0] != len(image_paths) or not np.isfinite(output).all():
            raise ValueError(f"invalid {self.name} feature matrix {output.shape}")
        return output


def build_uni2h_extractor(
    root: Path, *, device: str, dtype_name: str, checkpoint_sha256: str
) -> ImageFeatureExtractor:
    import torch
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    dtype = _torch_dtype(dtype_name)
    kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(pretrained=False, **kwargs)
    checkpoint = root / "pytorch_model.bin"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    config_path = root / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model.pretrained_cfg.update(config.get("pretrained_cfg", {}))
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    metadata = {
        "name": "uni2h",
        "architecture": "ViT-h/14-reg8",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "input_resolution": 224,
        "feature_policy": "raw_cls_embedding",
        "dtype": dtype_name,
        "transform": stable_transform_repr(transform),
    }
    return ImageFeatureExtractor("uni2h", model, transform, device, dtype, metadata)


def build_conch_extractor(
    root: Path,
    checkpoint: Path,
    *,
    device: str,
    dtype_name: str,
    checkpoint_sha256: str,
) -> ImageFeatureExtractor:
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from conch.open_clip_custom import create_model_from_pretrained

    dtype = _torch_dtype(dtype_name)
    model, transform = create_model_from_pretrained(
        "conch_ViT-B-16", checkpoint_path=str(checkpoint), device="cpu"
    )
    model.eval().requires_grad_(False).to(device=device, dtype=dtype)
    image_size = getattr(model.visual, "image_size", (448, 448))
    metadata = {
        "name": "conch",
        "architecture": "conch_ViT-B-16",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "input_resolution": list(image_size)
        if isinstance(image_size, (tuple, list))
        else int(image_size),
        "feature_policy": "encode_image(proj_contrast=False,normalize=False)",
        "dtype": dtype_name,
        "transform": stable_transform_repr(transform),
    }
    return ImageFeatureExtractor("conch", model, transform, device, dtype, metadata)


def _torch_dtype(name: str):
    import torch

    mapping = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc
