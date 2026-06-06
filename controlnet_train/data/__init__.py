"""Phase 5 data loading utilities for ControlNet training."""

from .common import LayeredSample, load_layered_dataset_samples
from .cross import CrossReconstructionDataset, build_cross_metadata
from .cross_v5_pairing import CrossV5PairingDataset, CrossV5PairingSampler, CrossV5PairingSamplerConfig
from .inpaint import InpaintDataset, build_inpaint_metadata
from .inpaint_synthesis import build_synthetic_inpaint_metadata as _build_synthetic_inpaint_metadata


def build_synthetic_inpaint_metadata(
    dataset_roots,
    output_dir,
    forced_mode: str = "mixed",
    forced_bucket: str | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    samples_per_dataset: int | None = None,
    max_attempts_per_sample: int | None = None,
):
    """Build synthetic inpaint metadata.

    The CLI forwards synthesis sizing knobs through this wrapper so the
    package surface stays stable while still reaching the synthesis helper.
    """

    return _build_synthetic_inpaint_metadata(
        dataset_roots=dataset_roots,
        output_dir=output_dir,
        forced_mode=forced_mode,
        forced_bucket=forced_bucket,
        val_ratio=val_ratio,
        seed=seed,
        samples_per_dataset=samples_per_dataset,
        max_attempts_per_sample=max_attempts_per_sample,
    )

__all__ = [
    "CrossReconstructionDataset",
    "CrossV5PairingDataset",
    "CrossV5PairingSampler",
    "CrossV5PairingSamplerConfig",
    "InpaintDataset",
    "LayeredSample",
    "build_cross_metadata",
    "build_inpaint_metadata",
    "build_synthetic_inpaint_metadata",
    "load_layered_dataset_samples",
]
