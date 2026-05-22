from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from dataset_config.unified_labels import COARSE_LABELS


SEGMENTATOR_CLASSES = tuple(COARSE_LABELS[idx].lower().replace(" ", "_") for idx in range(len(COARSE_LABELS)))


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Path
    sample_id: str


@dataclass(frozen=True)
class DatasetManifest:
    root: Path
    train: tuple[SampleRecord, ...]
    val: tuple[SampleRecord, ...]
    classes: tuple[str, ...] = SEGMENTATOR_CLASSES


@dataclass(frozen=True)
class BaselineConfig:
    image_size: int = 512
    num_classes: int = 8
    remap_invalid_to: int = 7
    batch_size: int = 2
    grad_accum_steps: int = 1
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    seed: int = 42
    train_split: int = 1000
    val_split: int = 200
    manifest_path: Path | None = None
    freeze_encoder: bool = True
    decoder: str = "upernet"
    mask2former_queries: int = 100
    amp: bool = True
    disable_cudnn: bool = False
    class_weighting: str = "none"
    export_val_predictions: bool = False
    export_val_tensors: bool = False
    boundary_width: int = 2
    output_dir: Path = field(default_factory=lambda: Path("segmentator_runs/stage4_baseline"))

    def resolve_output_dir(self) -> Path:
        return self.output_dir.expanduser().resolve()
