from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from dataset_config.unified_labels import COARSE_LABELS, FINE_LABELS


SEGMENTATOR_CLASSES = tuple(COARSE_LABELS[idx].lower().replace(" ", "_") for idx in range(len(COARSE_LABELS)))
SEGMENTATOR_FINE_CLASSES = tuple(FINE_LABELS[idx].lower().replace(" ", "_") for idx in range(len(FINE_LABELS)))


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    mask_path: Path
    sample_id: str
    dataset_id: str = "default"
    group_id: str = ""
    nuclei_path: Path | None = None


@dataclass(frozen=True)
class DatasetManifest:
    root: Path
    train: tuple[SampleRecord, ...]
    val: tuple[SampleRecord, ...]
    test: tuple[SampleRecord, ...] = ()
    classes: tuple[str, ...] = SEGMENTATOR_CLASSES


@dataclass(frozen=True)
class BaselineConfig:
    image_size: int = 512
    num_classes: int = 8
    remap_invalid_to: int = 7
    ignore_index: int = 255
    mask_remap: str = "auto"
    balanced_datasets: bool = False
    dataset_sampling_temperature: float = 0.5
    rare_class_sampling: bool = False
    rare_class_ids: tuple[int, ...] = (3, 4, 6)
    rare_class_sample_boost: float = 2.0
    samples_per_epoch: int | None = None
    batch_size: int = 2
    grad_accum_steps: int = 1
    num_workers: int = 0
    lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 20
    warmup_epochs: int = 1
    lr_scheduler: str = "cosine"
    backbone_unfreeze_epoch: int = -1
    backbone_unfreeze_blocks: int = 0
    min_free_gpu_memory_gb_before_unfreeze: float = 0.0
    gpu_memory_poll_seconds: float = 60.0
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-4
    checkpoint_boundary_weight: float = 0.25
    checkpoint_fine_weight: float = 0.25
    seed: int = 42
    train_split: int = 1000
    val_split: int = 200
    manifest_path: Path | None = None
    freeze_encoder: bool = True
    decoder: str = "upernet"
    mask2former_queries: int = 100
    mask2former_ignore_index: int = 255
    symmetric_padding: bool = False
    boundary_refinement: bool = False
    refinement_loss_weight: float = 1.0
    refinement_boundary_weight: float = 0.5
    refinement_boundary_widths: tuple[int, ...] = (2, 4, 8)
    refinement_boundary_ce_weight: float = 0.0
    refinement_consistency_weight: float = 0.0
    refinement_gate_width: int = 4
    refinement_gate_threshold: float = 0.15
    boundary_aware_sampling: bool = False
    boundary_sampling_boost: float = 3.0
    boundary_sampling_min_pixels: int = 512
    boundary_sampling_width: int = 4
    cellvit_mode: str = "none"
    cell_density_sigma: float = 8.0
    cell_prior_dropout: float = 0.2
    cell_aux_loss_weight: float = 0.2
    hierarchical_fine: bool = False
    fine_loss_weight: float = 1.0
    fine_class_weighting: str | None = None
    fine_class_weight_min: float = 0.5
    fine_class_weight_max: float = 4.0
    fine_supervision_sampling: bool = False
    fine_sampling_rare_class_boost: float = 4.0
    fine_sampling_min_valid_pixels: int = 1
    freeze_shared_for_fine: bool = False
    fine_only_loss: bool = False
    amp: bool = True
    disable_cudnn: bool = False
    class_weighting: str = "none"
    label_space_summary_path: Path | None = None
    stain_augmentation: str = "none"
    stain_augmentation_prob: float = 0.0
    augment_vflip: bool = False
    augment_rot90: bool = False
    augment_scale_crop: float = 0.0
    randstainna_root: Path = field(default_factory=lambda: Path("third_party/RandStainNA"))
    randstainna_yaml: Path | None = None
    randstainna_std_hyper: float = -0.3
    randstainna_distribution: str = "normal"
    export_val_predictions: bool = False
    export_val_tensors: bool = False
    boundary_width: int = 2
    metric_sample_limit: int = 2048
    ddp_timeout_seconds: float = 600.0
    rank_zero_validation: bool = False
    checkpoint_mode: str = "composite"
    checkpoint_coarse_miou_floor: float | None = None
    checkpoint_coarse_boundary_f1_4_floor: float | None = None
    checkpoint_fine_dataset_macro_floor: float | None = None
    trainable_scope: str = "all"
    refinement_only_loss: bool = False
    fine_sampling_require_nuclei: bool = False
    resume_from_checkpoint: str | None = None
    init_from_checkpoint: str | None = None
    output_dir: Path = field(default_factory=lambda: Path("segmentator_runs/stage4_baseline"))

    def resolve_output_dir(self) -> Path:
        return self.output_dir.expanduser().resolve()
