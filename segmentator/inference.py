from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .model import BaselineSegmenter


def load_checkpoint(
    checkpoint_path: str | Path,
    num_classes: int = 8,
    freeze_encoder: bool = True,
    local_repo: str | Path = "UNI-2h",
    decoder: str = "upernet",
    mask2former_queries: int = 100,
    mask2former_ignore_index: int = 255,
    feature_pyramid_source: str = "distinct_depths",
    symmetric_padding: bool = False,
    boundary_refinement: bool = False,
    refinement_gate_mode: str = "hard",
    cellvit_mode: str = "none",
    hierarchical_fine: bool = False,
    fine_supported_ids: tuple[int, ...] | None = None,
) -> BaselineSegmenter:
    model = BaselineSegmenter(
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        local_repo=local_repo,
        decoder=decoder,
        mask2former_queries=mask2former_queries,
        mask2former_ignore_index=mask2former_ignore_index,
        feature_pyramid_source=feature_pyramid_source,
        symmetric_padding=symmetric_padding,
        boundary_refinement=boundary_refinement,
        refinement_gate_mode=refinement_gate_mode,
        cellvit_mode=cellvit_mode,
        hierarchical_fine=hierarchical_fine,
        fine_supported_ids=fine_supported_ids,
    )
    state = torch.load(Path(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_image(
    model: BaselineSegmenter,
    image: torch.Tensor,
    nuclei_density: torch.Tensor | None = None,
    fine_allowed: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if nuclei_density is not None and nuclei_density.ndim == 3:
        nuclei_density = nuclei_density.unsqueeze(0)
    return model(image, nuclei_density=nuclei_density, fine_allowed=fine_allowed)


def save_prediction(mask: torch.Tensor, out_path: str | Path) -> None:
    array = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(array, mode="L").save(out_path)


def normalized_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Return Shannon entropy normalized to [0, 1] for a class distribution."""
    if probabilities.ndim < 2:
        raise ValueError(
            "probabilities must include batch and class dimensions"
        )
    class_count = int(probabilities.shape[1])
    if class_count < 2:
        return torch.zeros_like(probabilities[:, 0])
    safe = probabilities.float().clamp_min(1e-8)
    entropy = -(safe * safe.log()).sum(dim=1)
    return entropy / float(np.log(class_count))


def normalized_hierarchical_entropy(
    probabilities: torch.Tensor,
    parent_map: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    """Normalize entropy by the allowed child count for each parent pixel."""
    if probabilities.ndim != 4 or parent_map.ndim != 3 or allowed.ndim != 3:
        raise ValueError(
            "expected probabilities BCHW, parent_map BHW, and allowed BPF"
        )
    if probabilities.shape[0] != parent_map.shape[0]:
        raise ValueError("fine probabilities and parent map batch sizes differ")
    if probabilities.shape[0] != allowed.shape[0]:
        raise ValueError("fine probabilities and allowed batch sizes differ")
    safe_parent = parent_map.clamp(0, allowed.shape[1] - 1).long()
    batch_index = torch.arange(
        probabilities.shape[0], device=probabilities.device
    )[:, None, None]
    allowed_count = allowed[batch_index, safe_parent].sum(dim=-1)
    safe = probabilities.float().clamp_min(1e-8)
    entropy = -(safe * safe.log()).sum(dim=1)
    denominator = allowed_count.clamp_min(2).float().log()
    return torch.where(allowed_count > 1, entropy / denominator, 0.0)


def save_probability_tensor(
    probabilities: torch.Tensor,
    out_path: str | Path,
    *,
    class_ids: list[int] | tuple[int, ...],
) -> None:
    values = probabilities.squeeze(0).detach().cpu().numpy().astype(np.float16)
    if values.ndim != 3:
        raise ValueError(
            f"probability tensor must have CHW shape after squeezing, got {values.shape}"
        )
    np.savez_compressed(
        out_path,
        probabilities=values,
        class_ids=np.asarray(class_ids, dtype=np.int16),
        layout=np.asarray("CHW"),
    )
