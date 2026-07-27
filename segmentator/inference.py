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
    symmetric_padding: bool = False,
    boundary_refinement: bool = False,
    cellvit_mode: str = "none",
    hierarchical_fine: bool = False,
) -> BaselineSegmenter:
    model = BaselineSegmenter(
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        local_repo=local_repo,
        decoder=decoder,
        mask2former_queries=mask2former_queries,
        mask2former_ignore_index=mask2former_ignore_index,
        symmetric_padding=symmetric_padding,
        boundary_refinement=boundary_refinement,
        cellvit_mode=cellvit_mode,
        hierarchical_fine=hierarchical_fine,
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
