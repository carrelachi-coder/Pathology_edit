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
    decoder: str = "upernet",
    mask2former_queries: int = 100,
) -> BaselineSegmenter:
    model = BaselineSegmenter(
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
        decoder=decoder,
        mask2former_queries=mask2former_queries,
    )
    state = torch.load(Path(checkpoint_path), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model: BaselineSegmenter, image: torch.Tensor) -> dict[str, torch.Tensor]:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return model(image)


def save_prediction(mask: torch.Tensor, out_path: str | Path) -> None:
    array = mask.squeeze().detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(array, mode="L").save(out_path)
