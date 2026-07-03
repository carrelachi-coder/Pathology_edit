from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class StainAugmentationConfig:
    mode: str = "none"
    probability: float = 0.0
    randstainna_root: Path = Path("third_party/RandStainNA")
    randstainna_yaml: Path | None = None
    randstainna_std_hyper: float = -0.3
    randstainna_distribution: str = "normal"


class RandStainNAAdapter:
    """Thin adapter around the official RandStainNA implementation."""

    def __init__(self, config: StainAugmentationConfig) -> None:
        root = config.randstainna_root.expanduser().resolve()
        yaml_path = config.randstainna_yaml or (root / "CRC_LAB_randomTrue_n0.yaml")
        yaml_path = yaml_path.expanduser().resolve()
        if not (root / "randstainna.py").exists():
            raise FileNotFoundError(
                f"RandStainNA implementation not found under {root}. "
                "Expected third_party/RandStainNA/randstainna.py."
            )
        if not yaml_path.exists():
            raise FileNotFoundError(f"RandStainNA statistics YAML not found: {yaml_path}")
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "RandStainNA requires opencv-python. Install segmentator stage4 requirements "
                "or run: pip install opencv-python pyyaml scikit-image"
            ) from exc

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from randstainna import RandStainNA  # type: ignore
        except ImportError as exc:
            raise ImportError(f"Unable to import RandStainNA from {root}") from exc

        self._augment = RandStainNA(
            yaml_file=str(yaml_path),
            std_hyper=float(config.randstainna_std_hyper),
            probability=1.0,
            distribution=config.randstainna_distribution,
            is_train=True,
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        import cv2  # type: ignore

        bgr = self._augment(image.convert("RGB"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


def build_stain_augmenter(config: StainAugmentationConfig):
    mode = config.mode.lower()
    if mode == "none" or config.probability <= 0:
        return None
    if mode == "randstainna":
        return RandStainNAAdapter(config)
    raise ValueError(f"unsupported stain augmentation mode: {config.mode!r}")


def maybe_apply_stain_augmentation(image: Image.Image, augmenter, probability: float) -> Image.Image:
    if augmenter is None or random.random() >= probability:
        return image
    return augmenter(image)
