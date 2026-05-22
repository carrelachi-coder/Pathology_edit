"""Stage 4 tissue segmentation baseline package."""

from .config import BaselineConfig, DatasetManifest, SampleRecord

__all__ = [
    "BaselineConfig",
    "DatasetManifest",
    "SampleRecord",
    "BaselineSegmenter",
]


def __getattr__(name: str):
    if name == "BaselineSegmenter":
        from .model import BaselineSegmenter

        return BaselineSegmenter
    raise AttributeError(name)
