"""Conditioning modules for Phase 5 ControlNet training."""

from .change_mask_encoder import ChangeMaskEncoder
from .conditioning import build_cross_v0_condition, build_inpaint_condition
from .cross_v2_1_conditioning import (
    CROSS_V2_1_REFERENCE_WITH_REF,
    CROSS_V2_1_REFERENCE_ZERO_REF,
    CrossV21ControlSpec,
    apply_cross_v2_1_reference_mode,
    build_cross_v2_1_condition,
    normalize_cross_v2_1_reference_mode,
)
from .hte_embedding import HierarchicalTissueEmbedding
from .nuclei_condition_encoder import NucleiConditionEncoder
from .tissue_condition_downsampler import TissueConditionDownsampler

__all__ = [
    "ChangeMaskEncoder",
    "CROSS_V2_1_REFERENCE_WITH_REF",
    "CROSS_V2_1_REFERENCE_ZERO_REF",
    "CrossV21ControlSpec",
    "HierarchicalTissueEmbedding",
    "NucleiConditionEncoder",
    "TissueConditionDownsampler",
    "apply_cross_v2_1_reference_mode",
    "build_cross_v0_condition",
    "build_cross_v2_1_condition",
    "build_inpaint_condition",
    "normalize_cross_v2_1_reference_mode",
]
