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
from .cross_v3_conditioning import (
    CROSS_V3_PROMPT,
    CROSS_V3_REFERENCE_WITH_REF,
    CROSS_V3_REFERENCE_ZERO_REF,
    CROSS_V3_ROUTE_COARSE,
    CROSS_V3_ROUTE_FINE,
    CROSS_V3_ROUTE_NONE,
    CrossV3ControlSpec,
    CrossV3ReferenceContextEncoder,
    CrossV3ReferenceSpec,
    append_cross_v3_reference_context,
    apply_cross_v3_reference_mode,
    apply_cross_v3_reference_token_mode,
    build_cross_v3_reference_route_ids,
    build_cross_v3_control_condition,
    cross_v3_route_class_count,
    deterministic_latent_from_posterior as deterministic_cross_v3_latent_from_posterior,
    normalize_cross_v3_reference_route_mode,
    normalize_cross_v3_reference_mode,
    pack_cross_v3_reference_grid,
)
from .fixed_tissue_encoder import FixedOneHotTissueEncoder
from .hte_embedding import HierarchicalTissueEmbedding
from .nuclei_condition_encoder import NucleiConditionEncoder
from .tissue_condition_downsampler import TissueConditionDownsampler

__all__ = [
    "ChangeMaskEncoder",
    "CROSS_V2_1_REFERENCE_WITH_REF",
    "CROSS_V2_1_REFERENCE_ZERO_REF",
    "CROSS_V3_PROMPT",
    "CROSS_V3_REFERENCE_WITH_REF",
    "CROSS_V3_REFERENCE_ZERO_REF",
    "CROSS_V3_ROUTE_COARSE",
    "CROSS_V3_ROUTE_FINE",
    "CROSS_V3_ROUTE_NONE",
    "CrossV21ControlSpec",
    "CrossV3ControlSpec",
    "CrossV3ReferenceContextEncoder",
    "CrossV3ReferenceSpec",
    "FixedOneHotTissueEncoder",
    "HierarchicalTissueEmbedding",
    "NucleiConditionEncoder",
    "TissueConditionDownsampler",
    "append_cross_v3_reference_context",
    "apply_cross_v2_1_reference_mode",
    "apply_cross_v3_reference_mode",
    "apply_cross_v3_reference_token_mode",
    "build_cross_v0_condition",
    "build_cross_v2_1_condition",
    "build_cross_v3_control_condition",
    "build_cross_v3_reference_route_ids",
    "cross_v3_route_class_count",
    "deterministic_cross_v3_latent_from_posterior",
    "build_inpaint_condition",
    "normalize_cross_v2_1_reference_mode",
    "normalize_cross_v3_reference_mode",
    "normalize_cross_v3_reference_route_mode",
    "pack_cross_v3_reference_grid",
]
