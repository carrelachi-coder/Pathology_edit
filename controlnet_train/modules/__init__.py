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
from .cross_v2_2_conditioning import (
    CROSS_V2_2_REFERENCE_WITH_REF,
    CROSS_V2_2_REFERENCE_ZERO_REF,
    CrossV22ControlSpec,
    apply_cross_v2_2_reference_mode,
    build_cross_v2_2_block_bank_reference_latent,
    build_cross_v2_2_condition,
    normalize_cross_v2_2_reference_mode,
)
from .cross_v3_conditioning import (
    CROSS_V3_PROMPT,
    CROSS_V3_REFERENCE_WITH_REF,
    CROSS_V3_REFERENCE_ZERO_REF,
    CrossV3ControlSpec,
    CrossV3ReferenceContextEncoder,
    CrossV3ReferenceSpec,
    append_cross_v3_reference_context,
    apply_cross_v3_reference_mode,
    apply_cross_v3_reference_token_mode,
    build_cross_v3_control_condition,
    deterministic_latent_from_posterior as deterministic_cross_v3_latent_from_posterior,
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
    "CROSS_V2_2_REFERENCE_WITH_REF",
    "CROSS_V2_2_REFERENCE_ZERO_REF",
    "CROSS_V3_PROMPT",
    "CROSS_V3_REFERENCE_WITH_REF",
    "CROSS_V3_REFERENCE_ZERO_REF",
    "CrossV21ControlSpec",
    "CrossV22ControlSpec",
    "CrossV3ControlSpec",
    "CrossV3ReferenceContextEncoder",
    "CrossV3ReferenceSpec",
    "FixedOneHotTissueEncoder",
    "HierarchicalTissueEmbedding",
    "NucleiConditionEncoder",
    "TissueConditionDownsampler",
    "append_cross_v3_reference_context",
    "apply_cross_v2_1_reference_mode",
    "apply_cross_v2_2_reference_mode",
    "apply_cross_v3_reference_mode",
    "apply_cross_v3_reference_token_mode",
    "build_cross_v0_condition",
    "build_cross_v2_1_condition",
    "build_cross_v2_2_block_bank_reference_latent",
    "build_cross_v2_2_condition",
    "build_cross_v3_control_condition",
    "deterministic_cross_v3_latent_from_posterior",
    "build_inpaint_condition",
    "normalize_cross_v2_1_reference_mode",
    "normalize_cross_v2_2_reference_mode",
    "normalize_cross_v3_reference_mode",
    "pack_cross_v3_reference_grid",
]
