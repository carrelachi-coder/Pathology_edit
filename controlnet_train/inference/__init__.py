"""Inference utilities for the unified Phase 5.4 edit pipeline."""

from .pipeline import (
    CrossV0InferenceBundle,
    EditPipelineInputs,
    EditPipelineResult,
    InpaintInferenceBundle,
    load_cross_bundle,
    load_inpaint_bundle,
    resolve_prompt,
    run_cross_v0_bundle,
    run_edit_pipeline,
    run_inpaint_bundle,
)
from .pipeline_cross_v2_1 import (
    CROSS_V2_1_REFERENCE_WITH_REF,
    CROSS_V2_1_REFERENCE_ZERO_REF,
    CrossV21InferenceBundle,
    compute_cross_v2_1_fixed_timestep_losses,
    load_cross_v2_1_bundle,
    run_cross_v2_1_bundle,
)
from .router import (
    EditRoutingConfig,
    EditRoutingDecision,
    compute_change_region_mask,
    route_edit_request,
)

__all__ = [
    "CrossV0InferenceBundle",
    "CrossV21InferenceBundle",
    "CROSS_V2_1_REFERENCE_WITH_REF",
    "CROSS_V2_1_REFERENCE_ZERO_REF",
    "EditPipelineInputs",
    "EditPipelineResult",
    "EditRoutingConfig",
    "EditRoutingDecision",
    "InpaintInferenceBundle",
    "compute_change_region_mask",
    "compute_cross_v2_1_fixed_timestep_losses",
    "load_cross_bundle",
    "load_cross_v2_1_bundle",
    "load_inpaint_bundle",
    "resolve_prompt",
    "route_edit_request",
    "run_cross_v0_bundle",
    "run_cross_v2_1_bundle",
    "run_edit_pipeline",
    "run_inpaint_bundle",
]
