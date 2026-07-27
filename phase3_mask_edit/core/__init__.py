"""Core dataset-agnostic utilities for Phase 3 mask editing."""

from phase3_mask_edit.core.applicability import (
    EditApplicabilityDecision,
    assess_edit_applicability,
)
from phase3_mask_edit.core.config import (
    RecipeValidationError,
    load_recipe,
    validate_recipe_schema,
)
from phase3_mask_edit.core.candidates import (
    CandidateSelection,
    CandidateSelectionError,
    build_candidate_mask_by_priority,
)
from phase3_mask_edit.core.mask_io import (
    MaskIOError,
    id_to_rgb,
    load_change_region,
    load_id_mask,
    load_metadata,
    load_rgb_mask,
    rgb_to_id,
    save_change_region,
    save_edit_output,
    save_id_mask,
    save_metadata,
    save_rgb_mask,
)
from phase3_mask_edit.core.gland_region import (
    GLAS_GLAND_FINE_IDS,
    GLAS_WHOLE_GLAND_POLICY_VERSION,
    expand_region_to_intersecting_components,
    glas_gland_mask,
    glas_whole_gland_generation_region,
)
from phase3_mask_edit.core.context import (
    MaskEditContext,
    MaskEditContextError,
)
from phase3_mask_edit.core.intent import (
    EditIntent,
    IntentValidationError,
    resolve_reference_profile,
    validate_intent_against_recipe,
)
from phase3_mask_edit.core.labels import (
    MaskProfileSchema,
    MaskProfileSchemaError,
)
from phase3_mask_edit.core.morphology import (
    binary_dilate,
    binary_erode,
    boundary_ring,
    distance_to_boundary,
    distance_to_label,
    fill_small_holes,
    generate_islands,
    keep_only_touching,
    multi_scale_smooth_noise,
    nearest_label_backfill,
    remove_small_components,
    select_boundary_band_by_fraction,
    select_connected_region_by_fraction,
    select_region_by_fraction,
    signed_distance_field,
)
from phase3_mask_edit.core.validation import (
    MaskValidationError,
    ValidationCheck,
    ValidationResult,
    validate_edit_result,
)

__all__ = [
    "EditIntent",
    "GLAS_GLAND_FINE_IDS",
    "GLAS_WHOLE_GLAND_POLICY_VERSION",
    "EditApplicabilityDecision",
    "IntentValidationError",
    "MaskIOError",
    "CandidateSelection",
    "CandidateSelectionError",
    "MaskEditContext",
    "MaskEditContextError",
    "MaskProfileSchema",
    "MaskProfileSchemaError",
    "RecipeValidationError",
    "MaskValidationError",
    "ValidationCheck",
    "ValidationResult",
    "validate_edit_result",
    "assess_edit_applicability",
    "binary_dilate",
    "binary_erode",
    "build_candidate_mask_by_priority",
    "boundary_ring",
    "distance_to_boundary",
    "distance_to_label",
    "expand_region_to_intersecting_components",
    "fill_small_holes",
    "generate_islands",
    "glas_gland_mask",
    "glas_whole_gland_generation_region",
    "id_to_rgb",
    "keep_only_touching",
    "load_change_region",
    "load_id_mask",
    "load_metadata",
    "load_rgb_mask",
    "rgb_to_id",
    "load_recipe",
    "multi_scale_smooth_noise",
    "nearest_label_backfill",
    "remove_small_components",
    "resolve_reference_profile",
    "save_change_region",
    "save_edit_output",
    "save_id_mask",
    "save_metadata",
    "save_rgb_mask",
    "select_boundary_band_by_fraction",
    "select_connected_region_by_fraction",
    "select_region_by_fraction",
    "signed_distance_field",
    "validate_intent_against_recipe",
    "validate_recipe_schema",
]
