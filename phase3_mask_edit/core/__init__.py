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
    select_boundary_band_by_fraction,
    select_connected_region_by_fraction,
    select_region_by_fraction,
)

__all__ = [
    "EditIntent",
    "EditApplicabilityDecision",
    "IntentValidationError",
    "CandidateSelection",
    "CandidateSelectionError",
    "MaskEditContext",
    "MaskEditContextError",
    "MaskProfileSchema",
    "MaskProfileSchemaError",
    "RecipeValidationError",
    "assess_edit_applicability",
    "binary_dilate",
    "binary_erode",
    "build_candidate_mask_by_priority",
    "boundary_ring",
    "load_recipe",
    "resolve_reference_profile",
    "select_boundary_band_by_fraction",
    "select_connected_region_by_fraction",
    "select_region_by_fraction",
    "validate_intent_against_recipe",
    "validate_recipe_schema",
]
