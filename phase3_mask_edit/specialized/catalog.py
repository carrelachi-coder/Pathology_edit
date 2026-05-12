"""Dataset-specialized Phase 3 primitive catalog.

The catalog is intentionally declarative: each primitive states the fine IDs it
can read and write, while the shared fine-label transition executor supplies the
mask operation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _fine_transition(
    *,
    name: str,
    meaning: str,
    source_label: str,
    target_label: str,
    source_fine_ids: tuple[int, ...],
    target_fine_id: int,
    source_description: str,
    target_description: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "version": "v1",
        "execution_strategy": "id_transition",
        "pathology_meaning": meaning,
        "required_tissue_labels": [source_label],
        "required_context": ["source_fine_id_present"],
        "optional_tissue_labels": [],
        "mask_operation": {
            "type": "fine_label_transition",
            "source": source_label,
            "target": target_label,
            "source_fine_ids": list(source_fine_ids),
            "target_fine_id": target_fine_id,
            "source_description": source_description,
            "target_description": target_description,
        },
        "spatial_pattern": {
            "region": "in_place_source_fine_id_pixels",
            "geometry_policy": "preserve_existing_region_footprint",
        },
        "parameter_ranges": {
            "source_area_transition_fraction": {
                "mild": [0.08, 0.20],
                "moderate": [0.20, 0.40],
                "significant": [0.40, 0.70],
            }
        },
        "probnet_bias": {"default": "preserve_contextual_cell_mix"},
        "validation_rules": [
            "fine_transition_source_must_decrease",
            "fine_transition_target_must_increase",
            "change_region_must_match_source_fine_ids",
        ],
        "expected_failure_cases": [
            "source_fine_id_absent",
            "target_fine_id_not_in_dataset",
        ],
        "overlap_guard": {
            "prefer_tumor_burden_for_area_change": (
                "This primitive changes fine identity in place; use tumor burden "
                "primitives for large boundary growth or shrink."
            )
        },
    }


SPECIALIZED_PRIMITIVES: dict[str, tuple[dict[str, Any], ...]] = {
    "BCSS": (
        _fine_transition(
            name="dcis_invasion",
            meaning="DCIS focus becomes invasive tumor while retaining local footprint",
            source_label="Tumor",
            target_label="Tumor",
            source_fine_ids=(14,),
            target_fine_id=1,
            source_description="DCIS",
            target_description="invasive Tumor",
        ),
        _fine_transition(
            name="angioinvasion_emphasis",
            meaning="invasive tumor/DCIS focus gains angioinvasion subtype",
            source_label="Tumor",
            target_label="Tumor",
            source_fine_ids=(1, 14),
            target_fine_id=15,
            source_description="Tumor or DCIS",
            target_description="Angioinvasion",
        ),
    ),
    "PANDA": (
        _fine_transition(
            name="gleason_upgrade_3to4",
            meaning="Gleason pattern 3 upgrades to pattern 4",
            source_label="Tumor",
            target_label="Tumor",
            source_fine_ids=(8,),
            target_fine_id=9,
            source_description="Gleason 3",
            target_description="Gleason 4",
        ),
        _fine_transition(
            name="gleason_upgrade_4to5",
            meaning="Gleason pattern 4 upgrades to pattern 5",
            source_label="Tumor",
            target_label="Tumor",
            source_fine_ids=(9,),
            target_fine_id=10,
            source_description="Gleason 4",
            target_description="Gleason 5",
        ),
        _fine_transition(
            name="gleason_downgrade_4to3",
            meaning="Gleason pattern 4 downgrades to pattern 3",
            source_label="Tumor",
            target_label="Tumor",
            source_fine_ids=(9,),
            target_fine_id=8,
            source_description="Gleason 4",
            target_description="Gleason 3",
        ),
        _fine_transition(
            name="benign_to_gleason3",
            meaning="benign epithelium becomes low-grade malignant gland",
            source_label="Normal epithelium",
            target_label="Tumor",
            source_fine_ids=(5,),
            target_fine_id=8,
            source_description="benign epithelium",
            target_description="Gleason 3",
        ),
        _fine_transition(
            name="benign_atrophy",
            meaning="benign epithelium is replaced by stroma",
            source_label="Normal epithelium",
            target_label="Stroma",
            source_fine_ids=(5,),
            target_fine_id=2,
            source_description="benign epithelium",
            target_description="Stroma",
        ),
    ),
    "GLAS": (
        _fine_transition(
            name="normal_to_adenomatous",
            meaning="normal gland epithelium becomes adenomatous",
            source_label="Normal epithelium",
            target_label="Tumor",
            source_fine_ids=(5,),
            target_fine_id=11,
            source_description="normal gland",
            target_description="Adenomatous gland",
        ),
        _fine_transition(
            name="adenoma_to_carcinoma",
            meaning="adenomatous gland progresses to moderately differentiated carcinoma",
            source_label="Tumor",
            target_label="Tumor",
            source_fine_ids=(11,),
            target_fine_id=12,
            source_description="Adenomatous gland",
            target_description="Moderately differentiated",
        ),
        _fine_transition(
            name="grade_upgrade",
            meaning="moderately differentiated tumor becomes poorly differentiated",
            source_label="Tumor",
            target_label="Tumor",
            source_fine_ids=(12,),
            target_fine_id=13,
            source_description="Moderately differentiated",
            target_description="Poorly differentiated",
        ),
        _fine_transition(
            name="treatment_dedifferentiation",
            meaning="poorly differentiated tumor shifts toward moderate differentiation",
            source_label="Tumor",
            target_label="Tumor",
            source_fine_ids=(13,),
            target_fine_id=12,
            source_description="Poorly differentiated",
            target_description="Moderately differentiated",
        ),
    ),
}


def specialized_primitives_for(dataset: str) -> list[dict[str, Any]]:
    """Return deep-copied specialized primitive configs for a dataset."""

    return [deepcopy(item) for item in SPECIALIZED_PRIMITIVES.get(dataset.upper(), ())]


def specialized_primitive_names() -> tuple[str, ...]:
    names: list[str] = []
    for primitives in SPECIALIZED_PRIMITIVES.values():
        names.extend(str(primitive["name"]) for primitive in primitives)
    return tuple(dict.fromkeys(names))
