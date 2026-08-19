from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from phase3_joint_edit_refine.authority import (
    GLAND_AUTHORITY_DATASET_NATIVE,
    GLAND_AUTHORITY_SEMANTIC_PROXY,
    NUCLEUS_AUTHORITY_HYBRID,
    bind_gland_instance_authority_provenance,
    gland_instance_authority_status,
    semantic_gland_component_authority_metadata,
    summarize_nucleus_instance_authority,
    validate_mechanism_nucleus_authority,
    validate_nucleus_authority_floor,
)


def _digest(character: str) -> str:
    return character * 64


def _instance(
    instance_id: str,
    *,
    class_id: int,
    source: str,
    area_px: int = 9,
    complete: bool = True,
):
    return SimpleNamespace(
        instance_id=instance_id,
        class_id=class_id,
        source=source,
        area_px=area_px,
        touches_border=False,
        completeness_status="complete" if complete else "patch_boundary_censored",
        quality_flags=(),
    )


def test_semantic_gland_proxy_never_masquerades_as_original_instance_digest():
    source_digest = _digest("a")
    output_digest = _digest("b")
    record = {
        "producer_id": "joint-semantic-topology-auxiliary-v2",
        "producer_version": "joint-semantic-topology-auxiliary-v2",
        "source_tissue_mask_sha256": source_digest,
        "output_sha256": output_digest,
        **semantic_gland_component_authority_metadata(
            source_tissue_mask_sha256=source_digest,
            output_sha256=output_digest,
        ),
    }
    provenance = {
        "source_tissue_mask_sha256": source_digest,
        "original_instance_mask_digest": output_digest,
        "auxiliary_structure_sha256": {
            "native_gland_instance_map": output_digest,
        },
        "auxiliary_structure_provenance": {
            "native_gland_instance_map": record,
        },
    }

    bound = bind_gland_instance_authority_provenance(provenance)

    assert bound["gland_instance_authority_kind"] == (
        GLAND_AUTHORITY_SEMANTIC_PROXY
    )
    assert bound["derived_gland_component_map_sha256"] == output_digest
    assert "original_instance_mask_digest" not in bound
    assert bound["original_instance_mask_available_for_execution"] is False
    assert "removed_proxy_digest_from_original_instance_field" in bound[
        "authority_provenance_repairs"
    ]
    assert gland_instance_authority_status(bound)["valid"] is True


def test_dataset_native_gland_map_owns_original_instance_digest():
    source_digest = _digest("c")
    output_digest = _digest("d")
    provenance = {
        "source_tissue_mask_sha256": source_digest,
        "auxiliary_structure_sha256": {
            "native_gland_instance_map": output_digest,
        },
        "auxiliary_structure_provenance": {
            "native_gland_instance_map": {
                "producer_id": "dataset-gland-instance-loader-v1",
                "producer_version": "dataset-gland-instance-loader-v1",
                "source_tissue_mask_sha256": source_digest,
                "output_sha256": output_digest,
                "observation_scope": "dataset_native_annotation",
                "authority_kind": GLAND_AUTHORITY_DATASET_NATIVE,
            }
        },
    }

    bound = bind_gland_instance_authority_provenance(provenance)
    status = gland_instance_authority_status(bound)

    assert bound["original_instance_mask_digest"] == output_digest
    assert bound["original_instance_mask_available_for_execution"] is True
    assert "derived_gland_component_map_sha256" not in bound
    assert status["valid"] is True
    assert status["authority_kind"] == GLAND_AUTHORITY_DATASET_NATIVE


def test_hybrid_partition_is_reported_as_hybrid_not_native():
    instances = (
        _instance(
            "native-raster-cellvit-00001",
            class_id=1,
            source="instance_json_cellvit_seed",
            area_px=25,
        ),
        _instance(
            "native-raster-semantic-residual-00002",
            class_id=1,
            source="instance_json_semantic_seeded_residual",
            area_px=4,
        ),
        _instance(
            "native-raster-semantic-unseeded-00003",
            class_id=2,
            source="instance_json_semantic_unseeded",
            area_px=16,
        ),
    )

    summary = summarize_nucleus_instance_authority(instances)

    assert summary["observation_quality"] == NUCLEUS_AUTHORITY_HYBRID
    assert summary["native_seed_instance_count"] == 1
    assert summary["semantic_residual_instance_count"] == 2
    assert summary["native_complete_reference_count_by_class"] == {"1": 1}
    assert summary["native_seed_pixel_fraction"] == pytest.approx(25 / 45)


def test_add_only_no_fallback_mechanism_accepts_hybrid_native_references():
    instances = (
        _instance("native-1", class_id=1, source="instance_json_cellvit_seed"),
        _instance(
            "residual-1",
            class_id=1,
            source="instance_json_semantic_seeded_residual",
        ),
    )

    result = validate_mechanism_nucleus_authority(
        instances,
        allow_semantic_instance_fallback=False,
        required_cell_classes=(1,),
        actions=("retain", "add"),
    )

    assert result["passed"] is True
    assert result["observation_quality"] == NUCLEUS_AUTHORITY_HYBRID


def test_removal_no_fallback_mechanism_rejects_hybrid_partition():
    instances = (
        _instance("native-1", class_id=1, source="instance_json_cellvit_seed"),
        _instance(
            "residual-1",
            class_id=1,
            source="instance_json_semantic_unseeded",
        ),
    )

    result = validate_mechanism_nucleus_authority(
        instances,
        allow_semantic_instance_fallback=False,
        required_cell_classes=(1,),
        actions=("retain", "remove_whole"),
    )

    assert result["passed"] is False
    assert "hybrid_partition_cannot_authorize_removal" in result["reasons"]


def test_native_authority_floor_rejects_low_seed_coverage_and_missing_class1():
    summary = {
        "native_seed_instance_count": 5,
        "native_seed_instance_fraction": 5 / 150,
        "native_seed_pixel_fraction": 0.04,
        "native_complete_reference_count_by_class": {"1": 1},
    }

    reasons = validate_nucleus_authority_floor(
        summary,
        minimum_native_seed_count=4,
        minimum_native_seed_instance_fraction=0.05,
        minimum_native_seed_pixel_fraction=0.05,
        minimum_native_references_by_class={1: 4},
    )

    assert set(reasons) == {
        "native_seed_instance_fraction_below_threshold",
        "native_seed_pixel_fraction_below_threshold",
        "native_complete_class_1_reference_count_below_threshold",
    }


def test_tuple_instance_records_are_supported_for_runner_validation():
    native_component = np.zeros((8, 8), dtype=bool)
    native_component[1:4, 1:4] = True
    residual_component = np.zeros((8, 8), dtype=bool)
    residual_component[5:7, 5:7] = True

    summary = summarize_nucleus_instance_authority(
        (
            ("native-raster-cellvit-00001", 1, native_component),
            ("native-raster-semantic-residual-00002", 1, residual_component),
        )
    )

    assert summary["observation_quality"] == NUCLEUS_AUTHORITY_HYBRID
    assert summary["native_seed_pixels"] == 9
    assert summary["semantic_residual_pixels"] == 4


def test_hybrid_reference_count_uses_native_seeds_not_semantic_residuals():
    from phase3_joint_edit_refine.authority import (
        count_authoritative_complete_references,
    )

    instances = (
        _instance("native-1", class_id=1, source="instance_json_cellvit_seed"),
        _instance(
            "residual-1",
            class_id=1,
            source="instance_json_semantic_unseeded",
        ),
        _instance(
            "residual-2",
            class_id=1,
            source="instance_json_semantic_seeded_residual",
        ),
    )

    result = count_authoritative_complete_references(
        instances,
        allowed_cell_classes=(1,),
        allow_semantic_instance_fallback=True,
        library_reference_counts={1: 2},
    )

    assert result["same_patch_reference_count"] == 1
    assert result["library_reference_count"] == 2
    assert result["total_reference_count"] == 3


def test_pure_semantic_watershed_may_remain_a_declared_fallback_authority():
    from phase3_joint_edit_refine.authority import (
        count_authoritative_complete_references,
    )

    instances = (
        _instance(
            "semantic-1",
            class_id=2,
            source="semantic_distance_watershed",
        ),
    )

    result = count_authoritative_complete_references(
        instances,
        allowed_cell_classes=(2,),
        allow_semantic_instance_fallback=True,
    )

    assert result["same_patch_reference_count"] == 1
    assert result["total_reference_count"] == 1


def test_legacy_explicit_original_digest_infers_dataset_native_when_record_is_not_semantic():
    source_digest = _digest("e")
    output_digest = _digest("f")
    provenance = {
        "source_tissue_mask_sha256": source_digest,
        "original_instance_mask_digest": output_digest,
        "auxiliary_structure_sha256": {
            "native_gland_instance_map": output_digest,
        },
        "auxiliary_structure_provenance": {
            "native_gland_instance_map": {
                "producer_id": "fixture-native-loader-v1",
                "producer_version": "fixture-native-loader-v1",
                "source_tissue_mask_sha256": source_digest,
                "output_sha256": output_digest,
                "observation_scope": "native_annotation",
            }
        },
    }

    bound = bind_gland_instance_authority_provenance(provenance)

    assert bound["gland_instance_authority_kind"] == (
        GLAND_AUTHORITY_DATASET_NATIVE
    )
    assert gland_instance_authority_status(bound)["valid"] is True
