from __future__ import annotations

from phase3_joint_edit_refine.authority import (
    GLAND_AUTHORITY_DATASET_NATIVE,
    GLAND_AUTHORITY_SEMANTIC_PROXY,
    bind_gland_instance_authority_provenance,
    gland_instance_authority_status,
    semantic_gland_component_authority_metadata,
)


def _digest(character: str) -> str:
    return character * 64


def _native_fixture(*, nucleus_digest: str | None) -> dict:
    tissue_digest = _digest("a")
    gland_digest = _digest("b")
    provenance = {
        "source_tissue_mask_sha256": tissue_digest,
        "auxiliary_structure_sha256": {
            "native_gland_instance_map": gland_digest,
        },
        "auxiliary_structure_provenance": {
            "native_gland_instance_map": {
                "producer_id": "synthetic-native-gland-instance-fixture",
                "producer_version": "v1",
                "observation_scope": "native_instance",
                "source_tissue_mask_sha256": tissue_digest,
                "output_sha256": gland_digest,
            }
        },
    }
    if nucleus_digest is not None:
        provenance.update(
            {
                "source_nuclei_instances_sha256": nucleus_digest,
                "original_instance_mask_digest": nucleus_digest,
            }
        )
    return provenance


def test_native_gland_digest_does_not_overwrite_nucleus_instance_digest():
    nucleus_digest = _digest("c")
    bound = bind_gland_instance_authority_provenance(
        _native_fixture(nucleus_digest=nucleus_digest)
    )

    assert bound["gland_instance_authority_kind"] == (
        GLAND_AUTHORITY_DATASET_NATIVE
    )
    assert bound["original_instance_mask_digest"] == nucleus_digest
    assert bound["original_gland_instance_mask_digest"] == _digest("b")
    assert bound["original_gland_instance_mask_available_for_execution"] is True
    assert "preserved_distinct_nucleus_instance_digest" in bound[
        "authority_provenance_repairs"
    ]
    assert gland_instance_authority_status(bound)["valid"] is True


def test_legacy_native_gland_alias_is_retained_when_no_nucleus_digest_exists():
    bound = bind_gland_instance_authority_provenance(
        _native_fixture(nucleus_digest=None)
    )

    assert bound["original_gland_instance_mask_digest"] == _digest("b")
    assert bound["original_instance_mask_digest"] == _digest("b")
    assert bound["original_instance_mask_available_for_execution"] is True
    assert gland_instance_authority_status(bound)["valid"] is True


def test_semantic_proxy_preserves_a_distinct_nucleus_digest():
    tissue_digest = _digest("d")
    proxy_digest = _digest("e")
    nucleus_digest = _digest("f")
    record = {
        "producer_id": "joint-semantic-topology-auxiliary-v2",
        "producer_version": "joint-semantic-topology-auxiliary-v2",
        "observation_scope": "semantic_fine_mask_topology_only",
        "source_tissue_mask_sha256": tissue_digest,
        "output_sha256": proxy_digest,
        **semantic_gland_component_authority_metadata(
            source_tissue_mask_sha256=tissue_digest,
            output_sha256=proxy_digest,
        ),
    }
    provenance = {
        "source_tissue_mask_sha256": tissue_digest,
        "source_nuclei_instances_sha256": nucleus_digest,
        "original_instance_mask_digest": nucleus_digest,
        "auxiliary_structure_sha256": {
            "native_gland_instance_map": proxy_digest,
        },
        "auxiliary_structure_provenance": {
            "native_gland_instance_map": record,
        },
    }

    bound = bind_gland_instance_authority_provenance(provenance)

    assert bound["gland_instance_authority_kind"] == (
        GLAND_AUTHORITY_SEMANTIC_PROXY
    )
    assert bound["original_instance_mask_digest"] == nucleus_digest
    assert "original_gland_instance_mask_digest" not in bound
    assert bound["original_gland_instance_mask_available_for_execution"] is False
    assert gland_instance_authority_status(bound)["valid"] is True


def test_gland_authority_status_rejects_top_level_digest_detachment():
    bound = bind_gland_instance_authority_provenance(
        _native_fixture(nucleus_digest=_digest("c"))
    )
    tampered = {
        **bound,
        "gland_instance_authority_sha256": _digest("0"),
    }

    status = gland_instance_authority_status(tampered)

    assert status["valid"] is False
    assert "top_level_gland_authority_digest_mismatch" in status["violations"]
