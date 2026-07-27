from dataset_config import get_config, list_datasets
from phase3_mask_edit.core.labels import MaskProfileSchema


def test_unified_coarse_profile_is_registered_and_identity_mapped():
    cfg = get_config("unified_coarse")

    assert "UNIFIED_COARSE" in list_datasets()
    assert cfg.name == "UNIFIED_COARSE"
    assert cfg.cancer_type == "organ_agnostic"
    assert cfg.to_coarse_map == {idx: idx for idx in range(8)}
    assert cfg.to_fine_map == {idx: idx for idx in range(8)}
    assert cfg.tumor_ids == (1,)
    assert cfg.stroma_ids == (2,)
    assert cfg.cancer_type_index == -1


def test_unified_coarse_schema_exposes_shared_labels_without_dataset_claim():
    schema = MaskProfileSchema.from_reference_profile("UNIFIED_COARSE")

    assert schema.reference_profile == "UNIFIED_COARSE"
    assert schema.resolve_fine_ids("Tumor") == (1,)
    assert schema.resolve_fine_ids("Stroma") == (2,)
    assert "__profile__" in schema.semantic_warnings
