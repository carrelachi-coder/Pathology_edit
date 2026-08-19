---
name: glas-gland-v1
description: Interpret transformed GLaS masks with explicit separation between dataset-native gland instances and semantic connected-component exterior proxies. Use when annotation_profile_id is glas-gland-v1.
---

# glas-gland-v1 mask-only execution authority

1. Read `references/mask_contract.json`; execution uses digest-bound semantic masks, scene graphs, typed gland authorities, auxiliary maps, and deterministic certificates.
2. `dataset_native_instance` may own original GLaS gland identity. `semantic_connected_component_proxy` owns only transformed gland-support exterior topology and must never masquerade as an original instance mask.
3. The aligned nuclei mask is sufficient for deterministic mask-derived cell units and reusable same-patch morphology. Touching or imperfectly separated shapes are handled by packing/collision checks; failure to fit one template causes another certified template to be tried.
4. A CellViT/native nucleus-instance file is optional refinement, not an execution prerequisite.
5. Raw histology, overlays, crops, and reader boards are unavailable to the execution Planner and cannot invent unencoded structures or treatment effects.
6. Abstain only when the requested mask-level geometry, class, complete shape, count, or packing capacity cannot be certified.

## Gland versus nucleus digest namespace

- `original_instance_mask_digest` remains the nucleus-instance JSON/raster digest.
- `original_gland_instance_mask_digest` is reserved for a dataset-native gland-instance raster.
- `derived_gland_component_map_sha256` is reserved for a semantic gland-component exterior proxy.
- These fields are independent; one must never overwrite another.
