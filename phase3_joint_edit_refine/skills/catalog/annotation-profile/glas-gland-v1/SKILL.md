---
name: glas-gland-v1-joint
description: Compose joint tissue-nucleus edits for GLaS-derived gland masks while distinguishing dataset-native gland instances from semantic connected-component exterior proxies.
---

# GLaS joint annotation profile

1. Read the tissue profile at `../../../../../phase3_mask_edit_refine/skills/catalog/annotation-profile/glas-gland-v1/SKILL.md` for label semantics.
2. Treat canonical `Stroma` only as a heterogeneous non-gland complement; never assert pure fibrous stroma or a histologically proven invasive front.
3. Keep malignant-gland tissue growth closed without a dedicated whole-gland executor and receiving-validity map.
4. Distinguish `dataset_native_instance` from `semantic_connected_component_proxy`; the latter owns exterior topology only and is never written as `original_instance_mask_digest`.
5. The aligned source nuclei mask is the normal cell-instance and morphology source. Deterministic mask-derived units support complete-shape addition and whole-unit deletion; optional CellViT/native instances refine but do not gate execution.
6. Permit non-diagnostic periglandular scatter or compact two-to-four-cell foci only inside the selected gland-authority exterior annulus.
7. Preserve zero/background, every gland-support pixel, lumen/hole contours, normal glands, and all unrequested fine labels. Raw H&E cannot supply missing execution geometry.

## Gland versus nucleus digest namespace

- `original_instance_mask_digest` remains the nucleus-instance JSON/raster digest.
- `original_gland_instance_mask_digest` is reserved for a dataset-native gland-instance raster.
- `derived_gland_component_map_sha256` is reserved for a semantic gland-component exterior proxy.
- These fields are independent; one must never overwrite another.
