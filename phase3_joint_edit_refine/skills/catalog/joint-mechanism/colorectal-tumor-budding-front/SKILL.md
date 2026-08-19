---
name: colorectal-tumor-budding-front
description: Plan and audit non-diagnostic periglandular class-1 scatter or compact two-to-four-cell foci from semantic tissue and nuclei masks, with optional native instance refinement.
---

# Colorectal periglandular dispersion

1. Read `references/joint_contract.json` and require one digest-bound gland-exterior authority, outer-boundary certificate, and legal annulus certificate.
2. Accept `dataset_native_instance` for original gland-instance identity. Accept `semantic_connected_component_proxy` for semantic gland-support component exterior topology. Neither mode authorizes ITBCC, invasive-front, prognostic, or treatment claims.
3. Keep the operational non-gland tissue label pixel-exact; it is a heterogeneous receiving label, not histologically pure stroma.
4. The source nuclei mask is sufficient execution authority: deterministic per-class watershed/connected shapes may be copied, packed, or removed as complete mask-derived units. A supplied CellViT/native instance file may refine identity but is never required.
5. Use scatter for separated single-cell foci and small-cluster for multiple compact two-to-four-cell foci. Exact count, coordinates, packing, and annulus ownership remain compiler-controlled.
6. Exclude lumen/hole contours, zero, normal glands, every gland-support component, and every unrequested nucleus.
7. Reject bulk fill, solid bridges, remote cells, and diagnostic tumor-budding claims.

## Gland versus nucleus digest namespace

- `original_instance_mask_digest` remains the nucleus-instance JSON/raster digest.
- `original_gland_instance_mask_digest` is reserved for a dataset-native gland-instance raster.
- `derived_gland_component_map_sha256` is reserved for a semantic gland-component exterior proxy.
- These fields are independent; one must never overwrite another.
