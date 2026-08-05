---
name: lung-carcinoma-v1
description: Plan lung carcinoma mask edits with alveolar, bronchial, glandular, solid, necrotic, and stromal interface constraints. Use for pathology_domain_id lung-carcinoma-v1.
---

# Lung carcinoma mask-edit planning

1. Read `references/mask_contract.json` first; enforce protected labels, interface band, topology, and conditional auxiliary-structure constraints. Then read `references/rules.json` for H&E pattern interpretation.
2. Claim deterministic airspace/lumen/core protection only when these are encoded in native or auxiliary maps. H&E recognition alone supports Planner veto, not a mask guarantee.
3. Classify lepidic, acinar, papillary, micropapillary, solid, or squamous/keratinizing architecture and identify supporting structures.
4. Preserve encoded labels/maps, select one broad interface, and reject topology repair, holes, merges, remote foci, and short/deep edits.
5. Hand actual lung growth pattern and microscopic airspace/gland/core appearance to generation/post-generation audit. Cite constraints and rules; abstain when structure maps or site/remap provenance are insufficient.
