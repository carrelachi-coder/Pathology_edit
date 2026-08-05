---
name: oral-squamous-cell-carcinoma-v1
description: Plan oral squamous cell carcinoma mask edits using invasive nests, cords, keratinization, mucosa, muscle, salivary tissue, and tumor-front morphology. Use for pathology_domain_id oral-squamous-cell-carcinoma-v1.
---

# Oral SCC mask-edit planning

1. Read `references/mask_contract.json` first; enforce the coarse label contract, immutable background, one-interface band, and component topology. Then read `references/rules.json` for H&E interpretation.
2. Claim deterministic surface/keratin/nerve/vessel protection only when native or auxiliary maps encode them. ORCA Other tissue alone cannot supply these structures.
3. Classify surface/dysplastic epithelium, cohesive nests/cords, keratin pearls, satellites, muscle, salivary tissue, nerve, and vessel on H&E.
4. Preserve encoded/prohibited regions, use a broad carcinoma/tissue interface, and reject deep notches, background anchors, bridges, islands, splits, merges, and holes.
5. Hand nest/cord, keratin, satellite, nerve, and vessel appearance to generation/post-generation audit. Cite constraints and rules; abstain when the coarse mask cannot support raster protection.
