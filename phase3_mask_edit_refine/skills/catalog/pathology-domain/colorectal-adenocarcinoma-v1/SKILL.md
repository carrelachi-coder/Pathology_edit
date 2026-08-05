---
name: colorectal-adenocarcinoma-v1
description: Plan colorectal adenocarcinoma mask edits with gland boundary, lumen, differentiation, and tumor-stroma interface constraints. Use for pathology_domain_id colorectal-adenocarcinoma-v1.
---

# Colorectal adenocarcinoma mask-edit planning

1. Read `references/mask_contract.json` first; enforce its label, interface, topology, and native-instance requirements. Then read `references/rules.json` for H&E interpretation and rendering handoff.
2. Claim gland-object/lumen protection only when a native instance or auxiliary structure map is mounted. Treat H&E-only gland protection as conditional and abstain on ambiguous cuts.
3. Trace normal crypts, neoplastic glands, lumina, cribriform spaces, mucin, dirty necrosis, buds/discohesive cells, and desmoplastic stroma on H&E.
4. Preserve declared fine labels and mounted gland/lumen topology; use one broad interface band and reject split/merge/hole repair.
5. Hand epithelial continuity, mucin/debris identity, budding, and differentiation appearance to generation/post-generation audit. Cite both constraint and rule IDs.
