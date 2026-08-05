---
name: prostate-adenocarcinoma-v1
description: Plan prostate adenocarcinoma mask edits using gland, lumen, benign epithelium, stroma, and Gleason architectural constraints. Use only for pathology_domain_id prostate-adenocarcinoma-v1.
---

# Prostate adenocarcinoma mask-edit planning

1. Read `references/mask_contract.json` first; apply all active mask constraints. Then read `references/rules.json` for Gleason H&E interpretation and rendering handoff.
2. Treat fine-ID transitions and semantic topology as mask-enforceable. Treat gland/lumen preservation as conditional on a native or auxiliary structure map; H&E alone does not make it deterministic.
3. Classify the anchor as benign gland, pattern-3 well-formed gland, pattern-4 fused/poorly formed/cribriform/glomeruloid unit, pattern-5 non-gland-forming tumor, lumen, or fibromuscular stroma.
4. Preserve encoded fine IDs, gland/lumen maps, components, holes, and one broad directed interface. Reject implicit grade change and topology repair.
5. Hand actual pattern-3/4/5 microscopic architecture to generation/post-generation audit. Cite both constraint and pathology rule IDs; abstain when the required structure map is absent.
