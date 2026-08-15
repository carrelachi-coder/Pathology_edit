---
name: orca-semantic-v1
description: "Interpret masks following the ORCA oral cancer three-class protocol: non-tissue, mixed non-carcinoma tissue, and carcinoma. Use when annotation_profile_id is orca-semantic-v1 with any compatible pathology domain."
---

# orca-semantic-v1 mask-only execution authority

1. Read `references/mask_contract.json` first and enforce only digest-bound provenance, semantic masks, scene graphs, native annotations, auxiliary structure maps, and deterministic candidate certificates.
2. Raw histology, overlays, crops, reader boards, and renamed image panels are unavailable to the execution Planner and Critic.
3. Unencoded glands, lumina, epidermis, lung structures, invasive fronts, grades, and treatment effects cannot be inferred for anchor selection or veto.
4. Rules marked `reader_only_pathology_fact` may guide post-generation reader QA only and are excluded from the execution knowledge bundle.
5. Abstain whenever the required native or auxiliary authority is absent; all capabilities remain draft and shadow-only.
