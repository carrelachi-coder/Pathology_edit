---
name: oral-squamous-cell-carcinoma-v1
description: Plan oral squamous cell carcinoma mask edits using invasive nests, cords, keratinization, mucosa, muscle, salivary tissue, and tumor-front morphology. Use for pathology_domain_id oral-squamous-cell-carcinoma-v1.
---

# oral-squamous-cell-carcinoma-v1 mask-only execution authority

1. Read `references/mask_contract.json` first and enforce only digest-bound provenance, semantic masks, scene graphs, native annotations, auxiliary structure maps, and deterministic candidate certificates.
2. Raw histology, overlays, crops, reader boards, and renamed image panels are unavailable to the execution Planner and Critic.
3. Unencoded glands, lumina, epidermis, lung structures, invasive fronts, grades, and treatment effects cannot be inferred for anchor selection or veto.
4. Rules marked `reader_only_pathology_fact` may guide post-generation reader QA only and are excluded from the execution knowledge bundle.
5. Abstain whenever the required native or auxiliary authority is absent; all capabilities remain draft and shadow-only.
