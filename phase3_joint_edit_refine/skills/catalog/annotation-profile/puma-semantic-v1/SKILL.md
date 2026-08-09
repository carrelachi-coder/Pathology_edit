---
name: puma-semantic-v1-joint
description: Compose joint tissue-nucleus edits under the PUMA melanoma profile. Use when primary/metastatic context, epidermis, vessel, necrosis, and separate cell observations must be respected.
---

# PUMA joint annotation profile

1. Read the tissue profile at `../../../../../phase3_mask_edit_refine/skills/catalog/annotation-profile/puma-semantic-v1/SKILL.md`.
2. Read `references/joint_contract.json` and require primary/metastatic plus source-site provenance.
3. Preserve epidermal relationship for junctional mechanisms and protect vessel, necrosis, and background.
4. Do not synthesize an immune tissue label from cell annotations or canonical stroma.
5. Bind cell changes to the independent CellViT observation profile and melanoma population profile.
6. Abstain when a proposed mechanism conflicts with primary/metastatic or epidermal context.
