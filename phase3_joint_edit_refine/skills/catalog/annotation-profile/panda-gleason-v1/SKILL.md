---
name: panda-gleason-v1-joint
description: Compose joint tissue-nucleus edits under PANDA Gleason labels. Use when native Gleason pattern and lumen identity must remain bound to a prostate mechanism.
---

# PANDA joint annotation profile

1. Read the tissue profile at `../../../../../phase3_mask_edit_refine/skills/catalog/annotation-profile/panda-gleason-v1/SKILL.md`.
2. Read `references/joint_contract.json` and the bound native-pattern auxiliary provenance.
3. Match pattern 3, 4, or 5 mechanisms to their exact source fine label; never perform an implicit Gleason transition.
4. Protect lumina and internal cribriform spaces from tissue fill and nucleus placement.
5. Preserve unrequested pattern labels and background pixel-exactly.
6. Permit operational fine-2 stromal replacement only for an explicit post-treatment scenario, an existing Pattern-4/5 Tumor--Stroma interface, and a candidate that preserves Pattern 3 plus every unrequested Pattern-4/5 pixel. Fine-2 does not prove fibrosis or a tumor bed; raw H&E cannot supply treatment history.
7. Abstain when the source pattern or protected auxiliary structure is not observable.
