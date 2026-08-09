---
name: bcss-semantic-v1-joint
description: Compose joint tissue-nucleus edits under the BCSS semantic profile. Use for breast or cross-domain patches annotated with bcss-semantic-v1.
---

# BCSS joint annotation profile

1. Read the tissue profile at `../../../../../phase3_mask_edit_refine/skills/catalog/annotation-profile/bcss-semantic-v1/SKILL.md`.
2. Read `references/joint_contract.json` before exposing mechanisms.
3. Resolve native fine labels before using canonical stroma, vessel, necrosis, immune aggregate, or benign epithelium.
4. Preserve background and all unrequested fine labels pixel-exactly.
5. Allow stroma mechanisms only where the fine label and H&E support true stroma; glandular secretion is not stroma authority.
6. Keep added nuclei compatible with the resolved target fine label.
