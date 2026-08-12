---
name: glas-gland-v1-joint
description: Compose joint tissue-nucleus edits for GLaS-derived gland masks. Use whenever the annotation profile is glas-gland-v1, including colorectal edits that must distinguish gland labels from the heterogeneous non-gland complement.
---

# GLaS joint annotation profile

1. Read the tissue profile at `../../../../../phase3_mask_edit_refine/skills/catalog/annotation-profile/glas-gland-v1/SKILL.md` for label semantics.
2. Read `references/joint_contract.json` for executable tissue, cell-placement, support, and provenance restrictions.
3. Treat canonical `Stroma` only as a heterogeneous non-gland complement; never assert that it is pure fibrous stroma or a reliable invasive-front host.
4. Do not execute colorectal tumor budding or malignant-gland burden growth under this profile. GLaS does not provide an explicit stromal authority, and H&E review alone must not manufacture one.
5. Limit executable edits to bounded cell-population or cellularity modulation in an H&E-confirmed non-luminal tissue zone while preserving the tissue mask pixel-exactly.
6. Preserve zero/background, gland geometry, lumen candidates, normal crypt, and all unrequested fine labels.
7. Abstain when H&E cannot distinguish the cell-only host zone from lumen, mucin, debris, muscle, vessel, inflammation, or unresolved benign/malignant gland rims.
