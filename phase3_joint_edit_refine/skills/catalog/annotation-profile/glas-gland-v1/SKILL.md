---
name: glas-gland-v1-joint
description: Compose joint tissue-nucleus edits for GLaS-derived gland masks. Use whenever the annotation profile is glas-gland-v1, including colorectal edits that must distinguish gland labels from the heterogeneous non-gland complement.
---

# GLaS joint annotation profile

1. Read the tissue profile at `../../../../../phase3_mask_edit_refine/skills/catalog/annotation-profile/glas-gland-v1/SKILL.md` for label semantics.
2. Read `references/joint_contract.json` for executable tissue, cell-placement, support, and provenance restrictions.
3. Treat canonical `Stroma` only as a non-gland complement; never assert that it is pure fibrous stroma.
4. Require `gland_or_lumen_support` and its source-mask-bound producer provenance for gland-unit mechanisms.
5. Preserve zero/background, enclosed lumen, normal crypt, and all unrequested fine labels.
6. Abstain when H&E cannot distinguish true receiving tissue from lumen, mucin, debris, muscle, vessel, or inflammation.
