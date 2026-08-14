---
name: glas-gland-v1-joint
description: Compose joint tissue-nucleus edits for GLaS-derived gland masks. Use whenever the annotation profile is glas-gland-v1, including colorectal edits that must distinguish gland labels from the heterogeneous non-gland complement.
---

# GLaS joint annotation profile

1. Read the tissue profile at `../../../../../phase3_mask_edit_refine/skills/catalog/annotation-profile/glas-gland-v1/SKILL.md` for label semantics.
2. Read `references/joint_contract.json` for executable tissue, cell-placement, support, and provenance restrictions.
3. Treat canonical `Stroma` only as a heterogeneous non-gland complement; never assert that it is pure fibrous stroma or a reliable invasive-front host.
4. Keep malignant-gland burden growth closed without a dedicated whole-gland executor and receiving-validity map. Permit only explicitly downgraded, non-diagnostic periglandular scatter or one-to-four-cell synthetic foci when the native gland-instance exterior and legal annulus are digest-bound.
5. Limit executable population edits to compiler-certified tissue-mask components while preserving the tissue mask pixel-exactly.
6. Preserve zero/background, every gland-instance pixel, lumen/hole contours, normal glands, and all unrequested fine labels.
7. Abstain when the profile labels, native gland instances, or digest-bound auxiliary maps cannot certify a legal host zone; raw H&E cannot supply missing execution authority.
