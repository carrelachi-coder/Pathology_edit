---
name: orca-semantic-v1-joint
description: Compose joint tissue-nucleus edits under the ORCA semantic annotation convention. Use whenever annotation_profile_id is orca-semantic-v1, including cross-organ cases whose pathology domain is not oral SCC.
---

# ORCA joint annotation profile

1. Read the tissue profile at `../../../../../phase3_mask_edit_refine/skills/catalog/annotation-profile/orca-semantic-v1/SKILL.md`.
2. Read `references/joint_contract.json` for joint prohibitions and provenance.
3. Treat `Other tissue` as unresolved; never rename it stroma, immune tissue, normal epithelium, or necrosis.
4. Treat fragmented non-tissue/background as prohibited for tissue changes, nucleus placement, and generation support.
5. Permit only mechanisms whose source and target semantics survive this coarse ontology.
6. Abstain when a mechanism requires an unavailable tissue identity.
