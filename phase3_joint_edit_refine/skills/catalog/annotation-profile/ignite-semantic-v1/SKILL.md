---
name: ignite-semantic-v1-joint
description: Compose joint tissue-nucleus edits under the IGNITE lung annotation profile. Use when bronchial, alveolar, cartilage, muscle, fluid, inflammation, and other native structures must not collapse into generic stroma.
---

# IGNITE joint annotation profile

1. Read the tissue profile at `../../../../../phase3_mask_edit_refine/skills/catalog/annotation-profile/ignite-semantic-v1/SKILL.md`.
2. Read `references/joint_contract.json` and require source-site/specimen provenance.
3. Resolve the native fine label before authorizing any receiving interface.
4. Protect bronchial epithelium, alveolar framework, cartilage/bone, muscle, fluid/mucus, vessels, erythrocytes, macrophage regions, and background unless explicitly targeted by a supported primitive.
5. Require auxiliary structure evidence for lepidic and acinar/papillary mechanisms.
6. Abstain when canonical stroma hides an incompatible native structure.
