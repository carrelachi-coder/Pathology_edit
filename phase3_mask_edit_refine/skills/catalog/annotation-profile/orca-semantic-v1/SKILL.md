---
name: orca-semantic-v1
description: "Interpret masks following the ORCA oral cancer three-class protocol: non-tissue, mixed non-carcinoma tissue, and carcinoma. Use when annotation_profile_id is orca-semantic-v1 with any compatible pathology domain."
---

# ORCA annotation protocol

1. Read `references/mask_contract.json` first; enforce its provenance, fragmented-zero, three-class transition, and topology constraints. Then read `references/rules.json` for H&E interpretation.
2. Interpret only three native meanings: non-tissue, non-carcinoma tissue, and carcinoma. Keep `Other tissue` heterogeneous; never rename it stroma, immune, mucosa, muscle, salivary gland, vessel, or normal epithelium.
3. Exclude every non-tissue component from source selection, target support, edit bands, morphology kernels, and interface anchors. Preserve every zero pixel exactly, including fragmented internal holes and islands. Never bridge or fill across zero.
4. Intersect these annotation rules with the pathology-domain and primitive rules. Reject any primitive whose required tissue class is unavailable.
5. Put every active `constraint_id`, applied `rule_id`, and unresolved ambiguity into `EditPlan`. Hand nest/cord/keratin morphology to generation/post-generation audit; abstain if H&E cannot distinguish tissue from whitespace, tear, fold, or lumen.
