---
name: ignite-semantic-v1
description: Interpret masks following the repository's IGNITE-derived unified lung tissue protocol. Use when annotation_profile_id is ignite-semantic-v1, independently from pathology_domain_id.
---

# IGNITE annotation protocol

1. Read `references/mask_contract.json` first; enforce remap/site provenance, zero, declared transitions, and topology. Then read `references/rules.json` for native-tissue interpretation.
2. Preserve collapsed zero exactly because it may combine unannotated and background pixels.
3. Do not reverse-infer native tissue identity from coarse `Stroma`, `Normal epithelium`, `Immune infiltrate`, or `Other tissue`; consult H&E and retained native masks.
4. Retrieve statistics and references from matching primary/metastatic site, specimen type, center/scanner, and scale before broader fallbacks.
5. Cite constraints and rules, abstain when remap/site context is missing, and hand native lung-tissue appearance to generation rather than inferring it from unified IDs.
