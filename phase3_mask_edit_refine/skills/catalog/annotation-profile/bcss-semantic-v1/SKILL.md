---
name: bcss-semantic-v1
description: Interpret masks that follow the versioned BCSS-derived 22-class-to-unified semantic protocol, independently of the tissue organ. Use when annotation_profile_id is bcss-semantic-v1.
---

# BCSS annotation protocol

1. Read `references/mask_contract.json` first and enforce all active checker-backed constraints. Then read `references/rules.json` for remap interpretation and H&E caveats.
2. Preserve unified zero exactly: it collapses outside-ROI, excluded, and undetermined pixels and is not an editable tissue source.
3. Retain DCIS and angioinvasion fine identities. Do not infer a unique native subtype from merged `Stroma`, `Blood vessel`, or `Other tissue` labels.
4. Preserve every unrequested minority label, even when its component is smaller than a morphology kernel.
5. Intersect this profile with domain/primitive constraints, cite `constraint_id` and `rule_id`, and hand native-tissue appearance to generation rather than claiming it from the unified mask.
