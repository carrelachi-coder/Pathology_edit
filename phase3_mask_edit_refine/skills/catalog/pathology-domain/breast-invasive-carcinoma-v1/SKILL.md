---
name: breast-invasive-carcinoma-v1
description: Plan breast carcinoma mask edits using breast gland, duct, lobule, in-situ, invasive-front, and stromal morphology. Use only when pathology_domain_id is breast-invasive-carcinoma-v1.
---

# Breast carcinoma mask-edit planning

1. Read `references/mask_contract.json` first and treat every active mask constraint as mandatory. Then read `references/rules.json` for H&E interpretation and generation handoff.
2. Enforce only what the declared semantic/native/auxiliary masks expose. Mark H&E-only structures conditional; never present their preservation as a deterministic mask guarantee.
3. Classify the local structure on H&E as benign duct/lobule, DCIS-involved duct, cohesive invasive NST-like front, discohesive/single-file invasion, adipose front, desmoplastic stroma, or vascular space.
4. Preserve complete encoded benign/in-situ/vessel regions and fine IDs. Use broad boundary displacement; reject topology repair and unrequested transitions.
5. Hand microscopic duct, lobule, invasive-pattern, adipose, and vessel appearance to generation/post-generation audit. Cite both `constraint_id` and `rule_id`; abstain when protection cannot be rasterized reliably.
