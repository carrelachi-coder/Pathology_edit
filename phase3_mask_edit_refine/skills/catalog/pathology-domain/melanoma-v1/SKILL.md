---
name: melanoma-v1
description: Plan melanoma mask edits using epidermal, dermal, tumor-nest, stromal, vascular, necrotic, and regression morphology. Use for pathology_domain_id melanoma-v1.
---

# Melanoma mask-edit planning

1. Read `references/mask_contract.json` first; enforce encoded epidermis/necrosis/vessel labels, component topology, and interface geometry. Then read `references/rules.json` for H&E interpretation.
2. Treat junctional component, ulceration, regression, and small-vessel protection as conditional unless supplied by an auxiliary structure map. Never infer them from a semantic gap or Stroma label.
3. Classify the interface as epidermal/junctional, cohesive dermal nest/sheet, discohesive cells, ulcerated surface, regression bed, necrosis, vessel, or ordinary stroma.
4. Preserve encoded labels and one interface band; reject islands, bridges, splits, merges, holes, and unrequested tissue changes.
5. Hand cohesive/discohesive morphology, ulceration, regression, and junctional appearance to generation/post-generation audit. Cite constraints and rules; abstain when anatomical stratum or auxiliary protection is missing.
