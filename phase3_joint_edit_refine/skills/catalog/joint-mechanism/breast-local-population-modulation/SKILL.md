---
name: breast-local-population-modulation
description: Plan and audit bounded breast cancer cell abundance or total cellularity edits without changing tissue masks. Use for breast cell-type-abundance or cellularity primitives.
---

# Breast local population modulation

Read `references/joint_contract.json`. Select one mask-defined component population zone. Resolve an observable CellViT class explicitly for abundance edits; preserve source-observed local class proportions for cellularity edits. Add only complete local same-class shapes or remove complete instances dispersed across the zone. Reject zones that intersect a prohibited or protected annotation ID, lack complete-instance authority, lack meaningful extent, or cannot preserve the requested tissue mask pixel-exactly. Do not infer an unannotated lumen, adipose space, retraction cleft, or in-situ identity from H&E.
