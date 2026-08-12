---
name: cellularity-decrease-v1
description: Compile a local cell-only decrease in total cellularity without changing tissue labels. Use only when a reviewed mechanism and a certified mask-graph anchor define complete removable instances, a graded depletion field, and an unchanged outer reference.
---

# Cellularity decrease

Preserve tissue. Require Planner-selected component, interface and anchor IDs bound to a deterministic mask-graph certificate; abstain when no such anchor exists. Compile a spatial density field from interface span and local nucleus diameter: one bounded core, multiple outward transition subbands with monotonically decreasing removal fractions, and an unchanged outer reference. Derive the removal count from the observed complete-instance population inside that field; use case count metadata only as a safety interval. Remove only complete non-border instances, preserve residual cells and local class composition, and never turn the field into a cell-free hole. Do not use ProbNet for removal-only placement. Read [primitive_contract.json](references/primitive_contract.json).
