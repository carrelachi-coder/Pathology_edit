---
name: cellularity-decrease-v1
description: Compile a local cell-only decrease in total cellularity without changing tissue labels. Use only when a reviewed mechanism and visible H&E observation define a legal causal interface, complete removable instances, a graded depletion field and an unchanged outer reference.
---

# Cellularity decrease

Preserve tissue. Require Planner-selected component, interface and anchor IDs plus a visible observation; abstain when the image supports no such anchor. Compile non-overlapping core, transition and outer-reference bands from local nucleus diameter. Remove only complete non-border instances, thin the core more strongly than the transition, preserve a residual population in both bands, leave the outer reference unchanged and preserve local class composition. Do not use ProbNet for removal-only placement. Read [primitive_contract.json](references/primitive_contract.json).
