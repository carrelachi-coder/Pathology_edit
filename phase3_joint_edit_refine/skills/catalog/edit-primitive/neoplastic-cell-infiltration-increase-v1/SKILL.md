---
name: neoplastic-cell-infiltration-increase-v1
description: Compile and audit a cell-only increase of neoplastic infiltration while preserving the tissue mask. Use for budding, single-file, sparse junctional, dispersed-front, or satellite-cell instructions selected through a cancer-specific mechanism.
---

# Deprecated: neoplastic cell infiltration increase

This ID is retained only so legacy manifests fail with an explicit migration
message. It must not be selected or executed. Choose exactly one of:

- `neoplastic-microinfiltration-increase-v1` for cell-only spread;
- `invasive-front-expansion-v1` for coupled tissue/front displacement;
- `structural-void-spread-v1` for a verified native void such as airspace;
- `architecture-progression-v1` for a fine architectural identity transition.

1. Preserve the tissue mask pixel-exactly.
2. Require a verified tumor-to-host-tissue interface and a bounded cell-placement halo.
3. Use ProbNet only to rank legal anchors; use deterministic templates for bud, pair, file, cluster, or cord relationships.
4. Treat added cells as an explicit count/extent increment, never as tissue burden.
5. Read [primitive_contract.json](references/primitive_contract.json) for executable requirements.
