---
name: breast-cell-seeded-invasive-cord
description: Execute an explicitly requested Breast invasive cord as a source-scaled cell path followed by ordinary Tumor-mask derivation.
---

# Breast Cell-Seeded Invasive Cord

Use only semantic masks, nuclei masks, scene-graph features, and deterministic certificates.

- Require an external BCSS Tumor 1--operational Stroma 2 boundary.
- Build an extended, slightly curved invasion path in Stroma and arrange at least six tumor cells; keep most of the chain one to two cells wide and permit local proximal thickening to three cells.
- Derive fine-1 Tumor support from the proposed cell footprints with only cell-scale dilation and closing.
- Keep one connection to one parent component; reject symmetric tongues, capsules, wedges, broad boundary expansion, and detached islands.
- Reject a short or one-cell-thin result whose changed region is visually trivial even if its topology passes.
- The ordinary Tumor mask and nuclei mask remain the only generator conditions.
- Treat H&E appearance as a downstream veto, never as execution geometry authority.

Read every file in `references/` before promoting or evaluating this research mechanism.
