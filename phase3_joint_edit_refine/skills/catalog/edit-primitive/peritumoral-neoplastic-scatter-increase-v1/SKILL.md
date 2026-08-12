---
name: peritumoral-neoplastic-scatter-increase-v1
description: Add complete single class-1 nuclei to a certified outer Tumor--host annulus while preserving the tissue mask.
---

# Peritumoral Neoplastic Scatter Increase

Use this primitive for a cell-only increase of sparse complete neoplastic instances next to an annotated Tumor component.

- Preserve the tissue mask pixel-exactly.
- Bind an external Tumor--host interface and its certified receiving-side annulus.
- Use the `single` layout; ProbNet ranks legal centers and never owns counts or contours.
- Reject internal holes, remote foci, protected regions, overlap, truncation, or missing same-class shape authority.
- Do not report single-file invasion, tumor budding, EMT, or prognosis.

Read `references/primitive_contract.json` and `references/evidence.json` before execution.
