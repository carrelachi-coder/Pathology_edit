---
name: invasive-cord-formation-v1
description: Form a cell-seeded narrow invasive cord and derive its ordinary Tumor mask support from the placed tumor-cell footprints.
---

# Invasive Cord Formation

Use for an explicitly requested cord or strand at a certified external Tumor--host boundary.

- Execute `path -> cells -> Tumor mask`; do not start from a macroscopic protrusion.
- Lay out an extended, slightly curved, source-scaled chain with at least six realized tumor cells; keep most of it one to two cells wide and allow local proximal thickening up to three cells.
- Derive the ordinary Tumor tissue label only from the placed cell footprints using cell-scale dilation and closing.
- Allow mild width and spacing variation; reject a symmetric capsule, triangular wedge, needle tip, broad sheet, or detached island.
- Keep the cord connected to one parent Tumor component and keep every changed tissue pixel in the certified host compartment.
- Reject a visually trivial short chain even when its tip shape is otherwise valid.
- Use only existing Tumor tissue and nuclei channels; do not add a new support-mask channel.
- Report synthetic cord formation, not a diagnosed invasive subtype or dynamic invasion.

Read `references/primitive_contract.json` and `references/evidence.json` before execution.
