---
name: local-invasive-clearance-v1
description: Clear invasive tumor only inside an explicit user-supplied local ROI without claiming whole-lesion response.
---

# Local invasive clearance

Read `references/primitive_contract.json`. This conditional primitive requires
a digest-bound `local_clearance_roi`; no ROI means abstain. It may remove all
invasive tumor inside that ROI while preserving everything outside, but never
means pCR or complete patient response.
