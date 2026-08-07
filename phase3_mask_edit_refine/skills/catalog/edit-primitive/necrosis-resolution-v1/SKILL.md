---
name: necrosis-resolution-v1
description: Plan and audit replacement of explicitly annotated intratumoral necrosis by adjacent viable tumor. Use for primitive_id necrosis-resolution-v1 when the annotation profile contains Necrosis.
---

# Resolve intratumoral necrosis

Read `references/mask_contract.json`. Require a real Necrosis-to-Tumor interface and viable tumor evidence around the selected focus. Convert only Necrosis to Tumor along executable anchors. Reject necrosis adjacent only to stroma, lumen or non-tissue; v1 never guesses replacement tissue.
