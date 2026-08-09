---
name: necrosis-resolution-v1-joint
description: Compile a joint necrosis-resolution edit. Use when the user simply requests reduced necrosis and an adjacent viable tumor mechanism is representable.
---

# Joint necrosis resolution

1. Read `references/primitive_contract.json`.
2. Require an existing necrosis-to-viable-tumor interface; do not convert unrelated non-tumor tissue.
3. Remove only complete source instances owned by the changed region.
4. Use the mature source-first population pipeline to regenerate viable neoplastic nuclei in the resolved region.
5. Preserve exterior nuclei and all unrequested tissue labels.
6. Reject unsupported claims that the frozen H&E generator cannot express.
