---
name: necrosis-appearance-v1-joint
description: Compile a joint necrosis-appearance edit. Use when the user simply requests increased necrosis and an organ mechanism plus annotation profile can represent a tumor-to-necrosis transition.
---

# Joint necrosis appearance

1. Read `references/primitive_contract.json`.
2. Require tissue change from viable tumor into existing, verified intratumoral necrosis.
3. Remove only complete viable source nuclei intersecting the changed region.
4. Regenerate the sparse cell population permitted by the organ mechanism; necrotic tissue is not assumed acellular.
5. Delegate only non-nuclear debris, karyorrhexis, eosinophilia, and related appearance to bounded rendering.
6. Reject de-novo remote necrosis, border-censored shape reuse, or an unbounded empty hole.
