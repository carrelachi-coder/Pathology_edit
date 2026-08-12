---
name: residual-tumor-fragmentation-v1
description: Convert one invasive-tumor component into a bounded set of nonempty residual foci under explicit post-treatment residual-disease semantics.
---

# Residual-tumor fragmentation

Read `references/primitive_contract.json`. This primitive permits controlled
source splitting only inside the pre-existing invasive component. It requires
2–6 residual foci, minimum island size, residual area floor and no new empty
holes. It cannot invent a historical tumor bed.
