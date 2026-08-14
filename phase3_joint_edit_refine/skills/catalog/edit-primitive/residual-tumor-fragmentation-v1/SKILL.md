---
name: residual-tumor-fragmentation-v1
description: Convert one invasive-tumor component into a bounded set of nonempty residual foci under explicit post-treatment residual-disease semantics.
---

# Residual-tumor fragmentation

Read `references/primitive_contract.json`. This primitive permits controlled
source splitting only inside the pre-existing invasive component. It requires
three to six separated, area-meaningful residual foci, a visible converted
corridor fraction, balanced residual mass, minimum island size, residual area
floor and no new empty holes. A large retained mass with tiny satellites or a
local boundary notch is not fragmentation. It cannot invent a historical tumor
bed.
