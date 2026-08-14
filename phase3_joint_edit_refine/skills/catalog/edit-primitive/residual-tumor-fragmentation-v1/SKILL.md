---
name: residual-tumor-fragmentation-v1
description: Convert one invasive-tumor component into a bounded set of nonempty residual foci under explicit post-treatment residual-disease semantics.
---

# Residual-tumor fragmentation

Read `references/primitive_contract.json`. This primitive permits controlled
source splitting only inside the pre-existing invasive component. It requires
three to eight separated, area-meaningful residual foci, a visible converted
corridor fraction, balanced residual mass, an absolute island-size floor, a
2.5% relative focus floor, a residual-area floor and no new empty holes. The
relative threshold is an engineering proxy; the absolute floor still rejects
raster speckles. A large retained mass with tiny satellites or a local boundary
notch is not fragmentation. It cannot invent a historical tumor bed.
