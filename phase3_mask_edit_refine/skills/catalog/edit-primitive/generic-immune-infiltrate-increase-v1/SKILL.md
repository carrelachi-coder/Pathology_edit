---
name: generic-immune-infiltrate-increase-v1
description: Deterministically expand an existing Stroma-to-Immune-infiltrate interface without changing unrelated tissue labels.
---

# Generic immune compartment increase

Read `references/mask_contract.json`. Convert only Stroma to Immune infiltrate
from an existing selected interface. Preserve all unrequested labels and reject
remote islands, deep notches, topology damage or untraceable area fallback.
