---
name: generic-immune-infiltrate-decrease-v1
description: Deterministically retreat an existing Immune-infiltrate-to-Stroma interface without changing unrelated tissue labels.
---

# Generic immune compartment decrease

Read `references/mask_contract.json`. Convert only Immune infiltrate to Stroma
from an existing selected interface. Preserve all unrequested labels and reject
remote cuts, topology damage or untraceable area fallback.
