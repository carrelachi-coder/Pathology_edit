---
name: stroma-increase-v1
description: Compile and audit a joint stroma-area increase with target-stroma cell regeneration. Use when an instruction explicitly expands an available Stroma label and the annotation profile distinguishes stroma from generic other tissue.
---

# Stroma increase

This primitive has no standalone biological meaning. It may execute only after
the Planner selects an independent mechanism such as documented
treatment-associated fibrotic replacement and the annotation profile exposes
an explicit `Stroma` label. It must never be attached to a tumor growth skill,
nor may `Other tissue` be silently interpreted as stroma.

1. Require an explicit Stroma target label and a legal tumor-to-stroma interface.
2. Never use normal epithelium, non-tissue, lumen, or generic background as an implicit source.
3. Regenerate the changed region from the target-stroma population contract.
4. Do not claim desmoplasia from a coarse stroma mask; keep that as a render contract.
5. Read [primitive_contract.json](references/primitive_contract.json) for executable requirements.
