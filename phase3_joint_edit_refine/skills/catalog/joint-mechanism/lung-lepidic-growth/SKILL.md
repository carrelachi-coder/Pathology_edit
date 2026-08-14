---
name: lung-lepidic-growth
description: Joint tissue–cell execution contract for lung lepidic growth; use only after the required mask-profile, native-structure, nuclei, and certified-candidate observations are available.
---

# Lung Lepidic Growth

Use this skill to compile one atomic tissue–cell candidate pair. Never use it to infer an annotation profile, invent a missing auxiliary structure, or authorize pixels directly from free text.

## Execution

1. Load `references/joint_contract.json` and verify the recognition and representability contracts.
2. Intersect it with the active annotation, cell-observation and cell-population profiles.
3. Compile the tissue program and cell layout program; preserve complete source nuclei outside the authorized core/halo.
4. Run every required checker. Missing observations, auxiliary structures, checker implementations or source-matched cell shapes require abstention.
5. Treat render-only claims as post-generation critic obligations, never as mask guarantees.

Read `references/counterexamples.json` before candidate selection. `references/statistics.json` is deliberately non-production until cohort calibration and pathology review are complete.
