# Joint skill catalog v2

## Primitive scale is explicit

| Primitive | Tissue mask | Nuclei mask | Receiving structure | v1 execution |
|---|---|---|---|---|
| `neoplastic-microinfiltration-increase-v1` | unchanged | sparse mechanism-owned templates added | verified non-tumor host compartment | research executable |
| `invasive-front-expansion-v1` | source compartment becomes Tumor from a selected interface | complete source instances are handled and target population regenerated | true invasive interface | research executable through an audited tissue-tool adapter |
| `structural-void-spread-v1` | unchanged | separated templates inside a native void | airspace/lumen-like auxiliary structure, never implicit stroma | fail-closed pending void and generator validation |
| `architecture-progression-v1` | explicit fine identity transition | target-architecture population regenerated | one native architectural unit | fail-closed pending fine-label compiler and generator validation |

The deprecated `neoplastic-cell-infiltration-increase-v1` is retained only to
reject legacy manifests with an actionable migration reason.

## Stroma transition policy

`stroma-increase-v1` is not a growth mechanism. A `Tumor → Stroma` transition
requires an independently selected mechanism with compatible context, such as
documented post-treatment fibrotic/inflammatory replacement, plus an annotation
profile with explicit stromal authority. The catalog therefore removes the
primitive from lepidic, Gleason-pattern and other growth skills.

## Evidence authority

`evidence-governance-v2.json` classifies contract fields and sources into
exactly four non-interchangeable categories:

1. `pathology_fact` — biological/histologic recognition and contraindication;
2. `dataset_fact` — label protocol, revision, fine IDs and background meaning;
3. `engineering_proxy` — deterministic geometry, capacity and gate behavior;
4. `model_representability` — demonstrated response of the frozen condition and
   generator stack.

Source verification does not equal internal review. Current research skills
remain `draft`; uncalibrated statistics and the pending frozen-generator
evaluation keep production fail-closed.
