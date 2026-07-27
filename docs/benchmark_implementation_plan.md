# Pathology Edit Benchmark Implementation Plan

## 0. Current Scope and Frozen Decisions

This document is the active execution plan as of 2026-07-27. Historical
checkpoint-selection logs, superseded ProbNet count/density experiments, and
completed pilot commands are intentionally excluded from the active plan.

Current state:

| Workstream | Status | Remaining formal work |
|---|---|---|
| Mask semantic edit | Frozen and complete | Preserve artifacts and report the final Instruction/GT results; do not reopen model or threshold selection. |
| CellDistNet / nuclei layout | Engineering-accepted and closed | Keep CellDistNet as a tissue-aware placement prior, preserve the Section 5.8 artifacts, and do not reopen checkpoint or sampler selection. |
| Segmentator | Fine V2 accepted; gated ablations active | Finish the pre-registered Boundary and CellViT-teacher comparisons on real human tissue masks, then freeze one evaluator. |
| Generation consistency | Pilot implementation only | Run paired target-condition consistency, edit-local fidelity/preservation, and seed robustness with the final evaluator. |
| Patho-KID | Automatic-condition interim complete | Run the clinician-condition primary analysis and the paired automatic-condition sensitivity analysis on identical accepted sample IDs. |
| Representation insight | Paired U1/U2 analysis complete | Preserve the frozen tumor-burden and stromal-immune cohorts, report measured spatial overlap and cross-fitted axes, and do not expand into a general downstream-utility claim. |

Confirmed scope:

- Manuscript-facing nomenclature is `CellDistNet`. Existing code modules,
  artifact directories, configuration keys and checkpoint paths containing
  `probnet` are retained as legacy internal identifiers for reproducibility;
  they do not denote a different model.

- Organs/profiles are fixed as breast/BCSS, prostate/PANDA,
  colorectal/GlaS, lung/IGNITE, melanoma/PUMA, and oral/ORCA.
- The frozen mask-semantic benchmark covers all implemented generic
  primitives plus the PANDA and GlaS fine-label primitives. Its final result is
  documented in `docs/mask_edit_benchmark_v1_runbook.md`.
- The image-generation benchmark has two distinct scientific endpoints:
  condition consistency and realism. They must not be collapsed into one
  score.
- Local Inpaint and production Cross-v1 are separate generation strata.
  Changed-region fidelity is their common endpoint. Cross-v1 also reports
  unchanged-region drift as a main preservation endpoint; Inpaint reports
  unchanged drift and a narrow boundary ring as secondary safety diagnostics.
- Formal Inpaint outputs are evaluated as generated. Reference RGB is never
  pasted back after synthesis.
- CellDistNet is a supporting component. Counts and exact type quotas come from
  the frozen patch/profile statistical policy. The epoch-29 C3 checkpoint
  contributes only `P(nucleus) = 1 - P(background)` for spatial landing.
  Count-head, density-head, and inference-grid model selection are retired.
  Its required claim is engineering validity, not learned point-process
  superiority: it must realize valid layouts safely and provide a useful
  tissue-aware positional bias.
- The active paired candidate pool is the strict-Oral revision of
  `complex_paired_v3_1500`: 727 pairs / 1,454 directions. Clinician review
  must retain at least 500 pairs / 1,000 directions, and the final
  organ-balanced sensitivity set contains 300 directions.
- The clinician-condition run is the publication-primary generation result.
  The automatic-condition run is a paired sensitivity analysis on the exact
  same accepted sample IDs, checkpoints, prompts, seeds, normalization, and
  bootstrap draws.
- Pooled Patho-KID in UNI-2h and CONCH feature spaces is the primary realism
  metric. Organ composition is reported next to the pooled score; the
  300-direction organ-balanced result is secondary.
- Strict pixel/instance condition-consistency rankings include only methods
  that receive the corresponding target geometry. Text-only baselines remain
  eligible for Patho-KID and blinded realism review, not for strict spatial
  fidelity ranking.
- CellViT outputs are named `CellViT-derived nuclei consistency`, not human
  nuclei ground truth. The checkpoint, preprocessing, magnification, label
  map, and inference implementation are version-locked.
- The full project method is the frozen agentic system with at most two
  attempts. It is not a manually selected Inpaint or Cross result.
- All formal splits, sample IDs, masks, seeds, checkpoints, thresholds, and
  code revisions are frozen with hashes before official generation. Model
  selection and verifier calibration never use the official test outputs.

External generation baselines remain PixCell, MuPaD, PathLDM, UniPath, and
PathDiff, subject to their native input contracts and organ support. All large
remote jobs must be resumable and must write per-sample provenance.

## 1. Benchmark Layers

The benchmark should be separated into four layers. These layers answer different questions and should not be collapsed into one score.

### 1.1 Mask Semantic Edit Benchmark

Question:

- Does the mask edit result match the intended semantic edit?
- Does the system understand prompt-edit and semantic/instruction-edit modes?

Inputs:

- reference image, if available;
- reference mask;
- structured GT intent: `organ`, `profile`, `primitive`, `strength`, `source_labels`, `target_label`, `region_hint`;
- generated prompt pair:
  - old/new descriptive pathology reports for prompt mode;
  - free-text instruction for semantic/instruction mode.

Outputs:

- edited target mask;
- direction hit rate;
- on-target transition rate;
- spatial containment rate;
- ordinal monotonicity;
- magnitude bucket pass rate;
- off-target change rate;
- per-cell report by `organ x primitive x strength x mode`.

This layer already has a base implementation in `phase3_mask_edit.benchmark`:

- intent building: `phase3_mask_edit/benchmark/intents.py`
- prompt generation/checking: `phase3_mask_edit/benchmark/prompts.py`
- runner: `phase3_mask_edit/benchmark/runner.py`
- mask metrics: `phase3_mask_edit/benchmark/metrics.py`

### 1.2 Generation Condition-Consistency Benchmark

Question:

- Does a generated image express the requested tissue and nuclei condition?
- Does a local edit remain confined to the intended region?
- Is condition fidelity stable across generation seeds without forcing
  pixel-identical outputs?

The formal benchmark has three tracks.

#### Track G1: Paired target-condition consistency

Reuse the exact clinician-approved A-to-B/B-to-A directions used by Patho-KID.
This track adds no new generation bank. Re-segment every generated image with
the final frozen tissue evaluator and version-locked CellViT.

Tissue endpoints:

- non-background macro mIoU (primary);
- Boundary-F1@4 (co-primary boundary endpoint);
- macro Dice and per-class IoU;
- foreground area-fraction absolute error;
- class-presence recall;
- tissue area-distribution JSD.

For every doctor-labelled target, also run the evaluator on the withheld real
target RGB. Report `Segmentator(real target) vs doctor mask` as the
real-image evaluator ceiling next to `Segmentator(generated) vs doctor mask`.
The real target RGB remains unavailable to every generator.

Nuclei and tissue-nuclei endpoints are defined in Section 5.9. Strict spatial
rankings are restricted to methods that receive target geometry. PathDiff
native CoNIC/HoVer-Net spatial results are schema-specific and are not ranked
against CellViT-native spatial F1.

#### Track G2: Edit-local fidelity and preservation

Reuse the frozen 231-cell mask-edit target bank; the semantic mask-edit
benchmark itself remains closed. Define:

`changed_region = source_tissue_mask != target_tissue_mask`

`unchanged_region = source_tissue_mask == target_tissue_mask`

Primary endpoints are changed-region macro mIoU, changed-region accuracy,
target-class IoU, and boundary-ring consistency. Route-specific preservation
is reported without constructing a single aggregate score:

- Inpaint: unchanged-region drift and boundary-ring artifacts are secondary
  safety diagnostics.
- Cross-v1: unchanged-region tissue drift is a main preservation endpoint.
- Agentic system: retain and score both routed attempt 1 and selected final so
  that fallback gains and regressions remain visible.

Execution is phased:

- smoke: 3 frozen cases per valid cell (up to 693 conditions);
- focused formal run: 10 per valid cell for the project agentic system;
- fixed Inpaint, fixed Cross, and external geometry-conditioned baselines may
  use a prespecified balanced subset when full duplication is computationally
  prohibitive.

No 30- or 50-per-cell expansion is started until the 3-per-cell smoke passes
coverage, evaluator, provenance, and runtime QA.

#### Track G3: Seed robustness

Freeze 60-120 cases balanced across organ, edit size, route, tissue complexity,
and primitive. Generate 4-5 seeds per condition with all non-seed inputs fixed.

Report:

- mean, standard deviation, and lower-tail condition-consistency metrics;
- fraction of conditions that pass the frozen gate for every seed;
- changed-region LPIPS or pathology-embedding diversity;
- fidelity-diversity plots.

Seed consistency means stable semantic/structural adherence, not identical
pixels. Repeated seeds are nested within the same case and are never treated as
independent bootstrap units.

#### Automatic-condition versus clinician-condition sensitivity

Run both condition sources on identical clinician-accepted sample IDs.

- `DoctorCond` is publication-primary.
- `AutoCond` uses the final frozen Segmentator output and is a sensitivity
  analysis.
- A tissue-only ablation keeps the target nuclei condition bitwise identical
  and changes only the tissue mask.
- A full-pipeline ablation reruns the frozen statistical-count plus ProbNet
  spatial-sampling policy after the tissue condition changes.

If compute permits, run both versions on the complete accepted set. Otherwise
run DoctorCond on the complete set and both versions on the frozen balanced
300-direction subset. Use identical pair/WSI bootstrap draws for every
AutoCond-minus-DoctorCond comparison.

### 1.3 Generative Realism Benchmark

Question:

- Are generated images realistic in pathology feature space and under human review?

Automated metric:

- Patho-KID in UNI-2h feature space.
- Patho-KID in CONCH feature space.

Paired design:

- Each unordered pair contains real patches A and B from the same organ and WSI with similar tissue/cell composition.
- A and B have non-overlapping WSI coordinate boxes.
- Each pair creates two tasks: A as reference to generate B, and B as reference to generate A.
- The withheld real target RGB is the Patho-KID real sample. It is never passed to the generator.
- Target tissue/nuclei masks may be prepared from the real target offline because this is conditional generation, but the exact mask provenance must be logged.

Headline reporting:

- pooled Patho-KID over every direction in the frozen clinician-approved publication set is the primary realism metric, with at least 1,000 directions required;
- organ composition is always printed next to the pooled score;
- an organ-balanced 300-direction subset is a secondary sensitivity check;
- per-organ Patho-KID is descriptive only for the current set and is not the headline claim.

Human metric:

- small balanced blind review subset;
- expert ratings for realism, artifact, and structure/diagnosis plausibility;
- optional pairwise preference against the main model and strongest baseline.

Do not use generic natural-image FID/KID as the main claim. The formal claim should be feature-space Patho-KID.

### 1.4 Representation-Space Utility Insight

This workstream is a compact mechanistic insight, not a general downstream
utility benchmark. It does not claim improved diagnosis, grading, retrieval,
survival modelling, or synthetic-data augmentation.

Embedding models are frozen UNI-2h and CONCH. Raw embeddings form all
differences; only displacement vectors are normalized for cosine calculations.
All inference uses WSI-held-out fitting or case/WSI-cluster resampling.

#### Completed primitive U1: tumor-burden increase

The frozen U1 cohort contains 257 paired Original--Moderate--Significant
trajectories from 99 BCSS WSIs. Both local synthesis and reference-guided
global synthesis use the same tissue and nuclei targets at each strength. The
production nuclei policy and fresh image/embedding caches are frozen under:

`/data1/zhao/wqx/benchmarks/data/embedding_utility_bcss_paired257_frozen_sampling_v3_epoch29`

`/data1/zhao/wqx/benchmarks/runs/embedding_utility_bcss_paired257_frozen_sampling_v3_epoch29`

The supported claim is that prescribed tumor-burden increase produces a
dose-extending representation displacement reproduced across two synthesis
paths and two pathology encoders. WSI-held-out strict
`Original < Moderate < Significant` progression ranges from 72.76% to
88.72% across the four encoder/synthesis combinations. This remains a
BCSS/tumor-increase finding rather than evidence of general clinical utility.

#### Completed paired primitive U2: stromal immune infiltration

Use `stromal_immune_infiltration` as the single additional primitive. It
converts peritumoral stroma to immune tissue while preserving tumor area, so
its intended biological axis is immune-microenvironment activation rather than
tumor-burden expansion.

The completed 257-row U1 cohort is not suitable as the only candidate source:
168/257 original masks contain no immune tissue. Freeze a new paired cohort
using mask information only, before image generation or embedding inspection:

1. Scan BCSS for references with tumor, stroma, and immune tissue already
   present. Require immune >= 10%, tumor >= 10%, stroma >= 30%, and exactly
   zero Background and Other tissue pixels.
2. The strict filter yields 623 patches from 90 WSIs. Freeze 600 candidates by
   deterministic WSI round-robin ordering. Do not use RGB appearance,
   generation quality, nuclei output, or embeddings in this selection.
3. On every candidate, execute both `tumor_burden_increase` (U1) and
   `stromal_immune_infiltration` (U2) at Moderate and Significant strengths.
   Each of the four masks is an independent native multimodal LLM-contour
   proposal using the formal primitive recipe, `organic_v2` projection, and
   the validator-guided agent repair loop. The frozen contour model is
   `gpt-4.1-mini`, with at most two attempts and a 16-pixel tolerance only for
   clamping small out-of-bounds coordinate overshoots.
4. Do not make Significant a deterministic geometric expansion of Moderate.
   Require both strengths to pass their formal validators and require the
   realized Significant dose to exceed Moderate by at least 0.05 of the
   primitive denominator. Report Moderate/Significant IoU and containment as
   observed properties; do not use them as inclusion gates.
5. Do not force U1 and U2 changed masks to be disjoint. Both native mechanisms
   legitimately prioritize peritumoral stroma, and a disjointness or 16-pixel
   gap constraint would replace the formal primitives with an artificial
   geometry. Record per-strength U1/U2 IoU and bidirectional containment as
   descriptive covariates only; they are never selection criteria.
6. Freeze 300 eligible rows by deterministic WSI round-robin ordering after
   mask-side validation. If fewer than 300 survive, report the prespecified
   shortfall; do not inspect images or embeddings to choose replacements.

The frozen static candidate pool is:

`/data1/zhao/wqx/benchmarks/data/representation_two_primitive_bcss_immune_rich_candidates_v2`

The native mask run is:

`/data1/zhao/wqx/benchmarks/data/representation_two_primitive_bcss_immune_rich_llm_native_v1`

A 10-case preflight produced 10/10 eligible rows and 40/40 first-attempt
validated contours. U1 realized doses were 0.190/0.320 for
Moderate/Significant and U2 doses were 0.190/0.295. Natural
Moderate-in-Significant containment had medians 1.000 for U1 and 0.938 for U2;
no contour was geometrically reconciled.

For every retained reference, generate Moderate and Significant U1 and U2
images with the same local and reference-guided global synthesis checkpoints,
seed policy, MPP, frozen nuclei policy, and audit requirements used by the
completed U1 study. The two synthesis paths receive identical tissue and
nuclei targets within each `reference x primitive x strength` row.

The frozen 300-reference production bank is:

`/data1/zhao/wqx/benchmarks/data/representation_two_primitive_bcss_paired300_native_v1_sampling_v3_epoch29`

The completed synthesis run is:

`/data1/zhao/wqx/benchmarks/runs/representation_two_primitive_bcss_paired300_native_v1_sampling_v3_epoch29`

All 1,200 nuclei targets use independent direct-from-original strength
generation, epoch-29 ProbNet spatial ranking, and strict zero nucleus overlap.
The full audit placed 63,015/63,304 requested nuclei (`99.543%`), recorded all
289 unfilled placements, and verified retained-source nuclei bitwise for all
1,200 rows. Local and reference-guided global synthesis each completed
1,200/1,200 images with zero failures. The final 2,400-image audit verified
every `primitive x strength x backend` cell at n=300. The global path used
pix2pix epoch 26 / step 214895 with full-pyramid `local_histogram` reference
direction steering.

Primary U2 endpoints mirror U1:

- WSI-held-out `Original < Moderate < Significant` ordering;
- positive projection of `E(Significant) - E(Moderate)` on a WSI-held-out
  Moderate U2 direction;
- displacement-norm increase;
- within-primitive directional consistency for UNI-2h and CONCH.

Between-primitive endpoints test separability rather than assuming it:

- cosine between WSI-held-out U1 and U2 mean directions;
- matched-reference cosine between U1 and U2 displacements;
- a cross-fitted 2 x 2 direction matrix: each displacement projected on the
  held-out U1 and U2 axes;
- fraction of rows whose own-primitive projection exceeds the other-primitive
  projection;
- WSI-cluster bootstrap intervals and a WSI-level permutation test for
  direction labels.

Mask overlap is empirical, and embedding orthogonality is also an empirical
result. Do not select cases to force low mask overlap or a low embedding cosine,
and do not use the phrase `orthogonal direction` unless the frozen analysis
supports it. Report a high-overlap mask sensitivity analysis to show whether
between-primitive discrimination persists when the edited spatial support is
similar. The intended paper claim is limited to two distinguishable,
dose-extending edit directions when both within-primitive continuation and
between-primitive discrimination pass.

#### Frozen two-primitive representation result

Fresh UNI-2h and CONCH features were extracted from the final audited 2,400
images; no embedding from the earlier 257-case bank was reused. The analysis
contains the same 300 reference patches from 90 WSIs in every
`primitive x strength x synthesis` cell. Raw feature differences are used for
trajectory lengths, displacement vectors alone are normalized for cosine
endpoints, directions are fitted after leaving out the scored WSI, and
uncertainty is resampled by WSI.

The intended result holds in both encoders and both synthesis paths:

- WSI-held-out strict `Original < Moderate < Significant` progression is
  `90.7% / 86.7%` for U1 and `90.3% / 78.7%` for U2 in UNI-2h
  local/global synthesis. The corresponding CONCH values are
  `90.0% / 91.7%` for U1 and `84.0% / 77.7%` for U2.
- The mean Moderate-to-Significant projection is positive in all eight
  `encoder x primitive x synthesis` combinations, with the WSI-level
  one-sided sign-flip test at the Monte Carlo floor
  `P = 4.99975e-05`.
- Significant-to-Moderate displacement-norm ratios are above one in every
  combination, ranging from `1.066` to `1.409`. Matched
  Moderate/Significant direction cosine ranges from `0.621` to `0.896`.
- Local and reference-guided synthesis do not produce identical shifts, but
  reproduce a shared direction. Their global mean-direction cosine is
  `0.351-0.448` at Moderate and `0.478-0.553` at Significant; the paired
  backend-agreement WSI-bootstrap intervals are strictly positive in both
  encoders and both primitives.
- U1 and U2 are related rather than orthogonal: their cross-fitted shared-axis
  cosine is `0.841` in UNI-2h and `0.622` in CONCH. Nevertheless, the paired
  own-axis margin is positive for every encoder, strength, and synthesis path
  (`P = 4.99975e-05`). At Significant strength, both paired edits prefer their
  own held-out primitive axis in `70.7-86.3%` of references.
- This discrimination is not explained only by spatially disjoint masks. In
  the top mask-overlap quartile (75 references; IoU thresholds `0.513` at
  Moderate and `0.541` at Significant), every paired own-axis margin remains
  positive with a WSI-bootstrap interval excluding zero and
  `P = 4.99975e-05`.

The supported interpretation is therefore a shared tissue-remodelling
component plus primitive-specific residual directions: controlled tumor
expansion and stromal immune infiltration each form a dose-extending
representation trajectory that replicates across synthesis implementations
and pathology encoders. This is evidence for a reproducible representation
response to the prescribed edits, not proof of a universal biological
mechanism or downstream clinical utility.

Frozen artifacts:

`/data1/zhao/wqx/benchmarks/runs/representation_two_primitive_bcss_paired300_native_v1_sampling_v3_epoch29/two_primitive_trajectory/two_primitive_trajectory_report.json`

`/data1/zhao/wqx/benchmarks/runs/representation_two_primitive_bcss_paired300_native_v1_sampling_v3_epoch29/two_primitive_trajectory/figure/uni2h_conch_two_primitive_trajectory.pdf`

Trajectory figure caption:

> Representation-space response to controlled pathology edits. The same 300
> BCSS reference patches from 90 WSIs were edited for tumor-burden increase
> (U1) and stromal immune infiltration (U2) at Moderate and Significant
> strengths. Rows show frozen UNI-2h and CONCH feature spaces; columns show
> local synthesis and reference-guided global synthesis. Every original and
> generated patch retains its absolute position in a common three-dimensional
> projection: the first axis is the unit sum of the two synthesis-shared
> Moderate edit directions, the second is their unit difference, and the third
> is the leading reference variation orthogonal to both edit axes. No
> per-case centering is used and no edit displacement is magnified. All 300
> original points and all 1,200 generated points per panel are supplied to the
> plot; each display axis uses the panel-specific 1st--99th percentile window,
> so a small number of extreme points can fall outside the frame.
> The cubic axis box compresses only the display aspect and leaves every
> numeric coordinate unchanged. Axis limits are fitted separately within each
> synthesis panel, so cross-panel displacement length must be compared using
> numeric ticks rather than apparent screen length. Within each group, dense
> regions receive less per-point opacity to suppress overplotting.
> Foregrounded colored cohort centroids and thick mean paths trace
> Original--Moderate--Significant progression; reference-neighborhood bundles
> are omitted to prevent overplotting. Annotated estimates use raw
> features, directions fitted after leaving out the scored WSI, and
> WSI-cluster resampling. Both edits extend with prescribed strength across
> both synthesis paths and encoders, while positive paired own-axis margins
> identify primitive-specific components despite substantial mask and
> feature-direction overlap.

#### Reporting contract

Report one shared figure per encoder with U1 and U2 in a common cross-fitted
basis and identical axis limits across synthesis paths. Keep U1 and U2 cohort
counts, WSI counts, realized mask doses, generation failures, and bootstrap
units visible. The section title in the paper should be
`Representation-space response to controlled pathology edits`, not
`downstream utility benchmark`.

## 2. Primitive and Strength Matrix

### 2.1 Generic Primitives

Generic primitives are evaluated on all six organs/profiles when the recipe and candidate masks support them.

Primitives with 4 strengths:

- `tumor_burden_increase`: `mild`, `moderate`, `significant`, `xlarge_deid`
- `tumor_burden_decrease`: `mild`, `moderate`, `significant`, `xlarge_deid`
- `necrosis_appearance`: `mild`, `moderate`, `significant`, `xlarge_deid`
- `necrosis_resolution`: `mild`, `moderate`, `significant`, `xlarge_deid`
- `immune_infiltration_decrease`: `mild`, `moderate`, `significant`, `xlarge_deid`
- `stromal_desmoplasia`: `mild`, `moderate`, `significant`, `xlarge_deid`
- `stroma_decrease`: `mild`, `moderate`, `significant`, `xlarge_deid`

Primitives with 3 strengths:

- `stromal_immune_infiltration`: `mild`, `moderate`, `significant`
- `intratumoral_immune_infiltration`: `mild`, `moderate`, `significant`

Generic cells:

- 34 valid `primitive x strength` cells per organ.
- 6 organs -> 204 generic cells.

### 2.2 PANDA Specialized Primitives

PANDA primitives:

- `gleason_upgrade_3to4`
- `gleason_upgrade_4to5`
- `gleason_downgrade_4to3`
- `benign_to_gleason3`
- `benign_atrophy`

Strengths:

- `mild`, `moderate`, `significant`

PANDA specialized cells:

- 5 primitives x 3 strengths = 15 cells.

### 2.3 GlaS Specialized Primitives

GlaS primitives:

- `normal_to_adenomatous`
- `adenoma_to_carcinoma`
- `grade_upgrade`
- `treatment_dedifferentiation`

Strengths:

- `mild`, `moderate`, `significant`

GlaS specialized cells:

- 4 primitives x 3 strengths = 12 cells.

### 2.4 Total Formal Cells

Total benchmark cells:

- generic: 204
- PANDA specialized: 15
- GlaS specialized: 12
- total: 231 cells

Every automated generation benchmark report must be able to aggregate by these 231 cells, even if some cells are marked as shortfall because insufficient valid samples exist.

## 3. Dataset Sizes

### 3.1 Mask Semantic Benchmark

Formal target:

- 100 patches per valid `organ x primitive x strength` cell.
- 231 cells x 100 = 23,100 structured GT intents.

Modes:

- `gt`: structured GT execution sanity and target-mask creation.
- `prompt`: old/new pathology report comparison mode.
- `instruction`: free-text semantic instruction mode.

Formal rows:

- `gt`: 23,100 rows, used for sanity and target-mask bank.
- `prompt`: up to 23,100 rows.
- `instruction`: up to 23,100 rows.

The headline semantic benchmark should report `prompt` and `instruction` separately. `gt` should be reported as an execution-control ceiling, not as the main natural-language result.

### 3.2 Generation Consistency Sample Sizes

Paired target-condition consistency uses the same frozen clinician-approved
publication manifest as Patho-KID:

- minimum 500 accepted pairs / 1,000 directions;
- retain all accepted directions when more than 1,000 survive;
- DoctorCond is complete and publication-primary;
- AutoCond is complete when compute permits, otherwise paired on the frozen
  300-direction organ-balanced sensitivity set.

Edit-local fidelity uses the 231 valid
`organ x primitive x strength` cells from the mask-semantic bank:

- smoke: 3 cases per valid cell, up to 693 conditions;
- focused formal: 10 cases per valid cell for the project agentic system, up
  to 2,310 conditions;
- fixed-route and external-baseline duplication may use a prespecified
  organ/primitive/strength-balanced subset;
- every report exposes cell counts and shortfalls.

Seed robustness uses 60-120 frozen conditions with 4-5 seeds each. Balance
organ, primitive, strength, route, edit area, and tissue complexity. No seed is
selected after viewing a generated image or metric.

The previous 30/50-per-cell, six-model expansion is removed from the active
plan. It is not justified before the focused formal run establishes variance,
coverage, and compute requirements.

### 3.3 Segmentator Accuracy Benchmark

Segmentator is part of the generation condition-consistency evaluator. Unlike
CellViT, it should have local human validation because it is project-specific.

Recommended data:

- reuse clinician-corrected, individually usable tissue masks from the active
  1,454-patch annotation pool;
- target at least 1,200 usable patches across the six organs;
- do not equate a rejected A/B pair with two unusable segmentator-validation patches;
- prioritize individually usable patches from pair-level rejections for model
  selection so accepted official generation pairs remain untouched;
- use the 150 blinded redraws for independent agreement/anchoring analysis;
- reserve a WSI-disjoint human-mask subset that is not used for checkpoint,
  teacher/input-route, threshold, or early-stopping selection.

Sampling:

- independent WSI/patient-level split;
- balance tissue complexity and class presence;
- include hard cases with tumor/stroma/immune/necrosis/blood-vessel mixtures when available.

Outputs:

- per-class Dice;
- per-class IoU;
- macro Dice;
- mIoU;
- Boundary-F1@2/4/8 and HD95;
- confusion matrix;
- failure categories.

### 3.4 CellViT Nuclei Consistency

No new local nuclei human GT is required for the formal benchmark if CellViT literature/model evidence is cited.

Required controls:

- lock CellViT checkpoint;
- lock inference path;
- log CellViT version, model path, preprocessing, magnification assumption, and output label map;
- use the project convention:
  - `0`: background
  - `101`: neoplastic
  - `102`: inflammatory
  - `103`: connective
  - `104`: dead
  - `105`: epithelial

Recommended benchmark phrasing:

- "CellViT-derived nuclei consistency"
- not "human-GT nuclei accuracy"

### 3.5 Patho-KID Real Reference Sets

Automatic pilot source:

```text
/data1/zhao/wqx/benchmarks/data/complex_paired_v2_1018/
  pairs.csv
  directions.csv
  manual_review.csv
  summary.json
```

Validated pilot size:

- 509 unordered pairs;
- 1,018 A-to-B/B-to-A directions;
- 1,018 unique patches, each used exactly once as reference and exactly once as the real target across the two directions;
- 421 unique WSIs;
- zero selected-coordinate overlap within every WSI;
- zero patch reuse;
- all RGB, text, tissue-mask, and CellViT-mask assets present.

Organ mixture:

| Organ | Pairs | Directions / unique KID targets |
|---|---:|---:|
| breast | 122 | 244 |
| colorectal | 81 | 162 |
| head_neck | 25 | 50 |
| lung | 65 | 130 |
| prostate | 170 | 340 |
| skin | 46 | 92 |
| **Total** | **509** | **1,018** |

Publication annotation pool:

```text
/data1/zhao/wqx/benchmarks/data/complex_paired_v3_1500/
  pairs.csv
  directions.csv
  summary.json
  validation.json
  manifest_hashes.json
  annotation_package/
    images/
    tissue_masks_auto/
    labels_primary/
    labels_secondary/
    cellvit_masks_auto/
    captions_en/
    captions_zh/
    pair_previews/
    patch_annotation_manifest.csv
    double_annotation_manifest.csv
    pair_review.csv
    caption_manifest.csv
    caption_summary.json
```

Original validated annotation-pool size and composition before clinical-site filtering:

- 750 unordered pairs;
- 1,500 unique patches and A-to-B/B-to-A directions;
- 656 unique WSIs;
- no patch reuse and no selected-coordinate overlap within a WSI;
- organ water-filling is used rather than weakening pair thresholds to force equal quotas;
- `labels_primary` starts from segmentator pre-annotations and is editable;
- 150 score- and organ-stratified `labels_secondary` masks are blank independent redraws;
- CellViT masks remain automatic and are not clinician annotation targets.
- each annotation ID has a portable English source caption and a Simplified Chinese pathology translation;
- 12 Chinese captions reuse exact stem-matched prior translations and 1,488 were translated with `qwen-turbo` through the configured OpenAI-compatible endpoint;
- LLM translations were checked for exact stem/source alignment, CJK coverage, ordinary-English leakage, key pathology terminology, negation, and uncertainty; 39 entries were retried in smaller batches and 13 received deterministic terminology QA corrections;
- translation provenance is stored per annotation in `caption_manifest.csv`; temporary Google translations are not included in the formal package.

| Organ | Candidate pairs | Candidate patches / directions |
|---|---:|---:|
| breast | 151 | 302 |
| colorectal | 151 | 302 |
| head_neck | 57 | 114 |
| lung | 151 | 302 |
| prostate | 151 | 302 |
| skin | 89 | 178 |
| **Total** | **750** | **1,500** |

Active strict-Oral candidate set (2026-07-22):

- GDC `cases.primary_site` is frozen for every TCGA-HNSC case in `cohort_filters/strict_oral_20260722/case_primary_sites.csv`;
- retain only `Floor of mouth`, `Gum`, `Lip`, `Other and unspecified parts of mouth`, and `Other and unspecified parts of tongue`;
- exclude all `Larynx` cases and the ambiguous `Other and ill-defined sites in lip, oral cavity and pharynx` category;
- this removes 15 cases / 23 pairs / 46 directions and leaves 727 pairs / 1,454 directions;
- the internal `head_neck` value is retained only for path/config compatibility. Every active row under that label is strict Oral: 34 pairs / 68 directions;
- the active compatibility manifest is `paired_directions_auto_1500.json`; an explicitly named copy is `paired_directions_strict_oral_1454.json`;
- source metadata and removed annotation-package assets are auditable under `cohort_filters/strict_oral_20260722/`. Future UniPath and Cross generation must read the filtered active manifest and must not reintroduce rows by enumerating an old image directory.

| Active organ label | Pairs | Patches / directions |
|---|---:|---:|
| breast | 151 | 302 |
| colorectal | 151 | 302 |
| head_neck (strict Oral) | 34 | 68 |
| lung | 151 | 302 |
| prostate | 151 | 302 |
| skin | 89 | 178 |
| **Total** | **727** | **1,454** |

Primary real set:

- after clinician QA and pair revalidation, freeze `publication/directions.csv` from accepted pairs;
- require at least 500 accepted pairs / 1,000 directions;
- retain every accepted pair when more than 500 survive rather than discarding good samples to reach an exact round number;
- `target_image_path` from every row of the frozen publication manifest;
- every target path is unique;
- every model must generate the same frozen `sample_id` rows;
- generation failures must be rerun, not silently dropped from KID.

Primary report:

- pooled UNI-2h/CONCH Patho-KID over the complete clinician-approved publication set;
- this is a micro/distribution-mixture score weighted by the frozen post-QA organ counts;
- do not describe it as organ-balanced or as a macro average.

Organ-balanced sensitivity set:

- when each organ retains at least 25 pairs, freeze 25 pairs per organ, producing 50 directions per organ and 300 directions total;
- select by pair/WSI so both directions remain together;
- use the same sensitivity sample IDs for every model;
- report this result as secondary because its smaller sample size has wider uncertainty;
- if any organ falls below 25 accepted pairs, replenish and annotate replacements before freezing this sensitivity set.

Per-organ KID:

- optional and descriptive only;
- v2 pilot head/neck (`n=50`) and skin (`n=92`), and v3 pre-QA head/neck (`n=114`) and skin (`n=178`), are too small for strong standalone claims;
- never average unstable per-organ KID estimates into the primary result.

### 3.6 Doctor Review Set

Goal:

- keep doctor workload low while still producing useful evidence.

Recommended minimal design:

- 36 benchmark cases:
  - 6 organs x 6 semantic families;
  - strengths balanced across `mild`, `moderate`, `significant`, `xlarge_deid`;
  - include specialized PANDA/GlaS cases in the prostate/colorectal slots.
- 6 models per case:
  - project model;
  - PixCell;
  - MuPaD;
  - PathLDM;
  - UniPath;
  - PathDiff.
- 36 x 6 = 216 generated images.
- Add 10% duplicate/rater-reliability images:
  - about 238 images per doctor.

Recommended rater count:

- 2 doctors minimum if possible.
- 1 doctor acceptable only as pilot/qualitative evidence.

Doctor questions per image:

- realism: 1-5
- artifact severity: 1-5, lower is better
- pathology/structure plausibility for the stated edit: 1-5

Optional pairwise task:

- For the top 2-3 models after automated metrics, run 30-60 additional pairwise comparisons.
- Use this only if the Likert scores are too close.

Reports:

- mean and 95% CI by model;
- Wilcoxon signed-rank or paired bootstrap for model comparisons;
- inter-rater agreement:
  - weighted kappa for ordinal categories;
  - ICC for continuous/averaged ratings.

Doctor review is not expected to provide per-cell estimates for all 231 cells. Per-cell reporting is handled by automated metrics.

### 3.7 Clinician Tissue-Mask QA for the Paired Patho-KID Set

Why this is separate from generated-image review:

- Patho-KID itself uses RGB embeddings and does not directly consume a human tissue mask.
- Tissue masks still define generator conditioning and were used to screen A/B similarity.
- Segmentator errors can therefore make the conditional task wrong even when the KID implementation is correct.

Active annotation task:

- present all active 727 strict-Oral candidate pairs / 1,454 unique patches for review before official generation;
- record pair eligibility separately from A/B patch and tissue-mask usability;
- start from the frozen segmentator mask as a pre-annotation rather than drawing every mask from scratch;
- assign `pass`, `minor_correction`, `major_correction`, or `ungradable`;
- correct tissue labels/boundaries for all non-pass usable patches;
- reject an A/B pair from Patho-KID when either image is ungradable or the pair is clinically too dissimilar;
- retain individually usable, corrected patches from pair-level similarity rejections for segmentator validation;
- use the existing 8-class tissue schema (`0-7`, with `255` for ignore/ambiguous regions).

Reliability subset:

- independently redraw 150 patches (10% of the annotation pool), stratified by organ and pair score and blinded to the segmentator pre-annotation and primary correction;
- report per-class Dice/IoU and disagreement categories between annotators;
- adjudicate major disagreements before freezing the official mask.

After correction:

1. Recompute tissue proportions and pair distances from clinician-corrected masks.
2. Reapply the same pair thresholds and WSI-coordinate constraints.
3. Rerun matching if a corrected pair fails and draw replacements from the machine-screened reserve when the accepted set would fall below 500 pairs or an organ would fall below the balanced-sensitivity quota.
4. Freeze both auto-mask and clinician-mask paths, but use the clinician-corrected tissue mask as the official target condition.
5. Keep CellViT masks automatic and version-locked unless a separate nuclei-GT study is planned.
6. Freeze all accepted pairs, with a minimum of 500 pairs / 1,000 directions.

Optional operational ablation:

- run the project model once with auto masks and once with clinician-corrected masks on identical accepted IDs;
- use the complete accepted set when compute permits, otherwise use the frozen 300-row balanced subset;
- separate tissue-only sensitivity from the full-pipeline condition sensitivity;
- do not mix auto-mask and clinician-mask outputs inside one primary KID set.

Status rule:

- v2 automatic set: `technical_ready_pilot`;
- active 1,454-patch annotation pool before full clinician QA: `publication_mask_pending`;
- official set after correction/revalidation: `publication_ready`.

## 4. Manual Screening and Semantic Annotation Requirements

### 4.1 Requires Strict Manual Review

Prompt/report benchmark texts:

- old_prompt and new_prompt must be independent pathology-style descriptions;
- they must not contain edit commands or comparative language;
- organ, location, sentence order, and unrelated findings should reuse the same wording where possible; only the intended tissue-state observation should differ;
- identical shared wording is a benchmark control, but neither report may say that a finding remained stable, was preserved across reports, or changed relative to the other report;
- their semantic difference must match `organ`, `primitive`, direction, and location; intended strength is calibrated and reported at the same-reference ordinal-group level rather than treated as a unique per-row language label.

Doctor review subset:

- all images in the doctor deck must be manually screened for file integrity and gross display issues;
- image/model identity must be blinded and randomized.

Segmentator human GT:

- tissue labels must be clinician-corrected or independently drawn;
- reuse individually usable labels from the 1,500-patch paired annotation pool and keep the evaluation split WSI-disjoint from any segmentator retraining;
- ambiguous regions should be marked with a QA flag.

Paired Patho-KID tissue conditions:

- all 1,500 candidate patches should receive a pair decision and patch/mask QA status;
- all patches entering the frozen official generation set must have clinician-reviewed tissue masks;
- pre-annotation correction is preferred over independent full redraw for the complete set;
- the 150-patch double-annotation subset must be independent and blind to the pre-annotation.

A-B generation pairs for final paper figures:

- A and B must be manually checked for tissue quality, stain quality, and semantic fit.

### 4.2 Requires Strict Automated QC Plus Spot Manual Review

Large-scale mask intent set:

- capacity check;
- editable pixel threshold;
- legal label check;
- WSI/patient metadata;
- no empty/mostly-background patches;
- no severe blur or stain artifact.

Generated images:

- readable file;
- expected resolution;
- tissue content threshold;
- no blank/solid-color output;
- no extreme stain distribution outlier;
- no duplicate output for different seeds.

Patho-KID real reference set:

- metadata-controlled split;
- artifact screening;
- magnification/resolution check.
- automated masks are sufficient for pilot generation only; the formal clinician-QA status must be explicit;
- maintain a machine-screened reserve so rejected pairs can be replaced without weakening thresholds.

### 4.3 Semantic Labels Required in Manifests

Every benchmark sample should include:

- `sample_id`
- `organ`
- `profile`
- `wsi_id`
- `patient_id`, if available
- `image_path`
- `source_tissue_mask_path`
- `target_tissue_mask_path`
- `source_nuclei_mask_path`, if available
- `target_nuclei_mask_path`, if available
- `primitive`
- `strength`
- `edit_family`
- `region_hint`
- `source_labels`
- `target_label`
- `expected_direction`
- `expected_area_bucket`
- `is_specialized`
- `source_dataset`
- `magnification`
- `um_per_px`, if available
- `qc_status`
- `qc_notes`

Additional fields for paired Patho-KID rows:

- `pair_id`
- `direction`
- `reference_stem`
- `target_stem`
- `reference_x`, `reference_y`
- `target_x`, `target_y`
- `coordinate_span`
- `reference_image_path`
- `target_image_path`
- `auto_target_tissue_mask_path`
- `clinician_target_tissue_mask_path`, when available
- `official_target_tissue_mask_path`
- `target_cellvit_mask_path`
- `tissue_mask_provenance`
- `clinician_review_status`
- `clinician_id_hash`
- `adjudication_status`
- `pair_score`
- `tissue_jsd`
- `cell_jsd`
- `cell_density_diff`

## 5. Metrics

### 5.1 Direction Hit Rate

Purpose:

- Does the target class move in the expected increase/decrease direction?

For class `c_tgt`:

```text
delta_area(c_tgt) = area_after(c_tgt) - area_before(c_tgt)
direction_hit = 1[sign(delta_area(c_tgt)) == GT_direction]
DHR = mean(direction_hit)
```

For decrease operations where source class should shrink, compute direction over the source class.

Report:

- mean;
- 95% CI by WSI-level bootstrap;
- per `organ x primitive x strength x mode`.

Target:

- closer to 1 is better;
- below 0.9 suggests semantic direction misses.

### 5.2 On-Target Transition Rate

Purpose:

- Are changed pixels moving between the intended source/target categories rather than random labels?

```text
OTTR = count(changed pixels consistent with c_src -> c_tgt) / count(all changed pixels)
```

For multi-source primitives, source labels are the legal source set in the recipe.

Report:

- per sample;
- mean + 95% CI;
- per `organ x primitive x strength`.

Target:

- higher is better.

### 5.3 Off-Target Change Rate

Purpose:

- Penalize accidental changes in unrelated classes.

```text
OTCR = count(changed pixels outside legal source/target/constraint region) / count(all changed pixels)
```

Target:

- lower is better;
- use alongside OTTR because a model can get direction right while still damaging unrelated tissue.

### 5.4 Spatial Containment Rate

Purpose:

- Evaluate whether spatially constrained prompts are followed.

Only apply to prompts/intents with explicit spatial constraints:

- intratumoral;
- stromal;
- peritumoral;
- region hints such as upper-left/lower-right/center;
- explicit ROI constraints.

```text
SCR = count(changed pixels inside R_constraint) / count(all changed pixels)
```

Report:

- mean + CI;
- mark `NA` when there is no spatial constraint.

Target:

- higher is better;
- this is the key metric separating true spatial understanding from class-only editing.

### 5.5 Magnitude Bucket Pass Rate

Purpose:

- Does edit size match strength?

Each primitive defines strength buckets in the recipe.

```text
area_fraction = changed_pixels / primitive_denominator_pixels
bucket_pass = 1[lower - tolerance <= area_fraction <= upper + tolerance]
```

Mode-specific interpretation:

- `gt`: the structured strength is explicit, so bucket pass is a strict primary metric.
- `instruction`: the edit instruction names the requested magnitude, so bucket pass remains a strict primary metric.
- `prompt`: the two reports are independent, non-comparative descriptions. Their intended strength is a prompt-generation condition rather than a uniquely recoverable per-row language label. Report bucket agreement as a diagnostic only; do not use it to fail the primary semantic score.
- The `prompt` primary per-row score is `class_ok AND direction_ok AND location_ok`.
- Preserve the hidden intended bucket and its agreement result for calibration analyses; do not call this field strength accuracy.

Report:

- intended bucket agreement rate by strength, labeled by mode;
- measured area distribution by strength.

### 5.6 Ordinal Monotonicity

Purpose:

- Does `mild < moderate < significant < xlarge_deid` produce monotonically increasing edit size?

Important design rule:

- Use the same reference patch, same primitive, same source/target labels, and same region hint across strengths.
- Do not compute monotonicity across different patches.

For each same-ref group:

```text
rho = Spearman(strength_rank, measured_area_delta)
```

Report:

- Spearman rho;
- strict and nondecreasing monotonicity rates;
- pairwise concordance, tie, and reversal rates;
- number of valid same-ref groups;
- per primitive and organ.

For `prompt`, ordinal same-reference results are the primary strength evidence. Exact recovery of `mild`, `moderate`, `significant`, or `xlarge_deid` by a parser is not an accuracy metric unless the report pairs have first been independently calibrated by blinded human raters. If `significant` and `xlarge_deid` are not separable under that calibration, merge them for prompt-mode language claims while retaining `xlarge_deid` as a structured-GT stress condition.

Target:

- closer to 1 is better.

### 5.7 Tissue Generation-Consistency Metrics

#### Paired target-condition track

Given doctor-corrected target tissue mask `M_doctor`, automatic target mask
`M_auto`, withheld real target RGB `I_real`, and generated RGB `I_gen`,
run the final frozen Segmentator:

`S_real = Segmentator(I_real)`

`S_gen = Segmentator(I_gen)`

The publication-primary tissue endpoint is non-background macro mIoU between
`S_gen` and `M_doctor`, computed over classes present in target or
prediction. Boundary-F1@4 is the co-primary boundary endpoint. Also report
macro Dice, per-class IoU, class-presence recall, signed and absolute
area-fraction error, and tissue area JSD.

Report `S_real vs M_doctor` by organ and class as the real-image evaluator
ceiling. Do not divide generated scores by the ceiling or hide raw values; the
ceiling is interpretive context for Segmentator error.

Automatic-condition and clinician-condition runs use the same accepted sample
IDs. Report paired deltas for every tissue endpoint. The tissue-only sensitivity
holds the nuclei target fixed. The full-pipeline sensitivity allows the
production nuclei layout to change and is labelled as a joint tissue+nuclei
condition intervention.

Do not use full-image pixel accuracy as a headline metric. PathLDM, UniPath,
MuPaD, and other text-only methods do not enter strict pixel-level condition
ranking unless they receive a genuinely equivalent spatial condition.

#### Edit-local track

For the frozen mask-edit bank:

`changed_region = M_source != M_target`

`unchanged_region = M_source == M_target`

Common endpoints are changed-region accuracy, changed-region macro mIoU, and
target-class IoU. A frozen narrow ring around the edit measures boundary
artifacts.

Route-specific reporting is mandatory:

- Inpaint: evaluate the generated RGB directly; unchanged drift and the
  boundary ring are secondary safety diagnostics.
- Cross-v1: unchanged-region tissue drift is a main preservation endpoint.
- Agentic: score attempt 1 and selected final separately, retain
  `needs_review` rows in all denominators, and report recovery deltas.
- Fixed Inpaint, fixed Cross, routed first pass, and full agentic policies are
  compared only on identical sample IDs.

Do not average changed fidelity, preservation, boundary quality, and nuclei
consistency into one scalar score.

#### Seed-robustness track

For each frozen case with repeated seeds, report the case-level mean, standard
deviation, minimum or lower-tail tissue/nuclei consistency, and whether every
seed passes the frozen verifier gate. Report changed-region LPIPS or
pathology-embedding diversity as a separate diversity endpoint. Bootstrap by
case/WSI; seeds are repeated observations within a case.

All generated and real RGB inputs are normalized to
`512 x 512 @ 0.25 MPP` before evaluation.

### 5.8 Frozen Statistical-Count + Tissue-Aware CellDistNet Placement

#### 5.8.1 Scope

Production separates nuclei-layout responsibilities:

- the frozen patch/profile statistical policy sets total count and exact
  nucleus-type quotas;
- the immutable epoch-29 C3 checkpoint supplies only
  `P(nucleus) = 1 - P(background)` to rank spatial landing candidates;
- deterministic sampling policy allocates disconnected-component quotas,
  chooses same-class reference/library shapes, and retries failed placements.

The frozen checkpoint is:

`/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/best_epoch29_c29607f1b609accb.pt`

SHA256:

`c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211`

The benchmark does not evaluate or select a learned count head, density head,
type head, gamma grid, or inference-grid variant. The statistical count policy
is treated as a frozen production input. The active ProbNet evidence consists
only of Benchmark P1 and Benchmark P2 below.

#### 5.8.2 Benchmark P1: supporting spatial-prior ablation

Question:

- With count, type quotas, candidate support, and nucleus shapes held fixed,
  does the learned ProbNet field provide a useful tissue-aware positional bias
  beyond simple non-learned sampling?

P1 is supporting evidence about what the checkpoint contributes. It is not the
production engineering acceptance endpoint; that role belongs to P2.

Use leakage-safe grouped-test patches from the six native dataset schemas. For
each case, hide complete nuclei components inside one deterministic tissue-aware
reconstruction region. Supply every sampler with the same:

- reconstruction region and tissue mask;
- oracle hidden total count and exact hidden type quotas;
- disconnected-component quotas;
- Poisson candidate pool and retry pool;
- same-class reference/library shape sequence;
- overlap, boundary, and retry rules.

Only candidate ordering/selection changes. Compare:

1. `probnet`: rank candidates with the frozen scalar spatial field;
2. `poisson_only`: select from the common Poisson pool without learned
   weighting;
3. `uniform`: uniform candidate ordering inside eligible tissue;
4. `boundary_distance`: deterministic distance-to-boundary ranking.

Use identical case seeds and at least five repeated sampling seeds per case.
Seeds are repeated observations, not independent cases.

Primary spatial endpoints:

- nearest-neighbour-distance Wasserstein-1 error;
- pair-correlation or Ripley-K curve error over frozen physical radii;
- tissue-boundary-distance distribution error;
- disconnected-component occupancy error.

Safety and implementation endpoints:

- placement completion;
- exact tissue/type/component quota agreement;
- overlap and full-shape outside-valid-tissue rejection;
- complete preservation of retained source nuclei;
- same-class reference/library fallback rate.

Class-agnostic and class-aware point-matching F1 at a frozen micron tolerance
may be reported descriptively, but exact point recovery is not the primary
goal for a stochastic spatial layout.

Aggregate within case and dataset, then report an equal-dataset macro with
case/WSI-cluster bootstrap intervals. `uniform` is the publication-primary
simple-sampling comparator; `poisson_only` and `boundary_distance` are
secondary diagnostics. A claim that ProbNet learned clearly better cell
spatial structure requires all of the following, frozen before the strict
rerun:

- at least two of the four primary error endpoints improve over `uniform`
  with paired 95% confidence intervals excluding zero;
- each counted improvement reduces the corresponding equal-dataset macro
  error by at least 5%;
- at least one material improvement is NND Wasserstein error or Ripley-K
  error, rather than boundary location alone;
- no primary endpoint has a paired confidence interval showing regression;
- placement completion is at least 98% and no more than 0.5 percentage points
  below `uniform`;
- overlap pixels, outside-valid-tissue pixels, and retained-nucleus changes
  are exactly zero.

If this gate fails, the checkpoint may remain a production spatial prior, but
the report must state that clear learned spatial-structure superiority was not
demonstrated.

#### 5.8.3 Benchmark P2: geometry-only production engineering endpoint

Question:

- Can the frozen production sampler realize its requested nuclei layout as a
  valid nuclei mask under strict overlap and tissue-containment rules?

Use a fixed 120-case endpoint:

- six datasets;
- four dataset-relative planned-count load strata (`low`, `mid-low`,
  `mid-high`, and `high`);
- five cases per dataset-by-stratum cell;
- deterministic case/WSI ordering;
- no sample selected using ProbNet, generator, Segmentator, CellViT, or
  Patho-KID outputs.

Within each dataset, all eligible rows are ranked by frozen-policy planned
count with a stable hash tie-breaker and split into four disjoint, approximately
equal-size rank strata. Five rows are selected from each stratum by a separate
stable hash. Each row also retains its absolute planned-count bin
(`0`, `1-5`, `6-20`, or `>20`) for secondary reporting. Dataset-relative
strata are necessary because the common absolute cells are not all populated:
PANDA has only one eligible `>20` row and PUMA has only one eligible `0` row.

Run exactly:

`statistical count/type quotas -> ProbNet spatial ranking -> component-aware shape placement -> saved nuclei mask`

The P2 target is the saved planned layout produced by the frozen policy. P2
does not reopen the statistical count rule or compare it with a learned count
prediction. It does not synthesize H&E and does not run CellViT.

Report:

- requested, placed, and unfilled nuclei;
- placement completion and exact quota agreement;
- retained-source-nucleus preservation;
- outside-valid-tissue and overlap failures;
- reference versus library shape provenance.

Report dataset and planned-count-load-stratum breakdowns. Placement completion
must be at least 98%; overlap pixels and outside-valid-tissue pixels must both
be zero; retained source nuclei must be bitwise preserved; and every unfilled
placement must remain visible in the denominator. H&E synthesis and CellViT
outputs from the superseded 2026-07-24 run are invalid for ProbNet performance
claims and must not appear in formal ProbNet tables.

#### 5.8.4 Frozen runtime policy

The benchmark and production endpoint use:

`prob_count_weight = 0.0`

`density_scale = 1.0`

`density_scale_json = none`

`gamma = 1.5`

`local_density_direct_min_area = 20,000`

`local_density_direct_min_count = 10`

`minimum_mask_width = 33`

`component_quota_policy = area_largest_remainder`

`backfill_failed_placements = true`

`max_nucleus_overlap_fraction = 0.0`

`require_full_tissue_containment = true`

`retry_candidate_multiplier = 12`

`retry_candidate_floor = 64`

`dense_retry_quota_threshold = 20`

`dense_retry_occupancy_threshold = 0.12`

`dense_retry_candidate_multiplier = 24`

`dense_retry_candidate_floor = 128`

`placement_shape_trials = 4`

`placement_transform_trials = 12`

`dense_placement_shape_trials = 6`

`dense_placement_transform_trials = 24`

Count is independent of the ProbNet output. For tissue class `t`, define:

```text
rho_local,t = 10,000 * n_obs,t / A_obs,t

alpha_t = 1
          if A_obs,t >= 20,000 and n_obs,t >= 10
          else A_obs,t / A_patch

rho_target,t = alpha_t * rho_local,t
               + (1 - alpha_t) * rho_lib,t

N_raw,t = A_edit,t * rho_target,t / 10,000

N_target,t = round(clip(N_raw,t, 0, N_max,t))

N_target = sum_t N_target,t
```

Here `A_obs,t` and `n_obs,t` are the unedited reference area and complete
nucleus count for tissue `t`; `A_edit,t` is its target generation area;
`rho_lib,t` is the matching dataset-by-tissue profile density. The frozen
safety cap is
`N_max,t = min(900 * A_edit,t / 10,000,
2.5 * rho_lib,t * A_edit,t / 10,000)`.
The scalar ProbNet field, its integral, and any learned count/density head are
absent from this definition. A missing dataset-by-tissue density is a profile
coverage failure and must not silently reactivate a probability-derived count.

Position is sampled only after total, tissue, type, and disconnected-component
quotas are fixed. For each tissue-component pair, let
`C = C_poisson union C_retry` be its frozen candidate pool and define:

```text
P_nuc(x) = 1 - P_theta(background | x)

w(x) = P_nuc(x)^gamma + epsilon,  x in C

Pr(x_k = x | C_k) = w(x) / sum_{u in C_k} w(u)

C_{k+1} = C_k \ {x_k}
```

This is probability-weighted sampling without replacement with `gamma = 1.5`.
The benchmark implements the equivalent stochastic ordering
`rank_score(x) = log(w(x)) + g_x`, where
`g_x ~ Gumbel(0, 1)`, and tries candidates in decreasing score order. Thus
ProbNet changes only the order in which spatial candidates are attempted; it
does not change any count or quota.

Counts and type proportions use reliable patch-local observations directly.
Sparse observations shrink toward the matching
`dataset/reference_profile` library; BCSS is never a universal fallback.
Type and disconnected-component quotas use exact largest remainder. Same-class
reference shapes are preferred and the matching profile library supplies only
same-class shortages. A transformed shape `S` is accepted only when
`|S intersect N_current| = 0` and `S` is wholly contained in valid biological
tissue. Valid biological tissue is non-background tissue excluding
dataset-specific skipped labels; crossing between two valid tissue classes is
allowed. Any proposal sharing one or more pixels with an already retained or
newly placed nucleus, entering invalid/background tissue, or being truncated
at the patch edge is rejected. An immediately adjacent shape with no shared
pixel is not an overlap. Ordinary components use four same-class shape trials,
12 transforms, and a retry pool of at least `max(64, 12 * quota)`. Components
with quota at least 20 or expected nucleus occupancy at least 0.12 use six
shape trials, 24 transforms, and at least `max(128, 24 * quota)` candidates.
The sampler continues through the retry pool and never relaxes the zero-overlap
or full-containment rules. Exhausted slots remain explicitly unfilled.
Retained source nuclei must remain bitwise unchanged in the saved target mask.

Legacy `generation_profiles.json`, `phase4_probnet_workflow*.sh`, old
density-scale JSON overrides, count/density-head experiments, and historical
epoch-1 grids are outside the benchmark and must not appear in formal tables.

#### 5.8.5 Frozen output contract

```text
probnet_spatial_benchmark/
  provenance/
    frozen_config.yaml
    checkpoint_hashes.json
    split_and_endpoint_hashes.json
  p1_spatial_ablation/
    case_manifest.jsonl
    per_seed_rows.parquet
    paired_spatial_summary.json
    validation.json
  p2_geometry_endpoint/
    endpoint_120.jsonl
    planned_layouts/
    validation.json
  final_report.md
  final_conclusion.json
```

P1 and P2 must use separate output roots and manifests. No P1/P2 metric may
feed back into count rules, checkpoint choice, sampler parameters, generator
selection, or endpoint membership.

#### 5.8.6 Frozen-policy provenance

The spatial-only implementation is:

- `phase3_mask_edit/benchmark/probnet_spatial.py`;
- `scripts/run_probnet_spatial_benchmark.py`;
- `scripts/prepare_probnet_spatial_p2.py`;
- `scripts/launch_probnet_spatial_benchmark_epoch29.sh`;
- `scripts/launch_probnet_spatial_p2_epoch29.sh`;
- `scripts/write_probnet_spatial_geometry_report.py`.

The active run root is:

`/data1/zhao/wqx/benchmarks/runs/probnet_spatial_epoch29_strict_geometry_20260724`

P1 uses the grouped held-out test manifest with 600 source patches and two
reconstruction regions per patch, giving 1,200 case-regions. Each case-region
uses five repeated seeds and the four frozen samplers, for 24,000 per-seed
rows. Oracle hidden count and type quotas, area-largest-remainder component
quotas, candidate pools, same-class shape plans, and retry transforms are
shared within each case and seed. Only candidate ordering changes.

P2 first applies the frozen statistical count/type policy to all 9,950 grouped
held-out source rows without running ProbNet or any generator/evaluator. It
then ranks rows within each dataset by planned count, divides the ranking into
four disjoint load strata, and freezes five cases per dataset-by-stratum cell.
This gives the 120-case endpoint while preserving the absolute planned-count
bin as a separate field. The immutable checkpoint is loaded only after
endpoint membership is frozen, and contributes only the scalar
nucleus-presence field used for spatial ranking. P2 stops after the saved
nuclei mask and geometry validation; it has no H&E or CellViT stage.

#### 5.8.7 Completed result and final engineering interpretation

The completed run is:

`/data1/zhao/wqx/benchmarks/runs/probnet_spatial_epoch29_strict_geometry_20260724`

The engineering acceptance result is P2:

| Engineering endpoint | Result |
|---|---:|
| Completed cases | 120/120 |
| Requested / placed / unfilled nuclei | 1,457 / 1,457 / 0 |
| Placement completion | 100% |
| Exact type quota / component quota | 100% / 100% |
| Overlap pixels | 0 |
| Outside-valid-tissue pixels | 0 |
| Retained-nucleus preservation | 100% |

All six datasets and all four planned-count load strata passed. ORCA, the
dataset with the clearest strict-packing concern in P1, placed 249/249
requested nuclei. This demonstrates that the complete frozen policy can
convert an externally determined count/type plan into a valid instance mask
while preserving retained nuclei, enforcing zero overlap, and keeping every
complete shape inside valid tissue.

P1 contains exactly 24,000 unique per-seed rows: 1,200 case-regions, five
seeds, and four samplers. It isolates the positional value of ProbNet relative
to the primary `uniform` comparator:

| Supporting spatial endpoint | ProbNet | Uniform | Relative error reduction | Paired 95% CI for difference |
|---|---:|---:|---:|---:|
| NND W1 (µm) | 1.9056 | 1.9318 | 1.35% | `[-0.0596, 0.0051]` |
| Ripley-K normalized L1 | 0.1102 | 0.1108 | 0.55% | `[-0.0044, 0.0029]` |
| Tissue-boundary W1 (µm) | 4.1321 | 4.5258 | 8.70% | `[-0.6199, -0.1903]` |
| Component-occupancy L1 | 0.1241 | 0.1238 | -0.22% | `[-0.0007, 0.0012]` |

Only tissue-boundary distance was both statistically supported and at least
5% better. Neither core NND nor Ripley-K passed the frozen material-improvement
rule. The evidence therefore supports a narrow but useful role: epoch-29
ProbNet is a learned tissue- and boundary-aware prior for ordering otherwise
valid nucleus-placement candidates. It must not be described as learning
clearly superior higher-order cell spatial organization.

The P1 equal-dataset safety gate passed with exact zero overlap, exact zero
outside-valid-tissue pixels, and exact retained-nucleus preservation. However,
the ORCA oracle-reconstruction stress subset completed only 91.52% for ProbNet
and 91.51% for uniform; this is a shape-packing limitation shared by both
rankings and is not evidence against or in favor of learned candidate ranking.
It remains a disclosed limitation for artificial oracle-reconstruction loads.

Final project decision:

- keep ProbNet in the production pipeline as a supporting placement prior;
- attribute count and type decisions only to the statistical policy;
- attribute non-overlap, containment, and completion guarantees to the
  deterministic placement/retry rules, not to ProbNet;
- claim only that ProbNet adds a measurable boundary-aware bias while the full
  sampler is engineering-valid;
- do not make ProbNet a standalone scientific contribution and do not perform
  further checkpoint, count-head, density-head, or sampler tuning;
- do not create a standalone ProbNet results figure. The compact P2 acceptance
  table plus the P1 supporting table are sufficient; if space is limited, keep
  P2 in the main methods/results text and move P1 to supplementary material.

No H&E image, CellViT prediction, or CellViT-derived metric exists in the
strict run because image synthesis quality is outside this component's
responsibility.

Manuscript-ready Results text:

> ProbNet was used only as a tissue-aware prior for ranking candidate nucleus
> centers; nucleus counts and type quotas were fixed by an independent
> statistical policy. In a 120-case, six-dataset production endpoint, the
> complete sampler placed all 1,457 requested nuclei with exact quota
> agreement, zero overlap, zero outside-tissue pixels, and complete
> preservation of retained nuclei. In a controlled spatial ablation, ProbNet
> reduced tissue-boundary distribution error by 8.7% relative to uniform
> sampling, while NND and Ripley-K improvements were small and not
> statistically supported. These results establish ProbNet as an
> engineering-valid tissue- and boundary-aware placement prior, rather than a
> model of superior higher-order cellular organization.

### 5.9 CellViT-Derived Nuclei Metrics

This section applies to image-generation condition-consistency benchmarks, not
to the ProbNet spatial benchmark in Section 5.8.

Given:

- target nuclei mask `N_target`, if available;
- generated image `I_gen`;
- CellViT prediction `C_gen`.

Implementation warning: the current online helper named
`nuclei_density_relative_error` counts labelled pixels in semantic nuclei masks
and is therefore an occupied-nuclear-area proxy, not an instance count. Keep it
only as a backward-compatible online gate. Formal cell count, density, nearest
neighbour and matching metrics must consume CellViT instance JSON/coordinates
and the ProbNet placement diagnostics.

Common distribution metrics for Cross-v1, PixCell, and PathDiff only:

- total cell count and density absolute error;
- per-class count MAE and density absolute error;
- cell-type proportion JSD;
- inflammatory/TIL-like and dead-cell consistency when relevant.

Cell classes:

- neoplastic: 101
- inflammatory: 102
- connective: 103
- dead: 104
- epithelial: 105

Strict spatial metrics are evaluator-schema specific:

- Cross-v1 and PixCell: rerun the frozen CellViT evaluator and compare against the target CellViT instance condition with class-agnostic and class-aware matching F1, mean matched distance, and p95 matched distance at a frozen micron tolerance;
- PathDiff: rerun the frozen HoVer-Net CoNIC evaluator and compare against the actual CoNIC instance/type condition used for generation;
- never relabel a CellViT target into CoNIC types or compare PathDiff's CoNIC condition to CellViT as if their class schemas were identical;
- PathLDM, UniPath, MuPaD-text, and MuPaD-image do not receive the opposite patch's target geometry, so target cell density/count/type-distribution and strict spatial F1 are all `not_applicable`; evaluate them with Patho-KID and human realism instead.

Cross-level tissue-nuclei consistency is computed after mapping generated
CellViT centroids into the frozen Segmentator tissue prediction:

- nucleus density by tissue class;
- `P(cell type | tissue class)` JSD between target and generation;
- fraction of nuclei assigned to an incompatible or unintended tissue class;
- nucleus-to-tissue-boundary distance-distribution error;
- tumor/stroma/immune-region cell-composition error.

These cross-level values are secondary endpoints because both automatic
evaluators contribute error. They must be accompanied by tissue evaluator
ceiling results and may not be described as human-GT cellular organization.
For seed-robustness cases, compute nuclei and cross-level metrics per seed and
aggregate within case before case/WSI bootstrap.

The claim name is "segmentator/CellViT/HoVer-Net-derived condition consistency," not human-GT fidelity. Strict spatial metrics are only a main comparison between models conditioned on the corresponding target geometry.

### 5.10 Patho-KID

Feature extractors:

- UNI-2h
- CONCH

For each feature extractor `E`:

```text
X = E(withheld_real_target_images)
Y = E(generated_images)
Patho-KID = MMD^2(X, Y)
```

Use a polynomial kernel:

```text
k(x, y) = (x^T y / d + 1)^3
```

Use raw feature activations with the standard KID dimension scaling. For CONCH,
use the official image-only representation before the contrastive projection
head (`proj_contrast=False, normalize=False`). Do not combine row-wise L2
normalization with the `/d` scaling, because that suppresses the quadratic and
cubic terms.

Bootstrap:

- 100 bootstrap repeats;
- fixed random seed;
- sample size logged for each group;
- resample by WSI cluster, keeping both directions of a pair together;
- use identical bootstrap draws across models for paired model comparisons.
- report paired bootstrap `KID(model A) - KID(model B)` with its 95% interval and probability that model A is better.

Report:

- lower is better;
- primary: pooled Patho-KID over the complete frozen clinician-approved organ mixture, with at least 1,000 rows;
- secondary: organ-balanced Patho-KID on the frozen 300-row sensitivity set;
- optional: descriptive per-organ scores with sample-size warnings;
- include the six organ counts next to every pooled result;
- report missing/failed sample count and reject a headline score unless all models use the same accepted `sample_id` set.

The primary pooled score is valid because every model is compared on the same fixed target mixture. It answers which model best matches that mixed benchmark distribution. It does not estimate a macro-average organ performance.

The real set for a direction is its withheld `target_image_path`; the generated set is the image produced for the same `sample_id`. KID is distributional and does not use the A/B pairing in its kernel, but the pairing remains necessary to define fair generator inputs and to keep target images unique.

Do not merge UNI-2h and CONCH into one hidden score. Report both separately. This guards against evaluator coupling because PixCell uses UNI-2h as an input condition and UniPath RAG uses CONCH internally.

Real-vs-real calibration:

- use the same frozen 1,500-image real pool and organ mixture;
- run 100 repeated draws at `n=60, 100, 300, 500, 1000` per side;
- use two organ-preserving disjoint real sets whenever the pool can supply both sides;
- when disjoint sampling is impossible, use independent stratified empirical bootstrap and report source overlap; do not call that row a disjoint lower bound;
- show the calibration mean, standard deviation, and 95% interval next to model KID so a difference such as `0.4` versus `0.9` is interpreted relative to finite-sample variation.

The completed 1,500-image automatic-pool calibration is:

| Feature space | n per side | Sampling | Mean real-vs-real KID | SD |
|---|---:|---|---:|---:|
| CONCH | 60 | disjoint | `-4.54e-02` | `2.03e-02` |
| CONCH | 100 | disjoint | `-2.60e-02` | `1.47e-02` |
| CONCH | 300 | disjoint | `-9.00e-03` | `4.39e-03` |
| CONCH | 500 | disjoint | `-5.36e-03` | `2.47e-03` |
| CONCH | 1000 | independent bootstrap | `-2.73e-03` | `1.45e-03` |
| UNI-2h | 60 | disjoint | `-3.26e-03` | `1.69e-03` |
| UNI-2h | 100 | disjoint | `-1.78e-03` | `9.16e-04` |
| UNI-2h | 300 | disjoint | `-6.92e-04` | `3.55e-04` |
| UNI-2h | 500 | disjoint | `-3.72e-04` | `2.15e-04` |
| UNI-2h | 1000 | independent bootstrap | `-1.88e-04` | `9.62e-05` |

Small negative values are expected from the unbiased finite-sample estimator and should be interpreted as statistically near zero, not as a physically negative distribution distance.

Automatic-condition interim checkpoint (2026-07-24):

- this is a complete automatic-condition sensitivity result, not the
  publication-primary clinician-condition result;
- every model below uses the same strict-Oral common set of 1,421 directions,
  711 pair clusters, sample digest
  `c17f2a892605bda826a9c872989672327db15f2843efcf4985cd9a535be32fa3`,
  raw features, and 100 shared pair-cluster bootstrap draws;
- UniPath uses its completed 8K RAG and the frozen `30 steps / CFG 3.0`
  inference contract. Its native `384 x 384 @ 0.5 MPP` output is center-cropped
  to `[64, 64, 320, 320]` and resized to `512 x 512 @ 0.25 MPP`;
- lower Patho-KID is better. Brackets are 95% pair-cluster bootstrap
  intervals.

| Model | Fairness stratum | CONCH Patho-KID (rank) | UNI-2h Patho-KID (rank) |
|---|---|---:|---:|
| MuPaD-image auxiliary | reference-image auxiliary | `0.44346 [0.42595, 0.47272]` (1) | `0.03184 [0.03006, 0.03490]` (1) |
| MuPaD-text | text-only | `0.95403 [0.91393, 1.03871]` (2) | `0.04705 [0.04433, 0.05267]` (2) |
| PixCell-ControlNet | image + target cell geometry | `2.02206 [1.97427, 2.08858]` (3) | `0.11283 [0.10964, 0.11890]` (4) |
| UniPath-7B + RAG | text-only with internal RAG | `2.75615 [2.69523, 2.84624]` (4) | `0.08290 [0.08126, 0.08697]` (3) |
| PathDiff-CoNIC | text + target cell geometry | `3.21068 [3.14564, 3.31525]` (5) | `0.18047 [0.17498, 0.18746]` (5) |
| PathDiff-text 20x | text-only | `3.87364 [3.80122, 3.98425]` (6) | `0.19198 [0.18604, 0.20001]` (6) |

Every adjacent rank difference excludes zero in both feature spaces. Both
evaluators therefore agree on the text-only ordering
`MuPaD-text > UniPath > PathDiff-text` and that UniPath outperforms both
PathDiff modes. They disagree only on UniPath versus PixCell: CONCH favors
PixCell by paired `KID(PixCell) - KID(UniPath) =
-0.73127 [-0.79577, -0.66284]`, whereas UNI-2h favors UniPath by paired
`KID(UniPath) - KID(PixCell) =
-0.03000 [-0.03334, -0.02673]`. Do not resolve this disagreement with a hidden
average. PixCell consumes UNI-2h at inference and UniPath RAG consumes both
CONCH and UNI-2h-derived assets, so retain both evaluator spaces and disclose
the coupling.

Frozen reports:

- CONCH:
  `/data1/zhao/wqx/benchmarks/runs/pathokid_conch_generation_v3_plus_unipath_common1421_strict_oral_20260724`;
- UNI-2h:
  `/data1/zhao/wqx/benchmarks/runs/pathokid_uni2h_generation_v3_plus_unipath_common1421_strict_oral_20260724`.

### 5.11 Two-Primitive Representation-Response Metrics

For primitive `p`, strength `s`, synthesis path `b`, and encoder `E`:

`d[p,s,b,i] = E(I[p,s,b,i]) - E(I[original,i])`

Use raw embeddings for differences. Normalize individual displacement vectors
only when a cosine is calculated.

Within-primitive dose endpoints:

- leave-one-WSI-out alignment of Moderate and Significant displacements with
  the Moderate primitive direction;
- signed projection of
  `E(Significant) - E(Moderate)` on the held-out Moderate direction;
- fraction with positive incremental projection;
- displacement-norm change and ratio;
- strict cross-fitted `Original < Moderate < Significant` ordering.

Between-primitive endpoints:

- held-out U1/U2 mean-direction cosine;
- matched-reference U1/U2 displacement cosine;
- cross-fitted projection matrix on held-out U1 and U2 axes;
- own-axis-greater-than-other-axis fraction;
- WSI-level direction-label permutation test.

Fit every direction without any image from the scored WSI. Report UNI-2h and
CONCH separately and use WSI-cluster bootstrap intervals. Do not treat
quadratic pairs of cosine values as independent samples, do not select the U2
cohort using embeddings, and do not imply clinical downstream utility.

## 6. Model Baselines and Fairness

### 6.1 Models

Primary project model:

- frozen two-attempt agentic system with deterministic routing and verification.

Project ablations:

- fixed Inpaint;
- fixed production Cross-v1;
- routed attempt 1 without recovery.

External baselines:

- PixCell
- MuPaD
- PathLDM
- UniPath
- PathDiff / Pathdiff

### 6.2 Fairness Tiers

Tier A: directly comparable mask/edit-conditioned models

- Inputs include reference appearance and target structure/mask, or an equivalent conditioning signal.
- Eligible for strict generation condition-consistency comparison.

Tier B: image-conditioned pathology generators

- Inputs include a reference image embedding or image condition but may not accept exact target masks.
- Eligible for realism and partial semantic consistency.
- Image-mask fidelity should be reported only if target-mask conditioning is actually provided or approximated.

Tier C: text-only generators

- Inputs are text prompts only.
- Eligible for Patho-KID, human realism, and coarse semantic plausibility.
- Not eligible for strict spatial/mask fidelity ranking unless an official mask-conditioned mode exists.

The final report must clearly mark each metric as:

- `fair_main`: same or comparable inputs;
- `partial`: related but weaker conditioning;
- `not_applicable`: metric would be unfair or impossible.

### 6.3 Baseline Input Contracts

For the paired Patho-KID track, freeze the directional conditioning contract below. `A -> B` means that real patch `A` supplies any allowed reference-image condition while real patch `B` supplies only its offline target conditions; `B -> A` reverses those roles.

| Model | A -> B | B -> A |
|---|---|---|
| Project agentic | A + B masks/prompt, routed Inpaint/Cross | B + A masks/prompt, routed Inpaint/Cross |
| Fixed Cross-v1 ablation | A + B masks/prompt | B + A masks/prompt |
| PixCell-ControlNet | UNI(A) + binary mask(B) | UNI(B) + binary mask(A) |
| PathDiff | CoNIC mask(B) + prompt(B) | CoNIC mask(A) + prompt(A) |
| PathDiff-text | prompt(B) | prompt(A) |
| PathLDM/UniPath | prompt(B) | prompt(A) |
| MuPaD-text | prompt(B) | prompt(A) |
| MuPaD-image auxiliary | A | B |

Rules:

- `prompt(A)` and `prompt(B)` are independent descriptive pathology prompts for the corresponding real patches, not edit instructions or comparative text.
- `masks(A)` and `masks(B)` include the target tissue mask and target nuclei/cell condition required by the selected Cross-v1 generation route.
- PixCell receives the binary mask representation expected by its deployed ControlNet checkpoint; record the conversion source and foreground definition in metadata.
- PathDiff receives a CoNIC-style nuclei mask produced by the frozen CoNIC-compatible segmentator/conversion pipeline, not a CellViT mask relabeled without validation.
- The target RGB image is always withheld from every generator. Offline conditions derived from the target RGB are allowed only as listed above and must record their model/checkpoint and provenance.
- `MuPaD-image auxiliary` is an image-conditioned auxiliary experiment, not the main text-only MuPaD comparison, because it does not receive the target structure of the opposite patch. Its native input is a real `1024 @ 0.25 MPP` WSI crop centered on A and downsampled to `512 @ 0.5 MPP`; reflected padding is forbidden in the formal run.
- UniPath and MuPaD-text are explicitly text-to-image baselines. Their generated images are native 20x (`0.5 MPP`) outputs and must be center-cropped to the shared 128 um field before being resized to the `0.25 MPP` evaluation grid.
- PathLDM uses the released 10x TCGA-BRCA checkpoint and is therefore evaluated only on breast. It must not be shown for other organs.
- PathLDM's tumor/TIL categorical prefix is a required model-native text condition. The fixed pilot uses `High tumor; low TIL;`; any formal per-patch levels must be frozen before generation and recorded in metadata rather than inferred from the held-out target RGB at inference time.
- PixCell and PathDiff receive full-field 20x cell conditions: downsample the complete `512 @ 0.25 MPP` condition to `256 @ 0.5 MPP`; do not center-crop the condition or generated output.
- PathDiff's released `11-02T02-36-project.yaml` is the PathCap + CoNIC six-channel configuration. The formal loader must reject missing or unexpected checkpoint keys; the deployed `last.ckpt` passed this compatibility smoke.
- PathDiff-text must call the official `sampling.py::sample_one(..., mode='t2i')` on that same joint checkpoint. The branch uses a six-channel constant `NULL_MASK=10` for both conditional and unconditional control paths; it must not receive a generated CoNIC mask. PathCap training performs a 256-pixel center crop without MPP normalization, so the formal prompt explicitly freezes 20x objective magnification. Treat `0.5 MPP` as a prompt-conditioned nominal scale, retain the complete generated field, resize it `256 -> 512`, and never imply that the checkpoint enforces physical MPP exactly.
- PixCell uses the official UNI-2h image-conditioning branch and a binary foreground control derived from the target CellViT mask. Patho-KID must therefore show both UNI-2h and CONCH spaces separately rather than hiding evaluator coupling in one score.
- UniPath uses its official conversation template, CONCH-backed RAG/prototype retrieval, 30 sampling steps, and CFG `3.0`; its 384-pixel training/generation grid is 20x and is center-cropped to the shared physical field.
- MuPaD-text and MuPaD-image use 250 steps, guidance `2.5`, `guidance_high=0.75`, `guidance_low=0.0`, SDE mode, and linear path. The formal image run requires exact central reproduction of A from the WSI and excludes any direction whose 256-um reference context overlaps the held-out target patch or falls outside the WSI boundary. Patho-KID comparisons involving MuPaD-image use its final audited sample IDs as the common subset for every compared model. After the strict-Oral filter this common set contains 1,421 directions; the frozen CONCH and UNI-2h reports are under `pathokid_{conch,uni2h}_generation_v3_common1421_strict_oral_20260722`.

Create one YAML config per model:

```yaml
model_id: pixcell
display_name: PixCell
remote_host: amax2
entrypoint: /data1/zhao/wqx/benchmarks/scripts/run_pixcell_benchmark.sh
input_contract:
  reference_image: required
  target_tissue_mask: optional
  target_nuclei_mask: optional
  prompt: optional
  old_prompt: optional
  new_prompt: optional
  instruction: optional
output_contract:
  image_path: required
  metadata_json: required
resolution: 1024
seed_policy: fixed_from_manifest
notes: ""
```

Similar configs should be created for:

- `project_agentic`
- `fixed_inpaint`
- `fixed_cross_v1`
- `mupad`
- `pathldm`
- `unipath`
- `pathdiff`

The runner should never hard-code baseline-specific paths. It should read model configs.

## 7. Implementation Layout

Existing reusable implementation:

- generation contracts/runner:
  `phase3_mask_edit/benchmark/generation_models.py`,
  `phase3_mask_edit/cli/run_generation_benchmark.py`,
  `scripts/run_generation_baseline.py`;
- condition consistency:
  `phase3_mask_edit/benchmark/conditional_fidelity.py`,
  `phase3_mask_edit/cli/run_conditional_fidelity_benchmark.py`,
  `scripts/prepare_conditional_fidelity_predictions.py`;
- Patho-KID:
  `phase3_mask_edit/benchmark/pathokid.py`,
  `phase3_mask_edit/cli/run_pathokid_benchmark.py`,
  `phase3_mask_edit/cli/run_pathokid_calibration.py`;
- project agentic single-case workflow:
  `controlnet_train/inference/agentic.py`,
  `scripts/run_agentic_edit_workflow.py`;
- representation analysis:
  `phase3_mask_edit/benchmark/embedding_utility.py` and the existing
  dose-response/plot scripts;
- Segmentator and ProbNet evaluation utilities under `scripts/`.

Required additions before the formal run:

1. a paired-manifest batch adapter for `project_agentic` that calls the
   existing agentic workflow and writes attempt-1 plus selected-final
   manifests;
2. formal model configs for `project_agentic`, `fixed_inpaint`, and
   `fixed_cross_v1`; the current reuse-only 60-case Cross preview config is
   not the formal project method;
3. DoctorCond/AutoCond paired summary support and real-target evaluator-ceiling
   staging in the conditional-fidelity runner;
4. tissue-nuclei cross-level consistency metrics;
5. ProbNet P1 spatial-ablation and P2 endpoint CLIs/configs;
6. U2 cohort construction, disjoint-mask audit, two-primitive analysis, and
   shared-basis plotting;
7. `phase3_mask_edit/cli/export_human_review_deck.py`;
8. `phase3_mask_edit/cli/compile_benchmark_report.py`;
9. one machine-readable freeze-manifest validator.

Recommended formal run layout:

```text
runs/benchmark_v1/
  freeze/
    benchmark_freeze_manifest.json
    validation.json
  manifests/
    paired_doctor_condition.csv
    paired_automatic_condition.csv
    paired_balanced_300.csv
    edit_local_smoke_3_per_cell.jsonl
    edit_local_formal_10_per_cell.jsonl
    seed_robustness.jsonl
  generated/
    project_agentic/
      doctor_condition/
      automatic_condition/
      attempt_1/
      selected_final/
    fixed_inpaint/
    fixed_cross_v1/
    pixcell/
    pathdiff/
    text_only_baselines/
  generation_consistency/
    doctor_condition/
    automatic_condition/
    edit_local/
    seed_robustness/
    real_target_ceiling/
  probnet/
    p1_spatial_ablation/
    p2_layout_to_image/
  pathokid/
    pooled/
    balanced_300/
    condition_source_delta/
  representation_response/
    u1_tumor_burden/
    u2_stromal_immune/
    two_primitive_analysis/
  human_review/
  report/
```

Large immutable data/model roots may remain outside the repository, but every
formal row must resolve through the freeze manifest and preserve hashes.

## 8. Execution Plan

### Phase 0: Lock Active Formal Inputs

Before any remaining formal generation:

1. freeze the clinician-approved pooled and balanced-300 manifests;
2. freeze DoctorCond and AutoCond tissue-mask paths and hashes;
3. freeze the production statistical-count/ProbNet nuclei policy and every
   dataset/reference-profile library hash;
4. freeze the final Segmentator, CellViT, optional native HoVer-Net evaluator,
   preprocessing, MPP, and label maps;
5. freeze project router and verifier thresholds on a non-test calibration
   split;
6. freeze generator and feature-extractor checkpoints;
7. freeze seed policy, code revision, WSI/patient groups, annotation status,
   and adjudication rules.

Write one machine-readable freeze manifest before generation. It is the source
of truth for every later report.

### Phases 1-3: Mask Semantic Benchmark — Complete

The mask intent bank, checked prompts, target-mask bank, GT execution ceiling,
Instruction benchmark, same-reference ordinal analysis, and agentic contour
replanning report are frozen. Canonical results and paths are documented in
`docs/mask_edit_benchmark_v1_runbook.md`.

Do not rerun or retune mask-semantic models, prompts, magnitude thresholds, or
replanning policy as part of the remaining generation benchmark. Reuse the
frozen target masks for Track G2. Any operational re-export must preserve the
canonical sample IDs, mask hashes, and final metrics exactly.

### Phase 4: Freeze Generation Manifests

#### 4.1 Paired DoctorCond/AutoCond manifest

1. Import the active strict-Oral 727-pair / 1,454-direction candidate set.
2. Complete pair screening, patch-level tissue-mask correction, the 150 blinded
   independent redraws, and adjudication.
3. Recompute pair metrics with clinician masks and replace/review rows only
   when required to retain at least 500 pairs / 1,000 directions and at least
   25 pairs per organ for the balanced sensitivity set.
4. Freeze the complete accepted manifest and the 300-direction organ-balanced
   manifest.
5. Store both `auto_target_tissue_mask_path` and
   `clinician_target_tissue_mask_path`; resolve DoctorCond to the clinician
   mask and AutoCond to the final frozen Segmentator mask.
6. Freeze one tissue-only sensitivity manifest with identical nuclei targets
   and one full-pipeline sensitivity manifest whose nuclei layouts are rebuilt
   from each tissue condition.
7. Hash every RGB, mask, instance condition, prompt, checkpoint, seed policy,
   coordinate, and code/config revision.

Each row records pair/direction IDs, organ, WSI/patient, coordinates, reference
assets, withheld target RGB, both tissue-mask sources, nuclei-condition source,
clinician status, correction severity, fairness tier, and model-native
condition conversion. Do not start the official run with a pending/ungradable
row or a missing official mask.

#### 4.2 Edit-local consistency manifests

Build the 3-per-cell smoke and 10-per-cell project formal manifests from the
already-frozen mask-semantic target bank. Record source/target tissue and nuclei
masks, primitive, strength, changed/preserved regions, prompts, seeds,
WSI/patient groups, route-policy hash, and evaluator-policy hash.

#### 4.3 Seed-robustness manifest

Freeze 60-120 parent conditions and expand each to 4-5 seed rows. Parent
selection uses only pre-generation strata and may not use visual or metric
results.

### Phase 5: Run Generators

Command shape:

```bash
python -m phase3_mask_edit.cli.run_generation_benchmark \
  --manifest runs/benchmark_v1/manifests/paired_pathokid_directions.csv \
  --config-dir benchmark_configs/models \
  --output-root runs/benchmark_v1/generated \
  --execute
```

Remote behavior:

- The local runner should support a dry-run mode that prints each remote command.
- Remote commands should write one metadata JSON per generated image.
- If `amax2` SSH is unavailable, the runner should stop before modifying remote state and mark the model as `remote_unavailable`.

Required metadata per output:

- model id/version;
- checkpoint;
- seed;
- input paths;
- output path;
- runtime;
- GPU;
- command;
- status/error.

Paired consistency/Patho-KID requirements:

- every model receives exactly the same frozen clinician-approved `sample_id` rows, with at least 1,000 rows;
- the target RGB is never provided to inference;
- DoctorCond mask-conditioned models receive the frozen clinician target mask;
- AutoCond receives the final frozen Segmentator mask on identical accepted IDs;
- each target mask is converted only as required by a model's native input schema;
- text-only models receive the frozen target description/prompt without access to target RGB;
- failed rows are rerun until complete and are not removed independently per model.

#### Project-model full agentic track

The project method must be benchmarked as the complete frozen agentic system,
not as a manually selected output from either generation backend. For every
clinician-approved paired direction, the formal project-model path is:

```text
frozen reference image and masks
        +
frozen target tissue and nuclei masks
        +
frozen edit instruction
        |
        v
extract route features and select the primary generation route
        |
        v
generate the first candidate
        |
        v
re-segment with the frozen tissue segmentator and CellViT
        |
        v
compute changed-region accuracy / macro-IoU,
CellViT-derived nuclei consistency,
and route-specific preservation diagnostics
        |
        +-- passes frozen thresholds --> select as `validated`
        |
        +-- fails frozen thresholds --> choose one deterministic alternate route
                                           |
                                           v
                                   generate and verify once more
                                           |
                                           +-- passes --> select as recovered
                                           |
                                           +-- fails --> select the best scored
                                                         candidate but retain
                                                         `needs_review`
```

The batch benchmark runner must dispatch the project-model row to
`scripts/run_agentic_edit_workflow.py` (or a batch wrapper around the same
`controlnet_train/inference/agentic.py` implementation). It must not call
`inpaint` or production Cross directly and then manually choose the better
image. External baselines continue to use their frozen native inference
contracts and do not receive project-verifier-driven retry or model selection.

Frozen inputs and policy:

- freeze `sample_id`, reference RGB, reference tissue mask, reference nuclei
  mask, official target tissue mask, target nuclei mask/layout, edit region,
  preserved region, instruction, seed policy, and all file hashes before the
  official run;
- withhold the paired real target RGB from every inference process;
- freeze the router features and thresholds, verification thresholds,
  generator checkpoints, evaluator checkpoints, and fallback mapping before
  inspecting official benchmark outputs;
- set `max_attempts = 2`: one primary route and at most one alternate route;
- prohibit prompt changes, threshold changes, manual route overrides, extra
  seed searches, or additional attempts on individual scientific failures.

Required per-row execution:

1. Compute the tissue-normalized change ratio, connected components,
   bounding-box coverage, spatial distribution, and semantic label transitions
   from the frozen reference and target masks.
2. Select the primary route using the frozen router. Compact changes use
   `inpaint` first; large, multi-transition, or distributed changes use
   production Cross V1 plus pix2pix-v2 first; the frozen gray-zone rule is
   applied without per-sample intervention.
3. Generate attempt 1 and immediately run the frozen tissue segmentator and
   CellViT on the complete generated RGB image.
4. Compare the predicted tissue mask with the official target mask inside the
   edited region and compare the predicted nuclei map with the frozen target
   nuclei state. For Cross-v1, additionally compare the predicted tissue mask
   with the reference mask outside the edit. For inpaint, retain unchanged
   drift and a narrow boundary-ring diagnostic for QA, but do not mix either
   with target-region fidelity.
5. Record changed-region pixel accuracy, changed-region macro-IoU,
   CellViT-derived nuclei consistency, route-specific preservation metrics,
   failed checks, route, route reason, runtime, and artifact paths before
   making the next decision. Full-image accuracy is never used.
6. Apply route-specific frozen gates. Inpaint is gated by changed-region tissue
   fidelity and nuclei consistency. Cross-v1 uses those same gates plus the
   unchanged-region drift gate. If attempt 1 fails, target-structure or nuclei
   failures prefer production Cross, whereas Cross-v1 preservation failure
   prefers inpaint; a tool failure may use the remaining backend as specified
   by the production policy.
7. Generate and verify attempt 2 once. If it passes, select it as an
   alternate-route recovery. If it also fails, retain the highest-scoring
   valid candidate as the selected image but keep the final status
   `needs_review`. A `needs_review` row remains in all denominators and must not
   be silently removed from conditional-fidelity or realism evaluation.

Scientific verification failure and operational failure must be separated:

- a **scientific verification failure** has a valid generated image and valid
  evaluator outputs but fails one or more frozen acceptance thresholds; it is
  counted in first-pass, recovery, and residual-review statistics;
- an **operational failure** has no valid image or no valid evaluator result
  because of OOM, SSH interruption, corrupt output, missing dependency, or
  process failure; it remains `operational_incomplete` and is rerun with the
  same frozen inputs and policy until execution completes;
- the Phase 5 requirement that failed rows are rerun until complete refers to
  operational failures, not to unlimited retries after a scientific failure.

Required artifacts and metadata for every project-model row:

```text
generated/project_agentic/<sample_id>/
  attempt_01/
    generated_image.png
    verification/
    attempt_metadata.json
  attempt_02/                 # present only when recovery is triggered
    generated_image.png
    verification/
    attempt_metadata.json
  generated_image.png         # selected final candidate
  agentic_workflow.json        # append-safe history of every attempt
  pipeline_summary.json        # compact final decision
```

The metadata must preserve:

- policy version/hash, router thresholds, verification thresholds, and
  `max_attempts`;
- reference/target mask hashes and evaluator/checkpoint hashes;
- primary route, route features, route reason, generator checkpoint and seed;
- every attempt image, runtime, verification metrics, failed checks and error;
- fallback trigger, alternate route and recovery decision;
- `first_pass_status`, `recovery_used`, `selected_attempt`, `final_status`,
  total attempts and total runtime.

Benchmark outputs and reporting:

- the primary **full agentic system** result uses the selected final candidate
  for every frozen `sample_id`, including candidates marked `needs_review`;
- retain attempt 1 as a separate **first-pass routed generation** result rather
  than overwriting it when recovery succeeds;
- Phase 6 conditional-fidelity evaluation must be run on both attempt-1 and
  selected-final manifests so that recovery gains and regressions are visible;
- Phase 7 Patho-KID and blinded expert review use the complete selected-final
  manifest as the primary full-system result; first-pass outputs are retained
  as a prespecified sensitivity/ablation result;
- report route distribution, first-pass pass rate, alternate-route trigger
  rate, alternate-route recovery rate, final validated rate, `needs_review`
  rate, failure reason, mean attempts, and added runtime;
- compare fixed `inpaint` only, fixed production Cross only, routed first-pass
  without recovery, and the full agentic system on identical sample IDs;
- show inpaint and Cross-v1 as separate rows/panels. Use changed-region tissue
  and nuclei endpoints as the common comparison, report Cross-v1 preservation
  separately, and do not rank routes with the current single aggregate
  verification score;
- estimate false acceptance with blinded clinician/pathologist audit rather
  than defining it solely from the same automatic verifier that accepted the
  image.

Because the tissue segmentator and CellViT participate in both online routing
and Phase 6 conditional-fidelity measurement, scores from those exact models
are verifier-coupled evidence, not a fully independent confirmation. The paper
must therefore report first-pass and final outputs separately, validate the
tissue segmentator against clinician-corrected masks, describe CellViT metrics
as detector-derived nuclei consistency, and use blinded expert review plus
Patho-KID as complementary evidence. If an independently frozen evaluator is
available, add it as a sensitivity analysis without changing the online
verification policy.

### Phase 6: Run Generation Consistency Evaluators

#### 6.1 Freeze the evaluator

Select the final Segmentator only on real human tissue masks, using a
WSI/patient-disjoint calibration split that is not part of the official
generation test. Compare:

- image-only baseline;
- CellViT teacher with image-only inference;
- optional runtime CellViT input.

Runtime CellViT input is deployed only when it improves over teacher by at
least `+0.015` mIoU or `+0.03` Boundary-F1@4 on human tissue GT without a
material organ-level regression. It must also pass zero-prior and
missing/noisy-prior stress tests. Freeze the Segmentator, CellViT, thresholds,
preprocessing, and hashes before official generation.

#### 6.2 Paired target-condition consistency

Stage predictions for DoctorCond and AutoCond separately:

```bash
python scripts/prepare_conditional_fidelity_predictions.py \
  --manifest runs/benchmark_v1/manifests/paired_pathokid_directions.csv \
  --generated-root runs/benchmark_v1/generated_normalized/doctor_condition \
  --output-root runs/benchmark_v1/conditional_fidelity/doctor_condition

python scripts/prepare_conditional_fidelity_predictions.py \
  --manifest runs/benchmark_v1/manifests/paired_pathokid_directions.csv \
  --generated-root runs/benchmark_v1/generated_normalized/automatic_condition \
  --output-root runs/benchmark_v1/conditional_fidelity/automatic_condition
```

Run the common evaluator for each condition source and additionally stage the
withheld real target RGB through Segmentator to obtain the real-image ceiling.
Require identical accepted sample IDs for paired condition-source deltas.

Deliverables:

- generated and real-target tissue predictions;
- CellViT and native CoNIC/HoVer-Net predictions where applicable;
- tissue, nuclei, and tissue-nuclei per-sample metrics;
- real-image evaluator ceiling by organ/class;
- DoctorCond-minus-AutoCond paired summaries with shared pair/WSI bootstrap
  draws;
- explicit applicability/fairness labels.

#### 6.3 Edit-local fidelity and preservation

Run the 3-per-cell smoke before the 10-per-cell focused formal set. Produce
separate manifests for fixed Inpaint, fixed Cross, routed attempt 1, and
agentic selected final. Keep every scientific `needs_review` row in the
denominator and rerun only operationally incomplete rows.

#### 6.4 Seed robustness

Freeze 60-120 cases and 4-5 seeds per case. Store seed as a nested replicate
identifier and publish case-level fidelity distribution plus diversity.

### Phase 7: Run Paired Patho-KID

DoctorCond is the publication-primary run. AutoCond is a paired sensitivity
analysis on the same clinician-accepted IDs.

```bash
python -m phase3_mask_edit.cli.run_pathokid_benchmark \
  --config benchmark_configs/pathokid.yaml \
  --manifest runs/benchmark_v1/manifests/paired_pathokid_directions.csv \
  --real-path-field target_image_path \
  --sample-id-field sample_id \
  --bootstrap-group-field wsi_id \
  --feature-extractors conch uni2h \
  --model-root project_doctor=runs/benchmark_v1/generated/project_doctor \
  --model-root project_auto=runs/benchmark_v1/generated/project_auto \
  --model-root pixcell=runs/benchmark_v1/generated/pixcell \
  --output-root runs/benchmark_v1/pathokid/pooled
```

Run the frozen 300-direction balanced manifest as a second invocation. If the
complete dual-condition run is computationally prohibitive, DoctorCond remains
complete and AutoCond is restricted to this balanced subset.

Requirements:

- all compared models and condition sources use the same complete sample IDs
  within a report;
- cache raw embeddings with extractor/checkpoint and input digests;
- report pooled CONCH and UNI-2h KID before descriptive organ results;
- print organ, pair, direction, and WSI counts next to every score;
- use identical pair/WSI bootstrap draws for model and condition-source deltas;
- report `KID(DoctorCond) - KID(AutoCond)` with a paired interval;
- retain the real-vs-real calibration as estimator context;
- never interpret a Patho-KID change as condition fidelity without the Phase 6
  tissue/nuclei results;
- save exact sample IDs, bootstrap seed, normalization provenance, and failure
  audit for every score.

### Phase 8: Complete the Two-Primitive Representation Insight

The finalized tumor-burden-increase run in Section 1.4 is immutable. Do not
rerun historical epoch-1 or superseded nuclei-bank analyses.

For the paired U1/U2 cohort:

1. freeze the 600-reference immune-rich, tumor/stroma/immune-positive static
   candidate pool using the thresholds in Section 1.4;
2. execute independent native Moderate and Significant LLM contours for both
   U1 and U2, using at most two validator-guided attempts per mask;
3. apply only formal primitive validation and the mask-side Significant-minus-
   Moderate dose gate; measure strength containment and U1/U2 spatial overlap
   without forcing or selecting on either;
4. freeze 300 WSI-balanced eligible rows and all source/target/change-mask
   hashes before nuclei generation;
5. build both strengths and both primitives with the frozen production
   statistical-count plus ProbNet spatial policy;
6. generate local and reference-guided global images with identical target
   masks across backends for each row;
7. audit every image against the frozen nuclei and generation provenance;
8. extract fresh UNI-2h and CONCH caches;
9. run within-primitive dose continuation and cross-fitted U1-versus-U2
   direction separability, including a high-mask-overlap sensitivity;
10. render one shared-basis figure per encoder and a compact WSI-cluster
    statistical report.

No case may be included, excluded, or reassigned based on an embedding,
Patho-KID value, or visual preference. The result remains a
representation-space insight and is not promoted to a general downstream
utility benchmark.

### Phase 9: Doctor Review

Export deck:

```bash
python -m phase3_mask_edit.cli.export_human_review_deck \
  --manifest runs/benchmark_v1/manifests/paired_doctor_condition.csv \
  --generated-root runs/benchmark_v1/generated \
  --cases 36 \
  --models project_agentic pixcell mupad pathldm unipath pathdiff \
  --duplicate-rate 0.10 \
  --output runs/benchmark_v1/human_review
```

Deck requirements:

- randomized;
- blinded model id;
- image ids only;
- no model names in filenames shown to doctors;
- include reference/target mask context only if the review question requires it.

Preferred display:

- For realism-only rating: generated image only.
- For structure plausibility rating: generated image + target mask overlay or small side-by-side target mask, still model-blinded.

This phase rates generated images. Clinician tissue-mask QA for the paired Patho-KID inputs occurs before generation in Phase 4 and must not be deferred to this phase.

### Phase 10: Compile Final Report

Command shape:

```bash
python -m phase3_mask_edit.cli.compile_benchmark_report \
  --run-root runs/benchmark_v1 \
  --output runs/benchmark_v1/report
```

Final report sections:

1. Dataset and sample counts.
2. Mask semantic benchmark.
3. Image-mask fidelity.
4. Patho-KID.
5. Two-primitive representation-space response.
6. Doctor review.
7. Baseline fairness notes.
8. Failure analysis.
9. Shortfalls and limitations.

## 9. Statistical Reporting

Use WSI/patient-level bootstrap whenever WSI/patient IDs exist.

For proportions:

- report mean and 95% CI;
- Wilson CI acceptable for simple per-cell small-n summaries;
- bootstrap preferred for grouped results.

For continuous metrics:

- mean, median, IQR;
- 95% bootstrap CI.

For model comparison:

- paired bootstrap when models generated the same benchmark cases;
- use identical pair/WSI draws for DoctorCond versus AutoCond;
- Wilcoxon signed-rank for doctor Likert scores;
- multiple-comparison correction for many per-cell tests.

For ProbNet P1, compare samplers within case and seed, aggregate seeds within
case, and bootstrap case/WSI groups. For seed robustness, never count seeds as
independent samples.

For the two-primitive representation analysis, refit every direction while
holding out the scored WSI, bootstrap by WSI, and perform the direction-label
permutation at WSI level.

For paired Patho-KID:

- primary inference uses the complete pooled clinician-approved set, with at least 1,000 unique directions;
- use WSI-cluster bootstrap and identical resamples across models;
- report the frozen organ mixture explicitly;
- report the organ-balanced 300-row analysis as sensitivity, not as a replacement headline;
- do not claim organ-specific superiority when an organ has a small retained sample or a wide WSI-cluster interval.

Do not assign Patho-KID reliability from a fixed image-count threshold alone. For optional subgroup scores:

- report image count, pair count, WSI count, and the WSI-cluster bootstrap interval;
- treat small-organ subgroup scores as descriptive;
- suppress strong subgroup ranking when intervals are wide or only a few WSIs dominate;
- keep the complete pooled clinician-approved result as the primary claim.

## 10. Acceptance Criteria

Mask semantic benchmark:

- status remains frozen/complete;
- canonical sample IDs, artifacts, and metrics match the final runbook;
- no remaining workstream changes its prompts, target masks, or semantic
  thresholds.

Segmentator evaluator:

- selected only on real human tissue masks with WSI/patient separation;
- report six-organ and per-class mIoU/Dice, Boundary-F1, HD95, confusion, and
  failure categories;
- preserve an untouched human-mask evaluation split for the official report;
- runtime CellViT input, if selected, passes the predeclared gain threshold and
  missing/noisy/zero-prior stress tests;
- checkpoint, preprocessing, label map, CellViT version, and code hash are
  frozen before official generation.

Generation consistency:

- every generated image has complete status/provenance metadata and evaluator
  output;
- DoctorCond is publication-primary and AutoCond uses identical accepted IDs
  within every paired comparison;
- tissue-only and full-pipeline condition-source ablations are labelled
  separately;
- non-background macro mIoU and Boundary-F1@4 are the primary tissue endpoints;
- real-target Segmentator ceiling is reported by organ/class;
- strict nuclei spatial metrics use instance coordinates and retain native
  schema labels;
- text-only methods are not ranked on unavailable geometry;
- Inpaint and Cross are route-stratified; no full-image accuracy or hidden
  aggregate score is used;
- attempt 1, selected final, scientific failures, and operational failures
  remain distinguishable;
- seed replicates are nested within case and all lower-tail failures remain
  visible.

ProbNet spatial sampling and layout realization:

- immutable epoch-29 checkpoint, profile libraries, code revision, split,
  endpoint, baseline definitions, seeds, and configs are hashed;
- no count/density/type head or sampler hyperparameter is selected;
- P1 holds count, type quotas, candidates, shapes, component quotas, and retry
  rules fixed across spatial samplers;
- P1 includes ProbNet, Poisson-only, uniform, and boundary-distance baselines
  with paired case/WSI inference;
- a learned-spatial superiority claim requires a predeclared spatial endpoint
  improvement over the primary `uniform` comparator, including a material
  NND or Ripley-K improvement, without a material safety regression;
- P2 uses the fixed 120-case dataset-by-endpoint-count-stratum endpoint
  (within-dataset `low`, `mid-low`, `mid-high`, and `high` planned-count
  loads) and evaluates only geometry realization of the saved nuclei mask;
- each case resolves its own `dataset/reference_profile`; BCSS is never a
  global fallback;
- exact tissue/type/component quotas and instance-level shape provenance are
  recorded;
- placement completion is at least 98%, retained source nuclei are bitwise
  preserved, full shapes remain inside valid biological tissue, overlap is
  exactly zero, and every unfilled placement remains in the denominator;
- H&E synthesis and CellViT are excluded from the ProbNet benchmark.

Patho-KID:

- clinician-approved pooled manifest contains at least 1,000 directions and the
  300-direction balanced manifest is frozen;
- all compared rows share complete sample IDs and unique real targets;
- raw embedding caches, extractor versions, normalization, bootstrap draws,
  and organ/WSI counts are recorded;
- UNI-2h and CONCH are reported separately;
- DoctorCond primary and AutoCond sensitivity use paired pair/WSI intervals;
- no KID change is interpreted as condition adherence without Phase 6 metrics.

Representation-space insight:

- U1 remains immutable;
- the completed U1 result remains immutable, while the direct U1/U2 comparison
  uses a separately frozen 300-row immune-rich paired cohort;
- cohort selection uses source-mask thresholds, formal primitive validation,
  and realized dose only, never U1/U2 overlap, image appearance, or embeddings;
- both primitives use independent native Moderate and Significant contours,
  both synthesis paths, and both encoders;
- natural Moderate/Significant containment and U1/U2 mask overlap are reported,
  with a prespecified high-overlap sensitivity analysis;
- every direction is fitted without the scored WSI;
- within-primitive dose continuation and between-primitive discrimination are
  reported together;
- no clinical downstream-utility or embedding-orthogonality claim exceeds the
  frozen evidence.

Doctor review:

- deck is randomized and model-blinded;
- duplicate reliability images are included;
- at least one complete doctor score file is imported; two doctors are
  preferred for formal claims;
- false acceptance and `needs_review` examples are deliberately sampled.

Remote execution:

- jobs are resumable from per-sample metadata;
- an unavailable remote host causes no partial new run state;
- operational retries preserve frozen scientific inputs and policy.

## 11. Main Risks and Mitigations

Risk: generation volume expands before the evaluation contract is stable.

- Mitigation: require the 3-per-cell smoke before the 10-per-cell focused run;
  use the existing paired generation bank for target-condition consistency and
  Patho-KID instead of duplicating images unnecessarily.

Risk: text-only baselines are penalized for spatial conditions they never
receive.

- Mitigation: enforce metric applicability by native input contract. Strict
  tissue/nuclei spatial rankings contain only geometry-conditioned models.

Risk: Segmentator error contaminates consistency and the online verifier.

- Mitigation: select and validate Segmentator on WSI-disjoint human masks,
  report the real-image evaluator ceiling, keep official generation test masks
  out of model selection, and add an image-only sensitivity evaluator if
  runtime CellViT input is used.

Risk: automatic tissue masks define an incorrect scientific task.

- Mitigation: clinician-screen the active 727-pair pool, correct official
  conditions, freeze DoctorCond as primary, and quantify AutoCond sensitivity
  on identical accepted IDs.

Risk: CellViT-derived nuclei metrics are mistaken for human nuclei GT.

- Mitigation: lock the exact evaluator, preserve native schemas, name the
  endpoint detector-derived consistency, and use blinded human review for
  complementary evidence.

Risk: learned ProbNet spatial placement receives credit for statistical counts
or shape transfer.

- Mitigation: P1 holds count, type, candidates, shapes, quotas, and retry rules
  fixed; P2 separately measures geometry-only placement realization. Neither
  endpoint generates H&E or uses CellViT.

Risk: Patho-KID is interpreted as conditional fidelity.

- Mitigation: report KID only beside tissue/nuclei consistency and paired
  DoctorCond/AutoCond deltas. A realism gain does not imply mask adherence.

Risk: U2 appears separable because cases were selected in embedding space.

- Mitigation: select the paired cohort only with source-mask composition,
  formal mask validation, and dose gates; never select on cross-primitive mask
  overlap, generated RGB appearance, or embeddings. Freeze before embedding
  extraction and fit directions with held-out WSIs.

Risk: forced mask disjointness or forced Moderate/Significant containment
creates an artificially clean representation trajectory.

- Mitigation: use independent native LLM contours for every
  `primitive x strength` target. Treat natural containment and U1/U2 overlap as
  measured covariates, and report the high-overlap sensitivity rather than
  geometrically rewriting masks.

Risk: pooled Patho-KID is dominated by organ mixture.

- Mitigation: report organ/pair/WSI composition and the frozen
  300-direction organ-balanced sensitivity analysis.

Risk: WSI leakage or repeated seeds inflate confidence.

- Mitigation: split and bootstrap by patient/WSI, keep pair directions
  together, and nest all seeds within case.

## 12. Active Next Actions

Completed and closed:

- mask-semantic GT/Instruction benchmark and contour-replanning result;
- automatic paired condition bank and generation baseline implementation pilots;
- real-vs-real Patho-KID calibration;
- frozen production statistical-count plus epoch-29 ProbNet spatial policy;
- completed strict ProbNet P1/P2 geometry benchmark, with production geometry
  passing and learned spatial-structure superiority over uniform not shown;
- U1 tumor-burden trajectory and the paired U1/U2 representation analysis with
  production images, fresh UNI-2h/CONCH embeddings, WSI-held-out inference,
  and high-mask-overlap sensitivity.

Active critical path:

1. Complete final Segmentator model selection on real human tissue masks:
   image-only baseline, CellViT teacher, and optional runtime CellViT input.
2. Complete clinician review of the active 727-pair / 1,454-direction pool,
   including tissue correction, 150 blinded redraws, adjudication, and pair
   revalidation.
3. Freeze at least 500 accepted pairs / 1,000 directions plus the balanced
   300-direction sensitivity manifest.
4. Build one authoritative freeze manifest covering DoctorCond, AutoCond,
   nuclei conditions, checkpoints, seeds, thresholds, MPP, code revision, and
   sample hashes.
5. Integrate the complete two-attempt project agentic workflow into the formal
   paired generation runner; preserve attempt-1 and selected-final manifests.
6. Run generation consistency Track G1 on the accepted paired set, including
   the real-image Segmentator ceiling and tissue-nuclei coherence.
7. Run Track G2 first at 3 cases per valid cell; expand the project system to
   10 per cell only after smoke acceptance.
8. Run Track G3 on 60-120 cases with 4-5 frozen seeds.
9. Run DoctorCond Patho-KID as primary and AutoCond as the paired sensitivity
    analysis, using shared sample IDs and bootstrap draws.
10. Export the blinded doctor-review deck, import scores, and compile the final
    report with explicit fairness, coupling, failure, and limitation sections.

Do not start new ProbNet count/density heads, new mask-semantic model variants,
or a 30/50-per-cell generation expansion unless the frozen analyses above
identify a specific unresolved failure that changes the paper's main claim.
