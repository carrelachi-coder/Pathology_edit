# CellDistNet Frozen Spatial Sampling Plan

`CellDistNet` is the manuscript-facing model name. Existing source files,
configuration keys, checkpoint paths and benchmark artifacts retain `probnet`
as a legacy internal identifier so that released provenance remains stable.

## Production Decision

The production pipeline separates count, type, location and shape:

- total count is estimated from the unedited source patch; the edited target
  tissue mask supplies only the area to populate;
- the frozen ProbNet scalar field `P(nucleus) = 1 - P(background)` supplies
  spatial quality;
- the density head supplies exact type-quota evidence only for tissue/grade
  IDs genuinely created by the edit;
- unchanged target tissue preserves its pre-edit patch type mixture;
- component allocation, instance-shape selection, and placement retries remain
  explicit sampling-policy responsibilities.

The checkpoint count head remains non-authoritative. The density head is not a
total-count source.

## Frozen Checkpoint

Use the immutable C3 epoch-29 checkpoint:

```text
/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/
best_epoch29_c29607f1b609accb.pt
```

File SHA256:

```text
c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211
```

The checkpoint contains semantic, type, density, and count-related outputs.
Inference consumes the scalar nucleus-presence probability for spatial
placement and the normalized density-head class evidence for changed-tissue
type quotas. It never uses the checkpoint to set total count.

## Authoritative Code Contract

The contract is implemented in:

- `inpaint_cells/generate.py`;
- `inpaint_cells/sampling_policy.py`;
- `inpaint_cells/nuclei_library/library.py`;
- `scripts/run_phase3_inpaint_pipeline.py`;
- `scripts/run_phase3_manifest_pipeline.py`;
- `scripts/regenerate_embedding_utility_visual_candidate_nuclei.py`.

The relevant source-level entry points are
`compute_patch_adaptive_priors`, `allocate_area_proportional_counts`, and
`allocate_type_counts` in `inpaint_cells/generate.py`, plus
`NucleiLibrary.get_density` and `NucleiLibrary.get_type_distribution`.

`inpaint_cells/generate.py` resolves the dataset configuration with
`get_config(args.dataset)` and opens the caller-supplied dataset-specific
nuclei library. Production callers resolve that library from the current
`reference_profile`, normally as `nuclei_library/{profile}`. Its
`statistics.json` therefore represents the corresponding dataset/profile and
its organ or cancer type. A BCSS library must not be used as a universal prior
for PANDA, GlaS, IGNITE, ORCA, PUMA, or another organ profile.

`scripts/prepare_embedding_utility_nuclei.py` has a BCSS default because that
script serves the explicitly BCSS-only embedding-utility cohort. This default
is not the generic production fallback and must not be copied into multi-organ
execution.

## Wrapper Drift Found by Code Audit

The frozen policy above matches the defaults in `inpaint_cells/generate.py`,
the current `run_phase3_inpaint_pipeline.py` default, and the explicit
epoch-29 visual/final endpoint invocation.

The older shell/profile wrapper family is not production-authoritative:

- `inpaint_cells/configs/generation_profiles.json` and
  `scripts/phase4_probnet_workflow*.sh` still contain pre-freeze
  profile-specific values for `prob_count_weight`, `density_scale`, gamma
  lists, and density-scale JSON files.

Those JSON files are profile/organ-specific rather than BCSS-global, but they
are legacy calibration multipliers. Passing one multiplies the patch-adaptive
density scale after `compute_patch_adaptive_priors`, so those shell wrappers
are excluded from the frozen endpoint until their defaults are aligned. They
must not be used to claim that the policy in this document was executed.

The online product entrypoint `scripts/phase3_end_to_end_ui.py` is aligned with
the frozen endpoint: it pins the epoch-29 checkpoint by SHA256, uses gamma
`1.5`, leaves `density_scale_json` unset, and delegates agentic generation and
verification to `scripts/run_agentic_edit_workflow.py`.

## Frozen Sampling Policy

1. Preserve the exact semantic edit. For GLaS, any touched gland/tumor
   connected component becomes the complete destructive core.
2. Delete every complete source-nucleus instance that intersects the
   destructive core. Estimate the largest valid deleted equivalent diameter
   and set the generation support to the core plus a `1.5x` diameter buffer.
   Nuclei that only touch the buffer are retained as hard placement obstacles.
3. For every dataset and target tissue, measure patch-local density from the
   full pre-edit source tissue and source-nucleus centroids. Never remove the
   destructive core from this density reference.
4. Treat the exact target-tissue estimate as reliable when at least 20,000
   source pixels and 10 nuclei are observed. Use it directly.
5. For a sparse observation, shrink the local estimate toward the matching
   reference-profile tissue-density prior using
   `observed_tissue_area / full_patch_area` as confidence:

```text
reliable_t = (observed_area_t >= 20,000) and (local_count_t >= 10)
confidence_t = 1                                      if reliable_t
               observed_area_t / full_patch_area      otherwise
target_density_t =
    confidence_t * patch_local_density_t
    + (1 - confidence_t) * profile_tissue_density_t
```

   If a GLaS target grade is absent, retain the target grade's dataset prior and
   multiply it by a gland-family cellularity factor measured from the pre-edit
   patch.
6. For unchanged target tissue, use the pre-edit patch type distribution for
   exact quotas. For a genuinely changed target tissue/grade, normalize the
   density-head class evidence and use it for exact quotas; fall back to the
   matching profile prior only when head evidence is unavailable.
7. Convert type proportions to exact integer quotas with largest remainder.
   Per-center semantic type probabilities assign those exact quotas spatially;
   they do not change the quota totals.
8. Allocate each tissue-level target across disconnected components strictly
   by component area with largest remainder. Do not reserve one nucleus for
   every small component; a component with expected count below one may receive
   zero.
9. Generate candidates independently in every positive-quota component. Build
   the primary prefix greedily from
   `gamma * logit(P(nucleus)) + min(nearest_distance / radius, 1)`, where
   `radius = min(0.75 * sqrt(component_area / component_quota), 48)`.
   Preserve every unused candidate in a complete stable-descending ProbNet
   quality retry tail. Use `gamma=1.5`.
10. For unchanged tissue, draw same-class shapes from the corresponding
    source-patch component. Failed reference shapes return to the pool and are
    never resized during retries. A component-local shortage uses a library
    shape calibrated to that component's reference sizes. A real target
    tissue/grade transition uses exact `(target tissue, predicted type)`
    library shapes without source-patch size calibration.
11. Require zero overlap, full biological-tissue containment and one empty
    pixel between nuclei so two same-class placements remain separate
    instances.

Retry pools contain at least:

```text
max(32, 8 * component_quota)
```

Sparse pools are supplemented with alternative valid pixels from the same
component. Complete generated shapes may extend across the support boundary.

## Fixed Runtime Values

```text
prob_count_weight = 0.0
density_scale = 1.0
density_scale_json = none
gamma = 1.5
local_density_direct_min_area = 20,000
local_density_direct_min_count = 10
minimum_mask_width = 33
component_quota_policy = area_largest_remainder
backfill_failed_placements = true
retry_candidate_multiplier = 12
retry_candidate_floor = 64
dense_retry_quota_threshold = 20
dense_retry_occupancy_threshold = 0.12
dense_retry_candidate_multiplier = 24
dense_retry_candidate_floor = 128
quota_coverage_spacing_scale = 0.75
quota_coverage_max_radius = 48
retry_tail_policy = stable_descending_probnet_score
max_nucleus_overlap_fraction = 0.0
nucleus_spacing_margin_px = 1
type_density_head_weight_for_changed_tissue = 1.0
```

Strict zero overlap is part of the production contract. A proposed nucleus
that touches any retained or newly placed nucleus is rejected and the sampler
continues through its retry pool; accepted placements may not overwrite even a
single retained-source pixel.

## Evaluation

The frozen 9,950-source count-head audit remains a diagnostic explaining why
checkpoint count prediction is not authoritative. It must not select a
checkpoint or revive count-head training.

The product evaluation uses the frozen statistical count policy and reports:

- requested, placed, and unfilled nuclei;
- exact per-tissue, per-type, and per-component quota agreement;
- placement completion, with a minimum acceptable rate of 98%;
- complete-component retained-nucleus preservation;
- CellViT++ generated count error and signed bias;
- generated-versus-target type-proportion JSD;
- nearest-neighbour-distance Wasserstein error;
- same-class instance-area W1 and reference/library fallback rates.

The final 120-case cohort remains fixed at six datasets by four count bins by
five cases. It evaluates the immutable epoch-29 spatial field with this frozen
sampling policy. No density-head-only candidate is eligible for that endpoint.

## Verified Smoke

The formal entry-point smoke for case `048f9a87` is stored at:

```text
/data1/zhao/wqx/benchmarks/runs/
embedding_utility_sampling_policy_v3_formal_entry_smoke_048f9a87
```

It widened support from 53,987 to 97,167 pixels and placed all 40 requested
nuclei with exact tissue, type, and component quotas.

## Full Embedding-Utility Rerun

The frozen policy was subsequently applied to the full paired
moderate--significant BCSS utility cohort:

```text
/data1/zhao/wqx/benchmarks/data/
embedding_utility_bcss_paired257_frozen_sampling_v3_epoch29
```

The cohort contains 257 pairs from 99 WSIs. Moderate placed 9,047 of 9,053
requested nuclei and significant placed 5,968 of 5,972, corresponding to
`99.9337%` and `99.9330%` completion. The ten unfilled instances were recorded
explicitly as component-level spatial exhaustion; no fallback silently changed
the target density or type quota.

Both local synthesis and reference-guided global synthesis were regenerated at
both strengths (`1,028/1,028` H&E images, zero generation failures). The full
image audit verified that every output points to this nuclei bank and uses
`inpaint_change_region.png` as the common generation support. The global path
also verified Cross V1 without UNI or IP-Adapter and the orientation-adjusted
full-pyramid pix2pix checkpoint at epoch 26 / step 214895.
For reproduction, this pix2pix artifact must be selected explicitly through
`PATHOLOGY_PIX2PIX_CHECKPOINT`; historical filenames and default code paths are
not authoritative production selectors.

The completed run, audits, fresh UNI-2h/CONCH caches, dose-response reports, and
combined feature-space figure are stored at:

```text
/data1/zhao/wqx/benchmarks/runs/
embedding_utility_bcss_paired257_frozen_sampling_v3_epoch29
```

This full rerun predates the quota-aware coverage prefix introduced after the
18-case canary exposed collapse into a narrow high-score boundary band. It
remains useful as historical pipeline/completion evidence, but it is no longer
eligible as the final current-policy publication result and must not be mixed
with a rerun produced by the new policy.

## 18-Case Canary Correction

The earlier 18-case canary cell-spatial-quality conclusion is invalid. In the
d177 BCSS case, the superseded global-descending prefix preserved the requested
count but concentrated most generated instances near the changed-region
boundary. A same-case engineering regression with the generic quota-aware
coverage prefix preserved count and type quotas, produced independent
instances, and removed the visible boundary collapse.

This is implementation-regression evidence only, not a formal quality result.
No formal cell-quality conclusion should be reported until the fixed cohort is
rerun under code commit
`64de2f5d6d6637053bdb1ea050493fc6fbbfb27b`.
