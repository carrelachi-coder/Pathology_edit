# CellDistNet Frozen Spatial Sampling Plan

`CellDistNet` is the manuscript-facing model name. Existing source files,
configuration keys, checkpoint paths and benchmark artifacts retain `probnet`
as a legacy internal identifier so that released provenance remains stable.

## Production Decision

ProbNet does not determine the number or class composition of generated
nuclei. The production pipeline separates those responsibilities:

- the current patch and its matching reference-profile statistics determine
  total count and exact type quotas;
- the frozen ProbNet scalar field `P(nucleus) = 1 - P(background)` weights
  spatial landing positions;
- component allocation, instance-shape selection, and placement retries are
  deterministic sampling-policy responsibilities.

Count-head recalibration and density-head hyperparameter selection are retired.
They are not part of model selection, deployment, or the final 120-case
evaluation.

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

The checkpoint contains semantic, type, density, and count-related outputs,
but inference consumes only its scalar nucleus-presence probability for
location weighting. Checkpoint density integrals and type logits never set
the requested count or type mixture.

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

Two older wrapper families are not production-authoritative:

- `inpaint_cells/configs/generation_profiles.json` and
  `scripts/phase4_probnet_workflow*.sh` still contain pre-freeze
  profile-specific values for `prob_count_weight`, `density_scale`, gamma
  lists, and density-scale JSON files;
- `scripts/phase3_end_to_end_ui.py` still auto-fills a profile-specific
  `density_scale_{profile}.json` and currently falls back to gamma `1.0`.

Those JSON files are profile/organ-specific rather than BCSS-global, but they
are legacy calibration multipliers. Passing one multiplies the patch-adaptive
density scale after `compute_patch_adaptive_priors`, so these wrappers are
excluded from the frozen endpoint until their defaults are aligned. They must
not be used to claim that the policy in this document was executed.

## Frozen Sampling Policy

1. Widen locally thin edit branches to a 33-pixel minimum support width,
   preserve the semantic edit, and clip the support to biological foreground.
2. Delete a source nucleus as a complete component only when its centroid lies
   inside the generation support. Keep boundary-crossing nuclei whole when
   their centroids remain outside.
3. For each tissue type, measure patch-local density from source-nucleus
   centroids outside the full support.
4. Treat the local estimate as reliable when at least 20,000 unedited tissue
   pixels and 10 nuclei are observed. Use a reliable local density directly.
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

6. Use the reliable patch-local nucleus-type distribution for exact type
   quotas. If it is sparse, use the matching profile library's tissue-specific
   type distribution. If that distribution is unavailable, fall back first to
   sparse local types and only then to the explicit default class.
7. Convert type proportions to exact integer quotas with largest remainder.
   ProbNet type logits do not participate.
8. Allocate each tissue-level target across disconnected components strictly
   by component area with largest remainder. Do not reserve one nucleus for
   every small component; a component with expected count below one may receive
   zero.
9. Generate Poisson candidates independently in every positive-quota component
   and order/select locations with the frozen ProbNet scalar field using
   `gamma=1.5`.
10. Draw same-class instance shapes from the current patch first. Use the
    matching profile library only for same-class shortages. Continue through
    unused candidate centers until the quota is filled or the pool is
    exhausted.

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
retry_candidate_multiplier = 8
retry_candidate_floor = 32
max_nucleus_overlap_fraction = 0.0
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

The completed run, audits, fresh UNI-2h/CONCH caches, dose-response reports, and
combined feature-space figure are stored at:

```text
/data1/zhao/wqx/benchmarks/runs/
embedding_utility_bcss_paired257_frozen_sampling_v3_epoch29
```

This full rerun is the first utility result eligible for publication after the
sampling policy was frozen. Earlier epoch-1 utility banks remain historical
artifacts and must not be mixed with these embeddings.
