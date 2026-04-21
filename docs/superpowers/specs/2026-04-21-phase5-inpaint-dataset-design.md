# Phase 5 Inpaint Dataset Design

**Goal:** define a GT-supervised `local-preservation inpaint` dataset for Phase 5 that matches real tissue-edit region shapes without requiring real edited target images.

**Status:** approved design for first-pass implementation

## Problem

Phase 5 needs a dedicated `inpaint` training set that teaches the model two things at the same time:

- reconstruct missing content inside a local edit region
- preserve image content outside the edit region

We currently have real `gt_image`, `gt_tissue_mask`, and `gt_nuclei_mask`, but we do not have real post-edit target images for operations such as tissue replacement, expansion, or shrinkage. Because of that, the `inpaint` training target cannot be a semantically changed image. The dataset must instead stay GT-supervised while making the missing-region geometry look like real tissue edits.

## Non-Goals

- no attempt to supervise true edited-image appearance for Phase 3 edits
- no change to the existing `cross` dataset objective
- no generated nuclei mixing in the first dataset version
- no requirement to run the full Phase 3 editing pipeline before Phase 5 training

## Recommended Approach

Use a GT-preserving inpaint dataset:

- `target_image = gt_image`
- `target_tissue_mask = gt_tissue_mask`
- `target_nuclei_mask = gt_nuclei_mask`
- `change_region_mask = pseudo-edit region with realistic tissue-edit geometry`
- `erased_source_image = gt_image` with the change region erased

This keeps supervision internally consistent while still exposing the model to realistic local edit footprints.

Large semantic edits remain the responsibility of the `cross` model. The `inpaint` dataset should focus on small-to-medium local changes and strong outside-region preservation.

## Alternatives Considered

### 1. Random-hole inpainting

Pros:

- easiest to implement
- fully GT-supervised

Cons:

- edit regions do not resemble pathology tissue edits
- weak alignment with the eventual routing logic for `local-preservation`

Decision: rejected.

### 2. Use edited tissue masks directly as `inpaint` targets

Pros:

- best match to the intended deployment semantics

Cons:

- no real edited target image exists for supervision
- would create label conflict between changed target masks and unchanged GT image

Decision: rejected for first-pass training.

### 3. GT-preserving inpaint with pseudo-edit masks

Pros:

- supervision stays valid
- region shapes can be made close to real edit behavior
- matches current Phase 5 split between `inpaint` and `cross`

Cons:

- does not directly teach large semantic replacement

Decision: recommended.

## Data Definition

Each normalized inpaint record should continue to provide:

- `dataset`
- `sample_id`
- `case_id`
- `source_image`
- `erased_source_image`
- `target_image`
- `target_tissue_mask`
- `target_nuclei_mask`
- `change_region_mask`
- `prompt`
- `edit_type`
- `change_ratio`

The first implementation should add two explicit synthesis-trace fields:

- `mask_mode`
- `size_bucket`

Recommended meanings:

- `mask_mode`: one of `identity`, `near_identity`, `expand_band`, `shrink_band`, `replace_like_blob`
- `size_bucket`: one of `identity`, `small`, `medium`, `large`

These fields are metadata only. Training behavior does not change because of them in the first pass, but they support filtering, debugging, and metric breakdowns.

## Region Synthesis Rules

### Shared principles

All synthesized `change_region_mask` samples must satisfy:

- regions should be anchored to biologically meaningful tissue structure rather than random image positions
- regions should prefer editable tissues from `DatasetConfig`, especially `tumor_ids`, then stroma, gland, and necrosis-like tissues
- regions must avoid large overlap with `skip_tissues`
- regions should stay local and mostly contiguous
- regions should look like tissue-edit footprints, not free-floating holes

All synthesis happens on `gt_tissue_mask`. The synthesis step does not modify the target masks saved into metadata.

### Mode 1: `identity`

Definition:

- `change_region_mask` is all zeros
- `erased_source_image = source_image`
- `target_* = source_* = gt`

Purpose:

- teaches strict no-change behavior

### Mode 2: `near_identity`

Definition:

- non-zero but very small local change region
- `target_*` remains GT
- only `erased_source_image` is altered

Purpose:

- prevents the model from learning the shortcut that "no drift" only happens when the mask is exactly zero

### Mode 3: `expand_band`

Definition:

- choose a valid connected component from the tissue mask
- take a narrow exterior band around the component boundary
- clip the band to valid neighboring tissues that could plausibly be affected by expansion

Interpretation:

- mimics local outward growth or invasion

Constraints:

- the band should remain close to the chosen boundary
- avoid spanning across unrelated distant regions
- allow irregularity, but preserve locality

### Mode 4: `shrink_band`

Definition:

- choose a valid connected component
- erode inward from the component boundary
- use the removed inner band as the change region

Interpretation:

- mimics local regression or shrinkage

Constraints:

- do not collapse the full component
- preserve at least `30%-50%` of the original component area
- prefer components large enough to support meaningful inward erosion

### Mode 5: `replace_like_blob`

Definition:

- choose a sufficiently large connected component
- select a boundary segment or boundary seed points
- carve a blob from the boundary inward
- the carved region must stay attached to the original boundary

Interpretation:

- mimics local tissue replacement or rewriting

Constraints:

- never place the blob as an isolated hole in the center
- blob depth should stay within roughly `40%-60%` of the local thickness
- blob area should usually stay within `15%-35%` of the chosen component area
- avoid fragmenting the component into many small islands

## Size Buckets

Because real local edits often reach around `20%` of the patch, the inpaint dataset should use a less conservative range than a pure restoration setup.

Recommended buckets:

- `small`: `0.5%-10%`
- `medium`: `10%-22%`
- `large`: `22%-35%`

`large` samples are included for robustness, but should remain a minority because larger semantic change should usually route to `cross`.

## Sampling Distribution

Recommended global sampling mix:

- `identity`: `10%`
- `near_identity`: `10%`
- `expand_band`: `30%`
- `shrink_band`: `25%`
- `replace_like_blob`: `25%`

Recommended size distribution over non-identity samples:

- `small`: `35%`
- `medium`: `30%`
- `large`: `15%`

This preserves the `local-preservation` character of the dataset while still covering moderate local edits.

## Erasing Strategy

`erased_source_image` should be created from `gt_image` by masking the synthesized change region.

First-pass rule:

- fill erased pixels with a neutral constant value, matching the existing implementation style

Future extension, not part of this design:

- optional blurred or stain-aware erase fills

## Architecture Impact

No change is required to the current Phase 5 inpaint training interface:

- training still consumes `erased_source_image`
- training still consumes GT `target_tissue_mask`
- training still consumes GT `target_nuclei_mask`
- training still consumes `change_region_mask`
- supervision remains `target_image`

This keeps the current dataset loader and training path stable while improving how inpaint samples are synthesized.

## Proposed File Layout

Recommended responsibilities:

- `controlnet_train/data/inpaint.py`
  - keep normalized metadata schema and dataset loading
- `controlnet_train/data/inpaint_synthesis.py`
  - new pseudo-edit mask synthesis helpers
  - tissue-aware region selection
  - size-bucket targeting
  - erase-image materialization from GT
- `controlnet_train/cli/build_inpaint_dataset.py`
  - extend CLI so it can synthesize inpaint records directly from layered GT patch roots, not only normalize upstream edit jsonl files

This keeps synthesis logic separate from runtime dataset loading.

## Build Modes

The CLI should support two explicit input modes:

### Mode A: normalize existing upstream edit records

Current behavior:

- read user-provided jsonl
- normalize into `metadata_inpaint_{train,val}.jsonl`

### Mode B: synthesize GT-preserving inpaint records from layered patch roots

New behavior:

- read layered dataset roots
- sample GT patches
- synthesize `change_region_mask`
- materialize `erased_source_image`
- emit normalized inpaint metadata

The first pass should preserve Mode A and add Mode B rather than replacing existing behavior.

## Validation Rules

Each synthesized record should pass quality checks before being written:

- `change_ratio` is within the requested bucket
- change region overlaps valid foreground tissue
- change region has `1-3` connected components at most
- most changed pixels stay close to a tissue boundary for boundary-driven modes
- `replace_like_blob` is boundary-attached, not an interior hole
- `shrink_band` preserves the main source component
- overlap with `skip_tissues` stays below a configured threshold

Failed candidates should be re-sampled rather than silently emitted.

## Error Handling

The builder should fail fast with clear messages when:

- a dataset root is missing required layered files
- a sample has unsupported tissue or nuclei IDs
- a requested bucket cannot be satisfied after repeated attempts
- no valid connected component can be found for the requested mode

The builder should report per-mode acceptance and rejection statistics for debugging.

## Testing Strategy

The first implementation should add tests for:

- metadata rows include `mask_mode`, `size_bucket`, and `change_ratio`
- `identity` samples produce zero masks and unchanged erased images
- `near_identity` samples produce non-zero masks below the small upper bound
- `expand_band` masks lie near the exterior boundary of a component
- `shrink_band` masks lie inside the selected component boundary and preserve the component core
- `replace_like_blob` masks touch the boundary and do not form isolated center holes
- size buckets are respected within tolerance
- synthesized records still load through `InpaintDataset`

## Open Decisions Resolved In This Design

- use GT-preserving supervision for the first inpaint dataset version: yes
- model true edit geometry via pseudo-edit change masks: yes
- use real semantic target-mask edits in first inpaint training: no
- allow larger local regions up to about `35%`: yes
- treat edits around `20%` as still valid inpaint training samples: yes

## Future Extensions

Not part of this first pass:

- use Phase 3 edit operators directly to bias region-shape priors
- use generated nuclei inside edited regions during inpaint training
- use routing-aware curriculum shared with inference thresholds
- add dataset-specific mask-mode priors based on `available_edits`

## Success Criteria

The design is successful if:

- Phase 5 inpaint training can be built directly from existing GT layered data
- synthesized edit regions look meaningfully closer to real tissue edits than random holes
- metadata stays compatible with the current inpaint loader and training pipeline
- the resulting dataset strengthens outside-region preservation without requiring edited-image supervision
