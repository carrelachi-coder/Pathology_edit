# Phase 5.4 Edit Pipeline Design

Date: 2026-04-20

## Goal

Build a unified inference entrypoint for Phase 5 that accepts the current pathology state plus the desired target masks, automatically chooses `inpaint` or `cross`, and produces a final edited image.

This design is intentionally scoped to the first production-ready baseline:

- one unified CLI entrypoint
- automatic routing between `inpaint` and `cross`
- explicit external inputs only
- no internal reference retrieval
- no blending in V1
- no `cross V1` morphology branch

## Confirmed Inputs

The first version of the pipeline accepts exactly these core inputs:

- `reference_image`
- `reference_tissue_mask`
- `reference_nuclei_mask`
- `target_tissue_mask`
- `target_nuclei_mask`

Key semantic constraint:

- `reference_image` is the original image and also the current reference state
- there is no separate `source_image`
- `change_region_mask` is not user-provided in V1
- `change_region_mask` is computed internally from tissue-mask differences

This matches the intended editing setup:

- the user starts from the current image and current masks
- the user supplies the desired target tissue and nuclei states
- nuclei changes are assumed to be a subset of the tissue edit region

## High-Level Behavior

The unified pipeline performs three steps:

1. Compute `change_region_mask` and `change_ratio` from `reference_tissue_mask` and `target_tissue_mask`
2. Route to `inpaint` or `cross`
3. Run the selected model and save the final generated image

The first version does not blend generated output back into the reference image.

Rationale:

- blending can hide model quality problems
- pathology texture continuity is sensitive, and naive alpha blending may create artifacts
- the first goal is to verify that routing and raw model output are correct

## Recommended Code Layout

Inference should live in a dedicated package rather than being mixed into `cli/` or `training/`.

### New files

- `controlnet_train/inference/router.py`
- `controlnet_train/inference/pipeline.py`
- `controlnet_train/cli/edit_pipeline.py`

### Optional future file

- `controlnet_train/inference/blending.py`

This file should not be implemented in the first version unless debugging later shows that post-processing is necessary.

## Module Responsibilities

### `controlnet_train/inference/router.py`

Responsibilities:

- compute `change_region_mask`
- compute `change_ratio`
- choose `selected_mode`
- return debug metadata about the routing decision

Inputs:

- `reference_tissue_mask`
- `target_tissue_mask`
- optional threshold config

Outputs:

- `change_region_mask`
- `change_ratio`
- `selected_mode`
- `changed_tissue_ids_from`
- `changed_tissue_ids_to`

### `controlnet_train/inference/pipeline.py`

Responsibilities:

- load inference bundles
- invoke router
- run either `inpaint` or `cross`
- save result image and run summary

It should expose one public orchestration function, for example:

`run_edit_pipeline(...)`

This function should return a structured result object containing at least:

- `image`
- `selected_mode`
- `change_region_mask`
- `change_ratio`

### `controlnet_train/cli/edit_pipeline.py`

Responsibilities:

- parse CLI arguments
- call `run_edit_pipeline(...)`
- save final outputs to disk

It should not contain routing logic or model-loading logic directly.

## Routing Design

### Change-region computation

V1 defines:

`change_region_mask = (reference_tissue_mask != target_tissue_mask)`

This is the primary edit region.

Nuclei differences are not used to define the main region because the intended data invariant is:

- nuclei edits should remain inside the tissue edit region

### Change-ratio computation

`change_ratio = changed_pixels / total_pixels`

### Default thresholds

Recommended V1 thresholds:

- `t_inpaint = 0.12`
- `t_cross = 0.30`

Decision rule:

- `change_ratio <= 0.12` -> `inpaint`
- `change_ratio >= 0.30` -> `cross`
- `0.12 < change_ratio < 0.30` -> `inpaint`

This is intentionally conservative.

Rationale:

- `inpaint` is better aligned with local preservation
- `cross` should be reserved for more substantial structural changes
- the ambiguous middle region should default to the less disruptive path first

### Deferred routing features

The first version does not include:

- shape-complexity routing
- nuclei-based routing
- dataset-specific thresholds

These can be added later once real inference behavior is observed.

## Inference Bundles

The pipeline should load two explicit bundles rather than a single mixed object.

### Inpaint bundle

Contains:

- FLUX base inference components
- `ControlNet-Inpaint` checkpoint
- Phase 5 condition modules:
  - `HierarchicalTissueEmbedding`
  - `TissueConditionDownsampler`
  - `NucleiConditionEncoder`
  - `ChangeMaskEncoder`

### Cross bundle

Contains:

- FLUX base inference components
- `ControlNet-Cross` checkpoint
- Phase 5 condition modules:
  - `HierarchicalTissueEmbedding`
  - `TissueConditionDownsampler`
  - `NucleiConditionEncoder`

The first version should load them separately, even if that duplicates some components in memory.

Rationale:

- avoids coupling the two inference modes too early
- prevents config/channel mismatches
- keeps the implementation easier to debug

## Per-Mode Responsibilities

### `_run_inpaint(...)`

Inputs:

- `reference_image`
- `target_tissue_mask`
- `target_nuclei_mask`
- internally computed `change_region_mask`

Responsibilities:

- build `erased/reference image latent`
- encode target tissue condition via HTE + downsampler
- encode target nuclei condition
- encode change mask
- construct inpaint `controlnet_cond`
- run inpaint checkpoint
- return a full generated image

### `_run_cross_v0(...)`

Inputs:

- `reference_image`
- `reference_tissue_mask`
- `reference_nuclei_mask`
- `target_tissue_mask`
- `target_nuclei_mask`

Responsibilities:

- encode reference image latent
- encode reference tissue/nuclei conditions
- encode target tissue/nuclei conditions
- construct `cross V0` `controlnet_cond`
- run cross checkpoint
- return a full generated image

The first version only supports `cross V0`.

## CLI Contract

### Required arguments

- `--reference-image`
- `--reference-tissue-mask`
- `--reference-nuclei-mask`
- `--target-tissue-mask`
- `--target-nuclei-mask`
- `--pretrained-model-name-or-path`
- `--inpaint-checkpoint`
- `--cross-checkpoint`
- `--output-dir`

### Optional arguments

- `--force-mode inpaint|cross`
- `--save-debug-artifacts`
- `--device`
- `--seed`

### Default outputs

Always write:

- `final.png`
- `change_region_mask.png`
- `run_summary.json`

If `--save-debug-artifacts` is enabled, also write:

- `reference_tissue_mask.png`
- `target_tissue_mask.png`
- `reference_nuclei_mask.png`
- `target_nuclei_mask.png`

## Error Handling

The pipeline should fail early on inconsistent inputs.

### Required validation

- all five core input files exist
- image and mask spatial sizes match
- reference and target tissue masks have compatible label ranges
- reference and target nuclei masks have valid nuclei IDs
- required checkpoints exist
- router thresholds are valid (`t_inpaint <= t_cross`)

### Explicit first-version behavior

If inputs are inconsistent, raise clear user-facing errors rather than trying to auto-correct.

Examples:

- mismatched image and mask size
- unsupported nuclei ID
- missing checkpoint
- unsupported forced mode

## Testing Strategy

### 1. Router unit tests

Verify:

- `change_region_mask` is computed correctly
- `change_ratio` is correct
- `selected_mode` matches threshold rules
- `changed_tissue_ids_from/to` are returned correctly

### 2. Pipeline unit tests

Use stub bundles rather than real FLUX inference.

Verify:

- `inpaint` route calls `_run_inpaint(...)`
- `cross` route calls `_run_cross_v0(...)`
- result object contains expected fields
- debug summary is written correctly

### 3. CLI unit tests

Verify:

- argument parsing for `edit_pipeline.py`
- `--force-mode` behavior
- `--save-debug-artifacts` behavior

### 4. Later smoke tests

After implementation, run one real sample per mode:

- one small-change sample routed to `inpaint`
- one large-change sample routed to `cross`

The first smoke-test goal is behavioral correctness, not final visual quality benchmarking.

## Out of Scope for V1

The following are explicitly not part of the first Phase 5.4 implementation:

- blending
- seam refinement
- internal reference retrieval
- cross V1 morphology branch
- dataset-specific route thresholds
- nuclei-driven route decisions
- multiple candidate sampling and reranking

## Recommendation

Proceed with the modular design:

- `inference/router.py`
- `inference/pipeline.py`
- `cli/edit_pipeline.py`

Keep the first version strict and simple:

- tissue-diff-based routing
- no blending
- explicit external reference inputs
- full-image output from the selected model

This gives the cleanest path to implementation while preserving the ability to add blending or more advanced routing later without rewriting the inference entrypoint.
