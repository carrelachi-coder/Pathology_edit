# Phase 3 Mask Editing Strategy Design

## Background

Phase 5 training has been moved to the server. During the long training run, Phase 3 should define the mask editing strategies for all six datasets:

- BCSS
- PANDA
- GlaS
- IGNITE
- PUMA
- ORCA

The existing `mask_edit/` directory contains BCSS-oriented prototype scripts. These are useful references for morphology, pathology priors, validation, rule planning, and cell-retention/cell-fill logic, but the new Phase 3 scripts should live in a new folder to keep multi-dataset editing cleanly separated from the BCSS prototype.

## Hard Constraint

Generic strategies must produce valid edits on all six datasets.

This means a generic strategy cannot depend on labels that are missing from some datasets, such as necrosis, immune infiltrate, vessel, DCIS, Gleason grade, or gland differentiation. Generic strategies should only rely on label concepts that every dataset can represent:

- background or non-tissue
- tumor
- some form of non-tumor tissue

Dataset-specific or label-specific edits should be implemented as specialized strategies instead.

## Existing Code To Reuse Conceptually

The current `mask_edit/` codebase has three useful parts:

- `mask_edit/mask_data_generate/`: BCSS mask transform prototypes such as tumor dilation, tumor shrink, lymphocyte infiltration, necrosis replacement, necrosis fibrosis, stromal fibrosis, and end-to-end mask pair generation.
- `mask_edit/Prior_knowledge_of_pathology/`: pathology prior JSON generation and adjacency/area/co-occurrence prior tooling.
- `mask_edit/rule_engine/`: semantic-diff planning, operation ordering, tissue edit execution, cell-only adjustment logic, and cell refill hooks.

The strongest reusable ideas are:

- SDF-based boundary dilation and shrink.
- Multi-scale smoothed noise for natural boundaries.
- Boundary influence weights to localize edits.
- Edge fade masks to avoid patch-border artifacts.
- Nearest-neighbor or configured-class backfill after shrink.
- Topology cleanup for small fragments.
- Validation after every attempted edit.
- Separation between tissue-level edits and cell-level retention/fill.
- Operation logs and metadata for downstream training.

## Recommended New Directory

Create a new folder:

```text
phase3_mask_edit/
  README.md
  cli/
    generate_edits.py
  core/
    labels.py
    mask_io.py
    morphology.py
    validators.py
    pipeline.py
  generic/
    tumor_burden.py
    tumor_boundary_remodel.py
    tumor_position_bias.py
  specialized/
    bcss.py
    panda.py
    glas.py
    ignite.py
    puma.py
    orca.py
  recipes/
    generic.yaml
    bcss.yaml
    panda.yaml
    glas.yaml
    ignite.yaml
    puma.yaml
    orca.yaml
  tools/
    preview_edits.py
    summarize_editability.py
```

## Architecture

Use a "generic operator + dataset adapter + specialized strategy" architecture.

The generic layer should read `dataset_config.DatasetConfig` and operate in the unified label space. It should not redefine label IDs or dataset semantics. It should ask the dataset config:

- which fine IDs are tumor
- which labels count as valid non-tumor targets
- which labels should be skipped
- which original labels can map back from unified labels
- whether cell annotations are available

The specialized layer can use dataset-specific fine labels:

- PANDA Gleason 3/4/5
- GlaS adenomatous/moderately differentiated/poorly differentiated glands
- BCSS DCIS and angioinvasion
- PUMA epidermis and vessel-aware melanoma patterns
- IGNITE immune, necrosis, vessel, and lung tumor microenvironment labels
- ORCA minimal tumor versus non-tumor labels

## Core Modules

`core/labels.py`

- Loads `DatasetConfig`.
- Builds tumor masks from `cfg.tumor_ids`.
- Builds editable non-tumor masks from all available labels minus tumor and skip labels.
- Chooses the best fallback target for regression: stroma if available, otherwise other tissue, otherwise the largest valid non-tumor class in the current mask.
- Provides helpers for coarse/fine conversion.

`core/mask_io.py`

- Reads RGB masks and ID masks.
- Converts between dataset original IDs and unified fine IDs.
- Writes `src_mask.png`, `tar_mask.png`, `change_region.png`, `edited_tissue.png`, and preview images.
- Records color maps and label metadata.

`core/morphology.py`

- Shared SDF computation.
- Multi-scale noise generation.
- Boundary influence weights.
- Edge fade masks.
- Dilation, shrink, local protrusion, and local erosion primitives.
- Nearest-neighbor and configured-class backfill.
- Fragment cleanup.

`core/validators.py`

- Checks label legality.
- Checks non-empty change region.
- Checks minimum and maximum changed area.
- Checks no background leakage.
- Checks the edited mask remains in the dataset-supported label set.
- Checks topology constraints such as minimum connected component size.

`core/pipeline.py`

- Runs one edit strategy against one mask.
- Handles retry with different seeds/parameters.
- Preserves cells outside the change region.
- Optionally fills or adjusts cells inside edited regions.
- Writes outputs and metadata.

## Generic Strategies

Because generic strategies must work for all six datasets, they should be limited to universal tumor/non-tumor operations.

### 1. `tumor_burden_increase`

Goal: increase tumor extent.

Mechanism:

- Build a combined tumor mask from `cfg.tumor_ids`.
- Use SDF + multi-scale noise to expand tumor into adjacent editable non-tumor tissue.
- Select the replacement tumor ID from the local tumor boundary. For fine datasets, this preserves the nearest fine tumor subtype rather than collapsing everything to coarse tumor.
- Never expand into background or skip labels.
- Validate that the tumor area increases and the change region is non-empty.

Why it is generic:

- Every dataset has tumor labels.
- Every dataset has at least one non-tumor tissue class.
- ORCA works by expanding carcinoma into non-carcinoma tissue.

### 2. `tumor_burden_decrease`

Goal: decrease tumor extent.

Mechanism:

- Shrink the combined tumor mask inward using SDF + noise.
- Released tumor pixels are backfilled.
- Preferred backfill order: stroma, other tissue, normal epithelium, largest adjacent non-tumor class.
- For ORCA, released carcinoma becomes non-carcinoma tissue.
- Validate that tumor area decreases and released pixels are not background unless the original neighboring context is background-only.

Why it is generic:

- It only requires tumor and some valid non-tumor target.
- It can be implemented using dataset config fallbacks.

### 3. `tumor_boundary_remodel`

Goal: alter tumor morphology without requiring a net tumor burden change.

Mechanism:

- Apply local tumor protrusions and local erosions in different boundary zones.
- Keep total tumor area approximately stable within a small tolerance.
- Use nearest boundary tumor subtype for protrusions.
- Use adjacent non-tumor backfill for erosions.
- Validate that the change region is non-empty and tumor area remains within the configured tolerance.

Why it is generic:

- It only requires tumor and non-tumor labels.
- It creates a meaningful edit even for coarse datasets like ORCA.
- It gives Phase 3 one universal strategy that is not merely "bigger" or "smaller".

## Reclassified Semi-Generic Strategies

The earlier draft included `microenvironment_shift` as a generic strategy. Under the hard constraint above, it should not be treated as generic because some datasets lack the required labels.

It should instead become a reusable semi-generic module that specialized dataset recipes may enable when labels exist:

- immune/TIL increase or decrease
- necrosis appear or fibrosis
- angiogenesis or vessel-proximal invasion
- stromal desmoplasia

The code can still be shared, but the recipe is not universal.

## Specialized Strategies

### BCSS

Candidate strategies:

- `dcis_invasion`
- `angioinvasion_emphasis`
- `stromal_desmoplasia`
- `TIL_increase`
- `TIL_decrease`
- `necrosis_appear`
- `necrosis_fibrosis`

BCSS has fine tumor subtypes and rich tissue labels, so it can support the most pathology-specific edits.

### PANDA

Candidate strategies:

- `gleason_upgrade_3to4`
- `gleason_upgrade_4to5`
- `gleason_downgrade_4to3`
- `benign_to_gleason3`
- `benign_atrophy`
- `tumor_volume_increase`
- `tumor_volume_decrease`

PANDA-specific edits should mostly be in-place fine-label transitions plus tumor burden changes.

### GlaS

Candidate strategies:

- `normal_to_adenomatous`
- `adenoma_to_carcinoma`
- `grade_upgrade`
- `treatment_dedifferentiation`
- `tumor_gland_growth`
- `tumor_gland_regression`

GlaS edits should respect gland structure as much as possible. In-place fine-label edits may be safer than large morphology changes for grade transitions.

### IGNITE

Candidate strategies:

- `tumor_invasion`
- `tumor_regression`
- `necrosis_appear`
- `TIL_increase`
- `TIL_decrease`
- `stromal_desmoplasia`
- `angiogenesis`

IGNITE has coarse but useful microenvironment labels, so it is a good target for semi-generic necrosis/immune/vessel modules.

### PUMA

Candidate strategies:

- `tumor_epidermal_invasion`
- `epidermis_ulceration`
- `tumor_regression`
- `necrosis_appear`
- `perivascular_invasion`

PUMA has melanoma-specific epidermis and vessel labels. Its immune class is absent, so immune edits should not be enabled.

### ORCA

Candidate strategies:

- `tumor_invasion`
- `tumor_regression`
- `tumor_boundary_remodel`

ORCA only has tumor versus non-tumor tissue, so specialized edits should remain minimal. It should not receive synthetic necrosis, immune, stroma, or vessel edits that cannot be represented by its labels.

## Output Format

Every edit should write the same output structure:

```text
src_mask.png
tar_mask.png
change_region.png
edited_tissue.png
metadata.json
ops_log.json
preview.png
```

`metadata.json` should include:

- dataset name
- source mask path
- strategy name
- random seed
- parameters
- original label counts
- edited label counts
- changed pixel count
- validation status
- whether cells were retained, regenerated, or absent

`ops_log.json` should include:

- operation order
- attempted parameters
- accepted or rejected status
- rejection reason
- area changes
- retry count

## Preview And Review Tools

Add a preview script:

```text
phase3_mask_edit/tools/preview_edits.py
```

It should generate contact sheets similar to the current `edit_plan/overlay_visualizations` assets:

- original image
- source mask
- target mask
- change region
- overlay
- side-by-side panel

Use this before large-scale generation. For each dataset and each strategy, sample about 10 masks and create a contact sheet for quick visual review.

## Implementation Order

1. Create `phase3_mask_edit/core`.
2. Port morphology primitives from BCSS prototype scripts into dataset-agnostic helpers.
3. Implement `generic/tumor_burden.py` with increase and decrease.
4. Implement `generic/tumor_boundary_remodel.py`.
5. Implement `tools/preview_edits.py`.
6. Run the three generic strategies on all six datasets and inspect contact sheets.
7. Add specialized strategies for PANDA, GlaS, and BCSS.
8. Add specialized strategies for IGNITE, PUMA, and ORCA.
9. Add recipes for default per-dataset generation.
10. Freeze output metadata format for Phase 5 consumption.

## Design Decision To Carry Forward

The key design decision is that "generic" means universally effective, not merely universally callable.

Therefore:

- Generic strategies are limited to tumor/non-tumor morphology.
- Label-specific edits are specialized, even when their code can be reused.
- Missing labels should not silently produce no-op edits under a generic recipe.
- Dataset recipes may use shared modules, but their strategy names should honestly reflect dataset-specific label availability.
