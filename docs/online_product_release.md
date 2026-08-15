# Online pathology-edit product release

This is the human-readable source of truth for the online product aligned to
the approved G2 agentic implementation on 2026-07-31. The machine-readable companion is
`benchmark_configs/releases/online_agent_product_v1.json`. A runtime or UI
default that conflicts with that manifest is a bug.

## Authoritative Segmentator decision

As of 2026-08-03, the final production tissue evaluator remains
`segmentator-fine-legacy-anchor-v1`:

```text
/data1/zhao/wqx/segmentator_fine/legacy_anchor_fine_seed42/best_composite.pt
SHA256 5165e0fb20aca68f64fb06403eff846173fba3a3373deeedb36b3feb81727b3f
```

The online Agent, UI, formal image-edit run and evaluator replay must resolve
`benchmark_configs/releases/segmentator_fine_legacy_anchor.json`.  The C-line
checkpoints remain research comparisons and manual diagnostics; they are not
automatic fallbacks and cannot override the production release through an
implicit default.

This checkpoint was initialized from the historical coarse checkpoint
`/data1/zhao/wqx/segmentator/best_mIoU.pt` (SHA256
`5e4b5587359527e0e988428b8cd7fa453255de8729f0fde1ef375fb21d64f112`).
Its fine stage used `legacy_final_depth`, `freeze_shared_for_fine` and
`fine_only_loss`, so the shared/coarse parameters were not updated during that
stage. This provenance is intentional and must be reported rather than
describing the checkpoint as a C-line model. The old coarse training manifest
used a patch-level split and already included 545 of the 600 current G2 source
patches in training; its scores are therefore verifier-coupled engineering
evidence, not an independent paper endpoint.

The fixed `metadata_cross_val.json`/G2 cohort is evaluation metadata, not a
continued-training set. It contains grouped train, validation and test rows;
training on the complete file would leak the evaluator endpoint. Any
small-learning-rate continuation must therefore be a separately named
candidate, exclude frozen evaluation rows, and pass group-disjoint validation
plus a new clinician-labelled endpoint before it can replace this release.

## Production call chain

```text
phase3_end_to_end_ui.py
  -> tissue-mask edit
  -> target-nuclei synthesis
  -> run_agentic_edit_workflow.py
       -> route
       -> generate
       -> frozen legacy-anchor Segmentator + CellViT verification
       -> at most one alternate-backend recovery
       -> selected image and complete audit
```

The UI keeps the full four-stage front end. Only `agentic` is a production
generation mode; `dry-run` is a debugging mode.

## Implementation ownership

| Stage | Canonical implementation | Required behavior |
|---|---|---|
| UI input and mask editor | `scripts/phase3_end_to_end_ui.py` | Keeps prompt, direct instruction, multi-label manual contour/finalize, and auto-recommend. Source tissue auto-segmentation is release-driven; source nuclei auto-segmentation uses the frozen CellViT runtime. |
| Instruction parsing and planning | `phase3_mask_edit/parser/instruction_parser.py`, `phase3_mask_edit/rules/semantic_to_intent.py` | Multi-edit strengths remain clause-local; preservation clauses do not become edits; replacement/backfill remains a fallback. |
| Target nuclei construction | `scripts/run_phase3_inpaint_pipeline.py`, `inpaint_cells/generate.py`, `inpaint_cells/nuclei_library/library.py` | Count evidence always comes from the pre-edit source patch. ProbNet local type evidence is log-pooled with the target-tissue empirical prior and assigned with cumulative posterior balancing. Same-class component-local source shapes are preferred, with a calibrated library fallback. Generic edits delete and regenerate the semantic region while retaining buffer-only nuclei as obstacles. A GLaS structure edit uses the complete touched gland as the shared image, deletion, and nuclei-regeneration region so stale interior nuclei cannot survive a moved gland boundary. Each sampled mask is audited before image generation; the bounded feedback loop may correct spatial concentration once and otherwise changes only the seed. |
| Production orchestration | `scripts/run_agentic_edit_workflow.py` | This is the only online agent loop. The UI invokes it instead of maintaining a second generation/verification implementation. |
| Inpaint/Cross generation | `scripts/run_phase3_inpaint_pipeline.py`, `scripts/generate_cross_v1_no_ip_strict.py` | Runtime packages are validated before use. Cross retains the epoch-26 checkpoint's nuclei trust and identity adapter, applies the RGB/OD low-stain structure guard inside the generation region, and does not apply color matching. |
| Tissue verification | `segmentator/release.py`, `scripts/predict_segmentator_mask.py` | The frozen legacy-anchor release manifest reconstructs the architecture strictly; raw UI checkpoint/decoder selection is not supported. Evaluator policy `online-quality-evaluator-v2.4` first uses strict absolute semantic evidence and falls back to source-relative whole-region direction evidence when absolute source calibration fails. Cross may redraw stain, brightness, and texture over the full image. The evaluator estimates class-conditioned probability bias from stable unchanged-region anchors and removes that broad appearance effect before semantic direction and preservation checks; residual high-confidence class flips remain off-target drift, while insufficient stable anchors cause abstention. Boundary evidence also falls back from absolute BF1 to relative inner direction plus calibrated outer drift when source boundary accuracy is unreliable. Unsupported and Other labels never enter the score, and relative evidence does not claim target attainment. Absolute mode requires 0.80 evidence coverage. Relative mode requires 0.70, which still mandates relative semantic plus preservation and at least 0.15 independent nuclei or boundary evidence. |
| Nuclei verification | `scripts/run_cellvit_single_patch.py` | Frozen CellViT path, Python, SHA, 512-pixel input, and 0.25 MPP contract are shared by source preprocessing and online verification. |

The historical C-line training lineage is retained in
`scripts/train_segmentator_fine_boundary_v3_a800.sh`,
`scripts/train_segmentator_fine_cellvit_teacher_a800.sh`, and
`scripts/train_segmentator_fine_boundary_teacher_joint_a800.sh`. It remains a
manual safety comparison. The product evaluator uses
`segmentator_fine_legacy_anchor.json`, whose source-calibrated reliability
checks can abstain rather than misclassify evaluator error as generation
failure.

## Frozen components

| Part | Product selection | Runtime contract |
|---|---|---|
| Mask edit | semantic-diff schema 0.2, `gpt-4.1-mini`, `organic_v2` | Prompt, direct instruction, multi-label manual contour, and auto-recommend remain available. Independent instructions execute sequentially against the current mask. Replacement/backfill is a fallback, not an accidental second edit. |
| ProbNet / CellDistNet | `Qinxin11/pathology-probnet`, epoch 29 / step 33785, SHA256 `c29607f...571211` | G2 starts from `gamma=3`. Candidates are ordered by seeded Gumbel sampling from `odds(P(nucleus))^gamma`, without an added diversity or coverage-radius term. A failed spatial audit may change gamma once: over-concentration uses `gamma x 0.75`, under-following uses `gamma x 4/3`, clamped to `[1.5,5]`; the frozen evaluation distribution remains `gamma=3`. The remaining retry changes only the deterministic seed. Exact backfill inspects `128 x shortfall` candidates, clamped to `[512,4096]`, without relaxing count, type, shape, spacing, containment, or edit-region contracts. |
| Count | pre-edit patch adaptive | For every dataset and tissue, density evidence is measured from the source tissue/nuclei before editing. The edited target mask supplies only the target area. Reliable exact-target-tissue evidence is used directly; an absent GLaS target grade uses its grade-specific dataset prior multiplied by a source-patch gland-cellularity factor. |
| Nucleus shape | component-local reference-first | Unchanged target tissue uses same-class shapes from the corresponding source-patch connected component. A failed reference shape is quarantined for the current candidate while alternative references and library fallback are tried, then restored for later candidates; retry scaling cannot resize a reference shape. |
| Changed-tissue library shape | exact target tissue and type | A real tissue/grade transition samples the library by `(target tissue, density-head type)` without contaminated source-patch size calibration. An unchanged component-local shortage uses a library shape calibrated to that component's reference areas. |
| Placement integrity | full instance and one-pixel separation | Complete source instances intersecting the active deletion region are erased. For generic edits, buffer-only nuclei remain hard obstacles; for GLaS structure edits, the complete touched gland is intentionally cleared and regenerated. Generated nuclei use zero overlap, full biological-tissue containment, and a one-pixel spacing margin so same-class instances cannot merge and trigger false count backfill. |
| Inpaint | packaged `Qinxin11/pathology-inpaint-controlnet` | Packaged manifest commit and the released ControlNet weight size/SHA are checked before generation. |
| Cross V1 | packaged no-IP/no-UNI Cross release | Packaged manifest commit and the released ControlNet weight size/SHA are checked before generation. |
| Pix2pix | epoch 26 / step 214895, SHA256 `be5fe937...2ac` | `PATHOLOGY_PIX2PIX_CHECKPOINT` is mandatory. Full-pyramid local-histogram orientation steering, identity adapter, and `nuclei_reference_support_v2` come from the checkpoint. `cross_rgb_od_low_stain_v1` then preserves large low-OD, high-luma, low-chroma Cross components inside the generation region while excluding target nuclei. The guard records its mask, unprotected output and protected fraction in provenance; it uses no organ-specific rule and is not color matching. |
| Tissue evaluator | local frozen legacy anchor, SHA256 `5165e0f...27b3f` | Release `segmentator-fine-legacy-anchor-v1`, strict architecture reconstruction, image-only runtime. It is an engineering evaluator anchor, not an external clinical endpoint. |
| Nuclei evaluator | CellViT-SAM-H-x40-AMP-001 | SHA256 `356418f...c8ec4`, `512 x 512` at `0.25 MPP`. Root, model, and Python are centralized through `PATHOLOGY_CELLVIT_ROOT`, `PATHOLOGY_CELLVIT_MODEL`, and `PATHOLOGY_CELLVIT_PYTHON`. |
| Agent loop | canonical runner | The UI writes a per-run or per-patch `cell_fill_log.json`; the runner validates every nuclei attempt, reason code, bounded gamma update, frozen evaluation gamma, count, type routing, shape, erasure, buffer, spacing and checkpoint provenance before image generation. The nuclei loop allows at most three attempts and at most one parameter update. It also enforces `28` inference steps, guidance `3.5`, ControlNet scale `1.0`, `bf16`, seed `42`, routing thresholds `0.12/0.30`, the frozen legacy-anchor release and P1 `shadow`. Semantic routing normally uses the exact source/target tissue difference; a hash-locked joint-generation handoff instead makes its approved joint change and generation support authoritative so a nuclei-only edit cannot become a no-op. The subsequent image-generation loop still allows at most two attempts. |

Fresh online nuclei generation must satisfy the current v3 sampling audit and
feedback-loop contract. A prior audit policy can be consumed only as a frozen
replay artifact when its approval provenance, approved manifest entry, target
tissue mask, target nuclei mask, and cell-fill log all pass byte-level SHA-256
validation. This path cannot authorize a fresh or modified v2 sample and does
not change the online v3 default.

### Evaluator v2.1 region contract

The Agent and UI use two deliberately different regions. `semantic_change_region`
is exactly `source_tissue != target_tissue`; it remains the sole support for
semantic accuracy, transition and boundary metrics. `generation_change_region`
is a semantic superset used for full-structure regeneration and blending. It is
excluded only from the preservation set `U_far`. The Agent validates
`preservation_exclusion_region=full_generation_change_region` against the
machine-readable product release before loading a generation model. Drift in
the remaining `U_far` is still rejected above `0.08`.

An already rendered cohort does not need image generation again. After the
cohort is complete and frozen, `scripts/replay_product_quality_evaluator.py`
can recompute evaluator evidence and candidate selection from the stored
Segmentator probabilities and CellViT artifacts. The default mode is read-only
with respect to canonical run outputs. `--apply` requires an exact
`--expected-count`, completes and hashes a full-cohort preflight first, then
updates evaluator JSON, reports and selected-image pointers without invoking a
generation model. Do not apply a replay while image generation is still
writing the cohort.

### Nuclei sampling feedback loop

The online UI does not expose feedback controls. It reads the following
release-pinned contract and passes it through the canonical Phase 3 pipeline:

1. Generate with the frozen initial `gamma=3` and seed.
2. Require exact total count, patch-relative type agreement, valid raster
   instance count, boundary-distribution agreement and probability-mass
   coverage.
3. If the spatial audit fails, use a tie-aware probability-mass CDF to identify
   only the direction of the error. `PROBNET_OVERCONCENTRATED` permits one
   gamma decrease; `PROBNET_UNDERFOLLOW` permits one gamma increase. A flat or
   tied field does not manufacture a gamma direction.
4. Resample once more with a new seed if necessary. No fourth attempt and no
   second gamma update are allowed. If none of the three attempts passes, stop
   before H&E generation.

The loop never changes `target_count`, tissue or connected-component
allocation, deletion/generation regions, shape-source policy, or the one-pixel
nucleus spacing margin. It records all attempts and structured failure reasons
in `cell_fill_log.json`; `run_agentic_edit_workflow.py` rejects missing,
over-budget or reason-inconsistent traces.

The completed nuclei stage of
`/data1/zhao/wqx/benchmark_v1/g2_600_run_20260802_v2` is the attempt-budget and
regression evidence: 595/600 cases passed on attempt 1, 3 on attempt 2 and 2 on
attempt 3; no case required more than three attempts. These accepted generated
cases are not used to fit biological acceptance thresholds. The v3 directional
diagnostic passed an end-to-end replay of the prior three-attempt GLaS stress
case. A full 600-case v3 replay is not required: it would duplicate the 595
unchanged first-attempt cases and compete with the active image-generation
stage. The existing 600-case result therefore supports the attempt budget,
while the targeted replay validates the new feedback trace and Agent contract.

## Required production environment

```bash
export PATHOLOGY_INPAINT_CHECKPOINT=/models/pathology/pathology-inpaint-controlnet
export PATHOLOGY_CROSS_V1_CHECKPOINT=/models/pathology/pathology-cross-v1-pix2pix/cross_v1
export PATHOLOGY_PIX2PIX_CHECKPOINT=/models/pathology/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt
export PATHOLOGY_PROBNET_CHECKPOINT=/models/pathology/pathology-probnet/best_epoch29_c29607f1b609accb.pt
export PATHOLOGY_SEGMENTATOR_CHECKPOINT=/models/pathology/pathology-segmentator/legacy_anchor_fine_seed42_best_composite.pt
export PATHOLOGY_SEGMENTATOR_PYTHON=/home/lyw/anaconda3/envs/pathology-segmentator-mmseg/bin/python3.10
export PATHOLOGY_CELLVIT_ROOT=/home/lyw/wqx-DL/flow-edit/FlowEdit-main/CellViT-plus-plus-main/CellViT-plus-plus-main
export PATHOLOGY_CELLVIT_MODEL=$PATHOLOGY_CELLVIT_ROOT/checkpoints/CellViT-SAM-H-x40-AMP-001.pth
export PATHOLOGY_CELLVIT_PYTHON=/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python
```

Launch:

```bash
python scripts/phase3_end_to_end_ui.py
```

The machine-readable release pins the immutable ProbNet Hub checkpoint commit
`add6970449cf3a94997375a665c832e91b188251`. The production Segmentator is the
local legacy-anchor release pinned by SHA256 `5165e0f...27b3f`; the historical
C-line Hub revision is not a production selector. The default local paths
point at the packaged server releases. Environment variables are the supported
deployment override; editing source constants on a production machine is not.

## Explicitly retired

- pix2pix epoch 25 and use of `pilot_step001000.pt` as a runtime selector;
- disabling the epoch-26 checkpoint's high-resolution nuclei trust policy;
- LAB/histogram color matching after Cross;
- treating the Cross RGB/OD low-stain structure guard as a global color or
  stain-normalization postprocess;
- raw Segmentator checkpoint/decoder selection in the UI;
- `gpt-4o-all` as the default mask-edit model;
- Windows placeholder CellViT paths;
- library-first nucleus shape sampling;
- ProbNet checkpoint total-count output as a production count source;
- using the density head to overwrite unchanged-tissue type composition;
- estimating density from the edited target mask or excluding the deletion
  core from the pre-edit density reference;
- immediately recycling a failed reference shape inside the same candidate,
  consuming it permanently, or shrinking it during retries;
- unbounded exact-count backfill or relaxing count/type/shape constraints after
  a deterministic candidate budget is exhausted;
- using the superseded quality-plus-diversity prefix, coverage radius, or an
  unqualified stable-descending ProbNet queue as the primary quota prefix;
- organ-, tissue-, or dataset-specific hard placement bands that override the
  learned ProbNet ordering.

The GlaS-only boundary-center placement draft, CellViT-runtime-input
Segmentator ablation, multi-primitive paper benchmark draft, and temporary
online canary/deck scripts are not product components and are intentionally
absent from the release.

The online acceptance thresholds remain engineering thresholds pending blinded
G2 calibration. The product may report `engineering_pass_uncalibrated`; it must
not report formal clinical or publication validation from the online loop.
