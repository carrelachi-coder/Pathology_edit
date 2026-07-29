# Online pathology-edit product release

This is the human-readable source of truth for the online product assembled on
2026-07-29. The machine-readable companion is
`benchmark_configs/releases/online_agent_product_v1.json`. A runtime or UI
default that conflicts with that manifest is a bug.

## Production call chain

```text
phase3_end_to_end_ui.py
  -> tissue-mask edit
  -> target-nuclei synthesis
  -> run_agentic_edit_workflow.py
       -> route
       -> generate
       -> C-line Segmentator + CellViT verification
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
| Target nuclei construction | `scripts/run_phase3_inpaint_pipeline.py`, `inpaint_cells/generate.py`, `inpaint_cells/nuclei_library/library.py` | The UI passes source tissue, target tissue, source nuclei shapes, and the real edit region through the current six-argument interface. |
| Production orchestration | `scripts/run_agentic_edit_workflow.py` | This is the only online agent loop. The UI invokes it instead of maintaining a second generation/verification implementation. |
| Inpaint/Cross generation | `scripts/run_phase3_inpaint_pipeline.py`, `scripts/generate_cross_v1_no_ip_strict.py` | Runtime packages are validated before use. Cross retains the epoch-26 checkpoint's nuclei trust and identity adapter, applies the RGB/OD low-stain structure guard inside the generation region, and does not apply color matching. |
| Tissue verification | `segmentator/release.py`, `scripts/predict_segmentator_mask.py` | C-line release manifest reconstructs the architecture strictly; raw UI checkpoint/decoder selection is not supported. |
| Nuclei verification | `scripts/run_cellvit_single_patch.py` | Frozen CellViT path, Python, SHA, 512-pixel input, and 0.25 MPP contract are shared by source preprocessing and online verification. |

The exact C-line training lineage is retained in
`scripts/train_segmentator_fine_boundary_v3_a800.sh`,
`scripts/train_segmentator_fine_cellvit_teacher_a800.sh`, and
`scripts/train_segmentator_fine_boundary_teacher_joint_a800.sh`. The final
release is the joint epoch-2 checkpoint, not either initializer.

## Frozen components

| Part | Product selection | Runtime contract |
|---|---|---|
| Mask edit | semantic-diff schema 0.2, `gpt-4.1-mini`, `organic_v2` | Prompt, direct instruction, multi-label manual contour, and auto-recommend remain available. Independent instructions execute sequentially against the current mask. Replacement/backfill is a fallback, not an accidental second edit. |
| ProbNet / CellDistNet | `Qinxin11/pathology-probnet`, epoch 29 / step 33785, SHA256 `c29607f...571211` | ProbNet supplies only the spatial landing order. The complete legal retry queue is consumed by stable descending ProbNet score; a lower-score point is tried only after a higher-score placement fails. Patch-adaptive policy supplies counts and exact type quotas. Gamma is `1.5`; no density-scale JSON is used. |
| Nucleus shape | reference-first | Same-class shapes from the current source patch are sampled without replacement before the library is used. |
| Library fallback size | patch-calibrated | A library nucleus is resized to an empirical same-class area from the current patch. Linear scale is clamped to `0.5-2.0` with log-area jitter `0.05`. If no same-class reference exists, it remains unscaled and the diagnostic records that fact. |
| Inpaint | packaged `Qinxin11/pathology-inpaint-controlnet` | Packaged manifest commit and the released ControlNet weight size/SHA are checked before generation. |
| Cross V1 | packaged no-IP/no-UNI Cross release | Packaged manifest commit and the released ControlNet weight size/SHA are checked before generation. |
| Pix2pix | epoch 26 / step 214895, SHA256 `be5fe937...2ac` | `PATHOLOGY_PIX2PIX_CHECKPOINT` is mandatory. Full-pyramid local-histogram orientation steering, identity adapter, and `nuclei_reference_support_v2` come from the checkpoint. `cross_rgb_od_low_stain_v1` then preserves large low-OD, high-luma, low-chroma Cross components inside the generation region while excluding target nuclei. The guard records its mask, unprotected output and protected fraction in provenance; it uses no organ-specific rule and is not color matching. |
| Tissue evaluator | `Qinxin11/pathology-segmentator`, C-line joint epoch 2, SHA256 `ddb95a7...604c34` | Release `segmentator-fine-c-joint-epoch2-v1`, strict architecture reconstruction, image-only runtime. |
| Nuclei evaluator | CellViT-SAM-H-x40-AMP-001 | SHA256 `356418f...c8ec4`, `512 x 512` at `0.25 MPP`. Root, model, and Python are centralized through `PATHOLOGY_CELLVIT_ROOT`, `PATHOLOGY_CELLVIT_MODEL`, and `PATHOLOGY_CELLVIT_PYTHON`. |
| Agent loop | canonical runner | Semantic routing uses the exact source/target tissue difference. A wider generation support may be used for glands or thin regions. Source-image predictions are the preservation reference. P1 remains `shadow`; at most two generation attempts are allowed. |

## Required production environment

```bash
export PATHOLOGY_INPAINT_CHECKPOINT=/models/pathology/pathology-inpaint-controlnet
export PATHOLOGY_CROSS_V1_CHECKPOINT=/models/pathology/pathology-cross-v1-pix2pix/cross_v1
export PATHOLOGY_PIX2PIX_CHECKPOINT=/models/pathology/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt
export PATHOLOGY_PROBNET_CHECKPOINT=/models/pathology/pathology-probnet/best_epoch29_c29607f1b609accb.pt
export PATHOLOGY_SEGMENTATOR_CHECKPOINT=/models/pathology/pathology-segmentator/segmentator_fine_c_joint_epoch2_ddb95a7b.pt
export PATHOLOGY_SEGMENTATOR_PYTHON=/home/lyw/anaconda3/envs/pathology-segmentator-mmseg/bin/python3.10
export PATHOLOGY_CELLVIT_ROOT=/home/lyw/wqx-DL/flow-edit/FlowEdit-main/CellViT-plus-plus-main/CellViT-plus-plus-main
export PATHOLOGY_CELLVIT_MODEL=$PATHOLOGY_CELLVIT_ROOT/checkpoints/CellViT-SAM-H-x40-AMP-001.pth
export PATHOLOGY_CELLVIT_PYTHON=/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python
```

Launch:

```bash
python scripts/phase3_end_to_end_ui.py
```

The machine-readable release pins the immutable Hub checkpoint commits:
ProbNet `add6970449cf3a94997375a665c832e91b188251` and Segmentator
`afe195eaa3a4c2c1d24a41932669f5e55ac987bf`. The default local paths point at
the packaged server releases. Environment variables are the supported
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
- ProbNet checkpoint count, density, or type heads as production quota sources.
- organ-, tissue-, or dataset-specific hard placement bands that override the
  learned ProbNet ordering.

The GlaS-only boundary-center placement draft, CellViT-runtime-input
Segmentator ablation, multi-primitive paper benchmark draft, and temporary
online canary/deck scripts are not product components and are intentionally
absent from the release.

The online acceptance thresholds remain engineering thresholds pending blinded
G2 calibration. The product may report `engineering_pass_uncalibrated`; it must
not report formal clinical or publication validation from the online loop.
