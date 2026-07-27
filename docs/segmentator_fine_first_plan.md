# Segmentator Fine-First Plan

## Decision

The grouped epoch-7-style baseline remains the production coarse anchor:

`/data1/zhao/wqx/segmentator_improved/baseline_seed42/best_composite.pt`

The actual initialization checkpoint has grouped validation mIoU `0.616014`,
Boundary-F1@4 `0.510735`, and HD95 `85.6890`. The earlier Boundary-F1@4 value
`0.526891` belongs to a different epoch whose mIoU is `0.602737`; these two
checkpoints must not be combined into one preservation gate. The later
`training_seed42` coarse ablation is not accepted: its best mIoU is `0.598876`,
it missed the `-0.005` non-inferiority margin, and its intended rare-class
sampling was not enabled. Preserve that run as a negative ablation, but do not
use it to initialize Fine Segmentator.

The incomplete fine run initialized from that weaker checkpoint is also not a
valid baseline. Archive it and restart from the grouped coarse anchor.

## Phase 1: Coarse-Preserving Fine Probe

- Keep UNI2-h, Simple Feature Pyramid, and Mask2Former frozen. Train only the
  hierarchical fine head and use a fine-only loss during training.
- Match the grouped baseline inference contract: asymmetric reflect padding,
  equal-dataset sampling (`temperature=0`), and no coarse class weighting.
- Keep RandStainNA at `0.7`, horizontal/vertical flips, and 90-degree rotations.
  Do not use scale-crop in the probe.
- Separate coarse and fine weighting. Use softened inverse-square-root fine
  weights clamped to `[0.75, 2.0]`; keep unsupported fine ID 14 masked.
- Run the existing single-scale head for three epochs with LR `1e-4`, cosine
  decay, and rank-zero distributed validation. Select checkpoints by the macro
  mean of fine mIoU over BCSS, GlaS, and PANDA, subject to coarse preservation.
- Continue the single-scale model to at most eight epochs only if the three-epoch
  probe beats a deterministic majority-child fine baseline by at least `0.05`
  macro mIoU and all three fine datasets produce non-degenerate subtype output.
- Otherwise replace the single-scale head with a fine-only multiscale adapter
  that fuses all four frozen pyramid levels. Do not unfreeze the coarse branch.

Fine checkpoints are eligible only when coarse mIoU is at least `0.6159` and
Boundary-F1@4 is at least `0.5106`. These are small numerical-tolerance floors
around the metrics of the actual initialization checkpoint. A tensor audit of
the best Fine Probe checkpoint found zero differences across all 651 shared
tensors (703,925,705 elements), so the coarse branch is exactly frozen.

### Fine-supervision-aware sampling

The grouped training manifest contains 74,821 patches, but only BCSS, GlaS,
and PANDA define a branching fine target. Under equal-dataset sampling, half
of the sampled patches are therefore IGNITE, ORCA, or PUMA patches whose
fine target is entirely ignored. The formal Fine Segmentator run replaces
that probe sampler with the following frozen policy:

1. Scan the raw training masks and retain a patch in the sampling support only
   when it contains at least `256` valid dataset-specific fine pixels.
   Coarse-only and negligible edge patches receive exactly zero sampling weight.
2. Assign dataset probability mass proportional to
   `eligible_patch_count ** 0.5`. This upweights the small GlaS cohort without
   repeating it as aggressively as equal-dataset sampling.
3. Within each dataset, derive per-class presence multipliers from inverse
   square-root fine-pixel frequency. A patch receives the largest multiplier
   among its present fine classes, capped at `4x`, and the weights are then
   renormalized so rare-class boosting does not alter the dataset-level mass.
4. Keep the fine loss weights independently clamped to `[0.75, 2.0]`. Sampling
   and loss weighting solve different problems and both remain bounded.
5. Fine ID 14 (DCIS) currently has zero observed training pixels. Keep it
   unsupported and do not claim 16-class coverage until real DCIS supervision
   is added.

The July 24 patch-level audit found 14,799 eligible BCSS patches, 1,251 GlaS
patches, and 23,354 PANDA patches at the 256-pixel threshold, with no unreadable
masks. Relative to a one-pixel threshold, this removes only 108 BCSS edge
patches and no GlaS or PANDA patches. Temperature `0.5` assigns normalized
dataset mass `39.26% / 11.42% / 49.32%`, respectively. After filtering, BCSS
contains 53 Angioinvasion-positive patches; the bounded `4x` presence multiplier
keeps them in active rotation without changing BCSS's dataset-level mass.

The formal run continues from the best eligible probe checkpoint, trains the
fine head only with LR `5e-5`, runs for at most eight epochs, and uses
early-stopping patience three. It is launched automatically only when the
completed probe has no runtime errors, has an eligible checkpoint, beats the
per-dataset majority-child baseline by at least `0.05` macro mIoU, and produces
non-degenerate subtype predictions on all available fine datasets. A failed
gate is reported and no replacement training is started automatically.

The formal seed-42 run was launched on GPUs 3 and 5 from
`fine_probe_single_scale_seed42/best_fine_mIoU.pt`. Its output directory is:

`/data1/zhao/wqx/segmentator_fine/fine_supervised_sampling_seed42`

The run completed all eight epochs. Epoch 8 is the accepted Fine V2 checkpoint:

`/data1/zhao/wqx/segmentator_fine/fine_supervised_sampling_seed42/best_composite.pt`

Its grouped validation metrics are fine dataset-macro mIoU `0.647043`, overall
fine mIoU `0.585926`, coarse mIoU `0.616014`, Boundary-F1@4 `0.510736`, and
HD95 `85.6890`. The fine macro gain over the Fine Probe is `0.056519`; the
coarse branch remains unchanged.

## Phase 2: Boundary Refinement

Start from the accepted Fine V2 checkpoint. Freeze UNI2-h, Simple Feature
Pyramid, Mask2Former, and the Fine Head; train only the 150,792-parameter coarse
residual BoundaryRefinementHead.

This line uses all six datasets rather than the Fine-only support. It samples
39,404 patches per epoch with dataset temperature `0.5`, inverse-square-root
coarse class weights, and a bounded `2x` presence boost for vessel, necrosis,
and immune patches. It uses synchronized RandStainNA, flips, 90-degree
rotations, and a light `0.1` scale crop.

Select checkpoints by Boundary-F1@4 subject to coarse mIoU `>=0.6159` and fine
dataset-macro mIoU `>=0.6370`. Retain the line only when Boundary-F1@4 improves
by at least `0.03`, HD95 falls by at least `10%`, and no ORCA/PUMA mIoU drops
by more than `0.03`. Otherwise deploy the unmodified Fine V2 coarse output.

Script and seed-42 output:

```text
scripts/train_segmentator_fine_boundary_ablation_a800.sh
/data1/zhao/wqx/segmentator_fine/fine_boundary_ablation_seed42
```

## Phase 3: CellViT Teacher

Run Teacher concurrently with Boundary as an independent Fine V2 ablation; do
not initialize Teacher from Boundary. Freeze the entire coarse path and train
only the Fine Head, a zero-initialized fine-only residual adapter, and the
density head, totaling 811,030 parameters. The Fine Head and density head share
the adapter, so CellViT supervision can affect fine segmentation while the
coarse logits remain exactly fixed.

Convert each connected CellViT nucleus in labels `101-105` to a center-based
Gaussian heatmap, then append total-cell density as channel six. Use
`cell_aux_loss_weight=0.1`; mask missing or empty detections with
`nuclei_available`. The density head is training-only and is removed from the
inference contract, while the image-only adapter remains.

Use the same 39,404-patch Fine-aware support and temperature/rare-class policy
as Fine V2, with the additional requirement that every sampled record has a
CellViT mask. The July 25 full audit checked all 94,902 grouped train/val/test
records: every mask exists, no file has an unexpected label, and the small
number of empty detections is excluded from density loss rather than treated
as a true zero-cell target. The audit is stored at:

`/data1/zhao/wqx/segmentator_fine/cellvit_teacher_data_audit_seed42.json`

Retain teacher training only if it improves fine dataset-macro mIoU by `0.01`
without violating coarse mIoU `>=0.6159` and Boundary-F1@4 `>=0.5106`.
Runtime CellViT input remains deferred; it is considered only if it adds mIoU
`0.015` or Boundary-F1@4 `0.03` over teacher.

Script and seed-42 output:

```text
scripts/train_segmentator_fine_cellvit_teacher_a800.sh
/data1/zhao/wqx/segmentator_fine/fine_cellvit_teacher_seed42
```

Boundary and Teacher are intentionally run at the same time but remain
factor-isolated. Launch a third `Boundary + Teacher` combination only if each
individual line passes its own gate. This preserves a readable ablation:
Fine V2 versus Boundary-only versus Teacher-only, followed by the optional
combined model.

## Validation and Rollout

1. Run unit tests for parameter freezing, independent class weights, fine-only
   loss, unsupported-label masking, coarse-guarded checkpointing, and DDP
   rank-zero validation.
2. Run a two-GPU tiny-manifest smoke test before every full training stage.
   The Boundary and Teacher scopes both passed this test on July 25, including
   DDP, checkpoint initialization, rank-zero validation, and parameter-scope
   logging.
3. Run seed 42 for each gated ablation. Expand only the selected configuration
   to seeds 43 and 44.
4. Report grouped validation/test and the independent doctor-reviewed TCGA set.
5. Evaluate actual Fine Segmentator predictions through ProbNet against the
   coarse fallback using validation loss, count rMAE, signed relative error, and
   generated/target count ratio.
