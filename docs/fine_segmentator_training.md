# Hierarchical Fine Segmentator

## Purpose

The Fine Segmentator is a separate training line. It preserves the shared
8-class coarse task and adds a 16-class hierarchical prediction constrained by
the coarse parent at every pixel.

This is not a flat 16-class replacement. Coarse-only datasets continue to train
the shared UNI2-h, feature pyramid, and Mask2Former branch without generating
false fine supervision.

## Current supervision

- PANDA: Gleason 3/4/5 (fine IDs 8/9/10) under Tumor.
- GlaS: adenomatous/moderately differentiated/poorly differentiated glands
  (fine IDs 11/12/13) under Tumor.
- BCSS: generic tumor/DCIS/angioinvasion (fine IDs 1/14/15) under Tumor.
- IGNITE, ORCA, PUMA: coarse supervision only.

The head and dataset contract are parent-agnostic. A future stroma, immune,
vessel, or other-tissue subdivision only requires extending the unified label
mapping; the architecture does not need to change.

The current grouped training histogram contains BCSS ID 15 but no ID 14. ID 14
is therefore marked unsupported and masked at inference until DCIS training
pixels are restored or added. It must not be reported as a trained class.

## Outputs

- `pred`: compatible 8-class coarse prediction.
- `fine_pred`: unified 0-15 prediction after parent and dataset constraints.
- `hierarchical_pred`: deployment alias for the constrained fine prediction.

Fine loss is evaluated only where a dataset distinguishes multiple children of
the same parent. Validation reports overall and per-dataset fine mIoU, Dice,
accuracy, valid pixels, and per-class support.

## Initialization and launch

The formal run initializes shared parameters from the best leakage-free coarse
checkpoint. The fine head starts randomly. Optimizer, scheduler, and early-stop
state start fresh.

```bash
GPU_IDS=0,3 \
COARSE_CHECKPOINT=/data1/zhao/wqx/segmentator_improved/training_seed42/best_composite.pt \
bash scripts/train_segmentator_fine_hierarchical_a800.sh
```

The script resumes its own `checkpoint_last.pt` when present. It refuses random
initialization unless `ALLOW_RANDOM_INIT=1` is explicitly set.

## Evaluation order

1. Compare coarse metrics against the fixed coarse baseline.
2. Report fine metrics only for datasets and classes with explicit fine truth.
3. Evaluate ProbNet with ground-truth fine masks, collapsed coarse masks, and
   actual Fine Segmentator predictions.
4. Keep the hierarchical model only if coarse quality does not materially
   regress and fine predictions improve downstream ProbNet fidelity.
