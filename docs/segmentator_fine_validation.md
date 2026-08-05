# Segmentator Fine Validation and Production Selection

Last verified: 2026-08-03

This document separates the final production evaluator from the historical
C-line research candidate. Historical pooled unified-label metrics remain
useful for diagnostics, but they are not independent clinical validation.

## Final production checkpoint

The online Agent, UI and image-edit evaluator use:

```text
/data1/zhao/wqx/segmentator_fine/legacy_anchor_fine_seed42/best_composite.pt
```

- Release: `segmentator-fine-legacy-anchor-v1`.
- Release definition:
  `benchmark_configs/releases/segmentator_fine_legacy_anchor.json`.
- Size: `2,817,774,455` bytes.
- SHA256: `5165e0fb20aca68f64fb06403eff846173fba3a3373deeedb36b3feb81727b3f`.
- Runtime inputs: image only.
- Production selection date: 2026-08-03.

The checkpoint was initialized from the historical coarse checkpoint:

```text
/data1/zhao/wqx/segmentator/best_mIoU.pt
SHA256 5e4b5587359527e0e988428b8cd7fa453255de8729f0fde1ef375fb21d64f112
```

The fine stage used `feature_pyramid_source=legacy_final_depth`,
`freeze_shared_for_fine=true` and `fine_only_loss=true`. Therefore the coarse
shared parameters were not updated; the production checkpoint intentionally
retains the old coarse endpoint and adds the hierarchical fine head.
The old coarse model used
`segmentator_runs/stage4_multidataset_manifest.json`, a per-dataset patch-level
seed-42 split rather than a source-group-disjoint split. Of the current G2-600
source patches, 545 were in that coarse manifest's training split and 55 were
in its validation split. Segmentator-derived G2 measurements must therefore
remain labelled verifier-coupled engineering evidence, not independent
generalization results.

Its grouped validation record is:

| Metric | Legacy-anchor production checkpoint |
| --- | ---: |
| Coarse dataset-macro mIoU | `0.724986` |
| Pooled unified mIoU, diagnostic only | `0.645508` |
| Fine dataset-macro mIoU | `0.812287` |
| Boundary-F1@4 | `0.512012` |
| HD95 | `74.1973` |

## Frozen evaluation-data boundary

`phase5_runs/cross_meta/metadata_cross_val.json` is fixed evaluation metadata,
not a continued-training set. It contains samples assigned by
`grouped_seed42.json` to train, validation and test. An exact audit found
10,137 unique target/reference samples: 8,692 train, 433 validation, 991 test
and 21 unresolved ORCA reference identifiers. The frozen G2-600 cohort itself
contains 506 train, 32 validation and 62 test patches.

Training on the complete `eval_meta` would therefore expose validation/test
labels and invalidate same-endpoint comparisons. It is forbidden for the
production checkpoint. Any small-learning-rate continuation must use a new
candidate name, exclude frozen evaluation rows, preserve this checkpoint, and
be promoted only after group-disjoint validation and evaluation on a new
clinician-labelled endpoint. Reusing the `eval_meta` rows already assigned to
the grouped train split adds no new labelled images because those samples were
already available to the original fine training.

After the present paper benchmark has been frozen and reported, clinician
corrections from this endpoint may be used to train a separately versioned
post-benchmark product model. That later model must not be evaluated against
or substituted into the already reported benchmark.

### Safe small-learning-rate candidate

No continuation was launched as part of the 2026-08-03 production decision.
If a new coarse-improvement experiment is opened, it should:

- initialize from the frozen legacy-anchor checkpoint but write to a new
  output directory and release ID;
- train only on the grouped train split after excluding every frozen benchmark
  patch/group, never on complete `eval_meta`;
- use joint coarse and fine supervision rather than the existing
  `fine_only_loss`, because fine-only training cannot improve the old coarse
  endpoint;
- keep the UNI encoder frozen initially, use decoder learning rate
  `1e-6` to `5e-6`, run at most two to three epochs with early stopping, and
  reject any per-dataset/common-class regression;
- reserve a new source-group-disjoint, clinician-labelled endpoint before
  checkpoint selection.

This is a research-candidate protocol, not authorization to mutate the frozen
production checkpoint.

## Historical C-line research candidate

The Boundary + CellViT Teacher joint epoch-2 checkpoint remains a research
comparison and manual diagnostic, not the production selector:

```text
/data1/zhao/wqx/segmentator_fine/fine_boundary_teacher_joint_seed42/best_composite.pt
```

- Size: `2,820,003,238` bytes.
- SHA256: `ddb95a7be8a2d80f6d8a77672b265f261fbe79ca2c3e35215c9ed871da604c34`.
- Private Hub artifact:
  `Qinxin11/pathology-segmentator/segmentator_fine_c_joint_epoch2_ddb95a7b.pt`,
  checkpoint commit `afe195eaa3a4c2c1d24a41932669f5e55ac987bf`.
- Runtime inputs: image only. CellViT is a training teacher and is not required
  at inference.
- Historical safety-comparison source:
  `/data1/zhao/wqx/segmentator_fine/fine_cellvit_teacher_v2_seed42/best_composite.pt`.

## Metric contract

Canonical coarse mIoU is computed independently per dataset over only that
dataset's annotated biological classes. Background and other tissue are
excluded from the class average. The six dataset mIoUs are then averaged with
equal weight. The historical unified-eight-class value is reported only as
`pooled_unified_mIoU`.

Fine dataset-macro mIoU includes only BCSS, GlaS, and PANDA:

- BCSS supported fine classes: tumor and angioinvasion.
- GlaS supported fine classes: adenomatous, moderately differentiated, and
  poorly differentiated gland.
- PANDA supported fine classes: Gleason 3, 4, and 5.
- DCIS is outside the supported Segmentator fine task. It is excluded from
  macro metrics and checkpoint gates and remains an unevaluated diagnostic.
- IGNITE, ORCA, and PUMA have no fine supervision and never enter the fine
  dataset-macro.

Angioinvasion has training support but no validation or test support, so the
current held-out splits do not provide an Angioinvasion performance estimate.

## Historical C-line grouped validation

| Metric | Joint epoch 2 |
| --- | ---: |
| Coarse dataset-macro mIoU | `0.691264` |
| Pooled unified mIoU, diagnostic only | `0.619820` |
| Fine dataset-macro mIoU | `0.669412` |
| Boundary-F1@2 | `0.413745` |
| Boundary-F1@4 | `0.520756` |
| Boundary-F1@8 | `0.637244` |
| HD95 | `80.6823` |

Coarse validation mIoU by dataset:

| Dataset | Native-class mIoU |
| --- | ---: |
| BCSS | `0.530201` |
| GlaS | `0.812603` |
| IGNITE | `0.656132` |
| ORCA, tumor only | `0.849940` |
| PANDA | `0.568730` |
| PUMA | `0.729980` |

Fine validation mIoU by supervised dataset:

| Dataset | Supported-class mIoU |
| --- | ---: |
| BCSS | `0.875061` |
| GlaS | `0.703010` |
| PANDA | `0.430166` |

## Historical C-line held-out internal test

The test split contains `10,012` patches from `114` held-out source groups.
The grouped manifest has zero sample, image, and source-group overlap among
train, validation, and test. A source group may represent a WSI, patient, or
ROI depending on the source dataset; this is an internal grouped test, not a
uniform external-WSI cohort.

| Metric | Joint epoch 2 | Teacher V2 | Joint gain |
| --- | ---: | ---: | ---: |
| Coarse dataset-macro mIoU | `0.627313` | `0.602390` | `+0.024923` |
| Fine dataset-macro mIoU | `0.601549` | `0.577418` | `+0.024132` |

Joint epoch-2 test results by dataset:

| Dataset | Coarse native-class mIoU | Fine supported-class mIoU |
| --- | ---: | ---: |
| BCSS | `0.484067` | `0.860275` |
| GlaS | `0.750857` | `0.582575` |
| IGNITE | `0.527444` | not applicable |
| ORCA, tumor only | `0.857520` | not applicable |
| PANDA | `0.584624` | `0.361798` |
| PUMA | `0.559368` | not applicable |

The corrected resource-light test reevaluation did not recompute spatial
boundary metrics. Boundary-F1 and HD95 claims above therefore apply to grouped
validation only.

## Interpretation and release decision

- Joint epoch 2 remains a useful internal C-line research candidate because it
  improves both corrected coarse and supported-fine test macro over Teacher V2;
  it is not the selected online production evaluator.
- BCSS coarse mIoU is `0.026550` lower than Teacher V2 on the test split and
  remains the main regression risk.
- DCIS recognition is not claimed and DCIS-dependent downstream operations
  must remain disabled.
- The test split has now been inspected and must not be used for further
  checkpoint selection or hyperparameter tuning.
- Paper text may describe these values as grouped held-out internal test
  results. It must not describe them as external validation or uniformly
  unseen full-WSI evaluation.

## Reproducibility

Corrected evaluation artifacts:

```text
/data1/zhao/wqx/segmentator_fine/metric_recompute_annotated_v1/
  c_joint_epoch2_best_composite.corrected_val.json
  c_joint_epoch2_best_composite.corrected_test.json
  teacher_v2_epoch7_best_composite.corrected_test.json
```

The evaluator and training validation use the same supported-class contracts:

```text
scripts/eval_segmentator_per_class.py
segmentator/config.py
segmentator/metrics.py
segmentator/training.py
```
