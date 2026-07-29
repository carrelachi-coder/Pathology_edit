# Segmentator Fine Validation Report

Last verified: 2026-07-29

This document is the canonical metric card for the current Segmentator Fine
release candidate. Historical pooled unified-label metrics remain useful for
diagnostics, but they are not the primary checkpoint or paper metrics.

## Selected checkpoint

The selected model is Boundary + CellViT Teacher joint epoch 2:

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
- Safety fallback:
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

## Grouped validation

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

## Held-out internal test

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

- Joint epoch 2 remains the primary internal release candidate because it
  improves both corrected coarse and supported-fine test macro over Teacher V2.
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
