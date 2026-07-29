# G2 Edit-Local Consistency Runbook

## 0. Decision and scope

Track G2 is the immediate consistency benchmark for the second half of the
agentic pathology-edit workflow:

```text
frozen source RGB and masks
        +
frozen target tissue/nuclei masks
        +
semantic and generation edit regions
        |
        v
route to Inpaint or production Cross-v1
        |
        v
generate one candidate
        |
        v
re-segment with the frozen Segmentator and CellViT
        |
        +-- pass --> select attempt 1
        |
        +-- fail --> run the deterministic alternate route once
                          |
                          v
                 select attempt 2 if it passes;
                 otherwise retain one candidate as `needs_review`
```

G2 is sufficient for the immediate engineering/scientific question: given an
already accepted target mask and changed region, does routing, image
generation, automatic verification, and one bounded regeneration produce a
more target-consistent final image while preserving the unedited tissue?

G2 does **not** replace:

- G1 clinician-condition consistency against a real paired target;
- Patho-KID or blinded review for realism;
- G3 repeated-seed robustness;
- independent human assessment of verifier false acceptance.

Segmentator and CellViT-derived measurements are verifier-coupled evidence.
They must not be described as independent ground truth, and tissue fidelity,
preservation, boundary quality, nuclei consistency, and realism must not be
collapsed into one manuscript score.

This document is the normative execution contract for G2. The shorter G2
sections in `docs/benchmark_implementation_plan.md` are summaries and defer to
this runbook when details differ.

## 1. Frozen evaluator and generator contract

### 1.1 Segmentator

The engineering-frozen Segmentator C checkpoint is Boundary + CellViT
Teacher joint epoch 2:

```text
/data1/zhao/wqx/segmentator_fine/fine_boundary_teacher_joint_seed42/best_composite.pt
```

Checkpoint identity:

- size: `2,820,003,238` bytes;
- SHA256:
  `ddb95a7be8a2d80f6d8a77672b265f261fbe79ca2c3e35215c9ed871da604c34`;
- runtime input: image only; CellViT was a training teacher and is not an
  inference dependency.

Its corrected grouped validation metrics are:

- coarse dataset-native macro mIoU: `0.691264`;
- pooled unified-eight-class diagnostic mIoU: `0.619820`;
- Boundary-F1@2/4/8: `0.413745 / 0.520756 / 0.637244`;
- supported fine dataset-macro mIoU: `0.669412`;
- HD95: `80.6823`.

Its grouped held-out internal test metrics are coarse dataset-native macro
mIoU `0.627313` and supported fine dataset-macro mIoU `0.601549`. These are
internal grouped patch-test results, not external validation or a uniform
unseen full-WSI cohort. The complete metric contract is in
`docs/segmentator_fine_validation.md`.

The repository release definition is:

```text
benchmark_configs/releases/segmentator_fine_c_epoch2.json
```

It is deliberately marked
`engineering_frozen_pending_independent_doctor_review`. This allows G2
interface work and canary execution, but not final publication evaluation.
The release becomes `final_frozen` only after the implementation commit and
independent doctor-reviewed report path/hash are recorded.

Before any formal G2 generation, the evaluator release block in the freeze
manifest must contain:

```yaml
segmentator:
  release_status: final_frozen
  release_id: segmentator-fine-c-joint-epoch2-v1
  checkpoint: /data1/zhao/wqx/segmentator_fine/fine_boundary_teacher_joint_seed42/best_composite.pt
  checkpoint_sha256: ddb95a7be8a2d80f6d8a77672b265f261fbe79ca2c3e35215c9ed871da604c34
  code_commit: <release_implementation_git_sha>
  decoder: mask2former
  architecture:
    hierarchical_fine: true
    boundary_refinement: true
    refinement_gate_mode: learned_soft
    cellvit_mode: teacher
  input:
    image_size: [512, 512]
    mpp: 0.25
    color_mode: RGB
    normalization: imagenet-rgb-v1
    padding: mask2former-reflect-right-bottom-align56-v1
  output:
    coarse_class_ids: [0, 1, 2, 3, 4, 5, 6, 7]
    save_argmax_png: true
    save_probability_npz: true
    save_entropy_npy: true
    fine_output_policy: dataset_constrained
  human_validation_report: /absolute/path/to/doctor_review_report.json
  human_validation_report_sha256: <sha256>
```

No formal G2 job may resolve the Segmentator from an implicit code default.
The checkpoint, model-construction flags, preprocessing, label map, and hash
must come from the freeze manifest. A checkpoint that cannot be loaded with
`strict=True` under the frozen architecture is invalid.

The independent doctor-reviewed TCGA result must be reported separately from
the grouped validation metrics above. Until its path and hash are present in
the release block, G2 may run as an engineering smoke but not as the final
publication evaluation.

### 1.2 Coarse and fine output policy

Every G2 row receives a coarse eight-class prediction. Fine prediction is an
additional endpoint, not a replacement for the common coarse endpoint.

- Generic tissue edits use the coarse eight-class output.
- PANDA and GlaS specialized primitives additionally use the
  dataset-constrained hierarchical fine output.
- BCSS DCIS is unsupported by the frozen fine task. A DCIS row may contribute
  to a supported coarse parent-class analysis, but it must be marked
  `fine_applicability=unsupported_dcis` and excluded from fine-class
  consistency summaries.
- A dataset without explicit fine supervision is
  `fine_applicability=not_applicable`.
- Fine classes forbidden for a dataset are masked before argmax. An
  unconstrained 16-class argmax is not a valid G2 prediction.

### 1.3 Target nuclei condition

G2 consumes one frozen target nuclei mask per condition. It does not select or
retrain the nuclei-layout model.

The authoritative production policy is Section 5.8 of
`docs/benchmark_implementation_plan.md`:

- patch-local statistics, shrunk toward the matching dataset/profile library
  only when sparse, determine total count and exact type quotas;
- exact largest-remainder allocation determines tissue, type, and
  disconnected-component quotas;
- same-class shapes come from the current patch first and the matching
  profile library second;
- the immutable epoch-29 CellDistNet/ProbNet checkpoint contributes only
  `P(nucleus) = 1 - P(background)` for spatial candidate ordering;
- learned count, density, and type heads never set the requested condition;
- accepted shapes obey the frozen full-tissue-containment and zero-overlap
  rules.

The spatial checkpoint is:

```text
/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/
best_epoch29_c29607f1b609accb.pt
```

SHA256:

```text
c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211
```

Every G2 condition records requested, placed, and unfilled nuclei plus exact
quota/shape provenance. The saved placed layout is the image generator's
actual target and therefore the G2 nuclei target. Upstream unfilled placement
is reported separately and is not charged to the image generator.

The same bitwise target nuclei mask is used for fixed Inpaint, fixed Cross,
routed attempt 1, and the full agentic arm. It is never regenerated per route
or after seeing a generated image. A target layout whose hash changes creates a
new `g2_condition_id`.

### 1.4 CellViT

The version-locked CellViT evaluator uses:

```text
checkpoint:
/home/lyw/wqx-DL/flow-edit/FlowEdit-main/CellViT-plus-plus-main/
CellViT-plus-plus-main/checkpoints/CellViT-SAM-H-x40-AMP-001.pth

SHA256:
356418f19d9d478f164c7a31f85274584fefaa02355815c09f52346c658c8ec4
```

Evaluation is on `512 x 512 @ 0.25 MPP`. The project label convention is:

```text
0   background
101 neoplastic
102 inflammatory
103 connective
104 dead
105 epithelial
```

Formal nuclei metrics use connected target instances and CellViT-predicted
instances/centroids. Labelled nucleus-pixel occupancy may remain an online
engineering proxy, but it must be named
`nuclei_occupied_area_relative_error`; it is not a publication-level cell
count.

### 1.5 Generators

G2 freezes:

- the production local Inpaint checkpoint;
- production Cross V1 with IP-Adapter and UNI disabled;
- the orientation-adjusted pix2pix-v2 postprocessor;
- no Cross color-transfer or color-matching postprocess;
- all inference steps, guidance, prompts, seeds, and
  preprocessing.

Every production/formal job must explicitly set:

```bash
export PATHOLOGY_PIX2PIX_CHECKPOINT=/absolute/path/pix2pix/pix2pix_epoch26_step214895.pt
```

The resolved checkpoint must report:

```text
epoch: 26
global_step: 214895
trust_gate: nuclei_reference_support_v2
sha256: be5fe9376efdb5620a57481082f6d5738b6353796fb00fe6e58f6b212ba7c2ac
```

`pilot_step001000.pt`, an epoch-25 checkpoint, a historical run-directory
name, or a fallback in `model_paths.py` is not a valid production selector.

## 2. Evaluation frame and label validity

All source, target, and generated assets are evaluated in the same physical
frame:

```text
image size: 512 x 512
MPP:        0.25 micrometres/pixel
FOV:        128 x 128 micrometres
```

Nearest-neighbour interpolation is mandatory for tissue/nuclei ID masks and
binary regions. RGB resampling uses the frozen generation preprocessing.
Images or masks with inconsistent dimensions, MPP, orientation, or field of
view are `operational_incomplete`; they are not silently resized by the metric
runner.

The valid coarse labels are `0..7`, with `255` as ignore. Every metric must
apply the organ/dataset-valid class set. A predicted biological class that is
valid in the unified schema but invalid for the row's organ remains a
false-positive prediction in diagnostics and is not silently mapped to
`other_tissue`.

## 3. Region definitions

Let:

- `M_src` be the frozen source tissue mask;
- `M_tgt` be the frozen target tissue mask;
- `G` be the binary region sent to the generator;
- `R` be the semantic change region used for verification.

The semantic region is authoritative and is always recomputed as:

```text
R = (M_src != M_tgt)
```

An explicit `semantic_change_region` file is valid only if it is bitwise equal
to the recomputed `R`.

The generation region may be wider:

```text
R subset_of G
```

`G` may include whole-gland support for GlaS or a controlled margin around a
thin edit. It must never omit a semantic-change pixel. Routing uses `R` derived
from the tissue masks, never the wider `G`.

The preserved region is:

```text
U = valid_pixels and not R
```

For a fixed radius `r = 4` pixels (`1 micrometre` at `0.25 MPP`), define:

```text
B_in  = R     and dilate(not R, r)
B_out = not R and dilate(R, r)
B     = B_in or B_out
U_far = U and not B_out
```

`B_in` measures realization at the inner edit boundary. `B_out` measures
spillover immediately outside the intended edit. Preservation on `U_far`
separates broad off-target drift from a narrow transition artifact.

Rows with `R` empty are invalid for G2 and remain in a manifest-rejection log.
They are not converted to `noop` benchmark successes.

## 4. Manifest construction

### 4.1 Source bank

Build G2 from the canonical frozen structured-GT target-mask bank produced by
`phase3_mask_edit.cli.recompute_mask_edit_benchmark_metrics`. Do not rebuild
target masks, rerun prompts, change contour thresholds, or substitute the
controlled Instruction output.

The canonical source rows contain at least:

```text
sample_id
mode
organ
profile
primitive
strength
source_dataset
wsi_id
patient_id
source_mask_path
target_mask_path
change_region_path
metrics_path
```

Only completed canonical structured-GT rows whose asset hashes match the
frozen mask benchmark are eligible.

### 4.2 Required G2 fields

The enriched G2 manifest is JSONL. Every row must contain:

```text
schema_version
g2_condition_id
sample_id
organ
profile
source_dataset
wsi_id
patient_id
primitive
strength
is_specialized
fine_applicability

reference_image_path
source_tissue_mask_path
target_tissue_mask_path
source_nuclei_mask_path
target_nuclei_mask_path
semantic_change_region_path
generation_change_region_path

image_size
mpp
semantic_changed_pixels
semantic_change_ratio_image
semantic_change_ratio_tissue
generation_changed_pixels
generation_change_ratio_image
component_count
largest_component_fraction
bbox_fraction
transition_count
transition_pairs

router_policy_id
router_policy_sha256
verifier_policy_id
verifier_policy_sha256
generator_release_id
segmentator_release_id
cellvit_release_id
seed

reference_image_sha256
source_tissue_mask_sha256
target_tissue_mask_sha256
source_nuclei_mask_sha256
target_nuclei_mask_sha256
semantic_change_region_sha256
generation_change_region_sha256
```

`transition_pairs` is the sorted set of pixel-level
`(source_class_id, target_class_id)` pairs within `R`.

`g2_condition_id` is a stable digest of:

```text
sample_id
all five source/target asset hashes
semantic and generation region hashes
seed
router policy hash
verifier policy hash
generator release hash
```

Changing any field creates a new condition ID and invalidates reuse of old
generation outputs.

### 4.3 Deterministic selection

Selection uses only frozen pre-generation metadata. Within each valid
`organ/profile x primitive x strength` cell:

1. reject missing/corrupt/shape-mismatched assets and record the reason;
2. group by WSI/patient to expose available diversity;
3. assign each eligible row the stable rank
   `SHA256(selection_seed | sample_id)`;
4. select in rank order while minimizing repeated WSI/patient use;
5. never use generator output, Segmentator output, CellViT output, route
   success, or visual quality to select a row.

The frozen cohorts are:

- `g2_canary`: 18 conditions, three per organ, covering Inpaint-first,
  Cross-first, gray-zone, generic, and available fine-label paths;
- `g2_gate_calibration`: 60 conditions, ten per organ, balanced by planned
  route, edit-size stratum, primitive family, and tissue complexity;
- `g2_smoke_agentic`: up to three cases per each of 231 valid cells, maximum
  693 conditions;
- `g2_smoke_route_ablation`: the first ranked smoke row in each cell, maximum
  231 conditions;
- `g2_formal_agentic`: up to ten cases per valid cell, maximum 2,310
  conditions, containing every smoke row.

Short cells retain every available row and are listed in `shortfalls.csv`.
Reports always expose requested, available, and completed counts by cell.

The calibration cohort must not overlap the official smoke/formal test rows at
the WSI/patient level. If the source bank cannot support this separation, use
an independent calibration pool and document the reduced coverage.

## 5. Strategy arms and fairness

### 5.1 Required arms

The route-ablation cohort is generated once with both fixed routes:

1. `fixed_inpaint`;
2. `fixed_cross_epoch26`.

The same rows also produce:

3. `routed_attempt1`, using the frozen router only;
4. `agentic_selected_final`, using the frozen router, verifier, and at most
   one alternate-route recovery.

All four arms use identical sample IDs, source/target masks, prompts,
conditions, seed, and evaluator release. The only permitted difference is the
route/recovery policy.

The full smoke and focused formal cohorts require `routed_attempt1` and
`agentic_selected_final`. Fixed routes may be expanded beyond the 231-row
ablation cohort only when compute is available; any comparison remains paired
on the exact intersection of sample IDs.

External methods enter strict G2 spatial ranking only if they receive
equivalent target tissue and nuclei geometry. Text-only methods are not
assigned zero spatial consistency.

### 5.2 Frozen router

The initial routing policy is:

```text
Inpaint first:
  tissue-normalized change <= 0.12
  and component_count <= 4
  and bbox_fraction <= 0.45
  and transition_count <= 2

Cross first:
  tissue-normalized change >= 0.30
  or transition_count > 2
  or (component_count >= 8 and bbox_fraction >= 0.60)

Gray zone:
  Inpaint first, Cross as the only permitted fallback
```

The freeze manifest records every threshold. Calibration may replace these
pilot values before the official run, but no threshold may change after an
official generated image is inspected.

### 5.3 Maximum attempts and fallback

`max_attempts = 2`.

- An Inpaint attempt that fails target-tissue or nuclei fidelity falls back to
  production Cross.
- A Cross attempt that fails preservation falls back to Inpaint.
- A generation or evaluator tool error may use the remaining backend after
  the operational cause is recorded.
- No prompt editing, seed search, manual route override, or third attempt is
  permitted for an individual scientific failure.

If attempt 1 passes every required route-specific gate, it is selected and
attempt 2 is not generated in the production agentic arm.

If attempt 1 fails, attempt 2 is generated and verified once:

- if attempt 2 passes, select attempt 2 and mark `recovered`;
- if neither passes, retain `final_status=needs_review`;
- the retained image is selected by the frozen deterministic tie-break below,
  not by a human looking at both images.

The formal tie-break for two scientifically failed but operationally valid
candidates is:

1. fewer failed common target-tissue/nuclei gates;
2. higher changed-region macro mIoU;
3. higher target-versus-source probability margin;
4. lower prediction-relative preservation drift on `U_far`;
5. lower boundary spillover on `B_out`;
6. earlier attempt index.

This tie-break is an internal operational selector. It is not a manuscript
aggregate metric. The current weighted `verification.score` is pilot-only
until it is replaced by, or proven exactly equivalent to, this frozen rule.

## 6. Segmentator execution and evaluator calibration

### 6.1 Required predictions

Run the final Segmentator on the unedited real source image once per condition:

```text
S_src, P_src, H_src = Segmentator(I_src)
```

For every valid generated attempt:

```text
S_gen, P_gen, H_gen = Segmentator(I_gen)
```

where:

- `S` is the coarse argmax mask;
- `P` is the coarse class-probability tensor;
- `H` is per-pixel normalized entropy;
- available dataset-constrained fine outputs are saved separately.

The source prediction is mandatory. It supports prediction-relative
preservation, source-relative edit gain, and a prespecified evaluator-clean
sensitivity analysis.

Every inference output records:

```text
input image hash
checkpoint hash
model architecture flags
dataset/profile constraint
preprocessing hash
software commit
device/dtype
runtime
output hashes
```

### 6.2 Evaluator-clean sensitivity

Segmentator can fail on a real source image before generation. Such baseline
error must not be mislabelled as generator drift.

On the independent real-image calibration split, compute source-region
accuracy, source-class recall, normalized entropy, and boundary quality by
organ and source class. Freeze the lower-quality quantiles needed to identify
an `evaluator_uncertain` row. Prefer organ/source-class-specific cutoffs when
the stratum has sufficient WSI support; otherwise use the pooled cutoff.

The official G2 report contains:

- **all-case primary**: every scientifically valid generated row;
- **evaluator-clean sensitivity**: rows passing the frozen source-image
  evaluability rule;
- **evaluator-uncertain audit**: count and failure categories for excluded
  sensitivity rows.

`evaluator_uncertain` does not equal generator failure. The row remains in the
all-case denominator and the uncertainty flag is frozen from the source image
before generated-image metrics are inspected.

### 6.3 Gate calibration

The code defaults:

```text
changed-region accuracy >= 0.70
changed-region macro mIoU >= 0.55
Cross off-target drift <= 0.08
nuclei occupied-area relative error <= 0.35
```

are pilot values, not automatically valid formal thresholds.

Use the WSI/patient-disjoint `g2_gate_calibration` cohort. Generate both fixed
routes, producing 120 images from 60 conditions. Blind the route and ask
reviewers to record:

- requested tissue transition realized: yes/no/uncertain;
- unacceptable change outside the intended region: yes/no/uncertain;
- unacceptable boundary artifact: yes/no/uncertain;
- requested nuclei abundance/type state realized: yes/no/uncertain;
- overall usable for the requested edit: yes/no/uncertain.

Resolve disagreements using the frozen adjudication rule. Select automatic
thresholds only on this calibration cohort. The threshold-selection report
must expose confusion matrices, false-acceptance and false-rejection rates,
organ/route strata, and the exact selection objective. The formal policy
prioritizes low false acceptance; the accepted bound and any minimum
sensitivity are written to the freeze manifest before the official smoke.

If clinician calibration is unavailable, Segmentator/CellViT gates may drive
engineering retry, but automatic `validated` status must be described as a
verifier-defined state rather than a clinical success. No post-test threshold
tuning is allowed.

## 7. Tissue metrics

All metrics are computed per row and per attempt. `255` pixels are excluded.
The manuscript reports the metrics separately; it does not report the internal
selection tuple as one score.

### 7.1 Changed-region accuracy

```text
A_tgt =
  sum_{x in R} [S_gen(x) == M_tgt(x)]
  / |R|
```

This is a transparent pixel endpoint but is not sufficient alone.

### 7.2 Changed-region macro mIoU

Let `L_R` be the organ-valid labels present in either `M_tgt` or `S_gen`
inside `R`. Background is included only when it participates in an actual
target transition. For every `c` in `L_R`:

```text
IoU_c =
  |{M_tgt=c} intersect {S_gen=c} intersect R|
  / |({M_tgt=c} union {S_gen=c}) intersect R|
```

Then:

```text
mIoU_R = mean_c(IoU_c)
```

Using the union of target- and prediction-present classes penalizes a
hallucinated class. An implementation that averages only target-present
classes is not the formal G2 metric.

### 7.3 Target-class endpoints

For every actual target label present on changed pixels, report:

- target-class IoU;
- target-class recall;
- target-class precision;
- predicted and target area within `R`.

Also aggregate by pixel-level transition pair `(source_id, target_id)`.
Multi-transition rows retain every transition and use a pixel-weighted overall
summary plus an unweighted transition-macro sensitivity.

### 7.4 Source residual

```text
source_residual_rate =
  sum_{x in R} [S_gen(x) == M_src(x)]
  / |R|
```

Lower is better when `M_src(x) != M_tgt(x)`. Report it beside target
adherence so a candidate cannot appear successful merely because the target
class is common elsewhere in the region.

### 7.5 Source-relative target gain

The no-edit baseline is the frozen real source image:

```text
A_no_edit =
  sum_{x in R} [S_src(x) == M_tgt(x)]
  / |R|

target_gain_accuracy = A_tgt - A_no_edit
```

Also report:

```text
target_gain_mIoU =
  mIoU(S_gen, M_tgt, R)
  - mIoU(S_src, M_tgt, R)
```

Positive gain means the generated image moved the evaluator toward the
counterfactual target relative to doing nothing.

### 7.6 Soft target-versus-source margin

For each changed pixel, index the generated probability tensor by the exact
target and source IDs:

```text
soft_target_source_margin =
  mean_{x in R}[
    P_gen(x, M_tgt(x)) - P_gen(x, M_src(x))
  ]
```

Also report the corresponding no-edit margin from `P_src` and:

```text
soft_margin_gain =
  soft_margin_gen - soft_margin_source
```

These probability endpoints expose weak/uncertain movement hidden by argmax.
They require the frozen probability tensor and may not be reconstructed from a
PNG mask.

### 7.7 Prediction-relative preservation

The primary preservation endpoint compares generated and source predictions:

```text
preservation_drift_U =
  sum_{x in U} [S_gen(x) != S_src(x)]
  / |U|

preservation_drift_U_far =
  sum_{x in U_far} [S_gen(x) != S_src(x)]
  / |U_far|
```

This cancels stable Segmentator errors on the source image. For diagnostics,
also record:

```text
mask_relative_drift_U =
  sum_{x in U} [S_gen(x) != M_src(x)]
  / |U|
```

Do not use `mask_relative_drift_U` alone as evidence that the generator
changed preserved tissue.

- Cross-v1: `preservation_drift_U_far` is a required safety gate and main
  reported endpoint.
- Inpaint: preservation drift is reported as a secondary safety diagnostic.

### 7.8 Boundary endpoints

Compute class-aware Boundary-F1@4 between `S_gen` and `M_tgt`, restricted to
boundaries intersecting `B`. Matching tolerance is four pixels.

Additionally report:

```text
inner_ring_target_error =
  mean_{x in B_in}[S_gen(x) != M_tgt(x)]

outer_ring_spillover =
  mean_{x in B_out}[S_gen(x) != S_src(x)]
```

Boundary-F1, inner realization, and outer spillover remain separate. The
Segmentator's documented Boundary-F1@4 is approximately `0.51`, so boundary
failure must be interpreted with source-image calibration and may not be
silently merged into central-region mIoU.

### 7.9 Uncertainty

Normalize entropy by `log(K)` for `K` coarse classes. Report:

- mean normalized entropy on `R`, `B`, and `U_far`;
- 95th-percentile entropy on `R`;
- fraction of pixels above the frozen real-image high-entropy cutoff;
- fraction of changed pixels for which neither source nor target is in the
  Segmentator's top two classes.

Uncertainty is a diagnostic and sensitivity variable, not an automatic
substitute for fidelity.

## 8. Fine-label metrics

For supported PANDA/GlaS rows, repeat Sections 7.1-7.6 with the constrained
hierarchical prediction and fine target mask.

The common coarse endpoint remains present on the same row. Reports separate:

- `coarse_consistency`;
- `fine_consistency_supported`;
- `fine_not_applicable`;
- `fine_unsupported_dcis`.

Do not combine coarse and fine mIoU into one mean. Fine summaries are grouped
by dataset and primitive because the available child labels differ.

## 9. Nuclei and tissue-nuclei metrics

### 9.1 Target instances

Convert each connected component of target labels `101..105` to one instance
with:

- centroid;
- class ID;
- area;
- whether the centroid lies in `R`, `B`, or `U_far`.

CellViT predictions are parsed as instances in the same evaluation frame.
Region membership is decided by centroid. A component crossing the region
boundary is assigned once and is listed in the boundary diagnostic.

### 9.2 Changed-region nuclei endpoints

Inside `R`, report:

- target and predicted total instance count;
- total count absolute and relative error;
- per-class count absolute and relative error;
- cell-type distribution JSD;
- total and per-class density error per square millimetre;
- class-presence recall.

When the target count is zero, relative error is undefined. Report the target
and predicted counts plus a zero-target false-positive indicator rather than
dividing by one.

### 9.3 Spatial endpoints

When the generator receives the target nuclei geometry, match target and
predicted centroids with Hungarian assignment at `6 micrometres`:

- class-agnostic precision, recall, and F1;
- class-aware precision, recall, and F1;
- matched-distance mean and 95th percentile.

These metrics evaluate whether requested spatial geometry survives rendering.
They are not applied to a baseline that did not receive equivalent target
geometry.

### 9.4 Nuclei preservation

For Cross-v1, compare CellViT detections on generated and source RGB inside
`U_far`:

- total count relative drift;
- per-class count drift;
- cell-type JSD;
- optional spatial matching for source-preservation, clearly labelled as a
  source-relative endpoint.

Inpaint reports the same values as secondary safety diagnostics.

### 9.5 Tissue-nuclei coherence

Assign every predicted nucleus centroid the generated Segmentator tissue class
at that location. Report:

- fraction of nuclei assigned to an organ-invalid tissue class;
- target-versus-generated joint `(tissue_class, nucleus_class)` JSD;
- per-tissue nucleus density error where target support is nonzero.

This is Segmentator/CellViT-derived coherence, not human-GT cellular
organization.

## 10. Online verification and formal evaluation

The same frozen Segmentator and CellViT may be used online and offline, but the
roles are different:

- **online verifier**: decides pass/fallback/selection under the frozen gate;
- **formal evaluator**: emits the complete metric set for attempt 1 and the
  selected final image without dropping failures.

The formal evaluator always recomputes metrics from saved artifacts. It never
trusts only the compact online summary.

Required online gates after calibration:

- both routes: changed-region tissue fidelity;
- both routes: nuclei consistency when a target nuclei condition exists;
- Cross-v1: prediction-relative preservation on `U_far`;
- Inpaint: unchanged drift and boundary spillover are logged but are secondary
  unless the calibration report explicitly promotes one to a gate.

The gate policy must identify the metric version, comparison direction,
numeric threshold, applicability rule, and missing-value behavior. Missing a
required evaluator output is operational failure, not a scientific fail.

## 11. Status model

Store operational and scientific status separately.

Operational status:

```text
pending
running
complete
operational_incomplete
```

Scientific status:

```text
not_evaluated
validated_first_pass
recovered
needs_review
evaluator_uncertain
```

An evaluator-uncertain row may also be validated/recovered by the automatic
policy; preserve both fields rather than overwriting one with the other.

Definitions:

- `scientific_failure`: valid image and valid evaluator outputs, but one or
  more frozen gates fail;
- `operational_incomplete`: missing/corrupt image, OOM, dependency failure,
  evaluator crash, shape/MPP mismatch, or missing required metadata;
- `needs_review`: all permitted attempts completed but none passed;
- `recovered`: attempt 1 failed, the deterministic alternate route passed, and
  attempt 2 was selected.

Operational failures are rerun with identical inputs, hashes, seed, and
policy. Scientific failures are not granted additional attempts.

## 12. Artifact layout

```text
/data1/zhao/wqx/benchmark_v1/g2/
  freeze/
    g2_freeze_manifest.yaml
    g2_freeze_manifest.sha256
    segmentator_release.json
    cellvit_release.json
    generator_release.json
    router_policy.json
    verifier_policy.json
    selection_policy.json

  manifests/
    g2_canary.jsonl
    g2_gate_calibration.jsonl
    g2_smoke_agentic.jsonl
    g2_smoke_route_ablation.jsonl
    g2_formal_agentic.jsonl
    shortfalls.csv
    rejected_rows.jsonl

  source_predictions/
    <sample_id>/
      coarse_mask.png
      coarse_probabilities.npz
      entropy.npy
      fine_mask.png                 # supported rows only
      fine_probabilities.npz        # supported rows only
      cellvit.json
      provenance.json

  generated/
    project_agentic/<sample_id>/
      semantic_change_region.png
      generation_change_region.png
      attempt_01_<route>/
        generated_image.png
        generation_metadata.json
        verification/
          coarse_mask.png
          coarse_probabilities.npz
          entropy.npy
          fine_mask.png             # supported rows only
          cellvit.json
          online_verification.json
          provenance.json
      attempt_02_<route>/            # only if triggered
        ...
      generated_image.png            # selected final
      agentic_workflow.json
      pipeline_summary.json

    fixed_inpaint/<sample_id>/...
    fixed_cross_epoch26/<sample_id>/...

  evaluation/
    per_attempt_metrics.jsonl
    attempt1_manifest.jsonl
    selected_final_manifest.jsonl
    fixed_inpaint_manifest.jsonl
    fixed_cross_manifest.jsonl
    evaluator_uncertain.jsonl
    operational_incomplete.jsonl
    paired_recovery_deltas.csv
    cell_summary.csv
    organ_summary.csv
    primitive_summary.csv
    route_summary.csv
    bootstrap_report.json
    audit_deck/
    final_report.json
```

Every JSON/JSONL output includes schema version, command, code commit, config
hashes, input hashes, and creation time. Batch writers are append-safe or use
atomic temporary-file replacement so an interrupted job remains resumable.

## 13. Required command interfaces

The following are the target batch interfaces. A command marked
`required-new` must exist and pass the tests in Section 15 before the formal
run. Do not substitute an undocumented notebook.

### 13.1 Build and freeze manifests (`required-new`)

```bash
python scripts/build_g2_edit_local_manifests.py \
  --target-bank /absolute/path/to/canonical/target_mask_bank.jsonl \
  --intents /absolute/path/to/canonical/mask_semantic_intents.jsonl \
  --nuclei-root /absolute/path/to/frozen/target_nuclei \
  --freeze-config benchmark_configs/g2_edit_local.yaml \
  --output-root /data1/zhao/wqx/benchmark_v1/g2/manifests
```

The builder validates asset hashes, recomputes `R`, verifies `R subset_of G`,
computes router features, applies deterministic selection, and writes
shortfalls/rejections.

### 13.2 Source evaluator predictions (`required-new`)

```bash
python scripts/predict_g2_segmentator.py \
  --manifest /data1/zhao/wqx/benchmark_v1/g2/manifests/g2_smoke_agentic.jsonl \
  --release /data1/zhao/wqx/benchmark_v1/g2/freeze/segmentator_release.json \
  --image-field reference_image_path \
  --output-root /data1/zhao/wqx/benchmark_v1/g2/source_predictions \
  --save-probabilities \
  --save-entropy \
  --save-fine-when-applicable \
  --skip-existing
```

This batch interface must construct the exact frozen model, load with
`strict=True`, constrain fine outputs by profile, and save probability/entropy
artifacts. The current single-image predictor that saves only
`outputs["pred"]` is insufficient.

### 13.3 Single-condition agentic execution (existing entry point)

```bash
export PATHOLOGY_PIX2PIX_CHECKPOINT=/models/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt

conda run -n pathology-phase5-inpaint \
python scripts/run_agentic_edit_workflow.py \
  --profile BCSS \
  --reference-image /path/reference.png \
  --reference-tissue-mask /path/source_tissue.png \
  --reference-nuclei-mask /path/source_nuclei.png \
  --target-tissue-mask /path/target_tissue.png \
  --target-nuclei-mask /path/target_nuclei.png \
  --semantic-change-region /path/semantic_change_region.png \
  --generation-change-region /path/generation_change_region.png \
  --segmentator-checkpoint /path/final_segmentator.pt \
  --output /path/g2/generated/project_agentic/<sample_id> \
  --device cuda:0 \
  --segmentator-device cuda:1 \
  --cellvit-gpu 1
```

Before formal use, this entry point must consume the Segmentator release
architecture and calibrated verifier policy instead of relying on old
checkpoint/threshold defaults.

### 13.4 Batch generation (`required-new`)

```bash
python scripts/run_g2_edit_local_generation.py \
  --manifest /data1/zhao/wqx/benchmark_v1/g2/manifests/g2_smoke_agentic.jsonl \
  --freeze-manifest /data1/zhao/wqx/benchmark_v1/g2/freeze/g2_freeze_manifest.yaml \
  --arms routed_attempt1 agentic_selected_final \
  --output-root /data1/zhao/wqx/benchmark_v1/g2/generated \
  --resume
```

For the route-ablation cohort:

```bash
python scripts/run_g2_edit_local_generation.py \
  --manifest /data1/zhao/wqx/benchmark_v1/g2/manifests/g2_smoke_route_ablation.jsonl \
  --freeze-manifest /data1/zhao/wqx/benchmark_v1/g2/freeze/g2_freeze_manifest.yaml \
  --arms fixed_inpaint fixed_cross_epoch26 routed_attempt1 agentic_selected_final \
  --output-root /data1/zhao/wqx/benchmark_v1/g2/generated \
  --resume
```

### 13.5 Formal G2 evaluator (`required-new`)

```bash
python -m phase3_mask_edit.cli.run_g2_edit_local_consistency \
  --manifest /data1/zhao/wqx/benchmark_v1/g2/manifests/g2_smoke_agentic.jsonl \
  --freeze-manifest /data1/zhao/wqx/benchmark_v1/g2/freeze/g2_freeze_manifest.yaml \
  --source-pred-root /data1/zhao/wqx/benchmark_v1/g2/source_predictions \
  --generated-root /data1/zhao/wqx/benchmark_v1/g2/generated \
  --arms routed_attempt1 agentic_selected_final \
  --bootstrap-group-field wsi_id \
  --bootstrap-repeats 2000 \
  --output-root /data1/zhao/wqx/benchmark_v1/g2/evaluation
```

The existing conditional-fidelity evaluator computes paired/full-image G1
metrics and must not be used as if it already implements G2 regions,
source-relative preservation, boundary rings, or attempt-level recovery.

## 14. Statistical analysis and reporting

### 14.1 Unit of inference

Per-pixel metrics are first reduced to one value per generated condition.
Confidence intervals use WSI-clustered bootstrap with 2,000 frozen resamples.
Patient ID is preferred when multiple WSIs can belong to one patient. Multiple
attempts and strategy arms for one condition remain paired within every
resample.

Do not treat pixels, nuclei, attempts, directions from the same pair, or
generation seeds as independent observations.

### 14.2 Primary comparisons

On identical sample IDs, report:

```text
agentic selected final - routed attempt 1
fixed Inpaint          - fixed Cross
routed attempt 1       - each fixed route
agentic selected final - each fixed route
```

Primary agentic deltas are:

- changed-region macro mIoU;
- source-relative target gain;
- soft target-versus-source margin;
- Cross-route preservation drift;
- CellViT-derived nuclei count/type consistency.

Report mean/median paired delta, 95% WSI-clustered bootstrap interval, win/tie/
loss rate, and sample/WSI counts. Do not compare arm-level means from
different sample sets.

### 14.3 Recovery accounting

Report:

- initial route distribution and reasons;
- first-pass automatic validation rate;
- alternate-route trigger rate;
- recovery rate among triggered rows;
- selected-final automatic validation rate;
- residual `needs_review` rate;
- fallback regression rate;
- mean attempts and added runtime;
- failure reasons by initial/fallback route;
- attempt-2 minus attempt-1 metric deltas among triggered rows;
- all-cohort selected-final minus attempt-1 deltas, where rows without a
  second attempt have zero selection change.

A fallback regression is a triggered row whose selected-final target fidelity
or required route-specific safety endpoint is worse than attempt 1 under the
frozen clinically meaningful-difference rule.

### 14.4 Stratification

Always report counts and descriptive metrics by:

- organ/profile;
- primitive;
- strength;
- generic versus specialized;
- planned initial route;
- actual selected route;
- semantic change-size stratum;
- transition count;
- evaluator-clean versus evaluator-uncertain.

Cell-level estimates for very small cells are descriptive. The pooled frozen
cohort remains primary.

### 14.5 Realism

G2 Segmentator/CellViT metrics answer conditional fidelity, not image realism.
Patho-KID and blinded realism review remain separate analyses. G2 may report
obvious artifact-review rates and pathology-embedding diversity, but these are
not combined with conditional consistency.

## 15. Verification and phased execution

### 15.1 Required automated tests

Before canary generation, tests must cover:

- semantic region is bitwise `M_src != M_tgt`;
- generation region must contain the semantic region;
- routing ignores a widened generation region;
- same-shape/MPP/orientation enforcement;
- deterministic manifest selection and condition hashes;
- final Segmentator architecture reconstruction and strict checkpoint load;
- coarse probability sums, entropy range, and dataset-constrained fine output;
- DCIS fine exclusion;
- changed-region mIoU penalizes prediction-only hallucinated classes;
- source-relative target gain;
- prediction-relative preservation against `S_src`;
- `B_in`, `B_out`, `U_far`, and Boundary-F1@4;
- zero-target nuclei handling without fake division;
- attempt-1/final paired manifests;
- deterministic fallback/tie-break;
- scientific versus operational failure;
- WSI-clustered paired bootstrap;
- resume without changing successful row hashes;
- explicit epoch-26 pix2pix checkpoint/hash verification.

### 15.2 Phase A: release and interface freeze

Required before any new generation:

1. write the final Segmentator release block and human-validation report;
2. replace the pilot Segmentator in
   `benchmark_configs/conditional_fidelity.yaml` or introduce the dedicated
   `benchmark_configs/g2_edit_local.yaml`;
3. make batch/single inference consume all model architecture flags;
4. export coarse/fine probability and entropy artifacts;
5. implement the G2 metric runner and tests;
6. freeze numeric verifier gates from calibration;
7. write and hash the authoritative G2 freeze manifest.

### 15.3 Phase B: 18-condition canary

The canary must exercise:

- all six organs;
- Inpaint-first, Cross-first, and gray-zone routing;
- semantic/generation region inequality;
- at least one PANDA and one GlaS fine-label row;
- attempt-1 pass;
- tissue-fidelity fallback;
- preservation fallback;
- an injected operational evaluator failure and exact resume.

Canary acceptance requires complete artifact/provenance trees, deterministic
rerun hashes, correct attempt selection, and zero unresolved interface errors.
Canary scientific scores do not tune official thresholds.

### 15.4 Phase C: gate calibration

Run both fixed routes on the 60-condition calibration cohort, complete blinded
review, freeze thresholds, and archive the threshold-selection report. No
official smoke image may be used for this selection.

### 15.5 Phase D: 3-per-cell smoke

Run:

- project routed attempt 1 and selected final on up to 693 conditions;
- all four strategy arms on the deterministic up-to-231 route-ablation
  subset.

Rerun operational incompletes only. Do not alter scientific failures.

Smoke expansion gates:

- every row and artifact hash is traceable to the freeze manifest;
- all 231 cells expose coverage or an explicit source-bank shortfall;
- all operationally recoverable rows complete under identical inputs;
- evaluator and metric outputs are complete for source, attempt 1, and final;
- the blinded calibration false-acceptance policy is still satisfied on the
  prespecified smoke audit;
- selected final shows a prespecified non-inferior or positive paired
  target-fidelity result versus attempt 1 without an unacceptable preservation
  regression;
- fallback gains, regressions, and `needs_review` rows are fully reported.

The exact false-acceptance bound, minimum sensitivity, non-inferiority margin,
and preservation tolerance come from the calibration report and are copied
verbatim into the freeze manifest. They are not invented after smoke results.

### 15.6 Phase E: 10-per-cell focused formal run

Only after smoke acceptance, expand the project agentic system to up to 2,310
conditions. Do not start a 30/50-per-cell or six-model expansion. Repeat fixed
routes only if the smoke confidence intervals show that the 231-row ablation
is underpowered for the routing claim.

## 16. Required result tables

The final G2 report contains at least:

1. frozen cohort coverage and shortfalls by 231 cells;
2. Segmentator real-source evaluability by organ/class;
3. tissue fidelity for fixed Inpaint, fixed Cross, routed attempt 1, and
   selected final on paired IDs;
4. prediction-relative preservation by route;
5. boundary metrics and uncertainty;
6. supported fine-label consistency for PANDA/GlaS;
7. CellViT-derived nuclei count/type/spatial consistency;
8. attempt-1 to selected-final paired deltas;
9. route, recovery, regression, `needs_review`, and runtime accounting;
10. evaluator-clean sensitivity and evaluator-uncertain audit;
11. clinician audit of automatic acceptance/failure;
12. operational failures and exact rerun disposition.

The primary manuscript claim is supported only if the selected-final agentic
system improves or preserves target fidelity relative to routed attempt 1,
the improvement is not purchased through unacceptable off-target drift, and
automatic acceptance is consistent with the frozen blinded audit. A result
that improves an internal weighted score without these component-level results
does not establish G2 effectiveness.

## 17. Implementation status as of 2026-07-29

Already present:

- the benchmark-independent online self-audit package under
  `phase3_mask_edit/audit`, used directly by the production agent;
- P0 probability/confidence/entropy export and source-relative semantic audit
  as a non-destructive product layer;
- the conservative-island P1 candidate in product `shadow` mode, with
  protected semantic boundaries, source-stable components and a strict
  changed-pixel budget;
- deterministic exact-resume support for an already generated attempt;
- an 18-condition engineering canary builder/runner and a complete visual
  audit deck; the canary validates the product loop and cannot tune gates or
  change runtime behavior;
- semantic and generation change-region separation in
  `scripts/run_agentic_edit_workflow.py`;
- routing from the source/target tissue-mask difference;
- bounded two-attempt orchestration and deterministic alternate-route mapping;
- preservation-aware Inpaint versus Cross behavior;
- epoch-26 production pix2pix-v2 documentation;
- G1-oriented conditional-fidelity tissue/nuclei utilities;
- Segmentator human-GT evaluation utility;
- the C epoch-2 engineering release definition and strict release loader;
- single-image and JSONL Segmentator export of coarse `S/P/H`, plus
  dataset-constrained fine predictions/probabilities;
- source-image Segmentator execution before generation;
- P0 source-relative target gain, prediction-relative preservation,
  probability margin, entropy, confidence-coverage, and source-evaluator
  quality features;
- union-of-target-and-prediction changed-region mIoU, including penalties for
  prediction-only hallucinated classes.

The product/benchmark boundary is strict:

1. Online behavior lives in `phase3_mask_edit/audit` and
   `scripts/run_agentic_edit_workflow.py`.
2. G2 consumes the raw/P1 artifacts and decisions emitted by that product
   path.
3. The 18-condition canary checks route coverage, one real fallback, one
   preservation fallback, one evaluator failure with exact resume, and
   per-condition visual evidence.
4. The canary may reject or retain a P1 candidate for later calibration, but
   it cannot enable `enforce`. Enabling P1 requires the separate 60-condition
   calibration cohort plus blinded visual review.

Remaining blockers before formal G2:

1. The P0 confidence and evaluator-clean thresholds are intentionally
   uncalibrated; `evaluator_uncertain` is not assigned and formal
   `validated` status remains disabled until the 60-condition blinded
   calibration freezes them.
2. The current online nuclei proxy measures labelled pixel occupancy, not
   CellViT instance counts.
3. `run_conditional_fidelity_benchmark.py` implements full-image/G1-style
   evaluation and does not yet implement G2 changed/preserved/ring metrics or
   attempt-1 versus selected-final paired recovery in the final batch report.
4. The independent doctor-reviewed Segmentator result, implementation
   `code_commit`, and formal `final_frozen` release status are not yet
   available.
5. The authoritative freeze manifest, manifest builder, CellViT instance
   evaluator integration, deterministic formal tie-break, and clustered
   paired report still need their Phase-A implementations.

The 693-condition smoke must not start until gaps 1-5 are either implemented
or explicitly resolved in a hashed freeze manifest.
