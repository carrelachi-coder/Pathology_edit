# Production generation model release

This document pins the private inference artifacts used by the production
pathology edit workflow. The private Hub namespace is `Qinxin11`.

## Authoritative production pix2pix pin

`PATHOLOGY_PIX2PIX_CHECKPOINT` is the **only authoritative runtime selector**
for the production pix2pix postprocessor. Every production and formal
benchmark job must set it explicitly to the packaged orientation-adjusted
artifact:

```bash
export PATHOLOGY_PIX2PIX_CHECKPOINT=/models/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt
```

The resolved checkpoint must report epoch `26`, global step `214895`, trust
gate `nuclei_reference_support_v2`, and SHA256
`be5fe9376efdb5620a57481082f6d5738b6353796fb00fe6e58f6b212ba7c2ac`.
Do not infer the production version from a historical run-directory name, the
local source filename `pilot_step001000.pt`, an epoch-25 baseline reference, or
the Python fallback in `model_paths.py`. Those are implementation/provenance
details and cannot override `PATHOLOGY_PIX2PIX_CHECKPOINT`.

## Authoritative production Segmentator pin

The final online tissue evaluator is the local legacy-anchor release, not the
published C-line comparison:

```bash
export PATHOLOGY_SEGMENTATOR_CHECKPOINT=/models/pathology/pathology-segmentator/legacy_anchor_fine_seed42_best_composite.pt
```

Its authoritative server source is
`/data1/zhao/wqx/segmentator_fine/legacy_anchor_fine_seed42/best_composite.pt`,
size `2,817,774,455` bytes, SHA256
`5165e0fb20aca68f64fb06403eff846173fba3a3373deeedb36b3feb81727b3f`,
and release definition
`benchmark_configs/releases/segmentator_fine_legacy_anchor.json`.
`PATHOLOGY_SEGMENTATOR_CHECKPOINT` may relocate these exact bytes but may not
select a different model.

This checkpoint was initialized from
`/data1/zhao/wqx/segmentator/best_mIoU.pt` (SHA256
`5e4b5587359527e0e988428b8cd7fa453255de8729f0fde1ef375fb21d64f112`).
The subsequent fine stage used `legacy_final_depth`, froze shared/coarse
parameters and optimized only the hierarchical fine task. Consequently, its
coarse endpoint deliberately retains the historical coarse model. The C-line
joint epoch-2 artifact remains available only for research comparison and
manual diagnosis.

## Frozen nuclei-inference release

The production ProbNet artifact is
`Qinxin11/pathology-probnet/best_epoch29_c29607f1b609accb.pt`, pinned at Hub
checkpoint commit `add6970449cf3a94997375a665c832e91b188251`. It is epoch 29
(`epoch=28` in the zero-based checkpoint field), global step `33785`,
115,019,174 bytes, and SHA256
`c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211`.
The compatibility alias `best.pt` is byte-identical.

Only `P(nucleus) = 1 - P(background)` is consumed from this checkpoint as the
spatial landing mass; the local categorical posterior supplies type evidence.
Counts and density calibration come from the frozen pre-edit patch-adaptive
policy documented in `probnet_count_density_training.md`. Historical
density-scale files remaining in the Hub repository are provenance only and
are not production selectors. Candidates use seeded Gumbel ordering from
`odds(P(nucleus))^gamma` without a diversity or coverage-radius term. The
online controller starts at `gamma=3`, audits count/type/spatial agreement and
allows at most one bounded directional gamma update within three attempts.
This is a runtime policy and does not alter the epoch-29 checkpoint.

## Repositories

| Private repository | Contents | Source |
| --- | --- | --- |
| `Qinxin11/pathology-inpaint-controlnet` | FLUX ControlNet config, inference safetensors, Phase 5 conditioning | `/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_inpaint_all` |
| `Qinxin11/pathology-cross-v1-pix2pix` | Cross V1 no-IP/no-UNI export and pix2pix epoch 26 / step 214895 | `/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_cross_v1` and the provenance-only source `/data/wqx/flowedit/pix2pix_texture_transfer_lazy_ver4_wsi_identity_i0_local_full_pyramid_v3_ft/ckpt/pilot_step001000.pt`; production runtime selection still comes exclusively from `PATHOLOGY_PIX2PIX_CHECKPOINT` |
| `Qinxin11/pathology-probnet` | Versioned epoch-29 ProbNet plus byte-identical `best.pt` alias | `/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/best_epoch29_c29607f1b609accb.pt` |
| Local Segmentator release | Final legacy-anchor state dict and strict release JSON; the private Hub C-line artifact is historical comparison only | `/data1/zhao/wqx/segmentator_fine/legacy_anchor_fine_seed42/best_composite.pt` |

The release excludes FLUX.1-dev, UNI, IP-Adapter weights, optimizer/scheduler
state, training data and the dataset-specific nuclei instance libraries.

## Runtime configuration

Downloaded paths can be selected without editing code:

```bash
export PATHOLOGY_INPAINT_CHECKPOINT=/models/pathology/pathology-inpaint-controlnet
export PATHOLOGY_CROSS_V1_CHECKPOINT=/models/pathology/pathology-cross-v1-pix2pix/cross_v1
export PATHOLOGY_PIX2PIX_CHECKPOINT=/models/pathology/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt
export PATHOLOGY_PROBNET_CHECKPOINT=/models/pathology/pathology-probnet/best_epoch29_c29607f1b609accb.pt
export PATHOLOGY_SEGMENTATOR_CHECKPOINT=/models/pathology/pathology-segmentator/legacy_anchor_fine_seed42_best_composite.pt
```

Cross V1 production inference loads neither UNI nor IP-Adapter. Pix2pix loads
its architecture and all local full-pyramid steering/trust settings from the
checkpoint; the fixed high-resolution nuclei policy is
`nuclei_reference_support_v2` with unmatched scale `0.20`, matched floor
`0.60`, four sufficient reference tokens and reference pool size `8`.
Cross output is not passed through LAB transfer, histogram matching, or any
other color-matching postprocess. Same-WSI appearance conditioning remains the
checkpoint-learned identity adapter; it is independent of the nuclei trust
gate and remains enabled.

After Pix2pix, Cross generation applies the inference-only
`cross_rgb_od_low_stain_v1` guard inside the generation change region. It
identifies large, cell-free components using Cross RGB luma/chroma and optical
density, excludes a dilated target-nuclei mask, and feather-blends the
corresponding Stage 1 Cross pixels over the Pix2pix result. This protects
low-stain structures that Pix2pix may otherwise wash out. It is not global
color matching or stain normalization, contains no dataset/organ-specific
constraint, and writes the protection mask, unprotected image, thresholds and
protected fractions into generation provenance.

## Packaging and upload

On `amax2`, build inference-only release folders on the same filesystem as the
large ControlNet files so they can be hard-linked instead of duplicated:

```bash
python scripts/package_generation_models.py \
  --output-root /home/lyw/wqx-DL/flow-edit/hf_generation_release \
  --hf-namespace Qinxin11 \
  --git-commit '<release-commit>'
```

Create the three generation repositories with `--private`, then upload each
directory with `hf upload-large-folder`. For the production Segmentator, copy
the exact legacy-anchor checkpoint together with
`benchmark_configs/releases/segmentator_fine_legacy_anchor.json` as
`release.json`, and verify the SHA256 above after every transfer. The existing
private Hub C-line package must not be treated as the production selector.
Never place an access token in this document, the repository, process arguments
or committed shell scripts.

## Acceptance checks

- All manifest SHA256 values match the source and Hub LFS metadata.
- Hugging Face reports all four model repositories as private.
- Inpaint loads and generates one fixed sample.
- Cross V1 + pix2pix reports epoch `26`, global step `214895`, no IP/UNI, and
  trust gate `nuclei_reference_support_v2` on the fixed validation case.
- ProbNet loads the epoch-29 / step-33785 checkpoint and the online agent audits
  `probnet_odds_mass_without_replacement`, initial/evaluation gamma `3.0`, zero
  diversity/coverage contribution, every feedback attempt and reason code,
  the one-update/three-attempt budget, ProbNet-mass retry and pixel backfill,
  pre-edit count reference, cumulative posterior type routing, component-local
  shape policy, one-pixel spacing, full-instance erasure, buffer retention, its
  frozen SHA, and the absence of organ-specific placement constraints.
- Segmentator reconstructs the legacy-anchor Mask2Former architecture from
  `segmentator_fine_legacy_anchor.json`, strictly loads SHA256 `5165e0f...27b3f`,
  and accepts `PATHOLOGY_SEGMENTATOR_CHECKPOINT` only as a deployment-path
  override for those exact bytes.

## Release verification

- Generation pytest on `amax2`: 162 passed; ProbNet utility smoke tests: 5 passed.
- Hub API verification on 2026-07-29 reports the three generation repositories
  and the historical C-line comparison repository as private. The production
  legacy-anchor Segmentator is instead verified against its local release
  manifest and SHA256 after every transfer; the C-line Hub hash is not a
  production pin.
- The Hub-downloaded Inpaint package loaded with FLUX.1-dev and generated the
  fixed `TCGA-CN-4738...` validation sample.
- The Hub-downloaded Cross V1 package generated Stage 1 without constructing
  UNI or loading IP-Adapter, then reported pix2pix epoch 26 / step 214895 and
  trust gate `nuclei_reference_support_v2`.
- Pix2pix contains 233 tensors bitwise equal to the source checkpoint. Under
  deterministic CUDA, source-package and Hub-round-trip Stage 1 and Stage 2
  fixed-sample outputs are pixel-identical (maximum pixel difference `0`).
- Ordinary BF16 CUDA generation may show small run-to-run numeric variation;
  exact regression comparisons therefore use deterministic CUDA settings.
- The online product provenance tests verify the epoch-29 ProbNet SHA,
  odds-mass sampler, reason-directed bounded feedback trace and exact-count
  retry/backfill contract before the agent accepts a generated nuclei mask.
  The completed 600-case nuclei cohort supports the three-attempt budget
  (595/3/2 cases passing attempts 1/2/3) but is not threshold-calibration or
  independent biological ground truth. The production Segmentator release
  test verifies strict legacy-anchor architecture reconstruction and the
  portable checkpoint override.

The ProbNet runtime metadata and aggregate online product manifest pin
`64de2f5d6d6637053bdb1ea050493fc6fbbfb27b`, which adds the quota-aware
coverage prefix. Cross low-stain protection entered at
`ce7c065de806fdd923aa8d26a2321a39cd4021e6`. Neither change modifies a
released weight. Mutable repository-head revisions are not runtime selectors;
the machine-readable releases pin the immutable ProbNet checkpoint commit
`add6970449cf3a94997375a665c832e91b188251`; the production Segmentator is a
local checkpoint release pinned by its SHA256 rather than the historical
C-line Hub revision.

Primary released weight hashes:

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Inpaint ControlNet | 8,190,001,728 | `402c836c553410355cf2912518f69339d8eb61f1c9cc588d3020367121a6060c` |
| Cross V1 ControlNet | 8,192,950,848 | `b0442d93aa2b2649e3506620c36c4cc54ba55d377f4c7f767f19147ea83d276e` |
| Pix2pix epoch 26 / step 214895 | 230,809,783 | `be5fe9376efdb5620a57481082f6d5738b6353796fb00fe6e58f6b212ba7c2ac` |
| ProbNet epoch 29 / step 33785 | 115,019,174 | `c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211` |
| Production legacy-anchor Segmentator | 2,817,774,455 | `5165e0fb20aca68f64fb06403eff846173fba3a3373deeedb36b3feb81727b3f` |
