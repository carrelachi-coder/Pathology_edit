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

## Current frozen nuclei-inference override

The local production default for nuclei generation now supersedes the
historical ProbNet artifact in the Hub release table below:

`/data1/zhao/wqx/probnet_density/frozen/epoch29_C3_shape_group_total_count/best_epoch29_c29607f1b609accb.pt`

Its SHA256 is
`c29607f1b609accbb6ee0fceccb9ead02cd266cce67cec1d8df7c0b7da571211`.
Only `P(nucleus) = 1 - P(background)` is consumed from this checkpoint, as a
spatial landing weight. Counts, density calibration and exact nucleus-type
quotas come from the frozen patch-adaptive sampling policy documented in
`probnet_count_density_training.md`. The older Hub ProbNet verification below
is retained as release history and must not be interpreted as the current
runtime sampling contract.

## Repositories

| Private repository | Contents | Source |
| --- | --- | --- |
| `Qinxin11/pathology-inpaint-controlnet` | FLUX ControlNet config, inference safetensors, Phase 5 conditioning | `/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_inpaint_all` |
| `Qinxin11/pathology-cross-v1-pix2pix` | Cross V1 no-IP/no-UNI export and pix2pix epoch 26 / step 214895 | `/home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/controlnet_cross_v1` and the provenance-only source `/data/wqx/flowedit/pix2pix_texture_transfer_lazy_ver4_wsi_identity_i0_local_full_pyramid_v3_ft/ckpt/pilot_step001000.pt`; production runtime selection still comes exclusively from `PATHOLOGY_PIX2PIX_CHECKPOINT` |
| `Qinxin11/pathology-probnet` | ProbNet `best.pt` plus generation/density configs | `/home/lyw/wqx-DL/flow-edit/FlowEdit-main/inpaint_cells/checkpoints/best.pt` |

The release excludes FLUX.1-dev, UNI, IP-Adapter weights, optimizer/scheduler
state, training data and the dataset-specific nuclei instance libraries.

## Runtime configuration

Downloaded paths can be selected without editing code:

```bash
export PATHOLOGY_INPAINT_CHECKPOINT=/models/pathology-inpaint-controlnet
export PATHOLOGY_CROSS_V1_CHECKPOINT=/models/pathology-cross-v1-pix2pix/cross_v1
export PATHOLOGY_PIX2PIX_CHECKPOINT=/models/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt
export PATHOLOGY_PROBNET_CHECKPOINT=/models/pathology-probnet/best.pt
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

## Packaging and upload

On `amax2`, build inference-only release folders on the same filesystem as the
large ControlNet files so they can be hard-linked instead of duplicated:

```bash
python scripts/package_generation_models.py \
  --output-root /home/lyw/wqx-DL/flow-edit/hf_generation_release \
  --hf-namespace Qinxin11 \
  --git-commit '<release-commit>'
```

Create all three repositories with `--private`, then upload each directory with
`hf upload-large-folder`. Run the upload through the SSH reverse proxy when
`amax2` cannot reach Hugging Face directly. Never place an access token in this
document, the repository, process arguments or committed shell scripts.

## Acceptance checks

- All manifest SHA256 values match both the source and a Hub round-trip download.
- Hugging Face reports all three repositories as private.
- Inpaint loads and generates one fixed sample.
- Cross V1 + pix2pix reports epoch `26`, global step `214895`, no IP/UNI, and
  trust gate `nuclei_reference_support_v2` on the fixed validation case.
- ProbNet loads `best.pt` at epoch `61`, step `991504` and generates one mask
  using the existing local nuclei library.

## Release verification

- Generation pytest on `amax2`: 162 passed; ProbNet utility smoke tests: 5 passed.
- Hub API verification reports all three repositories as private. Every file
  listed in each manifest was downloaded from the Hub and rechecked for exact
  byte size and SHA256 equality with the packaged source.
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
- The Hub-downloaded ProbNet loaded epoch 61 / step 991504 and generated an
  ORCA mask with the existing 20,000-instance local nuclei library. Its output
  mask was byte-identical to the source-package smoke output (29/27 placements
  for gamma 2/3).

The manifests pin the production code commit
`6129422cc677d0183f3234ae17b049c76fc57024` on
`codex/generation-model-release`. Hub repository revisions are intentionally
not pinned here because publishing later validation metadata advances them.

Primary released weight hashes:

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Inpaint ControlNet | 8,190,001,728 | `402c836c553410355cf2912518f69339d8eb61f1c9cc588d3020367121a6060c` |
| Cross V1 ControlNet | 8,192,950,848 | `b0442d93aa2b2649e3506620c36c4cc54ba55d377f4c7f767f19147ea83d276e` |
| Pix2pix epoch 26 / step 214895 | 230,809,783 | `be5fe9376efdb5620a57481082f6d5738b6353796fb00fe6e58f6b212ba7c2ac` |
| ProbNet best.pt | 114,836,809 | `8efc4c0100fb0f013e70c64a8a01718ce5d6a2b2646af72878adf5e7726ee2d8` |
