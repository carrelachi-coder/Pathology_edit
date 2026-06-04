# Cross V4 Implementation Audit

Date: 2026-06-03

Reference design: `docs/superpowers/specs/2026-06-02-mask-guided-correspondence-cross-v4-design.md`

Update: 2026-06-04

## Summary

Cross V4 is implemented as the MVP requested by the design: target masks stay in the ControlNet path, reference image/masks become local FLUX context tokens with token-level metadata, learned coarse tissue prior tokens provide fallback, and a mask-guided additive bias is injected into selected FLUX double-block joint attention logits.

The current v4 launch defaults deliberately do not use ref-swap loss or self-reconstruction warmup. An earlier `SELF_RECONSTRUCTION_WARMUP_STEPS=500` default made the first 500 optimizer steps report `self_reconstruction_samples=1` and `cross_samples=0` for per-GPU batch size 1, which is unsafe for the correspondence MVP because it can train a self-copy shortcut before any cross-pair signal is observed. Ref-swap/perceptual losses are implemented, but they are not the current first move while the clean LR experiment is still running.

The 2026-06-04 branch adds a frozen same-WSI appearance perceptual loss. This is not an online GAN discriminator. It is a separately pretrained real-patch pair classifier/appearance encoder, frozen before Cross V4 generator training, and used through intermediate feature distances. The first pretraining run reached validation accuracy `0.877` after 5 epochs and is suitable for a first controlled Cross V4 run with `same_wsi_perceptual_weight=0.05`.

Latest same-class swap observation on the old low-LR run: `zero_ref` and normal reference outputs differ in stain, but reference A and reference B produce the same stain, density, and texture. This rules out a pure FLUX V bottleneck for that old 5e-7/early-checkpoint state, because the reference path can move low-frequency stain when a reference is present. It does not prove that a properly trained 5e-6 run will still ignore A/B. The active experiment must answer that question before adding auxiliary losses.

An early `checkpoint-2000` visual that looks like a natural landscape should be interpreted carefully. `checkpoint-2000` means 2000 training optimizer steps; it is not the diffusion sampling step count. The diagnostic/inference scripts default to `--num-inference-steps 28`. A poor image from `checkpoint-2000` is an early-checkpoint warning, not final proof that Cross V4 or the same-WSI perceptual branch failed. If `checkpoint-5000`/`checkpoint-8000` still produce natural-image/landscape outputs under the same fixed prompt and ControlNet scale, escalate to inference/checkpoint/conditioning diagnostics.

The runnable training path is:

```text
scripts/nohup_train_phase5_cross_v4.sh
  -> scripts/train_phase5_cross_v4.sh
  -> controlnet_train/cli/train_controlnet_flux_cross_v4.py
  -> controlnet_train/training/flux_phase5_cross_v3.py with args.cross_version="v4"
```

## Design Alignment Checklist

| Design requirement | Status | Code |
| --- | --- | --- |
| Target structure path is target-only ControlNet | Done | `build_cross_v4_control_condition()` delegates to v3 target-only condition in `controlnet_train/modules/cross_v4_conditioning.py` |
| Reference local tokens preserve space | Done | `CrossV4ReferenceContextEncoder.forward()` packs `z_ref + ref_tissue_feat + ref_nuclei_feat` with `pack_cross_v3_reference_grid()` |
| Reference token tissue/cell metadata | Done | `build_cross_v4_token_metadata()` returns fine/coarse IDs, confidence, cell histogram, and density |
| Per-class tissue priors | Done | `CrossV4PriorTokenBank`; default is 4 coarse tissue tokens per class |
| Cell prior/global style optional | Implemented but MVP-off | `CrossV4PriorTokenBank`; shell defaults are cell priors `0`, global style `0` |
| Context segment offsets | Done | `CrossV4ContextSegments`, `append_cross_v4_context()` |
| Mask-guided correspondence bias | Done | `build_cross_v4_correspondence_bias()` |
| Bias in FLUX logits, not just embeddings | Done | `FluxCrossV4AttnProcessor2_0._biased_forward()` adds bias to image-query/context-key logits |
| Single biased double block for MVP | Done | `--cross-v4-biased-double-blocks last` default |
| Route anchor off for clean MVP diagnosis | Done | `scripts/train_phase5_cross_v4.sh` default `REFERENCE_ROUTE_ANCHOR_MODE=none` |
| Bias warmup | Done | `_cross_v4_bias_scale()`; shell default warmup 1000 steps |
| Step-1 plus 500-2000 diagnostics | Done | `--cross-v4-diagnose-steps 1,10,100,500,1000,1500,2000`; JSONL at `cross_v4_diagnostics.jsonl` |
| Hard 80 GB memory guard | Done | `--max-cuda-memory-gb 80` with peak CUDA reserved memory enforcement |
| Checkpoint/inference config parity | Done | `_save_condition_modules()` saves `cross_v4_reference_spec` and `cross_v4_attention_bias`; `pipeline_cross_v4.py` loads them |
| Region-aware Cross V4 ref-swap loss | Implemented but gated | v4 ref-swap ranks normal reference below zero/same-class swapped references on target tokens covered by normal reference tissue classes; enable only if the 5e-6 LR experiment still has A/B collapse |
| Same-WSI appearance encoder pretraining | Done | `controlnet_train/cli/train_same_wsi_appearance.py`, `controlnet_train/training/same_wsi_appearance.py` |
| Same-WSI hard-negative sampling | Done | Different WSI/case negatives are sampled with tissue-composition matching to avoid tissue-only shortcuts |
| Frozen same-WSI perceptual loss in Cross V4 | Done | `--same-wsi-perceptual-*` args in `train_controlnet_flux_cross_v3.py`; loss integration in `flux_phase5_cross_v3.py` |
| Current-run loss plotting | Done | `scripts/plot_cross_v4_same_wsi_losses.py` filters fake skipped-step zeros and supports manifest/log time filtering |
| Same-class swap inference diagnostic | Done | `scripts/diagnose_cross_v4_same_class_swap.py` writes `run_config.json`, per-sample panels, and summary rows |

## Key Code Changes

### Training Entry

- `controlnet_train/cli/train_controlnet_flux_cross_v4.py`
  - Thin v4 entrypoint that reuses the v3 parser/training loop and forces `args.cross_version = "v4"`.
  - Renames default output/tracker names from v3 to v4.
  - Overrides v4 defaults to keep self-reconstruction and ref-swap loss off for the clean LR experiment. Explicitly enable `ref_swap_loss_weight=0.1`, `ref_swap_loss_interval=50-100`, `ref_swap_variants=zero,same_class` only after the 5e-6 run still shows A/B collapse at 2000+ steps.

- `controlnet_train/cli/train_controlnet_flux_cross_v3.py`
  - Adds Cross V4 prior/bias CLI options.
  - Adds early diagnosis options:
    - `--cross-v4-diagnose-steps`
    - `--cross-v4-diagnose-interval`
    - `--cross-v4-diagnose-jsonl`
  - Adds memory guard options:
    - `--max-cuda-memory-gb`
    - `--cuda-memory-check-interval`
  - Adds frozen same-WSI perceptual options:
    - `--same-wsi-perceptual-checkpoint`
    - `--same-wsi-perceptual-weight`
    - `--same-wsi-perceptual-interval`
    - `--same-wsi-perceptual-layers`
    - `--same-wsi-perceptual-min-pixels`

- `controlnet_train/cli/train_same_wsi_appearance.py`
  - Pretrains a same-WSI pair classifier/appearance encoder on real Cross metadata patch pairs.
  - Positives are same `dataset::case_id`; negatives are different WSI/case.
  - Hard negatives are chosen by approximate tissue-composition matching, controlled by `--hard-negative-prob`, `--hard-negative-pool-size`, and `--hard-negative-candidate-count`.

- `controlnet_train/training/same_wsi_appearance.py`
  - Defines `SameWSIAppearanceEncoder`, `SameWSIPairClassifier`, checkpoint helpers, and `same_wsi_perceptual_loss()`.
  - The generator loss uses frozen intermediate features, not the same/different classifier head.

### Conditioning And Bias

- `controlnet_train/modules/cross_v4_conditioning.py`
  - Adds `CrossV4ReferenceContextEncoder`.
  - Adds `CrossV4TokenMetadata` and `build_cross_v4_token_metadata()`.
  - Adds `CrossV4PriorTokenBank`.
  - Adds `append_cross_v4_context()` with stable segment offsets.
  - Adds `build_cross_v4_correspondence_bias()` with same-fine, same-coarse, mismatch, cell similarity/density, prior-present, prior-missing, wrong-prior, and cell-prior terms.

- `controlnet_train/training/cross_v4_attention.py`
  - Adds FLUX-compatible attention processor that accepts `cross_v4_bias`.
  - Applies bias only to `logits[:, :, image_queries, context_keys]`.
  - Records optional attention mass diagnostics from the injected layer.

### Training Loop

- `controlnet_train/training/flux_phase5_cross_v3.py`
  - Switches between v3 and v4 by `args.cross_version`.
  - Builds v4 control tensor, reference encoding, target metadata, prior tokens, context, and correspondence bias.
  - Installs v4 attention processors on selected FLUX double blocks.
  - Saves v4 conditioning/prior/bias config into `phase5_conditioning.pt`.
  - Logs early diagnosis snapshots to JSONL.
  - Enforces peak CUDA reserved memory limit.
  - Loads the frozen same-WSI appearance encoder when requested, reuses decoded RGB with the regional style loss path, and logs `same_wsi_perceptual_loss` plus `same_wsi_perceptual_layers`.
  - For Cross V4 ref-swap loss, computes the margin on packed target tokens whose tissue class is covered by the normal reference. `same_class` swap uses dataset-provided same-target/same-case alternate references when available, so batch size 1 still gets an A/B pressure signal.

### Inference And Plotting

- `controlnet_train/inference/pipeline_cross_v4.py`
  - Loads v4 control/reference specs and attention-bias config from `phase5_conditioning.pt`.
  - Installs the same selected v4 attention processors used by training.
  - Fixes `run_cross_v4_bundle(..., prompt=...)` to pass the caller prompt through to sampling instead of always forcing `CROSS_V4_PROMPT`.

- `scripts/diagnose_cross_v4_same_class_swap.py`
  - Runs normal reference, same-class swapped reference, and zero-reference variants for the same target masks.
  - Defaults `--prompt-source fixed` so inference matches the training prompt policy unless an ablation is explicitly requested.
  - Writes `run_config.json` with checkpoint, prompt, `num_inference_steps`, guidance, ControlNet scale, control/reference specs, and attention-bias config.

- `scripts/plot_cross_v4_same_wsi_losses.py`
  - Plots `denoise_loss`, `style_loss`, and `same_wsi_perceptual_loss`.
  - Filters skipped same-WSI loss steps using `same_wsi_perceptual_layers > 0`.
  - Supports `--since-manifest` and `--since-log` to avoid mixing TensorBoard events from reused output directories.

### Same-WSI Pretraining Result

First frozen backbone training command used Cross metadata real patch pairs with hard negatives:

```bash
python -m controlnet_train.cli.train_same_wsi_appearance \
  --train-metadata /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_meta/metadata_cross_train.json \
  --val-metadata /home/lyw/wqx-DL/flow-edit/FlowEdit-main/phase5_runs/cross_meta/metadata_cross_val.json \
  --output-dir /data/wqx/flowedit/same_wsi_appearance \
  --epochs 5 \
  --pairs-per-epoch 20000 \
  --batch-size 64 \
  --hard-negative-prob 0.8 \
  --hard-negative-pool-size 32 \
  --hard-negative-candidate-count 512 \
  --num-workers 4
```

Observed final epoch:

```text
epoch 5 train loss 0.3355 accuracy 0.8593
epoch 5 val   loss 0.3350 accuracy 0.8770
```

Interpretation: good enough for a first frozen perceptual run. It is not yet a final backbone; hard-negative validation and reference-swap sensitivity remain the important follow-up checks.

The most important early diagnostic fields are:

```text
cross_v4_attention_covered_ref_same_total
cross_v4_attention_covered_ref_all_local
cross_v4_attention_covered_ref_mismatch
cross_v4_attention_covered_tissue_prior_target
cross_v4_attention_missing_tissue_prior_target
cross_v4_attention_missing_ref_mismatch
ref_zero_minus_normal_denoise_loss
cross_v4_reference_encoder_grad_norm
cross_v4_prior_token_bank_grad_norm
cuda_peak_memory_reserved_gb
cross_v4_diagnostic_pass
```

Interpretation at step 1:

- Confirm attention diagnostics are present; otherwise the bias processor or diagnose hook is not connected.
- In `CROSS_V4_EXTREME_BIAS_SMOKE=1`, covered target tokens should be dominated by same-class reference mass and missing target tokens should be dominated by matching tissue-prior mass.
- `self_reconstruction_samples` should be 0 and `cross_samples` should be positive unless the dataloader sample itself is counterfactual.
- `cuda_peak_memory_reserved_gb` must stay below 80.

Interpretation at 500-2000 steps:

- Covered target tokens should put most reference-local mass on same-class reference tokens.
- Missing target tokens should prefer matching tissue prior over mismatch reference.
- `ref_zero_minus_normal_denoise_loss` should trend positive only in explicit ablation runs where zero-ref variants are enabled.
- Reference encoder and prior token bank grad norms should be non-zero on diagnostic steps.
- `cuda_peak_memory_reserved_gb` must stay below 80.

### Launch Scripts

- `scripts/train_phase5_cross_v4.sh`
  - Foreground training script.
  - Defaults to MVP-safe settings:
    - `REFERENCE_ROUTE_ANCHOR_MODE=none`
    - `CROSS_V4_CELL_PRIOR_TOKENS_PER_CLASS=0`
    - `CROSS_V4_GLOBAL_STYLE_TOKENS=0`
    - `CROSS_V4_BIASED_DOUBLE_BLOCKS=last`
    - `MAX_CUDA_MEMORY_GB=80`
    - `SELF_RECONSTRUCTION_WARMUP_STEPS=0`
    - `SELF_RECONSTRUCTION_SAMPLE_PROB=0`
    - `REF_SWAP_LOSS_WEIGHT=0`
    - `REF_SWAP_LOSS_INTERVAL=0`
    - `REF_SWAP_VARIANTS=` empty
    - `CROSS_V4_DIAGNOSE_STEPS=1,10,100,500,1000,1500,2000`
    - tissue-only bias: cell similarity, density gap, and cell-prior bias are `0`

- `scripts/nohup_train_phase5_cross_v4.sh`
  - One-command background launcher.
  - Writes:
    - nohup log
    - PID file
    - manifest JSON
    - training diagnostic JSONL path
  - Records effective MVP guard settings in the manifest: self-reconstruction, ref-swap, bias warmup, smoke mode, and memory limit.

Run:

```bash
bash scripts/nohup_train_phase5_cross_v4.sh
```

One-step extreme bias smoke test before a real run:

```bash
CROSS_V4_EXTREME_BIAS_SMOKE=1 bash scripts/nohup_train_phase5_cross_v4.sh
```

This forces:

```text
MAX_TRAIN_STEPS=1
SELF_RECONSTRUCTION_WARMUP_STEPS=0
SELF_RECONSTRUCTION_SAMPLE_PROB=0
REF_SWAP_LOSS_WEIGHT=0
CROSS_V4_DIAGNOSE_STEPS=1
CROSS_V4_BIAS_WARMUP_STEPS=0
CROSS_V4_SAME_FINE_BIAS=50
CROSS_V4_SAME_COARSE_BIAS=50
CROSS_V4_MISMATCH_BIAS=-50
CROSS_V4_PRIOR_MISSING_BIAS=50
CROSS_V4_PRIOR_WRONG_CLASS_BIAS=-50
```

The smoke run is a wiring test, not a training-quality test. It should produce one diagnostic JSONL row showing that the installed biased attention layer recorded attention mass and that the extreme same-class/prior logits affect the expected buckets.

Useful live checks:

```bash
tail -f /data/wqx/flowedit/controlnet_cross_v4_mask_guided/nohup_logs/<run>.log
tail -f /data/wqx/flowedit/controlnet_cross_v4_mask_guided/cross_v4_diagnostics.jsonl
```

Current-run loss plotting:

```bash
python scripts/plot_cross_v4_same_wsi_losses.py \
  --input /data/wqx/flowedit/controlnet_cross_v4_mask_guided/logs \
  --since-manifest /data/wqx/flowedit/controlnet_cross_v4_mask_guided/nohup_logs/<run>.manifest.json \
  --output-dir /data/wqx/flowedit/controlnet_cross_v4_mask_guided/loss_plots_<run> \
  --rolling-window 500 \
  --ylim 0,1 \
  --aux-ylim 0,0.15
```

Use `--since-manifest` or a run-specific output directory whenever the same `logs/` root has been reused. Otherwise TensorBoard event files from older runs can make the loss curves look mixed.

## Next Audit Plan

Do not wait until step 8000 and treat "texture did not transfer" as one undifferentiated failure. Cross V4 now has an earlier diagnostic gate: per-class stain transfer. If the correspondence bias routes target tumor/stroma/etc queries to the matching reference tissue tokens, the FLUX value projection should at least pass low-frequency reference statistics such as stain tone and brightness, even if it loses high-frequency nuclear texture. Therefore, stain that follows the reference by tissue class is evidence that the semantic routing is affecting the output at low frequency.

Current step-1500 readout is positive and should not trigger a restart for text-bias. From step 1000 to 1500, effective bias scale finished warmup (`0.667 -> 1.0`), covered reference bandwidth rose sharply (`ref_all_local 0.066 -> 0.165`, `ref_same_total 0.061 -> 0.163`), and text/global mass stayed nearly flat (`0.136 -> 0.137`). The important signal is not that text stayed present; it is that full-strength bias pulled reference bandwidth up by about 2.5x while keeping `ref_same_total / ref_all_local ~= 0.99`. That means the semantic route is working and the added reference mass is likely coming from image self-attention, not from text. Keep the current run going and let stain/texture decide whether text-bias is actually needed.

| Time | Signal | Pass condition | Decision value |
| --- | --- | --- | --- |
| Step 1 smoke | `cross_v4_diagnostic_pass`, same-class/prior attention mass, memory | Extreme bias smoke passes and `cuda_peak_memory_reserved_gb < 80` | Confirms the bias processor, segment offsets, and diagnostic hook are wired. |
| Around step 1500 | Attention mass in `cross_v4_diagnostics.jsonl`: covered same-class reference vs all reference vs mismatch/prior/text buckets | Covered target tokens put most reference-local mass on same-class reference tokens; missing target tokens prefer matching tissue priors over mismatch reference. For the current run, `ref_all_local ~= 0.165`, `ref_same_total ~= 0.163`, and text/global is stable at `~= 0.137`. | Confirms reference tokens are getting bandwidth and the bias is pointing to the intended semantic class. Do not add text-bias merely because text remains non-zero; `text_global ~= 0.137` is a normal equilibrium for `same_fine=+3` and no negative text bias. |
| Steps 2000-5000 | TensorBoard `style_tissue_loss`, plus `style_tissue_regions` | `style_tissue_regions > 0` and `style_tissue_loss` trends down from the early baseline | Early low-frequency gate. A downward per-region stain loss means generated tissue regions are moving toward the corresponding reference tissue regions. |
| Around step 5000 | Visual checkpoint with target tissue mask overlay | Tumor/stroma/etc regions have distinguishable stain when the reference regions differ, and each generated region visually tracks the matching reference class | Confirms the numeric stain signal is visible and per-class, not just whole-image color drift. |
| Step 8000+ | Visual texture inspection, reference swap, and zero/random-ref sensitivity if enabled | Nuclear morphology, chromatin granularity, and edge texture follow the reference while target structure remains controlled by the target mask | Final high-frequency test. Combine with the stain gate to decide whether the remaining failure is a V bottleneck or a routing failure. |

For a stricter sample-level check, compute mean color per generated target tissue mask and compare it with the mean color of the matching reference tissue mask. The same-class distance should be lower than mismatched-class distances. This is the visual/metric counterpart of `style_tissue_loss`, which already matches per-region stain statistics through mean/std/covariance terms.

### Step 4000 Output Readout

Current step-4000 visual/swap result is negative at the output level:

- Stain does not migrate toward the reference. There is visible perturbation, but the color shift is not directional toward the reference region.
- Texture is effectively identical across `ref_normal`, `ref_swap`, and `zero_ref`.

This does not contradict the earlier gradient and attention diagnostics. It means the plumbing can be alive and the attention route can be numerically biased, while the decoded image still does not use reference content in a perceptual way. Treat step 4000 as a stronger warning than the step-1500 attention readout, but not yet as final proof of architectural failure.

Two live hypotheses remain:

1. Training-time hypothesis: step 4000 may be too early for the model to learn the semantic meaning of the new reference route. Continue to step 8000 before deciding whether Cross V4 cannot use the reference.
2. Objective-mismatch hypothesis: the current denoise/L2-style objective may not align with human notions of "same stain", "same texture", or "reference migration". The model can reduce loss while preserving target reconstruction priors and ignoring reference appearance, especially when target/reference are same-case and the target GT already looks plausible without reference.

The proposed frozen same-WSI appearance discriminator is feasible and directly targets hypothesis 2. It should be treated as a separately pretrained perceptual backbone, not as an online GAN discriminator. Train it only on real patch pairs to classify or metrically separate same-WSI vs different-WSI pairs, freeze it, then use intermediate features as a pathology-specific perceptual loss between generated output regions and same-class reference regions. This gives the generator a feature space that is forced to retain stain, scanner/style, nuclear texture, and local tissue appearance cues that UNI2-h may discard.

This branch is now implemented and running as the first non-adversarial objective fix. The current launch used:

```text
same_wsi_perceptual_checkpoint=/data/wqx/flowedit/same_wsi_appearance/best.pt
same_wsi_perceptual_weight=0.05
same_wsi_perceptual_interval=2
same_wsi_perceptual_layers=1,2,3
```

`same_wsi_perceptual_weight=0.05` means the raw frozen-feature loss is multiplied by `0.05` before being added to total loss. It is not a guarantee that the contribution is 5% of denoise loss.

Immediate interpretation:

- Do not use this result to restart only for text-bias; the symptom is not "text too high" by itself.
- If step 8000 still shows `ref_normal == ref_swap == zero_ref`, the current Cross V4 MVP is failing to make reference appearance causally affect output.
- If `style_tissue_loss` is also not directionally improving toward reference, prioritize loss/supervision and reference-sensitivity diagnostics before tuning bias magnitudes.
- If attention routing remains healthy but output-level samples ignore reference appearance, prepare a frozen same-WSI perceptual-loss branch before escalating to adversarial training.

### Frozen Same-WSI Appearance Perceptual Branch

This branch is a good fit for Cross V4 because it aligns the loss with the actual training signal: two real patches from the same WSI should share H&E stain, scanner characteristics, preparation artifacts, and tissue texture statistics even when their spatial layouts differ.

Pretraining task:

```text
Input: patch_a, patch_b
Label: same WSI/case vs different WSI/case
Model: Siamese/two-tower appearance encoder plus a same/different head
Optional auxiliary loss: supervised contrastive or triplet loss
```

Sampling requirements:

- Positive pairs must come from different coordinates in the same WSI, with enough layout diversity to avoid memorizing local structure.
- Negative pairs must include same tissue-class pairs from different WSIs; otherwise the encoder may solve the task by tissue composition instead of WSI appearance.
- Use WSI-level train/validation split and report hard-negative performance, not only random-pair AUC.
- Use augmentations carefully: enough to suppress compression/background shortcuts, not so much that stain and texture cues are destroyed.

Generator loss after pretraining:

```text
E_wsi frozen
loss_wsi_perceptual =
  distance(E_wsi.mid(generated_rgb), E_wsi.mid(reference_rgb))
```

Use it region-aware by tissue class where possible: compare generated target tumor/stroma/etc regions against reference regions with the same class, and skip strong matching for target classes missing from the reference. Start with a small weight (`0.05-0.2`) and the same decode interval used by `style_tissue_loss`.

Why it is preferable to the failed UNI2-h perceptual loss:

- UNI2-h is optimized for high-level pathology representation and may intentionally collapse away stain/texture nuisance factors.
- The earlier Perceiver/general perceptual branch produced severe color collapse: outputs shifted into a dark-purple, visually frightening stain range. Treat that as evidence that the feature/loss space was pushing H&E appearance in the wrong direction, not merely as a small weight-tuning issue.
- The same-WSI task makes stain, scanner style, and local texture predictive, so intermediate features should preserve exactly the appearance details Cross V4 needs.
- Because the encoder is frozen and pretrained outside the generator loop, it avoids the instability of GAN-style adversarial training.

Risks and checks:

- If the encoder learns WSI identity shortcuts that do not survive generated images, the loss can be noisy or misleading.
- If it overweights global stain, it may duplicate `regional_stain_style_loss` instead of adding high-frequency texture pressure.
- Validate with reference-swap sensitivity: generated outputs should move in feature distance toward the chosen same-class reference and away from swapped/zero references.
- Keep this as an objective branch; if same-class attention is not routed correctly, fix Cross V4 bias/metadata first.

Current operational rule:

- Start at `0.05`, not `0.5`, for overnight runs.
- Compare `denoise_loss`, `style_loss`, and `same_wsi_perceptual_loss` on the same run-filtered plot before increasing the weight.
- If same-WSI loss is nonzero only every interval step and zero on skipped steps, that is expected logging behavior; use `same_wsi_perceptual_layers > 0` when plotting the actual active loss values.

## Inference Sanity And "Landscape" Outputs

The word "step" is overloaded:

```text
checkpoint-2000        means training optimizer step 2000
--num-inference-steps  means diffusion sampling steps, default 28
```

If a sample from `checkpoint-2000` looks like a natural landscape, classify it as an early-checkpoint inference warning first. Do not immediately blame the same-WSI loss or restart only for text-bias. The correct next checks are:

1. Confirm whether the image came from training-time preview or standalone inference with `--checkpoint .../checkpoint-2000`.
2. Inspect `run_config.json` from `scripts/diagnose_cross_v4_same_class_swap.py` for checkpoint, prompt, guidance, `controlnet_conditioning_scale`, Cross V4 specs, and bias config.
3. Keep `--prompt-source fixed` unless deliberately testing prompt ablations; it matches training's fixed `"histopathology image"` prompt.
4. Compare `--num-inference-steps 28` and `50` on the same checkpoint. If both are natural-image-like, the main issue is not sampler step count.
5. Compare `controlnet_conditioning_scale=0` and `1.0`. If outputs are visually or numerically nearly identical, the ControlNet/reference condition is not causally affecting sampling.
6. Re-run the same diagnostic at `checkpoint-5000` and `checkpoint-8000`. Persistent natural-image outputs at later checkpoints are a hard failure; `checkpoint-2000` alone is not.

Because the output directory has been reused, avoid ambiguous checkpoint paths. Prefer explicit paths such as:

```text
/data/wqx/flowedit/controlnet_cross_v4_mask_guided/checkpoint-2000
/data/wqx/flowedit/controlnet_cross_v4_mask_guided/checkpoint-5000
```

For future overnight runs, prefer a run-specific `CROSS_V4_OUTPUT_DIR` to avoid mixed logs/checkpoints.

## LR Experiment Gate

The 5e-6 learning-rate run is mandatory and must not be skipped. The prior token probe (`relative_l2 ~= 0.976`) and the observed `zero_ref`/normal stain difference plus A/B collapse were from the old low-LR state, especially `checkpoint-4000-old-version` with conditioning LR `5e-7`. Those results show what the undertrained model was doing; they do not answer whether a sufficiently trained 5e-6 model will learn to distinguish reference A/B by itself.

Estimated runtime note for the active run: at roughly `11.8 s/it` on 3 GPUs, 2000 optimizer steps is about 6-7 hours. Bias warmup must reach at least 1500 steps before the swap result is meaningful.

Old observed output pattern:

```text
zero_ref vs normal_ref: stain differs
normal reference A vs same-class reference B: stain, density, and texture are the same
```

What this proves for the old 5e-7 model:

- This is not the "all identical" signature of a hard V bottleneck. A hard V bottleneck would make zero reference and normal reference nearly identical even in low-frequency stain.
- The reference path is physically capable of affecting output, at least through low-frequency stain.
- In that old run, the model was not using reference content. It learned a content-independent "reference exists" offset, mostly a fixed stain shift.

What this does not prove:

- It does not prove stage 6 remains the root cause after conditioning LR is raised from `5e-7` to `5e-6`.
- It does not prove ref-swap loss is needed.
- It does not justify adding ref-swap/perceptual before the LR-only experiment produces a 2000+ step same-class swap result.

Current gate:

```text
1. Let the 5e-6 run reach >=1500 steps so bias warmup is complete.
2. At ~1500 steps, inspect ref_all / ref_same / text_global and confirm routing remains healthy.
3. At >=2000 steps, run no_grad same-class swap inference: normal A, same-class B, and zero_ref.
4. If A/B separate, diagnose the old collapse as undertraining and do not add ref-swap, perceptual, or K/V.
5. If A/B are still identical, then stage 6 objective pressure is confirmed for the high-LR run; only then enable region-aware ref-swap loss.
```

Operational decision:

- Do not interrupt or contaminate the running 5e-6 experiment with ref-swap/perceptual. The clean one-variable question is whether `5e-7 -> 5e-6` alone makes A/B separate.
- Do not add ref-swap loss until the 2000+ step swap still shows A/B collapse.
- If ref-swap makes A/B separate but high-frequency texture is still weak, then add region-aware same-WSI perceptual at small weight.
- Only revisit K/V if explicit supervision is active and A/B outputs still remain unchanged.

Report checkpoints for the active LR experiment:

```text
~1500 steps: report ref_all / ref_same / text_global to verify routing remains healthy.
>=2000 steps: run no_grad swap inference with normal A / same-class B / zero_ref.
```

## Failure Diagnosis And Strategy

| Observation | Diagnosis | Strategy |
| --- | --- | --- |
| Step-1 smoke fails, attention diagnostics are missing, or memory exceeds the guard | Wiring or runtime failure | Fix the v4 attention installation, segment offsets, saved config parity, or memory settings before any training interpretation. Do not add IP-Adapter for this case. |
| Around step 1500, reference bandwidth increases to about `0.15-0.17`, same-class ratio is near 1.0, and text/global stays stable | Healthy warmup completion | Let the run continue. This is the current case: bias is working, reference mass rose from image self-attention, and text is not yet proven harmful. |
| Around step 1500, reference-local attention has little bandwidth or same-class mass does not dominate mismatch mass | Bias route is not effective in real training | Audit `cross_v4_bias_*` stats, context segment boundaries, selected biased double blocks, and warmup scale. Then tune `same_fine`, `same_coarse`, `mismatch`, prior biases, or the number of biased blocks. Only test text-bias if the route is otherwise correct but reference bandwidth remains too low. |
| `style_tissue_loss` does not trend down by steps 2000-5000, and step-5000 samples show uniform stain or stain that does not follow reference tissue classes | Failure A: semantic partitioning is not reaching the image | Treat this as a bias/routing problem. Keep IP-Adapter off; first repair correspondence bias strength, block selection, tissue metadata, mask coverage, and target/reference class alignment. |
| `checkpoint-2000` standalone inference looks like a natural landscape | Early checkpoint or inference/config warning | Do not stop on this alone. Confirm `run_config.json`, distinguish training step from `--num-inference-steps`, compare 28 vs 50 sampling steps, and retest 5000/8000. |
| `checkpoint-5000`/`checkpoint-8000` still looks like a natural landscape under fixed prompt and scale 1.0 | Hard inference/conditioning or checkpoint failure | Check checkpoint mixing, ControlNet residual norms, scale 0 vs 1 sensitivity, reference context entry into transformer, and prompt/guidance dominance. |
| Old 5e-7 checkpoint: `zero_ref` and normal reference differ in stain, but same-class reference A/B produce identical stain, density, and texture | Old-run output-level reference collapse; likely undertraining vs objective pressure still unresolved for high LR | Do not add auxiliary loss yet. Let the 5e-6 run reach bias-warmup completion and perform the 2000+ no-grad same-class swap. |
| 5e-6 run at >=2000 steps: A/B separate under same-class swap | Old collapse was undertraining | Stop the failure branch. Keep the LR-only setup; do not add ref-swap, perceptual, or K/V. |
| 5e-6 run at >=2000 steps: A/B still identical | Stage 6 objective-pressure failure confirmed for the sufficiently trained run | Now enable sparse region-aware ref-swap loss. If it separates A/B but high-frequency texture remains weak, add small region-aware same-WSI perceptual. |
| Step 4000 shows stain perturbation but no directional movement toward reference, and `ref_normal`, `ref_swap`, `zero_ref` textures are identical | Output-level reference insensitivity; too early vs objective mismatch still unresolved | Continue to step 8000 for the timing hypothesis, but prepare loss/sensitivity audits. Check whether `style_tissue_loss` is genuinely directional by class, whether ref-swap is region-aware enough, and whether denoise/L2 training permits ignoring reference appearance. Start pretraining the frozen same-WSI appearance encoder as the likely non-adversarial objective fix. |
| Same-class attention is healthy, but UNI2-h perceptual/ref-swap/style losses do not create reference-following texture | Objective feature space is wrong for H&E appearance | Replace UNI2-h perceptual loss with the frozen same-WSI appearance perceptual loss. Use intermediate features and region-aware same-class matching; keep the encoder frozen and do not train it adversarially with the generator. |
| `style_tissue_loss` trends down, step-5000 samples show per-class stain transfer, but step-8000 samples still lack reference nuclear texture | Failure B: semantic routing works at low frequency, but the FLUX V projection is flattening high-frequency texture or the objective lacks high-frequency pressure | First try the frozen same-WSI appearance perceptual branch if not already tested. If feature loss confirms generated images still cannot move toward reference despite healthy routing, then add or ablate an IP-Adapter-style independent reference K/V path. |
| Step 8000+ texture fails and the earlier stain gate is also weak despite correct same-class attention | Likely bandwidth ceiling, not an IP-Adapter-first case | Now test a negative text-bias or stronger reference routing to raise reference bandwidth. This is the main case where text should be actively managed. |
| `style_tissue_loss` trends down, but visual samples do not show per-class stain transfer | Loss/mask artifact or whole-image color shortcut | Inspect decoded samples, tissue masks, region counts, and per-class color distances. Verify the loss is not being satisfied by a global stain shift or a mask-label mismatch. |
| Stain and texture both transfer by class | Cross V4 route is working | Continue normal quality tuning: sampler mix, bias strengths, checkpoint selection, and optional cell/global tokens as controlled ablations. |

Text-bias decision rule:

- Do not restart the current run only because `text_global` remains around `0.137`. With `same_fine=+3` and no negative text bias, that is expected softmax behavior, and the prompt still provides a useful histopathology prior.
- If texture transfers, text-bias was unnecessary.
- If stain transfers but texture does not, text-bias is still unlikely to fix the root cause; prioritize the independent reference K/V path.
- If both stain and texture are weak while same-class attention is otherwise correct, then add a `--cross-v4-text-bias`/negative text-bias ablation to test whether more reference bandwidth helps.

Non-invasive preparation for the next branch: add a `--cross-v4-text-bias` switch and a same-class swap diagnostic, but do not use either to interrupt the current run. These are contingency tools for the bandwidth-failure branch, not evidence that the current run should be reset.

Two caveats matter for interpretation:

- `style_tissue_loss` is an explicitly optimized objective (`REFERENCE_STYLE_LOSS_WEIGHT=1.0` by default), so its decline is evidence of per-region stain matching but not proof that attention alone caused the transfer. If this distinction becomes important, run a short ablation with `REFERENCE_STYLE_LOSS_WEIGHT=0` and check whether per-class stain still follows the reference.
- In the unlikely case that the value projection loses even low-frequency stain, stain failure could look like Failure A while the root cause is still V compression. The practical first move is still to audit bias/routing, because low-frequency stain is the easiest reference signal for a value projection to preserve.

## MVP-Off But Implemented

These pieces exist in code but are intentionally disabled by script defaults to keep the first 500-2000 step diagnosis clean and under 80 GB:

- route anchors: set `REFERENCE_ROUTE_ANCHOR_MODE=coarse` or `fine` for ablation.
- cell prior tokens: set `CROSS_V4_CELL_PRIOR_TOKENS_PER_CLASS=2`.
- global style tokens: set `CROSS_V4_GLOBAL_STYLE_TOKENS=2`.
- multiple biased double blocks: set `CROSS_V4_BIASED_DOUBLE_BLOCKS=-2,-1` or `all`, but re-check memory.

These pieces are implemented but must stay off until the 5e-6 LR gate fails:

- region-aware ref-swap loss: enable only after 5e-6 `>=2000` same-class swap still shows A/B collapse. Suggested first setting: `REF_SWAP_LOSS_WEIGHT=0.1`, `REF_SWAP_LOSS_INTERVAL=50-100`, `REF_SWAP_VARIANTS=zero,same_class`; use `--gradient-checkpointing` if peak memory is tight.

## Remaining Gaps

- Coverage-aware training now uses a `pair_difficulty` weighted sampler; monitor logs to confirm the effective target remains full/partial/low = 70/25/5.
- Attention visualizations are not yet saved as images; current implementation records numeric attention mass.
- Ref-swap is region-aware for normal-reference covered target tokens, but it is still a denoise-space ranking loss. Output-space A/B sensitivity should still be checked with `scripts/diagnose_cross_v4_same_class_swap.py`.
- Attention regularizer is not yet added as a training loss.
- Same-WSI appearance discriminator/perceptual encoder is implemented and integrated. Remaining work is better hard-negative validation, reference-swap feature-distance reporting, and weight/layer ablation.
- Cell histogram/density bias is available in the bias formula, but MVP diagnosis should first verify tissue-only behavior by setting cell-prior/global off.
- Checkpoint/log directory reuse can mix old and current runs. Prefer run-specific output directories, or always filter plots by manifest and point inference to explicit `checkpoint-*` directories.
