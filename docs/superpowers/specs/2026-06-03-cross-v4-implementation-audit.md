# Cross V4 Implementation Audit

Date: 2026-06-03

Reference design: `docs/superpowers/specs/2026-06-02-mask-guided-correspondence-cross-v4-design.md`

## Summary

Cross V4 is implemented as the MVP requested by the design: target masks stay in the ControlNet path, reference image/masks become local FLUX context tokens with token-level metadata, learned coarse tissue prior tokens provide fallback, and a mask-guided additive bias is injected into selected FLUX double-block joint attention logits.

The current v4 launch defaults deliberately do not use ref-swap loss or self-reconstruction warmup. An earlier `SELF_RECONSTRUCTION_WARMUP_STEPS=500` default made the first 500 optimizer steps report `self_reconstruction_samples=1` and `cross_samples=0` for per-GPU batch size 1, which is unsafe for the correspondence MVP because it can train a self-copy shortcut before any cross-pair signal is observed.

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

## Key Code Changes

### Training Entry

- `controlnet_train/cli/train_controlnet_flux_cross_v4.py`
  - Thin v4 entrypoint that reuses the v3 parser/training loop and forces `args.cross_version = "v4"`.
  - Renames default output/tracker names from v3 to v4.
  - Overrides v4 MVP defaults to disable ref-swap loss and cell bias unless the user explicitly passes ablation arguments.

- `controlnet_train/cli/train_controlnet_flux_cross_v3.py`
  - Adds Cross V4 prior/bias CLI options.
  - Adds early diagnosis options:
    - `--cross-v4-diagnose-steps`
    - `--cross-v4-diagnose-interval`
    - `--cross-v4-diagnose-jsonl`
  - Adds memory guard options:
    - `--max-cuda-memory-gb`
    - `--cuda-memory-check-interval`

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
- The same-WSI task makes stain, scanner style, and local texture predictive, so intermediate features should preserve exactly the appearance details Cross V4 needs.
- Because the encoder is frozen and pretrained outside the generator loop, it avoids the instability of GAN-style adversarial training.

Risks and checks:

- If the encoder learns WSI identity shortcuts that do not survive generated images, the loss can be noisy or misleading.
- If it overweights global stain, it may duplicate `regional_stain_style_loss` instead of adding high-frequency texture pressure.
- Validate with reference-swap sensitivity: generated outputs should move in feature distance toward the chosen same-class reference and away from swapped/zero references.
- Keep this as an objective branch; if same-class attention is not routed correctly, fix Cross V4 bias/metadata first.

## Failure Diagnosis And Strategy

| Observation | Diagnosis | Strategy |
| --- | --- | --- |
| Step-1 smoke fails, attention diagnostics are missing, or memory exceeds the guard | Wiring or runtime failure | Fix the v4 attention installation, segment offsets, saved config parity, or memory settings before any training interpretation. Do not add IP-Adapter for this case. |
| Around step 1500, reference bandwidth increases to about `0.15-0.17`, same-class ratio is near 1.0, and text/global stays stable | Healthy warmup completion | Let the run continue. This is the current case: bias is working, reference mass rose from image self-attention, and text is not yet proven harmful. |
| Around step 1500, reference-local attention has little bandwidth or same-class mass does not dominate mismatch mass | Bias route is not effective in real training | Audit `cross_v4_bias_*` stats, context segment boundaries, selected biased double blocks, and warmup scale. Then tune `same_fine`, `same_coarse`, `mismatch`, prior biases, or the number of biased blocks. Only test text-bias if the route is otherwise correct but reference bandwidth remains too low. |
| `style_tissue_loss` does not trend down by steps 2000-5000, and step-5000 samples show uniform stain or stain that does not follow reference tissue classes | Failure A: semantic partitioning is not reaching the image | Treat this as a bias/routing problem. Keep IP-Adapter off; first repair correspondence bias strength, block selection, tissue metadata, mask coverage, and target/reference class alignment. |
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
- ref-swap loss: set `REF_SWAP_LOSS_WEIGHT=0.1`, `REF_SWAP_LOSS_INTERVAL=1`, and `REF_SWAP_VARIANTS=zero,random` for explicit sensitivity ablation.

## Remaining Gaps

- Coverage-aware training now uses a `pair_difficulty` weighted sampler; monitor logs to confirm the effective target remains full/partial/low = 70/25/5.
- Attention visualizations are not yet saved as images; current implementation records numeric attention mass.
- Ref-swap loss is still image-level denoise loss, not region-level covered/missing loss.
- Ref-swap is disabled by default for v4 training; use it as an explicit ablation/diagnostic only.
- Attention regularizer is not yet added as a training loss.
- Same-WSI appearance discriminator/perceptual encoder is not yet implemented. It needs a real-patch pair pretraining pipeline, hard-negative validation, frozen feature extraction, and region-aware generator loss integration.
- Cell histogram/density bias is available in the bias formula, but MVP diagnosis should first verify tissue-only behavior by setting cell-prior/global off.
