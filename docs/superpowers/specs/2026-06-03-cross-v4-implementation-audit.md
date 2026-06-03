# Cross V4 Implementation Audit

Date: 2026-06-03

Reference design: `docs/superpowers/specs/2026-06-02-mask-guided-correspondence-cross-v4-design.md`

## Summary

Cross V4 is implemented as the MVP requested by the design: target masks stay in the ControlNet path, reference image/masks become local FLUX context tokens with token-level metadata, learned coarse tissue prior tokens provide fallback, and a mask-guided additive bias is injected into selected FLUX double-block joint attention logits.

The current v4 launch defaults keep self-reconstruction warmup disabled, but now enable a lightweight same-class ref-swap diagnostic before step 4000. An earlier `SELF_RECONSTRUCTION_WARMUP_STEPS=500` default made the first 500 optimizer steps report `self_reconstruction_samples=1` and `cross_samples=0` for per-GPU batch size 1, which is unsafe for the correspondence MVP because it can train a self-copy shortcut before any cross-pair signal is observed.

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
  - Overrides v4 MVP defaults to use lightweight same-class ref-swap (`weight=0.05`, `interval=100`, `variants=same_class`) and disable cell bias unless the user explicitly passes ablation arguments.

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
| Before step 4000 | Same-class ref-swap sensitivity: `ref_same_class_denoise_loss`, `ref_same_class_minus_normal_denoise_loss`, `ref_same_class_available`, and `ref_swap_loss` | `ref_same_class_available=1` and same-class swap creates a measurable loss delta from the normal reference. | Separates "reference affects the output but V/IP-Adapter may flatten texture" from "reference is effectively unused". This gate should run before interpreting step-5000 visuals. |
| Step 8000+ | Visual texture inspection, same-class reference swap, and zero/random-ref sensitivity if enabled | Nuclear morphology, chromatin granularity, and edge texture follow the reference while target structure remains controlled by the target mask | Final high-frequency test. Combine with the stain and pre-4000 swap gates to decide whether the remaining failure is a V bottleneck or a routing failure. |

For a stricter sample-level check, compute mean color per generated target tissue mask and compare it with the mean color of the matching reference tissue mask. The same-class distance should be lower than mismatched-class distances. This is the visual/metric counterpart of `style_tissue_loss`, which already matches per-region stain statistics through mean/std/covariance terms.

## Failure Diagnosis And Strategy

| Observation | Diagnosis | Strategy |
| --- | --- | --- |
| Step-1 smoke fails, attention diagnostics are missing, or memory exceeds the guard | Wiring or runtime failure | Fix the v4 attention installation, segment offsets, saved config parity, or memory settings before any training interpretation. Do not add IP-Adapter for this case. |
| Around step 1500, reference bandwidth increases to about `0.15-0.17`, same-class ratio is near 1.0, and text/global stays stable | Healthy warmup completion | Let the run continue. This is the current case: bias is working, reference mass rose from image self-attention, and text is not yet proven harmful. |
| Around step 1500, reference-local attention has little bandwidth or same-class mass does not dominate mismatch mass | Bias route is not effective in real training | Audit `cross_v4_bias_*` stats, context segment boundaries, selected biased double blocks, and warmup scale. Then tune `same_fine`, `same_coarse`, `mismatch`, prior biases, or the number of biased blocks. Only test text-bias if the route is otherwise correct but reference bandwidth remains too low. |
| `style_tissue_loss` does not trend down by steps 2000-5000, and step-5000 samples show uniform stain or stain that does not follow reference tissue classes | Failure A: semantic partitioning is not reaching the image | Treat this as a bias/routing problem. Keep IP-Adapter off; first repair correspondence bias strength, block selection, tissue metadata, mask coverage, and target/reference class alignment. |
| `style_tissue_loss` trends down, step-5000 samples show per-class stain transfer, but step-8000 samples still lack reference nuclear texture | Failure B: semantic routing works at low frequency, but the FLUX V projection is flattening high-frequency texture | This is the right case for an IP-Adapter-style independent reference K/V path. Keep Cross V4 bias as the semantic pointer, then add or ablate an IP-Adapter branch to preserve high-frequency reference appearance. |
| Step 8000+ texture fails and the earlier stain gate is also weak despite correct same-class attention | Likely bandwidth ceiling, not an IP-Adapter-first case | Now test a negative text-bias or stronger reference routing to raise reference bandwidth. This is the main case where text should be actively managed. |
| `style_tissue_loss` trends down, but visual samples do not show per-class stain transfer | Loss/mask artifact or whole-image color shortcut | Inspect decoded samples, tissue masks, region counts, and per-class color distances. Verify the loss is not being satisfied by a global stain shift or a mask-label mismatch. |
| Stain and texture both transfer by class | Cross V4 route is working | Continue normal quality tuning: sampler mix, bias strengths, checkpoint selection, and optional cell/global tokens as controlled ablations. |

Text-bias decision rule:

- Do not restart the current run only because `text_global` remains around `0.137`. With `same_fine=+3` and no negative text bias, that is expected softmax behavior, and the prompt still provides a useful histopathology prior.
- If texture transfers, text-bias was unnecessary.
- If stain transfers but texture does not, text-bias is still unlikely to fix the root cause; prioritize the independent reference K/V path.
- If both stain and texture are weak while same-class attention is otherwise correct, then add a `--cross-v4-text-bias`/negative text-bias ablation to test whether more reference bandwidth helps.

Non-invasive preparation for the next branch: add a `--cross-v4-text-bias` switch if both stain and same-class swap are weak. Same-class swap is now part of the default early diagnostic path so texture failures are not conflated with reference-unused failures.

Two caveats matter for interpretation:

- `style_tissue_loss` is an explicitly optimized objective (`REFERENCE_STYLE_LOSS_WEIGHT=1.0` by default), so its decline is evidence of per-region stain matching but not proof that attention alone caused the transfer. If this distinction becomes important, run a short ablation with `REFERENCE_STYLE_LOSS_WEIGHT=0` and check whether per-class stain still follows the reference.
- In the unlikely case that the value projection loses even low-frequency stain, stain failure could look like Failure A while the root cause is still V compression. The practical first move is still to audit bias/routing, because low-frequency stain is the easiest reference signal for a value projection to preserve.

## MVP-Off But Implemented

These pieces exist in code but are intentionally disabled by script defaults to keep the first 500-2000 step diagnosis clean and under 80 GB:

- route anchors: set `REFERENCE_ROUTE_ANCHOR_MODE=coarse` or `fine` for ablation.
- cell prior tokens: set `CROSS_V4_CELL_PRIOR_TOKENS_PER_CLASS=2`.
- global style tokens: set `CROSS_V4_GLOBAL_STYLE_TOKENS=2`.
- multiple biased double blocks: set `CROSS_V4_BIASED_DOUBLE_BLOCKS=-2,-1` or `all`, but re-check memory.
- stronger ref-swap loss: default is lightweight same-class swap (`REF_SWAP_LOSS_WEIGHT=0.05`, `REF_SWAP_LOSS_INTERVAL=100`, `REF_SWAP_VARIANTS=same_class`); set `REF_SWAP_LOSS_WEIGHT=0.1`, `REF_SWAP_LOSS_INTERVAL=1`, and `REF_SWAP_VARIANTS=same_class,zero,random` for explicit stronger sensitivity ablation.

## Remaining Gaps

- Coverage-aware training now uses a `pair_difficulty` weighted sampler; monitor logs to confirm the effective target remains full/partial/low = 70/25/5.
- Attention visualizations are not yet saved as images; current implementation records numeric attention mass.
- Ref-swap loss is still image-level denoise loss, not region-level covered/missing loss.
- Same-class ref-swap is enabled by default for v4 training as a pre-4000 diagnostic. Zero/random swaps remain explicit ablations.
- Attention regularizer is not yet added as a training loss.
- Cell histogram/density bias is available in the bias formula, but MVP diagnosis should first verify tissue-only behavior by setting cell-prior/global off.
