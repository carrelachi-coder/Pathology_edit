# Cross V4 Implementation Audit

Date: 2026-06-03

Reference design: `docs/superpowers/specs/2026-06-02-mask-guided-correspondence-cross-v4-design.md`

## Summary

Cross V4 is implemented as the MVP requested by the design: target masks stay in the ControlNet path, reference image/masks become local FLUX context tokens with token-level metadata, learned coarse tissue prior tokens provide fallback, and a mask-guided additive bias is injected into selected FLUX double-block joint attention logits.

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
| 500-2000 step diagnostics | Done | `--cross-v4-diagnose-steps 500,1000,1500,2000`; JSONL at `cross_v4_diagnostics.jsonl` |
| Hard 80 GB memory guard | Done | `--max-cuda-memory-gb 80` with peak CUDA reserved memory enforcement |
| Checkpoint/inference config parity | Done | `_save_condition_modules()` saves `cross_v4_reference_spec` and `cross_v4_attention_bias`; `pipeline_cross_v4.py` loads them |

## Key Code Changes

### Training Entry

- `controlnet_train/cli/train_controlnet_flux_cross_v4.py`
  - Thin v4 entrypoint that reuses the v3 parser/training loop and forces `args.cross_version = "v4"`.
  - Renames default output/tracker names from v3 to v4.

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

Interpretation at 500-2000 steps:

- Covered target tokens should put most reference-local mass on same-class reference tokens.
- Missing target tokens should prefer matching tissue prior over mismatch reference.
- `ref_zero_minus_normal_denoise_loss` should trend positive when zero-ref is enabled in `REF_SWAP_VARIANTS`.
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
    - `REF_SWAP_VARIANTS=zero,random`
    - `CROSS_V4_DIAGNOSE_STEPS=500,1000,1500,2000`

- `scripts/nohup_train_phase5_cross_v4.sh`
  - One-command background launcher.
  - Writes:
    - nohup log
    - PID file
    - manifest JSON
    - training diagnostic JSONL path

Run:

```bash
bash scripts/nohup_train_phase5_cross_v4.sh
```

Useful live checks:

```bash
tail -f /data/wqx/flowedit/controlnet_cross_v4_mask_guided/nohup_logs/<run>.log
tail -f /data/wqx/flowedit/controlnet_cross_v4_mask_guided/cross_v4_diagnostics.jsonl
```

## MVP-Off But Implemented

These pieces exist in code but are intentionally disabled by script defaults to keep the first 500-2000 step diagnosis clean and under 80 GB:

- route anchors: set `REFERENCE_ROUTE_ANCHOR_MODE=coarse` or `fine` for ablation.
- cell prior tokens: set `CROSS_V4_CELL_PRIOR_TOKENS_PER_CLASS=2`.
- global style tokens: set `CROSS_V4_GLOBAL_STYLE_TOKENS=2`.
- multiple biased double blocks: set `CROSS_V4_BIASED_DOUBLE_BLOCKS=-2,-1` or `all`, but re-check memory.

## Remaining Gaps

- Coverage-aware sampler ratio is not yet enforced in the dataloader.
- Attention visualizations are not yet saved as images; current implementation records numeric attention mass.
- Ref-swap loss is still image-level denoise loss, not region-level covered/missing loss.
- Attention regularizer is not yet added as a training loss.
- Cell histogram/density bias is available in the bias formula, but MVP diagnosis should first verify tissue-only behavior by setting cell-prior/global off.
