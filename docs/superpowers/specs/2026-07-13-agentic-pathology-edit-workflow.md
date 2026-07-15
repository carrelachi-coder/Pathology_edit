# Agentic Pathology Edit Workflow

## Production generation paths

- Local, preservation-sensitive edits: `inpaint`.
- Structural edits: Cross V1 with IP-Adapter and UNI disabled, followed by the
  production pix2pix-v2 postprocessor.
- Production pix2pix-v2 checkpoint:
  `/data/wqx/flowedit/pix2pix_texture_transfer_lazy_ver4_wsi_identity_ft/ckpt/epoch0025.pt`.

The checkpoint name is the source of truth for the pix2pix-v2 implementation.
The retired `controlnet_train/pix2pix_transfer_v2` DiT/flow-matching experiment
is unrelated to the production epoch25 postprocessor and has been removed.

Production inference deliberately ignores the historical `ref_trust_gate`
arguments stored in epoch25. The gate has no learned checkpoint weights and is
not constructed or applied. The learned WSI identity adapter remains enabled
because its FiLM and gamma parameters are part of the epoch25 model state.

The full-pyramid texture-steering checkpoint keeps the retired broad gate off,
but applies an inference-only nuclei support gate to reference cross-attention
at every scale. Tissue trust remains 1.0. Nuclei updates are attenuated when a
same-class reference is sparse or does not survive the checkpoint's pooled
high-resolution reference tokens, preventing a few cell tokens from being
repeated across many target cells. The WSI identity adapter remains unchanged.

## Control loop

1. Parse report differences or an edit instruction into structured edit intents.
2. Produce tissue and nuclei target masks through constrained mask tools.
3. Validate target masks before image generation.
4. Extract route features: image/tissue-normalized change ratio, connected
   components, bounding-box coverage, and semantic label transitions.
5. Run the primary generator.
6. Re-segment the generated image with the project segmentator and CellViT.
7. Check changed-region accuracy/macro-IoU, off-target drift, and CellViT-derived
   nuclei density error.
8. If verification fails, choose the alternate generator from the structured
   failure: off-target drift prefers inpaint; structure/IoU/nuclei failures
   prefer production cross; a tool error uses the remaining backend.
9. Run at most one alternate generator. If neither passes, return the best
   scored candidate with `needs_review` instead of silently accepting it.

The loop is implemented in `controlnet_train/inference/agentic.py`, not in an
LLM prompt. Every attempt immediately updates `agentic_workflow.json` with the
route reason, decision reason, artifact, verification metrics, and error. This
makes a partially completed run inspectable even if a verifier process fails.

## Default routing policy

- `change_ratio_tissue <= 0.12` plus compact geometry: inpaint first.
- `change_ratio_tissue >= 0.30`, more than two label transitions, or a widely
  distributed edit: production Cross V1 + pix2pix-v2 first.
- Gray zone: inpaint first and production cross as the single allowed fallback.

These are initial policy values. They should later be calibrated using paired
inpaint/cross benchmark rows rather than adjusted by prompts.

## Standalone entry point

`scripts/run_agentic_edit_workflow.py` runs the generation/verification loop
over already prepared target tissue and nuclei masks. It uses the formal no-IP
Cross V1 plus epoch25 pix2pix-v2 path, and requires CellViT runtime paths for
closed-loop nuclei verification.

The final image is written to `generated_image.png`; the compact run result is
written to `pipeline_summary.json`. The process exits with code `0` for
`validated`/`noop`, or `2` for `needs_review` so batch jobs can quarantine
unaccepted cases.

```bash
conda run -n pathology-phase5-inpaint python scripts/run_agentic_edit_workflow.py \
  --profile BCSS \
  --reference-image /path/ref.png \
  --reference-tissue-mask /path/ref_tissue.png \
  --reference-nuclei-mask /path/ref_nuclei.png \
  --target-tissue-mask /path/target_tissue.png \
  --target-nuclei-mask /path/target_nuclei.png \
  --output /path/agentic_run \
  --device cuda:0 \
  --segmentator-device cuda:1 \
  --cellvit-gpu 1
```
