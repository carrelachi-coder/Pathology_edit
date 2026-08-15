# Agentic Pathology Edit Workflow

## Production generation paths

- Local, preservation-sensitive edits: `inpaint`.
- Structural edits: Cross V1 with IP-Adapter and UNI disabled, followed by the
  production pix2pix-v2 postprocessor.
- Production pix2pix-v2 checkpoint:
  `pix2pix/pix2pix_epoch26_step214895.pt` from the frozen
  `Qinxin11/pathology-cross-v1-pix2pix` release, selected at runtime through
  `PATHOLOGY_PIX2PIX_CHECKPOINT`.

`PATHOLOGY_PIX2PIX_CHECKPOINT` is the authoritative source of truth for the
production pix2pix-v2 implementation. Production and formal benchmark jobs
must set it explicitly, and the resolved file must report epoch `26`, global
step `214895`, trust gate `nuclei_reference_support_v2`, and SHA256
`be5fe9376efdb5620a57481082f6d5738b6353796fb00fe6e58f6b212ba7c2ac`.
Neither a historical run-directory name, the provenance source filename
`pilot_step001000.pt`, an epoch-25 baseline reference, nor the fallback in
`model_paths.py` may override this environment variable.

The retired `controlnet_train/pix2pix_transfer_v2` DiT/flow-matching experiment
is unrelated to the production epoch-26 orientation-adjusted postprocessor and
has been removed.

Production inference deliberately ignores the historical `ref_trust_gate`
arguments inherited from the epoch-25 baseline. The gate has no learned
checkpoint weights and is
not constructed or applied. The learned WSI identity adapter remains enabled
because its FiLM and gamma parameters are part of the epoch-26 model state.

The epoch-26 full-pyramid texture-steering checkpoint includes local
orientation steering, keeps the retired broad gate off, and applies the frozen
`nuclei_reference_support_v2` support gate to reference cross-attention at
every scale. Tissue trust remains 1.0. Nuclei updates are attenuated when a
same-class reference is sparse or does not survive the checkpoint's pooled
high-resolution reference tokens, preventing a few cell tokens from being
repeated across many target cells. The WSI identity adapter remains unchanged.

## Control loop

The production loop below describes online execution. The complete formal G2
evaluation contract, including source-image Segmentator calibration,
prediction-relative preservation, changed-region and boundary formulas,
attempt-1 versus selected-final analysis, threshold calibration, manifest
fields, and phased acceptance, is
`docs/g2_edit_local_consistency_runbook.md`.

1. Parse report differences or an edit instruction into structured edit intents.
2. Produce tissue and nuclei target masks through constrained mask tools.
3. Validate target masks before image generation.
4. Extract route features: image/tissue-normalized change ratio, connected
   components, bounding-box coverage, and semantic label transitions.
5. Run the primary generator.
6. Re-segment the generated image with the project segmentator and CellViT.
7. Check changed-region accuracy/macro-IoU, source-relative target gain,
   prediction-relative off-target drift, and CellViT-derived nuclei
   consistency. Formal G2 compares generated and real-source Segmentator
   predictions outside the edit; source-mask-relative drift remains a
   diagnostic.
8. If verification fails, choose the alternate generator from the structured
   failure: off-target drift prefers inpaint; structure/IoU/nuclei failures
   prefer production cross; a tool error uses the remaining backend.
9. Run at most one alternate generator. If neither passes, apply the frozen G2
   lexicographic tie-break and return one candidate with `needs_review`
   instead of silently accepting it. The current weighted score remains a
   pilot implementation detail until the formal selector is implemented.

The loop is implemented in `controlnet_train/inference/agentic.py`, not in an
LLM prompt. Every attempt immediately updates `agentic_workflow.json` with the
route reason, decision reason, artifact, verification metrics, and error. This
makes a partially completed run inspectable even if a verifier process fails.

## Online semantic self-audit

Semantic self-audit is a product capability of the online agent. It is
implemented under `phase3_mask_edit/audit`; benchmark code may call this
package but must not define a different runtime policy.

For every generated candidate, the agent exports the Segmentator coarse
prediction, probabilities, confidence and normalized entropy. For BCSS, GlaS
and PANDA it also exports the dataset-native fine prediction and probabilities.
Metrics include only labels annotated by that dataset, always exclude
Background/Other, and exclude BCSS DCIS because the released Segmentator does
not support that class.

P0 is the non-destructive audit layer. It records source evaluator quality,
changed-region fidelity, off-target preservation, confidence coverage and
uncertainty without changing the prediction. P1 is a bounded postprocessor
candidate: it may relabel only tiny low-confidence, high-entropy islands when
one surrounding class has strong posterior support. It protects semantic
boundaries, source-stable components and a global changed-pixel budget. It
never updates model weights online.

The default product mode is `shadow`: both raw and P1 predictions are exported,
but raw metrics continue to drive pass/fallback. `enforce` remains unavailable
for release decisions until a disjoint calibration cohort and blinded visual
review freeze the P1 policy. G2 and its canaries validate this same online
loop; they do not alter agent functionality or tune scientific thresholds.

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
Cross V1 plus epoch-26 orientation-adjusted pix2pix-v2 path, and requires
CellViT runtime paths for closed-loop nuclei verification.

The online UI in `scripts/phase3_end_to_end_ui.py` keeps its full four-stage
front end (load inputs, edit tissue, synthesize target nuclei, generate), but
does not implement a second agent loop. Its agentic generation button invokes
this standalone entry point with the exact target masks and the separate
semantic/generation regions produced by the first three stages. The UI also:

- selects the versioned `segmentator-fine-legacy-anchor-v1` production release
  rather than a raw checkpoint or historical C-line comparison;
- pins the frozen epoch-29 ProbNet checkpoint by SHA256, uses gamma `3.0`, and
  passes no density-scale JSON;
- requires `PATHOLOGY_PIX2PIX_CHECKPOINT` to select the packaged epoch-26 /
  step-214895 artifact and validates its SHA256 and release metadata;
- surfaces `engineering_pass_uncalibrated` or `needs_review` from the runner
  instead of presenting pilot verifier gates as formal validation.

The CLI keeps two edit regions separate:

- `--semantic-change-region` is used only for verification and defaults to
  `reference_tissue != target_tissue`;
- `--generation-change-region` is passed to the inpaint generator and defaults
  to the semantic region. It may be a wider superset for GlaS whole-gland or
  locally thin edits;
- the historical `--change-region` remains as a deprecated alias for
  `--semantic-change-region`.

Routing normally uses the reference/target tissue-mask difference, independent
of either explicit region path. For a hash-locked
`joint-generation-handoff-v2/v3`, the approved `joint_change` is authoritative
for routing so a nuclei-only edit cannot be mistaken for a no-op. The same
handoff's approved `generation_support` remains authoritative for rendering;
the legacy context bound is not reapplied to that already audited support.

The final image is written to `generated_image.png`; the compact run result is
written to `pipeline_summary.json`. The process exits with code `0` for
`validated`/`noop`, or `2` for `needs_review` so batch jobs can quarantine
unaccepted cases.

```bash
export PATHOLOGY_PIX2PIX_CHECKPOINT=/models/pathology-cross-v1-pix2pix/pix2pix/pix2pix_epoch26_step214895.pt

conda run -n pathology-phase5-inpaint python scripts/run_agentic_edit_workflow.py \
  --profile BCSS \
  --reference-image /path/ref.png \
  --reference-tissue-mask /path/ref_tissue.png \
  --reference-nuclei-mask /path/ref_nuclei.png \
  --target-tissue-mask /path/target_tissue.png \
  --target-nuclei-mask /path/target_nuclei.png \
  --semantic-change-region /path/semantic_change_region.png \
  --generation-change-region /path/generation_change_region.png \
  --output /path/agentic_run \
  --device cuda:0 \
  --segmentator-device cuda:1 \
  --cellvit-gpu 1 \
  --semantic-postprocess-mode shadow
```

Each attempt retains `coarse_mask_raw.png`, `coarse_mask_p1.png`,
`p1_changed_pixels.png` and `online_semantic_audit.json`. Probability and
entropy artifacts remain available for online review, later calibration and
world-model self-audit replay.
