Production generation models
============================

The repository keeps four supported generation paths:

1. Phase 5 inpaint ControlNet for local, preservation-sensitive edits.
2. Cross V1 ControlNet in strict no-IP/no-UNI inference mode.
3. The checkpoint-faithful pix2pix postprocessor with local full-pyramid
   texture steering and nuclei-reference trust.
4. ProbNet for target nuclei probability prediction and placement.

Production inference
--------------------

The end-to-end entry point is `scripts/run_phase3_inpaint_pipeline.py`.
`--generation-mode auto` routes small changes to inpaint and larger changes to
Cross V1 followed by pix2pix. The direct Cross V1 entry point is
`scripts/generate_cross_v1_no_ip_strict.py`.

Model paths are defined in `controlnet_train/inference/model_paths.py` and can
be overridden without editing code:

- `PATHOLOGY_INPAINT_CHECKPOINT`
- `PATHOLOGY_CROSS_V1_CHECKPOINT`
- `PATHOLOGY_PIX2PIX_CHECKPOINT`
- `PATHOLOGY_PROBNET_CHECKPOINT`

Pix2pix architecture, steering, identity and nuclei trust settings are loaded
from the checkpoint. Production inference does not accept manual architecture
overrides.

Training
--------

- `scripts/train_phase5_inpaint.sh`
- `scripts/train_phase5_cross_v1.sh`
- `scripts/train_pix2pix_postprocess.sh`
- `scripts/phase4_probnet_workflow.sh`
- `scripts/phase4_probnet_workflow_all.sh`

The pix2pix launcher pins the approved epoch-26/full-pyramid continuation
recipe. Cross V1 training can use UNI/IP supervision, but production Cross V1
inference deliberately loads neither UNI nor IP-Adapter weights.

Inpaint data preparation accepts one or more ``--dataset-root`` inputs. The
synthetic edit manifest records the operation (for example
``replace_like_blob``) together with ``mask_mode``, ``size_bucket`` and
``change_ratio`` so every training pair can be traced and regenerated.

Model release
-------------

Private Hugging Face repository IDs, source checkpoints, hashes and validation
commands are recorded in `docs/generation_model_release.md`. FLUX.1-dev, UNI,
training data and optimizer state are external dependencies and are not
vendored into the released inference packages.
