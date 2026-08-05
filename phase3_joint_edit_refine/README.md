# Joint Pathology Edit Refine

This package is an independent research implementation of paired tissue and
nucleus condition editing. It imports the legacy deterministic tissue tools
through public Python interfaces but does not modify the legacy package,
entrypoints, configuration, or artifacts.

The offline CLI deliberately cannot approve a condition. It requires an
explicit `provenance.joint_mechanism_id`, generates at most twelve paired
candidates, runs inherited tissue gates and new joint gates, and returns
`review_required` after deterministic ranking. Production mode rejects every
draft or unreviewed skill.

The tissue budget is compiled provisionally and may be re-brokered once when
complete-instance removal necessarily adds cell-only pixels outside `T`. This
keeps the final union `J` close to the immutable target without partially
cutting a nucleus. Auxiliary gland/lumen, alveolar, epidermal or pattern maps
are accepted only as aligned files with SHA-256 provenance; a metadata string
alone never activates a protection rule.

G2 research manifests use `capacity_floor_policy=lower_to_proven_max_safe`:
19% remains the desired target and 14% the standard tissue floor, but a case
whose topology solver proves lower capacity may return only its exact maximum
safe tissue edit. The gate records this as a below-standard fallback; ordinary
under-editing or cell-area substitution still fails.

Reference nucleus shapes are instance-bound. Native instances are preferred;
semantic components are the explicit lower-confidence fallback. Any instance
touching the top, bottom, left, or right patch edge is crop-censored and is
excluded from the reusable library. Each placement records its source instance
ID, and `reference_shape_integrity` independently vetoes missing or rejected
template provenance.

```bash
python scripts/run_joint_edit_refine.py \
  --manifest joint_cases.jsonl \
  --output-root artifacts/joint_refine
```

Use repeated `--case-id` flags for a bounded retry or performance check. The
offline multi-interface Planner is research-only: it can compile several legal
source components, but it does not claim to have interpreted H&E.

An approved candidate emits `generation_handoff/manifest.json` plus target
tissue/nuclei, `T`, `C`, `J`, and generation-support `G` masks. `J` is the area
ledger; `G` is the frozen H&E generator erase/regeneration support and is not
counted toward the 19% budget.

`--agent-mode api` enables strict-schema multimodal tissue Planner, joint
Planner and independent joint critic adapters. It is opt-in and reads the key
only from `--api-key-env`; the default offline path makes no network call.
Cell execution has two explicit modes. `--cell-executor research` keeps the
deterministic layout executor and can optionally use `--probnet-checkpoint` only
as a legal-anchor ranker. `--cell-executor mature` requires an explicit
checkpoint, nucleus instance library and `--probnet-dataset`, then invokes the
unchanged mature `inpaint_cells.generate` pipeline through a read-only adapter.
The adapter preserves patch-adaptive density/type priors, context-stabilized
ProbNet sampling, source-first complete shapes, exact quotas and its sampling
audit. Production refuses the research substitute.

Every cell plan is compiled into four non-interchangeable regions: `E` removes
complete source instances, `P` admits centers, `V` contains complete new
footprints, and `S` supplies cleared model/render context. A footprint may cross
the edge of `P` while remaining inside `V`; this prevents the artificial
nucleus-free seam caused by treating the center region as a footprint region.

Cell-only primitives use a separate count/extent budget and a preserve-tissue
tool. Tumor budding, discohesive breast invasion, junctional melanoma spread
and dispersed oral-SCC fronts therefore do not create semantic tumor islands or
borrow the G2 tissue floor. Their single/pair/cluster/cord geometry is
deterministic; ProbNet may rank legal anchors but cannot choose cluster size or
silently alter the quota.

Generate the human/pathologist review document and capability matrix with:

```bash
python scripts/review_joint_edit_knowledge.py \
  --output artifacts/joint_knowledge_review.md \
  --matrix-output artifacts/joint_capability_matrix.json
```

Every catalog entry remains `draft` until full-cohort profile statistics and
internal pathology review are attached. The current source-relative NND,
component-area and Ripley-K checks are deliberately labeled an uncalibrated
research envelope and cannot pass production composition.
