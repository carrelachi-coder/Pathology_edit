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

The tissue budget is solved as a bounded fixed point. A provisional `T ∪ E`
can force an early re-broker only when every sibling already exceeds the hard
24% ceiling; it is not allowed to optimize the narrower desired interval
before ADD footprints exist. If fully executed cell candidates miss the
desired `J` interval, the workflow records the exact removed-instance spill
and newly placed footprint spill separately, re-brokers `T`, and re-runs both
the tissue tool and mature cell pipeline. If that exact-area replan loses the
previously certified packing capacity, the smallest otherwise-valid paired
candidate inside the hard range may be restored with an explicit
`joint_area_rebalance_exhausted` audit flag. It never crops a nucleus or
silently relaxes the joint-area gate after generation.
The canonical candidate manifest, gate report and review panels are rewritten
from that final in-memory batch before critic routing, so a restored closure
cannot leave pre-restoration failures in the authoritative audit files.
Auxiliary gland/lumen, alveolar, epidermal or pattern maps are accepted only as
aligned files with SHA-256 provenance; a metadata string alone never activates
a protection rule.

The executable scene is a typed hierarchy rather than a flat label graph:
`tissue_component -> structural_unit -> interface -> population_field ->
nucleus_instance`. GLaS gland/lumen and PANDA native-pattern/lumen units are
recovered only from versioned semantic-mask topology, bound to the source-mask
digest and exposed to Planner, tissue tools and gates through the same IDs. A
whole structural-unit turnover is available only as a reviewed
mechanism/primitive fallback after an ordinary interface-front plan has proved
the public area floor unreachable.

G2 research manifests use `capacity_floor_policy=lower_to_proven_max_safe`:
19% remains the desired target and 14% is a binding meaningful tissue floor.
A case below target may fall back only to its exact maximum safe tissue edit
inside the 14--19% interval. A paired condition above target may use only the
minimum safe union inside the 24% hard ceiling, and only after an executed
spill-driven exact-area replan proves non-executable. A lower proven capacity
is an auditable abstention. Ordinary under-editing and cell-area substitution
both fail.

Reference nucleus shapes are instance-bound. Native instances are preferred;
semantic components are the explicit lower-confidence fallback. Any instance
touching the top, bottom, left, or right patch edge is crop-censored and is
excluded from the reusable library. Each placement records its source instance
ID, and `reference_shape_integrity` independently vetoes missing or rejected
template provenance. Packing templates are also restricted to the same
per-class median-area interval enforced by the final gate. If stochastic
placement can meet the count only by preferentially accepting shrunken shapes,
execution switches to the certified complete-reference witness instead of
relaxing the local size contract.

Before ProbNet runs, the final compiled `E/P/V/C` rasters must admit an exact
packing certificate made from complete same-patch reference footprints. The
certificate freezes the total quota, exact class ledger, typed seam quota,
source instance IDs and witness centers. The seam quota is recomputed on the
final compiled continuity raster and intersected with both the local density
envelope and anchor-coverage requirement. It constrains only its declared cell
class; other compatible populations may still occupy that anatomical band, so
the seam is not misused as a cell-free exclusion zone. Contradictory candidates
are replanned, not left for post-sampling rejection. A preferred seam count
that exact footprints cannot pack may fall back only to the achieved count
when that count remains above the independently compiled density and geometric
minimums; the nominal target, hard minimum and fallback are all recorded.
Protected nuclei remain pixel-immutable but may serve as read-only shape
references when they are complete and non-border.

Packing and mature execution use the same 8-connected instance definition and
a Chebyshev one-pixel separation margin; diagonal contact can therefore never
turn two accepted centers into one raster component. CellViT class IDs are
converted explicitly to the mature sampler's 100-series type IDs in both
placement records and per-class reference-area statistics. ProbNet-selected
centers are retained whenever their local shape distribution passes; the fixed
witness is used only for a genuine count or size-contract shortfall.
If a mixed Q25/Q50/Q75 witness is capacity-limited, the certifier may retry
with the smallest complete same-patch shape that still lies inside the final
per-class median-area gate. The chosen source instance and the
`capacity_optimized_shape_fallback_used` flag remain explicit; border fragments
and out-of-distribution small shapes are never admitted.

A patch-local density prior is a finite-sample estimate, not an exact integer
biological law. A failed nominal cell count may therefore compile to the
maximum safe packing only when it remains within both 80% of the nominal count
and one square-root count deviation. The nominal count, lower bound and
effective count remain in the certificate; larger shortfalls still replan or
abstain.

```bash
python scripts/run_joint_edit_refine.py \
  --manifest joint_cases.jsonl \
  --output-root artifacts/joint_refine
```

Use repeated `--case-id` flags for a bounded retry or performance check. The
offline multi-interface Planner is research-only: it can compile several legal
source components, but it does not claim to have interpreted H&E.

If visual semantic resolution cannot distinguish two or three materially
different but executable primitives, the run returns exit code `3` and writes
`clarification_request.json`. The request is compiled only after skill and
deterministic feasibility filtering, so catalog-only or generator-unsupported
mechanisms are never exposed as choices. Bind the clinician/Codex answer with:

```bash
python scripts/resolve_joint_clarification.py \
  --request artifacts/joint_refine/CASE/clarification_request.json \
  --selected-option-id primitive:invasive-front-expansion-v1 \
  --responder clinician-id \
  --output clarification_decision.json

python scripts/run_joint_edit_refine.py \
  --manifest joint_cases.jsonl \
  --clarification-decisions clarification_decision.json \
  --output-root artifacts/joint_refine_resumed
```

The decision is bound to the instruction and all source/auxiliary digests. It
locks only the primitive; the visual Planner still selects the compatible
mechanism and interfaces. Changed inputs or capability options invalidate the
answer instead of silently replaying it.

An approved candidate emits `generation_handoff/manifest.json` plus target
tissue/nuclei, `T`, `C`, `J`, and generation-support `G` masks. `J` is the area
ledger; `G` is the frozen H&E generator erase/regeneration support and is not
counted toward the 19% budget.

Before promoting a new mechanism, freeze a three-arm generator ablation:

```bash
python scripts/prepare_joint_generator_paired_ablation.py \
  --handoff-manifest artifacts/CASE/generation_handoff/manifest.json \
  --output-root artifacts/generator_ablation/CASE \
  --generator-snapshot CHECKPOINT_DIGEST
```

The arms are source tissue+source nuclei, target tissue+source nuclei, and
target tissue+target nuclei. They share the source image, generation support,
route and fixed seed. Segmentator, CellViT, the preservation audit and an
independent visual critic must demonstrate target-condition contrast; missing
evaluators or an unresponsive cell condition returns `render_unsupported`.
Structural-void/STAS stays closed unless externally produced airspace and
alveolar maps are digest-bound and this generator ablation passes. The H&E
image itself is never used to fabricate those structural maps.

Freeze and replay-verify the materialized image/tissue/nuclei evidence for all
six annotation profiles with:

```bash
python scripts/freeze_joint_dataset_evidence.py \
  --grouped-manifest grouped_seed42.json \
  --output-root artifacts/dataset_evidence \
  --code-revision COMMIT_SHA
```

Every materialized file is hashed. A preprocessing revision absent from the
source manifest is recorded as missing evidence, never inferred from a path or
current checkout.

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

Every cell plan is compiled into five non-interchangeable regions: `T_pop` is
the target-tissue population area used for abundance and density, `E` removes
complete source instances, `P` admits centers, `V` contains complete new
footprints, and `S` supplies cleared model/render context. `T_pop` is never
replaced by the smaller legal-center domain `P`. A footprint may cross the edge
of `P` while remaining inside `V`; this prevents both under-counting and the
artificial nucleus-free seam caused by treating the center region as a
footprint region. `P` is not pre-eroded by a nominal nucleus radius: the exact
packing certificate applies each complete reference shape to `V` once, so the
final compiler cannot silently impose a second containment erosion.

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
