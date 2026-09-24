# R2 Figure 2D/E six-dataset Terra evaluation, version 3

The original 222-request cohort covered 74 dataset–primitive bindings with
three patches each. That design predated the final mask-edit contracts and
included requests with unavailable clinical premises, ROI authority, native
instance authority, or provider metadata. Version 3 uses 222 **new** requests:
37 each from BCSS, GLaS, ORCA, PANDA, IGNITE, and PUMA. It is a
capability-conditioned benchmark of ten specified bindings, not a replacement
estimate for all 74 original bindings. Report the original and version-3
results separately.

The allocation and selection algorithm are frozen in
`scripts/r2_de_six_freeze_cohort.py`. The selected cases are source-screened
using tissue and nuclei masks and dataset metadata, without inspecting any
Planner or executor outcome for the selected patch. There is no case
replacement after freeze. Requests use prebound benchmark semantics to
isolate Planner/executor performance; this experiment does not measure Parser
accuracy. Every Terra decision uses a fresh account-authenticated
`codex exec -m gpt-5.6-terra --ephemeral` call. The GPU server retains
candidate compilation, mask execution, and hard checks. The independent E
script recomputes raster preservation, support, label-transition, and ledger
checks for every validated output.

The PANDA `provider` value in this cohort means the **Radboud-style Gleason
annotation protocol** verified by source fine labels 8/9/10 and the repository
label map. The acquisition center of each slide is not independently
verified. This distinction matters because the [PANDA challenge
documentation](https://www.kaggle.com/c/prostate-cancer-grade-assessment/data)
specifies different label semantics for Radboud and Karolinska and places
institutional `data_provider` in the original `train.csv`. IGNITE site and
biopsy/resection values come from the exact patch entry in the local dataset
metadata. PUMA is limited to patches documented as **primary** melanoma; the
site is recorded at the dataset level as skin, while the exact anatomical
subsite is unavailable. The [PUMA dataset description](https://puma.grand-challenge.org/dataset/)
describes the primary/metastatic cohorts. These provenance scopes must remain
visible in any paper claim; neither raw H&E nor the LLM supplied them.

Filename-derived source groups cap contribution to eight patches per group.
They do not prove patient identity, and patch outcomes within a group are
correlated. Report both patch-level denominators and group counts. If a
confidence interval is needed, resample at source-group level.

The frozen source cohort and initial protocol are on the GPU server at
`/data1/lyw/pathology_edit_eval/r2_de_terra_six_v3_20260924`. Before the first
evaluation case, the wall-clock cap was amended from 600 to 1800 seconds to
allow slower tissue candidate compilation and the Terra queue. The initial
protocol is retained as `protocol_frozen_initial.json`; the executed setting
is in `protocol.json`. No source, request, skill, model checkpoint, or mask
budget changed in that amendment. Any unmet 200/222 target must be reported
as measured rather than hidden by post-hoc replacement.

After the first terminal case exposed a CLI-only schema-format issue, the
worker was amended to state stripped constraints in the prompt and allow up
to three **fresh** attempts for a schema violation. The original failed
response remains in the queue; the same frozen case may be rerun with an
explicit retry trace. See
`benchmarks/r2_de_six_v3/protocol_amendment_format_retry.json`. This
amendment does not loosen the Planner schema or any mask safety gate.

During the increase to four parallel local Terra CLI sessions, an active
session for `r2de6-001-bcss-cohesive-boundary-expansion` was interrupted.
Its first result is a Planner transport error attributable to that process
interruption. The first result for `r2de6-004-bcss-cohesive-boundary-expansion`
is a Planner output-format error. These two infrastructure outcomes are kept
distinct from scientific abstentions. Both frozen cases require a same-case
retry with their initial result and queue response retained in the audit
trail; no new patch or request may be substituted.

The early tissue cases also exposed repeated joint Planner contract errors:
mandatory mechanism rule IDs were omitted, and the pathology mechanism ID was
sometimes placed in the skill-owned cell program field. The prompt now states
the mandatory IDs directly and the output schema constrains the two fixed
cell program fields to the compiled skill layout. This is a post-start
software amendment, recorded in
`benchmarks/r2_de_six_v3/protocol_amendment_joint_contract_prompt.json`.
Per-case process start times identify which version was used. Failed cases
remain in the denominator and any rerun must keep the original attempt.

The local Terra queue consumer later exited on a transient SSH listing error;
it was restarted with retry-on-listing behavior. SSH/SCP transfer retries and
an early cohesive-boundary population-zone contract were also added. The
latter rejects a whole tumor-component cell zone for local boundary growth
before cell execution; the LLM still chooses among legal interface and anchor
options, and the independent gates are unchanged. The deployment times,
tests, and handling of affected frozen cases are recorded in
`benchmarks/r2_de_six_v3/protocol_amendment_execution_recovery.json`.

Several mature cell candidates were also rejected because the generic
adapter dropped its accepted placement ledger from the provenance trace and
omitted an already loaded calibrated shape-library authority. The adapter
now exports those existing records to the unchanged reference-shape gate.
This post-start evidence-handoff fix, a replay on one persisted candidate,
and the required same-case reruns are documented in
`benchmarks/r2_de_six_v3/protocol_amendment_mature_shape_provenance.json`.

The complete same-case retry of case 029 (attempt 3) subsequently passed the
unchanged program checks and independent E raster audit. Its original failed
attempt and failed attempt 2 remain on disk; this result is a recovery of the
same frozen request, not another cohort member. Transient Terra CLI transport
errors now receive up to three bounded fresh CLI attempts, with each error and
the total invocation count retained in the queue response. An isolated retry
queue may be selected without changing the frozen request. These changes and
the case-029 evidence are recorded in
`benchmarks/r2_de_six_v3/protocol_amendment_transport_and_retry_029.json`.
The live initial-attempt summary does not include recovered retries; final
per-case accounting must explicitly link and report both initial and accepted
retry outcomes.

The full same-case retry of BCSS case 014 (attempt 3) also passed the unchanged
program checks and independent E audit. Its attempt 2 failed before any mask
work because the launcher used the server's system Python rather than the
project environment; that infrastructure attempt is preserved. A fixed list
of 18 initial Planner transport, output-format, or wall-clock outcomes was
then frozen for sequential same-case retries in an isolated queue. The list,
initial outcomes, code version, per-attempt metadata, and batch progress are
retained under the evaluation root. This retry batch cannot change the source
cohort or the initial-attempt denominator.

Once initial GPU workers 0 and 2 finished their assigned cases, the retry
batch was split to use those idle GPUs. Cases 001, 004, and 019 had already
finished; case 020 continued in its own attempt process on GPU 3. The
remaining 14 previously selected IDs were partitioned into two disjoint
seven-case manifests, `retry_batch_infra_20260924_gpu0.json` and
`retry_batch_infra_20260924_gpu2.json`, each linking to the original retry
manifest digest. The original batch parent was stopped after its case-020
child had started; that child and all previous attempts were preserved.

`scripts/r2_de_six_reconcile.py` checks the frozen cohort digest, initial
result links, every retry's case identity and first-result digest, and the
independent E record for each accepted retry. It reports initial validated
cases separately from same-case recoveries and produces one auditable row per
frozen case. This prevents a retry from silently replacing an initial outcome
or being counted without an independently checked mask output.

Three additional BCSS cases (017, 018, 028) with initial reference-shape
provenance gate failures were frozen for same-case retries after the adapter
fix. Nine early BCSS cohesive-boundary cases that had stalled replanning or
candidate-local cell execution failures before the joint contract, population
zone, and provenance fixes were also frozen for a separate retry batch. Their
server manifests retain the exact IDs and selection reasons. These are
software-recovery audits, not additions to the 222-case cohort.
