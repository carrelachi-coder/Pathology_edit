# R2 D/E v4 amendments and retry audit

The frozen v4 cohort and all first-attempt outputs remain immutable. The
following repair was identified after observing first-attempt failures and is
therefore reported as a **recovery analysis**, never as the initial v4 rate.

## A1: minimum-only typed seam replay

BCSS cases `r2dev4-003` and `r2dev4-009` passed source screening and compiled
complete cell-packing witnesses, but their mature ProbNet execution exhausted
all three deterministic attempts with an exact-count shortfall. In both cases,
the witness placed additional target-class nuclei in the required tumor-front
band beyond the compiled **minimum** seam count. The executor filled the
minimum first, then excluded that entire band from the remainder even though
the contract supplied no maximum. This prevented replay of the certified
packing. The saved witness footprints were independently checked against the
retained nuclei and target tissue: they fit without overlap.

The repair in `inpaint_cells/generate.py` excludes the seam from same-class
remainder placement only when the request has an explicit maximum. With no
maximum, the remainder can use the band, and the unchanged count, spatial,
continuity, and mask gates still decide validity. The original v4 code copy
used for all first attempts is unchanged. Numbered same-case retries will use
the repaired code after the initial run and will retain links to the original
failure records.

The regression test covers a minimum-only seam with a larger compiled total;
the previously existing test covers a bounded seam. Syntax checks and an
isolated behavioral test in the server's scientific Python environment passed.
The actual failed cases still require full retry and independent E verification
before any recovery is counted.

The retry harness now accepts an explicit `--code-root`. Recovery attempts for
this repair will use a separate `code_92ddb4f` source copy and record hashes
for both the Planner modules and `inpaint_cells/generate.py`; the original
`code` directory continues to serve all first attempts.

## Queue interruption

The local Terra CLI queue worker exited when its interactive session ended.
The GPU workers and first-attempt result files continued running. The worker
was restarted as a user LaunchAgent so it persists across Codex turns. Planner
transport errors and wall-clock timeouts are retained as first-attempt
outcomes; any rerun will be numbered and linked to the initial result.

## A2: radial depletion class allocation

Several GLaS and ORCA cell-only first attempts met the prescribed deletion
count, spatial extent, radial density gradient, and whole-instance constraints
but failed the existing class-composition gate. The compiler rounded each
radial band's class quotas independently; those rounding errors accumulated
across bands. The repair performs deterministic **same-band class swaps** to
move the overall deletion mix toward the existing source-derived target. It
does not alter any radial deletion count, total budget, candidate mask, gate,
or tolerance. Where the source bands lack an eligible class, the residual
composition error remains and the ordinary gate can still reject the edit.

An isolated audit of eight saved failed cases found a feasible allocation
meeting the exact global class target for seven and an allocation within the
unchanged tolerance for the eighth. This is a feasibility check, not a
recovered outcome. A focused regression test and related density-field tests
passed in the scientific Python environment; full numbered retries and
independent E checks remain necessary.

## A3: explicit PANDA growth target fine ID

PANDA Pattern-4 growth case `r2dev4-123` selected a verified fine-9 tumor
front, and its compiled tool program authorized Stroma fine 2 to Pattern-4
fine 9. The first planning pass painted fine 9 correctly, but its second-pass
area replan fell back to fine 8, the first fine ID in the coarse Tumor class.
The unchanged fine-pattern gate correctly rejected all three joint candidates.
A separate fine-label binding
now applies the **single explicit mechanism-authorized target fine ID** to
the generic candidate's changed pixels, only when every changed source pixel
belongs to the explicitly authorized source fine IDs and the generic target
is the coarse class default. This leaves all unrelated fine labels untouched
and retains the normal tissue and joint gates after rebinding. A focused
regression test passed; full same-case retry is still required before recovery
can be counted.

## Numbered recovery observations and infrastructure queue

The first-attempt v4 run reached all 222 terminal outcomes before the
infrastructure recovery queue was frozen. Its unamended result was 156
validated, 49 abstained, 4 timed out, and 13 Planner transport errors;
independent E passed for all 156 validated outputs. These denominators remain
the primary v4 first-attempt result.

BCSS case `r2dev4-003` was rerun as attempt 2 under `code_e0438ad`. Its
executor reached the joint gate, but the new placement produced five inner
seam centers where the unchanged continuity contract allowed one to three.
The case remains a failed recovery; passing a packing preflight alone is not
evidence that the image-mask edit satisfies the complete contract. The
attempt-2 metadata and gate reports are retained. This observation also
limits A1: its conditional seam exclusion fixes a replay constraint, but does
not guarantee a feasible final density profile.

BCSS case `r2dev4-012`, whose first attempt failed from Planner transport,
was rerun as attempt 2 using the same frozen request and `code_e0438ad`.
It validated and passed independent E. The first-attempt transport failure
remains in the original result; the recovery is counted only in a separate
attempt-linked analysis.

The remaining initial Planner transport errors and wall-clock timeouts are
eligible for a frozen infrastructure recovery queue. The queue manifest
records each first-result SHA-256 and cohort digest, selects only those two
failure categories, and partitions numbered attempt-2 runs across workers.
Existing attempts are preserved and skipped, so a runner restart cannot
overwrite an outcome. Scientific abstentions are excluded and require
case-specific investigation before any amended run.

## A4: pass the source-composition policy through radial depletion

The first eight ORCA infrastructure retries reached cell-program compilation
and then all failed with the same `TypeError`: the caller supplied
`composition_include_outer_reference`, but
`_select_gradient_removal_instances` did not accept it. The downstream
`_select_density_field_instances` already accepted and used that parameter
for the existing class-composition target; the intermediate function simply
failed to forward it. This is a software interface error, not a biological
rejection. The repair adds the missing intermediate parameter and passes it
through unchanged. Every failed attempt-2 output remains preserved; any
affected case must receive an explicitly numbered attempt 3 under the
patched source, followed by independent E if validated.
