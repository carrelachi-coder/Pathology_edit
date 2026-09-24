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

## Queue interruption

The local Terra CLI queue worker exited when its interactive session ended.
The GPU workers and first-attempt result files continued running. The worker
was restarted as a user LaunchAgent so it persists across Codex turns. Planner
transport errors and wall-clock timeouts are retained as first-attempt
outcomes; any rerun will be numbered and linked to the initial result.
