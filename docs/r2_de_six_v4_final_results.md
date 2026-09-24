# R2 D/E six-dataset v4: final auditable mask results

The source-qualified v4 cohort was frozen before execution: 222 natural-language
requests, 37 per dataset, with cohort SHA-256
`f5d677ff400546eabaf23d7935a0dcaa180575511a538863b70200787df1fbaa`.
All 222 first attempts have terminal outcomes. The **primary first-attempt
result** is 156/222 validated (70.3%); 49 abstained, four reached the
prespecified 1,800-second limit, and 13 failed because of Planner transport.
All 156 initially validated masks passed the independent E raster audit.

After separately logged software and infrastructure recovery, **176/222
unique frozen requests** had a validated mask with an independent passing E
audit (79.3%). This is a recovery-analysis result, not a replacement for the
primary first-attempt rate. There were 37 numbered recovery attempts across
28 distinct cases; 20 additional cases validated. All original results,
failed attempts, and case identities remain preserved. No case was swapped
or selected post hoc, and no hard gate was weakened.

| Dataset | Requests | First-attempt validated | Added by recovery | Final validated | Unresolved |
|---|---:|---:|---:|---:|---:|
| BCSS | 37 | 25 | 3 | 28 | 9 |
| GLaS | 37 | 25 | 4 | 29 | 8 |
| IGNITE | 37 | 31 | 0 | 31 | 6 |
| ORCA | 37 | 22 | 10 | 32 | 5 |
| PANDA | 37 | 19 | 3 | 22 | 15 |
| PUMA | 37 | 34 | 0 | 34 | 3 |
| **Total** | **222** | **156** | **20** | **176** | **46** |

| Primitive | Requests | First-attempt validated | Final validated |
|---|---:|---:|---:|
| Cohesive boundary expansion | 73 | 46 | 51 |
| Cellularity decrease | 74 | 47 | 61 |
| Cell-type abundance decrease | 45 | 35 | 36 |
| Cell-type abundance increase | 15 | 14 | 14 |
| Generic immune infiltrate decrease | 10 | 9 | 9 |
| Neoplastic-cell abundance decrease | 5 | 5 | 5 |

The independent E audit measured every one of the 176 accepted outputs. It
found zero changed pixels outside the declared generation region, zero
protected-tissue changes, zero unauthorized tissue transitions, and zero
nuclear-label changes outside the declared cell region. These are **mask
contract checks**. They do not establish H&E realism or clinical validity;
the R2 D/E evaluation did not use physician ratings.

## Recovery accounting

Four initial infrastructure failures validated on numbered attempt 2. The
radial-depletion quota amendment validated all eight frozen GLaS/ORCA cases
that had failed the local-population-density gate. Of eight ORCA cases whose
attempt 2 exposed an interface-parameter `TypeError`, six validated on
attempt 3 and two remained infeasible or failed an unchanged gate. PANDA
case `r2dev4-123` validated after the explicit Pattern-4 fine label and its
candidate declaration were bound together. BCSS case `r2dev4-009` validated
after the minimum-only seam replay repair; `r2dev4-003` remained rejected by
the unchanged seam-density gate. The [amendment log](r2_de_six_v4_amendments.md)
documents each repair and its scope.

Of the 46 unresolved requests, 16 lacked an exact-capacity cell candidate,
11 exhausted distinct deterministic replan options, four failed a joint
gate, four failed source/skill feasibility, three failed a cell-condition
gate, two failed cell execution, five had other planning/compilation
failures, and one timed out on both its initial and numbered retry. These
are retained as failures or abstentions. In particular, the 16
exact-capacity cases must not be turned into successes by loosening the
specified cell-count range or selecting replacement patches.

## Source-patch dependence

The 222 requests use 210 distinct dataset-patch pairs. Twelve PUMA patches
were each reused for two different requests. On 11 of these patches both
requests validated, and on one patch one validated while the other failed.
Thus outcomes can depend on the requested operation even on the same source
patch; these repeated requests are not independent patch observations.
Recorded source-group counts were BCSS 17, GLaS 15, IGNITE 11, ORCA 18,
PANDA 15, and PUMA 10. The available grouping metadata does not establish
verified patient identity, so patient-level confidence intervals or
generalization claims are not justified from this audit.

The earlier v3 analysis (141/222 after its own recovery procedure) is
reported separately. v3 and v4 used different frozen source cohorts and
must not be treated as a paired improvement experiment.

## Audit artifacts

The frozen cohort and protocol are in
[`benchmarks/r2_de_six_v4`](../benchmarks/r2_de_six_v4/). The
[`results` directory](../benchmarks/r2_de_six_v4/results/) contains the
initial outcome CSV and summary, numbered-retry manifests, full case-level
reconciliation, final statistics, and initial/final failure audits. The
remote run retains all generated masks, gate reports, independent E files,
and individual retry metadata under
`/data1/lyw/pathology_edit_eval/r2_de_terra_six_v4_20260924` on `amax2`.

The branch `codex/de-terra-audit` contains the code changes and audit
scripts. The first-attempt runtime source copy remains unchanged; corrected
source copies are versioned separately and recorded in retry metadata.
