# Panel C follow-up: cumulative program generation

## What warranted a code change

Panel C used an existing validated two-step expansion program to obtain a larger
high condition. The final H&E inference needed a temporary composition script:
original reference + final masks + union of both generation-support masks. The
new `program_generator_adapter` and preparation CLI make that composition a
verified, reusable interface. The original single-step editor, samplers, gates,
weights and default routing thresholds remain unchanged.

The new interface does not claim that larger single-step requests always work.
In the figure exploration, 10% single-step execution failed after rebalancing;
12% (two seeds) and 16% attempts reached the 600-second limit without an approved
result. A timeout is not proof of infeasibility. Some attempts reported a
ProbNet target-count placement shortfall. The preliminary seam-exclusion change
tried during the earlier figure work was reverted and is not part of this patch:
it did not establish an end-to-end passing result or preservation of the frozen
panel cohort. No checker was relaxed to obtain the final figure.

## Observed example, not a benchmark estimate

| Quantity | Low | Revised high |
|---|---:|---:|
| Original tumour fraction | 49.2661% | 49.2661% |
| Target tumour fraction | 54.7% (rounded) | 60.1662% |
| Tumour-area change | +5.5 percentage points (rounded) | +10.9001 percentage points |
| Generation support G | 15% (rounded) | 23.6839% |
| Construction | one 6% joint-area budget | two sequential 6% joint-area budgets |
| Backend | local Inpaint | explicitly selected Cross-v1 |

The first step of the revised high condition has the same tissue and nucleus
mask pixels as the low condition. Each step passed the existing mask gates; the
program passed the state-chain audit. The final H&E used one global generation
with the cumulative support, not two successive image generations. Pixels
outside cumulative G matched the original. These are mask/engineering facts;
independent image segmentation and pathologist validation were not performed.

## Validation

Automated regressions cover retaining an earlier disjoint edit, original
reference/final target selection, one-step compatibility, source/chain/program
digest drift, selected gate failure, incomplete programs, missing support, route
selection, the forced-Cross boundary and avoiding an overwritten export.
The actual two-step Panel C audit is also replayed through the new adapter and
its cumulative masks compared with the figure's inference inputs. No claim is
made that this export interface changes image realism or repairs the broader
high-magnitude sampler failures.

Validation performed for this patch: **31 tests passed** (program adapter,
existing edit-program tests and the existing approved single-step generator
handoff regression). The real two-step case exported successfully with the new
CLI. Original H&E, original tissue/nucleus masks, final tissue/nucleus masks and
cumulative G all matched the figure inference's arrays exactly. The new generic
multi-step render prompt was not evaluated by a repeat image-generation run;
mask-input compatibility does not establish identical generated H&E.
