# Figure 2D/E: frozen six-dataset Terra mask-edit evaluation

The version-3 benchmark froze 222 requests before execution: 37 patches from
each of BCSS, GlaS, IGNITE, ORCA, PANDA, and PUMA. It uses ten
capability-conditioned dataset–primitive bindings, rather than the 74 bindings
in the earlier design. Requests use prebound semantics, so these results
evaluate the mask-aware Planner and executor, not natural-language Parser
accuracy. Each LLM decision was made with a fresh GPT-5.6 Terra CLI call.

The immutable cohort has SHA-256
`8693a26f8a453b3ed06136b674c0adc597bfbacbe55d2474ec2a60c364f856bc`.
All 222 initial attempts have a terminal result. Initial outcomes were 120
validated, 84 abstained, 14 Planner transport errors, three timeouts, and one
Planner output-format error. The initial validated rate is **120/222 (54.1%)**.
The frozen cohort was never replaced after seeing an outcome. Every retry
retains the initial result and a numbered same-case attempt with a source
digest and links to its independent E check.

| Dataset | Frozen patches | Initial validated | Same-case recoveries | Validated after retries | Source groups |
|---|---:|---:|---:|---:|---:|
| BCSS | 37 | 10 | 15 | 25 | 16 |
| GlaS | 37 | 23 | 0 | 23 | 14 |
| IGNITE | 37 | 28 | 2 | 30 | 12 |
| ORCA | 37 | 16 | 4 | 20 | 20 |
| PANDA | 37 | 11 | 0 | 11 | 15 |
| PUMA | 37 | 32 | 0 | 32 | 7 |
| **Total** | **222** | **120** | **21** | **141** | **84 dataset-wise groups** |

With the completed same-case retries, **141/222 (63.5%)** distinct requests
have a validated mask output. The prespecified aspiration of 200/222 was not
met; it is not defensible to reach it by swapping patches, weakening hard
constraints, or counting attempts as new cases. The other 81 requests retain
their failed or abstained outcomes. Among them, the *initial* failure audit
classified 41 as lacking exact source-instance removal capacity, 14 as
failing the local cell-condition gate, eight as source/skill infeasible, four
as stalled replanning, and the remainder across execution, joint-gate,
infrastructure, contract, format, and other errors. These categories are
diagnostic labels from the first attempt, not a claim that every subsequent
failure had the same cause. The full per-attempt files permit that distinction.

Independent E was run for **all 141 validated outputs**, including all
same-case recoveries. The independent checker recomputed mask changes from
the original and edited rasters, using the declared generation region,
protected tissue labels, allowed tissue transitions, and authorized nuclei
region. Each validated output passed. Across the six datasets, the maximum
changed pixels outside the generation region, protected-tissue changes,
unauthorized tissue transitions, and nuclei-label changes outside the declared
cell region were all zero. These are mask-contract checks; they do not measure
H&E synthesis quality, clinical plausibility, diagnostic accuracy, or doctor
agreement. The cell-instance count audit uses the executor's event ledger and
is not independent instance-level ground truth.

Several post-start software repairs were needed to make the same frozen
requests executable through the CLI. They repaired schema prompting, bounded
transport retries, preservation of mature-cell provenance, population-zone
prechecks, and oversized Planner input packets. All changes, retry manifests,
and packet-size evidence are documented in
[`r2_de_six_v3_protocol.md`](r2_de_six_v3_protocol.md) and the corresponding
`benchmarks/r2_de_six_v3/protocol_amendment_*.json` files. None relaxed a
pathology, capacity, or raster preservation gate. Targeted regression tests
for the packet fixes passed (23 tests and 15 subtests); the larger joint-edit
suite was not fully green and must not be cited as passing.

The case-level reconciliation and aggregate statistics are committed in
`benchmarks/r2_de_six_v3/final/`. Raw masks, CLI requests and responses,
initial results, numbered retry artifacts, and independent E files remain at
`/data1/lyw/pathology_edit_eval/r2_de_terra_six_v3_20260924` on `amax2`.
Filename-based source groups identify correlated patches, not verified unique
patients. PANDA's Radboud-style label schema does not establish the slide's
acquisition institution; PUMA's exact skin subsite is not available.
