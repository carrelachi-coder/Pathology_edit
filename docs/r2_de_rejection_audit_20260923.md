# Figure 2D/E frozen-cohort rejection audit (2026-09-23)

The original `7b123fb` run contains 222 fixed requests: 24 validated, 197
abstained, and one timed out. The 24 validated outputs all passed the independent
raster contract audit. The run used the deterministic offline Planner, so it is
**not** evidence for the requested GPT-5.6 Terra Planner.

The 197 abstentions were assigned one primary cause each, in priority order.
These are diagnostic buckets, not claims that every item within a bucket can be
repaired by the same change.

| Primary cause | Requests | Interpretation |
|---|---:|---|
| Resolved cell class lost between Program and step budget | 36 | Code defect; the Parser's `inflammatory` class resolves to CellViT class 2, but the budget read a missing `semantic_intent.resolved_cell_class_ids`. |
| Cell candidate hard gates | 30 | Requires case-level gate and placement audit; LLM selection cannot rescue a portfolio with no certified survivor. |
| Other capacity or exact packing limits | 38 | Requires case-level examination of biological host area, reference shapes and budget. |
| Explicit post-treatment premise absent | 36 | Current skills do not permit inventing treatment history from H&E or the mask. |
| Required local-clearance ROI absent | 6 | A model cannot invent a user-authorized ROI. |
| Required pathology review authority absent | 6 | Breast cellularity skill is intentionally review-gated. |
| Source provenance absent | 21 | Potentially recoverable only from source records, not by model inference. |
| Native nucleus-instance authority absent | 15 | Pixel class masks are not equivalent to trusted native instances. |
| Primitive detached from registered tissue adapter | 6 | Possible code/catalog integration defect; requires executable and gate proof. |
| Tissue topology gate failure | 1 | Candidate shape failed the boundary artifact gate. |
| Other | 2 | Requires case-level inspection. |

The 36 post-treatment, six ROI, and six review-authority refusals alone exceed
the 22 failures allowed by a 200/222 target. Under the current skill contracts
and *unchanged* requests, a Planner swap cannot make 200 cases valid. Raising
the count by filling in unobserved treatment history, inventing an ROI or
marking unreviewed mechanisms as approved would change the task's scientific
meaning. Any revised prompt/cohort or skill contract must be versioned and
reported separately from the frozen baseline.

The first concrete defect has been fixed in `program_workflow._bind_step_case`:
the observation-profile class resolution now reaches both the provenance
binding and the source-calibrated budget. The targeted regression test passes.
A server pilot using the same frozen GLaS request
`r2de-glas-cell-type-abundance-increase-01` progressed from the old class
resolution abstention to `validated`; a BCSS request progressed past that bug
but then correctly failed a separate legal-host-zone check. This is why 36
class-resolution errors must not be reported as 36 newly validated cases.

The Terra CLI bridge was subsequently exercised end to end on the fixed GLaS
pilot. Two independent `codex exec -m gpt-5.6-terra --ephemeral` sessions
selected the semantic option and certified cell candidate; the server executor
then validated the mask result. Independent E measurement found 24 requested
and 24 actual added instance events, with zero unauthorized tissue change and
zero change outside generation support. This is one integration pilot, not a
222-request Terra result. The full Terra protocol is in
`docs/r2_de_terra_cli_protocol.md`.

The six BCSS cord/nest adapter refusals were also investigated. Registering
the two primitive IDs with the generic tumor-increase mask bundle moved two
representative requests past bundle binding, but both still failed before the
Planner: the specialized `cell_seeded_cord` and
`peritumoral_tumor_island` mechanisms have no concrete tool in common with
that generic bundle. The provisional alias change was reverted. These six
cannot be counted as rescued by a name mapping; their dedicated tissue tool
contracts need a separately reviewed integration and executable gate proof.
