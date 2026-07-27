# Mask Edit Benchmark v1 Runbook

## Remote layout

- Repository: `/home/lyw/wqx-DL/flow-edit/FlowEdit-main`
- Manifests: `runs/benchmark_v1/manifests`
- Large execution outputs: `/data1/zhao/wqx/benchmark_v1`
- Health/build logs: `/data1/zhao/wqx/benchmark_v1/logs`
- Python: `/home/lyw/anaconda3/envs/pathology-phase5-inpaint/bin/python`

Large outputs must not be written to `/data`, which is nearly full. The run scripts use one pinned CPU, `nice=15`, idle-class IO, and refuse to start when `load1 / nproc > 0.5`.

## 1. Build formal intents

```bash
cd /home/lyw/wqx-DL/flow-edit/FlowEdit-main
scripts/build_mask_edit_benchmark_v1_amax2.sh
```

Expected outputs include:

- `mask_semantic_intents.jsonl/csv`
- `shortfalls.csv`
- `ordinal_groups.jsonl`
- `intent_qc.manual_review.csv`
- `build_summary.json`
- `effective_build_config.json`

The formal config requires image-mask pairing, records WSI/patient provenance, uses complete same-reference strength groups, and caps each WSI at 10 patches per cell.

## 2. Generate prompts

The full LLM generator/checker is resumable and checkpoints every 25 samples:

```bash
export BENCHMARK_ENV_FILE=/secure/path/benchmark.env
scripts/generate_mask_edit_benchmark_prompts_v1_amax2.sh
```

The env file must define `VISION_API_KEY`. It must not be committed or placed in a run directory.

For a deterministic no-API prompt bank, omit both LLM flags and invoke the Python CLI directly.

## 3. Run modes separately

```bash
MODE=gt RUN_NAME=mask_semantic_gt scripts/run_mask_edit_benchmark_v1_amax2.sh
MODE=instruction RUN_NAME=mask_semantic_instruction scripts/run_mask_edit_benchmark_v1_amax2.sh
```

Do not combine modes in the headline report. `gt` is the execution-control ceiling and target-mask source; `instruction` is the full-pipeline semantic evaluation.

## 4. Recompute an existing run

```bash
python -m phase3_mask_edit.cli.recompute_mask_edit_benchmark_metrics \
  --intents runs/mask_edit_semantic_benchmark_gt_v4/benchmark_intents.jsonl \
  --eval-results runs/mask_edit_semantic_benchmark_direct_gt_v4_full_20260630_163343/benchmark_eval_results.csv \
  --output runs/benchmark_v1/recomputed_v4_20260713 \
  --path-root /home/lyw/wqx-DL/flow-edit/FlowEdit-main
```

This produces DHR, OTTR, OTCR, SCR, magnitude-bucket results, WSI-clustered bootstrap confidence intervals, ordinal monotonicity, and a target-mask-bank manifest without modifying the original run.

## 5. Final full benchmark results (2026-07-22)

The frozen benchmark contains 14,288 cases per mode. The canonical final Instruction output is:

`/data1/zhao/wqx/benchmark_v1/full_v0_2/runs/instruction_v0_3_final`

The publication figure is generated at:

`outputs/mask_edit_benchmark_v0_3/figures/mask_edit_benchmark_overview.{png,pdf,svg}`

### Overall metrics

All values are percentages. Figure confidence intervals are 95% WSI-clustered bootstrap intervals based on 2,000 resamples. Semantic core is the conjunction of correct class transition, direction, and location. Edit-specificity ratios are calculated among completed cases.

| Metric | GT | Instruction |
| --- | ---: | ---: |
| Completion | 99.860 | 99.923 |
| Semantic core | 98.110 | 98.243 |
| Class transition | 99.860 | 99.923 |
| Direction | 99.860 | 99.923 |
| Location | 98.110 | 98.243 |
| On-target transition | 100.000 | 100.000 |
| Off-target preservation | 100.000 | 100.000 |
| Spatial containment | 100.000 | 100.000 |

Parser results:

- Instruction primitive exact: `14,288/14,288` (`100.000%`).
- Instruction-to-intent direction conflicts: `0`.

Strength is interpreted primarily through within-reference ordinal trajectories rather than a unique pixel-area target. Instruction has `83.932%` concordant, `15.267%` tied, and `0.801%` reversed pairs, with `96.071%` nondecreasing groups. The strict intended magnitude-bucket agreement (`78.010%`) and strict primary pass rate (`76.442%`) are retained as secondary calibration diagnostics.

### Agentic contour loop

| Metric | Result |
| --- | ---: |
| First-attempt success | 97.340% |
| Cumulative success at k=2 | 99.720% |
| Cumulative success at k=3 | 99.846% |
| Cumulative success at k=5 | 99.909% |
| Cumulative success at k=10 | 99.923% |
| Replanned cases | 380 |
| Recovered after replanning | 369 (97.105%) |
| Terminal failures | 11 (2.895%) |

The terminal failures comprise five `stroma_decrease` cases with no effective change and six fine-transition cases that did not satisfy the source-relative change-area constraint. Semantic replanning was not triggered in the corrected final Instruction benchmark because primitive parsing was exact for all 14,288 cases.

### Figure caption

**Figure X. Mask-edit benchmark performance and recovery through agentic contour replanning.** (A) Overall performance for ground-truth semantic intents (GT) and direct edit instructions (Instruction), with 14,288 cases per mode. Points show means, horizontal colored bars show 95% confidence intervals from 2,000 WSI-clustered bootstrap resamples, and gray connectors span the two mode estimates for each metric. Semantic core requires correct class transition, direction, and location; edit-specificity ratios are calculated among completed cases. (B) Ordinal strength trajectories for complete four-level groups, shown as the median and interquartile range after normalizing edit response to the maximum within each reference group (GT, n=2,584; Instruction, n=2,595). These trajectories assess within-reference ordering and do not assume a unique pixel-area ground truth for natural-language strength. (C) Cumulative contour execution success for the Instruction benchmark as the maximum number of allowed attempts increases from one to ten; the expanded y-axis displays absolute success rates. (D) Outcomes among the 380 Instruction cases that entered contour replanning: 369 were recovered (97.1%) and 11 remained terminal failures (2.9%). Instruction parsing and contour proposal used gpt-4.1-mini. Semantic replanning was not triggered in the corrected final Instruction set because primitive parsing was exact for all cases.
